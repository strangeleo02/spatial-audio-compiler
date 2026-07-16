import warnings
import logging
import os
import uuid
import shutil
import glob
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Suppress warnings and hub notices
warnings.filterwarnings("ignore", message=".*ffmpeg.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*avconv.*", category=RuntimeWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub._login").setLevel(logging.ERROR)
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# ──────────────────────────────────────────────
#  FFmpeg auto-discovery
# ──────────────────────────────────────────────
def _find_ffmpeg():
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    winget_base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    pattern = os.path.join(winget_base, "Gyan.FFmpeg*", "**", "ffmpeg.exe")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    for candidate in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None

_FFMPEG_PATH = _find_ffmpeg()
if _FFMPEG_PATH:
    _FFMPEG_DIR = os.path.dirname(_FFMPEG_PATH)
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

# ──────────────────────────────────────────────
#  FastAPI Application Setup
# ──────────────────────────────────────────────
app = FastAPI(title="3D Spatial Audio Mixer Server")

static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_folder, exist_ok=True)

STEMS_DIR = os.path.join(static_folder, "stems")
os.makedirs(STEMS_DIR, exist_ok=True)

# Mount the static directory for the stems
# Stems will be served directly at /stems/...
app.mount("/stems", StaticFiles(directory=STEMS_DIR), name="stems")

@app.get("/")
async def get_index():
    index_path = os.path.join(static_folder, "index.html")
    if not os.path.exists(index_path):
        # Fallback to serving a basic description if index.html isn't ready
        return JSONResponse({"status": "Server running. Static index.html not found yet."})
    return FileResponse(index_path)

@app.post("/api/separate")
async def api_separate(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    # Create a unique session ID for this separation task
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(STEMS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Save incoming audio file
    ext = os.path.splitext(file.filename)[1]
    input_path = os.path.join(session_dir, f"input{ext}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"\n[Server] Starting separation task for: {file.filename}")
    print(f"[Server] Session ID: {session_id}")

    try:
        import demucs.api
        separator = demucs.api.Separator(model="htdemucs", progress=True)
        _, separated = separator.separate_audio_file(input_path)

        stem_order = ["vocals", "drums", "bass", "other"]
        urls = {}
        for stem_name in stem_order:
            if stem_name not in separated:
                raise ValueError(f"Expected stem '{stem_name}' not found.")
            
            # Save audio file to session directory
            out_filename = f"{stem_name}.wav"
            out_path = os.path.join(session_dir, out_filename)
            demucs.api.save_audio(
                separated[stem_name], out_path, samplerate=separator.samplerate
            )
            print(f"[Server] Saved stem: {out_filename}")
            
            # Create public static URLs to fetch from client
            urls[stem_name] = f"/stems/{session_id}/{out_filename}"

        return {
            "success": True,
            "session_id": session_id,
            "stems": urls
        }

    except Exception as e:
        print(f"[Server] Error during separation: {e}")
        # Clean directory on failure
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Stem separation failed: {str(e)}")

    finally:
        # Clean up temporary source file
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass

# Mount generic static files at the end so it doesn't shadow /stems or routes
app.mount("/", StaticFiles(directory=static_folder, html=True), name="static")

if __name__ == "__main__":
    # Clean stems directory on launch
    if os.path.exists(STEMS_DIR):
        for item in os.listdir(STEMS_DIR):
            item_path = os.path.join(STEMS_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            
    print("\n[Server] 3D Auditorium Spatial Audio Server started.")
    print("Open http://localhost:5000 in your browser to view the 3D auditorium.\n")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="warning")
