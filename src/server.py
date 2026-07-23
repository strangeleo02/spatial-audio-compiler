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

from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import time

# ──────────────────────────────────────────────
#  Structured Logging Setup (Console & File)
# ──────────────────────────────────────────────
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "server.log")

logger = logging.getLogger("SpatialAudioBE")
logger.setLevel(logging.INFO)

if not logger.handlers:
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [BE] %(message)s",
        datefmt="%H:%M:%S"
    ))
    f_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    f_handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

# ──────────────────────────────────────────────
#  FastAPI Application Setup
# ──────────────────────────────────────────────
app = FastAPI(title="3D Spatial Audio Mixer Server")

# Enable CORS for all origins (Electron file:// & localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 1)
    if not request.url.path.startswith("/stems"):
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)")
    return response

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
        return JSONResponse({"status": "Server running. Static index.html not found yet."})
    return FileResponse(index_path)

import asyncio
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect

# Active WebSocket progress connections per session_id
progress_sockets: Dict[str, List[WebSocket]] = {}

@app.websocket("/ws/progress/{session_id}")
async def websocket_progress(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in progress_sockets:
        progress_sockets[session_id] = []
    progress_sockets[session_id].append(websocket)
    print(f"[WebSocket] Client connected for progress tracking: session {session_id}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if session_id in progress_sockets and websocket in progress_sockets[session_id]:
            progress_sockets[session_id].remove(websocket)
        print(f"[WebSocket] Client disconnected: session {session_id}")

async def broadcast_progress(session_id: str, data: dict):
    if session_id in progress_sockets:
        dead_sockets = []
        for ws in progress_sockets[session_id]:
            try:
                await ws.send_json(data)
            except Exception:
                dead_sockets.append(ws)
        for ds in dead_sockets:
            progress_sockets[session_id].remove(ds)

@app.post("/api/separate")
async def api_separate(file: UploadFile = File(...), session_id: str = Form(None)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    if not session_id:
        session_id = str(uuid.uuid4())

    session_dir = os.path.join(STEMS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    ext = os.path.splitext(file.filename)[1]
    input_path = os.path.join(session_dir, f"input{ext}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"Starting stem separation task for file: {file.filename} (Session: {session_id})")

    loop = asyncio.get_running_loop()

    last_pct = [-1]
    def demucs_callback(kwargs):
        try:
            step = kwargs.get("step", 0) if isinstance(kwargs, dict) else getattr(kwargs, "step", 0)
            total = kwargs.get("total", 1) if isinstance(kwargs, dict) else getattr(kwargs, "total", 1)
            if total > 0:
                pct = int((step / total) * 100)
                if pct != last_pct[0]:
                    last_pct[0] = pct
                    status_msg = f"Neural stem separation... ({pct}%)"
                    asyncio.run_coroutine_threadsafe(
                        broadcast_progress(session_id, {"percent": pct, "status": status_msg, "step": step, "total": total}),
                        loop
                    )
        except Exception:
            pass

    try:
        import torch
        import demucs.api

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            total_cores = os.cpu_count() or 4
            num_threads = max(1, total_cores - 2)
            torch.set_num_threads(num_threads)
            logger.info(f"Demucs CPU acceleration initialized with {num_threads} threads (2 cores reserved for UI).")
        else:
            logger.info("Demucs PyTorch CUDA GPU acceleration initialized.")

        await broadcast_progress(session_id, {"percent": 5, "status": "Loading Demucs neural model..."})

        separator = demucs.api.Separator(model="htdemucs", device=device, shifts=1, progress=True, callback=demucs_callback)
        
        await broadcast_progress(session_id, {"percent": 15, "status": "Separating audio stems..."})

        def run_demucs_inference():
            with torch.no_grad():
                return separator.separate_audio_file(input_path)

        _, separated = await asyncio.to_thread(run_demucs_inference)

        await broadcast_progress(session_id, {"percent": 85, "status": "Exporting stem audio files..."})

        stem_order = ["vocals", "drums", "bass", "other"]
        urls = {}
        for stem_name in stem_order:
            if stem_name not in separated:
                raise ValueError(f"Expected stem '{stem_name}' not found.")
            
            out_filename = f"{stem_name}.wav"
            out_path = os.path.join(session_dir, out_filename)
            stem_tensor = separated[stem_name]
            if hasattr(stem_tensor, "cpu"):
                stem_tensor = stem_tensor.cpu()

            try:
                demucs.api.save_audio(
                    stem_tensor, out_path, samplerate=separator.samplerate, clip="val", bits_per_sample=16
                )
            except Exception:
                demucs.api.save_audio(
                    stem_tensor, out_path, samplerate=separator.samplerate
                )
            
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise RuntimeError(f"Failed to write audio stem file: {out_filename}")

            logger.info(f"Saved stem audio file: {out_filename} ({round(os.path.getsize(out_path)/(1024*1024), 2)} MB)")
            urls[stem_name] = f"/stems/{session_id}/{out_filename}"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        await broadcast_progress(session_id, {"percent": 100, "status": "Stem separation complete!"})
        logger.info(f"Separation task finished successfully for session: {session_id}")

        return {
            "success": True,
            "session_id": session_id,
            "stems": urls
        }

    except Exception as e:
        logger.error(f"Error during stem separation task: {e}")
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
    import webbrowser
    import threading

    # Clean stems directory on launch
    if os.path.exists(STEMS_DIR):
        for item in os.listdir(STEMS_DIR):
            item_path = os.path.join(STEMS_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)

    def launch_browser():
        time.sleep(1.0)
        webbrowser.open("http://127.0.0.1:5000")

    logger.info("3D Auditorium Spatial Audio Server started.")
    logger.info("Opening http://127.0.0.1:5000 in your browser...")
    threading.Thread(target=launch_browser, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="warning")
