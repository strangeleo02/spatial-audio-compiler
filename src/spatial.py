import warnings
import logging
import os

# Suppress warnings and hub notices
warnings.filterwarnings("ignore", message=".*ffmpeg.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*avconv.*", category=RuntimeWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub._login").setLevel(logging.ERROR)
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import customtkinter as ctk
from tkinter import filedialog, messagebox
from pydub import AudioSegment
import numpy as np
import sounddevice as sd
import threading
import tempfile
import shutil
import glob
import sys
import urllib.request
import scipy.io
import scipy.signal

# ── Dynamic Path Configuration ────────────────────────────────────────────────
# Ensure the root folder (where setup.py compiles the .pyd) is in the search path.
# This makes imports work whether run from the project root or inside src/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import our compiled C++ Binaural DSP engine
try:
    import binaural_dsp
except ImportError:
    messagebox.showerror(
        "Import Error",
        "Could not load the compiled C++ DSP engine (binaural_dsp.pyd).\n"
        "Please compile it first using: python setup.py build_ext --inplace"
    )
    raise

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
    AudioSegment.converter = _FFMPEG_PATH
    AudioSegment.ffmpeg = _FFMPEG_PATH
    AudioSegment.ffprobe = os.path.join(_FFMPEG_DIR, "ffprobe.exe")


# ──────────────────────────────────────────────
#  Audio processing helpers
# ──────────────────────────────────────────────
def load_audio_tracks(file_paths):
    return [AudioSegment.from_file(path) for path in file_paths]


def separate_with_demucs(input_path, output_dir, progress_cb=None):
    import demucs.api

    print("\n[Demucs] Loading model: htdemucs")
    if progress_cb:
        progress_cb("Loading Demucs model (htdemucs)…")

    separator = demucs.api.Separator(model="htdemucs", progress=True)

    print(f"[Demucs] Separating: {os.path.basename(input_path)}")
    print("[Demucs] This may take 1–2 min on CPU…")
    if progress_cb:
        progress_cb("Separating stems… (watch terminal for progress)")

    _, separated = separator.separate_audio_file(input_path)

    print("[Demucs] Separation complete. Saving stems…")
    stem_order = ["vocals", "drums", "bass", "other"]
    stem_paths = []
    for stem_name in stem_order:
        if stem_name not in separated:
            raise ValueError(f"Expected stem '{stem_name}' not found.")
        out_path = os.path.join(output_dir, f"{stem_name}.wav")
        demucs.api.save_audio(
            separated[stem_name], out_path, samplerate=separator.samplerate
        )
        stem_paths.append(out_path)
        print(f"[Demucs]   ✓ {stem_name}.wav")
        if progress_cb:
            progress_cb(f"Saved {stem_name}.wav")

    print("[Demucs] All stems ready.\n")
    return stem_paths


# Set CustomTkinter theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")  # fallback theme

STEM_NAMES = ["Vocals", "Drums", "Bass", "Other"]


class SpatialAudioMixerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("3D Binaural Spatial Audio Mixer")
        self.geometry("820x960")
        self.resizable(False, False)

        # Thread safety lock for position updates and audio callback
        self.lock = threading.Lock()

        # Custom violet/accent theme
        self.COLOR_ACCENT = "#7c3aed"  # Modern violet
        self.COLOR_SUCCESS = "#10b981"  # Emerald green
        self.COLOR_DANGER = "#ef4444"  # Coral red
        self.COLOR_WARNING = "#f59e0b"  # Amber

        # ── Audio state ──────────────────────────────────────────────
        self.tracks = []        # raw AudioSegments for export
        self.track_arrays = []  # mono float32 NumPy arrays
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.is_seeking = False
        self.master_volume = 1.0
        self.stream = None

        # 4 C++ DSP Binaural Processors
        self.dsp_processors = [binaural_dsp.BinauralProcessor(44100.0) for _ in range(4)]
        # Track position settings (azimuth, elevation, distance)
        self.track_positions = [
            {"azimuth": 0.0, "elevation": 0.0, "distance": 1.0} for _ in range(4)
        ]

        self._demucs_tmpdir = None
        self.hrir_l = None
        self.hrir_r = None

        # ── Build GUI ───────────────────────────────────────────────
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ── Load KEMAR CIPIC Database ───────────────────────────────
        self.load_kemar_async()

    def create_widgets(self):
        # Background / Main Container
        self.configure(fg_color="#0f172a")  # Slate 900 background

        # Header Frame
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(20, 10))

        title_lbl = ctk.CTkLabel(
            header,
            text="3D BINAURAL SPATIAL AUDIO MIXER",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#f8fafc"
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = ctk.CTkLabel(
            header,
            text="AI-powered Demucs Stem Separation  •  High-Performance C++ Binaural Engine",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#94a3b8"
        )
        subtitle_lbl.pack(anchor="w", pady=(2, 0))

        # ── Loaded Stems panel ──────────────────────────────────────────
        stems_card = ctk.CTkFrame(self, fg_color="#1e293b", corner_radius=12)
        stems_card.pack(fill="x", padx=24, pady=8)

        stems_title = ctk.CTkLabel(
            stems_card,
            text="LOADED STEMS",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=self.COLOR_ACCENT
        )
        stems_title.pack(anchor="w", padx=16, pady=(12, 6))

        self.track_labels = []
        for i in range(4):
            lbl = ctk.CTkLabel(
                stems_card,
                text=f"●  {STEM_NAMES[i]}: (empty — load audio below)",
                font=ctk.CTkFont(family="Segoe UI", size=13),
                text_color="#cbd5e1",
                anchor="w"
            )
            lbl.pack(fill="x", padx=16, pady=2)
            self.track_labels.append(lbl)
        # Spacer
        ctk.CTkLabel(stems_card, text="", font=ctk.CTkFont(size=4)).pack()

        # ── Playback & Operations Card ──────────────────────────────────
        ops_card = ctk.CTkFrame(self, fg_color="#1e293b", corner_radius=12)
        ops_card.pack(fill="x", padx=24, pady=8)

        # Progress slider & Time label
        progress_frame = ctk.CTkFrame(ops_card, fg_color="transparent")
        progress_frame.pack(fill="x", padx=16, pady=(14, 8))

        self.progress_slider = ctk.CTkSlider(
            progress_frame,
            from_=0,
            to=100,
            number_of_steps=1000,
            button_color=self.COLOR_ACCENT,
            button_hover_color="#9333ea",
            progress_color=self.COLOR_ACCENT,
            fg_color="#334155",
            command=self.on_slider_seek
        )
        self.progress_slider.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self.progress_slider.bind("<Button-1>", self.on_seek_start)
        self.progress_slider.bind("<ButtonRelease-1>", self.on_seek_end)
        self.progress_slider.set(0)

        self.time_label = ctk.CTkLabel(
            progress_frame,
            text="00:00 / 00:00",
            font=ctk.CTkFont(family="Consolas", size=13, weight="bold"),
            text_color="#94a3b8"
        )
        self.time_label.pack(side="right")

        # Action Buttons Row
        btn_row = ctk.CTkFrame(ops_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=8)

        self.demucs_btn = ctk.CTkButton(
            btn_row,
            text="⬡  Separate with Demucs",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=self.COLOR_ACCENT,
            hover_color="#6d28d9",
            text_color="#ffffff",
            corner_radius=8,
            command=self.start_demucs_separation
        )
        self.demucs_btn.pack(side="left", padx=(0, 8))

        self.load_btn = ctk.CTkButton(
            btn_row,
            text="Load 4 Files Manually",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            fg_color="#475569",
            hover_color="#334155",
            text_color="#f1f5f9",
            corner_radius=8,
            command=self.load_files
        )
        self.load_btn.pack(side="left")

        self.export_btn = ctk.CTkButton(
            btn_row,
            text="Mix & Export",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color="#3b82f6",
            hover_color="#2563eb",
            text_color="#ffffff",
            corner_radius=8,
            command=self.mix_audio
        )
        self.export_btn.pack(side="right")

        # Transport Control Row
        transport_row = ctk.CTkFrame(ops_card, fg_color="transparent")
        transport_row.pack(fill="x", padx=16, pady=(4, 12))

        self.play_btn = ctk.CTkButton(
            transport_row,
            text="▶   Play",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=self.COLOR_SUCCESS,
            hover_color="#059669",
            text_color="#ffffff",
            corner_radius=8,
            state="disabled",
            command=self.toggle_play
        )
        self.play_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = ctk.CTkButton(
            transport_row,
            text="■   Stop",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=self.COLOR_DANGER,
            hover_color="#dc2626",
            text_color="#ffffff",
            corner_radius=8,
            state="disabled",
            command=self.stop_playback
        )
        self.stop_btn.pack(side="left", padx=(0, 16))

        # Volume slider inline
        ctk.CTkLabel(
            transport_row,
            text="Master Volume",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#94a3b8"
        ).pack(side="left", padx=(0, 10))

        self.volume_slider = ctk.CTkSlider(
            transport_row,
            from_=0,
            to=100,
            button_color=self.COLOR_SUCCESS,
            button_hover_color="#059669",
            progress_color=self.COLOR_SUCCESS,
            fg_color="#334155",
            command=self.update_volume
        )
        self.volume_slider.set(100)
        self.volume_slider.pack(side="left", fill="x", expand=True)

        # ── 3D Spatial Positioning Panel ────────────────────────────────
        spatial_panel = ctk.CTkFrame(self, fg_color="#1e293b", corner_radius=12)
        spatial_panel.pack(fill="both", expand=True, padx=24, pady=(8, 12))

        spatial_title = ctk.CTkLabel(
            spatial_panel,
            text="3D BINAURAL POSITIONING (PER STEM)",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=self.COLOR_ACCENT
        )
        spatial_title.pack(anchor="w", padx=16, pady=(12, 4))

        # Scrollable container for the 4 tracks
        scroll_container = ctk.CTkScrollableFrame(
            spatial_panel,
            fg_color="transparent",
            scrollbar_button_color=self.COLOR_ACCENT,
            scrollbar_button_hover_color="#9333ea"
        )
        scroll_container.pack(fill="both", expand=True, padx=8, pady=(4, 12))

        self.azimuth_sliders = []
        self.elevation_sliders = []
        self.distance_sliders = []

        self.azimuth_val_labels = []
        self.elevation_val_labels = []
        self.distance_val_labels = []

        for i in range(4):
            # Card for this specific track
            track_card = ctk.CTkFrame(scroll_container, fg_color="#0f172a", corner_radius=8)
            track_card.pack(fill="x", pady=6, padx=4)

            # Track title
            track_title = ctk.CTkLabel(
                track_card,
                text=f"{STEM_NAMES[i]} Position",
                font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                text_color="#f1f5f9"
            )
            track_title.grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(10, 4))

            # 1. Azimuth Slider Row
            ctk.CTkLabel(
                track_card,
                text="Azimuth",
                font=ctk.CTkFont(family="Segoe UI", size=12),
                text_color="#cbd5e1"
            ).grid(row=1, column=0, sticky="w", padx=14, pady=4)

            az_slider = ctk.CTkSlider(
                track_card,
                from_=-180.0,
                to=180.0,
                button_color=self.COLOR_ACCENT,
                button_hover_color="#9333ea",
                progress_color=self.COLOR_ACCENT,
                fg_color="#334155",
                command=lambda val, idx=i: self.update_azimuth(idx, val)
            )
            az_slider.set(0.0)
            az_slider.grid(row=1, column=1, sticky="ew", padx=10, pady=4)
            self.azimuth_sliders.append(az_slider)

            az_lbl = ctk.CTkLabel(
                track_card,
                text="0° (Front)",
                font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                text_color=self.COLOR_ACCENT,
                width=100,
                anchor="e"
            )
            az_lbl.grid(row=1, column=2, sticky="e", padx=14, pady=4)
            self.azimuth_val_labels.append(az_lbl)

            # 2. Elevation Slider Row
            ctk.CTkLabel(
                track_card,
                text="Elevation",
                font=ctk.CTkFont(family="Segoe UI", size=12),
                text_color="#cbd5e1"
            ).grid(row=2, column=0, sticky="w", padx=14, pady=4)

            el_slider = ctk.CTkSlider(
                track_card,
                from_=-90.0,
                to=90.0,
                button_color=self.COLOR_ACCENT,
                button_hover_color="#9333ea",
                progress_color=self.COLOR_ACCENT,
                fg_color="#334155",
                command=lambda val, idx=i: self.update_elevation(idx, val)
            )
            el_slider.set(0.0)
            el_slider.grid(row=2, column=1, sticky="ew", padx=10, pady=4)
            self.elevation_sliders.append(el_slider)

            el_lbl = ctk.CTkLabel(
                track_card,
                text="0° (Level)",
                font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                text_color=self.COLOR_ACCENT,
                width=100,
                anchor="e"
            )
            el_lbl.grid(row=2, column=2, sticky="e", padx=14, pady=4)
            self.elevation_val_labels.append(el_lbl)

            # 3. Distance Slider Row
            ctk.CTkLabel(
                track_card,
                text="Distance",
                font=ctk.CTkFont(family="Segoe UI", size=12),
                text_color="#cbd5e1"
            ).grid(row=3, column=0, sticky="w", padx=14, pady=4)

            dist_slider = ctk.CTkSlider(
                track_card,
                from_=0.1,
                to=10.0,
                button_color=self.COLOR_ACCENT,
                button_hover_color="#9333ea",
                progress_color=self.COLOR_ACCENT,
                fg_color="#334155",
                command=lambda val, idx=i: self.update_distance(idx, val)
            )
            dist_slider.set(1.0)
            dist_slider.grid(row=3, column=1, sticky="ew", padx=10, pady=(4, 12))
            self.distance_sliders.append(dist_slider)

            dist_lbl = ctk.CTkLabel(
                track_card,
                text="1.0m",
                font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                text_color=self.COLOR_ACCENT,
                width=100,
                anchor="e"
            )
            dist_lbl.grid(row=3, column=2, sticky="e", padx=14, pady=(4, 12))
            self.distance_val_labels.append(dist_lbl)

            # Configure grid resizing inside track card
            track_card.grid_columnconfigure(1, weight=1)

        # ── Status Bar ──────────────────────────────────────────────────
        self.status_lbl = ctk.CTkLabel(
            self,
            text="Ready. Select a song file or load 4 manual stems.",
            font=ctk.CTkFont(family="Segoe UI", size=12, slant="italic"),
            text_color="#64748b"
        )
        self.status_lbl.pack(side="bottom", pady=8)

    # ══════════════════════════════════════════════════════════════════════
    #  KEMAR CIPIC HRTF Dataset Loading
    # ══════════════════════════════════════════════════════════════════════
    def load_kemar_async(self):
        self._set_controls_state("disabled")
        threading.Thread(target=self._load_kemar_thread, daemon=True).start()

    def _load_kemar_thread(self):
        url = "https://raw.githubusercontent.com/amini-allight/cipic-hrtf-database/master/standard_hrir_database/subject_021/hrir_final.mat"
        dest_dir = os.path.join(_REPO_ROOT, "data")
        dest_path = os.path.join(dest_dir, "subject_021.mat")
        
        self.after(0, lambda: self.status_lbl.configure(
            text="Checking KEMAR HRTF dataset...",
            text_color=self.COLOR_WARNING
        ))

        try:
            if not os.path.exists(dest_path):
                self.after(0, lambda: self.status_lbl.configure(
                    text="Downloading KEMAR HRTF dataset (~5.5 MB)...",
                    text_color=self.COLOR_WARNING
                ))
                os.makedirs(dest_dir, exist_ok=True)
                
                # Download with progress updates
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response:
                    total_size = int(response.info().get('Content-Length', 0))
                    downloaded = 0
                    block_size = 8192
                    with open(dest_path, 'wb') as f:
                        while True:
                            buffer = response.read(block_size)
                            if not buffer:
                                break
                            downloaded += len(buffer)
                            f.write(buffer)
                            if total_size:
                                percent = int(downloaded * 100 / total_size)
                                self.after(0, lambda p=percent: self.status_lbl.configure(
                                    text=f"Downloading KEMAR HRTF dataset: {p}%...",
                                    text_color=self.COLOR_WARNING
                                ))

            self.after(0, lambda: self.status_lbl.configure(
                text="Loading KEMAR dataset standard matrices...",
                text_color=self.COLOR_WARNING
            ))

            mat = scipy.io.loadmat(dest_path)
            # CIPIC matrices shape is (25, 50, 200)
            hrir_l = mat['hrir_l']
            hrir_r = mat['hrir_r']
            
            # Check for resampling if target rate is not 44.1 kHz
            # (CIPIC original is 44.1 kHz, length 200)
            target_sr = 44100
            
            self.hrir_l = np.ascontiguousarray(hrir_l, dtype=np.float64)
            self.hrir_r = np.ascontiguousarray(hrir_r, dtype=np.float64)

            # Load into active processors
            with self.lock:
                for proc in self.dsp_processors:
                    proc.load_hrir_database(self.hrir_l, self.hrir_r)

            self.after(0, self.on_kemar_loaded)

        except Exception as e:
            print(f"[Error] Failed to load KEMAR CIPIC database: {e}")
            # Fallback to bypass / dummy HRIRs (Identity Impulse Response)
            dummy_l = np.zeros((25, 50, 200), dtype=np.float64)
            dummy_r = np.zeros((25, 50, 200), dtype=np.float64)
            dummy_l[:, :, 0] = 1.0
            dummy_r[:, :, 0] = 1.0

            self.hrir_l = dummy_l
            self.hrir_r = dummy_r

            with self.lock:
                for proc in self.dsp_processors:
                    proc.load_hrir_database(self.hrir_l, self.hrir_r)

            self.after(0, lambda: self.status_lbl.configure(
                text="KEMAR database error! Falling back to standard bypass panning.",
                text_color=self.COLOR_DANGER
            ))
            self.after(2000, self.on_kemar_loaded)

    def on_kemar_loaded(self):
        self._set_controls_state("normal")
        # Keep play/stop disabled if no tracks are loaded yet
        if not self.tracks:
            self.play_btn.configure(state="disabled")
            self.stop_btn.configure(state="disabled")
            self.status_lbl.configure(text="Ready. Select a song file or load 4 manual stems.", text_color="#64748b")
        else:
            self.status_lbl.configure(text="Mixer ready. Click Play.", text_color=self.COLOR_SUCCESS)

    # ══════════════════════════════════════════════════════════════════════
    #  Demucs Separation Workflow
    # ══════════════════════════════════════════════════════════════════════
    def start_demucs_separation(self):
        input_path = filedialog.askopenfilename(
            title="Select a song to separate",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.aiff"),
                ("All files", "*.*"),
            ],
        )
        if not input_path:
            return

        self._set_controls_state("disabled")
        self.status_lbl.configure(
            text="Initialising Demucs… (first run downloads model ~300 MB)",
            text_color=self.COLOR_WARNING
        )
        self.update()

        def _run():
            try:
                if self._demucs_tmpdir and os.path.isdir(self._demucs_tmpdir):
                    shutil.rmtree(self._demucs_tmpdir, ignore_errors=True)

                self._demucs_tmpdir = tempfile.mkdtemp(prefix="spatial_demucs_")

                def _progress(msg):
                    self.after(0, lambda m=msg: self.status_lbl.configure(
                        text=m, text_color=self.COLOR_WARNING
                    ))

                stem_paths = separate_with_demucs(
                    input_path, self._demucs_tmpdir, progress_cb=_progress
                )

                song_name = os.path.basename(input_path)
                self.after(0, lambda: self._on_demucs_done(stem_paths, song_name))

            except Exception as exc:
                self.after(0, lambda e=exc: self._on_demucs_error(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _on_demucs_done(self, stem_paths, song_name):
        display_names = [f"{song_name} [Demucs]" for _ in range(4)]
        self._activate_tracks(stem_paths, display_names)

    def _on_demucs_error(self, error_msg):
        self._set_controls_state("normal")
        self.status_lbl.configure(text="Demucs separation failed!", text_color=self.COLOR_DANGER)
        messagebox.showerror("Demucs Error", f"Stem separation failed:\n\n{error_msg}")

    # ══════════════════════════════════════════════════════════════════════
    #  Manual Loading Workflow
    # ══════════════════════════════════════════════════════════════════════
    def load_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Select exactly 4 audio tracks",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
                ("WAV files", "*.wav"),
            ],
        )
        if len(file_paths) != 4:
            messagebox.showwarning("Warning", "Please select exactly 4 audio files.")
            return

        display_names = [os.path.basename(p) for p in file_paths]
        self._activate_tracks(list(file_paths), display_names)

    # ══════════════════════════════════════════════════════════════════════
    #  Core Audio Loading & Stream Initialization
    # ══════════════════════════════════════════════════════════════════════
    def _activate_tracks(self, file_paths, display_names):
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            with self.lock:
                self.is_playing = False
                self.current_frame = 0

            self.status_lbl.configure(text="Loading stems into mixer…", text_color=self.COLOR_ACCENT)
            self.update()

            # Load via pydub and standardise to mono 44.1kHz
            self.tracks = [
                seg.set_frame_rate(44100).set_channels(1)
                for seg in load_audio_tracks(file_paths)
            ]

            # Convert to NumPy float32 arrays normalized to [-1.0, 1.0]
            self.track_arrays = []
            for track in self.tracks:
                samples = np.array(track.get_array_of_samples(), dtype=np.float32)
                max_val = float(2 ** (8 * track.sample_width - 1))
                samples /= max_val
                self.track_arrays.append(samples)

            # Pad tracks to the same length
            max_len = max(len(a) for a in self.track_arrays)
            self.track_arrays = [
                np.pad(a, (0, max_len - len(a)))
                for a in self.track_arrays
            ]
            self.total_frames = max_len

            # Reset C++ Binaural Processors
            for proc in self.dsp_processors:
                proc.reset()

            # Apply initial position parameters to DSP
            for idx, pos in enumerate(self.track_positions):
                self.dsp_processors[idx].set_position(
                    pos["azimuth"], pos["elevation"], pos["distance"]
                )

            # Initialize sounddevice Output Stream
            self.stream = sd.OutputStream(
                samplerate=44100,
                channels=2,
                callback=self.audio_callback,
                blocksize=1024
            )
            self.stream.start()

            # Update Labels
            for i, name in enumerate(display_names):
                self.track_labels[i].configure(
                    text=f"●  {STEM_NAMES[i]}: {name}",
                    text_color="#f8fafc"
                )

            self.progress_slider.configure(to=self.total_frames, state="normal")
            self.progress_slider.set(0)
            
            self.play_btn.configure(state="normal", text="▶   Play", fg_color=self.COLOR_SUCCESS)
            self.stop_btn.configure(state="normal")
            self._set_controls_state("normal")
            
            self.status_lbl.configure(text="Mixer ready. Click Play.", text_color=self.COLOR_SUCCESS)

            # Kick off UI update loop
            self.update_ui_loop()

        except Exception as e:
            self._set_controls_state("normal")
            self.status_lbl.configure(text="Failed to load tracks.", text_color=self.COLOR_DANGER)
            messagebox.showerror("Mixer Error", f"Failed to activate tracks:\n\n{e}")

    # ══════════════════════════════════════════════════════════════════════
    #  Binaural Mixing Callback (runs on high-priority audio thread)
    # ══════════════════════════════════════════════════════════════════════
    def audio_callback(self, outdata, frames, time_info, status):
        with self.lock:
            if not self.is_playing or self.current_frame >= self.total_frames:
                outdata.fill(0)
                if self.is_playing and self.current_frame >= self.total_frames:
                    self.is_playing = False
                    self.after(0, self.on_playback_finished)
                return

            chunk_size = min(frames, self.total_frames - self.current_frame)
            mixed = np.zeros((frames, 2), dtype=np.float32)
            vol = self.master_volume

            for idx, arr in enumerate(self.track_arrays):
                chunk = arr[self.current_frame : self.current_frame + chunk_size]
                if len(chunk) < frames:
                    # Pad tail chunk with zeros if necessary
                    chunk = np.pad(chunk, (0, frames - len(chunk)))

                # Process the mono chunk using our high-performance C++ DSP
                out_L, out_R = self.dsp_processors[idx].process(chunk)
                
                mixed[:, 0] += out_L * vol
                mixed[:, 1] += out_R * vol

            outdata[:] = mixed
            self.current_frame += chunk_size

    # ══════════════════════════════════════════════════════════════════════
    #  UI Refresh Loop
    # ══════════════════════════════════════════════════════════════════════
    def update_ui_loop(self):
        if not self.tracks:
            return

        with self.lock:
            curr = self.current_frame
            total = self.total_frames

        if not self.is_seeking:
            self.progress_slider.set(curr)

        def fmt(sec):
            return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"

        self.time_label.configure(
            text=f"{fmt(curr / 44100)} / {fmt(total / 44100)}"
        )

        self.after(100, self.update_ui_loop)

    # ══════════════════════════════════════════════════════════════════════
    #  UI Parameter Callback Handlers
    # ══════════════════════════════════════════════════════════════════════
    def update_azimuth(self, idx, val):
        angle = float(val)
        with self.lock:
            self.track_positions[idx]["azimuth"] = angle
            self.dsp_processors[idx].set_position(
                angle,
                self.track_positions[idx]["elevation"],
                self.track_positions[idx]["distance"]
            )

        # Label formatting
        if abs(angle) < 0.5:
            txt = "0° (Front)"
        elif angle > 0:
            txt = f"{int(angle)}° (Right)"
        else:
            txt = f"{int(abs(angle))}° (Left)"
        self.azimuth_val_labels[idx].configure(text=txt)

    def update_elevation(self, idx, val):
        angle = float(val)
        with self.lock:
            self.track_positions[idx]["elevation"] = angle
            self.dsp_processors[idx].set_position(
                self.track_positions[idx]["azimuth"],
                angle,
                self.track_positions[idx]["distance"]
            )

        if abs(angle) < 0.5:
            txt = "0° (Level)"
        elif angle > 0:
            txt = f"+{int(angle)}° (Up)"
        else:
            txt = f"-{int(abs(angle))}° (Down)"
        self.elevation_val_labels[idx].configure(text=txt)

    def update_distance(self, idx, val):
        dist = float(val)
        with self.lock:
            self.track_positions[idx]["distance"] = dist
            self.dsp_processors[idx].set_position(
                self.track_positions[idx]["azimuth"],
                self.track_positions[idx]["elevation"],
                dist
            )

        self.distance_val_labels[idx].configure(text=f"{dist:.1f}m")

    def update_volume(self, val):
        with self.lock:
            self.master_volume = float(val) / 100.0

    # ══════════════════════════════════════════════════════════════════════
    #  Transport Control Handlers
    # ══════════════════════════════════════════════════════════════════════
    def on_slider_seek(self, val):
        # Allow updating frame position live if dragging
        if not self.is_seeking:
            with self.lock:
                self.current_frame = min(int(val), self.total_frames)

    def on_seek_start(self, event):
        self.is_seeking = True

    def on_seek_end(self, event):
        val = self.progress_slider.get()
        with self.lock:
            self.current_frame = min(int(val), self.total_frames)
        self.is_seeking = False

    def toggle_play(self):
        if not self.tracks:
            return
        with self.lock:
            self.is_playing = not self.is_playing
            playing = self.is_playing

        if playing:
            self.play_btn.configure(text="⏸   Pause", fg_color=self.COLOR_WARNING, hover_color="#d97706")
            self.status_lbl.configure(text="Playing 3D Binaural Spatial Mix…", text_color=self.COLOR_SUCCESS)
        else:
            self.play_btn.configure(text="▶   Play", fg_color=self.COLOR_SUCCESS, hover_color="#059669")
            self.status_lbl.configure(text="Playback paused.", text_color="#64748b")

    def stop_playback(self):
        if not self.tracks:
            return
        with self.lock:
            self.is_playing = False
            self.current_frame = 0
            # Reset DSP filter states
            for proc in self.dsp_processors:
                proc.reset()
                
        self.play_btn.configure(text="▶   Play", fg_color=self.COLOR_SUCCESS, hover_color="#059669")
        self.progress_slider.set(0)
        self.status_lbl.configure(text="Playback stopped.", text_color="#64748b")

    def on_playback_finished(self):
        self.play_btn.configure(text="▶   Play", fg_color=self.COLOR_SUCCESS, hover_color="#059669")
        self.status_lbl.configure(text="Playback finished.", text_color="#64748b")

    # ══════════════════════════════════════════════════════════════════════
    #  Offline 3D Binaural Mix & Export
    # ══════════════════════════════════════════════════════════════════════
    def mix_audio(self):
        if len(self.tracks) != 4:
            messagebox.showwarning("Warning", "Please load or separate stems first.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save 3D Mixed Audio As",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
        )
        if not output_path:
            return

        try:
            self.status_lbl.configure(text="Rendering 3D Binaural Mix offline…", text_color=self.COLOR_ACCENT)
            self.update()

            # Render offline
            block_size = 1024
            total_samples = self.total_frames
            
            # Temporary DSP processors to keep real-time play states clean
            render_processors = [binaural_dsp.BinauralProcessor(44100.0) for _ in range(4)]
            with self.lock:
                for idx, pos in enumerate(self.track_positions):
                    if self.hrir_l is not None and self.hrir_r is not None:
                        render_processors[idx].load_hrir_database(self.hrir_l, self.hrir_r)
                    render_processors[idx].set_position(
                        pos["azimuth"], pos["elevation"], pos["distance"]
                    )

            output_L = np.zeros(total_samples, dtype=np.float32)
            output_R = np.zeros(total_samples, dtype=np.float32)

            for idx, arr in enumerate(self.track_arrays):
                cursor = 0
                while cursor < total_samples:
                    chunk = arr[cursor : cursor + block_size]
                    if len(chunk) < block_size:
                        chunk = np.pad(chunk, (0, block_size - len(chunk)))
                    
                    out_L, out_R = render_processors[idx].process(chunk)
                    
                    chunk_len = min(block_size, total_samples - cursor)
                    output_L[cursor : cursor + chunk_len] += out_L[:chunk_len]
                    output_R[cursor : cursor + chunk_len] += out_R[:chunk_len]
                    
                    cursor += block_size

            # Interleave L and R channels
            stereo_mix = np.vstack((output_L, output_R)).T  # (N, 2)
            
            # Normalize to avoid clipping
            max_peak = np.max(np.abs(stereo_mix))
            if max_peak > 1.0:
                stereo_mix /= max_peak
            
            # Convert back to standard 16-bit PCM integer samples for pydub
            int_samples = (stereo_mix * 32767.0).astype(np.int16)

            # Export stereo AudioSegment
            mixed_segment = AudioSegment(
                int_samples.tobytes(),
                frame_rate=44100,
                sample_width=2,
                channels=2
            )
            mixed_segment.export(output_path, format="wav")

            self.status_lbl.configure(
                text=f"Exported successfully to {os.path.basename(output_path)}",
                text_color=self.COLOR_SUCCESS
            )
            messagebox.showinfo("Export Successful", f"3D Spatial Binaural Mix saved to:\n{output_path}")

        except Exception as e:
            self.status_lbl.configure(text="Export failed!", text_color=self.COLOR_DANGER)
            messagebox.showerror("Export Error", f"Failed to export:\n\n{e}")

    # ══════════════════════════════════════════════════════════════════════
    #  Controls State Helper & Clean Exit
    # ══════════════════════════════════════════════════════════════════════
    def _set_controls_state(self, state):
        for widget in (
            self.demucs_btn,
            self.load_btn,
            self.play_btn,
            self.stop_btn,
            self.export_btn,
        ):
            widget.configure(state=state)

    def on_closing(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        if self._demucs_tmpdir and os.path.isdir(self._demucs_tmpdir):
            import shutil
            shutil.rmtree(self._demucs_tmpdir, ignore_errors=True)
        self.destroy()


if __name__ == "__main__":
    app = SpatialAudioMixerApp()
    app.mainloop()
