# 3D Spatial Audio Mixer

A project containing dual implementations of a 3D Spatial Audio Mixer:
1. A **Python GUI** application using `tkinter`, `pydub`, `numpy`, `sounddevice`, and **Demucs** for AI-powered stem separation.
2. A **C GUI** application using `GTK 3`, `libsndfile`, and `PortAudio`.

---

## Directory Layout

The repository is structured as follows:

```text
spatial-audio-compiler/
├── .gitignore              # Git ignore rules for caches and binaries
├── Makefile                # Build configuration for the C application
├── README.md               # Documentation (this file)
├── audio/                  # Input and output audio tracks (.wav files)
│   ├── bass.wav
│   ├── drums.wav
│   ├── instrumental.wav
│   ├── vocals.wav
│   └── output.wav
├── bin/                    # Compiled/executable binaries
│   └── spatial_mixer       # Pre-compiled C mixer executable (Linux ELF)
└── src/                    # Source files
    ├── spatial.py          # Python spatial mixer
    └── spatial_mixer.c     # C spatial mixer source
```

---

## 1. Python Spatial Audio Mixer

The Python version provides a real-time **3D Binaural Spatial Audio Mixer** with a premium dark-themed interface built using `customtkinter`. It features **AI-powered stem separation** via [Demucs](https://github.com/facebookresearch/demucs) and a high-performance **C++ DSP Engine** (`binaural_dsp.pyd`) for real-time acoustic spatialization.

### Features

- 🎵 **Demucs stem separation** — drop in any song (WAV, MP3, FLAC, OGG, M4A…) and the AI splits it into 4 stems
- 🧠 **C++ Measured KEMAR HRTF Engine** — high-performance 3D spatialization using pybind11:
  - **Measured HRTF Convolution** — real-time time-domain FIR convolution (200 taps) using the CIPIC Subject 021 dataset (KEMAR dummy head).
  - **Interaural-Polar Mapping** — converts standard spherical angles (Azimuth, Elevation) to interaural coordinates ($\theta$, $\phi$) to index standard databases.
  - **Bilinear Grid Interpolation** — interpolates filter coefficients across the 25 azimuth and 50 elevation CIPIC grid points.
  - **Anti-click protection** — sample-by-sample linear crossfading of impulse responses over the processing block.
  - **Distance Damping** — $1/d$ volume drop-off and distance-dependent air absorption (low-pass filtering).
- 🎚️ **CustomTkinter GUI** — sleek modern dark mode with Azimuth, Elevation, and Distance controls for each stem.
- ⏯️ **Play / Pause / Stop** transport controls with an interactive progress bar and timestamps.
- 🔉 **Master Volume** slider.
- 💾 **Mix & Export** — render your 3D spatial KEMAR HRTF mix offline into a stereo WAV file.

### Prerequisites

1. Ensure you have Python 3.9+ installed.
2. Install a C++ compiler (MSVC 2019+ on Windows, GCC/Clang on Linux/macOS).
3. Install required Python packages:
   ```bash
   pip install pydub numpy sounddevice demucs customtkinter scipy pybind11
   ```
   *Note: For Python 3.13 or newer, you must also install `audioop-lts` because the standard library `audioop` module was removed:*
   ```bash
   pip install audioop-lts
   ```

> **FFmpeg Note:** FFmpeg must be installed on your system. On Windows, if you install it using `winget install Gyan.FFmpeg`, the app will automatically discover and load the binary path at startup.
>
> **KEMAR Dataset Auto-Download:** The app automatically checks for and downloads the standard CIPIC `subject_021.mat` database file (~4.0 MB) from a public mirror at startup.
>
> **First Demucs run:** The `htdemucs` model weights (~300 MB) are downloaded automatically from Hugging Face on first use.

### Compilation

Before running the app, you **must compile the C++ Binaural DSP extension** (`binaural_dsp`):

Run this command from the repository root:
```bash
python setup.py build_ext --inplace
```

This compiles `src/binaural_dsp.cpp` and creates `binaural_dsp.cpXX-win_amd64.pyd` directly in the project root directory.

### Running the Python Application

Once compiled, run the application from the repository root:

```bash
python src/spatial.py
```

#### Workflow A — Demucs stem separation (recommended)

1. Click **⬡ Separate with Demucs** and select any song file (MP3, WAV, FLAC, M4A, etc.).
2. Watch progress in the terminal. The stems will load automatically when done.
3. Press **▶ Play** to start playback.
4. Move the **Azimuth**, **Elevation**, and **Distance** sliders for each stem to position them in 3D space:
   - **Azimuth**: Position sound left, right, front, or behind you ($-180^\circ$ to $+180^\circ$).
   - **Elevation**: Height cues ($-90^\circ$ below to $+90^\circ$ above).
   - **Distance**: Proximal volume and air damping ($0.1\text{m}$ to $10.0\text{m}$).
5. Click **Mix & Export** to save the spatialized mix as a WAV file.

#### Workflow B — Manual stem loading

1. Click **Load 4 Files Manually** and select exactly 4 audio files.
2. Press **▶ Play** and position them in 3D space.
3. Click **Mix & Export** to save your work.


---

## 2. C Spatial Audio Mixer

The C version uses `GTK 3` for the user interface, `libsndfile` to read WAV files, and `PortAudio` for real-time audio playback.

### Prerequisites

Ensure you have the required development libraries installed:

- **Debian/Ubuntu**:
  ```bash
  sudo apt-get install libgtk-3-dev libsndfile1-dev portaudio19-dev build-essential pkg-config
  ```
- **macOS** (using Homebrew):
  ```bash
  brew install gtk+3 libsndfile portaudio pkg-config
  ```

### Compiling the C Application

Compile using the provided Makefile:

```bash
make
```

This will compile the binary into `bin/spatial_mixer`.

### Running the C Application

Run the compiled executable, specifying the output file followed by 4 input `.wav` files:

```bash
./bin/spatial_mixer audio/output.wav audio/vocals.wav audio/drums.wav audio/bass.wav audio/instrumental.wav
```

Adjust the sliders in the GTK window to dynamically change the left/right panning of each track in real-time.
