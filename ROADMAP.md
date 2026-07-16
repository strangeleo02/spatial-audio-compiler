# 3D Spatial Audio Mixer — Development Roadmap

This document outlines the future milestones for transforming the 3D Spatial Audio Mixer from a desktop slider-based GUI into an interactive, visual 3D web application.

---

## Vision: The 3D Auditorium Mixer
Instead of configuring azimuth, elevation, and distance via individual numerical sliders, the user will be presented with a virtual **3D Auditorium** (rendered in the browser using `Three.js`). 

A listener's head sits in the center of the stage, and the 4 stems (Vocals, Drums, Bass, Other) are represented as drag-and-drop spheres. Positioning is updated in real-time by physically picking and placing the stems around the virtual stage.

```
                  [ Virtual Stage ]
                
                       Other
              Vocals     ●
                ●
                         Listener Head
                              👂
                            👃  👂
                      
                      ●           ●
                    Drums        Bass
```

---

## Development Milestones

### Phase 1: Web-Based 3D Stage (`Three.js`)
* **Interactive 3D Stage Canvas**:
  - Build a responsive 3D viewport of an auditorium using `Three.js` (WebGL).
  - Render a center mesh representing the listener's head (camera or reference point).
  - Render 4 distinct, color-coded interactive spheres for the stems.
* **Drag-and-Drop Raycasting**:
  - Implement a `PointerDragControls` mechanism to pick up, move, and place source spheres.
  - Constrain movement boundary to a realistic room model (e.g., maximum radius of 10m).
* **Live Coordinate Mapping**:
  - Automatically calculate 3D Cartesian coordinates $(x, y, z)$ of each sphere relative to the listener.
  - Convert coordinates to Spherical coordinates: **Azimuth** ($\theta$), **Elevation** ($\phi$), and **Distance** ($d$) in real-time.

### Phase 2: WebAssembly (WASM) Audio Engine
* **Emscripten WASM Port**:
  - Compile the C++ binaural spatial DSP engine (`src/binaural_dsp.cpp`) into a WebAssembly module using Emscripten.
  - Minimize JS-to-WASM overhead by passing shared raw audio frame buffers.
* **Web Audio API Integration**:
  - Run the WASM Binaural DSP engine inside a low-latency **AudioWorkletProcessor** thread.
  - Stream real-time parameters (Azimuth, Elevation, Distance) from the `Three.js` main thread directly to the AudioWorklet.
* **Server-Client Stem Architecture**:
  - Retain the Python backend with the Demucs separation pipeline.
  - User uploads a song to the local web server → server runs Demucs → streams separated stem MP3/WAV chunks to the browser client.

### Phase 3: Advanced Room Acoustics
* **Room Reflection Cues**:
  - Implement simple virtual room boundaries (walls, ceiling, floor) in the C++ engine.
  - Calculate early reflections (first-order image source model) to add realistic room acoustics and room presence.
* **Late Reverb Simulation**:
  - Integrate a feedback delay network (FDN) or convolution reverb to simulate late reverberation decay based on the size of the auditorium.
* **Visual Directional Wave Propagation**:
  - Render subtle expanding rings or particle waves emanating from the source spheres to visually represent active sound output and frequency amplitudes.
