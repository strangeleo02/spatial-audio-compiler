// ─────────────────────────────────────────────────────────────────────────────
//  3D Binaural Spatial Audio Mixer — Frontend Application Logic
// ─────────────────────────────────────────────────────────────────────────────

// Global State
let audioCtx = null;
let masterGain = null;
let isPlaying = false;
let stemsData = null;  // holds URLs of separated stems
let audioBuffers = {}; // holds loaded AudioBuffers
let sourceNodes = {};  // holds active AudioBufferSourceNodes
let pannerNodes = {};  // holds active PannerNodes
let startTime = 0;     // context time when playback started
let pausedTime = 0;    // offset time if paused (seconds)
let duration = 0;      // total length of audio tracks in seconds
let progressInterval = null;

// Three.js State
let scene, camera, renderer;
let listenerMesh;
let stemMeshes = {};
let activeStemsList = ['vocals', 'drums', 'bass', 'other'];

// Drag-and-drop state
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let planeXZ = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let intersectionPoint = new THREE.Vector3();
let selectedStem = null;
let isDragging = false;
let shiftPressed = false;

// Color Coding matching CSS
const STEM_COLORS = {
    vocals: 0xf43f5e, // Pink/Red
    drums: 0xf59e0b,  // Orange/Yellow
    bass: 0x10b981,   // Green
    other: 0x3b82f6   // Blue
};

// Initial stem positions in Three.js coordinates (X, Y, Z)
// Listener is at (0, 0, 0)
const INITIAL_POSITIONS = {
    vocals: { x: -3, y: 0, z: 3 },   // front-left-ish
    drums:  { x: 0,  y: 0, z: -4 },  // behind center
    bass:   { x: -2, y: 0, z: -3 },  // behind-left
    other:  { x: 3,  y: 0, z: 2 }    // front-right-ish
};

// ─────────────────────────────────────────────────────────────────────────────
//  Web Audio Initialization
// ─────────────────────────────────────────────────────────────────────────────

function initAudio() {
    if (audioCtx) return;
    
    // Create AudioContext (supports both Standard and Legacy browser implementations)
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass();
    
    // Create Master Gain Node
    masterGain = audioCtx.createGain();
    masterGain.gain.value = 0.8; // default volume matches master volume slider (80%)
    masterGain.connect(audioCtx.destination);
    
    // Configure Listener orientation (Facing forward along -Z, Up along +Y)
    const listener = audioCtx.listener;
    if (listener.forwardX) {
        // Modern Web Audio API
        listener.positionX.value = 0;
        listener.positionY.value = 0;
        listener.positionZ.value = 0;
        listener.forwardX.value = 0;
        listener.forwardY.value = 0;
        listener.forwardZ.value = -1;
        listener.upX.value = 0;
        listener.upY.value = 1;
        listener.upZ.value = 0;
    } else {
        // Legacy Web Audio API
        listener.setPosition(0, 0, 0);
        listener.setOrientation(0, 0, -1, 0, 1, 0);
    }
}

async function loadAudioBuffer(url) {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    return await audioCtx.decodeAudioData(arrayBuffer);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Three.js 3D Visualizer Setup
// ─────────────────────────────────────────────────────────────────────────────

function initThree() {
    const container = document.getElementById("canvas-container");
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // 1. Create Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a); // dark gray-blue background
    
    // 2. Create Camera
    camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.set(0, 8, 12);
    camera.lookAt(0, 0, 0);
    
    // 3. Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);
    
    // 4. Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 15, 7);
    dirLight.castShadow = true;
    scene.add(dirLight);
    
    // 5. Grid helper representing stage floor
    const gridHelper = new THREE.GridHelper(20, 20, 0x7c3aed, 0x334155);
    gridHelper.position.y = -0.01;
    scene.add(gridHelper);
    
    // 6. Draw Auditorium Stage Outline (arc representation)
    const stageGeo = new THREE.RingGeometry(0.1, 10, 32, 1, 0, Math.PI * 2);
    const stageMat = new THREE.MeshBasicMaterial({ color: 0x1e293b, side: THREE.DoubleSide, opacity: 0.15, transparent: true });
    const stageMesh = new THREE.Mesh(stageGeo, stageMat);
    stageMesh.rotation.x = Math.PI / 2;
    scene.add(stageMesh);

    // 7. Listener Mesh in center (Head + Nose/Direction pointer)
    const listenerGroup = new THREE.Group();
    
    // Head sphere
    const headGeo = new THREE.SphereGeometry(0.6, 32, 32);
    const headMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.5 });
    const headMesh = new THREE.Mesh(headGeo, headMat);
    listenerGroup.add(headMesh);
    
    // Nose pointer to show facing direction (-Z)
    const noseGeo = new THREE.ConeGeometry(0.15, 0.4, 4);
    const noseMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const noseMesh = new THREE.Mesh(noseGeo, noseMat);
    noseMesh.position.set(0, 0, -0.7);
    noseMesh.rotation.x = -Math.PI / 2;
    listenerGroup.add(noseMesh);

    scene.add(listenerGroup);
    
    // 8. Stem meshes (Draggable spheres)
    activeStemsList.forEach(stem => {
        const group = new THREE.Group();
        const pos = INITIAL_POSITIONS[stem];
        group.position.set(pos.x, pos.y, pos.z);
        group.name = stem;
        
        // Sphere represent source
        const sphereGeo = new THREE.SphereGeometry(0.4, 32, 32);
        const sphereMat = new THREE.MeshStandardMaterial({
            color: STEM_COLORS[stem],
            roughness: 0.2,
            metalness: 0.1,
            emissive: STEM_COLORS[stem],
            emissiveIntensity: 0.15
        });
        const sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
        sphereMesh.castShadow = true;
        group.add(sphereMesh);
        
        // Little light indicator above it
        const beaconGeo = new THREE.BoxGeometry(0.05, 0.2, 0.05);
        const beaconMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const beaconMesh = new THREE.Mesh(beaconGeo, beaconMat);
        beaconMesh.position.y = 0.55;
        group.add(beaconMesh);
        
        scene.add(group);
        stemMeshes[stem] = group;
        
        // Push initial updates to GUI monitors
        updateUIMonitor(stem, pos.x, pos.y, pos.z);
    });

    // 9. Resize handler
    window.addEventListener("resize", () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
    
    // Start Animation/Render loop
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    
    // Render loop animation cues (e.g. pulse active stems slightly when playing)
    if (isPlaying) {
        const time = Date.now() * 0.005;
        activeStemsList.forEach(stem => {
            const mesh = stemMeshes[stem];
            if (mesh) {
                // Subtle breathing animation
                const scale = 1.0 + Math.sin(time + INITIAL_POSITIONS[stem].x) * 0.05;
                mesh.children[0].scale.set(scale, scale, scale);
            }
        });
    }

    renderer.render(scene, camera);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Position Calculations & Monitoring Update
// ─────────────────────────────────────────────────────────────────────────────

function updateUIMonitor(stem, x, y, z) {
    // Web Audio coordinates map directly to Three.js coordinates:
    // Left/Right = X axis
    // Up/Down = Y axis
    // Front/Back = Z axis (Three.js -Z is front, +Z is back)
    
    // Azimuth calculation relative to front (-Z)
    // angle = Math.atan2(X, -Z)
    let azimuthRad = Math.atan2(x, -z);
    let azimuthDeg = Math.round(azimuthRad * 180 / Math.PI);
    
    // Elevation calculation
    // angle = Math.atan2(Y, Math.sqrt(X^2 + Z^2))
    let horizontalDist = Math.sqrt(x*x + z*z);
    let elevationRad = Math.atan2(y, horizontalDist);
    let elevationDeg = Math.round(elevationRad * 180 / Math.PI);
    
    // Distance
    let dist = Math.sqrt(x*x + y*y + z*z).toFixed(1);

    // Update labels
    let azTxt = azimuthDeg === 0 ? "0° (Front)" : (azimuthDeg > 0 ? `${azimuthDeg}° (Right)` : `${Math.abs(azimuthDeg)}° (Left)`);
    let elTxt = elevationDeg === 0 ? "0° (Level)" : (elevationDeg > 0 ? `+${elevationDeg}° (Up)` : `${elevationDeg}° (Down)`);

    document.getElementById(`coord-${stem}-az`).innerText = azTxt;
    document.getElementById(`coord-${stem}-el`).innerText = elTxt;
    document.getElementById(`coord-${stem}-dist`).innerText = `${dist}m`;
}

// Update PannerNode positions live
function updatePannerPosition(stem, x, y, z) {
    const panner = pannerNodes[stem];
    if (!panner) return;
    
    // Set parameters (handles immediate interpolation)
    if (panner.positionX) {
        panner.positionX.value = x;
        panner.positionY.value = y;
        panner.positionZ.value = z;
    } else {
        // Fallback for older browsers
        panner.setPosition(x, y, z);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pointer Drag Controls (Raycasting)
// ─────────────────────────────────────────────────────────────────────────────

const container = document.getElementById("canvas-container");

container.addEventListener("mousedown", onMouseDown, false);
container.addEventListener("mousemove", onMouseMove, false);
container.addEventListener("mouseup", onMouseUp, false);
window.addEventListener("keydown", (e) => { if (e.key === "Shift") shiftPressed = true; });
window.addEventListener("keyup", (e) => { if (e.key === "Shift") shiftPressed = false; });

function getMousePosition(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function onMouseDown(event) {
    getMousePosition(event);
    
    raycaster.setFromCamera(mouse, camera);
    
    // Check if we hit any of our stem mesh groups
    const targets = activeStemsList.map(s => stemMeshes[s]);
    const intersects = raycaster.intersectObjects(targets, true);
    
    if (intersects.length > 0) {
        // Find the top-level group node
        let obj = intersects[0].object;
        while (obj.parent && obj.parent !== scene) {
            obj = obj.parent;
        }
        
        selectedStem = obj.name;
        isDragging = true;
        
        // Highlight active card border slightly
        document.querySelectorAll('.stem-item').forEach(el => el.classList.remove('active'));
        document.querySelector(`.stem-item.${selectedStem}`).classList.add('active');
        
        // Capture initial offset plane intersection
        raycaster.ray.intersectPlane(planeXZ, intersectionPoint);
    }
}

function onMouseMove(event) {
    if (!isDragging || !selectedStem) return;
    
    getMousePosition(event);
    raycaster.setFromCamera(mouse, camera);
    
    const mesh = stemMeshes[selectedStem];
    
    if (shiftPressed) {
        // Drag up/down along Y-axis (Elevation)
        // Set up vertical plane facing camera to intersect ray
        const cameraNormal = new THREE.Vector3();
        camera.getWorldDirection(cameraNormal);
        cameraNormal.y = 0; // lock to horizontal normal
        cameraNormal.normalize();
        
        const planeY = new THREE.Plane(cameraNormal, -mesh.position.dot(cameraNormal));
        if (raycaster.ray.intersectPlane(planeY, intersectionPoint)) {
            // Clamp elevation range to [-5m, 5m] for physical sound elevation
            mesh.position.y = THREE.MathUtils.clamp(intersectionPoint.y, -5, 5);
        }
    } else {
        // Normal drag along Stage Floor X-Z plane
        if (raycaster.ray.intersectPlane(planeXZ, intersectionPoint)) {
            // Distance limit of 10m from center listener
            const distance = Math.sqrt(intersectionPoint.x * intersectionPoint.x + intersectionPoint.z * intersectionPoint.z);
            if (distance <= 10.0) {
                mesh.position.x = intersectionPoint.x;
                mesh.position.z = intersectionPoint.z;
            } else {
                // Project onto boundary edge
                const angle = Math.atan2(intersectionPoint.z, intersectionPoint.x);
                mesh.position.x = Math.cos(angle) * 10.0;
                mesh.position.z = Math.sin(angle) * 10.0;
            }
        }
    }
    
    // Update live coordinates in monitor labels and Audio engine
    updateUIMonitor(selectedStem, mesh.position.x, mesh.position.y, mesh.position.z);
    updatePannerPosition(selectedStem, mesh.position.x, mesh.position.y, mesh.position.z);
}

function onMouseUp() {
    isDragging = false;
    selectedStem = null;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Demucs API Connection & Progress Handler
// ─────────────────────────────────────────────────────────────────────────────

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("audio-file");
const fileLabel = document.getElementById("file-label");
const sepProgress = document.getElementById("separation-progress");
const progressBarFill = document.getElementById("progress-bar-fill");
const progressPercent = document.getElementById("progress-percent");
const progressStatusMsg = document.getElementById("progress-status-msg");
const btnSeparate = document.getElementById("btn-separate");

// Show filename on select
fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        fileLabel.innerHTML = `
            <span class="upload-icon">🎵</span>
            <span class="upload-title">${fileInput.files[0].name}</span>
            <span class="upload-subtitle">Ready to separate</span>
        `;
    }
});

uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (fileInput.files.length === 0) return;
    
    // Initialize Web Audio early due to browser security restrictions on click events
    initAudio();
    
    // Disable inputs
    fileInput.disabled = true;
    btnSeparate.disabled = true;
    
    // Reset and show progress bar
    sepProgress.classList.remove("hidden");
    progressBarFill.style.width = "0%";
    progressPercent.innerText = "0%";
    
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    
    // Start progress simulation (polling would be ideal, but since FastAPI blocks thread
    // for local Demucs separation, we mock an optimistic progress curve and hook completion)
    let simulatedProgress = 0;
    const progressTimer = setInterval(() => {
        if (simulatedProgress < 90) {
            simulatedProgress += (1.0 / (simulatedProgress * 0.15 + 1)) * 3; // slows down as it gets closer
            const roundedVal = Math.round(simulatedProgress);
            progressBarFill.style.width = `${roundedVal}%`;
            progressPercent.innerText = `${roundedVal}%`;
            
            if (roundedVal < 30) {
                progressStatusMsg.innerText = "Loading Demucs model... (weights ~300MB)";
            } else if (roundedVal < 75) {
                progressStatusMsg.innerText = "Running AI neural separation... (watch server console)";
            } else {
                progressStatusMsg.innerText = "Finalizing wav stems rendering...";
            }
        }
    }, 1500);

    try {
        const response = await fetch("/api/separate", {
            method: "POST",
            body: formData
        });
        
        clearInterval(progressTimer);
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Separation failed");
        }
        
        const data = await response.json();
        stemsData = data.stems;
        
        // Update progress bar to 100%
        progressBarFill.style.width = "100%";
        progressPercent.innerText = "100%";
        progressStatusMsg.innerText = "Stem separation complete!";
        
        // Hide progress card after delay
        setTimeout(() => sepProgress.classList.add("hidden"), 3000);
        
        // Download and decode tracks into AudioBuffers
        await loadAllStems();
        
    } catch (err) {
        clearInterval(progressTimer);
        sepProgress.classList.add("hidden");
        fileInput.disabled = false;
        btnSeparate.disabled = false;
        alert(`Error: ${err.message}`);
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  Web Audio playback core
// ─────────────────────────────────────────────────────────────────────────────

async function loadAllStems() {
    if (!stemsData) return;
    
    document.getElementById("status-text").innerText = "Downloading decoded audio stems...";
    document.getElementById("status-dot").classList.add("loading");
    
    try {
        for (const stem of activeStemsList) {
            const url = stemsData[stem];
            document.getElementById("status-text").innerText = `Downloading ${stem} stem...`;
            audioBuffers[stem] = await loadAudioBuffer(url);
        }
        
        // Get longest stem duration
        duration = Math.max(...activeStemsList.map(s => audioBuffers[s].duration));
        
        // Setup UI limits
        document.getElementById("timeline-slider").disabled = false;
        document.getElementById("timeline-slider").max = duration;
        document.getElementById("time-total").innerText = formatTime(duration);
        document.getElementById("time-current").innerText = "00:00";
        
        document.getElementById("btn-play").disabled = false;
        document.getElementById("btn-stop").disabled = false;
        
        document.getElementById("status-text").innerText = "All stems loaded successfully";
        document.getElementById("status-dot").classList.remove("loading");
        
        // Update HTML monitor track names
        activeStemsList.forEach(stem => {
            const filename = fileInput.files[0].name;
            document.getElementById(`file-label`).innerHTML = "Choose audio file";
            fileInput.disabled = false;
            btnSeparate.disabled = false;
            // Update labels
            const monitorLabel = document.querySelector(`.stem-item.${stem} .stem-name`);
            monitorLabel.innerHTML = `${stem.charAt(0).toUpperCase() + stem.slice(1)} <span style="font-size: 10px; font-weight:normal; color:#64748b">(${filename})</span>`;
        });
        
    } catch (err) {
        document.getElementById("status-text").innerText = "Failed to load stems";
        document.getElementById("status-dot").classList.remove("loading");
        alert(`Failed to load audio files: ${err.message}`);
    }
}

// Start playing all source nodes simultaneously
function playStems(offset = 0) {
    initAudio();
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    
    // Stop any currently running sources
    stopActiveSourceNodes();
    
    startTime = audioCtx.currentTime - offset;
    pausedTime = offset;
    
    activeStemsList.forEach(stem => {
        const buffer = audioBuffers[stem];
        if (!buffer) return;
        
        // 1. Create Buffer Source
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        
        // 2. Create PannerNode for 3D HRTF
        const panner = audioCtx.createPanner();
        panner.panningModel = 'HRTF';
        panner.distanceModel = 'inverse';
        panner.refDistance = 1.0;
        panner.maxDistance = 10.0;
        panner.rolloffFactor = 1.0;
        
        // Apply position from Three.js scene
        const mesh = stemMeshes[stem];
        if (panner.positionX) {
            panner.positionX.value = mesh.position.x;
            panner.positionY.value = mesh.position.y;
            panner.positionZ.value = mesh.position.z;
        } else {
            panner.setPosition(mesh.position.x, mesh.position.y, mesh.position.z);
        }
        
        // 3. Connect routing chain
        source.connect(panner);
        panner.connect(masterGain);
        
        // Start buffer
        source.start(0, offset);
        
        sourceNodes[stem] = source;
        pannerNodes[stem] = panner;
    });
    
    isPlaying = true;
    document.getElementById("btn-play").innerText = "⏸ Pause";
    
    // Timeline refresh interval
    progressInterval = setInterval(updateTimeline, 200);
}

function pauseStems() {
    if (!isPlaying) return;
    
    pausedTime = audioCtx.currentTime - startTime;
    stopActiveSourceNodes();
    
    isPlaying = false;
    document.getElementById("btn-play").innerText = "▶ Play";
    clearInterval(progressInterval);
}

function stopStems() {
    pausedTime = 0;
    stopActiveSourceNodes();
    
    isPlaying = false;
    document.getElementById("btn-play").innerText = "▶ Play";
    document.getElementById("timeline-slider").value = 0;
    document.getElementById("time-current").innerText = "00:00";
    clearInterval(progressInterval);
}

function stopActiveSourceNodes() {
    activeStemsList.forEach(stem => {
        const source = sourceNodes[stem];
        if (source) {
            try { source.stop(); } catch(e) {}
            delete sourceNodes[stem];
        }
    });
}

function updateTimeline() {
    if (!isPlaying) return;
    
    const elapsed = audioCtx.currentTime - startTime;
    if (elapsed >= duration) {
        stopStems();
        return;
    }
    
    document.getElementById("timeline-slider").value = elapsed;
    document.getElementById("time-current").innerText = formatTime(elapsed);
}

function formatTime(sec) {
    const min = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${min.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// ─────────────────────────────────────────────────────────────────────────────
//  UI Controls Bindings
// ─────────────────────────────────────────────────────────────────────────────

const btnPlay = document.getElementById("btn-play");
const btnStop = document.getElementById("btn-stop");
const volSlider = document.getElementById("volume-slider");
const volPercent = document.getElementById("vol-percent");
const timelineSlider = document.getElementById("timeline-slider");

btnPlay.addEventListener("click", () => {
    if (isPlaying) {
        pauseStems();
    } else {
        playStems(pausedTime);
    }
});

btnStop.addEventListener("click", () => {
    stopStems();
});

volSlider.addEventListener("input", () => {
    const val = volSlider.value;
    volPercent.innerText = `${val}%`;
    if (masterGain) {
        masterGain.gain.value = val / 100;
    }
});

timelineSlider.addEventListener("input", () => {
    const seekTime = parseFloat(timelineSlider.value);
    document.getElementById("time-current").innerText = formatTime(seekTime);
});

timelineSlider.addEventListener("change", () => {
    const seekTime = parseFloat(timelineSlider.value);
    if (isPlaying) {
        playStems(seekTime);
    } else {
        pausedTime = seekTime;
    }
});

// Launch visualizer scene on window load
window.addEventListener("DOMContentLoaded", () => {
    initThree();
});
