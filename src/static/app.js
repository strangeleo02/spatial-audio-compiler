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
let stemGainNodes = {};// holds active stem GainNodes
let startTime = 0;     // context time when playback started
let pausedTime = 0;    // offset time if paused (seconds)
let duration = 0;      // total length of audio tracks in seconds
let progressInterval = null;

// Mixer State (DAW-style)
let stemVolumes = { vocals: 0.8, drums: 0.8, bass: 0.8, other: 0.8 };
let stemMutes = { vocals: false, drums: false, bass: false, other: false };
let stemSolos = { vocals: false, drums: false, bass: false, other: false };

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
    
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass();
    
    // Create Master Gain Node
    masterGain = audioCtx.createGain();
    masterGain.gain.value = 0.8; // default matches master volume slider (80%)
    masterGain.connect(audioCtx.destination);
    
    // Configure Listener orientation (Facing forward along -Z, Up along +Y)
    const listener = audioCtx.listener;
    if (listener.forwardX) {
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
        listener.setPosition(0, 0, 0);
        listener.setOrientation(0, 0, -1, 0, 1, 0);
    }
}

async function loadAudioBuffer(url) {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    return await audioCtx.decodeAudioData(arrayBuffer);
}

// Update calculated effective gains based on Volume, Mute, and Solo states
function updateStemGains() {
    if (!audioCtx) return;
    
    const hasSolo = activeStemsList.some(s => stemSolos[s]);
    
    activeStemsList.forEach(stem => {
        const gainNode = stemGainNodes[stem];
        if (!gainNode) return;
        
        let targetGain = stemVolumes[stem];
        
        if (stemMutes[stem]) {
            targetGain = 0.0;
        } else if (hasSolo && !stemSolos[stem]) {
            targetGain = 0.0;
        }
        
        // Prevent pops/clicks using smooth ramp
        gainNode.gain.setTargetAtTime(targetGain, audioCtx.currentTime, 0.02);
    });
}

// ─────────────────────────────────────────────────────────────────────────────
//  Three.js 3D Visualizer Setup
// ─────────────────────────────────────────────────────────────────────────────

function initThree() {
    const container = document.getElementById("canvas-container");
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);
    
    camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.set(0, 8, 12);
    camera.lookAt(0, 0, 0);
    
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);
    
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 15, 7);
    dirLight.castShadow = true;
    scene.add(dirLight);
    
    const gridHelper = new THREE.GridHelper(20, 20, 0x7c3aed, 0x334155);
    gridHelper.position.y = -0.01;
    scene.add(gridHelper);
    
    const stageGeo = new THREE.RingGeometry(0.1, 10, 32, 1, 0, Math.PI * 2);
    const stageMat = new THREE.MeshBasicMaterial({ color: 0x1e293b, side: THREE.DoubleSide, opacity: 0.15, transparent: true });
    const stageMesh = new THREE.Mesh(stageGeo, stageMat);
    stageMesh.rotation.x = Math.PI / 2;
    scene.add(stageMesh);

    const listenerGroup = new THREE.Group();
    const headGeo = new THREE.SphereGeometry(0.6, 32, 32);
    const headMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.5 });
    const headMesh = new THREE.Mesh(headGeo, headMat);
    listenerGroup.add(headMesh);
    
    const noseGeo = new THREE.ConeGeometry(0.15, 0.4, 4);
    const noseMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const noseMesh = new THREE.Mesh(noseGeo, noseMat);
    noseMesh.position.set(0, 0, -0.7);
    noseMesh.rotation.x = -Math.PI / 2;
    listenerGroup.add(noseMesh);

    scene.add(listenerGroup);
    
    activeStemsList.forEach(stem => {
        const group = new THREE.Group();
        const pos = INITIAL_POSITIONS[stem];
        group.position.set(pos.x, pos.y, pos.z);
        group.name = stem;
        
        // Sphere representing source
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
        
        // Glowing visual 3D sound waves ring
        const waveGeo = new THREE.RingGeometry(0.4, 0.45, 32);
        const waveMat = new THREE.MeshBasicMaterial({
            color: STEM_COLORS[stem],
            side: THREE.DoubleSide,
            opacity: 0.0,
            transparent: true
        });
        const waveMesh = new THREE.Mesh(waveGeo, waveMat);
        waveMesh.rotation.x = Math.PI / 2;
        waveMesh.name = "wave";
        group.add(waveMesh);
        
        // Little light indicator above it
        const beaconGeo = new THREE.BoxGeometry(0.05, 0.2, 0.05);
        const beaconMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const beaconMesh = new THREE.Mesh(beaconGeo, beaconMat);
        beaconMesh.position.y = 0.55;
        group.add(beaconMesh);
        
        scene.add(group);
        stemMeshes[stem] = group;
        
        updateUIMonitor(stem, pos.x, pos.y, pos.z);
    });

    window.addEventListener("resize", () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
    
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    
    const hasSolo = activeStemsList.some(s => stemSolos[s]);

    activeStemsList.forEach(stem => {
        const mesh = stemMeshes[stem];
        if (!mesh) return;

        const wave = mesh.getObjectByName("wave");
        if (!wave) return;

        const isSilent = stemMutes[stem] || (hasSolo && !stemSolos[stem]);

        if (isPlaying && !isSilent) {
            // Pulse source sphere breathing size
            const time = Date.now() * 0.005;
            const scale = 1.0 + Math.sin(time + INITIAL_POSITIONS[stem].x) * 0.05;
            mesh.children[0].scale.set(scale, scale, scale);

            // Expand and fade out the sound wave ring
            wave.scale.addScalar(0.03);
            wave.material.opacity -= 0.015;

            if (wave.material.opacity <= 0) {
                wave.scale.set(1, 1, 1);
                wave.material.opacity = 0.7; // restart loop
            }
        } else {
            // Reset state
            mesh.children[0].scale.set(1, 1, 1);
            wave.scale.set(1, 1, 1);
            wave.material.opacity = 0.0;
        }
    });

    renderer.render(scene, camera);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Position Calculations & Monitoring Update
// ─────────────────────────────────────────────────────────────────────────────

function updateUIMonitor(stem, x, y, z) {
    let azimuthRad = Math.atan2(x, -z);
    let azimuthDeg = Math.round(azimuthRad * 180 / Math.PI);
    
    let horizontalDist = Math.sqrt(x*x + z*z);
    let elevationRad = Math.atan2(y, horizontalDist);
    let elevationDeg = Math.round(elevationRad * 180 / Math.PI);
    
    let dist = Math.sqrt(x*x + y*y + z*z).toFixed(1);

    let azTxt = azimuthDeg === 0 ? "0° (Front)" : (azimuthDeg > 0 ? `${azimuthDeg}° (Right)` : `${Math.abs(azimuthDeg)}° (Left)`);
    let elTxt = elevationDeg === 0 ? "0° (Level)" : (elevationDeg > 0 ? `+${elevationDeg}° (Up)` : `${elevationDeg}° (Down)`);

    document.getElementById(`coord-${stem}-az`).innerText = azTxt;
    document.getElementById(`coord-${stem}-el`).innerText = elTxt;
    document.getElementById(`coord-${stem}-dist`).innerText = `${dist}m`;
}

function updatePannerPosition(stem, x, y, z) {
    const panner = pannerNodes[stem];
    if (!panner) return;
    
    if (panner.positionX) {
        panner.positionX.value = x;
        panner.positionY.value = y;
        panner.positionZ.value = z;
    } else {
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
    
    const targets = activeStemsList.map(s => stemMeshes[s]);
    const intersects = raycaster.intersectObjects(targets, true);
    
    if (intersects.length > 0) {
        let obj = intersects[0].object;
        while (obj.parent && obj.parent !== scene) {
            obj = obj.parent;
        }
        
        selectedStem = obj.name;
        isDragging = true;
        
        document.querySelectorAll('.stem-item').forEach(el => el.classList.remove('active'));
        document.querySelector(`.stem-item.${selectedStem}`).classList.add('active');
        
        raycaster.ray.intersectPlane(planeXZ, intersectionPoint);
    }
}

function onMouseMove(event) {
    if (!isDragging || !selectedStem) return;
    
    getMousePosition(event);
    raycaster.setFromCamera(mouse, camera);
    
    const mesh = stemMeshes[selectedStem];
    
    if (shiftPressed) {
        const cameraNormal = new THREE.Vector3();
        camera.getWorldDirection(cameraNormal);
        cameraNormal.y = 0;
        cameraNormal.normalize();
        
        const planeY = new THREE.Plane(cameraNormal, -mesh.position.dot(cameraNormal));
        if (raycaster.ray.intersectPlane(planeY, intersectionPoint)) {
            mesh.position.y = THREE.MathUtils.clamp(intersectionPoint.y, -5, 5);
        }
    } else {
        if (raycaster.ray.intersectPlane(planeXZ, intersectionPoint)) {
            const distance = Math.sqrt(intersectionPoint.x * intersectionPoint.x + intersectionPoint.z * intersectionPoint.z);
            if (distance <= 10.0) {
                mesh.position.x = intersectionPoint.x;
                mesh.position.z = intersectionPoint.z;
            } else {
                const angle = Math.atan2(intersectionPoint.z, intersectionPoint.x);
                mesh.position.x = Math.cos(angle) * 10.0;
                mesh.position.z = Math.sin(angle) * 10.0;
            }
        }
    }
    
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
    
    initAudio();
    
    fileInput.disabled = true;
    btnSeparate.disabled = true;
    
    sepProgress.classList.remove("hidden");
    progressBarFill.style.width = "0%";
    progressPercent.innerText = "0%";
    
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    
    let simulatedProgress = 0;
    const progressTimer = setInterval(() => {
        if (simulatedProgress < 90) {
            simulatedProgress += (1.0 / (simulatedProgress * 0.15 + 1)) * 3;
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
        
        progressBarFill.style.width = "100%";
        progressPercent.innerText = "100%";
        progressStatusMsg.innerText = "Stem separation complete!";
        
        setTimeout(() => sepProgress.classList.add("hidden"), 3000);
        
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
        
        duration = Math.max(...activeStemsList.map(s => audioBuffers[s].duration));
        
        document.getElementById("timeline-slider").disabled = false;
        document.getElementById("timeline-slider").max = duration;
        document.getElementById("time-total").innerText = formatTime(duration);
        document.getElementById("time-current").innerText = "00:00";
        
        document.getElementById("btn-play").disabled = false;
        document.getElementById("btn-stop").disabled = false;
        document.getElementById("btn-export").disabled = false;
        
        document.getElementById("status-text").innerText = "All stems loaded successfully";
        document.getElementById("status-dot").classList.remove("loading");
        
        activeStemsList.forEach(stem => {
            const filename = fileInput.files[0].name;
            document.getElementById(`file-label`).innerHTML = "Choose audio file";
            fileInput.disabled = false;
            btnSeparate.disabled = false;
            
            const monitorLabel = document.querySelector(`.stem-item.${stem} .stem-name`);
            monitorLabel.innerHTML = `${stem.charAt(0).toUpperCase() + stem.slice(1)} <span style="font-size: 10px; font-weight:normal; color:#64748b">(${filename})</span>`;
        });
        
    } catch (err) {
        document.getElementById("status-text").innerText = "Failed to load stems";
        document.getElementById("status-dot").classList.remove("loading");
        alert(`Failed to load audio files: ${err.message}`);
    }
}

function playStems(offset = 0) {
    initAudio();
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    
    stopActiveSourceNodes();
    
    startTime = audioCtx.currentTime - offset;
    pausedTime = offset;
    
    activeStemsList.forEach(stem => {
        const buffer = audioBuffers[stem];
        if (!buffer) return;
        
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        
        const panner = audioCtx.createPanner();
        panner.panningModel = 'HRTF';
        panner.distanceModel = 'inverse';
        panner.refDistance = 1.0;
        panner.maxDistance = 10.0;
        panner.rolloffFactor = 1.0;
        
        const mesh = stemMeshes[stem];
        if (panner.positionX) {
            panner.positionX.value = mesh.position.x;
            panner.positionY.value = mesh.position.y;
            panner.positionZ.value = mesh.position.z;
        } else {
            panner.setPosition(mesh.position.x, mesh.position.y, mesh.position.z);
        }
        
        // Create per-stem GainNode to drive Mute/Solo and Volume
        const stemGain = audioCtx.createGain();
        
        source.connect(panner);
        panner.connect(stemGain);
        stemGain.connect(masterGain);
        
        source.start(0, offset);
        
        sourceNodes[stem] = source;
        pannerNodes[stem] = panner;
        stemGainNodes[stem] = stemGain;
    });
    
    updateStemGains();
    
    isPlaying = true;
    document.getElementById("btn-play").innerText = "⏸ Pause";
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
        delete pannerNodes[stem];
        delete stemGainNodes[stem];
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
//  Offline Render Mix & Export Logic
// ─────────────────────────────────────────────────────────────────────────────

async function exportMix() {
    if (!stemsData || duration <= 0) return;
    
    pauseStems();
    
    document.getElementById("status-text").innerText = "Rendering 3D mixdown offline...";
    document.getElementById("status-dot").classList.add("loading");
    
    // Create Offline Context
    const offlineCtx = new OfflineAudioContext(2, duration * 44100, 44100);
    
    // Create Master Gain
    const offlineMasterGain = offlineCtx.createGain();
    offlineMasterGain.gain.value = masterGain ? masterGain.gain.value : 0.8;
    offlineMasterGain.connect(offlineCtx.destination);
    
    const hasSolo = activeStemsList.some(s => stemSolos[s]);
    
    activeStemsList.forEach(stem => {
        const buffer = audioBuffers[stem];
        if (!buffer) return;
        
        const source = offlineCtx.createBufferSource();
        source.buffer = buffer;
        
        const panner = offlineCtx.createPanner();
        panner.panningModel = 'HRTF';
        panner.distanceModel = 'inverse';
        panner.refDistance = 1.0;
        panner.maxDistance = 10.0;
        panner.rolloffFactor = 1.0;
        
        const mesh = stemMeshes[stem];
        panner.positionX.value = mesh.position.x;
        panner.positionY.value = mesh.position.y;
        panner.positionZ.value = mesh.position.z;
        
        // Calculate Offline Stem Gain
        const stemGain = offlineCtx.createGain();
        let targetGain = stemVolumes[stem];
        if (stemMutes[stem] || (hasSolo && !stemSolos[stem])) {
            targetGain = 0.0;
        }
        stemGain.gain.value = targetGain;
        
        source.connect(panner);
        panner.connect(stemGain);
        stemGain.connect(offlineMasterGain);
        
        source.start(0);
    });
    
    try {
        const renderedBuffer = await offlineCtx.startRendering();
        document.getElementById("status-text").innerText = "Encoding WAV file...";
        
        const wavBlob = bufferToWav(renderedBuffer);
        
        // Trigger client-side download
        const url = URL.createObjectURL(wavBlob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "3d_spatial_mix.wav";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        document.getElementById("status-text").innerText = "Export successful!";
        document.getElementById("status-dot").classList.remove("loading");
        
        setTimeout(() => {
            document.getElementById("status-text").innerText = "Audio Engine Ready";
        }, 3000);
        
    } catch(err) {
        console.error(err);
        alert(`Export failed: ${err.message}`);
        document.getElementById("status-text").innerText = "Export failed";
        document.getElementById("status-dot").classList.remove("loading");
    }
}

// Convert AudioBuffer to 16-bit stereo WAV Blob
function bufferToWav(buffer) {
    let numOfChan = buffer.numberOfChannels,
        length = buffer.length * numOfChan * 2 + 44,
        bufferArr = new ArrayBuffer(length),
        view = new DataView(bufferArr),
        channels = [], i, sample,
        offset = 0,
        pos = 0;

    // write headers
    setUint32(0x46464952);                         // "RIFF"
    setUint32(length - 8);                         // file length - 8
    setUint32(0x45564157);                         // "WAVE"

    setUint32(0x20746d66);                         // "fmt " chunk
    setUint32(16);                                 // chunk length
    setUint16(1);                                  // sample format (raw PCM)
    setUint16(numOfChan);                          // channel count
    setUint32(buffer.sampleRate);                  // sample rate
    setUint32(buffer.sampleRate * 2 * numOfChan);  // byte rate
    setUint16(numOfChan * 2);                      // block align
    setUint16(16);                                 // bits per sample

    setUint32(0x61746164);                         // "data" chunk
    setUint32(buffer.length * numOfChan * 2);      // chunk length

    for(i=0; i<buffer.numberOfChannels; i++)
        channels.push(buffer.getChannelData(i));

    while(pos < buffer.length) {
        for(i=0; i<numOfChan; i++) {
            sample = Math.max(-1, Math.min(1, channels[i][pos]));
            sample = (sample < 0 ? sample * 0x8000 : sample * 0x7FFF);
            view.setInt16(44 + offset, sample, true);
            offset += 2;
        }
        pos++;
    }

    return new Blob([bufferArr], {type: "audio/wav"});

    function setUint16(data) {
        view.setUint16(44 - 44 + offset, data, true);
        offset += 2;
    }

    function setUint32(data) {
        view.setUint32(44 - 44 + offset, data, true);
        offset += 4;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  UI Controls & Mixer Bindings
// ─────────────────────────────────────────────────────────────────────────────

const btnPlay = document.getElementById("btn-play");
const btnStop = document.getElementById("btn-stop");
const btnExport = document.getElementById("btn-export");
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

btnExport.addEventListener("click", () => {
    exportMix();
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

// Configure Solo/Mute/Volume callbacks for each stem in right mixer panel
activeStemsList.forEach(stem => {
    const sBtn = document.getElementById(`solo-${stem}`);
    const mBtn = document.getElementById(`mute-${stem}`);
    const vSld = document.getElementById(`vol-${stem}`);
    const vPct = document.getElementById(`vol-pct-${stem}`);

    sBtn.addEventListener("click", () => {
        stemSolos[stem] = !stemSolos[stem];
        sBtn.classList.toggle("active", stemSolos[stem]);
        updateStemGains();
    });

    mBtn.addEventListener("click", () => {
        stemMutes[stem] = !stemMutes[stem];
        mBtn.classList.toggle("active", stemMutes[stem]);
        updateStemGains();
    });

    vSld.addEventListener("input", () => {
        const val = vSld.value;
        vPct.innerText = `${val}%`;
        stemVolumes[stem] = val / 100;
        updateStemGains();
    });
});

// Load Visualizer on layout mount
window.addEventListener("DOMContentLoaded", () => {
    initThree();
});
