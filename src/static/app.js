// ─────────────────────────────────────────────────────────────────────────────
//  Structured Frontend Logger Engine (Console & Live UI Output)
// ─────────────────────────────────────────────────────────────────────────────
const Logger = {
    logs: [],
    maxLogs: 150,
    info(msg, ...args) { this._emit('INFO', msg, '#38bdf8', ...args); },
    success(msg, ...args) { this._emit('SUCCESS', msg, '#10b981', ...args); },
    warn(msg, ...args) { this._emit('WARN', msg, '#f59e0b', ...args); },
    error(msg, ...args) { this._emit('ERROR', msg, '#ef4444', ...args); },
    audio(msg, ...args) { this._emit('AUDIO', msg, '#a855f7', ...args); },
    dsp(msg, ...args) { this._emit('DSP', msg, '#ec4899', ...args); },
    _emit(level, msg, color, ...args) {
        const time = new Date().toLocaleTimeString();
        console.log(`%c[${time}] [${level}] ${msg}`, `color: ${color}; font-weight: bold;`, ...args);
        
        const logLine = `[${time}] [${level}] ${msg}`;
        this.logs.push({ time, level, msg, color });
        if (this.logs.length > this.maxLogs) this.logs.shift();
        
        const logContainer = document.getElementById("ui-log-content");
        if (logContainer) {
            const entry = document.createElement("div");
            entry.className = `log-item log-${level.toLowerCase()}`;
            entry.style.color = color;
            entry.innerText = logLine;
            logContainer.appendChild(entry);
            logContainer.scrollTop = logContainer.scrollHeight;
        }
    }
};

window.addEventListener('error', (e) => {
    Logger.error(`Global Error: ${e.message} (${e.filename}:${e.lineno})`);
});
window.addEventListener('unhandledrejection', (e) => {
    Logger.error(`Unhandled Promise Rejection: ${e.reason ? (e.reason.message || e.reason) : 'Unknown Error'}`);
});

// Global Web Audio State
let audioCtx = null;
let masterGain = null;
let dryGainNode = null;
let reverbGainNode = null;
let convolverNode = null;
let isPlaying = false;
let stemsData = null;  // holds URLs of separated stems
let audioBuffers = {}; // holds loaded AudioBuffers
let sourceNodes = {};  // holds active AudioBufferSourceNodes
let pannerNodes = {};  // holds active PannerNodes
let toneFilterNodes = {}; // holds active BiquadFilterNodes (Tone EQ)
let stemGainNodes = {};// holds active stem GainNodes
let startTime = 0;     // context time when playback started
let pausedTime = 0;    // offset time if paused (seconds)
let duration = 0;      // total length of audio tracks in seconds
let progressInterval = null;

// Mixer State (DAW-style)
let stemVolumes = { vocals: 0.8, drums: 0.8, bass: 0.8, other: 0.8 };
let stemMutes = { vocals: false, drums: false, bass: false, other: false };
let stemSolos = { vocals: false, drums: false, bass: false, other: false };
let stemTones = { vocals: 20000, drums: 20000, bass: 20000, other: 20000 };

// Theater Presets & Acoustic Parameters
const THEATERS = {
    small_room: {
        name: "Small Studio Room",
        width: 10, depth: 10, height: 4,
        absorption: 0.7, t60: 0.3,
        floorColor: 0x332922, wallColor: 0x1e293b, accentColor: 0xf59e0b
    },
    theater: {
        name: "Medium Theater",
        width: 20, depth: 24, height: 7,
        absorption: 0.4, t60: 1.1,
        floorColor: 0x451a03, wallColor: 0x31100e, accentColor: 0xef4444
    },
    symphony: {
        name: "Symphony Hall",
        width: 30, depth: 36, height: 12,
        absorption: 0.25, t60: 2.2,
        floorColor: 0x1e1b4b, wallColor: 0x111827, accentColor: 0x8b5cf6
    },
    cathedral: {
        name: "Grand Cathedral",
        width: 40, depth: 60, height: 20,
        absorption: 0.12, t60: 4.8,
        floorColor: 0x0f172a, wallColor: 0x090d16, accentColor: 0x38bdf8
    }
};

let activeTheaterKey = 'theater';
let customHallWidth = 20;
let customHallDepth = 24;
let customHallHeight = 7;
let cameraMode = 'third_person'; // 'third_person', 'fps', 'top_down'
let pitchAngle = 0; // mouse look pitch for FPS view
let isMouseDownView = false;
let previousMousePosition = { x: 0, y: 0 };

let enableReflections = true;
let enableRays = true;
let reverbWetMix = 0.3;
let wallAbsorption = 0.4;

// Three.js State
let scene, camera, renderer;
let roomGroup = null;
let rayGroup = null;
let listenerGroup = null;
let stemMeshes = {};
let activeStemsList = ['vocals', 'drums', 'bass', 'other'];

// WASD & Listener Controls
let listenerPos = new THREE.Vector3(0, 0.5, 0);
let listenerHeading = 0; // facing direction angle in radians (0 = facing -Z)
const keysPressed = {};

function clearAllKeys() {
    for (const k in keysPressed) {
        delete keysPressed[k];
    }
    shiftPressed = false;
    updateHUDKeyActive("key-w", false);
    updateHUDKeyActive("key-a", false);
    updateHUDKeyActive("key-s", false);
    updateHUDKeyActive("key-d", false);
    updateHUDKeyActive("key-q", false);
    updateHUDKeyActive("key-e", false);
    updateHUDKeyActive("key-shift", false);
}

// Helper to get current dynamic hall dimensions
function getHallDimensions() {
    if (activeTheaterKey === 'custom') {
        return {
            name: "Custom Dynamic Hall",
            width: customHallWidth,
            depth: customHallDepth,
            height: customHallHeight,
            absorption: wallAbsorption,
            t60: calculateDynamicT60(customHallWidth, customHallDepth, customHallHeight, wallAbsorption),
            floorColor: 0x111827,
            wallColor: 0x0f172a,
            accentColor: 0xa855f7
        };
    }
    return THEATERS[activeTheaterKey] || THEATERS['theater'];
}

function calculateDynamicT60(w, d, h, absorption) {
    const volume = w * d * h;
    const surfaceArea = 2 * (w * d + w * h + d * h);
    const alpha = Math.max(0.05, absorption);
    // Sabine formula T60 = 0.161 * V / (S * alpha)
    const t60 = 0.161 * volume / (surfaceArea * alpha);
    return Math.min(8.0, Math.max(0.2, t60));
}

// Pointer / Raycasting Pick & Place State
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let planeXZ = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let intersectionPoint = new THREE.Vector3();
let selectedStem = null;
let isCarrying = false;
let shiftPressed = false;

// Color Coding matching UI
const STEM_COLORS = {
    vocals: 0xf43f5e, // Pink/Red
    drums: 0xf59e0b,  // Amber
    bass: 0x10b981,   // Emerald Green
    other: 0x3b82f6   // Blue
};

// Initial stem positions in Three.js coordinates (X, Y, Z)
const INITIAL_POSITIONS = {
    vocals: { x: -3, y: 0.5, z: 3 },
    drums:  { x: 0,  y: 0.5, z: -5 },
    bass:   { x: -3, y: 0.5, z: -4 },
    other:  { x: 4,  y: 0.5, z: 2 }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Web Audio Initialization & DSP Routing
// ─────────────────────────────────────────────────────────────────────────────

function initAudio() {
    if (audioCtx) {
        if (audioCtx.state === 'suspended') {
            audioCtx.resume().catch(() => {});
        }
        return;
    }
    
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass();
    if (audioCtx.state === 'suspended') {
        audioCtx.resume().catch(() => {});
    }
    
    Logger.audio(`Initialized Web Audio Context (SampleRate: ${audioCtx.sampleRate}Hz)`);

    // Create Master Gain Node
    masterGain = audioCtx.createGain();
    masterGain.gain.value = 0.8;
    masterGain.connect(audioCtx.destination);
    
    // Dry / Wet Reverb Gain Taps
    dryGainNode = audioCtx.createGain();
    dryGainNode.gain.value = 1.0 - reverbWetMix;
    dryGainNode.connect(masterGain);
    
    reverbGainNode = audioCtx.createGain();
    reverbGainNode.gain.value = reverbWetMix;
    reverbGainNode.connect(masterGain);
    
    // Convolver Node for Room Impulse Response
    convolverNode = audioCtx.createConvolver();
    try {
        updateReverbIR();
    } catch (e) {
        Logger.warn("Failed to set Reverb IR:", e);
    }
    convolverNode.connect(reverbGainNode);
    
    updateWebAudioListener();
}

function createReverbIR(duration, decay, sampleRate) {
    const sr = sampleRate || (audioCtx ? audioCtx.sampleRate : 44100);
    const safeDuration = Math.max(0.1, Math.min(3.0, duration || 1.0));
    const length = Math.floor(sr * safeDuration);
    const impulse = audioCtx.createBuffer(2, length, sr);
    const left = impulse.getChannelData(0);
    const right = impulse.getChannelData(1);
    const safeDecay = Math.max(0.1, decay || 1.0);

    for (let i = 0; i < length; i++) {
        const t = i / sr;
        const env = Math.exp(-t * 6.91 / safeDecay);
        left[i] = (Math.random() * 2 - 1) * env;
        right[i] = (Math.random() * 2 - 1) * env;
    }
    return impulse;
}

function updateReverbIR() {
    if (!audioCtx || !convolverNode) return;
    try {
        const currentTheater = getHallDimensions();
        const irBuffer = createReverbIR(currentTheater.t60, currentTheater.t60, audioCtx.sampleRate);
        convolverNode.buffer = irBuffer;
    } catch (err) {
        console.warn("Error updating Reverb IR:", err);
    }
}

function getApiUrl(path) {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) {
        return path;
    }
    const origin = (window.location.protocol.startsWith('http')) 
        ? window.location.origin 
        : 'http://127.0.0.1:5000';
    
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `${origin}${cleanPath}`;
}

function getWsUrl(path) {
    if (!path) return '';
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    if (window.location.protocol === 'https:') {
        return `wss://${window.location.host}${cleanPath}`;
    }
    if (window.location.protocol === 'http:') {
        return `ws://${window.location.host}${cleanPath}`;
    }
    return `ws://127.0.0.1:5000${cleanPath}`;
}

async function loadAudioBuffer(url) {
    const fullUrl = getApiUrl(url);
    const response = await fetch(fullUrl);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status} loading stem (${response.statusText})`);
    }
    let arrayBuffer = await response.arrayBuffer();
    
    initAudio();

    if (audioCtx && audioCtx.state === 'suspended') {
        try { await audioCtx.resume(); } catch (_) {}
    }

    try {
        const decoded = await audioCtx.decodeAudioData(arrayBuffer);
        arrayBuffer = null;
        return decoded;
    } catch (promiseErr) {
        return new Promise((resolve, reject) => {
            try {
                const bufferCopy = (arrayBuffer && arrayBuffer.byteLength > 0) ? arrayBuffer.slice(0) : new ArrayBuffer(0);
                audioCtx.decodeAudioData(
                    bufferCopy,
                    (decodedBuffer) => {
                        arrayBuffer = null;
                        resolve(decodedBuffer);
                    },
                    (err) => reject(new Error(`WebAudio decode error: ${err ? (err.message || err) : 'Failed to decode audio'}`))
                );
            } catch (fallbackErr) {
                reject(new Error(`Decoding error: ${fallbackErr.message || fallbackErr}`));
            }
        });
    }
}

function updateStemGains() {
    if (!audioCtx) return;
    
    const hasSolo = activeStemsList.some(s => stemSolos[s]);
    
    activeStemsList.forEach(stem => {
        const gainNode = stemGainNodes[stem];
        if (!gainNode) return;
        
        let targetGain = stemVolumes[stem];
        
        if (stemMutes[stem] || (hasSolo && !stemSolos[stem])) {
            targetGain = 0.0;
        }
        
        gainNode.gain.setTargetAtTime(targetGain, audioCtx.currentTime, 0.02);
    });
}

function updateWebAudioListener() {
    if (!audioCtx) return;
    const listener = audioCtx.listener;
    
    // Heading forward vector (-sin(heading), 0, -cos(heading))
    const fX = -Math.sin(listenerHeading);
    const fY = 0;
    const fZ = -Math.cos(listenerHeading);
    
    if (listener.positionX) {
        listener.positionX.value = listenerPos.x;
        listener.positionY.value = listenerPos.y;
        listener.positionZ.value = listenerPos.z;
        listener.forwardX.value = fX;
        listener.forwardY.value = fY;
        listener.forwardZ.value = fZ;
        listener.upX.value = 0;
        listener.upY.value = 1;
        listener.upZ.value = 0;
    } else {
        listener.setPosition(listenerPos.x, listenerPos.y, listenerPos.z);
        listener.setOrientation(fX, fY, fZ, 0, 1, 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Three.js 3D Theater & Visualizer Engine
// ─────────────────────────────────────────────────────────────────────────────

function initThree() {
    const container = document.getElementById("canvas-container");
    if (!container) return;
    if (typeof THREE === 'undefined') {
        console.error("[Three.js Error]: THREE is not loaded.");
        return;
    }

    const width = Math.max(100, container.clientWidth || 800);
    const height = Math.max(100, container.clientHeight || 600);
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090d16);
    
    camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 200);
    
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    renderer.domElement.addEventListener('webglcontextlost', (e) => {
        e.preventDefault();
        console.warn('[WebGL] WebGL Context Lost. Pausing rendering loop...');
    }, false);

    renderer.domElement.addEventListener('webglcontextrestored', () => {
        console.log('[WebGL] WebGL Context Restored. Rebuilding scene geometry...');
        buildTheaterGeometry();
    }, false);

    container.appendChild(renderer.domElement);
    
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(10, 20, 15);
    dirLight.castShadow = true;
    scene.add(dirLight);
    
    // Build initial Theater architecture
    buildTheaterGeometry();
    
    // Create Listener Avatar Group
    listenerGroup = new THREE.Group();
    const headGeo = new THREE.SphereGeometry(0.5, 32, 32);
    const headMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.3, metalness: 0.2 });
    const headMesh = new THREE.Mesh(headGeo, headMat);
    headMesh.castShadow = true;
    listenerGroup.add(headMesh);
    
    const noseGeo = new THREE.ConeGeometry(0.15, 0.4, 4);
    const noseMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const noseMesh = new THREE.Mesh(noseGeo, noseMat);
    noseMesh.position.set(0, 0, -0.6);
    noseMesh.rotation.x = -Math.PI / 2;
    listenerGroup.add(noseMesh);

    scene.add(listenerGroup);
    
    // Create 3D Speakers for each Stem
    activeStemsList.forEach(stem => {
        const pos = INITIAL_POSITIONS[stem];
        const speakerGroup = createSpeakerMesh(stem);
        speakerGroup.position.set(pos.x, pos.y, pos.z);
        scene.add(speakerGroup);
        stemMeshes[stem] = speakerGroup;
        
        updateUIMonitor(stem, pos.x, pos.y, pos.z);
    });

    window.addEventListener("resize", () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
    
    // Keyboard Event Listeners for WASD Navigation with Shift/Caps normalization and input filtering
    function isInputElement(target) {
        if (!target || !target.tagName) return false;
        const tag = target.tagName.toUpperCase();
        return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable;
    }

    window.addEventListener("keydown", (e) => {
        if (isInputElement(e.target)) return;
        keysPressed[e.key] = true;
        if (e.key) keysPressed[e.key.toLowerCase()] = true;
        if (e.code) keysPressed[e.code] = true;
        if (e.key === "Shift" || (e.code && e.code.startsWith("Shift"))) shiftPressed = true;
    });
    
    window.addEventListener("keyup", (e) => {
        if (e.key) {
            keysPressed[e.key] = false;
            keysPressed[e.key.toLowerCase()] = false;
        }
        if (e.code) keysPressed[e.code] = false;
        if (e.key === "Shift" || (e.code && e.code.startsWith("Shift"))) shiftPressed = false;
    });

    window.addEventListener("blur", clearAllKeys);
    window.addEventListener("focus", clearAllKeys);
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) clearAllKeys();
    });
    
    // Drag-and-drop listener
    container.addEventListener("mousedown", onMouseDown, false);
    container.addEventListener("mousemove", onMouseMove, false);
    container.addEventListener("mouseup", onMouseUp, false);
    
    animate();
}

// ─────────────────────────────────────────────────────────────────────────────
//  3D Speaker Enclosure Model Construction
// ─────────────────────────────────────────────────────────────────────────────

function createSpeakerMesh(stem) {
    const group = new THREE.Group();
    group.name = stem;
    const color = STEM_COLORS[stem];

    // Cabinet Enclosure Box
    const cabinetGeo = new THREE.BoxGeometry(0.8, 1.3, 0.7);
    const cabinetMat = new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        roughness: 0.4,
        metalness: 0.3
    });
    const cabinet = new THREE.Mesh(cabinetGeo, cabinetMat);
    cabinet.castShadow = true;
    cabinet.receiveShadow = true;
    group.add(cabinet);

    // Front Baffle Plate
    const baffleGeo = new THREE.PlaneGeometry(0.75, 1.25);
    const baffleMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.8 });
    const baffle = new THREE.Mesh(baffleGeo, baffleMat);
    baffle.position.z = 0.351;
    group.add(baffle);

    // Main Woofer Cone
    const wooferGeo = new THREE.CylinderGeometry(0.24, 0.1, 0.08, 32);
    const wooferMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.3, metalness: 0.5 });
    const woofer = new THREE.Mesh(wooferGeo, wooferMat);
    woofer.rotation.x = Math.PI / 2;
    woofer.position.set(0, -0.2, 0.36);
    woofer.name = "woofer";
    group.add(woofer);

    // Tweeter Dome
    const tweeterGeo = new THREE.SphereGeometry(0.08, 16, 16);
    const tweeterMat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.2, metalness: 0.8 });
    const tweeter = new THREE.Mesh(tweeterGeo, tweeterMat);
    tweeter.position.set(0, 0.3, 0.36);
    group.add(tweeter);

    // LED Status Light
    const ledGeo = new THREE.SphereGeometry(0.06, 16, 16);
    const ledMat = new THREE.MeshBasicMaterial({ color: color });
    const led = new THREE.Mesh(ledGeo, ledMat);
    led.position.set(0, 0.5, 0.36);
    group.add(led);

    // Sound Wave Pulse Ring
    const waveGeo = new THREE.RingGeometry(0.5, 0.55, 32);
    const waveMat = new THREE.MeshBasicMaterial({
        color: color,
        side: THREE.DoubleSide,
        opacity: 0.0,
        transparent: true
    });
    const waveMesh = new THREE.Mesh(waveGeo, waveMat);
    waveMesh.rotation.x = Math.PI / 2;
    waveMesh.name = "wave";
    group.add(waveMesh);

    // Floating 3D Text Label Sprite
    const labelSprite = createTextLabelSprite(stem.toUpperCase(), color);
    labelSprite.position.set(0, 1.0, 0);
    group.add(labelSprite);

    return group;
}

function createTextLabelSprite(text, hexColor) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
    const hexStr = `#${hexColor.toString(16).padStart(6, '0')}`;
    ctx.strokeStyle = hexStr;
    ctx.lineWidth = 4;
    
    ctx.beginPath();
    ctx.roundRect(4, 4, 248, 56, 10);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = hexStr;
    ctx.font = 'bold 22px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`🔊 ${text}`, 128, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(1.5, 0.375, 1);
    return sprite;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Theater Architecture Generator (Small Room to Cathedral)
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
//  Empty Concert Hall Architecture Generator
// ─────────────────────────────────────────────────────────────────────────────

function buildTheaterGeometry() {
    if (roomGroup) scene.remove(roomGroup);
    roomGroup = new THREE.Group();

    const t = getHallDimensions();
    const w = t.width;
    const d = t.depth;
    const h = t.height;

    // Floor Mesh (Polished hall floor)
    const floorGeo = new THREE.PlaneGeometry(w, d);
    const floorMat = new THREE.MeshStandardMaterial({
        color: t.floorColor,
        roughness: 0.4,
        metalness: 0.3
    });
    const floorMesh = new THREE.Mesh(floorGeo, floorMat);
    floorMesh.rotation.x = -Math.PI / 2;
    floorMesh.position.y = 0;
    floorMesh.receiveShadow = true;
    roomGroup.add(floorMesh);

    // Floor Grid lines (Architectural layout grid)
    const gridDivs = Math.max(w, d);
    const grid = new THREE.GridHelper(Math.max(w, d), gridDivs, t.accentColor, 0x334155);
    grid.position.y = 0.01;
    roomGroup.add(grid);

    // Outer Boundary Box Wireframe
    const wallGeo = new THREE.BoxGeometry(w, h, d);
    const wallEdges = new THREE.EdgesGeometry(wallGeo);
    const wallMat = new THREE.LineBasicMaterial({ color: t.accentColor, opacity: 0.7, transparent: true });
    const wallLines = new THREE.LineSegments(wallEdges, wallMat);
    wallLines.position.y = h / 2;
    roomGroup.add(wallLines);

    // Semi-transparent Hall Wall Shell
    const wallPlaneMat = new THREE.MeshPhysicalMaterial({
        color: t.wallColor,
        roughness: 0.7,
        transmission: 0.6,
        opacity: 0.18,
        transparent: true,
        side: THREE.BackSide
    });
    const wallBox = new THREE.Mesh(wallGeo, wallPlaneMat);
    wallBox.position.y = h / 2;
    roomGroup.add(wallBox);

    // Boundary Corner Pillars to define hall geometry
    const pillarGeo = new THREE.CylinderGeometry(0.3, 0.3, h, 16);
    const pillarMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.5 });
    [[-w/2, -d/2], [w/2, -d/2], [-w/2, d/2], [w/2, d/2]].forEach(([px, pz]) => {
        const pillar = new THREE.Mesh(pillarGeo, pillarMat);
        pillar.position.set(px, h / 2, pz);
        roomGroup.add(pillar);
    });

    scene.add(roomGroup);

    // Update HUD Stats
    const dimEl = document.getElementById("hud-hall-dimensions");
    const volEl = document.getElementById("hud-hall-volume");
    if (dimEl) dimEl.innerText = `${w}m × ${d}m × ${h}m`;
    if (volEl) {
        const vol = (w * d * h).toLocaleString();
        volEl.innerText = `${vol} m³`;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  WASD Free-Roam & Camera Controller
// ─────────────────────────────────────────────────────────────────────────────

function handleWASD() {
    const t = getHallDimensions();
    const halfW = t.width / 2 - 0.5;
    const halfD = t.depth / 2 - 0.5;
    const maxH = t.height - 0.5;

    const isSprint = keysPressed['Shift'] || keysPressed['shift'] || keysPressed['ShiftLeft'] || keysPressed['ShiftRight'] || shiftPressed;
    const MOVE_SPEED = isSprint ? 0.35 : 0.15;

    const forwardVec = new THREE.Vector3(-Math.sin(listenerHeading), 0, -Math.cos(listenerHeading));
    const rightVec = new THREE.Vector3(Math.cos(listenerHeading), 0, -Math.sin(listenerHeading));

    const isW = keysPressed['w'] || keysPressed['W'] || keysPressed['KeyW'] || keysPressed['ArrowUp'];
    const isS = keysPressed['s'] || keysPressed['S'] || keysPressed['KeyS'] || keysPressed['ArrowDown'];
    const isA = keysPressed['a'] || keysPressed['A'] || keysPressed['KeyA'] || keysPressed['ArrowLeft'];
    const isD = keysPressed['d'] || keysPressed['D'] || keysPressed['KeyD'] || keysPressed['ArrowRight'];
    const isQ = keysPressed['q'] || keysPressed['Q'] || keysPressed['KeyQ'];
    const isE = keysPressed['e'] || keysPressed['E'] || keysPressed['KeyE'];

    if (isW) listenerPos.addScaledVector(forwardVec, MOVE_SPEED);
    if (isS) listenerPos.addScaledVector(forwardVec, -MOVE_SPEED);
    if (isA) listenerPos.addScaledVector(rightVec, -MOVE_SPEED);
    if (isD) listenerPos.addScaledVector(rightVec, MOVE_SPEED);
    if (isQ) listenerPos.y = Math.max(0.3, listenerPos.y - MOVE_SPEED);
    if (isE) listenerPos.y = Math.min(maxH, listenerPos.y + MOVE_SPEED);

    // Update HUD active key caps
    updateHUDKeyActive("key-w", isW);
    updateHUDKeyActive("key-a", isA);
    updateHUDKeyActive("key-s", isS);
    updateHUDKeyActive("key-d", isD);
    updateHUDKeyActive("key-q", isQ);
    updateHUDKeyActive("key-e", isE);
    updateHUDKeyActive("key-shift", isSprint);

    // Clamp listener inside hall boundaries
    listenerPos.x = THREE.MathUtils.clamp(listenerPos.x, -halfW, halfW);
    listenerPos.z = THREE.MathUtils.clamp(listenerPos.z, -halfD, halfD);

    if (listenerGroup) {
        listenerGroup.position.copy(listenerPos);
        listenerGroup.rotation.y = listenerHeading;
    }

    updateWebAudioListener();

    // Camera positioning based on cameraMode
    if (cameraMode === 'fps') {
        // 1st-Person Listener FPS View
        camera.position.copy(listenerPos);
        camera.position.y += 0.2; // Eye height
        
        const targetLook = new THREE.Vector3(
            listenerPos.x - Math.sin(listenerHeading) * Math.cos(pitchAngle),
            listenerPos.y + 0.2 + Math.sin(pitchAngle),
            listenerPos.z - Math.cos(listenerHeading) * Math.cos(pitchAngle)
        );
        camera.lookAt(targetLook);
    } else if (cameraMode === 'top_down') {
        // Top-Down Stage View
        const camH = Math.max(t.width, t.depth) * 1.1;
        camera.position.set(0, camH, 0.1);
        camera.lookAt(0, 0, 0);
    } else {
        // 3rd-Person Follow View (Default)
        const camOffset = new THREE.Vector3(
            Math.sin(listenerHeading) * 8,
            5,
            Math.cos(listenerHeading) * 8
        );
        camera.position.copy(listenerPos).add(camOffset);
        camera.lookAt(listenerPos.x, listenerPos.y + 0.5, listenerPos.z);
    }

    updateWASDHUD();
}

function updateHUDKeyActive(elementId, isActive) {
    const el = document.getElementById(elementId);
    if (el) el.classList.toggle("active", Boolean(isActive));
}

let lastHUDPos = { x: -999, y: -999, z: -999, h: -999 };

function updateWASDHUD() {
    const headingDeg = Math.round(((listenerHeading * 180 / Math.PI) % 360 + 360) % 360);
    
    if (Math.abs(listenerPos.x - lastHUDPos.x) > 0.05 ||
        Math.abs(listenerPos.y - lastHUDPos.y) > 0.05 ||
        Math.abs(listenerPos.z - lastHUDPos.z) > 0.05 ||
        headingDeg !== lastHUDPos.h) {

        lastHUDPos = { x: listenerPos.x, y: listenerPos.y, z: listenerPos.z, h: headingDeg };
        const posTxt = `(${listenerPos.x.toFixed(1)}, ${listenerPos.y.toFixed(1)}, ${listenerPos.z.toFixed(1)})`;
        const posEl = document.getElementById("hud-listener-pos");
        const headEl = document.getElementById("hud-listener-heading");
        if (posEl) posEl.innerText = posTxt;
        if (headEl) headEl.innerText = `${headingDeg}°`;
    }
}

const rayLinePool = {};

function updateAcousticRays() {
    if (!enableRays || !enableReflections) {
        if (rayGroup) rayGroup.visible = false;
        return;
    }

    if (!rayGroup) {
        rayGroup = new THREE.Group();
        scene.add(rayGroup);
    }
    rayGroup.visible = true;

    const t = getHallDimensions();
    const halfW = t.width / 2;
    const halfD = t.depth / 2;

    activeStemsList.forEach((stem) => {
        const mesh = stemMeshes[stem];
        if (!mesh) return;

        const spkPos = mesh.position;
        const color = STEM_COLORS[stem];

        const wallBouncePoints = [
            new THREE.Vector3(-halfW, spkPos.y, (spkPos.z + listenerPos.z)/2),
            new THREE.Vector3(halfW, spkPos.y, (spkPos.z + listenerPos.z)/2),
            new THREE.Vector3((spkPos.x + listenerPos.x)/2, spkPos.y, -halfD),
            new THREE.Vector3((spkPos.x + listenerPos.x)/2, spkPos.y, halfD)
        ];

        if (!rayLinePool[stem]) {
            rayLinePool[stem] = [];
            const rayMat = new THREE.LineBasicMaterial({
                color: color,
                opacity: 0.35,
                transparent: true
            });

            for (let i = 0; i < 4; i++) {
                const rayGeo = new THREE.BufferGeometry();
                const positions = new Float32Array(9);
                rayGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                const line = new THREE.Line(rayGeo, rayMat);
                rayGroup.add(line);
                rayLinePool[stem].push({ line, geo: rayGeo });
            }
        }

        wallBouncePoints.forEach((bounce, idx) => {
            const entry = rayLinePool[stem][idx];
            if (!entry) return;

            const posAttr = entry.geo.attributes.position;
            const array = posAttr.array;

            // P0: Speaker
            array[0] = spkPos.x; array[1] = spkPos.y; array[2] = spkPos.z;
            // P1: Wall Bounce
            array[3] = bounce.x; array[4] = bounce.y; array[5] = bounce.z;
            // P2: Listener
            array[6] = listenerPos.x; array[7] = listenerPos.y; array[8] = listenerPos.z;

            posAttr.needsUpdate = true;
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main Animation Loop
// ─────────────────────────────────────────────────────────────────────────────

function animate() {
    requestAnimationFrame(animate);
    
    handleWASD();
    updateAcousticRays();
    
    const hasSolo = activeStemsList.some(s => stemSolos[s]);

    activeStemsList.forEach(stem => {
        const mesh = stemMeshes[stem];
        if (!mesh) return;

        const wave = mesh.getObjectByName("wave");
        const woofer = mesh.getObjectByName("woofer");
        
        const isSilent = stemMutes[stem] || (hasSolo && !stemSolos[stem]);

        if (isPlaying && !isSilent) {
            const time = Date.now() * 0.008;
            const pulse = 1.0 + Math.sin(time * 3 + INITIAL_POSITIONS[stem].x) * 0.08;
            
            if (woofer) woofer.scale.set(pulse, 1.0, pulse);

            if (wave) {
                wave.scale.addScalar(0.04);
                wave.material.opacity -= 0.02;

                if (wave.material.opacity <= 0) {
                    wave.scale.set(1, 1, 1);
                    wave.material.opacity = 0.7;
                }
            }
        } else {
            if (woofer) woofer.scale.set(1, 1, 1);
            if (wave) {
                wave.scale.set(1, 1, 1);
                wave.material.opacity = 0.0;
            }
        }
    });

    renderer.render(scene, camera);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Position Calculations & Monitoring Update
// ─────────────────────────────────────────────────────────────────────────────

function updateUIMonitor(stem, x, y, z) {
    const relX = x - listenerPos.x;
    const relY = y - listenerPos.y;
    const relZ = z - listenerPos.z;

    let azimuthRad = Math.atan2(relX, -relZ);
    let azimuthDeg = Math.round(azimuthRad * 180 / Math.PI);
    
    let horizontalDist = Math.sqrt(relX*relX + relZ*relZ);
    let elevationRad = Math.atan2(relY, horizontalDist);
    let elevationDeg = Math.round(elevationRad * 180 / Math.PI);
    
    let dist = Math.sqrt(relX*relX + relY*relY + relZ*relZ).toFixed(1);

    let azTxt = azimuthDeg === 0 ? "0° (Front)" : (azimuthDeg > 0 ? `${azimuthDeg}° (Right)` : `${Math.abs(azimuthDeg)}° (Left)`);
    let elTxt = elevationDeg === 0 ? "0° (Level)" : (elevationDeg > 0 ? `+${elevationDeg}° (Up)` : `${elevationDeg}° (Down)`);

    if (document.getElementById(`coord-${stem}-az`)) {
        document.getElementById(`coord-${stem}-az`).innerText = azTxt;
        document.getElementById(`coord-${stem}-el`).innerText = elTxt;
        document.getElementById(`coord-${stem}-dist`).innerText = `${dist}m`;
    }
}

let lastPannerPositions = {};

function updatePannerPosition(stem, x, y, z) {
    const panner = pannerNodes[stem];
    if (!panner) return;

    const last = lastPannerPositions[stem];
    if (last &&
        Math.abs(last.x - x) < 1e-4 &&
        Math.abs(last.y - y) < 1e-4 &&
        Math.abs(last.z - z) < 1e-4) {
        return;
    }

    lastPannerPositions[stem] = { x, y, z };

    if (panner.positionX) {
        panner.positionX.value = x;
        panner.positionY.value = y;
        panner.positionZ.value = z;
    } else {
        panner.setPosition(x, y, z);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pointer Pick & Place Drag Controls (Raycasting)
// ─────────────────────────────────────────────────────────────────────────────

function getMousePosition(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function onMouseDown(event) {
    getMousePosition(event);
    previousMousePosition = { x: event.clientX, y: event.clientY };
    
    raycaster.setFromCamera(mouse, camera);
    const targets = activeStemsList.map(s => stemMeshes[s]).filter(Boolean);
    const intersects = raycaster.intersectObjects(targets, true);
    
    if (intersects.length > 0) {
        let obj = intersects[0].object;
        while (obj.parent && obj.parent !== scene) {
            obj = obj.parent;
        }
        
        selectedStem = obj.name;
        isCarrying = true;
        isMouseDownView = false;
        
        document.querySelectorAll('.stem-item').forEach(el => el.classList.remove('active'));
        const el = document.querySelector(`.stem-item.${selectedStem}`);
        if (el) el.classList.add('active');
        
        raycaster.ray.intersectPlane(planeXZ, intersectionPoint);
    } else {
        isMouseDownView = true;
    }
}

function onMouseMove(event) {
    if (isMouseDownView) {
        const deltaX = event.clientX - previousMousePosition.x;
        const deltaY = event.clientY - previousMousePosition.y;
        
        listenerHeading += deltaX * 0.005;
        pitchAngle = THREE.MathUtils.clamp(pitchAngle - deltaY * 0.005, -Math.PI / 2.2, Math.PI / 2.2);
        
        previousMousePosition = { x: event.clientX, y: event.clientY };
        return;
    }

    if (!isCarrying || !selectedStem) return;
    
    getMousePosition(event);
    raycaster.setFromCamera(mouse, camera);
    
    const mesh = stemMeshes[selectedStem];
    const t = getHallDimensions();
    const halfW = t.width / 2 - 0.5;
    const halfD = t.depth / 2 - 0.5;
    
    if (shiftPressed) {
        const cameraNormal = new THREE.Vector3();
        camera.getWorldDirection(cameraNormal);
        cameraNormal.y = 0;
        cameraNormal.normalize();
        
        const planeY = new THREE.Plane(cameraNormal, -mesh.position.dot(cameraNormal));
        if (raycaster.ray.intersectPlane(planeY, intersectionPoint)) {
            mesh.position.y = THREE.MathUtils.clamp(intersectionPoint.y, 0.5, t.height - 0.5);
        }
    } else {
        if (raycaster.ray.intersectPlane(planeXZ, intersectionPoint)) {
            mesh.position.x = THREE.MathUtils.clamp(intersectionPoint.x, -halfW, halfW);
            mesh.position.z = THREE.MathUtils.clamp(intersectionPoint.z, -halfD, halfD);
        }
    }
    
    updateUIMonitor(selectedStem, mesh.position.x, mesh.position.y, mesh.position.z);
    updatePannerPosition(selectedStem, mesh.position.x, mesh.position.y, mesh.position.z);
}

function onMouseUp() {
    isCarrying = false;
    isMouseDownView = false;
    selectedStem = null;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Demucs Stem Separation Connection
// ─────────────────────────────────────────────────────────────────────────────

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("audio-file");
const fileLabel = document.getElementById("file-label");
const sepProgress = document.getElementById("separation-progress");
const progressBarFill = document.getElementById("progress-bar-fill");
const progressPercent = document.getElementById("progress-percent");
const progressStatusMsg = document.getElementById("progress-status-msg");
const btnSeparate = document.getElementById("btn-separate");

if (fileInput) {
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            fileLabel.innerHTML = `
                <span class="upload-icon">🎵</span>
                <span class="upload-title">${fileInput.files[0].name}</span>
                <span class="upload-subtitle">Ready to separate</span>
            `;
        }
    });
}

if (uploadForm) {
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!fileInput || fileInput.files.length === 0) return;
        
        initAudio();
        
        fileInput.disabled = true;
        btnSeparate.disabled = true;
        
        sepProgress.classList.remove("hidden");
        progressBarFill.style.width = "0%";
        progressPercent.innerText = "0%";
        progressStatusMsg.innerText = "Connecting live status...";
        
        const sessionId = 'session_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();
        
        Logger.info(`Starting separation for: ${fileInput.files[0].name} (Session: ${sessionId})`);
        
        let ws = null;
        try {
            const wsUrl = getWsUrl(`/ws/progress/${sessionId}`);
            ws = new WebSocket(wsUrl);
            ws.onopen = () => {
                Logger.info(`Live progress WebSocket connected for session: ${sessionId}`);
            };
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (typeof data.percent === 'number') {
                        progressBarFill.style.width = `${data.percent}%`;
                        progressPercent.innerText = `${data.percent}%`;
                    }
                    if (data.status) {
                        progressStatusMsg.innerText = data.status;
                        Logger.info(`[Separation Progress]: ${data.status}`);
                    }
                } catch (err) {
                    console.error("WebSocket message parse error:", err);
                }
            };
            await new Promise((res) => setTimeout(res, 150));
        } catch (wsErr) {
            Logger.warn("WebSocket progress connection fallback:", wsErr);
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        formData.append("session_id", sessionId);
        
        try {
            const separateUrl = getApiUrl("/api/separate");
            const response = await fetch(separateUrl, {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) {
                const data = await response.json().catch(() => ({ detail: "Separation request failed" }));
                throw new Error(data.detail || "Separation failed");
            }
            
            const data = await response.json();
            stemsData = data.stems;
            
            progressBarFill.style.width = "100%";
            progressPercent.innerText = "100%";
            progressStatusMsg.innerText = "Separation complete! Decoding audio...";
            Logger.success("Stem separation finished! Received stem URLs from server.");

            if (ws) {
                try { ws.close(); } catch (_) {}
            }

            setTimeout(() => sepProgress.classList.add("hidden"), 2500);
            
            await loadAllStems();
            
        } catch (err) {
            if (ws) {
                try { ws.close(); } catch (_) {}
            }
            sepProgress.classList.add("hidden");
            if (fileInput) fileInput.disabled = false;
            if (btnSeparate) btnSeparate.disabled = false;
            Logger.error(`Error during stem separation: ${err.message}`);
            alert(`Error during stem separation: ${err.message}`);
        } finally {
            clearAllKeys();
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
//  Audio Playback Engine
// ─────────────────────────────────────────────────────────────────────────────

async function loadAllStems() {
    if (!stemsData) return;
    
    document.getElementById("status-text").innerText = "Decoding separated audio stems...";
    document.getElementById("status-dot").classList.add("loading");
    Logger.audio("Decoding 4 separated audio stem tracks into Web Audio buffers...");

    try {
        // Stop active audio nodes & free old buffer references before loading new stems
        stopStems();
        audioBuffers = {};

        // Decode stems sequentially with GC pauses to prevent memory spike / WebAudio renderer crash
        for (const stem of activeStemsList) {
            if (stemsData[stem]) {
                document.getElementById("status-text").innerText = `Decoding ${stem} audio stem...`;
                Logger.audio(`Fetching & decoding ${stem} stem WAV file...`);
                
                // Allow V8 Garbage Collector to breathe between stem decodes
                await new Promise(res => setTimeout(res, 200));
                
                audioBuffers[stem] = await loadAudioBuffer(stemsData[stem]);
            }
        }
        
        duration = Math.max(0, ...activeStemsList.map(s => (audioBuffers[s] ? audioBuffers[s].duration : 0)));
        
        document.getElementById("timeline-slider").disabled = false;
        document.getElementById("timeline-slider").max = duration;
        document.getElementById("time-total").innerText = formatTime(duration);
        document.getElementById("time-current").innerText = "00:00";
        
        document.getElementById("btn-play").disabled = false;
        document.getElementById("btn-stop").disabled = false;
        document.getElementById("btn-export").disabled = false;
        
        document.getElementById("status-text").innerText = "All stems loaded & ready to mix!";
        document.getElementById("status-dot").classList.remove("loading");
        Logger.success(`All 4 stems decoded successfully! Total Duration: ${formatTime(duration)} (${duration.toFixed(1)}s)`);

        if (fileInput) fileInput.disabled = false;
        if (btnSeparate) btnSeparate.disabled = false;
        
    } catch (err) {
        document.getElementById("status-text").innerText = "Error decoding audio";
        document.getElementById("status-dot").classList.remove("loading");
        if (fileInput) fileInput.disabled = false;
        if (btnSeparate) btnSeparate.disabled = false;
        Logger.error(`Failed to decode stems: ${err.message}`);
        alert(`Failed to fetch audio stems: ${err.message}`);
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
        panner.maxDistance = 20.0;
        panner.rolloffFactor = 1.0;
        
        const mesh = stemMeshes[stem];
        const posX = (mesh && mesh.position) ? mesh.position.x : (INITIAL_POSITIONS[stem]?.x || 0);
        const posY = (mesh && mesh.position) ? mesh.position.y : (INITIAL_POSITIONS[stem]?.y || 0.5);
        const posZ = (mesh && mesh.position) ? mesh.position.z : (INITIAL_POSITIONS[stem]?.z || 0);

        if (panner.positionX) {
            panner.positionX.value = posX;
            panner.positionY.value = posY;
            panner.positionZ.value = posZ;
        } else {
            panner.setPosition(posX, posY, posZ);
        }
        
        // Tone EQ
        const toneFilter = audioCtx.createBiquadFilter();
        toneFilter.type = 'lowpass';
        toneFilter.frequency.value = stemTones[stem];
        
        // Stem Gain Node
        const stemGain = audioCtx.createGain();
        
        // Audio Signal Flow: Source -> Tone EQ -> HRTF Panner -> Stem Gain -> Dry/Wet -> Reverb/Master
        source.connect(toneFilter);
        toneFilter.connect(panner);
        panner.connect(stemGain);
        
        stemGain.connect(dryGainNode);
        stemGain.connect(convolverNode);
        
        source.start(0, offset);
        
        sourceNodes[stem] = source;
        pannerNodes[stem] = panner;
        toneFilterNodes[stem] = toneFilter;
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
        delete toneFilterNodes[stem];
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
//  Mix Export Logic
// ─────────────────────────────────────────────────────────────────────────────

async function exportMix() {
    if (!stemsData || duration <= 0) return;
    
    pauseStems();
    
    document.getElementById("status-text").innerText = "Rendering 3D theater mixdown offline...";
    document.getElementById("status-dot").classList.add("loading");
    
    const offlineCtx = new OfflineAudioContext(2, Math.ceil(duration * 44100), 44100);
    
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
        
        const mesh = stemMeshes[stem];
        panner.positionX.value = mesh.position.x;
        panner.positionY.value = mesh.position.y;
        panner.positionZ.value = mesh.position.z;
        
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
        const formatChoice = document.getElementById("select-bitdepth")?.value || "pcm16";
        const wavBlob = bufferToWav(renderedBuffer, formatChoice);
        
        const url = URL.createObjectURL(wavBlob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "3d_theater_mix.wav";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        document.getElementById("status-text").innerText = "Export successful!";
        document.getElementById("status-dot").classList.remove("loading");
        
    } catch(err) {
        console.error(err);
        alert(`Export failed: ${err.message}`);
        document.getElementById("status-text").innerText = "Export failed";
        document.getElementById("status-dot").classList.remove("loading");
    }
}

function bufferToWav(buffer, formatChoice = "pcm16") {
    let numOfChan = buffer.numberOfChannels;
    let bytesPerSample = formatChoice === "pcm24" ? 3 : (formatChoice === "float32" ? 4 : 2);
    let bitsPerSample = bytesPerSample * 8;
    let formatTag = formatChoice === "float32" ? 3 : 1;

    let dataLength = buffer.length * numOfChan * bytesPerSample;
    let totalLength = dataLength + 44;
    let bufferArr = new ArrayBuffer(totalLength);
    let view = new DataView(bufferArr);
    let channels = [];
    let i, sample, offset = 0, pos = 0;

    setUint32(0x46464952);
    setUint32(totalLength - 8);
    setUint32(0x45564157);

    setUint32(0x20746d66);
    setUint32(16);
    setUint16(formatTag);
    setUint16(numOfChan);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * bytesPerSample * numOfChan);
    setUint16(numOfChan * bytesPerSample);
    setUint16(bitsPerSample);

    setUint32(0x61746164);
    setUint32(dataLength);

    for (i = 0; i < buffer.numberOfChannels; i++) {
        channels.push(buffer.getChannelData(i));
    }

    while (pos < buffer.length) {
        for (i = 0; i < numOfChan; i++) {
            sample = Math.max(-1, Math.min(1, channels[i][pos]));
            
            if (formatChoice === "pcm24") {
                let s24 = Math.floor(sample < 0 ? sample * 8388608 : sample * 8388607);
                view.setUint8(offset, s24 & 0xFF);
                view.setUint8(offset + 1, (s24 >> 8) & 0xFF);
                view.setUint8(offset + 2, (s24 >> 16) & 0xFF);
            } else if (formatChoice === "float32") {
                view.setFloat32(offset, sample, true);
            } else {
                let s16 = Math.floor(sample < 0 ? sample * 0x8000 : sample * 0x7FFF);
                view.setInt16(offset, s16, true);
            }
            offset += bytesPerSample;
        }
        pos++;
    }

    return new Blob([bufferArr], { type: "audio/wav" });

    function setUint16(data) {
        view.setUint16(offset, data, true);
        offset += 2;
    }

    function setUint32(data) {
        view.setUint32(offset, data, true);
        offset += 4;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  UI Controls & Event Bindings
// ─────────────────────────────────────────────────────────────────────────────

const btnPlay = document.getElementById("btn-play");
const btnStop = document.getElementById("btn-stop");
const btnExport = document.getElementById("btn-export");
const volSlider = document.getElementById("volume-slider");
const volPercent = document.getElementById("vol-percent");
const timelineSlider = document.getElementById("timeline-slider");
const selectTheater = document.getElementById("select-theater");
const toggleReflections = document.getElementById("toggle-reflections");
const toggleRays = document.getElementById("toggle-rays");
const reverbWetSlider = document.getElementById("reverb-wet-slider");
const reverbWetVal = document.getElementById("reverb-wet-val");
const wallDampingSlider = document.getElementById("wall-damping-slider");
const wallDampingVal = document.getElementById("wall-damping-val");

if (btnPlay) btnPlay.addEventListener("click", () => {
    if (isPlaying) pauseStems();
    else playStems(pausedTime);
});

if (btnStop) btnStop.addEventListener("click", () => stopStems());
if (btnExport) btnExport.addEventListener("click", () => exportMix());

if (volSlider) {
    volSlider.addEventListener("input", () => {
        const val = volSlider.value;
        volPercent.innerText = `${val}%`;
        if (masterGain) masterGain.gain.value = val / 100;
    });
}

if (timelineSlider) {
    timelineSlider.addEventListener("input", () => {
        const seekTime = parseFloat(timelineSlider.value);
        document.getElementById("time-current").innerText = formatTime(seekTime);
    });

    timelineSlider.addEventListener("change", () => {
        const seekTime = parseFloat(timelineSlider.value);
        if (isPlaying) playStems(seekTime);
        else pausedTime = seekTime;
    });
}

// Dynamic Hall Controls & Presets
const sliderHallWidth = document.getElementById("slider-hall-width");
const sliderHallDepth = document.getElementById("slider-hall-depth");
const sliderHallHeight = document.getElementById("slider-hall-height");
const valHallWidth = document.getElementById("val-hall-width");
const valHallDepth = document.getElementById("val-hall-depth");
const valHallHeight = document.getElementById("val-hall-height");
const selectCamMode = document.getElementById("select-cam-mode");
const btnFullscreenStage = document.getElementById("btn-fullscreen-stage");
const stageViewportContainer = document.getElementById("stage-viewport-container");

function updateDynamicHallUIFromPreset() {
    const dims = getHallDimensions();
    customHallWidth = dims.width;
    customHallDepth = dims.depth;
    customHallHeight = dims.height;

    if (sliderHallWidth) sliderHallWidth.value = dims.width;
    if (sliderHallDepth) sliderHallDepth.value = dims.depth;
    if (sliderHallHeight) sliderHallHeight.value = dims.height;
    if (valHallWidth) valHallWidth.innerText = `${dims.width}m`;
    if (valHallDepth) valHallDepth.innerText = `${dims.depth}m`;
    if (valHallHeight) valHallHeight.innerText = `${dims.height}m`;
}

if (selectTheater) {
    selectTheater.addEventListener("change", (e) => {
        activeTheaterKey = e.target.value;
        updateDynamicHallUIFromPreset();
        buildTheaterGeometry();
        updateReverbIR();
    });
}

function handleDynamicSliderChange() {
    activeTheaterKey = 'custom';
    if (selectTheater) selectTheater.value = 'custom';

    if (sliderHallWidth) {
        customHallWidth = parseFloat(sliderHallWidth.value);
        if (valHallWidth) valHallWidth.innerText = `${customHallWidth}m`;
    }
    if (sliderHallDepth) {
        customHallDepth = parseFloat(sliderHallDepth.value);
        if (valHallDepth) valHallDepth.innerText = `${customHallDepth}m`;
    }
    if (sliderHallHeight) {
        customHallHeight = parseFloat(sliderHallHeight.value);
        if (valHallHeight) valHallHeight.innerText = `${customHallHeight}m`;
    }

    buildTheaterGeometry();
    updateReverbIR();
}

if (sliderHallWidth) sliderHallWidth.addEventListener("input", handleDynamicSliderChange);
if (sliderHallDepth) sliderHallDepth.addEventListener("input", handleDynamicSliderChange);
if (sliderHallHeight) sliderHallHeight.addEventListener("input", handleDynamicSliderChange);

if (selectCamMode) {
    selectCamMode.addEventListener("change", (e) => {
        cameraMode = e.target.value;
    });
}

if (btnFullscreenStage && stageViewportContainer) {
    btnFullscreenStage.addEventListener("click", () => {
        stageViewportContainer.classList.toggle("fullscreen");
        const isFS = stageViewportContainer.classList.contains("fullscreen");
        btnFullscreenStage.innerText = isFS ? "✕ Exit Fullscreen" : "⛶ Fullscreen";

        setTimeout(() => {
            const container = document.getElementById("canvas-container");
            if (container && renderer && camera) {
                const w = container.clientWidth;
                const h = container.clientHeight;
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
                renderer.setSize(w, h);
            }
        }, 100);
    });
}

// Wall Reflection & Acoustic Controls Binding
if (toggleReflections) {
    toggleReflections.addEventListener("change", (e) => {
        enableReflections = e.target.checked;
        if (reverbGainNode) reverbGainNode.gain.value = enableReflections ? reverbWetMix : 0.0;
    });
}

if (toggleRays) {
    toggleRays.addEventListener("change", (e) => {
        enableRays = e.target.checked;
    });
}

if (reverbWetSlider) {
    reverbWetSlider.addEventListener("input", (e) => {
        const val = e.target.value;
        reverbWetVal.innerText = `${val}%`;
        reverbWetMix = val / 100;
        if (dryGainNode) dryGainNode.gain.value = 1.0 - reverbWetMix;
        if (reverbGainNode && enableReflections) reverbGainNode.gain.value = reverbWetMix;
    });
}

if (wallDampingSlider) {
    wallDampingSlider.addEventListener("input", (e) => {
        const val = e.target.value;
        wallDampingVal.innerText = `${val}%`;
        wallAbsorption = val / 100;
    });
}

// Stem Mixer Bindings
activeStemsList.forEach(stem => {
    const sBtn = document.getElementById(`solo-${stem}`);
    const mBtn = document.getElementById(`mute-${stem}`);
    const vSld = document.getElementById(`vol-${stem}`);
    const vPct = document.getElementById(`vol-pct-${stem}`);

    if (sBtn) {
        sBtn.addEventListener("click", () => {
            stemSolos[stem] = !stemSolos[stem];
            sBtn.classList.toggle("active", stemSolos[stem]);
            updateStemGains();
        });
    }

    if (mBtn) {
        mBtn.addEventListener("click", () => {
            stemMutes[stem] = !stemMutes[stem];
            mBtn.classList.toggle("active", stemMutes[stem]);
            updateStemGains();
        });
    }

    if (vSld) {
        vSld.addEventListener("input", () => {
            const val = vSld.value;
            vPct.innerText = `${val}%`;
            stemVolumes[stem] = val / 100;
            updateStemGains();
        });
    }

    const tSld = document.getElementById(`tone-${stem}`);
    const tPct = document.getElementById(`tone-pct-${stem}`);
    if (tSld) {
        tSld.addEventListener("input", () => {
            const freq = parseFloat(tSld.value);
            stemTones[stem] = freq;
            tPct.innerText = freq >= 1000 ? `${(freq / 1000).toFixed(1)}kHz` : `${Math.round(freq)}Hz`;
            if (toneFilterNodes[stem] && audioCtx) {
                toneFilterNodes[stem].frequency.setValueAtTime(freq, audioCtx.currentTime);
            }
        });
    }
});

// Initialize 3D Engine on page load
window.addEventListener("DOMContentLoaded", () => {
    initThree();
});
