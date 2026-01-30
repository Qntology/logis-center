

// Import styles to ensure Vite bundles them
import "../styles.css";

// Declare global types for external libraries loaded via <script>
declare const jsQR: any;
declare const pako: any;
declare const QRCode: any;

// Utility to write to the log panel in HTML
function log(msg: string) {
    console.log(`[Main] ${msg}`);
    const panel = document.getElementById("log-panel");
    if (panel) {
        const line = document.createElement("div");
        line.textContent = `> ${msg}`;
        panel.appendChild(line);
        panel.scrollTop = panel.scrollHeight;
    }
}

log("Main module loaded. Initializing...");

// State
let videoStream: MediaStream | null = null;
let scanning = false;

// DOM Elements
const video = document.getElementById("v") as HTMLVideoElement;
const overlay = document.getElementById("scanner-overlay");

// --- Camera & Scanning Logic ---

async function startScanning() {
    if (scanning) return;
    log("Starting scanning sequence...");

    // Reset Chunk Buffer for new session
    receivedOfferChunks = [];
    expectedOfferTotal = 0;

    if (!video) {
        log("Error: Video element not found");
        return;
    }

    try {
        // 1. Request Camera
        const constraints = { video: { facingMode: "environment" } };
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = videoStream;
        
        // 2. Play Video
        // Required for iOS/Android to actually start playing
        video.setAttribute("playsinline", "true"); 
        await video.play();
        
        scanning = true;
        log("Camera active. Scanning for QR...");
        
        requestAnimationFrame(tick);
    } catch (err: any) {
        log(`Camera Error: ${err.message || err}`);
        alert("Camera failed: " + err);
    }
}

function stopScanning() {
    log("Stopping camera...");
    scanning = false;
    
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    
    if (video) {
        video.pause();
        video.srcObject = null;
    }
}

function tick() {
    if (!scanning) return;
    
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        // Create a temporary canvas to draw the video frame for analysis
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        
        if (ctx) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            
            // Attempt to find QR code
            if (typeof jsQR !== 'undefined') {
                const code = jsQR(imageData.data, imageData.width, imageData.height, {
                    inversionAttempts: "dontInvert",
                });

                if (code) {
                    // Log throttle to avoid spamming the UI on every frame
                    // log("QR Found: " + code.data.substring(0, 10) + "...");
                    
                    handleQrCode(code.data);
                    
                    // [Multipart Fix] Do NOT stop scanning here. 
                    // handleQrCode will call stopScanning() when all chunks are gathered.
                    
                    // Just return to avoid re-scanning the same frame immediately in this tick
                    // But we want to continue the loop
                }
            } else {
                // Throttle this log so it doesn't spam
                if (Math.random() < 0.01) log("Warning: jsQR not loaded");
            }
        }
    }
    
    requestAnimationFrame(tick);
}

// Chunk Buffer State
let receivedOfferChunks: string[] = [];
let expectedOfferTotal = 0;

function handleQrCode(data: string) {
    // 1. Check for Chunk Format: [index, total, chunk]
    try {
        if (data.trim().startsWith("[")) {
             const parsed = JSON.parse(data);
             if (Array.isArray(parsed) && parsed.length === 3 && typeof parsed[0] === 'number') {
                 const [idx, total, chunk] = parsed;
                 
                 // Init buffer
                 if (expectedOfferTotal === 0) {
                     expectedOfferTotal = total;
                     for(let i=0; i<total; i++) receivedOfferChunks.push("");
                     log(`Detected Multipart QR. Total: ${total}`);
                 }

                 // Store chunk
                 if (total === expectedOfferTotal && !receivedOfferChunks[idx]) {
                     receivedOfferChunks[idx] = chunk;
                     const count = receivedOfferChunks.filter(c => c).length;
                     log(`Received Part ${idx + 1}/${total}`);
                     
                     // Feedback
                     const statusArea = document.getElementById("status-area");
                     if (statusArea) {
                         statusArea.innerHTML = `<p style="font-size:3rem;">📡</p><p>Scanning... ${count}/${total}</p>`;
                     }
                 }

                 // Check Completion
                 if (receivedOfferChunks.every(c => c !== "")) {
                     log("All chunks received. Processing Offer...");
                     const fullSdp = receivedOfferChunks.join("");
                     
                     // Stop scanning and cleanup
                     stopScanning();
                     if (overlay) overlay.style.display = 'none';
                     
                     createPeerConnection({ type: 'offer', sdp: fullSdp });
                 }
                 return; // Keep scanning if not complete
             }
             // Legacy single array check could go here
        }
    } catch (e) {
        // Ignore parse errors, just keep scanning
    }
}

async function createPeerConnection(offer: any) {
    // 2. Create PC (No STUN needed for local network, or use default)
    const pc = new RTCPeerConnection({
        iceServers: [] // Local network only
    });
    
    // Log ICE Candidates for debugging
    pc.onicecandidate = (event) => {
        if (event.candidate) log(`ICE: ${event.candidate.candidate.substring(0, 20)}...`);
    };

    pc.oniceconnectionstatechange = () => {
        log(`ICE State: ${pc.iceConnectionState}`);
        if(pc.iceConnectionState === 'connected') {
            document.getElementById("answer-qr-container")!.style.display = 'none';
            const status = document.getElementById("status-area");
            if(status) status.innerHTML = `<p style="font-size:3rem;">🚀</p><p>Connected!</p>`;
        }
    };

    try {
        await pc.setRemoteDescription(new RTCSessionDescription(offer));
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);

        log("Gathering ICE candidates...");
        // Wait for ICE gathering to complete (max 2s) to ensure local IP is included
        await new Promise<void>(resolve => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const check = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', check);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', check);
                setTimeout(resolve, 2000); 
            }
        });

        // 3. Generate Answer QR (Multipart/Chunked)
        const sdp = pc.localDescription!.sdp;
        const CHUNK_COUNT = 4;
        const chunkSize = Math.ceil(sdp.length / CHUNK_COUNT);
        const chunks: string[] = [];
        
        for (let i = 0; i < CHUNK_COUNT; i++) {
            const chunk = sdp.substring(i * chunkSize, (i + 1) * chunkSize);
            chunks.push(JSON.stringify([i, CHUNK_COUNT, chunk]));
        }
        
        log(`Answer created. Split into ${CHUNK_COUNT} chunks. Displaying...`);
        
        showAnswerQrChunks(chunks);

    } catch (err: any) {
        log("WebRTC Error: " + err.message);
        alert("Connection Failed: " + err.message);
    }
}

function showAnswerQrChunks(chunks: string[]) {
    const container = document.getElementById("answer-qr-container");
    const qrDiv = document.getElementById("answer-qr");
    
    if (container && qrDiv) {
        container.style.display = "flex";
        
        let currentIdx = 0;
        const rotate = () => {
            qrDiv.innerHTML = "";
            
            // Header
            const h = document.createElement("div");
            h.innerText = `Part ${currentIdx + 1}/${chunks.length}`;
            h.style.fontWeight = "bold"; h.style.marginBottom = "10px";
            qrDiv.appendChild(h);

            // QR
            const q = document.createElement("div");
            qrDiv.appendChild(q);
            
            try {
                new QRCode(q, {
                    text: chunks[currentIdx],
                    width: 250, height: 250,
                    correctLevel: QRCode.CorrectLevel.L
                });
            } catch(e) {}
            
            currentIdx = (currentIdx + 1) % chunks.length;
        };
        
        rotate();
        setInterval(rotate, 700);
    }
}

// Deprecated function removed to fix build error

function showAnswerQr(data: string) {
    const container = document.getElementById("answer-qr-container");
    const qrDiv = document.getElementById("answer-qr");
    
    if (container && qrDiv) {
        qrDiv.innerHTML = ""; 
        try {
            new QRCode(qrDiv, {
                text: data,
                width: 250,
                height: 250,
                correctLevel: QRCode.CorrectLevel.L
            });
            container.style.display = "flex";
            log("Displaying Answer QR. Scan this with Laptop.");
        } catch(e) {
            log("QR Gen Error: " + e);
        }
    }
}

// --- Event Listeners ---

// Listen for events dispatched from index.html
window.addEventListener("request-camera-start", () => {
    startScanning();
});

window.addEventListener("request-camera-stop", () => {
    stopScanning();
});

log("Event listeners ready.");
