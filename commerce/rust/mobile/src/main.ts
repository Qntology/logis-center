

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
                    log("QR Found: " + code.data.substring(0, 20) + "...");
                    handleQrCode(code.data);
                    // Stop scanning after successful read (optional, depends on UX)
                    stopScanning();
                    
                    // Hide overlay
                    if (overlay) overlay.style.display = 'none';
                    return; // Stop the loop
                }
            } else {
                // Throttle this log so it doesn't spam
                if (Math.random() < 0.01) log("Warning: jsQR not loaded");
            }
        }
    }
    
    requestAnimationFrame(tick);
}

function handleQrCode(data: string) {
    log("Processing QR Data...");
    // Stop scanning while processing
    stopScanning();
    if (overlay) overlay.style.display = 'none';

    try {
        // 1. Parse Data (Support both JSON and Compressed Base64)
        let offerData: any;
        if (data.trim().startsWith("{")) {
             offerData = JSON.parse(data);
        } else {
             try {
                // Base64 -> Pako Inflate -> JSON
                const binaryString = atob(data);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                const inflated = pako.inflate(bytes, { to: 'string' });
                offerData = JSON.parse(inflated);
                log("Decompressed Offer data.");
             } catch (err) {
                 log("Parsing failed, assuming raw JSON failed too.");
                 throw err;
             }
        }

        log("Offer accepted. Generating Answer...");
        createPeerConnection(offerData);

    } catch (e: any) {
        log("Error processing QR: " + e);
        alert("Invalid QR Code.\nPlease scan a valid WebRTC Offer.");
        // Restart scanning if failed
        startScanning();
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

        // 3. Compress and Display Answer
        const finalAnswer = JSON.stringify(pc.localDescription);
        log("Answer created. Compressing...");
        
        // Pako Deflate -> Base64
        const binaryString = pako.deflate(finalAnswer, { to: 'string' });
        const base64Answer = btoa(binaryString);
        
        showAnswerQr(base64Answer);

    } catch (err: any) {
        log("WebRTC Error: " + err.message);
        alert("Connection Failed: " + err.message);
    }
}

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
