// --- Mobile Main (Desktop Parity) ---
const ethers = (window as any).ethers;
const blockies = (window as any).blockies;

interface ChatSession {
    hash: string;
    email?: string;
    address?: string;
}

let currentSession: ChatSession = { hash: "" };

// --- UI Elements ---
const btnSyncQr = document.getElementById("btn-sync-qr");
const btnScanPc = document.getElementById("btn-scan-pc");
const qrContainer = document.getElementById("nav-qr-container");
const qrTarget = document.getElementById("sync-qrcode");
const cameraContainer = document.getElementById("camera-container");
const videoEl = document.getElementById("camera-video") as HTMLVideoElement;
const btnCancelScan = document.getElementById("btn-cancel-scan");
const profileName = document.getElementById("nav-profile-name");
const profileFavicon = document.getElementById("nav-profile-favicon");

let stream: MediaStream | null = null;

// --- Initialization ---
async function initSession() {
    const saved = localStorage.getItem("chat_session");
    if (saved) {
        try { currentSession = JSON.parse(saved); } catch (e) {}
    }

    if (!currentSession.hash && ethers) {
        const w = ethers.Wallet.createRandom();
        currentSession.hash = w.address.toLowerCase().replace("0x", "");
        currentSession.address = w.address;
        localStorage.setItem("chat_session", JSON.stringify(currentSession));
    }
    
    updateProfileUI();
    console.log("[Mobile] Session initialized:", currentSession.hash);
}

function updateProfileUI() {
    if (currentSession.email) {
        if (profileName) profileName.innerText = currentSession.email.split('@')[0];
        if (profileFavicon && blockies) {
            const icon = blockies.create({ seed: currentSession.email, size: 8, scale: 4 });
            profileFavicon.innerHTML = ""; profileFavicon.appendChild(icon);
        }
    }
}

// --- QR & Scanning Logic ---
async function startScanning() {
    if (!cameraContainer || !videoEl) return;
    
    qrContainer?.classList.add("hidden");
    cameraContainer.classList.remove("hidden");

    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        videoEl.srcObject = stream;
        videoEl.play();
        
        if ('BarcodeDetector' in window) {
            const detector = new (window as any).BarcodeDetector({ formats: ['qr_code'] });
            const scanLoop = async () => {
                if (!stream) return;
                try {
                    const barcodes = await detector.detect(videoEl);
                    if (barcodes.length > 0) {
                        const data = barcodes[0].rawValue;
                        console.log("[Mobile] QR Detected:", data);
                        if (data.includes("webrtc-pair")) {
                            stopScanning();
                            showMobileQr();
                            return;
                        }
                    }
                } catch (e) { console.error("Detection error:", e); }
                requestAnimationFrame(scanLoop);
            };
            requestAnimationFrame(scanLoop);
        } else {
            // Fallback for non-supported browsers
            setTimeout(() => { stopScanning(); showMobileQr(); }, 3000);
        }
    } catch (e) {
        console.error("Camera error:", e);
        stopScanning();
    }
}

function stopScanning() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    cameraContainer?.classList.add("hidden");
}

function showMobileQr() {
    if (!qrContainer || !qrTarget) return;
    qrContainer.classList.remove("hidden");
    qrTarget.innerHTML = "";
    const pairingData = JSON.stringify({
        type: "webrtc-pair",
        hash: currentSession.hash,
        ts: Date.now(),
        role: "mobile"
    });
    new (window as any).QRCode(qrTarget, {
        text: pairingData,
        width: 180, height: 180, colorDark: "#000000", colorLight: "#ffffff",
        correctLevel: (window as any).QRCode.CorrectLevel.M
    });
}

// --- Event Listeners ---
btnScanPc?.addEventListener("click", startScanning);
btnCancelScan?.addEventListener("click", stopScanning);

btnSyncQr?.addEventListener("click", () => {
    const isHidden = qrContainer?.classList.contains("hidden");
    if (isHidden) showMobileQr();
    else qrContainer?.classList.add("hidden");
});

document.getElementById("btn-capture")?.addEventListener("click", () => {
    alert("Mobile Capture Feature - Pending WebRTC Link");
});

// Init
initSession();