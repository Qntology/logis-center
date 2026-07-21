// --- Mobile Main v2 (Stability Fix) ---
const ethers = window.ethers;
const blockies = window.blockies;

function log(msg, color = "#4ade80") {
    const el = document.getElementById("mobile-debug-log");
    if (el) {
        el.style.color = color;
        el.innerText = "[" + new Date().toLocaleTimeString() + "] " + msg;
        console.log("[Mobile-App] " + msg);
    }
}

let currentSession = { hash: "" };

const btnSyncQr = document.getElementById("btn-sync-qr");
const btnScanPc = document.getElementById("btn-scan-pc");
const qrContainer = document.getElementById("nav-qr-container");
const qrTarget = document.getElementById("sync-qrcode");
const cameraContainer = document.getElementById("camera-container");
const videoEl = document.getElementById("camera-video");
const btnCancelScan = document.getElementById("btn-cancel-scan");
const profileName = document.getElementById("nav-profile-name");
const profileFavicon = document.getElementById("nav-profile-favicon");

let stream = null;

async function initSession() {
    log("BUILD_VER_105: Initializing...");
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
    log("ID: " + currentSession.hash.substring(0, 8));
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

async function startScanning() {
    if (!cameraContainer || !videoEl) return;
    
    log("Initializing Camera...", "#fb923c");
    qrContainer.classList.add("hidden");
    cameraContainer.style.display = "block";

    try {
        if (stream) stopScanning(); // Clean up existing

        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: "environment",
                width: { ideal: 640 }, 
                height: { ideal: 480 } // Use lower resolution for stability
            } 
        });
        
        videoEl.srcObject = stream;
        videoEl.setAttribute("playsinline", "true");
        
        await videoEl.play();
        log("Camera Active", "#4ade80");
        
        // Give some time for video to settle
        setTimeout(startDetectionLoop, 1000);

    } catch (e) { 
        log("ERR: " + e.message, "#ef4444");
        // Don't stopScanning immediately, let user see error
    }
}

function startDetectionLoop() {
    if (!stream) return;

    if ('BarcodeDetector' in window) {
        log("Native Scanner ON", "#4ade80");
        const detector = new window.BarcodeDetector({ formats: ['qr_code'] });
        
        const scan = async () => {
            if (!stream) return;
            try {
                const barcodes = await detector.detect(videoEl);
                if (barcodes.length > 0) {
                    const data = barcodes[0].rawValue;
                    if (data.includes("webrtc-pair")) {
                        log("PC FOUND!", "#fb923c");
                        if (navigator.vibrate) navigator.vibrate(200);
                        stopScanning();
                        showMobileQr();
                        return;
                    }
                }
            } catch (e) { /* silent scan error */ }
            requestAnimationFrame(scan);
        };
        requestAnimationFrame(scan);
    } else {
        log("Scanner API Not Supported", "#aaa");
        // Stay in viewfinder mode anyway so user can see what's happening
    }
}

function stopScanning() {
    log("Stopping Camera", "#aaa");
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    if (videoEl) videoEl.srcObject = null;
    cameraContainer.style.display = "none";
}

function showMobileQr() {
    if (!qrContainer || !qrTarget) return;
    
    // Hide UI
    document.querySelectorAll(".nav-profile, .nav-section, .card").forEach(el => el.classList.add("hidden"));

    qrContainer.classList.remove("hidden");
    qrContainer.style.display = "block";
    qrTarget.innerHTML = "";
    
    const pairingData = JSON.stringify({
        type: "webrtc-pair",
        hash: currentSession.hash,
        ts: Date.now(),
        role: "mobile"
    });
    
    new window.QRCode(qrTarget, {
        text: pairingData,
        width: 200, height: 200, colorDark: "#000000", colorLight: "#ffffff",
        correctLevel: window.QRCode.CorrectLevel.H
    });
    
    log("PAIRING QR READY", "#4ade80");
}

btnScanPc.addEventListener("click", startScanning);
btnCancelScan.addEventListener("click", stopScanning);
btnSyncQr.addEventListener("click", () => {
    const isHidden = qrContainer.classList.contains("hidden");
    if (isHidden) showMobileQr();
    else {
        qrContainer.classList.add("hidden");
        document.querySelectorAll(".nav-profile, .nav-section, .card").forEach(el => el.classList.remove("hidden"));
    }
});

document.getElementById("btn-capture").addEventListener("click", () => {
    log("No Link - Pairing required", "#ef4444");
});

initSession();
