console.log("%c[MOBILE] V122 Fixed System Online", "color: #fb923c; font-weight: bold;");

// @ts-ignore
import { item2html } from "./lib/render";

// --- Global State ---
let peerConn: RTCPeerConnection | null = null;
let dataChannel: RTCDataChannel | null = null;
const API_HOST = "https://commerce.logis.center";

// --- UI Elements ---
const docListContainer = document.getElementById("doc-list") as HTMLElement;
const tabContents = document.querySelectorAll(".tab-content");
const handshakeLayer = document.getElementById("handshake-layer") as HTMLElement;
const connStatusText = document.getElementById("conn-status-text");
const connIndicator = document.getElementById("conn-indicator");
const detailView = document.getElementById("detail-view") as HTMLElement;
const detailTitle = document.getElementById("detail-title") as HTMLElement;
const detailContent = document.getElementById("detail-content") as HTMLElement;

// --- WebRTC Core ---
async function initWebRTC(sessionHash: string) {
    console.log("[WebRTC] Initializing with Session:", sessionHash);
    updateConnStatus(false, "NEGOTIATING...");

    peerConn = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
    });

    peerConn.onicecandidate = (e) => {
        if (e.candidate) {
            sendSignaling(sessionHash, { type: "ice", candidate: e.candidate });
        }
    };

    peerConn.ondatachannel = (event) => {
        console.log("[WebRTC] Data Channel Received");
        dataChannel = event.channel;
        setupDataChannel(dataChannel);
    };

    startOfferPoll(sessionHash);
}

function setupDataChannel(channel: RTCDataChannel) {
    channel.onopen = () => {
        console.log("[WebRTC] Data Channel OPEN!");
        updateConnStatus(true, "LIVE P2P LINKED");
        handshakeLayer.classList.add("hidden");
    };
    channel.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "sync_list") renderDocs(msg.data);
        else if (msg.type === "sync_detail") showDetailUI(msg.title, msg.content);
        else if (msg.type === "sync_chat") renderChatMessage(msg.data);
    };
}

async function startOfferPoll(hash: string) {
    const poll = async () => {
        if (peerConn?.remoteDescription) return;
        try {
            const resp = await fetch(`${API_HOST}/relay/${hash}`);
            const data = await resp.json();
            if (data && data.type === "offer") {
                await peerConn?.setRemoteDescription(new RTCSessionDescription(data.sdp));
                const answer = await peerConn?.createAnswer();
                await peerConn?.setLocalDescription(answer);
                await sendSignaling(hash, { type: "answer", sdp: answer });
            }
        } catch (e) {}
        setTimeout(poll, 2000);
    };
    poll();
}

async function sendSignaling(hash: string, payload: any) {
    try {
        await fetch(`${API_HOST}/relay/${hash}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
    } catch (e) {}
}

// --- Interaction ---
function renderDocs(docs: any[]) {
    docListContainer.innerHTML = "";
    if (docs.length === 0) {
        docListContainer.innerHTML = `<div style="text-align:center; padding:40px; color:#666;">No documents found.</div>`;
        return;
    }
    docs.forEach(doc => {
        const html = item2html(doc, false, "");
        const temp = document.createElement("div"); temp.innerHTML = html;
        const el = temp.firstElementChild as HTMLElement;
        el.addEventListener("click", () => {
            if (dataChannel?.readyState === "open") {
                dataChannel.send(JSON.stringify({ type: "get_detail", uuid: doc.uuid || doc.id }));
            }
        });
        docListContainer.appendChild(el);
    });
}

function showDetailUI(title: string, content: string) {
    detailView.style.display = "flex";
    detailTitle.textContent = title;
    detailContent.innerHTML = content;
}

function renderChatMessage(msg: any) {
    const talks = document.querySelector(".chat-talks");
    if (!talks) return;
    const div = document.createElement("div");
    div.className = `chat-talk ${msg.role === 'user' ? 'user' : 'system'}`;
    div.innerHTML = `<div class="chat-message">${msg.content}</div>`;
    talks.appendChild(div);
    talks.parentElement?.scrollTo({ top: talks.parentElement.scrollHeight, behavior: 'smooth' });
}

function updateConnStatus(linked: boolean, text: string) {
    if (connIndicator) connIndicator.style.background = linked ? "#4ade80" : "#fb923c";
    if (connStatusText) connStatusText.textContent = text;
}

function switchTab(name: string) {
    tabContents.forEach(c => c.classList.toggle("active", c.id === `tab-${name}`));
}

// --- Scanner ---
let scanStream: MediaStream | null = null;
const video = document.getElementById("v") as HTMLVideoElement;
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

async function startScan() {
    try {
        const overlay = document.getElementById("scanner-overlay");
        overlay?.classList.remove("hidden");
        scanStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        video.srcObject = scanStream; video.play();
        requestAnimationFrame(scanLoop);
    } catch (e) { alert("Camera Denied"); }
}

function stopScan() { 
    if (scanStream) scanStream.getTracks().forEach(t => t.stop()); 
    scanStream = null; 
    document.getElementById("scanner-overlay")?.classList.add("hidden"); 
}

function scanLoop() {
    if (!scanStream) return;
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);
        const img = ctx?.getImageData(0, 0, canvas.width, canvas.height);
        // @ts-ignore
        if (img) { const code = jsQR(img.data, img.width, img.height); if (code && code.data.includes("webrtc-pair")) { 
            stopScan(); handleScannedQR(code.data); return; 
        } }
    }
    requestAnimationFrame(scanLoop);
}

function handleScannedQR(dataStr: string) {
    const data = JSON.parse(dataStr);
    initWebRTC(data.hash);
    showHandshakeUI(data.hash);
}

function showHandshakeUI(hash: string) {
    handshakeLayer.classList.remove("hidden");
    const target = document.getElementById("qr-target");
    if (target) {
        target.innerHTML = "";
        const ansData = JSON.stringify({ type: "webrtc-pair", hash: hash, role: "mobile" });
        // @ts-ignore
        new QRCode(target, { text: ansData, width: 250, height: 250 });
    }
}

// --- Bindings ---
document.getElementById("btn-sync-toggle")?.addEventListener("click", startScan);
document.getElementById("btn-cancel-scan")?.addEventListener("click", stopScan);
document.getElementById("btn-settings")?.addEventListener("click", () => switchTab("settings"));
document.getElementById("btn-settings-back")?.addEventListener("click", () => switchTab("list"));
document.getElementById("btn-detail-back")?.addEventListener("click", () => { detailView.style.display = "none"; });
document.getElementById("btn-close-handshake")?.addEventListener("click", () => { handshakeLayer.classList.add("hidden"); });

const chatForm = document.querySelector("form[name='chat-form']") as HTMLFormElement;
chatForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    const input = chatForm.talk as HTMLInputElement;
    if (!input.value) return;
    renderChatMessage({ role: 'user', content: input.value });
    if (dataChannel?.readyState === "open") {
        dataChannel.send(JSON.stringify({ type: "chat_message", content: input.value }));
    }
    input.value = "";
});

// Boot
updateConnStatus(false, "READY TO LINK");