import { item2html } from "./lib/render";

declare const jsQR: any;
declare const QRCode: any;

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

log("Main module loaded. V140 (Final Transition Fix) Initializing...");

let videoStream: MediaStream | null = null;
let scanning = false;
let dataChannel: RTCDataChannel | null = null;
let peerConn: RTCPeerConnection | null = null;
let receivedOfferChunks: string[] = [];
let expectedOfferTotal = 0;
let qrInterval: any = null;

const video = document.getElementById("v") as HTMLVideoElement;
const searchInput = document.getElementById("global-search") as HTMLInputElement;
const tabContents = document.querySelectorAll<HTMLElement>(".tab-content");
const listView = document.getElementById("list-view") as HTMLElement;
const detailView = document.getElementById("detail-view") as HTMLElement;
const chatForm = document.querySelector('.chat-form') as HTMLFormElement;
const chatTalks = document.querySelector('.chat-talks') as HTMLElement;
const chatScroll = document.getElementById("chat-scroll") as HTMLElement;

// --- UI Logic ---
function switchTab(tabName: string) {
    log(`Switch Tab -> ${tabName}`);
    tabContents.forEach(c => {
        if (c.id === `tab-${tabName}`) {
            c.classList.add("active");
            c.style.display = 'flex';
        } else {
            c.classList.remove("active");
            c.style.display = 'none';
        }
    });
}

function showDetailView(show: boolean) {
    if (show) {
        listView.style.display = "none";
        detailView.style.display = "flex";
    } else {
        listView.style.display = "flex";
        detailView.style.display = "none";
    }
}

function finalizeConnectionUI() {
    log("🚀 Finalizing Connection UI...");
    
    // Hide all connection overlays
    const intro = document.getElementById("mobile-intro-overlay");
    if (intro) intro.style.display = 'none';
    
    document.getElementById("answer-qr-container")!.style.display = 'none';
    if (qrInterval) { clearInterval(qrInterval); qrInterval = null; }

    // Enable Search and UI
    if (searchInput) {
        searchInput.disabled = false;
        searchInput.placeholder = "Search or Ask Prompt";
    }
    document.querySelectorAll(".nav-icons .nav-btn").forEach(btn => (btn as HTMLButtonElement).disabled = false);

    // Switch to List Tab
    switchTab("list");
}

// --- Scanning ---
async function startScanning() {
    if (scanning) return;
    receivedOfferChunks = []; expectedOfferTotal = 0;
    if (qrInterval) { clearInterval(qrInterval); qrInterval = null; }
    try {
        const constraints = { video: { facingMode: "environment" } };
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = videoStream;
        await video.play();
        scanning = true;
        document.getElementById("scanner-overlay")!.style.display = 'block';
        requestAnimationFrame(tick);
    } catch (err) { log(`Camera Err: ${err}`); }
}

function stopScanning() {
    scanning = false;
    if (videoStream) videoStream.getTracks().forEach(t => t.stop());
    document.getElementById("scanner-overlay")!.style.display = 'none';
}

function tick() {
    if (!scanning) return;
    if (video.readyState === video.HAVE_ENOUGH_DATA) {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const code = jsQR(imgData.data, imgData.width, imgData.height);
            if (code) handleQrChunk(code.data);
        }
    }
    requestAnimationFrame(tick);
}

function handleQrChunk(data: string) {
    try {
        const parsed = JSON.parse(data);
        if (Array.isArray(parsed) && parsed.length === 3) {
            const [idx, total, chunk] = parsed;
            if (expectedOfferTotal === 0) {
                expectedOfferTotal = total;
                receivedOfferChunks = new Array(total).fill("");
            }
            if (!receivedOfferChunks[idx]) {
                receivedOfferChunks[idx] = chunk;
                log(`Offer ${idx+1}/${total}`);
                const introH3 = document.querySelector("#mobile-intro-overlay h3");
                if (introH3) introH3.textContent = `Scanning... ${receivedOfferChunks.filter(c => c).length}/${total}`;
            }
            if (receivedOfferChunks.every(c => c !== "")) {
                stopScanning();
                createPeerConnection(receivedOfferChunks.join(""));
            }
        }
    } catch(e) {}
}

// --- WebRTC ---
async function createPeerConnection(sdp: string) {
    peerConn = new RTCPeerConnection({ iceServers: [] });
    
    peerConn.ondatachannel = (e) => {
        dataChannel = e.channel;
        setupDataChannel(dataChannel);
    };

    peerConn.oniceconnectionstatechange = () => {
        const state = peerConn?.iceConnectionState;
        log(`ICE: ${state}`);
        if(state === 'connected' || state === 'completed') {
            finalizeConnectionUI();
        }
    };

    try {
        await peerConn.setRemoteDescription(new RTCSessionDescription({ type: 'offer', sdp: sdp }));
        const answer = await peerConn.createAnswer();
        await peerConn.setLocalDescription(answer);
        
        // Wait for ICE gathering
        await new Promise<void>(resolve => {
            const pc = peerConn!;
            if (pc.iceGatheringState === 'complete') resolve();
            else {
                const check = () => { if (pc.iceGatheringState === 'complete') { pc.removeEventListener('icegatheringstatechange', check); resolve(); } };
                pc.addEventListener('icegatheringstatechange', check);
                setTimeout(resolve, 5000); 
            }
        });

        const fullSdp = peerConn.localDescription!.sdp;
        const CHUNK_COUNT = 4;
        const size = Math.ceil(fullSdp.length / CHUNK_COUNT);
        const chunks = [];
        for(let i=0; i<CHUNK_COUNT; i++) {
            chunks.push(JSON.stringify([i, CHUNK_COUNT, fullSdp.substring(i*size, (i+1)*size)]));
        }
        showAnswerSlideQr(chunks);

    } catch (err: any) {
        log("WebRTC Error: " + err);
    }
}

function showAnswerSlideQr(chunks: string[]) {
    const container = document.getElementById("answer-qr-container")!;
    container.style.display = 'flex';
    const qrDiv = document.getElementById("answer-qr")!;
    let cur = 0;
    const rotate = () => {
        qrDiv.innerHTML = `<div style="font-weight:bold; margin-bottom:10px;">Answer ${cur+1}/${chunks.length}</div>`;
        const q = document.createElement("div");
        qrDiv.appendChild(q);
        new QRCode(q, { text: chunks[cur], width: 250, height: 250, correctLevel: 0 });
        cur = (cur + 1) % chunks.length;
    };
    if (qrInterval) clearInterval(qrInterval);
    rotate();
    qrInterval = setInterval(rotate, 1000);
}

function setupDataChannel(channel: RTCDataChannel) {
    channel.onopen = () => {
        log("Channel OPEN!");
        finalizeConnectionUI(); // Ensure UI switches even if ICE event was messy
        channel.send(JSON.stringify({ type: "search", query: "" }));
    };
    channel.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "sync_list") renderList(msg.data);
        else if (msg.type === "sync_detail") renderDetail(msg.title, msg.content);
        else if (msg.type === "sync_chat") renderChat(msg.data);
    };
}

function renderList(items: any[]) {
    const list = document.getElementById("doc-list");
    if (!list) return;
    list.innerHTML = items.map(item => item2html(item, false)).join("");
    list.querySelectorAll('.logis-result').forEach(el => {
        el.addEventListener("click", () => {
            if (dataChannel?.readyState === "open") {
                dataChannel.send(JSON.stringify({ type: "get_detail", uuid: el.id }));
            }
        });
    });
}

function renderDetail(title: string, content: string) {
    document.getElementById("detail-title")!.innerText = title;
    document.getElementById("detail-content")!.innerHTML = content;
    showDetailView(true);
}

function renderChat(data: any) {
    const div = document.createElement("div");
    div.className = `chat-talk ${data.role === 'user' ? 'user' : 'system'}`;
    div.innerHTML = `<div class="chat-message"><div class="content">${data.content}</div></div>`;
    chatTalks.appendChild(div);
    chatScroll.scrollTop = chatScroll.scrollHeight;
}

searchInput?.addEventListener("input", (e) => {
    const query = (e.target as HTMLInputElement).value;
    if (dataChannel?.readyState === "open") dataChannel.send(JSON.stringify({ type: "search", query: query }));
});
chatForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    const input = chatForm.querySelector('input[name="talk"]') as HTMLInputElement;
    if (input.value.trim()) {
        renderChat({ role: 'user', content: input.value });
        dataChannel?.send(JSON.stringify({ type: "chat_message", content: input.value }));
        input.value = "";
    }
});
document.getElementById("btn-settings")?.addEventListener("click", () => switchTab("settings"));
document.getElementById("btn-settings-back")?.addEventListener("click", () => switchTab("list"));
document.getElementById("btn-detail-back")?.addEventListener("click", () => showDetailView(false));
document.getElementById("list-refresh-btn")?.addEventListener("click", () => {
    if (dataChannel?.readyState === "open") dataChannel.send(JSON.stringify({ type: "search", query: searchInput?.value || "" }));
});
window.addEventListener("request-camera-start", () => startScanning());
window.addEventListener("request-camera-stop", () => stopScanning());