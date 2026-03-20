
import { item2html } from "./lib/render";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

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

listen("webrtc-offer", async (event) => {
    const [offerSdp, fromIp] = event.payload as [string, string];
    log(`Incoming offer from ${fromIp}`);
    
    peerConn = new RTCPeerConnection({ iceServers: [] });
    peerConn.ondatachannel = (e) => {
        dataChannel = e.channel;
        setupDataChannel(dataChannel);
    };

    await peerConn.setRemoteDescription({ type: 'offer', sdp: offerSdp });
    const answer = await peerConn.createAnswer();
    await peerConn.setLocalDescription(answer);
    
    // [FIXED] Send Answer back
    try {
        await invoke("submit_signal_answer", { targetIp: fromIp, sdp: answer.sdp });
        log(`Answer submitted to ${fromIp}`);
    } catch (e) {
        log(`Failed to submit answer: ${e}`);
    }
});

async function startWebRtcOfferer(targetIp: string, seed: number) {
    peerConn = new RTCPeerConnection({ iceServers: [] });
    dataChannel = peerConn.createDataChannel("logis-sync");
    setupDataChannel(dataChannel);
    
    const offer = await peerConn.createOffer();
    await peerConn.setLocalDescription(offer);
    
    // Wait for ICE
    await new Promise<void>(resolve => {
        if (peerConn?.iceGatheringState === 'complete') resolve();
        else peerConn?.addEventListener('icegatheringstatechange', () => {
            if (peerConn?.iceGatheringState === 'complete') resolve();
        });
        setTimeout(resolve, 2000);
    });

    const sdp = peerConn.localDescription?.sdp || "";
    const answerSdp = await invoke<string>("send_signal_offer", { targetIp, seed, sdp });
    await peerConn.setRemoteDescription({ type: 'answer', sdp: answerSdp });
    log("Connected via Seed Handshake!");
}

// --- WebRTC SDP Template for Compact Handshake ---
const SDP_TEMPLATE = `v=0
o=- {{sessId}} 2 IN IP4 {{ip}}
s=-
t=0 0
a=group:BUNDLE 0
a=msid-semantic: WMS
m=application 9 UDP/DTLS/SCTP webrtc-datachannel
c=IN IP4 {{ip}}
a=ice-ufrag:{{ufrag}}
a=ice-pwd:{{pwd}}
a=fingerprint:sha-256 {{fingerprint}}
a=setup:{{setup}}
a=mid:0
a=sctp-port:5000
a=max-message-size:262144`;

function extractSdp(sdp: string) {
    return {
        u: sdp.match(/a=ice-ufrag:(.*)/)?.[1] || "",
        p: sdp.match(/a=ice-pwd:(.*)/)?.[1] || "",
        f: sdp.match(/a=fingerprint:sha-256 (.*)/)?.[1] || "",
        s: sdp.match(/o=- (\d+) /)?.[1] || "0"
    };
}

function buildSdp(type: 'offer' | 'answer', ip: string, u: string, p: string, f: string, s: string) {
    return SDP_TEMPLATE
        .replace(/{{sessId}}/g, s)
        .replace(/{{ip}}/g, ip)
        .replace(/{{ufrag}}/g, u)
        .replace(/{{pwd}}/g, p)
        .replace(/{{fingerprint}}/g, f)
        .replace(/{{setup}}/g, type === 'offer' ? 'actpass' : 'active');
}

log("Main module loaded. V145 (Nav & Chat Sync) Initializing...");

let mySyncSeed = Math.floor(1000 + Math.random() * 9000); 

// Manual Connection UI Init
async function initManualConnectUI() {
    try {
        const myIp = await invoke<string>("get_mobile_ip");
        const myIpEl = document.getElementById("my-full-ip");
        if (myIpEl) myIpEl.innerText = myIp;

        const mySeedEl = document.getElementById("my-sync-seed");
        if (mySeedEl) mySeedEl.innerText = mySyncSeed.toString();

        const prefix = await invoke<string>("get_mobile_prefix");
        const prefixEl = document.getElementById("ip-prefix");
        if (prefixEl) prefixEl.innerText = prefix + ".";

        await invoke("start_listener_command", { seed: mySyncSeed });
    } catch (e) {
        log(`Network init err: ${e}`);
    }
}
initManualConnectUI();

document.getElementById("btn-manual-submit")?.addEventListener("click", async () => {
    const prefixEl = document.getElementById("ip-prefix");
    const p3 = (document.getElementById("target-part3") as HTMLInputElement).value;
    const p4 = (document.getElementById("target-part4") as HTMLInputElement).value;
    const tSeed = (document.getElementById("target-seed") as HTMLInputElement).value;
    
    if (!p3 || !p4 || !tSeed) {
        alert("Enter full target IP and Seed");
        return;
    }
    const prefix = prefixEl?.innerText || "";
    const fullTargetIp = `${prefix}${p3}.${p4}`;
    const seed = parseInt(tSeed);
    
    log(`Manual connection to ${fullTargetIp} with seed ${seed}`);
    try {
        await startWebRtcOfferer(fullTargetIp, seed);
    } catch (e) {
        log(`Err: ${e}`);
        alert("Err: " + e);
    }
});

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

    if (tabName === 'settings') {
        requestChatHistory();
    }
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
    log("🚀 Connection Finalized!");
    document.getElementById("mobile-intro-overlay")!.style.display = 'none';
    document.getElementById("answer-qr-container")!.style.display = 'none';
    if (qrInterval) { clearInterval(qrInterval); qrInterval = null; }

    if (searchInput) {
        searchInput.disabled = false;
        searchInput.placeholder = "Search or Ask Prompt";
    }
    document.querySelectorAll(".nav-icons .nav-btn").forEach(btn => (btn as HTMLButtonElement).disabled = false);
    switchTab("list");
}

// --- WebRTC Setup ---
function setupDataChannel(channel: RTCDataChannel) {
    channel.onopen = () => {
        log("Channel OPEN!");
        finalizeConnectionUI();
        channel.send(JSON.stringify({ type: "get_session" }));
        channel.send(JSON.stringify({ type: "get_navigation" }));
        channel.send(JSON.stringify({ type: "search", query: "" }));
    };
    channel.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            if (msg.type === "sync_list") renderList(msg.data);
            else if (msg.type === "sync_detail") renderDetail(msg.title, msg.content);
            else if (msg.type === "sync_chat") renderChat(msg.data);
            else if (msg.type === "sync_chat_history") renderChatHistory(msg.messages);
            else if (msg.type === "sync_session") updateSessionUI(msg.data);
            else if (msg.type === "sync_navigation") renderNavigationTree(msg.pages, msg.users);
            else if (msg.type === "extraction_progress") renderExtractionProgress(msg.payload);
        } catch(err) { log("Msg Parse Err: " + err); }
    };
}

// --- Navigation Rendering (Accordion Tree) ---
function renderNavigationTree(pages: any[], users: any[]) {
    log("Rendering Nav Tree...");
    const pageList = document.getElementById("nav-list-pages");
    const userList = document.getElementById("nav-list-users");
    
    if (pageList) pageList.innerHTML = renderAccordion(pages);
    if (userList) userList.innerHTML = renderAccordion(users);
}

function renderAccordion(nodes: any[]): string {
    if (!nodes || nodes.length === 0) return "<div style='color:#999; padding:5px; font-size:0.7rem;'>No items</div>";
    let html = `<ul class="logis-branch" style="list-style:none; padding-left:15px; margin:0;">`;
    nodes.forEach((node, i) => {
        const id = node.id || node.uuid || `node-${i}`;
        const name = node.name || (node.data && node.data.type) || "Page";
        const hasChildren = node.children && node.children.length > 0;
        
        html += `<li class="logis-parent" style="margin-bottom:8px;">`;
        html += `<div class="logis-label" style="font-size:0.8rem; cursor:pointer;" onclick="console.log('Nav to: ${id}')"><span>${name}</span></div>`;
        if (hasChildren) html += renderAccordion(node.children);
        html += `</li>`;
    });
    html += `</ul>`;
    return html;
}

// --- Chat History ---
function requestChatHistory() {
    if (dataChannel?.readyState === "open") {
        log("Fetching chat history...");
        dataChannel.send(JSON.stringify({ type: "get_chat_history" }));
    }
}

function renderChatHistory(messages: any[]) {
    if (!chatTalks) return;
    chatTalks.innerHTML = ""; // Clear for fresh history
    messages.forEach(msg => renderChat(msg));
}

// --- Session ---
function updateSessionUI(session: any) {
    if (searchInput) {
        searchInput.disabled = false;
        searchInput.placeholder = "Search or Ask Prompt";
    }
}

// --- Extraction Progress ---
function renderExtractionProgress(payload: any) {
    const detailTitle = document.getElementById("detail-title");
    const detailContent = document.getElementById("detail-content");
    if (detailTitle) detailTitle.innerText = "Task Progress";
    showDetailView(true);

    if (detailContent) {
        let logArea = document.getElementById("extraction-log-mobile");
        if (!logArea) {
            detailContent.innerHTML = `<div id="extraction-log-mobile" style="display:flex; flex-direction:column; gap:8px;"></div>`;
            logArea = document.getElementById("extraction-log-mobile");
        }
        const catId = payload.category ? payload.category.replace(/[^a-zA-Z0-9]/g, "") : "gen";
        const elementId = `prog-${catId}`;
        let p = document.getElementById(elementId);
        if (!p) {
            p = document.createElement("div");
            p.id = elementId; p.style.fontSize = "0.8rem";
            p.innerHTML = `<span class="spin-icon">⏳</span> <span class="txt">${payload.summary}</span>`;
            logArea?.appendChild(p);
        } else {
            p.querySelector(".txt")!.textContent = payload.summary;
        }
        if (payload.category === "Done") p.querySelector(".spin-icon")!.textContent = "✅";
    }
}

// --- Common Rendering ---
function renderList(items: any[]) {
    const list = document.getElementById("doc-list");
    if (!list) return;
    list.innerHTML = items.map(item => item2html(item, false)).join("");
    list.querySelectorAll('.logis-result').forEach(el => {
        el.addEventListener("click", () => {
            dataChannel?.send(JSON.stringify({ type: "get_detail", uuid: el.id }));
        });
    });
}

function renderDetail(title: string, content: string) {
    document.getElementById("detail-title")!.innerText = title;
    document.getElementById("detail-content")!.innerHTML = content;
    showDetailView(true);
}

function renderChat(data: any) {
    if (!chatTalks) return;
    // Handle both old and new data structures
    let content = data.content;
    if (typeof content === 'string' && content.startsWith('{')) {
        try { content = JSON.parse(content).summary || JSON.parse(content).text; } catch(e){}
    }
    const div = document.createElement("div");
    div.className = `chat-talk ${data.role === 'user' ? 'user' : 'system'}`;
    div.innerHTML = `<div class="chat-message"><div class="content">${content}</div></div>`;
    chatTalks.appendChild(div);
    chatScroll.scrollTop = chatScroll.scrollHeight;
}

// --- Standard Handlers (Scanning, Camera, etc) ---
async function startScanning() {
    if (scanning) return;
    receivedOfferChunks = []; expectedOfferTotal = 0;
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
        // Handle compact Offer
        if (parsed.t === "offer") {
            log("Compact Offer Received!");
            if (parsed.h) localStorage.setItem("last_laptop_hash", parsed.h);
            const sdp = buildSdp('offer', parsed.i, parsed.u, parsed.p, parsed.f, parsed.s);
            stopScanning();
            createPeerConnection(sdp);
            return;
        }
        // Fallback for legacy chunked format
        if (Array.isArray(parsed) && parsed.length >= 3) {
            const [idx, total, chunk, laptopHash] = parsed;
            if (laptopHash) localStorage.setItem("last_laptop_hash", laptopHash);
            if (expectedOfferTotal === 0) {
                expectedOfferTotal = total;
                receivedOfferChunks = new Array(total).fill("");
            }
            if (!receivedOfferChunks[idx]) {
                receivedOfferChunks[idx] = chunk;
                log(`Part ${idx+1}/${total}`);
            }
            if (receivedOfferChunks.every(c => c !== "")) {
                stopScanning();
                createPeerConnection(receivedOfferChunks.join(""));
            }
        }
    } catch(e) {}
}

async function createPeerConnection(sdp: string) {
    peerConn = new RTCPeerConnection({ iceServers: [] });
    peerConn.ondatachannel = (e) => {
        dataChannel = e.channel;
        setupDataChannel(dataChannel);
    };
    peerConn.oniceconnectionstatechange = () => {
        if(peerConn?.iceConnectionState === 'connected') finalizeConnectionUI();
    };
    await peerConn.setRemoteDescription(new RTCSessionDescription({ type: 'offer', sdp: sdp }));
    const answer = await peerConn.createAnswer();
    await peerConn.setLocalDescription(answer);
    
    // Finalize gathering then show Answer Slide
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
    const myIp = await invoke<string>("get_my_full_ip");
    const parts = extractSdp(fullSdp);
    
    const compactAnswer = {
        t: "answer",
        i: myIp,
        u: parts.u,
        p: parts.p,
        f: parts.f,
        s: parts.s
    };

    const qrData = JSON.stringify(compactAnswer);
    log(`Answer Generated. Compact Length: ${qrData.length}`);
    showAnswerSlideQr([qrData]); // Wrap in array to reuse rotation logic or modify it
}

function showAnswerSlideQr(chunks: string[]) {
    const container = document.getElementById("answer-qr-container")!;
    container.style.display = 'flex';
    const qrDiv = document.getElementById("answer-qr")!;
    let cur = 0;
    const rotate = () => {
        if (chunks.length > 1) {
            qrDiv.innerHTML = `<div style="font-weight:bold; margin-bottom:10px;">Part ${cur+1}/${chunks.length}</div>`;
        } else {
            qrDiv.innerHTML = `<div style="font-weight:bold; margin-bottom:10px;">Scan this on Desktop</div>`;
        }
        const q = document.createElement("div");
        qrDiv.appendChild(q);
        new (window as any).QRCode(q, { text: chunks[cur], width: 250, height: 250 });
        cur = (cur + 1) % chunks.length;
    };
    if (qrInterval) clearInterval(qrInterval);
    rotate();
    if (chunks.length > 1) {
        qrInterval = setInterval(rotate, 1000);
    }
}

// --- Listeners ---
searchInput?.addEventListener("input", (e) => {
    dataChannel?.send(JSON.stringify({ type: "search", query: (e.target as HTMLInputElement).value }));
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
    dataChannel?.send(JSON.stringify({ type: "search", query: searchInput?.value || "" }));
});

const fileInput = document.getElementById("mobile-file-input") as HTMLInputElement;
document.getElementById("nav-upload-btn")?.addEventListener("click", () => fileInput?.click());
fileInput?.addEventListener("change", async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
        const base64 = (reader.result as string).split(",")[1];
        if (dataChannel?.readyState === "open") dataChannel.send(JSON.stringify({ type: "mobile_upload", name: file.name, data: base64 }));
    };
    reader.readAsDataURL(file);
});

window.addEventListener("request-camera-start", () => startScanning());
window.addEventListener("request-camera-stop", () => stopScanning());

// Global Reconnect
(window as any).tryQuickConnect = async () => {
    const hash = localStorage.getItem("last_laptop_hash");
    if (!hash) return;
    log("Quick Reconnecting...");
    try {
        const res = await fetch(`https://commerce.logis.center/relay/${hash}`);
        const data = await res.json();
        if (data && data.type === 'offer') createPeerConnection(data.sdp);
        else alert("Desktop is not ready.");
    } catch (e) { log("Reconnect failed."); }
};

log("Event listeners ready.");
