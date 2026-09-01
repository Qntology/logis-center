
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

// 🌟 index.html 의 #build-tag / styles.css?v= 와 반드시 같은 숫자를 유지합니다.
//    세 곳이 어긋나면 캐시 문제인지 코드 문제인지 구분할 수 없습니다.
log("Main module loaded. V147 (Style Parity & Remote Query) Initializing...");

// 🌟 [SEED PERSISTENCE] 데스크톱(main.ts initSyncUI)과 동일한 계약으로 맞춥니다.
//  ── 무엇이 문제였나 ──
//   ① 매 부팅마다 Math.random() 으로 새로 만들어 저장하지 않았습니다.
//      앱을 껐다 켜면 시드가 바뀌어, PC 가 기억하고 있던 페어링이 즉시 끊깁니다.
//   ② 데스크톱은 2자리(10~99)를 쓰는데 모바일은 4자리(1000~9999)를 썼습니다.
//      자릿수 규약이 달라 시드 충돌 자동 양보(auth_reject → 재생성) 로직이
//      양쪽에서 서로 다른 범위를 만들어 내며 영원히 수렴하지 않습니다.
//  ── 해결 ──
//   localStorage 에 영구 보존하고, 데스크톱과 동일한 2자리 범위를 씁니다.
//   (모바일에는 Dexie 를 두지 않으므로 kv_store 대신 localStorage 를 사용합니다)
const SEED_STORAGE_KEY = "my_sync_seed";

function generateSyncSeed(): number {
    // 🌟 데스크톱 initSyncUI / 시드 충돌 양보 로직과 완전히 동일한 범위입니다.
    return Math.floor(10 + Math.random() * 90);
}

function loadOrCreateSyncSeed(): number {
    try {
        const saved = localStorage.getItem(SEED_STORAGE_KEY);
        if (saved) {
            const n = parseInt(saved, 10);
            if (!isNaN(n) && n > 0) return n;
        }
    } catch (_e) {}
    const fresh = generateSyncSeed();
    try { localStorage.setItem(SEED_STORAGE_KEY, fresh.toString()); } catch (_e) {}
    return fresh;
}

let mySyncSeed = loadOrCreateSyncSeed();

/**
 * 🌟 [SEED REGENERATE] 시드 충돌 시 새 번호를 발급하고 리스너에 즉시 반영합니다.
 *  Rust 쪽 start_signal_listener 는 ACTIVE_SEED 만 갱신하고 포트를 재바인딩하지
 *  않으므로(LISTENER_STARTED 가드) 10048 에러 없이 초고속으로 교체됩니다.
 */
async function regenerateSyncSeed(): Promise<number> {
    mySyncSeed = generateSyncSeed();
    try { localStorage.setItem(SEED_STORAGE_KEY, mySyncSeed.toString()); } catch (_e) {}
    const mySeedEl = document.getElementById("my-sync-seed");
    if (mySeedEl) mySeedEl.innerText = mySyncSeed.toString();
    try {
        await invoke("start_listener_command", { seed: mySyncSeed });
    } catch (e) {
        log(`Seed regen listener err: ${e}`);
    }
    log(`Seed regenerated → ${mySyncSeed}`);
    return mySyncSeed;
}

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
    // 🌟 [DISPLAY DELEGATION] 표시/숨김은 `.tab-content` / `.tab-content.active` CSS 가
    //    전담합니다. 인라인 display 를 쓰면 CSS 규칙이 영구히 무력화됩니다.
    tabContents.forEach(c => {
        if (c.id === `tab-${tabName}`) {
            c.classList.add("active");
        } else {
            c.classList.remove("active");
        }
        c.style.removeProperty("display");
    });
    if (tabName === 'settings') {
        requestChatHistory();
    }
    if (tabName === 'list') {
        // 🌟 [DETAIL RESET] #list-view / #detail-view 는 #tab-list 의 '내부' 요소라
        //    탭 전환만으로는 상태가 되돌아가지 않습니다.
        //    상세를 보다가 Settings 로 갔다 다시 List 로 오면
        //    상세 화면이 그대로 남아 목록을 볼 수 없었습니다.
        showDetailView(false);
        // 🌟 [REMOTE REFRESH] 모바일은 자체 DB 가 없어 화면에 남아 있는 것은
        //    '마지막으로 받은 스냅샷' 뿐입니다. PC 에서 추출/동기화가 진행되어도
        //    모바일이 요청하지 않으면 영원히 옛 데이터입니다.
        if (dataChannel?.readyState === "open") {
            requestRemoteList(true);
            requestQueueStatus();
        }
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
    // 🌟 [AUTH FLAG SYNC] 이 함수는 두 경로로 진입합니다.
    //   ① auth_success 수신 (정상 Zero-Trust 경로)
    //   ② createPeerConnection 의 oniceconnectionstatechange === 'connected'
    //  ②로 먼저 열렸는데 플래그를 세우지 않으면, 뒤늦게 auth_success 가 도착했을 때
    //  bootstrapRemoteState() 가 한 번 더 돌아 동일 요청이 중복됩니다.
    //  반대로 ②만 발생하고 auth_success 가 오지 않는 경우도 있으므로
    //  플래그를 세우면서 bootstrap 이 아직이면 여기서 보완합니다.
    const wasFinalized = isAuthFinalized;
    isAuthFinalized = true;
    // 🌟 [CRITICAL FIX] index.html 에 'mobile-intro-overlay' id 는 존재하지 않습니다.
    //    실제 인트로 패널의 id 는 'tab-intro' 이며, 기존 코드는 `!` 단언 때문에
    //    null 참조 예외로 즉시 터졌습니다. 그 아래 switchTab("list") 가 실행되지 않아
    //    채널이 열렸는데도 화면이 영원히 Disconnected 로 남았습니다.
    //    (로그: "Uncaught TypeError: Cannot read properties of null")
    const intro = document.getElementById("tab-intro");
    if (intro) {
        // 🌟 클래스만 내리면 `.tab-content { display:none }` 이 적용됩니다.
        intro.classList.remove("active");
        intro.style.removeProperty("display");
    }
    const answerQr = document.getElementById("answer-qr-container");
    if (answerQr) answerQr.style.display = "none";
    const scannerOverlay = document.getElementById("scanner-overlay");
    if (scannerOverlay) scannerOverlay.style.display = "none";
    if (qrInterval) { clearInterval(qrInterval); qrInterval = null; }
    if (searchInput) {
        searchInput.disabled = false;
        searchInput.placeholder = "Search or Ask Prompt";
    }
    document.querySelectorAll(".nav-icons .nav-btn").forEach(btn => (btn as HTMLButtonElement).disabled = false);
    switchTab("list");
    // 🌟 [ICE FALLBACK BOOTSTRAP] ②(ICE) 경로로 처음 열린 경우에는
    //    auth_success 가 오지 않을 수 있으므로 여기서 초기 상태를 당겨옵니다.
    //    ①(auth_success) 경로는 호출부에서 이미 bootstrapRemoteState() 를 실행하므로,
    //    wasFinalized 가 false 인 '최초 1회' 에만, 그리고 채널이 열려 있을 때만 보완합니다.
    //    (switchTab("list") 안의 requestRemoteList 와 겹쳐도
    //     remoteIsLoading 락이 중복 요청을 막아 줍니다)
    if (!wasFinalized && dataChannel?.readyState === "open") {
        setTimeout(() => {
            // auth_success 가 그 사이 도착해 bootstrap 을 마쳤다면 remoteSession 이 채워집니다.
            if (!remoteSession) {
                log("ICE-only finalize detected → bootstrapping remote state.");
                bootstrapRemoteState();
            }
        }, 300);
    }
}
// =====================================================================
// 🌟 [REMOTE RESOURCE PROTOCOL v2]
//  모바일은 자체 DB(Dexie / LanceDB)를 두지 않고 PC 의 자원을 원격으로 씁니다.
//  따라서 이 파일이 하는 일은 단 세 가지입니다.
//   ① 요청을 만들어 DataChannel 로 보낸다
//   ② PC 가 돌려준 스냅샷을 그린다
//   ③ 그 사이 대기열/진행률을 표시한다
//
//  ── v1 의 결함 ──
//   · 검색이 PC 의 단순 텍스트 매칭(Select["items"])만 탔습니다.
//     Dexie Plan / LanceDB ai_search_complex 경로를 전혀 쓰지 못했습니다.
//   · 모드(commerce / shipping / analytic) 개념이 없어 항상 전체를 봤습니다.
//   · 페이지네이션이 없어 21번째 문서부터 접근이 불가능했습니다.
//   · 대기열 상태를 볼 방법이 없어 PC 가 바쁜지 알 수 없었습니다.
//   · 인증 핸드셰이크(auth_request/success/reject)를 몰라 PC 가 영원히 대기했습니다.
// =====================================================================

// ── 원격 상태 (PC 가 소유한 값의 그림자) ──
// 🌟 [AUTH GATE] PC 가 나를 승인해 '개통' 이 끝났는지 여부입니다.
//    양쪽이 채널 open 직후 동시에 auth_request 를 보내는 구조라
//    이 플래그가 없으면 finalize / bootstrap 이 두 번 실행됩니다.
let isAuthFinalized = false;
// 🌟 [CANCEL TARGET] 현재 PC 에서 진행 중인 원격 작업 id. cancel_task 에 실어 보냅니다.
let activeRemoteTaskId: string | null = null;
// 🌟 [HEADER STATE] 결과 건수와 대기열 표시가 같은 h2 를 서로 덮어쓰지 않도록
//    값을 상태로 보관하고 renderListHeader() 한 곳에서만 그립니다.
let lastResultTotal = -1;
let remoteSession: any = null;
let remoteSearchMode: string = "commerce";
// 🌟 [MODE CONFIRMED] PC 가 sync_mode 로 실제 모드를 알려 줬는지 여부입니다.
//    false 인 동안 remoteSearchMode 는 '추정값' 이므로 화면에 라벨을 확정 표기하지 않습니다.
let isRemoteModeConfirmed = false;
let remotePage = 0;
let remoteHasMore = true;
let remoteIsLoading = false;
let remoteQueue: { busy: boolean; currentTaskId: string | null; pending: number } = {
    busy: false, currentTaskId: null, pending: 0
};

/** 🌟 채널이 열려 있을 때만 안전하게 보냅니다. 끊긴 채널에 send 하면 예외가 납니다. */
function sendRemote(payload: any): boolean {
    if (!dataChannel || dataChannel.readyState !== "open") {
        log("Send skipped: channel not open");
        return false;
    }
    try {
        dataChannel.send(JSON.stringify(payload));
        return true;
    } catch (e) {
        log(`Send err: ${e}`);
        return false;
    }
}

/**
 * 🌟 [REMOTE LIST] PC 의 Dexie(executeDexiePlan 경로)에서 목록을 당겨옵니다.
 *  reset=true 면 첫 페이지부터, false 면 다음 페이지를 이어 붙입니다.
 */
function requestRemoteList(reset: boolean = false) {
    if (remoteIsLoading) return;
    if (reset) {
        remotePage = 0;
        remoteHasMore = true;
    }
    if (!remoteHasMore) return;
    remoteIsLoading = true;
    sendRemote({
        type: "search",
        query: searchInput?.value || "",
        mode: remoteSearchMode,
        limit: 20,
        offset: remotePage * 20,
        reset: reset
    });
}

/**
 * 🌟 [REMOTE AI SEARCH] PC 의 GlobalTaskManager 큐에 AI 검색 작업을 등록합니다.
 *  ── 왜 별도 타입인가 ──
 *   'search' 는 Dexie 즉시 조회(무료)이고,
 *   'ai_search' 는 LanceDB + LLM 이 도는 무거운 작업이라 반드시 큐를 타야 합니다.
 *   큐를 우회하면 PC 가 이미 추출 중일 때 모델이 두 번 로드되어 메모리가 터집니다.
 */
function requestRemoteAiSearch(query: string) {
    if (!query.trim()) return;
    sendRemote({
        type: "ai_search",
        query: query.trim(),
        mode: remoteSearchMode
    });
}

/** 🌟 [QUEUE] PC 의 대기열 상태(실행 중 / 대기 N건)를 조회합니다. */
function requestQueueStatus() {
    sendRemote({ type: "get_queue_status" });
}

/** 🌟 [QUEUE] 진행 중인 원격 작업을 취소합니다. */
function requestCancelTask(taskId: string) {
    sendRemote({ type: "cancel_task", taskId: taskId });
}

// --- WebRTC Setup ---
function setupDataChannel(channel: RTCDataChannel) {
    channel.onopen = () => {
        log("Channel OPEN! Starting Zero-Trust Auth Handshake...");
        // 🌟 [AUTH HANDSHAKE] 데스크톱은 채널이 열리자마자 auth_request 를 보내고
        //    승인 전까지 어떤 데이터도 주지 않습니다. 모바일도 같은 규약을 지켜야
        //    양쪽 중 누가 Offerer 이든 대칭적으로 성립합니다.
        //    (기존에는 이 프로토콜을 몰라 PC 의 승인 팝업이 영원히 대기했습니다)
        channel.send(JSON.stringify({
            type: "auth_request",
            session: {
                hash: remoteSession?.hash || "",
                address: remoteSession?.address || "",
                email: remoteSession?.email || "",
                team: remoteSession?.team || "",
                name: remoteSession?.name || "Mobile Device"
            }
        }));
    };
    channel.onclose = () => {
        log("Channel CLOSED.");
        handleDisconnect();
    };
    channel.onerror = (e) => {
        log(`Channel ERROR: ${e}`);
    };
    channel.onmessage = async (e) => {
        try {
            const msg = JSON.parse(e.data);

            // ── ① 인증 핸드셰이크 ──
            if (msg.type === "auth_request") {
                // 🌟 [APPROVE ONLY] 상대(PC)가 먼저 인사한 경우입니다.
                //    모바일은 자원 제공자가 아니므로 PC 를 즉시 승인합니다.
                //
                //  ── 왜 여기서 bootstrap 하면 안 되는가 ──
                //   양쪽 모두 채널 open 즉시 auth_request 를 보내므로 두 요청이 교차합니다.
                //   ① 여기서 곧바로 데이터를 요청하면 PC 의 승인 팝업(ask)이 아직 떠 있는 동안
                //      세션·네비게이션·문서 목록이 전부 흘러나가 Zero-Trust 승인이 무의미해집니다.
                //   ② 잠시 뒤 도착하는 PC 의 auth_success 에서 한 번 더 bootstrap 되어
                //      동일 요청이 두 번 나가고 목록이 중복 렌더링됩니다.
                //   따라서 여기서는 '승인 회신' 만 하고, 실제 개통은 auth_success 에서만 합니다.
                log("Auth request received → approving desktop (waiting for its approval).");
                channel.send(JSON.stringify({ type: "auth_success" }));
                return;
            }
            if (msg.type === "auth_success") {
                // 🌟 [IDEMPOTENT] 재전송·중복 도착에도 개통은 단 한 번만 수행합니다.
                if (isAuthFinalized) {
                    log("Auth success received again → already finalized, skipping.");
                    return;
                }
                isAuthFinalized = true;
                log("Auth approved by desktop.");
                finalizeConnectionUI();
                bootstrapRemoteState();
                return;
            }
            if (msg.type === "auth_reject") {
                log(`Auth rejected: ${msg.reason}`);
                // 🌟 [SEED COLLISION YIELD] 데스크톱이 시드 충돌로 거절하면
                //    모바일이 양보하고 새 번호를 발급받습니다.
                //    (데스크톱은 멤버가 있으면 양보하지 않고 상대에게 재생성을 요구합니다)
                const reason = String(msg.reason || "");
                if (reason.toLowerCase().includes("seed") || reason.includes("regenerate")) {
                    const fresh = await regenerateSyncSeed();
                    alert(`시드 번호가 중복되어 새 번호(${fresh})로 자동 변경되었습니다.\n다시 연결해 주세요.`);
                } else {
                    alert(`연결이 거부되었습니다: ${reason}`);
                }
                peerConn?.close();
                handleDisconnect();
                return;
            }

            // ── ② 데이터 스냅샷 ──
            if (msg.type === "sync_list") {
                remoteIsLoading = false;
                // 🌟 [PAGINATION] PC 가 pageSize 미만을 돌려주면 마지막 페이지입니다.
                const rows = msg.data || [];
                remoteHasMore = rows.length >= 20;
                if (msg.reset) {
                    remotePage = 1;
                    renderList(rows, false);
                } else {
                    remotePage++;
                    renderList(rows, true);
                }
                if (typeof msg.total === "number") updateResultCount(msg.total);
                return;
            }
            if (msg.type === "sync_detail") { renderDetail(msg.title, msg.content); return; }
            if (msg.type === "sync_chat") { renderChat(msg.data); return; }
            // 🌟 [PEER CHAT] 데스크톱 채팅 폼은 입력 즉시 아래를 그대로 보냅니다.
            //      dataChannel.send({ type: "chat_message", content: query })
            //    이 타입을 모바일이 처리하지 않으면 "Unknown msg type" 으로 버려져
            //    PC 에서 보낸 대화가 모바일 화면에 영원히 뜨지 않습니다.
            //    발신 주체가 PC 이므로 상대방(system) 말풍선으로 그립니다.
            if (msg.type === "chat_message") {
                renderChat({ role: "system", content: msg.content });
                return;
            }
            if (msg.type === "sync_chat_history") { renderChatHistory(msg.messages); return; }
            if (msg.type === "sync_session") { updateSessionUI(msg.data); return; }
            if (msg.type === "sync_navigation") { renderNavigationTree(msg.pages, msg.users); return; }
            if (msg.type === "extraction_progress") { renderExtractionProgress(msg.payload); return; }

            // ── ③ 대기열 ──
            if (msg.type === "sync_queue_status") {
                remoteQueue = {
                    busy: !!msg.busy,
                    currentTaskId: msg.currentTaskId || null,
                    pending: Number(msg.pending || 0)
                };
                renderQueueStatus();
                return;
            }
            if (msg.type === "task_queued") {
                log(`Task queued on desktop: ${msg.taskId}`);
                renderExtractionProgress({
                    task_id: msg.taskId,
                    category: "Pending",
                    summary: msg.summary || "Waiting in desktop queue...",
                    status: 10
                });
                requestQueueStatus();
                return;
            }
            if (msg.type === "sync_mode") {
                const prevMode = remoteSearchMode;
                const wasConfirmed = isRemoteModeConfirmed;
                remoteSearchMode = msg.mode || "commerce";
                // 🌟 이제부터 remoteSearchMode 는 PC 가 확정해 준 실제 값입니다.
                isRemoteModeConfirmed = true;
                applyRemoteModeUI();
                // 🌟 [MODE RESYNC]
                //  ── 왜 필요한가 ──
                //   ① bootstrapRemoteState() 는 get_mode 와 search 를 연달아 보내지만
                //      응답이 비동기라 첫 목록 요청은 기본값 'commerce' 로 나갑니다.
                //      PC 가 Trading/Analytic 이면 개통 직후 화면이 통째로 어긋납니다.
                //   ② PC 에서 모드 탭을 바꾸면 sync_mode 만 날아오고 목록은 옛 모드 그대로입니다.
                //   모드가 '실제로 바뀐' 경우에만 다시 당겨 왕복을 낭비하지 않습니다.
                // 🌟 [RELOAD 판정]
                //  ① 확정 이후 모드가 실제로 바뀐 경우 (PC 탭 전환) → 반드시 재당김
                //  ② 최초 확정인데 값이 기본값과 다른 경우 → bootstrap 이 잘못된 모드로
                //     이미 요청을 보냈으므로 재당김
                //  ③ 최초 확정인데 값이 기본값과 같은 경우 → bootstrap 요청이 이미 옳으므로
                //     재당김하지 않고 라벨만 확정합니다. (불필요한 왕복 제거)
                const modeChanged = prevMode !== remoteSearchMode;
                const needsReload = wasConfirmed ? modeChanged : modeChanged;
                if (needsReload && dataChannel?.readyState === "open") {
                    log(`Remote mode changed ${prevMode} → ${remoteSearchMode}. Reloading list.`);
                    // 이전 모드로 나간 요청은 폐기하고 새 모드로 다시 잠급니다.
                    remoteIsLoading = false;
                    lastResultTotal = -1;
                    requestRemoteList(true);
                }
                return;
            }

            log(`Unknown msg type: ${msg.type}`);
        } catch(err) { log("Msg Parse Err: " + err); }
    };
}

/** 🌟 인증 승인 직후 PC 의 초기 상태를 한 번에 당겨옵니다. */
function bootstrapRemoteState() {
    sendRemote({ type: "get_session" });
    sendRemote({ type: "get_mode" });
    sendRemote({ type: "get_navigation" });
    sendRemote({ type: "get_queue_status" });
    requestRemoteList(true);
}

/** 🌟 채널이 끊겼을 때 UI 를 초기 상태로 되돌립니다. */
function handleDisconnect() {
    dataChannel = null;
    remoteIsLoading = false;
    // 🌟 [AUTH GATE RESET] 다음 페어링에서 다시 승인을 받아야 하므로 반드시 내립니다.
    //    내리지 않으면 재연결 시 auth_success 가 와도 isAuthFinalized 때문에
    //    finalizeConnectionUI / bootstrapRemoteState 가 통째로 무시되어
    //    화면이 Disconnected 에서 벗어나지 못합니다.
    isAuthFinalized = false;
    remoteSession = null;
    activeRemoteTaskId = null;
    lastResultTotal = -1;
    remoteQueue = { busy: false, currentTaskId: null, pending: 0 };
    // 🌟 [FINALIZED LEDGER RESET] 재연결 후에는 같은 task id 라도 다시 마감 처리해야
    //    목록/대기열이 갱신됩니다. 장부를 비우지 않으면 재접속 직후 도착한 Done 이
    //    "이미 마감함" 으로 조기 반환되어 화면이 옛 상태에 머뭅니다.
    finalizedRemoteTasks.clear();
    if (searchInput) {
        searchInput.disabled = true;
        searchInput.placeholder = "Disconnected...";
    }
    document.querySelectorAll(".nav-icons .nav-btn").forEach(btn => (btn as HTMLButtonElement).disabled = true);
    // 🌟 [DISPLAY DELEGATION] styles.css 의 `.tab-content.active { display:flex }` 가
    //    이미 표시 규칙을 갖고 있으므로, 인라인 display 는 제거하고 클래스만 토글합니다.
    //    인라인이 남아 있으면 우선순위가 더 높아 나중에 CSS 로 레이아웃을 바꿔도
    //    반영되지 않고, `.hidden { display:none !important }` 외에는 덮을 수 없습니다.
    const intro = document.getElementById("tab-intro");
    if (intro) {
        intro.classList.add("active");
        intro.style.removeProperty("display");
    }
    tabContents.forEach(c => {
        if (c.id !== "tab-intro") {
            c.classList.remove("active");
            c.style.removeProperty("display");
        }
    });
}

// 🌟 [HEADER SINGLE SOURCE]
//  ── 무엇이 문제였나 ──
//   renderQueueStatus 와 updateResultCount 가 같은 h2 를 각자 innerHTML 로 덮어썼습니다.
//   sync_list 로 개수를 그린 직후 sync_queue_status 가 도착하면 개수가 지워지고,
//   반대 순서면 대기열 표시가 지워집니다. 즉 도착 순서에 따라 화면이 달라집니다.
//   두 값을 상태로 보관하고 렌더링은 이 함수 하나만 담당합니다.
function renderListHeader() {
    const header = document.querySelector("#list-view .header-row h2") as HTMLElement;
    if (!header) return;
    const countPart = lastResultTotal >= 0
        ? ` <strong style="font-size:0.75rem; font-weight:normal; color:#888;">(${lastResultTotal})</strong>`
        : "";
    const busyPart = (remoteQueue.busy || remoteQueue.pending > 0)
        ? ` <span style="font-size:0.65rem; font-weight:normal; color:#f59e0b; margin-left:6px;">⚙️ PC busy${remoteQueue.pending > 0 ? ` (+${remoteQueue.pending} queued)` : ""}</span>`
        : "";
    header.innerHTML = `Results${countPart}${busyPart}`;
}

/** 🌟 [QUEUE UI] 헤더에 PC 대기열 상태를 표시합니다. */
function renderQueueStatus() {
    renderListHeader();
}

/** 🌟 [MODE UI] PC 의 현재 모드를 화면에 반영합니다. */
function applyRemoteModeUI() {
    const label: Record<string, string> = {
        commerce: "Commerce", shipping: "Trading", analytic: "Analytic"
    };
    if (searchInput) {
        searchInput.placeholder = `${label[remoteSearchMode] || remoteSearchMode} Search or Ask`;
    }
    log(`Remote mode = ${remoteSearchMode}`);
}

/** 🌟 결과 개수 표기 */
function updateResultCount(total: number) {
    lastResultTotal = Number(total) || 0;
    renderListHeader();
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
    // 🌟 [SESSION CACHE] 기존 구현은 받은 세션을 통째로 버리고 placeholder 만 바꿨습니다.
    //    그 결과 remoteSession 이 영원히 null 이라
    //    setupDataChannel 의 auth_request 가 빈 신분증(hash/address/team 전부 "")을 보냈고,
    //    PC 의 isCloudMember 판정과 시드 충돌 자동 양보 분기가
    //    항상 '알 수 없는 외부 기기' 로 떨어져 매번 수동 승인 팝업이 떴습니다.
    if (session && typeof session === "object") {
        remoteSession = session;
        log(`Session cached: ${session.email || session.address || "(anonymous)"}`);
    }
    if (searchInput) {
        searchInput.disabled = false;
    }
    // 🌟 [MODE PENDING] placeholder 는 모드 라벨과 일치해야 하지만,
    //    bootstrapRemoteState 는 get_session 을 get_mode 보다 먼저 보내므로
    //    이 시점의 remoteSearchMode 는 아직 기본값('commerce')일 수 있습니다.
    //    확정되지 않은 값을 그려 넣으면 잠깐 잘못된 라벨이 보이고,
    //    그 사이 사용자가 검색하면 어긋난 모드로 요청이 나갑니다.
    //    sync_mode 를 아직 못 받았다면 중립 문구를 쓰고, 도착하면 교정합니다.
    if (isRemoteModeConfirmed) {
        applyRemoteModeUI();
    } else if (searchInput) {
        searchInput.placeholder = "Connecting…";
    }
}

// --- Extraction Progress ---
const MOBILE_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
let mobileSpinnerTimer: any = null;

function startMobileSpinner() {
    if (mobileSpinnerTimer) return;
    let i = 0;
    mobileSpinnerTimer = setInterval(() => {
        const frame = MOBILE_SPINNER_FRAMES[i % MOBILE_SPINNER_FRAMES.length];
        document.querySelectorAll('.mobile-active-spinner').forEach(el => {
            (el as HTMLElement).textContent = frame;
        });
        i++;
    }, 80);
}

function stopMobileSpinnerIfIdle() {
    if (document.querySelectorAll('.mobile-active-spinner').length === 0 && mobileSpinnerTimer) {
        clearInterval(mobileSpinnerTimer);
        mobileSpinnerTimer = null;
    }
}

/**
 * 🌟 [PROGRESS RENDER v2]
 *  ── v1 의 결함 ──
 *   category === "Done" 일 때만 ✅ 로 바꿨습니다.
 *   Error / Stopped / Cancelled 는 영원히 ⏳ 로 남아
 *   사용자가 실패한 작업을 성공으로 오인했습니다.
 *   또한 데스크톱이 보내는 "List Extraction (3/12)" 의 분수 표기와
 *   퍼센트를 전혀 파싱하지 않았습니다.
 */
// 🌟 [PROGRESS DEDUP] 종료 이벤트가 반복 도착해도 후속 작업을 단 한 번만 수행하기 위한 장부입니다.
//  ── 무엇이 문제였나 ──
//   PC 는 extraction-progress 를 '모든 payload' 마다 그대로 릴레이합니다.
//   백엔드는 같은 task 의 Done 을 재전송하는 경로가 여러 개 있고(재시도 · 로그 복구 등),
//   그때마다 requestRemoteList(true) + requestQueueStatus() 가 다시 나가면
//   모바일이 PC 의 Dexie 를 반복 왕복하며 목록이 계속 리셋됩니다.
const finalizedRemoteTasks = new Set<string>();

/**
 * 🌟 [DETAIL AUTOOPEN GATE] 진행률 때문에 화면을 강제로 뺏지 않기 위한 판정입니다.
 *  ── 무엇이 문제였나 ──
 *   기존 구현은 무조건 showDetailView(true) 를 호출했습니다.
 *   그런데 PC 는 자기 화면에서 진행 중인 추출·검색의 진행률도 전부 릴레이하므로,
 *   모바일 사용자가 목록을 스크롤하는 중에 PC 가 백그라운드 작업만 시작해도
 *   화면이 상세로 튕겨 나가 조작이 불가능했습니다.
 *  ── 규칙 ──
 *   ① 내가 올린 작업(activeRemoteTaskId 와 일치)이면 연다
 *   ② 이미 상세 화면을 보고 있으면 그대로 둔다
 *   ③ 그 외에는 로그만 갱신하고 화면은 건드리지 않는다
 */
function shouldAutoOpenDetail(taskId: string): boolean {
    if (activeRemoteTaskId && taskId === activeRemoteTaskId) return true;
    return detailView && detailView.style.display !== "none";
}

function renderExtractionProgress(payload: any) {
    const taskId = String(payload.task_id || "");
    const detailTitle = document.getElementById("detail-title");
    const detailContent = document.getElementById("detail-content");
    if (detailTitle) detailTitle.innerText = "Task Progress";
    // 🌟 [GATE] 화면 강제 전환을 막습니다. (아래 로그 DOM 은 숨은 상태로도 정상 갱신됩니다)
    if (shouldAutoOpenDetail(taskId)) showDetailView(true);
    if (!detailContent) return;
    let logArea = document.getElementById("extraction-log-mobile");
    if (!logArea) {
        // 🌟 [CANCEL WIRE] requestCancelTask() 가 선언만 되어 있고 호출부가 없어
        //    모바일에서 잘못 건 무거운 작업(AI 검색·문서 추출)을 멈출 방법이 없었습니다.
        //    PC 의 cancel_task 핸들러와 짝을 맞춰 취소 버튼을 붙입니다.
        detailContent.innerHTML =
            `<div id="extraction-log-mobile" style="display:flex; flex-direction:column; gap:8px;"></div>` +
            `<button id="btn-mobile-cancel-task" style="display:none; margin-top:14px; padding:8px 16px;` +
            ` border:1px solid #ef4444; color:#ef4444; background:#fff; border-radius:6px;` +
            ` font-size:0.75rem; cursor:pointer;">작업 취소</button>`;
        logArea = document.getElementById("extraction-log-mobile");
        document.getElementById("btn-mobile-cancel-task")?.addEventListener("click", () => {
            const tid = activeRemoteTaskId || remoteQueue.currentTaskId;
            if (!tid) { log("Cancel skipped: no active task id."); return; }
            log(`Requesting cancel for ${tid}`);
            requestCancelTask(tid);
        });
    }
    // 🌟 취소 요청에 실을 대상 id 를 항상 최신으로 유지합니다.
    if (payload.task_id) activeRemoteTaskId = String(payload.task_id);
    const rawCat = String(payload.category || "general");
    const catId = rawCat.replace(/\s*\(.*?\)/g, "").replace(/[^a-zA-Z0-9]/g, "") || "gen";
    const elementId = `prog-${catId}`;

    // 🌟 [FRACTION] 데스크톱이 "List Extraction (3/12)" 형태로 보내는 진행 분수를 뽑습니다.
    let fractionStr = "";
    if (rawCat.includes("List Extraction")) {
        const m = rawCat.match(/\((\d+)\/(\d+)\)/);
        if (m) fractionStr = ` [${m[1]}/${m[2]}]`;
    }
    const summary = `${payload.summary || ""}${fractionStr}`;

    const lower = String(payload.summary || "").toLowerCase();
    const isDone = rawCat === "Done";
    const isError = rawCat === "Error";
    const isStopped = lower.includes("cancelled") || lower.includes("stopped");
    const isPending = rawCat === "Pending" || rawCat === "Cloud Sync" || rawCat === "Cloud Queue";

    let p = document.getElementById(elementId);
    if (!p) {
        p = document.createElement("div");
        p.id = elementId;
        p.style.fontSize = "0.8rem";
        p.style.display = "flex";
        p.style.alignItems = "center";
        p.style.gap = "8px";
        p.innerHTML = `<span class="spin-icon mobile-active-spinner" style="min-width:16px; font-family:monospace;">⠋</span><span class="txt"></span>`;
        logArea?.appendChild(p);
        // 새 단계가 시작되면 이전 단계들은 완료 처리합니다.
        logArea?.querySelectorAll('.mobile-active-spinner').forEach(s => {
            if (s.parentElement !== p) {
                s.classList.remove('mobile-active-spinner');
                (s as HTMLElement).textContent = "✅";
                (s as HTMLElement).style.color = "#4ade80";
            }
        });
    }
    const txtEl = p.querySelector(".txt") as HTMLElement;
    if (txtEl) txtEl.textContent = summary;

    const icon = p.querySelector(".spin-icon") as HTMLElement;
    if (icon) {
        if (isDone) {
            icon.classList.remove("mobile-active-spinner");
            icon.textContent = "✅";
            icon.style.color = "#4ade80";
        } else if (isError) {
            icon.classList.remove("mobile-active-spinner");
            icon.textContent = "❌";
            icon.style.color = "#ef4444";
        } else if (isStopped) {
            icon.classList.remove("mobile-active-spinner");
            icon.textContent = "🛑";
            icon.style.color = "#ef4444";
        } else if (isPending) {
            icon.classList.remove("mobile-active-spinner");
            icon.textContent = "📥";
            icon.style.color = "#999";
        } else {
            icon.classList.add("mobile-active-spinner");
            icon.style.color = "#6366f1";
        }
    }

    startMobileSpinner();
    stopMobileSpinnerIfIdle();
    // 🌟 [CANCEL VISIBILITY] 종료 상태에서는 취소 버튼을 숨깁니다.
    //    (이미 끝난 작업에 cancel_task 를 쏘면 PC 가 엉뚱한 currentTaskId 를 중단시킵니다)
    const cancelBtn = document.getElementById("btn-mobile-cancel-task") as HTMLButtonElement | null;
    if (cancelBtn) {
        cancelBtn.style.display = (isDone || isError || isStopped) ? "none" : "block";
    }
    // 🌟 [QUEUE REFRESH] 작업이 끝났으면 대기열 상태를 다시 물어봅니다.
    //    ── 중복 차단 ──
    //     같은 task 의 종료 이벤트가 여러 번 도착해도 후속 왕복은 단 한 번만 수행합니다.
    //     (재전송마다 목록을 리셋하면 사용자가 보고 있던 스크롤 위치가 계속 튕깁니다)
    if (isDone || isError || isStopped) {
        if (taskId && finalizedRemoteTasks.has(taskId)) {
            // 이미 마감 처리한 작업입니다. 아이콘만 갱신하고 왕복은 생략합니다.
            return;
        }
        if (taskId) {
            finalizedRemoteTasks.add(taskId);
            // 장부가 무한히 커지지 않도록 상한을 둡니다. (한 세션에서 수백 건이면 충분)
            if (finalizedRemoteTasks.size > 200) {
                const first = finalizedRemoteTasks.values().next().value;
                if (first) finalizedRemoteTasks.delete(first);
            }
        }
        if (activeRemoteTaskId === taskId) activeRemoteTaskId = null;
        requestQueueStatus();
        // 목록도 최신화 (PC 가 추출을 끝냈으므로 새 문서가 생겼을 수 있습니다)
        requestRemoteList(true);
    }
}

// --- Common Rendering ---
/**
 * 🌟 [LIST RENDER v2]
 *  append=false : 첫 페이지. 기존 목록을 비우고 새로 그립니다.
 *  append=true  : 다음 페이지. 뒤에 이어 붙입니다.
 *  ── v1 의 결함 ──
 *   무조건 innerHTML 을 덮어써서 페이지네이션이 원천적으로 불가능했습니다.
 *   또한 이벤트 리스너를 querySelectorAll 로 매번 전부 다시 붙여
 *   이어 붙이기를 하면 리스너가 중복 등록되는 구조였습니다.
 */
function renderList(items: any[], append: boolean = false) {
    const list = document.getElementById("doc-list");
    if (!list) return;
    if (!append) list.innerHTML = "";
    if (!items || items.length === 0) {
        if (!append) {
            list.innerHTML = `<div style="text-align:center; padding:20px; color:#999; font-size:0.8rem;">No documents found.</div>`;
        }
        return;
    }
    const html = items.map(item => item2html(item, false)).join("");
    list.insertAdjacentHTML("beforeend", html);
    // 🌟 새로 삽입된 카드에만 리스너를 붙입니다. (data-bound 마커로 중복 방지)
    list.querySelectorAll('.logis-result:not([data-bound])').forEach(el => {
        (el as HTMLElement).dataset.bound = "1";
        el.addEventListener("click", () => {
            sendRemote({ type: "get_detail", uuid: el.id });
        });
    });
}

/**
 * 🌟 [INFINITE SCROLL] 목록 끝에 닿으면 다음 페이지를 원격으로 요청합니다.
 *  모바일은 자체 DB 가 없으므로 '더 불러오기' 도 전부 PC 왕복입니다.
 */
function initRemoteInfiniteScroll() {
    const container = document.getElementById("list-scroll-container");
    if (!container) return;
    container.addEventListener("scroll", () => {
        if (remoteIsLoading || !remoteHasMore) return;
        const nearBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 80;
        if (nearBottom) {
            log("Reached bottom → requesting next page");
            requestRemoteList(false);
        }
    }, { passive: true });
}
initRemoteInfiniteScroll();

function renderDetail(title: string, content: string) {
    document.getElementById("detail-title")!.innerText = title;
    document.getElementById("detail-content")!.innerHTML = content;
    showDetailView(true);
}

function renderChat(data: any) {
    if (!chatTalks) return;
    // Handle both old and new data structures
    let content = data.content || data.text || "";
    if (typeof content === 'string' && content.startsWith('{')) {
        try { 
            const obj = JSON.parse(content);
            content = obj.summary || obj.text || obj.title || content; 
        } catch(e){}
    }
    if (!content || content === "undefined") content = "";
    
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
    // 🌟 [CRITICAL FIX] 모바일 Rust 는 get_mobile_ip 로 커맨드를 등록합니다.
    //    기존 코드는 데스크톱 이름(get_my_full_ip)을 호출해 예외가 발생했고,
    //    Answer QR 이 생성되지 않아 페어링이 100% 실패했습니다.
    //    lib.rs 에 별칭을 추가해 두었지만, 의도를 명확히 하기 위해 모바일 이름을 씁니다.
    const myIp = await invoke<string>("get_mobile_ip");
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
// 🌟 [DEBOUNCE] 기존에는 키 입력마다 즉시 send 하여 PC 의 Dexie 를 초당 수 회 두드렸습니다.
//    모바일 타이핑 속도 기준 한 단어에 5~10회 왕복이 발생합니다.
let mobileSearchDebounce: any = null;
searchInput?.addEventListener("input", () => {
    if (mobileSearchDebounce) clearTimeout(mobileSearchDebounce);
    mobileSearchDebounce = setTimeout(() => {
        requestRemoteList(true);
    }, 400);
});

// 🌟 [AI SEARCH] 엔터/돋보기는 단순 조회가 아니라 PC 의 작업 큐에 등록하는 무거운 경로입니다.
searchInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        const q = searchInput.value.trim();
        if (!q) return;
        if (mobileSearchDebounce) { clearTimeout(mobileSearchDebounce); mobileSearchDebounce = null; }
        requestRemoteAiSearch(q);
        searchInput.value = "";
    }
});
document.getElementById("btn-submit")?.addEventListener("click", () => {
    const q = searchInput?.value.trim() || "";
    if (!q) return;
    if (mobileSearchDebounce) { clearTimeout(mobileSearchDebounce); mobileSearchDebounce = null; }
    requestRemoteAiSearch(q);
    if (searchInput) searchInput.value = "";
});

chatForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    const input = chatForm.querySelector('input[name="talk"]') as HTMLInputElement;
    if (input.value.trim()) {
        // 🌟 낙관적 렌더링 후 PC 로 위임합니다.
        //    PC 쪽 chat_message 핸들러가 로컬 LanceDB 저장 + 서버 PUT 까지 수행합니다.
        //    (기존 PC 구현은 에코만 돌려주어 메시지가 아무 데도 저장되지 않았습니다 → §42 에서 수정)
        renderChat({ role: 'user', content: input.value });
        sendRemote({ type: "chat_message", content: input.value });
        input.value = "";
    }
});
document.getElementById("btn-settings")?.addEventListener("click", () => switchTab("settings"));
document.getElementById("btn-settings-back")?.addEventListener("click", () => switchTab("list"));
document.getElementById("btn-detail-back")?.addEventListener("click", () => showDetailView(false));
document.getElementById("list-refresh-btn")?.addEventListener("click", () => {
    // 🌟 새로고침은 첫 페이지부터 다시 당기고 네비게이션/대기열도 함께 갱신합니다.
    requestRemoteList(true);
    sendRemote({ type: "get_navigation" });
    requestQueueStatus();
});

const fileInput = document.getElementById("mobile-file-input") as HTMLInputElement;
document.getElementById("nav-upload-btn")?.addEventListener("click", () => fileInput?.click());
fileInput?.addEventListener("change", async (e) => {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    // 🌟 [QUEUE AWARE] PC 가 바쁘면 순번을 기다린다는 사실을 미리 알려 줍니다.
    //    (PC 는 GlobalTaskManager.addToQueue 로 등록하므로 요청이 유실되지는 않습니다)
    if (remoteQueue.busy || remoteQueue.pending > 0) {
        log(`Desktop is busy (+${remoteQueue.pending} queued). Upload will wait in line.`);
    }
    const reader = new FileReader();
    reader.onload = () => {
        const base64 = (reader.result as string).split(",")[1];
        // 🌟 sendRemote 로 통일해 채널이 닫힌 상태에서 예외가 나는 경로를 없앱니다.
        const ok = sendRemote({ type: "mobile_upload", name: file.name, data: base64 });
        if (ok) log(`Uploading '${file.name}' to desktop queue...`);
    };
    reader.readAsDataURL(file);
    // 🌟 같은 파일을 연속으로 올릴 수 있도록 값을 비웁니다.
    //    (비우지 않으면 동일 파일 재선택 시 change 이벤트가 발화하지 않습니다)
    input.value = "";
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
