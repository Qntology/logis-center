console.log("%c[WIDGET] MAIN.TS LOADED", "color: #00ff00; font-weight: bold; font-size: 1.2rem;");
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { open, ask } from '@tauri-apps/plugin-dialog';
import { listen, emit } from '@tauri-apps/api/event';
import { readFile } from '@tauri-apps/plugin-fs';

// Imports for Rendering & Shim
import { item2html, selector } from "./lib/render";
import { Select, Upsert } from "./lib/db";
import { hashId, time2text } from "./lib/utils";

// Access global libs
const ethers = (window as any).ethers;
const blockies = (window as any).blockies;

// --- Config ---
const API_HOST = "https://commerce.logis.center"; 
const WIDGET_WIDTH = 380;
const COLLAPSED_HEIGHT = 80;
const EXPANDED_HEIGHT = 600;

interface ChatSession {
    hash: string;
    token?: string;
    email?: string;
    team?: string;
    address?: string;
    name?: string;
    cc?: string;
    sender?: string;
}

// --- State ---
let currentSession: ChatSession = { hash: "", cc: "logis.center" };
let isExpanded = false;
let currentTab = "list";
let currentImage: string | null = null;
let currentDetectedUrl = "";
let isCurrentShop = false; 
let searchDebounceTimer: number | null = null;
let chatPollInterval: number | null = null;

// 🌟 [추가] 누락된 전역 상태 변수 선언
let isSearching = false;
let isExtracting = false;
let lastSearchedQuery = "";

// [통합 락 매니저 & 프론트엔드 큐 관리자]
import "./dexie.min.js";
const DexieLocal = (window as any).Dexie;

const appDb = new DexieLocal("LogisAppDB");
appDb.version(1).stores({
    ts_queue: 'taskId, type',
    kv_store: 'key' // 추가된 통합 키-값(Key-Value) 저장소
});

// 기존 localStorage를 대체할 Dexie 헬퍼 함수
async function kvGet(key: string): Promise<any> {
    const record = await appDb.table("kv_store").get(key);
    return record ? record.value : null;
}
async function kvSet(key: string, value: any) {
    await appDb.table("kv_store").put({ key, value });
}
async function kvRemove(key: string) {
    await appDb.table("kv_store").delete(key);
}

// [통합 락 매니저 & 프론트엔드 큐 관리자]
class GlobalTaskManager {
    static isBusy: boolean = false;
    static currentTaskId: string | null = null;
    static activeRefs: Set<string> = new Set();
    static queue: Array<{taskId: string, type: string, payload: any}> = [];

    // 🌟 [추가] 큐를 Dexie(IndexedDB)에 저장하여 새로고침 시에도 증발 방지
    static async saveQueue() {
        await appDb.table("ts_queue").clear();
        if (this.queue.length > 0) {
            await appDb.table("ts_queue").bulkAdd(this.queue);
        }
    }

    // 🌟 [수정] 앱 시작 시 Dexie에서 저장된 큐 복원
    static async loadQueue() {
        // 🌟 [CRITICAL FIX] 앱을 완전히 종료 후 재시작했을 때 대기열 자동 실행 방지
        // sessionStorage는 F5 새로고침 시에는 유지되지만, 앱 종료 시에는 초기화됩니다.
        if (!sessionStorage.getItem("app_running_session")) {
            sessionStorage.setItem("app_running_session", "true");
            await appDb.table("ts_queue").clear();
            console.log("[QUEUE] App restarted. Cleared persistent Dexie queue to mark as STOPPED.");
            this.queue = [];
            return;
        }

        try {
            const q = await appDb.table("ts_queue").toArray();
            if (q && q.length > 0) {
                this.queue = q;
                this.queue.forEach((task: any) => this.activeRefs.add(task.taskId));
                console.log(`[QUEUE] Restored ${this.queue.length} pending tasks from Dexie.`);
            } else {
                this.queue = [];
            }
        } catch(e) {
            console.error("[QUEUE] Failed to load queue from Dexie", e);
            this.queue = [];
        }
    }

    static async addToQueue(taskId: string, type: string, payload: any) {
        if (this.activeRefs.has(taskId)) return;
        this.queue.push({ taskId, type, payload });
        this.activeRefs.add(taskId);
        await this.saveQueue(); // 🌟 즉시 저장 (Dexie)
        
        // 🌟 [추가] 큐에 담기자마자 사용자에게 시각적 피드백 제공 (DB 등록 전 선행 렌더링)
        const startTime = parseInt(taskId.split('_')[1]) || Date.now();
        
        // 1. 사용자 질문 선행 렌더링 (검색인 경우)
        if (payload.query) {
            await renderMessage({
                id: `${taskId}_query`,
                role: "user",
                text: payload.query,
                status: 9,
                created_at: startTime - 100,
                updated_at: startTime - 100
            });
        }

        // 2. 시스템 대기열 말풍선 선행 렌더링
        await renderMessage({
            id: taskId,
            task_id: taskId,
            role: "system_task",
            text: payload.link || payload.image_path || "Waiting in queue...",
            status: 10, // Pending
            created_at: startTime,
            updated_at: startTime
        });

        console.log(`[QUEUE] Task ${taskId} (${type}) added. Current queue length: ${this.queue.length}`);
        await this.processNext();
    }

    // 다음 작업 실행 판단 로직
    static async processNext() {
        if (this.isBusy || this.queue.length === 0) return;

        this.isBusy = true;
        const task = this.queue.shift()!;
        await this.saveQueue(); // 🌟 큐에서 항목이 나갔으므로 갱신 (Dexie)
        
        this.currentTaskId = task.taskId;
        await kvSet("sys_lock", task.taskId);

        console.log(`[QUEUE] Starting Task: ${task.taskId}`);
        
        try {
            if (task.type === "ai_search") {
                await invoke("ai_search_complex", task.payload);
            } else {
                await emit("new-task-from-browser", task.payload);
            }
        } catch (e) {
            console.error(`[QUEUE] Task execution failed:`, e);
            await this.release(task.taskId, task.taskId);
        }
    }

    static async release(taskId: string, refOrQuery: string) {
        if (this.currentTaskId === taskId) {
            this.isBusy = false;
            this.currentTaskId = null;
        }
        this.activeRefs.delete(taskId);
        if (await kvGet("sys_lock") === taskId) {
            await kvRemove("sys_lock");
        }
        await this.saveQueue(); // 🌟 참조 목록(activeRefs)이 변했으므로 갱신 (Dexie)
        await this.processNext();
    }

    static async forceReset() {
        this.isBusy = false;
        this.currentTaskId = null;
        this.activeRefs.clear();
        this.queue = [];
        await kvRemove("sys_lock");
        await appDb.table("ts_queue").clear(); // 🌟 완전 초기화 시 Dexie도 비움
    }
}

// [TAG SYSTEM] Hashtag-style search state
interface SearchTag {
    id: string;
    label: string;
    type: 'domain' | 'type' | 'mode' | 'path';
    value: string;
}
let activeTags: SearchTag[] = [];

// List State
let cachedDocs: any[] = [];
let currentPage = 0;
const pageSize = 10;
let isLoading = false;
let hasMore = true;

// Chat Pagination State
let chatPage = 0;
let chatHasMore = true;
let isChatLoading = false;

// [NEW] Track first-load status for UI loaders
let isFirstNavRender = true;
let isFirstChatLoad = true;

// [NEW] Window Focus State (백그라운드 리소스 최적화용)
let isFocus = true;

// 🌟 [CRITICAL FIX] 새로고침 시 스텝 순서 꼬임 방지용 대기열
let isFetchingLogs = false;
let pendingLiveEvents: any[] = [];
const livePayloads = new Map<string, any>(); // 🌟 [CRITICAL FIX] 퍼센트(%) 지연 노출을 막기 위한 프론트엔드 초고속 캐시 메모리

// ==========================================
// [PARITY] Cloud front.js Core Utilities
// ==========================================
function isDiff(obj1: any, obj2: any): boolean {
    if (!obj1 && !obj2) return false;
    if (!obj1 || !obj2) return true;
    const keys1 = Object.keys(obj1);
    const keys2 = Object.keys(obj2);
    if (keys1.length !== keys2.length) return true;
    
    for (const key of keys1) {
        if (typeof obj1[key] === 'object' && typeof obj2[key] === 'object') {
            if (isDiff(obj1[key], obj2[key])) return true;
        } else if (obj1[key] !== obj2[key]) {
            return true;
        }
    }
    return false;
}

function safeClone(obj: any) {
    const seen = new WeakMap();
    function clone(value: any) {
        if (typeof value !== "object" || value === null) return value;
        if (seen.has(value)) return null; 
        const copy: any = Array.isArray(value) ? [] : {};
        seen.set(value, copy);
        for (const key in value) {
            copy[key] = clone(value[key]);
        }
        return copy;
    }
    return clone(obj);
}

function mergeNode(obj1: any, obj2: any) {
    const isEmpty = (value: any) => value === null || value === undefined || value === '' || value === 0;
    const merged = { ...obj1 };
    for (const key in obj2) {
        if (obj2.hasOwnProperty(key)) {
            const value2 = obj2[key];
            if (!isEmpty(value2)) {
                merged[key] = value2;
            }
        }
    }
    return merged;
}

const taskSteps = new Map<string, Map<string, number>>();
const taskTotalSteps = new Map<string, number>(); // 🌟 [CRITICAL FIX] 작업별 총 스텝 수를 기억하는 장부 추가

let selectedUuids = new Set<string>();
let currentDetailUuid: string | null = null;
let activeTaskId: string | null = null; 
// [DEPRECATED] 흩어져 있던 개별 락 변수들은 GlobalTaskManager로 대체되었습니다.
let spinnerInterval: number | null = null;
let qrSpinnerIndex = 0; 
let systemLogCount = 0;

function stepQrSpinner() {
    const el = document.getElementById("qr-auth-spinner");
    if (el) {
        qrSpinnerIndex = (qrSpinnerIndex + 1) % spinnerFrames.length;
        el.innerText = spinnerFrames[qrSpinnerIndex];
    }
}
// [NEW] Active navigation context for related logs/chat
let activeContext = {
    cc: "",
    bcc: "",
    ref: ""
};

// --- UI Elements ---
const contentPanel = document.getElementById("content-panel") as HTMLElement;
const searchInput = document.getElementById("global-search") as HTMLInputElement;
const btnSubmit = document.getElementById("btn-submit") as HTMLButtonElement; 
const btnExtract = document.getElementById("btn-extract") as HTMLButtonElement; 
const btnAutoLaunch = document.getElementById("btn-auto-launch") as HTMLButtonElement;
const settingsBtn = document.getElementById("btn-settings") as HTMLButtonElement;
const tabContents = document.querySelectorAll<HTMLElement>(".tab-content");

const navPreviewContainer = document.getElementById("nav-preview-container") as HTMLElement;
const navImgThumbnail = document.getElementById("nav-img-thumbnail") as HTMLImageElement;
const navImgClear = document.getElementById("nav-img-clear") as HTMLButtonElement;
const navUploadBtn = document.getElementById("nav-upload-btn");

const listView = document.getElementById("list-view") as HTMLElement;
const detailView = document.getElementById("detail-view") as HTMLElement;
const detailTitle = document.getElementById("detail-title") as HTMLElement;
const detailContent = document.getElementById("detail-content") as HTMLElement;
const btnDetailBack = document.getElementById("btn-detail-back") as HTMLButtonElement;
const btnListBack = document.getElementById("btn-list-back") as HTMLButtonElement;
const btnDetailDelete = document.getElementById("btn-detail-delete") as HTMLButtonElement;
const btnStopTask = document.getElementById("btn-stop-task") as HTMLButtonElement; 

// [CHANGED] Replaced table body with generic list container
const docListContainer = document.getElementById("doc-list") as HTMLElement;

const listRefreshBtn = document.getElementById("list-refresh-btn") as HTMLButtonElement;
const btnDeleteSelected = document.getElementById("btn-delete-selected") as HTMLButtonElement;
const btnSyncQr = document.getElementById("btn-sync-qr") as HTMLButtonElement;
const listScrollContainer = document.getElementById("list-scroll-container") as HTMLElement;
const headerLoading = document.getElementById("header-loading") as HTMLElement;

// 🌟 기존 loadingIndicator 대신 h2 태그를 선택합니다.
const listTitle = document.querySelector("#list-view .header-row h2") as HTMLElement;

const aiResultsArea = document.getElementById("ai-search-results") as HTMLElement;
const aiResultsTitle = document.getElementById("ai-results-title") as HTMLElement;
const aiResultsContent = document.getElementById("ai-results-content") as HTMLElement;

const chatTalks = document.querySelector('.chat-talks') as HTMLElement;
const chatForm = document.querySelector('form[name="chat-form"]') as HTMLFormElement;

// --- Settings Toggle Logic ---
const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
const settingsPanel = document.getElementById("settings-panel") as HTMLElement;
const docList = document.getElementById("doc-list") as HTMLElement;
// 🌟 nav-section은 여러 개이므로 querySelectorAll로 잡습니다.
const navSections = document.querySelectorAll(".nav-section"); 

settingsToggle?.addEventListener("change", (e) => {
    const isChecked = (e.target as HTMLInputElement).checked;
    const label = document.querySelector('label[for="settings-toggle"]') as HTMLElement;
    const listRefreshBtn = document.getElementById("list-refresh-btn"); // 🌟 버튼 참조 추가
    
    if (isChecked) {
        // 설정 켜짐: 설정 패널 표시, 리스트 및 네비게이션 숨김
        if (settingsPanel) settingsPanel.style.display = "block";
        if (docList) docList.style.display = "none";
        if (listRefreshBtn) listRefreshBtn.style.display = "none"; // 🌟 새로고침 버튼 숨김
        navSections.forEach(el => (el as HTMLElement).style.display = "none");
        
        if (label) {
            label.classList.add("on")
        }
    } else {
        // 설정 꺼짐: 설정 패널 숨김, 리스트 및 네비게이션 원상복구
        if (settingsPanel) settingsPanel.style.display = "none";
        if (docList) docList.style.display = ""; 
        if (listRefreshBtn) listRefreshBtn.style.display = "flex"; // 🌟 새로고침 버튼 다시 표시
        navSections.forEach(el => (el as HTMLElement).style.display = "");
        
        if (label) {
            label.classList.remove("on");
        }

        applySearchModeUI(); 
    }
});

// --- Spinner Logic ---
const spinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

function startSpinner() {
    if (spinnerInterval) clearInterval(spinnerInterval);
    
    if (settingsBtn) {
        settingsBtn.classList.add("active-spinner-mode");
        // 🌟 [CRITICAL FIX] 글로벌 스피너가 돌 때 번개 버튼을 무조건 숨기던 코드를 제거합니다! (대기열 큐잉 허용)
        // isSearching 변수가 Part 1에서 선언되었으므로 이제 에러가 발생하지 않습니다.
        if (isSearching && btnSubmit) btnSubmit.style.display = "none";
    }
    
    let i = 0;
    spinnerInterval = window.setInterval(() => {
        const char = spinnerFrames[i % spinnerFrames.length];
        if (settingsBtn) settingsBtn.innerText = char;
        
        document.querySelectorAll('.spinner, .active-spinner').forEach(el => {
            (el as HTMLElement).innerText = char;
        });
        i++;
    }, 80);
}

function stopSpinner() {
    // 🌟 [CRITICAL FIX] 추출(Extracting) 중이거나 검색(Searching) 중이면, 
    // 백그라운드 태스크가 함부로 글로벌 스피너를 끄지 못하도록 절대 방어합니다!
    if (isExtracting || isSearching) return;

    if (spinnerInterval) {
        clearInterval(spinnerInterval);
        spinnerInterval = null;
    }
    
    if (settingsBtn) {
        settingsBtn.classList.remove("active-spinner-mode");
        settingsBtn.innerText = settingsBtn.classList.contains('active') ? "💬" : "🗨️";
    }
    
    document.querySelectorAll('.spinner, .active-spinner').forEach(el => {
        if (!el.closest('#extraction-log')) {
            el.classList.remove('active-spinner');
            (el as HTMLElement).innerText = "";
        }
    });

    // 🌟 [수정] 스피너 정지 시, 이미지가 첨부되어 있거나 진행 중이지 않은 유효한 입력값이 존재하면 검색 버튼 노출
    if (btnSubmit) {
        const currentVal = searchInput?.value.trim() || "";
        // 스피너가 멈췄다는 건 작업이 끝났다는 의미이므로, isQueryActive(currentVal)가 false가 되어 버튼이 살아납니다.
        if (currentImage || (currentVal !== "" && !isQueryActive(currentVal))) {
            btnSubmit.style.display = "flex";
        } else {
            btnSubmit.style.display = "none";
        }
    }

    updateExtractButtonVisibility();
}

// --- Layout & Window Logic ---
async function setWindowSize(expanded: boolean) {
    const height = expanded ? EXPANDED_HEIGHT : COLLAPSED_HEIGHT;
    await invoke("resize_window", { width: WIDGET_WIDTH, height: height });
}

function switchTab(tabName: string) {
    tabContents.forEach(c => {
        if (c.id === `tab-${tabName}`) c.classList.add("active");
        else c.classList.remove("active");
    });
    currentTab = tabName;
    
    if (tabName === "settings") {
        settingsBtn?.classList.add("active-emoji", "active");
        
        // 🌟 [CRITICAL FIX] Settings 버튼을 누르면 3초를 기다리지 않고 즉시 1회 서버 통신(인증/동기화)을 강제 실행합니다!
        if (!currentSession.email) {
            checkAuthStatus();
        }

        // 🌟 [CRITICAL FIX] 검색 중(isSearching)이거나 추출 중(isExtracting)일 때는 
        // 탭을 전환하더라도 억지로 히스토리를 리셋하여 방금 작성한 말풍선을 날려버리지 않도록 방어합니다!
        if (!isSearching && !isExtracting) {
            fetchChatHistory();
        } else {
            // 🌟 진행 중인 작업 때문에 돔을 리셋(innerHTML="")할 수는 없지만,
            // 달랑 진행 중인 말풍선 1~2개만 있고 과거 내역이 안 불러와진 상태라면 과거 내역(isHistory=true)을 끌어와서 화면을 채웁니다!
            if (chatTalks && chatTalks.children.length < 10 && chatHasMore) {
                loadMoreChat(true, true);
            } else {
                // 이미 화면이 채워져 있다면 최신 상태 변경점(status)만 조용히 동기화합니다.
                loadMoreChat(false, true);
            }
        }
        
        // 🌟 탭이 열렸으므로 폴링 타이머를 새롭게 리셋하여 주기를 맞춥니다.
        startPolling();
    } else {
        settingsBtn?.classList.remove("active-emoji", "active");
    }
    
    if (tabName === "list") refreshList(); 
    if (tabName === "automation") initBrowserDropdown();
}

function openWidget(tabName: string = "list") {
    if (!isExpanded) {
        isExpanded = true;
        contentPanel.classList.add("open");
        settingsBtn.innerText = "💬";
        setWindowSize(true);
    }
    switchTab(tabName);
}

function collapseWidget() {
    isExpanded = false;
    contentPanel.classList.remove("open");
    setWindowSize(false);
    settingsBtn?.classList.remove("active-emoji", "active");
    settingsBtn.innerText = "🗨️";
}

// --- Mouse Passthrough Logic ---
const interactiveElements = ['.pill-nav', '#content-panel'];

function setupMousePassthrough() {
    // [FIX] 기본적으로 위젯은 클릭이 가능해야 합니다. 
    // 윈도우 크기(380x80)가 이미 작기 때문에 윈도우 밖은 자동으로 클릭이 통과됩니다.
    invoke('set_ignore_cursor_events', { ignore: false }).catch(console.error);

    interactiveElements.forEach(selector => {
        const el = document.querySelector(selector);
        if (el) {
            el.addEventListener('mouseenter', () => {
                invoke('set_ignore_cursor_events', { ignore: false }).catch(console.error);
            });
        }
    });
}

// Drag Logic
const pillNav = document.querySelector('.pill-nav') as HTMLElement;
if (pillNav) {
    setupMousePassthrough(); // Initialize passthrough with the nav
    let isMouseDown = false;
    let startX = 0, startY = 0;
    const DRAG_THRESHOLD = 5;

    pillNav.addEventListener('mousedown', (e) => {
        const target = e.target as HTMLElement;
        if (!target.closest('button, input') && e.button === 0) {
             isMouseDown = true; startX = e.clientX; startY = e.clientY;
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (!isMouseDown) return;
        if (Math.abs(e.clientX - startX) > DRAG_THRESHOLD || Math.abs(e.clientY - startY) > DRAG_THRESHOLD) {
            isMouseDown = false; invoke('start_drag').catch(console.error);
        }
    });
    window.addEventListener('mouseup', () => isMouseDown = false);
    pillNav.addEventListener('dblclick', (e) => {
        const target = e.target as HTMLElement;
        if (!target.closest('button, input')) invoke("move_to_top_center").catch(console.error);
    });
}

let extractClickLock = false; 

async function updateExtractButtonVisibility() {
    if (!btnExtract) return;

    // 🌟 [CRITICAL FIX] 대기열 추가를 허용하기 위해 isExtracting/isSearching 등의 전역 잠금을 해제합니다.
    // 오직 더블클릭 방지용 extractClickLock만 유지합니다.
    if (extractClickLock) {
        btnExtract.style.display = "none";
        return;
    }

    // 5초 고아 락 해제 로직은 유지하되 버튼 숨김은 제거합니다. (UI 정리를 위함)
    const currentLock = await kvGet("sys_lock");
    if (currentLock) {
        const lockEl = document.getElementById(currentLock);
        if (!lockEl) {
            const lockTime = parseInt(currentLock.split('_')[1] || "0");
            if (Date.now() - lockTime > 5000) {
                console.warn(`[LOCK] Orphaned lock detected: ${currentLock}. Releasing.`);
                await kvRemove("sys_lock");
            }
        }
    }

    // 1. 이미지가 올라와 있을 때 중복 대기열 방어 로직
    if (currentImage) {
        try {
            const ccHash = activeContext.cc || "";
            const imageRefHash = await hashId(currentImage); 

            const isImageActive = await invoke<boolean>("check_active_task", {
                payload: { cc: ccHash, ref: imageRefHash }
            });
            
            // 🌟 프론트엔드 대기열(큐)에 동일한 이미지가 이미 들어있는지 검사
            const isImageQueued = GlobalTaskManager.queue.some(q => q.payload && q.payload.ref === imageRefHash);

            if (isImageActive || isImageQueued) {
                btnExtract.style.display = "none";
            } else {
                btnExtract.style.display = "flex";
                btnExtract.innerHTML = "⚡";
                btnExtract.classList.remove("active-spinner");
                btnExtract.title = "Extract from Image";
            }
        } catch (e) {
            btnExtract.style.display = "flex";
        }
        return;
    }

    // 2. URL이 아예 감지되지 않았다면 숨김
    if (!currentDetectedUrl) {
        btnExtract.style.display = "none";
        return;
    }

    try {
        const currentUrlObj = new URL(currentDetectedUrl.toLowerCase());
        const currentHostname = currentUrlObj.hostname;

        // 🌟 [CRITICAL FIX] 백엔드에서 인증한 샵(isCurrentShop)이 아니더라도, 
        // 내가 가진 Pages 도메인 목록에 포함되어 있다면 추출을 허용합니다.
        let isAllowedDomain = isCurrentShop;
        
        if (!isAllowedDomain) {
            const pages = await Select["pages"]({});
            isAllowedDomain = pages.some((p: any) => {
                const data = p.data || p;
                if (!data.origin) return false;
                try {
                    const originHost = new URL(data.origin.toLowerCase()).hostname;
                    // 정확한 호스트명 일치 또는 서브도메인 여부만 검사하여 오작동 방지
                    return currentHostname === originHost || currentHostname.endsWith("." + originHost);
                } catch(e) {
                    return false;
                }
            });
        }

        // 허용되지 않은 도메인이면 버튼을 숨김
        if (!isAllowedDomain) {
            btnExtract.style.display = "none";
            return;
        }

        // 🌟 [신규] 3. Pages 네비게이션 검증 로직
        // 사용자가 사이드바에서 선택한 도메인 태그를 찾습니다.
        const domainTag = activeTags.find(t => t.type === 'domain');

        if (domainTag) {
            // 태그 값(예: "demofran.com")과 현재 호스트네임 비교
            if (!currentHostname.includes(domainTag.value)) {
                // 도메인이 다르면 추출 버튼을 숨기고 로직을 종료합니다.
                btnExtract.style.display = "none";
                return;
            }
        }

        // 4. 백엔드에 현재 페이지 작업 상태 질의 및 ID 생성
        const link = (currentUrlObj.pathname + currentUrlObj.search).toLowerCase();
        const ccHash = await hashId(currentHostname);
        // 🌟 [CRITICAL FIX] 클라우드 로직(index.ts)과 완벽히 동일하게 Team ID를 포함하여 고유 Ref ID를 생성합니다.
        const teamId = currentSession.team || "";
        const hashedRefId = await hashId(teamId + ccHash + link);

        btnExtract.title = `Extract from ${currentHostname}`;

        const isActive = await invoke<boolean>("check_active_task", {
            payload: { cc: ccHash, ref: hashedRefId }
        });

        // 🌟 프론트엔드 대기열(큐)에 동일한 주소가 이미 들어있는지 검사
        const isQueued = GlobalTaskManager.queue.some(q => q.payload && (q.payload.ref === hashedRefId || q.payload.link === link));

        if (isActive || isQueued) {
            btnExtract.style.display = "none";
        } else {
            btnExtract.style.display = "flex";
            btnExtract.innerHTML = "⚡";
            btnExtract.classList.remove("active-spinner");
        }
    } catch (e) {
        console.warn("[WIDGET] visibility check error:", e);
        btnExtract.style.display = "none"; // 에러 시에는 안전하게 숨김 처리
    }
}

listen("browser-match-found", async (event: any) => {
    const payload = event.payload;
    console.log("[WIDGET] Browser Match Found:", payload);
    
    currentDetectedUrl = payload.url;
    isCurrentShop = payload.is_client || payload.is_admin;

    // [FIX] 해당 이벤트가 발송되고 있다면 브라우저가 실행 중인 상태이므로 무조건 버튼을 숨김
    if (btnAutoLaunch) {
        btnAutoLaunch.style.display = "none";
    }
    
    await updateExtractButtonVisibility();
    
    // 🌟 [CRITICAL FIX] 크롬 브라우저에서 주소를 이동할 때마다, 
    // 해당 도메인에 맞는 Pages를 네비게이션에 즉시 다시 그려줍니다!
    await renderNavigation();
});

const handleSearchInteraction = () => {
    // 🌟 [추가] 검색창 클릭/포커스 시 세팅 패널이 열려있다면 강제로 스위치를 끄고 닫아줍니다.
    const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
    if (settingsToggle && settingsToggle.checked) {
        settingsToggle.checked = false;
        settingsToggle.dispatchEvent(new Event("change")); // UI 원상복구 이벤트 트리거
    }

    // [UI-FIX] If the panel is already expanded, don't refresh the navigation or clear the list.
    // This prevents annoying UI flickering when the user just wants to type in the search bar.
    if (isExpanded && currentTab === "list") {
        return;
    }

    openWidget("list");
    const navOverlay = document.getElementById("nav-categories");
    if (navOverlay) {
        navOverlay.classList.remove("hidden");
        navOverlay.classList.add("visible");
        renderNavigation();
        if (listScrollContainer) listScrollContainer.scrollTo({ top: 0, behavior: 'smooth' });
    }
    if (!searchInput.value) {
        if (docListContainer) docListContainer.innerHTML = "";
        cachedDocs = [];
        currentPage = 0;
        hasMore = true;
    }
};

searchInput?.addEventListener("focus", handleSearchInteraction);
searchInput?.addEventListener("click", handleSearchInteraction);

function hideNavigation() {
    const navOverlay = document.getElementById("nav-categories");
    if (navOverlay) {
        navOverlay.classList.add("hidden");
        navOverlay.classList.remove("visible");
    }
}

function addSearchTag(label: string, type: 'domain' | 'type' | 'mode' | 'path', value: string) {
    const id = `${type}:${value}`;
    if (activeTags.find(t => t.id === id)) return;
    activeTags.push({ id, label, type, value });
    updateTagsUI();
    if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
    searchDebounceTimer = window.setTimeout(() => loadMoreDocs(true), 300);
}

function removeSearchTag(id: string) {
    const tagToRemove = activeTags.find(t => t.id === id);
    if (tagToRemove) {
        // [FIX] Reset specific context when corresponding tags are removed
        if (tagToRemove.type === 'domain') activeContext.cc = "";
        if (tagToRemove.type === 'type') activeContext.ref = "";
        if (tagToRemove.type === 'path') activeContext.ref = "";
    }

    activeTags = activeTags.filter(t => t.id !== id);
    
    // If no more tags left, clear the entire active context
    if (activeTags.length === 0) {
        activeContext = { cc: "", bcc: "", ref: "" };
    }

    updateTagsUI();
    loadMoreDocs(true);
    
    // [FIX] Also refresh chat history to reflect cleared filters
    fetchChatHistory(true);
}

function updateTagsUI() {
    const container = document.getElementById("search-tags-container");
    if (!container) return;
    container.innerHTML = "";
    activeTags.forEach(tag => {
        const chip = document.createElement("div");
        chip.className = `search-chip ${tag.type}`;
        chip.innerHTML = `<span>${tag.label}</span><span class="remove-btn" onclick="document.dispatchEvent(new CustomEvent('remove-tag', {detail: '${tag.id}'}))">✕</span>`;
        container.appendChild(chip);
    });
}

document.addEventListener('remove-tag', (e: any) => removeSearchTag(e.detail));

// --- Tree Rendering Logic (Pages & Users) ---
// --- Original Logic Implementation from content.js ---
let navTmp: Record<string, boolean> = {};

async function renderAccordion(nodes: any[], level = 1): Promise<string> {
    let html = `<ul class="logis-branch">`;

    for (var n = 0; n < nodes.length; n++) {
        var node = nodes[n];
        var nodeId = node.id || node.uuid || `node-${level}-${n}`;
        var active = '';
        var host = '';
        var type = 'page';
        var content = '';
        var name = '';
        var desc: string[] = [];
        var _url: URL | null = null;

        // ONLY generate HTML if this node hasn't been rendered yet
        if (!navTmp[nodeId]) {
            navTmp[nodeId] = true;

            if (node.name) {
                type = node.type || "team";
                name = node.name;
                if (node.type === "team") {
                    var teamName = node.name;
                    if (node.from === currentSession.address && nodeId === node.to) {
                        teamName = "Members";
                    }
                    host = `<strong>${teamName}</strong>`;
                } else {
                    let cancelBtn = "";

                    // 🌟 [수정] 본인 계정만 (owner), 초대된 멤버나 펜딩 상태는 (member)로 표시
                    if (node.id === currentSession.address) {
                        desc.push("(owner)");
                        // 🌟 본인(Owner 또는 Self)일 경우 삭제/취소 버튼을 노출하지 않습니다.
                    } else {
                        desc.push("(member)");
                        // 🌟 타인(Member)일 경우에만 삭제/취소 버튼을 노출합니다.
                        cancelBtn = `<button class="btn-cancel-member" data-id="${nodeId}" data-name="${name}" style="background:none; border:none; color:#ef4444; font-size:0.85rem; cursor:pointer; padding:0 5px; margin-left:auto; display:flex; align-items:center; justify-content:center;" title="Remove / Cancel Invite">✕</button>`;
                    }

                    // 🌟 [수정] 버튼이 우측 끝에 붙도록 Flex 구조 적용
                    content = `<div style="display:flex; align-items:center; width:100%; gap:8px;">
                        <span>${name}${desc.length ? `<i>${desc.toString()}</i>` : ''}</span>
                        ${cancelBtn}
                    </div>`;
                }
                if (nodeId === activeContext.ref) active = "active";
            } else if (node.data || node.type) {
                type = 'page';
                const data = node.data || {};
                const nodeType = node.type || data.type || 'unknown';
                console.log("[DEBUG] renderAccordion Page Node:", { id: nodeId, type: nodeType, domain: node.domain, origin: data.origin, link: data.link });
                name = `<span>${nodeType}</span> <span>${(data.item ? " Draft" : " ")}</span>`;

                if (data.origin) {
                    _url = new URL(data.origin);
                    const domain = node.domain || _url.hostname;
                    if (!navTmp[domain] && data.item) {
                        // 🌟 [추가] Host명 옆에 일괄 Show Hidden 버튼을 숨김 상태로 렌더링
                        host = `<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                                  <strong style="margin-bottom:0;">${domain}<button class="btn-show-domain-hidden" data-domain="${domain}" style="display:none; margin-left: 10px; background:none; border:none; cursor:pointer; font-size:10px; font-style: italic; text-decoration:underline;">Show</button></strong>
                                </div>`;
                        navTmp[domain] = true;
                    }
                    if (nodeId === activeContext.ref || (currentDetectedUrl && currentDetectedUrl.includes(data.link))) {
                        active = "active";
                    }
                }
                var total = { draft: 0, count: 0 };
                const pagesStats = (currentSession as any).pages;
                const cc = node.cc || data.cc;
                
                // 🌟 [TRACKING LOG] Accordion 그릴 때 각 노드의 매칭 상태 확인
                if (nodeType !== 'team' && nodeType !== 'user' && nodeType !== 'member') {
                    console.log(`[TRACKING-4] 렌더링 노드 타입: ${nodeType}, 대상 CC: ${cc}`);
                    if (pagesStats && cc && pagesStats[cc] && pagesStats[cc][nodeType]) {
                        total = pagesStats[cc][nodeType];
                        console.log(`[TRACKING-5] 매칭 성공! 적용될 카운트:`, total);
                    } else {
                        console.log(`[TRACKING-FAIL] 매칭 실패. pagesStats 존재여부: ${!!pagesStats}, CC 존재여부: ${!!cc}`);
                    }
                }

                var recent = '';
                try {
                    const bcc = node.bcc || data.bcc;
                    if (bcc) {
                        // 🌟 [CRITICAL FIX 1] AI 검색(Select)을 타격하여 무한 스피너가 도는 현상을 원천 차단합니다.
                        const _items = await invoke<any[]>("get_all_documents", { limit: 1, offset: 0, filter: `bcc = '${bcc}'` });
                        if (_items.length && _items[0].created_at) {
                            recent = `<strong>${time2text(Number(_items[0].created_at))}</strong>`;
                        }
                    }
                } catch (err) {}

                // 🌟 [CRITICAL FIX] front.js 패리티: 리스트 페이지(data.item 존재)일 경우 'Draft' 문구와 draft 카운트를 정확히 표기합니다!
                name = `<span>${nodeType}</span> <span>${(data.item ? " Draft" : " ")}</span>`;
                
                var count = '';
                if (data.item) {
                    count = `<u>(${total.draft || 0})</u>`;
                } else {
                    count = `<u>(${total.count || 0})</u>`;
                }
                
                // 🌟 [추가] 숨김 처리 상태 아이콘 및 스타일 적용
                const isHidden = hiddenPages.includes(nodeId);
                const visibilityIcon = isHidden ? "show" : "hide";
                
                const visibilityBtn = `<button class="btn-toggle-visibility" data-id="${nodeId}">${visibilityIcon}</button>`;

                if(!isHidden){
                    content = `<div style="display:flex; align-items:center; width:100%; justify-content:space-between;">
                        ${visibilityBtn}
                        <div style="display:flex; align-items:center; gap:4px;"><span>${name} ${count}</span> ${recent}</div>
                    </div>`;
                }
            }

            var hasChildren = node.children && node.children.length > 0;
            const inputId = `${type}-${nodeId}`;

            html += `
                <input type="checkbox" name="${type}" id="${inputId}" ${hasChildren ? 'checked' : ''} style="display:none;" />
                <li class="logis-parent ${hasChildren ? 'has-children' : ''}" ${type}-id="${nodeId}">
                    ${host}
                    <label for="${inputId}" class="logis-label ${inputId} ${active}" 
                           data-id="${nodeId}" 
                           data-cc="${node.cc || (node.data && node.data.cc) || ''}" 
                           data-bcc="${node.bcc || (node.data && node.data.bcc) || ''}" 
                           data-ref="${node.ref || node.ref_val || (node.data && node.data.ref) || ''}"
                           data-domain="${node.domain || (_url ? _url.hostname : '')}" 
                           data-type="${node.type || (node.data && node.data.type) || ''}">${content}</label>
            `;

            if (hasChildren) {
                html += `<div class="logis-child ${inputId}">`;
                html += await renderAccordion(node.children, level + 1);
                html += `</div>`;
            }

            html += `</li>`;
        }
    }

    html += `</ul>`;
    return html;
}

async function renderNavigation() {
    const pageList = document.getElementById("nav-list-pages");
    const userList = document.getElementById("nav-list-users");
    const profileName = document.getElementById("nav-profile-name");
    const profileFavicon = document.getElementById("nav-profile-favicon");
    const btnSignin = document.getElementById("nav-signin");
    const btnSignout = document.getElementById("nav-signout");

    if (!pageList || !userList) return;

    // [FIX] Show spinner only on the very first navigation render
    if (isFirstNavRender) {
        startSpinner();
    }

    // Profile UI
    if (currentSession.email) {
        if (profileName) profileName.innerText = currentSession.email.split('@')[0];
        if (btnSignin) btnSignin.classList.add("hidden");
        if (btnSignout) btnSignout.classList.remove("hidden");
        if (profileFavicon && blockies) {
            const icon = blockies.create({ seed: currentSession.email, size: 8, scale: 4 });
            profileFavicon.innerHTML = ""; profileFavicon.appendChild(icon);
        }
    }

    try {
        navTmp = {}; // Reset for fresh render
        let _pages = await Select["pages"]({});
        
        // 🌟 [CRITICAL FIX] 크롬 브라우저의 현재 접속 도메인과 일치하는 페이지만 남깁니다.
        let currentDomain = "";
        if (currentDetectedUrl) {
            try {
                currentDomain = new URL(currentDetectedUrl.toLowerCase()).hostname;
            } catch(e) {}
        }

        if (currentDomain) {
            _pages = _pages.filter(p => {
                const data = p.data || p;
                return data.origin && data.origin.toLowerCase().includes(currentDomain);
            });
        }
        
        const navSection = pageList.closest('.nav-section') as HTMLElement;

        const isSettingsOpen = (document.getElementById("settings-toggle") as HTMLInputElement)?.checked;

        if (_pages.length === 0) {
            pageList.innerHTML = `<div class="empty">No shared pages found for this domain.</div>`;
            // 🌟 [CRITICAL FIX] 데이터가 없더라도 Commerce/Analytic 모드이면 "비어있음" 문구가 노출되도록 통일
            if (navSection) navSection.style.display = (isSettingsOpen || currentSearchMode === "shipping") ? "none" : "block";
        } else {
            // 🌟 일치하는 데이터가 있으면 섹션을 화면에 표시하되, 세팅/Shipping 상태에 맞춰 제어합니다.
            if (navSection) navSection.style.display = (isSettingsOpen || currentSearchMode === "shipping") ? "none" : "block";
            
            const branchs: Record<string, any> = {};

            // 1. Build branch Map (origin#type and id)
            for (var p = 0; p < _pages.length; p++) {
                var _page = _pages[p];
                const data = _page.data || _page;
                if (!data.origin) continue;

                const domain = new URL(data.origin).hostname;
                _page.domain = domain;
                _page.id = _page.id || _page.uuid; // Ensure ID exists

                if (data.item) {
                    const listKey = `${data.origin}#${_page.type}`;
                    if (!branchs[listKey]) {
                        branchs[listKey] = { ..._page, children: [] };
                    } else {
                        // If we already have a shell, upgrade it to a full List node
                        Object.assign(branchs[listKey], _page);
                    }
                }
                // Always keep a ref by ID for grouping details
                if (!branchs[_page.id]) {
                    branchs[_page.id] = { ..._page, children: [] };
                }
            }

            // 2. Logic Tree Assembly
            const tree: any[] = [];
            const processedIds = new Set();

            // First, process all nodes that are Lists (Parents)
            for (let key in branchs) {
                let node = branchs[key];
                if (key.includes('#')) { // It's an origin#type group
                    tree.push(node);
                    processedIds.add(node.id);
                }
            }

            // Second, attach Details to their respective Lists, or add as top-level if standalone
            for (var p = 0; p < _pages.length; p++) {
                let _page = _pages[p];
                const pageId = _page.id || _page.uuid;
                if (processedIds.has(pageId)) continue;

                const data = _page.data || _page;
                const parentKey = `${data.origin}#${_page.type}`;
                
                if (branchs[parentKey]) {
                    branchs[parentKey].children.push(_page);
                } else {
                    tree.push({ ..._page, children: [] });
                }
                processedIds.add(pageId);
            }

            // 🌟 [CRITICAL FIX] 네비게이션(Accordion)을 렌더링하기 직전에, 
            // 로컬 DB(users 테이블)에 저장된 AI의 최신 추출 통계(count)를 불러와 메모리에 덮어씌웁니다!
            try {
                const _usersForStats = await Select["users"]({});
                const teamDoc = _usersForStats.find(u => u.type === "team" || (u.data && u.data.type === "team"));
                if (teamDoc) {
                    // 🌟 [CRITICAL FIX] LanceDB(TradeDocument)와 Server(JSON)의 포맷 차이 완벽 호환
                    // TradeDocument의 경우 최신 데이터가 문자열 형태의 json_data에 들어있으므로 최우선으로 파싱합니다.
                    let teamData: any = null;
                    if (teamDoc.json_data && typeof teamDoc.json_data === "string") {
                        try { teamData = JSON.parse(teamDoc.json_data); } catch(e) {}
                    }
                    if (!teamData && teamDoc.data) {
                        teamData = typeof teamDoc.data === "string" ? JSON.parse(teamDoc.data) : teamDoc.data;
                    }
                    teamData = teamData || teamDoc;

                    // 🌟 [로그 추가] 검색창 클릭 및 네비게이션 렌더링 시 Dexie에서 로드된 통계를 출력합니다.
                    console.log("\n=====================================");
                    console.log("[DEBUG-UI] Dexie에서 로드된 Team 데이터:", teamDoc);
                    console.log("[DEBUG-UI] 화면에 렌더링될 Base 통계:", JSON.stringify(teamData.base, null, 2));
                    console.log("=====================================\n");

                    if (teamData.base && teamData.base.pages) {
                        (currentSession as any).pages = teamData.base.pages;
                    }
                }
            } catch(e) { 
                console.warn("[Navigation] Failed to load local stats:", e); 
            }

            // 3. Render
            pageList.innerHTML = await renderAccordion(tree);

            // 🌟 [추가] Show/Hide 토글 버튼 이벤트 바인딩
            pageList.querySelectorAll(".btn-toggle-visibility").forEach((btn: any) => {
                btn.onclick = async (e: Event) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const targetId = btn.dataset.id;
                    if (!targetId) return;

                    if (hiddenPages.includes(targetId)) {
                        hiddenPages = hiddenPages.filter(id => id !== targetId);
                    } else {
                        hiddenPages.push(targetId);
                    }
                    await kvSet("hidden_pages", JSON.stringify(hiddenPages));
                    await renderNavigation(); // UI 즉시 갱신
                };
            });

            // 🌟 [추가] 숨겨진 항목이 있는 Host(도메인)의 Show 버튼 노출 및 일괄 해제 이벤트 바인딩
            pageList.querySelectorAll(".logis-label").forEach((label: any) => {
                const id = label.dataset.id;
                const domain = label.dataset.domain;
                // 해당 도메인에 속한 아이템 중 하나라도 숨김(hidden) 상태라면 Host의 Show 버튼을 노출합니다.
                if (hiddenPages.includes(id)) {
                    const hostShowBtn = pageList.querySelector(`.btn-show-domain-hidden[data-domain="${domain}"]`) as HTMLElement;
                    if (hostShowBtn) hostShowBtn.style.display = "inline";
                }
            });

            pageList.querySelectorAll(".btn-show-domain-hidden").forEach((btn: any) => {
                btn.onclick = async (e: Event) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const domain = btn.dataset.domain;
                    
                    // 🌟 해당 도메인을 가진 모든 라벨을 찾아 hiddenPages 배열에서 전부 제거합니다.
                    pageList.querySelectorAll(`.logis-label[data-domain="${domain}"]`).forEach((label: any) => {
                        const id = label.dataset.id;
                        if (hiddenPages.includes(id)) {
                            hiddenPages = hiddenPages.filter(hId => hId !== id);
                        }
                    });
                    
                    await kvSet("hidden_pages", JSON.stringify(hiddenPages));
                    await renderNavigation(); // UI 즉시 갱신하여 숨겨졌던 모든 항목을 표시
                };
            });

            // 4. Bind Clicks manually to labels
            pageList.querySelectorAll(".logis-label").forEach((label: any) => {
                label.onclick = async (e: Event) => {
                    const ds = label.dataset;
                    if (!ds.id) return;

                    // 🌟 [기획 반영] 초대 패널이 열려있는지 확인
                    const inviteContainer = document.getElementById("nav-cloud-invite-container");
                    const isInviteMode = inviteContainer && !inviteContainer.classList.contains("hidden");

                    if (isInviteMode) {
                        // A. 초대 모드: 클래스 토글 및 이벤트 중단
                        e.preventDefault();
                        e.stopPropagation();
                        label.classList.toggle("selected");
                        console.log(`[INVITE-MODE] Page ${ds.id} selection toggled:`, label.classList.contains("selected"));
                        return; // 필터링 로직 실행 방지
                    }

                    // B. 일반 모드: 기존 필터링 및 내비게이션 로직
                    // 1. 컨텍스트 업데이트
                    activeContext.cc = ds.cc || "";
                    activeContext.bcc = ds.bcc || "";
                    activeContext.ref = ds.ref || "";
                    
                    activeTags = activeTags.filter(t => t.type !== 'type' && t.type !== 'domain' && t.type !== 'path');
                    
                    // 2. 검색 태그 추가
                    addSearchTag(`@${ds.domain}`, 'domain', ds.domain);
                    addSearchTag(`#${ds.type}`, 'type', ds.type);
                    updateTagsUI();
                    
                    // 3. 버튼(#btn-extract) 강제 업데이트 호출
                    await updateExtractButtonVisibility();

                    // 4. UI 갱신 및 닫기
                    fetchChatHistory(true);
                    hideNavigation();
                };
            });
        }

        // Users rendering (simplified parity)
        const localUserList = document.getElementById("nav-list-local-users");
        const users = await Select["users"]({});
        
        if (userList) userList.innerHTML = "";
        if (localUserList) localUserList.innerHTML = `<div class="empty">No local Members/Devices</div>`;

        if (users.length > 0) {
            // 1. 꼬리표를 기준으로 로컬/클라우드 유저 분할
            const localUsers = users.filter(u => u.data && u.data.is_device === true);
            const cloudUsers = users.filter(u => !u.data || u.data.is_device !== true);

            // 2. Cloud Team Members 렌더링 (중복 Row 제거 로직 추가)
            if (cloudUsers.length > 0 && userList) {
                const uniqueCloudMembers: any[] = [];
                const seenMemberMap = new Map();

                cloudUsers.forEach(u => {
                    if (u.type === "team") {
                        uniqueCloudMembers.push(u);
                    } else {
                        const memberKey = u.to || u.id;
                        if (!seenMemberMap.has(memberKey)) {
                            seenMemberMap.set(memberKey, true);
                            uniqueCloudMembers.push(u);
                        }
                    }
                });

                const teamNodes = uniqueCloudMembers.filter(u => u.type === "team").map(u => ({
                    ...u, 
                    children: uniqueCloudMembers.filter(m => (m.to === u.id || m.cc === u.id) && m.id !== u.id)
                }));
                userList.innerHTML = await renderAccordion(teamNodes);

                // 🌟 [수정] 방장(Owner)인 경우에만 ADD 버튼 노출 (폼은 HTML에 정적으로 존재)
                const myTeam = cloudUsers.find(u => u.type === "team" && u.from === currentSession.address && u.id === u.to);
                const btnCloudInvite = document.getElementById("btn-cloud-invite-toggle");
                
                if (myTeam) {
                    if (btnCloudInvite) btnCloudInvite.style.display = "inline-block";
                } else {
                    if (btnCloudInvite) btnCloudInvite.style.display = "none";
                }

                // 🌟 [추가] 멤버 삭제 및 초대 취소 이벤트 위임 (Event Delegation)
                userList.onclick = async (e: Event) => {
                    const target = e.target as HTMLElement;
                    const cancelBtn = target.closest('.btn-cancel-member') as HTMLElement;
                    if (!cancelBtn) return;

                    // 라벨의 기본 클릭 이벤트(검색 컨텍스트 전환) 방지
                    e.preventDefault();
                    e.stopPropagation();

                    const targetId = cancelBtn.dataset.id;
                    const targetName = cancelBtn.dataset.name;

                    // Tauri 네이브 ask 팝업으로 확인
                    const confirmed = await ask(`정말 '${targetName}' 멤버를 삭제하거나 초대를 취소하시겠습니까?`, { 
                        title: "멤버 삭제 확인", 
                        kind: "warning" 
                    });

                    if (confirmed && targetId) {
                        try {
                            // 로컬 및 클라우드(동기화 시)에서 데이터 삭제
                            await invoke("delete_document", { uuid: targetId });
                            console.log(`[AUTH] Member/Invite removed: ${targetId}`);
                            
                            // UI 즉시 새로고침
                            await renderNavigation();
                        } catch (err) {
                            console.error("Failed to remove member:", err);
                        }
                    }
                };
            }

            // 3. Local Devices 렌더링 (단일 리스트 구조)

            // 3. Local Devices 렌더링 (단일 리스트 구조)
            if (localUsers.length > 0 && localUserList) {
                // 로컬 기기는 자식(children)이 없는 플랫한 노드로 렌더링합니다.
                const localNodes = localUsers.map(u => ({ ...u, children: [] }));
                localUserList.innerHTML = await renderAccordion(localNodes);
            }
        }

    } catch (e) { 
        console.error("Nav render error:", e); 
    } finally {
        // [FIX] Navigation rendered (or failed), stop spinner if it was the first time
        if (isFirstNavRender) {
            isFirstNavRender = false;
            stopSpinner();
        }
    }
}

// --- Invite Logic ---
async function handleTeamInvite() {
    const emailInput = document.getElementById("invite-email-input") as HTMLInputElement;
    const email = emailInput?.value.trim();

    // 🌟 이메일 형식 검증을 위한 정규식 추가
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

    if (!email || !emailRegex.test(email)) {
        alert("Please enter a valid email address (e.g., user@example.com).");
        if (emailInput) {
            emailInput.focus();
            emailInput.style.outline = "2px solid #ef4444";
        }
        return;
    }

    // 검증 성공 시 스타일 초기화
    if (emailInput) emailInput.style.outline = "none";

    const btn = document.getElementById("btn-send-invite") as HTMLButtonElement;
    const originalText = btn.innerText;
    btn.innerText = "Wait...";
    btn.disabled = true;

try {
        const origin = "https://commerce.logis.center";
        const now = Date.now();
        const createdAt = now - timezoneOffset;

        // 🌟 [추가] Pages 영역에서 selected 클래스가 붙은 모든 라벨의 data-id 수집
        const selectedPages: string[] = [];
        const pageList = document.getElementById("nav-list-pages");
        if (pageList) {
            pageList.querySelectorAll(".logis-label.selected").forEach((label: any) => {
                if (label.dataset.id) selectedPages.push(label.dataset.id);
            });
        }
        
        const params = new URLSearchParams({
            origin: origin,
            created_at: createdAt.toString(),
            hash: currentSession.hash,
            token: currentSession.token || "",
            href: currentDetectedUrl || "https://commerce.logis.center/tracking",
            from: currentSession.team || "",
            to: currentSession.address || "",
            email: email,
            // 🌟 수집된 페이지 ID 배열을 JSON 문자열로 변환하여 전달
            ref: JSON.stringify(selectedPages)
        });
        
        const url = `${API_HOST}/?${params.toString()}`;
        
        const response = await invoke<any>("proxy_fetch", {
            url: url,
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            session_params: { hash: currentSession.hash, token: currentSession.token }
        });

        // 서버 응답에서 hook URL을 가져오거나, 기본 mailto 훅을 사용
        let hookUrl = `${currentSession.hash}.logis.center@oauth.email`;
        if (response.results && response.results.length > 0) {
            const invite = response.results[0];
            if (invite.hook) hookUrl = invite.hook;
        }

        showInviteQr(hookUrl, email);
        emailInput.value = "";

        // 🌟 [추가] 클라우드 멤버 리스트에 '대기 중(Pending)' 상태로 즉시 렌더링되도록 로컬 DB에 임시 주입합니다.
        try {
            const pendingMember = {
                id: `pending_invite_${Date.now()}`,
                type: "user", // users 테이블로 분류되어 아코디언 메뉴에 들어갑니다.
                name: `${email.split('@')[0]} (Pending ⏳)`,
                from: currentSession.address || "0x0000000000000000000000000000000000000000",
                to: currentSession.team || "0x0000000000000000000000000000000000000000",
                cc: currentSession.team || "",
                data: { origin: "cloud", is_pending: true, email: email }
            };
            
            await invoke("upsert_items", { items: [pendingMember] });
            await renderNavigation(); // UI 즉시 갱신
        } catch (err) {
            console.warn("[INVITE] Failed to add pending member to UI:", err);
        }

    } catch (e) {
        console.error("[INVITE] Failed:", e);
        alert("Error sending invite.");
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
}

function showInviteQr(hook: string, email: string) {
    if (!chatTalks) return;
    
    // 기존 열려있던 네비게이션 숨기고 세팅(채팅) 탭 열기
    hideNavigation();
    openWidget("settings");

    const existing = document.getElementById("msg-invite-qr");
    if (existing) existing.remove();
    
    const mailtoLink = `mailto:${encodeURIComponent(hook)}`;

    const html = `
        <div class="chat-talk system" id="msg-invite-qr" data-created-at="9999999999999">
            <div class="chat-message" style="padding:15px; background: #fff; color: #000; border:0; border-radius: 8px; text-align: center;">
                <div style="font-size:0.8rem; font-weight: bold; margin-bottom: 10px; color: #333;">
                    Invite <span style="color:var(--primary);">${email}</span>
                </div>
                <div style="font-size:0.65rem; color: #666; margin-bottom: 15px; line-height: 1.4;">
                    Scan this QR code with mobile camera<br>to send an invitation email.
                </div>
                <div id="invite-qr-target" style="display: inline-block; background: #fff; padding: 10px; border-radius: 8px; border: 1px solid #eee;"></div>
                <div style="margin-top: 15px;">
                    <a href="${mailtoLink}" style="display: inline-block; padding: 8px 16px; background: var(--primary); color: #000; text-decoration: none; border-radius: 4px; font-weight: bold; font-size: 0.7rem;">Open Mail App</a>
                </div>
            </div>
        </div>`;
        
    chatTalks.insertAdjacentHTML('beforeend', html);
    
    const qrTarget = document.getElementById("invite-qr-target");
    if (qrTarget) {
        qrTarget.innerHTML = "";
        new (window as any).QRCode(qrTarget, { 
            text: mailtoLink, 
            width: 300, 
            height: 300, 
            colorDark: "#000000", 
            colorLight: "#ffffff", 
            correctLevel: (window as any).QRCode.CorrectLevel.M 
        });
        const scroll = document.getElementById("chat-scroll");
        if (scroll) scroll.scrollTop = scroll.scrollHeight;
    }
}

// --- Sync Logic ---
// main.ts 내부
async function syncData() {
    if (!currentSession.hash || !currentSession.email) return;
    
    console.log("[SYNC] 1. 서버에 최신 데이터 요청 중...");
    try {
        const origin = "https://commerce.logis.center";
        const now = Date.now();
        const createdAt = now - timezoneOffset;
        
        // 🌟 [CRITICAL FIX] front.js 패리티: 서버가 나를 정확히 인지하도록 cc, type, 실제 href 파라미터를 추가합니다.
        const queryParams: any = {
            origin: origin,
            created_at: createdAt.toString(),
            hash: currentSession.hash,
            token: currentSession.token || "",
            href: currentDetectedUrl || "https://commerce.logis.center/tracking"
        };

        if (activeContext.cc) queryParams.cc = activeContext.cc;
        if (currentSearchMode && currentSearchMode !== "commerce") queryParams.type = currentSearchMode;

        const params = new URLSearchParams(queryParams);
        const url = `${API_HOST}/?${params.toString()}`;
        
        // 1. 서버 요청
        const response = await invoke<any>("proxy_fetch", {
            url: url,
            method: "GET",
            headers: { "Content-Type": "application/json" },
            // 🌟 서버가 cc를 헤더나 쿠키처럼 파싱할 수 있게 proxy_fetch 파라미터에도 주입합니다.
            session_params: { hash: currentSession.hash, token: currentSession.token, cc: activeContext.cc || "" }
        });

        stepQrSpinner();

        if (response.results && Array.isArray(response.results)) {
            // 🌟 [버그 수정] DOM에 렌더링된 요소뿐만 아니라, Dexie DB에 있는 백그라운드 통계(team) 객체의 최신 시간도 대조해야 합니다.
            // 안 그러면 방금 전처리 완료된 로컬 통계를 과거의 서버 통계로 덮어버리게 됩니다!
            const localUsers = await Select["users"]({});
            const localPages = await Select["pages"]({});
            const localMap = new Map();
            [...localUsers, ...localPages].forEach((item: any) => {
                localMap.set(item.id, item.updated_at_ts || item.updated_at || 0);
            });

            const filteredResults = response.results.filter((newItem: any) => {
                let localUpdated = 0;
                
                const existingEl = document.getElementById(newItem.id);
                if (existingEl) {
                    localUpdated = parseInt(existingEl.dataset.updatedAt || "0");
                } else if (localMap.has(newItem.id)) {
                    localUpdated = parseInt(localMap.get(newItem.id) || "0");
                }

                const serverUpdated = newItem.updated_at || 0;
                return serverUpdated > localUpdated; // 서버 데이터가 더 최신인 경우만 포함
            });

            if (filteredResults.length > 0) {
                console.log(`[SYNC] 2. 로컬 LanceDB 최신화 중... (${filteredResults.length} / ${response.results.length} 건 변경됨)`);
                await invoke("upsert_items", { items: filteredResults });
                
                // 🌟 [누락 복구] 서버 통계를 LanceDB에 덮어썼다면, 반드시 프론트엔드 Dexie DB 에도 동기화해줘야 화면이 바뀝니다!
                const newUsers = filteredResults.filter((r: any) => r.type === "team" || r.type === "user" || r.type === "member");
                const newPages = filteredResults.filter((r: any) => r.type === "pages" || r.type === "page");
                
                if (newUsers.length > 0) await Upsert["users"](newUsers);
                if (newPages.length > 0) await Upsert["pages"](newPages);
            } else {
                console.log(`[SYNC] 2. 변경된 데이터가 없어 DB 쓰기를 건너뜁니다.`);
            }

            // 🌟 [추가] '대기 중' 멤버 정화(Cleanup) 로직
            // 서버에서 받은 결과 중 정식 멤버(member/user)가 있는지 확인합니다.
            const realMembers = response.results.filter(item => item.type === "member" || item.type === "user");
            if (realMembers.length > 0) {
                const localUsers = await Select["users"]({});
                // 로컬에 저장된 'pending_invite_'로 시작하는 가짜 데이터들을 찾습니다.
                const pendingInvites = localUsers.filter(u => u.id && u.id.startsWith("pending_invite_"));

                for (const pending of pendingInvites) {
                    const pendingEmail = pending.data?.email;
                    // 서버에서 온 정식 멤버 중 이메일(혹은 이름)이 일치하는 사람이 있는지 대조
                    const isNowMember = realMembers.some(m => {
                        // 서버 데이터(m) 내부에 이메일 정보가 있거나, 이름이 이메일 아이디와 같은지 확인
                        return m.to === pending.from || (m.data && m.data.email === pendingEmail);
                    });

                    if (isNowMember) {
                        // 정식 멤버가 확인되었으므로 가짜(Pending) 데이터를 로컬 DB에서 삭제합니다.
                        await invoke("delete_document", { uuid: pending.id });
                        console.log(`[SYNC] Pending invite for ${pendingEmail} is now a real member. Placeholder removed.`);
                    }
                }
            }
            
            console.log("[SYNC] 3. 로컬 DB에서 데이터 불러와 메뉴 렌더링...");
            // 3. LanceDB 불러오기
            await renderNavigation();
            
            // 🌟 [CRITICAL FIX] 서버 데이터를 로컬 DB에 밀어넣었으니, 현재 보고 있는 탭에 맞춰 UI를 갱신합니다!
            if (currentTab === "list") {
                await loadMoreDocs(false, true); 
            } else if (currentTab === "settings") {
                await fetchChatHistory(false, true);
            }
        }
        
    } catch (e) { 
        console.error("[SYNC] 동기화 실패:", e); 
    } finally {
        if (!isExtracting && !isSearching) stopSpinner();
    }
}

// --- 기존 State 영역 어딘가에 추가 ---
let currentSearchMode = "commerce"; // 값은 initSession에서 Dexie 비동기로 덮어씌워집니다.

// 🌟 앱 시작 시 탭 UI 초기화 함수
function applySearchModeUI() {
    document.querySelectorAll('.mode-tab').forEach(btn => {
        const el = btn as HTMLElement;
        if (el.dataset.mode === currentSearchMode) {
            el.style.color = "#000";
            el.style.fontWeight = "bold";
            el.classList.add('active');
        } else {
            el.style.color = "#999";
            el.style.fontWeight = "bold";
            el.classList.remove('active');
        }
    });

    // 🌟 [추가] 선택된 모드의 첫 글자를 대문자로 변환하여 Placeholder에 즉시 반영!
    if (searchInput) {
        const capitalizedMode = currentSearchMode.charAt(0).toUpperCase() + currentSearchMode.slice(1);
        searchInput.placeholder = `${capitalizedMode} Search or Ask`;
    }

    // 🌟 [추가] Shipping 모드일 때 Pages 섹션 통째로 숨기기
    const pagesSection = document.getElementById("nav-list-pages")?.closest(".nav-section") as HTMLElement;
    const isSettingsOpen = (document.getElementById("settings-toggle") as HTMLInputElement)?.checked;
    
    if (pagesSection) {
        if (currentSearchMode === "shipping" || isSettingsOpen) {
            pagesSection.style.display = "none"; // Shipping이거나 세팅 패널이 열려있으면 숨김
        } else {
            pagesSection.style.display = "block"; // 🌟 명시적으로 block 처리하여 노출 보장
        }
    }
}

// DOM 로드 후 이벤트 리스너 추가
document.querySelectorAll('.mode-tab').forEach(btn => {
    btn.addEventListener('click', async (e) => {
        const target = e.target as HTMLElement;
        currentSearchMode = target.dataset.mode || "commerce";
        
        // 🌟 탭 클릭 시 상태 저장 및 UI 업데이트
        await kvSet("search_mode", currentSearchMode);
        applySearchModeUI();

        console.log(`[UI] Search mode changed to: ${currentSearchMode}. Refreshing list...`);
        await refreshList(); 
    });
});

// 파일이 로드될 때 즉시 UI 적용
applySearchModeUI();


// [NEW] Global Navigation Link Handler (from item2html)
document.addEventListener('nav-link', async (e: any) => {
    const targetLink = e.detail;
    console.log("[NAV] Internal Link Clicked:", targetLink);
    addSearchTag(targetLink, 'path', targetLink);
    openWidget("list");
    listView.style.display = "block";
    detailView.style.display = "none";
});

// 🌟 [추가] 현재 입력한 검색어가 이미 대기열(10)이나 진행 중(1)인지 확인하는 완벽한 헬퍼 함수
function isQueryActive(text: string): boolean {
    const query = text.trim();
    // 1. 프론트엔드 큐 배열 검사 (아직 UI에 안 그려진 찰나의 순간 방어)
    if (GlobalTaskManager.queue.some(q => q.type === "ai_search" && q.payload && q.payload.query === query)) return true;

    // 2. DOM 상태 검사 (현재 실행 중인 작업 및 대기열 포함)
    let active = false;
    const bubbles = document.querySelectorAll('.task-bubble');
    for (let i = 0; i < bubbles.length; i++) {
        const el = bubbles[i] as HTMLElement;
        const status = parseInt(el.dataset.status || "0");
        const taskId = el.id;
        // 🌟 상태가 1(Processing)이거나 10(Queued)일 때만 활성 상태로 간주
        if ((status === 1 || status === 10) && taskId.startsWith("search_")) {
            const queryEl = document.getElementById(`${taskId}_query`);
            if (queryEl) {
                const qText = queryEl.querySelector('.content')?.textContent || "";
                if (qText.trim() === query) {
                    active = true;
                    break;
                }
            }
        }
    }
    return active;
}

searchInput?.addEventListener("input", () => {
    // 🌟 [CRITICAL FIX] 입력값이 비어있지 않고, 현재 진행/대기 중인 검색어와 '다를 때만' 버튼을 노출합니다.
    if (btnSubmit) {
        const currentVal = searchInput.value.trim();
        if (currentVal !== "" && !isQueryActive(currentVal)) {
            btnSubmit.style.display = "flex";
        } else {
            btnSubmit.style.display = "none";
        }
    }

    // 🌟 [CRITICAL FIX] 추출 중(isExtracting)이거나 큐가 바쁠 때(GlobalTaskManager.isBusy)
    // 타이핑만으로 백그라운드 임베딩 로직이 몰래 실행되는 것을 원천 차단합니다!
    if (isSearching || isExtracting || GlobalTaskManager.isBusy) return; 
    if(searchDebounceTimer) clearTimeout(searchDebounceTimer);
    searchDebounceTimer = window.setTimeout(async () => {
        if (isSearching || isExtracting || GlobalTaskManager.isBusy) return; 
        await loadMoreDocs(true);
    }, 800);
});

// [신규] 검색창에서 엔터 키를 누르면 AI 검색(돋보기 버튼)을 강제로 실행하도록 연결
searchInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault(); 
        // 🌟 [CRITICAL FIX] isExtracting 검사를 삭제하여, 전처리 중에도 검색을 대기열에 넣을 수 있게 허용합니다!
        if (!isSearching) { 
            btnSubmit?.click(); 
        }
    }
});

// --- main.ts 소스 ---

btnSubmit?.addEventListener("click", async () => {
    const query = searchInput.value.trim();
    if (!query) return;

    // 🌟 [CRITICAL FIX] 다른 검색어가 진행 중이더라도 새로운 검색어를 큐에 추가할 수 있도록 허용하되,
    // 완전히 동일한 검색어가 이미 진행/대기 중일 때만 중복 실행을 방어합니다!
    if (isQueryActive(query)) {
        console.warn("[SEARCH] The exact same query is already in progress or queued.");
        return; 
    }

    if (searchDebounceTimer) {
        clearTimeout(searchDebounceTimer);
        searchDebounceTimer = null;
    }

    // 인풋창 비우기
    searchInput.value = "";

    // 🌟 [CRITICAL FIX] 검색 버튼 숨김 (번개 버튼은 독립적인 추출 대기열 노출 조건을 따르도록 강제 숨김 코드를 제거합니다)
    if (btnSubmit) btnSubmit.style.display = "none";

    const taskId = `search_${Date.now()}`;
    const startTime = Date.now();
    
    // 🌟 [수정] 검색 시 설정(채팅) 탭으로 화면을 전환합니다.
    openWidget("settings");

    // 3. 사용자 질문 말풍선 즉시 렌더링
    await renderMessage({
        id: `${taskId}_query`,
        role: "user", 
        text: query,
        status: 9, 
        created_at: startTime,
        updated_at: startTime
    });

    try {
        const devicePref = getDevicePref();
        // 🌟 큐에 추가 (스피너는 백엔드가 실제 작업을 픽업하면 renderProgressToUI가 켭니다)
        await GlobalTaskManager.addToQueue(taskId, "ai_search", { 
            taskId: taskId, 
            query: query, 
            language: "korean",
            devicePreference: devicePref,
            searchMode: currentSearchMode,
            cc: activeContext.cc || "",
            bcc: activeContext.bcc || "",
            refId: activeContext.ref || ""
        });
        
        // 🌟 [CRITICAL FIX] 검색을 대기열에 추가한 직후, 현재 주소가 전처리 중인지 여부를 재검사하여 번개 버튼을 확실히 숨깁니다.
        updateExtractButtonVisibility();

        // 🌟 [추가] 생성된 검색 테스크(질문) 말풍선 위치로 부드럽게 스크롤 이동
        setTimeout(() => {
            const taskEl = document.getElementById(`${taskId}_query`) || document.getElementById(taskId);
            const scrollEl = document.getElementById("chat-scroll");
            const container = document.querySelector(".chat-container") as HTMLElement;
            
            if (taskEl && scrollEl && container) {
                const maxScroll = Math.max(0, scrollEl.scrollHeight - container.clientHeight);
                let targetY = taskEl.offsetTop - (container.clientHeight / 2) + (taskEl.clientHeight / 2);
                
                if (targetY < 0) targetY = 0;
                if (targetY > maxScroll) targetY = maxScroll;
                
                currentY = targetY;
                scrollEl.style.transition = "transform 0.3s ease-out";
                updateTransform();
                
                setTimeout(() => { scrollEl.style.transition = ""; }, 300);
            }
        }, 100);

        // 🌟 [CRITICAL FIX] 여기서 isSearching = false를 하지 않습니다! 백엔드의 Done/Error 신호가 풀어줄 때까지 잠가둡니다.
    } catch(e) { 
        console.error("[SEARCH-ERROR]", e);
        if (aiResultsContent) aiResultsContent.innerHTML = "<div style='color:#ef4444;'>Error: " + e + "</div>"; 
        
        // 에러 발생 시에만 강제 해제
        isSearching = false; 
        if (btnSubmit) btnSubmit.style.display = "flex";
        stopSpinner(); 
        updateExtractButtonVisibility();
    } 
    // 🌟 [CRITICAL FIX] finally 블록을 통째로 삭제하여 isSearching이 조기 해제되어 큐가 뚫리는 치명적 버그를 차단했습니다.
});

document.addEventListener('show-doc', (e: any) => showDetail(e.detail));
document.addEventListener('view-task-log', () => { openWidget("list"); listView.style.display = "none"; detailView.style.display = "flex"; });


// 🌟 [CRITICAL FIX] 추출 버튼 더블클릭 완벽 방어 로직 적용
// 🌟 [CRITICAL FIX] 추출 버튼 더블클릭 방어 및 대기열(Queue) 다중 진입 허용
btnExtract?.addEventListener("click", async () => {
    // 1. 순수하게 더블클릭(extractClickLock)만 막고, 
    // 기존 작업이 돌아가고 있더라도 주소가 다르다면 큐에 넣을 수 있도록 조건 해제
    if (extractClickLock) {
        console.warn("[LOCK] Click locked to prevent double submission.");
        if (btnExtract) btnExtract.style.display = "none";
        return; 
    }
    
    // 2. 버튼 숨김 (isExtracting = true 는 백엔드 작업이 실제 픽업될 때 켜지도록 제외)
    extractClickLock = true;
    if (btnExtract) btnExtract.style.display = "none";

    console.log("[DEBUG] btnExtract clicked. currentDetectedUrl:", currentDetectedUrl, "currentImage:", currentImage);
    
    try {
        if (currentDetectedUrl || currentImage) {
            const logArea = document.getElementById("extraction-log");
            if (logArea) logArea.innerHTML = "";
            
            // 🌟 [CRITICAL FIX] 추출(Extract) 시 채팅창(settings) 탭으로 자동 이동합니다.
            openWidget("settings"); 

            const taskId = `task_${Date.now()}`;
            
            // 🌟 수동 renderMessage 및 startSpinner 제거: addToQueue가 대기열 UI(10번)를 예쁘게 그려줍니다.
            
            const isCloudMode = (document.getElementById("cloud-mode-toggle") as HTMLInputElement)?.checked;

            if (isCloudMode && currentSession.hash) {
                // ==========================================
                // ☁️ [SERVER MODE]
                // ==========================================
                console.log("[WIDGET] Routing task to Cloud Server...");
                let payloadBody = "";
                let format = "";

                if (currentImage) {
                    const contents = await readFile(currentImage);
                    const blob = new Blob([contents]);
                    const base64Data = await new Promise<string>((resolve) => {
                        const reader = new FileReader();
                        reader.onloadend = () => { resolve(reader.result as string); };
                        reader.readAsDataURL(blob);
                    });
                    
                    payloadBody = base64Data;
                    format = "image/png"; 
                } else {
                    payloadBody = await invoke<string>("extract_html_from_current_tab");
                    format = "text/html";
                }

                const requestData = {
                    id: taskId,
                    from: currentSession.address,
                    to: currentSession.team,
                    cc: activeContext.cc || "",
                    bcc: activeContext.bcc || "",
                    ref: activeContext.ref || "",
                    body: payloadBody,
                    link: currentDetectedUrl || "local",
                    type: currentImage ? "image_extraction" : "html_extraction"
                };

                const urlObj = new URL(API_HOST);
                urlObj.searchParams.append("from", currentSession.address || "");
                urlObj.searchParams.append("to", currentSession.team || "");
                if (format.includes("image")) {
                    urlObj.searchParams.append("format", encodeURIComponent(format));
                }

                renderProgressToUI({ task_id: taskId, category: "Cloud Sync", summary: "Sending data to Logis Center...", spinner: "⠋" });

                const response = await invoke<any>("proxy_fetch", {
                    url: urlObj.toString(),
                    method: "POST",
                    headers: { 
                        "Content-Type": "application/json",
                        "Content-Encoding": "gzip" 
                    },
                    body: requestData,
                    session_params: { hash: currentSession.hash, token: currentSession.token }
                });

                console.log("[SERVER MODE] Task accepted by server:", response);
                renderProgressToUI({ task_id: taskId, category: "Cloud Queue", summary: "Task queued on server. Processing remotely.", spinner: "☁️" });
                
            } else {
                // ==========================================
                // 💻 [LOCAL MODE]
                // ==========================================
                if (currentImage) {
                    console.log("[WIDGET] Queuing LOCAL IMAGE task...");
                    const imageRefHash = await hashId(currentImage);

                    // 🚀 큐에 등록
                    await GlobalTaskManager.addToQueue(taskId, "image_extraction", { 
                        id: taskId, type: "image_extraction", image_path: currentImage, 
                        ref: imageRefHash, 
                        cc: activeContext.cc || "",
                        bcc: activeContext.bcc || "",
                        link: "Local Image",
                        device_preference: getDevicePref(), search_mode: currentSearchMode
                    });
                } else {
                    console.log("[WIDGET] Queuing LOCAL HTML/ANALYTIC task...");
                    const html = await invoke<string>("extract_html_from_current_tab");
                    const urlObj = new URL(currentDetectedUrl.toLowerCase());
                    const cc = await hashId(urlObj.hostname);
                    const rawPath = urlObj.pathname + urlObj.search;
                    const teamId = currentSession.team || "";
                    const hashedRefId = await hashId(teamId + cc + rawPath.toLowerCase());
                    
                    // 🌟 [추가] 사용자의 모드 선택에 따라 전처리 파이프라인(Task Type) 분기
                    const extractType = currentSearchMode === "analytic" ? "analytic_extraction" : "html_extraction";
                    
                    // 🚀 큐에 등록
                    await GlobalTaskManager.addToQueue(taskId, extractType, { 
                        id: taskId, type: extractType, html: html, link: rawPath, 
                        cc: activeContext.cc || cc, 
                        ref: activeContext.ref || hashedRefId, 
                        bcc: activeContext.bcc || "", 
                        from: currentSession.address, to: currentSession.team,
                        device_preference: getDevicePref(), search_mode: currentSearchMode
                    });
                }
            }
            
            if (currentImage) {
                currentImage = null;
                if (navPreviewContainer) navPreviewContainer.classList.add("hidden");
                if (navUploadBtn) navUploadBtn.classList.remove("active-emoji");
                if (searchInput) searchInput.disabled = false;
                if (btnSubmit) btnSubmit.style.display = "flex";
            }
            console.log("[WIDGET] Task safely added to backend queue:", taskId);

            // 🌟 [CRITICAL FIX] 생성된 테스크 말풍선 위치로 부드럽게 스크롤 이동
            setTimeout(() => {
                const taskEl = document.getElementById(taskId);
                const scrollEl = document.getElementById("chat-scroll");
                const container = document.querySelector(".chat-container") as HTMLElement;
                
                if (taskEl && scrollEl && container) {
                    const maxScroll = Math.max(0, scrollEl.scrollHeight - container.clientHeight);
                    // 엘리먼트를 화면 중앙쯤에 오도록 Y값 계산
                    let targetY = taskEl.offsetTop - (container.clientHeight / 2) + (taskEl.clientHeight / 2);
                    
                    if (targetY < 0) targetY = 0;
                    if (targetY > maxScroll) targetY = maxScroll;
                    
                    currentY = targetY;
                    // 부드러운 스크롤 효과를 위해 transition 임시 적용
                    scrollEl.style.transition = "transform 0.3s ease-out";
                    updateTransform();
                    
                    // 이동 후 transition 제거 (원래 드래그를 위해 없는 상태 유지)
                    setTimeout(() => {
                        scrollEl.style.transition = "";
                    }, 300);
                }
            }, 100);
        }
    } catch (e) {
        console.error("[WIDGET] Extraction failed:", e);
        extractClickLock = false;
        // 다른 작업이 정상적으로 돌아가고 있을 수 있으므로 sys_lock이나 전역 스피너를 함부로 날리지 않습니다.
        updateExtractButtonVisibility();
    } finally {
        // 🌟 [CRITICAL FIX] Rust 백엔드(DB)에 작업이 완전히 등재되도록 1.5초간 여유를 줍니다.
        // 이 시간 동안은 버튼이 절대 부활하지 않으며, 1.5초 뒤 DB를 조회하여 정상적으로 큐에 등록되었다면 버튼은 계속 숨겨집니다.
        setTimeout(async () => {
            extractClickLock = false;
            await updateExtractButtonVisibility();
        }, 1500);
    }
});

// 🌟 [추가] Rust 백엔드에 성공적으로 등록되었을 때 가상 렌더링 내용을 실제 데이터로 덮어씌웁니다.
listen("task-db-registered", async (event: any) => {
    const p = event.payload;
    console.log(`[WIDGET] Task ${p.task_id} successfully registered in Backend DB.`);
    
    await renderMessage({
        id: p.task_id,
        task_id: p.task_id,
        role: "system_task",
        text: p.text,
        status: p.status,
        created_at: p.created_at,
        updated_at: Date.now()
    });
});

listen("extraction-progress", async (event: any) => { 
    const payload = event.payload;
    if (payload.task_id) livePayloads.set(payload.task_id, payload);

    const summary = (payload.summary || "").toLowerCase();
    const isTerminal = payload.category === "Done" || payload.category === "Error" || summary.includes("cancelled") || summary.includes("stopped");
    
    if (isTerminal && payload.task_id) {
        console.log(`[QUEUE] Terminal state reached for ${payload.task_id}. Releasing and checking next.`);
        
        // 🌟 [최종 교정] 전역 상태를 먼저 false로 변경해야 updateExtractButtonVisibility가 버튼을 그립니다.
        if (payload.task_id.startsWith("task_") || payload.task_id.startsWith("img_")) {
            isExtracting = false;
        } 
        if (payload.task_id.startsWith("search_")) {
            isSearching = false;
        }

        // 🌟 큐 매니저 릴리즈 (비동기로 Dexie 업데이트 후 processNext 호출됨)
        await GlobalTaskManager.release(payload.task_id, payload.task_id);
        
        // 🌟 버튼 UI 즉시 갱신 및 스피너 중단
        stopSpinner();
        updateExtractButtonVisibility();

        // 🌟 [추가] 검색 작업이 완료(Done)되었을 경우, 백엔드가 보내준 데이터를 결과창에 렌더링합니다.
        if (payload.task_id.startsWith("search_") && payload.category === "Done" && payload.data) {
            const response = payload.data; // 백엔드에서 넘겨준 결과 객체
            if (aiResultsArea && aiResultsContent) {
                aiResultsArea.style.display = "block";
                aiResultsTitle.innerText = "🧠 AI Deep Analysis";
                
                let html = `<div style="margin-bottom:15px; padding:10px; background:#222; border-left:3px solid var(--primary); font-size:0.8rem;"><strong style="display:block; margin-bottom:5px; color:#aaa;">Query Intent:</strong>`;
                if (response.structured && response.structured.context) {
                    response.structured.context.forEach((ctx: any) => {
                        html += `<div style="margin-bottom:5px;">• ${ctx.text} <span style="color:var(--primary)">[${ctx.type}]</span></div>`;
                    });
                }
                html += `</div>`;
                
                if(!response.results || response.results.length === 0) {
                    html += `<div class="empty">No matching data found</div>`;
                } else {
                    html += response.results.map((res: any) => 
                        `<div style="border-bottom:1px solid #333; padding:8px 0;">
                           <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                             <strong style="color:var(--primary)">${res.context_type} (Score: ${res.score.toFixed(2)})</strong>
                             <button class="link-btn" onclick="document.dispatchEvent(new CustomEvent('show-doc', {detail:'${res.id}'}))">View Detail</button>
                           </div>
                           <div style="color:#ddd; line-height:1.4;">${res.text}</div>
                         </div>`
                    ).join("");
                }
                aiResultsContent.innerHTML = html;
                aiResultsArea.scrollIntoView({ behavior: 'smooth' });
            }
        }
    }

    if (isFetchingLogs && payload.task_id === activeTaskId) {
        pendingLiveEvents.push(payload);
        return;
    }
    renderProgressToUI(payload); 
});

document.addEventListener('render-progress', (e: any) => { renderProgressToUI(e.detail); });

async function renderProgressToUI(payload: any, isRecovery: boolean = false) {
    payload.task_id = payload.task_id || activeTaskId || (document.getElementById("extraction-log")?.dataset.activeTaskId);
    const tId = payload.task_id;
    if (!tId) return;

    const summary = (payload.summary || "").toLowerCase();
    const isTerminal = payload.category === "Done" || payload.category === "Error" || summary.includes("cancelled") || summary.includes("stopped");
    const isNotification = payload.category === "Warning" || payload.category === "Info";

    // 🌟 [CRITICAL FIX 1] 상태 입양(Adopt) 범위 확대: 
    // 백엔드에서 날아오는 'Loading Model', 'Saving', 'Handover' 등 모든 활동을 
    // '진행 중'으로 인지하여 큐가 풀리지 않도록 락을 단단히 고정합니다!
    const isPayloadRunning = payload.category && !["Pending", "Cloud Sync", "Cloud Queue"].includes(payload.category);

    if (!isRecovery && !isTerminal && isPayloadRunning) {
        if (activeTaskId !== payload.task_id || !GlobalTaskManager.isBusy) {
            console.log("[WIDGET] Adopting/Confirming running background task:", payload.task_id);
            
            activeTaskId = payload.task_id;
            await kvSet("sys_lock", activeTaskId!);
            GlobalTaskManager.isBusy = true;
            GlobalTaskManager.currentTaskId = activeTaskId;
            
            if (payload.task_id && payload.task_id.startsWith("search_")) {
                isSearching = true;
                if (btnSubmit) btnSubmit.style.display = "none";
            } else {
                isExtracting = true;
            }
            startSpinner();
        }
    }

    const baseCategory = payload.category ? payload.category.replace(/\s*\(.*?\)/g, "") : "general";
    const catId = baseCategory.replace(/[^a-zA-Z0-9]/g, "");
    const elementId = `progress-${catId}`;
    
    let displaySummary = payload.summary || "";
    
    // 🌟 [CRITICAL FIX] 백엔드에서 텍스트(summary)가 없는 순수 로그 이벤트를 보냈을 때,
    // 기존에 화면에 떠있던 텍스트를 보존하여 말풍선이 텅 비어버리는 현상을 원천 차단합니다!
    if (tId) {
        const existingEl = document.getElementById(tId) as HTMLElement;
        if (!displaySummary && existingEl) {
            displaySummary = existingEl.querySelector('.content')?.textContent || "";
        }
    }
    
    if (!taskSteps.has(tId)) {
        taskSteps.set(tId, new Map());
    }
    const stepMap = taskSteps.get(tId)!;

    // 🌟 [UI 심플화] 복잡한 계산식을 모두 삭제하고, 오직 'List Extraction' 단계에서만 [N/M]을 보여줍니다!
    if (!isTerminal && !isNotification) {
        let rawSummary = payload.summary || "";
        const pctMatch = rawSummary.match(/\(\d+%\)/);
        const hasDots = rawSummary.endsWith("...");
        
        if (hasDots) rawSummary = rawSummary.slice(0, -3).trim();
        if (pctMatch) rawSummary = rawSummary.replace(pctMatch[0], '').trim();

        let fractionStr = "";
        if (payload.category && payload.category.includes("List Extraction")) {
            const match = payload.category.match(/\((\d+)\/(\d+)\)/);
            if (match) {
                fractionStr = ` [${match[1]}/${match[2]}]`; // 백엔드가 준 정확한 숫자만 사용
            }
        }
        
        displaySummary = `${rawSummary}${fractionStr}${pctMatch ? ' ' + pctMatch[0] : ''}${hasDots ? '...' : ''}`;
    } else if (isNotification) {
        displaySummary = payload.summary || "";
    }

    // 🌟 [CRITICAL FIX] 대기열 상태(10)와 진행 상태(1)를 엄격히 구분합니다.
    let statusCode = 1; 
    
    if (isTerminal) {
        if (payload.category === "Done") statusCode = 9;
        else if (payload.category === "Error") statusCode = 6;
        else statusCode = 3;
    } else if (summary.includes("cancelled") || summary.includes("stopped")) {
        statusCode = 3;
    } else {
        // 🌟 [CRITICAL FIX 2] 백엔드에서 날아오는 중간 과정들이 10번(QUEUED)으로 오해받아 
        // 텍스트가 지워지고 스피너가 멈추는 버그를 원천 차단합니다. 오직 Pending 계열만 10번을 부여합니다!
        if (payload.category === "Pending" || payload.category === "Cloud Sync" || payload.category === "Cloud Queue") {
            statusCode = 10;
        } else {
            statusCode = 1;
        }
    }
    
    if (payload.task_id) {
        const existingEl = document.getElementById(payload.task_id) as HTMLElement;
        let originalCreatedAt = Date.now();
        if (existingEl) {
            originalCreatedAt = parseInt(existingEl.dataset.createdAt || "0");
        } else {
            const match = payload.task_id.match(/_(\d+)$/);
            if (match) originalCreatedAt = parseInt(match[1]);
        }

        await renderMessage({ 
            id: payload.task_id, 
            role: "system_task", 
            // 🌟 [CRITICAL FIX 1] content 대신 text 속성을 명시적으로 사용하여 텍스트 증발 방지
            text: displaySummary, 
            status: statusCode, 
            created_at: originalCreatedAt, 
            updated_at: Date.now(),
            task_id: payload.task_id
        });
    }

    // 🌟 [CRITICAL FIX] 1차 스피너 및 전역 상태 종료 처리 
    // 사용자가 현재 무슨 화면을 보고 있든, 작업이 끝났다면 무조건 전역 락을 풀고 스피너를 정지시킵니다!
    if (isTerminal) {
        // 🌟 [Lock 해제 보강] 어떤 경로로든 종료 상태가 되면 락을 확실히 제거합니다.
        const currentLock = await kvGet("sys_lock");
        if (currentLock === tId || !currentLock) {
            await kvRemove("sys_lock");
        }
        
        isExtracting = false;
        isSearching = false;
        stopSpinner();
        
        if (btnExtract) { btnExtract.classList.remove("active-spinner"); btnExtract.innerText = "⚡"; }
        if (currentImage) {
            currentImage = null; 
            if (navPreviewContainer) navPreviewContainer.classList.add("hidden"); 
            if (navUploadBtn) navUploadBtn.classList.remove("active-emoji"); 
            if (searchInput) searchInput.disabled = false; 
            if (btnSubmit) btnSubmit.style.display = "flex"; 
        }
        updateExtractButtonVisibility(); 

        // 🌟 [CRITICAL FIX] 전처리가 완료되면 자동으로 메뉴 카운트와 리스트를 리프레시!
        if (payload.category === "Done") {
            // 🌟 [버그 수정] 서버 모드이든 로컬 모드이든 무조건 로컬 LanceDB의 최신 전처리 결과를 Dexie에 먼저 덮어써야 합니다!
            Promise.all([
                invoke<any[]>("get_known_users"),
                invoke<any[]>("get_known_pages") 
            ]).then(async ([users, pages]) => {
                console.log("\n[TRACKING-1] Rust(LanceDB)에서 가져온 get_known_users 목록 수:", users ? users.length : 0);
                if (users && users.length > 0) {
                    const teamDocs = users.filter(u => u.type === "team" || (u.data && u.data.type === "team"));
                    console.log("[TRACKING-2] 그 중 'team' 타입 문서 파악:", teamDocs);
                    if (teamDocs.length > 0) {
                        // 🌟 [CRITICAL FIX] 로그 출력을 위해 json_data 문자열을 객체로 안전하게 파싱합니다.
                        let tData: any = null;
                        if (teamDocs[0].json_data && typeof teamDocs[0].json_data === "string") {
                            try { tData = JSON.parse(teamDocs[0].json_data); } catch(e) {}
                        }
                        if (!tData && teamDocs[0].data) {
                            tData = typeof teamDocs[0].data === "string" ? JSON.parse(teamDocs[0].data) : teamDocs[0].data;
                        }
                        tData = tData || teamDocs[0];
                        
                        console.log("[TRACKING-3] Dexie에 덮어쓸 최신 Base 통계:", JSON.stringify(tData.base?.pages, null, 2));
                        await Upsert["users"](teamDocs);
                    } else {
                        console.warn("[TRACKING-WARN] get_known_users에 'team' 문서가 포함되지 않았습니다! (Limit 제한 의심)");
                    }
                }
                if (pages && pages.length > 0) await Upsert["pages"](pages);
                
                console.log("[SYNC] LanceDB의 최신 데이터(통계, 페이지)를 Dexie로 동기화 완료!");
                
                if (currentSession.email) {
                    syncData(); // 🌟 Dexie 업데이트 완료 후 서버 동기화 시작
                } else {
                    renderNavigation().then(() => {
                        if (currentTab === "list") refreshList();
                    });
                }
            }).catch(e => {
                console.error("[SYNC] Dexie 동기화 실패:", e);
                if (currentSession.email) {
                    syncData();
                } else {
                    renderNavigation().then(() => {
                        if (currentTab === "list") refreshList();
                    });
                }
            });
        }
    }

    // 🌟 이제 현재 열려있는 Detail View가 이 Task의 것인지 확인 후 내부 로그(DOM)를 업데이트합니다.
    const extractionLog = document.getElementById("extraction-log");
    const targetContainer = document.getElementById("progress-container") || extractionLog;

    if (extractionLog && detailView.style.display !== "none") {
        if (extractionLog.dataset.activeTaskId !== tId) {
            // 현재 보고 있는 화면이 다른 Task면 여기서 DOM 업데이트 중지! (버블은 이미 위에서 업데이트됨)
            return;
        }

        if (payload.category === "Processing" && stepMap.size > 0) {
            stepMap.clear();
            if (targetContainer) targetContainer.innerHTML = "";
            await kvRemove(`term_${tId}`);
            const termArea = document.getElementById("terminal-logs");
            if (termArea) { termArea.innerHTML = ""; termArea.style.display = "none"; }
        }

        if (!stepMap.has(elementId)) {
            stepMap.set(elementId, stepMap.size + 1);
        }

        if (isTerminal) {
            if (targetContainer) {
                 const existingSpinners = targetContainer.querySelectorAll('.active-spinner');
                 existingSpinners.forEach(s => {
                     s.classList.remove('active-spinner');
                     s.innerHTML = payload.category === "Error" ? "❌" : "✅";
                     (s as HTMLElement).style.color = payload.category === "Error" ? "#ef4444" : "#4ade80";
                 });
            }
            if (btnStopTask) btnStopTask.style.display = "none";
            if (btnDetailDelete) btnDetailDelete.style.display = "flex";
        }

        let p = document.getElementById(elementId);
        if (!p) {
            if (targetContainer && !isNotification) {
                const existingSpinners = targetContainer.querySelectorAll('.active-spinner');
                existingSpinners.forEach(s => {
                    s.classList.remove('active-spinner');
                    s.innerHTML = "✅";
                    (s as HTMLElement).style.color = "#4ade80";
                });
            }

            p = document.createElement("div"); p.id = elementId;
            p.className = "progress-item";
            p.style.borderBottom = "1px solid #eee"; p.style.padding = "6px 0"; p.style.fontSize = "0.8rem";
            p.style.display = "flex"; p.style.flexDirection = "column"; 
            const row = document.createElement("div"); row.className = "progress-row"; row.style.display = "flex"; row.style.alignItems = "center";
            
            const spinnerIcon = `<span class="active-spinner" style="color:var(--primary); margin-right:8px; font-family:monospace; min-width:15px;">⠋</span>`;
            row.innerHTML = `${spinnerIcon}<span class="summary-text">${displaySummary}</span>`;
            p.appendChild(row);
            const results = document.createElement("div"); results.className = "results-container"; p.appendChild(results);
            
            if (targetContainer) targetContainer.appendChild(p);
        }
        
        const summaryEl = p.querySelector(".summary-text") as HTMLElement;
        const spinnerEl = p.querySelector(".active-spinner") as HTMLElement;

        if (summaryEl && summaryEl.textContent !== displaySummary) {
            summaryEl.textContent = displaySummary;
        }

        if (payload.category === "Done") {
            const row = p.querySelector(".progress-row");
            if (row) {
                const s = row.querySelector(".active-spinner") as HTMLElement;
                if (s) { s.classList.remove("active-spinner"); s.innerHTML = "✅"; s.style.color = "#4ade80"; }
            }
        } else if (payload.category === "Error") {
            const row = p.querySelector(".progress-row");
            if (row) { 
                const s = row.querySelector(".active-spinner") as HTMLElement;
                if (s) { s.classList.remove("active-spinner"); s.innerHTML = "❌"; s.style.color = "#ef4444"; }
                (row as HTMLElement).style.color = "#ef4444"; 
            }
        } else if (isNotification) {
            if (spinnerEl) {
                spinnerEl.classList.remove("active-spinner");
                spinnerEl.innerHTML = payload.spinner || "⚠️";
                spinnerEl.style.color = "#fbbf24"; 
            }
        } else {
            if (spinnerEl && spinnerEl.innerHTML !== "✅" && spinnerEl.innerHTML !== "❌" && spinnerEl.innerHTML !== "⚠️") {
                const newIcon = payload.spinner || "⠋";
                if (spinnerEl.innerText !== newIcon) { spinnerEl.innerText = newIcon; }
                if (newIcon === "✅" || newIcon === "✔") {
                    spinnerEl.classList.remove("active-spinner"); spinnerEl.style.color = "#4ade80";
                } else if (newIcon === "❌") {
                    spinnerEl.classList.remove("active-spinner"); spinnerEl.style.color = "#ef4444";
                } else {
                    spinnerEl.classList.add("active-spinner");
                }
            }
        }
    }
}

btnStopTask?.addEventListener("click", async () => {
    if (await ask("Stop the current extraction/search? (The record will be deleted)", { title: "Stop Task", kind: "warning" })) {
        // 🌟 [통합 락 매니저] 강제 초기화 호출 (isBusy, currentTaskId, activeRefs, Dexie 일괄 정리)
        await GlobalTaskManager.forceReset();
        
        isExtracting = false; 
        isSearching = false; 
        stopSpinner();
        
        if (btnExtract) {
            btnExtract.classList.remove("active-spinner");
            btnExtract.innerText = "⚡";
            btnExtract.style.display = "flex";
        }
        if (btnStopTask) btnStopTask.style.display = "none";

        try {
            console.log("[WIDGET] Stopping task:", activeTaskId);
            await invoke<string>("stop_current_extraction", { taskId: activeTaskId });
            
            if (activeTaskId) {
                // 취소 시 UI 로그 데이터 및 엘리먼트 삭제
                await kvRemove(`term_${activeTaskId}`);
                const el = document.getElementById(activeTaskId);
                if (el) el.remove();
            }

            activeTaskId = null;
            detailTitle.innerText = "Cancelled";
            detailContent.innerHTML = "<div style='color:#ef4444; padding:20px;'>Extraction stopped and deleted by user.</div>";
            
            // 버튼 상태 즉시 재계산
            await updateExtractButtonVisibility();
        } catch (e) { 
            console.error("Stop failed:", e); 
        }
    }
});

// --- Browser Auto ---
btnAutoLaunch?.addEventListener("click", async () => { 
    try { 
        // [FIX] Hide immediately on click for better UX
        btnAutoLaunch.style.display = "none";
        await invoke("launch_best_browser", { url: "about:blank" }); 
    } catch (e) { 
        console.error("Launch error:", e); 
        btnAutoLaunch.style.display = "flex"; // Restore on error
    } 
});
const autoBrowser = document.getElementById("auto-browser") as HTMLSelectElement;
const autoUrl = document.getElementById("auto-url") as HTMLInputElement;
const autoBtn = document.getElementById("auto-btn") as HTMLButtonElement;

async function initBrowserDropdown() {
    if (!autoBrowser) return;
    try {
        const browsers = await invoke<any[]>("check_available_browsers");
        autoBrowser.innerHTML = "";
        browsers.forEach(b => {
            const opt = document.createElement("option");
            opt.value = b.name; opt.text = b.name + (b.needs_driver ? " (No Driver)" : "");
            autoBrowser.appendChild(opt);
        });
    } catch (e) { console.error("Dropdown error:", e); }
}

autoBtn?.addEventListener("click", async () => {
    if (!autoBrowser || !autoUrl) return;
    try { await invoke("launch_browser", { browser: autoBrowser.value, url: autoUrl.value, script: "" }); } catch (e) { console.error("Manual launch error:", e); }
});

listen("browser-status", async (event: any) => {
    const status = event.payload; 
    console.log("[WIDGET] Browser Status Changed:", status);
    if (btnAutoLaunch) btnAutoLaunch.style.display = (status === "running") ? "none" : "flex";
    if (status === "stopped") { currentDetectedUrl = ""; await updateExtractButtonVisibility(); }
});

// --- List Logic (Updated for Cards) ---
listRefreshBtn?.addEventListener("click", refreshList);

btnDeleteSelected?.addEventListener("click", async () => {
    if (selectedUuids.size === 0) return;
    if (await ask(`Delete ${selectedUuids.size} documents?`, { title: "Confirm Delete", kind: "warning" })) {
        try { await invoke("delete_documents", { uuids: Array.from(selectedUuids) }); refreshList(); } catch (e) { console.error(e); }
    }
});

btnSyncQr?.addEventListener("click", async () => {
    const qrContainer = document.getElementById("nav-qr-container");
    const navOverlay = document.getElementById("nav-categories");

    if (!qrContainer || !navOverlay) return;

    if (navOverlay.classList.contains("hidden")) {
        handleSearchInteraction();
    }

    const isHidden = qrContainer.classList.contains("hidden");
    if (isHidden) {
        qrContainer.classList.remove("hidden");
        if (btnSyncQr) btnSyncQr.innerText = "CLOSE"; // 🌟 [추가] 패널이 열리면 CLOSE로 변경
        await initSyncUI(); // [NEW] Initialize IP/Seed view
        listCurrentY = 0;
        updateListTransform(true);
    } else {
        qrContainer.classList.add("hidden");
        if (btnSyncQr) btnSyncQr.innerText = "ADD"; // 🌟 [추가] 패널이 닫히면 ADD로 원상복구
    }
});

// 🌟 [추가] Cloud Member 초대 패널 토글 로직 (Local Member와 동일한 구조)
document.getElementById("btn-cloud-invite-toggle")?.addEventListener("click", () => {
    const inviteContainer = document.getElementById("nav-cloud-invite-container");
    const btn = document.getElementById("btn-cloud-invite-toggle");
    const pageList = document.getElementById("nav-list-pages");
    
    if (!inviteContainer || !btn) return;

    if (inviteContainer.classList.contains("hidden")) {
        inviteContainer.classList.remove("hidden");
        btn.innerText = "CLOSE";
        listCurrentY = 0;
        updateListTransform(true);
    } else {
        inviteContainer.classList.add("hidden");
        btn.innerText = "ADD";
        // 🌟 [추가] 패널 닫을 때 선택된 클래스 일괄 제거 (기획 의도에 따라 생략 가능)
        if (pageList) {
            pageList.querySelectorAll(".logis-label.selected").forEach(el => el.classList.remove("selected"));
        }
    }
});

// 🌟 [추가] Cloud Member 초대 전송 이벤트 등록
document.getElementById("btn-send-invite")?.addEventListener("click", async () => {
    await handleTeamInvite();
});

// [NEW] Manual Connect Handler
document.getElementById("btn-manual-connect")?.addEventListener("click", async () => {
    const tSeed = (document.getElementById("target-seed") as HTMLInputElement).value;
    const btn = document.getElementById("btn-manual-connect") as HTMLButtonElement;
    
    if (!tSeed) {
        alert("Please enter target seed!");
        return;
    }

    // 1. 현재 PC의 전체 IP를 가져옵니다 (예: 192.168.45.115)
    const myFullIp = await invoke<string>("get_my_full_ip"); 
    const ipParts = myFullIp.split('.');
    
    if (ipParts.length !== 4) {
        alert("Could not determine local network subnet.");
        return;
    }

    // 2. 앞의 3자리만 잘라서 서브넷 베이스를 만듭니다 (예: 192.168.45)
    const baseIp = `${ipParts[0]}.${ipParts[1]}.${ipParts[2]}`; 
    const seed = parseInt(tSeed);

    console.log(`[SYNC] Auto-Scanning subnet ${baseIp}.1~254 with seed ${seed}...`);
    btn.innerText = "SCANNING...";
    btn.disabled = true;

    try {
        await startWebRtcOfferer(baseIp, seed);
    } catch (e) {
        alert("Connection failed. Device not found on this Wi-Fi network.");
    } finally {
        btn.innerText = "AUTO CONNECT";
        btn.disabled = false;
    }
});

// 🌟 [수정] 병렬 스캔 WebRTC 연결 함수 (이전 답변과 동일, 혹시 몰라 전체 첨부)
async function startWebRtcOfferer(baseIp: string, seed: number) {
    peerConn = new RTCPeerConnection({ iceServers: [] });
    dataChannel = peerConn.createDataChannel("logis-sync");
    setupDataChannel(dataChannel);
    
    const offer = await peerConn.createOffer();
    await peerConn.setLocalDescription(offer);
    
    // Wait for ICE gathering
    await new Promise<void>(resolve => {
        if (peerConn?.iceGatheringState === 'complete') resolve();
        else {
            const check = () => { if (peerConn?.iceGatheringState === 'complete') { peerConn?.removeEventListener('icegatheringstatechange', check); resolve(); } };
            peerConn?.addEventListener('icegatheringstatechange', check);
            setTimeout(resolve, 2000);
        }
    });

    const sdp = peerConn.localDescription?.sdp || "";
    
    // 🌟 병렬 연결 시도 (1.1부터 1.254까지 전부 핑을 날려서 가장 먼저 받는 놈과 연결)
    const scanPromises = [];
    for (let i = 1; i <= 254; i++) {
        const targetIp = `${baseIp}.${i}`;
        scanPromises.push(
            invoke<string>("send_signal_offer", { targetIp, seed, sdp })
                .then(answerSdp => ({ targetIp, answerSdp }))
        );
    }

    try {
        const result = await Promise.any(scanPromises);
        await peerConn.setRemoteDescription({ type: 'answer', sdp: result.answerSdp });
        console.log(`[SYNC] Connected to ${result.targetIp} successfully via Auto Scan!`);
    } catch (e) {
        peerConn.close();
        throw new Error("Scan failed");
    }
}

listen("webrtc-offer", async (event) => {
    const [offerSdp, fromIp] = event.payload as [string, string];
    console.log(`[SYNC] Incoming offer from ${fromIp}`);
    
    peerConn = new RTCPeerConnection({ iceServers: [] });
    peerConn.ondatachannel = (e) => setupDataChannel(e.channel);

    await peerConn.setRemoteDescription({ type: 'offer', sdp: offerSdp });
    const answer = await peerConn.createAnswer();
    await peerConn.setLocalDescription(answer);

    // [FIXED] Send Answer back via TCP stream through the backend
    try {
        await invoke("submit_signal_answer", { targetIp: fromIp, sdp: answer.sdp });
        console.log(`[SYNC] Answer submitted for ${fromIp}`);
    } catch (e) {
        console.error("[SYNC] Failed to submit answer:", e);
    }
});

let mySyncSeed = 0; 
let isListenerStarted = false; // 🌟 [추가] 리스너 중복 실행 방지용 플래그

async function initSyncUI() {
    // 🌟 [CRITICAL FIX] 시드 번호를 기기별로 고정(Fix)하기 위해 로컬 DB에서 불러오거나 최초 1회만 생성하여 저장합니다.
    if (mySyncSeed === 0) {
        const savedSeed = await kvGet("my_sync_seed");
        if (savedSeed) {
            mySyncSeed = parseInt(savedSeed);
        } else {
            // 🌟 [수정] 4자리 난수(1000~9999) 대신 2자리 난수(10~99)를 생성합니다.
            mySyncSeed = Math.floor(10 + Math.random() * 90);
            await kvSet("my_sync_seed", mySyncSeed.toString());
        }
    }

    const mySyncSeedEl = document.getElementById("my-sync-seed");
    const ipPrefixEl = document.getElementById("ip-prefix");

    if (mySyncSeedEl) {
        mySyncSeedEl.innerText = mySyncSeed.toString();
    }
    if (ipPrefixEl) {
        const prefix = await invoke("get_local_network_prefix") as string;
        ipPrefixEl.innerText = prefix + ".";
    }
    
    try {
        // 🌟 [CRITICAL FIX] 아직 리스너가 열리지 않았을 때만 딱 한 번 실행하도록 차단합니다.
        if (!isListenerStarted) {
            await invoke("start_listener_command", { seed: mySyncSeed });
            isListenerStarted = true;
        }
    } catch (e) { console.error(e); }
}

let peerConn: RTCPeerConnection | null = null;
let dataChannel: RTCDataChannel | null = null;
let desktopStream: MediaStream | null = null;
let qrRotationInterval: number | null = null;

// 🌟 [추가] 양측의 인증(검증)이 완료된 후 실제 데이터 동기화를 시작하는 헬퍼 함수
function finalizeWebRtcConnection(guestSession: any) {
    const profileName = document.getElementById("nav-profile-name");
    if (profileName) {
        profileName.textContent = "✅ Mobile Linked (P2P)";
        profileName.style.color = "#4ade80";
    }
    document.getElementById("nav-qr-container")?.classList.add("hidden");
    syncDataToMobile();

    try {
        const guestName = (guestSession && guestSession.email) ? guestSession.email.split('@')[0] : "📱 Linked Device";
        const guestAddr = (guestSession && guestSession.address) ? guestSession.address : "0x0000000000000000000000000000000000000000";

        const mobileUser = {
            id: `mobile_${Date.now()}`,
            type: "user",
            name: guestName,
            from: guestAddr, 
            to: currentSession.team || "0x0000000000000000000000000000000000000000",    
            data: { origin: "local", is_device: true } 
        };
        
        invoke("upsert_items", { items: [mobileUser] }).then(() => renderNavigation());
    } catch (e) {
        console.error("[WebRTC] Failed to add device to members:", e);
    }
}

function setupDataChannel(channel: RTCDataChannel) {
    channel.onopen = async () => {
        console.log("[WebRTC] Channel OPEN! Starting Zero-Trust Auth Handshake...");
        // 🌟 [핵심 1] 채널이 열리면 데이터를 즉시 붓지 않고, 내 세션(신분증)을 보내 통성명을 시작합니다.
        channel.send(JSON.stringify({ 
            type: "auth_request", 
            session: currentSession 
        }));
    };

    channel.onmessage = async (e) => {
        try {
            const msg = JSON.parse(e.data);
            console.log("[WebRTC] Received from Peer:", msg.type);
            
            // 🌟 [핵심 2] 상대방이 인증을 요청해옴 (내가 Host/수신자 역할일 때)
            if (msg.type === "auth_request") {
                const guest = msg.session;
                
                // a. 이미 클라우드 팀원인지 내 로컬 DB(LanceDB, 클라우드 동기화됨)에서 조회
                const users = await Select["users"]({});
                const isCloudMember = users.some(u => 
                    (u.id === guest.address || u.from === guest.address) &&
                    (u.to === currentSession.team || u.cc === currentSession.team)
                );

                if (isCloudMember) {
                    // 이미 클라우드에서 인증된 팀원이면 즉시 승인 및 동기화
                    console.log("[WebRTC] Guest is an authorized Cloud Member. Auto-approving.");
                    channel.send(JSON.stringify({ type: "auth_success" }));
                    finalizeWebRtcConnection(guest);
                } else {
                    // 🌟 [CRITICAL FIX] 시드 충돌 감지 및 0-멤버 자동 양보(Yield) 로직!
                    // 상대방(Guest)의 소속 팀과 내(Host) 소속 팀이 명확히 다른데 연결이 들어왔다면, 100% 시드 중복입니다.
                    if (guest.team && currentSession.team && guest.team !== currentSession.team) {
                        const myTeamMembers = users.filter(u => u.to === currentSession.team || u.cc === currentSession.team);
                        
                        // 내 팀에 나 혼자(1명 이하)밖에 없다면(초대한 멤버가 없다면) 내가 양보하고 시드를 바꿉니다.
                        if (myTeamMembers.length <= 1) {
                            console.warn("[WebRTC] Seed collision detected! I have no members. Auto-regenerating my seed...");
                            channel.send(JSON.stringify({ type: "auth_reject", reason: "Seed collision. Auto-yielding." }));
                            peerConn?.close();
                            
                            // 시드 강제 재생성 및 로컬 DB 영구 저장
                            // 🌟 [수정] 충돌 시 새로 부여받는 시드도 2자리 난수(10~99)로 통일합니다.
                            mySyncSeed = Math.floor(10 + Math.random() * 90);
                            await kvSet("my_sync_seed", mySyncSeed.toString());
                            
                            const mySyncSeedEl = document.getElementById("my-sync-seed");
                            if (mySyncSeedEl) mySyncSeedEl.innerText = mySyncSeed.toString();
                            
                            // 🌟 Rust 바인딩된 리스너의 시드만 초고속으로 업데이트 (10048 에러 없음!)
                            await invoke("start_listener_command", { seed: mySyncSeed });
                            
                            alert(`[Network] 동일한 와이파이 내에 시드 번호 충돌이 감지되었습니다.\n멤버가 없는 현재 PC의 시드가 새 번호(${mySyncSeed})로 자동 변경 및 양보되었습니다.`);
                            return;
                        } else {
                            // 내 팀에 멤버가 있다면, 상대방이 양보하도록 거절만 날려줍니다.
                            console.warn("[WebRTC] Seed collision detected, but I have members. Rejecting guest.");
                            channel.send(JSON.stringify({ type: "auth_reject", reason: "Wrong team. Please regenerate your seed." }));
                            peerConn?.close();
                            return;
                        }
                    }

                    // b. 충돌이 아니라 정상적인 외부 기기 연결이라면 화면에 팝업을 띄워 수동 승인 진행
                    const displayId = guest.email || guest.address || "Unknown Local Device";
                    const approved = await ask(`Incoming connection from '${displayId}'.\nAre you sure you want to approve this device and share local data?`, { title: "Peer Approval Required", kind: "warning" });
                    
                    if (approved) {
                        console.log("[WebRTC] Connection manually approved by peer.");
                        channel.send(JSON.stringify({ type: "auth_success" }));
                        finalizeWebRtcConnection(guest);
                        // [Option] 필요시 여기서 proxy_fetch를 날려 클라우드 DB에도 guest.address를 강제로 등록(PUT)시킬 수 있습니다.
                    } else {
                        console.log("[WebRTC] Connection rejected by peer.");
                        channel.send(JSON.stringify({ type: "auth_reject", reason: "Rejected by Team Member" }));
                        peerConn?.close();
                    }
                }
            } 
            // 🌟 [핵심 3] 상대방이 내 접속을 승인함 (내가 Guest/발신자 역할일 때)
            else if (msg.type === "auth_success") {
                console.log("[WebRTC] Auth Approved by Host Peer!");
                finalizeWebRtcConnection(null);
            }
            // 🌟 [핵심 4] 상대방이 내 접속을 거절함
            else if (msg.type === "auth_reject") {
                alert(`WebRTC Connection blocked: ${msg.reason}`);
                peerConn?.close();
            }
            // --- 기존 통신 로직 유지 ---
            else if (msg.type === "get_detail") {
                const doc = await invoke<any>("get_document", { uuid: msg.uuid });
                if (doc && dataChannel?.readyState === "open") {
                    dataChannel.send(JSON.stringify({
                        type: "sync_detail",
                        title: `${doc.doc_type || 'Detail'} ${doc.doc_number || ''}`,
                        content: `<div style="margin-bottom:15px;"><strong>Summary:</strong><br>${doc.text}</div><hr style="border-color:rgba(255,255,255,0.1);"><pre style="white-space: pre-wrap; font-size: 0.8rem; color:#fff; background:#000; padding:15px; border-radius:8px;">${doc.json_data}</pre>`
                    }));
                }
            } else if (msg.type === "get_session") {
                // Send current desktop session info to mobile
                if (dataChannel?.readyState === "open") {
                    dataChannel.send(JSON.stringify({ 
                        type: "sync_session", 
                        data: currentSession 
                    }));
                }
            } else if (msg.type === "get_navigation") {
                // Fetch pages and users for mobile tree
                const pages = await Select["pages"]({});
                const users = await Select["users"]({});
                if (dataChannel?.readyState === "open") {
                    dataChannel.send(JSON.stringify({ 
                        type: "sync_navigation", 
                        pages: pages,
                        users: users
                    }));
                }
            } else if (msg.type === "get_chat_history") {
                // Fetch last 20 messages for mobile
                const messages = await invoke<any[]>("get_chat_messages", { limit: 20, offset: 0 });
                if (dataChannel?.readyState === "open") {
                    dataChannel.send(JSON.stringify({ 
                        type: "sync_chat_history", 
                        messages: messages
                    }));
                }
            } else if (msg.type === "search") {
                // Perform local search for mobile
                console.log("[WebRTC] Remote Search Query:", msg.query);
                const docs = await Select["items"]({ 
                    value: msg.query || "", 
                    limit: 20, 
                    offset: 0 
                });
                if (dataChannel?.readyState === "open") {
                    dataChannel.send(JSON.stringify({ type: "sync_list", data: docs }));
                }
            } else if (msg.type === "chat_message") {
                // Echo for now, or could integrate with actual AI chat logic
                dataChannel?.send(JSON.stringify({ 
                    type: "sync_chat", 
                    data: { role: "system", content: "Hub: Received '" + msg.content + "'" } 
                }));
            } else if (msg.type === "mobile_upload") {
                console.log("[WebRTC] Receiving file from mobile:", msg.name);
                try {
                    // 1. Convert Base64 to Uint8Array
                    const binaryString = atob(msg.data);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }

                    // 2. Save to a temporary location using Tauri FS
                    // We'll use a specific name to identify mobile uploads
                    const tempPath = `mobile_upload_${Date.now()}_${msg.name}`;
                    const fullPath = await invoke<string>("save_mobile_temp_file", { 
                        filename: tempPath, 
                        data: Array.from(bytes) 
                    });

                    console.log("[WebRTC] Saved mobile upload to:", fullPath);

                    // 3. Trigger Desktop's existing Extraction Logic
                    const taskId = `task_mobile_${Date.now()}`;
                    await emit("new-task-from-browser", { 
                        id: taskId, 
                        type: "image_extraction", 
                        image_path: fullPath, 
                        ref: fullPath, 
                        link: "Mobile Upload",
                        device_preference: getDevicePref()
                    });

                    // 4. Relay progress to mobile
                    // (We'll handle this in the global progress listener below)

                } catch (err) {
                    console.error("[WebRTC] Mobile upload failed:", err);
                }
            }
        } catch (err) {
            console.error("[WebRTC] Message handle error:", err);
        }
    };
}

// --- Relay Desktop Progress to Mobile ---
listen("extraction-progress", (event: any) => {
    if (dataChannel && dataChannel.readyState === "open") {
        dataChannel.send(JSON.stringify({
            type: "extraction_progress",
            payload: event.payload
        }));
    }
});


const syncDataToMobile = () => {
    if (!dataChannel || dataChannel.readyState !== "open") return;
    console.log("[WebRTC] Syncing list to mobile...");
    const docs = Array.from(document.querySelectorAll('.logis-result')).map(el => {
        const card = el as HTMLElement;
        return {
            id: card.id, uuid: card.id,
            doc_type: card.dataset.type || "General",
            text: card.querySelector('.logis-info .value')?.textContent || "",
            created_at: parseInt(card.dataset.createdAt || "0"),
            updated_at: parseInt(card.dataset.updatedAt || "0")
        };
    });
    dataChannel.send(JSON.stringify({ type: "sync_list", data: docs }));
};

listen("task-console-log", async (event: any) => {
    const { task_id, text } = event.payload;
    const key = `term_${task_id}`;
    
    // 🌟 localStorage -> Dexie(appDb) 로 영구 보존!
    let logs = (await kvGet(key)) || "";
    logs += text;
    await kvSet(key, logs);

    const termArea = document.getElementById("terminal-logs");
    if (termArea && termArea.dataset.activeTaskId === task_id) {
        termArea.appendChild(document.createTextNode(text));
        termArea.style.display = "block"; // 🌟 [추가] 텍스트가 도착하면 까만 박스를 보여줍니다!
        termArea.scrollTop = termArea.scrollHeight; 
    }
});

async function handleTaskClick(el: HTMLElement) {
    const taskId = el.dataset.taskId;
    const status = parseInt(el.dataset.status || "0");
    if (!taskId) return;
    
    console.log("[Chat] Task clicked:", taskId);

    if (taskId.startsWith("search_") && status !== 1) {
        openWidget("list");
        listView.style.display = "block";
        detailView.style.display = "none";
        if (aiResultsArea) {
            aiResultsArea.style.display = "block";
            aiResultsArea.scrollIntoView({ behavior: 'smooth' });
        }
        return;
    }

    openWidget("list"); 
    listView.style.display = "none"; 
    detailView.style.display = "flex";
    
    if (status === 1) {
        if (btnStopTask) btnStopTask.style.display = "flex";
    } else {
        if (btnStopTask) btnStopTask.style.display = "none";
    }
    if (btnDetailDelete) btnDetailDelete.style.display = "none";

    detailTitle.innerText = taskId.startsWith("search_") ? "Search Progress" : "Task Progress";
    
    let logArea = document.getElementById("extraction-log");
    if (!logArea) {
        detailContent.innerHTML = `<div id="extraction-log"></div>`;
        logArea = document.getElementById("extraction-log");
    }

    if (logArea) {
        logArea.dataset.activeTaskId = taskId;
        
        const savedLogs = await kvGet(`term_${taskId}`);
        // 🌟 저장된 로그가 있을 때만 박스를 보여주고, 없으면 숨깁니다. (Connecting... 텍스트 제거)
        const displayStyle = savedLogs && savedLogs.trim() !== "" ? "block" : "none"; 
        
        logArea.innerHTML = `
            <div id="progress-container"></div>
            <div id="terminal-logs" data-active-task-id="${taskId}" style="display: ${displayStyle}; background: #0a0a0a; color: #4ade80; padding: 12px; font-family: monospace; font-size: 0.8rem; border-radius: 6px; max-height: 250px; overflow-y: auto; white-space: pre-wrap; border: 1px solid #333; box-shadow: inset 0 0 10px rgba(0,0,0,0.8); line-height: 1.4;">${savedLogs || ""}</div>
        `;
        
        const termArea = document.getElementById("terminal-logs");
        if (termArea && displayStyle === "block") termArea.scrollTop = termArea.scrollHeight;
        
        isFetchingLogs = true;
        pendingLiveEvents = [];

        invoke<any[]>("get_task_logs", { taskId: taskId }).then(async logs => {
            if (logArea!.dataset.activeTaskId !== taskId) {
                isFetchingLogs = false;
                return;
            }

            // 🌟 로컬 스토리지엔 없지만 백엔드에 로그가 남아있을 경우 복구하면서 박스를 노출합니다!
            if (!savedLogs && logs && logs.length > 0 && termArea) {
                const reconstructed = logs.map(l => `[${l.category ? l.category.toUpperCase() : 'SYSTEM'}] ${l.summary || ''}\n`).join("");
                if (reconstructed.trim() !== "") {
                    termArea.innerHTML = reconstructed;
                    termArea.style.display = "block"; // 숨겨뒀던 박스 노출!
                    await kvSet(`term_${taskId}`, reconstructed); 
                    termArea.scrollTop = termArea.scrollHeight;
                }
            }
            
            if (logs && logs.length > 0) {
                logs.forEach(payload => {
                    payload.task_id = payload.task_id || taskId; 
                    renderProgressToUI(payload, true);
                });
            } else if (status === 1) {
                const progContainer = document.getElementById("progress-container");
                if (progContainer) progContainer.insertAdjacentHTML('beforeend', `<div id="temp-spinner" style="padding: 10px; text-align: center; color: var(--primary);"><span class="spinner active-spinner">⠋</span> Generating Insights...</div>`);
            }

            if (status === 1 || status === 10) {
                const live = livePayloads.get(taskId);
                if (live) {
                    live.task_id = taskId;
                    renderProgressToUI(live, true);
                }
            }

            isFetchingLogs = false;
            pendingLiveEvents.forEach(p => renderProgressToUI(p, false));
            pendingLiveEvents = [];

        }).catch(err => {
            console.error(err);
            isFetchingLogs = false;
        });
    }
    
    activeTaskId = taskId; 
}

async function sendSignalingMessage(hash: string, payload: any) {
    try {
        await invoke("proxy_fetch", {
            url: `${API_HOST}/relay/${hash}`,
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: payload // payload is already JSON object or will be stringified
        });
    } catch (e) {
        console.error("[WebRTC] Relay send failed:", e);
    }
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

async function showPcPairingQr() {
    const qrTarget = document.getElementById("sync-qrcode");
    const pcView = document.getElementById("pc-qr-view");
    const mobileView = document.getElementById("mobile-scan-view");
    
    if (!qrTarget || !pcView || !mobileView) return;
    
    // Clear existing interval if any
    if (qrRotationInterval) {
        clearInterval(qrRotationInterval);
        qrRotationInterval = null;
    }

    pcView.classList.remove("hidden");
    mobileView.classList.add("hidden");
    stopDesktopCamera();

    qrTarget.innerHTML = "<div style='padding:20px;'><div class='spinner'></div><p>Generating P2P Offer...</p></div>";

    try {
        // 0. Get Local IP
        const myIp = await invoke<string>("get_my_full_ip");

        // 1. Initialize PeerConnection (No STUN for local only)
        peerConn = new RTCPeerConnection({ iceServers: [] });
        
        // 2. Create Data Channel (Must create before offer)
        dataChannel = peerConn.createDataChannel("logis-sync");
        setupDataChannel(dataChannel);

        // 3. Create Offer
        const offer = await peerConn.createOffer();
        await peerConn.setLocalDescription(offer);

        // 4. Wait for ICE Gathering (Essential for LAN connection)
        console.log("[WebRTC] Gathering ICE candidates (5s)...");
        await new Promise<void>(resolve => {
            if (peerConn?.iceGatheringState === 'complete') {
                resolve();
            } else {
                const check = () => {
                    if (peerConn?.iceGatheringState === 'complete') {
                        peerConn?.removeEventListener('icegatheringstatechange', check);
                        resolve();
                    }
                };
                peerConn?.addEventListener('icegatheringstatechange', check);
                setTimeout(resolve, 5000); // 5s timeout
            }
        });

        // Add 1 second stability delay
        await new Promise(r => setTimeout(r, 1000));

        // 5. Generate QR Data (Multipart/Chunked)
        const finalSdp = peerConn.localDescription?.sdp || "";
        const laptopHash = currentSession.hash;
        
        // [Relay] Also post to relay server so mobile can find us without scan next time
        sendSignalingMessage(laptopHash, { type: "offer", sdp: finalSdp });

        const parts = extractSdp(finalSdp);
        const compactOffer = { t: "offer", h: laptopHash, i: await invoke("get_my_full_ip"), u: parts.u, p: parts.p, f: parts.f, s: parts.s };
        const qrData = JSON.stringify(compactOffer);
        
        console.log(`[WebRTC] Offer Generated. Compact Length: ${qrData.length}`);

        // 6. Show Single QR
        qrTarget.innerHTML = ""; 
        const header = document.createElement("div");
        header.style.marginBottom = "10px";
        header.style.fontWeight = "bold";
        header.style.color = "var(--primary)";
        header.innerText = `Scan to Pair (P2P)`;
        qrTarget.appendChild(header);

        const qrDiv = document.createElement("div");
        qrTarget.appendChild(qrDiv);

        new (window as any).QRCode(qrDiv, {
            text: qrData,
            width: 250, height: 250, 
            colorDark: "#000000", colorLight: "#ffffff",
            correctLevel: (window as any).QRCode.CorrectLevel.M
        });
        // Clean up interval when view changes
        const cleanup = () => {
            if (qrRotationInterval) clearInterval(qrRotationInterval);
            document.getElementById("btn-switch-to-camera")?.removeEventListener("click", cleanup);
        };
        document.getElementById("btn-switch-to-camera")?.addEventListener("click", cleanup);

    } catch (e) {
        console.error("[WebRTC] Offer Generation Failed:", e);
        qrTarget.innerHTML = "<p style='color:red'>Failed to gen offer</p>";
    }
}

btnDetailDelete?.addEventListener("click", async () => {
    console.log("[WIDGET] Delete button clicked. UUID:", currentDetailUuid);
    if (!currentDetailUuid) {
        console.error("[WIDGET] No document UUID selected for deletion.");
        return;
    }
    
    try {
        const confirmed = await ask("Are you sure you want to delete this document?", { 
            title: "Confirm Delete", 
            kind: "warning" 
        });

        if (confirmed) {
            console.log("[WIDGET] Deletion confirmed for:", currentDetailUuid);
            const res = await invoke<string>("delete_document", { uuid: currentDetailUuid });
            console.log("[WIDGET] Delete response:", res);
            
            detailView.style.display = "none"; 
            listView.style.display = "block"; 
            refreshList(); 
        }
    } catch (e) { 
        console.error("[WIDGET] Deletion process failed:", e); 
    }
});

async function refreshList() {
    currentPage = 0; hasMore = true; cachedDocs = []; selectedUuids.clear();
    listCurrentY = 0; // Reset scroll
    if(docListContainer) docListContainer.innerHTML = "";
    await loadMoreDocs(true);
}

async function loadMoreDocs(reset: boolean = false, isSync: boolean = false) {
    if (reset) {
        currentPage = 0; hasMore = true;
        if (docListContainer) docListContainer.innerHTML = "";
        cachedDocs = [];
        listCurrentY = 0;
        updateListTransform();
    }

    if (isLoading || (!reset && !isSync && !hasMore)) {
        if (reset && !isSync) stopSpinner();
        return;
    }

    if (!isSync) startSpinner();
    isLoading = true;
    
    // 🌟 오직 숨겨둔 span 태그(headerLoading)만 살짝 켭니다. 
    // h2 태그(listTitle)를 덮어씌우는 코드는 완전히 삭제했습니다.
    if (headerLoading) {
        headerLoading.style.display = "inline-block";
    }
    
    try {
        // 🌟 [CRITICAL FIX] 현재 탭에 맞는 데이터만 필터링!
        let baseFilter = "";
        if (currentSearchMode === "shipping") {
            baseFilter = "type IN ('tracking', 'receiving', 'shipping', 'bl', 'awb', 'BL', 'AWB', 'CI', 'PI', 'PL', 'CO', 'LC', 'TRACKING', 'shipping_doc', 'Unknown')";
        } else if (currentSearchMode === "analytic") {
            // 🌟 [추가] Analytic 모드에서는 raw user interaction 로그와 리포트를 불러옵니다.
            baseFilter = "type IN ('click', 'hover', 'change', 'report')"; 
        } else {
            baseFilter = "type IN ('sales', 'goods', 'order', 'event', 'coupon', 'review', 'pages')";
        }

        // 기존 내비게이션 필터가 있다면 안전하게 괄호로 묶어서 AND 조건 추가
        if (activeContext.ref) baseFilter = `(${baseFilter}) AND ref = '${activeContext.ref}'`;
        else if (activeContext.bcc) baseFilter = `(${baseFilter}) AND bcc = '${activeContext.bcc}'`;
        else if (activeContext.cc) baseFilter = `(${baseFilter}) AND cc = '${activeContext.cc}'`;

        // 🌟 [CRITICAL FIX 4] 사이드바 태그(domain, type)는 위의 activeContext를 통해 SQL 필터로 100% 적용되었습니다.
        // 이 태그들을 텍스트로 엮어서 억지로 AI 검색에 던지면 검색 결과가 0건이 되고 VRAM이 폭발합니다. 삭제합니다!
        const textQuery = searchInput?.value.trim() || "";

        let finalFilter = baseFilter;
        let latestUpdateTime = 0;
        let oldestCreatedAt = 0;

        // [TIMESTAMPS] Scan UI for current range
        const allCards = docListContainer.querySelectorAll('.logis-result');
        allCards.forEach(el => {
            const up = parseInt((el as HTMLElement).dataset.updatedAt || "0");
            const cr = parseInt((el as HTMLElement).dataset.createdAt || "0");
            if (up > latestUpdateTime) latestUpdateTime = up;
            if (oldestCreatedAt === 0 || cr < oldestCreatedAt) oldestCreatedAt = cr;
        });

        if (isSync) {
            // [Top Pull] Newer than latest update
            const syncFilter = `updated_at > ${latestUpdateTime}`;
            finalFilter = baseFilter ? `${baseFilter} AND (${syncFilter})` : syncFilter;
        } else if (!reset && oldestCreatedAt > 0) {
            // [Bottom Pull] Older than oldest created
            const historyFilter = `created_at < ${oldestCreatedAt}`;
            finalFilter = baseFilter ? `${baseFilter} AND (${historyFilter})` : historyFilter;
        }

        let docs: any[] = [];
        
        if (textQuery) {
            // 🌟 텍스트 검색 시에도 프론트 래퍼를 버리고 Rust의 search_documents를 직접 타격합니다.
            const searchResults = await invoke<any[]>("search_documents", {
                query: textQuery,
                limit: pageSize,
                offset: 0,
                filter: finalFilter || null // 👈 이제 필터가 절대 증발하지 않고 Rust로 꽂힙니다!
            });
            
            // Rust의 search_documents는 [id, text, score] 형태의 배열만 반환하므로, 
            // 받아온 ID를 이용해 리스트 카드 생성에 필요한 원본 문서(Full Document)를 직접 꺼내옵니다.
            for (const res of searchResults) {
                const docId = res[0];
                const fullDoc = await invoke<any>("get_document", { uuid: docId });
                if (fullDoc) {
                    docs.push(fullDoc);
                }
            }
        } else {
            // 🌟 탭을 클릭하거나 일반 스크롤을 할 때는 Tauri Invoke를 직접 타격하여 완벽한 필터링을 보장합니다.
            docs = await invoke<any[]>("get_all_documents", {
                limit: pageSize,
                offset: 0,
                filter: finalFilter || null
            });
        }

        if (!isSync && docs.length < pageSize) hasMore = false;

        if (docs.length > 0) {
            const mode = isSync ? 'prepend' : 'append';
            upsertListItems(docs, mode);
            
            // [NEW] If syncing, also refresh the navigation tree (Pages / Users)
            if (isSync) {
                renderNavigation();
            }
        } else if (reset) {
            docListContainer.innerHTML = `<div class="empty">No documents found.</div>`;

        }
    } catch (e) { 
        console.error("[WIDGET] loadMoreDocs error:", e);
        if (reset && docListContainer) docListContainer.innerHTML = `<div style='text-align:center; padding:20px; color:#ef4444;'>Error loading data.</div>`;
    } 
    finally { 
        isLoading = false; 
        
        // 🌟 로딩 종료: Loading... 글자를 완전히 숨김 (아무것도 표시 안 됨)
        if (headerLoading) {
            headerLoading.style.display = "none";
        }
        
        if (!isSync) stopSpinner();
    }
}

function upsertListItems(docs: any[], mode: 'prepend' | 'append') {
    if (!docListContainer) return;

    const scrollEl = document.getElementById("list-scroll");
    const prevScrollHeight = scrollEl ? scrollEl.scrollHeight : 0;
    const wasAtTop = listCurrentY <= 10; 

    const sortedBatch = [...docs].sort((a, b) => b.created_at - a.created_at);
    const processBatch = mode === 'prepend' ? [...sortedBatch].reverse() : sortedBatch;

    processBatch.forEach(doc => {
        const docId = doc.id || doc.uuid || (doc.data && (doc.data.id || doc.data.uuid)) || doc.uuid_val || doc.ref || doc.index;
        const existingEl = docListContainer.querySelector(`[id="${docId}"]`) as HTMLElement;

        // 🌟 [CRITICAL FIX] item2html은 숨겨진 checkbox와 메인 카드(div) 2개의 요소를 생성합니다.
        const html = item2html(doc, false, currentDetectedUrl);
        const temp = document.createElement('div');
        temp.innerHTML = html;
        
        // 🌟 클래스 이름(.logis-result)이 누락되거나 충돌하는 상황을 원천 차단하기 위해 
        // 부여된 ID 값을 이용해 가장 확실하게 두 요소를 뜯어옵니다.
        const newCheckbox = temp.querySelector(`input#more-${docId}`) as HTMLElement || temp.querySelector('.toggle-more') as HTMLElement;
        const newCard = temp.querySelector(`div[id="${docId}"]`) as HTMLElement || temp.querySelector('.logis-result') as HTMLElement;

        if (existingEl) {
            const cachedUpdatedAt = parseInt(existingEl.dataset.updatedAt || "0");
            if (doc.updated_at > cachedUpdatedAt) {
                console.log(`[List] Updating item ${docId}`);
                
                // 체크박스와 카드를 각각 찾아서 안전하게 교체(Replace)합니다.
                const oldCheckbox = docListContainer.querySelector(`#more-${docId}`);
                if (oldCheckbox && newCheckbox) docListContainer.replaceChild(newCheckbox, oldCheckbox);
                
                if (newCard) {
                    docListContainer.replaceChild(newCard, existingEl);
                    bindCardEvents(newCard, doc);
                }
            }
        } else {
            // 새 카드를 삽입할 때도 체크박스와 카드를 순서대로 온전히 다 넣습니다.
            if (mode === 'prepend') {
                if (newCard) docListContainer.prepend(newCard);
                if (newCheckbox) docListContainer.prepend(newCheckbox);
            } else {
                if (newCheckbox) docListContainer.appendChild(newCheckbox);
                if (newCard) docListContainer.appendChild(newCard);
            }
            if (newCard) bindCardEvents(newCard, doc);
        }
    });

    if (mode === 'prepend' && scrollEl) {
        const newScrollHeight = scrollEl.scrollHeight;
        const heightDiff = newScrollHeight - prevScrollHeight;
        if (heightDiff > 0) {
            if (wasAtTop) listCurrentY = 0;
            else listCurrentY += heightDiff;
            updateListTransform();
        }
    }
}

function bindCardEvents(el: HTMLElement, doc: any) {
    const toggleCheckbox = el.querySelector('.toggle-more') as HTMLInputElement;
    const moreContent = el.querySelector('.more-content') as HTMLElement;
    const moreLabel = el.querySelector('.more-label') as HTMLElement;
    const relateContainer = el.querySelector('.logis-relate') as HTMLElement;

    // 🌟 [PARITY] 클라우드의 Relay(관계 병합) 아코디언 토글 이벤트
    if (toggleCheckbox && moreContent && moreLabel) {
        toggleCheckbox.addEventListener('change', async () => {
            if (toggleCheckbox.checked) {
                // 아코디언 열림
                moreContent.style.display = "block";
                moreLabel.innerHTML = "fold ▲";
                
                // 🌟 열릴 때 연관된 데이터(Foreign/Primary)를 DB에서 긁어와 병합합니다!
                if (relateContainer) {
                    await loadRelatedData(doc, relateContainer);
                }
            } else {
                // 아코디언 닫힘
                moreContent.style.display = "none";
                moreLabel.innerHTML = "more ▼";
            }
        });
    }

    el.addEventListener("click", (e) => {
        const target = e.target as HTMLElement;
        
        // 아코디언 래퍼나 내부 연관 데이터 클릭 시, 메인 상세 페이지로 넘어가지 않도록 차단
        if (target.closest('.toggle-more') || target.closest('.more-label') || target.closest('.more-content') || target.closest('.logis-relate')) {
            return;
        }

        const docId = doc.id || doc.uuid || (doc.data && (doc.data.id || doc.data.uuid)) || doc.uuid_val || doc.ref || doc.index;
        if (!target.closest('a') && !target.closest('input') && !target.closest('button')) {
            if (docId) showDetail(String(docId));
        }
    });
}

// 🌟 [PARITY] 클라우드 Relay 로직의 클라이언트 사이드 이식
async function loadRelatedData(doc: any, container: HTMLElement) {
    if (!container || container.dataset.loaded === "true") return;
    
    // 스피너 표시
    container.innerHTML = `<div style="padding:10px; text-align:center; font-size:0.8rem; color:var(--primary);"><span class="active-spinner">⠋</span> Loading related data...</div>`;
    
    try {
        const docId = doc.id || doc.uuid;
        const docRef = doc.ref;
        
        // 1. 나를 부모로 가지는 자식들 (ref = 내 ID)
        let filterStr = `ref = '${docId}'`; 
        
        // 2. 나와 같은 출신(링크)을 가진 형제들 (ref = 내 출처)
        if (docRef && docRef !== "") {
            filterStr += ` OR ref = '${docRef}'`; 
        }
        
        // 백엔드(LanceDB)에 쿼리 전송
        const relatedDocs = await invoke<any[]>("get_all_documents", {
            limit: 10,
            offset: 0,
            filter: filterStr
        });

        // 본인 제외 및 중복 제거
        const uniqueDocs = relatedDocs.filter(d => (d.id || d.uuid) !== docId);

        if (uniqueDocs.length > 0) {
            const relatedHtml = uniqueDocs.map(d => {
                // 🌟 하위 아이템은 무한 확장을 막기 위해 checked=true (펼쳐짐) 및 부가 정보 축소 형태로 렌더링
                return item2html(d, true, currentDetectedUrl);
            }).join("");
            
            // 연관 데이터 UI 주입
            container.innerHTML = `<div style="margin-top:15px; border-top:1px dashed rgba(255,255,255,0.2); padding-top:10px;">
                <strong style="font-size:0.8rem; color:#aaa; margin-bottom:10px; display:block;">🔗 Related Documents</strong>
                ${relatedHtml}
            </div>`;
            
            // 내부 연관 카드의 클릭 이벤트(상세 페이지 진입)도 재귀적으로 바인딩
            const newCards = container.querySelectorAll('.logis-result');
            newCards.forEach((card, idx) => {
                bindCardEvents(card as HTMLElement, uniqueDocs[idx]);
            });
        } else {
            // 연관 데이터가 없으면 깔끔하게 비움
            container.innerHTML = ""; 
        }
        
        container.dataset.loaded = "true"; // 불필요한 중복 쿼리 방지 (캐싱)
        
    } catch (e) {
        console.error("[Relay] Failed to load related data:", e);
        container.innerHTML = `<div style="color:#ef4444; font-size:0.7rem; padding:5px;">Failed to load related data.</div>`;
    }
}

function renderDocs(docs: any[]) {
    // This is now handled by upsertListItems for consistency
    upsertListItems(docs, 'append');
}

async function showDetail(uuid: string) {
    console.log("[WIDGET] Opening detail view for ID:", uuid);
    if (!uuid) {
        console.error("[WIDGET] Cannot open detail: ID is undefined");
        return;
    }
    currentDetailUuid = uuid;
    listView.style.display = "none";
    detailView.style.display = "flex";
    if (btnDetailDelete) btnDetailDelete.style.display = "flex";
    if (btnStopTask) btnStopTask.style.display = "none";

    detailTitle.innerText = "Loading...";
    detailContent.innerHTML = "Fetching details...";
    try {
        const doc = await invoke<any>("get_document", { uuid: uuid });
        if (doc) {
            detailTitle.innerText = `${doc.doc_type || 'Detail'} ${doc.doc_number || ''}`;
            let prettyJson = doc.json_data;
            try { prettyJson = JSON.stringify(JSON.parse(doc.json_data), null, 2); } catch(e) {}
            detailContent.innerHTML = `<div style="margin-bottom:10px;"><strong>Summary:</strong><br>${doc.text}</div><hr style="border-color:#444;"><pre style="white-space: pre-wrap; font-size: 0.8rem; color:#fff; background:#111; padding:10px;">${prettyJson}</pre>`;
        } else {
            detailContent.innerHTML = `<div class="empty">Document not found in database.</div>`;
        }
    } catch (e) { 
        console.error("[WIDGET] get_document failed:", e);
        detailContent.innerHTML = "Failed to load document details: " + e; 
    }
}

btnDetailBack?.addEventListener("click", () => { detailView.style.display = "none"; listView.style.display = "block"; });
document.getElementById("btn-settings-back")?.addEventListener("click", collapseWidget);

// 🌟 [수정] 세팅 패널이 열려있을 때는 세팅을 닫고 리스트로 복귀하며, 일반 리스트 상태일 때는 위젯을 닫습니다.
btnListBack?.addEventListener("click", () => {
    const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
    if (settingsToggle && settingsToggle.checked) {
        settingsToggle.checked = false;
        settingsToggle.dispatchEvent(new Event("change")); // 세팅 패널 닫기 이벤트 트리거
    } else {
        collapseWidget(); // 기존처럼 위젯 닫기
    }
});

document.getElementById("nav-signin")?.addEventListener("click", () => openWidget("settings"));
document.getElementById("nav-signout")?.addEventListener("click", () => { document.getElementById("btn-logout")?.click(); });

async function handleImageUpload(path: string) {
    currentImage = path;
    if (navPreviewContainer && navImgThumbnail) {
        navPreviewContainer.classList.remove("hidden");
        navUploadBtn?.classList.add("active-emoji");
        
        // 🌟 [수정] 이미지 업로드 시 검색창을 막고 버튼을 숨기던 로직을 제거합니다.
        searchInput.disabled = false;
        if (btnSubmit) btnSubmit.style.display = "flex";
        if (btnExtract) btnExtract.style.display = "flex";
        
        try {
            const contents = await readFile(currentImage);
            const blob = new Blob([contents]);
            const reader = new FileReader();
            reader.onloadend = () => { navImgThumbnail.src = reader.result as string; };
            reader.readAsDataURL(blob);
        } catch (e) { 
            navImgThumbnail.src = convertFileSrc(currentImage); 
        }

        console.log("[WIDGET] Image selected. Extraction button (⚡) is now visible.");
        
        // 🌟 [추가] 이미지 선택 시 설정(채팅) 탭으로 화면을 전환하고 스크롤을 맨 아래로 내립니다.
        openWidget("settings");
        setTimeout(() => {
            const scrollEl = document.getElementById("chat-scroll");
            const container = document.querySelector(".chat-container") as HTMLElement;
            if (scrollEl && container) {
                const maxScroll = Math.max(0, scrollEl.scrollHeight - container.clientHeight);
                currentY = maxScroll;
                scrollEl.style.transition = "transform 0.3s ease-out";
                updateTransform();
                setTimeout(() => { scrollEl.style.transition = ""; }, 300);
            }
        }, 100);
    }
}

navImgClear?.addEventListener("click", async () => {
    currentImage = null;
    navPreviewContainer.classList.add("hidden");
    navUploadBtn?.classList.remove("active-emoji");
    
    // 🌟 [유지] 검색창 활성화 및 조건부 검색 버튼 노출을 명시적으로 보장합니다.
    searchInput.disabled = false;
    if (btnSubmit) {
        const currentVal = searchInput.value.trim();
        if (currentVal !== "" && !isQueryActive(currentVal)) {
            btnSubmit.style.display = "flex";
        } else {
            btnSubmit.style.display = "none";
        }
    }
    
    await updateExtractButtonVisibility();
});

navUploadBtn?.addEventListener("click", async () => {
    const file = await open({ multiple: false, filters: [{ name: 'Image', extensions: ['png', 'jpg', 'jpeg'] }] });
    if (file) await handleImageUpload(file as string);
});

const ZERO_ADDRESS = "0x0000000000000000000000000000000000000000";
const timezoneOffset = new Date().getTimezoneOffset() * 60 * 1000;

async function checkAuthStatus() {
    if (!currentSession.hash) return;
    const origin = "https://commerce.logis.center"; 
    const now = Date.now();
    const createdAt = now - timezoneOffset; 
    try {
        // 🌟 [CRITICAL FIX] Tauri의 window.location.href는 'localhost'이므로 서버가 도메인(cc)을 파악하지 못합니다.
        // 브라우저에서 감지된 URL(currentDetectedUrl)이나 기본 클라우드 주소를 전달해야 완벽히 매칭됩니다!
        const queryParams: Record<string, string> = { 
            origin: origin, 
            created_at: createdAt.toString(), 
            hash: currentSession.hash, 
            href: currentDetectedUrl || "https://commerce.logis.center/tracking" 
        };
        if (currentSession.token) queryParams.token = currentSession.token;
        const params = new URLSearchParams(queryParams);
        const finalUrl = `${API_HOST}/?${params.toString()}`.toLowerCase();
        
        const data = await invoke<any>("proxy_fetch", { url: finalUrl, method: "GET", headers: { "Content-Type": "application/json" }, session_params: { hash: currentSession.hash, token: currentSession.token } });
        
        // [FIX] Step the spinner frame only when result arrives
        stepQrSpinner();

        let session = data.session || data; 
        if (session && session.hash) {
            const hashChanged = session.hash !== currentSession.hash;
            currentSession = { ...currentSession, ...session };
            await saveSession();
            if (hashChanged && !currentSession.email && currentTab === "settings") performQrAuth();
            if (currentSession.email) { 
                await invoke("initialize_hub", { address: currentSession.address, email: currentSession.email, flag: session.flag || "kr" }); 
                updateAuthUI(); fetchChatHistory(); syncData();
            }
        }
    } catch (e) { 
        console.warn("Auth check failed:", e); 
    }
}

function updateAuthUI() {
    const authStatus = document.getElementById("auth-status-text");
    const btnLogout = document.getElementById("btn-logout");
    const btnQrAuth = document.getElementById("btn-qr-auth");
    const chatForm = document.querySelector(".chat-form") as HTMLElement;
    const cloudToggle = document.getElementById("cloud-mode-toggle") as HTMLInputElement;
    
    // 🌟 [추가] Cloud Members 섹션을 통째로 잡습니다.
    const cloudMembersSection = document.getElementById("nav-list-users")?.closest(".nav-section") as HTMLElement;

    if (currentSession.email) {
        if (authStatus) authStatus.innerText = "Authenticated";
        if (btnLogout) btnLogout.style.display = "block";
        if (btnQrAuth) btnQrAuth.style.display = "none";
        if (chatForm) chatForm.classList.remove("hidden");
        const qrMsg = document.getElementById("msg-qr-auth");
        if (qrMsg) qrMsg.remove();
        
        if (cloudToggle) {
            cloudToggle.disabled = false;
            cloudToggle.title = "Cloud AI Mode is available";
        }

        // 🌟 [수정] 로그인 성공 시 Cloud Members 영역을 표시하되, 세팅 화면이 켜져있다면 숨김을 유지합니다.
        const isSettingsOpen = (document.getElementById("settings-toggle") as HTMLInputElement)?.checked;
        if (cloudMembersSection) cloudMembersSection.style.display = isSettingsOpen ? "none" : ""; 
        
    } else {
        if (authStatus) authStatus.innerText = "Waiting for Auth...";
        if (btnLogout) btnLogout.style.display = "none";
        if (btnQrAuth) btnQrAuth.style.display = "block";
        if (chatForm) chatForm.classList.add("hidden");
        
        if (cloudToggle) {
            cloudToggle.disabled = true;
            cloudToggle.checked = false;
            cloudToggle.title = "Login required to use Cloud AI";
        }

        // 🌟 [추가] 비로그인 시 Cloud Members 영역 완전히 숨김
        if (cloudMembersSection) cloudMembersSection.style.display = "none"; 
    }
}

async function performQrAuth() {
    if (!chatTalks || !currentSession.hash) return;
    const existing = document.getElementById("msg-qr-auth");
    if (existing) existing.remove();
    const html = `<div class="chat-talk system" id="msg-qr-auth" data-created-at="9999999999999"><div class="chat-message" style="padding:0; background: #fff; color: #000; border:0;"><div style="font-size:0.8rem; font-weight: bold; margin-bottom: 15px; color: #333;"><span id="qr-auth-spinner" class="active-spinner" style="margin-right:5px; font-family:monospace; color:#000; font-weight:bold;">⠋</span>Scan the QR code</div><div id="qr-code-target" style="display: inline-block; background: #fff; border-radius: 8px;"></div></div></div>`;
    chatTalks.insertAdjacentHTML('beforeend', html);
    const qrTarget = document.getElementById("qr-code-target");
    if (qrTarget) {
        qrTarget.innerHTML = "";
        new (window as any).QRCode(qrTarget, { text: `mailto:${encodeURIComponent(currentSession.hash + ".logis.center@oauth.email")}`, width: 300, height: 300, colorDark: "#000000", colorLight: "#ffffff", correctLevel: (window as any).QRCode.CorrectLevel.M });
        const scroll = document.getElementById("chat-scroll");
        if (scroll) scroll.scrollTop = scroll.scrollHeight;
    }
}

// 🌟 [PARITY] Window Focus/Blur 이벤트 리스너 추가
window.addEventListener("blur", () => {
    isFocus = false;
    if (chatPollInterval) {
        clearTimeout(chatPollInterval);
        chatPollInterval = null;
        console.log("[WIDGET] Window blurred. Polling paused to save resources.");
    }
});

window.addEventListener("focus", () => {
    isFocus = true;
    
    // 🌟 [CRITICAL FIX] 크롬 브라우저를 끄고 앱 화면으로 돌아왔을 때 즉시 브라우저 생존 여부를 검사하여 
    // 브라우저 런처 버튼 노출 및 번개 버튼 상태를 원상복구합니다.
    syncBrowserStatus();
    
    // 🌟 [CRITICAL FIX] 이메일(로그인)이 없는 상태에서도 QR 인증 대기를 위해 폴링이 무조건 재개되어야 합니다!
    if (!chatPollInterval) {
        console.log("[WIDGET] Window focused. Polling resumed.");
        // 창을 다시 봤을 때 즉시 1회 최신화 (로그인 된 상태일 때만)
        if (currentSession.email) {
            fetchChatHistory(false, true); 
        }
        startPolling();
    }
});

// 🌟 [PARITY] startPolling 함수 업그레이드 (setInterval -> 재귀적 setTimeout)
function startPolling() {
    if (chatPollInterval) {
        clearTimeout(chatPollInterval);
        chatPollInterval = null;
    }
    if (!isFocus) return; 
    
    const poll = async () => {
        if (!isFocus) return; 
        
        // 히스토리(Settings) 창이 열려있을 때만 서버에 인증/동기화 요청을 보냅니다!
        if (currentTab === "settings" && isExpanded) {
            try {
                if (!currentSession.email) {
                    await checkAuthStatus();
                } else {
                    // 🌟 [CRITICAL FIX] 로컬 DB만 조회하던 fetchChatHistory 대신, 
                    // front.js와 동일하게 실제 서버와 통신하는 syncData를 호출해야 합니다!
                    await syncData(); 
                }
            } catch (e) {
                console.error("[POLLING] Error during poll:", e);
            }
        }

        // 🌟 [핵심] Rust 백엔드(proxy_fetch)의 응답을 완전히 받은 후, 
        // 여전히 앱이 포커스 상태라면 다시 3초를 대기하고 다음 폴링을 예약합니다.
        if (isFocus) {
            chatPollInterval = window.setTimeout(poll, 3000);
        }
    };

    // 첫 시작 시 3초 대기 후 실행
    chatPollInterval = window.setTimeout(poll, 3000);
}



async function saveSession() { await kvSet("chat_session", JSON.stringify(currentSession)); }

// 🌟 [추가] Pages 숨김 처리 상태를 담을 전역 배열
let hiddenPages: string[] = [];

async function initSession() {
    // 🌟 [추가] Dexie에서 숨김 페이지 목록을 불러옵니다.
    const savedHiddenPages = await kvGet("hidden_pages");
    if (savedHiddenPages) {
        try { hiddenPages = JSON.parse(savedHiddenPages); } catch(e) {}
    }

    // 🌟 [CRITICAL FIX 1] 앱 최초 실행 시, Dexie에서 묵은 터미널 찌꺼기를 완벽 청소합니다!
    const allKeys = await appDb.table("kv_store").toCollection().primaryKeys();
    for (const key of allKeys) {
        if (typeof key === "string" && key.startsWith("term_")) {
            await kvRemove(key);
        }
    }

    // 🌟 search_mode 도 여기서 비동기로 불러와 초기화합니다.
    const savedSearchMode = await kvGet("search_mode");
    if (savedSearchMode) {
        currentSearchMode = savedSearchMode;
        applySearchModeUI(); // UI에 즉시 반영
    }

    const saved = await kvGet("chat_session");
    if (saved) { try { currentSession = { ...currentSession, ...JSON.parse(saved) }; } catch (e) {} } 
    else { const legacy = await kvGet("device_hash"); if (legacy) currentSession.hash = legacy; }
    
    if (!currentSession.hash && ethers) { 
        const w = ethers.Wallet.createRandom(); 
        currentSession.hash = w.address.toLowerCase().replace("0x", ""); 
        await saveSession(); 
    }
    
    await saveSession(); 
    currentSession.address = currentSession.address || ZERO_ADDRESS; 
    currentSession.team = currentSession.team || await hashId(ZERO_ADDRESS); 
    updateAuthUI(); 
    startPolling();

    try {
        console.log("[WIDGET] UI Ready handshake starting...");
        
        // 🌟 1. 새로고침 전 담아두었던 프론트엔드 대기열 먼저 복구 (Dexie 비동기 처리)
        await GlobalTaskManager.loadQueue();
        
        const data = await invoke<any>("mark_ui_ready");

        // 🌟 [CRITICAL FIX] 백엔드에서 실제로 실행 중인 작업이 있다면 프론트엔드 큐 매니저를 바쁨(Busy) 상태로 잠급니다!
        // 이렇게 해야 대기열에 있던 검색 작업이 새로고침 즉시 백엔드로 뚫고 들어가는 것을 막을 수 있습니다.
        const runningTask = data.tasks && data.tasks.find((t: any) => t.status === 1);
        if (runningTask) {
            GlobalTaskManager.isBusy = true;
            GlobalTaskManager.currentTaskId = runningTask.id;
            console.log(`[QUEUE] Backend is busy with ${runningTask.id}. Pausing frontend queue.`);
        }

        const currentLockId = await kvGet("sys_lock");
        if (currentLockId) {
            const isTaskStillAlive = data.tasks && data.tasks.some((t: any) => t.id === currentLockId && (t.status === 1 || t.status === 10));
            // 🌟 2. DB엔 없어도 TS Queue에 남아있는 녀석은 아직 Rust로 안 넘어간 정당한 대기열입니다.
            const isPendingInQueue = GlobalTaskManager.queue.some(q => q.taskId === currentLockId);
            
            if (!isTaskStillAlive && !isPendingInQueue) {
                console.log(`[LOCK] Zombie detected: ${currentLockId} is not active in Backend or Queue. Releasing.`);
                await kvRemove("sys_lock");
                await GlobalTaskManager.forceReset();
            } else {
                console.log(`[LOCK] Valid task detected: ${currentLockId}. Keeping lock.`);
                if (currentLockId.startsWith("search_")) isSearching = true;
                else isExtracting = true;
                activeTaskId = currentLockId;
                startSpinner();
            }
        }

        // 🌟 3. 큐 복구 후 밀린 작업이 있다면 자동 재개
        if (GlobalTaskManager.queue.length > 0 && !GlobalTaskManager.isBusy) {
            GlobalTaskManager.processNext();
        }

        // 🌟 4. DOM 청소 시 TS Queue 생존자도 보호
        const allBubbles = chatTalks.querySelectorAll('.task-bubble');
        allBubbles.forEach(el => {
            const bubbleId = el.id;
            const bubbleStatus = parseInt((el as HTMLElement).dataset.status || "0");
            if (bubbleStatus === 1 || bubbleStatus === 10) {
                const existsInDb = data.tasks && data.tasks.some((t: any) => t.id === bubbleId);
                const existsInQueue = GlobalTaskManager.queue.some(q => q.taskId === bubbleId);
                
                if (!existsInDb && !existsInQueue) {
                    console.log(`[UI] Removing zombie bubble from DOM: ${bubbleId}`);
                    el.remove();
                    const queryEl = document.getElementById(`${bubbleId}_query`);
                    if (queryEl) queryEl.remove();
                }
            }
        });

        // 🌟 새로고침 시 DB에 살아남은 진짜 대기열 목록만 복구
        if (data.tasks && data.tasks.length > 0) {
            // 🌟 [CRITICAL FIX] forEach의 콜백은 동기 함수이므로 내부에서 await를 쓸 수 없습니다. 비동기 순차 처리를 위해 for...of 루프로 변경합니다.
            for (const t of data.tasks) {
                if (t.status === 10 || t.status === 1) {
                    let taskQuery = "";
                    try {
                        // 🌟 t.data_json에 저장된 원본 query를 추출합니다.
                        const taskData = typeof t.data_json === 'string' ? JSON.parse(t.data_json) : t.data_json;
                        taskQuery = taskData.query || "";
                    } catch(e) {
                        console.warn("[WIDGET] Failed to parse task data for query recovery:", e);
                    }

                    // 1. 사용자 질문 말풍선 강제 복구 (100% DB 기반)
                    if (taskQuery) {
                        const userMsgId = `${t.id}_query`;
                        if (!document.getElementById(userMsgId)) {
                            await renderMessage({
                                id: userMsgId,
                                role: "user",
                                text: taskQuery,
                                status: 9,
                                // 🌟 [최종 수정] 시스템 태스크(t.created_at)보다 100ms 앞당겨 정렬 엔진의 충돌을 완벽히 회피합니다.
                                created_at: Number(t.created_at) - 100,
                                updated_at: Number(t.created_at) - 100
                            });
                        }
                    }

                    // 2. 시스템 대기열/진행 상태 말풍선 복구
                    if (!document.getElementById(t.id)) {
                        await renderMessage({
                            id: t.id,
                            task_id: t.id,
                            role: "system_task",
                            text: t.id.startsWith("search_") ? "Task Started: AI Search" : ("Task Started: " + (t.ref || "Local Source")),
                            status: t.status,
                            // 🌟 [핵심 수정] 기준 시간(t.created_at) 그대로 사용하여 질문 뒤에 오게 함
                            created_at: t.created_at,
                            updated_at: t.updated_at
                        });
                    }
                    
                    // 3. 진행 중(1)이거나 대기 중(10)인 작업에 대한 전역 상태 락 설정
                    // 🌟 [CRITICAL FIX] 검색 작업인데 프론트엔드 큐(TS Queue)에 존재하지 않는다면 실행될 가능성이 없는 유령(Ghost)입니다.
                    const isSearchGhost = t.id.startsWith("search_") && !GlobalTaskManager.queue.some(q => q.taskId === t.id);

                    if (!isSearchGhost) {
                        await kvSet("sys_lock", t.id);
                        
                        if (t.id.startsWith("search_")) {
                            isSearching = true;
                            if (btnSubmit) btnSubmit.style.display = "none";
                        } else {
                            isExtracting = true;
                        }
                        activeTaskId = t.id;
                        startSpinner();
                    } else {
                        console.log(`[WIDGET] Ignoring ghost search task: ${t.id}`);
                    }
                }
            }
            await updateExtractButtonVisibility();
        }

        // 브라우저 런처 상태 동기화
        if (btnAutoLaunch) {
            btnAutoLaunch.style.display = (data.browser_status === "running") ? "none" : "flex";
        }

        // 🌟 [CRITICAL FIX] 앱 새로고침 시 백엔드에서 감지 중인 브라우저 현재 URL 상태를 완벽 복구합니다.
        if (data.current_url) {
            currentDetectedUrl = data.current_url;
            isCurrentShop = data.is_client || data.is_admin;
        }

        // 아이템 리스트 캐싱
        if (data.items && data.items.length > 0 && cachedDocs.length === 0) {
            cachedDocs = data.items;
            renderDocs(data.items);
            currentPage = 1;
        }

        // 🌟 [핵심 분기 로직] 로그인 상태 확인 후 렌더링 방식 결정
        if (currentSession.email) {
            console.log("[WIDGET] 로그인 확인됨. 서버 데이터를 가져옵니다...");
            await syncData(); 
        } else {
            console.log("[WIDGET] 비로그인 상태. 로컬 LanceDB에서 메뉴를 불러옵니다...");
            await renderNavigation();
        }

    } catch (e) { 
        console.error("[WIDGET] Handshake failed:", e); 
    }
}

document.getElementById("btn-qr-auth")?.addEventListener("click", performQrAuth);

// 🌟 [수정] 로그아웃 시 Dexie DB의 세션까지 완벽히 제거합니다.
document.getElementById("btn-logout")?.addEventListener("click", async () => { 
    if (await ask("Are you sure you want to sign out?", { title: "Sign Out", kind: "warning" })) { 
        // 1. 메모리상의 세션 데이터 초기화
        currentSession = { hash: "", cc: "logis.center" };
        
        // 2. Dexie DB 내 저장된 세션 및 터미널 로그 영구 삭제
        await kvRemove("chat_session");
        
        // 3. sessionStorage 및 기타 상태 초기화
        sessionStorage.clear(); 
        
        // 4. 앱 강제 새로고침하여 초기 상태(새 해시 생성 등)로 복귀
        window.location.reload();
    } 
});

// 🌟 [추가] Dexie DB 초기화 및 앱 리셋 버튼 로직
document.getElementById("btn-reset-db")?.addEventListener("click", async () => {
    if (await ask("정말 로컬 데이터베이스를 초기화하시겠습니까?\n모든 로컬 큐 데이터와 캐시가 삭제되며 앱이 재시작됩니다.", { title: "Initialize Local DB", kind: "warning" })) {
        try {
            await appDb.delete(); // Dexie DB 완전 삭제
            sessionStorage.clear(); // 세션 스토리지 초기화
            window.location.reload(); // 앱 상태를 완전히 비우기 위해 강제 새로고침
        } catch (e) {
            console.error("DB Initialization failed:", e);
            alert("DB 초기화 중 오류가 발생했습니다.");
        }
    }
});

settingsBtn?.addEventListener("click", () => { if (currentTab === "settings" && isExpanded) collapseWidget(); else openWidget("settings"); });
document.getElementById("nav-to-auto")?.addEventListener("click", () => switchTab("automation"));
document.getElementById("unload-btn")?.addEventListener("click", async () => { 
    try { 
        // 🌟 메모리 강제 해제 시 진행 중인 프론트엔드 락도 함께 초기화합니다.
        await GlobalTaskManager.forceReset();
        isExtracting = false;
        isSearching = false;
        stopSpinner();

        await invoke("unload_model"); 
        alert("Memory cleared."); 
        
        // 버튼 상태 복구
        await updateExtractButtonVisibility();
        if (btnSubmit) btnSubmit.style.display = "flex";
    } catch (e) {
        console.error("[WIDGET] Unload failed:", e);
    } 
});

document.getElementById("invite-email-input")?.addEventListener("input", (e) => {
    const input = e.target as HTMLInputElement;
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    const btn = document.getElementById("btn-send-invite") as HTMLButtonElement;

    if (input.value.trim() === "") {
        input.style.outline = "none";
        if (btn) btn.disabled = false;
    } else if (!emailRegex.test(input.value.trim())) {
        input.style.outline = "1px solid #ef4444";
        // 형식이 맞지 않으면 전송 버튼을 비활성화하여 오전송 방지
        if (btn) btn.style.opacity = "0.5";
    } else {
        input.style.outline = "1px solid #4ade80";
        if (btn) {
            btn.disabled = false;
            btn.style.opacity = "1";
        }
    }
});

async function syncBrowserStatus() { 
    try { 
        // 🌟 문자열 대신 객체로 응답을 받아옵니다.
        const res = await invoke<any>("get_browser_status"); 
        const s = res.status;
        
        if (btnAutoLaunch) btnAutoLaunch.style.display = (s === "running") ? "none" : "flex"; 
        
        if (s === "stopped") { 
            currentDetectedUrl = ""; 
            await updateExtractButtonVisibility(); 
        } else if (s === "running") {
            // 🌟 [CRITICAL FIX] 휴면 상태에서 깨어났을 때 최신 URL 상태를 즉각 복구하여 추출 버튼을 노출시킵니다.
            if (res.url) {
                currentDetectedUrl = res.url;
                isCurrentShop = res.is_client || res.is_admin;
                await updateExtractButtonVisibility();
            }
        }
    } catch (e) {} 
}

// --- Device Preference Logic ---
const forceCpuToggle = document.getElementById("force-cpu-toggle") as HTMLInputElement;

// --- List Scroll & Pull Engine ---
let listCurrentY = 0;
let listPullY = 0;
let listPullTimer: number | null = null;
let listPushStartTime = 0;
let listPushDir: 'top' | 'bottom' | null = null;

function updateListTransform(resetting: boolean = false) {
    const scrollEl = document.getElementById("list-scroll");
    const container = document.getElementById("list-scroll-container");
    const topLoader = document.getElementById("list-pull-top");
    const bottomLoader = document.getElementById("list-pull-bottom");
    
    if (!scrollEl || !container || !topLoader || !bottomLoader) return;

    if (resetting) scrollEl.classList.add("resetting");
    else scrollEl.classList.remove("resetting");

    let effectiveOffset = listPullY;
    if (listPullY === 0 && listPushStartTime !== 0) {
        const pushElapsed = Date.now() - listPushStartTime;
        if (pushElapsed > 50) { 
            effectiveOffset = listPushDir === 'top' ? 50 : -50;
        }
    }

    scrollEl.style.transform = `translateY(${-listCurrentY + effectiveOffset}px)`;

    const loader = effectiveOffset !== 0 ? (effectiveOffset > 0 ? topLoader : bottomLoader) : null;
    
    if (loader) {
        loader.classList.add("visible");
        const absPull = Math.abs(effectiveOffset);
        loader.style.opacity = "1";
        
        if (absPull >= PULL_THRESHOLD) (loader as HTMLElement).classList.add("ready");
        else (loader as HTMLElement).classList.remove("ready");

        const spinner = (loader as HTMLElement).querySelector('.spinner') as HTMLElement;
        if (spinner) {
            const frameIndex = Math.floor(Date.now() / 80) % spinnerFrames.length;
            spinner.innerText = spinnerFrames[frameIndex];
        }
    } else {
        [topLoader, bottomLoader].forEach(el => {
            if (el) {
                el.classList.remove("visible", "ready");
                (el as HTMLElement).style.opacity = "0";
                const s = el.querySelector('.spinner') as HTMLElement;
                if (s && !el.classList.contains("loading")) s.innerText = "";
            }
        });
    }
}

function initListPullLogic() {
    const container = document.getElementById("list-scroll-container") as HTMLElement;
    const scrollEl = document.getElementById("list-scroll") as HTMLElement;
    const topLoader = document.getElementById("list-pull-top") as HTMLElement;
    const bottomLoader = document.getElementById("list-pull-bottom") as HTMLElement;
    
    if (!container || !scrollEl || !topLoader || !bottomLoader) return;

    let loopId: number | null = null;
    let lastTouchY = 0;

    const resetPull = () => {
        listPullY = 0;
        listPushStartTime = 0;
        listPushDir = null;
        updateListTransform(true);
        setTimeout(() => {
            scrollEl.classList.remove("resetting");
            topLoader.classList.remove("loading");
            bottomLoader.classList.remove("loading");
        }, 400);
    };

    const triggerAction = async (dir: 'top' | 'bottom') => {
        if (isLoading) return;
        const loader = dir === 'top' ? topLoader : bottomLoader;
        loader.classList.add("loading");
        
        listPullY = dir === 'top' ? 40 : -40;
        listPushStartTime = 0;
        updateListTransform(true);

        if (dir === 'top') {
            // [Top Pull] Sync Updates (opposite of chat)
            console.log("[List] Syncing latest updates...");
            await loadMoreDocs(false, true); 
        } else {
            // [Bottom Pull] Load More History (opposite of chat)
            console.log("[List] Loading more history...");
            await loadMoreDocs(false, false); 
        }

        resetPull();
    };

    const startAnimationLoop = () => {
        if (loopId) return;
        const tick = () => {
            const now = Date.now();
            if (listPushStartTime !== 0 && now - listPushStartTime >= 1000 && listPullY === 0) {
                const dir = listPushDir;
                if (dir) {
                    listPullY = dir === 'top' ? TRIGGER_THRESHOLD : -TRIGGER_THRESHOLD;
                    triggerAction(dir);
                }
            }
            updateListTransform();
            if (listPullY !== 0 || listPushStartTime !== 0 || isLoading) {
                loopId = requestAnimationFrame(tick);
            } else {
                loopId = null;
            }
        };
        loopId = requestAnimationFrame(tick);
    };

    const getMaxScroll = () => Math.max(0, scrollEl.scrollHeight - container.clientHeight);

    const handleDelta = (delta: number) => {
        // 🌟 Settings 패널 상태를 확인하여 열려있다면 모든 델타 계산을 중단합니다.
        const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
        if (currentTab !== "list" || (settingsToggle && settingsToggle.checked)) return;

        const maxScroll = getMaxScroll();
        const isAtTop = listCurrentY <= 0;
        const isAtBottom = listCurrentY >= maxScroll;

        if (!isLoading && (listPullY !== 0 || (isAtTop && delta < 0) || (isAtBottom && delta > 0))) {
            const currentDir = (isAtTop && delta < 0) ? 'top' : 'bottom';
            if (listPullY === 0) {
                if (listPushDir !== currentDir) {
                    listPushDir = currentDir;
                    listPushStartTime = Date.now();
                }
                startAnimationLoop(); 
                if (Date.now() - listPushStartTime < 1000) return; 
            }

            listPullY -= delta * FRICTION;
            if (listPullY > PULL_MAX) listPullY = PULL_MAX;
            if (listPullY < -PULL_MAX) listPullY = -PULL_MAX;
            
            if ((listPullY < 0 && listCurrentY <= 0) || (listPullY > 0 && listCurrentY >= maxScroll)) {
                resetPull();
            }
            startAnimationLoop();
        } 
        else {
            listPushDir = null;
            listPushStartTime = 0;
            listCurrentY += delta;
            if (listCurrentY < 0) listCurrentY = 0;
            else if (listCurrentY > maxScroll) listCurrentY = maxScroll;
        }
        updateListTransform();
    };

    container.addEventListener('wheel', (e) => {
        // 🌟 [CRITICAL CHECK] Settings 패널이 활성화되어 있는지 체크합니다.
        const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
        const isSettingsOpen = settingsToggle && settingsToggle.checked;

        // 리스트 탭이 아니거나 Settings 패널이 열려 있으면 리스트 전용 스크롤 로직을 완전히 차단합니다.
        if (currentTab !== "list" || isSettingsOpen) return;

        e.preventDefault();
        handleDelta(e.deltaY);
        if (listPullTimer) clearTimeout(listPullTimer);
        listPullTimer = window.setTimeout(() => {
            if (Math.abs(listPullY) >= PULL_THRESHOLD) triggerAction(listPullY > 0 ? 'top' : 'bottom');
            else if (listPushStartTime === 0 && !isLoading) resetPull();
        }, 200);
    }, { passive: false });

    container.addEventListener('touchstart', (e) => {
        lastTouchY = e.touches[0].pageY;
        scrollEl.classList.remove("resetting");
    }, { passive: true });

    container.addEventListener('touchmove', (e) => {
        // 🌟 [CRITICAL CHECK] Settings 패널이 활성화되어 있는지 체크합니다.
        const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
        const isSettingsOpen = settingsToggle && settingsToggle.checked;

        // Settings가 열려있다면 리스트의 Pull-to-refresh 로직이 간섭하지 못하게 합니다.
        if (currentTab !== "list" || isSettingsOpen) return;

        const currentTouchY = e.touches[0].pageY;
        handleDelta(lastTouchY - currentTouchY);
        lastTouchY = currentTouchY;
        e.preventDefault();
    }, { passive: false });

    container.addEventListener('touchend', () => {
        if (Math.abs(listPullY) >= PULL_THRESHOLD) triggerAction(listPullY > 0 ? 'top' : 'bottom');
        else if (listPushStartTime === 0) resetPull();
    });
}

async function initDevicePreference() {
    if (!forceCpuToggle) return;

    // 1. Check GPU Availability
    try {
        const hasGpu = await invoke<boolean>("check_gpu_availability");
        if (!hasGpu) {
            forceCpuToggle.disabled = true;
            forceCpuToggle.checked = true;
            const label = document.querySelector('label[for="force-cpu-toggle"]') as HTMLElement;
            if (label) label.innerText = "CPU Mode (No GPU detected)";
        } else {
            // 2. Load saved preference
            const savedPrefStr = await kvGet("force_cpu_mode");
            const savedPref = savedPrefStr === "true";
            forceCpuToggle.checked = savedPref;
        }
    } catch (e) {
        console.error("[WIDGET] Failed to check GPU status:", e);
    }

    // 3. Save on change
    forceCpuToggle.addEventListener("change", async () => {
        await kvSet("force_cpu_mode", forceCpuToggle.checked.toString());
        // [NOTE] The preference will be applied on the next model initialization.
        // Users can click "Free Memory" to force a reload if they want immediate effect.
    });
}

// --- Chat Virtual Scroll & Pull Engine ---
let currentY = 0; // Standard scroll position (positive)
let pullY = 0;    // Pull distance (positive for top, negative for bottom)
let pullTimer: number | null = null;
let pushStartTime = 0; // [NEW] Track hold time
let pushDir: 'top' | 'bottom' | null = null; 
const PULL_THRESHOLD = 50;
const PULL_MAX = 90;
const FRICTION = 0.3;
const TRIGGER_THRESHOLD = 50; 

function updateTransform(resetting: boolean = false) {
    const scrollEl = document.getElementById("chat-scroll");
    const container = document.querySelector(".chat-container") as HTMLElement;
    const topLoader = document.getElementById("chat-pull-top");
    const bottomLoader = document.getElementById("chat-pull-bottom");
    
    if (!scrollEl || !container || !topLoader || !bottomLoader) return;

    if (resetting) scrollEl.classList.add("resetting");
    else scrollEl.classList.remove("resetting");

    let effectiveOffset = pullY;
    if (pullY === 0 && pushStartTime !== 0) {
        const pushElapsed = Date.now() - pushStartTime;
        if (pushElapsed > 50) { 
            effectiveOffset = pushDir === 'top' ? 50 : -50; // Full 50px peek to show loader
        }
    }

    scrollEl.style.transform = `translateY(${-currentY + effectiveOffset}px)`;

    const loader = effectiveOffset !== 0 ? (effectiveOffset > 0 ? topLoader : bottomLoader) : null;
    
    if (loader) {
        loader.classList.add("visible");
        const absPull = Math.abs(effectiveOffset);
        loader.style.opacity = "1";
        
        if (absPull >= PULL_THRESHOLD) (loader as HTMLElement).classList.add("ready");
        else (loader as HTMLElement).classList.remove("ready");

        const spinner = (loader as HTMLElement).querySelector('.spinner') as HTMLElement;
        if (spinner) {
            const frameIndex = Math.floor(Date.now() / 80) % spinnerFrames.length;
            spinner.innerText = spinnerFrames[frameIndex];
        }
    } else {
        [topLoader, bottomLoader].forEach(el => {
            if (el) {
                el.classList.remove("visible", "ready");
                (el as HTMLElement).style.opacity = "0";
                const s = el.querySelector('.spinner') as HTMLElement;
                if (s && !el.classList.contains("loading")) s.innerText = "";
            }
        });
    }
}

function initChatPullLogic() {
    const container = document.querySelector(".chat-container") as HTMLElement;
    const scrollEl = document.getElementById("chat-scroll") as HTMLElement;
    const topLoader = document.getElementById("chat-pull-top") as HTMLElement;
    const bottomLoader = document.getElementById("chat-pull-bottom") as HTMLElement;
    
    if (!container || !scrollEl || !topLoader || !bottomLoader) return;

    let loopId: number | null = null;
    let lastTouchY = 0;

    const resetPull = () => {
        pullY = 0;
        pushStartTime = 0;
        pushDir = null;
        updateTransform(true);
        setTimeout(() => {
            scrollEl.classList.remove("resetting");
            topLoader.classList.remove("loading");
            bottomLoader.classList.remove("loading");
        }, 400);
    };

    const triggerAction = async (dir: 'top' | 'bottom') => {
        if (isChatLoading) return;
        const loader = dir === 'top' ? topLoader : bottomLoader;
        loader.classList.add("loading");
        
        pullY = dir === 'top' ? 40 : -40;
        pushStartTime = 0;
        updateTransform(true);

        if (dir === 'top') {
            // [Top Pull] Load Older History
            console.log("[Chat] Loading history (older than top)...");
            await loadMoreChat(true); 
        } else {
            // [Bottom Pull] Refresh/Load Latest Sync
            console.log("[Chat] Syncing latest/updated states...");
            await loadMoreChat(false); 
        }

        resetPull();
    };

    const startAnimationLoop = () => {
        if (loopId) return;
        const tick = () => {
            const now = Date.now();
            if (pushStartTime !== 0 && now - pushStartTime >= 1000 && pullY === 0) {
                const dir = pushDir;
                if (dir) {
                    pullY = dir === 'top' ? TRIGGER_THRESHOLD : -TRIGGER_THRESHOLD;
                    triggerAction(dir);
                }
            }
            updateTransform();
            if (pullY !== 0 || pushStartTime !== 0 || isChatLoading) {
                loopId = requestAnimationFrame(tick);
            } else {
                loopId = null;
            }
        };
        loopId = requestAnimationFrame(tick);
    };

    const getMaxScroll = () => Math.max(0, scrollEl.scrollHeight - container.clientHeight);

    const handleDelta = (delta: number) => {
        const maxScroll = getMaxScroll();
        const isAtTop = currentY <= 0;
        const isAtBottom = currentY >= maxScroll;

        if (!isChatLoading && (pullY !== 0 || (isAtTop && delta < 0) || (isAtBottom && delta > 0))) {
            const currentDir = (isAtTop && delta < 0) ? 'top' : 'bottom';
            if (pullY === 0) {
                if (pushDir !== currentDir) {
                    pushDir = currentDir;
                    pushStartTime = Date.now();
                }
                startAnimationLoop(); 
                if (Date.now() - pushStartTime < 1000) return; 
            }

            pullY -= delta * FRICTION;
            if (pullY > PULL_MAX) pullY = PULL_MAX;
            if (pullY < -PULL_MAX) pullY = -PULL_MAX;
            
            if ((pullY < 0 && currentY <= 0) || (pullY > 0 && currentY >= maxScroll)) {
                resetPull();
            }
            startAnimationLoop();
        } 
        else {
            pushDir = null;
            pushStartTime = 0;
            currentY += delta;
            if (currentY < 0) currentY = 0;
            else if (currentY > maxScroll) currentY = maxScroll;

            if (!isChatLoading && chatHasMore && currentY <= 50 && chatPage > 0) {
                loadMoreChat(false);
            }
        }
        updateTransform();
    };

    container.addEventListener('wheel', (e) => {
        e.preventDefault();
        handleDelta(e.deltaY);
        if (pullTimer) clearTimeout(pullTimer);
        pullTimer = window.setTimeout(() => {
            if (Math.abs(pullY) >= PULL_THRESHOLD) triggerAction(pullY > 0 ? 'top' : 'bottom');
            else if (pushStartTime === 0 && !isChatLoading) resetPull();
        }, 200);
    }, { passive: false });

    container.addEventListener('touchstart', (e) => {
        lastTouchY = e.touches[0].pageY;
        scrollEl.classList.remove("resetting");
    }, { passive: true });

    container.addEventListener('touchmove', (e) => {
        // 🌟 [CRITICAL CHECK] Settings 패널이 활성화되어 있는지 체크합니다.
        const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
        const isSettingsOpen = settingsToggle && settingsToggle.checked;

        // Settings가 열려있다면 리스트의 Pull-to-refresh 로직이 간섭하지 못하게 합니다.
        if (currentTab !== "list" || isSettingsOpen) return;

        const currentTouchY = e.touches[0].pageY;
        handleDelta(lastTouchY - currentTouchY);
        lastTouchY = currentTouchY;
        e.preventDefault();
    }, { passive: false });

    container.addEventListener('touchend', () => {
        if (Math.abs(pullY) >= PULL_THRESHOLD) triggerAction(pullY > 0 ? 'top' : 'bottom');
        else if (pushStartTime === 0) resetPull();
    });
}

// Call init functions
const getDevicePref = () => forceCpuToggle.checked ? "cpu" : null;
const talksScroll = document.getElementById("chat-scroll");
if (talksScroll) {
    initChatPullLogic();
}
const listScroll = document.getElementById("list-scroll");
if (listScroll) {
    initListPullLogic();
}
async function fetchChatHistory(reset: boolean = true, silent: boolean = false, shouldSnap: boolean = true) { 
    if (reset) { 
        chatPage = 0;
        chatHasMore = true;
        if (chatTalks) {
            chatTalks.innerHTML = "";
        }
    } 
    // Initial load is NOT history (isHistory = false)
    await loadMoreChat(false, silent); 
}

interface ChatMessage {
    id: string;
    role: string;
    text: string;
    updated_at: number;
    created_at: number;
    status: number;
    task_id?: string;
    content?: string | any;
}

async function upsertChatMessages(messages: ChatMessage[], mode: 'prepend' | 'append') {
    if (!chatTalks) return;

    const scrollEl = document.getElementById("chat-scroll");
    const prevScrollHeight = scrollEl ? scrollEl.scrollHeight : 0;

    for (const msg of messages) {
        let textContent = msg.text || "";
        const rawContent = msg.content || (msg as any).data;

        if (rawContent && rawContent !== "undefined") {
            try {
                const contentObj = typeof rawContent === 'string' ? JSON.parse(rawContent) : rawContent;
                textContent = contentObj.text || contentObj.title || contentObj.summary || textContent || (typeof contentObj === 'string' ? contentObj : JSON.stringify(contentObj));
            } catch (e) {
                if (!textContent) textContent = String(rawContent);
            }
        }

        const displayMsg: ChatMessage = { ...msg, text: textContent };
        const isTask = displayMsg.role === "system_task" || (displayMsg.role === "user" && !!displayMsg.task_id && displayMsg.task_id.startsWith("search_") && !displayMsg.id.endsWith("_query") && !displayMsg.task_id.endsWith("_query"));
        const domId = isTask ? (displayMsg.task_id || displayMsg.id) : displayMsg.id;
        
        const existingEl = chatTalks.querySelector(`[id="${domId}"]`) as HTMLElement;

        if (existingEl) {
            const cachedStatus = parseInt(existingEl.dataset.status || "0");
            
            // 🌟 [CRITICAL FIX 3] 한 번 진행 중(1)이 된 작업을 늦게 도착한 이벤트가 다시 대기(10)로 강등시키는 것을 원천 차단합니다!
            if ([1, 2, 6, 9].includes(cachedStatus) && msg.status === 10) {
                msg.status = cachedStatus; 
            }
            // 🌟 이미 종료 상태(2, 6, 9)인 메시지를 다시 진행(1)으로 되돌리는 것도 금지합니다.
            if ([2, 6, 9].includes(cachedStatus) && msg.status === 1) {
                msg.status = cachedStatus; 
            }

            const isTransitionFromVirtual = cachedStatus === 10 && displayMsg.status !== 10;
            const cachedUpdatedAt = parseInt(existingEl.dataset.updatedAt || "0");
            const cachedText = existingEl.querySelector('.content')?.textContent || "";

            // 🌟 [CRITICAL FIX 2] msg 대신 파싱이 완료된 displayMsg의 속성을 사용하여 안전하게 비교합니다.
            if (isTransitionFromVirtual || displayMsg.updated_at > cachedUpdatedAt || displayMsg.status !== cachedStatus || (displayMsg.text && cachedText !== displayMsg.text)) {
                
                // 1. 텍스트 내용 업데이트 (퍼센트 및 요약글)
                const contentEl = existingEl.querySelector('.content');
                // 🌟 [CRITICAL FIX 3] msg.text(undefined)가 아닌 displayMsg.text를 꽂아 넣어 빈칸 버그를 해결합니다!
                if (contentEl && contentEl.textContent !== displayMsg.text) {
                    contentEl.textContent = displayMsg.text;
                }

                // 2. 상태(Status) 및 아이콘 업데이트
                let finalStatus = displayMsg.status;
                
                // 🌟 [CRITICAL FIX] 좀비 방어: 현재 활성 작업(activeTaskId)이 아니더라도 큐가 돌리고 있는(currentTaskId) 정상 작업이면 STOPPED 처리를 면제합니다.
                if (finalStatus === 1 && !isSearching && !isExtracting && activeTaskId !== domId && GlobalTaskManager.currentTaskId !== domId) {
                    finalStatus = 2;
                }

                if (finalStatus !== cachedStatus) {
                    existingEl.dataset.status = finalStatus.toString();
                    
                    const currentLock = await kvGet("sys_lock");
                    if (currentLock === domId && [2, 6, 9].includes(finalStatus)) {
                        console.log(`[LOCK] Task ${domId} reached terminal state ${finalStatus}. Releasing lock.`);
                        await kvRemove("sys_lock");
                    }

                    const statusBar = existingEl.querySelector('.status-bar') as HTMLElement;
                    if (statusBar) {
                        const statusMap: any = {
                            1: { icon: "⠋", text: "PROCESSING", color: "#000" },
                            9: { icon: "✅", text: "DONE", color: "#22c55e" },
                            10: { icon: "📥", text: "QUEUED", color: "#999999" },
                            2: { icon: "❌", text: "STOPPED", color: "#ef4444" }, // 🌟 아이콘을 ❌로 변경하고 색상을 빨간색으로 고정
                            6: { icon: "❌", text: "ERROR", color: "#ef4444" }
                        };
                        // 🌟 finalStatus 변수를 참조하거나 msg.status를 직접 매핑에 사용하도록 보장합니다.
                        const s = statusMap[finalStatus] || statusMap[msg.status] || { icon: "⏳", text: "WAITING", color: "#999999" };
                        statusBar.style.color = s.color;
                        statusBar.innerHTML = `<span class="${(finalStatus === 1 || msg.status === 1) ? 'active-spinner' : ''}">${s.icon}</span> ${s.text}`;
                    }
                }
                existingEl.dataset.updatedAt = msg.updated_at.toString();
            }
        } else {
            const temp = document.createElement('div');
            temp.innerHTML = createMessageHTML(displayMsg);
            const newEl = temp.firstElementChild as HTMLElement;
            if (isTask) { newEl.onclick = () => handleTaskClick(newEl); }
            chatTalks.appendChild(newEl);
        }
    }

    // 🌟 [CRITICAL FIX] DOM 정렬 로직 강화 (시간 오름차순 및 질문 우선순위 고정)
    const sortedChildren = Array.from(chatTalks.children) as HTMLElement[];
    sortedChildren.sort((a, b) => {
        const timeA = Number(a.dataset.createdAt || 0);
        const timeB = Number(b.dataset.createdAt || 0);
        
        // 1. 시간이 다르면 시간순 정렬
        if (timeA !== timeB) {
            return timeA - timeB;
        }
        
        // 2. 시간이 동일할 경우, 질문(_query)이 작업 메시지보다 항상 앞에 오도록 배치
        const aId = a.id || "";
        const bId = b.id || "";
        const aIsQuery = aId.endsWith("_query") || aId.includes("_query");
        const bIsQuery = bId.endsWith("_query") || bId.includes("_query");
        
        if (aIsQuery && !bIsQuery) return -1;
        if (!aIsQuery && bIsQuery) return 1;
        
        // 3. 그 외에는 ID 문자열 순서로 고정 정렬
        return aId.localeCompare(bId);
    });

    // 🌟 [핵심 수정] 정렬된 리스트와 현재 DOM 순서를 비교하여 필요한 노드만 재배치
    sortedChildren.forEach((node, idx) => {
        if (chatTalks.children[idx] !== node) {
            chatTalks.insertBefore(node, chatTalks.children[idx] || null);
        }
    });

    // [Scroll Maintenance]
    if (mode === 'prepend' && scrollEl) {
        const newScrollHeight = scrollEl.scrollHeight;
        const heightDiff = newScrollHeight - prevScrollHeight;
        if (heightDiff > 0) {
            currentY += heightDiff;
            updateTransform();
        }
    } else if (mode === 'append' && scrollEl) {
        const container = document.querySelector(".chat-container") as HTMLElement;
        const maxScroll = Math.max(0, scrollEl.scrollHeight - (container?.clientHeight || 0));
        
        if (prevScrollHeight === 0 || (currentY >= prevScrollHeight - (container?.clientHeight || 0) - 50)) {
            currentY = maxScroll;
            updateTransform();
        }
    }
}

function createMessageHTML(msg: ChatMessage) {
    // 🌟 상태 2번에 대한 정의를 명시적으로 추가하여 'WAITING'으로 빠지는 것을 방지합니다.
    const statusMap: Record<number, { icon: string, text: string, color: string }> = {
        9: { icon: "✅", text: "DONE", color: "#22c55e" },
        0: { icon: "✅", text: "DONE", color: "#22c55e" },
        1: { icon: "⠋", text: "PROCESSING", color: "#000" },
        6: { icon: "❌", text: "ERROR", color: "#ef4444" },
        2: { icon: "❌", text: "STOPPED", color: "#ef4444" }, // 🌟 좀비 테스크(2)를 ERROR 아이콘과 색상으로 지정
        10: { icon: "📥", text: "PENDING", color: "#999999" },
        3: { icon: "🛑", text: "STOPPED", color: "#ef4444" }
    };
    
    const currentStatus = statusMap[msg.status] || { icon: "⏳", text: "WAITING", color: "#999999" };
    
    // Task Bubble 판단 로직 (ID와 Role 기준)
    const isTaskBubble = msg.role === "system_task" || (!!msg.task_id && msg.task_id.startsWith("search_") && !msg.id.endsWith("_query"));
    const roleClass = msg.role === "user" ? "user" : "system";
    const domId = isTaskBubble ? (msg.task_id || msg.id) : msg.id;
    
    const timeStr = new Date(Number(msg.created_at)).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const bubbleClass = isTaskBubble ? 'task-bubble' : '';

    // 🌟 핵심: msg.text가 비어있지 않도록 보장하여 새로고침 시에도 내용 표시
    const displayContent = msg.text && msg.text.trim() !== "" ? msg.text : "대기 중인 작업입니다...";

    return `<div id="${domId}" class="chat-talk ${roleClass} ${bubbleClass}" 
        data-task-id="${msg.task_id || msg.id}" 
        data-status="${msg.status}" 
        data-updated-at="${msg.updated_at}"
        data-created-at="${msg.created_at}"
        style="${isTaskBubble ? 'cursor:pointer;' : ''}">
        <div class="chat-message">
            <div style="font-size:0.8rem; opacity:0.5; margin-bottom:4px; display:flex; justify-content:space-between;">
                <span>${msg.role === 'user' ? '@YOU' : 'LOGIS AI'}</span>
                <span>${timeStr}</span>
            </div>
            <div class="content">${displayContent}</div>
            ${isTaskBubble && msg.status !== 0 ? `
                <div class="status-bar" style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.1); font-size: 0.65rem; font-weight: bold; color: ${currentStatus.color};">
                    <span class="${msg.status === 1 ? 'active-spinner' : ''}">${currentStatus.icon}</span> ${currentStatus.text}
                </div>` : ""}
        </div>
    </div>`;
}

async function loadMoreChat(isHistory: boolean = false, silent: boolean = false) {
    if (isChatLoading || (isHistory && !chatHasMore)) {
        if (!silent) stopSpinner();
        return;
    }

    if (!silent) startSpinner();
    isChatLoading = true;

    try {
        let baseFilter = "";
        if (activeContext.ref) baseFilter = `ref = '${activeContext.ref}'`;
        else if (activeContext.bcc) baseFilter = `bcc = '${activeContext.bcc}'`;
        else if (activeContext.cc) baseFilter = `cc = '${activeContext.cc}'`;
        
        let finalFilter = baseFilter;
        let oldestTime = 0;
        let latestUpdateTime = 0;

        const allMsgs = chatTalks.querySelectorAll('.chat-talk');
        allMsgs.forEach(el => {
            const up = parseInt((el as HTMLElement).dataset.updatedAt || "0");
            if (up > latestUpdateTime) latestUpdateTime = up;
        });

        if (isHistory) {
            const firstMsg = chatTalks.querySelector('.chat-talk:not(.chat-history-end)');
            if (firstMsg) {
                oldestTime = parseInt((firstMsg as HTMLElement).dataset.createdAt || "0");
            }
            
            if (oldestTime > 0) {
                let timeFilter = `created_at < ${oldestTime}`;
                if (latestUpdateTime > 0) {
                    timeFilter = `(${timeFilter}) OR (updated_at > ${latestUpdateTime})`;
                }
                finalFilter = baseFilter ? `${baseFilter} AND (${timeFilter})` : timeFilter;
            }
        } else if (latestUpdateTime > 0) {
            const syncFilter = `updated_at > ${latestUpdateTime}`;
            finalFilter = baseFilter ? `${baseFilter} AND ${syncFilter}` : syncFilter;
        }

        const limit = 10; 
        const offset = 0;

        let messages = await invoke<any[]>("get_chat_messages", { limit: limit, offset: offset, filter: finalFilter });
        
        // 🌟 [추가] 좀비 상태 보정 및 정렬 안정화 데이터 세탁
        messages = messages.map(m => {
            // 1. 유령 데이터 STOPPED 처리 강화: 
            // 현재 앱이 '초기화(Handshake) 완료 전'이거나, 백엔드 DB에서도 명시적으로 2인 경우만 중단 처리합니다.
            if ((m.status === 1 || m.status === 10) && !isSearching && !isExtracting) {
                // 단순히 검색/추출 중이 아니라고 해서 2로 바꾸지 않고, 
                // DB에서 넘어온 원본 status가 이미 2이거나 terminal state일 때만 UI를 고정합니다.
                return m; 
            }
            // 2. 사용자 질문의 경우 시스템 메시지와의 정렬 간격을 벌리기 위해 시간값 강제 보정
            if (m.role === "user" && m.id.endsWith("_query")) {
                return { ...m, created_at: Number(m.created_at) - 50 };
            }
            return m;
        });

        // 🌟 [CRITICAL FIX 2] 변수 스코프(Scope) 에러 해결! 
        let activeMemContext: any = null;
        try {
            activeMemContext = await invoke<any>("get_active_task_context");
        } catch (e) {}

        try {
            // 1. Rust 백엔드 DB에 저장된 활성 태스크 가져오기
            const activeTasks = await invoke<any[]>("get_active_tasks");
            
            // 🌟 2. [수정] 프론트엔드 큐 작업 병합 시, DB에서 이미 종료/중단된 ID는 제외합니다.
            const queuedTasks = GlobalTaskManager.queue.map(q => ({
                id: q.taskId,
                task_id: q.taskId,
                status: 10, // Pending
                created_at: parseInt(q.taskId.split('_')[1]) || Date.now(),
                data_json: q.payload,
                ref: q.payload.link || q.payload.image_path || "Queued Task"
            }));

            // 🌟 [핵심 로직] DB(activeTasks)에 있는 녀석이 10번이 아니라면(이미 2번 등으로 변했다면) 큐에서 부활시키지 않습니다.
            const combinedTasks = [...activeTasks];
            queuedTasks.forEach(qt => {
                const dbEquivalent = activeTasks.find(t => t.id === qt.id);
                // DB에 아예 없거나, DB에서도 여전히 Pending(10)인 경우에만 큐 정보를 신뢰합니다.
                if (!dbEquivalent) {
                    combinedTasks.push(qt);
                }
            });

            combinedTasks.forEach((t: any) => {
                let taskQuery = "";
                try {
                    const taskData = typeof t.data_json === 'string' ? JSON.parse(t.data_json) : t.data_json;
                    taskQuery = taskData.query || "";
                } catch(e) {}

                // 🌟 [CRITICAL FIX] 새로고침 시 질문 복구 로직 강화
                if (taskQuery) {
                    const userMsgId = `${t.id}_query`;
                    const userExistsInBatch = messages.some(m => m.id === userMsgId);
                    const userExistsInDom = document.getElementById(userMsgId);
                    
                    if (!userExistsInBatch && !userExistsInDom) {
                        messages.push({
                            id: userMsgId,
                            task_id: t.id,
                            role: "user",
                            text: taskQuery,
                            status: 9, 
                            // 🌟 initSession과 동일하게 100ms 시간차를 주어 정렬 순서를 물리적으로 강제합니다.
                            created_at: Number(t.created_at) - 100, 
                            updated_at: Number(t.created_at) - 100
                        });
                        console.log(`[RECOVERY] Restored missing user query for task: ${t.id}`);
                    }
                }

                const exists = messages.find(m => m.id === t.id || m.task_id === t.id);
                if (!exists) {
                    messages.push({
                        id: t.id,
                        task_id: t.id,
                        role: "system_task",
                        // 🌟 [UI 보강] DB에 아직 안 들어간 순수 대기열(status: 10) 상태임을 직관적으로 보여줍니다.
                        text: t.id.startsWith("search_") ? "Waiting in Queue: AI Search" : ("Waiting in Queue: " + (t.ref || "Local Source")),
                        status: t.status,
                        created_at: t.created_at + 1,
                        updated_at: t.updated_at + 1
                    });
                }
            });
        } catch (e) { }

        for (let m of messages) {
            if (m.status === 1 && (m.role === "system_task" || m.task_id)) {
                try {
                    const tId = m.task_id || m.id;
                    const logs = await invoke<any[]>("get_task_logs", { taskId: tId });
                    
                    let lastLog = null;
                    if (logs && logs.length > 0) {
                        lastLog = logs[logs.length - 1];
                    }
                    
                    let rawSummary = "Processing...";
                    const live = livePayloads.get(tId);
                    
                    if (live && live.summary) {
                        rawSummary = live.summary;
                    } else if (lastLog && lastLog.summary) {
                        rawSummary = lastLog.summary;
                    } else if (activeMemContext && activeMemContext.id === tId && activeMemContext.summary) {
                        rawSummary = activeMemContext.summary;
                    }

                    const pctMatch = rawSummary.match(/\(\d+%\)/);
                    const hasDots = rawSummary.endsWith("...");
                    if (hasDots) rawSummary = rawSummary.slice(0, -3).trim();
                    if (pctMatch) rawSummary = rawSummary.replace(pctMatch[0], '').trim();
                    
                    let fractionStr = "";
                    const targetCat = (live && live.category) ? live.category : (lastLog && lastLog.category ? lastLog.category : "");

                    // 🌟 [UI 심플화] 채팅방 히스토리에도 오직 List Extraction 단계에서만 [N/M]을 보여줍니다.
                    if (targetCat.includes("List Extraction")) {
                        const match = targetCat.match(/\((\d+)\/(\d+)\)/);
                        if (match) {
                            fractionStr = ` [${match[1]}/${match[2]}]`;
                        }
                    }
                    
                    m.text = `${rawSummary}${fractionStr}${pctMatch ? ' ' + pctMatch[0] : ''}${hasDots ? '...' : ''}`;
                    m.updated_at = Date.now();
                    
                } catch (e) {}
            }
        }

        const scrollEl = document.getElementById("chat-scroll") as HTMLElement;

        if (chatTalks) {
            if (messages && messages.length > 0) {
                const mode = isHistory ? 'prepend' : 'append';
                upsertChatMessages(messages, mode);
                if (isHistory && messages.length < limit) chatHasMore = false;
            } else { 
                if (isHistory) chatHasMore = false;
                // 🌟 [보강] 이미 no-msg 엘리먼트가 존재한다면 추가하지 않도록 방어합니다.
                const hasNoMsgEl = chatTalks.querySelector('.no-msg');
                if (!isHistory && chatTalks.querySelectorAll('.chat-talk').length === 0 && !hasNoMsgEl) {
                    chatTalks.insertAdjacentHTML('beforeend', "<div class='no-msg' data-created-at=\"0\" style='text-align:center; padding:20px; color:#999; font-size:0.8rem;'>No messages yet.</div>");
                }
            }

            if (isHistory && !chatHasMore && !chatTalks.querySelector('.chat-history-end')) {
                const endHtml = `<div class="chat-talk system chat-history-end" data-created-at="0" style="text-align:center; opacity:0.4; font-size:0.8rem; padding:15px 10px;">
                    <div style="border-top:1px solid rgba(255,255,255,0.05); margin-bottom:10px;"></div>
                    <span>No more older messages</span>
                </div>`;
                chatTalks.insertAdjacentHTML('afterbegin', endHtml);
            }

            if (!currentSession.email && currentTab === "settings") {
                performQrAuth();
            }
        }
    } catch (e) { 
        console.error(e); 
    } finally { 
        isChatLoading = false; 
        if (!silent) stopSpinner();
    }
}

async function renderMessage(msg: any, shouldScroll: boolean = true, isPrepend: boolean = false) {
    if (!chatTalks) return;
    // Single message upsert (Real-time is always append/newest in Slack style)
    await upsertChatMessages([msg], isPrepend ? 'prepend' : 'append');
}

// --- Initialize ---
initSession();
setWindowSize(false);
syncBrowserStatus();
initDevicePreference();

function stopDesktopCamera() {
    if (desktopStream) {
        desktopStream.getTracks().forEach(track => track.stop());
        desktopStream = null;
    }
}

async function startMobileScanning(video: HTMLVideoElement) {
    if (!video || !(video instanceof HTMLVideoElement)) {
        console.error('Invalid video element provided to startMobileScanning');
        return;
    }

    try {
        console.log("Starting desktop camera stream...");
        desktopStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        video.srcObject = desktopStream;
        await video.play();
        
        document.getElementById("mobile-scan-view")?.classList.remove("hidden");
        document.getElementById("pc-qr-view")?.classList.add("hidden");
    } catch (err) {
        console.error("Failed to start desktop camera:", err);
        alert("Camera start failed: " + err);
        return;
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    const receivedChunks: string[] = [];
    let expectedTotal = 0;
    
    const scanLoop = async () => {
        if (!video || video.paused || video.ended) return;
        try {
            if (video.readyState >= 2) {
                canvas.width = video.videoWidth; canvas.height = video.videoHeight;
                if (ctx && canvas.width > 0 && canvas.height > 0) {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    // @ts-ignore
                    const code = jsQR(imageData.data, imageData.width, imageData.height);
                    if (code) {
                        try {
                            const data = JSON.parse(code.data);
                            // Handle compact Answer
                            if (data.t === "answer") {
                                const sdp = buildSdp('answer', data.i, data.u, data.p, data.f, data.s);
                                const answer = new RTCSessionDescription({ type: 'answer', sdp });
                                if (peerConn) {
                                    await peerConn.setRemoteDescription(answer);
                                    stopDesktopCamera();
                                    const profileName = document.getElementById("nav-profile-name");
                                    if (profileName) {
                                        profileName.textContent = "✅ Mobile Connected";
                                        profileName.style.color = "#4ade80";
                                    }
                                    document.getElementById("nav-qr-container")?.classList.add("hidden");
                                }
                                return;
                            }
                            // Fallback for legacy chunked format
                            if (Array.isArray(data) && data.length === 3) {
                                const [idx, total, chunkStr] = data;
                                if (expectedTotal === 0) {
                                    expectedTotal = total;
                                    for(let i=0; i<total; i++) receivedChunks.push(""); 
                                }
                                if (!receivedChunks[idx]) {
                                    receivedChunks[idx] = chunkStr;
                                    const profileName = document.getElementById("nav-profile-name");
                                    if (profileName) {
                                        const count = receivedChunks.filter(c => c).length;
                                        profileName.textContent = `Scanning... ${count}/${total}`;
                                    }
                                }
                                if (receivedChunks.every(c => c !== "")) {
                                    const answer = new RTCSessionDescription({ type: 'answer', sdp: receivedChunks.join("") });
                                    if (peerConn) {
                                        await peerConn.setRemoteDescription(answer);
                                        stopDesktopCamera();
                                        const profileName = document.getElementById("nav-profile-name");
                                        if (profileName) {
                                            profileName.textContent = "✅ Mobile Connected";
                                            profileName.style.color = "#4ade80";
                                        }
                                        document.getElementById("nav-qr-container")?.classList.add("hidden");
                                    }
                                    return;
                                }
                            }
                        } catch (e) {}
                    }
                }
            }
        } catch (e) {}
        requestAnimationFrame(scanLoop);
    };
    requestAnimationFrame(scanLoop);
}

document.getElementById("btn-switch-to-camera")?.addEventListener("click", () => {
    const video = document.getElementById("desktop-camera-video") as HTMLVideoElement;
    if (video) startMobileScanning(video);
});
document.getElementById("btn-switch-to-qr")?.addEventListener("click", () => {
    const video = document.getElementById("desktop-camera-video") as HTMLVideoElement;
    if (video) stopDesktopCamera();
    showPcPairingQr();
});