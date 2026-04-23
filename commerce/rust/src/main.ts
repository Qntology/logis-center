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
let isExtracting = false; 
let isSearching = false; // 🌟 [CRITICAL FIX] 검색 중복 방지 및 스피너 보호용 락(Lock)
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
    
    if (isChecked) {
        // 설정 켜짐: 설정 패널 표시, 리스트 및 네비게이션 숨김
        if (settingsPanel) settingsPanel.style.display = "block";
        if (docList) docList.style.display = "none";
        navSections.forEach(el => (el as HTMLElement).style.display = "none");
        
        // 라벨 UI 강조 효과 (선택사항)
        if (label) {
            label.classList.add("on")
        }
    } else {
        // 설정 꺼짐: 설정 패널 숨김, 리스트 및 네비게이션 원상복구
        if (settingsPanel) settingsPanel.style.display = "none";
        if (docList) docList.style.display = ""; 
        navSections.forEach(el => (el as HTMLElement).style.display = "");
        
        // 라벨 UI 원상복구
        if (label) {
            label.classList.remove("on");
        }

        // 🌟 [CRITICAL FIX] 패널을 닫고 UI가 복구될 때, 
        // 현재 탭이 Shipping 이라면 Shared Pages가 다시 보이지 않도록 즉시 재적용!
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

    // 🌟 [CRITICAL FIX] 이미지가 올라와 있을 때는 검색(🔍) 버튼을 숨기고 번개(⚡) 버튼을 복구하도록 예외 처리를 추가합니다!
    if (currentImage) {
        if (btnSubmit) btnSubmit.style.display = "none";
    } else {
        if (btnSubmit) btnSubmit.style.display = "flex";
    }

    updateExtractButtonVisibility(); // 숨겨졌던 번개 버튼 상태를 안전하게 다시 계산해서 화면에 띄웁니다.
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
        fetchChatHistory();
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
    // 🌟 락이 걸려있을 때는 백그라운드 폴링이나 다른 함수가 함부로 버튼을 부활시키지 못하게 방어!
    if (!btnExtract || extractClickLock) return; 

    // 1. 이미지가 올라와 있을 때 중복 대기열 방어 로직
    if (currentImage) {
        try {
            const ccHash = activeContext.cc || "";
            const imageRefHash = await hashId(currentImage); 

            const isImageActive = await invoke<boolean>("check_active_task", {
                payload: { cc: ccHash, ref: imageRefHash }
            });

            if (isImageActive) {
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

    // 2. 분석 가능한 샵 도메인이나 허용된 페이지가 아니면 숨김
    if (!currentDetectedUrl || !isCurrentShop) {
        btnExtract.style.display = "none";
        return;
    }

    try {
        const urlObj = new URL(currentDetectedUrl.toLowerCase());
        const hostname = urlObj.hostname;
        const link = (urlObj.pathname + urlObj.search).toLowerCase();
        const ccHash = await hashId(hostname);
        const hashedRefId = await hashId(ccHash + link);

        btnExtract.title = `Extract from ${hostname}`;

        // 3. 백엔드에 현재 페이지 작업 상태 질의
        const isActive = await invoke<boolean>("check_active_task", {
            payload: { cc: ccHash, ref: hashedRefId }
        });

        if (isActive) {
            btnExtract.style.display = "none";
        } else {
            btnExtract.style.display = "flex";
            btnExtract.innerHTML = "⚡";
            btnExtract.classList.remove("active-spinner");
        }
    } catch (e) {
        console.warn("[WIDGET] visibility check error:", e);
        btnExtract.style.display = "flex";
        btnExtract.innerHTML = "⚡";
        btnExtract.classList.remove("active-spinner");
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
});

const handleSearchInteraction = () => {
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
                    if (node.from === currentSession.address) desc.push("owner");
                    content = `<span>${name}${desc.length ? `<i>${desc.toString()}</i>` : ''}</span>`;
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
                        host = `<strong>${domain}</strong>`;
                        navTmp[domain] = true;
                    }
                    if (nodeId === activeContext.ref || (currentDetectedUrl && currentDetectedUrl.includes(data.link))) {
                        active = "active";
                    }
                }
                var total = { draft: 0, count: 0 };
                const pagesStats = (currentSession as any).pages;
                const cc = node.cc || data.cc;
                if (pagesStats && cc && pagesStats[cc] && pagesStats[cc][nodeType]) {
                    total = pagesStats[cc][nodeType];
                }

                var recent = '';
                try {
                    const bcc = node.bcc || data.bcc;
                    if (bcc) {
                        const _items = await Select['items']({ key: 'bcc', value: bcc, limit: 1 });
                        if (_items.length) {
                            recent = `<strong>${time2text(_items[0].created_at)}</strong>`;
                        }
                    }
                } catch (err) {}

                var count = data.item ? `<u>(${total.draft})</u>` : `<u>(${total.count})</u>`;
                content = `<span>${name} ${count}</span> ${recent}`;
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
                           data-type="${node.type || (node.data && node.data.type) || ''}">
                        ${content}
                    </label>
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
        
        if (_pages.length === 0) {
            pageList.innerHTML = "<div>No shared pages found.</div>";
        } else {
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

            // 3. Render
            pageList.innerHTML = await renderAccordion(tree);

            // 4. Bind Clicks manually to labels
            pageList.querySelectorAll(".logis-label").forEach((label: any) => {
                label.onclick = (e: Event) => {
                    const ds = label.dataset;
                    if (!ds.id) return;

                    activeContext.cc = ds.cc || "";
                    activeContext.bcc = ds.bcc || "";
                    activeContext.ref = ds.ref || "";

                    addSearchTag(`@${ds.domain}`, 'domain', ds.domain);
                    addSearchTag(`#${ds.type}`, 'type', ds.type);
                    
                    fetchChatHistory(true);
                    hideNavigation();
                };
            });
        }

        // Users rendering (simplified parity)
        const localUserList = document.getElementById("nav-list-local-users");
        const users = await Select["users"]({});
        
        if (userList) userList.innerHTML = "";
        if (localUserList) localUserList.innerHTML = "<div style='padding-left:1em; font-size:0.75rem; color:#666;'>No local devices</div>";

        if (users.length > 0) {
            // 1. 꼬리표를 기준으로 로컬/클라우드 유저 분할
            const localUsers = users.filter(u => u.data && u.data.is_device === true);
            const cloudUsers = users.filter(u => !u.data || u.data.is_device !== true);

            // 2. Cloud Team Members 렌더링 (기존 트리 구조 유지)
            if (cloudUsers.length > 0 && userList) {
                const teamNodes = cloudUsers.filter(u => u.type === "team").map(u => ({
                    ...u, 
                    children: cloudUsers.filter(m => m.to === u.id && m.id !== u.id)
                }));
                userList.innerHTML = await renderAccordion(teamNodes);
            }

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

// --- Sync Logic ---
// main.ts 내부
async function syncData() {
    if (!currentSession.hash || !currentSession.email) return;
    
    console.log("[SYNC] 1. 서버에 최신 데이터 요청 중...");
    try {
        const origin = "https://commerce.logis.center";
        const now = Date.now();
        const createdAt = now - timezoneOffset;
        
        const params = new URLSearchParams({
            origin: origin,
            created_at: createdAt.toString(),
            hash: currentSession.hash,
            token: currentSession.token || "",
            href: window.location.href
        });
        
        const url = `${API_HOST}/?${params.toString()}`;
        
        // 1. 서버 요청
        const response = await invoke<any>("proxy_fetch", {
            url: url,
            method: "GET",
            headers: { "Content-Type": "application/json" },
            session_params: { hash: currentSession.hash, token: currentSession.token }
        });

        stepQrSpinner();

        if (response.results && Array.isArray(response.results)) {
            console.log(`[SYNC] 2. 로컬 LanceDB 최신화 중... (데이터 ${response.results.length}건)`);
            // 2. LanceDB 최신화 (Rust의 upsert_items 호출)
            await invoke("upsert_items", { items: response.results });
            
            console.log("[SYNC] 3. 로컬 DB에서 데이터 불러와 메뉴 렌더링...");
            // 3. LanceDB 불러오기 (내부적으로 Select["pages"], Select["users"]를 통해 로컬 DB를 읽음)
            await renderNavigation();
            
            // 리스트 뷰도 최신화
            if (currentTab === "list") {
                await loadMoreDocs(false, true); 
            }
        }
        
    } catch (e) { 
        console.error("[SYNC] 동기화 실패:", e); 
    } finally {
        if (!isExtracting) stopSpinner();
    }
}

// --- 기존 State 영역 어딘가에 추가 ---
let currentSearchMode = localStorage.getItem("search_mode") || "commerce";

// 🌟 앱 시작 시 탭 UI 초기화 함수
function applySearchModeUI() {
    document.querySelectorAll('.mode-tab').forEach(btn => {
        const el = btn as HTMLElement;
        if (el.dataset.mode === currentSearchMode) {
            el.style.color = "var(--primary)";
            el.style.fontWeight = "bold";
            el.classList.add('active');
        } else {
            el.style.color = "#666";
            el.style.fontWeight = "normal";
            el.classList.remove('active');
        }
    });

    // 🌟 [추가] 선택된 모드의 첫 글자를 대문자로 변환하여 Placeholder에 즉시 반영!
    if (searchInput) {
        const capitalizedMode = currentSearchMode.charAt(0).toUpperCase() + currentSearchMode.slice(1);
        searchInput.placeholder = `${capitalizedMode} Search or Ask`;
    }

    // 🌟 [추가] Shipping 모드일 때 Shared Pages 섹션 통째로 숨기기
    const pagesSection = document.getElementById("nav-list-pages")?.closest(".nav-section") as HTMLElement;
    if (pagesSection) {
        if (currentSearchMode === "shipping") {
            pagesSection.style.display = "none"; // Shipping이면 숨김
        } else {
            pagesSection.style.display = "";     // 그 외(Commerce, Analytic)면 복구
        }
    }
}

// DOM 로드 후 이벤트 리스너 추가
document.querySelectorAll('.mode-tab').forEach(btn => {
    btn.addEventListener('click', async (e) => {
        const target = e.target as HTMLElement;
        currentSearchMode = target.dataset.mode || "commerce";
        
        // 🌟 탭 클릭 시 상태 저장 및 UI 업데이트
        localStorage.setItem("search_mode", currentSearchMode);
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

// --- List Logic (Updated for Cards) ---
searchInput?.addEventListener("input", () => {
    if (isSearching) return; // 🌟 [CRITICAL FIX] 전처리(isExtracting) 중이어도 자동완성 벡터 검색은 즉각 실행되도록 허용!
    if(searchDebounceTimer) clearTimeout(searchDebounceTimer);
    searchDebounceTimer = window.setTimeout(async () => {
        if (isSearching) return; // 🌟 [CRITICAL FIX] 
        await loadMoreDocs(true);
    }, 800);
});

// [신규] 검색창에서 엔터 키를 누르면 AI 검색(돋보기 버튼)을 강제로 실행하도록 연결
searchInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault(); 
        if (!isSearching && !isExtracting) { 
            btnSubmit?.click(); 
        }
    }
});

btnSubmit?.addEventListener("click", async () => {
    // 🌟 [CRITICAL FIX] 이미 검색 중이거나 추출 중이면 무조건 무시!
    if (isSearching) return

    // 🌟 [CRITICAL FIX] 예약되어 있던 라이브 텍스트 검색 타이머를 박살 내어 GPU 충돌을 원천 차단합니다!
    if (searchDebounceTimer) {
        clearTimeout(searchDebounceTimer);
        searchDebounceTimer = null;
    }

    const query = searchInput.value;
    if (!query) return;

    isSearching = true; // 🌟 락 온!
    if (btnSubmit) btnSubmit.style.display = "none";

    const taskId = `search_${Date.now()}`;
    const startTime = Date.now();
    
    openWidget("settings");
    startSpinner();

    renderMessage({
        id: taskId,
        role: "user", 
        text: query,
        status: 10, // 🌟 [CRITICAL FIX] UI에 즉시 📥 PENDING 상태로 띄웁니다!
        created_at: startTime,
        updated_at: startTime,
        task_id: taskId
    });

    try {
        const devicePref = forceCpuToggle.checked ? "cpu" : null;
        const response = await invoke<any>("ai_search_complex", { 
            taskId: taskId, 
            query: query, 
            language: "korean",
            devicePreference: devicePref,
            searchMode: currentSearchMode,
            // 🌟 [CRITICAL FIX] 히스토리 증발 방지: 현재 사용자가 보고 있는 위치(Context)를 백엔드에 전달합니다!
            cc: activeContext.cc || "",
            bcc: activeContext.bcc || "",
            refId: activeContext.ref || ""
        });

        renderMessage({
            id: taskId,
            role: "user",
            text: query, 
            status: 9, 
            created_at: startTime,
            updated_at: Date.now(),
            task_id: taskId
        });

        if (aiResultsArea && aiResultsContent) {
            aiResultsArea.style.display = "block";
            aiResultsTitle.innerText = "🧠 AI Deep Analysis";
            
            let html = `<div style="margin-bottom:15px; padding:10px; background:#222; border-left:3px solid var(--primary); font-size:0.75rem;"><strong style="display:block; margin-bottom:5px; color:#aaa;">Query Intent:</strong>`;
            if (response.structured && response.structured.context) {
                response.structured.context.forEach((ctx: any) => {
                    html += `<div style="margin-bottom:5px;">• ${ctx.text} <span style="color:var(--primary)">[${ctx.type}]</span></div>`;
                });
            }
            html += `</div>`;
            if(response.results.length === 0) {
                html += "No matching data found.";
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
        }
    } catch(e) { 
        if (aiResultsContent) {
            aiResultsContent.innerHTML = "<div style='color:#ef4444;'>Error: " + e + "</div>"; 
        }
        renderMessage({
            id: taskId,
            role: "system_task",
            text: `AI Search Error: ${query}`,
            status: 6, 
            created_at: startTime,
            updated_at: Date.now(),
            task_id: taskId
        });
    } finally {
        isSearching = false; // 🌟 락 오프!
        if (btnSubmit) btnSubmit.style.display = "flex";
        stopSpinner(); 
    }
});

document.addEventListener('show-doc', (e: any) => showDetail(e.detail));
document.addEventListener('view-task-log', () => { openWidget("list"); listView.style.display = "none"; detailView.style.display = "flex"; });


// 🌟 [CRITICAL FIX] 추출 버튼 더블클릭 완벽 방어 로직 적용
btnExtract?.addEventListener("click", async () => {
    // 🌟 이미 락이 걸려있다면 0.001초만에 들어온 광클도 무조건 튕겨냅니다!
    if (extractClickLock) return; 
    extractClickLock = true;
    
    // 🌟 클릭 즉시 버튼을 시각적, 물리적으로 증발시킵니다.
    btnExtract.style.display = "none"; 

    console.log("[DEBUG] btnExtract clicked. currentDetectedUrl:", currentDetectedUrl, "currentImage:", currentImage);
    
    try {
        if (currentDetectedUrl || currentImage) {
            const wasExtracting = isExtracting;
            isExtracting = true;

            if (!wasExtracting) {
                const logArea = document.getElementById("extraction-log");
                if (logArea) logArea.innerHTML = "";
                openWidget("settings");
                startSpinner();
            }
            const taskId = `task_${Date.now()}`;

            renderMessage({
                id: taskId,
                role: "system_task",
                text: currentImage ? "Task Started: Local Image" : "Task Started: " + (currentDetectedUrl || "Unknown URL"),
                status: 10, 
                created_at: Date.now(),
                updated_at: Date.now(),
                task_id: taskId
            });
            
            const isCloudMode = (document.getElementById("cloud-mode-toggle") as HTMLInputElement)?.checked;

            if (isCloudMode && currentSession.hash) {
                // ==========================================
                // ☁️ [SERVER MODE]
                // ==========================================
                try {
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
                    isExtracting = false;

                } catch (e) {
                    console.error("[SERVER MODE] Failed to send task:", e);
                    renderProgressToUI({ task_id: taskId, category: "Error", summary: `Cloud upload failed: ${e}`, spinner: "❌" });
                    isExtracting = false;
                }
                
            } else {
                // ==========================================
                // 💻 [LOCAL MODE]
                // ==========================================
                if (currentImage) {
                    console.log("[WIDGET] Queuing LOCAL IMAGE task...");
                    const imageRefHash = await hashId(currentImage);

                    await emit("new-task-from-browser", { 
                        id: taskId, type: "image_extraction", image_path: currentImage, 
                        ref: imageRefHash, 
                        cc: activeContext.cc || "",
                        bcc: activeContext.bcc || "",
                        link: "Local Image",
                        device_preference: getDevicePref(), search_mode: currentSearchMode
                    });
                } else {
                    console.log("[WIDGET] Queuing LOCAL HTML task...");
                    try {
                        const html = await invoke<string>("extract_html_from_current_tab");
                        const urlObj = new URL(currentDetectedUrl.toLowerCase());
                        const cc = await hashId(urlObj.hostname);
                        const rawPath = urlObj.pathname + urlObj.search;
                        const hashedRefId = await hashId(cc + rawPath.toLowerCase());
                        
                        await emit("new-task-from-browser", { 
                            id: taskId, type: "html_extraction", html: html, link: rawPath, 
                            cc: activeContext.cc || cc, 
                            ref: activeContext.ref || hashedRefId, 
                            bcc: activeContext.bcc || "", 
                            from: currentSession.address, to: currentSession.team,
                            device_preference: getDevicePref()
                        });
                    } catch (e) {
                        isExtracting = false;
                    }
                }
            }
            
            if (currentImage) {
                currentImage = null;
                if (navPreviewContainer) navPreviewContainer.classList.add("hidden");
                if (navUploadBtn) navUploadBtn.classList.remove("active-emoji");
                if (searchInput) searchInput.disabled = false;
                if (btnSubmit) btnSubmit.style.display = "flex";
            }

            if (wasExtracting) {
                console.log("[WIDGET] Task safely added to backend queue:", taskId);
            }
        }
    } finally {
        // 🌟 [CRITICAL FIX] Rust 백엔드(DB)에 작업이 완전히 등재되도록 1.5초간 여유를 줍니다.
        // 이 시간 동안은 버튼이 절대 부활하지 않으며, 1.5초 뒤 DB를 조회하여 정상적으로 큐에 등록되었다면 버튼은 계속 숨겨집니다.
        setTimeout(async () => {
            extractClickLock = false;
            await updateExtractButtonVisibility();
        }, 1500);
    }
});

listen("extraction-progress", async (event: any) => { 
    const payload = event.payload;
    if (payload.task_id) livePayloads.set(payload.task_id, payload); // 🌟 즉시 캐싱!

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

    if (!isRecovery && !isExtracting && !isTerminal && !isSearching) {
        if (payload.category === "Processing" || payload.category === "Preparation" || payload.category === "Vision" || payload.category === "Shipping" || payload.category === "Analytic") {
            console.log("[WIDGET] Queue task started, resuming extraction UI state.");
            isExtracting = true;
            startSpinner();
        }
    }

    const baseCategory = payload.category ? payload.category.replace(/\s*\(.*?\)/g, "") : "general";
    const catId = baseCategory.replace(/[^a-zA-Z0-9]/g, "");
    const elementId = `progress-${catId}`;
    
    let displaySummary = payload.summary || "";
    
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

    // 🌟 [CRITICAL FIX] 조기 종료(return) 방지! 다른 상세 페이지가 열려있더라도 채팅방 버블(말풍선)은 무조건 최신화합니다!
    let statusCode = 1; 
    if (isTerminal) {
        if (payload.category === "Done") statusCode = 9;
        else if (payload.category === "Error") statusCode = 6;
        else statusCode = 3;
    } else if (summary.includes("cancelled") || summary.includes("stopped")) {
        statusCode = 3;
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

        renderMessage({ 
            id: payload.task_id, 
            role: "system_task", 
            content: displaySummary, 
            status: statusCode, 
            created_at: originalCreatedAt, 
            updated_at: Date.now(),
            task_id: payload.task_id
        });
    }

    // 🌟 1차 스피너 및 전역 상태 종료 처리 (현재 활성화된 작업일 때만 버튼 UI 리셋)
    if (isTerminal && activeTaskId === tId) {
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
    }

    // 🌟 이제 현재 열려있는 Detail View가 이 Task의 것인지 확인 후 화면(DOM)을 업데이트합니다.
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
            localStorage.removeItem(`term_${tId}`);
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
            p.style.borderBottom = "1px solid #eee"; p.style.padding = "6px 0"; p.style.fontSize = "0.75rem";
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
                // 🌟 취소 시 localStorage 데이터 삭제
                localStorage.removeItem(`term_${activeTaskId}`);
                const el = document.getElementById(activeTaskId);
                if (el) el.remove();
            }

            activeTaskId = null;
            detailTitle.innerText = "Cancelled";
            detailContent.innerHTML = "<div style='color:#ef4444; padding:20px;'>Extraction stopped and deleted by user.</div>";
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
        await initSyncUI(); // [NEW] Initialize IP/Seed view
        listCurrentY = 0;
        updateListTransform(true);
    } else {
        qrContainer.classList.add("hidden");
    }
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

let mySyncSeed = Math.floor(1000 + Math.random() * 9000); 

async function initSyncUI() {
    const myFullIpEl = document.getElementById("my-full-ip");
    const mySyncSeedEl = document.getElementById("my-sync-seed");
    const ipPrefixEl = document.getElementById("ip-prefix");

    if (myFullIpEl) {
        const fullIp = await invoke("get_my_full_ip") as string;
        myFullIpEl.innerText = fullIp;
    }
    if (mySyncSeedEl) {
        mySyncSeedEl.innerText = mySyncSeed.toString();
    }
    if (ipPrefixEl) {
        const prefix = await invoke("get_local_network_prefix") as string;
        ipPrefixEl.innerText = prefix + ".";
    }
    
    try {
        await invoke("start_listener_command", { seed: mySyncSeed });
    } catch (e) { console.error(e); }
}

let peerConn: RTCPeerConnection | null = null;
let dataChannel: RTCDataChannel | null = null;
let desktopStream: MediaStream | null = null;
let qrRotationInterval: number | null = null;

function setupDataChannel(channel: RTCDataChannel) {
    channel.onopen = async () => {
        console.log("[WebRTC] Channel OPEN!");
        const profileName = document.getElementById("nav-profile-name");
        if (profileName) {
            profileName.textContent = "✅ Mobile Linked (P2P)";
            profileName.style.color = "#4ade80";
        }
        document.getElementById("nav-qr-container")?.classList.add("hidden");
        syncDataToMobile();

        // 🌟 [수정] 모바일 기기 정보를 DB에 넣을 때 로컬 꼬리표(is_device) 부착
        try {
            const mobileUser = {
                id: `mobile_${Date.now()}`,
                type: "user",
                name: "📱 Linked Mobile",
                from: currentSession.address || "0x0000000000000000000000000000000000000000", 
                to: currentSession.team || "0x0000000000000000000000000000000000000000",    
                data: { origin: "local", is_device: true } // 👈 여기서 구분합니다!
            };
            
            await invoke("upsert_items", { items: [mobileUser] });
            await renderNavigation();
        } catch (e) {
            console.error("[WebRTC] Failed to add mobile to members:", e);
        }
    };

    channel.onmessage = async (e) => {
        try {
            const msg = JSON.parse(e.data);
            console.log("[WebRTC] Received from Mobile:", msg.type);
            
            if (msg.type === "get_detail") {
                const doc = await invoke<any>("get_document", { uuid: msg.uuid });
                if (doc && dataChannel?.readyState === "open") {
                    dataChannel.send(JSON.stringify({
                        type: "sync_detail",
                        title: `${doc.doc_type || 'Detail'} ${doc.doc_number || ''}`,
                        content: `<div style="margin-bottom:15px;"><strong>Summary:</strong><br>${doc.text}</div><hr style="border-color:rgba(255,255,255,0.1);"><pre style="white-space: pre-wrap; font-size: 0.75rem; color:#fff; background:#000; padding:15px; border-radius:8px;">${doc.json_data}</pre>`
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

listen("task-console-log", (event: any) => {
    const { task_id, text } = event.payload;
    const key = `term_${task_id}`;
    
    // 🌟 sessionStorage -> localStorage 로 영구 보존!
    let logs = localStorage.getItem(key) || "";
    logs += text;
    localStorage.setItem(key, logs);

    const termArea = document.getElementById("terminal-logs");
    if (termArea && termArea.dataset.activeTaskId === task_id) {
        termArea.appendChild(document.createTextNode(text));
        termArea.style.display = "block"; // 🌟 [추가] 텍스트가 도착하면 까만 박스를 보여줍니다!
        termArea.scrollTop = termArea.scrollHeight; 
    }
});

function handleTaskClick(el: HTMLElement) {
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
        
        const savedLogs = localStorage.getItem(`term_${taskId}`);
        // 🌟 저장된 로그가 있을 때만 박스를 보여주고, 없으면 숨깁니다. (Connecting... 텍스트 제거)
        const displayStyle = savedLogs && savedLogs.trim() !== "" ? "block" : "none"; 
        
        logArea.innerHTML = `
            <div id="progress-container"></div>
            <div id="terminal-logs" data-active-task-id="${taskId}" style="display: ${displayStyle}; background: #0a0a0a; color: #4ade80; padding: 12px; font-family: monospace; font-size: 0.75rem; border-radius: 6px; max-height: 250px; overflow-y: auto; white-space: pre-wrap; border: 1px solid #333; box-shadow: inset 0 0 10px rgba(0,0,0,0.8); line-height: 1.4;">${savedLogs || ""}</div>
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
                    localStorage.setItem(`term_${taskId}`, reconstructed); 
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
            baseFilter = "type IN ('sales', 'event', 'users', 'pages')"; 
        } else {
            baseFilter = "type IN ('sales', 'goods', 'order', 'event', 'coupon', 'review', 'pages')";
        }

        // 기존 내비게이션 필터가 있다면 안전하게 괄호로 묶어서 AND 조건 추가
        if (activeContext.ref) baseFilter = `(${baseFilter}) AND ref = '${activeContext.ref}'`;
        else if (activeContext.bcc) baseFilter = `(${baseFilter}) AND bcc = '${activeContext.bcc}'`;
        else if (activeContext.cc) baseFilter = `(${baseFilter}) AND cc = '${activeContext.cc}'`;

        const textInput = searchInput?.value.toLowerCase() || "";
        let queryParts = activeTags.map(t => {
            if (t.type === 'domain') return `host:${t.value}`;
            if (t.type === 'type') return `type:${t.value.toLowerCase()}`;
            if (t.type === 'mode') return `mode:${t.value.toLowerCase()}`;
            return t.value;
        });
        if (textInput) queryParts.push(textInput);
        const textQuery = queryParts.join(" ");

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
            
            // [NEW] If syncing, also refresh the navigation tree (Shared Pages / Users)
            if (isSync) {
                renderNavigation();
            }
        } else if (reset) {
            docListContainer.innerHTML = "<div style='text-align:center; padding:20px; color:#999;'>No documents found.</div>";
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

    // Capture scroll state for prepend maintenance
    const scrollEl = document.getElementById("list-scroll");
    const prevScrollHeight = scrollEl ? scrollEl.scrollHeight : 0;
    const wasAtTop = listCurrentY <= 10; // Check if user was near the top

    // Sort: Newest created_at first (Descending)
    const sortedBatch = [...docs].sort((a, b) => b.created_at - a.created_at);
    
    // For 'prepend' (Sync), we iterate from Oldest to Newest in the batch 
    // and prepend each, so the absolute Newest ends up at the very top.
    const processBatch = mode === 'prepend' ? [...sortedBatch].reverse() : sortedBatch;

    processBatch.forEach(doc => {
        const docId = doc.id || doc.uuid || (doc.data && (doc.data.id || doc.data.uuid)) || doc.uuid_val || doc.ref || doc.index;
        const existingEl = docListContainer.querySelector(`[id="${docId}"]`) as HTMLElement;

        if (existingEl) {
            // [DIFF] Update if modified
            const cachedUpdatedAt = parseInt(existingEl.dataset.updatedAt || "0");
            if (doc.updated_at > cachedUpdatedAt) {
                console.log(`[List] Updating item ${docId}`);
                existingEl.outerHTML = item2html(doc, false, currentDetectedUrl);
                const newEl = document.getElementById(String(docId));
                if (newEl) bindCardEvents(newEl, doc);
            }
        } else {
            // [INSERT]
            const html = item2html(doc, false, currentDetectedUrl);
            const temp = document.createElement('div');
            temp.innerHTML = html;
            const newEl = temp.firstElementChild as HTMLElement;
            bindCardEvents(newEl, doc);

            if (mode === 'prepend') {
                docListContainer.prepend(newEl);
            } else {
                docListContainer.appendChild(newEl);
            }
        }
    });

    // [Scroll Maintenance]
    if (mode === 'prepend' && scrollEl) {
        const newScrollHeight = scrollEl.scrollHeight;
        const heightDiff = newScrollHeight - prevScrollHeight;
        if (heightDiff > 0) {
            if (wasAtTop) {
                // [FIX] If user was at the top, keep them at the top to see new items
                listCurrentY = 0;
            } else {
                // If they were scrolled down, maintain visual position of old items
                listCurrentY += heightDiff;
            }
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
    container.innerHTML = `<div style="padding:10px; text-align:center; font-size:0.75rem; color:var(--primary);"><span class="active-spinner">⠋</span> Loading related data...</div>`;
    
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
                <strong style="font-size:0.75rem; color:#aaa; margin-bottom:10px; display:block;">🔗 Related Documents</strong>
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
            detailContent.innerHTML = `<div style="margin-bottom:10px;"><strong>Summary:</strong><br>${doc.text}</div><hr style="border-color:#444;"><pre style="white-space: pre-wrap; font-size: 0.75rem; color:#fff; background:#111; padding:10px;">${prettyJson}</pre>`;
        } else {
            detailContent.innerHTML = "Document not found in database.";
        }
    } catch (e) { 
        console.error("[WIDGET] get_document failed:", e);
        detailContent.innerHTML = "Failed to load document details: " + e; 
    }
}

btnDetailBack?.addEventListener("click", () => { detailView.style.display = "none"; listView.style.display = "block"; });
document.getElementById("btn-settings-back")?.addEventListener("click", collapseWidget);
btnListBack?.addEventListener("click", collapseWidget);

document.getElementById("nav-signin")?.addEventListener("click", () => openWidget("settings"));
document.getElementById("nav-signout")?.addEventListener("click", () => { document.getElementById("btn-logout")?.click(); });

async function handleImageUpload(path: string) {
    currentImage = path;
    if (navPreviewContainer && navImgThumbnail) {
        navPreviewContainer.classList.remove("hidden");
        navUploadBtn?.classList.add("active-emoji");
        searchInput.disabled = true; 
        btnSubmit.style.display = "none"; 
        btnExtract.style.display = "flex";
        
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
    }
}

navImgClear?.addEventListener("click", async () => {
    currentImage = null; navPreviewContainer.classList.add("hidden"); navUploadBtn?.classList.remove("active-emoji");
    searchInput.disabled = false; btnSubmit.style.display = "flex"; 
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
        const queryParams: Record<string, string> = { origin: origin, created_at: createdAt.toString(), hash: currentSession.hash, href: window.location.href };
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
            saveSession();
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

        // 🌟 [추가] 로그인 성공 시 Cloud Members 영역 표시
        if (cloudMembersSection) cloudMembersSection.style.display = ""; 
        
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
    const html = `<div class="chat-talk system" id="msg-qr-auth" data-created-at="9999999999999"><div class="chat-message" style="padding:0; background: #fff; color: #000; border:0;"><div style="font-size:0.75rem; font-weight: bold; margin-bottom: 15px; color: #333;"><span id="qr-auth-spinner" class="active-spinner" style="margin-right:5px; font-family:monospace; color:var(--primary); font-weight:bold;">⠋</span>Scan the QR code</div><div id="qr-code-target" style="display: inline-block; background: #fff; border-radius: 8px;"></div></div></div>`;
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
        clearInterval(chatPollInterval);
        chatPollInterval = null;
        console.log("[WIDGET] Window blurred. Polling paused to save resources.");
    }
});

window.addEventListener("focus", () => {
    isFocus = true;
    if (!chatPollInterval && currentSession.email) {
        console.log("[WIDGET] Window focused. Polling resumed.");
        // 창을 다시 봤을 때 즉시 1회 최신화
        fetchChatHistory(false, true); 
        startPolling();
    }
});

// 🌟 [PARITY] startPolling 함수 업그레이드
function startPolling() {
    if (chatPollInterval) clearInterval(chatPollInterval);
    if (!isFocus) return; 
    
    chatPollInterval = window.setInterval(() => {
        if (!isFocus) return; 
        
        // 🌟 [사용자 요청 완벽 반영] 히스토리(Settings) 창이 열려있을 때만 서버에 인증/동기화 요청을 보냅니다!
        if (currentTab !== "settings" || !isExpanded) return;

        if (!currentSession.email) checkAuthStatus();
        else fetchChatHistory(false, true); 
    }, 3000);
}



function saveSession() { localStorage.setItem("chat_session", JSON.stringify(currentSession)); }

async function initSession() {
    // 🌟 [CRITICAL FIX 1] 앱 최초 실행 시, 묵은 localStorage 터미널 찌꺼기를 완벽 청소합니다!
    Object.keys(localStorage).forEach(key => {
        if (key.startsWith("term_")) {
            localStorage.removeItem(key);
        }
    });

    const saved = localStorage.getItem("chat_session");
    if (saved) { try { currentSession = { ...currentSession, ...JSON.parse(saved) }; } catch (e) {} } 
    else { const legacy = localStorage.getItem("device_hash"); if (legacy) currentSession.hash = legacy; }
    
    if (!currentSession.hash && ethers) { 
        const w = ethers.Wallet.createRandom(); 
        currentSession.hash = w.address.toLowerCase().replace("0x", ""); 
        saveSession(); 
    }
    
    saveSession(); 
    currentSession.address = currentSession.address || ZERO_ADDRESS; 
    currentSession.team = currentSession.team || await hashId(ZERO_ADDRESS); 
    updateAuthUI(); 
    startPolling();

    try {
        console.log("[WIDGET] UI Ready handshake starting...");
        const data = await invoke<any>("mark_ui_ready");

        // 브라우저 런처 상태 동기화
        if (btnAutoLaunch) {
            btnAutoLaunch.style.display = (data.browser_status === "running") ? "none" : "flex";
        }
        if (data.current_url) {
            currentDetectedUrl = data.current_url;
            isCurrentShop = data.is_client || data.is_admin;
            updateExtractButtonVisibility();
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
            // 서버 -> DB 저장 -> 화면 렌더링 흐름을 순차적으로 실행
            await syncData(); 
        } else {
            console.log("[WIDGET] 비로그인 상태. 로컬 LanceDB에서 메뉴를 불러옵니다...");
            // 1. LanceDB 불러오기 (오프라인/비로그인 모드)
            await renderNavigation();
        }

        // 🌟 [CRITICAL FIX] 앱(창) 새로고침 시, DB가 아닌 백엔드 메모리/JSON 백업에서 진행 중인 작업을 확실하게 복구합니다!
        try {
            const activeTask = await invoke<any>("get_active_task_context");
            if (activeTask && activeTask.id && (activeTask.status === 1 || activeTask.status === 10)) {
                console.log("[WIDGET] Resuming active task from fallback:", activeTask.id);
                
                renderMessage({
                    id: activeTask.id,
                    task_id: activeTask.id,
                    role: "system_task",
                    // 🌟 [CRITICAL FIX] 메모리에 최신 퍼센트 요약본이 있다면 그걸 쓰고, 아니면 기본 문구를 씁니다!
                    text: activeTask.summary || ("Resuming Task: " + (activeTask.link || "Local Source")),
                    status: activeTask.status,
                    created_at: activeTask.created_at || Date.now(),
                    updated_at: activeTask.updated_at || Date.now()
                });
                
                // 프론트엔드의 진행 상태 락(Lock)을 다시 걸어주고 스피너를 돌립니다.
                isExtracting = true;
                activeTaskId = activeTask.id; // 현재 활성화된 작업 ID 복구
                startSpinner();
                
                // 진행 중 버튼 상태 동기화
                await updateExtractButtonVisibility();
            }
        } catch (err) {
            console.warn("[WIDGET] No active task to resume or failed to fetch:", err);
        }

    } catch (e) { 
        console.error("[WIDGET] Handshake failed:", e); 
    }
}

document.getElementById("btn-qr-auth")?.addEventListener("click", performQrAuth);
document.getElementById("btn-logout")?.addEventListener("click", async () => { if (await ask("Are you sure?", { title: "Sign Out", kind: "warning" })) { currentSession.email = undefined; updateAuthUI(); } });
settingsBtn?.addEventListener("click", () => { if (currentTab === "settings" && isExpanded) collapseWidget(); else openWidget("settings"); });
document.getElementById("nav-to-auto")?.addEventListener("click", () => switchTab("automation"));
document.getElementById("unload-btn")?.addEventListener("click", async () => { try { await invoke("unload_model"); alert("Memory cleared."); } catch (e) {} });
async function syncBrowserStatus() { try { const s = await invoke<string>("get_browser_status"); if (btnAutoLaunch) btnAutoLaunch.style.display = (s === "running") ? "none" : "flex"; } catch (e) {} }
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
            const savedPref = localStorage.getItem("force_cpu_mode") === "true";
            forceCpuToggle.checked = savedPref;
        }
    } catch (e) {
        console.error("[WIDGET] Failed to check GPU status:", e);
    }

    // 3. Save on change
    forceCpuToggle.addEventListener("change", () => {
        localStorage.setItem("force_cpu_mode", forceCpuToggle.checked.toString());
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

function upsertChatMessages(messages: ChatMessage[], mode: 'prepend' | 'append') {
    if (!chatTalks) return;

    const scrollEl = document.getElementById("chat-scroll");
    const prevScrollHeight = scrollEl ? scrollEl.scrollHeight : 0;

    messages.forEach(msg => {
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
        const isTask = displayMsg.role === "system_task" || (displayMsg.role === "user" && !!displayMsg.task_id && displayMsg.task_id.startsWith("search_"));
        const domId = isTask ? (displayMsg.task_id || displayMsg.id) : displayMsg.id;
        
        const existingEl = chatTalks.querySelector(`[id="${domId}"]`) as HTMLElement;

        if (existingEl) {
            const cachedUpdatedAt = parseInt(existingEl.dataset.updatedAt || "0");
            const cachedStatus = parseInt(existingEl.dataset.status || "0");
            
            if (msg.updated_at > cachedUpdatedAt || msg.status !== cachedStatus || parseInt(existingEl.dataset.createdAt || "0") > Date.now()) {
                console.log(`[Chat] Updating ${domId}`);
                existingEl.outerHTML = createMessageHTML(displayMsg);
                
                const newEl = chatTalks.querySelector(`[id="${domId}"]`) as HTMLElement;
                if (newEl) {
                    newEl.classList.add("updated-flash");
                    setTimeout(() => newEl?.classList.remove("updated-flash"), 1000);
                    if (isTask) { newEl.onclick = () => handleTaskClick(newEl); }
                }
            }
        } else {
            const temp = document.createElement('div');
            temp.innerHTML = createMessageHTML(displayMsg);
            const newEl = temp.firstElementChild as HTMLElement;
            if (isTask) { newEl.onclick = () => handleTaskClick(newEl); }
            chatTalks.appendChild(newEl);
        }
    });

    // 🌟 [CRITICAL FIX] 복잡하고 버그가 많았던 insertBefore를 완전히 삭제!
    // DOM에 렌더링된 모든 자식들을 한 번에 메모리에 올린 후, 시간을 기준으로 100% 완벽하게 재정렬하여 덮어씌웁니다.
    const children = Array.from(chatTalks.children) as HTMLElement[];
    children.sort((a, b) => {
        const timeA = parseInt(a.dataset.createdAt || "0");
        const timeB = parseInt(b.dataset.createdAt || "0");
        return timeA - timeB;
    });
    children.forEach(c => chatTalks.appendChild(c));

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
    const statusMap: Record<number, { icon: string, text: string, color: string }> = {
        0: { icon: "✅", text: "done", color: "#22c55e" },
        1: { icon: "⏳", text: "processing", color: "var(--primary)" },
        2: { icon: "🛑", text: "stopped", color: "#ef4444" },
        3: { icon: "🚫", text: "cancelled", color: "#666" },
        6: { icon: "❌", text: "error", color: "#ef4444" },
        9: { icon: "✅", text: "done", color: "#22c55e" },
        10: { icon: "📥", text: "pending", color: "#999" }
    };
    const currentStatus = statusMap[msg.status] || statusMap[0];
    
    const isTaskBubble = msg.role === "system_task" || (msg.role === "user" && !!msg.task_id && msg.task_id.startsWith("search_"));
    const roleClass = msg.role === "user" ? "user" : "system";
    const domId = isTaskBubble ? (msg.task_id || msg.id) : msg.id;

    // 🌟 [CRITICAL FIX] 화면이 지워졌다 켜져도 절대 흔들리지 않는 불변의 생성 시간(태초의 시간)을 Task ID에서 직접 뽑아냅니다!
    let trueCreatedAt = msg.created_at;
    const match = domId.match(/_(\d+)$/);
    if (match) {
        trueCreatedAt = parseInt(match[1]);
    }
    const timeStr = new Date(trueCreatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    return `<div id="${domId}" class="chat-talk ${roleClass} ${isTaskBubble ? 'task-bubble' : ''}" 
        data-task-id="${msg.task_id || msg.id}" 
        data-status="${msg.status}" 
        data-updated-at="${msg.updated_at || trueCreatedAt}"
        data-created-at="${trueCreatedAt}"
        style="${isTaskBubble ? 'cursor:pointer;' : ''}">
        <div class="chat-message">
            <div style="font-size:0.6rem; opacity:0.5; margin-bottom:4px; display:flex; justify-content:space-between;">
                <span>${msg.role === 'user' ? '@YOU' : '🤖 LOGIS AI'}</span>
                <span>${timeStr}</span>
            </div>
            <div class="content">${msg.text}</div>
            ${isTaskBubble && msg.status !== 0 ? `<div class="status-bar" style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.1); font-size:0.65rem; font-weight:bold; color:${currentStatus.color};"><span class="${msg.status === 1 ? 'active-spinner' : ''}">${currentStatus.icon}</span> ${currentStatus.text.toUpperCase()}</div>` : ""}
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

        const messages = await invoke<any[]>("get_chat_messages", { limit: limit, offset: offset, filter: finalFilter });
        
        // 🌟 [CRITICAL FIX 2] 변수 스코프(Scope) 에러 해결! 
        // try 블록 바깥으로 빼내어 for 루프 안에서 변수를 찾지 못해 Processing이 멈추는 버그를 원천 차단합니다.
        let activeTask: any = null;
        try {
            activeTask = await invoke<any>("get_active_task_context");
            if (activeTask && activeTask.id) {
                const exists = messages.find(m => m.id === activeTask.id || m.task_id === activeTask.id);
                if (!exists) {
                    messages.push({
                        id: activeTask.id,
                        task_id: activeTask.id,
                        role: "system_task",
                        text: "Task Started: " + (activeTask.link || "Local Source"),
                        status: activeTask.status || 1,
                        created_at: activeTask.created_at || Date.now(),
                        updated_at: activeTask.updated_at || Date.now()
                    });
                }
            }
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
                    } else if (activeTask && activeTask.id === tId && activeTask.summary) {
                        rawSummary = activeTask.summary;
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
                if (!isHistory && chatTalks.querySelectorAll('.chat-talk').length === 0) {
                    chatTalks.insertAdjacentHTML('beforeend', "<div class='no-msg' data-created-at=\"0\" style='text-align:center; padding:20px; color:#999; font-size:0.75rem;'>No messages yet.</div>");
                }
            }

            if (isHistory && !chatHasMore && !chatTalks.querySelector('.chat-history-end')) {
                const endHtml = `<div class="chat-talk system chat-history-end" data-created-at="0" style="text-align:center; opacity:0.4; font-size:0.6rem; padding:15px 10px;">
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

function renderMessage(msg: any, shouldScroll: boolean = true, isPrepend: boolean = false) {
    if (!chatTalks) return;
    // Single message upsert (Real-time is always append/newest in Slack style)
    upsertChatMessages([msg], isPrepend ? 'prepend' : 'append');
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