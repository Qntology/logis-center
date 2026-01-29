console.log("%c[WIDGET] MAIN.TS LOADED", "color: #00ff00; font-weight: bold; font-size: 1.2rem;");
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { open, ask } from '@tauri-apps/plugin-dialog';
import { listen, emit } from '@tauri-apps/api/event';
import { readFile } from '@tauri-apps/plugin-fs';

// Imports for Rendering & Shim
import { item2html, selector } from "./lib/render";
import { Select, Upsert } from "./lib/db";
import { hashId } from "./lib/utils";

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

let selectedUuids = new Set<string>();
let currentDetailUuid: string | null = null;
let activeTaskId: string | null = null; // [NEW] Track current extraction task
let isExtracting = false; 
let spinnerInterval: number | null = null;
let qrSpinnerIndex = 0; // [NEW] Track discrete frame for QR spinner
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
const listScrollContainer = document.getElementById("list-scroll-container") as HTMLElement;
const loadingIndicator = document.getElementById("loading-indicator") as HTMLElement;

const aiResultsArea = document.getElementById("ai-search-results") as HTMLElement;
const aiResultsTitle = document.getElementById("ai-results-title") as HTMLElement;
const aiResultsContent = document.getElementById("ai-results-content") as HTMLElement;

const chatTalks = document.querySelector('.chat-talks') as HTMLElement;
const chatForm = document.querySelector('form[name="chat-form"]') as HTMLFormElement;

// --- Spinner Logic ---
const spinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

function startSpinner() {
    if (spinnerInterval) clearInterval(spinnerInterval);
    
    // [FIX] Use btnSettings as the spinner container
    if (settingsBtn) {
        settingsBtn.classList.add("active-spinner-mode");
        // Ensure the extraction button is hidden while the global spinner is active
        if (btnExtract) btnExtract.style.display = "none";
    }
    
    let i = 0;
    spinnerInterval = window.setInterval(() => {
        const char = spinnerFrames[i % spinnerFrames.length];
        if (settingsBtn) settingsBtn.innerText = char;
        
        // [NEW] Update pull-loader spinners
        document.querySelectorAll('.chat-pull-loader .spinner').forEach(el => {
            (el as HTMLElement).innerText = char;
        });

        document.querySelectorAll('.active-spinner').forEach(el => {
            (el as HTMLElement).innerText = char;
        });
        i++;
    }, 80);
}

function stopSpinner() {
    if (spinnerInterval) {
        clearInterval(spinnerInterval);
        spinnerInterval = null;
    }
    
    // [FIX] Only restore the icon if we are NOT currently extracting data
    if (!isExtracting && settingsBtn) {
        settingsBtn.classList.remove("active-spinner-mode");

        if(settingsBtn.classList.contains('active')){
            settingsBtn.innerText = "💬";
        }else{
            settingsBtn.innerText = "🗨️";
        }
        
    }
    
    // [FIX] Comprehensive cleanup of all active-spinner classes and elements
    document.querySelectorAll('.active-spinner').forEach(el => {
        el.classList.remove('active-spinner');
        if (el.tagName === 'SPAN' && (el as HTMLElement).innerText.length <= 2) {
             (el as HTMLElement).innerText = ""; // Clear icon
        }
    });
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

// --- Search & Main Nav ---
async function updateExtractButtonVisibility() {
    if (!btnExtract) return;
    
    // [UI-FIX] If already extracting or the global spinner is active, keep the button hidden
    if (isExtracting || spinnerInterval) {
        btnExtract.style.display = "none";
        btnExtract.classList.remove("active-spinner");
        return;
    }

    if (currentImage) {
        btnExtract.style.display = "flex";
        btnExtract.innerHTML = "⚡";
        btnExtract.classList.remove("active-spinner");
        btnExtract.title = "Extract from Image";
        return;
    }

    // [FIX] Only show extract button if it's a confirmed shop domain
    if (!currentDetectedUrl || !isCurrentShop) {
        btnExtract.style.display = "none";
        return;
    }

    // [UI-FIX] Show the button immediately before performing async checks
    btnExtract.style.display = "flex";
    btnExtract.innerHTML = "⚡";
    btnExtract.classList.remove("active-spinner");

    try {
        const urlObj = new URL(currentDetectedUrl.toLowerCase());
        const hostname = urlObj.hostname; 
        const link = (urlObj.pathname + urlObj.search).toLowerCase();
        const ccHash = await hashId(hostname); 
        const hashedRefId = await hashId(ccHash + link);
        
        btnExtract.title = `Extract from ${hostname}`;

        const isActive = await invoke<boolean>("check_active_task", {
                    payload: { cc: ccHash, ref: hashedRefId }
                });
        
        // [FIX] Priority Guard: Even if server says active, trust the local isExtracting flag
        // especially during the transition period after clicking Stop button.
        if (isActive === true && isExtracting) {
            btnExtract.classList.add("active-spinner");
            btnExtract.title = "Extraction in progress...";
            if (!spinnerInterval) startSpinner(); 
        } else if (!isExtracting) {
            // Ensure UI is reset if we are definitely not extracting
            btnExtract.classList.remove("active-spinner");
            btnExtract.innerText = "⚡";
        }
    } catch (e) { 
        console.warn("[WIDGET] visibility check error:", e);
    }
}

listen("browser-match-found", async (event: any) => {
    const payload = event.payload;
    console.log("[WIDGET] Browser Match Found:", payload);
    
    currentDetectedUrl = payload.url;
    isCurrentShop = payload.is_client || payload.is_admin;

    if (isCurrentShop) {
        // If a match is found, the browser must be running, so hide launch button
        if (btnAutoLaunch) btnAutoLaunch.style.display = "none";
    } else {
        // Even if not a shop, if we have a URL, the browser is running
        if (currentDetectedUrl && btnAutoLaunch) btnAutoLaunch.style.display = "none";
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
                name = `<span>${nodeType}</span> <span>${(data.item ? " Draft" : " ")}</span>`;

                if (data.origin) {
                    var _url = new URL(data.origin);
                    if (!navTmp[_url.host] && data.item) {
                        host = `<strong>${_url.host}</strong>`;
                        navTmp[_url.host] = true;
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
                           data-domain="${node.domain || ''}" 
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
        let _pages = await Select["pages"]();
        
        if (_pages.length === 0) {
            pageList.innerHTML = "<div style='color:#999; padding:10px; font-size:0.75rem;'>No shared pages found.</div>";
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

            // [FIX] Navigation rendered, stop spinner if it was the first time
            if (isFirstNavRender) {
                isFirstNavRender = false;
                stopSpinner();
            }

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
        const users = await Select["users"]();
        userList.innerHTML = "";
        if (users.length > 0) {
            const teamNodes = users.filter(u => u.type === "team").map(u => ({...u, children: users.filter(m => m.to === u.id && m.id !== u.id)}));
            userList.innerHTML = await renderAccordion(teamNodes);
        }

    } catch (e) { console.error("Nav render error:", e); }
}

// --- Sync Logic ---
async function syncData() {
    if (!currentSession.hash || !currentSession.email) return;
    
    console.log("[SYNC] Starting data synchronization...");
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
        
        const response = await invoke<any>("proxy_fetch", {
            url: url,
            method: "GET",
            headers: { "Content-Type": "application/json" },
            session_params: { hash: currentSession.hash, token: currentSession.token }
        });

        // [FIX] Advance spinner one step
        stepQrSpinner();

        if (response.results && Array.isArray(response.results)) {
            await Upsert["items"](response.results);
            console.log("[SYNC] Data upserted.");
            
            // [REACTIVE] Re-render navigation immediately after sync
            await renderNavigation();
            // Also refresh list if on list tab
            if (currentTab === "list") await loadMoreDocs(true);
        }
        
    } catch (e) { 
        console.error("[SYNC] Failed:", e); 
    } finally {
        // [FIX] Stop spinner
        if (!isExtracting) stopSpinner();
    }
}

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
    if(searchDebounceTimer) clearTimeout(searchDebounceTimer);
    searchDebounceTimer = window.setTimeout(async () => {
        await loadMoreDocs(true);
    }, 800);
});

btnSubmit?.addEventListener("click", async () => {
    const query = searchInput.value;
    if (!query) return;
    openWidget("list"); 
    if (aiResultsArea && aiResultsContent) {
        aiResultsArea.style.display = "block";
        aiResultsTitle.innerText = "🧠 AI Deep Analysis";
        aiResultsContent.innerHTML = "<div class='spinner'></div> 🤖 Analyzing your query...";
        try {
            const devicePref = forceCpuToggle.checked ? "cpu" : null;
            const response = await invoke<any>("ai_search_complex", { 
                query: query, 
                language: "korean",
                devicePreference: devicePref
            });
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
        } catch(e) { aiResultsContent.innerHTML = "<div style='color:#ef4444;'>Error: " + e + "</div>"; }
    }
});

document.addEventListener('show-doc', (e: any) => showDetail(e.detail));
document.addEventListener('view-task-log', () => { openWidget("list"); listView.style.display = "none"; detailView.style.display = "flex"; });

btnExtract?.addEventListener("click", async () => {
    if (currentDetectedUrl) {
        try {
            const normalizedUrl = currentDetectedUrl.toLowerCase();
            const urlObj = new URL(normalizedUrl);
            const hostname = urlObj.hostname;
            const link = (urlObj.pathname + urlObj.search).toLowerCase();
            const ccHash = await hashId(hostname); 
            const hashedRefId = await hashId(ccHash + link);
            const isActive = await invoke<boolean>("check_active_task", {
                        payload: { cc: ccHash, ref: hashedRefId }
                    });            if (isActive) {
                alert("This page is already in the queue or being processed.");
                openWidget("settings");
                await updateExtractButtonVisibility();
                return;
            }
        } catch(e) { console.error("[WIDGET] Pre-click check error:", e); }
    }

    btnExtract.style.opacity = "0.5";
    const logArea = document.getElementById("extraction-log");
    
    // [UI-RESET] Clear previous task's debris and status messages
    if (logArea) {
        logArea.innerHTML = "";
        logArea.dataset.activeTaskId = ""; // Clear active task tracking
    }
    if (detailTitle) detailTitle.innerText = "Task Progress";
    if (detailContent && logArea) {
        // Keep only the log area, remove any error/stop messages from previous run
        detailContent.innerHTML = "";
        detailContent.appendChild(logArea);
    }

    setTimeout(() => { if (btnExtract) btnExtract.style.opacity = "1"; }, 300);

    openWidget("settings");
    startSpinner();
    
    if (currentImage) {
        isExtracting = true;
        const taskId = `img_${Date.now()}`;
        activeTaskId = taskId; // [NEW]
        if (logArea) logArea.dataset.activeTaskId = taskId; // [CRITICAL-FIX]
        try {
            await emit("new-task-from-browser", { 
                id: taskId, type: "image_extraction", image_path: currentImage, 
                ref: currentImage, link: "Local Image",
                device_preference: forceCpuToggle?.checked ? "cpu" : null
            });
            renderMessage({ id: taskId, role: "system_task", content: `Queued: Image Analysis`, status: 10, task_id: taskId, created_at: Date.now() });
            await updateExtractButtonVisibility();
        } catch (e) { isExtracting = false; await updateExtractButtonVisibility(); }
    } else {
        isExtracting = true;
        let taskId = `task_${Date.now()}`;
        if (logArea) logArea.dataset.activeTaskId = taskId; // [CRITICAL-FIX]
        try {
            const html = await invoke<string>("extract_html_from_current_tab");
            const normalizedUrl = currentDetectedUrl.toLowerCase();
            const urlObj = new URL(normalizedUrl);
            const cc = await hashId(urlObj.hostname);
            const rawPath = urlObj.pathname + urlObj.search;
            const hashedRefId = await hashId(cc + rawPath.toLowerCase());
            activeTaskId = hashedRefId; // [NEW] In case of HTML, we often use the hash as ref
            
            await emit("new-task-from-browser", { 
                id: taskId, type: "html_extraction", html: html, link: rawPath, 
                cc: cc, ref: hashedRefId, from: currentSession.address, to: currentSession.team,
                device_preference: forceCpuToggle?.checked ? "cpu" : null
            });
            renderMessage({ id: taskId, role: "system_task", content: `Queued: ${urlObj.hostname}`, status: 10, task_id: hashedRefId, created_at: Date.now() });
            await updateExtractButtonVisibility();
        } catch (e) { isExtracting = false; await updateExtractButtonVisibility(); }
    }
});

listen("extraction-progress", async (event: any) => { renderProgressToUI(event.payload); });
document.addEventListener('render-progress', (e: any) => { renderProgressToUI(e.detail); });

async function renderProgressToUI(payload: any, isRecovery: boolean = false) {
    const summary = (payload.summary || "").toLowerCase();
    const isTerminal = payload.category === "Done" || payload.category === "Error" || summary.includes("cancelled") || summary.includes("stopped");

    // [CRITICAL-FIX] If the user manually stopped extraction, ignore ALL incoming progress events 
    // unless they are the final terminal signals confirming the stop/error/completion.
    // [RECOVERY-BYPASS] If we are recovering logs, bypass this guard.
    if (!isRecovery && !isExtracting && !isTerminal) {
        console.log("[WIDGET] Ignoring late progress event after stop:", payload.category);
        return; 
    }

    const catId = payload.category ? payload.category.replace(/[^a-zA-Z0-9]/g, "") : "general";
    const elementId = `progress-${catId}`;
    const extractionLog = document.getElementById("extraction-log");
    
    if (extractionLog && extractionLog.dataset.activeTaskId) {
        if (payload.task_id && payload.task_id !== extractionLog.dataset.activeTaskId) return;
    }

    if (payload.category === "Done" || payload.category === "Error") {
        isExtracting = false; // Ensure flag is synced
        stopSpinner();
        if (btnExtract) {
            btnExtract.classList.remove("active-spinner");
            btnExtract.innerText = "⚡";
        }
        await updateExtractButtonVisibility();
        
        // [FIX] Continue to mapping but we'll return at the end of terminal mapping
    }

    if (payload.task_id) {
        let statusCode = 1; 
        const summary = (payload.summary || "").toLowerCase();
        
        // [FIX] Correctly map status by checking summary text first
        if (summary.includes("cancelled") || summary.includes("stopped")) {
            statusCode = 3; // Cancelled
            isExtracting = false;
            stopSpinner();
        } else if (payload.category === "Done") {
            statusCode = 9; // Success
        } else if (payload.category === "Error") {
            statusCode = 6; // Error
        }
        
        renderMessage({ 
            id: payload.task_id, 
            role: "system_task", 
            content: payload.summary, 
            status: statusCode, 
            created_at: Date.now() 
        });

        // [FIX] If terminal status, don't proceed to draw new spinners in the log
        if (statusCode === 3 || statusCode === 9 || statusCode === 6) {
            if (btnStopTask) btnStopTask.style.display = "none";
            return; // [CRITICAL] Stop processing immediately for terminal states
        }
    }

    if (extractionLog && detailView.style.display !== "none") {
         let p = document.getElementById(elementId);
         if (!p) {
             p = document.createElement("div"); p.id = elementId;
             p.className = "progress-item";
             p.style.borderBottom = "1px solid #eee"; p.style.padding = "6px 0"; p.style.fontSize = "0.75rem";
             p.style.display = "flex"; p.style.flexDirection = "column"; 
             const row = document.createElement("div"); row.className = "progress-row"; row.style.display = "flex"; row.style.alignItems = "center";
             
             // Initial creation
             const spinnerIcon = `<span class="active-spinner" style="color:var(--primary); margin-right:8px; font-family:monospace; min-width:15px;">⠋</span>`;
             row.innerHTML = `${spinnerIcon}<span class="summary-text">${payload.summary || ""}</span>`;
             p.appendChild(row);
             const results = document.createElement("div"); results.className = "results-container"; p.appendChild(results);
             extractionLog.appendChild(p);
         }
         
         const summaryEl = p.querySelector(".summary-text") as HTMLElement;
         const spinnerEl = p.querySelector(".active-spinner") as HTMLElement;
         const resultsContainer = p.querySelector(".results-container");

         // [SMART-UPDATE] Only update DOM if the content actually changed
         if (summaryEl && summaryEl.textContent !== payload.summary) {
             summaryEl.textContent = payload.summary || "";
         }

         if (payload.category === "Done") {
             if (btnStopTask) btnStopTask.style.display = "none";
             if (btnDetailDelete) btnDetailDelete.style.display = "flex";
             if (globalNavSpinner) globalNavSpinner.style.display = "none"; 
             
             // Finalize this specific row
             const row = p.querySelector(".progress-row");
             if (row) {
                 const s = row.querySelector(".active-spinner");
                 if (s) {
                     s.classList.remove("active-spinner");
                     s.innerHTML = "✅";
                     s.style.color = "#4ade80";
                 }
             }
         } else if (payload.category === "Error") {
             const row = p.querySelector(".progress-row");
             if (row) { 
                 const s = row.querySelector(".active-spinner");
                 if (s) {
                     s.classList.remove("active-spinner");
                     s.innerHTML = "❌";
                 }
                 (row as HTMLElement).style.color = "#ef4444"; 
             }
         } else {
             // Intermediate update: ensure spinner is animating if it's still processing
             if (spinnerEl) {
                 const newIcon = payload.spinner || "⠋";
                 if (spinnerEl.innerText !== newIcon) {
                     spinnerEl.innerText = newIcon;
                 }
                 if (newIcon === "✅" || newIcon === "✔") {
                     spinnerEl.classList.remove("active-spinner");
                     spinnerEl.style.color = "#4ade80";
                 } else if (newIcon === "❌") {
                     spinnerEl.classList.remove("active-spinner");
                     spinnerEl.style.color = "#ef4444";
                 } else {
                     spinnerEl.classList.add("active-spinner");
                 }
             }
         }
    }
}

btnStopTask?.addEventListener("click", async () => {
    if (await ask("Stop the current extraction? (The record will be deleted)", { title: "Stop Task", kind: "warning" })) {
        // [UI-FIRST] Reset everything immediately to prevent race conditions
        isExtracting = false; 
        stopSpinner();
        
        if (btnExtract) {
            btnExtract.classList.remove("active-spinner");
            btnExtract.innerText = "⚡";
            btnExtract.style.display = "flex";
        }
        if (btnStopTask) btnStopTask.style.display = "none";

        try {
            console.log("[WIDGET] Stopping task:", activeTaskId);
            // [FIX] Pass activeTaskId accurately
            await invoke<string>("stop_current_extraction", { taskId: activeTaskId });
            
            // Remove the message bubble immediately from UI
            if (activeTaskId) {
                const msgId = `msg-task-${activeTaskId}`;
                const el = document.getElementById(msgId);
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
    if(docListContainer) docListContainer.innerHTML = "";
    await loadMoreDocs();
}

async function loadMoreDocs(reset: boolean = false) {
    if (reset) {
        currentPage = 0; hasMore = true;
        if (docListContainer) docListContainer.innerHTML = "";
        cachedDocs = [];
    }

    if (isLoading || !hasMore) {
        if (reset) stopSpinner(); // Stop if already loading
        return;
    }

    // [FIX] Always start spinner for any document fetching
    startSpinner();
    isLoading = true;
    if (loadingIndicator) loadingIndicator.style.display = "block";
    
    try {
        let docs: any[] = [];
        const textInput = searchInput?.value.toLowerCase() || "";
        let queryParts = activeTags.map(t => {
            if (t.type === 'domain') return `host:${t.value}`;
            if (t.type === 'type') return `type:${t.value.toLowerCase()}`;
            if (t.type === 'mode') return `mode:${t.value.toLowerCase()}`;
            return t.value;
        });
        if (textInput) queryParts.push(textInput);
        const finalQuery = queryParts.join(" ");

        docs = await Select["items"]({ value: finalQuery, limit: pageSize, offset: currentPage * pageSize });

        if (docs.length < pageSize) hasMore = false;
        if (docs.length > 0) {
            cachedDocs = [...cachedDocs, ...docs];
            renderDocs(docs);
            currentPage++;
        } else if (currentPage === 0) {
            if (docListContainer) docListContainer.innerHTML = "<div style='text-align:center; padding:20px; color:#999;'>No documents found.</div>";
        }
    } catch (e) { 
        console.error("[WIDGET] loadMoreDocs error:", e);
        if (currentPage === 0 && docListContainer) docListContainer.innerHTML = `<div style='text-align:center; padding:20px; color:#ef4444;'>Error loading data.</div>`;
    } 
    finally { 
        isLoading = false; 
        if (loadingIndicator) loadingIndicator.style.display = "none"; 
        // [FIX] Always stop spinner when loading attempt finishes
        stopSpinner();
    }
}

function renderDocs(docs: any[]) {
    if (!docListContainer) return;
    docs.forEach(doc => {
        const html = item2html(doc, false, currentDetectedUrl);
        docListContainer.insertAdjacentHTML('beforeend', html);
        const lastEl = docListContainer.lastElementChild as HTMLElement;
        if (lastEl) {
            lastEl.addEventListener("click", (e) => {
                const target = e.target as HTMLElement;
                // If user clicked 'toggle-more' or its label, let CSS/HTML handle it
                if (target.closest('.toggle-more') || target.closest('.more-label')) return;
                
                // [STRICT-ID] Try every possible ID field name
                const docId = doc.id || doc.uuid || (doc.data && (doc.data.id || doc.data.uuid)) || doc.uuid_val || doc.ref || doc.index;
                
                if (!target.closest('a') && !target.closest('input') && !target.closest('button')) {
                    if (docId) {
                        showDetail(String(docId));
                    } else {
                        console.warn("[WIDGET] Item clicked but no valid ID found:", doc);
                    }
                }
            });
        }
    });
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
        searchInput.disabled = true; btnSubmit.style.display = "none"; btnExtract.style.display = "flex";
        try {
            const contents = await readFile(currentImage);
            const blob = new Blob([contents]);
            const reader = new FileReader();
            reader.onloadend = () => { navImgThumbnail.src = reader.result as string; };
            reader.readAsDataURL(blob);
        } catch (e) { navImgThumbnail.src = convertFileSrc(currentImage); }
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
    if (currentSession.email) {
        if (authStatus) authStatus.innerText = "Authenticated";
        if (btnLogout) btnLogout.style.display = "block";
        if (btnQrAuth) btnQrAuth.style.display = "none";
        if (chatForm) chatForm.classList.remove("hidden");
        const qrMsg = document.getElementById("msg-qr-auth");
        if (qrMsg) qrMsg.remove();
    } else {
        if (authStatus) authStatus.innerText = "Waiting for Auth...";
        if (btnLogout) btnLogout.style.display = "none";
        if (btnQrAuth) btnQrAuth.style.display = "block";
        if (chatForm) chatForm.classList.add("hidden");
    }
}

async function performQrAuth() {
    if (!chatTalks || !currentSession.hash) return;
    const existing = document.getElementById("msg-qr-auth");
    if (existing) existing.remove();
    const html = `<div class="chat-talk system" id="msg-qr-auth"><div class="chat-message" style="padding:0; background: #fff; color: #000; border:0;"><div style="font-size:0.75rem; font-weight: bold; margin-bottom: 15px; color: #333;"><span id="qr-auth-spinner" class="active-spinner" style="margin-right:5px; font-family:monospace; color:var(--primary); font-weight:bold;">⠋</span>Scan the QR code</div><div id="qr-code-target" style="display: inline-block; background: #fff; border-radius: 8px;"></div></div></div>`;
    chatTalks.insertAdjacentHTML('beforeend', html);
    const qrTarget = document.getElementById("qr-code-target");
    if (qrTarget) {
        qrTarget.innerHTML = "";
        new (window as any).QRCode(qrTarget, { text: `mailto:${encodeURIComponent(currentSession.hash + ".logis.center@oauth.email")}`, width: 300, height: 300, colorDark: "#000000", colorLight: "#ffffff", correctLevel: (window as any).QRCode.CorrectLevel.M });
        const scroll = document.getElementById("chat-scroll");
        if (scroll) scroll.scrollTop = scroll.scrollHeight;
    }
}

function startPolling() {
    if (chatPollInterval) clearInterval(chatPollInterval);
    chatPollInterval = window.setInterval(() => {
        if (!currentSession.email) checkAuthStatus();
        else fetchChatHistory();
    }, 3000);
}

function renderMessage(msg: any, shouldScroll: boolean = true) {
    if (!chatTalks) return;
    const msgId = msg.role === "system_task" ? `msg-task-${msg.task_id || msg.id}` : `msg-${msg.id}`;
    let existing = document.getElementById(msgId);
    const isSystemTask = msg.role === "system_task";
    const roleClass = msg.role === "user" ? "user" : "system";
    if (isSystemTask && !existing) {
        systemLogCount++;
        const countEl = document.querySelector('.system-label .count');
        if (countEl) countEl.innerHTML = `(${systemLogCount})`;
    }
    const statusMap: Record<number, { icon: string, text: string, color: string }> = { 1: { icon: "⏳", text: "processing", color: "var(--primary)" }, 2: { icon: "🛑", text: "stopped", color: "#ef4444" }, 3: { icon: "🚫", text: "cancelled", color: "#666" }, 6: { icon: "❌", text: "error", color: "#ef4444" }, 9: { icon: "✅", text: "done", color: "#22c55e" }, 10: { icon: "📥", text: "pending", color: "#999" } };
    const currentStatus = statusMap[msg.status] || statusMap[1];
    const timeStr = new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    let displayHtml = "";
    let itemData: any = null;
    try { itemData = typeof msg.content === 'string' ? JSON.parse(msg.content) : msg.content; } catch (e) { displayHtml = msg.content; }
    if (itemData && typeof itemData === 'object') displayHtml = itemData.text || itemData.title || itemData.summary || JSON.stringify(itemData);
    const htmlContent = `<div class="chat-message"><div style="font-size:0.6rem; opacity:0.5; margin-bottom:4px; display:flex; justify-content:space-between;"><span>${msg.role === 'user' ? '@YOU' : '🤖 LOGIS AI'}</span><span>${timeStr}</span></div><div class="content">${displayHtml}</div>${isSystemTask ? `<div class="status-bar" style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.1); font-size:0.65rem; font-weight:bold; color:${currentStatus.color};"><span class="${msg.status === 1 ? 'active-spinner' : ''}">${currentStatus.icon}</span> ${currentStatus.text.toUpperCase()}</div>` : ""}</div>`;
    if (existing) {
        existing.className = `chat-talk ${roleClass} ${isSystemTask ? 'task-bubble' : ''}`;
        existing.dataset.status = msg.status; existing.innerHTML = htmlContent;
    } else {
        const messageDiv = document.createElement("div"); messageDiv.id = msgId; messageDiv.className = `chat-talk ${roleClass} ${isSystemTask ? 'task-bubble' : ''}`;
        messageDiv.dataset.taskId = msg.task_id || msg.id; messageDiv.dataset.status = msg.status; messageDiv.innerHTML = htmlContent;
        if (isSystemTask) {
            messageDiv.style.cursor = "pointer";
            messageDiv.onclick = async () => {
                const taskId = messageDiv.dataset.taskId; const status = parseInt(messageDiv.dataset.status || "0");
                if (status === 9 && taskId) showDetail(taskId);
                else if (taskId) {
                    openWidget("list"); listView.style.display = "none"; detailView.style.display = "flex"; detailTitle.innerText = "Task Progress";
                    const logArea = document.getElementById("extraction-log");
                    if (logArea && logArea.dataset.activeTaskId !== taskId) {
                        logArea.innerHTML = "<div style='padding:10px;'>📡 Loading task history...</div>"; logArea.dataset.activeTaskId = taskId;
                        const logs = await invoke<any[]>("get_task_logs", { taskId }); logArea.innerHTML = ""; logs.forEach(l => renderProgressToUI(l));
                    }
                }
            };
        }
        chatTalks.insertAdjacentElement('beforeend', messageDiv);
    }
    const scroll = document.getElementById("chat-scroll"); 
    if (scroll && shouldScroll && !existing) scroll.scrollTop = scroll.scrollHeight;
}

function saveSession() { localStorage.setItem("chat_session", JSON.stringify(currentSession)); }
async function initSession() {
    const saved = localStorage.getItem("chat_session");
    if (saved) { try { currentSession = { ...currentSession, ...JSON.parse(saved) }; } catch (e) {} } 
    else { const legacy = localStorage.getItem("device_hash"); if (legacy) currentSession.hash = legacy; }
    if (!currentSession.hash && ethers) { const w = ethers.Wallet.createRandom(); currentSession.hash = w.address.toLowerCase().replace("0x", ""); saveSession(); }
    saveSession(); currentSession.address = currentSession.address || ZERO_ADDRESS; currentSession.team = currentSession.team || await hashId(ZERO_ADDRESS); updateAuthUI(); startPolling();

    // [SUPER-HANDSHAKE] Get all initial data in one go
    try {
        console.log("[WIDGET] UI Ready handshake starting...");
        const data = await invoke<any>("mark_ui_ready");
        console.log("[WIDGET] Initial data received:", data);

        // 1. Browser & URL Status
        if (btnAutoLaunch) {
            btnAutoLaunch.style.display = (data.browser_status === "running") ? "none" : "flex";
        }
        
        if (data.current_url) {
            currentDetectedUrl = data.current_url;
            isCurrentShop = data.is_client || data.is_admin; // [FIX] Sync shop status on startup
            // Trigger UI update based on the detected URL
            updateExtractButtonVisibility();
        }

        // 2. Pre-populate items list (if empty)
        if (data.items && data.items.length > 0 && cachedDocs.length === 0) {
            cachedDocs = data.items;
            renderDocs(data.items);
            currentPage = 1;
        }

        // 3. Trigger tree renders background
        // (renderNavigation will now use cached data if we optimize it further, 
        // but for now, the DB is warm and ready)
        renderNavigation();

        // 4. If there are active tasks, start spinner
        if (data.tasks && data.tasks.length > 0) {
            const lastTask = data.tasks[data.tasks.length - 1];
            renderMessage({ 
                id: lastTask.id, role: "system_task", 
                content: `Resuming: ${lastTask.id}`, 
                status: 1, created_at: Date.now() 
            });
            startSpinner();
        }
    } catch (e) { console.error("[WIDGET] Handshake failed:", e); }
}

document.getElementById("btn-qr-auth")?.addEventListener("click", performQrAuth);
document.getElementById("btn-logout")?.addEventListener("click", async () => { if (await ask("Are you sure?", { title: "Sign Out", kind: "warning" })) { currentSession.email = undefined; updateAuthUI(); } });
listScrollContainer?.addEventListener("scroll", () => { if (listScrollContainer.scrollTop + listScrollContainer.clientHeight >= listScrollContainer.scrollHeight - 20) loadMoreDocs(); });
settingsBtn?.addEventListener("click", () => { if (currentTab === "settings" && isExpanded) collapseWidget(); else openWidget("settings"); });
document.getElementById("nav-to-auto")?.addEventListener("click", () => switchTab("automation"));
document.getElementById("unload-btn")?.addEventListener("click", async () => { try { await invoke("unload_model"); alert("Memory cleared."); } catch (e) {} });
async function syncBrowserStatus() { try { const s = await invoke<string>("get_browser_status"); if (btnAutoLaunch) btnAutoLaunch.style.display = (s === "running") ? "none" : "flex"; } catch (e) {} }
// --- Device Preference Logic ---
const forceCpuToggle = document.getElementById("force-cpu-toggle") as HTMLInputElement;

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
const PULL_THRESHOLD = 50;
const PULL_MAX = 90;
const FRICTION = 0.3;

function initChatPullLogic() {
    const container = document.querySelector(".chat-container") as HTMLElement;
    const scrollEl = document.getElementById("chat-scroll") as HTMLElement;
    const topLoader = document.getElementById("chat-pull-top") as HTMLElement;
    const bottomLoader = document.getElementById("chat-pull-bottom") as HTMLElement;
    
    if (!container || !scrollEl || !topLoader || !bottomLoader) return;

    const updateTransform = (resetting: boolean = false) => {
        if (resetting) scrollEl.classList.add("resetting");
        else scrollEl.classList.remove("resetting");

        // Final Position = -(Scroll Position) + Pull Offset
        scrollEl.style.transform = `translateY(${-currentY + pullY}px)`;

        // Show/Hide Loaders
        if (pullY > 0) {
            topLoader.classList.add("visible");
            topLoader.style.opacity = Math.min(1, pullY / 20).toString();
            if (pullY >= PULL_THRESHOLD) topLoader.classList.add("ready");
            else topLoader.classList.remove("ready");
        } else if (pullY < 0) {
            bottomLoader.classList.add("visible");
            bottomLoader.style.opacity = Math.min(1, Math.abs(pullY) / 20).toString();
            if (Math.abs(pullY) >= PULL_THRESHOLD) bottomLoader.classList.add("ready");
            else bottomLoader.classList.remove("ready");
        } else {
            topLoader.classList.remove("visible", "ready");
            bottomLoader.classList.remove("visible", "ready");
            topLoader.style.opacity = "0";
            bottomLoader.style.opacity = "0";
        }
    };

    const getMaxScroll = () => Math.max(0, scrollEl.scrollHeight - container.clientHeight);

    const handleDelta = (delta: number) => {
        const maxScroll = getMaxScroll();

        // 1. Pulling Logic
        if (pullY !== 0 || (currentY <= 0 && delta < 0) || (currentY >= maxScroll && delta > 0)) {
            pullY -= delta * FRICTION;
            if (pullY > PULL_MAX) pullY = PULL_MAX;
            if (pullY < -PULL_MAX) pullY = -PULL_MAX;
            if ((pullY < 0 && currentY <= 0) || (pullY > 0 && currentY >= maxScroll)) pullY = 0;
        } 
        // 2. Normal Scrolling
        else {
            currentY += delta;
            if (currentY < 0) {
                const leftover = currentY;
                currentY = 0;
                pullY = -leftover * FRICTION;
            } else if (currentY > maxScroll) {
                const leftover = currentY - maxScroll;
                currentY = maxScroll;
                pullY = -leftover * FRICTION;
            }

            // [INFINITE SCROLL] Auto-load when near bottom
            if (!isChatLoading && chatHasMore && currentY >= maxScroll - 100) {
                loadMoreChat();
            }
        }
        updateTransform();
    };

    const resetPull = () => {
        pullY = 0;
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
        
        // Hold pull position
        pullY = dir === 'top' ? 40 : -40;
        updateTransform(true);

        if (dir === 'top') await fetchChatHistory(true);
        else await loadMoreChat();

        resetPull();
    };

    // 1. Wheel
    container.addEventListener('wheel', (e) => {
        if (isChatLoading) return;
        e.preventDefault();
        handleDelta(e.deltaY);

        if (pullTimer) clearTimeout(pullTimer);
        pullTimer = window.setTimeout(() => {
            if (Math.abs(pullY) >= PULL_THRESHOLD) triggerAction(pullY > 0 ? 'top' : 'bottom');
            else resetPull();
        }, 200);
    }, { passive: false });

    // 2. Touch
    let lastTouchY = 0;
    container.addEventListener('touchstart', (e) => {
        lastTouchY = e.touches[0].pageY;
        scrollEl.classList.remove("resetting");
    }, { passive: true });

    container.addEventListener('touchmove', (e) => {
        if (isChatLoading) return;
        const currentTouchY = e.touches[0].pageY;
        const delta = lastTouchY - currentTouchY;
        lastTouchY = currentTouchY;
        
        handleDelta(delta);
        e.preventDefault();
    }, { passive: false });

    container.addEventListener('touchend', () => {
        if (Math.abs(pullY) >= PULL_THRESHOLD) triggerAction(pullY > 0 ? 'top' : 'bottom');
        else resetPull();
    });
}

// Call init functions
const getDevicePref = () => forceCpuToggle.checked ? "cpu" : null;
const talksScroll = document.getElementById("chat-scroll");
if (talksScroll) {
    initChatPullLogic();
}
async function fetchChatHistory(reset: boolean = true) { 
    if (reset) { 
        chatPage = 0; 
        chatHasMore = true; 
        if (chatTalks) chatTalks.innerHTML = ""; 
    } 
    await loadMoreChat(); 
}

async function loadMoreChat() {
    if (isChatLoading || !chatHasMore) {
        stopSpinner();
        return;
    }

    // [FIX] Always start spinner for any chat fetching
    startSpinner();
    isChatLoading = true;

    try {
        let sqlFilter = null;
        if (activeContext.ref) sqlFilter = `ref = '${activeContext.ref}'`;
        else if (activeContext.bcc) sqlFilter = `bcc = '${activeContext.bcc}'`;
        else if (activeContext.cc) sqlFilter = `cc = '${activeContext.cc}'`;
        const messages = await invoke<any[]>("get_chat_messages", { limit: pageSize, offset: chatPage * pageSize, filter: sqlFilter });
        
        if (chatTalks) {
            if (messages && messages.length > 0) {
                if (messages.length < pageSize) chatHasMore = false;
                messages.sort((a, b) => a.created_at - b.created_at);
                messages.forEach(msg => renderMessage(msg, false)); chatPage++;
            } else { 
                chatHasMore = false; 
                if (chatPage === 0) {
                    chatTalks.innerHTML = "<div style='text-align:center; padding:20px; color:#999; font-size:0.75rem;'>No messages yet.</div>";
                    // [QR-RECOVERY] If no messages and not authenticated, show QR code immediately
                    if (!currentSession.email && currentTab === "settings") performQrAuth();
                }
            }
            // [QR-RECOVERY] Also trigger if we are on page 1 and no email exists
            if (!currentSession.email && currentTab === "settings" && chatPage <= 1) performQrAuth();
        }
    } catch (e) { 
        console.error(e); 
    } finally { 
        isChatLoading = false; 
        stopSpinner();
    }
}

// --- Initialize ---
initSession();
setWindowSize(false);
syncBrowserStatus();
initDevicePreference();