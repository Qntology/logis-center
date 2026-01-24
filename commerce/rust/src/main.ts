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

let selectedUuids = new Set<string>();
let currentDetailUuid: string | null = null;
let isExtracting = false; 
let spinnerInterval: number | null = null;
let systemLogCount = 0;

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
const globalNavSpinner = document.getElementById("global-nav-spinner") as HTMLElement;

const btnSpinnerAction = document.getElementById("btn-spinner-action") as HTMLButtonElement;

function startSpinner() {
    if (spinnerInterval) clearInterval(spinnerInterval);
    if (globalNavSpinner) {
        globalNavSpinner.style.display = "inline-block";
        globalNavSpinner.onclick = () => {
            openWidget("list");
            listView.style.display = "none";
            detailView.style.display = "flex";
            detailTitle.innerText = "Task Progress";
            if (btnStopTask) btnStopTask.style.display = "flex";
            if (btnDetailDelete) btnDetailDelete.style.display = "none";
        };
    }
    
    let i = 0;
    spinnerInterval = window.setInterval(() => {
        const char = spinnerFrames[i % spinnerFrames.length];
        if (globalNavSpinner) globalNavSpinner.innerText = char;
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
    if (globalNavSpinner) globalNavSpinner.style.display = "none";
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
        setWindowSize(true);
    }
    switchTab(tabName);
}

function collapseWidget() {
    isExpanded = false;
    contentPanel.classList.remove("open");
    setWindowSize(false);
    settingsBtn?.classList.remove("active-emoji", "active");
}

// Drag Logic
const pillNav = document.querySelector('.pill-nav') as HTMLElement;
if (pillNav) {
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
    if (currentImage) {
        btnExtract.style.display = "flex";
        btnExtract.title = "Extract from Image";
        return;
    }
    if (!currentDetectedUrl) {
        btnExtract.style.display = "none";
        return;
    }
    try {
        const urlObj = new URL(currentDetectedUrl.toLowerCase());
        const hostname = urlObj.hostname; 
        const link = (urlObj.pathname + urlObj.search).toLowerCase();
        const ccHash = await hashId(hostname); 
        const hashedRefId = await hashId(ccHash + link);
        const isActive = await invoke<boolean>("check_active_task", { 
            payload: { cc: ccHash, refId: hashedRefId } 
        });
        if (isActive === true) {
            btnExtract.style.display = "none";
        } else {
            btnExtract.style.display = "flex";
            btnExtract.title = `Extract from ${hostname}`;
        }
    } catch (e) { btnExtract.style.display = "flex"; }
}

listen("browser-match-found", async (event: any) => {
    const payload = event.payload;
    if (payload.is_client || payload.is_admin) {
        currentDetectedUrl = payload.url;
    } else {
        currentDetectedUrl = "";
    }
    await updateExtractButtonVisibility();
});

const handleSearchInteraction = () => {
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
    activeTags = activeTags.filter(t => t.id !== id);
    updateTagsUI();
    loadMoreDocs(true);
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
async function renderNavigation() {
    const pageList = document.getElementById("nav-list-pages");
    const userList = document.getElementById("nav-list-users");
    const profileName = document.getElementById("nav-profile-name");
    const profileFavicon = document.getElementById("nav-profile-favicon");
    const btnSignin = document.getElementById("nav-signin");
    const btnSignout = document.getElementById("nav-signout");

    if (!pageList || !userList) return;

    if (currentSession.email) {
        if (profileName) profileName.innerText = currentSession.email.split('@')[0];
        if (btnSignin) btnSignin.classList.add("hidden");
        if (btnSignout) btnSignout.classList.remove("hidden");
        if (profileFavicon && blockies) {
            const icon = blockies.create({ seed: currentSession.email, size: 8, scale: 4 });
            profileFavicon.innerHTML = ""; profileFavicon.appendChild(icon);
            icon.style.borderRadius = "4px"; icon.style.width = "100%"; icon.style.height = "100%";
        }
    }

    try {
        // 1. Pages Tree (Domain -> Type -> Page)
        let pages = await Select["pages"](); 
        
        // [FIX] Inject Current Page if not present
        if (currentDetectedUrl) {
            try {
                const urlObj = new URL(currentDetectedUrl);
                const domain = urlObj.hostname;
                const link = urlObj.pathname + urlObj.search;
                const origin = urlObj.origin;
                
                // Check if already exists to avoid duplicate
                const exists = pages.some(p => {
                    const d = p.data || p;
                    return (d.link === link && (d.origin || "").includes(domain));
                });

                if (!exists) {
                    console.log("[NAV] Injecting current page:", link);
                    pages.unshift({
                        id: "current-page",
                        type: "tracking", // Default to tracking/active type
                        data: {
                            origin: origin,
                            link: link,
                            title: "Current Page",
                            item: true // Treat as list item for icon
                        }
                    });
                }
            } catch (e) { console.error("Current URL parse error:", e); }
        }

        pageList.innerHTML = "";
        
        if (pages.length === 0) {
            pageList.innerHTML = "<div style='color:#999; padding:10px; font-size:0.75rem;'>No shared pages found.</div>";
        } else {
            // Grouping Logic
            const tree: Record<string, Record<string, any[]>> = {};
            pages.forEach(p => {
                let data = p.data || p; // Normalize
                const domain = (data.origin || "unknown").replace(/^https?:\/\//, "");
                const type = (data.type || "general").toUpperCase();
                const isDetail = data.detail === true;
                
                if (!tree[domain]) tree[domain] = {};
                if (!tree[domain][type]) tree[domain][type] = [];
                
                tree[domain][type].push({ 
                    uuid: p.id, 
                    link: data.link || "/", 
                    // [FIX] Fallback title logic: text -> title -> link
                    text: data.detail ? "Detail" : "List", 
                    // [FIX] Detail logic: Explicit detail flag OR missing item selector (List has item selector)
                    isDetail: data.detail === true || !data.item,
                    domain,
                    type,
                    active: currentDetectedUrl.includes(data.link)
                });
            });

            // HTML Generation
            for (const [domain, types] of Object.entries(tree)) {
                const branchDiv = document.createElement("div");
                branchDiv.className = "logis-branch";
                
                let html = `
                    <div class="logis-parent">
                        <div class="logis-favicon"></div> <!-- Placeholder for favicon -->
                        <strong>${domain}</strong>
                    </div>
                `;

                for (const [type, items] of Object.entries(types)) {
                    html += `<div class="logis-children">
                        <div class="logis-type-header"># ${type}</div>`;
                    
                    items.forEach(it => {
                        const icon = it.isDetail ? '◈' : '☰';
                        const activeClass = it.active ? 'active' : '';
                        const childClass = it.isDetail ? 'child' : '';
                        // [FIX] Ensure text content is safe and visible
                        const displayText = it.text || it.link; 
                        
                        html += `
                            <a href="#" class="logis-page ${activeClass} ${childClass}" 
                               data-id="${it.uuid}" data-domain="${it.domain}" data-type="${it.type}" data-mode="${it.isDetail ? 'Detail' : 'List'}">
                                <span class="icon">${icon}</span>
                                <span class="text" title="${displayText}">${displayText}</span>
                            </a>
                        `;
                    });
                    html += `</div>`;
                }
                branchDiv.innerHTML = html;
                
                // Bind Clicks
                branchDiv.querySelectorAll(".logis-page").forEach((link: any) => {
                    link.onclick = (e: Event) => {
                        e.preventDefault();
                        e.stopPropagation();
                        const ds = link.dataset;
                        
                        // [UX] Log navigation action to chat history
                        renderMessage({
                            id: `nav-${Date.now()}`,
                            role: "system_task",
                            content: `Navigated to: ${ds.domain} - ${ds.type}`,
                            status: 9, // Done
                            created_at: Date.now()
                        });

                        addSearchTag(`@${ds.domain}`, 'domain', ds.domain);
                        addSearchTag(`#${ds.type}`, 'type', ds.type);
                        addSearchTag(`[${ds.mode}]`, 'mode', ds.mode);
                        hideNavigation();
                    };
                });
                
                // Add Favicon (Async)
                const faviconUrl = `https://${domain}/favicon.ico`;
                const favEl = branchDiv.querySelector(".logis-favicon") as HTMLElement;
                if(favEl) favEl.style.backgroundImage = `url(${faviconUrl})`;

                pageList.appendChild(branchDiv);
            }
        }

        // 2. Users Tree (Team -> Members)
        const users = await Select["users"]();
        userList.innerHTML = "";
        
        if (users.length === 0) {
            userList.innerHTML = "<div style='color:#999; padding:10px; font-size:0.75rem;'>No team members.</div>";
        } else {
            // Group by Team
            const teamMap: Record<string, any> = {};
            const membersMap: Record<string, any[]> = {};

            users.forEach(u => {
                let data = u.data || u;
                if (data.type === "team") {
                    teamMap[u.id] = data;
                } else {
                    const teamId = u.to || "unknown";
                    if (!membersMap[teamId]) membersMap[teamId] = [];
                    membersMap[teamId].push(data);
                }
            });

            // If no explicit teams found, group under 'Unknown' or inferred
            for (const [teamId, members] of Object.entries(membersMap)) {
                const teamName = teamMap[teamId]?.name || "Team " + teamId.slice(0,6);
                
                const branchDiv = document.createElement("div");
                branchDiv.className = "logis-branch";
                
                let html = `
                    <div class="logis-parent">
                        <strong>${teamName}</strong>
                    </div>
                    <div class="logis-children">
                `;
                
                members.forEach(m => {
                    html += `
                        <div class="logis-page" style="cursor:default;">
                            <span class="icon">👤</span>
                            <span class="text">${m.name || m.id.slice(0,8)}</span>
                        </div>
                    `;
                });
                
                html += `</div>`;
                branchDiv.innerHTML = html;
                userList.appendChild(branchDiv);
            }
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

        if (response.results && Array.isArray(response.results)) {
            await Upsert["items"](response.results);
            console.log("[SYNC] Data upserted.");
            
            // [REACTIVE] Re-render navigation immediately after sync
            await renderNavigation();
            // Also refresh list if on list tab
            if (currentTab === "list") await loadMoreDocs(true);
        }
        
    } catch (e) { console.error("[SYNC] Failed:", e); }
}

// ... (List Logic) ...

// [NEW] Global Navigation Link Handler (from item2html)
document.addEventListener('nav-link', async (e: any) => {
    const targetLink = e.detail;
    console.log("[NAV] Internal Link Clicked:", targetLink);
    
    // Add chip for the specific path to filter items
    addSearchTag(targetLink, 'path', targetLink);
    
    // Move to List View
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
            const response = await invoke<any>("ai_search_complex", { query: query, language: "korean" });
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
    let pathname = "/";
    let cc = "";
    if (currentDetectedUrl) {
        try {
            const normalizedUrl = currentDetectedUrl.toLowerCase();
            const urlObj = new URL(normalizedUrl);
            const hostname = urlObj.hostname;
            const link = (urlObj.pathname + urlObj.search).toLowerCase();
            
            const ccHash = await hashId(hostname); 
            const hashedRefId = await hashId(ccHash + link);
            
            const isActive = await invoke<boolean>("check_active_task", { 
                payload: { cc: ccHash, refId: hashedRefId } 
            });
            if (isActive) {
                alert("This page is already in the queue or being processed.");
                openWidget("settings");
                await updateExtractButtonVisibility();
                return;
            }
        } catch(e) { console.error("[WIDGET] Pre-click check error:", e); }
    }

    btnExtract.style.opacity = "0.5";
    const logArea = document.getElementById("extraction-log");
    if (logArea) logArea.innerHTML = "";
    setTimeout(() => { if (btnExtract) btnExtract.style.opacity = "1"; }, 300);

    openWidget("settings");
    startSpinner();
    
    if (currentImage) {
        isExtracting = true;
        const taskId = `img_${Date.now()}`;
        try {
            await emit("new-task-from-browser", { id: taskId, type: "image_extraction", image_path: currentImage, ref_id: currentImage, link: "Local Image" });
            renderMessage({ id: taskId, role: "system_task", content: `Queued: Image Analysis`, status: 10, task_id: taskId, created_at: Date.now() });
            await updateExtractButtonVisibility();
        } catch (e) { isExtracting = false; await updateExtractButtonVisibility(); }
    } else {
        isExtracting = true;
        let taskId = `task_${Date.now()}`;
        try {
            const html = await invoke<string>("extract_html_from_current_tab");
            const normalizedUrl = currentDetectedUrl.toLowerCase();
            const urlObj = new URL(normalizedUrl);
            const cc = await hashId(urlObj.hostname);
            const rawPath = urlObj.pathname + urlObj.search; // [STRICT] Real path for link
            const hashedRefId = await hashId(cc + rawPath.toLowerCase());
            
            await emit("new-task-from-browser", { 
                id: taskId, 
                type: "html_extraction", 
                html: html, 
                link: rawPath, // [FIX] Send raw path
                cc: cc, 
                ref_id: hashedRefId, // [FIX] Send hash as ref_id
                from: currentSession.address,
                to: currentSession.team
            });
            renderMessage({ id: taskId, role: "system_task", content: `Queued: ${urlObj.hostname}`, status: 10, task_id: hashedRefId, created_at: Date.now() });
            await updateExtractButtonVisibility();
        } catch (e) { isExtracting = false; await updateExtractButtonVisibility(); }
    }
});

listen("extraction-progress", async (event: any) => { renderProgressToUI(event.payload); });
document.addEventListener('render-progress', (e: any) => { renderProgressToUI(e.detail); });

async function renderProgressToUI(payload: any) {
    const catId = payload.category ? payload.category.replace(/[^a-zA-Z0-9]/g, "") : "general";
    const elementId = `progress-${catId}`;
    const extractionLog = document.getElementById("extraction-log");
    
    if (extractionLog && extractionLog.dataset.activeTaskId) {
        if (payload.task_id && payload.task_id !== extractionLog.dataset.activeTaskId) {
            return;
        }
    }

    if (payload.category === "Done" || payload.category === "Error") {
        stopSpinner();
        isExtracting = false;
        if (btnExtract) setTimeout(() => { btnExtract.innerText = "⚡"; }, 2000);
        await updateExtractButtonVisibility();
    }

    if (payload.task_id) {
        let statusCode = 1; 
        if (payload.category === "Done") statusCode = 9;
        else if (payload.category === "Error") statusCode = 6;
        else if (payload.summary?.toLowerCase().includes("cancelled")) statusCode = 3;
        renderMessage({ id: payload.task_id, role: "system_task", content: payload.summary, status: statusCode, created_at: Date.now() });
    }

    if (extractionLog && detailView.style.display !== "none") {
         let p = document.getElementById(elementId);
         if (!p) {
             p = document.createElement("div"); p.id = elementId;
             p.style.borderBottom = "1px solid #eee"; p.style.padding = "6px 0"; p.style.fontSize = "0.75rem";
             p.style.display = "flex"; p.style.flexDirection = "column"; 
             const row = document.createElement("div"); row.className = "progress-row"; row.style.display = "flex"; row.style.alignItems = "center";
             row.innerHTML = `<span class="active-spinner" style="color:var(--primary); margin-right:8px; font-family:monospace; min-width:15px;">⠋</span><span class="summary-text">${payload.summary || ""}</span>`;
             p.appendChild(row);
             const results = document.createElement("div"); results.className = "results-container"; p.appendChild(results);
             extractionLog.appendChild(p);
         }
         
         const summary = p.querySelector(".summary-text") as HTMLElement;
         const spinner = p.querySelector(".active-spinner") as HTMLElement;
         const resultsContainer = p.querySelector(".results-container");

         if (payload.category === "Done") {
             isExtracting = false;
             if (btnStopTask) btnStopTask.style.display = "none";
             if (btnDetailDelete) btnDetailDelete.style.display = "flex";
             extractionLog.querySelectorAll(".progress-row").forEach(row => {
                 const s = row.querySelector(".active-spinner");
                 if (s) {
                     s.classList.remove("active-spinner");
                     row.innerHTML = `<span style="margin-right:8px; color:#4ade80;">✅</span> <span>${row.querySelector(".summary-text")?.textContent || "Complete"}</span>`;
                 }
             });
         } else if (payload.category === "Error") {
             isExtracting = false;
             const row = p.querySelector(".progress-row");
             if (row) { 
                 const s = row.querySelector(".active-spinner");
                 if (s) s.classList.remove("active-spinner");
                 row.innerHTML = `<span style="margin-right:8px;">❌</span> <span>${payload.summary || "Error"}</span>`; 
                 (row as HTMLElement).style.color = "#ef4444"; 
             }
         } else {
             if (spinner) {
                 spinner.innerText = payload.spinner || "⠋";
                 if (payload.spinner === "✅" || payload.spinner === "✔") {
                     spinner.classList.remove("active-spinner");
                     spinner.style.color = "#4ade80";
                 }
             }
             if (summary) summary.innerText = payload.summary || "";
             if((payload.display_text || payload.data) && resultsContainer) {
                  const pre = document.createElement("pre");
                  pre.style.whiteSpace = "pre-wrap"; pre.style.fontSize = "0.7rem"; pre.style.color = "#aaa"; pre.style.background = "#252525"; pre.style.padding = "5px"; pre.style.borderRadius = "3px"; pre.style.marginTop = "5px"; pre.style.borderLeft = "2px solid var(--primary)";
                  pre.innerText = payload.display_text || JSON.stringify(payload.data, null, 2);
                  resultsContainer.appendChild(pre);
                  detailContent.scrollTop = detailContent.scrollHeight;
             }
         }
    }
}

btnStopTask?.addEventListener("click", async () => {
    if (await ask("Stop the current extraction?", { title: "Stop Task", kind: "warning" })) {
        try {
            await invoke<string>("stop_current_extraction");
            isExtracting = false; stopSpinner();
            if (btnStopTask) btnStopTask.style.display = "none";
            detailTitle.innerText = "Stopped";
            detailContent.innerHTML = "<div style='color:#ef4444; padding:20px;'>Extraction stopped by user.</div>";
            await updateExtractButtonVisibility();
        } catch (e) { console.error("Stop failed:", e); }
    }
});

// --- Browser Auto ---
btnAutoLaunch?.addEventListener("click", async () => { try { await invoke("launch_best_browser", { url: "https://google.com" }); } catch (e) { console.error("Launch error:", e); } });
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
    if (!currentDetailUuid) return;
    if (await ask("Delete this document?", { title: "Confirm Delete", kind: "warning" })) {
        try { await invoke("delete_document", { uuid: currentDetailUuid }); detailView.style.display = "none"; listView.style.display = "block"; refreshList(); } catch (e) { console.error(e); }
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

    if (isLoading || !hasMore) return;
    isLoading = true;
    if (loadingIndicator) loadingIndicator.style.display = "block";
    
    // Construct query
    let queryParts = activeTags.map(t => {
        if (t.type === 'domain') return `host:${t.value}`;
        if (t.type === 'type') return `type:${t.value.toLowerCase()}`;
        if (t.type === 'mode') return `mode:${t.value.toLowerCase()}`;
        return t.value;
    });
    
    const textInput = searchInput?.value.toLowerCase() || "";
    if (textInput) queryParts.push(textInput);
    const finalQuery = queryParts.join(" ");
    
    try {
        let docs: any[] = [];
        
        // Final Query string from tags + text
        const textInput = searchInput?.value.toLowerCase() || "";
        let queryParts = activeTags.map(t => {
            if (t.type === 'domain') return `host:${t.value}`;
            if (t.type === 'type') return `type:${t.value.toLowerCase()}`;
            return t.value;
        });
        if (textInput) queryParts.push(textInput);
        const finalQuery = queryParts.join(" ");

        // [STRICT] If we have structured tags, the DB shim will use SQL filter.
        // If it's pure text, it will use FTS search.
        docs = await Select["items"]({ 
            value: finalQuery, 
            limit: pageSize, 
            offset: currentPage * pageSize 
        });

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
        if (currentPage === 0 && docListContainer) {
            docListContainer.innerHTML = `<div style='text-align:center; padding:20px; color:#ef4444;'>Error loading data.</div>`;
        }
    } 
    finally { 
        isLoading = false; 
        if (loadingIndicator) loadingIndicator.style.display = "none"; 
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
                if (target.closest('.toggle-more') || target.closest('.more-label')) return;
                if (!target.closest('a') && !target.closest('input')) {
                    showDetail(doc.id);
                }
            });
        }
    });
}

// ... (showDetail, etc.)

async function showDetail(uuid: string) {
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
        }
    } catch (e) { detailContent.innerHTML = "Failed: " + e; }
}

btnDetailBack?.addEventListener("click", () => { detailView.style.display = "none"; listView.style.display = "block"; });
document.getElementById("btn-settings-back")?.addEventListener("click", collapseWidget);
btnListBack?.addEventListener("click", collapseWidget);

// [NEW] Tree Profile Actions
document.getElementById("nav-signin")?.addEventListener("click", () => openWidget("settings"));
document.getElementById("nav-signout")?.addEventListener("click", () => { document.getElementById("btn-logout")?.click(); });

// Image Logic
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

// Auth & Chat
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
        const qrAuthSpinner = document.getElementById("qr-auth-spinner");
        if (qrAuthSpinner) { let idx = 0; qrAuthSpinner.innerText = spinnerFrames[idx++ % spinnerFrames.length]; }
        let session = data.session || data; 
        if (session && session.hash) {
            const hashChanged = session.hash !== currentSession.hash;
            currentSession = { ...currentSession, ...session };
            saveSession();
            if (hashChanged && !currentSession.email && currentTab === "settings") performQrAuth();
            if (currentSession.email) { 
                await invoke("initialize_hub", { address: currentSession.address, email: currentSession.email, flag: session.flag || "kr" }); 
                updateAuthUI(); 
                fetchChatHistory();
                syncData(); // [NEW] Sync data after auth
            }
        }
    } catch (e) { console.warn("Auth check failed:", e); }
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
    const html = `<div class="chat-talk system" id="msg-qr-auth"><div class="chat-message" style="padding:0; background: #fff; color: #000; border:0;"><div style="font-size:0.75rem; font-weight: bold; margin-bottom: 15px; color: #333;"><span id="qr-auth-spinner" style="margin-right:5px; font-family:monospace; color:var(--primary); font-weight:bold;">⠋</span>Scan the QR code</div><div id="qr-code-target" style="display: inline-block; background: #fff; border-radius: 8px;"></div></div></div>`;
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

function renderMessage(msg: any) {

    if (!chatTalks) return;

    

    const msgId = `msg-${msg.id}`;

    let existing = document.getElementById(msgId);

    

    const isSystemTask = msg.role === "system_task";

    const roleClass = msg.role === "user" ? "user" : "system";



    // [NEW] Update Preprocessing Log Count

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
    if (itemData) {
        displayHtml = typeof itemData === 'string' ? itemData : (itemData.text || itemData.title || JSON.stringify(itemData)); 
    }
    const html = `<div class="chat-talk ${roleClass} ${isSystemTask ? 'task-bubble' : ''}" id="${msgId}" data-task-id="${msg.task_id || msg.id}" data-status="${msg.status}" style="cursor: ${isSystemTask ? 'pointer' : 'default'};"><div class="chat-message"><div style="font-size:0.6rem; opacity:0.5; margin-bottom:4px; display:flex; justify-content:space-between;"><span>${msg.role === 'user' ? '@YOU' : '🤖 LOGIS AI'}</span><span>${timeStr}</span></div><div class="content">${displayHtml}</div>${isSystemTask ? `<div class="status-bar" style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.1); font-size:0.65rem; font-weight:bold; color:${currentStatus.color};">${currentStatus.icon} ${currentStatus.text.toUpperCase()}</div>` : ""}</div></div>`;
    if (existing) existing.outerHTML = html;
    else chatTalks.insertAdjacentHTML('beforeend', html);
}

function saveSession() { localStorage.setItem("chat_session", JSON.stringify(currentSession)); }
async function initSession() {
    const saved = localStorage.getItem("chat_session");
    if (saved) { try { currentSession = { ...currentSession, ...JSON.parse(saved) }; } catch (e) {} } 
    else { const legacy = localStorage.getItem("device_hash"); if (legacy) currentSession.hash = legacy; }
    if (!currentSession.hash && ethers) { const w = ethers.Wallet.createRandom(); currentSession.hash = w.address.toLowerCase().replace("0x", ""); saveSession(); }
    saveSession();
    currentSession.address = currentSession.address || ZERO_ADDRESS;
    currentSession.team = currentSession.team || await hashId(ZERO_ADDRESS);
    updateAuthUI(); startPolling();
}

document.getElementById("btn-qr-auth")?.addEventListener("click", performQrAuth);
document.getElementById("btn-logout")?.addEventListener("click", async () => { if (await ask("Are you sure?", { title: "Sign Out", kind: "warning" })) { currentSession.email = undefined; updateAuthUI(); } });

listScrollContainer?.addEventListener("scroll", () => { if (listScrollContainer.scrollTop + listScrollContainer.clientHeight >= listScrollContainer.scrollHeight - 20) loadMoreDocs(); });
settingsBtn?.addEventListener("click", () => { if (currentTab === "settings" && isExpanded) collapseWidget(); else openWidget("settings"); });
document.getElementById("nav-to-auto")?.addEventListener("click", () => switchTab("automation"));
document.getElementById("unload-btn")?.addEventListener("click", async () => { try { await invoke("unload_model"); alert("Memory cleared."); } catch (e) {} });

async function syncBrowserStatus() { try { const s = await invoke<string>("get_browser_status"); if (btnAutoLaunch) btnAutoLaunch.style.display = (s === "running") ? "none" : "flex"; } catch (e) {} }
initSession(); setWindowSize(false); syncBrowserStatus();

const talksScroll = document.getElementById("chat-scroll");
if (talksScroll) {
    talksScroll.addEventListener('wheel', (e: WheelEvent) => { talksScroll.scrollTop -= e.deltaY; e.preventDefault(); }, { passive: false });
    talksScroll.addEventListener('scroll', () => { if (talksScroll.scrollTop + talksScroll.clientHeight >= talksScroll.scrollHeight - 20) loadMoreChat(); });
}

async function fetchChatHistory(reset: boolean = true) {
    if (reset) { chatPage = 0; chatHasMore = true; if (chatTalks) chatTalks.innerHTML = ""; }
    await loadMoreChat();
}

async function loadMoreChat() {
    if (isChatLoading || !chatHasMore) return;
    isChatLoading = true;
    try {
        const messages = await invoke<any[]>("get_chat_messages", { limit: pageSize, offset: chatPage * pageSize });
        if (chatTalks) {
            if (messages && messages.length > 0) {
                if (messages.length < pageSize) chatHasMore = false;
                messages.sort((a, b) => a.created_at - b.created_at);
                messages.forEach(msg => renderMessage(msg));
                chatPage++;
            } else { chatHasMore = false; if (chatPage === 0) chatTalks.innerHTML = "<div style='text-align:center; padding:20px; color:#999; font-size:0.75rem;'>No messages yet.</div>"; }
            if (!currentSession.email && currentTab === "settings" && chatPage === 1) performQrAuth();
        }
    } catch (e) { console.error(e); } finally { isChatLoading = false; }
}
