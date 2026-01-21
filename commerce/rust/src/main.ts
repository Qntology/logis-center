console.log("%c[WIDGET] MAIN.TS LOADED", "color: #00ff00; font-weight: bold; font-size: 1.2rem;");
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { open, ask } from '@tauri-apps/plugin-dialog';
import { listen, emit } from '@tauri-apps/api/event';
import { readFile } from '@tauri-apps/plugin-fs';

// Access global libs
const ethers = (window as any).ethers;
const QRCode = (window as any).QRCode;
const marked = (window as any).marked;
const pako = (window as any).pako;
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

// List State
let cachedDocs: any[] = [];
let currentPage = 0;
const pageSize = 10;
let isLoading = false;
let hasMore = true;
let selectedUuids = new Set<string>();
let currentDetailUuid: string | null = null;
let isExtracting = false; 
let spinnerInterval: number | null = null;

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

const docTableBody = document.getElementById("doc-tbody") as HTMLElement;
const listRefreshBtn = document.getElementById("list-refresh-btn") as HTMLButtonElement;
const btnDeleteSelected = document.getElementById("btn-delete-selected") as HTMLButtonElement;
const selectAllCheckbox = document.getElementById("select-all-checkbox") as HTMLInputElement;
const listScrollContainer = document.getElementById("list-scroll-container") as HTMLElement;
const loadingIndicator = document.getElementById("loading-indicator") as HTMLElement;

const aiResultsArea = document.getElementById("ai-search-results") as HTMLElement;
const aiResultsTitle = document.getElementById("ai-results-title") as HTMLElement;
const aiResultsContent = document.getElementById("ai-results-content") as HTMLElement;

const chatTalks = document.querySelector('.chat-talks') as HTMLElement;
const chatForm = document.querySelector('form[name="chat-form"]') as HTMLFormElement;
const chatInput = chatForm?.querySelector('input[name="talk"]') as HTMLInputElement;

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
    if (btnSpinnerAction) {
        btnSpinnerAction.style.display = "inline-block";
        btnSpinnerAction.onclick = () => {
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
        
        // 1. Animate Global Nav Spinner
        if (globalNavSpinner) globalNavSpinner.innerText = char;
        
        // 2. Animate All Active Log/Chat Spinners
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
    if (btnSpinnerAction) btnSpinnerAction.style.display = "none";
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
        if (!currentSession.email) performQrAuth();
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

// Drag Logic (Manual)
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

// --- 2. Search & Main Nav ---
async function updateExtractButtonVisibility() {
    if (!btnExtract) return;

    // If an image is selected, we always show the button for image extraction
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
        
        // Always query backend for tasks related to THIS specific URL
        const isActive = await invoke<boolean>("check_active_task", { 
            payload: { cc: ccHash, refId: hashedRefId } 
        });
        
        console.log(`[WIDGET] URL Check: cc=${ccHash}, ref_id=${hashedRefId}, isActive=${isActive}`);

        if (isActive === true) {
            // [STRICT] If a task is pending or in progress for this URL, hide the button
            btnExtract.style.display = "none";
        } else {
            btnExtract.style.display = "flex";
            btnExtract.title = `Extract from ${hostname}`;
        }
    } catch (e) {
        console.error("[WIDGET] Visibility check failed:", e);
        // Fallback to showing button if we can't verify, to avoid getting stuck
        btnExtract.style.display = "flex";
    }
}

// Update the listener to be more responsive
listen("browser-match-found", async (event: any) => {
    const payload = event.payload;
    if (payload.is_client || payload.is_admin) {
        currentDetectedUrl = payload.url;
    } else {
        currentDetectedUrl = "";
    }
    // Immediately check if we should show or hide the ⚡ button for this new tab
    await updateExtractButtonVisibility();
});

const handleSearchInteraction = () => {
    openWidget("list");
    
    const navOverlay = document.getElementById("nav-categories");
    if (navOverlay) {
        navOverlay.classList.remove("hidden");
        // [FIX] Add 'visible' class to trigger opacity and pointer-events
        navOverlay.classList.add("visible");
        renderNavigation();
        
        // [FIX] Scroll to the very top to make sure Tree/Accordion is visible
        if (listScrollContainer) {
            listScrollContainer.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }

    // [STRICT] Before search, the results area should be empty.
    // Only show navigation tree (users/pages).
    if (!searchInput.value) {
        if (docTableBody) docTableBody.innerHTML = "";
        cachedDocs = [];
        currentPage = 0;
        hasMore = true;
    }
};

searchInput?.addEventListener("focus", handleSearchInteraction);
searchInput?.addEventListener("click", handleSearchInteraction);

// Helper to hide navigation when clicking away or selecting something
function hideNavigation() {
    const navOverlay = document.getElementById("nav-categories");
    if (navOverlay) {
        navOverlay.classList.add("hidden");
        navOverlay.classList.remove("visible");
    }
}

async function renderNavigation() {
    const pageList = document.getElementById("nav-list-pages");
    const userList = document.getElementById("nav-list-users");
    const profileName = document.getElementById("nav-profile-name");
    const profileFavicon = document.getElementById("nav-profile-favicon");
    const btnSignin = document.getElementById("nav-signin");
    const btnSignout = document.getElementById("nav-signout");

    if (!pageList || !userList) return;

    // 0. Update Profile Section
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
        const pages = await invoke<any[]>("get_known_pages");
        pageList.innerHTML = "";
        
        if (pages.length === 0) {
            pageList.innerHTML = "<div style='color:#999; padding:10px; font-size:0.75rem;'>No shared pages found.</div>";
        } else {
            // [STRICT PARITY] Grouping: Domain -> Type -> (List or Detail)
            const tree: Record<string, any> = {};
            
            pages.forEach(p => {
                let data: any = {};
                try { data = JSON.parse(p.json_data); } catch(e) { data = { origin: "unknown", type: p.doc_type }; }
                
                const domain = (data.origin || "unknown").replace(/^https?:\/\//, "");
                const type = (p.doc_type || "general").toUpperCase();
                const isDetail = data.detail === true; // From before_server schema
                
                if (!tree[domain]) tree[domain] = {};
                if (!tree[domain][type]) tree[domain][type] = { lists: [], details: [] };
                
                const item = { uuid: p.uuid, link: data.link || "/", text: data.text || "Untitled" };
                if (isDetail) tree[domain][type].details.push(item);
                else tree[domain][type].lists.push(item);
            });

            for (const [domain, types] of Object.entries(tree)) {
                const domainDiv = document.createElement("div");
                domainDiv.className = "nav-domain-group";
                domainDiv.style.marginBottom = "10px";
                domainDiv.innerHTML = `<div style="font-size:0.65rem; color:#999; padding:0 5px 5px; font-weight:bold; border-bottom:1px solid #eee; margin-bottom:5px;">${domain}</div>`;
                
                for (const [type, categories] of Object.entries<any>(types)) {
                    const typeSection = document.createElement("div");
                    typeSection.className = "nav-accordion";
                    
                    // Combined content for the accordion: List items then Detail items
                    const buildItemsHtml = (items: any[], label: string, color: string) => {
                        if (items.length === 0) return "";
                        return `
                            <div style="font-size:0.6rem; color:${color}; padding:4px 10px; text-transform:uppercase; letter-spacing:1px; opacity:0.6;">${label}</div>
                            ${items.map(it => `
                                <div class="nav-link" data-id="${it.uuid}" style="padding:6px 20px; font-size:0.75rem; color:#555; cursor:pointer; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${it.link}">
                                    • ${it.link}
                                </div>
                            `).join("")}
                        `;
                    };

                    typeSection.innerHTML = `
                        <div class="nav-header" style="display:flex; justify-content:space-between; padding:8px 10px; background:#fafafa; border-radius:6px; cursor:pointer; border:1px solid #f0f0f0; margin-bottom:2px;">
                            <span style="font-size:0.7rem; font-weight:bold; color:var(--primary);">${type}</span>
                            <span class="arrow" style="font-size:0.6rem; color:#ccc;">▼</span>
                        </div>
                        <div class="nav-content hidden" style="padding:5px 0;">
                            ${buildItemsHtml(categories.lists, "Lists", "#2563eb")}
                            ${buildItemsHtml(categories.details, "Details", "#db2777")}
                        </div>
                    `;
                    
                    const header = typeSection.querySelector(".nav-header") as HTMLElement;
                    const content = typeSection.querySelector(".nav-content") as HTMLElement;
                    header.onclick = () => {
                        content.classList.toggle("hidden");
                        header.querySelector(".arrow")!.innerHTML = content.classList.contains("hidden") ? "▼" : "▲";
                    };
                    
                    typeSection.querySelectorAll(".nav-link").forEach((link: any) => {
                        link.onclick = (e: Event) => {
                            e.stopPropagation();
                            hideNavigation();
                            // Logic parity: If List item, show list. If Detail item, show detail.
                            const uuid = (e.currentTarget as HTMLElement).dataset.id;
                            if (uuid) {
                                // If it's a detail link, we could potentially go straight to showDetail
                                // but for search consistency, we'll filter first.
                                searchInput.value = `type:${type.toLowerCase()} path:${link.innerText.replace("• ", "").trim()}`;
                                btnSubmit.click();
                            }
                        };
                    });
                    domainDiv.appendChild(typeSection);
                }
                pageList.appendChild(domainDiv);
            }
        }

        // 2. Render Users (Team Members)
        const users = await invoke<any[]>("get_known_users");
        userList.innerHTML = "";
        if (users.length > 0) {
            users.forEach(u => {
                if (u.doc_type === "team") return;
                const item = document.createElement("div");
                item.className = "member-item";
                item.style.display = "flex"; item.style.alignItems = "center"; item.style.gap = "10px";
                item.style.padding = "8px"; item.style.borderRadius = "8px"; item.style.cursor = "pointer";
                const icon = blockies.create({ seed: u.uuid, size: 8, scale: 3 });
                icon.style.borderRadius = "50%";
                item.innerHTML = `<div class="avatar" style="width:24px; height:24px;"></div><div style="font-size:0.75rem; color:#333; font-weight:500;">${u.uuid.slice(0,10)}...</div>`;
                item.querySelector(".avatar")!.appendChild(icon);
                item.onclick = () => { hideNavigation(); openWidget("settings"); };
                userList.appendChild(item);
            });
        }
    } catch (e) { console.error("Nav error:", e); }
}

searchInput?.addEventListener("input", () => {
    if(searchDebounceTimer) clearTimeout(searchDebounceTimer);
    searchDebounceTimer = window.setTimeout(async () => {
        const keyword = searchInput.value.toLowerCase();
        if (!keyword) { refreshList(); return; }
        try {
            const results = await invoke<[string, string, number][]>("search_documents", { query: keyword });
            if (docTableBody) {
                docTableBody.innerHTML = "";
                if (results.length === 0) {
                    docTableBody.innerHTML = "<tr><td colspan='4' style='text-align:center; padding:20px;'>No AI matches.</td></tr>";
                } else {
                    results.forEach(([id, text, score]) => {
                        const tr = document.createElement("tr");
                        tr.style.cursor = "pointer";
                        tr.innerHTML = `<td style='text-align:center;'>✨</td><td>Result</td><td title="${text}">${id.slice(0,8)}...</td><td>${score.toFixed(2)}</td>`;
                        tr.addEventListener("click", () => showDetail(id));
                        docTableBody.appendChild(tr);
                    });
                }
            }
        } catch (e) { console.error("Filter error:", e); }
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

document.addEventListener('view-task-log', () => {
    openWidget("list");
    listView.style.display = "none";
    detailView.style.display = "flex";
});

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
        } catch(e) {
            console.error("[WIDGET] Pre-click check error:", e);
        }
    }

    // [NEW] Visual feedback: pulse the button
    btnExtract.style.opacity = "0.5";
    
    // [NEW] Clear logs
    const logArea = document.getElementById("extraction-log");
    if (logArea) logArea.innerHTML = "";

    setTimeout(() => { if (btnExtract) btnExtract.style.opacity = "1"; }, 300);

    // Initial View setup: Go to Settings (Chat) to see history
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
            const link = urlObj.pathname + urlObj.search;
            const hashedRefId = await hashId(cc + link);
            
            await emit("new-task-from-browser", { 
                id: taskId, 
                type: "html_extraction", 
                html: html, 
                link: currentDetectedUrl, 
                cc: cc, 
                ref_id: hashedRefId,
                from: currentSession.address,
                to: currentSession.team
            });
            // Task ID and hashedRefId are both important: taskId for Task, hashedRefId for Result Document
            renderMessage({ id: taskId, role: "system_task", content: `Queued: ${urlObj.hostname}`, status: 10, task_id: hashedRefId, created_at: Date.now() });
            
            await updateExtractButtonVisibility();
        } catch (e) { isExtracting = false; await updateExtractButtonVisibility(); }
    }
});

listen("extraction-progress", async (event: any) => {
    const payload = event.payload;
    const catId = payload.category ? payload.category.replace(/[^a-zA-Z0-9]/g, "") : "general";
    const elementId = `progress-${catId}`;
    const extractionLog = document.getElementById("extraction-log");
    
    if (payload.category === "Done" || payload.category === "Error") {
        stopSpinner();
        isExtracting = false;
        if (btnExtract) setTimeout(() => { btnExtract.innerText = "⚡"; }, 2000);
        // [NEW] Refresh button visibility once task is finished
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
         // [FIXED] Transition ONLY previous categories to 'Done'
         if (payload.category !== "Done" && payload.category !== "Error") {
             const allRows = extractionLog.querySelectorAll(".progress-row");
             allRows.forEach(row => {
                 const container = row.closest('[id^="progress-"]');
                 const spinner = row.querySelector(".active-spinner");
                 // If this is an old category container, mark it as done
                 if (container && container.id !== elementId && spinner) {
                     row.innerHTML = `<span style="margin-right:8px; color:#4ade80;">✅</span> <span style="color:#888;">${row.querySelector(".summary-text")?.textContent || ""}</span>`;
                 }
             });
         }

         let p = document.getElementById(elementId);
         if (!p) {
             p = document.createElement("div"); p.id = elementId;
             p.style.borderBottom = "1px solid #333"; p.style.padding = "6px 0"; p.style.fontSize = "0.75rem";
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
                 if (s) row.innerHTML = `<span style="margin-right:8px; color:#4ade80;">✅</span> <span>${row.querySelector(".summary-text")?.textContent || "Complete"}</span>`;
             });
         } else if (payload.category === "Error") {
             isExtracting = false;
             const row = p.querySelector(".progress-row");
             if (row) { row.innerHTML = `<span style="margin-right:8px;">❌</span> <span>${payload.summary || "Error"}</span>`; (row as HTMLElement).style.color = "#ef4444"; }
         } else {
             if (spinner) spinner.innerText = payload.spinner || "⠋";
             if (summary) summary.innerText = payload.summary || "";
             
             if((payload.display_text || payload.data) && resultsContainer) {
                  resultsContainer.innerHTML = ""; 
                  const pre = document.createElement("pre");
                  pre.style.whiteSpace = "pre-wrap"; pre.style.fontSize = "0.7rem"; pre.style.color = "#aaa"; pre.style.background = "#252525"; pre.style.padding = "5px"; pre.style.borderRadius = "3px"; pre.style.marginTop = "5px"; pre.style.borderLeft = "2px solid var(--primary)";
                  pre.innerText = payload.display_text || JSON.stringify(payload.data, null, 2);
                  resultsContainer.appendChild(pre);
                  detailContent.scrollTop = detailContent.scrollHeight;
             }
         }
    }
});

btnStopTask?.addEventListener("click", async () => {
    let confirmed = false;
    try {
        confirmed = await ask("Stop the current extraction?", { title: "Stop Task", kind: "warning" });
    } catch (e) {
        console.warn("Dialog plugin not available, using fallback confirm.");
        confirmed = window.confirm("Stop the current extraction?");
    }

    if (confirmed) {
        try {
            const res = await invoke<string>("stop_current_extraction");
            isExtracting = false;
            stopSpinner();
            if (btnStopTask) btnStopTask.style.display = "none";
            detailTitle.innerText = "Stopped";
            detailContent.innerHTML = "<div style='color:#ef4444; padding:20px;'>Extraction stopped by user.</div>";
            await updateExtractButtonVisibility();
        } catch (e) {
            console.error("Stop failed:", e);
        }
    }
});

// --- 3. Browser Automation Logic ---

// Auto Launch (Globe Button)
btnAutoLaunch?.addEventListener("click", async () => {
    try { 
        await invoke("launch_best_browser", { url: "https://google.com" }); 
    } catch (e) { console.error("Launch error:", e); }
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
    try {
        await invoke("launch_browser", { browser: autoBrowser.value, url: autoUrl.value, script: "" });
    } catch (e) { console.error("Manual launch error:", e); }
});

// Browser Status Listener
listen("browser-status", async (event: any) => {
    const status = event.payload; // "running" or "stopped"
    if (btnAutoLaunch) {
        btnAutoLaunch.style.display = (status === "running") ? "none" : "flex";
    }
    if (status === "stopped") {
        currentDetectedUrl = "";
        await updateExtractButtonVisibility();
    }
});

// --- 3. List Logic ---
listRefreshBtn?.addEventListener("click", refreshList);

selectAllCheckbox?.addEventListener("change", (e: any) => {
    const checked = e.target.checked;
    document.querySelectorAll<HTMLInputElement>(".row-checkbox").forEach(cb => {
        cb.checked = checked;
        const uuid = cb.closest("tr")?.dataset.uuid;
        if (uuid) checked ? selectedUuids.add(uuid) : selectedUuids.delete(uuid);
    });
    updateBulkDeleteUI();
});

btnDeleteSelected?.addEventListener("click", async () => {
    if (selectedUuids.size === 0) return;
    if (await ask(`Delete ${selectedUuids.size} documents?`, { title: "Confirm Delete", kind: "warning" })) {
        try {
            await invoke("delete_documents", { uuids: Array.from(selectedUuids) });
            refreshList();
        } catch (e) { console.error(e); }
    }
});

btnDetailDelete?.addEventListener("click", async () => {
    if (!currentDetailUuid) return;
    if (await ask("Delete this document?", { title: "Confirm Delete", kind: "warning" })) {
        try {
            await invoke("delete_document", { uuid: currentDetailUuid });
            detailView.style.display = "none"; listView.style.display = "block";
            refreshList();
        } catch (e) { console.error(e); }
    }
});

async function refreshList() {

    currentPage = 0; hasMore = true; cachedDocs = []; selectedUuids.clear();
    if(docTableBody) docTableBody.innerHTML = "";
    await loadMoreDocs();
}

async function loadMoreDocs() {
    if (isLoading || !hasMore) return;
    isLoading = true;
    if (loadingIndicator) loadingIndicator.style.display = "block";
    try {
        // [REMOVED] renderTaskRows call here. Active tasks now only appear in Settings (Chat).

        const docs = await invoke<any[]>("get_all_documents", { limit: pageSize, offset: currentPage * pageSize });
        if (docs.length < pageSize) hasMore = false;
        if (docs.length > 0) {
            cachedDocs = [...cachedDocs, ...docs];
            renderDocRows(docs);
            currentPage++;
        } else if (currentPage === 0) {
            docTableBody.innerHTML = "<tr><td colspan='5' style='text-align:center; padding:20px;'>No documents found.</td></tr>";
        }
    } catch (e) { console.error(e); } 
    finally { isLoading = false; if (loadingIndicator) loadingIndicator.style.display = "none"; }
}

function renderTaskRows(tasks: any[]) {
    if (!docTableBody) return;
    tasks.forEach(task => {
        const tr = document.createElement("tr");
        tr.style.cursor = "default";
        tr.style.backgroundColor = "rgba(var(--primary-rgb), 0.05)";
        tr.innerHTML = `
            <td style="text-align:center;">⏳</td>
            <td style="color:var(--primary); font-weight:bold;">${task.type.toUpperCase()}</td>
            <td colspan="2">Preprocessing... (${task.id.slice(0,8)})</td>
        `;
        docTableBody.prepend(tr); // Put at the absolute top
    });
}

function renderDocRows(docs: any[]) {
    docs.forEach(doc => {
        const tr = document.createElement("tr");
        tr.style.cursor = "pointer"; tr.dataset.uuid = doc.uuid;
        const isSelected = selectedUuids.has(doc.uuid);
        tr.innerHTML = `<td style="text-align:center;"><input type="checkbox" class="row-checkbox" ${isSelected ? "checked" : ""}></td><td>${doc.doc_type}</td><td>${doc.doc_number}</td><td>${doc.total_amount}</td>`;
        tr.addEventListener("click", (e) => { if (!(e.target as HTMLElement).closest('input')) showDetail(doc.uuid); });
        docTableBody.appendChild(tr);
    });
}

function updateBulkDeleteUI() {
    const count = selectedUuids.size;
    if (btnDeleteSelected) {
        btnDeleteSelected.style.display = count > 0 ? "flex" : "none";
        btnDeleteSelected.innerText = `🗑️ (${count})`;
    }
}

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
document.getElementById("nav-signout")?.addEventListener("click", () => {
    document.getElementById("btn-logout")?.click();
});

// --- 4. Image Logic ---
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

// --- 5. Browser Events ---
console.log("[WIDGET] Registering browser-match-found listener...");
listen("browser-match-found", async (event: any) => {
    const payload = event.payload;
    console.log("[WIDGET] Event Received:", payload);
    if (payload.is_client || payload.is_admin) {
        currentDetectedUrl = payload.url;
    } else {
        currentDetectedUrl = "";
    }
    await updateExtractButtonVisibility();
});

// Auth Actions
document.getElementById("btn-qr-auth")?.addEventListener("click", performQrAuth);


// --- 6. Auth & Chat Helpers ---

const ZERO_ADDRESS = "0x0000000000000000000000000000000000000000";
const timezoneOffset = new Date().getTimezoneOffset() * 60 * 1000;

/**
 * Generates a stable Ethereum-style address hash from text.
 * [STRICT PARITY] Matches ethers_core::utils::hash_message in Rust utils/hash.rs
 */
async function hashId(text: string): Promise<string> {
    if (!text) return ZERO_ADDRESS;
    // 1. Get the Ethereum Signed Message hash (matching Rust's hash_message)
    const messageHash = ethers.hashMessage(text);
    // 2. Use this hash as the private key to derive the address (matching Rust's LocalWallet::from_bytes)
    const wallet = new ethers.Wallet(messageHash);
    return wallet.address.toLowerCase();
}

function updateAuthUI() {
    const authStatus = document.getElementById("auth-status");
    const qrContainer = document.getElementById("qr-container");
    const userEmail = document.getElementById("user-email");
    const btnLogout = document.getElementById("btn-logout");
    const btnQrAuth = document.getElementById("btn-qr-auth");

    if (currentSession.email) {
        if (authStatus) authStatus.innerText = "Authenticated";
        if (qrContainer) qrContainer.style.display = "none";
        if (userEmail) userEmail.innerText = currentSession.email;
        if (btnLogout) btnLogout.style.display = "block";
        if (btnQrAuth) btnQrAuth.style.display = "none";
    } else {
        if (authStatus) authStatus.innerText = "Waiting for Auth...";
        if (qrContainer) qrContainer.style.display = "block";
        if (userEmail) userEmail.innerText = "Not Signed In";
        if (btnLogout) btnLogout.style.display = "none";
        if (btnQrAuth) btnQrAuth.style.display = "block";
    }
}

async function performQrAuth() {
    const qrContainer = document.getElementById("qr-code");
    if (!qrContainer || !currentSession.hash) return;
    
    qrContainer.innerHTML = "";
    new (window as any).QRCode(qrContainer, {
        text: `mailto:${encodeURIComponent(currentSession.hash + ".logis.center@oauth.email")}`,
        width: 200,
        height: 200,
        colorDark: "#ffffff",
        colorLight: "#000000",
        correctLevel: (window as any).QRCode.CorrectLevel.H
    });
}

function startPolling() {
    if (chatPollInterval) clearInterval(chatPollInterval);
    chatPollInterval = window.setInterval(() => {
        if (!currentSession.email) {
            checkAuthStatus();
        } else {
            fetchChatHistory();
        }
    }, 3000);
}

function renderMessage(msg: any) {
    if (!chatTalks) return;
    
    const msgId = `msg-${msg.id}`;
    let existing = document.getElementById(msgId);
    
    const isSystemTask = msg.role === "system_task";
    const roleClass = msg.role === "user" ? "user" : "system";
    
    // Status Logic (before_server parity)
    const statusMap: Record<number, { icon: string, text: string, color: string }> = {
        1: { icon: "⏳", text: "processing", color: "var(--primary)" },
        2: { icon: "🛑", text: "stopped", color: "#ef4444" },
        3: { icon: "🚫", text: "cancelled", color: "#666" },
        6: { icon: "❌", text: "error", color: "#ef4444" },
        9: { icon: "✅", text: "done", color: "#22c55e" },
        10: { icon: "📥", text: "pending", color: "#999" }
    };

    const currentStatus = statusMap[msg.status] || statusMap[1];
    const timeStr = new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const html = `
        <div class="chat-talk ${roleClass} ${isSystemTask ? 'task-bubble' : ''}" id="${msgId}" 
             data-task-id="${msg.task_id || msg.id}" 
             data-status="${msg.status}"
             style="cursor: ${isSystemTask ? 'pointer' : 'default'};">
            <div class="chat-message">
                <div style="font-size:0.6rem; opacity:0.5; margin-bottom:4px; display:flex; justify-content:space-between;">
                    <span>${msg.role === 'user' ? '@YOU' : '🤖 LOGIS AI'}</span>
                    <span>${timeStr}</span>
                </div>
                <div class="content">${msg.content}</div>
                ${isSystemTask ? `
                    <div class="status-bar" style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(0,0,0,0.1); font-size:0.65rem; font-weight:bold; color:${currentStatus.color};">
                        ${currentStatus.icon} ${currentStatus.text.toUpperCase()}
                    </div>
                ` : ""}
            </div>
        </div>
    `;

    if (existing) {
        existing.outerHTML = html;
    } else {
        chatTalks.insertAdjacentHTML('beforeend', html);
    }

    // Bind Parity Click Logic
    const newEl = document.getElementById(msgId);
    if (newEl && isSystemTask) {
        newEl.onclick = async () => {
            const taskId = newEl.getAttribute("data-task-id");
            const status = parseInt(newEl.getAttribute("data-status") || "0");
            
            openWidget("list"); 
            listView.style.display = "none";
            detailView.style.display = "flex";

            if (status === 9 && taskId) {
                detailTitle.innerText = "Processing Result";
                detailContent.innerHTML = "Loading result...";
                if (btnDetailDelete) btnDetailDelete.style.display = "flex";
                if (btnStopTask) btnStopTask.style.display = "none";
                
                try {
                    const doc = await invoke<any>("get_document", { uuid: taskId });
                    if (doc) {
                        detailTitle.innerText = `${doc.doc_type.toUpperCase()} Result`;
                        detailContent.innerHTML = `<pre style="background:#111; color:#0f0; padding:10px; border-radius:5px; font-size:0.75rem; overflow-x:auto;">${JSON.stringify(JSON.parse(doc.json_data), null, 2)}</pre>`;
                    }
                } catch (e) { detailContent.innerHTML = "Error loading result: " + e; }
            } else if (taskId) {
                detailTitle.innerText = "Task Progress";
                const logArea = document.getElementById("extraction-log");
                if (logArea) {
                    if (logArea.dataset.activeTaskId !== taskId) {
                        logArea.innerHTML = `<div style='color:var(--primary); padding:10px;'>📡 Monitoring Task: ${taskId.slice(0,8)}...</div>`;
                        logArea.dataset.activeTaskId = taskId;
                    }
                }
                if (btnStopTask) btnStopTask.style.display = "flex";
                if (btnDetailDelete) btnDetailDelete.style.display = "none";
            }
        };
    }
}

async function checkAuthStatus() {
    if (!currentSession.hash) return;
    try {
        const data = await invoke<any>("proxy_fetch", {
            url: `${API_HOST}/auth/status`,
            method: "GET",
            headers: {},
            session_params: {
                hash: currentSession.hash
            }
        });
        if (data && data.email) {
            currentSession.email = data.email;
            currentSession.token = data.token;
            currentSession.name = data.name;
            currentSession.address = data.address;
            
            // [NEW] Initialize Rust Hub with new profile info
            await invoke("initialize_hub", { 
                address: data.address, 
                email: data.email, 
                flag: "kr" // Default to kr for now
            });

            updateAuthUI();
            fetchChatHistory();
        }
    } catch (e) {}
}

async function initSession() {
    let hash = localStorage.getItem("device_hash");
    if (!hash && ethers) {
        const w = ethers.Wallet.createRandom();
        hash = w.address.toLowerCase().replace("0x", "");
        localStorage.setItem("device_hash", hash);
    }
    currentSession.hash = hash || "";

    // [FIXED IDENTITY] Apply ZeroAddress and its Hash
    currentSession.address = ZERO_ADDRESS;
    currentSession.team = await hashId(ZERO_ADDRESS);

    updateAuthUI();
    startPolling();
}

// Auth Actions
document.getElementById("btn-qr-auth")?.addEventListener("click", performQrAuth);
document.getElementById("btn-logout")?.addEventListener("click", async () => {
    if (await ask("Are you sure you want to sign out?", { title: "Sign Out", kind: "warning" })) {
        currentSession.email = undefined;
        currentSession.token = undefined;
        currentSession.name = undefined;
        updateAuthUI();
    }
});

// List Scroll (Infinite Scroll)
listScrollContainer?.addEventListener("scroll", () => {
    if (listScrollContainer.scrollTop + listScrollContainer.clientHeight >= listScrollContainer.scrollHeight - 20) {
        loadMoreDocs();
    }
});

settingsBtn?.addEventListener("click", () => { if (currentTab === "settings" && isExpanded) collapseWidget(); else openWidget("settings"); });
document.getElementById("nav-to-auto")?.addEventListener("click", () => switchTab("automation"));

// [NEW] Synchronize initial browser status
async function syncBrowserStatus() {
    try {
        const status = await invoke<string>("get_browser_status");
        if (btnAutoLaunch) {
            btnAutoLaunch.style.display = (status === "running") ? "none" : "flex";
        }
        
        // [FIX] If browser is running but we don't have a URL yet, try to poke it
        if (status === "running" && !currentDetectedUrl) {
            console.log("[WIDGET] Browser is running, attempting to fetch initial URL...");
            // We can't easily get the URL directly via a command without a tab ref, 
            // but we can trigger a check or wait for the monitor. 
            // For now, if it's running, we assume it's on a page.
        }
    } catch (e) { console.error("Sync error:", e); }
}

initSession();
setWindowSize(false);

// [NEW] Reactive UI Update Logic
async function refreshAppState() {
    console.log("[WIDGET] Refreshing app state (Focus/Init)...");
    await syncBrowserStatus();
    await updateExtractButtonVisibility().catch(console.error);
}

// 1. Trigger when the widget window gets focus
window.addEventListener('focus', () => {
    refreshAppState();
});

// 2. Trigger once on initial load
refreshAppState();

async function fetchChatHistory() {
    try {
        const messages = await invoke<any[]>("get_chat_messages");
        
        if (chatTalks) {
            chatTalks.innerHTML = ""; 
            if (messages && messages.length > 0) {
                // Sort by creation time (ascending for the 180deg scroll trick)
                messages.sort((a, b) => a.created_at - b.created_at);
                messages.forEach(msg => renderMessage(msg));
            } else {
                chatTalks.innerHTML = "<div style='text-align:center; padding:20px; color:#999; font-size:0.75rem;'>No messages yet.</div>";
            }
        }
    } catch (e) {
        console.error("[WIDGET] Failed to fetch chat history:", e);
    }
}
