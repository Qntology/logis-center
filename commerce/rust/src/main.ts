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
let isExtracting = false; // Track any extraction task

// --- UI Elements ---
const contentPanel = document.getElementById("content-panel") as HTMLElement;
const searchInput = document.getElementById("global-search") as HTMLInputElement;
const btnSubmit = document.getElementById("btn-submit") as HTMLButtonElement; // Search
const btnExtract = document.getElementById("btn-extract") as HTMLButtonElement; // Extract
const btnAutoLaunch = document.getElementById("btn-auto-launch") as HTMLButtonElement;
const settingsBtn = document.getElementById("btn-settings") as HTMLButtonElement;
const tabContents = document.querySelectorAll<HTMLElement>(".tab-content");

// Nav Preview
const navPreviewContainer = document.getElementById("nav-preview-container") as HTMLElement;
const navImgThumbnail = document.getElementById("nav-img-thumbnail") as HTMLImageElement;
const navImgClear = document.getElementById("nav-img-clear") as HTMLButtonElement;
const navUploadBtn = document.getElementById("nav-upload-btn");

// Views
const listView = document.getElementById("list-view") as HTMLElement;
const detailView = document.getElementById("detail-view") as HTMLElement;
const detailTitle = document.getElementById("detail-title") as HTMLElement;
const detailContent = document.getElementById("detail-content") as HTMLElement;
const btnDetailBack = document.getElementById("btn-detail-back") as HTMLButtonElement;
const btnListBack = document.getElementById("btn-list-back") as HTMLButtonElement;
const btnDetailDelete = document.getElementById("btn-detail-delete") as HTMLButtonElement;

// List Elements
const docTableBody = document.getElementById("doc-tbody") as HTMLElement;
const listRefreshBtn = document.getElementById("list-refresh-btn") as HTMLButtonElement;
const btnDeleteSelected = document.getElementById("btn-delete-selected") as HTMLButtonElement;
const selectAllCheckbox = document.getElementById("select-all-checkbox") as HTMLInputElement;
const listScrollContainer = document.getElementById("list-scroll-container") as HTMLElement;
const loadingIndicator = document.getElementById("loading-indicator") as HTMLElement;

// AI Results
const aiResultsArea = document.getElementById("ai-search-results") as HTMLElement;
const aiResultsTitle = document.getElementById("ai-results-title") as HTMLElement;
const aiResultsContent = document.getElementById("ai-results-content") as HTMLElement;

// Chat Elements
const chatTalks = document.querySelector('.chat-talks') as HTMLElement;
const chatForm = document.querySelector('form[name="chat-form"]') as HTMLFormElement;
const chatInput = chatForm?.querySelector('input[name="talk"]') as HTMLInputElement;

// --- 1. Layout & Window Logic ---

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

searchInput?.addEventListener("focus", () => {
    openWidget("list");
});

searchInput?.addEventListener("input", () => {
    if(searchDebounceTimer) clearTimeout(searchDebounceTimer);
    searchDebounceTimer = window.setTimeout(() => {
        const keyword = searchInput.value.toLowerCase();
        filterListLocally(keyword);
    }, 1000);
});

// SUBMIT: Regular Search + AI
btnSubmit?.addEventListener("click", async () => {
    const query = searchInput.value;
    if (!query) return;
    
    openWidget("list"); 
    if (aiResultsArea && aiResultsContent) {
        aiResultsArea.style.display = "block";
        aiResultsTitle.innerText = "🔍 AI Search Results";
        aiResultsContent.innerHTML = "🔍 AI Searching documents...";
        try {
            const results = await invoke<[string, string, number][]>("search_documents", { query: query });
            if(results.length === 0) aiResultsContent.innerHTML = "No AI matches found.";
            else {
                aiResultsContent.innerHTML = results.map(([_, text, score]) => 
                    `<div style="border-bottom:1px solid #444; padding:6px 0;">
                       <strong style="color:var(--primary)">${score.toFixed(2)}</strong> ${text}
                     </div>`
                ).join("");
            }
        } catch(e) { aiResultsContent.innerHTML = "Error: " + e; }
    }
});

// EXTRACT: Image or Browser Task
btnExtract?.addEventListener("click", async () => {
    openWidget("list");
    
    // Switch to Detail View for Progress
    listView.style.display = "none";
    detailView.style.display = "flex";

    // Block re-execution if already running
    if (isExtracting) {
        return;
    }
    
    if (currentImage) {
        // --- 1. Image Extraction ---
        isExtracting = true;
        detailTitle.innerText = "⚡ Analyzing Image...";
        detailContent.innerHTML = `<div id="extraction-log" style="display:flex; flex-direction:column; gap:5px; padding-bottom:20px;"></div>`;
        try {
            // Generate Task ID
            const taskId = `img_${Date.now()}`;
            
            // Send to Backend Scheduler
            await emit("new-task-from-browser", {
                id: taskId,
                type: "image_extraction",
                image_path: currentImage, 
                ref_id: currentSession.hash || "manual"
            });
            
        } catch (e) { 
            isExtracting = false;
            detailTitle.innerText = "Error";
            detailContent.innerHTML = `<div style="color:red;">Failed to queue: ${e}</div>`; 
        }
    } else {
        // --- 2. Browser Extraction ---
        isExtracting = true;
        detailTitle.innerText = "⚡ Browser Extraction...";
        detailContent.innerHTML = `<div id="extraction-log" style="display:flex; flex-direction:column; gap:5px; padding-bottom:20px;">
            <div style="color:#888; font-size:0.8rem;">Initializing task...</div>
        </div>`;
        
        try {
            const html = await invoke<string>("extract_html_from_current_tab");
            
            // Generate Task ID
            let taskId = `task_${Date.now()}`;
            if (currentDetectedUrl) {
                try {
                    const urlObj = new URL(currentDetectedUrl);
                    const link = urlObj.pathname + urlObj.search;
                    const cc = currentSession.cc || "logis.center";
                    taskId = await hashId(cc + link);
                } catch (e) {}
            }

            // Send to Backend
            await emit("new-task-from-browser", {
                id: taskId,
                type: "html_extraction",
                html: html, 
                link: currentDetectedUrl, 
                ref_id: currentSession.hash || "manual"
            });
            
            // Note: The 'extraction-progress' listener below will handle UI updates
            
        } catch (e) {
            detailTitle.innerText = "Error";
            detailContent.innerHTML = `<div style="color:red;">Browser Error: ${e}</div>`;
        }
    }
});

// --- 3. List Logic ---

async function refreshList() {
    currentPage = 0; hasMore = true; cachedDocs = []; selectedUuids.clear();
    updateBulkDeleteUI(); if(docTableBody) docTableBody.innerHTML = "";
    await loadMoreDocs();
}

async function loadMoreDocs() {
    if (isLoading || !hasMore) return;
    isLoading = true;
    if (loadingIndicator) loadingIndicator.style.display = "block";
    try {
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

function renderDocRows(docs: any[]) {
    docs.forEach(doc => {
        const tr = document.createElement("tr");
        tr.style.cursor = "pointer"; tr.dataset.uuid = doc.uuid;
        const isSelected = selectedUuids.has(doc.uuid);
        tr.innerHTML = `<td style="text-align:center;"><input type="checkbox" class="row-checkbox" ${isSelected ? "checked" : ""}></td><td>${doc.doc_type}</td><td>${doc.doc_number}</td><td>${doc.total_amount}</td>`;
        tr.addEventListener("click", (e) => { if (!(e.target as HTMLElement).closest('input')) showDetail(doc.uuid); });
        tr.querySelector(".row-checkbox")?.addEventListener("change", (e:any) => {
            if (e.target.checked) selectedUuids.add(doc.uuid); else selectedUuids.delete(doc.uuid); updateBulkDeleteUI();
        });
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
    detailTitle.innerText = "Loading...";
    detailContent.innerHTML = "Fetching details...";
    
    try {
        const doc = await invoke<any>("get_document", { uuid: uuid });
        if (doc) {
            const type = doc.doc_type || "Unknown";
            const no = doc.doc_number || "No Number";
            detailTitle.innerText = `${type} ${no}`;
            let prettyJson = doc.json_data;
            try { prettyJson = JSON.stringify(JSON.parse(doc.json_data), null, 2); } catch(e) {}
            
            detailContent.innerHTML = `
                <div style="margin-bottom:10px;"><strong>Summary:</strong><br>${doc.text}</div>
                <hr style="border-color:#444;">
                <pre style="white-space: pre-wrap; font-size: 0.75rem; color:#000;">${prettyJson}</pre>
            `;
        } else {
            detailTitle.innerText = "Error"; detailContent.innerHTML = "Document not found.";
        }
    } catch (e) { detailTitle.innerText = "Error"; detailContent.innerHTML = "Failed: " + e; }
}

btnDetailBack?.addEventListener("click", () => { detailView.style.display = "none"; listView.style.display = "block"; });
btnListBack?.addEventListener("click", collapseWidget);

// --- 4. Image Logic ---

async function handleImageUpload(path: string) {
    currentImage = path;
    if (navPreviewContainer && navImgThumbnail) {
        navPreviewContainer.classList.remove("hidden");
        navUploadBtn?.classList.add("active-emoji");
        
        searchInput.disabled = true; searchInput.placeholder = "Image selected"; searchInput.style.opacity = "0.5";
        btnSubmit.style.display = "none"; btnExtract.style.display = "flex";
        
        try {
            const contents = await readFile(currentImage);
            const blob = new Blob([contents]);
            const reader = new FileReader();
            reader.onloadend = () => { navImgThumbnail.src = reader.result as string; };
            reader.readAsDataURL(blob);
        } catch (e) { navImgThumbnail.src = convertFileSrc(currentImage); }
    }
}

navImgClear?.addEventListener("click", () => {
    currentImage = null;
    navPreviewContainer.classList.add("hidden");
    navUploadBtn?.classList.remove("active-emoji");
    
    searchInput.disabled = false; searchInput.placeholder = "Search keywords..."; searchInput.style.opacity = "1";
    btnSubmit.style.display = "flex"; btnExtract.style.display = "none";
});

navUploadBtn?.addEventListener("click", async () => {
    try {
        const file = await open({ multiple: false, filters: [{ name: 'Image', extensions: ['png', 'jpg', 'jpeg'] }] });
        if (file) await handleImageUpload(file as string);
    } catch (err) { console.error(err); }
});

// --- 5. Browser Automation & Events ---

listen("browser-match-found", (event: any) => {
    const payload = event.payload;
    if (payload.is_client || payload.is_admin) {
        currentDetectedUrl = payload.url;
        if (btnExtract) {
            btnExtract.style.display = "flex";
            btnExtract.title = `Extract from ${new URL(payload.url).hostname}`;
        }
    } else {
        // Hide only if NO image selected AND NOT currently extracting via browser
        if (!currentImage && btnExtract && !isBrowserExtracting) btnExtract.style.display = "none";
    }
});

listen("extraction-progress", (event: any) => {
    const payload = event.payload;
    const catId = payload.category ? payload.category.replace(/[^a-zA-Z0-9]/g, "") : "general";
    const elementId = `progress-${catId}`;

    const extractionLog = document.getElementById("extraction-log");
    
    // Update the button spinner
    if (btnExtract) {
        if (payload.spinner) btnExtract.innerText = payload.spinner;
        if (payload.category === "Done") {
            setTimeout(() => { btnExtract.innerText = "⚡"; }, 2000);
        }
    }

    // Only update if we are in the detail view watching logs
    if (extractionLog && detailView.style.display !== "none") {
         
         // 1. Visual Cleanup: Mark previous steps as done if focus moves to a new category
         Array.from(extractionLog.children).forEach((child: any) => {
             // If it is a progress row, is NOT the current one, and doesn't have a checkmark yet
             if (child.id.startsWith("progress-") && child.id !== elementId && !child.innerHTML.includes("✅")) {
                 const textSpan = child.querySelector("span:last-child");
                 const text = textSpan ? textSpan.innerText : "Completed";
                 child.innerHTML = `<span style="margin-right:8px; color:#666;">✅</span> <span style="color:#888;">${text}</span>`;
             }
         });

         let p = document.getElementById(elementId);
         if (!p) {
             p = document.createElement("div");
             p.id = elementId;
             p.style.borderBottom = "1px solid #333";
             p.style.padding = "6px 0";
             p.style.fontSize = "0.75rem";
             p.style.display = "flex";
             p.style.alignItems = "center";
             extractionLog.appendChild(p);
         }
         
         if (payload.category === "Done" || payload.data) {
             isExtracting = false; // Reset state
             const successText = payload.category === "Done" ? "Extraction Complete" : `<strong>${payload.category}</strong> extracted.`;
             p.innerHTML = `<span style="margin-right:8px;">✅</span> <span>${successText}</span>`;
             p.style.color = "#4ade80"; 
             
             if(payload.category === "Done" && payload.data) {
                  // Show final JSON
                  const pretty = JSON.stringify(payload.data, null, 2);
                  const pre = document.createElement("pre");
                  pre.style.whiteSpace = "pre-wrap"; pre.style.fontSize = "0.75rem";
                  pre.style.color = "#e5e5e5"; pre.style.background = "#1e1e1e";
                  pre.style.padding = "10px"; pre.style.borderRadius = "5px"; pre.style.marginTop = "10px";
                  pre.innerText = pretty;
                  extractionLog.appendChild(pre);
             }
         } else {
             // Progress
             p.innerHTML = `<span style="color:var(--primary); margin-right:8px; font-family:monospace; min-width:15px;">${payload.spinner}</span> <span>${payload.summary}</span>`;
         }
    }
});

// Browser Status Listener (Active State)
listen("browser-status", (event: any) => {
    const status = event.payload; // "running" or "stopped"
    if (btnAutoLaunch) {
        if (status === "running") {
            btnAutoLaunch.style.display = "none";
        } else {
            btnAutoLaunch.style.display = "flex";
        }
    }
});

// Auto Launch
btnAutoLaunch?.addEventListener("click", async () => {
    try { await invoke("launch_best_browser", { url: "https://google.com" }); } 
    catch (e) { console.error(e); }
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
    } catch (e) { console.error(e); }
}

autoBtn?.addEventListener("click", async () => {
    try {
        await invoke("launch_browser", { browser: autoBrowser.value, url: autoUrl.value, script: "" });
    } catch (e) { console.error(e); }
});

// --- 6. Auth & Chat Helpers ---

async function hashId(text?: string): Promise<string> {
    if (!ethers) return "";
    if (typeof text === "undefined") {
        const account = ethers.Wallet.createRandom();
        text = account.privateKey;
    }
    const hashMessage = ethers.hashMessage(text);
    return ethers.computeAddress(hashMessage).toLowerCase();
}

function updateAuthUI() {
    const isLoggedIn = !!currentSession.email;
    const name = currentSession.name || "Sign In";
    const navName = document.getElementById("nav-profile-name");
    const navSignin = document.getElementById("nav-signin");
    const navSignout = document.getElementById("nav-signout");
    const navEdit = document.getElementById("nav-profile-edit");
    const navFavicon = document.getElementById("nav-profile-favicon");

    if (navName) navName.innerText = name;
    if (blockies && currentSession.address && navFavicon) {
        navFavicon.style.backgroundImage = `url(${blockies.create({ seed: currentSession.address.toLowerCase() }).toDataURL()})`;
    }

    if (isLoggedIn) {
        navSignin?.classList.add("hidden"); navSignout?.classList.remove("hidden"); navEdit?.classList.remove("hidden");
        document.getElementById("auth-status-text")!.innerText = `🟢 ${name}`;
        document.querySelector('form[name="chat-form"]')?.classList.remove("hidden");
    } else {
        navSignin?.classList.remove("hidden"); navSignout?.classList.add("hidden"); navEdit?.classList.add("hidden");
        document.getElementById("auth-status-text")!.innerText = "🔴 Anonymous";
        document.querySelector('form[name="chat-form"]')?.classList.add("hidden");
    }
}

function performQrAuth() {
    if (!currentSession.hash) return;
    const email = `${currentSession.hash}.logis.center@oauth.email`;
    const mailto = `mailto:${encodeURIComponent(email)}?subject=Login&body=Hash:${currentSession.hash}`;
    
    document.getElementById("auth-status-text")!.innerText = "📧 Scan QR to Login...";
    const qrContainer = document.querySelector('.qrcode') as HTMLElement;
    if (qrContainer && QRCode) {
        qrContainer.innerHTML = ""; qrContainer.style.display = "block";
        new QRCode(qrContainer, { text: mailto, width: 200, height: 200 });
    }
}

// Init
async function initSession() {
    // ... (Existing init logic slightly simplified for brevity, assume localstorage load)
    let hash = localStorage.getItem("device_hash");
    if (!hash) {
        if(ethers) {
            const w = ethers.Wallet.createRandom();
            hash = w.address.toLowerCase().replace("0x", "");
            localStorage.setItem("device_hash", hash);
        }
    }
    currentSession.hash = hash || "";
    updateAuthUI();
}

settingsBtn?.addEventListener("click", () => {
    if (currentTab === "settings" && isExpanded) collapseWidget();
    else openWidget("settings");
});
document.getElementById("nav-to-auto")?.addEventListener("click", () => switchTab("automation"));
document.getElementById("nav-back-list")?.addEventListener("click", () => switchTab("list"));

initSession();
setWindowSize(false);