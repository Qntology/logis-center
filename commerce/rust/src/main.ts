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
let spinnerInterval: number | null = null; // UI Spinner Animation Timer

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
const btnStopTask = document.getElementById("btn-stop-task") as HTMLButtonElement; // Added

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

// --- Spinner Logic ---
const spinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
function startSpinner() {
    if (spinnerInterval) clearInterval(spinnerInterval);
    let i = 0;
    spinnerInterval = window.setInterval(() => {
        const char = spinnerFrames[i % spinnerFrames.length];
        
        // 1. Animate Main Button
        if (btnExtract) btnExtract.innerText = char;
        
        // 2. Animate All Active Log Spinners
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
}

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
    // Ensure list is populated when focusing
    if (cachedDocs.length === 0) {
        refreshList();
    }
});

searchInput?.addEventListener("input", () => {
    if(searchDebounceTimer) clearTimeout(searchDebounceTimer);
    searchDebounceTimer = window.setTimeout(async () => {
        const keyword = searchInput.value.toLowerCase();
        if (!keyword) {
            refreshList();
            return;
        }
        
        // --- REAL-TIME VECTOR FILTERING ---
        try {
            const results = await invoke<[string, string, number][]>("search_documents", { query: keyword });
            // Update table with matches
            if (docTableBody) {
                docTableBody.innerHTML = "";
                if (results.length === 0) {
                    docTableBody.innerHTML = "<tr><td colspan='4' style='text-align:center; padding:20px;'>No AI matches.</td></tr>";
                } else {
                    // Try to find full doc info from cachedDocs or just render what we have
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

searchInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        btnSubmit?.click();
    }
});

// SUBMIT: Complex AI Search (para2graph + graph2contexts)
btnSubmit?.addEventListener("click", async () => {
    const query = searchInput.value;
    if (!query) return;
    
    openWidget("list"); 
    if (aiResultsArea && aiResultsContent) {
        aiResultsArea.style.display = "block";
        aiResultsTitle.innerText = "🧠 AI Deep Analysis";
        aiResultsContent.innerHTML = "<div class='spinner'></div> 🤖 Thinking and analyzing your query...";
        
        try {
            const response = await invoke<any>("ai_search_complex", { query: query, language: "korean" });
            
            // Render Structured Context
            let html = `<div style="margin-bottom:15px; padding:10px; background:#222; border-left:3px solid var(--primary); font-size:0.75rem;">
                <strong style="display:block; margin-bottom:5px; color:#aaa;">Query Intent:</strong>`;
            
            if (response.structured && response.structured.context) {
                response.structured.context.forEach((ctx: any) => {
                    html += `<div style="margin-bottom:5px;">• ${ctx.text} <span style="color:var(--primary)">[${ctx.type}]</span></div>`;
                });
            }
            html += `</div>`;

            // Render Results
            if(response.results.length === 0) {
                html += "No matching data found for these contexts.";
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
        } catch(e) { aiResultsContent.innerHTML = "<div style='color:#ef4444;'>AI Analysis Error: " + e + "</div>"; }
    }
});

// Add global listener for View Detail button in AI results
document.addEventListener('show-doc', (e: any) => {
    showDetail(e.detail);
});

// EXTRACT: Image or Browser Task
btnExtract?.addEventListener("click", async () => {
    // [PREVENTION] Check for existing task first
    if (currentDetectedUrl) {
        try {
            const isActive = await invoke<boolean>("check_active_task", { ref_id: currentDetectedUrl });
            if (isActive) {
                alert("An extraction for this URL is already in progress. Check the dashboard (Config).");
                openWidget("settings");
                return;
            }
        } catch(e) {}
    }

    openWidget("list");
    
    // Switch to Detail View for Progress
    listView.style.display = "none";
    detailView.style.display = "flex";

    // Toggle Buttons (Ensure Stop button is visible if running)
    if (btnDetailDelete) btnDetailDelete.style.display = "none";
    if (btnStopTask) btnStopTask.style.display = "flex";

    // Block re-execution if already running (just show view)
    if (isExtracting) return;

    // [UI-UPDATE] Separate Spinner from Button
    // We now keep the ⚡ button as is, and the animation happens in the chat/log
    startSpinner();
    
    if (currentImage) {
        isExtracting = true;
        detailTitle.innerText = "⚡ Analyzing Image...";
        detailContent.innerHTML = `<div id="extraction-log" style="display:flex; flex-direction:column; gap:5px; padding-bottom:20px;"></div>`;
        try {
            const taskId = `img_${Date.now()}`;
            await emit("new-task-from-browser", {
                id: taskId,
                type: "image_extraction",
                image_path: currentImage, 
                ref_id: currentImage, // Using path as unique ref
                link: "Local Image"
            });
            // Show in chat too
            addChatMessage({ id: taskId, role: "system_task", content: `Task Started: Image Analysis`, status: "processing" });
        } catch (e) { 
            isExtracting = false;
            detailTitle.innerText = "Error";
        }
    } else {
        isExtracting = true;
        detailTitle.innerText = "⚡ Browser Extraction...";
        detailContent.innerHTML = `<div id="extraction-log" style="display:flex; flex-direction:column; gap:5px; padding-bottom:20px;">
            <div style="color:#888; font-size:0.8rem;">Initializing task...</div>
        </div>`;
        
        try {
            const html = await invoke<string>("extract_html_from_current_tab");
            let taskId = `task_${Date.now()}`;
            
            await emit("new-task-from-browser", {
                id: taskId,
                type: "html_extraction",
                html: html, 
                link: currentDetectedUrl, 
                ref_id: currentDetectedUrl
            });
            
            // Show in chat dashboard
            addChatMessage({ id: taskId, role: "system_task", content: `Task Started: ${new URL(currentDetectedUrl).hostname}`, status: "processing", url: currentDetectedUrl });
            
        } catch (e) {
            detailTitle.innerText = "Error";
        }
    }
});

function addChatMessage(msg: { id: string, role: string, content: string, status?: string, url?: string }) {
    if (!chatTalks) return;
    
    let el = document.getElementById(`msg-${msg.id}`);
    if (!el) {
        el = document.createElement("div");
        el.id = `msg-${msg.id}`;
        el.className = `talk ${msg.role === 'user' ? 'me' : 'ai'}`;
        chatTalks.appendChild(el);
    }

    if (msg.role === "system_task") {
        const isDone = msg.status === "done";
        const isError = msg.status === "error";
        
        el.style.borderLeft = `3px solid ${isDone ? '#4ade80' : (isError ? '#ef4444' : 'var(--primary)')}`;
        el.style.padding = "10px";
        el.style.background = "#222";
        el.style.marginBottom = "10px";
        el.style.borderRadius = "5px";
        el.style.fontSize = "0.8rem";

        el.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div style="flex:1;">
                    <div style="color:#aaa; font-size:0.7rem; margin-bottom:4px;">EXTRACTION TASK</div>
                    <div style="font-weight:bold; margin-bottom:4px;">${msg.content}</div>
                    ${msg.url ? `<div style="color:#666; font-size:0.7rem; word-break:break-all;">${msg.url}</div>` : ''}
                </div>
                <div style="margin-left:10px;">
                    ${isDone ? '✅' : (isError ? '❌' : '<span class="active-spinner">⠋</span>')}
                </div>
            </div>
            <div style="margin-top:10px; display:flex; gap:8px;">
                ${!isDone && !isError ? `<button class="stop-btn" style="background:#ef4444; border:none; color:white; padding:4px 8px; border-radius:3px; cursor:pointer; font-size:0.7rem;">Stop 🛑</button>` : `<span style="color:#4ade80;">Completed</span>`}
            </div>
        `;

        const stopBtn = el.querySelector(".stop-btn");
        if (stopBtn) {
            stopBtn.addEventListener("click", async () => {
                if (confirm("Stop this task?")) {
                    await invoke("stop_current_extraction");
                    msg.status = "error";
                    msg.content = "Task Stopped by User";
                    addChatMessage(msg); // Re-render
                }
            });
        }
    } else {
        // Normal Chat Message
        el.innerText = msg.content;
    }

    // Scroll to bottom
    const chatScroll = document.getElementById("chat-scroll");
    if (chatScroll) chatScroll.scrollTop = chatScroll.scrollHeight;
}

// Update chat from background events
listen("extraction-progress", (event: any) => {
    const payload = event.payload;
    if (payload.task_id) {
        // Find existing message or create summary
        const msgId = payload.task_id;
        if (payload.category === "Done") {
            addChatMessage({ id: msgId, role: "system_task", content: payload.summary, status: "done" });
        } else if (payload.category === "Error") {
            addChatMessage({ id: msgId, role: "system_task", content: payload.summary, status: "error" });
        }
    }
    // ... existing progress log logic ...

// Stop Task
btnStopTask?.addEventListener("click", async () => {
    // 1. Confirm First
    if (!confirm("Stop current extraction?")) {
        return;
    }

    // 2. Update UI immediately to show feedback
    btnStopTask.innerText = "Stopping...";
    if (detailTitle) detailTitle.innerText = "🛑 Stopping...";
    
    try {
        // 3. Send command
        await invoke("stop_current_extraction");
        
        // 4. Force UI Reset 
        isExtracting = false;
        stopSpinner();
        if (btnStopTask) btnStopTask.style.display = "none"; 
        
        // [UI-FIX] Hide delete button if interrupted, as there might not be a valid document to delete
        if (btnDetailDelete) btnDetailDelete.style.display = "none";
        
        // Reset Extract Button
        if (btnExtract) {
            btnExtract.innerText = "⚡";
        }

    } catch(e) { 
        console.error(e); 
        if (btnStopTask) btnStopTask.innerText = "Error";
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
        // Force hide if no match, unless an image is manually selected OR extraction is running
        if (!currentImage && !isExtracting) {
             if (btnExtract) btnExtract.style.display = "none";
        }
    }
});

listen("extraction-progress", (event: any) => {
    const payload = event.payload;
    const catId = payload.category ? payload.category.replace(/[^a-zA-Z0-9]/g, "") : "general";
    const elementId = `progress-${catId}`;

    const extractionLog = document.getElementById("extraction-log");
    
    // Update the button spinner
    // Note: Button animation is now handled by startSpinner() / stopSpinner() independent of payload
    if (btnExtract) {
        // If the task ended (Done or Error), stop spinner and reset
        if (payload.category === "Done" || payload.category === "Error") {
            stopSpinner();
            setTimeout(() => { 
                if (btnExtract) btnExtract.innerText = "⚡"; 
            }, 2000);
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
                 // Remove active-spinner class and set to checkmark
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
             p.style.flexDirection = "column"; // Stack text and results
             p.style.alignItems = "flex-start";
             
             // Container for the row (spinner + text)
             const row = document.createElement("div");
             row.className = "progress-row";
             row.style.display = "flex";
             row.style.alignItems = "center";
             row.innerHTML = `<span class="active-spinner" style="color:var(--primary); margin-right:8px; font-family:monospace; min-width:15px;">${payload.spinner || "⠋"}</span> 
                              <span class="summary-text">${payload.summary || ""}</span>`;
             p.appendChild(row);
             
             // Container for data/results
             const results = document.createElement("div");
             results.className = "results-container";
             results.style.width = "100%";
             p.appendChild(results);
             
             extractionLog.appendChild(p);
         }
         
         // 1. Done State
         if (payload.category === "Done") {
             isExtracting = false; // Reset global state
             
             // Reset Buttons
             if (btnStopTask) { 
                 btnStopTask.style.display = "none"; 
                 btnStopTask.innerText = "🛑"; 
             }
             if (btnDetailDelete) btnDetailDelete.style.display = "flex";
             
             // Hide Extract Button if no image selected
             if (!currentImage && btnExtract) {
                 btnExtract.style.display = "none";
             }

             const summary = payload.summary || "";
             const isCancelled = summary.includes("Cancelled");
             const successText = isCancelled ? "Task Cancelled" : "Extraction Complete";
             
             const row = p.querySelector(".progress-row");
             if (row) {
                 row.innerHTML = `<span style="margin-right:8px;">${isCancelled ? "🛑" : "✅"}</span> <span>${successText}</span>`;
                 (row as HTMLElement).style.color = isCancelled ? "#ef4444" : "#4ade80"; 
             }
             
             if(payload.data) {
                  const pretty = JSON.stringify(payload.data, null, 2);
                  const pre = document.createElement("pre");
                  pre.style.whiteSpace = "pre-wrap"; pre.style.fontSize = "0.75rem";
                  pre.style.color = "#e5e5e5"; pre.style.background = "#1e1e1e";
                  pre.style.padding = "10px"; pre.style.borderRadius = "5px"; pre.style.marginTop = "10px";
                  pre.innerText = pretty;
                  p.appendChild(pre);
             }
         } 
         // 2. Error State
         else if (payload.category === "Error") {
             isExtracting = false;
             if (btnStopTask) btnStopTask.style.display = "none";
             if (!currentImage && btnExtract) btnExtract.style.display = "none";
             
             const row = p.querySelector(".progress-row");
             if (row) {
                 row.innerHTML = `<span style="margin-right:8px;">❌</span> <span>${payload.summary || "Unknown Error"}</span>`;
                 (row as HTMLElement).style.color = "#ef4444";
             }
         } 
         // 3. Progress State (including intermediate items)
         else {
             // Update Summary
             const spinner = p.querySelector(".active-spinner") as HTMLElement;
             const summary = p.querySelector(".summary-text") as HTMLElement;
             if (spinner) spinner.innerText = payload.spinner || "⠋";
             if (summary) summary.innerText = payload.summary || "";
             
             // Append Intermediate Data (if not already appended or just keep adding)
             if(payload.display_text || payload.data) {
                  const resultsContainer = p.querySelector(".results-container");
                  if (resultsContainer) {
                      const pre = document.createElement("pre");
                      pre.style.whiteSpace = "pre-wrap"; pre.style.fontSize = "0.7rem";
                      pre.style.color = "#aaa"; pre.style.background = "#252525";
                      pre.style.padding = "5px"; pre.style.borderRadius = "3px"; pre.style.marginTop = "5px";
                      pre.style.borderLeft = "2px solid var(--primary)";
                      
                      // Use display_text (narrative) if available, otherwise use raw data (JSON)
                      const content = payload.display_text || JSON.stringify(payload.data, null, 2);
                      pre.innerText = content;
                      
                      resultsContainer.appendChild(pre);
                      
                      // Auto-scroll to latest result in detail view
                      detailContent.scrollTop = detailContent.scrollHeight;
                  }
             }
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

async function fetchChatHistory() {
    if (!chatTalks) return;

    try {
        // 1. Fetch from Local LanceDB (Tasks)
        const localTasks = await invoke<any[]>("get_chat_messages");
        
        // 2. Fetch from Cloud Server (if logged in)
        let cloudMessages: any[] = [];
        if (currentSession.email && currentSession.token) {
            try {
                const response = await fetch(`${API_HOST}/talks?hash=${currentSession.hash}`, {
                    headers: { "Authorization": `Bearer ${currentSession.token}` }
                });
                if (response.ok) {
                    const json = await response.ok ? await response.json() : { results: [] };
                    cloudMessages = json.results || [];
                }
            } catch (e) { console.warn("Cloud sync failed, using local only."); }
        }

        // 3. Merge and Sort by Date
        const allMessages = [...localTasks, ...cloudMessages].sort((a, b) => {
            const timeA = a.created_at || a.time || 0;
            const timeB = b.created_at || b.time || 0;
            return timeA - timeB;
        });

        // 4. Render All
        chatTalks.innerHTML = "";
        allMessages.forEach(msg => renderMessage(msg));
        
        // Scroll to bottom
        const chatScroll = document.getElementById("chat-scroll");
        if (chatScroll) chatScroll.scrollTop = chatScroll.scrollHeight;

    } catch (err) {
        console.error("Chat history error:", err);
    }
}

function renderMessage(msg: any) {
    if (!chatTalks) return;

    const role = msg.role || (msg.is_ai ? "assistant" : "user");
    const content = msg.content || msg.text || "";
    const id = msg.id || msg.uuid || `msg-${Date.now()}-${Math.random()}`;

    let el = document.getElementById(`msg-${id}`);
    if (!el) {
        el = document.createElement("div");
        el.id = `msg-${id}`;
        el.className = `talk ${role === 'user' ? 'me' : 'ai'}`;
        chatTalks.appendChild(el);
    }

    if (role === "system_task") {
        const isDone = msg.status === "done";
        const isError = msg.status === "error";
        
        el.className = "talk ai task-card";
        el.style.borderLeft = `3px solid ${isDone ? '#4ade80' : (isError ? '#ef4444' : 'var(--primary)')}`;
        el.style.background = "#1a1a1a";
        el.style.padding = "12px";
        el.style.marginBottom = "12px";
        el.style.borderRadius = "8px";

        el.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div style="flex:1;">
                    <div style="color:var(--primary); font-size:0.65rem; font-weight:bold; margin-bottom:4px; letter-spacing:1px;">WEB EXTRACTION</div>
                    <div style="font-weight:bold; margin-bottom:4px; font-size:0.85rem; color:#fff;">${content}</div>
                    ${msg.url ? `<div style="color:#666; font-size:0.7rem; word-break:break-all; font-family:monospace;">${msg.url}</div>` : ''}
                </div>
                <div style="margin-left:10px; font-size:1.2rem;">
                    ${isDone ? '✅' : (isError ? '❌' : '<span class="active-spinner">⠋</span>')}
                </div>
            </div>
            <div style="margin-top:12px; display:flex; align-items:center; gap:10px;">
                ${!isDone && !isError 
                    ? `<button class="stop-btn" data-task="${msg.task_id}" style="background:#331111; border:1px solid #ef4444; color:#ef4444; padding:4px 10px; border-radius:4px; cursor:pointer; font-size:0.7rem; font-weight:bold;">STOP TASK 🛑</button>` 
                    : `<span style="color:#4ade80; font-size:0.7rem; font-weight:bold;">${isDone ? 'COMPLETED' : 'STOPPED'}</span>`
                }
                <div style="flex:1;"></div>
                <span style="font-size:0.65rem; color:#444;">${new Date(msg.created_at).toLocaleTimeString()}</span>
            </div>
        `;

        const stopBtn = el.querySelector(".stop-btn") as HTMLElement;
        if (stopBtn) {
            stopBtn.onclick = async (e) => {
                e.stopPropagation();
                if (await ask("Stop this extraction task?", { title: "Logis AI", kind: "warning" })) {
                    await invoke("stop_current_extraction");
                    // Local UI update will happen via polling or immediate re-fetch
                    fetchChatHistory();
                }
            };
        }
    } else {
        // Standard Chat Bubble
        el.innerHTML = `<div class="content">${marked.parse(content)}</div>
                        <div class="meta">${new Date(msg.created_at || msg.time).toLocaleTimeString()}</div>`;
    }
}

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