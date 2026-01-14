import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { open, ask } from '@tauri-apps/plugin-dialog';
import { listen } from '@tauri-apps/api/event';
import { readFile } from '@tauri-apps/plugin-fs';

// --- Configuration ---
const WIDGET_WIDTH = 380;
const COLLAPSED_HEIGHT = 80; // Pill (64px) + Padding
const EXPANDED_HEIGHT = 600; // Expanded content

// --- Drag Logic (Manual Fallback with Threshold) ---
const pillNav = document.querySelector('.pill-nav') as HTMLElement;

if (pillNav) {
    let isMouseDown = false;
    let startX = 0;
    let startY = 0;
    const DRAG_THRESHOLD = 5;

    pillNav.addEventListener('mousedown', (e) => {
        const target = e.target as HTMLElement;
        const isButton = target.closest('button');
        const isInput = target.closest('input'); // Ignore input
        
        if (!isButton && !isInput && e.button === 0) {
             isMouseDown = true;
             startX = e.clientX;
             startY = e.clientY;
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (!isMouseDown) return;
        
        const dx = Math.abs(e.clientX - startX);
        const dy = Math.abs(e.clientY - startY);
        
        if (dx > DRAG_THRESHOLD || dy > DRAG_THRESHOLD) {
            isMouseDown = false; // Reset so we don't trigger multiple times
            invoke('start_drag').catch(e => console.error("Rust Drag Error:", e));
        }
    });

    window.addEventListener('mouseup', () => {
        isMouseDown = false;
    });

    // Double Click to Center Top
    pillNav.addEventListener('dblclick', async (e) => {
        const target = e.target as HTMLElement;
        const isButton = target.closest('button');
        const isInput = target.closest('input');
        if (!isButton && !isInput) {
             console.log("Double click detected - centering window");
             invoke("move_to_top_center").catch(e => console.error("Move Error:", e));
        }
    });
}

// --- State ---
let isExpanded = false;
let currentTab = "list"; // Default is list now
let currentImage: string | null = null;
let searchDebounceTimer: number | null = null;

// --- UI Elements ---
const contentPanel = document.getElementById("content-panel") as HTMLElement;
const searchInput = document.getElementById("global-search") as HTMLInputElement;
const submitBtn = document.getElementById("btn-submit") as HTMLButtonElement;
const extractBtnNav = document.getElementById("btn-extract") as HTMLButtonElement;
const autoLaunchBtn = document.getElementById("btn-auto-launch") as HTMLButtonElement;
const settingsBtn = document.getElementById("btn-settings") as HTMLButtonElement;
const tabContents = document.querySelectorAll<HTMLElement>(".tab-content");

const navPreviewContainer = document.getElementById("nav-preview-container") as HTMLElement;
const navImgThumbnail = document.getElementById("nav-img-thumbnail") as HTMLImageElement;
const navImgClear = document.getElementById("nav-img-clear") as HTMLButtonElement;

const aiResultsArea = document.getElementById("ai-search-results") as HTMLElement;
const aiResultsTitle = document.getElementById("ai-results-title") as HTMLElement;
const aiResultsContent = document.getElementById("ai-results-content") as HTMLElement;

// --- 1. Layout & Window Management ---
// ... (previous layout logic) ...

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

    // Toggle Settings Active State
    if (tabName === "settings") {
        settingsBtn?.classList.add("active-emoji", "active");
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
    
    // Clear Settings Active State on Collapse
    settingsBtn?.classList.remove("active-emoji", "active");
}

// --- 2. Search & Navigation Logic ---

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

// SUBMIT: Regular Search Only
submitBtn?.addEventListener("click", async () => {
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

// EXTRACT: Image Analysis Only
extractBtnNav?.addEventListener("click", async () => {
    if (!currentImage) return;
    
    openWidget("list");
    
    // Switch to Detail View for Progress Overlay
    listView.style.display = "none";
    detailView.style.display = "flex";
    detailTitle.innerText = "⚡ Extracting...";
    detailContent.innerHTML = `<div id="extraction-log" style="display:flex; flex-direction:column; gap:5px; padding-bottom:20px;"></div>`;
    
    try {
        const result = await invoke<string>("summarize_image", { imagePath: currentImage });
        const parsed = JSON.parse(result);
        
        detailTitle.innerText = `${parsed.header?.doc_type || 'Unknown'} ${parsed.header?.document_number || ''}`;
        
        // Render Final Result
        const prettyJson = JSON.stringify(parsed, null, 2);
        detailContent.innerHTML = `
            <div style="margin-bottom:10px; color: #4ade80;"><strong>✅ Analysis Complete</strong></div>
            <div style="margin-bottom:10px; font-size: 0.8rem;">
               <strong>Type:</strong> ${parsed.header?.doc_type || 'Unknown'}<br>
               <strong>No:</strong> ${parsed.header?.document_number || 'N/A'}
            </div>
            <hr style="border-color:#444;">
            <pre style="white-space: pre-wrap; font-size: 0.75rem; color:#e5e5e5; background:#1e1e1e; padding:10px; border-radius:5px;">${prettyJson}</pre>
        `;
    } catch (e) { 
        detailTitle.innerText = "Error";
        detailContent.innerHTML = `<div style="color:red;">Failed to extract: ${e}</div>`; 
    }
});

// AUTO: Launch Best Browser
autoLaunchBtn?.addEventListener("click", async () => {
    console.log("Launching best browser...");
    try {
        await invoke("launch_best_browser", { url: "https://google.com" }); 
    } catch (e) {
        console.error("Auto Launch Failed:", e);
        // Minimal visual feedback if needed
    }
});

// Handle Image logic
async function handleImageUpload(path: string) {
    currentImage = path;
    const navUploadBtn = document.getElementById("nav-upload-btn");
    
    if (navPreviewContainer && navImgThumbnail) {
        navPreviewContainer.classList.remove("hidden");
        navUploadBtn?.classList.add("active-emoji"); // Activate Emoji
        
        // UI Interaction Change: Disable Search, Show Extract
        searchInput.disabled = true;
        searchInput.placeholder = "Image selected";
        searchInput.style.opacity = "0.5";
        
        submitBtn.style.display = "none";
        extractBtnNav.style.display = "flex";
        
        try {
            const contents = await readFile(currentImage);
            const blob = new Blob([contents]);
            const reader = new FileReader();
            reader.onloadend = () => { navImgThumbnail.src = reader.result as string; };
            reader.readAsDataURL(blob);
        } catch (e) {
            navImgThumbnail.src = convertFileSrc(currentImage);
        }
    }
}

navImgClear?.addEventListener("click", () => {
    currentImage = null;
    navPreviewContainer.classList.add("hidden");
    document.getElementById("nav-upload-btn")?.classList.remove("active-emoji"); // Deactivate Emoji
    
    // Restore Search UI
    searchInput.disabled = false;
    searchInput.placeholder = "Search keywords...";
    searchInput.style.opacity = "1";
    searchInput.value = "";
    
    submitBtn.style.display = "flex";
    extractBtnNav.style.display = "none";
});

document.getElementById("nav-upload-btn")?.addEventListener("click", async () => {
    try {
        const file = await open({
            multiple: false,
            filters: [{ name: 'Image', extensions: ['png', 'jpg', 'jpeg'] }]
        });
        if (file) await handleImageUpload(file as string);
    } catch (err) { console.error(err); }
});

// Settings & Navigation
settingsBtn?.addEventListener("click", () => {
    if (currentTab === "settings" && isExpanded) collapseWidget();
    else openWidget("settings");
});

document.getElementById("nav-to-auto")?.addEventListener("click", () => switchTab("automation"));
document.getElementById("nav-back-list")?.addEventListener("click", () => switchTab("list"));

// Initialize
setWindowSize(false);

// --- 3. Features (List, Auto) ---

const docTableBody = document.getElementById("doc-tbody") as HTMLElement;
const listRefreshBtn = document.getElementById("list-refresh-btn") as HTMLButtonElement;
const btnDeleteSelected = document.getElementById("btn-delete-selected") as HTMLButtonElement;
const selectAllCheckbox = document.getElementById("select-all-checkbox") as HTMLInputElement;
const listScrollContainer = document.getElementById("list-scroll-container") as HTMLElement;
const loadingIndicator = document.getElementById("loading-indicator") as HTMLElement;

// State
let cachedDocs: any[] = []; 
let currentPage = 0;
const pageSize = 10;
let isLoading = false;
let hasMore = true;
let selectedUuids = new Set<string>();

// Detail View Elements
const listView = document.getElementById("list-view") as HTMLElement;
const detailView = document.getElementById("detail-view") as HTMLElement;
const detailTitle = document.getElementById("detail-title") as HTMLElement;
const detailContent = document.getElementById("detail-content") as HTMLElement;
const btnDetailBack = document.getElementById("btn-detail-back") as HTMLButtonElement;
const btnListBack = document.getElementById("btn-list-back") as HTMLButtonElement;
const btnDetailDelete = document.getElementById("btn-detail-delete") as HTMLButtonElement;
let currentDetailUuid: string | null = null;

async function refreshList() {
    currentPage = 0;
    hasMore = true;
    cachedDocs = [];
    selectedUuids.clear();
    updateBulkDeleteUI();
    if(docTableBody) docTableBody.innerHTML = "";
    await loadMoreDocs();
}

async function loadMoreDocs() {
    if (isLoading || !hasMore) return;
    isLoading = true;
    if (loadingIndicator) loadingIndicator.style.display = "block";

    try {
        const offset = currentPage * pageSize;
        const docs = await invoke<any[]>("get_all_documents", { limit: pageSize, offset: offset });
        
        if (docs.length < pageSize) {
            hasMore = false;
        }
        
        if (docs.length > 0) {
            cachedDocs = [...cachedDocs, ...docs];
            renderDocRows(docs);
            currentPage++;
        } else if (currentPage === 0) {
            docTableBody.innerHTML = "<tr><td colspan='5' style='text-align:center; padding:20px;'>No documents found.</td></tr>";
        }
    } catch (e) {
        console.error("Failed to load docs:", e);
    } finally {
        isLoading = false;
        if (loadingIndicator) loadingIndicator.style.display = "none";
    }
}

function renderDocRows(docs: any[]) {
    docs.forEach(doc => {
        const tr = document.createElement("tr");
        tr.style.cursor = "pointer";
        tr.dataset.uuid = doc.uuid;
        
        const isSelected = selectedUuids.has(doc.uuid);
        
        tr.innerHTML = `
            <td style="text-align:center;">
                <input type="checkbox" class="row-checkbox" ${isSelected ? "checked" : ""}>
            </td>
            <td>${doc.doc_type}</td>
            <td>${doc.doc_number}</td>
            <td>${doc.total_amount}</td>
        `;
        
        // Row Click -> Show Detail
        tr.addEventListener("click", (e) => {
             const target = e.target as HTMLElement;
             if (target.closest('input[type="checkbox"]')) return;
             showDetail(doc.uuid);
        });

        // Checkbox Logic
        const checkbox = tr.querySelector(".row-checkbox") as HTMLInputElement;
        checkbox.addEventListener("change", (e) => {
            if (checkbox.checked) selectedUuids.add(doc.uuid);
            else selectedUuids.delete(doc.uuid);
            updateBulkDeleteUI();
        });
        
        docTableBody.appendChild(tr);
    });
}

function updateBulkDeleteUI() {
    const count = selectedUuids.size;
    if (btnDeleteSelected) {
        if (count > 0) {
            btnDeleteSelected.style.display = "flex";
            btnDeleteSelected.innerText = `🗑️ (${count})`;
        } else {
            btnDeleteSelected.style.display = "none";
        }
    }
    
    // Update "Select All" state
    if (selectAllCheckbox) {
        // Simple heuristic: if all loaded docs are selected
        const loadedCount = cachedDocs.length;
        selectAllCheckbox.checked = loadedCount > 0 && loadedCount === count;
        // Note: "Select All" usually implies selecting ALL in DB, but here we'll stick to "loaded" or just "current page" behavior?
        // Let's make "Select All" toggle currently loaded items for simplicity.
    }
}

// Select All Listener
selectAllCheckbox?.addEventListener("change", () => {
    const isChecked = selectAllCheckbox.checked;
    const checkboxes = docTableBody.querySelectorAll(".row-checkbox") as NodeListOf<HTMLInputElement>;
    
    checkboxes.forEach(cb => {
        cb.checked = isChecked;
        // Trigger change event logic manually or update set
        const tr = cb.closest("tr");
        const uuid = tr?.dataset.uuid;
        if (uuid) {
            if (isChecked) selectedUuids.add(uuid);
            else selectedUuids.delete(uuid);
        }
    });
    updateBulkDeleteUI();
});

// Bulk Delete Action
btnDeleteSelected?.addEventListener("click", async () => {
    if (selectedUuids.size === 0) return;
    
    const confirmed = await ask(`Delete ${selectedUuids.size} selected items?`, { title: 'Delete Documents', kind: 'warning' });
    if (confirmed) {
        try {
            await invoke("delete_documents", { uuids: Array.from(selectedUuids) });
            refreshList();
        } catch (e) {
            console.error("Bulk delete failed:", e);
            alert("Failed to delete items.");
        }
    }
});

// Infinite Scroll Listener
listScrollContainer?.addEventListener("scroll", () => {
    const { scrollTop, scrollHeight, clientHeight } = listScrollContainer;
    if (scrollTop + clientHeight >= scrollHeight - 20) { // Threshold 20px
        loadMoreDocs();
    }
});

async function showDetail(uuid: string) {
    currentDetailUuid = uuid;

    // 1. Switch View
    listView.style.display = "none";
    detailView.style.display = "flex";
    detailTitle.innerText = "Loading...";
    detailContent.innerHTML = "Fetching details...";
    
    // 2. Fetch Data
    try {
        const doc = await invoke<any>("get_document", { uuid: uuid });
        if (doc) {
            // 3. Render Header: "Type No"
            const type = doc.doc_type || "Unknown";
            const no = doc.doc_number || "No Number";
            detailTitle.innerText = `${type} ${no}`;
            
            // 4. Render Content (Formatted JSON)
            // Try to parse json_data string if it exists
            let prettyJson = doc.json_data;
            try {
                const parsed = JSON.parse(doc.json_data);
                prettyJson = JSON.stringify(parsed, null, 2);
            } catch(e) {}
            
            detailContent.innerHTML = `
                <div style="margin-bottom:10px;"><strong>Summary:</strong><br>${doc.text}</div>
                <hr style="border-color:#444;">
                <pre style="white-space: pre-wrap; font-size: 0.75rem; color:#000;">${prettyJson}</pre>
            `;
        } else {
            detailTitle.innerText = "Error";
            detailContent.innerHTML = "Document not found.";
        }
    } catch (e) {
        detailTitle.innerText = "Error";
        detailContent.innerHTML = "Failed to load: " + e;
    }
}

// Detail Delete Button Logic
btnDetailDelete?.addEventListener("click", async () => {
    if (!currentDetailUuid) return;
    
    const confirmed = await ask("Delete this document?", { title: 'Confirm Deletion', kind: 'warning' });
    if (confirmed) {
        try {
            await invoke("delete_document", { uuid: currentDetailUuid });
            // Close detail view and refresh list
            detailView.style.display = "none";
            listView.style.display = "block";
            refreshList();
        } catch (e) {
            console.error("Delete failed:", e);
            alert("Failed to delete document.");
        }
    }
});

// Back Button Logic
btnDetailBack?.addEventListener("click", () => {
    detailView.style.display = "none";
    listView.style.display = "block";
});

btnListBack?.addEventListener("click", () => {
    collapseWidget();
});

function filterListLocally(keyword: string) {
    // Note: Local filtering with infinite scroll is tricky because we only have partial data.
    // For now, we will just filter the *loaded* data.
    // Ideal solution: Server-side search (we already have 'search_documents' for that).
    // Let's keep it simple: Filter cachedDocs.
    
    if (!docTableBody) return;
    docTableBody.innerHTML = "";
    
    let filtered = cachedDocs;
    if (keyword) {
        filtered = cachedDocs.filter(d => 
            (d.doc_type && d.doc_type.toLowerCase().includes(keyword)) ||
            (d.doc_number && d.doc_number.toLowerCase().includes(keyword)) ||
            (d.total_amount && d.total_amount.toString().includes(keyword))
        );
    }
    
    renderDocRows(filtered);
}

listRefreshBtn?.addEventListener("click", refreshList);

// Automation
const autoBtn = document.getElementById("auto-btn") as HTMLButtonElement;
const autoBrowser = document.getElementById("auto-browser") as HTMLSelectElement;
const autoUrl = document.getElementById("auto-url") as HTMLInputElement;

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
        await invoke("launch_browser", { 
            browser: autoBrowser.value, url: autoUrl.value, script: "" 
        });
    } catch (e) { console.error(e); }
});

// Listeners
listen("extraction-progress", (event: any) => {
    const payload = event.payload;
    const catId = payload.category ? payload.category.replace(/[^a-zA-Z0-9]/g, "") : "general";
    const elementId = `progress-${catId}`;

    // 1. Priority: Detail View (Extraction Log)
    const extractionLog = document.getElementById("extraction-log");
    if (extractionLog) {
         let p = document.getElementById(elementId);
         if (!p) {
             p = document.createElement("div");
             p.id = elementId;
             p.style.borderBottom = "1px solid #333";
             p.style.padding = "6px 0";
             p.style.fontSize = "0.75rem";
             p.style.display = "flex";
             p.style.alignItems = "center";
             extractionLog.appendChild(p); // Append for sequential order
         }
         
         if (payload.data) {
             // Success / Final for a category
             const successText = payload.category === "Processing" ? "Processing complete." : `<strong>${payload.category}</strong> extracted.`;
             p.innerHTML = `<span style="margin-right:8px;">✅</span> <span>${successText}</span>`;
             p.style.color = "#4ade80"; 
         } else {
             // Progress
             const progressText = payload.category === "Processing" ? "Processing..." : `Extracting ${payload.category}...`;
             p.innerHTML = `<span style="color:var(--primary); margin-right:8px; font-family:monospace; min-width:15px;">${payload.spinner}</span> <span>${progressText}</span>`;
         }
         return; 
    }

    // 2. Fallback: AI Results Area
    if (aiResultsContent && payload.spinner) {
         let p = document.getElementById(`ai-${elementId}`);
         if (!p) {
            p = document.createElement("div");
            p.id = `ai-${elementId}`;
            p.style.fontSize = "0.7rem";
            aiResultsContent.prepend(p);
         }
         p.innerText = `${payload.spinner} ${payload.category || 'Working'}...`;
    }
});

// Browser Status Listener (Active State)
listen("browser-status", (event: any) => {
    const status = event.payload; // "running" or "stopped"
    if (autoLaunchBtn) {
        if (status === "running") {
            autoLaunchBtn.classList.add("active-emoji");
            autoLaunchBtn.classList.add("active"); // Optional: reuse existing active style
        } else {
            autoLaunchBtn.classList.remove("active-emoji");
            autoLaunchBtn.classList.remove("active");
        }
    }
});
