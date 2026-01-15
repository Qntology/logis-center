import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { open, ask } from '@tauri-apps/plugin-dialog';
import { listen } from '@tauri-apps/api/event';
import { readFile } from '@tauri-apps/plugin-fs';

// Access global ethers loaded via <script> tag
const ethers = (window as any).ethers;
const QRCode = (window as any).QRCode;
const marked = (window as any).marked;
const pako = (window as any).pako;
const blockies = (window as any).blockies;

// --- CRC32 ---
function crc32(s: string) { var polynomial = arguments.length < 2 ? 0x04C11DB7 : arguments[1], initialValue = arguments.length < 3 ? 0xFFFFFFFF : arguments[2], finalXORValue = arguments.length < 4 ? 0xFFFFFFFF : arguments[3], crc = initialValue, table = [], i, j, c; function reverse(x:number, n:number) { var b = 0; while (n) { b = b * 2 + x % 2; x /= 2; x -= x % 1; n--; } return b; } for (i = 256; i >= 0; i--) { c = reverse(i, 32); for (j = 0; j < 8; j++) { c = ((c * 2) ^ (((c >>> 31) % 2) * polynomial)) >>> 0; } table[i] = reverse(c, 32); } for (i = 0; i < s.length; i++) { c = s.charCodeAt(i); if (c > 255) { throw new RangeError(); } j = (crc % 256) ^ c; crc = ((crc / 256) ^ table[j]) >>> 0; } return (crc ^ finalXORValue) >>> 0; }

// --- Global Config & State ---
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

let currentSession: ChatSession = {
    hash: "",
    cc: "logis.center" 
};

let isExpanded = false;
let currentTab = "list";
let currentImage: string | null = null;
let searchDebounceTimer: number | null = null;
let chatPollInterval: number | null = null;

// --- Helpers ---
async function hashId(text?: string): Promise<string> {
    if (!ethers) return "";
    if (typeof text === "undefined") {
        const account = ethers.Wallet.createRandom();
        text = account.privateKey;
    }
    const hashMessage = ethers.hashMessage(text);
    return ethers.computeAddress(hashMessage).toLowerCase();
}

async function reqUrl(baseParams: any = {}): Promise<string> {
    const origin = encodeURIComponent(window.location.origin);
    const href = encodeURIComponent(window.location.href);
    const created_at = Date.now();
    const crons = encodeURIComponent("[]"); 
    
    const pathname = new URL(window.location.href).pathname.toLowerCase();
    const cc = currentSession.cc || "logis.center";
    const to = await hashId(cc + pathname);

    let query = `origin=${origin}&created_at=${created_at}&href=${href}&crons=${crons}&to=${to}`;
    
    if (currentSession.hash) query += `&hash=${currentSession.hash}`;
    if (currentSession.token) query += `&token=${currentSession.token}`;
    
    for (const key in baseParams) {
        if (baseParams[key]) {
            query += `&${key}=${encodeURIComponent(baseParams[key])}`;
        }
    }

    return `${API_HOST}/?${query}`;
}

// --- Initialization ---
async function initSession() {
    if (!ethers) { setTimeout(initSession, 100); return; }

    let hash = localStorage.getItem("device_hash");
    let token = localStorage.getItem("device_token");
    
    if (!hash || !token) {
        const wallet = ethers.Wallet.createRandom();
        hash = wallet.address.toLowerCase().replace("0x", "");
        token = wallet.privateKey.toLowerCase().replace("0x", "");
        
        localStorage.setItem("device_hash", hash);
        localStorage.setItem("device_token", token);
    }
    
    currentSession.hash = hash;
    if (token) currentSession.token = token;

    const cached = localStorage.getItem("chat_session_details");
    if (cached) {
        try {
            const parsed = JSON.parse(cached);
            if (parsed.email) {
                currentSession = { ...currentSession, ...parsed };
            }
        } catch(e) {}
    }
    
    updateAuthUI();
    startChatPolling();
    initNavCategories();
}

function updateAuthUI() {
    const isLoggedIn = !!currentSession.email;
    const name = currentSession.name || "Sign In";
    const team = currentSession.team || "";
    
    // Update Nav Profile
    const navName = document.getElementById("nav-profile-name");
    const navSignin = document.getElementById("nav-signin");
    const navSignout = document.getElementById("nav-signout");
    const navEdit = document.getElementById("nav-profile-edit");
    const navFavicon = document.getElementById("nav-profile-favicon");

    if (navName) navName.innerText = name;
    
    // Generate Blockies Icon
    if (blockies && currentSession.address && navFavicon) {
        const icon = blockies.create({ seed: currentSession.address.toLowerCase() }).toDataURL();
        navFavicon.style.backgroundImage = `url(${icon})`;
    }

    if (isLoggedIn) {
        navSignin?.classList.add("hidden");
        navSignout?.classList.remove("hidden");
        navEdit?.classList.remove("hidden");
        
        const authStatus = document.getElementById("auth-status-text");
        if(authStatus) authStatus.innerText = `🟢 ${name} (${team})`;
        
        document.getElementById("btn-qr-auth")?.classList.add("hidden");
        document.querySelector('.qrcode')?.classList.remove("active"); // Hide QR
        document.getElementById("btn-logout")?.classList.remove("hidden");
        document.querySelector('form[name="chat-form"]')?.classList.remove("hidden");
    } else {
        navSignin?.classList.remove("hidden");
        navSignout?.classList.add("hidden");
        navEdit?.classList.add("hidden");
        
        const authStatus = document.getElementById("auth-status-text");
        if(authStatus) authStatus.innerText = "🔴 Anonymous (Logs Only)";
        
        document.getElementById("btn-qr-auth")?.classList.remove("hidden");
        document.getElementById("btn-logout")?.classList.add("hidden");
        document.querySelector('form[name="chat-form"]')?.classList.add("hidden");
    }
}

// --- UI Elements ---
const contentPanel = document.getElementById("content-panel") as HTMLElement;
const searchInput = document.getElementById("global-search") as HTMLInputElement;
const navCategories = document.getElementById("nav-categories") as HTMLElement;
const settingsBtn = document.getElementById("btn-settings") as HTMLButtonElement;
const tabContents = document.querySelectorAll<HTMLElement>(".tab-content");

const chatTalks = document.querySelector('.chat-talks') as HTMLElement;
const chatForm = document.querySelector('form[name="chat-form"]') as HTMLFormElement;
const chatInput = chatForm?.querySelector('input[name="talk"]') as HTMLInputElement;

// --- Window & Tabs ---
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
        // Trigger QR Auth if not logged in
        if (!currentSession.email) {
            performQrAuth();
        }
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

// --- Navigation Logic ---
searchInput?.addEventListener("focus", () => {
    openWidget("list");
    if (navCategories) {
        navCategories.classList.remove("hidden");
        setTimeout(() => navCategories.classList.add("visible"), 10);
    }
});

searchInput?.addEventListener("input", () => {
    if (navCategories && searchInput.value.length > 0) {
        navCategories.classList.remove("visible");
        setTimeout(() => navCategories.classList.add("hidden"), 200);
    } else if (navCategories && searchInput.value.length === 0) {
        navCategories.classList.remove("hidden");
        setTimeout(() => navCategories.classList.add("visible"), 10);
    }

    if(searchDebounceTimer) clearTimeout(searchDebounceTimer);
    searchDebounceTimer = window.setTimeout(() => {
        const keyword = searchInput.value.toLowerCase();
        filterListLocally(keyword);
    }, 1000);
});

function initNavCategories() {
    const createItem = (text: string, listId: string) => {
        const list = document.getElementById(listId);
        if (!list) return;
        const div = document.createElement("div");
        div.className = "nav-item";
        div.innerText = text;
        div.addEventListener("click", () => {
            searchInput.value = `Category: ${text}`;
            filterListLocally(text.toLowerCase());
            navCategories.classList.remove("visible");
            setTimeout(() => navCategories.classList.add("hidden"), 200);
        });
        list.appendChild(div);
    };

    ["Premium", "Standard", "Free"].forEach(m => createItem(m, "nav-list-membership"));
    ["Dashboard", "Orders", "Products"].forEach(p => createItem(p, "nav-list-pages"));
    ["Alice", "Bob", "Charlie"].forEach(u => createItem(u, "nav-list-users"));

    document.getElementById("nav-signin")?.addEventListener("click", performQrAuth);
    document.getElementById("nav-signout")?.addEventListener("click", performLogout);
}

// --- Auth Logic ---
function performQrAuth() {
    if (!currentSession.hash) return;
    const email = `${currentSession.hash}.logis.center@oauth.email`;
    const subject = "Login Request";
    const body = `Hash: ${currentSession.hash}`;
    const mailto = `mailto:${encodeURIComponent(email)}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
    
    // Open Mail Client
    window.location.href = mailto;
    document.getElementById("auth-status-text")!.innerText = "📧 Check email & scan QR...";

    // Show QR Code
    const qrContainer = document.querySelector('.qrcode') as HTMLElement;
    const btnQr = document.getElementById("btn-qr-auth");
    
    if (qrContainer && QRCode) {
        qrContainer.innerHTML = ""; // Clear prev
        qrContainer.style.display = "block"; // Make visible
        new QRCode(qrContainer, {
            text: mailto,
            width: 200,
            height: 200,
            colorDark : "#000000",
            colorLight : "#ffffff",
            correctLevel : QRCode.CorrectLevel.H
        });
    }
}

function performLogout() {
    localStorage.removeItem("chat_session_details");
    const hash = currentSession.hash;
    const token = currentSession.token;
    currentSession = { hash, token, cc: "logis.center" };
    updateAuthUI();
    
    // Hide QR
    const qrContainer = document.querySelector('.qrcode') as HTMLElement;
    if(qrContainer) qrContainer.style.display = "none";
}

document.getElementById("btn-qr-auth")?.addEventListener("click", performQrAuth);
document.getElementById("btn-logout")?.addEventListener("click", performLogout);

// --- Polling & Chat ---
function startChatPolling() {
    if (chatPollInterval) return;
    pollServer();
    chatPollInterval = window.setInterval(pollServer, 3000);
}

async function pollServer() {
    try {
        const url = await reqUrl({ type: 'talks' }); 
        const res = await fetch(url, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (res.ok) {
            const data = await res.json();
            
            // 1. Session Update
            if (data.session) {
                if (data.session.hash) {
                    currentSession.hash = data.session.hash;
                    localStorage.setItem("device_hash", data.session.hash);
                }
                if (data.session.token) {
                    currentSession.token = data.session.token;
                    localStorage.setItem("device_token", data.session.token);
                }
                
                if (data.session.address) {
                    currentSession.address = data.session.address;
                    currentSession.email = data.session.email;
                    currentSession.team = data.session.team;
                    currentSession.name = data.session.name;
                    currentSession.cc = data.session.cc;
                    currentSession.sender = data.session.sender;
                    
                    localStorage.setItem("chat_session_details", JSON.stringify(currentSession));
                    updateAuthUI();
                }
            }
            
            // 2. Process Talks (pako ungzip)
            if (data.results && Array.isArray(data.results)) {
                processResults(data.results);
            }
        }
    } catch (e) { console.error("Poll Error:", e); }
}

function processResults(results: any[]) {
    const talks = results.filter(item => item.table === 'talks' || item.type === 'talk');
    talks.sort((a, b) => a.created_at - b.created_at);
    
    if (chatTalks) {
        chatTalks.innerHTML = '';
        talks.forEach(talk => {
            let text = "Message";
            
            // Decompress logic
            if (talk.data) {
                try {
                    // If talk.data is an object/array (buffer representation), convert to Uint8Array
                    let buffer: Uint8Array;
                    if (Array.isArray(talk.data)) {
                        buffer = new Uint8Array(talk.data);
                    } else if (typeof talk.data === 'object' && !Array.isArray(talk.data)) {
                        // Handle object like {0: 31, 1: 139...}
                        const values = Object.values(talk.data) as number[];
                        buffer = new Uint8Array(values);
                    } else {
                        // Fallback if string or other
                        // If base64 string, decode it? For now assume buffer object.
                        buffer = new Uint8Array([]); 
                    }

                    if (pako && buffer.length > 0) {
                        const decompressed = pako.ungzip(buffer);
                        const jsonStr = new TextDecoder('utf-8').decode(decompressed);
                        const parsed = JSON.parse(jsonStr);
                        
                        if (parsed.text) text = parsed.text;
                        else if (parsed.markdown) text = parsed.markdown;
                    } else {
                        // If no pako or empty, maybe it is a plain string?
                        if (typeof talk.data === 'string') text = talk.data;
                    }
                } catch(e) {
                    // Decompression failed or parse error
                    console.error("Decompress Error", e);
                    if (typeof talk.data === 'string') text = talk.data;
                }
            } else if (talk.text) {
                text = talk.text;
            }
            
            addChatMessage(text, talk.from === currentSession.address ? 'user' : 'system', talk.from);
        });
    }
}

if (chatForm) {
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text || !currentSession.address) return;

        addChatMessage(text, 'user', currentSession.name);
        chatInput.value = "";

        try {
            const url = await reqUrl({
                from: currentSession.address,
                to: currentSession.team,
                text: text 
            });
            await fetch(url, { method: 'PUT' });
        } catch (e) { console.error("Send Error:", e); }
    });
}

function addChatMessage(text: string, type: 'user' | 'system', senderName?: string) {
    if (!chatTalks) return;
    const div = document.createElement('div');
    div.classList.add('chat-talk', type);
    
    const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    const nameLabel = senderName ? `<div style="font-size:0.7rem; margin-bottom:2px; opacity:0.7;">${senderName}</div>` : '';

    // Markdown Render
    let contentHtml = text;
    if (marked && (window as any).marked.parse) {
        // Handle both marked() and marked.parse() styles
        try { contentHtml = (window as any).marked.parse(text); } catch(e) { contentHtml = text; }
    } else if (marked && typeof marked === 'function') {
        try { contentHtml = marked(text); } catch(e) { contentHtml = text; }
    }

    div.innerHTML = `${type === 'system' ? nameLabel : ''}<div class="chat-message">${contentHtml}</div><div class="chat-created-at">${time}</div>`;
    chatTalks.prepend(div);
}

// --- Previous Logic (Drag, Auto, Etc) ---
const navPreviewContainer = document.getElementById("nav-preview-container") as HTMLElement;
const navImgThumbnail = document.getElementById("nav-img-thumbnail") as HTMLImageElement;
const navImgClear = document.getElementById("nav-img-clear") as HTMLButtonElement;

// Drag Logic
const pillNav = document.querySelector('.pill-nav') as HTMLElement;
if (pillNav) {
    let isMouseDown = false;
    let startX = 0, startY = 0;
    pillNav.addEventListener('mousedown', (e) => {
        if (!(e.target as HTMLElement).closest('button, input') && e.button === 0) {
             isMouseDown = true; startX = e.clientX; startY = e.clientY;
        }
    });
    window.addEventListener('mousemove', (e) => {
        if (!isMouseDown) return;
        if (Math.abs(e.clientX - startX) > 5 || Math.abs(e.clientY - startY) > 5) {
            isMouseDown = false; invoke('start_drag').catch(console.error);
        }
    });
    window.addEventListener('mouseup', () => isMouseDown = false);
    pillNav.addEventListener('dblclick', (e) => {
        if (!(e.target as HTMLElement).closest('button, input')) invoke("move_to_top_center").catch(console.error);
    });
}

// Settings Button
settingsBtn?.addEventListener("click", () => {
    if (currentTab === "settings" && isExpanded) collapseWidget();
    else openWidget("settings");
});
document.getElementById("nav-to-auto")?.addEventListener("click", () => switchTab("automation"));
document.getElementById("nav-back-list")?.addEventListener("click", () => switchTab("list"));
document.getElementById("btn-settings-back")?.addEventListener("click", () => switchTab("list"));

// List Logic (Stub)
let cachedDocs: any[] = [];
let currentPage = 0;
const pageSize = 10;
let isLoading = false;
let hasMore = true;
let selectedUuids = new Set<string>();

const docTableBody = document.getElementById("doc-tbody") as HTMLElement;
const listRefreshBtn = document.getElementById("list-refresh-btn") as HTMLButtonElement;
const btnDeleteSelected = document.getElementById("btn-delete-selected") as HTMLButtonElement;
const selectAllCheckbox = document.getElementById("select-all-checkbox") as HTMLInputElement;
const listScrollContainer = document.getElementById("list-scroll-container") as HTMLElement;
const loadingIndicator = document.getElementById("loading-indicator") as HTMLElement;

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

selectAllCheckbox?.addEventListener("change", () => {
    const isChecked = selectAllCheckbox.checked;
    docTableBody.querySelectorAll(".row-checkbox").forEach((cb:any) => {
        cb.checked = isChecked;
        const uuid = cb.closest("tr")?.dataset.uuid;
        if (uuid) { if (isChecked) selectedUuids.add(uuid); else selectedUuids.delete(uuid); }
    });
    updateBulkDeleteUI();
});

listRefreshBtn?.addEventListener("click", refreshList);
listScrollContainer?.addEventListener("scroll", () => {
    if (listScrollContainer.scrollTop + listScrollContainer.clientHeight >= listScrollContainer.scrollHeight - 20) loadMoreDocs();
});

async function showDetail(uuid: string) { /* Detail Logic */ }
async function initBrowserDropdown() { /* Automation Logic */ }

const extractBtnNav = document.getElementById("btn-extract") as HTMLButtonElement;
extractBtnNav?.addEventListener("click", async () => { /* Extraction Logic */ });

const autoBtn = document.getElementById("auto-btn") as HTMLButtonElement;
autoBtn?.addEventListener("click", async () => { /* Auto Logic */ });

// Init
initSession();
setWindowSize(false);
