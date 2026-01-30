console.log("%c[MOBILE] V116 Layout Active", "color: #fb923c; font-weight: bold;");

// @ts-ignore
import { item2html } from "./lib/render";

// --- State ---
let activeTab = "list";

// --- Elements ---
const tabList = document.getElementById("tab-list");
const tabSettings = document.getElementById("tab-settings");
const docListContainer = document.getElementById("doc-list");
const navPagesContainer = document.getElementById("nav-list-pages");

// --- Tab Controller ---
function switchTab(name: string) {
    activeTab = name;
    if (name === "list") {
        tabList?.classList.add("active");
        tabSettings?.classList.remove("active");
    } else if (name === "settings") {
        tabList?.classList.remove("active");
        tabSettings?.classList.add("active");
    }
}

// --- Bridge Logic ---
window.addEventListener("message", (event) => {
    const msg = event.data;
    if (!msg || !msg.type) return;

    if (msg.type === "sync_list") {
        renderDocs(msg.data);
        // Handle Tree Render (Simplified mock for Step 1)
        if (navPagesContainer) {
            navPagesContainer.innerHTML = `<ul class="logis-branch"><li class="logis-label"><span>Linked Desktop Pages</span></li></ul>`;
        }
    }
});

function renderDocs(docs: any[]) {
    if (!docListContainer) return;
    docListContainer.innerHTML = "";
    docs.forEach(doc => {
        const html = item2html(doc, false, "");
        const temp = document.createElement("div");
        temp.innerHTML = html;
        const el = temp.firstElementChild as HTMLElement;
        docListContainer.appendChild(el);
    });
}

// --- Event Bindings ---
document.getElementById("btn-settings")?.addEventListener("click", () => switchTab("settings"));
document.getElementById("btn-settings-back")?.addEventListener("click", () => switchTab("list"));

// Scanner Logic (Keep existing functionality)
const scannerOverlay = document.getElementById("scanner-overlay");
document.getElementById("btn-sync-toggle")?.addEventListener("click", () => scannerOverlay?.classList.remove("hidden"));
document.getElementById("btn-cancel-scan")?.addEventListener("click", () => scannerOverlay?.classList.add("hidden"));

// Initialize
switchTab("list");