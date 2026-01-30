import "../styles.css";

console.log("[MOBILE] V126 Logic Start");

const logPanel = document.getElementById("log-panel");
function debug(m: string) {
    if (logPanel) {
        const d = document.createElement("div");
        d.textContent = `> ${m}`;
        logPanel.appendChild(d);
        logPanel.scrollTop = logPanel.scrollHeight;
    }
}

// --- Global Handlers for HTML ---
(window as any).switchTab = (name: string) => {
    debug("Switch Tab: " + name);
    document.getElementById("tab-list")?.classList.toggle("hidden", name !== "list");
    document.getElementById("tab-settings")?.classList.toggle("hidden", name !== "settings");
};

(window as any).startCamera = async () => {
    debug("Camera Clicked");
    const overlay = document.getElementById("scanner-overlay");
    const video = document.getElementById("v") as HTMLVideoElement;
    if (!overlay || !video) return;

    try {
        overlay.classList.remove("hidden");
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        video.srcObject = stream;
        video.play();
        debug("Camera Active");
    } catch (e) {
        debug("Camera Err: " + e);
    }
};

(window as any).stopCamera = () => {
    debug("Stopping Camera");
    const video = document.getElementById("v") as HTMLVideoElement;
    if (video.srcObject) {
        (video.srcObject as MediaStream).getTracks().forEach(t => t.stop());
    }
    document.getElementById("scanner-overlay")?.classList.add("hidden");
};

(window as any).sendChat = () => {
    const input = document.getElementById("chat-input") as HTMLInputElement;
    if (!input || !input.value) return;
    debug("User: " + input.value);
    input.value = "";
};

debug("System Ready. V126");