// ============================================================
//  commerce/src/modes/oauth.ts
//  🌟 [OAUTH SITE REGISTRY] analytic 트랙 전용 사이트 등록/조회/통계 계층
//
//  ── 왜 analytic 하위인가 ──
//   api.oauth.network 는 '어떤 도메인의 행동 로그를 볼 권한이 있는가' 를 판정하는
//   자격증명 저장소입니다. console.logis.center 는 client_id 검증을 통과한 요청에만
//   로그를 돌려주므로, 이 파일이 없으면 analytic 트랙은 영구히 0건입니다.
//   commerce / shipping 트랙과는 접점이 전혀 없습니다.
// ============================================================
import { invoke } from "@tauri-apps/api/core";
import { ask } from '@tauri-apps/plugin-dialog';
import { rt } from "./runtime";

// Access global libs (index.html 의 ethers.umd.min.js)
const ethers = (window as any).ethers;

export const OAUTH_API_HOST = "https://api.oauth.network";

export function normalizeOAuthHost(raw: any): string {
    if (!raw || typeof raw !== "string") return "";
    let s = raw.trim().toLowerCase();
    if (!s) return "";
    if (!/^https?:\/\//.test(s)) s = "https://" + s;
    try {
        const u = new URL(s);
        if (!u.hostname) return "";
        return "https://" + u.host;
    } catch (_e) {
        return "";
    }
}

function extractBalancedJson(text: string, fromIndex: number): string {
    const start = text.indexOf("{", fromIndex);
    if (start === -1) return "";
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (let i = start; i < text.length; i++) {
        const ch = text[i];
        if (inStr) {
            if (esc) { esc = false; }
            else if (ch === "\\") { esc = true; }
            else if (ch === '"') { inStr = false; }
            continue;
        }
        if (ch === '"') { inStr = true; continue; }
        if (ch === "{") depth++;
        else if (ch === "}") {
            depth--;
            if (depth === 0) return text.substring(start, i + 1);
        }
    }
    return "";
}

/** api/index.js 의 HTML 응답에서 JSON 페이로드를 추출합니다. */
function parseOAuthApiResponse(raw: any): { rows: any[]; cookies: any; count: number; query: any } {
    let text = "";
    if (typeof raw === "string") {
        text = raw;
    } else if (raw && typeof raw === "object") {
        if (raw.text) {
            text = raw.text;
        } else if (raw.rows) {
            // 이미 JSON 으로 파싱된 경우 (향후 api/index.js 수정 시)
            return { rows: raw.rows || [], cookies: raw.cookies || {}, count: raw.count || 0, query: raw.query || {} };
        }
    }
    if (!text) {
        return { rows: [], cookies: {}, count: 0, query: {} };
    }
    // <script> 내부의 parent.postMessage(JSON.stringify({...}), ...) 에서 JSON 추출
    const anchor = text.indexOf("JSON.stringify(");
    const jsonText = anchor === -1
        ? extractBalancedJson(text, 0)
        : extractBalancedJson(text, anchor + "JSON.stringify(".length);
    if (!jsonText) {
        console.warn("[OAUTH] 응답에서 postMessage 페이로드를 찾지 못했습니다.");
        return { rows: [], cookies: {}, count: 0, query: {} };
    }
    try {
        const parsed = JSON.parse(jsonText);
        // ⚠️ index.js 는 cookies 를 '이중 stringify' 해서 내려줍니다.
        //    JSON.stringify(JSON.stringify(req.cookies)) → 문자열 리터럴이므로 한 번 더 파싱합니다.
        let cookies: any = {};
        if (typeof parsed.cookies === "string") {
            try { cookies = JSON.parse(parsed.cookies); } catch (_e) { cookies = {}; }
        } else if (parsed.cookies && typeof parsed.cookies === "object") {
            cookies = parsed.cookies;
        }
        return {
            rows: Array.isArray(parsed.rows) ? parsed.rows : [],
            cookies,
            count: parsed.count || 0,
            query: parsed.query || {}
        };
    } catch (e) {
        console.warn("[OAUTH] postMessage 페이로드 JSON 파싱 실패:", e);
        return { rows: [], cookies: {}, count: 0, query: {} };
    }
}

export async function oauthApiFetch(
    query: Record<string, any>,
    opts: { method?: "GET" | "POST"; body?: any } = {}
): Promise<{ rows: any[]; cookies: any; count: number; query: any }> {
    const method = opts.method || "GET";
    const sp = new URLSearchParams();
    for (const k of Object.keys(query)) {
        const v = query[k];
        if (v === undefined || v === null || v === "") continue;
        if (Array.isArray(v)) {
            for (const item of v) sp.append(k, String(item));
        } else {
            sp.append(k, String(v));
        }
    }
    const qs = sp.toString();
    const url = qs ? `${OAUTH_API_HOST}/?${qs}` : `${OAUTH_API_HOST}/`;
    const headers: Record<string, string> = {
        "Content-Type": method === "POST"
            ? "application/x-www-form-urlencoded"
            : "application/json",
        "Referer": "https://oauth.network/"
    };
    const args: any = {
        url,
        method,
        headers,
        session_params: null
    };
    if (opts.body) args.body = opts.body;
    const response = await invoke<any>("proxy_fetch", args);
    return parseOAuthApiResponse(response);
}

export async function submitOAuthRegistration(
    hostUrl: string,
    credentials?: { client_id: string; client_secret: string }
): Promise<{
    success: boolean;
    client_id: string;
    client_secret: string;
    error: string;
    removed: boolean;
}> {
    const R = rt();
    const S = R.getSession();
    const isRemove = !!(credentials && credentials.client_id && credentials.client_secret);
    if (!S.hash || !S.token) {
        return { success: false, client_id: "", client_secret: "", error: "로그인이 필요합니다.", removed: false };
    }
    const host = normalizeOAuthHost(hostUrl);
    if (!host) {
        return { success: false, client_id: "", client_secret: "", error: "도메인 형식이 올바르지 않습니다. (예: https://example.com)", removed: false };
    }
    try {
        const body: any = {
            host: host,
            hash: S.hash,
            token: S.token
        };
        if (isRemove) {
            body.client_id = credentials!.client_id;
            body.client_secret = credentials!.client_secret;
        }
        console.log(`[OAUTH] POST ${isRemove ? "삭제" : "등록/재발급"} 요청: host='${host}'`);
        const parsed = await oauthApiFetch({}, { method: "POST", body });
        if (parsed.rows.length > 0) {
            const row = parsed.rows[0];
            if (row.client_id && row.client_secret) {
                await fetchOAuthRegisteredSites();
                return {
                    success: true,
                    client_id: String(row.client_id),
                    client_secret: String(row.client_secret),
                    error: "",
                    removed: isRemove
                };
            }
        }
        const email = S.email || "";
        let expectedAddr = String(parsed.cookies?.client_id || "");
        if (!expectedAddr && email && typeof ethers !== "undefined") {
            try {
                expectedAddr = ethers.computeAddress(ethers.hashMessage(email)).toLowerCase();
            } catch (_e) { /* ignore */ }
        }
        const head = isRemove
            ? "서버 삭제 실패. 삭제도 사이트 소유 확인을 통과해야 합니다."
            : "사이트 소유 확인 실패.";
        const errMsg = expectedAddr
            ? `${head}\n사이트 <head>에 아래 세 태그가 그대로 있는지 확인하세요:\n<meta name="oauth-network-verification" content="${expectedAddr}" />\n<meta name="privacy" content="..." />\n<meta name="terms" content="..." />`
            : `${head}\n메타 태그를 사이트 <head>에 추가하세요.`;
        return { success: false, client_id: "", client_secret: "", error: errMsg, removed: isRemove };
    } catch (e: any) {
        return { success: false, client_id: "", client_secret: "", error: String(e), removed: isRemove };
    }
}

export async function fetchOAuthRegisteredSites(): Promise<void> {
    const R = rt();
    const S = R.getSession();
    if (!S.hash || !S.token) return;
    try {
        const parsed = await oauthApiFetch({
            hash: S.hash,
            token: S.token
        });
        const cookies = parsed.cookies || {};
        const authed = !!(cookies.email || cookies.client_id || cookies["#length"] !== undefined);
        if (!authed) {
            const cookieKeys = Object.keys(cookies || {});
            console.warn(
                "[OAUTH] ⚠️ 서버 세션이 성립하지 않아 목록을 갱신하지 않습니다. " +
                `(hash='${(S.hash || "").slice(0, 8)}...', ` +
                `token 유무=${!!S.token}, ` +
                `응답 cookies 키=[${cookieKeys.join(", ")}]). ` +
                "로컬 kv_store 목록은 그대로 유지합니다."
            );
            return;
        }
        const len = parseInt(cookies["#length"] || "0", 10) || 0;
        const sites: any[] = [];
        for (let i = 0; i < len; i++) {
            const raw = cookies[`#${i}`];
            if (!raw || typeof raw !== "string") continue;
            try {
                // 형식: "0x주소:프라이빗키@호스트"
                const u = new URL("https://" + raw);
                if (!u.username || !u.hostname) continue;
                sites.push({
                    host: "https://" + u.host,
                    client_id: decodeURIComponent(u.username),
                    client_secret: decodeURIComponent(u.password),
                    registered_at: Date.now()
                });
            } catch (_e) {
                continue;
            }
        }
        if (cookies.client_id) {
            await R.kvSet("oauth_client_address", String(cookies.client_id));
        }
        // 서버가 유일한 진실 공급원입니다. 병합하지 않고 통째로 교체합니다.
        await R.kvSet("oauth_registered_sites", sites);
        console.log(
            `[OAUTH] 서버 조회 완료. 등록된 사이트 ${sites.length}건 → kv_store 교체. ` +
            `${sites.map((s: any) => s.host).join(", ")}`
        );
    } catch (e) {
        console.warn("[OAUTH] fetchOAuthRegisteredSites failed:", e);
    }
}

export async function fetchOAuthSitePaths(referer: string): Promise<string[]> {
    const host = normalizeOAuthHost(referer);
    if (!host) return [];
    try {
        const parsed = await oauthApiFetch({
            referer: host,
            distinct: "Cc",
            id: "#LOG"
        });
        return parsed.rows
            .map((r: any) => r.Cc)
            .filter((c: string) => !!c);
    } catch (_e) {
        return [];
    }
}

export async function fetchOAuthSiteCount(referer: string, hoursBack: number): Promise<number> {
    const host = normalizeOAuthHost(referer);
    if (!host) return 0;
    try {
        const now = Date.now();
        const from = new Date(now - hoursBack * 3600 * 1000).toISOString();
        const to = new Date(now).toISOString();
        const parsed = await oauthApiFetch({
            referer: host,
            id: "#LOG",
            cnt: "true",
            date: [from, to]
        });
        return parsed.count || 0;
    } catch (_e) {
        return 0;
    }
}

/**
 * 🌟 [CREDENTIAL LOOKUP] analytic 트랙이 console.logis.center 를 호출할 때
 *  origin 에 대응하는 client_id / client_secret 을 찾습니다.
 *  이 값이 없으면 Worker 가 로그를 돌려주지 않으므로 analytic 이 영구히 0건이 됩니다.
 */
export async function getOAuthCredentialForOrigin(origin: string): Promise<{ client_id: string; client_secret: string } | null> {
    const R = rt();
    const toHost = (raw: any): string => {
        if (!raw || typeof raw !== "string") return "";
        let s = raw.trim().toLowerCase();
        if (!s) return "";
        if (!/^https?:\/\//.test(s)) s = "https://" + s;
        try {
            return new URL(s).host;
        } catch (_e) {
            return "";
        }
    };
    const targetHost = toHost(origin);
    if (!targetHost) return null;
    try {
        const sites = await R.kvGet("oauth_registered_sites");
        if (!Array.isArray(sites)) return null;
        for (const s of sites) {
            if (!s) continue;
            if (toHost(s.host) !== targetHost) continue;
            if (!s.client_id) continue;
            return {
                client_id: String(s.client_id),
                client_secret: String(s.client_secret || "")
            };
        }
    } catch (e) {
        console.warn("[OAUTH] getOAuthCredentialForOrigin failed:", e);
    }
    return null;
}

let isOAuthSitesRendering = false;

export async function renderOAuthSitesUI(pageList: HTMLElement) {
    const R = rt();
    if (!pageList) return;
    if (R.getSearchMode() !== "analytic") return;
    if (!R.getSession().email) return;
    if (isOAuthSitesRendering) return;
    isOAuthSitesRendering = true;
    try {
        const existingItems = pageList.querySelectorAll(".oauth-site-item");
        if (existingItems.length > 0) {
            existingItems.forEach((el: Element) => el.remove());
        }
        // 🌟 [OAUTH SYNC] 렌더링 전 서버에서 최신 목록을 조회합니다.
        //    다른 기기에서 등록/삭제한 내역도 이 시점에 반영됩니다.
        await fetchOAuthRegisteredSites();
        const registeredSites = await R.kvGet("oauth_registered_sites") || [];
        if (!Array.isArray(registeredSites) || registeredSites.length === 0) return;
        let oauthHtml = '';
        for (let si = 0; si < registeredSites.length; si++) {
            const site = registeredSites[si];
            const siteHost = site.host || "";
            const clientId = site.client_id || "";
            const clientSecret = site.client_secret || "";
            const siteId = `oauth_site_${si}`;
            let displayHost = siteHost;
            try {
                displayHost = new URL(siteHost).hostname;
            } catch (_e) {}
            oauthHtml += `
<div class="oauth-site-item" id="${siteId}" style="position:relative; padding:6px 0; border-bottom:1px solid #f0f0f0;">
    <div style="display:flex; align-items:center; gap:6px;">
        <span style="font-size:0.8rem; font-weight:600; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${displayHost}</span>
        <button class="btn-oauth-more" data-site-idx="${si}" style="background:none; border:none; cursor:pointer; font-size:10px; font-style:italic; text-decoration:underline; color:#6366f1; padding:0 4px;">more</button>
        <button class="btn-oauth-hide" data-site-idx="${si}" style="background:none; border:none; cursor:pointer; font-size:10px; text-decoration:underline; color:#888; padding:0 4px;">hide</button>
    </div>
    <div class="oauth-site-tokens" data-site-idx="${si}" style="display:none; margin-top:6px; padding:8px; background:#f8f9fa; border-radius:4px; font-size:0.68rem; line-height:1.6; word-break:break-all;">
        <div><strong>Client Id:</strong> ${clientId}</div>
        <div style="margin-top:4px;"><strong>Client Secret:</strong> ${clientSecret}</div>
        <div style="margin-top:8px; text-align:right; display:flex; gap:6px; justify-content:flex-end;">
            <button class="btn-oauth-reissue" data-site-idx="${si}" style="background:none; border:1px solid #6366f1; color:#6366f1; border-radius:4px; padding:3px 10px; font-size:0.65rem; cursor:pointer;">재발급</button>
            <button class="btn-oauth-delete" data-site-idx="${si}" style="background:none; border:1px solid #ef4444; color:#ef4444; border-radius:4px; padding:3px 10px; font-size:0.65rem; cursor:pointer;">Delete</button>
        </div>
    </div>
    <div class="oauth-site-stats" data-site-host="${siteHost}" style="margin-top:15px; border:1px solid #ddd; border-radius:8px; position:relative;">
        <span style="position:absolute; left: 8px; top: -5px; font-size:0.6rem; color:#999; font-weight:100; background:#fff; padding:0 4px;">시간별 접속량</span>
        <table style="width:100%; border-collapse:collapse;">
            <tbody>
                <tr>
                    <td class="stat-hour" style="padding:8px; width:25%; text-align:center; border-right:1px solid #ddd;">
                        <cnt><span style="font-size:1em; font-weight:bold; text-decoration:underline;">0</span><span style="display:block; margin-top:3px; font-size:10px; font-weight:100;">hour</span></cnt>
                    </td>
                    <td class="stat-day" style="padding:8px; width:25%; text-align:center; border-right:1px solid #ddd;">
                        <cnt><span style="font-size:1em; font-weight:bold; text-decoration:underline;">0</span><span style="display:block; margin-top:3px; font-size:10px; font-weight:100;">day</span></cnt>
                    </td>
                    <td class="stat-week" style="padding:8px; width:25%; text-align:center; border-right:1px solid #ddd;">
                        <cnt><span style="font-size:1em; font-weight:bold; text-decoration:underline;">0</span><span style="display:block; margin-top:3px; font-size:10px; font-weight:100;">week</span></cnt>
                    </td>
                    <td class="stat-month" style="padding:8px; width:25%; text-align:center;">
                        <cnt><span style="font-size:1em; font-weight:bold; text-decoration:underline;">0</span><span style="display:block; margin-top:3px; font-size:10px; font-weight:100;">month</span></cnt>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
</div>`;
        }
        pageList.insertAdjacentHTML("beforeend", oauthHtml);
        // "more" 버튼: 토큰 정보 + 재발급/삭제 버튼 토글
        pageList.querySelectorAll(".btn-oauth-more").forEach((btn: any) => {
            btn.onclick = (e: Event) => {
                e.preventDefault();
                e.stopPropagation();
                const idx = btn.dataset.siteIdx;
                const tokenDiv = pageList.querySelector(`.oauth-site-tokens[data-site-idx="${idx}"]`) as HTMLElement;
                if (tokenDiv) {
                    const isVisible = tokenDiv.style.display !== "none";
                    tokenDiv.style.display = isVisible ? "none" : "block";
                    btn.textContent = isVisible ? "more" : "fold";
                }
            };
        });
        // "hide" 버튼: 사이트 항목 숨김 (화면 전용, 서버 영향 없음)
        pageList.querySelectorAll(".btn-oauth-hide").forEach((btn: any) => {
            btn.onclick = async (e: Event) => {
                e.preventDefault();
                e.stopPropagation();
                const idx = btn.dataset.siteIdx;
                const siteItem = document.getElementById(`oauth_site_${idx}`);
                if (siteItem) {
                    siteItem.style.display = "none";
                }
            };
        });
        // 🌟 [REISSUE] host 만 보내면 index.js 가 재발급 분기로 진입합니다.
        pageList.querySelectorAll(".btn-oauth-reissue").forEach((btn: any) => {
            btn.onclick = async (e: Event) => {
                e.preventDefault();
                e.stopPropagation();
                const idx = parseInt(btn.dataset.siteIdx, 10);
                const sites = (await R.kvGet("oauth_registered_sites")) || [];
                if (!Array.isArray(sites) || idx >= sites.length) return;
                const site = sites[idx];
                const confirmed = await ask(
                    `'${site.host}' 의 client_id / client_secret 을 재발급하시겠습니까?\n기존 키는 즉시 무효화됩니다.`,
                    { title: "재발급 확인", kind: "warning" }
                );
                if (!confirmed) return;
                btn.textContent = "처리 중...";
                btn.disabled = true;
                const res = await submitOAuthRegistration(site.host);
                if (res.success) {
                    await R.renderNavigation();
                } else {
                    btn.textContent = "재발급";
                    btn.disabled = false;
                    alert(res.error);
                }
            };
        });
        pageList.querySelectorAll(".btn-oauth-delete").forEach((btn: any) => {
            btn.onclick = async (e: Event) => {
                e.preventDefault();
                e.stopPropagation();
                const idx = parseInt(btn.dataset.siteIdx, 10);
                const sites = (await R.kvGet("oauth_registered_sites")) || [];
                if (!Array.isArray(sites) || idx >= sites.length) return;
                const site = sites[idx];
                const confirmed = await ask(
                    `'${site.host}' 등록을 삭제하시겠습니까?\n\n` +
                    "· api.oauth.network 서버에서 client_id / client_secret 이 영구 삭제됩니다.\n" +
                    "· 서버가 삭제 시에도 소유 확인을 하므로, 사이트 <head> 의 메타 태그는 아직 남아 있어야 합니다.",
                    { title: "사이트 삭제 확인", kind: "warning" }
                );
                if (!confirmed) return;
                btn.textContent = "삭제 중...";
                btn.disabled = true;
                const res = await submitOAuthRegistration(site.host, {
                    client_id: site.client_id,
                    client_secret: site.client_secret
                });
                if (res.success) {
                    console.log(`[OAUTH] 🗑️ 서버에서 '${site.host}' 등록을 삭제했습니다.`);
                    await R.renderNavigation();
                } else {
                    btn.textContent = "Delete";
                    btn.disabled = false;
                    alert(res.error);
                }
            };
        });
        // 🌟 [STATS] 각 사이트의 시간별 접속량을 비동기로 조회합니다.
        //    hash / token 을 싣지 않으므로 index.js 의 referer 분기가 실제로 실행됩니다.
        for (let si = 0; si < registeredSites.length; si++) {
            const site = registeredSites[si];
            const siteHost = site.host || "";
            if (!siteHost) continue;
            const statsDiv = pageList.querySelector(`.oauth-site-stats[data-site-host="${siteHost}"]`) as HTMLElement;
            if (!statsDiv) continue;
            // hour: 최근 1시간
            fetchOAuthSiteCount(siteHost, 1).then(cnt => {
                const el = statsDiv.querySelector(".stat-hour cnt span") as HTMLElement;
                if (el) el.textContent = String(cnt);
            });
            // day: 최근 24시간
            fetchOAuthSiteCount(siteHost, 24).then(cnt => {
                const el = statsDiv.querySelector(".stat-day cnt span") as HTMLElement;
                if (el) el.textContent = String(cnt);
            });
            // week: 최근 7일 (168시간)
            fetchOAuthSiteCount(siteHost, 168).then(cnt => {
                const el = statsDiv.querySelector(".stat-week cnt span") as HTMLElement;
                if (el) el.textContent = String(cnt);
            });
            // month: 최근 30일 (720시간)
            fetchOAuthSiteCount(siteHost, 720).then(cnt => {
                const el = statsDiv.querySelector(".stat-month cnt span") as HTMLElement;
                if (el) el.textContent = String(cnt);
            });
        }
    } catch (oauthErr) {
        console.warn("[NAV] OAuth registered sites render failed:", oauthErr);
    } finally {
        // 🌟 [LOCK RELEASE] 예외 여부과 무관하게 락을 해제해야
        //    다음 renderNavigation() 에서 재렌더링이 가능해집니다.
        isOAuthSitesRendering = false;
    }
}

export function renderOAuthRegistrationForm() {
    const R = rt();
    const existing = document.getElementById("oauth-registration-modal");
    if (existing) existing.remove();
    const modal = document.createElement("div");
    modal.id = "oauth-registration-modal";
    modal.style.cssText = "position: fixed; inset: 121px 11px 11px; border-bottom-left-radius: 1em; border-bottom-right-radius: 1em; z-index: 99999; display: flex; align-items: center; justify-content: center; background: rgba(255, 255, 255, 0.88); pointer-events: initial;";
    modal.innerHTML = `
<div style="background:#fff;border-radius:12px;padding:24px;width:90%;max-width:480px;max-height:85vh;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <h3 style="margin:0;font-size:1.1rem;font-weight:700;">사이트 등록 (Analytic)</h3>
        <button id="oauth-modal-close" style="background:none;border:none;font-size:1.2rem;cursor:pointer;color:#666;">✕</button>
    </div>
    <!-- Step 1: 도메인 입력 -->
    <div style="margin-bottom:16px;">
        <label style="display:block;margin-bottom:6px;font-size:0.85rem;font-weight:600;">사이트 도메인</label>
        <input id="oauth-reg-host" type="url" placeholder="https://example.com" style="width:100%;padding:10px 12px;border:1px solid #ddd;border-radius:6px;box-sizing:border-box;font-size:0.9rem;">
        <p style="margin:4px 0 0;font-size:0.75rem;color:#888;">예시: (O) https://example.com (X) https://www.example.com</p>
    </div>
    <!-- Step 2: 소유 확인 메타 태그 (도메인 입력 시 즉시 생성) -->
    <div id="oauth-reg-meta" style="display:none;margin-bottom:16px;">
        <label style="display:block;margin-bottom:6px;font-size:0.85rem;font-weight:600;">사이트 소유 확인 메타 태그</label>
        <p style="margin:0 0 6px;font-size:0.75rem;color:#666;">
            아래 메타 태그를 복사하여 사이트 홈페이지의 <code>&lt;head&gt;</code> 섹션에 붙여넣으세요.<br>
            등록 버튼 클릭 시 서버가 이 태그의 존재 여부를 검증합니다.
        </p>
        <textarea id="oauth-reg-meta-tag" readonly style="width:100%;min-height:100px;padding:10px;border:1px solid #ddd;border-radius:6px;font-size:0.72rem;resize:none;box-sizing:border-box;background:#f8f9fa;line-height:1.6;"></textarea>
        <button id="oauth-meta-copy" style="margin-top:6px;padding:6px 14px;border:1px solid #6366f1;border-radius:4px;background:#fff;color:#6366f1;font-size:0.78rem;cursor:pointer;">태그 복사</button>
    </div>
    <!-- 결과 메시지 -->
    <div id="oauth-reg-result" style="display:none;margin-bottom:12px;padding:12px;border-radius:6px;font-size:0.85rem;"></div>
    <!-- 등록 성공 후 Client 정보 -->
    <div id="oauth-reg-credentials" style="display:none;margin-bottom:16px;">
        <label style="display:block;margin-bottom:6px;font-size:0.85rem;font-weight:600;">Client Id (Address)</label>
        <input id="oauth-reg-client-id" readonly style="width:100%;padding:8px 12px;border:1px solid #ddd;border-radius:6px;box-sizing:border-box;font-size:0.8rem;margin-bottom:8px;">
        <label style="display:block;margin-bottom:6px;font-size:0.85rem;font-weight:600;">Client Secret (Private Key)</label>
        <input id="oauth-reg-client-secret" readonly style="width:100%;padding:8px 12px;border:1px solid #ddd;border-radius:6px;box-sizing:border-box;font-size:0.8rem;">
    </div>
    <button id="oauth-reg-submit" style="width:100%;padding:12px;border:none;border-radius:8px;background:#eee;color:#000;font-size:0.95rem;font-weight:700;cursor:pointer;">추가</button>
</div>`;
    document.body.appendChild(modal);
    // 닫기
    document.getElementById("oauth-modal-close")!.addEventListener("click", () => modal.remove());
    modal.addEventListener("click", (e) => { if (e.target === modal) modal.remove(); });
    const hostInput = document.getElementById("oauth-reg-host") as HTMLInputElement;
    const metaDiv = document.getElementById("oauth-reg-meta") as HTMLDivElement;
    const metaTextarea = document.getElementById("oauth-reg-meta-tag") as HTMLTextAreaElement;
    const metaCopyBtn = document.getElementById("oauth-meta-copy") as HTMLButtonElement;
    let metaDebounce: number | null = null;
    hostInput.addEventListener("input", () => {
        if (metaDebounce) clearTimeout(metaDebounce);
        metaDebounce = window.setTimeout(() => {
            const url = hostInput.value.trim();
            if (!url || !url.startsWith("http")) {
                metaDiv.style.display = "none";
                return;
            }
            try {
                const email = R.getSession().email || "";
                if (!email) {
                    metaDiv.style.display = "none";
                    return;
                }
                const hashMessage = ethers.hashMessage(email);
                const clientAddress = ethers.computeAddress(hashMessage).toLowerCase();
                // 🌟 [DEBUG] 생성된 주소를 콘솔에 출력하여 사이트 태그와 대조 가능하게 합니다.
                console.log(`[OAUTH] Meta tag generated for email='${email}' → content="${clientAddress}"`);
                const tags = `<meta name="oauth-network-verification" content="${clientAddress}" />
    <meta name="privacy" content="/개인정보약관 경로/" />
    <meta name="terms" content="/이용약관 경로/" />`;
                metaTextarea.value = tags;
                metaDiv.style.display = "block";
            } catch (e) {
                metaDiv.style.display = "none";
            }
        }, 300);
    });
    // 태그 복사 버튼
    metaCopyBtn.addEventListener("click", () => {
        const text = metaTextarea.value;
        if (!text) return;
        navigator.clipboard.writeText(text).then(() => {
            metaCopyBtn.textContent = "복사됨 ✓";
            setTimeout(() => { metaCopyBtn.textContent = "태그 복사"; }, 2000);
        }).catch(() => {
            // clipboard API 실패 시 textarea 선택 방식
            metaTextarea.focus();
            metaTextarea.select();
            document.execCommand("copy");
            metaCopyBtn.textContent = "복사됨 ✓";
            setTimeout(() => { metaCopyBtn.textContent = "태그 복사"; }, 2000);
        });
    });
    // 제출
    document.getElementById("oauth-reg-submit")!.addEventListener("click", async () => {
        const resultDiv = document.getElementById("oauth-reg-result") as HTMLDivElement;
        const credDiv = document.getElementById("oauth-reg-credentials") as HTMLDivElement;
        const submitBtn = document.getElementById("oauth-reg-submit") as HTMLButtonElement;
        const hostUrl = hostInput.value.trim();
        if (!hostUrl) {
            resultDiv.style.display = "block";
            resultDiv.style.background = "#fef2f2";
            resultDiv.style.color = "#dc2626";
            resultDiv.textContent = "도메인을 입력하세요.";
            return;
        }
        // 메타 태그가 아직 생성되지 않은 경우 생성 유도
        if (metaDiv.style.display === "none") {
            const email = R.getSession().email || "";
            const hashMessage = ethers.hashMessage(email);
            const clientAddress = ethers.computeAddress(hashMessage).toLowerCase();
            metaTextarea.value = `<meta name="oauth-network-verification" content="${clientAddress}" />
<meta name="privacy" content="/개인정보약관 경로/" />
<meta name="terms" content="/이용약관 경로/" />`;
            metaDiv.style.display = "block";
            resultDiv.style.display = "block";
            resultDiv.style.background = "#fffbeb";
            resultDiv.style.color = "#d97706";
            resultDiv.textContent = "위 메타 태그를 사이트 <head>에 추가한 후 다시 등록하세요.";
            return;
        }
        submitBtn.disabled = true;
        submitBtn.textContent = "등록 중...";
        resultDiv.style.display = "none";
        const res = await submitOAuthRegistration(hostUrl);
        submitBtn.disabled = false;
        submitBtn.textContent = "추가";
        if (res.success) {
            resultDiv.style.display = "block";
            resultDiv.style.background = "#f0fdf4";
            resultDiv.style.color = "#16a34a";
            resultDiv.textContent = "등록 완료!";
            metaDiv.style.display = "none";
            credDiv.style.display = "block";
            (document.getElementById("oauth-reg-client-id") as HTMLInputElement).value = res.client_id;
            (document.getElementById("oauth-reg-client-secret") as HTMLInputElement).value = res.client_secret;
            hostInput.value = "";
            // 네비게이션 갱신 (Pages 섹션에 등록된 사이트가 즉시 반영됩니다)
            await R.renderNavigation();
        } else {
            resultDiv.style.display = "block";
            resultDiv.style.background = "#fef2f2";
            resultDiv.style.color = "#dc2626";
            resultDiv.textContent = res.error;
            // 소유 확인 실패 시 메타 태그를 다시 강조 표시
            metaDiv.style.display = "block";
            metaTextarea.style.border = "2px solid #ef4444";
            setTimeout(() => { metaTextarea.style.border = "1px solid #ddd"; }, 3000);
        }
    });
}