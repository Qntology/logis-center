// ============================================================
//  commerce/src/modes/analytic.ts
//  🌟 [ANALYTIC TRACK] 사용자 행동 로그 수집 · 구조화 계층
//
//  ── 파이프라인 ──
//   ① fetchAnalyticsOrigin : console.logis.center 에서 원시 이벤트를 당겨옵니다.
//      (client_id 검증을 통과해야 하므로 oauth.ts 의 자격증명이 필수입니다)
//   ② runAnalyticStructuring : 원시 이벤트를 Qwen 이 시맨틱 문장으로 구조화합니다.
//   ③ finalizeAnalyticBubbles : 구조화 진행 말풍선을 Done 으로 마감합니다.
//   ④ runLocalEmbeddingSync : 구조화된 문장을 로컬에서 임베딩합니다. (main.ts 소유)
//   ①만 서버가 필요합니다. 세션이 없으면 폴링이 runAnalyticLocalTick 으로 ②~④만 돌립니다.
//
//  ── DELTA GUARD 계약 ──
//   updated_at == 0  → 아직 구조화되지 않은 원시 이벤트
//   updated_at >  0  → 구조화 완료. 서버 원시 데이터로 절대 덮어쓰지 않습니다.
// ============================================================
import { invoke } from "@tauri-apps/api/core";
import { hashId } from "../lib/utils";
import { modeOfType } from "./types";
import { rt, decodeAnalyticBlob, getSyncIntervalMs } from "./runtime";
import { getOAuthCredentialForOrigin } from "./oauth";

// 🌟 [ANALYTICS TRACK] 관리자(Console) 기능 및 사용자 행동 로그 동기화 전용 Client Worker
export const ANALYTIC_API_HOST = "https://console.logis.center";

let isAnalyticsSyncRunning = false;
let lastAnalyticsSyncAt = 0;
let isAnalyticStructuring = false;

/**
 * 🌟 [THROTTLE RESET] analytic 탭으로 전환한 직후 폴링 주기를 기다리지 않고
 *  즉시 1회 당겨오기 위한 스로틀 해제입니다.
 *  (구 main.ts 의 `lastAnalyticsSyncAt = 0;` 직접 대입을 대체합니다)
 */
export function resetAnalyticThrottle() {
    lastAnalyticsSyncAt = 0;
}

export async function finalizeAnalyticBubbles(processedCount: number): Promise<void> {
    const chatTalks = document.querySelector('.chat-talks') as HTMLElement;
    if (!chatTalks) return;
    const bubbles = chatTalks.querySelectorAll('.chat-talk.task-bubble') as NodeListOf<HTMLElement>;
    let finalized = 0;
    for (const el of Array.from(bubbles)) {
        const taskId = el.dataset.taskId || el.id || "";
        // analytic_sync 로 시작하는 말풍선만 대상
        if (!taskId.startsWith("analytic_sync")) continue;
        const status = parseInt(el.dataset.status || "0");
        // 이미 완료/에러/정지된 것은 건드리지 않음
        if ([2, 3, 6, 9].includes(status)) continue;
        // status → 9 (Done)
        el.dataset.status = "9";
        const contentEl = el.querySelector('.content') as HTMLElement;
        if (contentEl) {
            contentEl.textContent = `Analytic structuring complete (${processedCount} event(s)).`;
        }
        const statusBar = el.querySelector('.status-bar') as HTMLElement;
        if (statusBar) {
            statusBar.style.color = "#22c55e";
            statusBar.innerHTML = `<span>✅</span> DONE`;
        }
        // active-spinner 클래스 제거
        const spinner = el.querySelector('.active-spinner') as HTMLElement;
        if (spinner) spinner.classList.remove('active-spinner');
        finalized++;
    }
    if (finalized > 0) {
        console.log(`[ANALYTIC] ✅ ${finalized}개의 구조화 말풍선을 Done 상태로 전환했습니다.`);
    }
}

export async function runAnalyticStructuring(): Promise<number> {
    const R = rt();
    if (isAnalyticStructuring) return 0;
    if (R.isBusy()) return 0;
    isAnalyticStructuring = true;
    try {
        const res = await invoke<any>("structure_pending_analytics", {
            limit: 20,
            devicePreference: R.getDevicePref(),
        });
        const processed = res && res.processed ? res.processed : 0;
        if (processed > 0) {
            console.log(
                `[ANALYTIC] 🧠 ${processed}건의 행동 로그를 시맨틱 문장으로 구조화했습니다.`
            );
            if (R.appDb) {
                try {
                    const nowTs = Date.now();
                    const rows = await R.appDb
                        .table("items")
                        .where("mode")
                        .equals("analytic")
                        .filter((r: any) => Number(r.updated_at || 0) === 0)
                        .toArray();
                    for (const r of rows) {
                        if (
                            r.data &&
                            (r.data.summary ||
                                (typeof r.data.action === "string" && r.data.action))
                        ) {
                            await R.appDb
                                .table("items")
                                .where("id")
                                .equals(r.id)
                                .modify({
                                    updated_at: nowTs,
                                    "data.updated_at": nowTs,
                                });
                        }
                    }
                    console.log(
                        `[ANALYTIC] Dexie updated_at 갱신 완료 (후보 ${rows.length}건)`
                    );
                } catch (e) {
                    console.warn("[ANALYTIC] Dexie updated_at sync failed:", e);
                }
            }
            if (R.getSearchMode() === "analytic") {
                await R.renderNavigation();
                if (R.getCurrentTab() === "list") {
                    await R.loadMoreDocs(false, true);
                }
            }
        } else if (res && res.skipped) {
            console.log(`[ANALYTIC] 구조화 스킵: 사유=${res.skipped}`);
        }
        return processed;
    } catch (e) {
        console.warn("[ANALYTIC] structure_pending_analytics failed:", e);
        return 0;
    } finally {
        isAnalyticStructuring = false;
    }
}

async function resolveAnalyticsOrigins(): Promise<string[]> {
    const R = rt();
    const origins = new Set<string>();
    const push = (raw: any) => {
        if (!raw || typeof raw !== "string") return;
        let s = raw.trim().toLowerCase();
        if (!s) return;
        if (!/^https?:\/\//.test(s)) s = "https://" + s;
        try {
            const u = new URL(s);
            if (!u.hostname) return;
            if (u.hostname === "localhost" || u.hostname === "127.0.0.1") return;
            if (u.hostname.endsWith("logis.center")) return;
            origins.add(u.origin);
        } catch (e) {}
    };
    try {
        const sites = await R.kvGet("oauth_registered_sites");
        if (Array.isArray(sites)) {
            for (const s of sites) push(s && s.host);
        }
    } catch (e) {}
    push(R.getDetectedUrl());
    try {
        if (R.appDb) {
            const rows = await R.appDb.table("items").where("mode").equals("analytic").limit(2000).toArray();
            for (const r of rows) push(r && r.data && r.data.origin);
        }
    } catch (e) {}
    return Array.from(origins);
}

export function extractAnalyticText(parsed: any): string {
    const pick = (v: any): string => (typeof v === "string" ? v.trim() : "");
    return pick(parsed?.action)
        || pick(parsed?.summary)
        || pick(parsed?.cross_action_flow)
        || pick(parsed?.intent_evolution)
        || pick(parsed?.text);
}

async function fetchAnalyticsOrigin(origin: string, cursor: number): Promise<number> {
    const R = rt();
    const S = R.getSession();
    let expectedCc = "";
    try {
        expectedCc = await hashId(new URL(origin).host);
    } catch (e) {}
    const cred = await getOAuthCredentialForOrigin(origin);
    if (!cred) {
        console.warn(
            `[SYNC-ANALYTIC] ⏭️ '${origin}' 은 등록된 API 키가 없어 건너뜁니다. ` +
            `Analytic 탭의 '+ 사이트 등록' 으로 먼저 등록하세요. ` +
            `(console.logis.center 는 client_id 검증을 통과한 요청에만 로그를 반환합니다)`
        );
        return 0;
    }
    const params = new URLSearchParams({
        origin: "https://console.logis.center",
        created_at: cursor.toString(),
        hash: S.hash,
        token: S.token || "",
        href: origin + "/"
    });
    if (expectedCc) params.append("cc", expectedCc);
    params.append("client_id", cred.client_id);
    if (cred.client_secret) params.append("client_secret", cred.client_secret);
    let response: any = null;
    try {
        response = await invoke<any>("proxy_fetch", {
            url: `${ANALYTIC_API_HOST}/?${params.toString()}`,
            method: "GET",
            headers: { "Content-Type": "application/json" },
            session_params: { hash: S.hash, token: S.token, cc: expectedCc }
        });
    } catch (e) {
        console.warn(`[SYNC-ANALYTIC] ❌ '${origin}' 조회 실패 (cc=${expectedCc}):`, e);
        return 0;
    }
    R.stepQrSpinner();
    const verifySession = (response && response.session) ? response.session : {};
    if (verifySession.verify_reason && verifySession.verified === false) {
        console.warn(
            `[SYNC-ANALYTIC] 🔐 '${origin}' API 키 검증 실패: reason='${verifySession.verify_reason}' ` +
            `(client_id='${cred.client_id}'). Analytic 탭에서 '재발급' 후 다시 시도하거나, ` +
            `사이트 <head> 의 oauth-network-verification 메타 태그를 확인하세요.`
        );
    }
    if (!response || !response.results || !Array.isArray(response.results)) {
        console.log(`[SYNC-ANALYTIC] '${origin}' (cc=${expectedCc}) → 응답에 results 없음`);
        return 0;
    }
    if (response.results.length === 0) {
        console.log(
            `[SYNC-ANALYTIC] '${origin}' (cc=${expectedCc}) → 0건. ` +
            `Worker 조회 조건은 cc = hashId('${(() => { try { return new URL(origin).host; } catch (e) { return "?"; } })()}') ` +
            `AND created_at < ${cursor} 이며, client_id 검증을 통과해야 합니다. ` +
            `(verify='${verifySession.verify_reason || "unknown"}')`
        );
        return 0;
    }
    const now = Date.now();
    const items: any[] = [];
    // 🌟 [DELTA GUARD] 이미 같은 updated_at 으로 로컬에 있는 행은 쓰기를 생략합니다.
    const localMap = new Map<string, number>();
    try {
        if (R.appDb) {
            const ids = response.results.map((r: any) => r && r.id).filter(Boolean);
            if (ids.length > 0) {
                const rows = await R.appDb.table("items").where("id").anyOf(ids).toArray();
                for (const r of rows) localMap.set(String(r.id), Number(r.updated_at || 0));
            }
        }
    } catch (e) {}
    let skipped = 0;
    for (let i = 0; i < response.results.length; i++) {
        const row = response.results[i];
        if (!row || !row.id) continue;
        const serverUpdated = Number(row.updated_at || 0);
        const localUpdated = localMap.get(String(row.id));
        if (localUpdated !== undefined && serverUpdated <= localUpdated) {
            skipped++;
            continue;
        }
        if (localUpdated !== undefined && localUpdated > 0) {
            skipped++;
            console.log(
                `[SYNC-ANALYTIC] ⚪ '${row.id}' 는 이미 구조화 완료되어 서버 원시 데이터로 덮어쓰지 않습니다.`
            );
            continue;
        }
        const parsed: any = decodeAnalyticBlob(row.data) || {};
        if (!parsed.origin) parsed.origin = origin;
        const textVal = extractAnalyticText(parsed);
        const rowType = String(row.type || "click");
        const rowCc = String(row.cc || expectedCc || "");
        let rowBcc = String(row.bcc || "");
        if (!rowBcc && rowCc) {
            rowBcc = await hashId(rowType + rowCc);
        }
        // 🌟 [MODE TAGGING] modeOfType() 단일 판정 경로를 그대로 사용합니다.
        const rowMode = modeOfType(rowType);
        const isRawEvent = Array.isArray(parsed?.action) || Array.isArray(parsed?.relate);
        items.push({
            id: row.id,
            type: rowType,
            flag: row.flag || String((S as any).flag || ""),
            from: row.from || "",
            to: row.to || "",
            cc: rowCc,
            bcc: rowBcc,
            ref: row.ref || "",
            status: 9,
            mode: rowMode,
            created_at: Number(row.created_at || now),
            updated_at: isRawEvent ? 0 : serverUpdated,
            text: textVal,
            masked_text: textVal,
            data: {
                ...parsed,
                id: row.id,
                type: rowType,
                mode: rowMode,
                updated_at: isRawEvent ? 0 : serverUpdated,
                text: textVal,
                masked_text: textVal
            }
        });
    }
    if (items.length === 0) {
        console.log(`[SYNC-ANALYTIC] '${origin}' 수신 ${response.results.length}건 → 전부 최신 상태(스킵 ${skipped}건)`);
        return 0;
    }
    await invoke("upsert_items", { items });
    if (R.appDb) {
        await R.appDb.table("items").bulkPut(R.normalizeEnvelope(items)).catch(() => null);
    }
    const typeBrief: Record<string, number> = {};
    for (const it of items) typeBrief[it.type] = (typeBrief[it.type] || 0) + 1;
    console.log(
        `[SYNC-ANALYTIC] ✅ '${origin}' (cc=${expectedCc}) → 수신 ${response.results.length}건 / ` +
        `저장 ${items.length}건 / 스킵 ${skipped}건 ${JSON.stringify(typeBrief)}`
    );
    return items.length;
}

/**
 * 로컬에 남은 원시 이벤트(updated_at == 0)를 구조화하고 임베딩을 예약합니다. 서버에는 접속하지 않습니다.
 * `received` 는 이번에 서버에서 새 행을 받았는지입니다 (받았으면 구조화 결과와 무관하게 임베딩을 예약).
 */
async function structurePendingAnalytics(received: boolean): Promise<number> {
    const structured = await runAnalyticStructuring();
    if (structured > 0) {
        await finalizeAnalyticBubbles(structured);
    }
    if (received || structured > 0) {
        rt().runLocalEmbeddingSync();
    }
    return structured;
}

export async function syncAnalyticsData() {
    const R = rt();
    const S = R.getSession();
    if (!S.hash) return;
    if (isAnalyticsSyncRunning) return;
    isAnalyticsSyncRunning = true;
    try {
        const origins = await resolveAnalyticsOrigins();
        let totalStored = 0;
        if (origins.length === 0) {
            console.warn(
                "[SYNC-ANALYTIC] 조회할 추적 대상 사이트가 없습니다. " +
                "Analytic 탭의 '+ 사이트 등록' 으로 도메인을 등록하거나, " +
                "브라우저로 추적 대상 사이트를 열어 두세요. " +
                "(Worker 는 cc 파라미터가 아니라 href 의 host 로 조회 대상을 결정합니다)"
            );
        } else {
            const now = Date.now();
            const cursor = Math.max(now, now - R.timezoneOffset) + 60_000;
            console.log(`[SYNC-ANALYTIC] 대상 사이트 ${origins.length}곳 조회 시작: ${JSON.stringify(origins)} | cursor=${cursor}`);
            for (const origin of origins) {
                totalStored += await fetchAnalyticsOrigin(origin, cursor);
            }
            if (totalStored > 0 && R.getSearchMode() === "analytic") {
                await R.renderNavigation();
                if (R.getCurrentTab() === "list") {
                    await R.loadMoreDocs(false, true);
                }
            }
        }
        // 🌟 [LOCAL STRUCTURING] 조회할 사이트가 없거나 새로 받은 행이 없어도 로컬에 남은 원시 이벤트는 구조화합니다.
        //    (예전에는 사이트가 0곳이면 여기까지 오지 않았습니다)
        await structurePendingAnalytics(totalStored > 0);
        console.log(`[SYNC-ANALYTIC] 완료. 총 저장 ${totalStored}건.`);
    } catch (e) {
        console.warn("[SYNC-ANALYTIC] Failed:", e);
    } finally {
        isAnalyticsSyncRunning = false;
        lastAnalyticsSyncAt = Date.now();
        // 🌟 main.ts 의 stopSpinner() 내부가 `if (isExtracting || isSearching) return;`
        //    으로 이미 방어하므로 구 코드의 가드와 동작이 동일합니다.
        R.stopSpinner();
        // 🌟 [SYNC DONE → SUBMIT RESTORE] 구 btnSubmit 복사본을 단일 함수로 대체합니다.
        R.restoreSubmitButton();
    }
}

/**
 * 🌟 [LOCAL-ONLY] 세션이 없을 때(오프라인 · 로그인 전 · 로그아웃) 폴링이 부르는 로컬 단계입니다.
 *  원시 이벤트 구조화는 로컬 Qwen 과 LanceDB 만 쓰는데, 예전에는 서버 동기화(syncAnalyticsData) 안에서만 불려
 *  세션이 없으면 로컬에 쌓인 이벤트가 구조화되지 않았습니다. 서버 동기화와 같은 스로틀(30초 이상)을 씁니다.
 *  대상이 없으면 Rust 가 LanceDB 조회 한 번으로 끝내고 모델을 올리지 않습니다.
 */
export async function runAnalyticLocalTick(): Promise<number> {
    if (isAnalyticsSyncRunning) return 0;
    const throttleMs = Math.max(30_000, getSyncIntervalMs());
    if (Date.now() - lastAnalyticsSyncAt < throttleMs) return 0;
    isAnalyticsSyncRunning = true;
    try {
        return await structurePendingAnalytics(false);
    } finally {
        isAnalyticsSyncRunning = false;
        lastAnalyticsSyncAt = Date.now();
    }
}

export async function syncAnalyticsInBackground() {
    const S = rt().getSession();
    if (!S.hash) return;
    if (isAnalyticsSyncRunning) return;
    const throttleMs = Math.max(30_000, getSyncIntervalMs());
    if (Date.now() - lastAnalyticsSyncAt < throttleMs) return;
    await syncAnalyticsData();
}