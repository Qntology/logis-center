// ============================================================
//  commerce/src/modes/trading.ts
//  🌟 [TRADING TRACK] shipping(무역 서식) 모드 전용 동기화 계층
//
//  ── 왜 별도 Worker / 별도 파일인가 ──
//   commerce D1 의 items 에는 mode / flag 컬럼이 없어 클라이언트가
//   MODE TAGGING / FLAG RECOVERY 로 사후 보강해 왔고, 그 보강 누락이
//   무역 서식 19종이 mode='commerce' 로 굳은 직접 원인이었습니다.
//   trading D1 은 두 컬럼을 물리 컬럼으로 갖고 있어 그 해킹이 필요 없습니다.
//   따라서 동기화 규칙 자체가 commerce 와 달라 파일을 분리합니다.
// ============================================================
import { invoke } from "@tauri-apps/api/core";
import {
    rt,
    decodeAnalyticBlob,
    updateSyncBackoff,
    getSyncIntervalMs
} from "./runtime";

export const TRADING_API_HOST = "https://trading.logis.center";

let isTradingSyncRunning = false;
let lastTradingSyncAt = 0;

/**
 * 🌟 [THROTTLE RESET] 사용자가 shipping 탭으로 전환한 직후 폴링 주기를 기다리지 않고
 *  즉시 1회 당겨오기 위한 스로틀 해제입니다.
 *  (구 main.ts 의 `lastTradingSyncAt = 0;` 직접 대입을 대체합니다)
 */
export function resetTradingThrottle() {
    lastTradingSyncAt = 0;
}

export async function tradingApiFetch(
    query: Record<string, any>,
    opts: { method?: "GET" | "POST" | "DELETE"; body?: any; gzip?: boolean } = {}
): Promise<any> {
    const S = rt().getSession();
    if (!S.hash || !S.token) return null;
    const method = opts.method || "GET";
    const sp = new URLSearchParams();
    sp.append("hash", S.hash);
    sp.append("token", S.token);
    if (S.team) sp.append("to", S.team);
    for (const k of Object.keys(query)) {
        const v = query[k];
        if (v === undefined || v === null || v === "") continue;
        sp.append(k, String(v));
    }
    const url = `${TRADING_API_HOST}/?${sp.toString()}`;
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (opts.gzip) headers["Content-Encoding"] = "gzip";
    const args: any = { url, method, headers, session_params: null };
    if (opts.body !== undefined) args.body = opts.body;
    try {
        return await invoke<any>("proxy_fetch", args);
    } catch (e) {
        console.warn(`[SYNC-TRADING] ${method} 실패:`, e);
        return null;
    }
}

/** 서버 → 로컬. 델타 커서는 서버가 확정해 준 cursor 를 그대로 씁니다. */
export async function pullTradingData(): Promise<number> {
    const R = rt();
    const since = Number((await R.kvGet("trading_sync_cursor")) || 0);
    const now = Date.now();
    // 🌟 [CURSOR] created_at 은 '상한' 입니다. now - timezoneOffset 을 쓰면
    //    UTC- 지역에서 최근 문서가 통째로 잘립니다. 반드시 미래여야 합니다.
    const upper = Math.max(now, now - R.timezoneOffset) + 60_000;
    const res = await tradingApiFetch({ since: since, created_at: upper, limit: 1000 });
    if (!res || !Array.isArray(res.results)) return 0;
    if (res.results.length === 0) {
        if (res.cursor) await R.kvSet("trading_sync_cursor", String(res.cursor));
        return 0;
    }
    const tombs = await R.loadItemTombstones();
    let tombBlocked = 0;
    const items: any[] = [];
    for (const row of res.results) {
        if (!row || !row.id) continue;
        if (tombs.has(String(row.id))) { tombBlocked++; continue; }
        const parsed: any = decodeAnalyticBlob(row.data) || {};
        // 서버 봉투가 진실의 원천입니다. data 안의 값보다 우선합니다.
        parsed.id = row.id;
        parsed.type = row.type;
        parsed.mode = row.mode || "shipping";
        parsed.flag = row.flag || "";
        parsed.cc = row.cc || "";
        parsed.bcc = row.bcc || "";
        parsed.ref = row.ref || "";
        parsed.digest = row.digest || "";
        parsed.created_at = Number(row.created_at || 0);
        parsed.updated_at = Number(row.updated_at || 0);
        const textVal =
            (typeof parsed.text === "string" ? parsed.text.trim() : "")
            || (typeof parsed.summary === "string" ? parsed.summary.trim() : "");
        items.push({
            id: row.id,
            table: "items",
            type: row.type,
            flag: row.flag || "",
            from: row.from || "",
            to: row.to || "",
            cc: row.cc || "",
            bcc: row.bcc || "",
            ref: row.ref || "",
            mode: row.mode || "shipping",
            digest: row.digest || "",
            status: 9,
            created_at: Number(row.created_at || 0),
            updated_at: Number(row.updated_at || 0),
            text: textVal,
            masked_text: textVal,
            data: parsed
        });
    }
    if (tombBlocked > 0) {
        console.log(`[SYNC-TRADING] 🪦 삭제된 문서 ${tombBlocked}건의 재삽입을 차단했습니다.`);
    }
    if (items.length > 0) {
        await invoke("upsert_items", { items });
        if (R.appDb) {
            await R.appDb.table("items").bulkPut(R.normalizeEnvelope(items)).catch(() => null);
        }
        const brief: Record<string, number> = {};
        for (const it of items) brief[it.type] = (brief[it.type] || 0) + 1;
        console.log(`[SYNC-TRADING] ⬇️ 수신 ${res.results.length}건 / 저장 ${items.length}건 ${JSON.stringify(brief)}`);
    }
    if (res.cursor) await R.kvSet("trading_sync_cursor", String(res.cursor));
    return items.length;
}

export async function pushTradingData(): Promise<number> {
    const R = rt();
    if (!R.appDb) return 0;
    const pushedCursor = Number((await R.kvGet("trading_push_cursor")) || 0);
    let rows: any[] = [];
    try {
        rows = await R.appDb.table("items").where("mode").equals("shipping").toArray();
    } catch (e) {
        console.warn("[SYNC-TRADING] 로컬 조회 실패:", e);
        return 0;
    }
    const stampOf = (r: any) => Math.max(Number(r.updated_at || 0), Number(r.created_at || 0));
    const candidates = rows
        .filter((r: any) => r && r.id && stampOf(r) > pushedCursor)
        .sort((a: any, b: any) => stampOf(a) - stampOf(b));
    if (candidates.length === 0) return 0;
    const batch = candidates.slice(0, 200);
    const payload = {
        items: batch.map((r: any) => ({
            id: r.id,
            type: r.type,
            flag: r.flag || "",
            digest: r.digest || (r.data && r.data.digest) || "",
            created_at: Number(r.created_at || 0),
            updated_at: Number(r.updated_at || 0),
            data: r.data || {}
        }))
    };
    const res = await tradingApiFetch({}, { method: "POST", body: payload, gzip: true });
    if (!res) return 0;
    const accepted = Number(res.accepted || 0);
    const skipped = Number(res.skipped || 0);
    const rejected = Number(res.rejected || 0);
    if (accepted + skipped >= batch.length) {
        const maxStamp = batch.reduce((m: number, r: any) => Math.max(m, stampOf(r)), pushedCursor);
        await R.kvSet("trading_push_cursor", String(maxStamp));
    }
    if (Array.isArray(res.results) && res.results.length > 0) {
        const byId = new Map<string, any>();
        for (const r of batch) byId.set(String(r.id), r);
        const adopted: any[] = [];
        for (const srv of res.results) {
            const local = byId.get(String(srv.id));
            if (!local) continue;
            if (local.ref === srv.ref && local.bcc === srv.bcc && local.cc === srv.cc) continue;
            const data = { ...(local.data || {}) };
            data.cc = srv.cc;
            data.bcc = srv.bcc;
            data.ref = srv.ref;
            data.mode = srv.mode;
            data.flag = srv.flag;
            adopted.push({
                id: srv.id,
                table: "items",
                type: srv.type,
                flag: srv.flag,
                from: srv.from,
                to: srv.to,
                cc: srv.cc,
                bcc: srv.bcc,
                ref: srv.ref,
                mode: srv.mode,
                digest: srv.digest,
                created_at: srv.created_at,
                updated_at: srv.updated_at,
                text: local.data?.text || "",
                masked_text: local.data?.masked_text || local.data?.text || "",
                data
            });
        }
        if (adopted.length > 0) {
            await invoke("upsert_items", { items: adopted });
            if (R.appDb) await R.appDb.table("items").bulkPut(R.normalizeEnvelope(adopted)).catch(() => null);
            console.log(`[SYNC-TRADING] 🔗 서버 확정 봉투(ref/bcc) ${adopted.length}건을 로컬에 반영했습니다.`);
        }
    }
    console.log(`[SYNC-TRADING] ⬆️ 후보 ${candidates.length}건 중 ${batch.length}건 전송 → 저장 ${accepted} / 스킵 ${skipped} / 거부 ${rejected}`);
    return accepted;
}

export async function syncTradingData() {
    const R = rt();
    const S = R.getSession();
    if (!S.hash || !S.token) return;
    if (isTradingSyncRunning) return;
    isTradingSyncRunning = true;
    try {
        const pushedCount = await pushTradingData();
        const pulledCount = await pullTradingData();
        if (pushedCount > 0 || pulledCount > 0) {
            updateSyncBackoff(true);
            if (R.getSearchMode() === "shipping") {
                await R.renderNavigation();
                if (R.getCurrentTab() === "list") {
                    await R.loadMoreDocs(false, true);
                }
            }
            R.runLocalEmbeddingSync();
        } else {
            updateSyncBackoff(false);
        }
    } catch (e) {
        console.warn("[SYNC-TRADING] Failed:", e);
    } finally {
        isTradingSyncRunning = false;
        lastTradingSyncAt = Date.now();
        // 🌟 main.ts 의 stopSpinner() 는 내부에서 `if (isExtracting || isSearching) return;`
        //    로 이미 방어하므로, 구 코드의 `if (!isExtracting && !isSearching)` 가드와
        //    동작이 100% 동일합니다.
        R.stopSpinner();
        // 🌟 [SYNC DONE → SUBMIT RESTORE] 구 코드의 btnSubmit 4벌 복사본을 단일 함수로 대체합니다.
        R.restoreSubmitButton();
    }
}

export async function syncTradingInBackground() {
    const S = rt().getSession();
    if (!S.hash || !S.token) return;
    if (isTradingSyncRunning) return;
    const throttleMs = Math.max(30_000, getSyncIntervalMs());
    if (Date.now() - lastTradingSyncAt < throttleMs) return;
    await syncTradingData();
}