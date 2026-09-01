// ============================================================
//  commerce/src/modes/commerce.ts
//  🌟 [COMMERCE TRACK] commerce.logis.center D1 동기화 계층
//
//  ── 이 트랙만의 특수 사정 ──
//   commerce D1 의 items 테이블에는 mode / flag 물리 컬럼이 없습니다.
//   그래서 수신 직후 클라이언트가
//     ① MODE TAGGING  : modeOfType() 으로 mode 를 확정하고
//     ② FLAG RECOVERY : 세션 flag(GeoIP 국가코드)로 flag 를 보강하고
//     ③ ROOT ABSORB   : 루트 물리 컬럼을 data 로 내려 인덱스를 살립니다.
//   trading D1 은 두 컬럼을 물리 컬럼으로 갖고 있어 이 보강이 전혀 필요 없습니다.
//   보강 규칙이 근본적으로 다르기 때문에 파일을 분리합니다.
// ============================================================
import { invoke } from "@tauri-apps/api/core";
import { hashId } from "../lib/utils";
import { Select } from "../lib/db";
import { modeOfType } from "./types";
import { rt, getRootDomain, ENVELOPE_ROOT_KEYS, updateSyncBackoff } from "./runtime";

export const COMMERCE_API_HOST = "https://commerce.logis.center";

let isCommerceSyncRunning = false;

export async function syncCommerceInBackground() {
    const R = rt();
    const S = R.getSession();
    if (isCommerceSyncRunning) return;
    if (!S.hash || !S.email) return;
    isCommerceSyncRunning = true;
    try {
        const origin = "https://commerce.logis.center";
        const now = Date.now();
        const createdAt = now - R.timezoneOffset;
        let targetHref = R.getDetectedUrl() || "https://commerce.logis.center/tracking";
        if (targetHref.includes("localhost") || targetHref.includes("127.0.0.1") || targetHref === "about:blank") {
            targetHref = "https://commerce.logis.center/tracking";
        }
        const queryParams: any = {
            origin: origin,
            created_at: createdAt.toString(),
            hash: S.hash,
            token: S.token || "",
            href: targetHref
        };
        const ctx = R.getContext();
        let syncEffectiveCc = ctx.cc;
        const activeTags = R.getActiveTags();
        const isDefaultForced = activeTags.some(t => t.value === "logis.center" && t.type === "domain");
        if (!syncEffectiveCc || (!isDefaultForced && activeTags.length === 0)) {
            try {
                const urlObj = new URL(targetHref.toLowerCase());
                const rootDomain = getRootDomain(urlObj.hostname);
                syncEffectiveCc = await hashId(rootDomain);
            } catch(e) {}
        }
        if (syncEffectiveCc) queryParams.cc = syncEffectiveCc;
        const params = new URLSearchParams(queryParams);
        const url = `${COMMERCE_API_HOST}/?${params.toString()}`;
        const response = await invoke<any>("proxy_fetch", {
            url: url,
            method: "GET",
            headers: { "Content-Type": "application/json" },
            session_params: { hash: S.hash, token: S.token, cc: ctx.cc || "" }
        });
        if (response.results && Array.isArray(response.results)) {
            try {
                const pako = (window as any).pako;
                for (let i = 0; i < response.results.length; i++) {
                    let item = response.results[i];
                    if (item.data && typeof item.data === 'object' && !item.data.text && !item.data.title) {
                        let arrData = item.data.data || item.data;
                        let arr: Uint8Array | null = null;
                        if (Array.isArray(arrData)) {
                            arr = new Uint8Array(arrData);
                        } else if (arrData.buffer) {
                            arr = new Uint8Array(arrData.buffer);
                        } else if (Object.keys(arrData).length > 0 && !isNaN(Number(Object.keys(arrData)[0]))) {
                            arr = new Uint8Array(Object.values(arrData) as number[]);
                        }
                        if (arr) {
                            try {
                                if (pako) {
                                    const decompressed = pako.ungzip(arr, { to: 'string' });
                                    item.data = JSON.parse(decompressed);
                                } else {
                                    const decompressed = new TextDecoder('utf-8').decode(arr);
                                    item.data = JSON.parse(decompressed);
                                }
                            } catch (e) {
                                try {
                                    const decompressed = new TextDecoder('utf-8').decode(arr);
                                    item.data = JSON.parse(decompressed);
                                } catch (err) {}
                            }
                        }
                    } else if (typeof item.data === 'string' && item.data.length > 50) {
                        // 🌟 [BASE64 GZIP PATH] syncAnalyticsData 와 동일한 처리.
                        //    data 가 base64(gzip) 문자열로 내려오는 경우를 커버합니다.
                        try {
                            const rawBytes = Uint8Array.from(atob(item.data), c => c.charCodeAt(0));
                            if (rawBytes[0] === 0x1f && rawBytes[1] === 0x8b) {
                                item.data = JSON.parse(pako ? pako.ungzip(rawBytes, { to: 'string' }) : new TextDecoder('utf-8').decode(rawBytes));
                            } else {
                                item.data = JSON.parse(new TextDecoder('utf-8').decode(rawBytes));
                            }
                        } catch (_strErr) {
                            // JSON 도 base64 도 아니면 원본 문자열을 그대로 둡니다.
                        }
                    }
                }
            } catch (err) {
                console.warn("[SYNC-BG] pako decompression failed:", err);
            }
            // 🌟 [TOMBSTONE GUARD] analytic 탭에서도 commerce D1 을 백그라운드로 긁으므로
            //    syncCommerceData 와 동일한 부활 경로가 열려 있습니다. 같은 게이트를 적용합니다.
            const bgTombstones = await R.loadTalkTombstones();
            const bgItemTombs = await R.loadItemTombstones();
            let bgTombBlocked = 0;
            let bgItemTombBlocked = 0;
            const filteredResults = response.results.filter((newItem: any) => {
                if (bgTombstones.has(String(newItem.id))) {
                    bgTombBlocked++;
                    return false;
                }
                // 🌟 [ITEM TOMBSTONE CHECK] 문서 삭제도 백그라운드 재삽입을 차단합니다.
                if (bgItemTombs.has(String(newItem.id))) {
                    bgItemTombBlocked++;
                    return false;
                }
                const existingEl = document.getElementById(newItem.id);
                if (!existingEl) return true;
                let localUpdated = parseInt(existingEl.dataset.updatedAt || existingEl.dataset.createdAt || "0");
                const serverUpdated = newItem.updated_at || newItem.created_at || 0;
                return serverUpdated > localUpdated;
            });
            if (bgTombBlocked > 0) {
                console.log(`[TOMBSTONE] 🪦 [BG] 삭제된 메시지 ${bgTombBlocked}건의 백그라운드 재삽입을 차단했습니다.`);
            }
            if (bgItemTombBlocked > 0) {
                console.log(`[ITEM-TOMBSTONE] 🪦 [BG] 삭제된 문서 ${bgItemTombBlocked}건의 백그라운드 재삽입을 차단했습니다.`);
            }
            if (filteredResults.length > 0) {
                for (const r of filteredResults) {
                    if (!r) continue;
                    if (r.mode) continue;
                    r.mode = modeOfType(String(r.type || ""));
                }
                const sessionFlag = String((S as any).flag || "");
                if (sessionFlag) {
                    for (const r of filteredResults) {
                        if (!r) continue;
                        if (r.flag) continue;
                        const inner = (r.data && typeof r.data === 'object') ? r.data.flag : undefined;
                        if (inner) { r.flag = inner; continue; }
                        r.flag = sessionFlag;
                    }
                }
                for (const r of filteredResults) {
                    if (!r) continue;
                    if (!r.data || typeof r.data !== 'object') continue;
                    for (const k in r) {
                        if (!Object.prototype.hasOwnProperty.call(r, k)) continue;
                        if (ENVELOPE_ROOT_KEYS.has(k)) continue;
                        const v = r[k];
                        if (v === undefined || v === null) continue;
                        if (typeof v === 'function') continue;
                        if (r.data[k] === undefined || r.data[k] === null || r.data[k] === "") {
                            r.data[k] = v;
                        }
                    }
                }
                // 🌟 [BACKOFF RESET] 백그라운드 commerce 동기화에서 변경이 있었으므로 리셋합니다.
                updateSyncBackoff(true);
                console.log(`[SYNC-BG] Commerce D1 background sync: ${filteredResults.length} item(s)`);
                await invoke("upsert_items", { items: filteredResults });
                // 🌟 [PAGE CACHE GUARD] syncCommerceData 와 동일한 규칙으로 페이지 셀렉터 캐시를 분리합니다.
                //    기존에는 node / item 체크 없이 type 만으로 필터링하여
                //    { table:'pages', type:'tracking', node:1, item:true } 문서를
                //    items 테이블에 잘못 저장했습니다.
                //    (로그: "[DEBUG] Syncing item - ID: , Type: tracking" 의 직접 원인)
                const bgNewPages = filteredResults.filter((r: any) => {
                    if (r.table === "pages" || r.table === "page") return true;
                    if (r.type === "pages" || r.type === "page") return true;
                    const d = typeof r.data === 'string' ? JSON.parse(r.data) : (r.data || r);
                    return !!d.node || !!d.item;
                });
                if (bgNewPages.length > 0 && R.appDb) {
                    await R.appDb.table("pages").bulkPut(R.normalizeEnvelope(bgNewPages)).catch(() => null);
                }
                const newItems = filteredResults.filter((r: any) =>
                    r.type !== "team" && r.type !== "user" && r.type !== "member"
                    && r.type !== "pages" && r.type !== "page"
                    && r.type !== "talk" && r.table !== "talks"
                    && !bgNewPages.includes(r)
                );
                if (newItems.length > 0 && R.appDb) {
                    await R.appDb.table("items").bulkPut(R.normalizeEnvelope(newItems)).catch(() => null);
                }
                // 현재 탭이 commerce/list 일 때만 UI 갱신
                if (R.getSearchMode() === "commerce" && R.getCurrentTab() === "list") {
                    await R.loadMoreDocs(false, true);
                }
                R.runLocalEmbeddingSync();
            }
        }
    } catch (e) {
        console.warn("[SYNC-BG] Commerce background sync failed:", e);
    } finally {
        isCommerceSyncRunning = false;
    }
}

/**
 * 🌟 [COMMERCE FOREGROUND] 구 main.ts syncData() 의 commerce 본문입니다.
 *  ── 라우팅 계약 ──
 *   main.ts 의 syncData() 가 analytic / shipping 분기를 먼저 처리하고,
 *   commerce 트랙일 때만 이 함수를 await 합니다.
 *   따라서 이 함수는 진입 시점에 hash / email 이 확정되어 있습니다.
 */
export async function syncCommerceData() {
    const R = rt();
    const S = R.getSession();
    console.log("[SYNC] 1. 서버에 최신 데이터 요청 중...");
    try {
        const origin = "https://commerce.logis.center";
        const now = Date.now();
        const createdAt = now - R.timezoneOffset;

        // 🌟 [CRITICAL FIX] front.js 패리티: 서버가 나를 정확히 인지하도록 cc, type, 실제 href 파라미터를 추가합니다.
        let targetHref = R.getDetectedUrl() || "https://commerce.logis.center/tracking";
        if (targetHref.includes("localhost") || targetHref.includes("127.0.0.1") || targetHref === "about:blank") {
            targetHref = "https://commerce.logis.center/tracking";
        }
        const queryParams: any = {
            origin: origin,
            created_at: createdAt.toString(),
            hash: S.hash,
            token: S.token || "",
            href: targetHref
        };
        // 🌟 [SENDER IMPRINT] checkAuthStatus 와 동일한 이유입니다.
        //    syncCommerceData 는 email 이 확정된 뒤에만 실행되므로
        //    여기서 보내는 sender 는 반드시 서버 user.data 에 각인됩니다.
        //    이 값이 있어야 PUT(talks) / POST(tasks) 가 cookies.sender 게이트를 통과합니다.
        const syncSender = S.email || S.name || "";
        if (syncSender) queryParams.sender = syncSender;
        // 🌟 [CRITICAL FIX] chrome.js 패리티: 서버 동기화 시, 강제 지정된 사이드바 메뉴가 없다면 현재 URL의 도메인(CC)을 최우선으로 서버에 전달합니다.
        const ctx = R.getContext();
        let syncEffectiveCc = ctx.cc;
        const activeTags = R.getActiveTags();
        const isDefaultForced = activeTags.some(t => t.value === "logis.center" && t.type === "domain");
        if (!syncEffectiveCc || (!isDefaultForced && activeTags.length === 0)) {
            try {
                const urlObj = new URL(targetHref.toLowerCase());
                const rootDomain = getRootDomain(urlObj.hostname);
                syncEffectiveCc = await hashId(rootDomain);
            } catch(e) {}
        }
        if (syncEffectiveCc) queryParams.cc = syncEffectiveCc;
        const currentSearchMode = R.getSearchMode();
        if (currentSearchMode && currentSearchMode !== "commerce") queryParams.type = currentSearchMode;
        const params = new URLSearchParams(queryParams);
        const url = `${COMMERCE_API_HOST}/?${params.toString()}`;

        // 1. 서버 요청
        const response = await invoke<any>("proxy_fetch", {
            url: url,
            method: "GET",
            headers: { "Content-Type": "application/json" },
            // 🌟 서버가 cc를 헤더나 쿠키처럼 파싱할 수 있게 proxy_fetch 파라미터에도 주입합니다.
            session_params: { hash: S.hash, token: S.token, cc: ctx.cc || "" }
        });
        R.stepQrSpinner();
        console.log('response',response);
        if (response.results && Array.isArray(response.results)) {
            // 🌟 [추가] 서버에서 압축된 gzip 데이터를 그대로 보낼 경우, chrome.js와 동일하게 pako로 압축 해제
            try {
                const pako = (window as any).pako;
                for (let i = 0; i < response.results.length; i++) {
                    let item = response.results[i];
                    if (item.data && typeof item.data === 'object' && !item.data.text && !item.data.title) {
                        let arrData = item.data.data || item.data;
                        let arr: Uint8Array | null = null;

                        if (Array.isArray(arrData)) {
                            arr = new Uint8Array(arrData);
                        } else if (arrData.buffer) {
                            arr = new Uint8Array(arrData.buffer);
                        } else if (Object.keys(arrData).length > 0 && !isNaN(Number(Object.keys(arrData)[0]))) {
                            arr = new Uint8Array(Object.values(arrData) as number[]);
                        }
                        if (arr) {
                            try {
                                if (pako) {
                                    const decompressed = pako.ungzip(arr, { to: 'string' });
                                    item.data = JSON.parse(decompressed);
                                } else {
                                    const decompressed = new TextDecoder('utf-8').decode(arr);
                                    item.data = JSON.parse(decompressed);
                                }
                            } catch (e) {
                                try {
                                    const decompressed = new TextDecoder('utf-8').decode(arr);
                                    item.data = JSON.parse(decompressed);
                                } catch (err) {}
                            }
                        }
                    }
                }
            } catch (err) {
                console.warn("[SYNC] pako decompression failed:", err);
            }
            // 🌟 [버그 수정] DOM에 렌더링된 요소뿐만 아니라, Dexie DB에 있는 백그라운드 통계(team) 객체의 최신 시간도 대조해야 합니다.
            const localUsers = await Select["users"]({});
            const localPages = await Select["pages"]({});
            const localTalks = await R.appDb.table("talks").toArray(); // 🌟 채팅(talks) 데이터도 로컬 맵에 반드시 포함
            const localMap = new Map();
            [...localUsers, ...localPages, ...localTalks].forEach((item: any) => {
                // 🌟 updated_at이 0인 레거시 데이터를 대비해 created_at을 백업으로 사용
                localMap.set(item.id, item.updated_at_ts || item.updated_at || item.created_at_ts || item.created_at || 0);
            });
            // 🌟 [TOMBSTONE GUARD] 내가 삭제한 메시지는 서버에 행이 남아 있으므로
            //    매 폴링마다 다시 내려옵니다. 그때 로컬에는 DOM 도 Dexie 도 없으니
            //    바로 아래 `!existingEl && !localMap.has(id)` 조건을 통과해 재삽입됩니다.
            //    묘비 조회는 메모리 Set 이라 폴링 비용이 사실상 0 입니다.
            const tombstones = await R.loadTalkTombstones();
            // 🌟 [ITEM TOMBSTONE] 문서(items) 삭제도 동일하게 서버 재삽입을 차단합니다.
            const itemTombs = await R.loadItemTombstones();
            let tombBlocked = 0;
            let itemTombBlocked = 0;
            const filteredResults = response.results.filter((newItem: any) => {
                // 🌟 삭제 의사가 기록된 id 는 어떤 조건보다 먼저, 무조건 차단합니다.
                if (tombstones.has(String(newItem.id))) {
                    tombBlocked++;
                    return false;
                }
                // 🌟 [ITEM TOMBSTONE CHECK] talk 이 아닌 일반 문서도 차단합니다.
                if (itemTombs.has(String(newItem.id))) {
                    itemTombBlocked++;
                    return false;
                }
                const existingEl = document.getElementById(newItem.id);
                // 🌟 완전 신규 데이터 (DOM에도 없고 로컬 DB 캐시에도 없는 경우)는 조건 없이 즉시 통과
                if (!existingEl && !localMap.has(newItem.id)) {
                    return true;
                }
                let localUpdated = 0;
                if (existingEl) {
                    localUpdated = parseInt(existingEl.dataset.updatedAt || existingEl.dataset.createdAt || "0");
                } else if (localMap.has(newItem.id)) {
                    localUpdated = parseInt(localMap.get(newItem.id) || "0");
                }
                // 🌟 서버의 updated_at이 0일 수 있으므로(기존 index.ts 특성), created_at을 백업 비교값으로 활용
                const serverUpdated = newItem.updated_at || newItem.created_at || 0;
                return serverUpdated > localUpdated; // 서버 데이터가 더 최신인 경우만 포함
            });
            if (tombBlocked > 0) {
                console.log(`[TOMBSTONE] 🪦 삭제된 메시지 ${tombBlocked}건의 서버 재삽입을 차단했습니다.`);
            }
            if (itemTombBlocked > 0) {
                console.log(`[ITEM-TOMBSTONE] 🪦 삭제된 문서 ${itemTombBlocked}건의 서버 재삽입을 차단했습니다.`);
            }
            if (filteredResults.length > 0) {
                updateSyncBackoff(true);
                // 🌟 [MODE TAGGING v2] 클라우드 D1 에는 mode 컬럼이 없습니다.
                //    mode 는 '동기화 시점에 클라이언트가 확정하는 값' 이며,
                //    판정은 modes/types.ts 의 modeOfType() 단일 함수가 전담합니다.
                //
                //    ── v1 의 결함 ──
                //     TRADING_TYPES 에 소문자 'tracking' 이 들어 있어
                //     proxy 의 Relay("tracking","order") 가 만든 commerce 배송추적 문서가
                //     전부 mode='shipping' 으로 오염되었고,
                //     loadMoreDocs 의 mode 필터에서 commerce 목록이 통째로 비었습니다.
                for (const r of filteredResults) {
                    if (!r) continue;
                    if (r.mode) continue;
                    r.mode = modeOfType(String(r.type || ""));
                }
                // 🌟 [DRAFT PRESERVE] 서버 행에 updated_at 이 없으면
                //    Rust upsert_item 이 현재 시각을 부여해 draft → count 가 됩니다.
                //    data 내부에도 없으면 여기서 0 을 명시해 draft 계약을 보존합니다.
                for (const r of filteredResults) {
                    if (!r) continue;
                    const hasRoot = r.updated_at !== undefined && r.updated_at !== null;
                    const inner = (r.data && typeof r.data === 'object') ? r.data.updated_at : undefined;
                    const hasInner = inner !== undefined && inner !== null;
                    if (!hasRoot && !hasInner) {
                        r.updated_at = 0;
                    }
                }
                // 🌟 [FLAG RECOVERY] commerce D1 의 items 테이블에는 flag 컬럼이 없습니다.
                //    (analytics D1 · LanceDB · Dexie 에는 모두 존재)
                //    그대로 두면 commerce 트랙 전 문서의 flag 가 영구 공백이 되어
                //    지역 스코프 조회가 성립하지 않습니다.
                //    세션 flag(= Client Worker 가 GeoIP 로 확정한 국가 코드)로 보강합니다.
                const sessionFlag = String((S as any).flag || "");
                if (sessionFlag) {
                    let flagFilled = 0;
                    for (const r of filteredResults) {
                        if (!r) continue;
                        if (r.flag) continue;
                        // data 안에 이미 flag 가 있으면(users/team 문서) 그쪽을 신뢰합니다.
                        const inner = (r.data && typeof r.data === 'object') ? r.data.flag : undefined;
                        if (inner) { r.flag = inner; continue; }
                        r.flag = sessionFlag;
                        flagFilled++;
                    }
                    if (flagFilled > 0) {
                        console.log(`[SYNC] flag 컬럼 부재 문서 ${flagFilled}건을 세션 flag('${sessionFlag}')로 보강했습니다.`);
                    }
                }
                // 🌟 [ROOT ABSORB / RUST] Rust 의 upsert_items 는 item.data 객체의 알맹이를
                //    루트로 끌어올리는 방향만 처리합니다. 반대로 '루트에만 있는 물리 컬럼' 은
                //    data 로 내려 주지 않으므로, 여기서 미리 합쳐 보냅니다.
                //    (normalizeEnvelope 의 ROOT ABSORB 와 동일 규칙이어야 두 저장소가 일치합니다)
                for (const r of filteredResults) {
                    if (!r) continue;
                    if (!r.data || typeof r.data !== 'object') continue;
                    for (const k in r) {
                        if (!Object.prototype.hasOwnProperty.call(r, k)) continue;
                        if (ENVELOPE_ROOT_KEYS.has(k)) continue;
                        const v = r[k];
                        if (v === undefined || v === null) continue;
                        if (typeof v === 'function') continue;
                        if (r.data[k] === undefined || r.data[k] === null || r.data[k] === "") {
                            r.data[k] = v;
                        }
                    }
                }
                console.log(`[SYNC] 2. 로컬 LanceDB 최신화 중... (${filteredResults.length} / ${response.results.length} 건 변경됨)`);
                await invoke("upsert_items", { items: filteredResults });

                // 🌟 [누락 복구] 서버 통계를 LanceDB에 덮어썼다면, 반드시 프론트엔드 Dexie DB 에도 동기화해줘야 화면이 바뀝니다!
                const newUsers = filteredResults.filter((r: any) => r.type === "team" || r.type === "user" || r.type === "member");

                // 🌟 [ROUTING FIX] 서버(Client Worker)는 페이지 캐시 행에 table:'pages' 를 실어 보냅니다.
                //    기존처럼 data.node / data.item 의 '존재 여부' 로 추정하면
                //    canonicalize_data 가 node / detail 을 0 으로 시딩하는 순간 전 아이템이 참이 되어
                //    일반 상품/주문까지 Dexie pages 테이블로 오분류됩니다.
                //    따라서 서버가 명시한 table 을 1순위로 신뢰하고,
                //    없을 때만 셀렉터 마커의 '값이 truthy 인지' 를 봅니다.
                const newPages = filteredResults.filter((r: any) => {
                    if (r.table === "pages" || r.table === "page") return true;
                    if (r.type === "pages" || r.type === "page") return true;
                    const d = typeof r.data === 'string' ? JSON.parse(r.data) : (r.data || r);
                    return !!d.node || !!d.item;
                });

                // chrome.js 형태의 talks(채팅) 메시지 추출
                const newTalks = filteredResults.filter((r: any) => r.type === "talk" || r.table === "talks");
                const newItems = filteredResults.filter((r: any) => !newUsers.includes(r) && !newPages.includes(r) && !newTalks.includes(r));
                // 🌟 v4 : users / pages 도 봉투 정규화를 거칩니다.
                //  talks 는 스키마가 다르므로(role/task_id/status) 그대로 넣습니다.
                if (newUsers.length > 0) await R.appDb.table("users").bulkPut(R.normalizeEnvelope(newUsers));
                if (newPages.length > 0) await R.appDb.table("pages").bulkPut(R.normalizeEnvelope(newPages));
                if (newTalks.length > 0) await R.appDb.table("talks").bulkPut(newTalks);
                if (newItems.length > 0) await R.appDb.table("items").bulkPut(R.normalizeEnvelope(newItems));
                // 🌟 [CRITICAL FIX] 서버에서 가져온 데이터는 이미 윗줄에서 invoke("upsert_items")를 통해 Rust(LanceDB)에
                // 일괄 저장되었습니다. 프론트엔드가 이를 다시 백엔드로 밀어넣는 병목 루프를 삭제합니다.
            } else {
                updateSyncBackoff(false);

                console.log(`[SYNC] 2. 변경된 데이터가 없어 DB 쓰기를 건너뜁니다.`);
            }
            // 🌟 [추가] '대기 중' 멤버 정화(Cleanup) 로직
            // 서버에서 받은 결과 중 정식 멤버(member/user)가 있는지 확인합니다.
            const realMembers = response.results.filter((item: any) => item.type === "member" || item.type === "user");
            if (realMembers.length > 0) {
                const localUsersForCleanup = await Select["users"]({});
                // 로컬에 저장된 'pending_invite_'로 시작하는 가짜 데이터들을 찾습니다.
                const pendingInvites = localUsersForCleanup.filter(u => u.id && u.id.startsWith("pending_invite_"));
                for (const pending of pendingInvites) {
                    const pendingEmail = pending.data?.email;
                    // 서버에서 온 정식 멤버 중 이메일(혹은 이름)이 일치하는 사람이 있는지 대조
                    const isNowMember = realMembers.some((m: any) => {
                        // 서버 데이터(m) 내부에 이메일 정보가 있거나, 이름이 이메일 아이디와 같은지 확인
                        return m.to === pending.from || (m.data && m.data.email === pendingEmail);
                    });
                    if (isNowMember) {
                        // 정식 멤버가 확인되었으므로 가짜(Pending) 데이터를 로컬 DB에서 삭제합니다.
                        await invoke("delete_document", { uuid: pending.id });
                        console.log(`[SYNC] Pending invite for ${pendingEmail} is now a real member. Placeholder removed.`);
                    }
                }
            }

            // 🌟 [CLOUD TASK LIFECYCLE] 서버 tasks 목록과 대조하여 클라우드 작업 완료 여부를 판정합니다.
            const cloudPendingTasks = R.getCloudPendingTasks();
            if (cloudPendingTasks.size > 0) {
                const serverTaskIds = new Set<string>();
                for (const r of response.results) {
                    if (!r) continue;
                    if (r.table === "tasks" && r.id) {
                        serverTaskIds.add(r.id);
                    }
                }
                for (const [localTid, meta] of Array.from(cloudPendingTasks.entries()) as Array<[string, any]>) {
                    const stillRunning = meta.serverId ? serverTaskIds.has(meta.serverId) : false;
                    if (stillRunning) {
                        await R.renderProgressToUI({
                            task_id: localTid,
                            category: "Cloud Queue",
                            summary: "Processing on Logis Center...",
                            spinner: "☁️"
                        });
                        continue;
                    }
                    // 서버 등록 직후의 레이스 컨디션 방어(최소 5초 유예)
                    if (Date.now() - meta.createdAt < 5000) continue;
                    cloudPendingTasks.delete(localTid);
                    await R.renderProgressToUI({
                        task_id: localTid,
                        category: "Done",
                        summary: meta.kind === "search"
                            ? "Cloud AI search complete."
                            : "Cloud AI extraction complete.",
                        spinner: "✅"
                    });
                    console.log(`[CLOUD] Task ${localTid} (server: ${meta.serverId}) finished on Logis Center.`);
                }
            }
            console.log("[SYNC] 3. 로컬 DB에서 데이터 불러와 메뉴 렌더링...");
            // 🌟 [RENDER GATE] 변경된 데이터가 없으면 네비게이션/리스트 재렌더링을 건너뜁니다.
            //    기존에는 매 폴링마다 renderNavigation + loadMoreDocs 가 돌아
            //    동일 데이터를 반복 적재하고 콘솔 로그를 낭비했습니다.
            if (filteredResults.length > 0) {
                // 3. LanceDB 불러오기
                await R.renderNavigation();
                // 🌟 [CRITICAL FIX] 서버 데이터를 로컬 DB에 밀어넣었으니, 현재 보고 있는 탭에 맞춰 UI를 갱신합니다!
                if (R.getCurrentTab() === "list") {
                    await R.loadMoreDocs(false, true);
                } else if (R.getCurrentTab() === "settings") {
                    await R.fetchChatHistory(false, true);
                }
            } else {
                console.log("[SYNC] 3. 변경 없음 → 네비게이션/리스트 재렌더링을 건너뜁니다.");
            }
            // 🌟 [CLIENT-SIDE EMBEDDING] 클라우드는 구조화만 했으므로 임베딩은 여기서 로컬로 수행합니다.
            //    runLocalEmbeddingSync 내부의 2초 디바운스가 initSession의 4초 타이머와
            //    겹치는 중복 호출을 자동으로 병합하여 1회만 실행합니다.
            console.log("[SYNC] 4. 로컬 임베딩 파이프라인 스케줄링 (2초 디바운스 적용)...");
            R.runLocalEmbeddingSync();
        }

    } catch (e) {
        console.error("[SYNC] 동기화 실패:", e);
    } finally {
        // 🌟 main.ts 의 stopSpinner() 내부가 `if (isExtracting || isSearching) return;`
        //    으로 이미 방어하므로 구 코드의 가드와 동작이 동일합니다.
        R.stopSpinner();
        // 🌟 [SYNC DONE → SUBMIT RESTORE] 동기화가 끝나면 검색 입력 상태를 재평가합니다.
        //    기존에는 stopSpinner() 내부에서만 조건부로 노출했지만,
        //    syncCommerceData 가 백그라운드에서 돌 때 btnSubmit 이 숨겨진 채
        //    복귀하지 않는 경로가 있었습니다.
        R.restoreSubmitButton();
    }
}