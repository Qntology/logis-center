import { invoke } from "@tauri-apps/api/core";
import { parseItemData, hashId } from "./utils";

// --- Types ---
interface DbQuery {
    select?: string;
    upsert?: string;
    delete?: string;
    key?: string;
    value?: any;
    limit?: number;
    offset?: number;
    from?: string;
    to?: string;
    ref?: string;
    type?: string;
}

export const Select: Record<string, (query: any) => Promise<any[]>> = {};
export const Upsert: Record<string, (value: any) => Promise<any>> = {};
export const Delete: Record<string, (query: any) => Promise<any>> = {};

// 🌟 Dexie 인스턴스 전역 참조 획득
const getAppDb = () => (window as any).appDb;

// Helper: Parse tags into SQL filter string for LanceDB
async function parseQueryToFilter(queryStr: string): Promise<string | null> {
    if (!queryStr) return null;
    
    const filters: string[] = [];
    const parts = queryStr.split(' ');
    
    for (const part of parts) {
        if (part.startsWith('host:')) {
            const host = part.replace('host:', '');
            const cc = await hashId(host);
            filters.push(`cc = '${cc}'`);
        } else if (part.startsWith('type:')) {
            const type = part.replace('type:', '').toLowerCase();
            filters.push(`type = '${type}'`);
        } else if (part.startsWith('mode:')) {
            // mode:list or mode:detail (mapping logic can be added if needed)
        }
    }
    
    return filters.length > 0 ? filters.join(' AND ') : null;
}

// 🌟 [KEY PATH v4] 외부에서 넘어오는 키를 Dexie 인덱스 경로로 정규화합니다.
//  기존 호출부가 'no' / 'index' / 'tracking_number' 같은 루트 키를 그대로 쓰고 있었는데,
//  v4 에서는 루트 호이스팅이 사라져 data.* 로 옮겨졌습니다.
//  봉투 키(id/type/cc/...)는 그대로, 나머지는 data. 접두사를 붙입니다.
const ENVELOPE_KEYS = new Set([
    'id', 'type', 'flag', 'from', 'to', 'cc', 'bcc', 'ref', 'mode', 'created_at', 'updated_at'
]);
function toDexiePath(key: string): string {
    if (!key) return key;
    if (key.startsWith('data.')) return key;
    if (ENVELOPE_KEYS.has(key)) return key;
    return `data.${key}`;
}

// 🌟 [ROW → DOC] Dexie 봉투 행을 렌더러가 기대하는 형태로 펼칩니다.
//  render.ts 의 Tpl 은 item[key] 와 item.data[key] 를 모두 시도하므로
//  data 를 그대로 두고 봉투 필드만 루트에 남겨 두면 됩니다.
function envelopeToDoc(row: any): any {
    if (!row) return row;
    const parsed = (row.data && typeof row.data === 'object')
        ? row.data
        : parseItemData(row.json_data || row.data);
    return {
        id: row.id,
        uuid: row.id,
        type: row.type,
        flag: row.flag,
        from: row.from,
        to: row.to,
        cc: row.cc,
        bcc: row.bcc,
        ref: row.ref,
        mode: row.mode,
        created_at: row.created_at ?? parsed.created_at ?? 0,
        updated_at: row.updated_at ?? parsed.updated_at ?? 0,
        text: parsed.text ?? "",
        data: parsed
    };
}

// 1. ITEMS (Main Documents)
Select["items"] = async function(query: DbQuery = {}) {
    try {
        let results: any[] = [];
        const appDb = getAppDb();

        // Case A: Specific Key-Value lookup (Dexie 중첩 인덱스 고속 eq 쿼리)
        if (query.key && typeof query.value !== 'undefined') {
            if (appDb) {
                const path = toDexiePath(query.key);
                // 🌟 canonicalize 가 식별자를 String 으로 확정했으므로 값도 String 으로 맞춥니다.
                const val = ENVELOPE_KEYS.has(path) ? query.value : String(query.value);
                const cachedDocs = await appDb.table('items').where(path).equals(val).toArray();
                if (cachedDocs && cachedDocs.length > 0) {
                    return cachedDocs.map(envelopeToDoc);
                }
            }

            // Dexie 캐시에 없으면 fallback to Rust Backend
            if (query.key === 'id') {
                const doc = await invoke<any>("get_document", { uuid: query.value });
                if (doc) {
                    const finalDoc = envelopeToDoc({
                        ...doc,
                        created_at: doc.created_at ?? doc.created_at_ts,
                        updated_at: doc.updated_at ?? doc.updated_at_ts
                    });
                    if (appDb) {
                        await appDb.table('items').put({
                            id: finalDoc.id, type: finalDoc.type, flag: finalDoc.flag,
                            from: finalDoc.from, to: finalDoc.to, cc: finalDoc.cc,
                            bcc: finalDoc.bcc, ref: finalDoc.ref, mode: finalDoc.mode,
                            created_at: finalDoc.created_at, updated_at: finalDoc.updated_at,
                            data: finalDoc.data
                        }).catch(()=>null);
                    }
                    results.push(finalDoc);
                }
                return results;
            }
        }

        // Case B: Filtered or General Search
        const limit = query.limit || 50;
        const offset = query.offset || 0;

        // 🌟 [SCOPE ONLY] 태그(host:/type:)는 봉투 컬럼만 생성하므로 그대로 SQL 로 내려도 안전합니다.
        const sqlFilter = await parseQueryToFilter(String(query.value || ''));

        if (sqlFilter || !query.value) {
            const docs = await invoke<any[]>("get_all_documents", {
                limit,
                offset,
                filter: sqlFilter
            });
            results = docs.map(doc => envelopeToDoc({
                ...doc,
                created_at: doc.created_at ?? doc.created_at_ts,
                updated_at: doc.updated_at ?? doc.updated_at_ts
            }));
        } else {
            const searchRes = await invoke<[string, string, number][]>("search_documents", {
                query: String(query.value),
                limit,
                offset,
                filter: null
            });

            results = searchRes.map(([id, jsonData, score]) => {
                const parsed = parseItemData(jsonData);
                return {
                    id,
                    uuid: id,
                    type: parsed.type || "unknown",
                    mode: parsed.mode || "commerce",
                    created_at: parsed.created_at || 0,
                    updated_at: parsed.updated_at || 0,
                    text: parsed.text || "",
                    score,
                    data: parsed
                };
            });
        }

        return results;
    } catch (e) {
        console.error("[DB Shim] Select['items'] error:", e);
        return [];
    }
};

// 2. PAGES
Select["pages"] = async function(query: DbQuery = {}) {
    try {
        const appDb = getAppDb();
        let pageRows: any[] = [];

        if (appDb) {
            pageRows = await appDb.table('pages').toArray();
        }

        if (pageRows.length === 0) {
            try {
                const fromRust = await invoke<any[]>("get_known_pages", { filter: null });
                pageRows = fromRust.map(d => {
                    const parsed = parseItemData(d.json_data || d.data);
                    return {
                        id: d.id, type: d.type, flag: d.flag ?? "",
                        from: d.from, to: d.to, cc: d.cc, bcc: d.bcc, ref: d.ref,
                        mode: d.mode ?? "commerce",
                        created_at: d.created_at ?? d.created_at_ts ?? 0,
                        updated_at: d.updated_at ?? d.updated_at_ts ?? 0,
                        data: parsed
                    };
                });
                if (appDb && pageRows.length > 0) {
                    await appDb.table('pages').bulkPut(pageRows).catch(()=>null);
                }
            } catch (e) { /* 무시 */ }
        }

        // 🌟 [TITLE ENRICH] 페이지 카드에 표시할 대표 제목을 items 에서 가져옵니다.
        //  기존에는 get_all_documents(200) 를 매번 호출해 Rust 왕복이 발생했는데,
        //  v4 에서는 Dexie 의 ref 인덱스로 즉시 조회합니다.
        const titleMap = new Map<string, string>();
        if (appDb) {
            const refs = pageRows.map(p => p.ref).filter(Boolean);
            if (refs.length > 0) {
                const linked = await appDb.table('items').where('ref').anyOf(refs).toArray();
                for (const it of linked) {
                    if (!titleMap.has(it.ref)) {
                        titleMap.set(it.ref, it.data?.title || it.data?.text || "");
                    }
                }
            }
        }

        // 🌟 [PAGE FILTER] pages 테이블에 들어온 행 중 실제 셀렉터 캐시만 남깁니다.
        //  판정 기준은 data.origin 존재 여부입니다. (기존과 동일)
        let results = pageRows
            .filter(p => {
                const d = p.data || {};
                return !!(d.origin || d.node !== undefined || d.item !== undefined);
            })
            .map(p => {
                const d = p.data || {};
                const realTitle = titleMap.get(p.ref) || d.title || d.text || "";
                return {
                    id: p.id,
                    uuid: p.id,
                    // 🌟 실제 도메인 타입(goods/order/...)이 data.type 에 있습니다.
                    //    봉투 type 은 'pages' 일 수 있으므로 data 를 우선합니다.
                    type: (d.type || p.type || "").toLowerCase(),
                    flag: p.flag,
                    from: p.from, to: p.to, cc: p.cc, bcc: p.bcc, ref: p.ref,
                    mode: p.mode,
                    created_at: p.created_at,
                    updated_at: p.updated_at,
                    title: realTitle,
                    data: { ...d, title: realTitle }
                };
            });

        if (query.key === 'id' && query.value) {
            results = results.filter(r => r.id === query.value);
        }
        return results;
    } catch (e) {
        console.error("[DB Shim] Select['pages'] error:", e);
        return [];
    }
};

// 3. USERS
Select["users"] = async function(query: DbQuery = {}) {
    try {
        const appDb = getAppDb();
        let rows: any[] = [];

        if (appDb) {
            rows = await appDb.table('users').toArray();
        }

        if (rows.length === 0) {
            const fromRust = await invoke<any[]>("get_known_users");
            rows = fromRust.map(d => {
                const parsed = parseItemData(d.json_data || d.data);
                return {
                    id: d.id, type: d.type, flag: d.flag ?? "",
                    from: d.from, to: d.to, cc: d.cc, bcc: d.bcc, ref: d.ref,
                    created_at: d.created_at ?? d.created_at_ts ?? 0,
                    updated_at: d.updated_at ?? d.updated_at_ts ?? 0,
                    data: parsed
                };
            });
            if (appDb && rows.length > 0) {
                await appDb.table('users').bulkPut(rows).catch(()=>null);
            }
        }

        // 🌟 [FLAT VIEW] 네비게이션 렌더러(renderAccordion)가 user.name / user.data.base 를
        //  둘 다 참조하므로, 봉투는 루트에 두고 data 는 그대로 노출합니다.
        //  기존처럼 parsed 를 스프레드해서 봉투 필드를 덮어쓰는 일이 없어졌습니다.
        return rows.map(row => {
            const d = row.data || {};
            return {
                id: row.id,
                uuid: row.id,
                type: d.type || row.type || "",
                flag: row.flag,
                from: row.from, to: row.to, cc: row.cc, bcc: row.bcc, ref: row.ref,
                created_at: row.created_at,
                updated_at: row.updated_at,
                name: d.name || "",
                data: d
            };
        });
    } catch (e) {
        console.error("[DB Shim] Select['users'] error:", e);
        return [];
    }
};

// 4. CRONS
Select["crons"] = async function(query: DbQuery = {}) {
    try {
        const tasks = await invoke<any[]>("get_active_tasks");
        if (query.key === 'ref' && query.value) {
            // 🌟 [CRITICAL FIX] Task 구조체에는 ref_id가 아니라 ref 속성이 존재합니다.
            return tasks.filter((t: any) => t.ref === query.value || (t.data_json && JSON.parse(t.data_json).ref === query.value));
        }
        return tasks;
    } catch (e) { return []; }
};

async function handleUpsert(value: any) {
    if (!value) return;
    const items = Array.isArray(value) ? value : [value];
    try {
        // 1. 백엔드(Rust) 업데이트
        await invoke("upsert_items", { items });

        // 2. 프론트엔드 로컬 Dexie DB 동시 업데이트
        const appDb = getAppDb();
        if (appDb) {
            // 🌟 v4 : main.ts 가 전역으로 노출한 normalizeEnvelope 를 재사용합니다.
            //  db.ts 안에 별도 enrich 복사본을 두면 두 규칙이 어긋나 인덱스가 깨집니다.
            const normalize = (window as any).normalizeEnvelope as ((docs: any[]) => any[]) | undefined;
            const prep = (docs: any[]) => normalize ? normalize(docs) : docs;

            const pages = items.filter(r =>
                r.type === "pages" || r.type === "page" ||
                (r.data && (r.data.node !== undefined || r.data.item !== undefined || r.data.origin !== undefined))
            );
            const users = items.filter(r => r.type === "team" || r.type === "user" || r.type === "member");
            const generalItems = items.filter(r => !pages.includes(r) && !users.includes(r));

            if (users.length > 0) await appDb.table("users").bulkPut(prep(users)).catch(()=>null);
            if (pages.length > 0) await appDb.table("pages").bulkPut(prep(pages)).catch(()=>null);
            if (generalItems.length > 0) await appDb.table("items").bulkPut(prep(generalItems)).catch(()=>null);
        }

        return items;
    } catch (e) {
        console.error("[DB Shim] handleUpsert error:", e);
        return [];
    }
}

Upsert["items"] = handleUpsert;
Upsert["pages"] = handleUpsert;
Upsert["users"] = handleUpsert;
Upsert["crons"] = handleUpsert;

Delete["items"] = async (q) => { return {}; };