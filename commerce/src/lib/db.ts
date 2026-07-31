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

// 1. ITEMS (Main Documents)
Select["items"] = async function(query: DbQuery = {}) {
    try {
        let results: any[] = [];
        const appDb = getAppDb();

        // Case A: Specific Key-Value lookup (Dexie DB 기반 고속 eq 쿼리)
        if (query.key && typeof query.value !== 'undefined') {
            if (appDb) {
                const cachedDocs = await appDb.table('items').where(query.key).equals(query.value).toArray();
                if (cachedDocs && cachedDocs.length > 0) {
                    return cachedDocs.map((doc: any) => {
                        const parsed = parseItemData(doc.json_data || doc.data);
                        // 🌟 [CRITICAL FIX] Dexie 객체를 리턴할 때 data 필드를 원본 JSON 객체로 확실히 꽂아줍니다.
                        return { ...parsed, ...doc, id: doc.id || doc.uuid, data: parsed };
                    });
                }
            }
            
            // Dexie 캐시에 없으면 fallback to Rust Backend
            if (query.key === 'id') {
                const doc = await invoke<any>("get_document", { uuid: query.value });
                if (doc) {
                    const parsed = parseItemData(doc.json_data);
                    const finalDoc = { ...parsed, ...doc, id: doc.uuid };
                    if (appDb) await appDb.table('items').put(finalDoc).catch(()=>null);
                    results.push(finalDoc); 
                }
                return results;
            }
        }

        // Case B: Filtered or General Search
        const limit = query.limit || 50;
        const offset = query.offset || 0;
        
        // Construct SQL filter from tags (e.g., host:..., type:...)
        const sqlFilter = await parseQueryToFilter(String(query.value || ''));

        if (sqlFilter || !query.value) {
            // [OPTIMIZED] Use get_all_documents with SQL filter for exact matches (Navigation clicks)
            const docs = await invoke<any[]>("get_all_documents", { 
                limit, 
                offset, 
                filter: sqlFilter 
            });
            
            results = docs.map(doc => {
                const parsed = parseItemData(doc.json_data);
                const docId = doc.id || doc.uuid;
                return { ...parsed, ...doc, id: docId, uuid: docId };
            });
        } else {
            // [OPTIMIZED] Use search_documents for fuzzy text search
            // We no longer call get_document in a loop! 
            // search_items in Rust returns (id, json_data, score).
            const searchRes = await invoke<[string, string, number][]>("search_documents", {
                query: String(query.value),
                limit,
                offset,
                filter: null
            });
            
            results = searchRes.map(([id, jsonData, score]) => {
                const parsed = parseItemData(jsonData);
                return { ...parsed, id, score };
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
        let pageDocs: any[] = [];
        let itemDocs: any[] = [];
        const appDb = getAppDb();

        if (appDb) {
            pageDocs = await appDb.table('pages').toArray();
        }
        
        if (pageDocs.length === 0) {
            try { pageDocs = await invoke<any[]>("get_known_pages", { filter: null }); } catch (e) {}
            if (appDb && pageDocs.length > 0) await appDb.table('pages').bulkPut(pageDocs).catch(()=>null);
        }

        try { itemDocs = await invoke<any[]>("get_all_documents", { limit: 200, offset: 0, filter: null }); } catch (e) {}

        const itemsMap = new Map();
        itemDocs.forEach(doc => {
            const parsed = parseItemData(doc.json_data);
            if (doc.ref) {
                if (!itemsMap.has(doc.ref)) itemsMap.set(doc.ref, parsed.title || parsed.text || "");
            }
        });

        const unique = new Map();
        const combined = [...pageDocs, ...itemDocs];

        combined.forEach(doc => {
            const docId = doc.id || doc.uuid;
            if (docId && !unique.has(docId)) {
                const parsed = parseItemData(doc.json_data);
                const typeStr = (parsed.type || doc.type || doc.doc_type || "").toLowerCase();
                const isPage = (doc.type === 'pages' || doc.doc_type === 'pages' || typeStr === 'pages') || (parsed.origin || (parsed.data && parsed.data.origin));
                
                if (isPage) {
                    const data = parsed.data || parsed;
                    const realTitle = itemsMap.get(doc.ref) || data.title || data.text || "";
                    
                    // 🌟 [CRITICAL FIX] before.ts 패리티: JSON 내부에 실수로 비어있는 cc, bcc, ref가 있다면 
                    // LanceDB 원본 Row가 가진 확실한 값을 유지하도록 병합 순서 및 검증을 강화합니다.
                    const safeParsed = { ...parsed };
                    if (!safeParsed.cc) delete safeParsed.cc;
                    if (!safeParsed.bcc) delete safeParsed.bcc;
                    if (!safeParsed.ref) delete safeParsed.ref;

                    unique.set(docId, {
                        ...doc, // Spread original DB fields first (guarantees valid cc, bcc, ref)
                        ...safeParsed, // Overwrite with parsed fields safely
                        id: docId,
                        type: typeStr,
                        title: realTitle,
                        data: { ...data, title: realTitle }
                    });
                }
            }
        });

        let results = Array.from(unique.values());
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
        let docs: any[] = [];
        if (appDb) {
            docs = await appDb.table('users').toArray();
        }
        if (docs.length === 0) {
            docs = await invoke<any[]>("get_known_users");
            if (appDb && docs.length > 0) await appDb.table('users').bulkPut(docs).catch(()=>null);
        }
        
        return docs.map(doc => {
            const parsed = parseItemData(doc.json_data || doc.data);
            return {
                ...parsed,
                ...doc, // 🌟 [CRITICAL FIX] Rust의 TradeDocument 속성(id, type)을 유실하지 않고 완벽하게 병합합니다.
                id: parsed.id || doc.id || doc.uuid,
                type: parsed.type || doc.type || doc.doc_type,
                // 🌟 [CRITICAL FIX] data 속성이 이중으로 중첩되는 것을 방지하고 순수한 parsed 객체를 할당합니다.
                data: parsed.data ? parsed.data : parsed
            };
        });
    } catch (e) { return []; }
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
        
        // 2. 프론트엔드 로컬 Dexie DB 동시 업데이트 (브릿지 연동 완료)
        const appDb = getAppDb();
        if (appDb) {
            const enrichForIndex = (docs: any[]) => docs.map(d => {
                const parsed = typeof d.json_data === 'string' ? (JSON.parse(d.json_data) || {}) : (d.data || d || {});
                return { 
                    ...d, 
                    no: parsed.no?.toString(), 
                    code: parsed.code?.toString(), 
                    tracking_number: parsed.tracking_number?.toString(), 
                    goods: parsed.goods?.toString(), 
                    order: parsed.order?.toString(), 
                    tracking: parsed.tracking?.toString(), 
                    stock_keeping_unit: parsed.stock_keeping_unit?.toString(), 
                    barcode: parsed.barcode?.toString(), 
                    index: parsed.index?.toString(),
                    // 🌟 [CRITICAL FIX] LanceDB의 물리적 컬럼 데이터를 Dexie 루트에 100% 동기화
                    status: d.status ?? parsed.status ?? 0,
                    amount: d.amount ?? parsed.amount ?? parsed.total_amount ?? 0,
                    mode: d.mode ?? parsed.mode ?? 'commerce',
                    is_masked: d.is_masked ?? parsed.is_masked ?? false,
                    text: d.text ?? parsed.text ?? "",
                    masked_text: d.masked_text ?? parsed.masked_text ?? "",
                    created_at: d.created_at_ts ?? d.created_at ?? parsed.created_at ?? 0,
                    updated_at: d.updated_at_ts ?? d.updated_at ?? parsed.updated_at ?? 0,
                    data: parsed
                };
            });

            const pages = items.filter(r => r.type === "pages" || r.type === "page" || (r.data && (r.data.node !== undefined || r.data.item !== undefined)));
            const users = items.filter(r => r.type === "team" || r.type === "user" || r.type === "member");
            const generalItems = items.filter(r => !pages.includes(r) && !users.includes(r));
            
            if (users.length > 0) await appDb.table("users").bulkPut(users).catch(()=>null);
            if (pages.length > 0) await appDb.table("pages").bulkPut(pages).catch(()=>null);
            if (generalItems.length > 0) await appDb.table("items").bulkPut(enrichForIndex(generalItems)).catch(()=>null);
        }
        
        return items;
    } catch (e) { return []; }
}

Upsert["items"] = handleUpsert;
Upsert["pages"] = handleUpsert;
Upsert["users"] = handleUpsert;
Upsert["crons"] = handleUpsert;

Delete["items"] = async (q) => { return {}; };