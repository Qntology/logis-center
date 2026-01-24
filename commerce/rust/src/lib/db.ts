import { invoke } from "@tauri-apps/api/core";
import { parseItemData } from "./utils";

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

// 1. ITEMS (Main Documents)
Select["items"] = async function(query: DbQuery = {}) {
    try {
        let results: any[] = [];
        if (query.key === 'id' || query.key === 'ref') {
            if (query.key === 'id' && typeof query.value === 'string') {
                const doc = await invoke<any>("get_document", { uuid: query.value });
                if (doc) {
                    const parsed = parseItemData(doc.json_data);
                    results.push({ ...parsed, ...doc }); 
                }
            } else {
                const searchRes = await invoke<[string, string, number][]>("search_documents", {
                    query: String(query.value),
                    limit: 50,
                    offset: 0
                });
                for (const [id] of searchRes) {
                    const doc = await invoke<any>("get_document", { uuid: id });
                    if (doc) {
                        const parsed = parseItemData(doc.json_data);
                        results.push({ ...parsed, ...doc });
                    }
                }
            }
        } 
        else {
            const limit = query.limit || 50;
            const offset = query.offset || 0;
            const docs = await invoke<any[]>("get_all_documents", { limit, offset });
            results = docs.map(doc => {
                const parsed = parseItemData(doc.json_data);
                parsed.id = parsed.id || doc.uuid;
                return parsed;
            });
            if (query.type) {
                results = results.filter(r => r.type === query.type);
            }
        }
        return results;
    } catch (e) {
        console.error("[DB Shim] Select['items'] error:", e);
        return [];
    }
};

// 2. PAGES (Routing/Navigation Nodes with Join Logic)
Select["pages"] = async function(query: DbQuery = {}) {
    try {
        // [HOP 1] Fetch Page definitions (Schema & Link)
        let pageDocs: any[] = [];
        try { pageDocs = await invoke<any[]>("get_known_pages"); } catch (e) {}

        // [HOP 2] Fetch Item instances to get real Titles (Join Emulation)
        // We fetch a larger batch of items to find matching references
        let itemDocs: any[] = [];
        try { itemDocs = await invoke<any[]>("get_all_documents", { limit: 200, offset: 0 }); } catch (e) {}

        const itemsMap = new Map();
        itemDocs.forEach(doc => {
            const parsed = parseItemData(doc.json_data);
            // Store the latest title for this reference (ref could be a path or hash)
            if (doc.ref) {
                if (!itemsMap.has(doc.ref)) itemsMap.set(doc.ref, parsed.title || parsed.text || "");
            }
        });

        const unique = new Map();
        const combined = [...pageDocs, ...itemDocs];

        combined.forEach(doc => {
            if (!unique.has(doc.uuid)) {
                const parsed = parseItemData(doc.json_data);
                const typeStr = (parsed.type || doc.doc_type || "").toLowerCase();
                
                // Hallmarks of a page: 
                // 1. In 'pages' table
                // 2. Explicit type 'pages'
                // 3. Has origin/link data
                const isPage = (doc.doc_type === 'pages') || (typeStr === 'pages') || (parsed.origin || (parsed.data && parsed.data.origin));
                
                if (isPage) {
                    const data = parsed.data || parsed;
                    // [JOIN] Try to find a real title from itemsMap using the reference
                    // In legacy, doc.ref is the join key.
                    const realTitle = itemsMap.get(doc.ref) || data.title || data.text || "";

                    unique.set(doc.uuid, {
                        ...parsed,
                        id: parsed.id || doc.uuid,
                        type: parsed.type || doc.doc_type || "page",
                        title: realTitle, // [FIX] Overwrite with joined title
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

// 3. USERS (Team Members)
Select["users"] = async function(query: DbQuery = {}) {
    try {
        const docs = await invoke<any[]>("get_known_users");
        return docs.map(doc => {
            const parsed = parseItemData(doc.json_data);
            return {
                ...parsed,
                id: parsed.id || doc.uuid,
                type: parsed.type || doc.doc_type,
                data: parsed.data || parsed
            };
        });
    } catch (e) { return []; }
};

// 4. CRONS
Select["crons"] = async function(query: DbQuery = {}) {
    try {
        const tasks = await invoke<any[]>("get_active_tasks");
        if (query.key === 'ref' && query.value) {
            return tasks.filter(t => t.ref_id === query.value);
        }
        return tasks;
    } catch (e) { return []; }
};

async function handleUpsert(value: any) {
    if (!value) return;
    const items = Array.isArray(value) ? value : [value];
    try {
        await invoke("upsert_items", { items });
        return items;
    } catch (e) { return []; }
}

Upsert["items"] = handleUpsert;
Upsert["pages"] = handleUpsert;
Upsert["users"] = handleUpsert;
Upsert["crons"] = handleUpsert;

Delete["items"] = async (q) => { return {}; };
