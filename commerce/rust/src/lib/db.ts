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
    // content.js specific filters
    from?: string;
    to?: string;
    ref?: string;
    type?: string;
}

// --- Shim Implementation ---

export const Select: Record<string, (query: any) => Promise<any[]>> = {};
export const Upsert: Record<string, (value: any) => Promise<any>> = {};
export const Delete: Record<string, (query: any) => Promise<any>> = {};

// 1. ITEMS (Main Documents)
Select["items"] = async function(query: DbQuery = {}) {
    try {
        let results: any[] = [];

        // Case A: Search by ID/Ref specifically (Simulated via search or get_document)
        if (query.key === 'id' || query.key === 'ref') {
            // Use get_document for precise ID lookup
            if (query.key === 'id' && typeof query.value === 'string') {
                const doc = await invoke<any>("get_document", { uuid: query.value });
                if (doc) {
                    const parsed = parseItemData(doc.json_data);
                    // Merge top-level doc properties
                    results.push({ ...parsed, ...doc }); 
                }
            } else {
                // If searching by ref, we might need a search query
                // For now, fallback to search_documents with the ref string
                const searchRes = await invoke<[string, string, number][]>("search_documents", {
                    query: String(query.value),
                    limit: 50,
                    offset: 0
                });
                
                // Fetch full docs for these results
                for (const [id] of searchRes) {
                    const doc = await invoke<any>("get_document", { uuid: id });
                    if (doc) {
                        const parsed = parseItemData(doc.json_data);
                        results.push({ ...parsed, ...doc });
                    }
                }
            }
        } 
        // Case B: General List / Full Fetch
        else {
            const limit = query.limit || 50;
            const offset = query.offset || 0;

            const docs = await invoke<any[]>("get_all_documents", { limit, offset });
            
            results = docs.map(doc => {
                const parsed = parseItemData(doc.json_data);
                // Ensure ID parity
                parsed.id = parsed.id || doc.uuid;
                return parsed;
            });

            // Client-side filtering for properties not supported by get_all_documents
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

// 2. PAGES (Routing/Navigation Nodes)
Select["pages"] = async function(query: DbQuery = {}) {
    // [FIX] Independent fetching to prevent one failure from blocking all results
    let pageDocs: any[] = [];
    let itemDocs: any[] = [];

    try {
        pageDocs = await invoke<any[]>("get_known_pages");
    } catch (e) {
        console.warn("[DB Shim] Failed to fetch from 'pages' table (might be empty/missing):", e);
    }

    try {
        itemDocs = await invoke<any[]>("get_all_documents", { limit: 200, offset: 0 });
    } catch (e) {
        console.warn("[DB Shim] Failed to fetch from 'items' table:", e);
    }

    try {
        const combined = [...pageDocs, ...itemDocs];
        const unique = new Map();

        combined.forEach(doc => {
            if (!unique.has(doc.uuid)) {
                const parsed = parseItemData(doc.json_data);
                
                // Check if it's a page-like item
                // 1. Explicitly in 'pages' table (doc_type check logic in backend might tag them)
                // 2. Has 'origin' or 'link' properties (the hallmark of a page node)
                // 3. Has type 'pages' or 'page'
                const typeStr = (parsed.type || doc.doc_type || "").toLowerCase();
                const isPage = (doc.doc_type === 'pages') || 
                               (typeStr === 'pages' || typeStr === 'page') || 
                               (parsed.origin || (parsed.data && parsed.data.origin));
                
                if (isPage) {
                    unique.set(doc.uuid, {
                        ...parsed,
                        id: parsed.id || doc.uuid,
                        type: parsed.type || doc.doc_type || "page",
                        data: parsed.data || parsed // Ensure data structure consistency
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
        console.error("[DB Shim] Select['pages'] aggregation error:", e);
        return [];
    }
};

// 3. USERS (Team Members)
Select["users"] = async function(query: DbQuery = {}) {
    try {
        const docs = await invoke<any[]>("get_known_users");
        
        let results = docs.map(doc => {
            const parsed = parseItemData(doc.json_data);
            return {
                ...parsed,
                id: parsed.id || doc.uuid,
                type: parsed.type || doc.doc_type, // team vs user
                data: parsed.data || parsed
            };
        });

        if (query.key === 'id' && query.value) {
            results = results.filter(r => r.id === query.value);
        }

        return results;
    } catch (e) {
        console.error("[DB Shim] Select['users'] error:", e);
        return [];
    }
};

// 4. CRONS (Tasks) - Stub for now or map to active tasks
Select["crons"] = async function(query: DbQuery = {}) {
    // Rust has 'get_active_tasks'
    try {
        const tasks = await invoke<any[]>("get_active_tasks");
        if (query.key === 'ref' && query.value) {
            return tasks.filter(t => t.ref_id === query.value);
        }
        return tasks;
    } catch (e) {
        console.error("[DB Shim] Select['crons'] error:", e);
        return [];
    }
};

// --- Upsert Implementation ---
// Generic upsert handler using the new Rust command
async function handleUpsert(value: any) {
    if (!value) return;
    const items = Array.isArray(value) ? value : [value];
    try {
        // [NEW] Call the Rust command to save items
        await invoke("upsert_items", { items });
        return items;
    } catch (e) {
        console.error("[DB Shim] Upsert error:", e);
        return [];
    }
}

Upsert["items"] = handleUpsert;
Upsert["pages"] = handleUpsert;
Upsert["users"] = handleUpsert;
Upsert["crons"] = handleUpsert; // Will go to tasks logic in backend

Delete["items"] = async (q) => { console.warn("Delete not implemented in frontend shim"); return {}; };