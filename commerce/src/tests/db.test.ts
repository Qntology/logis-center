import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

import { invoke } from "@tauri-apps/api/core";
import { Delete, Select, Upsert, envelopeToDoc, parseQueryToFilter } from "../lib/db";
import { makeDb, type FakeDb, type Row } from "./helpers/fakedb";
import { callsOf } from "./helpers/runtime";

const invokeMock = vi.mocked(invoke) as unknown as ReturnType<typeof vi.fn>;
const w = window as any;

/** db.ts reads the Dexie instance from window.appDb at call time. */
function useDb(seed: Record<string, Row[]> = {}): FakeDb {
  const db = makeDb(seed);
  w.appDb = db;
  return db;
}

/** Route invoke() by command name. */
function serve(routes: Record<string, (args: any) => unknown>) {
  invokeMock.mockImplementation(async (cmd: string, args: any) => (routes[cmd] ? routes[cmd](args) : undefined));
}

afterEach(() => {
  delete w.appDb;
  delete w.normalizeEnvelope;
});

describe("parseQueryToFilter", () => {
  it("turns host:/type: tags into an envelope SQL filter (host hashed with hashId)", async () => {
    expect(await parseQueryToFilter("host:example.com type:BL shanghai")).toBe(
      "cc = '0x5134846d336b9828abe829c91bdaa2d07fea6a28' AND type = 'bl'",
    );
  });

  it("returns null when there are no tags", async () => {
    expect(await parseQueryToFilter("")).toBeNull();
    expect(await parseQueryToFilter("plain words")).toBeNull();
    expect(await parseQueryToFilter("mode:list")).toBeNull();
  });
});

describe("envelopeToDoc", () => {
  it("keeps object data and lifts envelope fields to the root", () => {
    const data = { title: "T", text: "body", created_at: 3 };
    expect(envelopeToDoc({ id: "i1", type: "goods", mode: "commerce", ref: "R", updated_at: 9, data })).toEqual({
      id: "i1",
      uuid: "i1",
      type: "goods",
      flag: undefined,
      from: undefined,
      to: undefined,
      cc: undefined,
      bcc: undefined,
      ref: "R",
      mode: "commerce",
      created_at: 3,
      updated_at: 9,
      text: "body",
      data,
    });
  });

  it("parses json_data strings and defaults timestamps/text", () => {
    expect(envelopeToDoc({ id: "r1", type: "BL", json_data: '{"vessel":"V"}' })).toMatchObject({
      id: "r1",
      uuid: "r1",
      created_at: 0,
      updated_at: 0,
      text: "",
      data: { vessel: "V" },
    });
    expect(envelopeToDoc(null)).toBeNull();
  });
});

describe("Select.items", () => {
  const goods = { id: "g1", type: "goods", mode: "commerce", created_at: 1, updated_at: 2, data: { index: 5, no: "A-1", title: "T" } };

  it("answers key lookups from Dexie without touching Rust", async () => {
    useDb({ items: [goods] });
    const docs = await Select.items({ key: "no", value: "A-1" });
    expect(docs).toHaveLength(1);
    expect(docs[0]).toMatchObject({ id: "g1", uuid: "g1", type: "goods", text: "", data: { title: "T" } });
    expect((await Select.items({ key: "type", value: "goods" })).map((d) => d.id)).toEqual(["g1"]);
    expect(invokeMock).not.toHaveBeenCalled();
  });

  it("falls back to get_document for an id miss and caches the result in Dexie", async () => {
    const db = useDb();
    serve({ get_document: ({ uuid }) => (uuid === "d9" ? { id: "d9", type: "BL", json_data: '{"vessel":"V"}', created_at_ts: 10 } : null) });
    expect(await Select.items({ key: "id", value: "nope" })).toEqual([]);
    const [doc] = await Select.items({ key: "id", value: "d9" });
    expect(doc).toMatchObject({ id: "d9", type: "BL", created_at: 10, data: { vessel: "V" } });
    expect(callsOf(invokeMock, "get_document")).toEqual([{ uuid: "nope" }, { uuid: "d9" }]);
    expect(db.ids("items")).toEqual(["d9"]);
  });

  it("lists documents via get_all_documents (paging + *_ts timestamps)", async () => {
    useDb();
    serve({ get_all_documents: () => [{ id: "r1", type: "BL", json_data: '{"vessel":"V"}', created_at_ts: 10, updated_at_ts: 20 }] });
    expect(await Select.items({ limit: 3, offset: 6 })).toEqual([
      { id: "r1", uuid: "r1", type: "BL", flag: undefined, from: undefined, to: undefined, cc: undefined, bcc: undefined, ref: undefined, mode: undefined, created_at: 10, updated_at: 20, text: "", data: { vessel: "V" } },
    ]);
    expect(callsOf(invokeMock, "get_all_documents")).toEqual([{ limit: 3, offset: 6, filter: null }]);
  });

  it("sends tag queries as a SQL filter", async () => {
    useDb();
    serve({ get_all_documents: () => [] });
    await Select.items({ value: "host:example.com type:BL" });
    expect(callsOf(invokeMock, "get_all_documents")).toEqual([
      { limit: 50, offset: 0, filter: "cc = '0x5134846d336b9828abe829c91bdaa2d07fea6a28' AND type = 'bl'" },
    ]);
  });

  it("runs free text through search_documents and maps the (id, json, score) tuples", async () => {
    useDb();
    serve({ search_documents: () => [["s1", '{"type":"goods","title":"x"}', 0.9]] });
    expect(await Select.items({ value: "shoes" })).toEqual([
      { id: "s1", uuid: "s1", type: "goods", mode: "commerce", created_at: 0, updated_at: 0, text: "", score: 0.9, data: { type: "goods", title: "x" } },
    ]);
    expect(callsOf(invokeMock, "search_documents")).toEqual([{ query: "shoes", limit: 50, offset: 0, filter: null }]);
  });

  it("returns [] when the backend throws", async () => {
    useDb();
    invokeMock.mockRejectedValue(new Error("db down"));
    expect(await Select.items({ value: "x" })).toEqual([]);
  });

  it.fails("BUG-DB-1 db.ts:100 data.* lookup values are String()-ed, so numeric canonical fields (data.index = 5) never match", async () => {
    useDb({ items: [goods] });
    serve({ search_documents: () => [], get_all_documents: () => [] });
    expect((await Select.items({ key: "index", value: 5 })).map((d) => d.id)).toEqual(["g1"]);
  });

  it.fails("BUG-DB-2 db.ts:138 a mixed query ('type:BL shanghai') silently drops the free-text part", async () => {
    useDb();
    serve({ search_documents: () => [], get_all_documents: () => [] });
    await Select.items({ value: "type:BL shanghai" });
    const searchedText = invokeMock.mock.calls.some(([cmd, args]: any[]) => cmd === "search_documents" && String(args.query).includes("shanghai"));
    expect(searchedText).toBe(true);
  });

  it.fails("BUG-DB-3 db.ts:157-158 one search row whose JSON is 'null' makes the whole result empty", async () => {
    useDb();
    serve({ search_documents: () => [["s1", "null", 0.9], ["s2", '{"type":"goods"}', 0.8]] });
    expect((await Select.items({ value: "shoes" })).map((d) => d.id)).toContain("s2");
  });
});

describe("Select.pages", () => {
  it("keeps selector-cache rows, lower-cases data.type and borrows titles from linked items", async () => {
    useDb({
      pages: [
        { id: "p1", type: "pages", ref: "R1", data: { origin: "https://a.com", type: "Goods" } },
        { id: "p2", type: "pages", data: { node: 0 } },
        { id: "p3", type: "pages", data: {} },
      ],
      items: [{ id: "i1", ref: "R1", data: { title: "Linked Title" } }],
    });
    const pages = await Select.pages({});
    expect(pages.map((p) => [p.id, p.type, p.title])).toEqual([
      ["p1", "goods", "Linked Title"],
      ["p2", "pages", ""],
    ]);
    expect(pages[0].data.title).toBe("Linked Title");
    expect((await Select.pages({ key: "id", value: "p2" })).map((p) => p.id)).toEqual(["p2"]);
  });

  it("loads from get_known_pages on a cold cache and fills Dexie", async () => {
    const db = useDb();
    serve({ get_known_pages: () => [{ id: "k1", type: "pages", json_data: '{"origin":"https://a.com","item":true}', created_at_ts: 4 }] });
    const pages = await Select.pages({});
    expect(pages.map((p) => p.id)).toEqual(["k1"]);
    expect(pages[0]).toMatchObject({ mode: "commerce", created_at: 4 });
    expect(callsOf(invokeMock, "get_known_pages")).toEqual([{ filter: null }]);
    expect(db.ids("pages")).toEqual(["k1"]);
  });
});

describe("Select.users / Select.crons", () => {
  it("maps Dexie user rows to the flat navigation view", async () => {
    useDb({ users: [{ id: "u1", type: "user", from: "F", to: "T", data: { name: "Kim", type: "member" } }] });
    expect(await Select.users({})).toEqual([
      { id: "u1", uuid: "u1", type: "member", flag: undefined, from: "F", to: "T", cc: undefined, bcc: undefined, ref: undefined, created_at: undefined, updated_at: undefined, name: "Kim", data: { name: "Kim", type: "member" } },
    ]);
  });

  it("falls back to get_known_users without a local DB", async () => {
    serve({ get_known_users: () => [{ id: "u2", type: "team", json_data: '{"name":"Team"}', created_at_ts: 3 }] });
    expect(await Select.users({})).toMatchObject([{ id: "u2", type: "team", name: "Team", flag: "", created_at: 3, updated_at: 0 }]);
  });

  it("filters active tasks by ref (root field or data_json.ref)", async () => {
    serve({ get_active_tasks: () => [{ id: "t1", ref: "R" }, { id: "t2", data_json: '{"ref":"R"}' }, { id: "t3", ref: "X" }] });
    expect((await Select.crons({ key: "ref", value: "R" })).map((t: Row) => t.id)).toEqual(["t1", "t2"]);
    expect(await Select.crons({})).toHaveLength(3);
  });

  it.fails("BUG-DB-4 db.ts:318 one task with malformed data_json makes Select.crons return []", async () => {
    serve({ get_active_tasks: () => [{ id: "t1", ref: "R" }, { id: "t3", data_json: "not json" }] });
    expect((await Select.crons({ key: "ref", value: "R" })).map((t: Row) => t.id)).toEqual(["t1"]);
  });
});

describe("Upsert / Delete", () => {
  it("writes to Rust first, then routes users/pages/items to their Dexie tables via window.normalizeEnvelope", async () => {
    const db = useDb();
    w.normalizeEnvelope = vi.fn((docs: Row[]) => docs.map((d) => ({ ...d, normalized: true })));
    serve({ upsert_items: () => "ok" });
    const items = [
      { id: "u", type: "member", data: {} },
      { id: "pg", type: "page", data: {} },
      { id: "plain", type: "goods", data: { title: "x" } },
    ];
    expect(await Upsert.items(items)).toEqual(items);
    expect(callsOf(invokeMock, "upsert_items")).toEqual([{ items }]);
    expect(db.ids("users")).toEqual(["u"]);
    expect(db.ids("pages")).toEqual(["pg"]);
    expect(db.ids("items")).toEqual(["plain"]);
    expect(db.rows("items")[0].normalized).toBe(true);
  });

  it("returns undefined for empty input and [] when Rust rejects", async () => {
    useDb();
    expect(await Upsert.items(null)).toBeUndefined();
    invokeMock.mockRejectedValue(new Error("x"));
    expect(await Upsert.items([{ id: "a" }])).toEqual([]);
  });

  it.fails("BUG-DB-5 db.ts:339-342 ordinary documents with data.node = 0 or data.origin are routed to the pages table", async () => {
    const db = useDb();
    serve({ upsert_items: () => "ok" });
    await Upsert.items([
      { id: "g", type: "goods", data: { node: 0 } },
      { id: "a", type: "click", data: { origin: "https://shop.com" } },
    ]);
    expect(db.ids("items").sort()).toEqual(["a", "g"]);
  });

  it("Delete by id removes the document from Rust and every Dexie table", async () => {
    const db = useDb({ items: [{ id: "g" }], pages: [{ id: "g" }], users: [{ id: "g" }, { id: "keep" }] });
    serve({ delete_document: () => "ok" });
    expect(await Delete.items({ key: "id", value: "g" })).toEqual({ success: true });
    expect(callsOf(invokeMock, "delete_document")).toEqual([{ uuid: "g" }]);
    expect([db.ids("items"), db.ids("pages"), db.ids("users")]).toEqual([[], [], ["keep"]]);
  });

  it("Delete reports failure when Rust rejects", async () => {
    useDb();
    invokeMock.mockRejectedValue(new Error("x"));
    expect(await Delete.items({ key: "id", value: "g" })).toEqual({ success: false });
  });

  it.fails("BUG-DB-6 db.ts:363-377 Delete with a non-id key deletes nothing but still reports success", async () => {
    const db = useDb({ items: [{ id: "g", data: { no: "x" } }] });
    serve({ delete_document: () => "ok", delete_documents: () => "ok" });
    const res = await Delete.items({ key: "no", value: "x" });
    const deleted = !db.ids("items").includes("g");
    // Either outcome is acceptable once fixed: report failure, or actually delete the matching doc.
    expect(res.success ? deleted : true).toBe(true);
  });
});
