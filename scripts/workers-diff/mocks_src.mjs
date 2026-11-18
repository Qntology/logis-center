// 가짜 Workers AI · Vectorize · D1 래퍼 — 원본 TS 와 Rust 두 진입 모듈이 env 를 Proxy 로 감싸 이것을 쥐여 준다.
// 글(모듈 소스)로 내보낸다: 하네스가 out/ 아래 두 쪽에 mocks.mjs 로 쓴다. ORIGINAL=raw 면 D1 bind 의 undefined 를 그대로 둔다.
export const mocksSource = () => `
const VECTORS = ${JSON.stringify(["commerce-logis-apac1-event", "commerce-logis-apac1-items", "commerce-logis-apac1-sales", "commerce-logis-apac1-tracking"])};
function vec(t) { let h = 2166136261; for (const c of String(t)) { h = Math.imul(h ^ c.codePointAt(0), 16777619) >>> 0; }
  const v = []; for (let i = 0; i < 768; i++) v.push(Math.round(Math.sin(h + i * 0.37) * 1e6) / 1e6); return v; }
function mockAI() {
  return {
    async run(model, input) {
      await fetch("http://record/ai", { method: "POST", body: JSON.stringify({ model, input }) });
      const texts = Array.isArray(input && input.text) ? input.text : [input && input.text];
      return { shape: [texts.length, 768], data: texts.map(vec) };
    },
  };
}
const store = new Map();
function cos(a, b) { let d = 0, x = 0, y = 0; for (let i = 0; i < Math.min(a.length, b.length); i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return x && y ? d / Math.sqrt(x * y) : 0; }
function pass(meta, filter) { if (!filter) return true; for (const k of Object.keys(filter)) { const f = filter[k];
  if (f && typeof f === "object") { if ("$eq" in f && meta[k] !== f.$eq) return false; if ("$ne" in f && meta[k] === f.$ne) return false;
    if ("$in" in f && !f.$in.includes(meta[k])) return false; if ("$gte" in f && !(meta[k] >= f.$gte)) return false; if ("$lte" in f && !(meta[k] <= f.$lte)) return false; }
  else if (meta[k] !== f) return false; } return true; }
function mockVec(name) {
  if (!store.has(name)) store.set(name, new Map());
  const s = store.get(name);
  return {
    async upsert(vectors) {
      await fetch("http://record/vec", { method: "POST", body: JSON.stringify({ name, op: "upsert", vectors }) });
      for (const v of vectors) s.set(v.id, v);
      return { mutationId: "m" + s.size, count: vectors.length, ids: vectors.map((v) => v.id) };
    },
    async query(vector, options = {}) {
      await fetch("http://record/vec", { method: "POST", body: JSON.stringify({ name, op: "query", vector: Array.from(vector || []).slice(0, 4), options }) });
      const topK = options.topK ?? 5;
      const matches = [...s.values()].filter((v) => pass(v.metadata || {}, options.filter))
        .map((v) => ({ id: v.id, score: Math.round(cos(vector || [], v.values || []) * 1e6) / 1e6, v }))
        .sort((a, b) => b.score - a.score || (a.id < b.id ? -1 : 1)).slice(0, topK)
        .map(({ id, score, v }) => { const m = { id, score }; if (options.returnMetadata) m.metadata = v.metadata || {}; if (options.returnValues) m.values = v.values; if (v.namespace) m.namespace = v.namespace; return m; });
      return { matches, count: matches.length };
    },
  };
}
const ai = mockAI();
const vecs = Object.fromEntries(VECTORS.map((n) => [n, mockVec(n)]));
// fix-bind-undefined: D1 bind 의 undefined 를 null 로 (원본은 D1_TYPE_ERROR). ORIGINAL=raw 면 감싸지 않는다
const RAW = ${JSON.stringify(process.env.ORIGINAL === "raw")};
const real = new WeakMap();
function wrapStmt(stmt) {
  const p = new Proxy(stmt, { get(t, prop) {
    if (prop === "bind") return (...args) => wrapStmt(t.bind(...args.map((a) => (a === undefined ? null : a))));
    const v = Reflect.get(t, prop); return typeof v === "function" ? v.bind(t) : v; } });
  real.set(p, stmt);
  return p;
}
function wrapD1(db) {
  if (RAW || !db || typeof db.prepare !== "function") return db;
  return new Proxy(db, { get(t, prop) {
    if (prop === "prepare") return (sql) => wrapStmt(t.prepare(sql));
    if (prop === "batch") return (list) => t.batch(list.map((s) => real.get(s) || s));
    const v = Reflect.get(t, prop); return typeof v === "function" ? v.bind(t) : v; } });
}
export function wrapEnv(env) {
  return new Proxy(env, { get(target, prop) { if (prop === "AI") return ai; if (prop in vecs) return vecs[prop]; return wrapD1(Reflect.get(target, prop)); } });
}`;
