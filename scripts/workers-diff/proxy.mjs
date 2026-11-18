// proxy 차분 시험: 원본 TS 와 Rust(wasm) 를 같은 D1 · 같은 가짜 LLM/AI/Vectorize · 같은 가상 시계 · 같은 난수로 돌려
// 응답 · 바깥 요청(LLM) · AI/Vectorize 호출 · D1 끝 상태를 견준다.
// usage: node proxy.mjs <logis-center/workers 경로> [시나리오 이름]
import { Miniflare } from "miniflare";
import * as esbuild from "esbuild";
import { gzipSync, gunzipSync } from "node:zlib";
import fs from "node:fs";
import path from "node:path";
// 운영 Workers 의 시간대는 UTC 다: workerd(Miniflare)가 PC 시간대(KST)로 날짜를 읽지 않게
process.env.TZ = "UTC";
import { execFileSync } from "node:child_process";
import { patchOriginal } from "./fixes.mjs";
import { mocksSource } from "./mocks_src.mjs";
// 원본 TypeScript 는 저장소 이력에서 꺼낸다 (workers-rs 로 바꾸기 직전 커밋)
const ORIGINAL_COMMIT = "801b0ce0";
const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "..", "..");
async function original(name) {
  const file = path.join(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "out", "ts", `${name}.ts`);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const raw = execFileSync("git", ["-C", REPO, "show", `${ORIGINAL_COMMIT}:workers/${name}/src/index.ts`]).toString("utf8");
  // ORIGINAL=raw 면 버그를 고치기 전의 원본 그대로, 아니면 fixes.mjs 의 수정을 덧댄 "고친 원본"
  const { source, applied } = process.env.ORIGINAL === "raw" ? { source: raw, applied: [] } : await patchOriginal(name, raw);
  if (applied.length) console.log(`[${name}] fixes applied to the original: ${applied.join(", ")}`);
  fs.writeFileSync(file, source);
  return file;
}
import { SCENARIOS } from "./proxy_scenarios.mjs";

const NL = String.fromCharCode(10);
const ENTRY_TS = ['import "./prelude.mjs";', 'import W from "./proxy-ts.mjs";', 'import { wrapEnv } from "./mocks.mjs";',
  'export default { fetch(req, env, ctx) { return W.fetch(req, wrapEnv(env), ctx); } };', ''].join(NL);
const ENTRY_RS = ['import "./prelude.mjs";', 'import W from "./index.js";', 'import { wrapEnv } from "./mocks.mjs";',
  'export default { fetch(req, env, ctx) { return new W(ctx, wrapEnv(env)).fetch(req); } };', ''].join(NL);
const WORKERS = process.argv[2] && !process.argv[2].startsWith("-") && fs.existsSync(process.argv[2]) ? process.argv[2] : path.join(REPO, "workers");
const ONLY = process.env.ONLY || (process.argv[2] && !fs.existsSync(process.argv[2]) ? process.argv[2] : process.argv[3]);
const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const OUT = path.join(HERE, "out");
fs.mkdirSync(OUT, { recursive: true });

await esbuild.build({
  entryPoints: [await original("proxy")],
  bundle: true, format: "esm", platform: "browser", mainFields: ["browser", "module", "main"], conditions: ["workerd", "worker", "browser"],
  outfile: path.join(OUT, "proxy-ts.mjs"), logLevel: "error",
});
fs.copyFileSync(path.join(HERE, "prelude.mjs"), path.join(OUT, "prelude.mjs"));
fs.writeFileSync(path.join(OUT, "proxy-ts-entry.mjs"), ENTRY_TS);

const RS = path.join(OUT, "proxy-rs");
fs.mkdirSync(RS, { recursive: true });
fs.copyFileSync(path.join(WORKERS, "proxy", "build", "index_bg.wasm"), path.join(RS, "index_bg.wasm"));
{
  const js = fs.readFileSync(path.join(WORKERS, "proxy", "build", "index.js"), "utf8");
  fs.writeFileSync(path.join(RS, "index.js"), js.replace(/import source (\w+) from"\.\/index_bg\.wasm"/, 'import $1 from"./index_bg.wasm"'));
}
fs.copyFileSync(path.join(HERE, "prelude.mjs"), path.join(RS, "prelude.mjs"));
fs.writeFileSync(path.join(RS, "entry.mjs"), ENTRY_RS);

const MIGRATION = fs.readFileSync(path.join(WORKERS, "migrations", "0001_create_comments_table.sql"), "utf8");
const D1 = [
  "commerce_logis_apac", "commerce_logis_apac1_items", "commerce_logis_apac1_event", "commerce_logis_apac1_sales",
  "commerce_logis_apac1_talks", "commerce_logis_apac1_tracking", "commerce_logis_center", "DB",
];
const VECTORS = ["commerce-logis-apac1-event", "commerce-logis-apac1-items", "commerce-logis-apac1-sales", "commerce-logis-apac1-tracking"];

// 가짜 Workers AI · Vectorize — 두 워커 모두 같은 진입 모듈이 env 를 Proxy 로 감싸 이것을 쥐여 준다
const MOCKS = mocksSource();

fs.writeFileSync(path.join(OUT, "mocks.mjs"), MOCKS);
fs.writeFileSync(path.join(RS, "mocks.mjs"), MOCKS);
const T0 = 1760000000000;
// 원본(pako 가 D1 의 숫자 배열을 읽음)이 읽을 수 있는 gzip 을 고른다: 저장 블록이 생기면 수준을 바꾸고, 그래도 안 되면 채움 칸을 붙인다.
// 행에 "unreadable: true" 를 주면 일부러 못 읽는 것을 넣는다 (원본 버그 재현용)
import pako from "pako";
const readable = (b) => { try { pako.ungzip(Array.from(b)); return true; } catch { return false; } };
const gz = (o, unreadable = false) => {
  if (unreadable) { for (const level of [0, 1, 6, 9]) { const b = gzipSync(Buffer.from(JSON.stringify(o)), { level }); if (!readable(b)) return b; } }
  let obj = o;
  for (let pad = 0; pad < 8; pad++) {
    for (const level of [6, 9, 1]) { const b = gzipSync(Buffer.from(JSON.stringify(obj)), { level }); if (readable(b)) return b; }
    obj = { ...o, _pad: "abcabcabc".repeat(20 * (pad + 1)) };
  }
  throw new Error("no readable gzip for " + JSON.stringify(o).slice(0, 80));
};
const ungz = (b) => { try { return JSON.parse(gunzipSync(Buffer.from(b)).toString("utf8")); } catch { return { raw: Buffer.from(b).toString("base64") }; } };

async function run(variant, scenario) {
  const records = [];
  const isTs = variant === "ts";
  const outbound = async (req) => {
    const url = new URL(req.url);
    const text = await req.text();
    if (url.hostname === "record") {
      records.push({ kind: url.pathname.slice(1), ...JSON.parse(text) });
      return new Response("ok");
    }
    let body; try { body = JSON.parse(text); } catch { body = text; }
    const auth = req.headers.get("authorization");
    records.push({ kind: "fetch", url: req.url, method: req.method, auth, body });
    const reply = scenario.llm({ url: req.url, body, records });
    return new Response(JSON.stringify(reply), { headers: { "content-type": "application/json" } });
  };
  const mf = new Miniflare({
    workers: [
      {
        name: "main",
        modules: true,
        scriptPath: isTs ? path.join(OUT, "proxy-ts-entry.mjs") : path.join(RS, "entry.mjs"),
        modulesRules: [{ type: "CompiledWasm", include: ["**/*.wasm"], fallthrough: true }, { type: "ESModule", include: ["**/*.js", "**/*.mjs"] }],
        compatibilityDate: "2025-12-01",
        d1Databases: Object.fromEntries(D1.map((n) => [n, `${n}-${variant}-${scenario.name}`])),
        outboundService: outbound,
      },
    ],
  });
  for (const n of D1) {
    const db = await mf.getD1Database(n, "main");
    for (const stmt of MIGRATION.split(";").map((s) => s.trim()).filter(Boolean)) await db.prepare(stmt).run();
  }
  for (const row of scenario.rows) {
    const db = await mf.getD1Database(row.db, "main");
    const cols = Object.keys(row.values);
    const vals = cols.map((c) => (c === "data" && row.values[c] !== null && typeof row.values[c] === "object" ? gz(row.values[c], row.unreadable) : row.values[c]));
    await db.prepare(`INSERT OR REPLACE INTO ${row.table} (${cols.map((c) => `"${c}"`).join(",")}) VALUES (${cols.map((_, i) => `?${i + 1}`).join(",")})`).bind(...vals).run();
  }
  let response;
  try {
    const body = scenario.request === null ? undefined : gz(scenario.request);
    const res = await mf.dispatchFetch("https://proxy.commerce.logis.center/", { method: "POST", body });
    response = { status: res.status, type: res.headers.get("content-type"), body: await res.text() };
  } catch (e) {
    response = { error: String(e) };
  }
  const state = {};
  for (const n of D1) {
    const db = await mf.getD1Database(n, "main");
    for (const table of ["tasks", "users", "pages", "items", "sales", "event", "tracking", "talks", "console", "views", "crons"]) {
      const { results } = await db.prepare(`SELECT * FROM ${table} ORDER BY ${table === "console" ? "log, created_at" : "id"}`).all();
      if (results.length) state[`${n}.${table}`] = results.map((r) => ({ ...r, data: r.data ? ungz(r.data) : r.data }));
    }
  }
  await mf.dispose();
  return { response, records, state };
}

let fail = 0;
for (const scenario of SCENARIOS.filter((s) => !ONLY || s.name === ONLY)) {
  const ts = await run("ts", scenario);
  const rs = await run("rs", scenario);
  // hashId() 의 무작위 id(원본: ethers.Wallet.createRandom)는 실행마다 다르다: 한쪽에만 있는 0x 주소는 나온 순서대로 자리표로
  const ADDR = /0x[0-9a-f]{40}/g;
  const ra = JSON.stringify(ts, null, 1), rb = JSON.stringify(rs, null, 1);
  const inA = new Set(ra.match(ADDR) || []), inB = new Set(rb.match(ADDR) || []);
  const norm = (s, other) => { const seen = new Map(); return s.replace(ADDR, (x) => (other.has(x) ? x : (seen.has(x) ? seen.get(x) : (seen.set(x, "<random#" + seen.size + ">"), seen.get(x))))); };
  const a = norm(ra, inB), b = norm(rb, inA);
  fs.writeFileSync(path.join(OUT, `proxy-${scenario.name}-ts.json`), a);
  fs.writeFileSync(path.join(OUT, `proxy-${scenario.name}-rs.json`), b);
  const same = a === b;
  if (!same) fail++;
  const kinds = (r) => r.records.map((x) => x.kind[0]).join("");
  console.log(`${same ? "SAME" : "DIFF"}  ${scenario.name}  status=${ts.response.status}/${rs.response.status}  records=${kinds(ts)}/${kinds(rs)}  tables=${Object.keys(ts.state).length}/${Object.keys(rs.state).length}`);
}
console.log(fail ? `${fail} scenario(s) differ` : "all scenarios identical");
process.exit(fail ? 1 : 0);
