// apac 차분 시험: 원본 TS 와 Rust(wasm) 를 같은 D1 · 같은 가상 시계 · 같은 난수로 돌려 바깥으로 나간 요청과 D1 끝 상태를 견준다.
// usage: node apac.mjs <logis-center/workers 경로>
import { Miniflare } from "miniflare";
import * as esbuild from "esbuild";
import { gzipSync, gunzipSync } from "node:zlib";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { patchOriginal } from "./fixes.mjs";
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

const WORKERS = process.argv[2] && !process.argv[2].startsWith("-") && fs.existsSync(process.argv[2]) ? process.argv[2] : path.join(REPO, "workers");
const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const OUT = path.join(HERE, "out");
fs.mkdirSync(OUT, { recursive: true });

// 원본 TS 묶기
await esbuild.build({
  entryPoints: [await original("apac")],
  bundle: true, format: "esm", platform: "neutral", mainFields: ["module", "main"],
  outfile: path.join(OUT, "apac-ts.mjs"), logLevel: "error",
});
fs.copyFileSync(path.join(HERE, "prelude.mjs"), path.join(OUT, "prelude.mjs"));
fs.writeFileSync(path.join(OUT, "apac-ts-entry.mjs"), `import "./prelude.mjs";\nexport { default } from "./apac-ts.mjs";\n`);

// Rust 빌드 산출물 (worker-build) 을 out/rs 로
const RS = path.join(OUT, "apac-rs");
fs.mkdirSync(RS, { recursive: true });
fs.copyFileSync(path.join(WORKERS, "apac", "build", "index_bg.wasm"), path.join(RS, "index_bg.wasm"));
// Miniflare 의 모듈 해석기는 `import source X from "*.wasm"`(소스 단계 가져오기)를 못 읽는다. CompiledWasm 의 기본 가져오기도 같은
// WebAssembly.Module 이므로 시험 사본에서만 바꾼다.
{
  const js = fs.readFileSync(path.join(WORKERS, "apac", "build", "index.js"), "utf8");
  const n = (js.match(/import source (\w+) from"\.\/index_bg\.wasm"/g) || []).length;
  if (n > 1) throw new Error(`expected at most one source-phase wasm import, found ${n}`);
  fs.writeFileSync(path.join(RS, "index.js"), js.replace(/import source (\w+) from"\.\/index_bg\.wasm"/, 'import $1 from"./index_bg.wasm"'));
}
fs.copyFileSync(path.join(HERE, "prelude.mjs"), path.join(RS, "prelude.mjs"));
fs.writeFileSync(path.join(RS, "entry.mjs"), `import "./prelude.mjs";\nexport * from "./index.js";\nexport { default } from "./index.js";\n`);

const MIGRATION = fs.readFileSync(path.join(WORKERS, "migrations", "0001_create_comments_table.sql"), "utf8");

const T0 = 1760000000000;
const gz = (o) => gzipSync(Buffer.from(JSON.stringify(o)));

// 시나리오: D1 에 넣을 행
const team = { id: "0xteam", from: "0xowner", data: { base: { pages: {} }, address: [] } };
const SCENARIOS = {
  empty: [],
  two_tasks_prev_done: [
    { table: "tasks", id: "T1", created_at: T0 - 50000, updated_at: 5, data: { id: "T1", ref: "r1", created_at: T0 - 50000, team, flag: "kr", method: "x" } },
    { table: "tasks", id: "T2", created_at: T0 - 40000, updated_at: 0, data: { id: "T2", ref: "r2", created_at: T0 - 40000, team, flag: "kr" } },
    { table: "users", id: "0xteam", type: "team", created_at: T0 - 90000, updated_at: 0, data: team.data },
  ],
  team_changed: [
    { table: "tasks", id: "T1", created_at: T0 - 50000, updated_at: 5, data: { id: "T1", ref: "r1", created_at: T0 - 50000, team, flag: "kr" } },
    { table: "tasks", id: "T2", created_at: T0 - 40000, updated_at: 0, data: { id: "T2", ref: "r2", created_at: T0 - 40000, team, flag: "kr" } },
    { table: "users", id: "0xteam", type: "team", created_at: T0 - 90000, updated_at: 0, data: { base: { pages: { x: 1 } }, address: [] } },
  ],
  lock_first: [
    { table: "tasks", id: "lock", created_at: 1, updated_at: 0, data: null },
    { table: "tasks", id: "T2", created_at: T0 - 40000, updated_at: 0, data: { id: "T2", ref: "r2", created_at: T0 - 40000, team, flag: "kr" } },
  ],
  fresh_tasks: [
    { table: "tasks", id: "T1", created_at: T0 - 50000, updated_at: 0, data: { id: "T1", ref: "r1", created_at: T0 - 50000, team, flag: "kr" } },
    { table: "tasks", id: "T2", created_at: T0 - 40000, updated_at: 0, data: { id: "T2", ref: "r2", created_at: T0 - 40000, team, flag: "kr" } },
  ],
};

async function run(variant, scenario) {
  console.log(`=== ${variant} ${scenario}`);
  const outbound = [];
  const isTs = variant === "ts";
  const mf = new Miniflare({
    modules: true,
    scriptPath: isTs ? path.join(OUT, "apac-ts-entry.mjs") : path.join(RS, "entry.mjs"),
    modulesRules: [{ type: "CompiledWasm", include: ["**/*.wasm"], fallthrough: true }, { type: "ESModule", include: ["**/*.js", "**/*.mjs"] }],
    compatibilityDate: "2025-12-01",
    d1Databases: { commerce_logis_apac: `apac-${variant}-${scenario}`, DB: `db-${variant}-${scenario}` },
    bindings: { region: "commerce_logis_apac", gemini1: "G1", gemini2: "G2", deepinfra: "DI" },
    d1Persist: false,
    outboundService: async (req) => {
      const buf = Buffer.from(await req.arrayBuffer());
      let body;
      try { body = JSON.parse(gunzipSync(buf).toString("utf8")); } catch { body = buf.toString("base64"); }
      outbound.push({ url: req.url, method: req.method, contentType: req.headers.get("content-type"), encoding: req.headers.get("content-encoding"), body });
      const limits = { ...(body.limits || {}) };
      if (body.id) limits[String(body.id).toUpperCase()] = true;
      return new Response(JSON.stringify({ models: body.models, limits }), { headers: { "content-type": "application/json" } });
    },
  });
  const db = await mf.getD1Database("commerce_logis_apac");
  for (const stmt of MIGRATION.split(";").map((s) => s.trim()).filter(Boolean)) await db.prepare(stmt).run();
  for (const row of SCENARIOS[scenario]) {
    const data = row.data === null ? null : gz(row.data);
    await db.prepare(`INSERT OR REPLACE INTO ${row.table} ("id","type","data","created_at","updated_at") VALUES (?1,?2,?3,?4,?5)`)
      .bind(row.id, row.type ?? "", data, row.created_at, row.updated_at).run();
  }
  const worker = await mf.getWorker();
  let error = null;
  try { await worker.scheduled({ cron: "* * * * *", scheduledTime: new Date(T0) }); } catch (e) { error = String(e); }
  const state = {};
  for (const table of ["tasks", "users"]) {
    const { results } = await db.prepare(`SELECT * FROM ${table} ORDER BY id`).all();
    state[table] = results.map((r) => ({ ...r, data: r.data ? JSON.parse(gunzipSync(Buffer.from(r.data)).toString("utf8")) : r.data }));
  }
  await mf.dispose();
  return { error, outbound, state };
}

let fail = 0;
for (const scenario of Object.keys(SCENARIOS).filter((s) => !process.env.ONLY || s === process.env.ONLY)) {
  const ts = await run("ts", scenario);
  const rs = await run("rs", scenario);
  const a = JSON.stringify(ts, null, 1), b = JSON.stringify(rs, null, 1);
  fs.writeFileSync(path.join(OUT, `apac-${scenario}-ts.json`), a);
  fs.writeFileSync(path.join(OUT, `apac-${scenario}-rs.json`), b);
  const same = a === b;
  if (!same) fail++;
  console.log(`${same ? "SAME" : "DIFF"}  ${scenario}  outbound=${ts.outbound.length}/${rs.outbound.length}  tasks=${ts.state.tasks.length}/${rs.state.tasks.length}  err=${ts.error}/${rs.error}`);
}
console.log(fail ? `${fail} scenario(s) differ` : "all scenarios identical");
process.exit(fail ? 1 : 0);
