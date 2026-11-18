// commerce 차분 시험: 원본 TS(commerce-logis-center fd1e991, 짝 없는 `}else if` 한 곳만 고침)와 Rust(wasm)를
// 같은 D1 · 같은 가짜 S3/AI/Vectorize · 같은 가상 시계 · 같은 난수로 돌려 응답(상태 · 머리글 · 본문) · 바깥 요청 · D1 끝 상태 · S3 끝 상태를 견준다.
// 시나리오는 요청 여러 개를 차례로 보낸다(앞 응답의 Set-Cookie 를 뒤 요청에 실을 수 있다).
// usage: node commerce.mjs [시나리오 이름]   (원본 저장소 위치: COMMERCE_REPO, 기본 logis-center 옆 commerce-logis-center)
import { Miniflare } from "miniflare";
import * as esbuild from "esbuild";
import { gzipSync, gunzipSync } from "node:zlib";
import fs from "node:fs";
import path from "node:path";
process.env.TZ = "UTC";
import { execFileSync } from "node:child_process";
import { AwsV4Signer } from "aws4fetch";
import { patchOriginal } from "./fixes.mjs";
import { mocksSource } from "./mocks_src.mjs";
import { SCENARIOS, AWS } from "./commerce_scenarios.mjs";

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const REPO = path.resolve(HERE, "..", "..");
const WORKERS = path.join(REPO, "workers");
const OUT = path.join(HERE, "out");
const ONLY = process.env.ONLY || process.argv[2];
const NL = String.fromCharCode(10);

// 원본: 별도 저장소의 고정 커밋. HEAD 는 1594 줄의 `}else if(request.method == 'PUT'){` 가 짝이 없어 구문 오류라, 이 한 곳은 raw 에서도 고친다
const ORIGINAL_REPO = process.env.COMMERCE_REPO || path.resolve(REPO, "..", "commerce-logis-center");
const ORIGINAL_COMMIT = "fd1e991";
const SYNTAX_FIX = { find: "}else if(request.method == 'PUT'){", replace: "if(request.method == 'PUT'){" };
async function original() {
  const dir = path.join(OUT, "commerce-src");
  fs.mkdirSync(dir, { recursive: true });
  const show = (f) => execFileSync("git", ["-C", ORIGINAL_REPO, "show", `${ORIGINAL_COMMIT}:src/${f}`]).toString("utf8");
  let raw = show("index.ts");
  if (raw.split(SYNTAX_FIX.find).length !== 2) throw new Error("syntax fix: expected exactly one match");
  raw = raw.replace(SYNTAX_FIX.find, SYNTAX_FIX.replace);
  const { source, applied } = process.env.ORIGINAL === "raw" ? { source: raw, applied: [] } : await patchOriginal("commerce", raw);
  if (applied.length) console.log(`[commerce] fixes applied to the original: ${applied.join(", ")}`);
  fs.writeFileSync(path.join(dir, "index.ts"), source);
  fs.writeFileSync(path.join(dir, "renderHtml.ts"), show("renderHtml.ts"));
  return path.join(dir, "index.ts");
}

const ENTRY_TS = ['import "./prelude.mjs";', 'import W from "./commerce-ts.mjs";', 'import { wrapEnv } from "./mocks.mjs";',
  'export default { fetch(req, env, ctx) { return W.fetch(req, wrapEnv(env), ctx); } };', ''].join(NL);
const ENTRY_RS = ['import "./prelude.mjs";', 'import W from "./index.js";', 'import { wrapEnv } from "./mocks.mjs";',
  'export default { fetch(req, env, ctx) { return new W(ctx, wrapEnv(env)).fetch(req); } };', ''].join(NL);

fs.mkdirSync(OUT, { recursive: true });
await esbuild.build({
  entryPoints: [await original()],
  bundle: true, format: "esm", platform: "browser", mainFields: ["browser", "module", "main"], conditions: ["workerd", "worker", "browser"],
  outfile: path.join(OUT, "commerce-ts.mjs"), logLevel: "error",
});
fs.copyFileSync(path.join(HERE, "prelude.mjs"), path.join(OUT, "prelude.mjs"));
fs.writeFileSync(path.join(OUT, "commerce-ts-entry.mjs"), ENTRY_TS);
fs.writeFileSync(path.join(OUT, "mocks.mjs"), mocksSource());

const RS = path.join(OUT, "commerce-rs");
fs.mkdirSync(RS, { recursive: true });
fs.copyFileSync(path.join(WORKERS, "commerce", "build", "index_bg.wasm"), path.join(RS, "index_bg.wasm"));
{
  const js = fs.readFileSync(path.join(WORKERS, "commerce", "build", "index.js"), "utf8");
  fs.writeFileSync(path.join(RS, "index.js"), js.replace(/import source (\w+) from"\.\/index_bg\.wasm"/, 'import $1 from"./index_bg.wasm"'));
}
fs.copyFileSync(path.join(HERE, "prelude.mjs"), path.join(RS, "prelude.mjs"));
fs.writeFileSync(path.join(RS, "entry.mjs"), ENTRY_RS);
fs.writeFileSync(path.join(RS, "mocks.mjs"), mocksSource());

const MIGRATION = fs.readFileSync(path.join(WORKERS, "migrations", "0001_create_comments_table.sql"), "utf8");
const D1 = [
  "commerce_logis_apac", "commerce_logis_apac1_items", "commerce_logis_apac1_event", "commerce_logis_apac1_sales",
  "commerce_logis_apac1_talks", "commerce_logis_apac1_tracking", "commerce_logis_center", "DB",
];
const TABLES = ["tasks", "users", "pages", "items", "sales", "event", "tracking", "talks", "views", "crons", "zones", "usage"];

// 원본이 읽을 수 있는 gzip (proxy.mjs 와 같은 까닭: pako 가 D1 의 숫자 배열에서 저장 블록을 못 읽음)
import pako from "pako";
const readable = (b) => { try { pako.ungzip(Array.from(b)); return true; } catch { return false; } };
const gz = (o) => {
  let obj = o;
  for (let pad = 0; pad < 8; pad++) {
    for (const level of [6, 9, 1]) { const b = gzipSync(Buffer.from(JSON.stringify(obj)), { level }); if (readable(b)) return b; }
    obj = { ...o, _pad: "abcabcabc".repeat(20 * (pad + 1)) };
  }
  throw new Error("no readable gzip for " + JSON.stringify(o).slice(0, 80));
};
const ungz = (b) => { try { return JSON.parse(gunzipSync(Buffer.from(b)).toString("utf8")); } catch { return { raw: Buffer.from(b).toString("base64") }; } };

// gzip 바이트는 두 구현(pako · miniz_oxide)이 다르게 압축한다: 응답 · 상태 안의 gzip 바이트(숫자 배열 · {"0":31,…} 꼴)는 풀어서 견준다
const isGzNums = (a) => a.length >= 18 && a[0] === 31 && a[1] === 139 && a.every((x) => Number.isInteger(x) && x >= 0 && x < 256);
function decodeGz(v) {
  if (Array.isArray(v)) return isGzNums(v) ? { gz: ungz(v) } : v.map(decodeGz);
  if (v && typeof v === "object") {
    const keys = Object.keys(v);
    if (keys.length >= 18 && keys.every((k, i) => k === String(i))) {
      const nums = keys.map((k) => v[k]);
      if (isGzNums(nums)) return { gz: ungz(nums), shape: "Uint8Array" };
    }
    return Object.fromEntries(keys.map((k) => [k, decodeGz(v[k])]));
  }
  return v;
}

// 가짜 S3 (aws4fetch 가 보낸 서명을 같은 시각으로 다시 계산해 맞는지 본다 — 무작위 토큰이 서명에 들어가 값 자체는 견줄 수 없다)
async function s3Mock(req, url, store, records) {
  const p = decodeURIComponent(url.pathname.slice(1));
  const meta = {};
  for (const [k, v] of req.headers) if (k.startsWith("x-amz-meta-")) meta[k.slice(11)] = v;
  const auth = req.headers.get("authorization") || "";
  let sig = "none";
  const m = auth.match(/SignedHeaders=([^,]+), Signature=([0-9a-f]+)/);
  if (m) {
    const headers = {};
    for (const h of m[1].split(";")) if (h !== "host") headers[h] = req.headers.get(h);
    const signer = new AwsV4Signer({
      url: req.url, method: req.method, headers, accessKeyId: AWS.aws_access_key_id, secretAccessKey: AWS.aws_secret_access_key,
      service: "s3", region: AWS.aws_region, datetime: req.headers.get("x-amz-date"),
    });
    const signed = await signer.sign();
    sig = signed.headers.get("authorization") === auth ? "ok" : "bad";
  }
  const rec = { kind: "s3", method: req.method, path: p, sig };
  if (Object.keys(meta).length) rec.meta = meta;
  records.push(rec);
  if (req.method === "HEAD" || req.method === "GET") {
    if (!store.has(p)) return new Response(null, { status: 404 });
    const h = new Headers();
    for (const [k, v] of Object.entries(store.get(p))) h.set(`x-amz-meta-${k}`, String(v));
    return new Response(req.method === "HEAD" ? null : "", { status: 200, headers: h });
  }
  if (req.method === "PUT") { store.set(p, meta); return new Response("", { status: 200 }); }
  if (req.method === "DELETE") { store.delete(p); return new Response(null, { status: 204 }); }
  return new Response("", { status: 405 });
}

const DEFAULT_CF = { country: "KR", city: "Seoul", continent: "AS", timezone: "Asia/Seoul", region: "Seoul", regionCode: "11", postalCode: "04524", latitude: "37.56", longitude: "126.97", asOrganization: "Test ISP" };

async function run(variant, scenario) {
  const records = [];
  const s3 = new Map(Object.entries(scenario.s3 || {}).map(([k, v]) => [k, { ...v }]));
  const outbound = async (req) => {
    const url = new URL(req.url);
    if (url.hostname === "record") {
      records.push({ kind: url.pathname.slice(1), ...JSON.parse(await req.text()) });
      return new Response("ok");
    }
    if (/\.s3\.[^.]+\.amazonaws\.com$/.test(url.hostname)) return s3Mock(req, url, s3, records);
    const text = await req.text();
    let body; try { body = JSON.parse(text); } catch { body = text; }
    records.push({ kind: "fetch", url: req.url, method: req.method, body });
    return new Response("{}", { headers: { "content-type": "application/json" } });
  };
  const mf = new Miniflare({
    workers: [{
      name: "main", modules: true,
      scriptPath: variant === "ts" ? path.join(OUT, "commerce-ts-entry.mjs") : path.join(RS, "entry.mjs"),
      modulesRules: [{ type: "CompiledWasm", include: ["**/*.wasm"], fallthrough: true }, { type: "ESModule", include: ["**/*.js", "**/*.mjs"] }],
      compatibilityDate: "2025-12-01",
      bindings: { ...AWS },
      d1Databases: Object.fromEntries(D1.map((n) => [n, `${n}-${variant}-${scenario.name}`])),
      outboundService: outbound,
    }],
  });
  for (const n of D1) {
    const db = await mf.getD1Database(n, "main");
    for (const stmt of MIGRATION.split(";").map((s) => s.trim()).filter(Boolean)) await db.prepare(stmt).run();
  }
  for (const row of scenario.rows || []) {
    const db = await mf.getD1Database(row.db, "main");
    const cols = Object.keys(row.values);
    const vals = cols.map((c) => (c === "data" && row.values[c] !== null && typeof row.values[c] === "object" ? gz(row.values[c]) : row.values[c]));
    await db.prepare(`INSERT OR REPLACE INTO ${row.table} (${cols.map((c) => `"${c}"`).join(",")}) VALUES (${cols.map((_, i) => `?${i + 1}`).join(",")})`).bind(...vals).run();
  }
  const responses = [];
  const jar = {};
  for (const r of scenario.requests) {
    const headers = { ...(r.headers || {}) };
    const cookies = { ...(r.jar ? jar : {}), ...(r.cookies || {}) };
    if (Object.keys(cookies).length) headers.Cookie = Object.entries(cookies).map(([k, v]) => `${k}=${v}`).join("; ");
    let body;
    if (r.gzip !== undefined) { body = gzipSync(Buffer.from(typeof r.gzip === "string" ? r.gzip : JSON.stringify(r.gzip))); headers["Content-Encoding"] = "gzip"; }
    else if (r.body !== undefined) body = r.body;
    let out;
    try {
      const res = await mf.dispatchFetch(`https://commerce.logis.center${r.path || "/"}`, { method: r.method || "GET", headers, body, cf: { ...DEFAULT_CF, ...(r.cf || {}) } });
      const setCookies = res.headers.getSetCookie();
      for (const c of setCookies) { const [kv] = c.split(";"); const i = kv.indexOf("="); jar[kv.slice(0, i).trim()] = kv.slice(i + 1).trim(); }
      const text = await res.text();
      let parsed; try { parsed = decodeGz(JSON.parse(text)); } catch { parsed = text; }
      const hs = [...res.headers].filter(([k]) => k !== "set-cookie").sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
      out = { status: res.status, headers: Object.fromEntries(hs), setCookie: setCookies, body: parsed };
    } catch (e) {
      out = { error: String(e) };
    }
    responses.push(out);
  }
  const state = {};
  for (const n of D1) {
    const db = await mf.getD1Database(n, "main");
    for (const table of TABLES) {
      let results;
      try { ({ results } = await db.prepare(`SELECT * FROM ${table} ORDER BY id`).all()); } catch { continue; }
      if (results.length) state[`${n}.${table}`] = results.map((r) => ({ ...r, ...(r.data ? { data: ungz(r.data) } : {}) }));
    }
  }
  await mf.dispose();
  // S3 끝 상태는 넣은 순서 그대로 (키에 무작위 hash 가 들어가 이름순 정렬은 실행마다 바뀐다)
  return { responses, records, state, s3: [...s3].map(([key, meta]) => ({ key, meta })) };
}

// 무작위 값(새 지갑의 주소 · 개인 키 = hash · token 쿠키, hashId() 의 무작위 id)은 실행마다 다르다:
// 한쪽에만 있는 40/64자리 16진 값은 나온 순서대로 자리표로 바꾼다 (양쪽에 다 있는 값 — 결정적 해시 — 은 그대로)
const RANDOM = /(?:0x)?(?:[0-9a-fA-F]{64}|[0-9a-fA-F]{40})(?![0-9a-fA-F])/g;
function normalize(a, b) {
  const inA = new Set((a.match(RANDOM) || []).map((x) => x.toLowerCase().replace(/^0x/, "")));
  const inB = new Set((b.match(RANDOM) || []).map((x) => x.toLowerCase().replace(/^0x/, "")));
  const norm = (s, other) => {
    const seen = new Map();
    return s.replace(RANDOM, (x) => {
      const key = x.toLowerCase().replace(/^0x/, "");
      if (other.has(key)) return x;
      if (!seen.has(key)) seen.set(key, `<random#${seen.size}>`);
      return seen.get(key);
    });
  };
  return [norm(a, inB), norm(b, inA)];
}

let fail = 0;
for (const scenario of SCENARIOS.filter((s) => !ONLY || s.name === ONLY)) {
  const ts = await run("ts", scenario);
  const rs = await run("rs", scenario);
  const [a, b] = normalize(JSON.stringify(ts, null, 1), JSON.stringify(rs, null, 1));
  fs.writeFileSync(path.join(OUT, `commerce-${scenario.name}-ts.json`), a);
  fs.writeFileSync(path.join(OUT, `commerce-${scenario.name}-rs.json`), b);
  const same = a === b;
  if (!same) fail++;
  const st = (x) => x.responses.map((r) => r.status ?? "E").join(",");
  const kinds = (x) => x.records.map((r) => r.kind[0]).join("");
  console.log(`${same ? "SAME" : "DIFF"}  ${scenario.name}  status=${st(ts)}/${st(rs)}  records=${kinds(ts)}/${kinds(rs)}  tables=${Object.keys(ts.state).length}/${Object.keys(rs.state).length}`);
}
console.log(fail ? `${fail} scenario(s) differ` : "all scenarios identical");
process.exit(fail ? 1 : 0);
