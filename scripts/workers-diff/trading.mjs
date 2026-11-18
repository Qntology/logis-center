// trading 차분 시험: 원본 TS(trading-logis-center/src/index.ts) 와 Rust(wasm) 를 같은 D1 · 같은 가상 시계 · 같은 난수 ·
// 같은 가짜 S3 · commerce 로 돌려 응답(상태 · 머리글 · 모든 Set-Cookie · 본문) · 바깥 요청(주소 · 방법 · 모든 머리글 · 본문) ·
// D1 끝 상태를 견준다.
// usage: node trading.mjs [시나리오 이름]   (ONLY=<이름> 도 됨)
//   TRADING_SRC=<trading-logis-center 경로>  원본 위치 (기본: C:\Users\lp\Documents\GitHub\trading-logis-center)
//   MASK_SIG=1   (선택) 바깥 요청의 authorization · x-amz-date 값을 양쪽 모두 비운다. 기본은 서명 그대로 견주고
//                가짜 S3 가 AwsV4Signer 로 다시 계산해 sig: ok/bad 를 남긴다
//   WORKERS=<logis-center/workers 경로>  Rust 빌드(trading/build) 위치
import { Miniflare } from "miniflare";
import * as esbuild from "esbuild";
import { gzipSync, gunzipSync } from "node:zlib";
import fs from "node:fs";
import path from "node:path";
// 운영 Workers 의 시간대는 UTC 다
process.env.TZ = "UTC";
import { AwsV4Signer } from "aws4fetch";
import { SCENARIOS } from "./trading_scenarios.mjs";

const HERE = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const REPO = path.resolve(HERE, "..", "..");
const WORKERS = process.env.WORKERS || path.join(REPO, "workers");
const SRC = process.env.TRADING_SRC || (fs.existsSync("C:\\Users\\lp\\Documents\\GitHub\\trading-logis-center")
  ? "C:\\Users\\lp\\Documents\\GitHub\\trading-logis-center" : path.resolve(REPO, "..", "trading-logis-center"));
const ONLY = process.env.ONLY || process.argv[2];
const MASK_SIG = process.env.MASK_SIG === "1";
const OUT = path.join(HERE, "out");
fs.mkdirSync(path.join(OUT, "ts"), { recursive: true });

// 원본 TS 를 복사해 묶는다 (pako · ethers · aws4fetch 는 이 폴더의 node_modules)
const TS_COPY = path.join(OUT, "ts", "trading.ts");
fs.copyFileSync(path.join(SRC, "src", "index.ts"), TS_COPY);
await esbuild.build({
  entryPoints: [TS_COPY],
  bundle: true, format: "esm", platform: "browser", mainFields: ["browser", "module", "main"], conditions: ["workerd", "worker", "browser"],
  outfile: path.join(OUT, "trading-ts.mjs"), logLevel: "error",
});
// prelude: 인자 없는 new Date() 도 가상 시계를 따르게 (aws4fetch 의 x-amz-date 가 두 실행에서 같게 — 이미 그렇다면 그대로)
const PRELUDE = fs.readFileSync(path.join(HERE, "prelude.mjs"), "utf8").replace("if (args.length === 0) super();", "if (args.length === 0) super(t);");
if (!PRELUDE.includes("super(t)")) throw new Error("prelude.mjs: cannot pin new Date() to the virtual clock");
fs.writeFileSync(path.join(OUT, "prelude.mjs"), PRELUDE);

// 요청에 x-harness-no-cf 가 있으면 cf 없는 Request 로 바꿔 넘긴다 (Miniflare 는 늘 cf 를 채운다)
const STRIP = `
export function strip(req) {
  if (!req.headers.has("x-harness-no-cf")) return req;
  const h = new Headers(req.headers); h.delete("x-harness-no-cf");
  const body = req.method === "GET" || req.method === "HEAD" ? undefined : req.body;
  return new Request(req.url, { method: req.method, headers: h, body });
}`;
const NL = String.fromCharCode(10);
fs.writeFileSync(path.join(OUT, "strip.mjs"), STRIP);
fs.writeFileSync(path.join(OUT, "trading-ts-entry.mjs"), ['import "./prelude.mjs";', 'import W from "./trading-ts.mjs";', 'import { strip } from "./strip.mjs";',
  'export default { fetch(req, env, ctx) { return W.fetch(strip(req), env, ctx); } };', ""].join(NL));

const RS = path.join(OUT, "trading-rs");
fs.mkdirSync(RS, { recursive: true });
fs.copyFileSync(path.join(WORKERS, "trading", "build", "index_bg.wasm"), path.join(RS, "index_bg.wasm"));
{
  // Miniflare 는 `import source X from "*.wasm"` 를 못 읽는다 (apac.mjs 참고)
  const js = fs.readFileSync(path.join(WORKERS, "trading", "build", "index.js"), "utf8");
  fs.writeFileSync(path.join(RS, "index.js"), js.replace(/import source (\w+) from"\.\/index_bg\.wasm"/, 'import $1 from"./index_bg.wasm"'));
}
fs.writeFileSync(path.join(RS, "prelude.mjs"), PRELUDE);
fs.writeFileSync(path.join(RS, "strip.mjs"), STRIP);
fs.writeFileSync(path.join(RS, "entry.mjs"), ['import "./prelude.mjs";', 'import W from "./index.js";', 'import { strip } from "./strip.mjs";',
  'export default { fetch(req, env, ctx) { return new W(ctx, env).fetch(strip(req)); } };', ""].join(NL));

// 마이그레이션: 주석 줄을 빼고 ; 로 나눈다
const MIGRATION = fs.readFileSync(path.join(WORKERS, "trading", "migrations", "0001_create_items_table.sql"), "utf8")
  .split(/\r?\n/).filter((l) => !l.trim().startsWith("--")).join("\n")
  .split(";").map((s) => s.trim()).filter(Boolean);

const BUCKET = "logis-bucket", REGION = "ap-northeast-2";
const S3_HOST = `${BUCKET}.s3.${REGION}.amazonaws.com`;
const ENV = { aws_access_key_id: "AKIDEXAMPLE", aws_secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", aws_region: REGION, aws_bucket: BUCKET };

// 0x 없는 40 · 64 자리 hex 와 0x 주소 (원본은 ethers.Wallet.createRandom 이 crypto 로 만든다 — 실행마다 다름)
const RANDOM = /(?<![0-9a-fA-F])(?:0x)?(?:[0-9a-f]{64}|[0-9a-f]{40})(?![0-9a-fA-F])/g;
// 응답 JSON 의 data(base64 gzip) 는 풀어서 견준다 (gzip 바이트는 pako ↔ miniz_oxide 가 다르다)
const unBlob = (b64) => { try { return "gz:" + gunzipSync(Buffer.from(b64, "base64")).toString("utf8"); } catch { return b64; } };
const ungz = (b) => { try { return gunzipSync(Buffer.from(b)).toString("utf8"); } catch { return { raw: Buffer.from(b).toString("base64") }; } };

function encodeBody(r) {
  if (r.body === undefined) return undefined;
  let bytes = typeof r.body === "string" ? Buffer.from(r.body) : Buffer.isBuffer(r.body) ? r.body : Buffer.from(JSON.stringify(r.body));
  if (r.gzip === true) bytes = gzipSync(bytes);
  if (typeof r.gzip === "function") bytes = r.gzip(bytes);
  return bytes;
}

async function run(variant, scenario) {
  const outbound = [];
  const isTs = variant === "ts";
  const handler = async (req) => {
    const url = new URL(req.url);
    const body = req.method === "GET" || req.method === "HEAD" ? "" : Buffer.from(await req.arrayBuffer()).toString("utf8");
    const headers = [...req.headers.entries()].map(([k, v]) => [k, MASK_SIG && (k === "authorization" || k === "x-amz-date") ? "" : v]);
    // SigV4 를 aws4fetch 의 AwsV4Signer 로 같은 x-amz-date 에서 다시 계산해 맞는지 (commerce.mjs 와 같은 방식).
    // 무작위 토큰(bootstrap PUT 의 x-amz-meta-token)이 서명에 들어가 그 요청의 서명 값은 실행마다 다르다 — 값은 자리표로, 맞는지는 sig 로 견준다
    let sig = "none";
    const m = (req.headers.get("authorization") || "").match(/SignedHeaders=([^,]+), Signature=([0-9a-f]+)/);
    if (m) {
      const signed = {};
      for (const h of m[1].split(";")) if (h !== "host") signed[h] = req.headers.get(h);
      const again = await new AwsV4Signer({
        url: req.url, method: req.method, headers: signed, accessKeyId: ENV.aws_access_key_id, secretAccessKey: ENV.aws_secret_access_key,
        service: "s3", region: ENV.aws_region, datetime: req.headers.get("x-amz-date"),
      }).sign();
      sig = again.headers.get("authorization") === req.headers.get("authorization") ? "ok" : "bad";
    }
    outbound.push({ url: req.url, method: req.method, headers, body, sig });
    if (url.host === S3_HOST) {
      const hash = decodeURIComponent(url.pathname.replace(/^\/hash\//, ""));
      if (req.method === "PUT") return new Response("", { status: 200, headers: { etag: '"e"' } });
      const entry = (scenario.s3 || {})[hash];
      if (!entry) return new Response(null, { status: 404 });
      return new Response(null, { status: entry.status ?? 200, headers: entry.headers || {} });
    }
    if (url.host === "commerce.logis.center") {
      const reply = scenario.commerce ? scenario.commerce(url) : { body: { session: {} } };
      const text = typeof reply.body === "string" ? reply.body : JSON.stringify(reply.body);
      return new Response(text, { status: reply.status ?? 200, headers: { "content-type": "application/json" } });
    }
    return new Response("unexpected outbound", { status: 599 });
  };
  const mf = new Miniflare({
    workers: [{
      name: "main",
      modules: true,
      scriptPath: isTs ? path.join(OUT, "trading-ts-entry.mjs") : path.join(RS, "entry.mjs"),
      modulesRules: [{ type: "CompiledWasm", include: ["**/*.wasm"], fallthrough: true }, { type: "ESModule", include: ["**/*.js", "**/*.mjs"] }],
      compatibilityDate: "2025-12-01",
      d1Databases: { DB: `trading-${variant}-${scenario.name}` },
      bindings: { ...ENV, ...(scenario.env || {}) },
      outboundService: handler,
    }],
  });
  const db = await mf.getD1Database("DB", "main");
  for (const stmt of MIGRATION) await db.prepare(stmt).run();
  for (const row of scenario.rows || []) {
    const cols = Object.keys(row);
    const vals = cols.map((c) => (c === "data" && row[c] !== null && typeof row[c] === "object" && !Buffer.isBuffer(row[c]) ? gzipSync(Buffer.from(JSON.stringify(row[c]))) : row[c]));
    await db.prepare(`INSERT INTO items (${cols.map((c) => `"${c}"`).join(",")}) VALUES (${cols.map((_, i) => `?${i + 1}`).join(",")})`).bind(...vals).run();
  }
  // 시나리오가 주는 추가 SQL (예: 배치 실패를 일으킬 트리거)
  for (const sql of scenario.setup || []) await db.prepare(sql).run();
  const responses = [];
  for (const r of scenario.requests) {
    const headers = { ...(r.headers || {}) };
    if (r.cookie) headers.cookie = r.cookie;
    if (r.noCf) headers["x-harness-no-cf"] = "1";
    try {
      const res = await mf.dispatchFetch(`https://trading.logis.center/${r.query ? "?" + r.query : ""}`, {
        method: r.method || "GET", headers, body: encodeBody(r), cf: r.cf === undefined ? { country: "KR" } : r.cf,
      });
      const text = await res.text();
      let body = text;
      try {
        const j = JSON.parse(text);
        if (j && Array.isArray(j.results)) for (const x of j.results) if (x && typeof x.data === "string" && x.data) x.data = unBlob(x.data);
        body = j;
      } catch {}
      // 처리기 밖으로 던진 예외(500)는 첫 줄만 (Miniflare 의 오류 쪽에는 스택이 붙는다)
      if (res.status === 500 && typeof body === "string") body = body.split("\n")[0];
      responses.push({ status: res.status, headers: [...res.headers.entries()].filter(([k]) => k !== "set-cookie"), setCookie: res.headers.getSetCookie(), body });
    } catch (e) {
      responses.push({ error: String(e).split("\n")[0] });
    }
  }
  const { results } = await db.prepare(`SELECT * FROM items ORDER BY id`).all();
  const state = results.map((r) => ({ ...r, data: r.data && typeof r.data === "object" ? ungz(r.data) : r.data }));
  await mf.dispose();
  return { responses, outbound, state };
}

let fail = 0;
const list = SCENARIOS.filter((s) => !ONLY || s.name === ONLY);
if (!list.length) throw new Error(`no scenario named ${ONLY}`);
for (const scenario of list) {
  const ts = await run("ts", scenario);
  const rs = await run("rs", scenario);
  const ra = JSON.stringify(ts, null, 1), rb = JSON.stringify(rs, null, 1);
  const inA = new Set(ra.match(RANDOM) || []), inB = new Set(rb.match(RANDOM) || []);
  const norm = (s, other) => { const seen = new Map(); return s.replace(RANDOM, (x) => (other.has(x) ? x : (seen.has(x) ? seen.get(x) : (seen.set(x, "<random#" + seen.size + ">"), seen.get(x))))); };
  const a = norm(ra, inB), b = norm(rb, inA);
  fs.writeFileSync(path.join(OUT, `trading-${scenario.name}-ts.json`), a);
  fs.writeFileSync(path.join(OUT, `trading-${scenario.name}-rs.json`), b);
  const same = a === b;
  if (!same) fail++;
  const st = (r) => r.responses.map((x) => x.status ?? "E").join(",");
  const sigs = (r) => r.outbound.filter((o) => o.sig !== "none").map((o) => (o.sig === "ok" ? "✓" : "✗")).join("");
  console.log(`${same ? "SAME" : "DIFF"}  ${scenario.name.padEnd(28)} status=${st(ts)}/${st(rs)}  outbound=${ts.outbound.length}/${rs.outbound.length}  sig=${sigs(ts) || "-"}/${sigs(rs) || "-"}  rows=${ts.state.length}/${rs.state.length}`);
}
console.log(fail ? `${fail} scenario(s) differ` : `all ${list.length} scenarios identical${MASK_SIG ? " (MASK_SIG=1: authorization · x-amz-date blanked)" : ""}`);
process.exit(fail ? 1 : 0);
