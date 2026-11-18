// 원본 TS(801b0ce0)에 버그 수정을 덧대 "고친 원본"을 만든다. Rust 판의 같은 수정과 1:1 로 견주기 위해서다.
// 각 수정은 { id, worker: "apac" | "proxy" | "both", find, replace } — find 의 공백(탭 · 줄바꿈 수)은 무시하고 맞추며,
// 정확히 한 번(count 를 주면 그 수만큼) 맞아야 한다(worker 가 both 면 각 파일에서 한 번). 설명은 docs/workers-rs.md 의 "고친 버그" 표와 같은 id 를 쓴다.
// D1 의 undefined 바인딩(fix-bind-undefined)은 글 패치가 아니라 진입 모듈의 env 래퍼가 맡는다(mocks: wrapD1).
import fs from "node:fs";
import path from "node:path";

const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const loose = (s) => new RegExp(escape(s.trim()).replace(/\s+/g, "\\s+"), "g");

export const FIXES = [
  {
    id: "fix-pako-array", worker: "both",
    find: "import { gzip, ungzip } from 'pako'",
    replace: "import { gzip, ungzip as pakoUngzip } from 'pako'\nconst ungzip = (d) => pakoUngzip(d instanceof ArrayBuffer || ArrayBuffer.isView(d) ? d : new Uint8Array(d))",
  },
  {
    id: "fix-isdiff-buffer", worker: "both",
    find: "if (key === 'data' && Buffer.isBuffer(obj1[key]) && Buffer.isBuffer(obj2[key])) {",
    replace: "if (false) {",
  },
  { id: "fix-clean-number", worker: "proxy", find: 'str = str.replace(/./gi,"")', replace: 'str = str.replace(/\\./gi,"")' },
  {
    id: "fix-apac-lock-first", worker: "apac",
    find: `var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(cron.data))

				var task = JSON.parse(decompressedJsonString)

				if(cron.id == 'lock'){
					tasks = []

					break;
				}`,
    replace: `if(cron.id == 'lock'){
					tasks = []

					break;
				}

				var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(cron.data))

				var task = JSON.parse(decompressedJsonString)`,
  },
  { id: "fix-apac-delete-quote", worker: "apac", find: 'DELETE FROM tasks WHERE id = "${task.id}"', replace: "DELETE FROM tasks WHERE id = '${task.id}'" },
  {
    id: "fix-apac-gemini-quota", worker: "apac",
    find: `models[\`\${env.gemini1}-gemini-2.5-flash-lite\`] = 4000
		models[\`\${env.gemini2}-gemini-2.5-flash-lite\`] = 4000
		models[\`\${env.gemini1}-gemini-2.0-flash-lite\`] = 4000
		models[\`\${env.gemini2}-gemini-2.0-flash-lite\`] = 4000`,
    replace: `models[\`\${env.gemini1}-gemini-flash-lite-latest\`] = 4000
		models[\`\${env.gemini2}-gemini-flash-lite-latest\`] = 4000`,
  },
];

/** fixes/ 폴더의 묶음(proxy 구역별)도 읽는다: export default [ ...같은 꼴 ] */
async function moreFixes() {
  const dir = path.join(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "fixes");
  if (!fs.existsSync(dir)) return [];
  const out = [];
  for (const f of fs.readdirSync(dir).filter((f) => f.endsWith(".mjs")).sort()) {
    const mod = await import(new URL(`./fixes/${f}`, import.meta.url));
    out.push(...mod.default);
  }
  return out;
}

/** 원본 글에 수정을 덧댄다. 맞는 수가 1 이 아니면 던진다 */
export async function patchOriginal(worker, source) {
  let s = source;
  const applied = [];
  for (const fix of [...FIXES, ...(await moreFixes())]) {
    // "both" 는 apac · proxy 둘. commerce · trading 은 이름으로만 (배열이면 그 안의 워커들)
    const targets = Array.isArray(fix.worker) ? fix.worker : fix.worker === "both" ? ["apac", "proxy"] : [fix.worker];
    if (!targets.includes(worker)) continue;
    const re = fix.find instanceof RegExp ? fix.find : loose(fix.find);
    const n = (s.match(re) || []).length;
    const want = fix.count ?? 1;
    if (n !== want) throw new Error(`fix ${fix.id}: ${n} matches in ${worker} (want ${want})`);
    s = s.replace(re, () => fix.replace);
    applied.push(fix.id);
  }
  return { source: s, applied };
}
