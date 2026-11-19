// commerce 원본(fd1e991)에 덧대는 수정 — Rust 판(workers/commerce)이 고친 것과 1:1 로 맞추기 위해. id 는 docs/workers-rs.md 의 "고친 버그" 표와 같다.
// 긴 수정(JS 코드 블록)은 백틱 · ${} 를 거르지 않고 읽게 commerce_patches.txt 에 두고 여기서 { id, worker, find, replace, count } 로 바꾼다.
import fs from "node:fs";

const PATCHES = new URL("./commerce_patches.txt", import.meta.url);

/** commerce_patches.txt: "@@ id" · "@@ count n" · "@@ regex" 줄 뒤 "<<< find" … "=== replace" … ">>>" */
function readPatches() {
  const lines = fs.readFileSync(PATCHES, "utf8").replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let head = null;
  let mode = null;
  let find = [];
  let replace = [];
  for (const line of lines) {
    if (mode === null) {
      const m = line.match(/^@@ (\S+)(?: (\S+))?$/);
      if (m) {
        if (m[1] === "count") head.count = Number(m[2]);
        else if (m[1] === "regex") head.regex = true;
        else head = { id: m[1] };
      } else if (line === "<<< find") {
        if (!head) throw new Error("commerce_patches.txt: find block without @@ id");
        mode = "find";
      }
    } else if (mode === "find") {
      if (line === "=== replace") mode = "replace";
      else find.push(line);
    } else if (mode === "replace") {
      if (line === ">>>") {
        const text = find.join("\n");
        out.push({
          id: head.id, worker: "commerce",
          find: head.regex ? new RegExp(text.trim(), "g") : text,
          replace: replace.join("\n"),
          ...(head.count ? { count: head.count } : {}),
        });
        head = null; mode = null; find = []; replace = [];
      } else replace.push(line);
    }
  }
  if (mode !== null) throw new Error("commerce_patches.txt: unterminated block");
  return out;
}

// COMMERCE_SKIP=fix-a,fix-b 면 그 수정을 빼고 덧댄다 — 수정 하나가 어느 시나리오를 바꾸는지 보려고 (Rust 와 DIFF 가 나야 한다)
const SKIP = new Set((process.env.COMMERCE_SKIP || "").split(",").filter(Boolean));

export default [
  // D1 이 BLOB 을 숫자 배열로 주면 pako 가 저장 블록을 못 읽는다 (apac · proxy 와 같은 수정)
  {
    id: "fix-pako-array", worker: "commerce",
    find: "import { gzip, ungzip } from 'pako'",
    replace: "import { gzip, ungzip as pakoUngzip } from 'pako'\nconst ungzip = (d) => pakoUngzip(d instanceof ArrayBuffer || ArrayBuffer.isView(d) ? d : new Uint8Array(d))",
  },
  ...readPatches(),
].filter((fix) => !SKIP.has(fix.id));
