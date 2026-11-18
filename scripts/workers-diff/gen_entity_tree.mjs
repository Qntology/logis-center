// entities 7.0.1 의 htmlDecodeTree 를 Rust 배열로 (workers/proxy/src/dom/entity_tree.rs)
import { htmlDecodeTree } from './node_modules/htmlparser2/node_modules/entities/dist/esm/generated/decode-data-html.js';
import { writeFileSync } from 'node:fs';
const out = process.argv[2];
const t = Array.from(htmlDecodeTree);
let s = '//! entities 7.0.1 `htmlDecodeTree` (generated/decode-data-html.js 의 base64 를 풀어 둔 u16 배열) — 손으로 고치지 말 것\n';
s += '//! 만든 곳: 하네스 `gen_entity_tree.mjs`\n\n';
s += `pub(super) static HTML_DECODE_TREE: [u16; ${t.length}] = [\n`;
for (let i = 0; i < t.length; i += 16) s += '    ' + t.slice(i, i + 16).join(', ') + ',\n';
s += '];\n';
writeFileSync(out, s);
console.log(t.length);
