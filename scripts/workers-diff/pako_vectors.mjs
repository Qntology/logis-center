// pako.ungzip(숫자 배열) 이 던지는지: 여러 글 · 여러 압축기(pako · node zlib 여러 수준)
import pako from "pako";
import { gzipSync } from "node:zlib";
import fs from "node:fs";
const texts = [];
const words = ["부산광역시 해운대구 1", "서울특별시 강남구 테헤란로 1", "{}", "{\"a\":1}", "x", "", "hello world hello world hello world"];
for (const w of words) texts.push(JSON.stringify({ sender: w }));
for (let n = 1; n < 40; n++) texts.push(JSON.stringify({ base: { pages: {} }, address: Array.from({ length: n }, (_, i) => "주소" + i + "번지" + "가나다".repeat(i % 5)) }));
for (let n = 0; n < 30; n++) { let s = ""; let h = n * 2654435761 >>> 0; for (let i = 0; i < 20 + n * 37; i++) { h = Math.imul(h ^ i, 16777619) >>> 0; s += String.fromCharCode(32 + (h % 95)); } texts.push(s); }
for (let n = 0; n < 10; n++) texts.push("한글".repeat(n * 50) + "abc".repeat(n * 300));
const cases = [];
for (const t of texts) {
  for (const [name, gz] of [["pako", pako.gzip(t)], ["zlib6", gzipSync(Buffer.from(t))], ["zlib1", gzipSync(Buffer.from(t), { level: 1 })], ["zlib0", gzipSync(Buffer.from(t), { level: 0 })], ["zlib9", gzipSync(Buffer.from(t), { level: 9 })]]) {
    const arr = Array.from(gz);
    let throws = false;
    try { pako.ungzip(arr); } catch (e) { throws = /subarray/.test(e.message); }
    cases.push({ gz: Buffer.from(gz).toString("base64"), throws, by: name });
  }
}
fs.writeFileSync(process.argv[2], JSON.stringify(cases));
console.log(cases.length, "cases,", cases.filter((c) => c.throws).length, "throw");
