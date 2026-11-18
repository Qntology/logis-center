// aws4fetch 1.0.20 의 sign() 결과(주소 · 방법 · 머리글 반복 순서 · 본문)와 ethers 6 의 isAddress · computeAddress 기준값.
// node aws_eth_vectors.mjs ../../workers/common/tests/aws_eth_vectors.json
// 값 표기: 없는 칸 = undefined, null = null. 본문은 { str } (문자열) · { b64 } (Uint8Array).
import { AwsClient } from "aws4fetch";
import { ethers } from "ethers";
import fs from "node:fs";

// new Date() 가 정해진 시각을 내게 (aws4fetch 는 new Date().toISOString() 으로 서명 시각을 잡는다)
const RealDate = Date;
let FIXED = 0;
globalThis.Date = class extends RealDate {
  constructor(...a) {
    if (a.length === 0) super(FIXED);
    else super(...a);
  }
  static now() {
    return FIXED;
  }
};

const KEY = "AKIAIOSFODNN7EXAMPLE";
const SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
const B = "logis-bucket";
const RG = "ap-northeast-2";
const S3 = `https://${B}.s3.${RG}.amazonaws.com`;
const T0 = RealDate.UTC(2026, 10, 19, 12, 34, 56, 789);

const base = { accessKeyId: KEY, secretAccessKey: SECRET, service: "s3", region: RG };
const meta = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [`x-amz-meta-${k}`, v]));
const boot = {
  "Content-Type": "text/plain",
  ...meta({ balance: "1000", host: "logis.center", token: "ab".repeat(32), email: "", ip: "203.0.113.7", phone: "", address: "" }),
};

// [이름, client, url, init, now]
const cases = [];
const add = (name, client, url, init, now = T0) => cases.push({ name, client, url, init, now });

add("commerce head hash", base, `${S3}/hash/0123456789abcdef0123456789abcdef01234567`, { method: "HEAD" });
add("commerce head address", base, `${S3}/address/0x56d67386939607c11bd60bb009eb02f4dd29c318`, { method: "HEAD" });
add("commerce put bootstrap", base, `${S3}/hash/aa`, { method: "PUT", headers: boot, body: "", aws: { service: "s3", region: RG } });
add("trading put bootstrap", base, `${S3}/hash/bb`, { method: "PUT", headers: { ...boot, "x-amz-meta-host": "trading.logis.center" }, body: "", aws: { service: "s3", region: RG } });
add("commerce delete", base, `${S3}/hash/cc`, { method: "DELETE" });
// env.aws_region 이 없을 때: 주소에 "undefined" 가 들어가고, 지역은 주소에서 짐작
const noRegion = { ...base, region: undefined };
add("undefined region head", noRegion, `https://${undefined}.s3.${undefined}.amazonaws.com/hash/x`, { method: "HEAD" });
add("undefined region put", noRegion, `https://${B}.s3.${undefined}.amazonaws.com/hash/x`, { method: "PUT", headers: boot, body: "", aws: { service: "s3", region: undefined } });
add("undefined region delete", noRegion, `https://${B}.s3.${undefined}.amazonaws.com/hash/x`, { method: "DELETE" });
add("empty region", { ...base, region: "" }, `${S3}/hash/x`, { method: "HEAD" });
add("undefined key", { ...base, accessKeyId: undefined }, `${S3}/hash/x`, { method: "HEAD" });
add("undefined secret", { ...base, secretAccessKey: undefined }, `${S3}/hash/x`, { method: "HEAD" });
add("null key", { ...base, accessKeyId: null }, `${S3}/hash/x`, { method: "HEAD" });
add("number key", { ...base, accessKeyId: 12345 }, `${S3}/hash/x`, { method: "HEAD" });
add("unicode secret", { ...base, secretAccessKey: "비밀/키+é" }, `${S3}/hash/x`, { method: "HEAD" });
add("undefined service s3 host", { ...base, service: undefined }, `${S3}/hash/x`, { method: "HEAD" });
add("undefined service r2", { ...base, service: undefined, region: undefined }, `https://acc.r2.cloudflarestorage.com/b/k`, { method: "GET" });
// 열쇠 · 경로 부호화
for (const [n, p] of [
  ["space", "/hash/a b  c"],
  ["unicode", "/hash/한글키/é"],
  ["plus", "/hash/a+b+c"],
  ["percent lone", "/hash/100%"],
  ["percent bad hex", "/hash/a%zzb"],
  ["encoded slash", "/hash/a%2Fb%2fc"],
  ["encoded letters", "/hash/%41%62%7e%2a"],
  ["rfc3986 marks", "/hash/a!b'c(d)e*f~g"],
  ["double slash", "/hash//a///b"],
  ["dot segments", "/hash/a/./b/../c"],
  ["emoji", "/hash/😀/x"],
  ["bad utf8", "/hash/%C3%28"],
  ["truncated utf8", "/hash/%E0%A4%A"],
  ["encoded plus", "/hash/a%2Bb"],
  ["encoded space", "/hash/a%20b"],
  ["reserved", "/hash/a:b@c$d&e=f,g;h"],
  ["empty path", ""],
  ["trailing slash", "/hash/"],
]) {
  add(`s3 path ${n}`, base, `${S3}${p}`, { method: "HEAD" });
}
add("non-s3 path double encode", { ...base, service: "execute-api" }, `https://abc.execute-api.us-east-1.amazonaws.com//prod//a b/%41/é+x`, { method: "GET" });
// 질의
for (const [n, q] of [
  ["sorted", "?b=2&a=1&a=0&c"],
  ["empty key", "?=x&a=&&b=1"],
  ["plus and space", "?q=a+b%20c&r=%2B"],
  ["unicode", "?이름=값&é=ü"],
  ["dup keys", "?a=2&a=1&b=1&a=3"],
  ["marks", "?k=!'()*~&x=a/b"],
  ["eq in value", "?a=b=c&d"],
  ["bad percent", "?a=%zz&b=%"],
  ["fragment", "?a=1#frag"],
]) {
  add(`s3 query ${n}`, base, `${S3}/hash/x${q}`, { method: "GET" });
}
add("non-s3 query dup keys", { ...base, service: "execute-api" }, `https://abc.execute-api.us-east-1.amazonaws.com/p?a=2&a=1`, { method: "GET" });
// 주소 꼴
add("port", base, `https://${B}.s3.${RG}.amazonaws.com:8443/hash/x`, { method: "HEAD" });
add("default port", base, `https://${B}.s3.${RG}.amazonaws.com:443/hash/x`, { method: "HEAD" });
add("upper host", base, `https://LOGIS-Bucket.S3.${RG}.AmazonAWS.com/hash/x`, { method: "HEAD" });
add("invalid url", base, `not a url`, { method: "HEAD" });
add("credentials url", base, `https://user:pw@${B}.s3.${RG}.amazonaws.com/hash/x`, { method: "HEAD" });
add("spaces around url", base, `  ${S3}/hash/x  `, { method: "HEAD" });
// 방법 · 본문
add("get no body", base, `${S3}/hash/x`, { method: "GET" });
add("get empty body", base, `${S3}/hash/x`, { method: "GET", body: "" });
add("head body", base, `${S3}/hash/x`, { method: "HEAD", body: "x" });
add("post string body", base, `${S3}/hash/x`, { method: "POST", body: '{"a":1}' });
add("post bytes body", base, `${S3}/hash/x`, { method: "POST", body: new Uint8Array([1, 2, 3, 250]) });
add("put empty no type", base, `${S3}/hash/x`, { method: "PUT", body: "" });
add("put empty bytes", base, `${S3}/hash/x`, { method: "PUT", body: new Uint8Array([]) });
add("no method body", base, `${S3}/hash/x`, { body: "x" });
add("no method no body", base, `${S3}/hash/x`, {});
add("no method empty body", base, `${S3}/hash/x`, { body: "" });
add("lower put", base, `${S3}/hash/x`, { method: "put", body: "" });
add("lower delete", base, `${S3}/hash/x`, { method: "delete" });
add("patch lower", base, `${S3}/hash/x`, { method: "patch", body: "p" });
add("custom method", base, `${S3}/hash/x`, { method: "Purge" });
add("trace method", base, `${S3}/hash/x`, { method: "TRACE" });
add("bad method", base, `${S3}/hash/x`, { method: "BAD METHOD" });
// 머리글
add("whitespace values", base, `${S3}/hash/x`, { method: "PUT", body: "", headers: { "Content-Type": "text/plain", "x-amz-meta-a": "  a   b  ", "x-amz-meta-b": "x\ty\t\tz", "X-Amz-Meta-C": "\t lead" } });
add("latin1 spaces", base, `${S3}/hash/x`, { method: "HEAD", headers: { "x-amz-meta-a": "a  b", "x-amz-meta-b": "a\u000bb\u000cc", "x-amz-meta-c": "é" } });
add("user content sha", base, `${S3}/hash/x`, { method: "PUT", body: "abc", headers: { "X-Amz-Content-Sha256": "deadbeef" } });
add("empty content sha", base, `${S3}/hash/x`, { method: "PUT", body: "abc", headers: { "X-Amz-Content-Sha256": "" } });
add("user date", base, `${S3}/hash/x`, { method: "HEAD", headers: { "X-Amz-Date": "19990101T000000Z" } });
add("host header", base, `${S3}/hash/x`, { method: "HEAD", headers: { Host: "evil.example", "x-amz-meta-z": "1" } });
add("unsignable", base, `${S3}/hash/x`, { method: "GET", headers: { Range: "bytes=0-9", "User-Agent": "ua", Expect: "100-continue", "X-Amzn-Trace-Id": "t", Authorization: "old", "Presigned-Expires": "1" } });
add("combined dup", base, `${S3}/hash/x`, { method: "HEAD", headers: { "X-Amz-Meta-A": "1", "x-amz-meta-a": "2", "x-AMZ-meta-A": "3" } });
add("set-cookie dup", base, `${S3}/hash/x`, { method: "HEAD", headers: { "Set-Cookie": "a=1", "set-cookie": "b=2" } });
add("session-like header", base, `${S3}/hash/x`, { method: "HEAD", headers: { "X-Amz-Security-Token": "tok" } });
add("non latin1 value", base, `${S3}/hash/x`, { method: "HEAD", headers: { "x-amz-meta-a": "한" } });
add("bad header name", base, `${S3}/hash/x`, { method: "HEAD", headers: { "bad name": "1" } });
add("newline value", base, `${S3}/hash/x`, { method: "HEAD", headers: { "x-amz-meta-a": "a\nb" } });
add("edge newline value", base, `${S3}/hash/x`, { method: "HEAD", headers: { "x-amz-meta-a": "\r\nab\n" } });
// 서비스 · 지역 짐작 (service · region 이 없을 때)
const guess = { accessKeyId: KEY, secretAccessKey: SECRET };
for (const [n, u, init] of [
  ["dynamodb", "https://dynamodb.us-west-2.amazonaws.com/", { method: "POST", body: '{"TableName":"t"}', headers: { "Content-Type": "application/x-amz-json-1.0", "X-Amz-Target": "DynamoDB_20120810.ListTables" } }],
  ["lambda url", "https://abcdef.lambda-url.eu-west-1.on.aws/path", { method: "GET" }],
  ["bad on.aws", "https://foo.on.aws/", { method: "GET" }],
  ["backblaze", "https://s3.us-west-004.backblazeb2.com/b/k", { method: "GET" }],
  ["s3 global", "https://bucket.s3.amazonaws.com/k", { method: "GET" }],
  ["s3 accelerate", "https://bucket.s3-accelerate.amazonaws.com/k", { method: "GET" }],
  ["s3 dualstack", "https://bucket.s3.dualstack.eu-central-1.amazonaws.com/k", { method: "GET" }],
  ["s3 legacy dash", "https://s3-us-west-2.amazonaws.com/b/k", { method: "GET" }],
  ["s3 external", "https://s3-external-1.amazonaws.com/b/k", { method: "GET" }],
  ["us-gov", "https://ec2.us-gov.amazonaws.com/", { method: "GET" }],
  ["iot", "https://iot.us-east-1.amazonaws.com/things", { method: "GET" }],
  ["iot data", "https://abc.iot.us-east-1.amazonaws.com/mqtt", { method: "GET" }],
  ["fips", "https://sqs-fips.us-east-1.amazonaws.com/", { method: "POST", body: "Action=ListQueues" }],
  ["swapped", "https://us-east-1.s3-2.amazonaws.com/", { method: "GET" }],
  ["queue", "https://queue.amazonaws.com/", { method: "GET" }],
  ["china", "https://ec2.cn-north-1.amazonaws.com.cn/", { method: "GET" }],
  ["autoscaling", "https://autoscaling.us-east-1.amazonaws.com/", { method: "POST", body: "{}", headers: { "X-Amz-Target": "AnyScaleFrontendService.Describe" } }],
  ["other host", "https://example.com/a", { method: "GET" }],
]) {
  add(`guess ${n}`, guess, u, init);
}
add("non-s3 bytes body", { ...base, service: "execute-api" }, "https://abc.execute-api.us-east-1.amazonaws.com/p", { method: "POST", body: new Uint8Array([0, 255, 7]) });
add("aws override region", base, `${S3}/hash/x`, { method: "HEAD", aws: { region: "eu-west-1" } });
add("aws override service", base, `${S3}/hash/x`, { method: "HEAD", aws: { service: "sts" } });
// 시각
for (const [n, t] of [
  ["epoch2000", RealDate.UTC(2000, 0, 1)],
  ["leap day", RealDate.UTC(2024, 1, 29, 23, 59, 59, 999)],
  ["year end", RealDate.UTC(2026, 11, 31, 23, 59, 59, 0)],
  ["fraction", RealDate.UTC(2026, 0, 2, 3, 4, 5, 6) + 0.75],
]) {
  add(`date ${n}`, base, `${S3}/hash/x`, { method: "HEAD" }, t);
}

const b64 = (u8) => Buffer.from(u8).toString("base64");
const enc = (init) => {
  const o = {};
  if ("method" in init) o.method = init.method;
  if (init.headers) o.headers = Object.entries(init.headers);
  if ("body" in init) o.body = typeof init.body === "string" ? { str: init.body } : { b64: b64(init.body) };
  if (init.aws) o.aws = init.aws;
  return o;
};
const encClient = (c) => {
  const o = {};
  for (const k of ["accessKeyId", "secretAccessKey", "service", "region"]) if (k in c && c[k] !== undefined) o[k] = c[k];
  return o;
};
const err = (e) => ({ name: e.name, message: e.message });

const awsOut = [];
for (const c of cases) {
  FIXED = c.now;
  const rec = { name: c.name, client: encClient(c.client), url: c.url, init: enc(c.init), now: c.now };
  let client;
  try {
    client = new AwsClient(c.client);
  } catch (e) {
    rec.ctor_throws = err(e);
  }
  if (client) {
    try {
      const req = await client.sign(c.url, { ...c.init, headers: c.init.headers ? { ...c.init.headers } : undefined });
      rec.out = { url: req.url, method: req.method, headers: [...req.headers], body: b64(new Uint8Array(await req.arrayBuffer())) };
    } catch (e) {
      rec.throws = err(e);
    }
  }
  awsOut.push(rec);
}

// ethers
const keys = [
  "0x0000000000000000000000000000000000000000000000000000000000000001",
  "0x0000000000000000000000000000000000000000000000000000000000000002",
  "0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140",
  "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318",
  ethers.keccak256(ethers.toUtf8Bytes("logis")),
  ethers.keccak256(ethers.toUtf8Bytes("commerce")),
  ethers.keccak256(ethers.toUtf8Bytes("trading")),
  ethers.hashMessage("hello"),
];
const computed = keys.map((k) => ({ key: k, address: ethers.computeAddress(k) }));

const flip = (s, i) => s.slice(0, i) + (s[i] === s[i].toUpperCase() ? s[i].toLowerCase() : s[i].toUpperCase()) + s.slice(i + 1);
const firstLetter = (s) => [...s].findIndex((ch, i) => i >= 2 && /[a-fA-F]/.test(ch));
const ibanLookup = (ch) => (/[0-9]/.test(ch) ? ch : String(ch.charCodeAt(0) - 55));
const icapSum = (body) => {
  const digits = (body.toUpperCase() + "XE00").split("").map(ibanLookup).join("");
  return String(98n - (BigInt(digits) % 97n)).padStart(2, "0");
};
const inputs = [];
const S = (v) => inputs.push(v);
for (const { address } of computed) {
  S(address);
  S(address.toLowerCase());
  S("0x" + address.slice(2).toUpperCase());
  S(address.slice(2));
  S(address.slice(2).toLowerCase());
  const i = firstLetter(address);
  if (i >= 0) S(flip(address, i));
  S(ethers.getIcapAddress(address));
}
for (const a of [
  "0x8ba1f109551bD432803012645Ac136ddd64DBA72",
  "0x8Ba1f109551bD432803012645Ac136ddd64DBA72",
  "0x8ba1f109551bd432803012645ac136ddd64dba72",
  "0X8ba1f109551bd432803012645ac136ddd64dba72",
  "0x8ba1f109551bd432803012645ac136ddd64dba7",
  "0x8ba1f109551bd432803012645ac136ddd64dba723",
  "0x8ba1f109551bd432803012645ac136ddd64dba7g",
  "0x8ba1f109551bd432803012645ac136ddd64dba72 ",
  " 0x8ba1f109551bd432803012645ac136ddd64dba72",
  "0x8ba1f109551bd432803012645ac136ddd64dba72\n",
  "0x0000000000000000000000000000000000000000",
  "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
  "0xffffffffffffffffffffffffffffffffffffffff",
  "0x",
  "",
  "0",
  "xx8ba1f109551bd432803012645ac136ddd64dba72",
  "0x0x8ba1f109551bd432803012645ac136ddd64dba",
  "0x８ba1f109551bd432803012645ac136ddd64dba72",
  "XE65GB6LDNXYOFTX0NSV3FUWKOWIXAMJK36",
  "XE65GB6LDNXYOFTX0NSV3FUWKOWIXAMJK37",
  "XE65gb6ldnxyoftx0nsv3fuwkowixamjk36",
  "xe65GB6LDNXYOFTX0NSV3FUWKOWIXAMJK36",
  "XE65GB6LDNXYOFTX0NSV3FUWKOWIXAMJK3",
  "XE65GB6LDNXYOFTX0NSV3FUWKOWIXAMJK36A",
  "XE6GB6LDNXYOFTX0NSV3FUWKOWIXAMJK36",
  "XE65GB6LDNXYOFTX0NSV3FUWKOWIXAMJK3!",
  "XE00",
  "undefined",
  "null",
]) {
  S(a);
}
// ICAP: 30 · 31 자, 160비트를 넘는 31자(ethers 는 받는다), 체크섬을 다시 맞춘 것
for (const body of ["000000000000000000000000000001", "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ", "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", "123456789ABCDEFGHIJKLMNOPQRSTUV", "0000000000000000000000000000000"]) {
  S("XE" + icapSum(body) + body);
  S("XE" + String((Number(icapSum(body)) + 1) % 100).padStart(2, "0") + body);
}
// 글이 아닌 값 (JSON 으로 적을 수 있는 것 + undefined)
const nonStrings = [null, 0, 1, true, false, ["0x8ba1f109551bd432803012645ac136ddd64dba72"], { address: "0x8ba1f109551bd432803012645ac136ddd64dba72" }, 1.5e48];
const addrOut = [];
for (const v of inputs) addrOut.push({ v, ok: ethers.isAddress(v) });
for (const v of nonStrings) addrOut.push({ v, ok: ethers.isAddress(v) });
addrOut.push({ undef: true, ok: ethers.isAddress(undefined) });
const checksum = inputs.filter((v) => ethers.isAddress(v)).map((v) => ({ v, address: ethers.getAddress(v) }));

const out = { aws: awsOut, is_address: addrOut, compute_address: computed, checksum };
fs.writeFileSync(process.argv[2], JSON.stringify(out, null, 1) + "\n");
console.log(
  awsOut.length, "aws cases (", awsOut.filter((r) => r.out).length, "signed,", awsOut.filter((r) => r.throws || r.ctor_throws).length, "throw ),",
  addrOut.length, "isAddress (", addrOut.filter((r) => r.ok).length, "true ),",
  computed.length, "computeAddress,", checksum.length, "getAddress",
);
