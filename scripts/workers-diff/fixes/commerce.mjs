// commerce 원본(fd1e991)에 덧대는 수정 — Rust 판(공용 층)이 이미 고친 것과 1:1 로 맞추기 위해. id 는 docs/workers-rs.md 의 "고친 버그" 표와 같다.
export default [
  // D1 이 BLOB 을 숫자 배열로 주면 pako 가 저장 블록을 못 읽는다 (apac · proxy 와 같은 수정)
  {
    id: "fix-pako-array", worker: "commerce",
    find: "import { gzip, ungzip } from 'pako'",
    replace: "import { gzip, ungzip as pakoUngzip } from 'pako'\nconst ungzip = (d) => pakoUngzip(d instanceof ArrayBuffer || ArrayBuffer.isView(d) ? d : new Uint8Array(d))",
  },
];
