//! ethers v6 의 쓰는 부분: `ethers.isAddress(v)` · `ethers.Wallet.createRandom()` (주소 · 개인 키).
//! (옮기는 중: 몸통은 공용 층 담당이 채운다)

use crate::jsv::V;

/// `ethers.isAddress(v)` — 0x + 40 hex(대소문자가 섞이면 EIP-55 체크섬이 맞아야), ICAP(XE..) 도 받는다. 글이 아니면 거짓
pub fn is_address(v: &V) -> bool {
    let _ = v;
    false
}

/// `ethers.Wallet.createRandom()` 의 (`address`(체크섬 대소문자), `privateKey`(0x + 소문자 hex 64))
pub fn random_wallet() -> (String, String) {
    (String::new(), String::new())
}
