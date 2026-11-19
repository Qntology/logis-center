//! ethers v6 의 쓰는 부분: `ethers.isAddress(v)` · `ethers.getAddress` · `ethers.computeAddress` · `ethers.Wallet.createRandom()` (주소 · 개인 키).
//! keccak · secp256k1 은 premium-discount-identity 와 같은 크레이트(tiny-keccak · k256)다. 대조는 `tests/aws_eth.rs`.

use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::SecretKey;
use tiny_keccak::{Hasher, Keccak};

use crate::jsv::V;

/// keccak256
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut k = Keccak::v256();
    k.update(data);
    let mut out = [0u8; 32];
    k.finalize(&mut out);
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// ethers 의 getChecksumAddress: `0x` 뒤 글을 소문자로, 앞 40자의 keccak 으로 대소문자를 정한다 (EIP-55).
/// 40자를 넘으면(160비트를 넘는 ICAP) 나머지는 소문자로 붙는다 — ethers 그대로
fn checksum(address: &str) -> String {
    let mut chars: Vec<char> = address[2..].to_lowercase().chars().collect();
    let expanded: Vec<u8> = chars.iter().take(40).map(|&c| c as u8).collect();
    let hashed = keccak256(&expanded);
    for i in (0..40).step_by(2) {
        if hashed[i >> 1] >> 4 >= 8 {
            chars[i] = chars[i].to_ascii_uppercase();
        }
        if hashed[i >> 1] & 0x0f >= 8 {
            chars[i + 1] = chars[i + 1].to_ascii_uppercase();
        }
    }
    format!("0x{}", chars.into_iter().collect::<String>())
}

/// ICAP 의 IBAN 체크섬 (두 자리)
fn iban_checksum(address: &str) -> String {
    let a = address.to_ascii_uppercase();
    let moved = format!("{}{}00", &a[4..], &a[..2]);
    let mut r: u32 = 0;
    for c in moved.chars() {
        let digits = if c.is_ascii_digit() { c.to_string() } else { (c as u32 - 55).to_string() };
        for d in digits.bytes() {
            r = (r * 10 + (d - b'0') as u32) % 97;
        }
    }
    format!("{:02}", 98 - r)
}

/// 36진수 → 16진수 (BigInt(…).toString(16), 앞의 0 없음)
fn base36_to_hex(s: &str) -> String {
    // 큰 수를 바이트 배열로 (큰 자리 먼저)
    let mut n: Vec<u8> = Vec::new();
    for c in s.to_ascii_lowercase().chars() {
        let d = c.to_digit(36).unwrap_or(0);
        let mut carry = d;
        for b in n.iter_mut().rev() {
            let x = *b as u32 * 36 + carry;
            *b = (x & 0xff) as u8;
            carry = x >> 8;
        }
        while carry > 0 {
            n.insert(0, (carry & 0xff) as u8);
            carry >>= 8;
        }
    }
    let h = hex(&n);
    let h = h.trim_start_matches('0');
    if h.is_empty() {
        "0".into()
    } else {
        h.into()
    }
}

/// `ethers.getAddress(s)` — 던지면 None
pub fn get_address(s: &str) -> Option<String> {
    // /^(0x)?[0-9a-fA-F]{40}$/
    let body = s.strip_prefix("0x").unwrap_or(s);
    if body.len() == 40 && body.bytes().all(|b| b.is_ascii_hexdigit()) {
        let address = format!("0x{body}");
        let result = checksum(&address);
        let mixed = body.bytes().any(|b| b.is_ascii_uppercase()) && body.bytes().any(|b| b.is_ascii_lowercase());
        // 대소문자가 섞였으면 체크섬이어야 한다
        return (!mixed || result == address).then_some(result);
    }
    // /^XE[0-9]{2}[0-9A-Za-z]{30,31}$/ (ICAP 직접 형식)
    let b = s.as_bytes();
    if s.starts_with("XE")
        && (34..=35).contains(&b.len())
        && b[2..4].iter().all(|c| c.is_ascii_digit())
        && b[4..].iter().all(|c| c.is_ascii_alphanumeric())
    {
        if s[2..4] != iban_checksum(s) {
            return None;
        }
        let h = base36_to_hex(&s[4..]);
        return Some(checksum(&format!("0x{h:0>40}")));
    }
    None
}

/// `ethers.isAddress(v)` — 0x + 40 hex(대소문자가 섞이면 EIP-55 체크섬이 맞아야), ICAP(XE..) 도 받는다. 글이 아니면 거짓
pub fn is_address(v: &V) -> bool {
    match v {
        V::Str(s) => get_address(s).is_some(),
        _ => false,
    }
}

/// `ethers.computeAddress(privateKey)` (체크섬 대소문자). 개인 키 범위 밖이면 None
pub fn address_from_private_key(key: &[u8]) -> Option<String> {
    let secret = SecretKey::from_slice(key).ok()?;
    let point = secret.public_key().to_encoded_point(false);
    let digest = keccak256(&point.as_bytes()[1..]);
    Some(checksum(&format!("0x{}", hex(&digest[12..]))))
}

/// `ethers.Wallet.createRandom()` 의 (`address`(체크섬 대소문자), `privateKey`(0x + 소문자 hex 64)).
/// 원본은 BIP-39 니모닉에서 유도하지만, 나오는 키는 같은 꼴의 무작위 32바이트다(crypto.getRandomValues)
pub fn random_wallet() -> (String, String) {
    loop {
        let mut key = [0u8; 32];
        getrandom::fill(&mut key).expect("crypto.getRandomValues");
        if let Some(address) = address_from_private_key(&key) {
            return (address, format!("0x{}", hex(&key)));
        }
    }
}
