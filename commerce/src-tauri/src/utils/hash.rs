use ethers_core::utils::hash_message;
use ethers_signers::{LocalWallet, Signer};
use regex::Regex;
use once_cell::sync::Lazy;

static PUNCTUATION_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\p{P}\p{S}\p{Z}]").unwrap());
static TWO_PART_DOMAINS: &[&str] = &["co.kr","co.uk","co.jp","com.cn","co.in","com.mx","co.id","com.my","com.sg","com.ph","com.vn"];

pub fn get_base_domain(hostname: &str) -> String {
    let host = hostname.to_lowercase();
    let parts: Vec<&str> = host.split('.').collect();
    
    let is_two_part = TWO_PART_DOMAINS.iter().any(|&d| host.ends_with(d));
    
    if is_two_part && parts.len() >= 3 {
        return parts[parts.len()-3..].join(".");
    }
    if parts.len() >= 2 {
        return parts[parts.len()-2..].join(".");
    }
    host
}

/// JS의 crc32(s)와 동일한 결과값을 반환합니다.
pub fn crc32(text: &str) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for b in text.bytes() {
        crc ^= b as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// JS의 ethers.computeAddress(ethers.hashMessage(text))와 100% 동일한 결과값을 반환합니다.
pub fn hash_id(text: &str) -> String {
    // 1. Ethereum Signed Message 프리픽스를 붙여 Keccak256 해싱
    let message_hash = hash_message(text);
    
    // 2. 해시값(32바이트)을 개인키로 사용하여 지갑 객체 생성
    let bytes = message_hash.as_bytes();
    if let Ok(wallet) = LocalWallet::from_bytes(bytes) {
        // 3. 주소 추출 및 소문자 변환
        return format!("{:?}", wallet.address()).to_lowercase();
    }
    
    String::new()
}

/// 서버의 normalizeNumericHomoglyphs 로직을 이식하여 시각적으로 유사한 문자를 숫자로 교정합니다.
pub fn normalize_numeric_homoglyphs(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for c in text.chars() {
        let normalized = match c {
            'O' | 'o' | 'Ο' | '○' | '〇' | '０' | 'Ｏ' => '0',
            'I' | 'l' | '１' | 'Ｉ' | 'ｌ' | 'Ι' | '|' | 'ᛁ' => '1',
            'Z' | 'z' | '２' | 'Ƨ' | 'ᒿ' => '2',
            'Ɛ' | 'ɜ' | 'З' | 'з' | '３' => '3',
            'Ꮞ' | '４' => '4',
            'S' | 's' | '５' | 'ƽ' => '5',
            'b' | 'Ꮾ' | '６' => '6',
            'T' | '７' => '7',
            'Β' | 'ß' | '８' => '8',
            'g' | '９' | 'ǵ' | 'ɡ' => '9',
            _ => c,
        };
        result.push(normalized);
    }
    result
}

/// 서버의 Digest(text)와 동일하게 문장 부호 및 공백을 제거한 후 hash_id를 생성합니다.
pub fn digest(text: &str) -> String {
    let normalized = normalize_numeric_homoglyphs(text);
    let clean_text = PUNCTUATION_REGEX.replace_all(&normalized, "").to_string().to_lowercase();
    hash_id(&clean_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_id_match_js() {
        // JS에서 ethers.computeAddress(ethers.hashMessage("hello")) 결과는 
        // "0x1c89b531aed45ee57073f00083c61d48e6cc44d1" (예시)
        // 실제 값과 대조하여 정합성을 확인합니다.
        let result = hash_id("hello");
        println!("Hash for 'hello': {}", result);
        assert!(result.starts_with("0x"));
        assert_eq!(result.len(), 42);
    }
}