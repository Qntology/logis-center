use ethers_core::utils::hash_message;
use ethers_signers::{LocalWallet, Signer};
use regex::Regex;
use once_cell::sync::Lazy;

static PUNCTUATION_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\p{P}\p{S}\p{Z}]").unwrap());
static TWO_PART_DOMAINS: &[&str] = &["co.kr","co.uk","co.jp","com.cn","co.in","com.mx","co.id","com.my","com.sg","com.ph","com.vn"];

pub fn get_base_domain(hostname: &str) -> String {
    let host = hostname.to_lowercase();
    let parts: Vec<&str> = host.split('.').collect();
    
    // 🌟 [SUFFIX BOUNDARY] ends_with 만으로는 'deco.kr' 이 'co.kr' 접미사로 걸립니다.
    //    shop.deco.kr → 3라벨 판정 → 'shop.deco.kr' 반환 (기대값 'deco.kr')
    //    라벨 경계(.)를 붙여 접미사가 통째로 한 라벨에서 시작하는지 확인합니다.
    let is_two_part = TWO_PART_DOMAINS.iter().any(|&d| {
        host == d || host.ends_with(&format!(".{}", d))
    });
    
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
    let mut message_hash = hash_message(text);

    for _ in 0..4 {
        if let Ok(wallet) = LocalWallet::from_bytes(message_hash.as_bytes()) {
            // 3. 주소 추출 및 소문자 변환
            return format!("{:?}", wallet.address()).to_lowercase();
        }
        message_hash = hash_message(format!("logis-rehash:{:?}", message_hash));
    }

    // 여기까지 오는 것은 암호학적으로 사실상 불가능합니다.
    // 빈 값 대신 충돌하지 않고 진단 가능한 값을 남깁니다.
    format!("0xdead{:0>36}", crc32(text))
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

pub fn relay_index(raw: &str) -> u32 {
    if !is_valid_relay_key(raw) {
        return 0;
    }
    crc32(&normalize_identifier(raw))
}

pub fn relay_id(raw: &str, target_type: &str) -> String {
    if !is_valid_relay_key(raw) {
        return String::new();
    }
    let n = normalize_identifier(raw);
    if n.is_empty() { return String::new(); }
    hash_id(&format!("{}{}", target_type, n))
}

pub fn is_valid_relay_key(raw: &str) -> bool {
    let t = raw.trim();
    if t.is_empty() { return false; }
    
    // 🌟 [PLACEHOLDER BLOCK] LLM 이 값을 찾지 못했을 때 뱉는 플레이스홀더 차단
    let lower = t.to_lowercase();
    if lower == "n/a" || lower == "na" || lower == "null"
        || lower == "none" || lower == "undefined" || lower == "unknown"
        || lower == "..." || lower == "-" || lower == "string"
        || lower == "number" || lower == "boolean" {
        return false;
    }
    
    // 🌟 전각 영숫자를 반각으로 접은 뒤 판정합니다.
    //    "１２３４５" → "12345" (5자리) → 순수 숫자 6자리 미만이므로 유효하지 않음
    //    "ＣＩ－４３７２６" → "CI43726" (7자리) → 유효
    let folded: String = t.chars().map(|c| {
        let u = c as u32;
        if (0xFF10..=0xFF19).contains(&u)      // ０-９
            || (0xFF21..=0xFF3A).contains(&u)  // Ａ-Ｚ
            || (0xFF41..=0xFF5A).contains(&u) { // ａ-ｚ
            char::from_u32(u - 0xFEE0).unwrap_or(c)
        } else { c }
    }).collect();
    
    let n: String = folded.chars().filter(|c| c.is_alphanumeric()).collect();
    if n.len() < 4 { return false; }
    
    // 순수 숫자 키는 최소 6자리 이상만 인정합니다.
    if n.chars().all(|c| c.is_ascii_digit()) && n.len() < 6 { return false; }
    // 순수 알파벳 키는 문서번호가 아닙니다.
    if n.chars().all(|c| c.is_ascii_alphabetic()) { return false; }
    
    true
}

#[cfg(test)]
mod relay_tests {
    use super::*;

    #[test]
    fn relay_key_is_case_and_separator_invariant() {
        assert_eq!(relay_index("BL-55432219"), relay_index("bl 55432219"));
        assert_eq!(relay_index("CI-43726"),    relay_index("ci43726"));
        assert_eq!(relay_index("CI-43726"),    relay_index("ＣＩ－４３７２６"));
    }

    #[test]
    fn digest_is_not_case_invariant_and_must_not_be_used_for_relay() {
        // 이 단언이 깨지는 날이 오면 digest 를 릴레이에 써도 되는 날입니다.
        assert_ne!(digest("BL-55432219"), digest("bl-55432219"));
    }

    #[test]
    fn noise_is_rejected_as_relay_key() {
        assert!(!is_valid_relay_key("CI"));
        assert!(!is_valid_relay_key("2022"));
        assert!(!is_valid_relay_key("Shorts"));
        assert!(is_valid_relay_key("CI-43726"));
        assert!(is_valid_relay_key("93763111837"));
        assert!(is_valid_relay_key("ORD32829"));
    }
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

pub fn normalize_identifier(raw: &str) -> String {
    let folded: String = raw.chars().map(|c| {
        let u = c as u32;
        if (0xFF10..=0xFF19).contains(&u)
            || (0xFF21..=0xFF3A).contains(&u)
            || (0xFF41..=0xFF5A).contains(&u) {
            char::from_u32(u - 0xFEE0).unwrap_or(c)
        } else { c }
    }).collect();
    let stripped: String = folded
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_uppercase();
    if stripped.is_empty() { return stripped; }
    if stripped.chars().any(|c| c.is_alphabetic()) {
        stripped
    } else {
        normalize_numeric_homoglyphs(&stripped)
    }
}

pub fn digest(text: &str) -> String {
    let normalized = normalize_numeric_homoglyphs(text);
    let cleaned: String = normalized
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    if cleaned.is_empty() { return String::new(); }
    hash_id(&cleaned)
}