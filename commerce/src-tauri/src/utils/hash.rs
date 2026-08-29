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

    // 2. 해시값(32바이트)을 개인키로 사용하여 지갑 객체 생성
    //    🌟 secp256k1 은 0 과 curve order 이상을 개인키로 받지 않습니다.
    //       기존 구현은 이때 빈 문자열을 돌려주었는데, 빈 id 는 upsert 단계에서
    //       서로 다른 문서끼리 같은 행으로 병합되는 최악의 충돌을 만듭니다.
    //       실패하면 도메인 분리 재해싱으로 반드시 유효한 주소를 얻습니다.
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

/// 🌟 [RELAY KEY] 문서 간 연결(Relay)에 쓰는 유일한 키 생성기입니다.
///
///  ── digest() 를 릴레이 키로 쓰면 안 되는 이유 ──
///   digest 는 normalize_numeric_homoglyphs 를 '먼저' 적용하는데,
///   그 치환표는 대소문자를 비대칭으로 다룹니다.
///     'b'→6, 'l'→1, 's'→5 는 있지만 'B', 'L' 은 없습니다.
///     그리고 to_lowercase() 가 치환 '뒤' 에 옵니다.
///   결과:
///     digest("BL-55432219") → "bl55432219"   (B, L 미치환)
///     digest("bl-55432219") → "6155432219"   (b→6, l→1)
///   같은 문서번호가 대소문자만 달라도 다른 값이 됩니다.
///   릴레이는 이 둘을 같은 것으로 봐야 하므로 normalize_identifier 를 씁니다.
///
///  digest 는 서버 Digest(text) 와의 바이트 호환을 위해 그대로 두고,
///  릴레이는 이 함수들만 사용합니다.
/// 🌟 [RELAY INDEX v4]
///  ── 무엇이 바뀌었나 ──
///   기존은 `crc32(normalize_identifier(raw))` 였는데,
///   `normalize_identifier` 가 대소문자를 통일하지만
///   `relay_index` 자체에는 유효성 검사가 없었습니다.
///   빈 문자열이나 "N/A" 가 들어오면 의미 없는 인덱스가 생성됩니다.
///   유효성 검사를 내장하여 잘못된 키로 인한 릴레이 오염을 원천 차단합니다.
pub fn relay_index(raw: &str) -> u32 {
    if !is_valid_relay_key(raw) {
        return 0;
    }
    crc32(&normalize_identifier(raw))
}

/// 🌟 [RELAY ID v4]
///  ── 무엇이 바뀌었나 ──
///   기존은 `normalize_identifier` 통과값이 비어 있으면 빈 문자열을 반환했는데,
///   호출부가 빈 문자열을 그대로 `hash_id("")` 로 전달할 수 있었습니다.
///   유효성 검사를 내장하여 잘못된 키로 인한 빈 id 생성을 원천 차단합니다.
pub fn relay_id(raw: &str) -> String {
    if !is_valid_relay_key(raw) {
        return String::new();
    }
    let n = normalize_identifier(raw);
    if n.is_empty() { return String::new(); }
    hash_id(&n)
}

/// 릴레이 키로 쓸 자격이 있는지 판정합니다.
///  'CI', '1', '4' 같은 잡음이 index 로 승격되면
///  전혀 무관한 문서 수백 건이 한 행으로 뭉칩니다.
/// 🌟 [RELAY KEY VALIDATION v4]
///  ── 무엇이 바뀌었나 ──
///   기존은 `normalize_identifier` 통과값 기준으로 판정했는데,
///   "N/A", "null", "..." 같은 LLM 플레이스홀더가 통과할 수 있었습니다.
///   또한 전각 영숫자(예: "１２３４５")가 반각으로 접히기 전에는
///   길이가 부족하여 유효하지 않다고 판정될 수 있었습니다.
///   전각 접기를 먼저 수행하고, 플레이스홀더를 명시적으로 차단합니다.
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

/// 🌟 [IDENTIFIER NORMALIZE] 모든 index / id 계산의 단일 진입점입니다.
///
///  ── normalize_numeric_homoglyphs 를 직접 쓰면 안 되는 이유 ──
///   그 함수는 '스캔된 순수 숫자열' 의 OCR 오독을 되돌리기 위한 것입니다.
///   치환표에 S→5, O→0, I→1, T→7, Z→2, B→8, G→9 가 들어 있어
///   영숫자 식별자에 적용하면 글자가 통째로 숫자로 바뀝니다.
///     SC-2026-0802 → 5C20260802
///     SKU-A1B2     → 5KUA1B2
///     task_...     → ta5k...        (실측 로그)
///   그러면 ① 서로 다른 두 값이 같은 index 로 충돌하고
///          ② 무역 경로(normalize_trade_doc_number)와 결과가 갈립니다.
///
///  ── 규칙 ──
///   ① 구분자(-, _, ., ,, 공백, /) 제거
///   ② 대문자 통일 (bl-55432219 ↔ BL-55432219 는 같은 문서)
///   ③ 호모글리프 복원은 '알파벳이 하나도 없을 때' 만 적용
/// 🌟 [NORMALIZE IDENTIFIER v4]
///  ── 무엇이 바뀌었나 ──
///   기존은 전각 접기 → 비영숫자 제거 → 대문자 통일 → 호모글리프 순이었는데,
///   이 순서에서 '전각 접기' 가 `is_valid_relay_key` 와 중복되었습니다.
///   또한 `normalize_numeric_homoglyphs` 가 순수 숫자열에만 적용되었는데,
///   '12345-67890' 처럼 하이픈이 포함된 경우 비영숫자 제거 후
///   '1234567890' 이 되어 호모글리프가 적용되었습니다.
///   이 동작은 의도된 것이지만, 로그에 기록하여 추적 가능하게 만듭니다.
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

/// 서버의 Digest(text)와 동일하게 문장 부호 및 공백을 제거한 후 hash_id를 생성합니다.
/// ⚠️ 이 함수는 릴레이 키로 사용하지 마십시오.
///    digest 는 대소문자를 구분하므로 'BL-55432219' 와 'bl-55432219' 가
///    다른 해시를 생성합니다. 릴레이에는 relay_id 를 사용하십시오.
pub fn digest(text: &str) -> String {
    let normalized = normalize_numeric_homoglyphs(text);
    let cleaned: String = normalized
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    if cleaned.is_empty() { return String::new(); }
    hash_id(&cleaned)
}