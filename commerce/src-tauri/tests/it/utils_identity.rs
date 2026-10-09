//! 식별자 계층 검증 — hash / crypto / compression / json_utils
//!
//! - `hash_id`, `crc32` 는 프론트엔드(ethers 6.6.2 `computeAddress(hashMessage(s))`, JS crc32)와
//!   문자 단위로 같아야 합니다. 아래 벡터는 실제 ethers / zlib 실행 결과입니다.
//! - relay_* 는 "같은 문서번호 → 같은 index/id" 를 보장하는 축이므로 표기 변형(대소문자, 구분자, 전각)에
//!   불변이어야 합니다.
//! - 순수 함수만 다룹니다. 파일/네트워크 부작용 없음.

use crate::common::pattern;
use serde_json::json;
use tauri_app_lib::utils::compression::{compress_value, decompress_to_value};
use tauri_app_lib::utils::crypto::{decrypt_data, encrypt_data};
use tauri_app_lib::utils::hash::{
    crc32, digest, get_base_domain, hash_id, is_valid_relay_key, normalize_identifier,
    normalize_numeric_homoglyphs, relay_id, relay_index,
};
use tauri_app_lib::utils::json_utils::merge_node;

// ───────────────────────────── hash_id ─────────────────────────────

#[test]
fn hash_id_matches_frontend_ethers_vectors() {
    // ethers 6.6.2 실행 결과 (coordinator 제공)
    let vectors = [
        ("logis.center", "0x45a2a96bf8ae28042073444f3ec1dd7acaa8a961"),
        ("example.com", "0x5134846d336b9828abe829c91bdaa2d07fea6a28"),
        ("example.co.kr", "0xb2e020367adcbfdf936a3266bb8ab4515fb0623b"),
        ("a@b.com", "0xf7932301cf309f69791a5e67be5a400b0c26106d"),
        ("", "0x7404358246d491ed2f9dac694f7dea789037424c"),
        (
            "0x0000000000000000000000000000000000000000",
            "0x4d9b9d714f7c42cacb353cf4861bb8bfd88a0420",
        ),
    ];
    for (input, expected) in vectors {
        assert_eq!(hash_id(input), expected, "hash_id({input:?})");
    }
}

#[test]
fn hash_id_more_vectors_and_address_shape() {
    // 순수 파이썬 keccak256 + secp256k1 구현으로 교차 계산한 값 (위 ethers 벡터 6개로 구현 자체를 검증함)
    let vectors = [
        ("hello", "0x56d67386939607c11bd60bb009eb02f4dd29c318"),
        ("안녕", "0xd6581ff7b7c9820caea226135acb3a98b34003c7"),
        ("trading.logis.center", "0xa276e12f1939e2a9ad4412ab059e2cdacc659283"),
        ("orderORD32829", "0x47d099fd247bf09a87dc6265e385b1977f49947d"),
    ];
    for (input, expected) in vectors {
        let h = hash_id(input);
        assert_eq!(h, expected, "hash_id({input:?})");
        // 주소 형태: 0x + 40자리 소문자 hex
        assert_eq!(h.len(), 42);
        assert!(h.starts_with("0x"));
        assert!(h[2..].chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)), "{h}");
    }
    // 결정적이어야 함
    assert_eq!(hash_id("logis.center"), hash_id("logis.center"));
    assert_ne!(hash_id("a"), hash_id("A"));

    // 무역 트랙 cc 는 trading Worker 와 문자 단위로 같아야 하는 고정값
    assert_eq!(tauri_app_lib::parsing::TRADING_HOST, "trading.logis.center");
    assert_eq!(
        tauri_app_lib::parsing::trading_cc(),
        "0xa276e12f1939e2a9ad4412ab059e2cdacc659283"
    );
}

// ───────────────────────────── crc32 / relay ─────────────────────────────

#[test]
fn crc32_matches_frontend_and_zlib_vectors() {
    let vectors: [(&str, u32); 7] = [
        ("", 0),
        ("a", 3904355907),
        ("abc", 891568578),
        ("123456789", 3421780262), // CRC-32 표준 check 값 0xCBF43926
        ("hello", 907060870),
        ("The quick brown fox jumps over the lazy dog", 1095738169),
        ("CI43726", 1963561162),
    ];
    for (input, expected) in vectors {
        assert_eq!(crc32(input), expected, "crc32({input:?})");
    }
    assert_eq!(crc32("123456789"), 0xCBF4_3926);
}

#[test]
fn relay_index_is_invariant_to_case_separators_and_fullwidth() {
    assert_eq!(relay_index("BL-55432219"), 3958018455);
    assert_eq!(relay_index("bl 55432219"), 3958018455);
    assert_eq!(relay_index("CI-43726"), 1963561162);
    assert_eq!(relay_index("ci43726"), 1963561162);
    assert_eq!(relay_index("ＣＩ－４３７２６"), 1963561162);
    assert_eq!(relay_index("ORD32829"), 2444371425);
    assert_eq!(relay_index("93763111837"), 1447238541);
    // 유효하지 않은 키는 0 (= 릴레이 없음)
    assert_eq!(relay_index("CI"), 0);
    assert_eq!(relay_index("N/A"), 0);
    // 정의: crc32(normalize_identifier(raw))
    for raw in ["PO-99281A", "MSKU1234567", "BL55432219"] {
        assert_eq!(relay_index(raw), crc32(&normalize_identifier(raw)), "{raw}");
    }
}

#[test]
fn relay_id_and_digest() {
    // relay_id = hash_id(target_type + normalize_identifier(raw))
    assert_eq!(
        relay_id("CI-43726", "reference_invoice"),
        "0x379a5d1b29644ed9beb9a512dcd39dd03db58800"
    );
    assert_eq!(relay_id("CI-43726", "reference_invoice"), hash_id("reference_invoiceCI43726"));
    assert_eq!(relay_id("ord-32829", "order"), "0x47d099fd247bf09a87dc6265e385b1977f49947d");
    // 같은 번호라도 대상 타입이 다르면 다른 id
    assert_ne!(relay_id("CI-43726", "order"), relay_id("CI-43726", "reference_invoice"));
    // 무효 키 → 빈 문자열
    assert_eq!(relay_id("N/A", "order"), "");
    assert_eq!(relay_id("2022", "order"), "");

    // digest: 'b'→'6', 'l'→'1' 호모글리프 교정 때문에 대소문자가 다른 digest 를 만듭니다 (릴레이에 쓰면 안 되는 이유)
    assert_eq!(digest("BL-55432219"), hash_id("BL55432219"));
    assert_eq!(digest("BL-55432219"), "0x8cd8ec5ed854f374b2804dadc26155251362df0e");
    assert_eq!(digest("bl-55432219"), hash_id("6155432219"));
    assert_eq!(digest("bl-55432219"), "0x0d0e2b2e56cd62cb479a8c0ba7c5afee371ebe69");
    assert_ne!(digest("BL-55432219"), digest("bl-55432219"));
    assert_eq!(digest("--"), "");
    assert_eq!(digest(""), "");
}

#[test]
fn normalize_identifier_and_homoglyphs() {
    assert_eq!(normalize_identifier("ＣＩ－４３７２６"), "CI43726");
    assert_eq!(normalize_identifier("ab-12"), "AB12");
    assert_eq!(normalize_identifier("123-456 789"), "123456789");
    assert_eq!(normalize_identifier(" -/ "), "");

    assert_eq!(normalize_numeric_homoglyphs("SO-l2Z"), "50-122");
    assert_eq!(normalize_numeric_homoglyphs("Bg"), "B9");
    assert_eq!(normalize_numeric_homoglyphs("１２３"), "123");
    assert_eq!(normalize_numeric_homoglyphs("한국"), "한국");
}

#[test]
fn is_valid_relay_key_rejects_placeholders_and_noise() {
    for bad in ["", "   ", "CI", "2022", "Shorts", "N/A", "null", "none", "...", "１２３４５"] {
        assert!(!is_valid_relay_key(bad), "{bad:?} should be rejected");
    }
    for good in ["CI-43726", "93763111837", "ORD32829", "123456", "AB12", "ＣＩ－４３７２６"] {
        assert!(is_valid_relay_key(good), "{good:?} should be accepted");
    }
}

#[test]
#[ignore = "BUG(B10): is_valid_relay_key counts UTF-8 bytes and only rejects ASCII-alphabetic keys, so 2-char Hangul/Latin-1 words pass"]
fn is_valid_relay_key_rejects_short_non_ascii_words() {
    // "가나" 는 2글자 순수 문자 → 문서번호가 아님 (n.len() 이 바이트 수 6 이라 통과됨)
    assert!(!is_valid_relay_key("가나"));
    // "éé" 도 2글자 순수 문자 (바이트 4)
    assert!(!is_valid_relay_key("éé"));
}

#[test]
fn get_base_domain_respects_label_boundaries() {
    let cases = [
        ("www.example.com", "example.com"),
        ("shop.deco.kr", "deco.kr"), // 'deco.kr' 은 'co.kr' 접미사가 아님
        ("www.naver.co.kr", "naver.co.kr"),
        ("a.b.c.co.uk", "c.co.uk"),
        ("co.kr", "co.kr"),
        ("localhost", "localhost"),
        ("WWW.Example.COM", "example.com"),
        ("", ""),
    ];
    for (host, expected) in cases {
        assert_eq!(get_base_domain(host), expected, "get_base_domain({host:?})");
    }
}

// ───────────────────────────── crypto ─────────────────────────────

#[test]
fn crypto_roundtrip_prepends_nonce() -> anyhow::Result<()> {
    for (n, salt) in [(0usize, 0u8), (1, 1), (63, 2), (64, 3), (65, 4), (1000, 5)] {
        let plain = pattern(n, salt);
        let enc = encrypt_data(&plain)?;
        assert_eq!(enc.len(), n + 8, "8-byte nonce prefix (n={n})");
        assert_eq!(decrypt_data(&enc)?, plain, "roundtrip n={n}");
    }
    // 같은 평문이라도 nonce 가 달라 암호문이 달라야 함
    let plain = pattern(256, 9);
    let a = encrypt_data(&plain)?;
    let b = encrypt_data(&plain)?;
    assert_ne!(a, b);
    assert_ne!(&a[8..], plain.as_slice(), "payload must not be stored in clear");
    Ok(())
}

#[test]
fn crypto_rejects_truncated_input() -> anyhow::Result<()> {
    let err = decrypt_data(&[1, 2, 3]).err().expect("shorter than nonce must fail");
    assert!(err.to_string().contains("too short"), "{err}");
    // nonce 만 있는 입력은 빈 평문
    assert!(decrypt_data(&[0u8; 8])?.is_empty());
    Ok(())
}

// ───────────────────────────── compression ─────────────────────────────

#[test]
fn compression_roundtrip_and_garbage_rejection() -> anyhow::Result<()> {
    let v = json!({
        "title": "니트 가디건 Knit",
        "price": 15000,
        "ratio": 0.25,
        "tags": ["a", "b", "가"],
        "nested": {"ok": true, "none": null, "deep": [{"x": 1}, {"y": [1, 2, 3]}]}
    });
    let bin = compress_value(&v)?;
    assert_eq!(&bin[..2], &[0x1f, 0x8b], "gzip magic");
    assert_eq!(decompress_to_value(&bin)?, v);
    // 스칼라도 그대로
    assert_eq!(decompress_to_value(&compress_value(&json!("s"))?)?, json!("s"));

    // gzip 이 아니거나 비어 있으면 에러
    assert!(decompress_to_value(b"not gzip at all").is_err());
    assert!(decompress_to_value(&[]).is_err());
    Ok(())
}

// ───────────────────────────── json_utils::merge_node ─────────────────────────────

#[test]
fn merge_node_skips_empty_values_but_keeps_false_and_empty_collections() {
    let a = json!({"a": 1, "b": "x"});
    let b = json!({"b": "", "c": null, "d": false, "e": []});
    assert_eq!(merge_node(&a, &b), json!({"a": 1, "b": "x", "d": false, "e": []}));
    // 일반 덮어쓰기
    assert_eq!(merge_node(&json!({"k": "old"}), &json!({"k": "new"})), json!({"k": "new"}));
    // 객체가 아니면 첫 인자를 그대로 돌려줌
    assert_eq!(merge_node(&json!([1, 2]), &json!({"a": 1})), json!([1, 2]));
    assert_eq!(merge_node(&json!({"a": 1}), &json!("str")), json!({"a": 1}));
}

#[test]
#[ignore = "BUG(B29): merge_node treats numeric 0 as empty, so a field can never be updated to 0 (e.g. stock sold out)"]
fn merge_node_can_set_numeric_zero() {
    assert_eq!(merge_node(&json!({"stock": 5}), &json!({"stock": 0})), json!({"stock": 0}));
}
