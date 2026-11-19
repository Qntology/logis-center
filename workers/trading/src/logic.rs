//! 원본 1~353 줄: 상수 · hashId · base64 · encodeData · stableStringify · contentDigest · decodeData · normalizeDocType ·
//! isTradeDocType · entityBcc · normalizeIdentifier · findRelayValue · resolveHubKey · tradingRef. (resolveTeam 은 `team.rs`)

use std::sync::OnceLock;

use logis_worker_common::console_log;
use logis_worker_common::gzip::{decode_utf8, gzip};
use logis_worker_common::jsv::{js_trim, num_to_string, parse_str, quote, stringify, stringify_checked, to_lower, to_upper, R, V};
use logis_worker_common::util;
use regex::Regex;

use crate::jsrt;

/// 무역 트랙의 합성 도메인 (30)
pub const TRADING_HOST: &str = "trading.logis.center";
/// 31
pub const TRADING_MODE: &str = "shipping";

/// 35~46: bias_schema.rs 의 TRADE_DOC_TYPES 55종
pub const TRADE_DOC_TYPES: &[&str] = &[
    // 기존 27종
    "BL", "AWB", "CI", "PI", "PL", "PO", "SC", "LC", "CO",
    "SA", "DO", "AN", "BC", "ED", "ID", "CINV",
    "IC", "WC", "CA", "PHYTO", "HC", "BEN_CERT",
    "DGD", "MSDS", "POA", "BIZ_LIC", "INS",
    // overlay 18종
    "HBL", "FCR", "POD", "LG", "TR", "LLC", "TI", "CP", "CM",
    "CCC", "CNM", "PC", "COA", "FI", "CDR", "ICF", "SOA", "EL",
    // 공용 사전 10종
    "BE", "SR", "BK", "WR", "CSI", "SWB", "IP", "DN", "CN", "FC",
];

/// 47~53: `TRADE_DOC_TYPE_MAP[t]` (키는 `toUpperCase()`). 평범한 객체라 프로토타입 이름도 읽히지만 키가 대문자라
/// Object.prototype 의 이름(모두 소문자가 섞임)과 겹치지 않는다
fn trade_doc_type_map(t: &str) -> Option<&'static str> {
    TRADE_DOC_TYPES.iter().find(|x| to_upper(x) == t).copied()
}

/// 57~62: logic.rs 의 TRADE_HUB_TYPES 와 같은 순서
pub const HUB_PRIORITY: &[(&str, &str)] = &[("PO", "reference_po"), ("CI", "reference_invoice"), ("BL", "reference_bl"), ("LC", "reference_lc")];

/// 75~84: parsing.rs 의 TRADE_RELAY_FIELDS 와 같은 집합 · 같은 순서
pub const TRADE_RELAY_FIELDS: &[(&str, &[&str])] = &[
    ("self", &["doc_number", "reference_number", "reference_invoice"]),
    ("transport", &["reference_bl", "reference_master_bl", "bl_number", "awb_number", "airway_bill_number", "tracking_number"]),
    ("booking", &["reference_booking", "booking_number"]),
    ("order", &["reference_po", "po_number", "order_number"]),
    ("contract", &["reference_contract", "reference_sr", "contract_number"]),
    ("credit", &["reference_lc", "lc_number"]),
    ("container", &["container_number"]),
];

/// 86~91: `Hello[language]`
pub fn hello(language: &V) -> V {
    let key = language.to_key();
    let v = match key.as_str() {
        "Korean" => "무역 서류를 동기화합니다",
        "English" => "Trading documents are being synchronized",
        "Japanese" => "貿易書類を同期します",
        "Chinese" => "正在同步贸易单据",
        // 고침 (fix-trading-proto-keys): 자기 속성만 본다 (Object.prototype 의 이름은 없는 것)
        _ => return V::Undef,
    };
    V::from(v)
}

/// 92~96: `languageCode[FLAG]`
pub fn language_code(flag: &V) -> V {
    let key = flag.to_key();
    let v = match key.as_str() {
        "kr" => "Korean",
        "jp" => "Japanese",
        "cn" | "tw" | "hk" => "Chinese",
        "us" | "ca" | "gb" | "au" | "nz" | "sg" | "in" | "ph" => "English",
        // 고침 (fix-trading-proto-keys): 'constructor' · '__proto__' 같은 FLAG 는 없는 나라 (→ English)
        _ => return V::Undef,
    };
    V::from(v)
}

/// 고침 (fix-trading-cors): 자격증명(쿠키)과 함께 응답을 읽어도 되는 출처. 그 밖의 Origin 에는 CORS 허가 머리글을 붙이지 않는다.
/// 실제 소비자(데스크톱 앱의 proxy_fetch · reqwest)는 Origin 을 보내지 않는다 — 여기 있는 것은 logis.center 의 웹 출처뿐
pub const ALLOWED_ORIGINS: &[&str] = &["https://logis.center", "https://commerce.logis.center", "https://trading.logis.center"];

/// 고침 (fix-trading-preload-chunk): digest 사전 조회 한 번에 묶는 id 수 (D1 바인딩 상한 100)
pub const PRELOAD_CHUNK: usize = 99;

/// 고침 (fix-trading-html-escape): `escapeHtml(s)` — `& < > " '` 를 엔티티로
pub fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// 100~107: `hashId(text)`
pub fn hash_id(text: &V) -> R<String> {
    util::hash_id(text)
}

/// 113~120: `bytesToBase64(bytes)`
pub fn bytes_to_base64(bytes: &[u8]) -> String {
    jsrt::btoa_bytes(bytes)
}

/// 121~128: `base64ToBytes(b64)` (atob 이 던지면 던진다)
pub fn base64_to_bytes(b64: &str) -> R<Vec<u8>> {
    jsrt::atob(b64)
}

/// 131~133: `encodeData(obj)` → gzip 한 ArrayBuffer
pub fn encode_data(obj: &V) -> R<V> {
    let target = if obj.truthy() { obj.clone() } else { V::new_obj() };
    let text = stringify_checked(&target)?.unwrap_or_default();
    Ok(V::bytes(gzip(text.as_bytes())))
}

/// 151
pub const DIGEST_VOLATILE_KEYS: &[&str] = &["created_at", "updated_at", "digest", "embed", "time", "views", "score", "tier"];

/// 153~170: `stableStringify(v)` — JS 의 undefined 반환은 None
pub fn stable_stringify(v: &V) -> Option<String> {
    match v {
        V::Arr(_) => {
            // `parts.join(',')` 은 undefined 를 빈 글로 쓴다
            let parts: Vec<String> = v.items().iter().map(|x| stable_stringify(x).unwrap_or_default()).collect();
            Some(format!("[{}]", parts.join(",")))
        }
        // ArrayBuffer · JS 객체: Object.keys 가 비어 "{}" (원본에서는 오지 않는다)
        V::Bytes(_) | V::Ext(_) => Some("{}".into()),
        V::Obj(_) => {
            let mut keys: Vec<String> = v.keys().into_iter().filter(|k| !DIGEST_VOLATILE_KEYS.contains(&k.as_str())).collect();
            // `.sort()` 기본 비교: UTF-16 단위 사전순
            keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            let out: Vec<String> = keys
                .iter()
                // `JSON.stringify(k) + ':' + undefined` 는 "…:undefined"
                .map(|k| format!("{}:{}", quote(k), stable_stringify(&v.get(k)).unwrap_or_else(|| "undefined".into())))
                .collect();
            Some(format!("{{{}}}", out.join(",")))
        }
        other => stringify(other),
    }
}

/// 172~185: `contentDigest(data)` — FNV-1a 64bit (32bit 두 개)
pub fn content_digest(data: &V) -> String {
    let target = if data.truthy() { data.clone() } else { V::new_obj() };
    // 원본 그대로: 객체가 아닌 참 값이면 stableStringify 가 undefined 를 돌려 s.length 에서 던지지만, 부르는 곳은 늘 객체를 넘긴다
    let s = stable_stringify(&target).unwrap_or_default();
    let mut h1: u32 = 0x811c9dc5;
    let mut h2: u32 = 0x811c9dc5;
    for (i, c) in s.encode_utf16().enumerate() {
        let c = c as u32;
        h1 ^= c & 0xff;
        h1 = h1.wrapping_mul(0x01000193);
        h2 ^= ((c >> 8) & 0xff) ^ (i as u32 & 0xff);
        h2 = h2.wrapping_mul(0x01000193);
    }
    format!("{h1:08x}{h2:08x}")
}

/// 188~213: `decodeData(raw)` — D1 BLOB / base64 / number[] → data 객체 (풀 수 없으면 `{}`)
pub fn decode_data(raw: &V) -> V {
    if !raw.truthy() {
        return V::new_obj();
    }
    let r: R<V> = (|| {
        let bytes: Option<Vec<u8>> = if let V::Str(s) = raw {
            Some(base64_to_bytes(s)?)
        } else if raw.is_arr() {
            Some(jsrt::uint8array_new(raw)?)
        } else if raw.get("buffer").truthy() {
            Some(jsrt::uint8array_new(&raw.get("buffer"))?)
        } else if let V::Bytes(b) = raw {
            Some(b.borrow().clone())
        } else {
            let keys = raw.keys();
            if !keys.is_empty() && !jsrt::is_nan(&V::from(keys[0].as_str())) {
                // `new Uint8Array(Object.values(raw))`
                Some(jsrt::uint8array_new(&V::from(keys.iter().map(|k| raw.get(k)).collect::<Vec<_>>()))?)
            } else {
                None
            }
        };
        let Some(bytes) = bytes else {
            return Ok(V::new_obj());
        };
        parse_str(&decode_utf8(&jsrt::pako_ungzip(&bytes)?))
    })();
    match r {
        Ok(v) => v,
        Err(err) => {
            console_log!("[TRADING] decodeData err", err);
            V::new_obj()
        }
    }
}

/// `String(raw ? raw : '').trim()`
fn str_or_empty_trim(raw: &V) -> String {
    let s = if raw.truthy() { raw.to_str() } else { String::new() };
    js_trim(&s).to_string()
}

/// 224~234: `normalizeDocType(raw)` — 무역 서식이면 대문자 표준형, 그 외는 소문자
pub fn normalize_doc_type(raw: &V) -> String {
    let t = str_or_empty_trim(raw);
    if t.is_empty() {
        return "unknown".into();
    }
    if let Some(canon) = trade_doc_type_map(&to_upper(&t)) {
        return canon.into();
    }
    to_lower(&t)
}

/// 235~244: `isTradeDocType(raw)`
pub fn is_trade_doc_type(raw: &V) -> bool {
    let t = to_upper(&str_or_empty_trim(raw));
    if t.is_empty() {
        return false;
    }
    if t == "SHIPPING_DOC" {
        return true;
    }
    trade_doc_type_map(&t).is_some()
}

/// 247~249: `entityBcc(type, cc)` = hashId(type + cc)
pub fn entity_bcc(type_: &V, cc: &V) -> R<String> {
    let a = if type_.truthy() { type_.to_str() } else { String::new() };
    let b = if cc.truthy() { cc.to_str() } else { String::new() };
    hash_id(&V::from(a + &b))
}

/// 265~282: `normalizeIdentifier(raw)` — 전각 영숫자를 반각으로, `[^\p{L}\p{N}]` 제거, 대문자
pub fn normalize_identifier(raw: &V) -> String {
    if raw.is_nullish() {
        return String::new();
    }
    let s = raw.to_str();
    let folded: String = s
        .chars()
        .map(|c| {
            let code = c as u32;
            if (0xFF10..=0xFF19).contains(&code) || (0xFF21..=0xFF3A).contains(&code) || (0xFF41..=0xFF5A).contains(&code) {
                char::from_u32(code - 0xFEE0).unwrap_or(c)
            } else {
                c
            }
        })
        .collect();
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"[^\p{L}\p{N}]").unwrap());
    to_upper(&re.replace_all(&folded, ""))
}

/// `asText(v)` (290~294)
fn as_text(v: &V) -> String {
    match v {
        V::Str(s) if !js_trim(s).is_empty() => js_trim(s).to_string(),
        V::Num(n) => num_to_string(*n),
        _ => String::new(),
    }
}

/// 289~316: `findRelayValue(data, key)` — 루트 → 카테고리 객체 → 배열 원소 순으로 얕게
pub fn find_relay_value(data: &V, key: &str) -> String {
    if data.truthy() && data.type_of() == "object" {
        let direct = as_text(&data.get(key));
        if !direct.is_empty() {
            return direct;
        }
        // 고침 (fix-trading-relay-order): 앱(logis-pug parsing.rs find_relay_value)처럼 키를 정렬 순서(코드 포인트 =
        // UTF-8 바이트 순)로 훑는다. JS 의 for-in 순서(정수 키 먼저 · 넣은 순서)는 보낸 JSON 의 키 순서에 따라 바뀌어
        // 서버 ref 와 로컬 ref 가 갈렸다
        let mut keys = data.keys();
        keys.sort();
        for k in keys {
            let sub = data.get(&k);
            if !sub.truthy() || sub.type_of() != "object" {
                continue;
            }
            if sub.is_arr() {
                for item in sub.items() {
                    if item.truthy() && item.type_of() == "object" {
                        let av = as_text(&item.get(key));
                        if !av.is_empty() {
                            return av;
                        }
                    }
                }
            } else {
                let sv = as_text(&sub.get(key));
                if !sv.is_empty() {
                    return sv;
                }
            }
        }
    }
    String::new()
}

fn norm(s: &str) -> String {
    normalize_identifier(&V::from(s))
}

/// 318~347: `resolveHubKey(docType, data, fallbackId)`
pub fn resolve_hub_key(doc_type: &V, data: &V, fallback_id: &V) -> String {
    // ① 허브 참조 (PO → CI → BL → LC)
    for (_, field) in HUB_PRIORITY {
        let v = norm(&find_relay_value(data, field));
        if !v.is_empty() {
            return v;
        }
    }
    // ② 자기 자신이 허브 타입이면 자기 doc_number
    let code = to_upper(&str_or_empty_trim(doc_type));
    let self_no = norm(&find_relay_value(data, "doc_number"));
    for (hub, _) in HUB_PRIORITY {
        if *hub == code && !self_no.is_empty() {
            return self_no;
        }
    }
    // ③ 아무 참조라도 (doc_number 는 ②에서)
    for (_role, fields) in TRADE_RELAY_FIELDS {
        for f in fields.iter() {
            if *f == "doc_number" {
                continue;
            }
            let rv = norm(&find_relay_value(data, f));
            if !rv.is_empty() {
                return rv;
            }
        }
    }
    // ④ 자기 번호 → id
    if !self_no.is_empty() {
        return self_no;
    }
    normalize_identifier(fallback_id)
}

/// 350~352: `tradingRef(team, cc, hubKey)` = hashId(team + cc + '#' + hubKey)
pub fn trading_ref(team: &V, cc: &V, hub_key: &str) -> R<String> {
    hash_id(&V::from(format!("{}{}#{}", team.to_str(), cc.to_str(), hub_key)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use logis_worker_common::{arr, obj};

    #[test]
    fn digest_and_ids() {
        let d = obj! { "b" => 1, "a" => arr!["x", V::Null], "created_at" => 5, "가" => obj!{ "z" => true } };
        assert_eq!(stable_stringify(&d).unwrap(), r#"{"a":["x",null],"b":1,"가":{"z":true}}"#);
        assert_eq!(content_digest(&V::new_obj()), {
            // FNV-1a of "{}" 를 손으로
            let mut h1: u32 = 0x811c9dc5;
            let mut h2: u32 = 0x811c9dc5;
            for (i, c) in "{}".bytes().enumerate() {
                h1 = (h1 ^ c as u32).wrapping_mul(0x01000193);
                h2 = (h2 ^ (i as u32)).wrapping_mul(0x01000193);
            }
            format!("{h1:08x}{h2:08x}")
        });
        assert_eq!(normalize_identifier(&V::from("ＢＬ-１２ 가#a")), "BL12가A");
        assert_eq!(normalize_doc_type(&V::from(" bl ")), "BL");
        assert_eq!(normalize_doc_type(&V::from("Invoice")), "invoice");
        assert_eq!(normalize_doc_type(&V::Undef), "unknown");
        assert!(is_trade_doc_type(&V::from("shipping_doc")));
        assert!(!is_trade_doc_type(&V::from("unknown")));
        let data = obj! { "header" => obj!{ "doc_number" => " ci-1 " }, "items" => arr![obj!{ "reference_bl" => 77 }] };
        assert_eq!(find_relay_value(&data, "reference_bl"), "77");
        assert_eq!(resolve_hub_key(&V::from("CI"), &data, &V::from("id1")), "77");
        assert_eq!(resolve_hub_key(&V::from("CI"), &obj! { "doc_number" => "ci-1" }, &V::from("id1")), "CI1");
        // fix-trading-relay-order: 넣은 순서 · 정수 키와 상관없이 키 정렬 순서 ("10" < "2" < "b" < "가")
        let data = obj! { "가" => obj!{ "reference_bl" => "K" }, "b" => obj!{ "reference_bl" => "B" }, "2" => arr![obj!{ "reference_bl" => "TWO" }], "10" => obj!{ "reference_bl" => "TEN" } };
        assert_eq!(find_relay_value(&data, "reference_bl"), "TEN");
        let data = obj! { "가" => obj!{ "reference_bl" => "K" }, "b" => obj!{ "reference_bl" => "B" } };
        assert_eq!(find_relay_value(&data, "reference_bl"), "B");
        assert_eq!(escape_html(r#"<a href="x">&'</a>"#), "&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;");
    }
}
