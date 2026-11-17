//! 추출 병합의 순수 규칙: 스키마 설명 되울림 판정 · 필드 주인 라벨 뱅크. 예전 model/merge.rs 에서 옮김 (PUG 파서 · 자연어 변환이 씀).

pub fn is_schema_echo(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return true;
    }
    // {String} / {Number} / <value> 같은 플레이스홀더 표기
    if (t.starts_with('{') && t.ends_with('}')) || (t.starts_with('<') && t.ends_with('>')) {
        return true;
    }
    let lower = t.to_lowercase();
    matches!(
        lower.as_str(),
        "..." | "null" | "n/a" | "na" | "none" | "undefined" | "unknown"
            | "string" | "number" | "boolean" | "array" | "object" | "integer" | "float"
            | "yyyy-mm-dd" | "yyyy-mm-ddthh:mm:ss" | "iso8601" | "iso 8601"
            | "not specified" | "not available" | "not found"
            | "nicht angegeben" | "nicht verfügbar" | "keine angabe" | "unbekannt" | "entfällt"
            | "non spécifié" | "non renseigné" | "non disponible" | "inconnu" | "sans objet"
            | "no especificado" | "no disponible" | "desconocido" | "no aplica" | "sin datos"
            | "non specificato" | "non disponibile" | "sconosciuto" | "non applicabile"
            | "não especificado" | "não disponível" | "desconhecido" | "não aplicável"
            | "niet opgegeven" | "niet beschikbaar" | "onbekend" | "niet van toepassing"
            | "neuvedeno" | "nedostupné" | "neznámé" | "nevztahuje se"
            | "غير محدد" | "غير متوفر" | "غير معروف" | "لا ينطبق"
            | "指定なし" | "該当なし" | "不明" | "未記入" | "なし"
            | "未指定" | "不适用" | "未知" | "无" | "未提供"
            | "해당 없음" | "해당없음" | "정보 없음" | "정보없음" | "미기재" | "없음" | "미상"
    )
}

pub fn plural_equivalent(short: &str, long: &str) -> bool {
    let sc = short.chars().count();
    let lc = long.chars().count();
    if sc == 0 || lc < sc { return false; }
    if lc > sc {
        return match long.strip_prefix(short) {
            Some("S") => true,
            Some(rest) => sc >= 3 && PLURAL_SUFFIXES_ML.iter().any(|s| *s == rest),
            None => false,
        };
    }
    if sc < 4 { return false; }
    let mut a: Vec<char> = short.chars().collect();
    let mut b: Vec<char> = long.chars().collect();
    let la = match a.pop() { Some(c) => c, None => return false };
    let lb = match b.pop() { Some(c) => c, None => return false };
    if a != b { return false; }
    PLURAL_VOWEL_SWAPS_ML
        .iter()
        .any(|(s, p)| (*s == la && *p == lb) || (*s == lb && *p == la))
}

/// 소유권 경쟁·복구 게이트가 쓰는 라벨 뱅크. 다국어 + 보강표를 합칩니다.
///
///  ── 왜 함수로 빼는가 ──
///   이 뱅크를 만드는 곳과 그 구의 임베딩을 미리 계산해 두는 곳(vision.rs 의 lookup)이
///   서로 다른 파일에 있습니다. 두 곳이 각자 뱅크를 조립하면 한쪽만 다국어가 되는 순간
///   lookup 에 없는 구가 조용히 건너뛰어져, 구를 늘렸는데 점수가 그대로인 상태가 됩니다.
pub fn owner_label_bank(doc_lang: &str, bank_type: &str, field: &str) -> (Vec<String>, Vec<f32>) {
    let (mut ph, mut wt) =
        crate::utils::ai_utils::label_phrase_bank_multilingual(doc_lang, bank_type, field);
    let sup = crate::logic::trade_label_supplement(field);
    if !sup.is_empty() {
        crate::logic::merge_phrase_bank(&mut ph, &mut wt, &sup, 1.0);
    }
    (ph, wt)
}

pub const PLURAL_SUFFIXES_ML: [&str; 6] = ["S", "ES", "N", "EN", "X", "Y"];

pub const PLURAL_VOWEL_SWAPS_ML: [(char, char); 4] = [('O', 'I'), ('A', 'E'), ('E', 'I'), ('A', 'Y')];
