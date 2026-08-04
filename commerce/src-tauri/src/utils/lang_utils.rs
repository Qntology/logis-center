pub fn detect_document_language(text: &str) -> String {
    let mut is_korean = false;
    let mut is_japanese = false;
    let mut is_chinese_char = false;
    let mut is_russian = false;
    let mut is_arabic = false;
    let mut is_thai = false;
    let mut is_hindi = false;
    let mut is_bengali = false;
    let mut is_greek = false;
    let mut is_hebrew = false;
    let mut is_vietnamese = false;
    let mut has_latin = false;

    for c in text.chars() {
        let u = c as u32;
        // 1. 유니코드 블록이 명확한 언어들
        if (u >= 0xAC00 && u <= 0xD7A3) || (u >= 0x1100 && u <= 0x11FF) || (u >= 0x3130 && u <= 0x318F) { is_korean = true; }
        else if (u >= 0x3040 && u <= 0x309F) || (u >= 0x30A0 && u <= 0x30FF) { is_japanese = true; }
        else if u >= 0x4E00 && u <= 0x9FFF { is_chinese_char = true; }
        else if u >= 0x0400 && u <= 0x04FF { is_russian = true; }
        else if u >= 0x0600 && u <= 0x06FF { is_arabic = true; }
        else if u >= 0x0E00 && u <= 0x0E7F { is_thai = true; }
        else if u >= 0x0900 && u <= 0x097F { is_hindi = true; }
        else if u >= 0x0980 && u <= 0x09FF { is_bengali = true; }
        else if u >= 0x0370 && u <= 0x03FF { is_greek = true; }
        else if u >= 0x0590 && u <= 0x05FF { is_hebrew = true; }
        else if u >= 0x1EA0 && u <= 0x1EF9 { is_vietnamese = true; }
        // 2. 라틴 알파벳 영역 (whatlang 세부 판별 필요)
        else if (u >= 0x0041 && u <= 0x005A) || (u >= 0x0061 && u <= 0x007A) || (u >= 0x00C0 && u <= 0x024F) {
            has_latin = true;
        }
    }

    // 우선순위 판별: 확실한 유니코드가 있다면 whatlang 호출 없이 즉시 결정
    let local_language = if is_korean {
        "ko".to_string()
    } else if is_japanese {
        "ja".to_string()
    } else if is_thai {
        "th".to_string()
    } else if is_russian {
        "ru".to_string()
    } else if is_arabic {
        "ar".to_string()
    } else if is_hindi {
        "hi".to_string()
    } else if is_bengali {
        "bn".to_string()
    } else if is_greek {
        "el".to_string()
    } else if is_hebrew {
        "he".to_string()
    } else if is_vietnamese {
        "vi".to_string()
    } else if is_chinese_char {
        "zh-hans".to_string()
    } else if has_latin {
        // 유니코드로 구분이 힘든 라틴 알파벳 계열 언어만 whatlang으로 정밀 분류
        match whatlang::detect(text) {
            Some(info) => match info.lang() {
                whatlang::Lang::Fra => "fr",
                whatlang::Lang::Deu => "de",
                whatlang::Lang::Spa => "es",
                whatlang::Lang::Ita => "it",
                whatlang::Lang::Por => "pt",
                whatlang::Lang::Nld => "nl",
                _ => "en",
            }.to_string(),
            None => "en".to_string(),
        }
    } else {
        "en".to_string()
    };

    let mut detected_languages_vec = vec![local_language.clone()];
    if local_language != "en" {
        detected_languages_vec.push("en".to_string());
    }

    if let Some(pos) = detected_languages_vec.iter().position(|l| l == &local_language) {
        let local = detected_languages_vec.remove(pos);
        detected_languages_vec.insert(0, local);
    }
    
    detected_languages_vec[0].clone()
}