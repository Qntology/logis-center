pub fn page_type_prompt(search_mode: &str) -> String {
    match search_mode {
        "shipping" => {
            r###"[TASK]
Based on the provided Pug template, identify the primary trade document category.

[SCHEMA DEFINITIONS]
- type: The main trade document category. Must be one of:
  - "PO": Purchase Order, buyer issues to seller.
  - "PI": Proforma Invoice, quotation or preliminary invoice.
  - "SC": Sales Contract, agreement between seller and buyer.
  - "LC": Letter of Credit, documentary credit from issuing bank.
  - "CI": Commercial Invoice, seller bills buyer for goods.
  - "PL": Packing List, carton details and weights.
  - "BL": Bill of Lading, ocean carrier document.
  - "AWB": Air Waybill, airline transport document.
  - "SA": Shipping Advice, shipment notification to buyer.
  - "DO": Delivery Order, release cargo to consignee.
  - "AN": Arrival Notice, cargo arrival notification.
  - "BC": Booking Confirmation, space booking with carrier.
  - "ED": Export Declaration, customs export filing.
  - "ID": Import Declaration, customs import filing.
  - "CINV": Customs Invoice, invoice for customs valuation.
  - "CO": Certificate of Origin, country of origin declaration.
  - "IC": Inspection Certificate, quality inspection result.
  - "WC": Weight Certificate, certified weight measurement.
  - "CA": Certificate of Analysis, laboratory test result.
  - "PHYTO": Phytosanitary Certificate, plant health.
  - "HC": Health Certificate, sanitary certificate.
  - "BEN_CERT": Beneficiary Certificate, beneficiary statement.
  - "DGD": Dangerous Goods Declaration, hazardous materials.
  - "MSDS": Material Safety Data Sheet, chemical hazard info.
  - "POA": Power of Attorney, authorization letter.
  - "BIZ_LIC": Business License, company registration.
  - "INS": Insurance Policy, marine cargo insurance.
  - "TRACKING": Courier label, parcel waybill.
  - "Unknown": If none of the above match.
- language: ISO 639-1 language code.

[OUTPUT FORMAT]
{ "type": "String", "language": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###.to_string()
        },
        "analytic" => {
            r###"[TASK]
Based on the provided Pug template, identify the user interaction type.

[SCHEMA DEFINITIONS]
- type: The interaction type. Must be one of:
  - "click": User pressed or selected an element.
  - "hover": User lingered over an element without pressing.
  - "change": User typed, toggled, or picked an option.
  - "report": A synthesized behavioural summary.
  - "": If none of the above match.
- language: ISO 639-1 language code.

[OUTPUT FORMAT]
{ "type": "String", "language": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###.to_string()
        },
        _ => {
            r###"[TASK]
Based on the provided Pug template, identify the primary category.

[SCHEMA DEFINITIONS]
- type: The main category. Must be one of:
  - "order": Order list, Order history, Order details, Checkout success.
  - "goods": Product list, product detail.
  - "tracking": Shipment tracking status, delivery history.
  - "review": Product reviews, feedback list.
  - "coupon": Coupon list, discount events.
  - "event": Promotion pages, event announcements.
  - "": If none of the above match.
- language: ISO 639-1 language code.

[OUTPUT FORMAT]
{ "type": "String", "language": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###.to_string()
        },
    }
}

pub fn extract_titles_prompt(page_type: &str) -> String {
    let (category_desc, titles_desc, title_desc) = match page_type {
        "goods" => ("product", "product titles", "product title"),
        "order" => ("product", "order product titles", "order product title"),
        "tracking" => ("product", "tracking product titles", "tracking product title"),
        "review" => ("title", "review titles", "review title"),
        "coupon" => ("title", "coupon titles", "coupon title"),
        "event" => ("title", "event titles", "event title"),
        _ => ("title", "titles", "title"),
    };

    let template = r###"[TASK]
Find all the {TITLES} from the following PUG/HTML content.

[SCHEMA DEFINITIONS]
{ {CATEGORY}: ["{TITLE}"] }

[OUTPUT FORMAT]
{ {CATEGORY}: [...] }

RETURN JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{CATEGORY}", category_desc)
            .replace("{TITLES}", titles_desc)
            .replace("{TITLE}", title_desc)
            .replace("{TYPE}", page_type)
}

// pub fn get_trade_doc_classification_prompt() -> String {
//     r###"Classify document type. Choose strictly from: PI, CI, BL, AWB, PL, CO, LC, TRACKING, Unknown. 
// Return JSON exactly like: {"doc_type": "BL"}
// NO EXPLANATION."###.to_string()
// }

pub fn get_trade_doc_classification_prompt() -> String {
    // 🌟 [CLASSIFIER v3 / VISION-FIRST FALLBACK]
    //  ── 호출 조건이 바뀌었습니다 ──
    //   v2 는 이 프롬프트가 '항상' 호출되는 1차 분류기였습니다.
    //   v3 에서는 SigLIP2 패치 코사인(vision_encoder::classify_doc_type)이 1차이며,
    //   그 판정의 1위-2위 마진이 사실상 동률일 때만 이 프롬프트가 1회 호출됩니다.
    //   벡터 근거는 get_trade_doc_classification_prompt_with_evidence 가 동봉합니다.
    //
    //  ── 이 함수는 언제 쓰이는가 ──
    //   SigLIP2 텍스트 인코더 로드 실패 등으로 비전 판정 자체가 불가능할 때의
    //   최후 폴백입니다. 근거 없이 목록만 제시하므로 정확도가 낮으며,
    //   정상 경로에서는 도달하지 않습니다.
    //
    //  ── 오분류 완화 ──
    //   2B 비전 모델이 27갈래를 정확히 가르기는 어렵습니다.
    //   다만 같은 그룹은 bias.json 의 trade_schema 를 공유하므로
    //   (CI|PI|SC / ED|ID|CINV / IC|WC|CA|PHYTO|HC|BEN_CERT / DGD|MSDS / POA|BIZ_LIC|INS)
    //   그룹 내 혼동은 추출 품질에 영향을 주지 않습니다.
    //   그래서 아래 프롬프트도 '그룹 → 코드' 순서로 제시합니다.
    r###"Classify this trade document. Return the single closest code.

[GROUPS]
1. Contract & Payment
   PO  = Purchase Order
   PI  = Proforma Invoice
   SC  = Sales Contract
   LC  = Letter of Credit
2. Shipping & Transport
   CI  = Commercial Invoice
   PL  = Packing List
   BL  = Bill of Lading
   AWB = Air Waybill
   SA  = Shipping Advice
   DO  = Delivery Order
   AN  = Arrival Notice
   BC  = Booking Confirmation
3. Customs
   ED   = Export Declaration
   ID   = Import Declaration
   CINV = Customs Invoice
   CO   = Certificate of Origin
4. Inspection & Certificates
   IC       = Inspection Certificate
   WC       = Weight Certificate
   CA       = Certificate of Analysis
   PHYTO    = Phytosanitary Certificate
   HC       = Health Certificate
   BEN_CERT = Beneficiary Certificate
5. Special & Legal
   DGD     = Dangerous Goods Declaration
   MSDS    = Material Safety Data Sheet
   POA     = Power of Attorney
   BIZ_LIC = Business License
   INS     = Insurance Policy
6. Parcel
   TRACKING = Courier label / parcel waybill

If none fit, return "Unknown".

[OUTPUT FORMAT]
{"doc_type": "BL"}

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###.to_string()
}

/// 🌟 [VISION EVIDENCE CLASSIFIER] 비전 코사인 근거를 동봉한 재판정 프롬프트.
///
///  ── 호출 조건 ──
///   siglip2::vision_encoder::classify_doc_type 의 코드 판정에서
///   1위와 2위가 사실상 동률(margin ≈ 0)일 때만 1회 호출됩니다.
///   마진이 충분하면 LLM 을 아예 부르지 않습니다.
///
///  ── 왜 근거를 동봉하는가 ──
///   scheduler.rs STEP A 가 [VECTOR EVIDENCE] 를 실어 보내는 것과 같은 이유입니다.
///   후보 목록만 주면 2B 모델이 목록 첫 항목이나 가장 흔한 서식(BL / CI)으로
///   쏠립니다. 코사인 점수를 함께 주면 모델은 '이미 좁혀진 선택지 중 하나' 를
///   고르는 작업만 하게 되어 창작 여지가 사라집니다.
///
///  ── 후보 제한 ──
///   candidates 는 비전이 통과시킨 코드만 담습니다.
///   모델이 그 밖의 코드를 반환하면 호출부가 폐기합니다.
pub fn get_trade_doc_classification_prompt_with_evidence(
    group: &str,
    candidates: &[(String, f32)],
) -> String {
    let mut cands = String::new();
    for (code, score) in candidates.iter().take(8) {
        cands.push_str(&format!(
            "- \"{}\" (vision cosine surprisal {:+.4}) — {}\n",
            code,
            score,
            crate::logic::trade_code_anchor(code)
        ));
    }

    let template = r###"[TASK]
The vision encoder already narrowed this document down. Pick the single closest code.

[VISION VERDICT]
Document group: "{GROUP}"

[CANDIDATE CODES]
{CANDIDATES}

[RULES]
1. Choose exactly ONE code from [CANDIDATE CODES]. Never invent a code that is not listed.
2. The scores come from patch-level cosine matching against each code's concept anchor.
   A higher score means more of the page visually matched that concept.
   Two nearly identical scores mean the page is genuinely ambiguous — decide by what you actually see.
3. Judge by the printed title and the structural layout of the page, not by which candidate is listed first.
4. If none of the candidates match what you see, return "Unknown".

[OUTPUT FORMAT]
{ "doc_type": String }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;

    template
        .replace("{GROUP}", group)
        .replace("{CANDIDATES}", &cands)
}

/// 🌟 [TRADE SCHEMA v2 / BASE + OVERLAY]
///  ── v1 의 결함 ──
///   시그니처가 `_doc_type` 이었습니다. 즉 27종 서식에 전부 같은 27개 필드를
///   물어봤습니다. L/C 의 tenor, DGD 의 un_number, CA 의 result_value 처럼
///   그 서식에만 존재하는 축은 추출 자체가 불가능했습니다.
///
///  ── 왜 bias.json 인가 ──
///   app-logis-center 의 get_category_schema 는 400줄짜리 doc_type 하드코딩입니다.
///   그대로 옮기면 새 서식마다 Rust 를 고치고 재빌드해야 합니다.
///   이 코드베이스가 이미 path_alias / multilingual_value_anchor / abstract_bridge 를
///   bias.json 으로 옮긴 것과 같은 이유로, 스키마도 데이터로 취급합니다.
///
///  ── 필드 이름 ──
///   base 는 extract_shipping_conditions(검색)와 '같은 이름' 을 씁니다.
///   그래야 저장과 조회가 alias 를 거치지 않고 바로 만납니다.
///   레거시 데이터는 path_alias 가 흡수합니다.
/// 🌟 [TYPE MARKER SPLIT] 설명 문자열 끝의 타입 표기를 값에서 떼어냅니다.
///
///  ── 무엇이 문제였나 ──
///   구버전 스키마는 `"voyage_number": "Voyage or flight leg number {String}"` 를
///   모델에게 그대로 보여 주었습니다. 그러면 값 자리에 문자열이 이미 들어 있으므로
///   2B 모델은 그 마지막 토큰을 값으로 복사합니다.
///   (실측 로그: `"voyage_number": "{String}"`)
///   타입은 '설명' 이지 '값' 이 아니므로 값 위치에서 물리적으로 제거해야 합니다.
fn split_type_marker(desc: &str) -> (String, &'static str) {
    let d = desc.trim();
    for (marker, ty) in [
        ("{String}", "String"),
        ("{Number}", "Number"),
        ("{Boolean}", "Boolean"),
        ("{Array}", "Array"),
    ] {
        if let Some(pos) = d.rfind(marker) {
            let head = d[..pos].trim().trim_end_matches(':').trim();
            let tail = d[pos + marker.len()..].trim();
            let joined = if tail.is_empty() {
                head.to_string()
            } else {
                format!("{} {}", head, tail)
            };
            return (joined.trim().to_string(), ty);
        }
    }
    (d.to_string(), "String")
}

/// 🌟 [EXAMPLE TOKEN HARVEST v2] 설명 안의 '예시 값' 을 뽑아냅니다.
///
///  ── 왜 뽑아내는가 ──
///   `"incoterms": "FOB, CIF, EXW, DDP, DAP {String}"` 를 보여 주면
///   2B 모델은 목록의 첫 항목을 답으로 복사합니다.
///   (실측: 정답 DAP 인데 FOB 반환 / 문서에 없는데 CTN 반환)
///   설명에서 예시를 지우면 '어떤 종류의 값인가' 를 모르게 되므로 지울 수는 없습니다.
///   대신 그 토큰들을 [FORBIDDEN VALUES] 로 명시해
///   "이 목록은 값이 아니라 예시" 라는 사실을 프롬프트 안에서 못박습니다.
///
///  ── v1 이 놓친 것 ──
///   v1 은 '대문자·숫자 비율' 하나만 봤습니다. bias.json 의 실제 44개 문자열에 대입하면
///     "Sea, Air, Road, Rail"                 → Sea(대문자 1/알파벳 3) 전량 미검출
///     "Freight Prepaid or Freight Collect"   → 15자 상한 초과로 미검출
///     "(e.g. PO-99281A)"                     → 14자 상한 초과로 미검출
///     "Tenor: at sight, 30 days, ..."        → 라벨 접두가 붙어 미검출
///   전부 모델이 그대로 복사할 수 있는 형태입니다.
///
///  ── v2 판정 근거 (전부 구조, 어휘 사전 없음) ──
///   R1 괄호 안 : 설명문의 괄호는 예시 열거입니다.
///   R2 예시 마커: 마침표로 끝나는 4자 이하 선행 토큰(e.g. / eg. / ex.)은 마커이므로 제거합니다.
///   R3 짧은 열거: 본문이 2조각 이상으로 쪼개지고 모든 조각이 3단어 이하면
///                그 본문은 설명이 아니라 값 열거입니다.
///                ("Sea, Air, Road, Rail" ✓ / "Subtotal before tax and charges" 는 1조각이라 미해당)
///   R4 라벨 접두: 조각에 ':' 가 있으면 뒤쪽만 취합니다. ("Tenor: at sight" → "at sight")
///   R5 대문자·숫자 우세: v1 규칙 유지. FOB / 20GP / UN1263 / YYYY-MM-DD 를 잡습니다.
fn extract_example_tokens(desc: &str) -> Vec<String> {
    /// 조각 하나를 정규화합니다. (R2 마커 제거 + R4 라벨 접두 제거 + 양끝 비영숫자 제거)
    fn normalize_segment(raw: &str) -> String {
        let mut s = raw.trim().to_string();

        // R4 : "Tenor: at sight" → "at sight"
        if let Some(p) = s.find(':') {
            let tail = s[p + 1..].trim().to_string();
            if !tail.is_empty() {
                s = tail;
            }
        }

        // R2 : "e.g. PO-99281A" → "PO-99281A"
        let toks: Vec<&str> = s.split_whitespace().collect();
        if toks.len() >= 2 {
            let head = toks[0];
            if head.ends_with('.') && head.chars().count() <= 4 {
                s = toks[1..].join(" ");
            }
        }

        s.trim()
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string()
    }

    /// 이 조각이 '값 예시로 쓰일 수 있는 생김새' 인지 최소 자격만 봅니다.
    fn is_usable(t: &str) -> bool {
        if t.is_empty() {
            return false;
        }
        if t.chars().count() > 24 {
            return false;
        }
        t.chars().filter(|c| c.is_alphanumeric()).count() >= 2
    }

    /// R5 : 대문자 + 숫자 비중이 절반 이상인 짧은 토큰
    fn is_code_like(t: &str) -> bool {
        if t.chars().count() > 12 {
            return false;
        }
        let alnum = t.chars().filter(|c| c.is_alphanumeric()).count();
        if alnum < 2 {
            return false;
        }
        let upper = t
            .chars()
            .filter(|c| c.is_uppercase() || c.is_ascii_digit())
            .count();
        upper * 2 >= alnum
    }

    /// 콤마 / " or " / '/' 를 공통 나열 구분자로 보고 쪼갭니다.
    fn split_enum(seg: &str) -> Vec<String> {
        seg.replace(" or ", ",")
            .replace('/', ",")
            .split(',')
            .map(normalize_segment)
            .filter(|s| !s.is_empty())
            .collect()
    }

    let mut out: Vec<String> = Vec::new();
    let mut push = |t: String, out: &mut Vec<String>| {
        if !is_usable(&t) {
            return;
        }
        if !out.iter().any(|e| e == &t) {
            out.push(t);
        }
    };

    // ── R1 : 괄호/대괄호 안은 무조건 예시 열거로 봅니다 ──
    let mut depth = 0usize;
    let mut buf = String::new();
    let mut paren_segs: Vec<String> = Vec::new();
    for ch in desc.chars() {
        match ch {
            '(' | '[' => {
                depth += 1;
                buf.clear();
            }
            ')' | ']' => {
                if depth > 0 {
                    depth -= 1;
                    if !buf.trim().is_empty() {
                        paren_segs.push(buf.clone());
                    }
                    buf.clear();
                }
            }
            _ => {
                if depth > 0 {
                    buf.push(ch);
                }
            }
        }
    }
    for seg in paren_segs.iter() {
        for part in split_enum(seg) {
            // 괄호 안이라도 5단어 이상이면 설명문입니다. ("4 letters + 7 digits")
            if part.split_whitespace().count() > 4 {
                continue;
            }
            push(part, &mut out);
        }
    }

    // ── 괄호를 제거한 본문 ──
    let flat: String = desc
        .chars()
        .map(|c| if c == '(' || c == ')' || c == '[' || c == ']' { ' ' } else { c })
        .collect();
    let body_parts = split_enum(&flat);

    // ── R3 : 2조각 이상 + 모든 조각 3단어 이하 → 본문 전체가 값 열거 ──
    let is_enumeration = body_parts.len() >= 2
        && body_parts
            .iter()
            .all(|p| p.split_whitespace().count() <= 3);

    for part in body_parts.iter() {
        if is_enumeration {
            push(part.clone(), &mut out);
            continue;
        }
        // ── R5 : 열거가 아니어도 코드형 토큰은 개별 수확 ──
        if is_code_like(part) {
            push(part.clone(), &mut out);
            continue;
        }
        // 조각 내부의 단어 단위로도 코드형 토큰을 찾습니다. ("UN number UN1263" 대비)
        for w in part.split_whitespace() {
            let t = w.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
            if is_code_like(&t) {
                push(t, &mut out);
            }
        }
    }

    out
}

/// 🌟 [DOC-SPECIFIC IDENTITY RULES] doc_type 이 확정된 상태에서
///    이 서식에 실제로 인쇄되는 라벨과 참조 축만 알려줍니다.
///
///  ── 왜 범용 접두어 매칭을 버리는가 ──
///   45종 × 7개 접두어 = 315개 조합을 매번 나열하면
///   2B 모델의 주의력이 분산되어 순수 숫자("93763111837")가
///   reference_bl 로 흘러가는 사고가 발생합니다.
///   이 서식에 실제로 있는 라벨만 알려주면 물리적으로 오배정이 불가능합니다.
///
///  ── 반환 형식 ──
///   (doc_number 라벨 목록, [(필드명, 인쇄 라벨 목록)], 자기참조 필드명)
fn trade_doc_identity_context(doc_type: &str) -> (Vec<&'static str>, Vec<(&'static str, Vec<&'static str>)>, &'static str) {
    // 자기참조 필드 (이 서식이 절대 참조하지 않는 축)
    let self_ref = crate::logic::trade_reference_field_of(doc_type).unwrap_or("");

    // 이 서식의 doc_number 에 붙는 실제 인쇄 라벨
    let doc_labels: Vec<&'static str> = match doc_type {
        "CI"       => vec!["INVOICE NUMBER", "Invoice No.", "INV NO.", "COMMERCIAL INVOICE NO"],
        "CINV"     => vec!["CUSTOMS INVOICE NO", "INVOICE NO."],
        "CSI"      => vec!["CONSULAR INVOICE NO", "INVOICE NO."],
        "TI"       => vec!["TAX INVOICE NO", "세금계산서 번호"],
        "FI"       => vec!["FREIGHT INVOICE NO", "INVOICE NO."],
        "PL"       => vec!["PACKING LIST NO.", "P/L NO.", "PACKING LIST NUMBER"],
        "BL"       => vec!["B/L NO.", "BILL OF LADING NO.", "B/L NUMBER", "OCEAN B/L NO."],
        "HBL"      => vec!["HOUSE B/L NO.", "HBL NO.", "HOUSE BILL OF LADING NO."],
        "SWB"      => vec!["SEA WAYBILL NO.", "WAYBILL NO.", "SWB NO."],
        "AWB"      => vec!["AIR WAYBILL NO.", "AWB NO.", "AIRWAYBILL NUMBER"],
        "PO"       => vec!["P/O NO.", "PURCHASE ORDER NO.", "ORDER NO.", "PO NUMBER"],
        "PI"       => vec!["PROFORMA INVOICE NO.", "PI NO.", "PROFORMA NO."],
        "SC"       => vec!["CONTRACT NO.", "SALES CONTRACT NO.", "S/C NO."],
        "LC"       => vec!["L/C NO.", "LETTER OF CREDIT NO.", "CREDIT NO.", "DOCUMENTARY CREDIT NO."],
        "LLC"      => vec!["LOCAL L/C NO.", "LOCAL LETTER OF CREDIT NO."],
        "CP"       => vec!["CONFIRMATION NO.", "PURCHASE CONFIRMATION NO."],
        "BE"       => vec!["DRAFT NO.", "BILL OF EXCHANGE NO.", "EXCHANGE NO."],
        "TR"       => vec!["TRUST RECEIPT NO.", "TR NO."],
        "LG"       => vec!["GUARANTEE NO.", "LETTER OF GUARANTEE NO.", "L/G NO."],
        "EL"       => vec!["LICENSE NO.", "EXPORT LICENSE NO.", "LICENCE NO."],
        "SA"       => vec!["ADVICE NO.", "SHIPPING ADVICE NO."],
        "DO"       => vec!["DELIVERY ORDER NO.", "D/O NO."],
        "AN"       => vec!["NOTICE NO.", "ARRIVAL NOTICE NO."],
        "BC" | "BK"=> vec!["BOOKING NO.", "BOOKING CONFIRMATION NO.", "BKG NO."],
        "SR"       => vec!["SHIPPING REQUEST NO.", "S/R NO."],
        "FCR"      => vec!["FCR NO.", "FORWARDER CERTIFICATE OF RECEIPT NO."],
        "POD"      => vec!["POD NO.", "PROOF OF DELIVERY NO."],
        "CM"       => vec!["MANIFEST NO.", "CARGO MANIFEST NO."],
        "WR"       => vec!["WAREHOUSE RECEIPT NO.", "W/R NO."],
        "ED"       => vec!["DECLARATION NO.", "EXPORT DECLARATION NO."],
        "ID"       => vec!["DECLARATION NO.", "IMPORT DECLARATION NO."],
        "CO"       => vec!["CERTIFICATE NO.", "CERTIFICATE OF ORIGIN NO."],
        "CNM"      => vec!["CERTIFICATE NO.", "NON-MANIPULATION CERTIFICATE NO."],
        "CCC"      => vec!["CERTIFICATE NO.", "CUSTOMS CLEARANCE CERTIFICATE NO."],
        "IC"       => vec!["CERTIFICATE NO.", "INSPECTION CERTIFICATE NO."],
        "COA" | "CA" => vec!["CERTIFICATE NO.", "CERTIFICATE OF ANALYSIS NO."],
        "WC"       => vec!["CERTIFICATE NO.", "WEIGHT CERTIFICATE NO."],
        "PHYTO" | "PC" => vec!["CERTIFICATE NO.", "PHYTOSANITARY CERTIFICATE NO."],
        "FC"       => vec!["CERTIFICATE NO.", "FUMIGATION CERTIFICATE NO."],
        "HC"       => vec!["CERTIFICATE NO.", "HEALTH CERTIFICATE NO."],
        "BEN_CERT" => vec!["CERTIFICATE NO.", "BENEFICIARY CERTIFICATE NO."],
        "CDR"      => vec!["REPORT NO.", "SURVEY REPORT NO."],
        "DGD"      => vec!["DECLARATION NO.", "DANGEROUS GOODS DECLARATION NO."],
        "MSDS"     => vec!["MSDS NO.", "SDS NO."],
        "POA"      => vec!["POWER OF ATTORNEY NO.", "POA NO."],
        "BIZ_LIC"  => vec!["LICENSE NO.", "BUSINESS LICENSE NO.", "REGISTRATION NO."],
        "INS" | "IP" => vec!["POLICY NO.", "INSURANCE POLICY NO."],
        "ICF"      => vec!["CLAIM NO.", "INSURANCE CLAIM NO."],
        "SOA"      => vec!["STATEMENT NO.", "STATEMENT OF ACCOUNT NO."],
        "DN"       => vec!["DEBIT NOTE NO.", "D/N NO."],
        "CN"       => vec!["CREDIT NOTE NO.", "C/N NO."],
        _          => vec!["DOCUMENT NO.", "NO.", "NUMBER"],
    };

    // 이 서식에 실제로 인쇄되는 참조 축과 그 인쇄 라벨
    // (자기참조 필드는 제외 — 이미 위에서 파악)
    let mut refs: Vec<(&'static str, Vec<&'static str>)> = Vec::new();

    let mut try_add = |field: &'static str, labels: Vec<&'static str>| {
        if field == self_ref { return; }
        refs.push((field, labels));
    };

    match doc_type {
        "CI" => {
            try_add("reference_po", vec!["P/O NO.", "ORDER NO.", "PURCHASE ORDER NO.", "YOUR ORDER"]);
            try_add("reference_bl", vec!["B/L NO.", "AIRWAYBILL / BILL OF LADING", "BILL OF LADING NO."]);
            try_add("reference_lc", vec!["L/C NO.", "LETTER OF CREDIT NO.", "CREDIT NO."]);
        },
        "CINV" | "CSI" => {
            try_add("reference_invoice", vec!["INVOICE NO.", "COMMERCIAL INVOICE NO."]);
            try_add("reference_po", vec!["P/O NO.", "ORDER NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
        },
        "PL" => {
            try_add("reference_invoice", vec!["INVOICE NO.", "INV NO."]);
            try_add("reference_bl", vec!["B/L NO.", "BILL OF LADING NO."]);
            try_add("reference_po", vec!["P/O NO.", "ORDER NO."]);
        },
        "BL" => {
            try_add("reference_invoice", vec!["INVOICE NO.", "COMMERCIAL INVOICE NO."]);
            try_add("reference_booking", vec!["BOOKING NO.", "BKG NO."]);
            try_add("reference_po", vec!["P/O NO.", "ORDER NO."]);
            try_add("reference_lc", vec!["L/C NO."]);
        },
        "HBL" => {
            try_add("reference_master_bl", vec!["MASTER B/L NO.", "MBL NO.", "OCEAN B/L NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_booking", vec!["BOOKING NO."]);
        },
        "SWB" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
            try_add("reference_lc", vec!["L/C NO."]);
        },
        "AWB" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_po", vec!["P/O NO.", "ORDER NO."]);
            try_add("reference_lc", vec!["L/C NO."]);
        },
        "PO" => {
            try_add("reference_lc", vec!["L/C NO.", "LETTER OF CREDIT NO."]);
            try_add("reference_proforma", vec!["PROFORMA INVOICE NO.", "PI NO."]);
        },
        "PI" => {
            try_add("reference_po", vec!["P/O NO.", "ORDER NO."]);
        },
        "SC" => {
            try_add("reference_proforma", vec!["PROFORMA INVOICE NO.", "PI NO."]);
            try_add("reference_po", vec!["P/O NO.", "ORDER NO."]);
        },
        "LC" => {
            try_add("reference_po", vec!["P/O NO.", "ORDER NO.", "PURCHASE ORDER NO."]);
        },
        "LLC" => {
            try_add("reference_lc", vec!["MASTER L/C NO.", "L/C NO."]);
            try_add("reference_purchase_confirm", vec!["CONFIRMATION NO.", "CP NO."]);
        },
        "CP" => {
            try_add("reference_export_decl", vec!["EXPORT DECLARATION NO.", "ED NO."]);
            try_add("reference_po", vec!["P/O NO."]);
        },
        "BE" => {
            try_add("reference_lc", vec!["L/C NO.", "DRAWN UNDER L/C NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "TR" => {
            try_add("reference_lg", vec!["GUARANTEE NO.", "L/G NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "LG" => {
            try_add("reference_bl", vec!["B/L NO.", "ORIGINAL B/L NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "EL" => {
            try_add("reference_po", vec!["P/O NO.", "ORDER NO."]);
            try_add("reference_contract", vec!["CONTRACT NO.", "S/C NO."]);
        },
        "SA" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
        },
        "DO" => {
            try_add("reference_bl", vec!["B/L NO."]);
            try_add("reference_arrival_notice", vec!["ARRIVAL NOTICE NO.", "A/N NO."]);
        },
        "AN" => {
            try_add("reference_bl", vec!["B/L NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "BC" | "BK" => {
            try_add("reference_sr", vec!["SHIPPING REQUEST NO.", "S/R NO."]);
            try_add("reference_po", vec!["P/O NO."]);
        },
        "SR" => {
            try_add("reference_booking", vec!["BOOKING NO."]);
            try_add("reference_po", vec!["P/O NO."]);
        },
        "FCR" => {
            try_add("reference_hbl", vec!["HOUSE B/L NO.", "HBL NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "POD" => {
            try_add("reference_do", vec!["DELIVERY ORDER NO.", "D/O NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
        },
        "CM" => {
            try_add("reference_bl", vec!["B/L NO."]);
            try_add("reference_export_decl", vec!["EXPORT DECLARATION NO."]);
        },
        "WR" => {
            try_add("reference_do", vec!["DELIVERY ORDER NO."]);
            try_add("reference_po", vec!["P/O NO."]);
        },
        "FI" => {
            try_add("reference_booking", vec!["BOOKING NO.", "BKG NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "ED" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_po", vec!["P/O NO."]);
            try_add("reference_lc", vec!["L/C NO."]);
        },
        "ID" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
            try_add("reference_origin", vec!["CERTIFICATE OF ORIGIN NO.", "C/O NO."]);
        },
        "CO" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
        },
        "CNM" => {
            try_add("reference_origin", vec!["CERTIFICATE OF ORIGIN NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
        },
        "CCC" => {
            try_add("reference_import_decl", vec!["IMPORT DECLARATION NO."]);
            try_add("reference_origin", vec!["CERTIFICATE OF ORIGIN NO."]);
        },
        "IC" => {
            try_add("reference_po", vec!["P/O NO.", "ORDER NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "COA" | "CA" => {
            try_add("reference_po", vec!["P/O NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_inspection", vec!["INSPECTION CERTIFICATE NO.", "IC NO."]);
        },
        "WC" => {
            try_add("reference_bl", vec!["B/L NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "PHYTO" | "PC" => {
            try_add("reference_fumigation", vec!["FUMIGATION CERTIFICATE NO.", "FC NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
        },
        "FC" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
        },
        "HC" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "BEN_CERT" => {
            try_add("reference_lc", vec!["L/C NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "CDR" => {
            try_add("reference_bl", vec!["B/L NO."]);
            try_add("reference_policy", vec!["INSURANCE POLICY NO.", "POLICY NO."]);
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "DGD" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
        },
        "MSDS" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "POA" => {
            try_add("reference_biz_license", vec!["BUSINESS LICENSE NO."]);
        },
        "BIZ_LIC" => {},
        "INS" | "IP" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
        },
        "ICF" => {
            try_add("reference_policy", vec!["INSURANCE POLICY NO.", "POLICY NO."]);
            try_add("reference_survey", vec!["SURVEY REPORT NO.", "CDR NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
        },
        "SOA" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_debit_note", vec!["DEBIT NOTE NO.", "D/N NO."]);
            try_add("reference_credit_note", vec!["CREDIT NOTE NO.", "C/N NO."]);
        },
        "DN" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
        },
        "CN" => {
            try_add("reference_invoice", vec!["INVOICE NO."]);
        },
        "TI" => {
            try_add("reference_purchase_confirm", vec!["CONFIRMATION NO.", "CP NO."]);
            try_add("reference_local_lc", vec!["LOCAL L/C NO."]);
        },
        _ => {
            // 폴백: 범용 참조 라벨
            try_add("reference_invoice", vec!["INVOICE NO."]);
            try_add("reference_bl", vec!["B/L NO."]);
            try_add("reference_po", vec!["P/O NO."]);
        },
    }

    (doc_labels, refs, self_ref)
}

pub fn get_trade_category_schema(category: &str, doc_type: &str) -> String {
    use serde_json::Value;

    // ── 코드 폴백 base : bias.json 에 trade_schema 노드가 없어도 동작해야 합니다 ──
    fn fallback_base(category: &str) -> Vec<(&'static str, &'static str)> {
        match category {
            "header" => vec![
                ("doc_type",           "Document kind code {String}"),
                ("doc_number",         "Primary identifier of THIS document (B/L No, Invoice No, PO No) {String}"),
                ("issue_date",         "Date of issue (YYYY-MM-DD) {String}"),
                ("reference_po",       "Referenced Purchase Order number printed on this document {String}"),
                ("reference_invoice",  "Referenced Commercial Invoice number printed on this document {String}"),
                ("reference_bl",       "Referenced Bill of Lading number printed on this document {String}"),
                ("reference_lc",       "Referenced Letter of Credit number printed on this document {String}"),
                ("reference_booking",  "Referenced Booking number printed on this document {String}"),
                ("reference_contract", "Referenced Sales Contract number printed on this document {String}"),
                ("reference_number",   "Any OTHER reference number printed that does not fit the fields above {String}"),
            ],
            "parties" => vec![
                ("sender_name",       "Shipper, Seller, Exporter {String}"),
                ("sender_address",    "Address of sender {String}"),
                ("recipient_name",    "Consignee, Buyer, Importer {String}"),
                ("recipient_address", "Address of recipient {String}"),
                ("notify_party_name", "Notify party name {String}"),
            ],
            "logistics" => vec![
                ("vessel",         "Vessel name or Flight number {String}"),
                ("voyage_number",  "Voyage or flight leg number {String}"),
                ("pol",            "Port of Loading / Airport of Departure {String}"),
                ("pod",            "Port of Discharge / Airport of Destination {String}"),
                ("etd",            "Estimated time of departure {String}"),
                ("eta",            "Estimated time of arrival {String}"),
                ("transport_mode", "Sea, Air, Road, Rail {String}"),
            ],
            "conditions" => vec![
                ("incoterms",            "FOB, CIF, EXW, DDP, DAP {String}"),
                ("payment_terms",        "T/T, L/C, Net30 {String}"),
                ("freight_payment_term", "Freight Prepaid or Freight Collect {String}"),
            ],
            "financials" => vec![
                ("currency",        "ISO 4217 currency code {String}"),
                ("amount",          "Grand total amount {Number}"),
                ("amount_subtotal", "Subtotal before tax and charges {Number}"),
                ("amount_tax",      "Tax or VAT amount {Number}"),
            ],
            "cargo" => vec![
                ("package_count", "Total number of packages (NOT money) {Number}"),
                ("package_unit",  "Package unit (CTN, PLT, PKG) {String}"),
                ("weight_gross",  "Total gross weight {Number}"),
                ("weight_net",    "Total net weight {Number}"),
                ("volume",        "Total volume in CBM {Number}"),
                ("marks_numbers", "Marks and numbers {String}"),
            ],
            "items" => vec![
                ("description", "Description of goods {String}"),
                ("quantity",    "Line item quantity {Number}"),
                ("unit",        "Unit of measure {String}"),
                ("hs_code",     "HS code / tariff number {String}"),
                ("unit_price",  "Unit price {Number}"),
                ("total_price", "Line total {Number}"),
            ],
            "containers" => vec![
                ("container_number", "Container number (4 letters + 7 digits) {String}"),
                ("seal_number",      "Seal number {String}"),
                ("type_size",        "Size and type (20GP, 40HC) {String}"),
            ],
            _ => vec![],
        }
    }

    // ── bias.json 에서 { field: desc } 맵을 읽습니다 ──
    fn read_node(path: &[&str]) -> Option<serde_json::Map<String, Value>> {
        let mut cur: &Value = &crate::parsing::BIAS_DICT;
        for p in path {
            cur = cur.get(*p)?;
        }
        cur.as_object().cloned()
    }

    // 1) base
    let mut fields: Vec<(String, String)> = Vec::new();
    if let Some(obj) = read_node(&["trade_schema", "base", category]) {
        for (k, v) in obj {
            fields.push((k, v.as_str().unwrap_or("{String}").to_string()));
        }
    } else {
        for (k, d) in fallback_base(category) {
            fields.push((k.to_string(), d.to_string()));
        }
    }

    // 2) overlay : 이 서식에만 존재하는 축을 덧붙입니다.
    //    같은 이름이면 overlay 설명이 이깁니다(서식별 뉘앙스가 더 정확하므로).
    if let Some(obj) = read_node(&["trade_schema", "overlay", doc_type, category]) {
        for (k, v) in obj {
            let desc = v.as_str().unwrap_or("{String}").to_string();
            if let Some(slot) = fields.iter_mut().find(|(n, _)| n == &k) {
                slot.1 = desc;
            } else {
                fields.push((k, desc));
            }
        }
    }

    // 🌟 [SELF-REFERENCE FIELD DROP] 자기 자신을 가리키는 참조 축은 이 문서에 없습니다.
    //
    //  ── 실측 사고 ──
    //   CI 인보이스의 header 스키마에 doc_number 와 reference_invoice 가 함께 제시되었고,
    //   2B 모델은 눈에 보이는 'INVOICE NUMBER' 라벨을 reference_invoice 쪽으로 복사한 뒤
    //   doc_number 를 null 로 반환했습니다.
    //     [Qwen3.5-DECODING]  ": null,number      ← doc_number
    //     [Qwen3.5-DECODING]  ",VAT/EORInce      ← reference_invoice
    //   doc_number 가 비면 문서 기본키가 내용 합성 키로 폴백되어 릴레이가 영구히 끊깁니다.
    //
    //  ── 근거 ──
    //   CI 가 자기 자신을 참조하는 일은 정의상 없습니다.
    //   서식 코드 → 참조 필드 사전은 logic.rs 가 이미 소유하므로 그대로 재사용합니다.
    //   (scheduler.rs 의 [SELF-REFERENCE DROP] 과 같은 판정을 프롬프트 생성 단계로 앞당깁니다)
    if let Some(self_ref) = crate::logic::trade_reference_field_of(doc_type) {
        fields.retain(|(n, _)| n != self_ref);
    }

    // 🌟 [IDENTITY FIRST] doc_number 를 항상 첫 필드로 올립니다.
    //   2B 모델은 [FIELD DEFINITIONS] 앞쪽 항목에 더 강하게 반응합니다.
    //   문서 기본키가 되는 축은 14개 중 아무 자리가 아니라 첫 줄에 있어야 합니다.
    if category == "header" {
        if let Some(pos) = fields.iter().position(|(n, _)| n == "doc_number") {
            let f = fields.remove(pos);
            fields.insert(0, f);
        }
    }

    if fields.is_empty() {
        return format!(
            "RULES: Output JSON ONLY. MISSION: Extract data for category '{}'.\nSCHEMA:\n{{}}",
            category.to_uppercase()
        );
    }

    // 3) 렌더링 v2 : '정의' 와 '값 자리' 를 물리적으로 분리합니다.
    //
    //  ── 구버전이 만들던 프롬프트 ──
    //     SCHEMA:
    //     {
    //       "incoterms": "FOB, CIF, EXW, DDP, DAP {String}",
    //       "package_unit": "Package unit (CTN, PLT, PKG) {String}"
    //     }
    //   값 자리에 이미 문자열이 채워져 있으므로 2B 모델은 그것을 복사합니다.
    //   실측 결과: 정답 DAP 인데 FOB, 문서에 없는데 CTN, 그리고 "{String}" 원문 그대로.
    //
    //  ── 새 프롬프트 ──
    //     [FIELD DEFINITIONS]
    //     - "incoterms" (String): FOB, CIF, EXW, DDP, DAP
    //     [FORBIDDEN VALUES]
    //     ... FOB, CIF, EXW, DDP, DAP ...
    //     SCHEMA:
    //     { "incoterms": null }
    //   값 자리는 전부 null 이라 복사할 대상이 없고,
    //   예시는 정의 블록으로 옮겨 '값이 아님' 을 명시합니다.
    //
    //  ⚠️ [CONTRACT] 아래 두 계약은 반드시 유지해야 합니다.
    //     ① get_trade_doc_categories / process_trading_task 가
    //        "SCHEMA:\n{}" 문자열로 빈 스키마를 판정합니다.
    //     ② process_trading_task 가 SCHEMA 블록에서 trim_start 후 '"' 로 시작하는
    //        라인 수로 필드 개수를 셉니다. 정의/금지 블록은 '-' 로 시작시켜 제외합니다.
    let is_array = category == "items" || category == "containers";

    let parsed: Vec<(String, String, &'static str)> = fields
        .iter()
        .map(|(k, d)| {
            let (desc, ty) = split_type_marker(d);
            (k.clone(), desc, ty)
        })
        .collect();

    let defs = parsed
        .iter()
        .map(|(k, d, t)| format!("- \"{}\" ({}): {}", k, t, d.replace('"', "'")))
        .collect::<Vec<_>>()
        .join("\n");

    let mut forbidden: Vec<String> = Vec::new();
    for (_, d, _) in parsed.iter() {
        for tok in extract_example_tokens(d) {
            if !forbidden.iter().any(|e| e == &tok) {
                forbidden.push(tok);
            }
        }
    }
    // 타입 표기 자체도 금지 목록에 넣습니다. (실측: "voyage_number": "{String}")
    for t in ["String", "Number", "Boolean", "Array"] {
        let m = format!("{{{}}}", t);
        if !forbidden.iter().any(|e| e == &m) {
            forbidden.push(m);
        }
    }

    let forbidden_block = if forbidden.is_empty() {
        String::new()
    } else {
        format!(
            "\n[FORBIDDEN VALUES]\n\
             The tokens below appear in [FIELD DEFINITIONS] only as EXAMPLES of the kind of value.\n\
             They are NOT the answer. Returning one of them without reading it off the image is a fabrication.\n\
             - {}\n\
             Return one of these ONLY when you can actually read that exact token in the image.",
            forbidden.join(", ")
        )
    };

    let body = parsed
        .iter()
        .map(|(k, _, _)| format!("  \"{}\": null", k))
        .collect::<Vec<_>>()
        .join(",\n");

    let schema = if is_array {
        // 🌟 [ARRAY SHAPE ENFORCEMENT] 원소를 두 개 제시해 '반복 구조' 를 눈으로 보여 줍니다.
        //    구버전은 원소 하나짜리 `[ { ... } ]` 만 보여 줬고,
        //    2B 모델은 표에서 한 행만 읽으면 대괄호를 벗겨 객체를 반환했습니다.
        //    (실측: T-Shirt 행만 객체로 반환 → Shorts 행 소실)
        //    원소를 두 개 세워 두면 '행마다 원소 하나' 라는 계약이 형태로 전달됩니다.
        format!("[\n  {{\n{}\n  }},\n  {{\n{}\n  }}\n]", body, body)
    } else {
        format!("{{\n{}\n}}", body)
    };

    // 🌟 [PROMPT EXAMPLE PURGE] 규칙 3번에 박혀 있던 예시 문서번호
    //    (PO-99281A, CI-2026-08001, BL-55432219, LC-88492011) 를 삭제했습니다.
    //    value_grounding.rs 의 주석이 지목한 환각
    //      ① reference_invoice = "CI-2026-08001" — 문서 어디에도 없는 값
    //    의 출처는 bias.json 뿐만이 아니라 이 프롬프트였습니다.
    //    아래 [FORBIDDEN VALUES] 블록이 '설명문의 예시' 는 막지만
    //    '규칙문의 예시' 는 forbidden 목록에 들어가지 않아 그대로 새어 나갑니다.
    //
    // 🌟 [IDENTITY RULE] doc_number 를 '먼저 읽으라' 고 규칙 0번으로 못박습니다.
    //    문서 자신의 번호는 제목 바로 아래 식별 블록에 인쇄된다는 레이아웃 사실은
    //    run_title_gate 가 상단 30% 를 제목 밴드로 보는 것과 같은 근거입니다.
    let reference_rule = if category == "header" {
        let (doc_labels, refs, self_ref) = trade_doc_identity_context(doc_type);

        // ── doc_number 라벨 목록 ──
        let doc_label_str = doc_labels.join(" / ");

        // ── 이 서식에 실제로 있는 참조 축과 그 인쇄 라벨 ──
        let mut ref_lines = String::new();
        for (field, labels) in refs.iter() {
            let label_str = labels.join(" / ");
            ref_lines.push_str(&format!(
                "   - \"{}\" — printed under label: {}\n",
                field, label_str
            ));
        }
        if ref_lines.is_empty() {
            ref_lines.push_str(
                "   - (This document type has no reference_* fields. Omit all reference_* keys.)\n",
            );
        }

        // ── 자기참조 경고 ──
        let self_ref_warning = if self_ref.is_empty() {
            String::new()
        } else {
            format!(
                "   - WARNING: \"{}\" is THIS document's own number. It is NEVER printed on this page as a reference. \
    If you see a number that looks like this document's own ID, it belongs in \"doc_number\", NOT in \"{}\".\n",
                self_ref, self_ref
            )
        };

        format!(
            "
        IDENTITY RULE (read this before anything else):
        \
        0. This page is a {doc}. \"doc_number\" is the number printed in the identity block at the TOP of the page, \
        directly under or beside the printed document title. Read that block first and fill \"doc_number\" before any other field. \
        The label for doc_number on THIS document is: {doc_labels}.
        \
        1. \"doc_number\" MUST contain the FULL number including its prefix. \
        Copy it character-for-character. Never strip the prefix, never drop hyphens, never re-type from memory.
        \
        REFERENCE RULES (only these reference fields exist on THIS document type):
        \
        {ref_lines}\
        {self_ref_warning}\
        2. Any number printed NEXT TO or BELOW a reference label belongs to the matching reference_* field, NOT to doc_number. \
        The reference labels available on this document are listed above. Use ONLY those.
        \
        3. Match the reference number to the CORRECT field by the LABEL printed above or beside it. \
        If the label is not listed above, use \"reference_number\". \
        NEVER guess a reference_* field from the number's prefix alone — use the printed LABEL.
        \
        4. Copy each reference number EXACTLY as printed, including its prefix and hyphens. Never re-type it from memory.
        \
        5. Never copy the same number into two different fields. If unsure which reference_* field fits, use \"reference_number\".
        \
        6. Omit any reference_* field that is not printed on this page. An omitted field is correct data; a guessed one corrupts the document graph.
        \
        7. A caption or heading printed on the page is a LABEL, not a value. \
        Return the value printed NEXT TO it, never the label text itself.
        \
        8. Pure digit strings with no alphabetic prefix are NOT reference numbers unless they appear \
        directly under one of the reference labels listed above. Match them to the label they are printed under.",
            doc = doc_type,
            doc_labels = doc_label_str,
            ref_lines = ref_lines,
            self_ref_warning = self_ref_warning,
        )
    } else {
        String::new()
    };

    // 🌟 [ARRAY RULE] 표는 '행이 몇 개인지' 를 모델이 스스로 세어야 합니다.
    //    실측에서 상품 표에 T-Shirt / Shorts 두 행이 있는데 한 행만 반환되었고,
    //    두 번째 행은 파이프라인 어디에서도 복구할 수 없었습니다.
    let array_rule = if is_array {
        "\n[TABLE RULES]\n\
         1. This category is a TABLE. Output one array element per printed row, in top-to-bottom order.\n\
         2. Count the rows before you write. If you see three rows, the array has three elements.\n\
         3. The two elements shown in SCHEMA are a shape example, not a row count.\n\
         4. Never merge two rows into one element. Never split one row into two.\n\
         5. If a cell in a row is blank, that element's field is null — do not borrow the value from the row above."
    } else {
        ""
    };

    format!(
        "RULES: Output JSON ONLY. Every value in SCHEMA is null on purpose — replace a null ONLY with text you can actually read in the image.\n\
         MISSION: Extract data for category '{}' of a {} document.{}{}\n\
         [FIELD DEFINITIONS]\n{}{}\n\
         SCHEMA:\n{}",
        category.to_uppercase(),
        doc_type,
        reference_rule,
        array_rule,
        defs,
        forbidden_block,
        schema
    )
}

/// 🌟 [VISION CROP PROMPT] 정밀 크롭 이미지 전용 스키마 프롬프트.
///
///  ── get_trade_category_schema 와 무엇이 다른가 ──
///   기존 프롬프트는 '전체 페이지 또는 고정 세로 슬라이스' 를 전제로 합니다.
///   그래서 모델은 화면 어딘가에 값이 있을 것이라 가정하고 탐색하며,
///   찾지 못하면 근처 숫자를 끌어와 채워 넣습니다.
///
///   이 프롬프트의 입력은 SigLIP2 코사인 히트맵이 지목한 '그 카테고리 영역만' 입니다.
///   즉 "여기 없으면 문서 어디에도 없다" 가 성립합니다.
///   그 사실을 명시해야 모델이 null 을 자신 있게 돌려줍니다.
///
///  ── 왜 null 이 중요한가 ──
///   무역 문서 그래프는 참조 번호로 연결됩니다.
///   없는 번호를 창작하면 잘못된 문서끼리 relay 되어 그래프가 통째로 오염됩니다.
///   빈 값은 되돌릴 수 있지만 잘못된 연결은 되돌릴 수 없습니다.
pub fn get_trade_crop_prompt(
    category: &str,
    doc_type: &str,
    top_field: &str,
    score: f32,
    claimed: &[(String, String)],
) -> String {
    let base = get_trade_category_schema(category, doc_type);

    let evidence = if top_field.is_empty() {
        String::new()
    } else {
        format!(
            "\n[VISION EVIDENCE]\nThe vision encoder located this region by matching the concept \"{}\" \
             (cosine surprisal {:+.4}). The value for that field is almost certainly printed inside this crop.",
            top_field, score
        )
    };

    // 🌟 [ALREADY CLAIMED] 앞선 크롭이 이미 확정한 값 목록.
    //
    //  ── 왜 필요한가 ──
    //   크롭은 카테고리별로 순차 호출되므로 뒤 크롭은 앞 크롭의 결과를 모릅니다.
    //   그래서 같은 숫자를 두 필드가 각각 자기 값으로 가져가는 사고가 납니다.
    //   (실측: financials 가 2000.00 을 잡았는데 cargo 도 근처 숫자를 끌어옴)
    //   scheduler.rs 의 커머스 추출이 [ALREADY CLAIMED VALUES] 로 같은 문제를 막는 것과
    //   동일한 원리를 비전 크롭에도 적용합니다.
    let claimed_block = if claimed.is_empty() {
        String::new()
    } else {
        let mut s = String::from(
            "\n[ALREADY CLAIMED VALUES]\n\
             Previous crops of this same document already確 locked these values to other fields.\n\
             Never return any of them for a field in this crop:\n",
        );
        for (k, v) in claimed.iter().take(24) {
            s.push_str(&format!("- \"{}\" = \"{}\"\n", k, v));
        }
        s
    };

    format!(
        "[INPUT NOTICE]\n\
         The image you receive is NOT the whole page. It is a precise crop that the vision encoder \
         identified as the '{}' region of a {} document. Everything relevant to this category is inside this crop.\n\
         \n\
         [HOW TO ANSWER]\n\
         1. Read the crop first. List in your head only the text you can actually SEE.\n\
         2. Fill a field ONLY from that seen text. If the field's value is not printed in this crop, return null.\n\
         3. Never take a value from [FIELD DEFINITIONS] or [FORBIDDEN VALUES]. Those are descriptions, not data.\n\
         4. Never take a value from a neighbouring field just because it is the only number nearby.\n\
         5. A null field is correct data. A fabricated one silently corrupts the document graph and can never be undone.{}{}\n\n{}",
        category.to_uppercase(),
        doc_type,
        evidence,
        claimed_block,
        base
    )
}

/// 🌟 [TRADE CONDITION — DEPTH 1] 질의 청크가 어느 '조건 카테고리' 인지 1갈래만 고릅니다.
///  ── 왜 쪼개는가 ──
///   기존 extract_shipping_conditions 는 44개 필드 + 변환 규칙 + 값 예시를
///   한 프롬프트에 통째로 넣고 2B 모델에게 "알아서 골라라" 라고 시켰습니다.
///   scheduler.rs STEP A 가 27개 서식 코드를 '그룹 → 코드' 2뎁스로 좁히는 것과
///   정반대 구조였고, 그래서 Cross References 3축이 44축으로 늘어나는 순간
///   프롬프트 길이만 폭증하고 정확도는 오히려 떨어집니다.
///
///  ── 호출 조건 ──
///   이 프롬프트는 '항상' 호출되지 않습니다.
///   model.rs 의 SURPRISAL 게이트가 1위-2위 마진을 확보하면 LLM 없이 확정되고,
///   사실상 동률일 때만 이 초소형 프롬프트가 1회 호출됩니다.
///
///  ── 벡터 근거 동봉 ──
///   scheduler.rs STEP A 의 [VECTOR EVIDENCE] 와 동일하게, 코사인 점수를
///   후보 목록에 함께 실어 모델이 근거 없이 창작하지 못하게 만듭니다.
pub fn trade_condition_category_prompt(
    chunk: &str,
    full_query: &str,
    scored: &[(String, f32)],
) -> String {
    let mut cands = String::new();
    for (cat, score) in scored.iter() {
        let desc = match cat.as_str() {
            "identity"  => "the document itself: its kind, its own number, its status, its issue or expiry date",
            "transport" => "how the cargo moves: vessel, flight, voyage, ports, ETD, ETA, transport mode",
            "parties"   => "who is involved: shipper, exporter, consignee, importer, notify party",
            "terms"     => "commercial terms: incoterms, payment terms, freight prepaid or collect, currency, amounts",
            "cargo"     => "the goods themselves: container, seal, packages, weight, volume, HS code, marks",
            "reference" => "a number belonging to ANOTHER document that this one refers to",
            "hub"       => "trace one number across EVERY related document, without naming which reference field it is",
            _           => "other",
        };
        cands.push_str(&format!("- \"{}\" (vector score {:+.4}) — {}\n", cat, score, desc));
    }

    let template = r###"[TASK]
Decide which SINGLE condition category the highlighted chunk belongs to.

[FULL QUERY]
{FULL_QUERY}

[CHUNK TO CLASSIFY]
{CHUNK}

[CANDIDATE CATEGORIES]
{CANDIDATES}

[RULES]
1. Choose exactly ONE category from [CANDIDATE CATEGORIES]. Never invent a category name.
2. "identity" is about THIS document. "reference" is about ANOTHER document that this one points to.
   "The invoice number is CI-2026-08001" on an invoice  -> identity
   "against invoice CI-2026-08001" on a bill of lading  -> reference
3. Choose "hub" ONLY when the query asks to trace one number across every related document
   and does NOT say which reference role it plays.
   "show me everything under PO-99281A"  -> hub
   "referenced purchase order is PO-99281A" -> reference
4. If the chunk carries no filtering meaning at all, return "".

[OUTPUT FORMAT]
{ "category": String }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;

    template
        .replace("{FULL_QUERY}", full_query)
        .replace("{CHUNK}", chunk)
        .replace("{CANDIDATES}", &cands)
}

/// 🌟 [TRADE CONDITION — DEPTH 2] 확정된 카테고리 안에서 파라미터 1개를 고릅니다.
///  ── 핵심 ──
///   후보 목록에 '승리한 카테고리의 필드' 만 들어갑니다.
///   reference 카테고리라면 44축이지만, identity 라면 6축뿐입니다.
///   모델이 볼 수 있는 선택지 자체를 결정론으로 좁혀 오답 경로를 물리적으로 없앱니다.
///
///  ── 호출 조건 ──
///   exclusive_assign_by_score 가 확정하면 호출되지 않습니다.
///   1위-2위가 사실상 동률일 때만 1회 호출됩니다.
pub fn trade_condition_field_prompt(
    chunk: &str,
    full_query: &str,
    category: &str,
    scored: &[(String, String, f32)],
) -> String {
    let mut cands = String::new();
    for (field, desc, score) in scored.iter() {
        cands.push_str(&format!("- \"{}\" (vector score {:+.4}) — {}\n", field, score, desc));
    }

    let reference_note = if category == "reference" {
        "\n[REFERENCE FIELD NOTE]\n\
         The prefix of the number itself is the strongest clue.\n\
         PO- -> reference_po / CI- -> reference_invoice / BL- -> reference_bl / LC- -> reference_lc\n\
         HBL- -> reference_hbl / SWB- -> reference_swb / AWB- -> reference_awb / BK- -> reference_booking\n\
         DO- -> reference_do / POD- -> reference_pod / CM- -> reference_manifest / FI- -> reference_freight_invoice\n\
         ED- -> reference_export_decl / ID- -> reference_import_decl / CO- -> reference_origin\n\
         IC- -> reference_inspection / WC- -> reference_weight / COA- -> reference_analysis\n\
         IP- -> reference_policy / LG- -> reference_lg / TR- -> reference_tr / CDR- -> reference_survey\n\
         If the printed prefix names a field in [CANDIDATE FIELDS], choose that field."
    } else {
        ""
    };

    let template = r###"[TASK]
Decide which SINGLE field inside the '{CATEGORY}' category the highlighted chunk maps to.

[FULL QUERY]
{FULL_QUERY}

[CHUNK TO MAP]
{CHUNK}

[CANDIDATE FIELDS]
{CANDIDATES}{REFERENCE_NOTE}

[RULES]
1. Choose exactly ONE field from [CANDIDATE FIELDS]. Never invent a field name.
2. Judge by what the chunk MEANS, not by which candidate is listed first.
3. If none of the candidates fit, return "".

[OUTPUT FORMAT]
{ "field": String }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;

    template
        .replace("{CATEGORY}", category)
        .replace("{FULL_QUERY}", full_query)
        .replace("{CHUNK}", chunk)
        .replace("{CANDIDATES}", &cands)
        .replace("{REFERENCE_NOTE}", reference_note)
}

/// 🌟 [TRADE CONDITION — DEPTH 3] 확정된 필드 하나의 값과 연산자만 뽑습니다.
///  ── 왜 값까지 LLM 에게 맡기지 않는가 ──
///   값은 '벡터가 짚어준 원문 청크' 그 자체입니다.
///   deterministic_condition_value / split_numeric_and_comparator 가
///   Rust 에서 결정론으로 뽑아내므로, 이 프롬프트는 그 결정론이 실패했을 때만
///   호출되는 최후 보루입니다.
///
///  ── 연산자 고정 ──
///   trade_default_operator 가 필드 형식으로 기본 연산자를 확정합니다.
///   모델은 그것을 '바꿀 근거가 있을 때만' 바꿉니다.
pub fn trade_condition_value_prompt(
    chunk: &str,
    full_query: &str,
    field: &str,
    field_desc: &str,
    default_operator: &str,
) -> String {
    let template = r###"[TASK]
Extract the filter VALUE for the field '{FIELD}' from the highlighted chunk.

[FULL QUERY]
{FULL_QUERY}

[CHUNK]
{CHUNK}

[FIELD DEFINITION]
"{FIELD}": {FIELD_DESC}

[DEFAULT OPERATOR]
"{DEFAULT_OP}"

[RULES]
1. "value" MUST be an exact literal substring of [CHUNK]. Never translate, reformat, round, or re-type it.
2. Copy document numbers EXACTLY as printed, including prefix and hyphens. Never strip "PO-", "CI-", "BL-", "LC-".
3. Keep the [DEFAULT OPERATOR] unless the chunk explicitly demands a comparison.
   Only these are allowed: "eq", "gt", "gte", "lt", "lte", "contains".
   "over 5000"  -> "gt"    "under 5000" -> "lt"
   "at least"   -> "gte"   "up to"      -> "lte"
   "after 2026-08-01" -> "gte"   "before 2026-09-01" -> "lte"
4. Strip the operator words from "value". "over 5000 USD" -> value "5000", operator "gt".
5. If [CHUNK] holds no usable value for this field, return null for both keys.
   null is correct data; an invented value corrupts the filter.

[OUTPUT FORMAT]
{ "operator": String, "value": String }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;

    template
        .replace("{FIELD}", field)
        .replace("{FIELD_DESC}", field_desc)
        .replace("{FULL_QUERY}", full_query)
        .replace("{CHUNK}", chunk)
        .replace("{DEFAULT_OP}", default_operator)
}

/// 🌟 [FALLBACK ONLY] 이 프롬프트는 더 이상 정상 경로가 아닙니다.
///  ── 언제 호출되는가 ──
///   parse_shipping_query 의 Depth 1 SURPRISAL 게이트를 통과한 청크가
///   단 하나도 없을 때(= 벡터 근거가 전무할 때) 최후 1회만 호출됩니다.
///
///  ── 왜 축소했는가 ──
///   v2 는 44개 필드를 전부 나열했습니다. 그러면 프롬프트 길이만 폭증하고
///   2B 모델이 "어느 칸에든 하나는 채워야 한다" 고 오인해 근거 없는 조건을 창작합니다.
///   정상 경로는 trade_condition_category_prompt → trade_condition_field_prompt →
///   trade_condition_value_prompt 3단으로 이미 좁혀지므로,
///   여기서는 카테고리 대표 축만 남겨 '아무것도 못 뽑는 상황' 을 면하는 역할만 합니다.
///
///  ⚠️ 여기서 뽑힌 조건은 전부 Dexie(executeDexiePlan)가 data.* 경로로 실행합니다.
///     LanceDB 는 봉투 스코프(mode/type/cc)만 담당하므로,
///     이 목록에 필드를 추가해도 Rust 스키마나 SQL 을 고칠 필요가 전혀 없습니다.
pub fn extract_shipping_conditions(query: &str, language: &str) -> String {
    let template = r###"Task: Act as a deterministic shipping and trade logistics semantic parser.
The vector engine found NO reliable evidence in this query, so extract only what is unmistakably printed.

[SCHEMA DEFINITION — representative axes only]
# Document Identity
- "doc_type": Document kind code.
- "doc_number": Primary identifier OF THE DOCUMENT ITSELF.
- "no": Tracking number or generic reference number.
- "status": Document / shipping status.
- "issue_date": Date the document was issued.

# Transport
- "vessel": Vessel name or Flight number.
- "pol": Port of Loading.
- "pod": Port of Discharge.
- "etd": Estimated Time of Departure.
- "eta": Estimated Time of Arrival.

# Parties
- "sender_name": Shipper, Seller, Exporter.
- "recipient_name": Consignee, Buyer, Importer.

# Commercial Terms
- "incoterms": Incoterms code.
- "currency": ISO 4217 currency code.
- "amount": Total financial amount.

# Cargo
- "container_number": Container number.
- "weight_gross": Gross weight.
- "hs_code": HS Code.

# Hub Reference
- "hub_reference": A document number to trace ACROSS every related document, when the query does NOT say which reference role it plays.

[TRANSFORMATION LOGIC]
For EVERY extracted field, wrap it in an operator object:
{ "operator": "eq" | "gt" | "lt" | "gte" | "lte" | "contains", "value": <extracted_value> }
- Use "eq" for strict identifiers: doc_number, container_number, hs_code, no, status, doc_type, incoterms, currency.
- Use "contains" for free text and for hub_reference.
- Use "gte" / "lte" for date ranges and numeric ranges.
- Copy document numbers EXACTLY as printed, including prefix and hyphens.
- Omit any field that is NOT explicitly present in the query. Returning an empty object is correct
  when nothing is printed; an invented condition silently destroys recall.

[QUERY]
{QUERY}

[OUTPUT FORMAT]
{ "<property_name>": { "operator": "...", "value": "..." } }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{QUERY}", query).replace("{LANGUAGE}", language)
}

pub fn get_image_extraction_prompt(region: &str, language: &str, page_type: &str, address: &str) -> String {
    if page_type == "tracking" {
        // 🌟 [SCHEMA ECHO 방어] 값 자리의 "string" 은 그대로 복사될 위험이 있습니다.
        //    실측에서 무역 경로가 "{String}" 을 그대로 뱉은 것과 같은 구조입니다.
        //    값 자리는 null 로 비우고, 타입은 정의 블록으로 옮깁니다.
        let template = r###"[TASK]
Read this shipping label image and fill the structured JSON format.

[CONTEXT]
Region: {REGION}
Recipient Address: {ADDRESS}
Current Language: {LANGUAGE}

[FIELD DEFINITIONS]
- "tracking_number" (String): the tracking / waybill number printed on the label
- "recipient_match" (Boolean): true only when the address on the label matches [CONTEXT] Recipient Address
- "barcodes" (Array of String): every barcode value you can read on the label

[HOW TO ANSWER]
1. Every value in [OUTPUT FORMAT] is null on purpose. Replace a null ONLY with text read off the image.
2. Never answer with a type name such as "string", "number", "boolean", or "{String}".
3. Copy digits EXACTLY as printed. Never reformat or insert separators.
4. If a value is not printed on the label, return null. An empty array is correct when no barcode is readable.

[OUTPUT FORMAT]
{ "tracking_number": null, "recipient_match": null, "barcodes": [] }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;
        template.replace("{REGION}", region).replace("{ADDRESS}", address).replace("{LANGUAGE}", language)
    } else if page_type == "goods" {
        // 🌟 [COMMERCE GOODS] 기존에는 tracking 과 같은 템플릿을 공유해
        //    상품 이미지에서도 tracking_number / barcodes 만 물었습니다.
        //    상품명·가격·색상 같은 실제 커머스 축이 통째로 빠져 있었습니다.
        let template = r###"[TASK]
Read this product image and fill the structured JSON format.

[CONTEXT]
Region: {REGION}
Current Language: {LANGUAGE}

[FIELD DEFINITIONS]
- "title" (String): product name as printed
- "code" (String): product code / SKU as printed
- "price" (Number): digits only
- "currency" (String): ISO 4217 code or the printed symbol
- "color" (String): colour name as printed
- "brand_name" (String): brand as printed
- "barcodes" (Array of String): every readable barcode value

[HOW TO ANSWER]
1. Every value in [OUTPUT FORMAT] is null on purpose. Replace a null ONLY with text read off the image.
2. Never answer with a type name such as "String", "Number", or "{String}".
3. Copy every value EXACTLY as printed. Never translate or reformat.
4. "price" holds digits only. Put the currency symbol or code in "currency".
5. If a field is not visible, return null. A null field is correct data; a guessed one is corrupted data.

[OUTPUT FORMAT]
{ "title": null, "code": null, "price": null, "currency": null, "color": null, "brand_name": null, "barcodes": [] }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;
        template.replace("{REGION}", region).replace("{LANGUAGE}", language)
    } else {
        String::new()
    }
}

/// 🌟 [COMMERCE CROP PROMPT] 커머스 정밀 크롭 이미지 전용 프롬프트.
///
///  ── 커머스에도 비전 크롭이 필요한 이유 ──
///   상품 상세 캡처나 주문서 스크린샷은 한 장에
///   상품명 / 가격 / 옵션 / 배송지 / 결제수단이 전부 들어 있습니다.
///   전체를 통째로 물으면 2B 모델이 가격과 배송비를 섞고,
///   상품명 자리에 카테고리 배지를 넣습니다.
///   무역과 동일하게 필드 앵커 히트맵으로 영역을 잘라 하나씩 묻습니다.
///
///  ── 필드 목록의 출처 ──
///   bias_schema::get_detail_schema_fields(page_type) 가 돌려주는
///   필드명과 설명을 그대로 씁니다. 여기서 새로 나열하지 않습니다.
pub fn get_commerce_crop_prompt(
    page_type: &str,
    fields: &[(String, String)],
    language: &str,
    top_field: &str,
    score: f32,
    claimed: &[(String, String)],
) -> String {
    // 🌟 [SCHEMA ECHO 방어] 무역 크롭과 동일하게 '정의' 와 '값 자리' 를 분리합니다.
    //    구버전은 `  "price": Number {Number}` 처럼 값 자리에 타입 표기를 노출해
    //    모델이 그것을 그대로 복사할 수 있었습니다.
    let mut defs = String::new();
    let mut body = String::new();
    let mut forbidden: Vec<String> = Vec::new();

    for (name, desc) in fields.iter() {
        let (clean, ty) = split_type_marker(desc);
        defs.push_str(&format!("- \"{}\" ({}): {}\n", name, ty, clean.replace('"', "'")));
        if !body.is_empty() {
            body.push_str(",\n");
        }
        body.push_str(&format!("  \"{}\": null", name));
        for tok in extract_example_tokens(&clean) {
            if !forbidden.iter().any(|e| e == &tok) {
                forbidden.push(tok);
            }
        }
    }
    if body.is_empty() {
        defs.push_str("- \"title\" (String): Product title as printed\n");
        body.push_str("  \"title\": null");
    }
    for t in ["String", "Number", "Boolean", "Array"] {
        let m = format!("{{{}}}", t);
        if !forbidden.iter().any(|e| e == &m) {
            forbidden.push(m);
        }
    }

    let forbidden_block = if forbidden.is_empty() {
        String::new()
    } else {
        format!(
            "\n[FORBIDDEN VALUES]\n\
             These tokens appear in [FIELD DEFINITIONS] only as EXAMPLES. They are not the answer.\n\
             - {}\n",
            forbidden.join(", ")
        )
    };

    let claimed_block = if claimed.is_empty() {
        String::new()
    } else {
        let mut s = String::from(
            "\n[ALREADY CLAIMED VALUES]\n\
             Previous crops of this same screen already locked these values to other fields.\n\
             Never return any of them for a field in this crop:\n",
        );
        for (k, v) in claimed.iter().take(24) {
            s.push_str(&format!("- \"{}\" = \"{}\"\n", k, v));
        }
        s
    };

    let evidence = if top_field.is_empty() {
        String::new()
    } else {
        format!(
            "\n[VISION EVIDENCE]\nThe vision encoder located this region by matching the concept \"{}\" \
             (cosine surprisal {:+.4}).",
            top_field, score
        )
    };

    let template = r###"[INPUT NOTICE]
The image you receive is NOT the whole screen. It is a precise crop that the vision encoder
identified as holding the fields listed below. Everything relevant is inside this crop.{EVIDENCE}

[CONTEXT]
Page Type: {TYPE}
Output Language: {LANGUAGE}

[HOW TO ANSWER]
1. Read the crop first. Use only the text you can actually SEE.
2. Every value in [SCHEMA] is null on purpose. Replace a null ONLY with text read off the image.
3. Never take a value from [FIELD DEFINITIONS] or [FORBIDDEN VALUES]. Those are descriptions, not data.
4. Copy every value EXACTLY as printed. Never translate, round, or reformat.
5. Numeric fields hold digits only. Strip currency symbols and thousand separators.
6. Return null for anything not printed in this crop. A null field is correct data.

[FIELD DEFINITIONS]
{DEFS}{FORBIDDEN}{CLAIMED}
[SCHEMA]
{
{BODY}
}

[ACTION] JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

    template
        .replace("{EVIDENCE}", &evidence)
        .replace("{TYPE}", page_type)
        .replace("{LANGUAGE}", language)
        .replace("{DEFS}", &defs)
        .replace("{FORBIDDEN}", &forbidden_block)
        .replace("{CLAIMED}", &claimed_block)
        .replace("{BODY}", &body)
}

pub fn extract_table_structure_prompt(page_type: &str, item_selector: &str, pug_content: &str, reference_row: &str) -> String {
    let template = r###"[PUG CONTENT]
{PUG_CONTENT}

[Reference: Row Structure]
{REFERENCE_ROW}

[Instruction]
Locate the main table wrapper, its body container, and its corresponding header container within the [PUG CONTENT].

[Rules]
1. Tag Agnostic: Do NOT assume traditional <table> tags. The structure could be built using <div>, <ul>/<li>, or other semantic tags. Analyze logically.
2. Fill out the `table` selector FIRST to logically establish the common parent wrapper that encompasses both the header (thead) and the items (tbody).
3. The `tbody` selector is exactly "{ITEM_SELECTOR}". Return it as provided.
4. Provide the final exact CSS selector for the `thead` based on your analysis within that table wrapper.

[OUTPUT FORMAT]
{ "{TYPE}": { "tbody": { "selector": "{ITEM_SELECTOR}" }, "table": { "selector": "..." }, "thead": { "selector": "..." } } }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{TYPE}", page_type)
            .replace("{ITEM_SELECTOR}", item_selector)
            .replace("{PUG_CONTENT}", pug_content)
            .replace("{REFERENCE_ROW}", reference_row)
}

pub fn analytic_report_prompt() -> String {
    r###"[TASK]
You are a User Behavior Analysis Expert. Interpret raw HTML interactions to understand the user's specific intent and analyze the selection context within a list or a group of items.

Analyze the parallel arrays of 'Clicked HTML' (the selected element) and 'Related HTML' (the surrounding structure).
If 'Previous Analysis' is provided, use it to infer the user's behavioral flow and connect past actions with the current click.

[ANALYSIS GUIDELINES & CHAIN OF THOUGHT]
Fill out the JSON keys in the exact order specified below. Use 'analysis_*' keys to logically establish the context before finalizing the outputs.

1. analysis_target: Identify the primary entity name and its key attributes from the Clicked HTML.
2. analysis_surroundings: Identify the neighboring items or alternatives displayed in the Related HTML that were NOT selected.
3. action: Determine the specific user intent for clicking the item. Must explicitly include the primary entity name and key attributes. Output as a short verb phrase.
4. relate: Summarize the surrounding unselected items to capture the context of the choice. Do not summarize the clicked item itself in this field.
5. summary: Provide a detailed explanation of what the user aimed to accomplish on this page. Must explicitly reference the extracted primary entity and its key attributes.

[OUTPUT FORMAT]
{
    "actions": {
        "https://hostname.com/pathname?search=parameter": {
            "records": [
                {
                    "id": "...",
                    "analysis_target": "...",
                    "analysis_surroundings": "...",
                    "relate": [...],
                    "action": "..."
                }
            ],
            "summary": "..."
        }
    },
    "cross_action_flow": "...",
    "intent_evolution": "...",
    "consistent_preferences": "..."
}

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###.to_string()
}

// 🌟 [ANALYTIC SEMANTIC] 원시 outerHTML 을 PUG 로 변환하고 속성을 전부 제거한 뒤,
//    '태그 구조 + 화면 텍스트' 만 남은 상태에서 Qwen3.5 2B 가 의미를 요약합니다.
//    ── 왜 속성을 지우는가 ──
//     class="btn_prd_option_2 on" / id="cnt_capa_1" 같은 값은 사이트마다 제각각이라
//     LLM 이 그것을 '의미' 로 오인해 존재하지 않는 속성을 지어냅니다.
//     속성을 제거하면 남는 신호가 (a / li / td / 텍스트) 뿐이므로
//     모델은 화면에 실제로 인쇄된 문자열만 근거로 삼게 됩니다.
pub fn analytic_semantic_prompt(
    event_type: &str,
    link: &str,
    lang: &str,
    target_pug: &str,
    related_pug: &str,
) -> String {
    let template = r###"[TASK]
You are a User Behavior Analysis Expert. The raw HTML has already been converted into an ATTRIBUTE-FREE PUG tree, so ONLY the semantic tag structure and the visible text remain. Interpret that structure and describe what the user did.

[CONTEXT]
Event Type: {EVENT_TYPE}
Page: {LINK}
Output Language: {LANG}

[TARGET ELEMENT — the element the user actually interacted with]
{TARGET_PUG}

[SURROUNDING ELEMENTS — siblings shown next to it that the user did NOT choose]
{RELATED_PUG}

[RULES]
1. "action" MUST name the concrete entity in [TARGET ELEMENT] (product title, option value, menu label, typed value, price) EXACTLY as it is printed there. Never invent a name that is not printed.
2. "relate" describes only the NEIGHBOURING items in [SURROUNDING ELEMENTS]. Never describe the target itself here. Return an empty array when there is no sibling.
3. "summary" is ONE sentence explaining what the user was trying to accomplish on this page, and MUST reuse the same entity name used in "action".
4. Copy every proper noun, product name, code, price and number EXACTLY as printed. Do not translate, round, or reformat them.
5. If [TARGET ELEMENT] carries no readable text at all, return null for every key. A null answer is correct data; an invented one is corrupted data.

[OUTPUT FORMAT]
{ "action": String, "relate": [String], "summary": String }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

    template
        .replace("{EVENT_TYPE}", event_type)
        .replace("{LINK}", link)
        .replace("{LANG}", lang)
        .replace("{TARGET_PUG}", target_pug)
        .replace("{RELATED_PUG}", related_pug)
}

// 🌟 [ANALYTIC FLOW] 같은 사용자·같은 페이지에서 구조화된 여러 행동을 묶어
//    흐름(cross_action_flow) / 의도 변화(intent_evolution) / 반복 성향(consistent_preferences)을
//    한 번에 합성합니다. analytics-logis-center 의 Cron 산출물과 동일한 3축입니다.
pub fn analytic_flow_prompt(lang: &str, records_json: &str) -> String {
    let template = r###"[TASK]
You are a User Behavior Analysis Expert. Below is a time-ordered list of ALREADY STRUCTURED user actions taken by ONE user. Synthesize them into a behavioural narrative.

[CONTEXT]
Output Language: {LANG}

[STRUCTURED ACTION RECORDS]
{RECORDS}

[RULES]
1. Use ONLY the facts present in [STRUCTURED ACTION RECORDS]. Never invent a page, product or option that is not listed.
2. Keep every proper noun, product name and number EXACTLY as printed in the records.
3. "cross_action_flow": describe the overall path in order (what was viewed, what was compared, what was chosen).
4. "intent_evolution": describe how the goal shifted from the first action to the last one.
5. "consistent_preferences": describe the attributes the user repeatedly gravitated toward. Return an empty string when nothing repeats.

[OUTPUT FORMAT]
{ "cross_action_flow": String, "intent_evolution": String, "consistent_preferences": String }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

    template
        .replace("{LANG}", lang)
        .replace("{RECORDS}", records_json)
}

// 🌟 [ANALYTIC QUERY PARSER] parse_commerce_query 의 para2graph + extract_time_intent_prompt
//    구조를 분석(Analytic) 도메인에 맞춰 하나로 압축한 질의 파서입니다.
//    ── 왜 별도 파서인가 ──
//     commerce 는 sale_price / tracking_number 처럼 '컬럼 조건' 을 뽑아야 하지만,
//     analytic 의 저장 축은 action / summary / relate 세 개의 자유 서술 문장뿐입니다.
//     따라서 수치 조건 추출은 불필요하고, 실제로 필요한 것은
//       ① 기간(time_intent / season_intent)
//       ② 이벤트 종류(click / hover / change / report)
//       ③ 의미 키워드
//     세 가지입니다. 기간은 LLM 값을 그대로 쓰지 않고 Rust 가 다시 epoch 로 확정합니다.
pub fn analytic_query_prompt(query: &str, time_context: &str, lang: &str) -> String {
    let mut time_keys: Vec<String> = crate::parsing::BIAS_DICT
        .get("time_filters")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_else(|| vec![
            "today".to_string(), "yesterday".to_string(),
            "this_month".to_string(), "last_month".to_string(),
            "this_year".to_string(), "last_year".to_string(),
            "recently".to_string()
        ]);
    time_keys.push("".to_string());
    let time_arr_str = serde_json::to_string(&time_keys).unwrap_or_else(|_| "[]".to_string());

    let mut season_keys: Vec<String> = crate::parsing::BIAS_DICT
        .get("season_filters")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_else(|| vec![
            "spring".to_string(), "summer".to_string(),
            "autumn".to_string(), "winter".to_string()
        ]);
    season_keys.push("".to_string());
    let season_arr_str = serde_json::to_string(&season_keys).unwrap_or_else(|_| "[]".to_string());

    let template = r###"[TASK]
Act as a deterministic semantic parser for a USER BEHAVIOUR LOG search engine.
Split the natural language query into a period, an event kind, and the semantic keywords.

[SYSTEM TIME & LOCALE CONTEXT]
{TIME_CONTEXT}

[AVAILABLE TIME INTENTS]
{TIME_ARRAY}

[AVAILABLE SEASON INTENTS]
{SEASON_ARRAY}

[AVAILABLE EVENT TYPES]
- "click"  : the user pressed / selected something
- "hover"  : the user lingered on something without pressing it
- "change" : the user typed a value, picked a select option, toggled a checkbox
- "report" : a synthesized behavioural report over several actions
- ""       : the query does not restrict the event kind

[QUERY]
{QUERY}

[RULES]
1. "time_intent" MUST be chosen from [AVAILABLE TIME INTENTS]. Return "" when the query contains NO explicit temporal word. Never guess a period from context.
2. "season_intent" MUST be chosen from [AVAILABLE SEASON INTENTS]. Return "" when no season word is printed. A clothing name is NOT a season word.
3. "event_types" is an array chosen from [AVAILABLE EVENT TYPES]. Return an empty array when the query does not restrict the kind.
4. "keywords" holds the meaning-bearing chunks of the query in the ORIGINAL language, with every temporal word removed. Never include verbs such as "show me", "find", "tell me".
5. "target" is one short sentence, in {LANG}, restating what behaviour the user wants to see. It is used as the semantic search sentence.
6. "original_text" is the query copied character for character.

[OUTPUT FORMAT]
{ "original_text": String, "time_intent": String, "season_intent": String, "event_types": [String], "keywords": [String], "target": String }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template
        .replace("{TIME_CONTEXT}", time_context)
        .replace("{TIME_ARRAY}", &time_arr_str)
        .replace("{SEASON_ARRAY}", &season_arr_str)
        .replace("{QUERY}", query)
        .replace("{LANG}", lang)
}

// 🌟 [ANALYTIC REPORT] 벡터 검색으로 회수한 '구조화된 행동 기록' 목록을 근거로
//    사용자 질의에 답하는 리포트를 작성합니다. 결과는 JSON 이 아니라 마크다운 본문입니다.
pub fn analytic_report_answer_prompt(
    query: &str,
    time_context: &str,
    scope: &str,
    records_json: &str,
    lang: &str,
) -> String {
    let template = r###"[TASK]
You are a User Behavior Analyst. Answer the user's question using ONLY the retrieved behaviour records below.

[SYSTEM TIME & LOCALE CONTEXT]
{TIME_CONTEXT}

[SEARCH SCOPE]
{SCOPE}

[USER QUESTION]
{QUERY}

[RETRIEVED BEHAVIOUR RECORDS]
{RECORDS}

[RULES]
1. Every sentence you write MUST be supported by a record above. If the records do not answer the question, say so plainly instead of inventing an answer.
2. Copy product names, option values, prices and numbers EXACTLY as printed in the records.
3. Represent users as User A, User B, User C ... Never print the raw address / hash of a user.
4. Write in {LANG}.
5. Structure the answer as:
   - one short headline sentence that directly answers the question
   - a bullet list of the concrete supporting actions (what, where, when)
   - one closing sentence on the pattern or the recommended follow-up
6. Do NOT output JSON. Output plain readable text (markdown bullets are allowed).
7. FORBIDDEN OUTPUT SHAPES: your reply MUST NOT start with '{' or '['. It MUST NOT contain
   any key/value pair such as "headline": or "supporting_actions": or "closing":.
   It MUST NOT be wrapped in a code fence.
8. The FIRST character of your reply MUST be a normal word character of {LANG}.

[EXAMPLE OF A CORRECT REPLY SHAPE]
<one headline sentence>
- <supporting action 1>
- <supporting action 2>
<one closing sentence>

[ACTION] WRITE THE REPORT ONLY. NO PREAMBLE. NO CODE FENCE. NO JSON. /no_think"###;

    template
        .replace("{TIME_CONTEXT}", time_context)
        .replace("{SCOPE}", scope)
        .replace("{QUERY}", query)
        .replace("{RECORDS}", records_json)
        .replace("{LANG}", lang)
}

pub fn is_detail_prompt(page_type: &str, title: &str, lang: &str) -> String {
    let (list_hints, form_hints) = crate::parsing::get_layout_prompt_hints(page_type, lang);

    let template = r###"[TASK]
Analyze the provided PUG/HTML content from top to bottom.

[ENTITY CONTEXT: {TYPE}]
Language Context: {LANGUAGE}
You are evaluating a page managing this specific domain entity. Use this context to conceptually understand the abstract structures:
- has_form: A property configuration interface. It features a large overarching form dedicated to inputting or updating the specific attributes of ONE primary entity.{FORM_HINTS}
- has_list: A catalog or inventory interface dedicated to displaying, filtering, or batch-processing multiple DIFFERENT primary entities.{LIST_HINTS}

[FORCED DOCUMENT SCANNING LOGIC]
Read the entire document from top to bottom, applying the following strict filters and evaluations:

1. IGNORE:
   - Strictly ignore global navigation, menus, headers, footers, aside, search, filter.
2. TARGET:
   - Focus purely on the main data payload where "{TYPE}", or actual items are listed.
3. EVALUATE:
   - You MUST evaluate the concluding elements at the very bottom of the main content area first. Check for the following:
     A. Does the page terminate with dataset navigation (pagination, "next/prev") or bulk-action execution elements?
     B. Does the main data area consist of a repeating multi-entity grid?
     C. Does the main data area contain an extensive configuration/input form (inputs, textareas, image uploads, save buttons) for a single entity?

[SCHEMA DEFINITIONS]
- {TYPE}:
    - has_header: Boolean. True if the document contains a header.
    - title: String. Default '{TITLE}'.
    - has_footer: Boolean. True if the document contains a footer.
    - language: String. Default '{LANGUAGE}'.
    - has_list: Boolean. True if the document contains a multi-entity grid, OR if the bottom of main content area has dataset navigation/bulk controls.
    - has_form: Boolean. True if the main data payload is heavily composed of data entry fields (text, select, radio, file uploads) dedicated to creating or updating a single entity.

[OUTPUT FORMAT]
{ "{TYPE}": { "has_header": Boolean, "title": String, "has_footer": Boolean, "language": String, "has_list": Boolean, "has_form": Boolean } }

JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;
    
    template.replace("{TYPE}", page_type)
        .replace("{TITLE}", title)
        .replace("{LANGUAGE}", lang)
        .replace("{FORM_HINTS}", &form_hints)
        .replace("{LIST_HINTS}", &list_hints)
}

pub fn para2graph(language: &str) -> String {
    let template = r###"Translate and convert the given natural language search query into English, then segment it into the specified JSON dataset structure.

[DOCUMENT SCANNING & STRICT SEGMENTATION LOGIC]
1. EXACT COPY: Copy the full original input into 'original_text' without changing anything.
2. TRANSLATE & TAGGED PIPE PLANNING: Translate the query into English. In the 'segmented_plan' field, prefix every translated segment with its assigned type tag in brackets, separated by pipes ('|'). Structure it strictly as '[tag1] english chunk1 | [tag2] english chunk2'.
3. MAXIMAL GROUPING: Group all contiguous words belonging to the same type into a SINGLE English segment. DO NOT split subjects from their numeric conditions. Break the segment ONLY when the type logically shifts.
4. STRICT ARRAY MAPPING: For EVERY tagged English segment in 'segmented_plan', create exactly one object in the 'context' array sequentially.

[SCHEMA DEFINITIONS]
- original_text: String. The exact, unaltered full natural language input.
- segmented_plan: String. Translated English text with '[type] english text | ' format inserted strictly at type boundaries.
- context:
  - 'text': String. The translated English chunk.
  - 'language': String. Default '{LANG}'.
  - 'type': String. Choose one:
    * 'order': Intent to measure sales performance or direct transactions. Triggers: conversion rate, sales volume, checkout, payment, cancellation, refund. (RULE: If the context measures buying success or revenue, classify as 'order' even if the word 'product' or 'item' is present).
    * 'goods': Intent to describe product catalog data, exposure, or traffic metrics. Triggers: page views, clicks, physical attributes, stock limits, unit prices. (RULE: Focuses on item specifications and customer traffic before the actual purchase).
    * 'tracking': Intent to manage logistics and fulfillment. Triggers: shipment status, dispatch, delivery duration, courier information.
    * 'review': Intent to analyze the voice of the customer. Triggers: feedback, ratings, reviews, CS messages, complaints.
    * 'coupon': Intent to manage specific discount vouchers. Triggers: coupon codes, issuance limits, discount amounts applied via coupons.
    * 'event': Intent to manage marketing campaigns or analyze broad operational trends. Triggers: promotions, exhibitions, seasonal sales, overarching managerial analysis requests.
    * '': If none logically apply.

[OUTPUT FORMAT]
{ "original_text": "String", "segmented_plan": "String", "context": [...] }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;
    template.replace("{LANG}", language)
}

pub fn extract_time_intent_prompt(text: &str, time_context: &str, first_choice: &str, first_score: f32, alternatives: &[(String, f32)]) -> String {
    let mut time_keys: Vec<String> = crate::parsing::BIAS_DICT
        .get("time_filters")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_else(|| vec!["today".to_string(), "yesterday".to_string(), "this_month".to_string(), "last_month".to_string(), "this_year".to_string(), "last_year".to_string(), "recently".to_string()]);
    
    // 시간 의도가 아닐 경우를 대비해 빈 문자열 선택지를 추가합니다.
    time_keys.push("".to_string());

    let time_arr_str = serde_json::to_string(&time_keys).unwrap_or_else(|_| "[]".to_string());

    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for (prop, score) in alternatives.iter() {
        cands_str.push_str(&format!("- \"{}\" (Vector Score: {:.4})\n", prop, score));
    }

    let template = r###"[TASK]
Analyze the given text and extract the exact relative time intent based on the Current Time Context.
You MUST strictly choose ONLY from the provided array. If none logically apply, return "". Do not invent any other values.

[SYSTEM TIME & LOCALE CONTEXT]
{TIME_CONTEXT}

[AVAILABLE TIME INTENTS]
{TIME_ARRAY}

[TEXT TO ANALYZE]
Text: "{TEXT}"

[CANDIDATE INTENTS]
{CANDIDATES}

[INSTRUCTIONS]
1. STRICT RULE: You MUST return "" (empty string) if the Text DOES NOT explicitly contain temporal words. Do NOT guess or infer time based on context.
2. Evaluate all [CANDIDATE INTENTS] equally. If one of them matches the explicit text perfectly, return it.
3. If none of the candidates match, but the text explicitly mentions time, choose the best fit from [AVAILABLE TIME INTENTS]. Otherwise, return "".

[OUTPUT FORMAT]
{ "time_intent": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{TIME_CONTEXT}", time_context)
            .replace("{TIME_ARRAY}", &time_arr_str)
            .replace("{TEXT}", text)
            .replace("{CANDIDATES}", &cands_str)
}

pub fn extract_season_intent_prompt(text: &str, first_choice: &str, first_score: f32, alternatives: &[(String, f32)]) -> String {
    let mut season_keys: Vec<String> = crate::parsing::BIAS_DICT
        .get("season_filters")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_else(|| vec!["spring".to_string(), "summer".to_string(), "autumn".to_string(), "winter".to_string()]);
    
    // 계절 의도가 아닐 경우를 대비해 빈 문자열 선택지를 추가합니다.
    season_keys.push("".to_string());

    let season_arr_str = serde_json::to_string(&season_keys).unwrap_or_else(|_| "[]".to_string());

    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for (prop, score) in alternatives.iter() {
        cands_str.push_str(&format!("- \"{}\" (Vector Score: {:.4})\n", prop, score));
    }

    let template = r###"[TASK]
Analyze the given text and extract the exact seasonal intent.
You MUST strictly choose ONLY from the provided array. If none logically apply, return "". Do not invent any other values.

[AVAILABLE SEASON INTENTS]
{SEASON_ARRAY}

[TEXT TO ANALYZE]
Text: "{TEXT}"

[CANDIDATE INTENTS]
{CANDIDATES}

[INSTRUCTIONS]
1. STRICT RULE: You MUST return "" (empty string) if the Text DOES NOT explicitly contain season-related words. Do NOT guess the season just because it's a specific clothing or item name.
2. Evaluate all [CANDIDATE INTENTS] equally. If one of them matches the explicit text perfectly, return it.
3. If none of the candidates match, but the text explicitly mentions a season, choose the best fit from [AVAILABLE SEASON INTENTS]. Otherwise, return "".

[OUTPUT FORMAT]
{ "season_intent": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{SEASON_ARRAY}", &season_arr_str)
            .replace("{TEXT}", text)
            .replace("{CANDIDATES}", &cands_str)
}

pub fn extract_numeric_conditions(current: &str, seg_type: &str, metrics_json: &str, vector_guide: &str, time_context: &str, lang: &str, value_type: &str) -> String {
    let (deterministic_time, _) = crate::parsing::get_deterministic_time_guide(vector_guide, lang);
    
    let final_time_context = if !deterministic_time.is_empty() {
        format!("{}\n{}", time_context, deterministic_time)
    } else {
        time_context.to_string()
    };

    let template = r###"[Task]
Act as a deterministic semantic parser.
You must extract, transform, and normalize numeric and property conditions from the natural language input into the strictly defined JSON output format.

[SYSTEM TIME & LOCALE CONTEXT]
{TIME_CONTEXT}
- If explicit exact dates are mentioned in the text, use them.

[DATABASE METRICS CONTEXT]
Metrics: {METRICS}
- CRITICAL RULES FOR FUZZY ADJECTIVES (Translate from any language):
  * If the query implies "many", "often", "popular", "best", or "high" without a specific number, you MUST IGNORE the Vector Guide's operator and map the operator to 'top' and set percent_total to 20.
  * If the query implies "few", "rarely", "unpopular", "worst", or "low" without a specific number, you MUST IGNORE the Vector Guide's operator and map the operator to 'bottom' and set percent_total to 20.
  * You MUST use the Metrics data to calculate the exact absolute threshold for these percentiles.

[VECTOR MATCHING GUIDE (HINT)]
The system has pre-calculated vector similarities for properties, operators, and metric types, including 1st choices and Alternatives.
- If the 1st choice operator/metric makes semantic sense, use it.
- If the 1st choice is wrong, consider the Alternatives provided.
- Metric Type gives a crucial hint about the data (date, time, price, discount, quantity, ratio). 
- If Metric Type is 'ratio', extract a percentage logic. If 'date', extract a date logic, etc.
- TEMPORAL & SEASON CORRECTION RULES:
  1. Vectors often hallucinate seasons. If the text explicitly contains a season word, IGNORE the Vector Guide's Season Intent and select the exact season yourself from the [LOCALE CALENDAR REFERENCE].
  2. If a Season is detected, check the Time Intent (or explicit time text):
     - If Time Intent implies the past, map the season to the PREVIOUS year's dates.
     - If Time Intent implies the present, map the season to the CURRENT year's dates.
     - Output BOTH 'started_at' (gte) and 'expired_at' (lte) to form a date range.
{GUIDE}

[SCHEMA DEFINITION]
Extract the following numeric/property conditions if semantically present in the text:
condition:
  - property: String.
  - is_percent: Boolean.
  - operator: String. 'gt' | 'gte' | 'lt' | 'lte' | 'eq' | 'contains' | 'top' | 'bottom'
  - percent_total: Number.
  - value: {VALUE_TYPE}.

[CURRENT CHUNK TO ANALYZE]
{CURRENT}

[OUTPUT FORMAT]
{ "condition": condition }

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{CURRENT}", current)
            .replace("{TYPE}", seg_type)
            .replace("{METRICS}", metrics_json)
            .replace("{GUIDE}", vector_guide)
            .replace("{TIME_CONTEXT}", &final_time_context)
            .replace("{VALUE_TYPE}", value_type)
}
pub fn extract_status_intent_prompt(current_text: &str, seg_type: &str, first_choice: &str, first_score: f32, alternatives: &[(String, f32)]) -> String {
    let status_options = match seg_type {
        "tracking" => r#"* 'draft': Shipment preparation or pending pickup.
* 'progress': Currently in transit or out for delivery.
* 'return': Returning to sender.
* 'complete': Successfully delivered to the recipient.
* '': If none logically apply."#,

        "goods" => r#"* 'draft': Product is being created, not yet published.
* 'show': Visible and available for sale on storefront.
* 'hide': Hidden from the storefront.
* 'progress': Currently being restocked or updated.
* 'stop': Sales temporarily suspended.
* 'cancel': Product discontinued or cancelled.
* 'refund': Related to refunded inventory.
* 'return': Related to returned inventory.
* 'exchange': Related to exchanged inventory.
* 'expire': Product expired.
* 'complete': Completely sold out or finished lifecycle.
* '': If none logically apply."#,

        "order" => r#"* 'draft': Pending payment or in cart.
* 'progress': Order processing or preparing for shipment.
* 'stop': Order on hold.
* 'cancel': Order cancelled before fulfillment.
* 'refund': Payment refunded.
* 'return': Items returned by customer.
* 'exchange': Items being exchanged.
* 'expire': Payment window expired.
* 'complete': Order fully fulfilled and closed.
* '': If none logically apply."#,

        "coupon" | "event" => r#"* 'show': Visible to customers.
* 'progress': Currently active and running.
* 'hide': Hidden from customers.
* 'stop': Temporarily paused.
* 'cancel': Terminated early.
* 'expire': Passed its expiration date.
* 'complete': Successfully finished its run.
* '': If none logically apply."#,

        "review" => r#"* 'progress': Under moderation or pending approval.
* 'stop': Blocked or suspended review.
* 'cancel': Deleted or withdrawn by user.
* 'refund': Associated with a refunded order.
* 'return': Associated with a returned order.
* 'exchange': Associated with an exchanged order.
* 'expire': Review period expired.
* 'complete': Published and visible.
* '': If none logically apply."#,

        _ => r#"* 'show': Visible state.
* 'progress': Active/Processing state.
* 'remove': Deleted state.
* 'hide': Hidden state.
* 'stop': Paused/Stopped state.
* 'cancel': Cancelled state.
* 'refund': Refunded state.
* 'return': Returned state.
* 'exchange': Exchanged state.
* 'expire': Expired state.
* 'complete': Finished/Completed state.
* '': If none logically apply."#,
    };

    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for (prop, score) in alternatives.iter() {
        cands_str.push_str(&format!("- \"{}\" (Vector Score: {:.4})\n", prop, score));
    }

    let template = r###"[TASK]
Analyze the given text and extract the exact semantic intent for status.
You MUST strictly choose ONLY from the provided array. If none logically apply, return "". Do not invent any other values.

[SCHEMA DEFINITIONS]
- status: String. Choose one:
{STATUS_OPTIONS}

[TEXT TO ANALYZE]
Text: "{TEXT}"

[CANDIDATE INTENTS]
{CANDIDATES}

[INSTRUCTIONS]
1. Evaluate all [CANDIDATE INTENTS] equally. If one of them is semantically correct for this text, return it.
2. If none of the candidates match, but the text explicitly dictates a status state, choose a valid intent from the [SCHEMA DEFINITIONS] array.
3. Otherwise, return "".

[OUTPUT FORMAT]
{ "status": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template
        .replace("{STATUS_OPTIONS}", status_options)
        .replace("{TEXT}", current_text)
        .replace("{CANDIDATES}", &cands_str)
}

// 🌟 [추가] scheduler.rs 전용 3개 인자 레거시 함수
pub fn extract_status_intent_legacy_prompt(current_text: &str, seg_type: &str, vector_guide: &str) -> String {
    let status_options = match seg_type {
        "tracking" => r#"* 'draft': Shipment preparation or pending pickup.
* 'progress': Currently in transit or out for delivery.
* 'return': Returning to sender.
* 'complete': Successfully delivered to the recipient.
* '': If none logically apply."#,

        "goods" => r#"* 'draft': Product is being created, not yet published.
* 'show': Visible and available for sale on storefront.
* 'hide': Hidden from the storefront.
* 'progress': Currently being restocked or updated.
* 'stop': Sales temporarily suspended.
* 'cancel': Product discontinued or cancelled.
* 'refund': Related to refunded inventory.
* 'return': Related to returned inventory.
* 'exchange': Related to exchanged inventory.
* 'expire': Product expired.
* 'complete': Completely sold out or finished lifecycle.
* '': If none logically apply."#,

        "order" => r#"* 'draft': Pending payment or in cart.
* 'progress': Order processing or preparing for shipment.
* 'stop': Order on hold.
* 'cancel': Order cancelled before fulfillment.
* 'refund': Payment refunded.
* 'return': Items returned by customer.
* 'exchange': Items being exchanged.
* 'expire': Payment window expired.
* 'complete': Order fully fulfilled and closed.
* '': If none logically apply."#,

        "coupon" | "event" => r#"* 'show': Visible to customers.
* 'progress': Currently active and running.
* 'hide': Hidden from customers.
* 'stop': Temporarily paused.
* 'cancel': Terminated early.
* 'expire': Passed its expiration date.
* 'complete': Successfully finished its run.
* '': If none logically apply."#,

        "review" => r#"* 'progress': Under moderation or pending approval.
* 'stop': Blocked or suspended review.
* 'cancel': Deleted or withdrawn by user.
* 'refund': Associated with a refunded order.
* 'return': Associated with a returned order.
* 'exchange': Associated with an exchanged order.
* 'expire': Review period expired.
* 'complete': Published and visible.
* '': If none logically apply."#,

        _ => r#"* 'show': Visible state.
* 'progress': Active/Processing state.
* 'remove': Deleted state.
* 'hide': Hidden state.
* 'stop': Paused/Stopped state.
* 'cancel': Cancelled state.
* 'refund': Refunded state.
* 'return': Returned state.
* 'exchange': Exchanged state.
* 'expire': Expired state.
* 'complete': Finished/Completed state.
* '': If none logically apply."#,
    };

    let template = r###"[TASK]
Analyze the given text and extract the exact semantic intent for status.
You MUST strictly choose ONLY from the provided array and use the Vector Matching Guide. Do not invent any other values.

[VECTOR MATCHING GUIDE]
{VECTOR_GUIDE}

[TEXT TO ANALYZE]
{TEXT}

[SCHEMA DEFINITIONS]
- status: String. Choose one:
{STATUS_OPTIONS}
  * '': If none logically apply.

[OUTPUT FORMAT]
{ "status": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template
        .replace("{STATUS_OPTIONS}", status_options)
        .replace("{TEXT}", current_text)
        .replace("{VECTOR_GUIDE}", vector_guide)
}

pub fn extract_substantial_intent_prompt(current_text: &str, first_choice: &str, first_score: f32, alternatives: &[(String, f32)]) -> String {
    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for (prop, score) in alternatives.iter() {
        cands_str.push_str(&format!("- \"{}\" (Vector Score: {:.4})\n", prop, score));
    }

    let template = r###"[TASK]
Analyze the given text and extract the exact semantic intent for substantial.
You MUST strictly choose ONLY from the provided array. If none logically apply, return "". Do not invent any other values.

[SCHEMA DEFINITIONS]
- substantial: String. Choose one:
  * 'size': Physical dimensions or volume.
  * 'weight': Mass or heaviness.
  * 'shipping_fee': Cost of delivery.
  * 'shipping_duration': Time taken for delivery.
  * 'sale_price': Final selling price to the customer.
  * 'supply_price': Wholesale or original cost.
  * 'low_stock_threshold': Minimum inventory alert level.
  * 'discount': Amount or percentage of price reduction.
  * 'min_order_amount': Minimum spend required to trigger an action.
  * 'max_discount_amount': Maximum cap for a discount.
  * 'usage_limit': Maximum number of times usable globally.
  * 'usage_per': Maximum number of times usable per user.
  * '': If none logically apply.

[TEXT TO ANALYZE]
Text: "{TEXT}"

[CANDIDATE INTENTS]
{CANDIDATES}

[INSTRUCTIONS]
1. Evaluate all [CANDIDATE INTENTS] equally. If one of them is semantically correct for this text, return it.
2. If none of the candidates match, but the text explicitly dictates a substantial state, choose a valid intent from the [SCHEMA DEFINITIONS] array.
3. Otherwise, return "".

[OUTPUT FORMAT]
{ "substantial": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{TEXT}", current_text)
            .replace("{CANDIDATES}", &cands_str)
}

pub fn extract_find_intent_prompt(current_text: &str, first_choice: &str, first_score: f32, alternatives: &[(String, f32)]) -> String {
    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for (prop, score) in alternatives.iter() {
        cands_str.push_str(&format!("- \"{}\" (Vector Score: {:.4})\n", prop, score));
    }

    let template = r###"[TASK]
Analyze the given text and extract the exact semantic intent for find.
You MUST strictly choose ONLY from the provided array. If none logically apply, return "". Do not invent any other values.

[SCHEMA DEFINITIONS]
- find: String. Choose one:
  * 'many': High quantity, count, or volume.
  * 'few': Low quantity, count, or volume.
  * 'much': High financial value, price, or amount.
  * 'little': Low financial value, price, or amount.
  * 'heavy': High physical weight.
  * 'light': Low physical weight.
  * '': If none logically apply.

[TEXT TO ANALYZE]
Text: "{TEXT}"

[CANDIDATE INTENTS]
{CANDIDATES}

[INSTRUCTIONS]
1. Evaluate all [CANDIDATE INTENTS] equally. If one of them is semantically correct for this text, return it.
2. If none of the candidates match, but the text explicitly dictates a find state, choose a valid intent from the [SCHEMA DEFINITIONS] array.
3. Otherwise, return "".

[OUTPUT FORMAT]
{ "find": "String" }

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{TEXT}", current_text)
            .replace("{CANDIDATES}", &cands_str)
}



pub fn extract_single_field_prompt(page_type: &str, field_name: &str, field_desc: &str, language: &str, metadata: &str, target_data: &str) -> String {
    let mut dynamic_output_keys = String::new();
    for key in field_name.split(',') {
        dynamic_output_keys.push_str(&format!("  \"{}\": <literal substring of [TARGET DATA], or null>,\n", key.trim()));
    }
    let dynamic_output_keys = dynamic_output_keys.trim_end_matches(",\n");

    // 🌟 기존 7번 SHAPE RULES 전체는 detect_field_format / value_matches_format 의
    //    사전 형식 게이트와 사후 [FORMAT REJECT] 검증이 코드로 강제하므로 프롬프트에서 제거했습니다.
    let template = r###"[TASK]
Copy ONE property set from [TARGET DATA]. You are a copier, not a writer.

[CONTEXT]
Page Type: {TYPE} / Output Language: {LANGUAGE}
Column labels (LABELS, never answers): {METADATA}

[SCHEMA DEFINITIONS]
{FIELDS}

[TARGET DATA]
{TARGET_DATA}

[RULES]
1. The answer MUST be an exact literal substring of [TARGET DATA]. Never translate, reformat, round, or re-type it.
2. Never answer with a column label, a format placeholder ("yyyy-MM-ddThh:mm:ss", "string", "...", "N/A"), or a value listed under [ALREADY CLAIMED VALUES].
3. NEVER answer with an HTML/PUG tag name.
4. If [VECTOR MATCH RESULT], [LINK CANDIDATES] or [DATE CANDIDATES] is given, the answer MUST come from it.
5. If nothing in [TARGET DATA] fits the schema, return null. null is correct data; a wrong value is corrupted data.

[OUTPUT FORMAT]
{ {DYNAMIC_KEYS} }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

    template.replace("{TYPE}", page_type)
            .replace("{LANGUAGE}", language)
            .replace("{METADATA}", metadata)
            .replace("{FIELDS}", field_desc)
            .replace("{TARGET_DATA}", target_data)
            .replace("{DYNAMIC_KEYS}", &dynamic_output_keys)
}

// 🌟 [NEW] insight / summary / analysis 계열 합성 필드 전용 프롬프트.
// 기존 extract_single_field_prompt 는 "리터럴 복사" 지시라, 합성 필드에 쓰면
// LLM 이 어쩔 수 없이 셀 하나("89000", "본사")를 그대로 뱉게 됩니다.
// 또한 원문 언어(doc_lang)를 명시적으로 전달하여, 한국어 문서에서도
// 고유명사/코드/숫자는 원문 그대로 보존하고 문장만 영어로 합성하도록 강제합니다.
pub fn extract_synthesis_field_prompt(page_type: &str, field_name: &str, field_desc: &str, doc_lang: &str, source_data: &str) -> String {
    let mut dynamic_output_keys = String::new();
    for key in field_name.split(',') {
        dynamic_output_keys.push_str(&format!("  \"{}\": <one sentence written by you, or null>,\n", key.trim()));
    }
    let dynamic_output_keys = dynamic_output_keys.trim_end_matches(",\n");

    let template = r###"[TASK]
Write ONE analytic sentence for the requested field. This field is a SUMMARY you compose, not a value you copy.

[CONTEXT]
Page Type: {TYPE}
Source Document Language: {DOC_LANG}

[SCHEMA DEFINITIONS]
{FIELDS}

[SOURCE DATA]
{SOURCE_DATA}

[WRITING RULES]
1. Read ALL of [SOURCE DATA] before writing. NEVER answer with a single cell value such as a bare number, a status word, a person name, a branch name, or a column label.
2. The sentence MUST combine at least two different facts taken from [SOURCE DATA].
3. Do NOT invent facts. Only restate and connect what is present in [SOURCE DATA].
4. Keep every proper noun, product name, code, identifier, and number EXACTLY as written in [SOURCE DATA]. Never translate, transliterate, or reformat them, whatever the source language is.
5. Write the connecting sentence in English, while keeping the copied literals in their original script.
6. If [SOURCE DATA] has no usable content for this field, return null. A null summary is correct; a fabricated one is corrupted data.

[OUTPUT FORMAT]
{ {DYNAMIC_KEYS} }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

    template.replace("{TYPE}", page_type)
            .replace("{DOC_LANG}", doc_lang)
            .replace("{FIELDS}", field_desc)
            .replace("{SOURCE_DATA}", source_data)
            .replace("{DYNAMIC_KEYS}", &dynamic_output_keys)
}

// 🌟 [NEW] 상태(status) 컨트롤의 CSS selector 를 찾는 전용 프롬프트.
// 코사인으로 '옵션 집합이 생애주기 상태인지'를 판정했는데 마진이 부족해 애매할 때만 호출됩니다.
// 값을 뱉게 하지 않고 '선택자'만 뱉게 하여, 실제 값은 우리가 selected 옵션에서 결정론적으로 읽습니다.
pub fn extract_status_selector_prompt(page_type: &str, lang: &str, candidates_json: &str) -> String {
    let template = r###"[TASK]
Pick the ONE form control that holds the CURRENT LIFECYCLE STATE of this single {TYPE} record.

[CONTEXT]
Page Type: {TYPE}
Document Language: {LANG}

[CANDIDATES]
Every entry below is a real <select> element found in this document.
- "selector": its exact CSS selector
- "role"    : the words attached to the control (its name / id / row header)
- "options" : every option label inside it
{CANDIDATES}

[RULES]
1. The correct control lists MUTUALLY EXCLUSIVE LIFECYCLE STATES of ONE record.
   Examples of lifecycle states: pending, preparing, in transit, delivered, completed,
   cancelled, returned, exchanged, refunded, expired, on hold.
2. It is NOT a control that lists organizations or catalogue values:
   couriers, delivery companies, banks, account numbers, card issuers, payment gateways,
   categories, brands, countries, quantities, dates, or addresses.
3. Judge by the OPTION SET, not by the control name. A control whose options are company
   names is never the state control, however its name reads.
4. "selector" MUST be copied character for character from [CANDIDATES]. Never invent one.
5. If no candidate lists lifecycle states, return null. null is correct data; a wrong
   selector silently corrupts every record.

[OUTPUT FORMAT]
{ "status_selector": <one selector copied verbatim from [CANDIDATES], or null> }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

    template.replace("{TYPE}", page_type)
            .replace("{LANG}", lang)
            .replace("{CANDIDATES}", candidates_json)
}

pub fn verify_property_mapping_prompt(text: &str, property: &str) -> String {
    let template = r###"[TASK]
Given the text '{TEXT}' and the current property '{PROPERTY}', suggest the most accurate property name(s) from the schema.
If the current property is already correct, just return it in the array.

[OUTPUT FORMAT]
{"suggested_properties": ["String", "String"]}

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{TEXT}", text)
            .replace("{PROPERTY}", property)
}

pub fn verify_property_with_alternatives_prompt(
    text: &str,
    first_choice: &str,
    first_score: f32,
    alternatives: &[(String, f32)],
) -> String {
    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for (prop, score) in alternatives.iter() {
        cands_str.push_str(&format!("- \"{}\" (Vector Score: {:.4})\n", prop, score));
    }

    let template = r###"[TASK]
Text: "{TEXT}"

[CANDIDATE PROPERTIES]
{CANDIDATES}

Instructions:
1. Evaluate all [CANDIDATE PROPERTIES] equally based on the text.
2. Return the best-fitting property as "suggested_property".
3. If none of the candidates are correct, suggest a completely different property from the schema.

[OUTPUT FORMAT]
{ "suggested_property": String }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{TEXT}", text)
            .replace("{CANDIDATES}", &cands_str)
}

pub fn verify_operator_mapping_prompt(text: &str, property: &str, operator: &str) -> String {
    let valid_ops: Vec<String> = crate::parsing::BIAS_DICT
        .get("operators")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_else(|| vec![
            "eq".to_string(), "neq".to_string(), "gt".to_string(), "gte".to_string(), 
            "lt".to_string(), "lte".to_string(), "contains".to_string(), 
            "not_contains".to_string(), "top".to_string(), "bottom".to_string()
        ]);
    
    let valid_ops_str = valid_ops.join(", ");

    let template = r###"[TASK]
Given the text "{TEXT}", the property '{PROPERTY}' currently has the operator '{OPERATOR}'.
Suggest the most correct operator based on the context of the text.
If the current operator is already correct, just return it.
Valid operators: {VALID_OPS}

[OUTPUT FORMAT]
{ "suggested_operator": String }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{TEXT}", text)
            .replace("{PROPERTY}", property)
            .replace("{OPERATOR}", operator)
            .replace("{VALID_OPS}", &valid_ops_str)
}

pub fn verify_category_with_alternatives_prompt(
    text: &str,
    first_choice: &str,
    first_score: f32,
    alternatives: &[String],
) -> String {
    let mut cands_str = format!("- \"{}\" (Vector Score: {:.4})\n", first_choice, first_score);
    for prop in alternatives.iter() {
        if prop != first_choice {
            cands_str.push_str(&format!("- \"{}\"\n", prop));
        }
    }

    let template = r###"[TASK]
Text: "{TEXT}"

[CANDIDATE CATEGORIES]
{CANDIDATES}

Instructions:
1. Evaluate all [CANDIDATE CATEGORIES] equally based on the text.
2. Choose the category that best matches the text context.
3. Return the best-fitting category as "suggested_category".

[OUTPUT FORMAT]
{ "suggested_category": String }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{TEXT}", text)
            .replace("{CANDIDATES}", &cands_str)
}

pub fn transliteration_prompt(source_value: &str, target_language: &str) -> String {
    // 🌟 [SPECIAL CHAR PRE-STRIPPED] 호출부(build_transliteration_prompt)에서
    //    이미 특수문자가 공백으로 치환된 source_value 가 전달됩니다.
    //    여기서도 방어적으로 한 번 더 공백 정규화를 수행합니다.
    let cleaned = source_value
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    // 🌟 [WORD-KEY JSON] 각 단어를 독립 키로 갖는 객체 구조를 생성합니다.
    //    LLM 이 단어 단위로 독립 음차하므로 중간 맥락 끊김이 발생하지 않습니다.
    // 🌟 [FULL SOURCE FIRST] 최초 첫 번째 키로 특수문자 제거된 전체 SOURCE 를 배치하여
    //    LLM 이 전체 문맥을 먼저 파악한 뒤 단어별 음차를 수행하도록 합니다.
    let mut word_keys: Vec<String> = Vec::with_capacity(words.len() + 1);
    // word_keys.push(format!("\"{}\": String", cleaned));
    for w in &words {
        word_keys.push(format!("\"{}\": String", w));
    }
    let transliteration_obj = format!("{{ {} }}", word_keys.join(", "));
    let template = r###"[TASK]
You are a sound-based respelling engine.
write how that word sounds in the [TARGET LANGUAGE] writing system.

[TARGET LANGUAGE]
{TARGET_LANGUAGE}

[RULES]
- Digits inside a word must be copied exactly as they appear.

[OUTPUT FORMAT]
{ "language": "{TARGET_LANGUAGE}", "transcription": { "{SOURCE}" : String }, "transliteration": {TRANSLITERATION_OBJ} }

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;
    template.replace("{SOURCE}", &cleaned)
        .replace("{TARGET_LANGUAGE}", target_language)
        .replace("{TRANSLITERATION_OBJ}", &transliteration_obj)
}