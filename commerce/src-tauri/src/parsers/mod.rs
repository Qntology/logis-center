// 🌟 문서 파일(엑셀, 워드, 한글, CSV 등) 텍스트 추출 헬퍼
pub fn extract_document_text(file_path: &str) -> anyhow::Result<String> {
    use std::path::Path;
    let path = Path::new(file_path);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    // 표 데이터를 (헤더명: 값) 형태의 Key-Value 문맥으로 변환하는 지능형 헬퍼 함수
    fn parse_table_to_kv_string(mut grid: Vec<Vec<String>>) -> String {
        if grid.is_empty() { return String::new(); }
        let rows = grid.len();
        let cols = grid.iter().map(|r| r.len()).max().unwrap_or(0);
        
        // 그리드 규격 정규화
        for r in &mut grid { r.resize(cols, String::new()); }

        // 1. 데이터 패턴 분석을 통한 헤더/본문 경계선(Boundary) 동적 유추를 먼저 수행합니다.
        let mut data_start_row = 1;
        let mut data_start_col = 0; // [개선] 기본적으로 0번 열을 데이터 영역으로 강제하여 유실 방지

        // 1-1. 가로 헤더 경계 (어디서부터 본문 행이 시작되는가?)
        let re_date = regex::Regex::new(r"^\d{2,4}[-/.]\d{1,2}[-/.]\d{1,2}").unwrap();
        for r in 0..rows {
            let mut data_like_count = 0;
            for c in 0..cols {
                let val = grid[r][c].trim();
                if !val.is_empty() {
                    let is_numeric = val.replace(",", "").parse::<f64>().is_ok();
                    let is_date = re_date.is_match(val);
                    
                    if is_numeric || is_date {
                        data_like_count += 1;
                    }
                }
            }
            
            if r > 0 && data_like_count > 0 {
                data_start_row = r;
                break;
            }
        }

        // 테이블 전체가 문자열이라서 본문을 찾지 못했다면 2번째 줄(1)을 본문으로 지정
        if data_start_row >= rows {
            data_start_row = if rows > 1 { 1 } else { 0 };
        }

        // 1-2. 세로 헤더 경계 유추 로직 안정화
        let mut col0_empty_header = false;
        for r in 0..data_start_row {
            if grid[r][0].trim().is_empty() {
                col0_empty_header = true;
            }
        }
        
        if col0_empty_header {
            let mut col0_text_only = true;
            for r in data_start_row..rows {
                let val = grid[r][0].trim();
                if !val.is_empty() && (val.replace(",", "").parse::<f64>().is_ok() || re_date.is_match(val)) {
                    col0_text_only = false;
                    break;
                }
            }
            if col0_text_only && rows > data_start_row {
                data_start_col = 1;
            }
        }

        // 🌟 [CRITICAL FIX] 본문 데이터가 오염되지 않도록, 셀 병합 해제(빈칸 채우기)는 오직 '헤더 영역'에만 적용합니다.
        // colspan 대응 (가로 방향 확산)
        for r in 0..data_start_row {
            for c in 1..cols {
                if grid[r][c].trim().is_empty() { 
                    let left_val = grid[r][c-1].clone();
                    if !left_val.trim().is_empty() { grid[r][c] = left_val; }
                }
            }
        }
        
        // rowspan 대응 (세로 방향 확산)
        for c in 0..cols {
            for r in 1..data_start_row {
                if grid[r][c].trim().is_empty() { 
                    let top_val = grid[r-1][c].clone();
                    if !top_val.trim().is_empty() { grid[r][c] = top_val; }
                }
            }
        }

        // 테이블 전체가 문자열이라서 본문을 찾지 못했다면 2번째 줄(1)을 본문으로 지정
        if data_start_row >= rows {
            data_start_row = if rows > 1 { 1 } else { 0 };
        }

        // 3-2. 세로 헤더 경계 유추 로직 안정화
        let mut col0_empty_header = false;
        for r in 0..data_start_row {
            if grid[r][0].trim().is_empty() {
                col0_empty_header = true;
            }
        }
        
        // [개선] 가로 헤더의 0번 열이 의도적으로 비워져 있고, 본문 0번 열이 순수 텍스트 카테고리일 때만 세로 헤더로 취급
        if col0_empty_header {
            let mut col0_text_only = true;
            for r in data_start_row..rows {
                let val = grid[r][0].trim();
                if !val.is_empty() && (val.replace(",", "").parse::<f64>().is_ok() || re_date.is_match(val)) {
                    col0_text_only = false;
                    break;
                }
            }
            if col0_text_only && rows > data_start_row {
                data_start_col = 1;
            }
        }

        let mut parsed_result = String::new();
        
        // 4. 경계선을 바탕으로 확정된 영역 안에서만 헤더 수집 및 데이터 매핑
        for r in data_start_row..rows {
            let mut row_has_data = false;
            let mut row_result = String::new();
            
            for c in data_start_col..cols {
                let val = grid[r][c].trim();
                if !val.is_empty() {
                    // 현재 셀 기준 확정된 '가로 헤더 영역(0 ~ data_start_row)'의 명칭만 수집
                    let mut col_headers = Vec::new();
                    for hr in 0..data_start_row {
                        let h_val = grid[hr][c].trim();
                        if !h_val.is_empty() && !col_headers.contains(&h_val) && h_val != val {
                            col_headers.push(h_val);
                        }
                    }
                    
                    // 현재 셀 기준 확정된 '세로 헤더 영역(0 ~ data_start_col)'의 명칭만 수집
                    let mut row_headers = Vec::new();
                    for hc in 0..data_start_col {
                        let h_val = grid[r][hc].trim();
                        if !h_val.is_empty() && !row_headers.contains(&h_val) && h_val != val {
                            row_headers.push(h_val);
                        }
                    }
                    
                    // 가로 세로 헤더 조합
                    let mut combined_headers = col_headers;
                    for rh in row_headers {
                        if !combined_headers.contains(&rh) {
                            combined_headers.push(rh);
                        }
                    }
                    
                    let clean_val = val.replace("\"", "'").replace("\n", " ");
                    if combined_headers.is_empty() {
                        row_result.push_str(&format!("\"Column_{}\": \"{}\", ", c, clean_val));
                    } else {
                        let clean_header = combined_headers.join(" - ").replace("\"", "'");
                        row_result.push_str(&format!("\"{}\": \"{}\", ", clean_header, clean_val));
                    }
                    row_has_data = true;
                }
            }
            if row_has_data {
                let clean_row = row_result.trim_end_matches(", ");
                parsed_result.push_str(&format!("{{ {} }}\n", clean_row));
            }
        }
        
        // 1행/1열 구조이거나 모든 매칭이 실패한 경우 단순 결합으로 Fallback
        if parsed_result.trim().is_empty() {
            let mut fallback = String::new();
            for r in grid {
                let mut row_json = String::new();
                for (i, s) in r.into_iter().enumerate() {
                    let val = s.trim();
                    if !val.is_empty() {
                        row_json.push_str(&format!("\"Column_{}\": \"{}\", ", i, val.replace("\"", "'").replace("\n", " ")));
                    }
                }
                if !row_json.is_empty() {
                    let clean_row = row_json.trim_end_matches(", ");
                    fallback.push_str(&format!("{{ {} }}\n", clean_row));
                }
            }
            return fallback;
        }
        parsed_result
    }

    match ext.as_str() {
        "csv" => {
            let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(path).map_err(|e| anyhow::anyhow!("CSV error: {:?}", e))?;
            let mut result = String::new();

            let mut grid: Vec<Vec<String>> = Vec::new();
            for result_row in rdr.records() {
                if let Ok(record) = result_row {
                    let row_str: Vec<String> = record.iter().map(|s| s.to_string()).collect();
                    grid.push(row_str);
                }
            }
            if !grid.is_empty() {
                let csv_parsed = parse_table_to_kv_string(grid);
                result.push_str(&csv_parsed);
            }
            
            Ok(result)
        },
        "xlsx" | "xls" | "xlsm" | "xlsb" => {
            use calamine::{Reader, open_workbook_auto, Data};
            let mut excel = open_workbook_auto(path).map_err(|e| anyhow::anyhow!("Excel error: {:?}", e))?;
            let mut result = String::new();

            let sheets = excel.sheet_names().to_owned();
            for sheet in sheets {
                if let Ok(range) = excel.worksheet_range(&sheet) {
                    let mut grid: Vec<Vec<String>> = Vec::new();
                    for row in range.rows() {
                        let row_str: Vec<String> = row.iter().map(|cell| {
                            match cell {
                                Data::String(s) => s.to_string(),
                                Data::Float(f) => f.to_string(),
                                Data::Int(i) => i.to_string(),
                                Data::Bool(b) => b.to_string(),
                                _ => "".to_string(),
                            }
                        }).collect();
                        grid.push(row_str);
                    }
                    if !grid.is_empty() {
                        let sheet_parsed = parse_table_to_kv_string(grid);
                        result.push_str(&sheet_parsed);
                        result.push('\n');
                    }
                }
            }
            Ok(result)
        },
        "docx" | "docs" | "doc" | "hwpx" => {
            // DOCX와 HWPX는 모두 내부가 XML로 이루어진 ZIP 압축 파일입니다.
            let file = std::fs::File::open(path)?;
            let mut archive = zip::ZipArchive::new(file).map_err(|e| anyhow::anyhow!("Zip error: {:?}", e))?;
            let mut result = String::new();

            for i in 0..archive.len() {
                let mut file = archive.by_index(i).unwrap();
                let name = file.name().to_string();
                
                // docx: word/document.xml, hwpx: Contents/section*.xml
                let is_docx = ext == "docx" || ext == "docs" || ext == "doc";
                if (is_docx && name == "word/document.xml") || 
                   (ext == "hwpx" && name.starts_with("Contents/section") && name.ends_with(".xml")) {
                    let mut xml_content = String::new();
                    use std::io::Read;
                    if file.read_to_string(&mut xml_content).is_ok() {
                        let mut reader = quick_xml::Reader::from_str(&xml_content);
                        // reader.trim_text(true); // Removed for quick-xml 0.41.0 compatibility

                        let mut current_grid: Vec<Vec<String>> = Vec::new();
                        let mut current_row: Vec<String> = Vec::new();
                        let mut current_cell = String::new();

                        let mut in_tbl = false;
                        let mut in_tr = false;
                        let mut in_tc = false;
                        let mut in_t = false;

                        loop {
                            match reader.read_event() {
                                Ok(quick_xml::events::Event::Start(ref e)) => {
                                    let name = e.name();
                                    let tag_name = name.as_ref();
                                    if tag_name == b"w:tbl" || tag_name == b"hp:tbl" {
                                        in_tbl = true;
                                        current_grid = Vec::new();
                                    } else if tag_name == b"w:tr" || tag_name == b"hp:tr" {
                                        in_tr = true;
                                        current_row = Vec::new();
                                    } else if tag_name == b"w:tc" || tag_name == b"hp:tc" {
                                        in_tc = true;
                                        current_cell = String::new();
                                    } else if tag_name == b"w:t" || tag_name == b"hp:t" {
                                        in_t = true;
                                    }
                                },
                                Ok(quick_xml::events::Event::Text(e)) => {
                                    if in_t {
                                        // quick-xml API 버전에 의존하지 않고 안전하게 문자열 디코딩 및 특수기호 언이스케이프 처리
                                        let raw_str = String::from_utf8_lossy(&e);
                                        let text = raw_str.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&quot;", "\"").replace("&apos;", "'");
                                        
                                        if in_tc {
                                            current_cell.push_str(&text);
                                        } else {
                                            let trimmed = text.trim();
                                            if !trimmed.is_empty() {
                                                result.push_str(trimmed);
                                                result.push(' ');
                                            }
                                        }
                                    }
                                },
                                Ok(quick_xml::events::Event::End(ref e)) => {
                                    let name = e.name();
                                    let tag_name = name.as_ref();
                                    if tag_name == b"w:tbl" || tag_name == b"hp:tbl" {
                                        in_tbl = false;
                                        if !current_grid.is_empty() {
                                            // 🌟 테이블을 만나면 엑셀과 동일한 Key-Value 생성기를 태워 완벽한 독립 데이터 블록으로 변환
                                            let parsed_table = parse_table_to_kv_string(current_grid.clone());
                                            result.push_str("\n\n");
                                            result.push_str(&parsed_table);
                                            result.push_str("\n\n");
                                        }
                                    } else if tag_name == b"w:tr" || tag_name == b"hp:tr" {
                                        in_tr = false;
                                        if !current_row.is_empty() {
                                            current_grid.push(current_row.clone());
                                        }
                                    } else if tag_name == b"w:tc" || tag_name == b"hp:tc" {
                                        in_tc = false;
                                        current_row.push(current_cell.trim().to_string());
                                    } else if tag_name == b"w:t" || tag_name == b"hp:t" {
                                        in_t = false;
                                    } else if tag_name == b"w:p" || tag_name == b"hp:p" {
                                        if in_tc {
                                            current_cell.push(' '); // 셀 내부의 줄바꿈을 공백으로 치환하여 보존
                                        } else {
                                            result.push('\n'); // 테이블 바깥 본문의 줄바꿈 보존
                                        }
                                    }
                                },
                                Ok(quick_xml::events::Event::Eof) => break,
                                Err(_) => break,
                                _ => (),
                            }
                        }
                    }
                }
            }
            Ok(result)
        },
        "hwp" => {
            // 구형 HWP 바이너리 포맷의 경우 순수 Rust에서 파편화된 스트링 풀만 추출합니다.
            let bytes = std::fs::read(path)?;
            let mut result = String::new();
            let mut current_str = String::new();
            
            for b in bytes {
                if (b >= 32 && b <= 126) || b > 127 {
                    current_str.push(b as char);
                } else {
                    if current_str.len() > 5 {
                        result.push_str(&current_str);
                        result.push(' ');
                    }
                    current_str.clear();
                }
            }
            Ok(format!("[주의: HWP 바이너리 포맷은 추출이 불안정할 수 있으므로 HWPX 변환을 권장합니다]\n{}", result))
        },
        "txt" | "md" | "json" => {
            std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!(e))
        },
        _ => Err(anyhow::anyhow!("Unsupported file extension for text extraction: {}", ext)),
    }
}