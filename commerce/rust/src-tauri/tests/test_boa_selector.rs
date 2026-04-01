use boa_engine::{Context, Source};
use scraper::Html;
use serde_json::json;
use std::fs;
use std::collections::HashMap;
use tauri_app_lib::parsing::{self, PugMode};

#[test]
fn test_automated_extraction_pipeline() {
    let html_content = fs::read_to_string("tests/temp_clean_html.txt").expect("Failed to read HTML content");
    let document = Html::parse_document(&html_content);
    let mut nodes_json = Vec::new();
    let mut node_to_idx = HashMap::new();

    for (idx, node) in document.tree.root().descendants().enumerate() {
        node_to_idx.insert(node.id(), idx);
    }

    for (idx, node) in document.tree.root().descendants().enumerate() {
        if let Some(el) = node.value().as_element() {
            let parent_idx = node.parent().and_then(|p| node_to_idx.get(&p.id())).map(|&i| i as i32).unwrap_or(-1);
            let text: String = node.children().filter_map(|child| child.value().as_text().map(|t| t.to_string())).collect::<Vec<_>>().join(" ").trim().to_string();
            nodes_json.push(json!({ 
                "index": idx, 
                "parentIndex": parent_idx, 
                "tagName": el.name().to_string(), 
                "id": el.id().unwrap_or("").to_string(), 
                "classes": el.attr("class").unwrap_or("").split_whitespace().collect::<Vec<_>>(), 
                "text": text 
            }));
        } else {
            nodes_json.push(json!(null));
        }
    }

    // [INPUT] 오직 타겟 텍스트만 주어짐
    let titles = vec!["테스트 상품3".to_string()];
    let nodes_str = serde_json::to_string(&nodes_json).unwrap();
    let titles_str = serde_json::to_string(&titles).unwrap();

    let mut context = Context::default();
    let js_template = r##"
        const nodes = NODES_PLACEHOLDER;
        const titles = TITLES_PLACEHOLDER;
        
        function cleanClassList(classes, stripNumbers = false) {
            if (!classes) return [];
            const skip = ['active', 'selected', 'on', 'current', 'focus', 'hover', 'enabled', 'disabled'];
            return classes
                .filter(c => {
                    const lowerC = c.toLowerCase();
                    return !skip.includes(lowerC) && c.indexOf('__') === -1 && !/^[a-z0-9]{8,}$/.test(c);
                })
                .map(c => stripNumbers ? c.replace(/\d+$/, '') : c)
                .sort();
        }

        function getSignature(node, includeId = true) {
            if (!node || !node.tagName) return "";
            let s = node.tagName;
            if (includeId && node.id) s += "#" + node.id;
            const cls = cleanClassList(node.classes);
            if (cls.length > 0) s += "." + cls.join(".");
            return s;
        }

        function getChildren(pIdx) { 
            return nodes.filter(n => n && n.parentIndex === pIdx); 
        }

        // tracker.js 핵심: 유사도 점수 계산 (태그 일치 + 클래스 일부 일치)
        // [FIX] 숫자를 제거한 클래스명으로 비교하여 zebra stripes(list0, list1) 대응
        function calculateSimilarity(nodeA, nodeB) {
            if (nodeA.tagName !== nodeB.tagName) return 0;
            const clsA = cleanClassList(nodeA.classes, true);
            const clsB = cleanClassList(nodeB.classes, true);
            if (clsA.length === 0 && clsB.length === 0) return 100;
            
            let matchCount = 0;
            clsA.forEach(c => { if (clsB.includes(c)) matchCount++; });
            return clsA.length ? (matchCount / clsA.length) * 100 : 0;
        }

        function detectList(tIdx) {
            let curIdx = tIdx;
            let debugLogs = [];
            for (let i = 0; i < 15; i++) {
                const node = nodes[curIdx];
                if (!node) break;
                
                const pIdx = node.parentIndex;
                if (pIdx === undefined || pIdx === -1) break;
                
                const parentNode = nodes[pIdx];
                const siblings = getChildren(pIdx);
                
                debugLogs.push(`Level ${i}: Parent <${parentNode.tagName}> Class: [${parentNode.classes.join(", ")}]`);
                
                const similarSiblings = siblings.filter(s => {
                    const score = calculateSimilarity(node, s);
                    if (score >= 20) {
                        debugLogs.push(`  - Sibling <${s.tagName}> Original Class: [${s.classes.join(", ")}] Score (stripped): ${score.toFixed(1)}`);
                    }
                    return score >= 60;
                });

                if (similarSiblings.length >= 2) {
                    // 리스트 부모 찾기 (ID나 Table 선호)
                    let finalParent = parentNode;
                    let walkIdx = pIdx;
                    for(let j=0; j<5; j++) {
                        let gIdx = nodes[walkIdx] ? nodes[walkIdx].parentIndex : -1;
                        if (gIdx !== -1 && nodes[gIdx]) {
                            const grand = nodes[gIdx];
                            if (grand.id || ["table", "ul", "ol", "nav"].includes(grand.tagName)) {
                                finalParent = grand;
                                if (grand.id || grand.tagName === "table") break;
                            }
                            walkIdx = gIdx;
                        }
                    }

                    const parentSig = getSignature(finalParent, true);
                    
                    // 아이템 셀렉터 유추: 수집된 모든 유사 형제의 클래스를 분석
                    const uniqueSigs = [];
                    similarSiblings.forEach(s => {
                        const sig = getSignature(s, false);
                        if (!uniqueSigs.includes(sig)) uniqueSigs.push(sig);
                    });

                    // 콤마로 연결된 최종 셀렉터 생성
                    const fullSelector = uniqueSigs.map(sig => parentSig + " " + sig).join(", ");

                    return { 
                        parent: parentSig, 
                        itemSelector: fullSelector,
                        matchCount: similarSiblings.length,
                        debug: debugLogs
                    };
                }
                curIdx = pIdx;
            }
            return { error: "No list detected", debug: debugLogs };
        }

        const findText = titles[0].toLowerCase().replace(/\s+/g, ' ');
        const matches = nodes.filter(n => n && n.text && n.text.toLowerCase().replace(/\s+/g, ' ').includes(findText));
        
        if (matches.length > 0) {
            const res = detectList(matches[0].index);
            JSON.stringify(res);
        } else {
            JSON.stringify({ error: "Text not found" });
        }
    "##;

    let js_code = js_template.replace("NODES_PLACEHOLDER", &nodes_str).replace("TITLES_PLACEHOLDER", &titles_str);
    let detection_res = match context.eval(Source::from_bytes(js_code.as_bytes())) {
        Ok(val) => val.as_string().unwrap().to_std_string_escaped(),
        Err(e) => panic!("JS Error: {:?}", e),
    };

    let res_json: serde_json::Value = serde_json::from_str(&detection_res).unwrap();
    
    if let Some(debug) = res_json.get("debug").and_then(|v| v.as_array()) {
        println!("\n--- [DETECTION LOGS] ---");
        for log in debug {
            println!("{}", log.as_str().unwrap_or(""));
        }
    }

    if let Some(selector) = res_json.get("itemSelector").and_then(|v| v.as_str()) {
        println!("\n[PHASE 1] Final Result: {}", detection_res);
        println!("[PHASE 2] Using Detected Selector: {}", selector);
        
        let pug_list = parsing::split_html_to_pug_list(&html_content, selector, PugMode::FullContent);
        println!("[PHASE 3] Found {} items via Pug conversion.", pug_list.len());
        
        // 검증: list1이 포함되어 있는지 확인
        assert!(selector.contains("list1"), "Selector should contain list1");
        assert!(pug_list.len() >= 13, "Should find all items (at least 13 based on HTML)");
    } else {
        println!("Error: {}", res_json.get("error").and_then(|v| v.as_str()).unwrap_or("Unknown"));
        panic!("Detection failed");
    }
}
