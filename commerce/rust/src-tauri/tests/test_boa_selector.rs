use boa_engine::{Context, Source};
use scraper::Html;
use serde_json::json;
use std::fs;
use std::collections::HashMap;

#[test]
fn test_boa_selector_detection() {
    let html_content = fs::read_to_string("tests/temp_clean_html.txt").expect("Failed to read HTML content");
    let document = Html::parse_document(&html_content);
    let mut nodes_json = Vec::new();
    let mut node_to_idx = HashMap::new();

    // 1단계: 모든 노드의 ID를 인덱스에 매핑 (부모 참조를 위해)
    for (idx, node) in document.tree.root().descendants().enumerate() {
        node_to_idx.insert(node.id(), idx);
    }

    // 2단계: 노드 정보 수집
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
            // 태그가 아닌 노드(Text 등)도 인덱스 유지를 위해 null로 채움
            nodes_json.push(json!(null));
        }
    }

    let titles = vec!["F7093".to_string()];
    let nodes_str = serde_json::to_string(&nodes_json).unwrap();
    let titles_str = serde_json::to_string(&titles).unwrap();

    let mut context = Context::default();
    let js_template = r##"
        const nodes = NODES_PLACEHOLDER;
        const titles = TITLES_PLACEHOLDER;
        
        function cleanClassList(classes) {
            if (!classes) return [];
            return classes
                .map(c => c.toLowerCase().replace(/\d+$/, ''))
                .filter(c => !['active', 'rows'].includes(c))
                .sort();
        }

        function getSignature(node, includeId = true) {
            if (!node || !node.tagName) return "";
            let s = node.tagName;
            if (includeId && node.id) s += "#" + node.id;
            const cls = cleanClassList(node.classes);
            if (cls.length > 0) s += "." + [...new Set(cls)].join(".");
            return s;
        }

        function getChildren(pIdx) { 
            return nodes.filter(n => n && n.parentIndex === pIdx); 
        }

        function detect(tIdx) {
            let cur = tIdx;
            let history = [];
            for (let i = 0; i < 15; i++) {
                const node = nodes[cur];
                if (!node) { 
                    // 부모가 null인 경우(Text 노드 등) 한 단계 더 위로
                    cur = (nodes[cur] && nodes[cur].parentIndex) || -1;
                    if (cur === -1) break;
                    continue;
                }
                
                const pIdx = node.parentIndex;
                if (pIdx === undefined || pIdx === -1) break;
                
                const sibs = getChildren(pIdx);
                const parentNode = nodes[pIdx];
                
                if (sibs.length >= 2) {
                    const sigs = sibs.map(s => getSignature(s, false)).filter(s => s !== "");
                    const counts = {}; sigs.forEach(s => counts[s] = (counts[s] || 0) + 1);
                    const entries = Object.entries(counts).sort((a,b) => b[1]-a[1]);
                    
                    if (entries.length > 0) {
                        const topPattern = entries[0][0];
                        const confidence = entries[0][1] / sibs.length;
                        history.push({ i, tagName: parentNode.tagName, sig: getSignature(parentNode), confidence, topPattern });

                        if (confidence >= 0.2 && (topPattern.startsWith("tr") || topPattern.startsWith("li") || topPattern.startsWith("div"))) {
                            // 테이블/리스트 구조 발견
                            let finalParent = parentNode;
                            // table이나 div#id 가 나올 때까지 2단계 더 상향 탐색
                            let walkIdx = pIdx;
                            for(let j=0; j<3; j++) {
                                let gIdx = nodes[walkIdx] ? nodes[walkIdx].parentIndex : -1;
                                if (gIdx !== -1 && nodes[gIdx]) {
                                    const grand = nodes[gIdx];
                                    if (grand.id || ["table", "ul", "ol"].includes(grand.tagName)) {
                                        finalParent = grand;
                                        if (grand.tagName === "table") break;
                                    }
                                    walkIdx = gIdx;
                                }
                            }
                            return { parent: getSignature(finalParent, true), item: topPattern, confidence, history };
                        }
                    }
                }
                cur = pIdx;
            }
            return { error: "Structure Not Recognized", history };
        }

        const matches = nodes.filter(n => n && n.text && titles.some(t => n.text.includes(t)));
        let res = { matchCount: matches.length };
        if (matches.length > 0) {
            const d = detect(matches[0].index);
            res = { ...res, ...d };
        }
        JSON.stringify(res);
    "##;

    let js_code = js_template.replace("NODES_PLACEHOLDER", &nodes_str).replace("TITLES_PLACEHOLDER", &titles_str);
    match context.eval(Source::from_bytes(js_code.as_bytes())) {
        Ok(val) => { println!("RESULT: {}", val.as_string().unwrap().to_std_string_escaped()); },
        Err(e) => { panic!("JS Error: {:?}", e); }
    }
}
