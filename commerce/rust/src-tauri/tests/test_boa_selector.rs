use boa_engine::{Context, Source};
use scraper::Html;
use serde_json::json;
use std::fs;
use std::collections::HashMap;

#[test]
fn test_boa_selector_detection() {
    let pug_content = fs::read_to_string("tests/temp_pug_content.txt").expect("Failed to read PUG content");
    let document = Html::parse_document(&pug_content);
    let mut nodes_json = Vec::new();
    let mut node_to_idx = HashMap::new();

    for (idx, node) in document.tree.root().descendants().enumerate() {
        node_to_idx.insert(node.id(), idx);
        if let Some(el) = node.value().as_element() {
            let parent_idx = node.parent().and_then(|p| node_to_idx.get(&p.id())).map(|&i| i as i32).unwrap_or(-1);
            let text: String = node.children().filter_map(|child| child.value().as_text().map(|t| t.to_string())).collect::<Vec<_>>().join(" ").trim().to_string();
            nodes_json.push(json!({ "index": idx, "parentIndex": parent_idx, "tagName": el.name().to_string(), "id": el.id().unwrap_or("").to_string(), "classes": el.attr("class").unwrap_or("").split_whitespace().collect::<Vec<_>>(), "text": text }));
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
            // rows, list0, list1 같은 반복되는 클래스는 유지해야 패턴 매칭이 됨
            const ignore = ['active', 'selected', 'on', 'current', 'focus', 'hover'];
            return classes.filter(c => !ignore.includes(c.toLowerCase()) && c.indexOf('__') === -1).sort();
        }

        function getSignature(node, includeId = true) {
            if (!node) return "";
            let s = node.tagName;
            if (includeId && node.id) s += "#" + node.id;
            const cls = cleanClassList(node.classes);
            if (cls.length > 0) s += "." + cls.join(".");
            return s;
        }

        function findElements(textToFind) {
            const lower = textToFind.toLowerCase();
            return nodes.filter(n => n.text && n.text.toLowerCase().includes(lower));
        }

        function getChildren(pIdx) { return nodes.filter(n => n && n.parentIndex === pIdx); }

        function detect(tIdx) {
            let cur = tIdx;
            // 부모 탐색을 더 높게 (10단계까지)
            for (let i = 0; i < 10 && cur !== -1; i++) {
                const node = nodes[cur];
                if (!node) break;
                const pIdx = node.parentIndex;
                if (pIdx === -1) break;
                
                const sibs = getChildren(pIdx);
                // 형제가 2개 이상이고 시그니처가 어느 정도 반복되면 리스트로 인정
                if (sibs.length >= 2) {
                    const sigs = sibs.map(s => getSignature(s, false)).filter(s => s !== "");
                    const counts = {}; sigs.forEach(s => counts[s] = (counts[s] || 0) + 1);
                    const entries = Object.entries(counts).sort((a,b) => b[1]-a[1]);
                    
                    // 가장 많은 패턴이 전체의 30% 이상이면 리스트 후보 (복합 구조 고려하여 낮춤)
                    if (entries.length && (entries[0][1] / sibs.length) >= 0.3) {
                        const parentNode = nodes[pIdx];
                        // 부모가 table이나 tbody면 더 올라가야 할 수도 있음
                        if (["tbody", "thead"].includes(parentNode.tagName)) {
                            cur = pIdx;
                            continue;
                        }
                        return { "parent": getSignature(parentNode, true), "item": entries[0][0], "confidence": (entries[0][1] / sibs.length) };
                    }
                }
                cur = pIdx;
            }
            return null;
        }

        const matches = findElements(titles[0]);
        let res = { "parent": "body", "item": "div", "matchCount": matches.length, "matches": [] };
        if (matches.length > 0) {
            res.matches = matches.slice(0, 3).map(m => ({ signature: getSignature(m), text: m.text.substring(0, 20) }));
            const d = detect(matches[0].index);
            if (d) { res.parent = d.parent; res.item = d.item; res.confidence = d.confidence; }
        }
        JSON.stringify(res);
    "##;

    let js_code = js_template.replace("NODES_PLACEHOLDER", &nodes_str).replace("TITLES_PLACEHOLDER", &titles_str);
    match context.eval(Source::from_bytes(js_code.as_bytes())) {
        Ok(val) => { println!("RESULT: {}", val.as_string().unwrap().to_std_string_escaped()); },
        Err(e) => { panic!("JS Error: {:?}", e); }
    }
}
