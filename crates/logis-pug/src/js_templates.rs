pub fn get_boa_js_template() -> &'static str {
    r##"
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
            if (cls.length > 0) {
                s += "." + [...new Set(cls)].join(".");
            }
            return s;
        }

        function getChildren(pIdx) { 
            return nodes.filter(n => n && n.parentIndex === pIdx); 
        }

        function calculateSimilarity(nodeA, nodeB) {
            if (nodeA.tagName !== nodeB.tagName) return 0;
            
            
            // (예: 일반 데이터 행 vs colspan=10 인 안내/합계 행)
            if (nodeA.tagName === 'tr') {
                const aKids = getChildren(nodeA.index).filter(n => n.tagName === 'td' || n.tagName === 'th');
                const bKids = getChildren(nodeB.index).filter(n => n.tagName === 'td' || n.tagName === 'th');
                
                const aColspan = aKids.reduce((sum, k) => sum + parseInt(k.colspan || '1', 10), 0);
                const bColspan = bKids.reduce((sum, k) => sum + parseInt(k.colspan || '1', 10), 0);
                
                // 두 행의 가로 칸 수(colspan 총합)가 2칸 이상 차이난다면 구조가 아예 다른 것입니다.
                if (aColspan > 0 && bColspan > 0 && Math.abs(aColspan - bColspan) > 1) {
                    return 0;
                }
            }
            
            
            if (nodeA.tagName === 'td' || nodeA.tagName === 'th') {
                if (nodeA.colspan !== nodeB.colspan || nodeA.rowspan !== nodeB.rowspan) return 0;
            }

            const clsA = cleanClassList(nodeA.classes, true);
            const clsB = cleanClassList(nodeB.classes, true);
            if (clsA.length === 0 && clsB.length === 0) return 100;
            
            let matchCount = 0;
            clsA.forEach(c => { if (clsB.includes(c)) matchCount++; });
            return clsA.length ? (matchCount / clsA.length) * 100 : 0;
        }

        function detect(tIdx) {
            let cur = tIdx;
            for (let i = 0; i < 15; i++) {
                const node = nodes[cur];
                if (!node) break;
                
                const pIdx = node.parentIndex;
                if (pIdx === undefined || pIdx === -1) break;
                
                if (node.tagName === "td" || node.tagName === "th") {
                    
                    // 이는 단일 항목이 아니라 복잡한 그리드의 부속품입니다. 묻지도 따지지도 않고 부모(tr)로 올라갑니다.
                    if (parseInt(node.colspan || '1', 10) > 1 || parseInt(node.rowspan || '1', 10) > 1) {
                        cur = pIdx;
                        continue;
                    }
                    
                    const pNode = nodes[pIdx];
                    if (pNode && pNode.tagName === "tr") {
                        const gpIdx = pNode.parentIndex; 
                        if (gpIdx !== undefined && gpIdx !== -1) {
                            const trSiblings = getChildren(gpIdx);
                            const similarTrs = trSiblings.filter(s => calculateSimilarity(pNode, s) >= 60);
                            
                            // 부모(tr)가 유사한 구조의 다른 형제(tr)들을 여럿 거느리고 있다면 진짜 세로 리스트입니다.
                            if (similarTrs.length >= 2) {
                                cur = pIdx;
                                continue;
                            }
                        }
                    }
                }

                const parentNode = nodes[pIdx];
                const siblings = getChildren(pIdx);
                
                const similarSiblings = siblings.filter(s => calculateSimilarity(node, s) >= 60);

                if (similarSiblings.length >= 2) {
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
                    const uniqueSigs = [];
                    similarSiblings.forEach(s => {
                        const sig = getSignature(s, false);
                        if (!uniqueSigs.includes(sig)) uniqueSigs.push(sig);
                    });

                    const fullSelector = uniqueSigs.map(sig => parentSig + " " + sig).join(", ");

                    return { 
                        parent: parentSig, 
                        itemSelector: fullSelector,
                        matchCount: similarSiblings.length
                    };
                }
                cur = pIdx;
            }
            return null;
        }

        let bestResult = { "parent": "body", "itemSelector": "div", "matchCount": 0 };
        let firstMatchD = null;
        
        for (let i = 0; i < titles.length; i++) {
            let t = titles[i].toLowerCase().replace(/\s+/g, ' ');
            let potentialMatches = [];
            
            // 깨진 문자(\uFFFD)가 포함되어 있는지 확인합니다.
            if (t.includes('\uFFFD')) {
                // 깨진 경우: 쪼개서 조각들로 유연하게 검색
                let chunks = t.split(/[\uFFFD]+/).map(c => c.trim()).filter(c => c.length > 1);
                if (chunks.length === 0) continue;
                
                potentialMatches = nodes.filter(n => {
                    if (!n || !n.text) return false;
                    let nText = n.text.toLowerCase().replace(/\s+/g, ' ');
                    return chunks.every(chunk => nText.includes(chunk));
                });
            } else {
                // 온전한 경우: 전체 문자열을 하나의 컬렉션으로 취급하여 정확하게 포함 여부 검색
                potentialMatches = nodes.filter(n => {
                    if (!n || !n.text) return false;
                    let nText = n.text.toLowerCase().replace(/\s+/g, ' ');
                    return nText.includes(t);
                });
            }
            
            if (potentialMatches.length > 0) {
                // 부모 노드(body, tr 등)를 배제하고, 텍스트 길이가 가장 짧은(가장 타이트한) 진짜 제목 단일 노드만 추출합니다.
                potentialMatches.sort((a, b) => a.text.length - b.text.length);
                let d = detect(potentialMatches[0].index);
                if (d) {
                    if (!firstMatchD) firstMatchD = d;
                    // 형제 노드가 가장 많은(리스트에 가장 가까운) 결과를 최우선으로 저장
                    if (d.matchCount > bestResult.matchCount) {
                        bestResult = { "parent": d.parent, "itemSelector": d.itemSelector, "matchCount": d.matchCount };
                    }
                }
            }
        }
        
        // 반복 구조(matchCount > 0)를 하나도 못 찾았다면 첫 번째로 찾은 구조라도 사용
        if (bestResult.matchCount === 0 && firstMatchD) {
            bestResult = { "parent": firstMatchD.parent, "itemSelector": firstMatchD.itemSelector, "matchCount": firstMatchD.matchCount };
        }
        
        JSON.stringify(bestResult);
    "##
}

// 🌟 [CRITICAL OPTIMIZATION] 여러 개의 텍스트 타겟을 한 번의 JS 엔진 구동으로 일괄(Batch) 역추적하여 반환하는 템플릿입니다.
pub fn get_boa_block_extractor_template() -> &'static str {
    r##"
        const nodes = NODES_PLACEHOLDER;
        const targetTitles = TARGET_TITLES_PLACEHOLDER;
        
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
            if (cls.length > 0) {
                s += "." + [...new Set(cls)].join(".");
            }
            return s;
        }

        function getChildren(pIdx) { 
            return nodes.filter(n => n && n.parentIndex === pIdx); 
        }

        function calculateSimilarity(nodeA, nodeB) {
            if (nodeA.tagName !== nodeB.tagName) return 0;
            if (nodeA.tagName === 'tr') {
                const aKids = getChildren(nodeA.index).filter(n => n.tagName === 'td' || n.tagName === 'th');
                const bKids = getChildren(nodeB.index).filter(n => n.tagName === 'td' || n.tagName === 'th');
                const aColspan = aKids.reduce((sum, k) => sum + parseInt(k.colspan || '1', 10), 0);
                const bColspan = bKids.reduce((sum, k) => sum + parseInt(k.colspan || '1', 10), 0);
                if (aColspan > 0 && bColspan > 0 && Math.abs(aColspan - bColspan) > 1) { return 0; }
            }
            if (nodeA.tagName === 'td' || nodeA.tagName === 'th') {
                if (nodeA.colspan !== nodeB.colspan || nodeA.rowspan !== nodeB.rowspan) return 0;
            }
            const clsA = cleanClassList(nodeA.classes, true);
            const clsB = cleanClassList(nodeB.classes, true);
            if (clsA.length === 0 && clsB.length === 0) return 100;
            let matchCount = 0;
            clsA.forEach(c => { if (clsB.includes(c)) matchCount++; });
            return clsA.length ? (matchCount / clsA.length) * 100 : 0;
        }

        function detect(tIdx) {
            let cur = tIdx;
            let fallbackParent = null;
            for (let i = 0; i < 15; i++) {
                const node = nodes[cur];
                if (!node) break;
                const pIdx = node.parentIndex;
                if (pIdx === undefined || pIdx === -1) break;

                const parentNode = nodes[pIdx];
                if (parentNode && !fallbackParent) {
                    // 🌟 단독으로 배치된 Form 이나 고유 ID 클래스가 있는 시맨틱 컨테이너를 백업 부모로 동적 추적합니다.
                    if (parentNode.id || (parentNode.classes && parentNode.classes.length > 0) || ["form", "table", "nav", "tbody", "fieldset"].includes(parentNode.tagName)) {
                        fallbackParent = parentNode;
                    }
                }
                
                if (node.tagName === "td" || node.tagName === "th") {
                    if (parseInt(node.colspan || '1', 10) > 1 || parseInt(node.rowspan || '1', 10) > 1) {
                        cur = pIdx; continue;
                    }
                    const pNode = nodes[pIdx];
                    if (pNode && pNode.tagName === "tr") {
                        const gpIdx = pNode.parentIndex; 
                        if (gpIdx !== undefined && gpIdx !== -1) {
                            const trSiblings = getChildren(gpIdx);
                            const similarTrs = trSiblings.filter(s => calculateSimilarity(pNode, s) >= 60);
                            if (similarTrs.length >= 2) { cur = pIdx; continue; }
                        }
                    }
                }
                const siblings = getChildren(pIdx);
                const similarSiblings = siblings.filter(s => calculateSimilarity(node, s) >= 60);
                if (similarSiblings.length >= 2) {
                    let finalParent = parentNode;
                    let walkIdx = pIdx;
                    for(let j=0; j<5; j++) {
                        let gIdx = nodes[walkIdx] ? nodes[walkIdx].parentIndex : -1;
                        if (gIdx !== -1 && nodes[gIdx]) {
                            const grand = nodes[gIdx];
                            if (grand.id || ["table", "ul", "ol", "nav", "form"].includes(grand.tagName)) {
                                finalParent = grand;
                                if (grand.id || grand.tagName === "table" || grand.tagName === "form") break;
                            }
                            walkIdx = gIdx;
                        }
                    }
                    return getSignature(finalParent, true);
                }
                cur = pIdx;
            }
            // 🌟 형제 노드 반복 패턴이 없는 독자적인 레이아웃일 경우 추적 수집된 시맨틱 백업 부모 블록을 안전하게 반환합니다.
            if (fallbackParent) { return getSignature(fallbackParent, true); }
            return null;
        }

        let finalResults = [];
        for (let k = 0; k < targetTitles.length; k++) {
            let t = targetTitles[k].toLowerCase().replace(/\s+/g, ' ');
            let potentialMatches = [];
            if (t.includes('\uFFFD')) {
                let chunks = t.split(/[\uFFFD]+/).map(c => c.trim()).filter(c => c.length > 1);
                if (chunks.length > 0) {
                    potentialMatches = nodes.filter(n => {
                        if (!n || !n.text) return false;
                        let nText = n.text.toLowerCase().replace(/\s+/g, ' ');
                        return chunks.every(chunk => nText.includes(chunk));
                    });
                }
            } else {
                potentialMatches = nodes.filter(n => {
                    if (!n || !n.text) return false;
                    let nText = n.text.toLowerCase().replace(/\s+/g, ' ');
                    return nText.includes(t);
                });
            }
            
            let parentSel = "";
            if (potentialMatches.length > 0) {
                potentialMatches.sort((a, b) => a.text.length - b.text.length);
                const d = detect(potentialMatches[0].index);
                if (d && d !== "body") { parentSel = d; }
            }
            finalResults.push(parentSel);
        }
        JSON.stringify(finalResults);
    "##
}