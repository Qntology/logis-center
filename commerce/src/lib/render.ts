// ==========================================
// [PARITY] Cloud front.js Renderer Engine
// ==========================================

export const selector = {
    app: "logis-app",
    mobile: "logis-mobile",
    desktop: "logis-desktop",
    result: "logis-result",
    info: "logis-info",
    relate: "logis-relate",
    active: "active",
    visited: "visited",
    completed: "completed",
    checkbox: "logis-checkbox",
    label: "logis-label",
    created_at: "field-created-at",
    status: "field-status",
    title: "field-title",
    currency: "field-currency",
    more: "more-content" // 클라우드의 동적 more 클래스를 대체하는 정적 클래스
};

export function parseStatus(status: any): string {
    if (status == 1) return 'progress';
    if (status == 2) return 'stop';
    if (status == 3) return 'cancel';
    if (status == 4) return 'refund';
    if (status == 5) return 'return';
    if (status == 6) return 'error';
    if (status == 7) return 'expire';
    if (status == 8) return 'exchange';
    if (status == 9) return 'complete';
    if (status == 10) return 'draft';
    if (status == 11) return 'show';
    if (status == 12) return 'hide';
    return status?.toString() || '';
}

export function time2text(dateVal: any): string {
    const date = new Date(dateVal);
    const seconds = Math.floor((new Date().getTime() - date.getTime()) / 1000);
    
    let interval = seconds / 31536000;
    if (interval > 1) return Math.floor(interval) + " years";
    
    interval = seconds / 2592000;
    if (interval > 1) return Math.floor(interval) + " months";
    
    interval = seconds / 86400;
    if (interval > 1) return Math.floor(interval) + " days";
    
    interval = seconds / 3600;
    if (interval > 1) return Math.floor(interval) + " hours";
    
    interval = seconds / 60;
    if (interval > 1) return Math.floor(interval) + " minutes";
    
    return Math.floor(seconds) + " seconds";
}

function isAlmostEqual(obj1: any, obj2: any): boolean {
    if (!obj1 || !obj2) return false;
    if (Object.keys(obj1).length === 0 || Object.keys(obj2).length === 0) return false;
    const keys1 = Object.keys(obj1);
    const keys2 = Object.keys(obj2);
    if (keys1.length !== keys2.length) return false;
    
    let diffCount = 0;
    for (const key of keys1) {
        if (obj2.hasOwnProperty(key)) {
            if (obj1[key] !== obj2[key]) diffCount++;
            if (diffCount > 1) return false; 
        }
    }
    return true; 
}

export function item2html(item: any, checked: boolean = false, currentUrl: string = ""): string {
    let href = '';
    if (item.data && item.data.link) {
        href = item.data.link;
    } else if (item.link) {
        href = item.link;
    }

    // `front.js`의 URL 파라미터 비교를 통한 자동 확장 로직 (간소화)
    let more = true; // 기본적으로 확장 데이터 렌더링 허용
    if (href && currentUrl) {
        try {
            const itemUrl = new URL(href, 'http://localhost');
            const footUrl = new URL(currentUrl);
            if (itemUrl.pathname === footUrl.pathname) more = true;
        } catch(e) {}
    }

    const docId = item.id || item.uuid || (item.data && item.data.id) || item.index || Math.random().toString(36).substr(2, 9);
    
    // 🌟 v4 : 봉투 루트를 우선 읽되, 구버전 데이터를 위해 data.* 폴백을 둡니다.
    const createdTs = item.created_at ?? item.data?.created_at ?? 0;
    const updatedTs = item.updated_at ?? item.data?.updated_at ?? 0;
    const modeStr = item.mode ?? item.data?.mode ?? 'commerce';

    let body = `<input type="checkbox" id="more-${docId}" class="toggle-more" ${checked ? 'disabled checked' : ''} style="display:none;" />`;
    body += `<div id="${docId}" class="${selector.result}" data-type="${item.type || ''}" data-mode="${modeStr}" data-created-at="${createdTs}" data-updated-at="${updatedTs}">`;

    let itemType = item.type || "unknown";

    // 🌟 [TRADING FULL SET] app-logis-center 의 get_slice_config 가 다루던 전 서식을 흡수합니다.
    //  기존 7종만 있어서 통관(ED/ID/CINV)·검사증(IC/WC/CA/PHYTO/HC)·
    //  위험물(DGD/MSDS)·법무(POA/BIZ_LIC/INS) 문서가 'unknown' 으로 떨어졌고,
    //  그 결과 status/title/created_at 3줄만 렌더링되어 실무 정보가 전부 소실됐습니다.
    const tradeDocs = [
        'BL', 'AWB', 'CI', 'PI', 'PL', 'CO', 'LC', 'PO', 'SC',
        'SA', 'DO', 'AN', 'BC', 'ED', 'ID', 'CINV',
        'IC', 'WC', 'CA', 'PHYTO', 'HC', 'BEN_CERT',
        'DGD', 'MSDS', 'POA', 'BIZ_LIC', 'INS',
        'shipping_doc', 'shipping'
    ];
    // 🌟 [ANALYTICS ADMIN] 사용자 행동 로그 / 리포트 타입
    const analyticDocs = ['click', 'hover', 'change', 'report'];

    if (analyticDocs.includes(item.type)) {
        itemType = "analytic";
    } else if (item.type === "sales" || item.type === "goods" || item.type === "order") {
        itemType = "sales";
        // 🌟 [CRITICAL FIX] 화면 표시용 타입을 무조건 'order'로 덮어씌우던 원흉(하드코딩) 제거!
        // 이제 DB에 저장된 실제 타입(goods 등)이 UI에 그대로 노출됩니다.
    } else if (item.type === "event" || item.type === "coupon") {
        itemType = "event";
    } else if (tradeDocs.includes(item.type) || tradeDocs.includes(item.type?.toUpperCase())) {
        itemType = "shipping"; // 🌟 무역/선적 문서 전용 타입
    } else if (item.type === "receiving" || item.type === "tracking") {
        itemType = "tracking";
    }

    // 템플릿 생성 함수 (front.js의 Tpl 함수와 100% 동일한 로직)
    function Tpl(itm: any, key: string, unitStr?: string) {
        let _value: any = '';
        let _unit = '';
        let _name = key.replace(/_/gi, " ");

        if (typeof itm[key] !== "undefined") _value = itm[key];
        else if (itm.data && typeof itm.data[key] !== "undefined") _value = itm.data[key];

        if (_value && key === "status") {
            _value = parseStatus(_value) || _value;
        }

        if (unitStr) {
            if (typeof itm[unitStr] !== "undefined") _unit = ` (${itm[unitStr]})`;
            else if (itm.data && typeof itm.data[unitStr] !== "undefined") _unit = ` (${itm.data[unitStr]})`;
        }

        let props = '';
        let tagName = 'div';

        if (key === 'title') {
            tagName = 'a';
            if (itm.data && itm.data.link) {
                // 클릭 시 외부 링크가 아닌 내부 앱 이벤트가 동작하도록 바인딩
                props = `href="javascript:void(0);" onclick="document.dispatchEvent(new CustomEvent('nav-link', {detail: '${itm.data.link}'}));"`;
            }
        }

        // 🌟 [DATE KEYS v2] 무역 서식의 날짜 축(etd/eta/expiry_date)을 추가합니다.
        //    이 목록에 없으면 원시 타임스탬프(1735689600000)가 그대로 노출됩니다.
        const DATE_KEYS = [
            "created_at", "updated_at", "started_at", "expired_at",
            "issue_date", "etd", "eta", "expiry_date",
            "shipped_on_board_date", "declaration_date", "contract_date"
        ];
        if (DATE_KEYS.includes(key)) {
            if (_value) _value = time2text(_value);
            if (key === "created_at") {
                _name = _value; 
                _value = `<label for="more-${docId}" class="more-label" style="cursor:pointer;">More</label>`;
            }
        }

        if (key === "status") {
            _name = itm.type || "status";
        }

        if (key !== "created_at") {
            let input_type = 'text';
            if (typeof _value === "string") {
                // XSS 쉴드 (front.js 로직 완벽 이식)
                _value = _value.replace(/\\/g, '\\\\')
                               .replace(/&/g, '&amp;')
                               .replace(/</g, '&lt;')
                               .replace(/>/g, '&gt;')
                               .replace(/"/g, '&quot;')
                               .replace(/'/g, '&#39;');
                if (key.indexOf('date') > -1) input_type = 'date';
            } else if (typeof _value === "number") {
                input_type = 'number';
            }
            
            // 원래는 input 박스를 그렸지만, 읽기 전용 앱이므로 span 렌더링 유지
            _value = `<span class="value">${_value}</span>`;
        }

        if (!_value || _value === `<span class="value"></span>` || _value === `<span class="value">null</span>`) return '';

        return `
            <${tagName} ${props} class="${selector.info} ${key}">
                <strong>${_name}</strong>
                <span>${_value}<i class="unit">${_unit}</i></span>
            </${tagName}>
        `;
    }

    // --- 타입별 HTML 조립 (front.js 패리티) ---
    // 🌟 [ANALYTICS ADMIN] 사용자 행동 로그 / 리포트 전용 UI
    //    (기존 Client Front SDK(content.js)의 ._item / ._item._user 렌더링을 이관)
    if (itemType === "analytic") {
        body += `
            ${Tpl(item, "action")}
            ${Tpl(item, "summary")}
            ${Tpl(item, "cross_action_flow")}
        `;
        body += `<div class="${selector.more}">`;
        if (more) {
            body += `
                ${Tpl(item, "intent_evolution")}
                ${Tpl(item, "consistent_preferences")}
                ${Tpl(item, "relate")}
                ${Tpl(item, "href")}
            `.trim();
        }
        body += `</div>${Tpl(item, "created_at")}`;

    // 🌟 Shipping / Trading 전용 UI
    } else if (itemType === "shipping") {
        // 🌟 v4 : status 는 봉투가 아니라 data.status 입니다.
        //    canonicalize 가 정수 코드로 확정했으므로 parseStatus 로 문자열화합니다.
        if (item.data && item.data.status !== undefined) {
            item.data.status = parseStatus(item.data.status);
        }

        // 🌟 [DOC TYPE BADGE] 무역 실무자는 'B/L 인가 AWB 인가' 를 가장 먼저 봅니다.
        //    doc_type 이 없으면 봉투 type 을 폴백으로 씁니다.
        if (item.data && !item.data.doc_type && item.type) {
            item.data.doc_type = item.type;
        }

        body += `
            ${Tpl(item, "doc_type")}
            ${Tpl(item, "status")}
            ${Tpl(item, "doc_number")}
            ${Tpl(item, "no")}
            ${Tpl(item, "vessel")}
        `;
        body += `<div class="${selector.more}">`;
        if (more) {
            body += `
                ${Tpl(item, "voyage_number")}
                ${Tpl(item, "pol")}
                ${Tpl(item, "pod")}
                ${Tpl(item, "place_receipt")}
                ${Tpl(item, "place_delivery")}
                ${Tpl(item, "etd")}
                ${Tpl(item, "eta")}
                ${Tpl(item, "transport_mode")}
                ${Tpl(item, "incoterms")}
                ${Tpl(item, "incoterms_place")}
                ${Tpl(item, "payment_terms")}
                ${Tpl(item, "freight_payment_term")}
                ${Tpl(item, "sender_name")}
                ${Tpl(item, "sender_address")}
                ${Tpl(item, "recipient_name")}
                ${Tpl(item, "recipient_address")}
                ${Tpl(item, "notify_party_name")}
                ${Tpl(item, "amount", "currency")}
                ${Tpl(item, "subtotal_amount", "currency")}
                ${Tpl(item, "tax_amount", "currency")}
                ${Tpl(item, "freight_amount", "currency")}
                ${Tpl(item, "insurance_amount", "currency")}
                ${Tpl(item, "local_charges", "currency")}
                ${Tpl(item, "container_number")}
                ${Tpl(item, "seal_number")}
                ${Tpl(item, "package_count", "package_unit")}
                ${Tpl(item, "weight_gross")}
                ${Tpl(item, "weight_net")}
                ${Tpl(item, "volume")}
                ${Tpl(item, "marks_numbers")}
                ${Tpl(item, "hs_code")}
                ${Tpl(item, "origin_criterion")}
                ${Tpl(item, "reference_invoice")}
                ${Tpl(item, "reference_lc")}
                ${Tpl(item, "reference_booking")}
                ${Tpl(item, "issue_date")}
                ${Tpl(item, "expiry_date")}
            `.trim();
        }
        body += `</div>${Tpl(item, "created_at")}`;

    } else if (itemType === "sales") {
        body += `
            ${Tpl(item, "status")}
            ${Tpl(item, "title")}
            ${Tpl(item, "sale_price", "currency")}
        `;
        body += `<div class="${selector.more}">`;
        if (more) {
            body += `
                ${Tpl(item, "price", "currency")}
                ${Tpl(item, "quantity")}
                ${Tpl(item, "width")}
                ${Tpl(item, "height")}
                ${Tpl(item, "length")}
                ${Tpl(item, "weight")}
                ${Tpl(item, "supply_price", "currency")}
                ${Tpl(item, "discount", "currency")}
                ${Tpl(item, "reward_point")}
                ${Tpl(item, "shipping_fee", "currency")}
                ${Tpl(item, "shipping_method")}
                ${Tpl(item, "shipping_duration")}
                ${Tpl(item, "tax_included")}
                ${Tpl(item, "release_date")}
                ${Tpl(item, "manufacture_date")}
                ${Tpl(item, "expiration_date")}
            `.trim();
        }
        body += `</div>${Tpl(item, "created_at")}`;

    } else if (itemType === "tracking") {
        if (item.data && item.data.status !== undefined) {
            item.data.status = parseStatus(item.data.status);
        }
        body += `
            ${Tpl(item, "status")}
            ${Tpl(item, "text")}
            ${Tpl(item, "title")}
        `;
        body += `<div class="${selector.more}">`;
        if (item.data || more) {
            body += `
                ${Tpl(item, "sender_name")}
                ${Tpl(item, "sender_address")}
                ${Tpl(item, "sender_phone")}
                ${Tpl(item, "recipient_name")}
                ${Tpl(item, "recipient_address")}
                ${Tpl(item, "recipient_phone")}
            `.trim();
        }
        body += `</div>${Tpl(item, "created_at")}`;

    } else if (itemType === "event") {
        if (item.data && item.data.status !== undefined) {
            item.data.status = parseStatus(item.data.status);
        }
        body += `
            ${Tpl(item, "status")}
            ${Tpl(item, "title")}
            ${Tpl(item, "discount")}
        `;
        body += `<div class="${selector.more}">`;
        if (more) {
            body += `
                ${Tpl(item, "code")}
                ${Tpl(item, "quantity")}
                ${Tpl(item, "usage_per")}
                ${Tpl(item, "usage_limit")}
                ${Tpl(item, "new_customer_only")}
                ${Tpl(item, "min_order_amount")}
                ${Tpl(item, "max_discount_amount")}
                ${Tpl(item, "first_purchase_only")}
                ${Tpl(item, "region_restrictions")}
            `.trim();
        }
        body += `</div>${Tpl(item, "created_at")}`;
    } else {
        // Fallback for Unknown Types
        body += `
            ${Tpl(item, "status")}
            ${Tpl(item, "title")}
            ${Tpl(item, "created_at")}
        `;
    }

    body += `<input type="hidden" readonly name="${selector.created_at}" value="${createdTs || 'undefined'}" />`;

    // 🌟 [SEARCH BADGE] Dexie 플랜이 어떤 조건으로 이 문서를 통과시켰는지 표시합니다.
    if (item.data && item.data.search_badge) {
        body += `<div class="${selector.info} search-badge" style="opacity:0.7;">
            <strong>match</strong>
            <span><span class="value">${item.data.search_badge}</span></span>
        </div>`;
    }

    // 🌟 [RELAY ANCHOR] v4 : 연관 키가 data.* 로 이동했습니다.
    const d = item.data || {};
    body += `<div class="${selector.relate}" index="${d.index ?? ''}" event="${d.event ?? ''}" views="${d.views ?? ''}" goods="${d.goods ?? ''}" tracking="${d.tracking ?? ''}"></div>`;

    body += `</div>`; // Close .logis-result

    return body;
}