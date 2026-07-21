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
    
    let body = `<input type="checkbox" id="more-${docId}" class="toggle-more" ${checked ? 'disabled checked' : ''} style="display:none;" />`;
    body += `<div id="${docId}" class="${selector.result}" data-type="${item.type || ''}" data-created-at="${item.created_at || 0}" data-updated-at="${item.updated_at || 0}">`;

    let itemType = item.type || "unknown";
    const tradeDocs = ['BL', 'AWB', 'CI', 'PI', 'PL', 'CO', 'LC', 'shipping_doc', 'shipping'];

    if (item.type === "sales" || item.type === "goods" || item.type === "order") {
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

        // 🌟 issue_date 추가
        if (key === "created_at" || key === "updated_at" || key === "started_at" || key === "expired_at" || key === "issue_date") {
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
    // 🌟 Shipping 전용 UI 추가
    if (itemType === "shipping") {
        if(item.status) item.status = parseStatus(item.status);
        body += `
            ${Tpl(item, "status")}
            ${Tpl(item, "no")}
            ${Tpl(item, "vessel")}
        `;
        body += `<div class="${selector.more}">`;
        if (more) {
            body += `
                ${Tpl(item, "pol")}
                ${Tpl(item, "pod")}
                ${Tpl(item, "incoterms")}
                ${Tpl(item, "sender_name")}
                ${Tpl(item, "recipient_name")}
                ${Tpl(item, "amount", "currency")}
                ${Tpl(item, "issue_date")}
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
        if(item.status) item.status = parseStatus(item.status);
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
        if(item.status) item.status = parseStatus(item.status);
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

    body += `<input type="hidden" readonly name="${selector.created_at}" value="${item.created_at || 'undefined'}" />`;
    
    // Relay 연동을 위한 메타데이터 앵커 태그
    body += `<div class="${selector.relate}" index="${item.index}" event="${item.event}" views="${item.views}" goods="${item.goods}" tracking="${item.tracking}"></div>`;
    
    body += `</div>`; // Close .logis-result

    return body;
}