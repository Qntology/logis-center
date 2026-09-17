import { parseStatus, time2text } from "./utils";

// --- Static Selectors ---
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
    more: "more-content"
};

const TRADE_DOC_TYPES = [
    'BL', 'AWB', 'CI', 'PI', 'PL', 'CO', 'LC', 'PO', 'SC',
    'SA', 'DO', 'AN', 'BC', 'ED', 'ID', 'CINV',
    'IC', 'WC', 'CA', 'PHYTO', 'HC', 'BEN_CERT',
    'DGD', 'MSDS', 'POA', 'BIZ_LIC', 'INS',
    'shipping_doc', 'shipping'
];
const ANALYTIC_DOC_TYPES = ['click', 'hover', 'change', 'touch', 'report'];
const DATE_KEYS = [
    "created_at", "updated_at", "started_at", "expired_at",
    "issue_date", "etd", "eta", "expiry_date", "release_date",
    "shipped_on_board_date", "declaration_date", "contract_date"
];

// --- Main Rendering Function ---
export function item2html(item: any, checked: boolean = false, currentUrl: string = ""): string {
    let href = "";
    if (item.data && item.data.link) {
        href = item.data.link;
    } else if (item.link) {
        href = item.link;
    }
    let more = true;
    if (href && currentUrl) {
        try {
            const itemUrl = new URL(href, "http://localhost");
            const footUrl = new URL(currentUrl);
            if (itemUrl.pathname === footUrl.pathname) more = true;
        } catch (e) {}
    }

    const docId = item.id || item.uuid || (item.data && item.data.id) || item.index
        || Math.random().toString(36).substr(2, 9);
    const createdTs = item.created_at ?? item.data?.created_at ?? 0;
    const updatedTs = item.updated_at ?? item.data?.updated_at ?? 0;
    const modeStr = item.mode ?? item.data?.mode ?? "commerce";

    const rawType = String(item.type || item.doc_type || "unknown");
    let itemType = rawType;
    if (ANALYTIC_DOC_TYPES.includes(rawType)) {
        itemType = "analytic";
    } else if (rawType === "sales" || rawType === "goods" || rawType === "order") {
        itemType = "sales";
    } else if (rawType === "event" || rawType === "coupon") {
        itemType = "event";
    } else if (TRADE_DOC_TYPES.includes(rawType) || TRADE_DOC_TYPES.includes(rawType.toUpperCase())) {
        itemType = "shipping";
    } else if (rawType === "receiving" || rawType === "tracking") {
        itemType = "tracking";
    } else {
        itemType = "unknown";
    }

    function Tpl(itm: any, key: string, unitStr?: string): string {
        let _value: any = "";
        let _unit = "";
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

        let props = "";
        let tagName = "div";
        if (key === "title") {
            tagName = "a";
            const targetLink = (itm.data && itm.data.link) || itm.link || "";
            if (targetLink) {
                props = `href="javascript:void(0);" onclick="document.dispatchEvent(new CustomEvent('nav-link', {detail: '${targetLink}'}));"`;
            }
        }

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
            if (typeof _value === "string") {
                _value = _value.replace(/\\/g, '\\\\')
                               .replace(/&/g, '&amp;')
                               .replace(/</g, '&lt;')
                               .replace(/>/g, '&gt;')
                               .replace(/"/g, '&quot;')
                               .replace(/'/g, '&#39;');
            }
            _value = `<span class="value">${_value}</span>`;
        }

        if (!_value || _value === `<span class="value"></span>` || _value === `<span class="value">null</span>`) return "";

        return `
            <${tagName} ${props} class="${selector.info} ${key}">
                <strong>${_name}</strong>
                <span>${_value}<i class="unit">${_unit}</i></span>
            </${tagName}>
        `;
    }

    let body = `<input type="checkbox" id="more-${docId}" class="toggle-more" ${checked ? 'disabled checked' : ''} style="display:none;" />`;
    body += `<div id="${docId}" class="${selector.result} ${itemType}" data-type="${rawType}" data-mode="${modeStr}" data-created-at="${createdTs}" data-updated-at="${updatedTs}">`;

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
    } else if (itemType === "shipping") {
        if (item.data && item.data.status !== undefined) {
            item.data.status = parseStatus(item.data.status) || item.data.status;
        }
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
                ${Tpl(item, "payment_terms")}
                ${Tpl(item, "freight_payment_term")}
                ${Tpl(item, "sender_name")}
                ${Tpl(item, "recipient_name")}
                ${Tpl(item, "notify_party_name")}
                ${Tpl(item, "amount", "currency")}
                ${Tpl(item, "freight_amount", "currency")}
                ${Tpl(item, "insurance_amount", "currency")}
                ${Tpl(item, "container_number")}
                ${Tpl(item, "seal_number")}
                ${Tpl(item, "package_count", "package_unit")}
                ${Tpl(item, "weight_gross")}
                ${Tpl(item, "weight_net")}
                ${Tpl(item, "volume")}
                ${Tpl(item, "hs_code")}
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
                ${Tpl(item, "supply_price", "currency")}
                ${Tpl(item, "discount", "currency")}
                ${Tpl(item, "stock_keeping_unit")}
                ${Tpl(item, "shipping_fee", "currency")}
                ${Tpl(item, "shipping_method")}
                ${Tpl(item, "shipping_duration")}
                ${Tpl(item, "tax_included")}
                ${Tpl(item, "release_date")}
            `.trim();
        }
        body += `</div>${Tpl(item, "created_at")}`;
    } else if (itemType === "tracking") {
        if (item.data && item.data.status !== undefined) {
            item.data.status = parseStatus(item.data.status) || item.data.status;
        }
        body += `
            ${Tpl(item, "status")}
            ${Tpl(item, "text")}
            ${Tpl(item, "title")}
        `;
        body += `<div class="${selector.more}">`;
        if (more) {
            body += `
                ${Tpl(item, "carrier")}
                ${Tpl(item, "tracking_number")}
                ${Tpl(item, "sender_name")}
                ${Tpl(item, "sender_address")}
                ${Tpl(item, "recipient_name")}
                ${Tpl(item, "recipient_address")}
            `.trim();
        }
        body += `</div>${Tpl(item, "created_at")}`;
    } else if (itemType === "event") {
        if (item.data && item.data.status !== undefined) {
            item.data.status = parseStatus(item.data.status) || item.data.status;
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
                ${Tpl(item, "min_order_amount")}
                ${Tpl(item, "max_discount_amount")}
                ${Tpl(item, "expired_at")}
            `.trim();
        }
        body += `</div>${Tpl(item, "created_at")}`;
    } else {
        body += `
            ${Tpl(item, "status")}
            ${Tpl(item, "title")}
            ${Tpl(item, "text")}
            ${Tpl(item, "created_at")}
        `;
    }

    body += `<input type="hidden" readonly name="${selector.created_at}" value="${createdTs || 'undefined'}" />`;

    if (item.data && item.data.search_badge) {
        body += `<div class="${selector.info} search-badge" style="opacity:0.7;">
            <strong>match</strong>
            <span><span class="value">${item.data.search_badge}</span></span>
        </div>`;
    }

    const d = item.data || {};
    body += `<div class="${selector.relate}" index="${d.index ?? ''}" event="${d.event ?? ''}" views="${d.views ?? ''}" goods="${d.goods ?? ''}" tracking="${d.tracking ?? ''}"></div>`;
    body += `</div>`;
    return body;
}