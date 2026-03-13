import { parseStatus, time2text } from "./utils";

// --- Static Selectors ---
export const selector = {
    app: "logis-app",
    mobile: "logis-mobile",
    desktop: "logis-desktop",
    result: "logis-result",
    info: "logis-info",
    relate: "logis-relate",
    more: "logis-more",
    active: "active",
    visited: "visited",
    completed: "completed",
    checkbox: "logis-checkbox",
    label: "logis-label",
    created_at: "field-created-at",
    status: "field-status",
    title: "field-title",
    currency: "field-currency"
};

// --- Main Rendering Function ---
export function item2html(item: any, checked: boolean, currentUrl: string = ""): string {
    let more = false;
    let href = "";
    
    if (item.data && item.data.link) {
        href = item.data.link;
    } else if (item.link) {
        href = item.link;
    }

    if (href && currentUrl) {
        if (currentUrl.includes(href)) more = true;
    }

    let itemType = item.type || item.doc_type || "unknown";
    if (itemType === "sales" || itemType === "goods" || itemType === "order") {
        itemType = "sales";
    } else if (itemType === "event" || itemType === "coupon") {
        itemType = "event";
    } else if (itemType === "receiving" || itemType === "shipping") {
        itemType = "tracking";
    }

    function Tpl(data: any, key: string, unit: string = ""): string {
        let value = "";
        let unitText = "";
        let name = key.replace(/_/g, " ");

        if (data[key] !== undefined) value = data[key];
        else if (data.data && data.data[key] !== undefined) value = data.data[key];

        if (value === "" || value === null || value === undefined) return "";

        if (key === "status") {
            value = parseStatus(value);
            name = data.type || "Status";
        }

        if (unit) {
            if (data[unit] !== undefined) unitText = ` (${data[unit]})`;
            else if (data.data && data.data[unit] !== undefined) unitText = ` (${data.data[unit]})`;
        }

        if (["created_at", "updated_at", "started_at", "expired_at", "release_date"].includes(key)) {
            value = time2text(value);
        }

        let tagName = "div";
        let props = "";
        let content = "";

        if (key === "title") {
            tagName = "a";
            if (data.link || (data.data && data.data.link)) {
                const targetLink = data.link || data.data.link;
                props = `href="#" onclick="document.dispatchEvent(new CustomEvent('nav-link', {detail: '${targetLink}'})); return false;"`;
            }
        }

        content = `<span class="value">${value}</span><i class="unit">${unitText}</i>`;

        return `
            <${tagName} ${props} class="${selector.info} ${key}">
                <strong>${name}</strong>
                ${content}
            </${tagName}>
        `;
    }

    let body = `<div class="${selector.result} ${itemType}" id="${item.uuid || item.id}">`;
    const uniqueId = `more-${item.id || Math.random().toString(36).substr(2, 9)}`;
    body += `<input type="checkbox" id="${uniqueId}" class="toggle-more" ${more ? 'checked' : ''} style="display:none;" />`;

    if (itemType === "sales") {
        body += `
            ${Tpl(item, "status")}
            ${Tpl(item, "title")}
            ${Tpl(item, "sale_price", "currency")}
            ${Tpl(item, "created_at")}
            <label for="${uniqueId}" class="more-label">▼ details</label>
            <div class="${selector.more}">
                ${Tpl(item, "price", "currency")}
                ${Tpl(item, "quantity")}
                ${Tpl(item, "stock_keeping_unit")}
                ${Tpl(item, "shipping_fee", "currency")}
                ${Tpl(item, "shipping_method")}
                ${Tpl(item, "tax_included")}
            </div>
        `;
    } else if (itemType === "tracking") {
        body += `
            ${Tpl(item, "status")}
            ${Tpl(item, "title")}
            ${Tpl(item, "carrier")}
            ${Tpl(item, "created_at")}
            <label for="${uniqueId}" class="more-label">▼ details</label>
            <div class="${selector.more}">
                ${Tpl(item, "text")}
                ${Tpl(item, "sender_name")}
                ${Tpl(item, "recipient_name")}
                ${Tpl(item, "tracking_number")}
            </div>
        `;
    } else if (itemType === "event") {
        body += `
            ${Tpl(item, "status")}
            ${Tpl(item, "title")}
            ${Tpl(item, "discount")}
            ${Tpl(item, "expired_at")}
            <label for="${uniqueId}" class="more-label">▼ details</label>
            <div class="${selector.more}">
                ${Tpl(item, "code")}
                ${Tpl(item, "min_order_amount")}
                ${Tpl(item, "usage_limit")}
            </div>
        `;
    } else {
        body += `
            ${Tpl(item, "id")}
            ${Tpl(item, "type")}
            ${Tpl(item, "created_at")}
            <div style="font-size:0.8rem; padding:10px; color:#666;">${item.text || ""}</div>
        `;
    }

    body += `<input type="hidden" readonly name="${selector.created_at}" value="${item.created_at}" />`;
    body += `</div>`;

    return body;
}