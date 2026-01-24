import { parseStatus, time2text } from "./utils";

// --- Static Selectors (Option B) ---
export const selector = {
    // Layout
    app: "logis-app",
    mobile: "logis-mobile",
    desktop: "logis-desktop",
    
    // Components
    result: "logis-result",
    info: "logis-info",
    relate: "logis-relate",
    more: "logis-more",
    
    // States
    active: "active",
    visited: "visited",
    completed: "completed",
    
    // Input/Form
    checkbox: "logis-checkbox",
    label: "logis-label",
    
    // Specific Fields (for styling hooks)
    created_at: "field-created-at",
    status: "field-status",
    title: "field-title",
    currency: "field-currency"
};

// --- Helper: Check if two URLs/Objects are "almost" equal (for 'more' logic) ---
function isAlmostEqual(obj1: any, obj2: any): boolean {
    if (!obj1 || !obj2) return false;
    const k1 = Object.keys(obj1);
    const k2 = Object.keys(obj2);
    if (k1.length !== k2.length) return false;
    
    let diff = 0;
    for (const key of k1) {
        if (obj1[key] !== obj2[key]) diff++;
        if (diff > 1) return false;
    }
    return true;
}

// --- Main Rendering Function ---
export function item2html(item: any, checked: boolean, currentUrl: string = ""): string {
    // 1. Determine "More" visibility based on URL context
    // In content.js this checked window.location against item link.
    // Here we use the passed `currentUrl` state.
    let more = false;
    let href = "";
    
    if (item.data && item.data.link) {
        href = item.data.link;
    } else if (item.link) {
        href = item.link;
    }

    if (href && currentUrl) {
        // Simple inclusion check for now (Can be enhanced with full URL parsing)
        if (currentUrl.includes(href)) more = true;
    }

    // 2. Normalize Types
    let itemType = item.type || "unknown";
    if (itemType === "sales" || itemType === "goods" || itemType === "order") {
        itemType = "sales";
    } else if (itemType === "event" || itemType === "coupon") {
        itemType = "event";
    } else if (itemType === "receiving" || itemType === "shipping") {
        itemType = "tracking";
    }

    // 3. Template Helper
    function Tpl(data: any, key: string, unit: string = ""): string {
        let value = "";
        let unitText = "";
        let name = key.replace(/_/g, " ");

        // Resolve Value
        if (data[key] !== undefined) value = data[key];
        else if (data.data && data.data[key] !== undefined) value = data.data[key];

        if (value === "" || value === null || value === undefined) return "";

        // Format Status
        if (key === "status") {
            value = parseStatus(value);
            name = data.type || "Status"; // Use type as label for status row
        }

        // Resolve Unit
        if (unit) {
            if (data[unit] !== undefined) unitText = ` (${data[unit]})`;
            else if (data.data && data.data[unit] !== undefined) unitText = ` (${data.data[unit]})`;
        }

        // Format Dates
        if (["created_at", "updated_at", "started_at", "expired_at", "release_date"].includes(key)) {
            value = time2text(value);
        }

        // Construct HTML
        let tagName = "div";
        let props = "";
        let content = "";

        if (key === "title") {
            tagName = "a";
            // If it's a link, we might want to trigger a navigation event instead of href
            if (data.link || (data.data && data.data.link)) {
                const targetLink = data.link || data.data.link;
                props = `href="#" onclick="document.dispatchEvent(new CustomEvent('nav-link', {detail: '${targetLink}'})); return false;"`;
            }
        }

        // Input vs Text logic (Simplified for Read-Only view)
        // content.js used inputs. We will use spans for cleaner display unless editable.
        content = `<span class="value">${value}</span><i class="unit">${unitText}</i>`;

        return `
            <${tagName} ${props} class="${selector.info} ${key}">
                <strong>${name}</strong>
                ${content}
            </${tagName}>
        `;
    }

    // 4. Build Body
    let body = `<div class="${selector.result} ${itemType}">`;
    
    // Checkbox for expansion (UI state)
    const uniqueId = `more-${item.id || Math.random().toString(36).substr(2, 9)}`;
    // We use a hidden checkbox hack for CSS-only expansion toggling
    body += `<input type="checkbox" id="${uniqueId}" class="toggle-more" ${more ? 'checked' : ''} style="display:none;" />`;

    // --- Content by Type ---
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
        // Fallback generic
        body += `
            ${Tpl(item, "id")}
            ${Tpl(item, "type")}
            ${Tpl(item, "created_at")}
        `;
    }

    // Hidden meta fields for indexing/relations
    body += `<input type="hidden" readonly name="${selector.created_at}" value="${item.created_at}" />`;
    body += `<div class="${selector.relate}" index="${item.index}" event="${item.event}"></div>`;

    body += `</div>`; // Close card

    return body;
}
