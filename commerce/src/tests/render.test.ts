import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { isAlmostEqual, item2html, parseStatus, selector, time2text } from "../lib/render";

const NOW = Date.UTC(2026, 9, 9, 12, 0, 0); // 2026-10-09T12:00:00Z
const DAY = 86_400_000;
const XSS = "<img src=x onerror=alert(1)>";

/** Parse into an inert <template> (no resource loading / handlers) for DOM assertions. */
function render(item: any, checked = false): DocumentFragment {
  const tpl = document.createElement("template");
  tpl.innerHTML = item2html(item, checked);
  return tpl.content;
}
const text = (root: ParentNode, sel: string) => root.querySelector(sel)?.textContent ?? null;

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW);
});
afterEach(() => vi.useRealTimers());

describe("item2html", () => {
  it("renders a goods card: data attributes, parsed status, linked title, priced unit, More toggle", () => {
    const root = render({
      id: "d1",
      type: "goods",
      created_at: NOW - 150_000,
      updated_at: NOW - 1_000,
      data: { title: "Shirt", sale_price: 1000, currency: "KRW", status: 9, link: "/p/1", quantity: 0, width: null, weight: "null" },
    });
    const card = root.querySelector<HTMLElement>(`#d1.${selector.result}`)!;
    expect(card).not.toBeNull();
    expect(card.dataset.type).toBe("goods");
    expect(card.dataset.mode).toBe("commerce");
    expect(card.dataset.createdAt).toBe(String(NOW - 150_000));
    expect(card.dataset.updatedAt).toBe(String(NOW - 1_000));
    expect(root.querySelector("input#more-d1.toggle-more")).not.toBeNull();

    expect(text(card, ".logis-info.status strong")).toBe("goods");
    expect(text(card, ".logis-info.status .value")).toBe("complete");
    expect(card.querySelector("a.logis-info.title")!.getAttribute("onclick")).toContain("detail: '/p/1'");
    expect(text(card, ".logis-info.title .value")).toBe("Shirt");
    expect(text(card, ".logis-info.sale_price .unit")).toBe(" (KRW)");
    expect(text(card, ".more-content .logis-info.quantity .value")).toBe("0"); // 0 is a real value
    expect(card.querySelector(".logis-info.width")).toBeNull(); // null is skipped
    expect(card.querySelector(".logis-info.weight")).toBeNull(); // "null" is skipped

    expect(text(card, ".logis-info.created_at strong")).toBe("2 minutes");
    expect(card.querySelector("label.more-label")!.getAttribute("for")).toBe("more-d1");
    expect(card.querySelector<HTMLInputElement>('input[type="hidden"][name="field-created-at"]')!.value).toBe(
      String(NOW - 150_000),
    );
  });

  it("renders a B/L trading document: doc_type badge, parsed status, dates, units and relate anchor", () => {
    const root = render({
      id: "b1",
      type: "BL",
      created_at: NOW - 2 * DAY,
      data: {
        status: 9,
        vessel: "EVER GIVEN",
        etd: "2026-10-01T00:00:00Z",
        amount: 5,
        currency: "USD",
        package_count: 10,
        package_unit: "CTN",
        index: 7,
        goods: 3,
      },
    });
    const card = root.querySelector("#b1")!;
    expect(text(card, ".doc_type .value")).toBe("BL");
    expect(text(card, ".status strong")).toBe("BL");
    expect(text(card, ".status .value")).toBe("complete");
    expect(text(card, ".vessel .value")).toBe("EVER GIVEN");
    expect(text(card, ".more-content .etd .value")).toBe("8 days");
    expect(text(card, ".amount .unit")).toBe(" (USD)");
    expect(text(card, ".package_count .unit")).toBe(" (CTN)");
    expect(text(card, ".created_at strong")).toBe("2 days");
    const relate = card.querySelector(`.${selector.relate}`)!;
    expect(relate.getAttribute("index")).toBe("7");
    expect(relate.getAttribute("goods")).toBe("3");
    expect(relate.getAttribute("event")).toBe("");
  });

  it("routes types to their templates (trade code in any case, coupon, receiving, click, unknown)", () => {
    expect(text(render({ id: "b2", type: "bl", data: { doc_number: "X1" } }), ".doc_number .value")).toBe("X1");

    const ev = render({ id: "e1", type: "coupon", data: { status: 1, title: "Sale", discount: 10, code: "C1" } });
    expect(text(ev, ".status .value")).toBe("progress");
    expect(text(ev, ".discount .value")).toBe("10");
    expect(text(ev, ".more-content .code .value")).toBe("C1");

    const tr = render({ id: "t1", type: "receiving", data: { status: "2", text: "In transit", sender_name: "A" } });
    expect(text(tr, ".status .value")).toBe("stop");
    expect(text(tr, ".text .value")).toBe("In transit");
    expect(text(tr, ".more-content .sender_name .value")).toBe("A");

    const an = render({ id: "a1", type: "click", mode: "analytic", data: { action: "clicked buy", summary: "S" } });
    expect(an.querySelector<HTMLElement>("#a1")!.dataset.mode).toBe("analytic");
    expect(text(an, ".action .value")).toBe("clicked buy");
    expect(text(an, ".summary .value")).toBe("S");

    const unknown = render({ id: "u1", type: "weird", data: { status: 3, title: "T", vessel: "V" } });
    expect(text(unknown, ".status strong")).toBe("weird");
    expect(text(unknown, ".status .value")).toBe("cancel");
    expect(text(unknown, ".title .value")).toBe("T");
    expect(unknown.querySelector(".vessel")).toBeNull();
    expect(unknown.querySelector(".more-content")).toBeNull();
  });

  it("prefers root fields over data.* and falls back to data.mode / data.created_at", () => {
    expect(text(render({ id: "p1", type: "goods", title: "ROOT", data: { title: "DATA" } }), ".title .value")).toBe("ROOT");
    const card = render({ id: "m1", type: "goods", data: { mode: "shipping", created_at: 5, updated_at: 6 } }).querySelector<HTMLElement>("#m1")!;
    expect(card.dataset.mode).toBe("shipping");
    expect(card.dataset.createdAt).toBe("5");
    expect(card.dataset.updatedAt).toBe("6");
  });

  it("derives the element id from id -> uuid -> data.id -> index -> random", () => {
    const idOf = (item: any) => render(item).querySelector(`.${selector.result}`)!.id;
    expect(idOf({ uuid: "U9", type: "goods", data: {} })).toBe("U9");
    expect(idOf({ type: "goods", data: { id: "D9" } })).toBe("D9");
    expect(idOf({ type: "goods", index: 42, data: {} })).toBe("42");
    vi.spyOn(Math, "random").mockReturnValue(0.123456789);
    expect(idOf({ type: "goods", data: {} })).toBe("4fzzzxjyl");
  });

  it("checked=true renders the More toggle as disabled+checked", () => {
    const toggle = render({ id: "c1", type: "goods", data: { title: "T" } }, true).querySelector<HTMLInputElement>("#more-c1")!;
    expect(toggle.hasAttribute("disabled")).toBe(true);
    expect(toggle.hasAttribute("checked")).toBe(true);
  });

  it("HTML-escapes string field values", () => {
    const html = item2html({ id: "x", type: "goods", data: { title: "<script>alert(1)</script>", sale_price: "\"&'" } });
    expect(html).toContain("&lt;script&gt;alert(1)&lt;/script&gt;");
    expect(html).not.toContain("<script>");
    expect(html).toContain("&quot;&amp;&#39;");
  });

  it.fails("BUG-RENDER-1 render.ts:110,187 item.type is interpolated unescaped (data-type attribute and status label) -> XSS", () => {
    expect(item2html({ id: "x1", type: XSS, data: { status: 1 } })).not.toContain(XSS);
  });

  it.fails("BUG-RENDER-2 render.ts:391-395 data.search_badge is interpolated unescaped -> XSS", () => {
    expect(item2html({ id: "x2", type: "goods", data: { search_badge: XSS } })).not.toContain(XSS);
  });

  it.fails("BUG-RENDER-3 render.ts:192-206 non-string values (arrays/objects) bypass escaping -> XSS", () => {
    expect(item2html({ id: "x3", type: "click", data: { action: "a", relate: [XSS] } })).not.toContain(XSS);
  });

  it.fails("BUG-RENDER-4 render.ts:167 data.link is pasted into an inline onclick JS string; a quote breaks out", () => {
    const html = item2html({ id: "l1", type: "goods", data: { title: "T", link: "x'});alert(1);//" } });
    expect(html).not.toContain("x'});alert(1);//");
  });

  it.fails("BUG-RENDER-5 render.ts:194 backslashes are doubled in displayed values ('C:\\path' shows 'C:\\\\path')", () => {
    expect(text(render({ id: "bs", type: "goods", data: { title: "C:\\path" } }), ".title .value")).toBe("C:\\path");
  });

  it.fails("BUG-RENDER-6 render.ts:179 future dates (eta 14 days ahead) render as negative seconds", () => {
    const eta = text(render({ id: "f1", type: "BL", data: { eta: NOW + 14 * DAY } }), ".eta .value");
    expect(eta).not.toBeNull();
    expect(eta).not.toMatch(/^-/);
  });

  it.fails("BUG-RENDER-7 render.ts:243-251 item2html mutates the caller's item.data (status -> string, doc_type added)", () => {
    const item = { id: "b9", type: "BL", data: { status: 9 } };
    const before = structuredClone(item);
    item2html(item);
    expect(item).toEqual(before);
  });
});

describe("parseStatus (loose ==, render.ts)", () => {
  it.each<[unknown, string]>([
    [9, "complete"],
    ["9", "complete"],
    [12, "hide"],
    ["abc", "abc"],
    [13, "13"],
    [0, "0"],
    [null, ""],
    [undefined, ""],
  ])("parseStatus(%j) -> %j", (input, expected) => {
    expect(parseStatus(input)).toBe(expected);
  });
});

describe("time2text (render.ts)", () => {
  it("formats elapsed time like utils.time2text", () => {
    expect(time2text(NOW - 150_000)).toBe("2 minutes");
    expect(time2text(NOW - 3 * DAY)).toBe("3 days");
  });
});

describe("isAlmostEqual", () => {
  it("treats records that differ only in one value (the id) as the same message", () => {
    expect(isAlmostEqual({ role: "user", text: "hi", id: "talk_1" }, { role: "user", text: "hi", id: "0xabc" })).toBe(true);
    expect(isAlmostEqual({ a: 1 }, { a: 1 })).toBe(true);
  });

  it("rejects two differing values, different key counts, empty and nullish inputs", () => {
    expect(isAlmostEqual({ role: "user", text: "hi", id: "talk_1" }, { role: "system", text: "hi", id: "0xabc" })).toBe(false);
    expect(isAlmostEqual({ a: 1 }, { a: 1, b: 2 })).toBe(false);
    expect(isAlmostEqual({}, {})).toBe(false);
    expect(isAlmostEqual(null, { a: 1 })).toBe(false);
  });

  it.fails("BUG-RENDER-8 render.ts:75-80 keys missing from obj2 are not counted, so disjoint objects compare 'almost equal'", () => {
    expect(isAlmostEqual({ x: 1 }, { y: 2 })).toBe(false);
  });
});
