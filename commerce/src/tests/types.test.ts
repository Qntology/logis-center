import { describe, expect, it } from "vitest";
import {
  ANALYTIC_TYPE_SET,
  COMMERCE_TYPE_SET,
  MODE_LABEL,
  TRADING_DOC_CODES,
  TRADING_DOC_TYPE_SET,
  TYPE_SETS,
  modeLabel,
  modeOfType,
} from "../modes/types";

describe("modeOfType", () => {
  it.each([
    ["click", "analytic"],
    ["hover", "analytic"],
    ["report", "analytic"],
    ["touch", "analytic"],
    ["question", "analytic"],
    ["answer", "analytic"],
    ["sales", "commerce"],
    ["goods", "commerce"],
    ["order", "commerce"],
    // commerce delivery-tracking docs must stay commerce (see the comment in commerce.ts MODE TAGGING v2)
    ["tracking", "commerce"],
    ["receiving", "commerce"],
    ["shipping", "commerce"],
    ["talk", "commerce"],
    ["team", "commerce"],
    ["page", "commerce"],
    ["TRACKING", "shipping"],
    ["BL", "shipping"],
    ["bl", "shipping"],
    ["CINV", "shipping"],
    ["cinv", "shipping"],
    ["BEN_CERT", "shipping"],
    ["shipping_doc", "shipping"],
    ["unknown", "shipping"],
    ["Unknown", "shipping"],
    ["Bl", "commerce"],
    ["Goods", "commerce"],
    ["something-else", "commerce"],
    ["", "commerce"],
  ])("modeOfType(%j) -> %s", (type, mode) => {
    expect(modeOfType(type)).toBe(mode);
  });

  it("treats nullish input as an empty type (commerce)", () => {
    expect(modeOfType(undefined as unknown as string)).toBe("commerce");
    expect(modeOfType(null as unknown as string)).toBe("commerce");
  });
});

describe("modeLabel", () => {
  it.each([
    ["commerce", "Commerce"],
    ["shipping", "Trading"],
    ["analytic", "Analytic"],
    ["foo", "Foo"],
    ["", ""],
  ])("modeLabel(%j) -> %j", (mode, label) => {
    expect(modeLabel(mode)).toBe(label);
  });

  it("MODE_LABEL covers exactly the three tracks", () => {
    expect(Object.keys(MODE_LABEL).sort()).toEqual(["analytic", "commerce", "shipping"]);
  });

  it.fails("BUG-TYPES-1 types.ts:76 modeLabel('toString') returns Object.prototype.toString (a function) instead of a label", () => {
    expect(modeLabel("toString")).toBe("ToString");
  });
});

describe("type taxonomy invariants", () => {
  it("TRADING_DOC_CODES holds 55 unique upper-case codes", () => {
    expect(TRADING_DOC_CODES).toHaveLength(55);
    expect(new Set(TRADING_DOC_CODES).size).toBe(55);
    for (const code of TRADING_DOC_CODES) expect(code).toBe(code.toUpperCase());
  });

  it("TRADING_DOC_TYPE_SET registers every code in both cases plus the 4 extra markers", () => {
    for (const code of TRADING_DOC_CODES) {
      expect(TRADING_DOC_TYPE_SET.has(code)).toBe(true);
      expect(TRADING_DOC_TYPE_SET.has(code.toLowerCase())).toBe(true);
    }
    expect(TRADING_DOC_TYPE_SET.size).toBe(55 * 2 + 4);
  });

  it("TYPE_SETS (read side) agrees with modeOfType (write side)", () => {
    for (const t of TYPE_SETS.analytic) expect(modeOfType(t)).toBe("analytic");
    for (const t of TYPE_SETS.commerce) expect(modeOfType(t)).toBe("commerce");
    for (const t of TYPE_SETS.shipping) {
      // tracking / receiving / shipping are deliberately listed for both tracks
      const expected = COMMERCE_TYPE_SET.has(t) ? "commerce" : "shipping";
      expect(modeOfType(t)).toBe(expected);
    }
  });

  it("analytic and commerce sets are disjoint", () => {
    for (const t of ANALYTIC_TYPE_SET) expect(COMMERCE_TYPE_SET.has(t)).toBe(false);
  });
});
