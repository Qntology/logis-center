console.log("%c[WIDGET] MAIN.TS LOADED", "color: #00ff00; font-weight: bold; font-size: 1.2rem;");
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { open, ask } from '@tauri-apps/plugin-dialog';
import { listen, emit } from '@tauri-apps/api/event';
import { readFile } from '@tauri-apps/plugin-fs';

// Imports for Rendering & Shim
// 🌟 isAlmostEqual : 낙관적 로컬 talk 행 ↔ 서버 발급 talk 행 승계 판정에 사용합니다.
//    (render.ts 에 존재하지만 그동안 한 번도 호출되지 않던 죽은 함수였습니다)
import { item2html, selector, isAlmostEqual } from "./lib/render";
import { Select, Upsert } from "./lib/db";
import { hashId, time2text } from "./lib/utils";

// 🌟 [CANONICALIZE] data 안의 값 타입을 확정합니다.
//  IndexedDB 인덱스는 타입이 혼재하면(123 vs "123") equals 가 절반을 놓치고,
//  boolean / undefined 는 아예 인덱스에서 조용히 빠집니다.
//  따라서 '쓰기 시점'에 딱 한 번 정규화해서 저장합니다.
//  (Rust upsert_item 도 동일 규칙을 적용해야 양쪽 결과가 일치합니다 — Part 2 참조)
// 🌟 [CANONICAL CONTRACT / TS]
//  ⚠️ 이 파일의 판정은 store.rs 가 쓰는 utils/canonical.rs 의 kind_of() 와
//     '비트 단위로 동일' 해야 합니다. 한쪽만 바뀌면 같은 값이
//     LanceDB 에는 Number, Dexie 에는 String 으로 저장되어
//     where('data.xxx').equals(...) 가 절반을 놓칩니다.
//
//  ── 왜 이름 목록을 없앴나 ──
//   기존에는 ID/NUM/BOOL 배열에 필드명을 일일이 등록해야 했습니다.
//   이제 접미사/부분일치 규칙으로 자동 판정하므로,
//   Dexie 에 새 필드를 추가해도 이 목록을 건드릴 필요가 없습니다.
type CanonKind = 'id' | 'num' | 'bool' | 'tags' | 'free';

// ── ① 명시 예외 : 규칙만으로 판정 불가능한 이름 (새 필드로 커지지 않습니다) ──
const FORCE_ID = new Set(['id', 'no', 'digest']);
const FORCE_NUM = new Set(['status', 'views', 'created_at', 'updated_at', 'index', 'goods', 'order', 'tracking']);
const FORCE_BOOL = new Set(['detail', 'node', 'embed']);

// ── ② 접미사 / 부분일치 규칙 : 새 필드는 여기에 자동으로 걸립니다 ──
const ID_SUFFIX = ['_no', '_code', '_number', '_id', '_sku', '_barcode', '_gtin', '_mpn'];
const ID_CONTAINS = ['code', 'barcode', 'gtin', 'mpn', 'sku', 'reference_', 'container', 'seal'];
const NUM_SUFFIX = [
    '_price', '_amount', '_fee', '_rate', '_count', '_qty', '_at',
    '_weight', '_volume', '_duration', '_limit', '_threshold', '_charges',
    // 🌟 [UNIT / CURRENCY SUFFIX] 무역 서식은 값 이름에 단위를 붙이는 관례가 있습니다.
    //    total_gross_weight_kg / measurement_cbm / entered_value_usd / amount_krw
    //    이 넷은 기존 어느 규칙에도 걸리지 않아 'free' 로 떨어졌고,
    //    LanceDB 에는 Number, Dexie 에는 String 으로 갈라져
    //    where('data.xxx').between(...) 이 통째로 실패했습니다.
    //    ⚠️ canonical.rs 의 NUM_SUFFIX 와 반드시 같은 집합이어야 합니다.
    '_kg', '_cbm', '_m3', '_usd', '_krw', '_eur', '_jpy', '_cny', '_gbp'
];
const NUM_CONTAINS = [
    'price', 'amount', 'quantity', 'discount', 'weight', 'volume',
    'shipping_fee', 'usage_', 'threshold', 'exchange_rate', 'package_count',
    'local_charges', 'number_of_',
    // 🌟 [PACKAGE FAMILY] total_packages / packages_delivered / packages_received /
    //    total_pieces / number_of_pieces 는 전부 개수입니다.
    //    'package_count' 완전일치만 있어서 나머지가 전부 새고 있었습니다.
    'packages', 'pieces',
    // 🌟 [MEASUREMENT / VALUE] measurement / premium / duty / dutiable / balance /
    //    flash_point 는 값이 항상 수치입니다.
    'measurement', 'premium', 'duty_', 'dutiable', 'balance', 'flash_point',
    'tare_weight', 'chargeable'
];
// 🌟 단독 명사형 수치 축. canonical.rs 의 NUM_EXACT 와 동일 집합입니다.
const NUM_EXACT = new Set([
    'width', 'height', 'length',
    'premium', 'rate', 'debit', 'credit', 'dosage'
]);
const BOOL_PREFIX = ['is_', 'has_', 'allow_', 'use_'];
// 🌟 [TRADING INDEX PREFIX] canonical.rs 의 NUM_PREFIX 와 반드시 동일해야 합니다.
//    trading_index_column() 이 만드는 'rel_ci' / 'rel_bl' 은 crc32 숫자 인덱스입니다.
const NUM_PREFIX = ['rel_'];
// 🌟 [_shipping 제거] 이 접미사에 걸리는 실제 필드는 bundle_shipping 하나뿐인데,
//    추출값이 "묶음배송가능" / "불가" 같은 자연어 문자열이라
//    bool 변환식이 두 값을 모두 0 으로 만들어 구분을 통째로 없앴습니다.
//    bias_schema.rs 도 add("bundle_shipping", "String", ...) 로 선언하므로
//    문자열로 두는 것이 Rust 쪽과도 일치합니다.
const BOOL_SUFFIX = ['_only', '_included', '_allowed', '_match'];

function kindOf(key: string): CanonKind {
    const k = key.toLowerCase();

    if (k === 'tags') return 'tags';

    if (FORCE_ID.has(k)) return 'id';
    if (FORCE_NUM.has(k)) return 'num';
    if (FORCE_BOOL.has(k)) return 'bool';

    // 🌟 [TRADING INDEX] 'rel_ci' / 'rel_bl' 은 crc32 숫자 인덱스입니다.
    //    ID_CONTAINS 의 'reference_' 보다 먼저 검사해야
    //    'rel_' 이 id(String)로 오분류되지 않습니다.
    if (NUM_PREFIX.some(p => k.startsWith(p))) return 'num';

    // 🌟 Boolean 을 수치보다 먼저 검사합니다.
    //    ('recipient_match' 처럼 실제 참/거짓인 필드만 여기에 걸립니다)
    if (BOOL_PREFIX.some(p => k.startsWith(p))) return 'bool';
    if (BOOL_SUFFIX.some(s => k.endsWith(s))) return 'bool';

    if (NUM_EXACT.has(k)) return 'num';
    if (NUM_SUFFIX.some(s => k.endsWith(s))) return 'num';

    // 🌟 식별자를 수치보다 먼저 봅니다.
    //    'doc_number' 는 '_number' 로 수치에도 걸리지만
    //    실제로는 'ABCD1234567' 영숫자 혼합이므로 String 이어야 합니다.
    if (ID_SUFFIX.some(s => k.endsWith(s))) return 'id';
    if (ID_CONTAINS.some(c => k.includes(c))) return 'id';

    if (NUM_CONTAINS.some(c => k.includes(c))) return 'num';

    return 'free';
}

// 🌟 [SEED KEYS] Dexie stores() 의 data.* 인덱스 중 기본값이 필요한 키만 나열합니다.
//    store.rs 의 SEED_KEYS 와 동일해야 합니다.
const SEED_KEYS: Array<[string, CanonKind]> = [
    ['id', 'id'], ['no', 'id'], ['code', 'id'],
    ['tracking_number', 'id'], ['stock_keeping_unit', 'id'], ['barcode', 'id'], ['digest', 'id'],
    ['index', 'num'], ['goods', 'num'], ['order', 'num'], ['tracking', 'num'],
    ['status', 'num'], ['created_at', 'num'], ['updated_at', 'num'],
    ['embed', 'bool'],
    ['tags', 'tags']
];

// 🌟 [STATUS PARITY] store.rs 의 crate::logic::parse_status 와 1:1 로 동일한 표입니다.
//    두 표가 어긋나면 같은 문서가 LanceDB 에서는 9, Dexie 에서는 0 으로 저장되어
//    data.status 인덱스 조회가 절반을 놓칩니다.
const STATUS_CODE: Record<string, number> = {
    progress: 1, stop: 2, cancel: 3, refund: 4, return: 5,
    error: 6, expire: 7, exchange: 8, complete: 9,
    draft: 10, show: 11, hide: 12
};

// 🌟 기본값 시딩을 하지 않는 타입. store.rs 의 `matches!(target, "users" | "pages")` 와 대응합니다.
//    🌟 [ANALYTICS] click / hover / change / report 는 행동 로그이므로
//       commerce 도메인 필드(sale_price / tracking_number ...)를 가질 이유가 전혀 없습니다.
//       시딩하면 문서당 48개의 무의미한 키가 붙어 저장 용량과 인덱스를 낭비합니다.
//       Dexie 는 없는 키를 인덱스에서 조용히 제외할 뿐 에러를 내지 않으므로 시딩이 불필요합니다.
const NON_SEED_TYPES = new Set([
    'team', 'user', 'member', 'users', 'pages', 'page',
    'click', 'hover', 'change', 'report', 'question', 'answer'
]);

// 🌟 [ISO DATE] Rust 의 iso_to_epoch_ms 와 동일 규칙.
//    scheduler 가 started_at / expired_at 을 "2024-01-01T12:00:00" 으로 만듭니다.
//    Number("2024-01-01120000") = NaN → 0 이 되어 기간 조건이 통째로 죽었습니다.
function isoToEpochMs(t: string): number | null {
    if (!/^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?/.test(t)) return null;
    let ms: number;
    if (t.length === 10) {
        ms = Date.parse(t); // "2024-01-01" 은 명세상 UTC 로 해석됩니다.
    } else {
        const hasTz = /[Zz]$|[+\-]\d{2}:?\d{2}$/.test(t);
        const norm = t.includes('T') ? t : t.replace(' ', 'T');
        // 타임존이 없으면 UTC 로 강제해야 Rust(and_utc) 결과와 1ms 도 어긋나지 않습니다.
        ms = Date.parse(hasTz ? norm : norm + 'Z');
    }
    return isNaN(ms) ? null : ms;
}

function canonicalizeData(parsed: any, seedDefaults: boolean = true): any {
    if (!parsed || typeof parsed !== 'object') return {};
    const out: any = { ...parsed };

    // ── ① 기존 키 전량 정규화 (규칙 기반) ──
    //    새 필드도 여기서 자동 처리되므로 이 함수는 확장 시 수정할 필요가 없습니다.
    for (const k of Object.keys(out)) {
        const kind = kindOf(k);
        if (kind === 'free') continue;

        const v = out[k];

        if (kind === 'id') {
            if (v === undefined || v === null) continue;
            // 배열/객체는 식별자가 될 수 없으므로 건드리지 않습니다.
            if (typeof v === 'object') continue;
            out[k] = String(v);
            continue;
        }

        if (kind === 'num') {
            if (v === undefined || v === null || v === "") continue;
            if (typeof v === 'object') continue;
            if (typeof v === 'number') { out[k] = v; continue; }
            if (typeof v === 'boolean') { out[k] = v ? 1 : 0; continue; }

            const s = String(v).trim();
            // 🌟 [STATUS PARITY] status 는 'complete' 같은 상태 문자열로 들어올 수 있습니다.
            if (k === 'status') {
                const mapped = STATUS_CODE[s.toLowerCase()];
                if (mapped !== undefined) { out[k] = mapped; continue; }
            }
            const ms = isoToEpochMs(s);
            if (ms !== null) { out[k] = ms; continue; }

            const n = Number(s.replace(/[^\d.\-]/g, ''));
            out[k] = isNaN(n) ? 0 : n;
            continue;
        }

        if (kind === 'bool') {
            if (v === undefined || v === null) continue;
            if (typeof v === 'object') continue;
            // 🌟 boolean 은 IDB 키가 아니므로 반드시 0|1 로 내립니다.
            out[k] = (v === true || v === 1 || v === "1" || v === "true") ? 1 : 0;
            continue;
        }

        if (kind === 'tags') {
            if (v === undefined || v === null) continue;
            if (!Array.isArray(v)) {
                out[k] = [String(v)].filter(Boolean);
            } else {
                out[k] = v
                    .map((t: any) => (typeof t === 'object' && t !== null ? (t.tag ?? "") : String(t)))
                    .filter(Boolean);
            }
            continue;
        }
    }

    // ── ② 조회 축 기본값 시딩 ──
    if (seedDefaults) {
        for (const [k, kind] of SEED_KEYS) {
            if (out[k] !== undefined && out[k] !== null) continue;
            out[k] = kind === 'id' ? "" : kind === 'tags' ? [] : 0;
        }
    }

    return out;
}

// =====================================================================
// 🌟 [MODE CONTRACT v1] 모드 ↔ 타입 매핑 단일 진실 공급원
// ---------------------------------------------------------------------
//  ── 왜 단일화하는가 ──
//   기존에는 syncData 의 TRADING_TYPES 와 loadMoreDocs 의 TYPE_SETS 가
//   각자 하드코딩되어 있었고, 두 목록 모두 'tracking' 을 포함했습니다.
//   그 결과 proxy/index.ts 의 Relay("tracking","order") 가 만든
//   'commerce 주문 배송추적' 문서가 mode='shipping' 으로 오염되어
//   commerce 목록에서 통째로 사라졌습니다.
//
//  ── 두 목록의 역할이 다르다 ──
//   MODE_OF_TYPE  : "이 문서를 어느 트랙에 소속시킬 것인가" (쓰기 시점, 배타적)
//   TYPE_SETS     : "이 모드에서 어떤 타입을 보여줄 것인가" (읽기 시점, 중첩 허용)
//   tracking 은 소속은 commerce 이지만, shipping 모드에서도 조회 대상입니다.
//   (mode 컬럼이 이미 걸러 주므로 조회 목록에 중복 등재해도 무해합니다)
//
//  ⚠️ D1(commerce / analytics) 어디에도 mode 컬럼이 없습니다.
//     따라서 mode 는 '동기화 시점에 클라이언트가 확정하는 값' 입니다.
// =====================================================================

// ── 무역 서식 코드 : app-logis-center 의 get_slice_config 분류 전량 ──
//    ① 계약·결제 ② 선적·운송 ③ 통관·신고 ④ 검사·증명 ⑤ 특수·법무
// 🌟 [TRADING DOC CODES v2] 27종 → 55종
//
//  ── 무엇이 문제였나 ──
//   modeOfType() 은 이 목록에 없는 타입을 `return 'commerce'` 로 떨굽니다.
//   그래서 HBL / SOA / TI / CDR / ICF / LLC 등 28종의 무역 서식이
//   mode='commerce' 로 태깅되어 Trading 탭에서 통째로 사라졌습니다.
//   TYPE_SETS.shipping 도 이 배열에서 파생되므로 타입 필터에서도 탈락하여
//   두 탭 어디에서도 보이지 않는 고아 문서가 됩니다.
//
//  ⚠️ src-tauri/src/logic.rs 의 TRADE_GROUP_CODES 및
//     bias_schema.rs 의 canonical_bias_type 매치 목록과 같은 집합이어야 합니다.
const TRADING_DOC_CODES = [
    // 계약 · 결제
    'PO', 'PI', 'SC', 'LC', 'LLC', 'CP', 'BE', 'TR', 'LG', 'EL',
    // 선적 · 운송
    'CI', 'PL', 'BL', 'HBL', 'SWB', 'AWB', 'SA', 'DO', 'AN',
    'BC', 'BK', 'SR', 'FCR', 'POD', 'CM', 'FI', 'WR',
    // 통관 · 신고
    'ED', 'ID', 'CINV', 'CO', 'CCC', 'CNM', 'CSI',
    // 검사 · 증명
    'IC', 'WC', 'CA', 'COA', 'PHYTO', 'PC', 'HC', 'BEN_CERT', 'FC', 'CDR',
    // 특수 · 법무 · 보험
    'DGD', 'MSDS', 'POA', 'BIZ_LIC', 'INS', 'IP', 'ICF',
    // 정산
    'SOA', 'DN', 'CN', 'TI'
];

// 🌟 LLM 분류기는 대문자('BL')로, scheduler 의 page_type 은 소문자('tracking')로
//    뱉는 두 경로가 공존하므로 양쪽을 모두 등재합니다.
const TRADING_DOC_TYPE_SET = new Set<string>([
    ...TRADING_DOC_CODES,
    ...TRADING_DOC_CODES.map(c => c.toLowerCase()),
    'shipping_doc', 'TRACKING', 'Unknown', 'unknown'
]);

// ── commerce 도메인 타입 : proxy/index.ts 의 Relay 대상 전량 ──
const COMMERCE_TYPE_SET = new Set<string>([
    'sales', 'goods', 'order', 'tracking', 'event', 'coupon', 'review',
    'receiving', 'shipping',
    'member', 'team', 'user', 'users', 'pages', 'page', 'talk', 'prompt', 'ai_search'
]);

// ── analytics 행동 로그 / 관리자 Q&A ──
const ANALYTIC_TYPE_SET = new Set<string>([
    'click', 'hover', 'change', 'report', 'touch', 'question', 'answer'
]);

/**
 * 🌟 [MODE TAGGING] D1 응답 행의 type 만 보고 소속 트랙을 확정합니다.
 *  commerce 를 먼저 판정해야 'tracking' 이 shipping 으로 새어 나가지 않습니다.
 *  (TRADING_DOC_TYPE_SET 에는 소문자 'tracking' 이 없으므로 충돌하지 않지만,
 *   순서를 고정해 두면 이후 코드가 추가되어도 안전합니다)
 */
function modeOfType(t: string): 'commerce' | 'shipping' | 'analytic' {
    const s = String(t || '');
    if (ANALYTIC_TYPE_SET.has(s)) return 'analytic';
    if (COMMERCE_TYPE_SET.has(s)) return 'commerce';
    if (TRADING_DOC_TYPE_SET.has(s)) return 'shipping';
    return 'commerce';
}

/**
 * 🌟 [READ SCOPE] 각 모드에서 목록/검색에 노출할 타입 목록입니다.
 *  mode 컬럼이 이미 트랙을 격리하므로 여기서는 중첩을 허용합니다.
 */
const TYPE_SETS: Record<string, string[]> = {
    shipping: [
        'tracking', 'receiving', 'shipping', 'shipping_doc', 'TRACKING',
        ...TRADING_DOC_CODES,
        ...TRADING_DOC_CODES.map(c => c.toLowerCase()),
        'Unknown', 'unknown'
    ],
    // 🌟 [Q&A VISIBLE] question / answer 는 콘솔에서 오간 관리자 Q&A 입니다.
    //    기존에는 syncAnalyticsData 가 아예 버렸고 이 목록에도 없어
    //    앱 어디에서도 확인할 수 없었습니다. 이제 저장하므로 목록에도 노출합니다.
    //    (검색 스코프에서는 parse_analytic_query 가 별도로 제외하므로 충돌하지 않습니다)
    // 🌟 [TOUCH VISIBLE] bias.json analytic_event_filters 에 정의된 touch 를
    //    목록에 포함합니다. 이 목록에서 빠지면 목록 조회에서 통째로 탈락합니다.
    analytic: ['click', 'hover', 'change', 'report', 'touch', 'question', 'answer'],
    // 🌟 [ORPHAN TYPE FIX] proxy/index.ts 가 택배 라벨에 붙이는 'receiving' / 'shipping' 은
    //    COMMERCE_TYPE_SET 에 있어 modeOfType 이 mode='commerce' 로 태깅하는데,
    //    이 읽기 목록에는 없어서 commerce 탭에서 조회되지 않았습니다.
    //    shipping 탭에는 타입은 있으나 mode 가 달라 역시 탈락 →
    //    결과적으로 두 탭 어디에서도 보이지 않는 고아 문서가 되었습니다.
    //    (mode 컬럼이 트랙을 이미 격리하므로 읽기 목록 중첩은 무해합니다)
    commerce: ['sales', 'goods', 'order', 'tracking', 'event', 'coupon', 'review',
        'receiving', 'shipping']
};

/**
 * 🌟 [MODE LABEL] 내부 코드값과 사용자 표기를 분리합니다.
 *  저장/쿼리 계약은 여전히 mode='shipping' 이므로 DB · Rust 변경이 없습니다.
 *  ⚠️ 이 표가 유일한 정의입니다. applySearchModeUI 와 검색 결과 헤더가 공유합니다.
 */
const MODE_LABEL: Record<string, string> = {
    commerce: 'Commerce',
    shipping: 'Trading',
    analytic: 'Analytic'
};

function modeLabel(m: string): string {
    return MODE_LABEL[m] || (m.charAt(0).toUpperCase() + m.slice(1));
}

// 🌟 [NORMALIZE ENVELOPE] Rust(TradeDocument) / Cloudflare(D1 row) / 로컬 생성 객체를
//  하나의 봉투 형태로 통일합니다. 루트에는 봉투 12개만 남기고, 나머지는 전부 data 로 내립니다.
//  → 값이 2벌 저장되던 문제(enrichForIndex 호이스팅)가 사라집니다.
//  → 새 도메인 필드가 생겨도 이 함수는 영원히 그대로입니다.
// 🌟 [ENVELOPE KEYS] 루트에 남겨 둘 봉투 12개.
//    이 집합에 없는 루트 키는 전부 '확장 필드' 로 간주해 data 로 하강시킵니다.
const ENVELOPE_ROOT_KEYS = new Set([
    'id', 'uuid', 'type', 'doc_type', 'flag', 'from', 'to', 'cc', 'bcc',
    'ref', 'ref_val', 'mode', 'created_at', 'updated_at',
    'created_at_ts', 'updated_at_ts',
    // 아래 4개는 data 안으로 별도 승격되므로 하강 루프에서 제외합니다.
    'data', 'json_data', 'text', 'masked_text',
    // 클라우드 응답 전용 라우팅 힌트 (도메인 값이 아님)
    'table', 'digest', 'current', 'score', 'name'
]);

const normalizeEnvelope = (docs: any[]) => docs.map(d => {
    let parsed: any = {};
    if (typeof d.json_data === 'string') {
        try { parsed = JSON.parse(d.json_data) || {}; } catch (e) { parsed = {}; }
    } else if (d.data && typeof d.data === 'object') {
        parsed = d.data;
    } else if (typeof d.data === 'string') {
        try { parsed = JSON.parse(d.data) || {}; } catch (e) { parsed = {}; }
    } else {
        parsed = {};
    }

    // 🌟 [ROOT ABSORB] Cloudflare D1 의 sales / tracking / event 테이블은
    //    확장 필드를 gzip data 가 아니라 '물리 컬럼' 으로만 저장합니다.
    //    (proxy/index.ts 의 INSERT INTO sales(... sale_price, quantity, weight ...) 참고)
    //    그래서 GET 응답에서는 그 값들이 행 최상위에 실려 옵니다.
    //    이 루프가 없으면 봉투 12개만 취하고 나머지를 통째로 버려서
    //    Dexie 의 data.sale_price / data.weight 조건이 전부 오판합니다.
    //    로컬 추출 경로(scheduler → upsert_item)는 이미 data 에 전부 넣으므로
    //    parsed 에 값이 있으면 그쪽을 우선하고, 없을 때만 루트에서 끌어옵니다.
    for (const k in d) {
        if (!Object.prototype.hasOwnProperty.call(d, k)) continue;
        if (ENVELOPE_ROOT_KEYS.has(k)) continue;
        const v = d[k];
        if (v === undefined || v === null) continue;
        // 함수/DOM 등 직렬화 불가 값은 IndexedDB structured clone 에서 터지므로 제외합니다.
        if (typeof v === 'function') continue;
        if (parsed[k] === undefined || parsed[k] === null || parsed[k] === "") {
            parsed[k] = v;
        }
    }

    // 검색/표시용 텍스트도 data 안으로 통일합니다.
    if (parsed.text === undefined) parsed.text = d.text ?? "";
    if (parsed.masked_text === undefined) parsed.masked_text = d.masked_text ?? parsed.text ?? "";
    if (parsed.mode === undefined) parsed.mode = d.mode ?? 'commerce';
    if (parsed.digest === undefined) parsed.digest = d.digest ?? "";
    const created = d.created_at_ts ?? d.created_at ?? parsed.created_at ?? 0;
    // 🌟 [DRAFT PRESERVE] updated_at 폴백 체인에서 created_at 으로 빠지는 경로를 제거합니다.
    //    기존에는 updated_at 이 없으면 created_at(현재 시각) 으로 폴백되어
    //    draft(updated_at=0) 가 count 로 승격되었습니다.
    //    updated_at 이 어디에도 없으면 0(draft) 으로 남겨야
    //    renderNavigation 의 Dexie 카운트가 올바릅니다.
    const updatedRaw = d.updated_at_ts ?? d.updated_at ?? parsed.updated_at;
    const updated = updatedRaw !== undefined && updatedRaw !== null ? updatedRaw : 0;
    parsed.created_at = Number(created) || 0;
    parsed.updated_at = Number(updated) || 0;

    return {
        // ── 봉투 12개. 이 목록은 앞으로 절대 늘어나지 않습니다 ──
        id: String(d.id ?? d.uuid ?? parsed.id ?? ""),
        type: String(d.type ?? d.doc_type ?? parsed.type ?? "unknown"),
        flag: String(d.flag ?? parsed.flag ?? ""),
        from: String(d.from ?? parsed.from ?? ""),
        to: String(d.to ?? parsed.to ?? ""),
        cc: String(d.cc ?? parsed.cc ?? ""),
        bcc: String(d.bcc ?? parsed.bcc ?? ""),
        ref: String(d.ref ?? d.ref_val ?? parsed.ref ?? ""),
        mode: String(d.mode ?? parsed.mode ?? 'commerce'),
        created_at: Number(created) || 0,
        updated_at: Number(updated) || 0,
        // ── 확장 영역. 여기에 뭘 넣든 스키마 변경 없음 ──
        //    users / team / pages 는 시딩을 끕니다. (통계 문서 오염 방지)
        data: canonicalizeData(parsed, !NON_SEED_TYPES.has(String(d.type ?? parsed.type ?? "")))
    };
});

// 🌟 [BACK-COMPAT] 기존 호출부(enrichForIndex(...))를 그대로 살려 둡니다.
//  호출부 치환은 Part 3 에서 일괄 정리합니다.
const enrichForIndex = normalizeEnvelope;

// Access global libs
const ethers = (window as any).ethers;
const blockies = (window as any).blockies;

// --- Config ---
const API_HOST = "https://commerce.logis.center"; 
// 🌟 [ANALYTICS TRACK] 관리자(Console) 기능 및 사용자 행동 로그 동기화 전용 Client Worker
const ANALYTICS_API_HOST = "https://console.logis.center";
const WIDGET_WIDTH = 380;
const COLLAPSED_HEIGHT = 80;
const EXPANDED_HEIGHT = 600;

// 🌟 [CRITICAL FIX] chrome.js와 완벽히 동일한 루트 도메인 추출 로직 추가 (필터 엇갈림 원천 차단)
const twoPartDomains = ["co.kr","co.uk","co.jp","com.cn","co.in","com.mx","co.id","com.my","com.sg","com.ph","com.vn"];
function getRootDomain(hostname: string) {
    const host = hostname.split('.');
    const isTwoPart = twoPartDomains.some(domain => hostname.endsWith(domain));
    if (isTwoPart && host.length >= 3) {
        return host[host.length - 3] + "." + host[host.length - 2] + "." + host[host.length - 1];
    } else if (host.length >= 2) {
        return host[host.length - 2] + "." + host[host.length - 1];
    }
    return hostname;
}

interface ChatSession {
    hash: string;
    token?: string;
    email?: string;
    team?: string;
    address?: string;
    name?: string;
    cc?: string;
    sender?: string;
    // 🌟 [FLAG] Client Worker 가 GeoIP 로 확정해 내려주는 국가 코드입니다.
    //    commerce D1 items 테이블에 flag 컬럼이 없어, 동기화 시 이 값으로 보강합니다.
    flag?: string;
}

// --- State ---
let currentSession: ChatSession = { hash: "", cc: "logis.center" };
let isExpanded = false;
let currentTab = "list";
let currentImage: string | null = null;
let currentDetectedUrl = "";
let isCurrentShop = false; 
let searchDebounceTimer: number | null = null;
let chatPollInterval: number | null = null;

// 🌟 [추가] 누락된 전역 상태 변수 선언
// =====================================================================
// 🌟 [OAUTH-NETWORK REGISTRATION] api.oauth.network 사이트 등록 기능
// ---------------------------------------------------------------------
//  www/docs/index.html 의 <form name="api.oauth.network"> 기능을
//  Tauri proxy_fetch 경유로 이식합니다.
//
//  통신 대상: https://api.oauth.network (Vercel, api/index.js)
//  목적: mode=analytic 에서 특정 도메인의 데이터를 조회하기 위한
//        Client 등록 절차 (client_id / client_secret 발급)
//
//  CORS: proxy_fetch 는 Rust 백엔드(reqwest) 에서 요청을 보내므로
//        브라우저 CORS 정책이 적용되지 않습니다. 별도 대응 불필요.
//
//  응답 파싱: api/index.js 는 HTML+postMessage 형식으로 응답합니다.
//        proxy_fetch 가 JSON 파싱 실패 시 { text: "..." } 로 래핑하므로
//        프론트엔드에서 <script> 내부 JSON 을 추출합니다.
// =====================================================================

const OAUTH_API_HOST = "https://api.oauth.network";

/**
 * 🌟 [HOST NORMALIZE] 등록 사이트 식별자를 `https://{host}` 하나로 통일합니다.
 *
 *  ── 왜 필요한가 ──
 *   서버 메타데이터는 "client_id:client_secret@example.com" 처럼 '스킴 없는 host' 로 저장하고,
 *   사용자는 등록 폼에 "https://example.com" 을 입력합니다.
 *   두 표기를 그대로 두면 중복 판정(s.host === hostUrl)이 항상 실패해
 *   같은 사이트가 두 줄로 쌓입니다. (이슈 2 의 직접 원인)
 *
 *  www/docs 의 레퍼런스 구현이 `host : "https://"+url.host` 로 정규화하므로
 *  같은 규칙을 그대로 따릅니다.
 */
function normalizeOAuthHost(raw: any): string {
    if (!raw || typeof raw !== "string") return "";
    let s = raw.trim().toLowerCase();
    if (!s) return "";
    if (!/^https?:\/\//.test(s)) s = "https://" + s;
    try {
        const u = new URL(s);
        if (!u.hostname) return "";
        return "https://" + u.host;
    } catch (_e) {
        return "";
    }
}

/**
 * 🌟 [BALANCED EXTRACT] `parent.postMessage(JSON.stringify({...}), "...")` 에서
 *  첫 번째 JSON 객체를 '괄호 균형' 으로 잘라냅니다.
 *  기존 정규식 /(\{[\s\S]*?\})\)/ 은 비탐욕 매칭이라 rows 값 안에 '})' 조합이
 *  먼저 등장하면 잘린 조각을 파싱해 조용히 실패했습니다.
 */
function extractBalancedJson(text: string, fromIndex: number): string {
    const start = text.indexOf("{", fromIndex);
    if (start === -1) return "";
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (let i = start; i < text.length; i++) {
        const ch = text[i];
        if (inStr) {
            if (esc) { esc = false; }
            else if (ch === "\\") { esc = true; }
            else if (ch === '"') { inStr = false; }
            continue;
        }
        if (ch === '"') { inStr = true; continue; }
        if (ch === "{") depth++;
        else if (ch === "}") {
            depth--;
            if (depth === 0) return text.substring(start, i + 1);
        }
    }
    return "";
}

/** api/index.js 의 HTML 응답에서 JSON 페이로드를 추출합니다. */
function parseOAuthApiResponse(raw: any): { rows: any[]; cookies: any; count: number; query: any } {
    let text = "";
    if (typeof raw === "string") {
        text = raw;
    } else if (raw && typeof raw === "object") {
        if (raw.text) {
            text = raw.text;
        } else if (raw.rows) {
            // 이미 JSON 으로 파싱된 경우 (향후 api/index.js 수정 시)
            return { rows: raw.rows || [], cookies: raw.cookies || {}, count: raw.count || 0, query: raw.query || {} };
        }
    }
    if (!text) {
        return { rows: [], cookies: {}, count: 0, query: {} };
    }

    // <script> 내부의 parent.postMessage(JSON.stringify({...}), ...) 에서 JSON 추출
    const anchor = text.indexOf("JSON.stringify(");
    const jsonText = anchor === -1
        ? extractBalancedJson(text, 0)
        : extractBalancedJson(text, anchor + "JSON.stringify(".length);

    if (!jsonText) {
        console.warn("[OAUTH] 응답에서 postMessage 페이로드를 찾지 못했습니다.");
        return { rows: [], cookies: {}, count: 0, query: {} };
    }

    try {
        const parsed = JSON.parse(jsonText);
        // ⚠️ index.js 는 cookies 를 '이중 stringify' 해서 내려줍니다.
        //    JSON.stringify(JSON.stringify(req.cookies)) → 문자열 리터럴이므로 한 번 더 파싱합니다.
        let cookies: any = {};
        if (typeof parsed.cookies === "string") {
            try { cookies = JSON.parse(parsed.cookies); } catch (_e) { cookies = {}; }
        } else if (parsed.cookies && typeof parsed.cookies === "object") {
            cookies = parsed.cookies;
        }
        return {
            rows: Array.isArray(parsed.rows) ? parsed.rows : [],
            cookies,
            count: parsed.count || 0,
            query: parsed.query || {}
        };
    } catch (e) {
        console.warn("[OAUTH] postMessage 페이로드 JSON 파싱 실패:", e);
        return { rows: [], cookies: {}, count: 0, query: {} };
    }
}

/**
 * 🌟 [BRANCH CONTRACT] api/index.js 의 GET 분기는 아래 순서로 '배타적' 입니다.
 *
 *    ① isAddress(to/from)                 → 지갑 주소 기반 조회
 *    ② req.query.hash && req.query.token  → 등록 사이트 목록 (cookies.#0, #1 ...)
 *    ③ req.query.referer                  → rows / count 조회
 *
 *  즉 referer 계열(경로 목록·접속량 통계)에 hash·token 을 실으면
 *  ②에 먼저 걸려 ③에 '영원히' 도달하지 못합니다. (통계가 항상 0 이던 원인)
 *
 *  그런데 Rust proxy_fetch 는 session_params 가 있으면
 *  hash / token / href 를 쿼리에 무조건 덧붙입니다(lib.rs DETAIL 1 블록).
 *  따라서 이 헬퍼는 session_params 를 '항상 null' 로 고정하고,
 *  필요한 파라미터만 호출부가 직접 명시하도록 강제합니다.
 *
 *  값이 배열이면 같은 키를 반복 append 합니다.
 *  (index.js 의 `typeof req.query.date == "object"` 기간 필터가 반복 파라미터를 요구합니다)
 */
async function oauthApiFetch(
    query: Record<string, any>,
    opts: { method?: "GET" | "POST"; body?: any } = {}
): Promise<{ rows: any[]; cookies: any; count: number; query: any }> {
    const method = opts.method || "GET";

    const sp = new URLSearchParams();
    for (const k of Object.keys(query)) {
        const v = query[k];
        if (v === undefined || v === null || v === "") continue;
        if (Array.isArray(v)) {
            for (const item of v) sp.append(k, String(item));
        } else {
            sp.append(k, String(v));
        }
    }

    const qs = sp.toString();
    const url = qs ? `${OAUTH_API_HOST}/?${qs}` : `${OAUTH_API_HOST}/`;

    // ⚠️ index.js 최상단 게이트: `if(req.referer){ ... } else { res.status(500) }`
    //    Referer 헤더가 없으면 무조건 500 입니다.
    const headers: Record<string, string> = {
        "Content-Type": method === "POST"
            ? "application/x-www-form-urlencoded"
            : "application/json",
        "Referer": "https://oauth.network/"
    };

    const args: any = {
        url,
        method,
        headers,
        session_params: null
    };
    if (opts.body) args.body = opts.body;

    const response = await invoke<any>("proxy_fetch", args);
    return parseOAuthApiResponse(response);
}

/**
 * 🌟 api.oauth.network 에 POST 를 보냅니다. 서버 계약상 이 한 경로가 3가지 동작을 겸합니다.
 *
 *    body { host }                                  → 신규 발급
 *    body { host } (이미 등록된 host)                → 재발급
 *    body { host, client_id, client_secret }        → 삭제 (index.js 의 trim 경로)
 *
 *  ⚠️ DELETE 메서드는 쓸 수 없습니다.
 *     index.js 의 DELETE 블록은 `ethers.hashMessage(uri.host)` 에서
 *     uri(= req.query.referer 로만 생성)가 undefined 라 무조건 500 이며,
 *     로직 자체도 '사이트 1건 삭제' 가 아니라 '계정 탈퇴' 입니다.
 *
 *  ⚠️ 삭제 역시 서버가 사이트 <head> 의 oauth-network-verification 메타 태그를
 *     다시 검증한 뒤에야 수행합니다. 태그를 먼저 지우면 서버 삭제가 실패합니다.
 */
async function submitOAuthRegistration(
    hostUrl: string,
    credentials?: { client_id: string; client_secret: string }
): Promise<{
    success: boolean;
    client_id: string;
    client_secret: string;
    error: string;
    removed: boolean;
}> {
    const isRemove = !!(credentials && credentials.client_id && credentials.client_secret);

    if (!currentSession.hash || !currentSession.token) {
        return { success: false, client_id: "", client_secret: "", error: "로그인이 필요합니다.", removed: false };
    }

    const host = normalizeOAuthHost(hostUrl);
    if (!host) {
        return { success: false, client_id: "", client_secret: "", error: "도메인 형식이 올바르지 않습니다. (예: https://example.com)", removed: false };
    }

    try {
        const body: any = {
            host: host,
            hash: currentSession.hash,
            token: currentSession.token
        };
        if (isRemove) {
            body.client_id = credentials!.client_id;
            body.client_secret = credentials!.client_secret;
        }

        console.log(`[OAUTH] POST ${isRemove ? "삭제" : "등록/재발급"} 요청: host='${host}'`);

        const parsed = await oauthApiFetch({}, { method: "POST", body });

        if (parsed.rows.length > 0) {
            const row = parsed.rows[0];
            if (row.client_id && row.client_secret) {
                // 🌟 [SERVER TRUTH] 로컬 배열에 push 하지 않습니다.
                //    www/docs 레퍼런스가 성공 시 reload 로 목록을 다시 읽는 것과 동일하게,
                //    서버 목록을 통째로 다시 당겨와 kv_store 를 교체합니다.
                //    → 같은 주소를 두 번 등록해도 절대 중복되지 않습니다.
                await fetchOAuthRegisteredSites();

                return {
                    success: true,
                    client_id: String(row.client_id),
                    client_secret: String(row.client_secret),
                    error: "",
                    removed: isRemove
                };
            }
        }

        // 🌟 [DETAIL ERROR] 서버 응답에 rows 가 비어 있으면 소유 확인 실패입니다.
        //    사용자가 대조할 수 있도록 '서버가 기대하는 주소' 를 함께 안내합니다.
        const email = currentSession.email || "";
        let expectedAddr = String(parsed.cookies?.client_id || "");
        if (!expectedAddr && email && typeof ethers !== "undefined") {
            try {
                expectedAddr = ethers.computeAddress(ethers.hashMessage(email)).toLowerCase();
            } catch (_e) { /* ignore */ }
        }

        const head = isRemove
            ? "서버 삭제 실패. 삭제도 사이트 소유 확인을 통과해야 합니다."
            : "사이트 소유 확인 실패.";

        const errMsg = expectedAddr
            ? `${head}\n사이트 <head>에 아래 세 태그가 그대로 있는지 확인하세요:\n<meta name="oauth-network-verification" content="${expectedAddr}" />\n<meta name="privacy" content="..." />\n<meta name="terms" content="..." />`
            : `${head}\n메타 태그를 사이트 <head>에 추가하세요.`;

        return { success: false, client_id: "", client_secret: "", error: errMsg, removed: isRemove };
    } catch (e: any) {
        return { success: false, client_id: "", client_secret: "", error: String(e), removed: isRemove };
    }
}

/**
 * 🌟 api.oauth.network 에서 로그인된 사용자의 등록 사이트 목록을 조회합니다.
 *
 *  ⚠️ 이 호출만 hash / token 을 실어야 합니다. (index.js 의 2번 분기)
 *     referer 를 함께 보내면 안 됩니다. 보내도 2번 분기가 먼저 잡아먹습니다.
 *
 *  응답의 cookies["#0"], cookies["#1"] ... 은 "client_id:client_secret@host" 이며,
 *  레퍼런스 구현과 동일하게 `new URL("https://" + entry)` 의 자격증명 파싱으로 읽습니다.
 */
async function fetchOAuthRegisteredSites(): Promise<void> {
    if (!currentSession.hash || !currentSession.token) return;
    try {
        const parsed = await oauthApiFetch({
            hash: currentSession.hash,
            token: currentSession.token
        });

        const cookies = parsed.cookies || {};

        // 🌟 [TRUST GATE] index.js 세션 블록이 예외로 빠지면 req.cookies = {} 가 되어
        //    빈 응답이 옵니다. 그 상태로 교체하면 멀쩡한 로컬 목록이 지워지므로,
        //    '세션이 실제로 성립한 응답' 일 때만 교체합니다.
        const authed = !!(cookies.email || cookies.client_id || cookies["#length"] !== undefined);
        if (!authed) {
            // 🌟 [DIAGNOSTIC] 소멸 버그의 원인 추적용 상세 로그를 추가합니다.
            //    cookies 가 완전히 비어 있으면 세션 블록 자체가 예외로 빠진 것이고,
            //    일부만 비어 있으면 hash/token 쌍 불일치입니다.
            const cookieKeys = Object.keys(cookies || {});
            console.warn(
                "[OAUTH] ⚠️ 서버 세션이 성립하지 않아 목록을 갱신하지 않습니다. " +
                `(hash='${(currentSession.hash || "").slice(0, 8)}...', ` +
                `token 유무=${!!currentSession.token}, ` +
                `응답 cookies 키=[${cookieKeys.join(", ")}]). ` +
                "로컬 kv_store 목록은 그대로 유지합니다."
            );
            return;
        }

        const len = parseInt(cookies["#length"] || "0", 10) || 0;

        const sites: any[] = [];
        for (let i = 0; i < len; i++) {
            const raw = cookies[`#${i}`];
            if (!raw || typeof raw !== "string") continue;
            try {
                // 형식: "0x주소:프라이빗키@호스트"
                const u = new URL("https://" + raw);
                if (!u.username || !u.hostname) continue;
                sites.push({
                    host: "https://" + u.host,
                    client_id: decodeURIComponent(u.username),
                    client_secret: decodeURIComponent(u.password),
                    registered_at: Date.now()
                });
            } catch (_e) {
                continue;
            }
        }

        if (cookies.client_id) {
            await kvSet("oauth_client_address", String(cookies.client_id));
        }

        // 서버가 유일한 진실 공급원입니다. 병합하지 않고 통째로 교체합니다.
        await kvSet("oauth_registered_sites", sites);

        console.log(
            `[OAUTH] 서버 조회 완료. 등록된 사이트 ${sites.length}건 → kv_store 교체. ` +
            `${sites.map((s: any) => s.host).join(", ")}`
        );
    } catch (e) {
        console.warn("[OAUTH] fetchOAuthRegisteredSites failed:", e);
    }
}

/**
 * 🌟 api.oauth.network 에서 등록된 사이트의 Cc(경로) 목록을 조회합니다.
 *  ⚠️ hash / token 을 절대 싣지 않습니다. (index.js 의 3번 분기로 가야 합니다)
 */
async function fetchOAuthSitePaths(referer: string): Promise<string[]> {
    const host = normalizeOAuthHost(referer);
    if (!host) return [];
    try {
        const parsed = await oauthApiFetch({
            referer: host,
            distinct: "Cc",
            id: "#LOG"
        });
        return parsed.rows
            .map((r: any) => r.Cc)
            .filter((c: string) => !!c);
    } catch (_e) {
        return [];
    }
}

/**
 * 🌟 api.oauth.network 에서 시간별 접속량 통계를 조회합니다.
 *  ⚠️ hash / token 을 절대 싣지 않습니다.
 *  ⚠️ date 는 '같은 키를 두 번' 보내야 index.js 가 배열로 인식해 기간 필터를 겁니다.
 */
async function fetchOAuthSiteCount(referer: string, hoursBack: number): Promise<number> {
    const host = normalizeOAuthHost(referer);
    if (!host) return 0;
    try {
        const now = Date.now();
        const from = new Date(now - hoursBack * 3600 * 1000).toISOString();
        const to = new Date(now).toISOString();
        const parsed = await oauthApiFetch({
            referer: host,
            id: "#LOG",
            cnt: "true",
            date: [from, to]
        });
        return parsed.count || 0;
    } catch (_e) {
        return 0;
    }
}

// 🌟 [OAUTH RENDER LOCK] 비동기 양보 구간에서 두 번째 renderNavigation()이
//    끼어들어 insertAdjacentHTML 이 중복 실행되는 것을 차단합니다.
let isOAuthSitesRendering = false;

async function renderOAuthSitesUI(pageList: HTMLElement) {
    if (!pageList) return;
    if (currentSearchMode !== "analytic") return;
    if (!currentSession.email) return;

    // 🌟 [CONCURRENT GUARD] 이미 렌더링 중이면 중복 진입을 즉시 차단합니다.
    //    폴링(3초) + browser-match-found + syncAnalyticsData 가 동시에
    //    renderNavigation() 을 발동하는 레이스 컨디션에서
    //    insertAdjacentHTML 이 2회 실행되어 아이템이 복제되던 직접 원인입니다.
    if (isOAuthSitesRendering) return;
    isOAuthSitesRendering = true;

    try {
        // 🌟 [IDEMPOTENT CLEANUP] 기존 .oauth-site-item 노드를 전부 제거합니다.
        //    기존 코드는 insertAdjacentHTML("beforeend") 만 있어서
        //    renderNavigation() 이 여러 번 호출될 때마다 노드가 누적되었습니다.
        //    renderAccordion(tree) 가 pageList.innerHTML 을 교체하더라도,
        //    그 '이후' 에 이 함수가 비동기로 실행되므로
        //    이전 라운드의 노드가 남아 있을 수 있습니다.
        const existingItems = pageList.querySelectorAll(".oauth-site-item");
        if (existingItems.length > 0) {
            existingItems.forEach((el: Element) => el.remove());
        }

        // 🌟 [OAUTH SYNC] 렌더링 전 서버에서 최신 목록을 조회합니다.
        //    다른 기기에서 등록/삭제한 내역도 이 시점에 반영됩니다.
        await fetchOAuthRegisteredSites();
        const registeredSites = await kvGet("oauth_registered_sites") || [];
        if (!Array.isArray(registeredSites) || registeredSites.length === 0) return;

        let oauthHtml = '';
        for (let si = 0; si < registeredSites.length; si++) {
            const site = registeredSites[si];
            const siteHost = site.host || "";
            const clientId = site.client_id || "";
            const clientSecret = site.client_secret || "";
            const siteId = `oauth_site_${si}`;

            let displayHost = siteHost;
            try {
                displayHost = new URL(siteHost).hostname;
            } catch (_e) {}

            oauthHtml += `
<div class="oauth-site-item" id="${siteId}" style="position:relative; padding:6px 0; border-bottom:1px solid #f0f0f0;">
    <div style="display:flex; align-items:center; gap:6px;">
        <span style="font-size:0.8rem; font-weight:600; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${displayHost}</span>
        <button class="btn-oauth-more" data-site-idx="${si}" style="background:none; border:none; cursor:pointer; font-size:10px; font-style:italic; text-decoration:underline; color:#6366f1; padding:0 4px;">more</button>
        <button class="btn-oauth-hide" data-site-idx="${si}" style="background:none; border:none; cursor:pointer; font-size:10px; text-decoration:underline; color:#888; padding:0 4px;">hide</button>
    </div>
    <div class="oauth-site-tokens" data-site-idx="${si}" style="display:none; margin-top:6px; padding:8px; background:#f8f9fa; border-radius:4px; font-size:0.68rem; line-height:1.6; word-break:break-all;">
        <div><strong>Client Id:</strong> ${clientId}</div>
        <div style="margin-top:4px;"><strong>Client Secret:</strong> ${clientSecret}</div>
        <div style="margin-top:8px; text-align:right; display:flex; gap:6px; justify-content:flex-end;">
            <button class="btn-oauth-reissue" data-site-idx="${si}" style="background:none; border:1px solid #6366f1; color:#6366f1; border-radius:4px; padding:3px 10px; font-size:0.65rem; cursor:pointer;">재발급</button>
            <button class="btn-oauth-delete" data-site-idx="${si}" style="background:none; border:1px solid #ef4444; color:#ef4444; border-radius:4px; padding:3px 10px; font-size:0.65rem; cursor:pointer;">Delete</button>
        </div>
    </div>
    <div class="oauth-site-stats" data-site-host="${siteHost}" style="margin-top:15px; border:1px solid #ddd; border-radius:8px; position:relative;">
        <span style="position:absolute; left: 8px; top: -5px; font-size:0.6rem; color:#999; font-weight:100; background:#fff; padding:0 4px;">시간별 접속량</span>
        <table style="width:100%; border-collapse:collapse;">
            <tbody>
                <tr>
                    <td class="stat-hour" style="padding:8px; width:25%; text-align:center; border-right:1px solid #ddd;">
                        <cnt><span style="font-size:1em; font-weight:bold; text-decoration:underline;">0</span><span style="display:block; margin-top:3px; font-size:10px; font-weight:100;">hour</span></cnt>
                    </td>
                    <td class="stat-day" style="padding:8px; width:25%; text-align:center; border-right:1px solid #ddd;">
                        <cnt><span style="font-size:1em; font-weight:bold; text-decoration:underline;">0</span><span style="display:block; margin-top:3px; font-size:10px; font-weight:100;">day</span></cnt>
                    </td>
                    <td class="stat-week" style="padding:8px; width:25%; text-align:center; border-right:1px solid #ddd;">
                        <cnt><span style="font-size:1em; font-weight:bold; text-decoration:underline;">0</span><span style="display:block; margin-top:3px; font-size:10px; font-weight:100;">week</span></cnt>
                    </td>
                    <td class="stat-month" style="padding:8px; width:25%; text-align:center;">
                        <cnt><span style="font-size:1em; font-weight:bold; text-decoration:underline;">0</span><span style="display:block; margin-top:3px; font-size:10px; font-weight:100;">month</span></cnt>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
</div>`;
        }

        pageList.insertAdjacentHTML("beforeend", oauthHtml);

        // "more" 버튼: 토큰 정보 + 재발급/삭제 버튼 토글
        pageList.querySelectorAll(".btn-oauth-more").forEach((btn: any) => {
            btn.onclick = (e: Event) => {
                e.preventDefault();
                e.stopPropagation();
                const idx = btn.dataset.siteIdx;
                const tokenDiv = pageList.querySelector(`.oauth-site-tokens[data-site-idx="${idx}"]`) as HTMLElement;
                if (tokenDiv) {
                    const isVisible = tokenDiv.style.display !== "none";
                    tokenDiv.style.display = isVisible ? "none" : "block";
                    btn.textContent = isVisible ? "more" : "fold";
                }
            };
        });

        // "hide" 버튼: 사이트 항목 숨김 (화면 전용, 서버 영향 없음)
        pageList.querySelectorAll(".btn-oauth-hide").forEach((btn: any) => {
            btn.onclick = async (e: Event) => {
                e.preventDefault();
                e.stopPropagation();
                const idx = btn.dataset.siteIdx;
                const siteItem = document.getElementById(`oauth_site_${idx}`);
                if (siteItem) {
                    siteItem.style.display = "none";
                }
            };
        });

        // 🌟 [REISSUE] host 만 보내면 index.js 가 재발급 분기로 진입합니다.
        pageList.querySelectorAll(".btn-oauth-reissue").forEach((btn: any) => {
            btn.onclick = async (e: Event) => {
                e.preventDefault();
                e.stopPropagation();

                const idx = parseInt(btn.dataset.siteIdx, 10);
                const sites = (await kvGet("oauth_registered_sites")) || [];
                if (!Array.isArray(sites) || idx >= sites.length) return;

                const site = sites[idx];
                const confirmed = await ask(
                    `'${site.host}' 의 client_id / client_secret 을 재발급하시겠습니까?\n기존 키는 즉시 무효화됩니다.`,
                    { title: "재발급 확인", kind: "warning" }
                );
                if (!confirmed) return;

                btn.textContent = "처리 중...";
                btn.disabled = true;

                const res = await submitOAuthRegistration(site.host);

                if (res.success) {
                    await renderNavigation();
                } else {
                    btn.textContent = "재발급";
                    btn.disabled = false;
                    alert(res.error);
                }
            };
        });

        // 🌟 [DELETE] 서버에 host + client_id + client_secret 을 함께 POST 해야
        //    index.js 의 trim 경로(#length--, 테이블 row delete)가 실행됩니다.
        //    기존 구현은 kv_store 에서 splice 만 해서 서버에는 그대로 남아 있었고,
        //    다음 fetchOAuthRegisteredSites 에서 그대로 되살아났습니다.
        pageList.querySelectorAll(".btn-oauth-delete").forEach((btn: any) => {
            btn.onclick = async (e: Event) => {
                e.preventDefault();
                e.stopPropagation();

                const idx = parseInt(btn.dataset.siteIdx, 10);
                const sites = (await kvGet("oauth_registered_sites")) || [];
                if (!Array.isArray(sites) || idx >= sites.length) return;

                const site = sites[idx];

                const confirmed = await ask(
                    `'${site.host}' 등록을 삭제하시겠습니까?\n\n` +
                    "· api.oauth.network 서버에서 client_id / client_secret 이 영구 삭제됩니다.\n" +
                    "· 서버가 삭제 시에도 소유 확인을 하므로, 사이트 <head> 의 메타 태그는 아직 남아 있어야 합니다.",
                    { title: "사이트 삭제 확인", kind: "warning" }
                );
                if (!confirmed) return;

                btn.textContent = "삭제 중...";
                btn.disabled = true;

                const res = await submitOAuthRegistration(site.host, {
                    client_id: site.client_id,
                    client_secret: site.client_secret
                });

                if (res.success) {
                    console.log(`[OAUTH] 🗑️ 서버에서 '${site.host}' 등록을 삭제했습니다.`);
                    await renderNavigation();
                } else {
                    btn.textContent = "Delete";
                    btn.disabled = false;
                    alert(res.error);
                }
            };
        });

        // 🌟 [STATS] 각 사이트의 시간별 접속량을 비동기로 조회합니다.
        //    hash / token 을 싣지 않으므로 index.js 의 referer 분기가 실제로 실행됩니다.
        for (let si = 0; si < registeredSites.length; si++) {
            const site = registeredSites[si];
            const siteHost = site.host || "";
            if (!siteHost) continue;
            const statsDiv = pageList.querySelector(`.oauth-site-stats[data-site-host="${siteHost}"]`) as HTMLElement;
            if (!statsDiv) continue;

            // hour: 최근 1시간
            fetchOAuthSiteCount(siteHost, 1).then(cnt => {
                const el = statsDiv.querySelector(".stat-hour cnt span") as HTMLElement;
                if (el) el.textContent = String(cnt);
            });
            // day: 최근 24시간
            fetchOAuthSiteCount(siteHost, 24).then(cnt => {
                const el = statsDiv.querySelector(".stat-day cnt span") as HTMLElement;
                if (el) el.textContent = String(cnt);
            });
            // week: 최근 7일 (168시간)
            fetchOAuthSiteCount(siteHost, 168).then(cnt => {
                const el = statsDiv.querySelector(".stat-week cnt span") as HTMLElement;
                if (el) el.textContent = String(cnt);
            });
            // month: 최근 30일 (720시간)
            fetchOAuthSiteCount(siteHost, 720).then(cnt => {
                const el = statsDiv.querySelector(".stat-month cnt span") as HTMLElement;
                if (el) el.textContent = String(cnt);
            });
        }
    } catch (oauthErr) {
        console.warn("[NAV] OAuth registered sites render failed:", oauthErr);
    } finally {
        // 🌟 [LOCK RELEASE] 예외 여부과 무관하게 락을 해제해야
        //    다음 renderNavigation() 에서 재렌더링이 가능해집니다.
        isOAuthSitesRendering = false;
    }
}

function renderOAuthRegistrationForm() {
    const existing = document.getElementById("oauth-registration-modal");
    if (existing) existing.remove();

    const modal = document.createElement("div");
    modal.id = "oauth-registration-modal";
    modal.style.cssText = "position: fixed; inset: 121px 11px 11px; border-bottom-left-radius: 1em; border-bottom-right-radius: 1em; z-index: 99999; display: flex; align-items: center; justify-content: center; background: rgba(255, 255, 255, 0.88); pointer-events: initial;";
    modal.innerHTML = `
<div style="background:#fff;border-radius:12px;padding:24px;width:90%;max-width:480px;max-height:85vh;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <h3 style="margin:0;font-size:1.1rem;font-weight:700;">사이트 등록 (Analytic)</h3>
        <button id="oauth-modal-close" style="background:none;border:none;font-size:1.2rem;cursor:pointer;color:#666;">✕</button>
    </div>

    <!-- Step 1: 도메인 입력 -->
    <div style="margin-bottom:16px;">
        <label style="display:block;margin-bottom:6px;font-size:0.85rem;font-weight:600;">사이트 도메인</label>
        <input id="oauth-reg-host" type="url" placeholder="https://example.com" style="width:100%;padding:10px 12px;border:1px solid #ddd;border-radius:6px;box-sizing:border-box;font-size:0.9rem;">
        <p style="margin:4px 0 0;font-size:0.75rem;color:#888;">예시: (O) https://example.com (X) https://www.example.com</p>
    </div>

    <!-- Step 2: 소유 확인 메타 태그 (도메인 입력 시 즉시 생성) -->
    <div id="oauth-reg-meta" style="display:none;margin-bottom:16px;">
        <label style="display:block;margin-bottom:6px;font-size:0.85rem;font-weight:600;">사이트 소유 확인 메타 태그</label>
        <p style="margin:0 0 6px;font-size:0.75rem;color:#666;">
            아래 메타 태그를 복사하여 사이트 홈페이지의 <code>&lt;head&gt;</code> 섹션에 붙여넣으세요.<br>
            등록 버튼 클릭 시 서버가 이 태그의 존재 여부를 검증합니다.
        </p>
        <textarea id="oauth-reg-meta-tag" readonly style="width:100%;min-height:100px;padding:10px;border:1px solid #ddd;border-radius:6px;font-size:0.72rem;resize:none;box-sizing:border-box;background:#f8f9fa;line-height:1.6;"></textarea>
        <button id="oauth-meta-copy" style="margin-top:6px;padding:6px 14px;border:1px solid #6366f1;border-radius:4px;background:#fff;color:#6366f1;font-size:0.78rem;cursor:pointer;">태그 복사</button>
    </div>

    <!-- 결과 메시지 -->
    <div id="oauth-reg-result" style="display:none;margin-bottom:12px;padding:12px;border-radius:6px;font-size:0.85rem;"></div>

    <!-- 등록 성공 후 Client 정보 -->
    <div id="oauth-reg-credentials" style="display:none;margin-bottom:16px;">
        <label style="display:block;margin-bottom:6px;font-size:0.85rem;font-weight:600;">Client Id (Address)</label>
        <input id="oauth-reg-client-id" readonly style="width:100%;padding:8px 12px;border:1px solid #ddd;border-radius:6px;box-sizing:border-box;font-size:0.8rem;margin-bottom:8px;">
        <label style="display:block;margin-bottom:6px;font-size:0.85rem;font-weight:600;">Client Secret (Private Key)</label>
        <input id="oauth-reg-client-secret" readonly style="width:100%;padding:8px 12px;border:1px solid #ddd;border-radius:6px;box-sizing:border-box;font-size:0.8rem;">
    </div>

    <button id="oauth-reg-submit" style="width:100%;padding:12px;border:none;border-radius:8px;background:#eee;color:#000;font-size:0.95rem;font-weight:700;cursor:pointer;">추가</button>
</div>`;
    document.body.appendChild(modal);

    // 닫기
    document.getElementById("oauth-modal-close")!.addEventListener("click", () => modal.remove());
    modal.addEventListener("click", (e) => { if (e.target === modal) modal.remove(); });

    // 🌟 [META TAG GENERATION] 도메인 입력 시 즉시 메타 태그를 생성합니다.
    //    api/index.js 의 검증 로직:
    //      clientAddress = ethers.computeAddress(ethers.hashMessage(email))
    //    즉, 현재 로그인한 사용자의 email 기반으로 해시를 만들어
    //    사이트 <head> 에 메타 태그로 삽입해야 소유 증명이 됩니다.
    //
    //    그러나 Tauri 앱에서는 currentSession.email 이 있으므로
    //    사전에 태그를 만들어 보여줄 수 있습니다.
    //    (api/index.js 는 서버에서 같은 값으로 검증합니다)
    const hostInput = document.getElementById("oauth-reg-host") as HTMLInputElement;
    const metaDiv = document.getElementById("oauth-reg-meta") as HTMLDivElement;
    const metaTextarea = document.getElementById("oauth-reg-meta-tag") as HTMLTextAreaElement;
    const metaCopyBtn = document.getElementById("oauth-meta-copy") as HTMLButtonElement;

    let metaDebounce: number | null = null;
    hostInput.addEventListener("input", () => {
        if (metaDebounce) clearTimeout(metaDebounce);
        metaDebounce = window.setTimeout(() => {
            const url = hostInput.value.trim();
            if (!url || !url.startsWith("http")) {
                metaDiv.style.display = "none";
                return;
            }
            try {
                // api/index.js 검증 로직과 동일한 해시 생성:
                //   clientAddress = ethers.computeAddress(ethers.hashMessage(email))
                // Tauri 에서는 currentSession.email 이 있으므로 사전 계산 가능
                const email = currentSession.email || "";

                // 🌟 [EMAIL GUARD] 이메일이 없으면 메타 태그를 생성하지 않습니다.
                //    빈 문자열로 계산된 주소는 서버 검증과 절대 일치하지 않으므로
                //    잘못된 태그를 붙여넣게 되는 것을 사전에 차단합니다.
                if (!email) {
                    metaDiv.style.display = "none";
                    return;
                }

                const hashMessage = ethers.hashMessage(email);
                const clientAddress = ethers.computeAddress(hashMessage).toLowerCase();

                // 🌟 [DEBUG] 생성된 주소를 콘솔에 출력하여 사이트 태그와 대조 가능하게 합니다.
                console.log(`[OAUTH] Meta tag generated for email='${email}' → content="${clientAddress}"`);

                const tags = `<meta name="oauth-network-verification" content="${clientAddress}" />
    <meta name="privacy" content="/개인정보약관 경로/" />
    <meta name="terms" content="/이용약관 경로/" />`;

                metaTextarea.value = tags;
                metaDiv.style.display = "block";
            } catch (e) {
                metaDiv.style.display = "none";
            }
        }, 300);
    });

    // 태그 복사 버튼
    metaCopyBtn.addEventListener("click", () => {
        const text = metaTextarea.value;
        if (!text) return;
        navigator.clipboard.writeText(text).then(() => {
            metaCopyBtn.textContent = "복사됨 ✓";
            setTimeout(() => { metaCopyBtn.textContent = "태그 복사"; }, 2000);
        }).catch(() => {
            // clipboard API 실패 시 textarea 선택 방식
            metaTextarea.focus();
            metaTextarea.select();
            document.execCommand("copy");
            metaCopyBtn.textContent = "복사됨 ✓";
            setTimeout(() => { metaCopyBtn.textContent = "태그 복사"; }, 2000);
        });
    });

    // 제출
    document.getElementById("oauth-reg-submit")!.addEventListener("click", async () => {
        const resultDiv = document.getElementById("oauth-reg-result") as HTMLDivElement;
        const credDiv = document.getElementById("oauth-reg-credentials") as HTMLDivElement;
        const submitBtn = document.getElementById("oauth-reg-submit") as HTMLButtonElement;

        const hostUrl = hostInput.value.trim();
        if (!hostUrl) {
            resultDiv.style.display = "block";
            resultDiv.style.background = "#fef2f2";
            resultDiv.style.color = "#dc2626";
            resultDiv.textContent = "도메인을 입력하세요.";
            return;
        }

        // 메타 태그가 아직 생성되지 않은 경우 생성 유도
        if (metaDiv.style.display === "none") {
            const email = currentSession.email || "";
            const hashMessage = ethers.hashMessage(email);
            const clientAddress = ethers.computeAddress(hashMessage).toLowerCase();
            metaTextarea.value = `<meta name="oauth-network-verification" content="${clientAddress}" />
<meta name="privacy" content="/개인정보약관 경로/" />
<meta name="terms" content="/이용약관 경로/" />`;
            metaDiv.style.display = "block";
            resultDiv.style.display = "block";
            resultDiv.style.background = "#fffbeb";
            resultDiv.style.color = "#d97706";
            resultDiv.textContent = "위 메타 태그를 사이트 <head>에 추가한 후 다시 등록하세요.";
            return;
        }

        submitBtn.disabled = true;
        submitBtn.textContent = "등록 중...";
        resultDiv.style.display = "none";

        const res = await submitOAuthRegistration(hostUrl);

        submitBtn.disabled = false;
        submitBtn.textContent = "추가";

        if (res.success) {
            resultDiv.style.display = "block";
            resultDiv.style.background = "#f0fdf4";
            resultDiv.style.color = "#16a34a";
            resultDiv.textContent = "등록 완료!";

            metaDiv.style.display = "none";
            credDiv.style.display = "block";
            (document.getElementById("oauth-reg-client-id") as HTMLInputElement).value = res.client_id;
            (document.getElementById("oauth-reg-client-secret") as HTMLInputElement).value = res.client_secret;

            // 🌟 [NO LOCAL PUSH] submitOAuthRegistration 이 성공 직후
            //    fetchOAuthRegisteredSites() 로 서버 목록을 통째로 다시 읽어
            //    kv_store 를 교체합니다. 같은 주소로 다시 눌러도 절대 중복되지 않습니다.
            //    (기존에는 여기서 로컬 배열에 push 했고, 서버 목록은 스킴 없는 host 라
            //     중복 판정이 항상 실패했습니다)
            hostInput.value = "";

            // 네비게이션 갱신 (Pages 섹션에 등록된 사이트가 즉시 반영됩니다)
            await renderNavigation();
        } else {
            resultDiv.style.display = "block";
            resultDiv.style.background = "#fef2f2";
            resultDiv.style.color = "#dc2626";
            resultDiv.textContent = res.error;
            // 소유 확인 실패 시 메타 태그를 다시 강조 표시
            metaDiv.style.display = "block";
            metaTextarea.style.border = "2px solid #ef4444";
            setTimeout(() => { metaTextarea.style.border = "1px solid #ddd"; }, 3000);
        }
    });
}

let isSearching = false;
let isExtracting = false;

// 🚀 모델 다운로드 관련 상태 관리 변수 추가
let modelStatus: Record<string, boolean> = {};
const TARGET_MODELS = [
    'Qwen3', 'Qwen3.5', 'Embedding', 'Granite', 'SigLIP2',
    'stanza_korean', 'stanza_english', 'stanza_japanese', 'stanza_chinese',
    'stanza_french', 'stanza_german', 'stanza_spanish', 'stanza_italian',
    'stanza_portuguese', 'stanza_dutch', 'stanza_russian', 'stanza_arabic',
    'stanza_thai', 'stanza_hindi', 'stanza_bengali', 'stanza_greek',
    'stanza_hebrew', 'stanza_vietnamese'
];
export let lastSearchedQuery = "";
// 🌟 [CRITICAL FIX] 프론트엔드 상태 토글 및 중복 전송 방어용 락
let isBrowserRunning = false;
let isAutoLaunchLocked = false; // 🌟 런처 클릭 후 stopped 시그널 전까지 버튼 강제 숨김 락

/*
    🌟 [CLOUD AI TRACK]
    - Cloud 모드로 보낸 작업은 로컬 GPU 큐를 점유하지 않습니다. (락 미사용)
    - 서버 tasks 테이블에서 해당 task 가 사라지면 = 처리 완료로 간주하여 말풍선을 Done 처리합니다.
    - 임베딩은 GPU 유무와 무관하게 항상 로컬(Client App)에서 수행합니다.
*/
interface CloudPendingMeta {
    serverId: string;
    kind: "extract" | "search";
    createdAt: number;
}
let cloudPendingTasks = new Map<string, CloudPendingMeta>();
let isReindexing = false;


// 🌟 [EMBED DEBOUNCE] runLocalEmbeddingSync 중복 호출 방지를 위한 스케줄링 변수
let reindexScheduled = false;
let reindexDebounceTimer: number | null = null;

// 🌟 클라우드에서 내려온(벡터 없는) 아이템을 로컬 임베딩 모델로 벡터화 + 청크 인덱싱합니다.
//    디바운스(2초)를 적용하여 initSession + syncData에서 연속 호출되어도 1회만 실행됩니다.
async function runLocalEmbeddingSync() {
    if (isReindexing || reindexScheduled) return;
    if (isSearching || isExtracting || GlobalTaskManager.isBusy) return;
    // 🌟 [DEBOUNCE] 2초 내 재호출 시 타이머를 리셋하여 마지막 호출만 실행
    reindexScheduled = true;
    if (reindexDebounceTimer) clearTimeout(reindexDebounceTimer);
    reindexDebounceTimer = window.setTimeout(async () => {
        reindexScheduled = false;
        reindexDebounceTimer = null;
        // 디바운스 후에도 여전히 바쁘면 스킵
        if (isReindexing || isSearching || isExtracting || GlobalTaskManager.isBusy) return;
        isReindexing = true;
        try {
            // 🌟 [ALL-TRACK EMBEDDING]
            //  ── 무엇이 문제였나 ──
            //   기존에는 currentSearchMode 하나만 넘겼습니다.
            //   그래서 사용자가 Trading 탭을 한 번도 열지 않으면
            //   무역 문서가 영원히 벡터화되지 않았고,
            //   syncAnalyticsData 직후 호출도 현재 탭이 commerce 면
            //   analytic 문서를 건너뛰었습니다.
            //   벡터가 없으면 search_items 는 0벡터와 비교하고
            //   search_chunks 는 청크가 없어 검색이 구조적으로 0건이 됩니다.
            //  ── 비용 ──
            //   백엔드는 대상이 0건이면 임베딩 모델을 올리기 전에
            //   'no_pending' 으로 즉시 반환하므로(LanceDB 조회 1회),
            //   세 트랙을 순회해도 유휴 시 부담이 사실상 없습니다.
            //   현재 탭을 먼저 처리해 체감 지연을 최소화합니다.
            const trackOrder = [currentSearchMode,
                ...["commerce", "shipping", "analytic"].filter(m => m !== currentSearchMode)];

            let totalProcessed = 0;
            for (const track of trackOrder) {
                if (isSearching || isExtracting || GlobalTaskManager.isBusy) break;

                const res = await invoke<any>("reindex_pending_embeddings", {
                    limit: 20,
                    devicePreference: getDevicePref(),
                    mode: track
                });

                if (res && res.processed && res.processed > 0) {
                    totalProcessed += res.processed;
                    console.log(`[EMBED] Locally embedded ${res.processed} item(s). (mode: ${res.mode || track})`);
                } else if (res && res.skipped) {
                    console.log(`[EMBED] 트랙 '${track}' 임베딩 스킵: 사유=${res.skipped}`);
                }
            }

            if (totalProcessed > 0) {
                await renderNavigation();
                if (currentTab === "list") {
                    await loadMoreDocs(false, true);
                }
            }
        } catch (e) {
            console.warn("[EMBED] reindex_pending_embeddings failed:", e);
        } finally {
            isReindexing = false;
        }
    }, 2000);
}

/**
 * 🌟 [ANALYTIC BUBBLE DONE] 구조화 완료 후 채팅 화면에 남아 있는
 *    analytic_sync 관련 말풍선들을 전부 Done(9) 상태로 전환합니다.
 *
 *  ── 무엇이 문제였나 ──
 *   구조화 진행 중에는 renderProgressToUI 가 "Extracting data (100%)..."
 *   같은 중간 상태로 말풍선을 갱신합니다. 그런데 구조화가 끝나면
 *   더 이상 해당 task_id 로 이벤트가 오지 않아
 *   data-status="1" (PROCESSING) + 스피너가 영원히 멈춰 있습니다.
 *
 *  ── 처리 대상 ──
 *   · analytic_sync_sem_*  (개별 이벤트 구조화)
 *   · analytic_sync         (흐름 리포트 합성)
 *   · analytic_sync_flow    (cross_action_flow)
 *   이 세 접두사로 시작하는 모든 .task-bubble 을 잡습니다.
 */
async function finalizeAnalyticBubbles(processedCount: number): Promise<void> {
    if (!chatTalks) return;
    const bubbles = chatTalks.querySelectorAll('.chat-talk.task-bubble') as NodeListOf<HTMLElement>;
    let finalized = 0;
    for (const el of Array.from(bubbles)) {
        const taskId = el.dataset.taskId || el.id || "";
        // analytic_sync 로 시작하는 말풍선만 대상
        if (!taskId.startsWith("analytic_sync")) continue;
        const status = parseInt(el.dataset.status || "0");
        // 이미 완료/에러/정지된 것은 건드리지 않음
        if ([2, 3, 6, 9].includes(status)) continue;
        // status → 9 (Done)
        el.dataset.status = "9";
        const contentEl = el.querySelector('.content') as HTMLElement;
        if (contentEl) {
            contentEl.textContent = `Analytic structuring complete (${processedCount} event(s)).`;
        }
        const statusBar = el.querySelector('.status-bar') as HTMLElement;
        if (statusBar) {
            statusBar.style.color = "#22c55e";
            statusBar.innerHTML = `<span>✅</span> DONE`;
        }
        // active-spinner 클래스 제거
        const spinner = el.querySelector('.active-spinner') as HTMLElement;
        if (spinner) spinner.classList.remove('active-spinner');
        finalized++;
    }
    if (finalized > 0) {
        console.log(`[ANALYTIC] ✅ ${finalized}개의 구조화 말풍선을 Done 상태로 전환했습니다.`);
    }
}

// =====================================================================
// 🌟 [ANALYTIC STRUCTURING] 원시 행동 로그(HTML) → 시맨틱 문장 확정
// ---------------------------------------------------------------------
//  순서가 중요합니다.
//    ① structure_pending_analytics : HTML → PUG → 속성 제거 → Qwen3.5 2B 요약
//    ② reindex_pending_embeddings  : 확정된 문장을 임베딩 + item_chunks 인덱싱
//  ①을 건너뛰면 text 가 비어 있어 ②의 RAW GUARD 가 그 문서를 통째로 보류합니다.
// =====================================================================
let isAnalyticStructuring = false;

async function runAnalyticStructuring(): Promise<number> {
    if (isAnalyticStructuring) return 0;
    if (isSearching || isExtracting || GlobalTaskManager.isBusy) return 0;

    isAnalyticStructuring = true;

    try {
        const res = await invoke<any>("structure_pending_analytics", {
            limit: 20,
            devicePreference: getDevicePref(),
        });

        const processed = res && res.processed ? res.processed : 0;

        if (processed > 0) {
            console.log(
                `[ANALYTIC] 🧠 ${processed}건의 행동 로그를 시맨틱 문장으로 구조화했습니다.`
            );

            // 🌟 [DEXIE SYNC] 구조화 후 프론트엔드 Dexie의 updated_at을 갱신합니다.
            //    기존에는 Rust LanceDB만 updated_at = now_ts로 갱신하고,
            //    프론트엔드 Dexie는 갱신하지 않아 다음 폴링에서
            //    '로컬 0 < 서버 updated_at'이 성립하여 구조화 데이터를
            //    서버 원시 데이터로 덮어쓰는 무한 루프가 발생했습니다.
            //    이제 Dexie에서 summary가 있는(=구조화 완료) 항목의
            //    updated_at을 현재 시각으로 갱신하여
            //    다음 폴링에서 '이미 처리 완료'로 판단되게 합니다.
            if (appDb) {
                try {
                    const nowTs = Date.now();
                    const rows = await appDb
                        .table("items")
                        .where("mode")
                        .equals("analytic")
                        .filter((r: any) => Number(r.updated_at || 0) === 0)
                        .toArray();

                    for (const r of rows) {
                        if (
                            r.data &&
                            (r.data.summary ||
                                (typeof r.data.action === "string" && r.data.action))
                        ) {
                            await appDb
                                .table("items")
                                .where("id")
                                .equals(r.id)
                                .modify({
                                    updated_at: nowTs,
                                    "data.updated_at": nowTs,
                                });
                        }
                    }
                    console.log(
                        `[ANALYTIC] Dexie updated_at 갱신 완료 (후보 ${rows.length}건)`
                    );
                } catch (e) {
                    console.warn("[ANALYTIC] Dexie updated_at sync failed:", e);
                }
            }

            if (currentSearchMode === "analytic") {
                await renderNavigation();
                if (currentTab === "list") {
                    await loadMoreDocs(false, true);
                }
            }
        } else if (res && res.skipped) {
            console.log(`[ANALYTIC] 구조화 스킵: 사유=${res.skipped}`);
        }

        return processed;
    } catch (e) {
        console.warn("[ANALYTIC] structure_pending_analytics failed:", e);
        return 0;
    } finally {
        isAnalyticStructuring = false;
    }
}

// =====================================================================
// 🌟 [ANALYTICS TRACK v2] console.logis.center(Client Worker) ↔ 로컬 동기화
// ---------------------------------------------------------------------
//  ── Worker 계약 (console-logis-center/src/index.ts 실측) ──
//    cookies.href = decodeURIComponent(req.query.href).toLowerCase()
//    var url      = new URL(cookies.href)
//    cookies.cc   = hashId(url.host)            ← req.query.cc 는 '전혀' 읽지 않음
//    GET(JSON)    = SELECT * FROM items
//                   WHERE "cc" = <cookies.cc>
//                     AND "updated_at" > 0
//                     AND "created_at" < <req.query.created_at>
//                   ORDER BY created_at DESC LIMIT 1000
//
//  ── 그래서 무엇이 바뀌었나 ──
//   ① [TDZ 사망] 기존 코드는 `const params` 선언 '이전' 에 params.append() 를 호출해
//      매 호출마다 ReferenceError 로 즉사했고, 바깥 catch 가 조용히 삼켰습니다.
//      HTTP 요청이 단 한 번도 나가지 않았습니다.
//   ② [cc 불일치] Worker 는 cc 파라미터를 무시하고 href 의 host 로 cc 를 만듭니다.
//      브라우저가 꺼져 있으면 href 가 "https://console.logis.center/" 로 폴백되어
//      cc = hashId("console.logis.center") 를 조회 → 영원히 0건이었습니다.
//      이제 '추적 대상 사이트의 origin' 을 해석해 사이트마다 1회씩 조회합니다.
//   ③ [풀 호스트] Worker 는 hashId(url.host) 즉 'abc.cafe24.com' 을 씁니다.
//      getRootDomain() 이 만드는 hashId('cafe24.com') 과 애초에 다른 값이므로
//      로컬 계산 규칙도 Worker 와 동일하게 풀 호스트로 통일합니다.
//   ④ [시간대] created_at 은 '상한 커서' 입니다. now - timezoneOffset 은
//      UTC- 지역(예: EST, offset=+300)에서 now-5h 가 되어 최근 5시간 이벤트를
//      통째로 잘라냈습니다. 반드시 미래 시각이어야 합니다.
//   ⑤ [question/answer] 버리지 않고 analytic 아이템으로 보존합니다.
// =====================================================================

let isAnalyticsSyncRunning = false;
let lastAnalyticsSyncAt = 0;

// =====================================================================
// 🌟 [ADAPTIVE POLLING / BACKOFF] 동일 데이터 반복 동기화 방지
// ---------------------------------------------------------------------
//  연속으로 '변경 없음' 이 감지되면 폴링 간격을 1.5배씩 늘립니다.
//  실제 변경이 오면 즉시 기본 간격(3초)으로 리셋합니다.
//  최대 30초를 넘기지 않습니다.
// =====================================================================
const SYNC_BASE_INTERVAL_MS = 3_000;   // 기본 폴링 간격
const SYNC_BACKOFF_FACTOR = 1.5;       // 증가 배수
const SYNC_MAX_INTERVAL_MS = 30_000;   // 최대 간격
let syncConsecutiveNoChange = 0;       // 연속 '변경 없음' 카운터
let syncCurrentIntervalMs = SYNC_BASE_INTERVAL_MS; // 현재 적용 중인 간격

/**
 * 🌟 폴링 간격을 계산합니다.
 *  변경 없음 카운터가 올라갈수록 1.5배씩 증가, 최대 30초.
 */
function computeSyncInterval(): number {
    if (syncConsecutiveNoChange === 0) return SYNC_BASE_INTERVAL_MS;
    const interval = SYNC_BASE_INTERVAL_MS * Math.pow(SYNC_BACKOFF_FACTOR, syncConsecutiveNoChange);
    return Math.min(interval, SYNC_MAX_INTERVAL_MS);
}

/**
 * 🌟 동기화 결과에 따라 백오프 상태를 갱신합니다.
 * @param hasChange 실제 데이터 변경이 있었는지 여부
 */
function updateSyncBackoff(hasChange: boolean): void {
    if (hasChange) {
        if (syncConsecutiveNoChange > 0) {
            console.log(
                `[SYNC-BACKOFF] 🔄 변경 감지. 폴링 간격을 ${syncCurrentIntervalMs}ms → ${SYNC_BASE_INTERVAL_MS}ms 로 리셋합니다.`
            );
        }
        syncConsecutiveNoChange = 0;
        syncCurrentIntervalMs = SYNC_BASE_INTERVAL_MS;
    } else {
        syncConsecutiveNoChange++;
        syncCurrentIntervalMs = computeSyncInterval();
        console.log(
            `[SYNC-BACKOFF] ⏳ 변경 없음 (${syncConsecutiveNoChange}회 연속). ` +
            `다음 폴링까지 ${syncCurrentIntervalMs}ms 대기합니다.`
        );
    }
}

/**
 * 🌟 [ORIGIN RESOLUTION] 이벤트를 조회할 '추적 대상 사이트' 목록을 확정합니다.
 *  Worker 가 cc 를 href 의 host 로 만들기 때문에, 조회 단위는 곧 origin 입니다.
 *
 *  수집 소스 (우선순위 무관, 중복 제거)
 *   ① kv_store 의 oauth_registered_sites  : 사용자가 명시적으로 등록한 사이트
 *   ② currentDetectedUrl                  : 지금 브라우저가 보고 있는 사이트
 *   ③ Dexie 의 기존 analytic 문서 data.origin : 과거에 한 번이라도 받아온 사이트
 *
 *  ⚠️ origin(스킴+호스트)만 사용합니다. 경로/쿼리를 붙이면 Worker 의
 *     decodeURIComponent(req.query.href) 가 2중 디코딩되어 '%' 가 들어간 경로에서
 *     URIError 로 418 을 맞습니다. (cc 는 host 로만 결정되므로 경로는 불필요)
 */
async function resolveAnalyticsOrigins(): Promise<string[]> {
    const origins = new Set<string>();

    const push = (raw: any) => {
        if (!raw || typeof raw !== "string") return;
        let s = raw.trim().toLowerCase();
        if (!s) return;
        // 🌟 [SCHEME TOLERANCE] api.oauth.network 메타데이터는 'example.com' 처럼
        //    스킴 없이 저장됩니다. 기존 startsWith("http") 검사는 그 항목을 전부 버려
        //    '등록된 사이트' 가 조회 대상에서 통째로 빠졌습니다.
        if (!/^https?:\/\//.test(s)) s = "https://" + s;
        try {
            const u = new URL(s);
            if (!u.hostname) return;
            if (u.hostname === "localhost" || u.hostname === "127.0.0.1") return;
            // logis.center 계열은 content.js 의 isShop 게이트에서 애초에 제외되므로
            // 이벤트가 존재할 수 없습니다. 조회해 봐야 0건이라 왕복만 낭비합니다.
            if (u.hostname.endsWith("logis.center")) return;
            origins.add(u.origin);
        } catch (e) {}
    };

    try {
        const sites = await kvGet("oauth_registered_sites");
        if (Array.isArray(sites)) {
            for (const s of sites) push(s && s.host);
        }
    } catch (e) {}

    push(currentDetectedUrl);

    try {
        if (appDb) {
            const rows = await appDb.table("items").where("mode").equals("analytic").limit(2000).toArray();
            for (const r of rows) push(r && r.data && r.data.origin);
        }
    } catch (e) {}

    return Array.from(origins);
}

/**
 * 🌟 [TEXT EXTRACT] Cron Worker 가 구조화한 문장을 골라냅니다.
 *  원시 이벤트의 action 은 [outerHTML] '배열' 이므로 문자열일 때만 채택합니다.
 *  (배열을 그대로 text 로 쓰면 HTML 덩어리가 FTS/임베딩에 들어갑니다)
 */
function extractAnalyticText(parsed: any): string {
    const pick = (v: any): string => (typeof v === "string" ? v.trim() : "");
    return pick(parsed?.action)
        || pick(parsed?.summary)
        || pick(parsed?.cross_action_flow)
        || pick(parsed?.intent_evolution)
        || pick(parsed?.text);
}

/** 🌟 D1 BLOB(number[] / ArrayBuffer / base64) → JSON 객체로 복원합니다. */
function decodeAnalyticBlob(rawData: any): any {
    const pako = (window as any).pako;
    try {
        if (rawData && typeof rawData === "object") {
            const raw = rawData.data || rawData;
            let arr: Uint8Array | null = null;

            if (Array.isArray(raw)) {
                arr = new Uint8Array(raw);
            } else if (raw.buffer) {
                arr = new Uint8Array(raw.buffer);
            } else if (Object.keys(raw).length > 0 && !isNaN(Number(Object.keys(raw)[0]))) {
                arr = new Uint8Array(Object.values(raw) as number[]);
            }

            if (arr) {
                try {
                    return JSON.parse(pako ? pako.ungzip(arr, { to: 'string' }) : new TextDecoder('utf-8').decode(arr));
                } catch (e) {
                    return JSON.parse(new TextDecoder('utf-8').decode(arr));
                }
            }
            return raw;
        }

        if (typeof rawData === "string") {
            // 🌟 [BASE64 GZIP PATH] Worker 가 BLOB 을 base64 로 직렬화했거나
            //    구버전 Rust upsert_items 가 base64(gzip) 로 저장한 경우를 처리합니다.
            try {
                return JSON.parse(rawData);
            } catch (_jsonErr) {
                try {
                    const rawBytes = Uint8Array.from(atob(rawData), c => c.charCodeAt(0));
                    if (rawBytes.length > 50 && rawBytes[0] === 0x1f && rawBytes[1] === 0x8b) {
                        return JSON.parse(pako ? pako.ungzip(rawBytes, { to: 'string' }) : new TextDecoder('utf-8').decode(rawBytes));
                    }
                    return JSON.parse(new TextDecoder('utf-8').decode(rawBytes));
                } catch (_b64Err) {
                    return {};
                }
            }
        }
    } catch (e) {}
    return {};
}

/**
 * 🌟 [API KEY LOOKUP] origin 에 대응하는 client_id / client_secret 을 kv_store 에서 찾습니다.
 *
 *  ── 왜 필요한가 ──
 *   console.logis.center 의 로그 조회는 이제 '그 사이트에 발급된 API 키' 를 요구합니다.
 *   키가 없으면 서버가 0건을 돌려주므로, 아예 왕복하지 않고 건너뛰는 편이 낫습니다.
 *
 *  ── 호스트 표기 흔들림 ──
 *   kv_store 의 host 는 등록 경로에 따라 'https://example.com' 또는 'example.com' 으로
 *   섞여 저장될 수 있습니다. 양쪽 모두 URL 로 정규화한 뒤 host(도메인+포트)로 비교합니다.
 */
async function getOAuthCredentialForOrigin(origin: string): Promise<{ client_id: string; client_secret: string } | null> {
    const toHost = (raw: any): string => {
        if (!raw || typeof raw !== "string") return "";
        let s = raw.trim().toLowerCase();
        if (!s) return "";
        if (!/^https?:\/\//.test(s)) s = "https://" + s;
        try {
            return new URL(s).host;
        } catch (_e) {
            return "";
        }
    };

    const targetHost = toHost(origin);
    if (!targetHost) return null;

    try {
        const sites = await kvGet("oauth_registered_sites");
        if (!Array.isArray(sites)) return null;

        for (const s of sites) {
            if (!s) continue;
            if (toHost(s.host) !== targetHost) continue;
            if (!s.client_id) continue;
            return {
                client_id: String(s.client_id),
                client_secret: String(s.client_secret || "")
            };
        }
    } catch (e) {
        console.warn("[OAUTH] getOAuthCredentialForOrigin failed:", e);
    }

    return null;
}

/** 🌟 origin 하나에 대해 GET 1회를 수행하고 저장한 건수를 돌려줍니다. */
async function fetchAnalyticsOrigin(origin: string, cursor: number): Promise<number> {
    // Worker 와 '동일한 규칙' 으로 cc 를 계산합니다. (풀 호스트, 루트 도메인 아님)
    let expectedCc = "";
    try {
        expectedCc = await hashId(new URL(origin).host);
    } catch (e) {}

    // 🌟 [API KEY GATE] Worker 가 이제 client_id 로 사이트 소유를 검증합니다.
    //    키가 없으면 서버가 0건을 돌려주므로 왕복 자체를 생략합니다.
    const cred = await getOAuthCredentialForOrigin(origin);

    if (!cred) {
        console.warn(
            `[SYNC-ANALYTIC] ⏭️ '${origin}' 은 등록된 API 키가 없어 건너뜁니다. ` +
            `Analytic 탭의 '+ 사이트 등록' 으로 먼저 등록하세요. ` +
            `(console.logis.center 는 client_id 검증을 통과한 요청에만 로그를 반환합니다)`
        );
        return 0;
    }

    const params = new URLSearchParams({
        origin: "https://console.logis.center",
        created_at: cursor.toString(),
        hash: currentSession.hash,
        token: currentSession.token || "",
        // ⚠️ 경로 없는 origin + '/' 만 보냅니다. Worker 의 2중 디코딩 대비.
        href: origin + "/"
    });
    if (expectedCc) params.append("cc", expectedCc);

    // 🌟 client_id 는 필수, client_secret 은 있으면 쌍까지 대조합니다.
    params.append("client_id", cred.client_id);
    if (cred.client_secret) params.append("client_secret", cred.client_secret);

    let response: any = null;
    try {
        response = await invoke<any>("proxy_fetch", {
            url: `${ANALYTICS_API_HOST}/?${params.toString()}`,
            method: "GET",
            headers: { "Content-Type": "application/json" },
            session_params: { hash: currentSession.hash, token: currentSession.token, cc: expectedCc }
        });
    } catch (e) {
        console.warn(`[SYNC-ANALYTIC] ❌ '${origin}' 조회 실패 (cc=${expectedCc}):`, e);
        return 0;
    }

    stepQrSpinner();

    // 🌟 [VERIFY ECHO] Worker 가 session.verified / session.verify_reason 을 실어 보냅니다.
    //    실패 원인이 클라이언트에서 바로 보이도록 표면화합니다.
    const verifySession = (response && response.session) ? response.session : {};
    if (verifySession.verify_reason && verifySession.verified === false) {
        console.warn(
            `[SYNC-ANALYTIC] 🔐 '${origin}' API 키 검증 실패: reason='${verifySession.verify_reason}' ` +
            `(client_id='${cred.client_id}'). Analytic 탭에서 '재발급' 후 다시 시도하거나, ` +
            `사이트 <head> 의 oauth-network-verification 메타 태그를 확인하세요.`
        );
    }

    if (!response || !response.results || !Array.isArray(response.results)) {
        console.log(`[SYNC-ANALYTIC] '${origin}' (cc=${expectedCc}) → 응답에 results 없음`);
        return 0;
    }

    if (response.results.length === 0) {
        console.log(
            `[SYNC-ANALYTIC] '${origin}' (cc=${expectedCc}) → 0건. ` +
            `Worker 조회 조건은 cc = hashId('${(() => { try { return new URL(origin).host; } catch (e) { return "?"; } })()}') ` +
            `AND created_at < ${cursor} 이며, client_id 검증을 통과해야 합니다. ` +
            `(verify='${verifySession.verify_reason || "unknown"}')`
        );
        return 0;
    }

    const now = Date.now();
    const items: any[] = [];

    // 🌟 [DELTA GUARD] 이미 같은 updated_at 으로 로컬에 있는 행은 쓰기를 생략합니다.
    const localMap = new Map<string, number>();
    try {
        if (appDb) {
            const ids = response.results.map((r: any) => r && r.id).filter(Boolean);
            if (ids.length > 0) {
                const rows = await appDb.table("items").where("id").anyOf(ids).toArray();
                for (const r of rows) localMap.set(String(r.id), Number(r.updated_at || 0));
            }
        }
    } catch (e) {}

    let skipped = 0;
    for (let i = 0; i < response.results.length; i++) {
        const row = response.results[i];
        if (!row || !row.id) continue;

        const serverUpdated = Number(row.updated_at || 0);
        const localUpdated = localMap.get(String(row.id));

        if (localUpdated !== undefined && serverUpdated <= localUpdated) {
            skipped++;
            continue;
        }

        if (localUpdated !== undefined && localUpdated > 0) {
            skipped++;
            console.log(
                `[SYNC-ANALYTIC] ⚪ '${row.id}' 는 이미 구조화 완료되어 서버 원시 데이터로 덮어쓰지 않습니다.`
            );
            continue;
        }

        const parsed: any = decodeAnalyticBlob(row.data) || {};

        // 🌟 [ORIGIN SEED] question / answer 의 data 에는 origin 이 없습니다.
        //    다음 라운드의 resolveAnalyticsOrigins() 가 이 사이트를 계속 발견하려면
        //    반드시 채워 두어야 합니다.
        if (!parsed.origin) parsed.origin = origin;

        const textVal = extractAnalyticText(parsed);

        // 🌟 [BCC RECONSTRUCT] analytics D1 스키마에는 bcc 컬럼이 없습니다.
        //    (id/type/flag/from/to/cc/ref/data/created_at/updated_at 뿐)
        //    commerce 와 동일한 규칙 bcc = hashId(type + cc) 로 클라이언트가 재구성합니다.
        const rowType = String(row.type || "click");
        const rowCc = String(row.cc || expectedCc || "");
        let rowBcc = String(row.bcc || "");
        if (!rowBcc && rowCc) {
            rowBcc = await hashId(rowType + rowCc);
        }

        // 🌟 [MODE TAGGING] modeOfType() 단일 판정 경로를 그대로 사용합니다.
        const rowMode = modeOfType(rowType);

        // 🌟 [RAW MARKER] 아직 구조화되지 않은 원시 이벤트인지 판정합니다.
        //    content.js 는 action / relate 를 outerHTML '배열' 로 올리므로
        //    배열이면 곧 '구조화 전' 이라는 구조적 사실입니다.
        const isRawEvent = Array.isArray(parsed?.action) || Array.isArray(parsed?.relate);

        items.push({
            id: row.id,
            type: rowType,
            // 🌟 [FLAG] analytics D1 은 flag 를 실제로 채워 보냅니다. 비었을 때만 세션 flag 로 보강.
            flag: row.flag || String((currentSession as any).flag || ""),
            from: row.from || "",
            to: row.to || "",
            cc: rowCc,
            bcc: rowBcc,
            ref: row.ref || "",
            status: 9,
            mode: rowMode,
            created_at: Number(row.created_at || now),
            // ⚠️ updated_at 은 draft/count 계약이자 '구조화 대기' 마커입니다.
            //    ── 왜 now 로 덮으면 안 되는가 ──
            //     Rust 의 structure_pending_analytics 는
            //       mode = 'analytic' AND updated_at = 0
            //     로 구조화 대상을 찾습니다. 여기서 now 를 넣으면
            //     원시 outerHTML 이 영원히 요약되지 않아
            //     text 가 빈 채로 남고, 검색이 구조적으로 0건이 됩니다.
            updated_at: isRawEvent ? 0 : serverUpdated,
            text: textVal,
            masked_text: textVal,
            data: {
                ...parsed,
                id: row.id,
                type: rowType,
                mode: rowMode,
                updated_at: isRawEvent ? 0 : serverUpdated,
                text: textVal,
                masked_text: textVal
            }
        });
    }

    if (items.length === 0) {
        console.log(`[SYNC-ANALYTIC] '${origin}' 수신 ${response.results.length}건 → 전부 최신 상태(스킵 ${skipped}건)`);
        return 0;
    }

    await invoke("upsert_items", { items });
    if (appDb) {
        await appDb.table("items").bulkPut(normalizeEnvelope(items)).catch(() => null);
    }

    const typeBrief: Record<string, number> = {};
    for (const it of items) typeBrief[it.type] = (typeBrief[it.type] || 0) + 1;

    console.log(
        `[SYNC-ANALYTIC] ✅ '${origin}' (cc=${expectedCc}) → 수신 ${response.results.length}건 / ` +
        `저장 ${items.length}건 / 스킵 ${skipped}건 ${JSON.stringify(typeBrief)}`
    );

    return items.length;
}

async function syncAnalyticsData() {
    if (!currentSession.hash) return;
    if (isAnalyticsSyncRunning) return;

    isAnalyticsSyncRunning = true;

    try {
        const origins = await resolveAnalyticsOrigins();

        if (origins.length === 0) {
            console.warn(
                "[SYNC-ANALYTIC] 조회할 추적 대상 사이트가 없습니다. " +
                "Analytic 탭의 '+ 사이트 등록' 으로 도메인을 등록하거나, " +
                "브라우저로 추적 대상 사이트를 열어 두세요. " +
                "(Worker 는 cc 파라미터가 아니라 href 의 host 로 조회 대상을 결정합니다)"
            );
            return;
        }

        // 🌟 [CURSOR] created_at 은 상한 커서입니다. 반드시 '현재보다 미래' 여야
        //    UTC- 지역에서 최근 이벤트가 잘려 나가지 않습니다.
        const now = Date.now();
        const cursor = Math.max(now, now - timezoneOffset) + 60_000;

        console.log(`[SYNC-ANALYTIC] 대상 사이트 ${origins.length}곳 조회 시작: ${JSON.stringify(origins)} | cursor=${cursor}`);

        let totalStored = 0;
        for (const origin of origins) {
            totalStored += await fetchAnalyticsOrigin(origin, cursor);
        }

        if (totalStored > 0) {
            if (currentSearchMode === "analytic") {
                await renderNavigation();
                if (currentTab === "list") {
                    await loadMoreDocs(false, true);
                }
            }
            // 🌟 [STRUCTURE FIRST] 원시 outerHTML 을 먼저 시맨틱 문장으로 확정합니다.
            //    이 단계가 끝나야 text 가 채워지고, 그때서야 임베딩이 의미를 갖습니다.
            const structuredCount = await runAnalyticStructuring();
            // 🌟 [STRUCTURE BUBBLE DONE] 구조화 말풍선들을 완료 처리합니다.
            //    analytic_sync_sem_* / analytic_sync / analytic_sync_flow 가
            //    status=1(PROCESSING) 에서 영원히 멈춰 있는 것을 방지합니다.
            if (structuredCount > 0) {
                await finalizeAnalyticBubbles(structuredCount);
            }
            // 🌟 [CLIENT-SIDE EMBEDDING] 서버는 벡터를 만들지 않으므로 로컬에서 즉시 임베딩합니다.
            runLocalEmbeddingSync();
        } else {
            // 🌟 새로 받은 건이 없어도, 이전 라운드에서 구조화하지 못하고 남은
            //    원시 이벤트가 있으면 이어서 처리합니다.
            //    (LLM 로드 비용은 대상 0건일 때 백엔드가 probe 단계에서 차단합니다)
            const structured = await runAnalyticStructuring();
            if (structured > 0) {
                await finalizeAnalyticBubbles(structured);
                runLocalEmbeddingSync();
            }
        }

        console.log(`[SYNC-ANALYTIC] 완료. 총 저장 ${totalStored}건.`);

    } catch (e) {
        console.warn("[SYNC-ANALYTIC] Failed:", e);
    } finally {
        isAnalyticsSyncRunning = false;
        lastAnalyticsSyncAt = Date.now();
        if (!isExtracting && !isSearching) stopSpinner();
        // 🌟 [SYNC DONE → SUBMIT RESTORE] syncData 와 동일하게 검색 버튼 상태를 복원합니다.
        if (btnSubmit) {
            const currentVal = searchInput?.value.trim() || "";
            if (currentVal !== "" && !isQueryActive(currentVal)) {
                btnSubmit.style.display = "flex";
            } else {
                btnSubmit.style.display = "none";
            }
        }
    }
}

// 🌟 [ANALYTICS BACKGROUND SYNC] commerce / shipping 탭에 있어도 analytics 이벤트가
//    계속 흘러 들어오도록 하는 역방향 경로입니다.
//    (기존에는 syncCommerceInBackground 만 있고 이 방향이 없어서,
//     Analytic 탭 + 채팅 화면 + 위젯 확장 3조건이 동시에 성립할 때만 동기화되었습니다)
//    폴링 주기(3초)마다 N개 사이트를 왕복하면 과하므로 30초 스로틀을 겁니다.
async function syncAnalyticsInBackground() {
    if (!currentSession.hash) return;
    if (isAnalyticsSyncRunning) return;
    // 🌟 [ADAPTIVE THROTTLE] 기존 고정 30초 대신 백오프 간격을 적용합니다.
    //    기본 3초에서 시작해 변경 없음이 연속되면 1.5배씩 증가, 최대 30초.
    const throttleMs = Math.max(30_000, syncCurrentIntervalMs);
    if (Date.now() - lastAnalyticsSyncAt < throttleMs) return;
    await syncAnalyticsData();
}

// 🌟 [COMMERCE BACKGROUND SYNC] analytic 모드에서도 commerce.logis.center D1 과
//    양방향 동기화를 수행합니다. syncData() 의 commerce 경로를 재사용하되,
//    UI 갱신(renderNavigation / loadMoreDocs)은 현재 탭이 commerce 일 때만 수행합니다.
let isCommerceSyncRunning = false;
async function syncCommerceInBackground() {
    if (isCommerceSyncRunning) return;
    if (!currentSession.hash || !currentSession.email) return;
    isCommerceSyncRunning = true;
    try {
        const origin = "https://commerce.logis.center";
        const now = Date.now();
        const createdAt = now - timezoneOffset;
        let targetHref = currentDetectedUrl || "https://commerce.logis.center/tracking";
        if (targetHref.includes("localhost") || targetHref.includes("127.0.0.1") || targetHref === "about:blank") {
            targetHref = "https://commerce.logis.center/tracking";
        }
        const queryParams: any = {
            origin: origin,
            created_at: createdAt.toString(),
            hash: currentSession.hash,
            token: currentSession.token || "",
            href: targetHref
        };
        let syncEffectiveCc = activeContext.cc;
        const isDefaultForced = activeTags.some(t => t.value === "logis.center" && t.type === "domain");
        if (!syncEffectiveCc || (!isDefaultForced && activeTags.length === 0)) {
            try {
                const urlObj = new URL(targetHref.toLowerCase());
                const rootDomain = getRootDomain(urlObj.hostname);
                syncEffectiveCc = await hashId(rootDomain);
            } catch(e) {}
        }
        if (syncEffectiveCc) queryParams.cc = syncEffectiveCc;
        const params = new URLSearchParams(queryParams);
        const url = `${API_HOST}/?${params.toString()}`;
        const response = await invoke<any>("proxy_fetch", {
            url: url,
            method: "GET",
            headers: { "Content-Type": "application/json" },
            session_params: { hash: currentSession.hash, token: currentSession.token, cc: activeContext.cc || "" }
        });
        if (response.results && Array.isArray(response.results)) {
            try {
                const pako = (window as any).pako;
                for (let i = 0; i < response.results.length; i++) {
                    let item = response.results[i];
                    if (item.data && typeof item.data === 'object' && !item.data.text && !item.data.title) {
                        let arrData = item.data.data || item.data;
                        let arr: Uint8Array | null = null;
                        if (Array.isArray(arrData)) {
                            arr = new Uint8Array(arrData);
                        } else if (arrData.buffer) {
                            arr = new Uint8Array(arrData.buffer);
                        } else if (Object.keys(arrData).length > 0 && !isNaN(Number(Object.keys(arrData)[0]))) {
                            arr = new Uint8Array(Object.values(arrData) as number[]);
                        }
                        if (arr) {
                            try {
                                if (pako) {
                                    const decompressed = pako.ungzip(arr, { to: 'string' });
                                    item.data = JSON.parse(decompressed);
                                } else {
                                    const decompressed = new TextDecoder('utf-8').decode(arr);
                                    item.data = JSON.parse(decompressed);
                                }
                            } catch (e) {
                                try {
                                    const decompressed = new TextDecoder('utf-8').decode(arr);
                                    item.data = JSON.parse(decompressed);
                                } catch (err) {}
                            }
                        }
                    } else if (typeof item.data === 'string' && item.data.length > 50) {
                        // 🌟 [BASE64 GZIP PATH] syncAnalyticsData 와 동일한 처리.
                        //    data 가 base64(gzip) 문자열로 내려오는 경우를 커버합니다.
                        try {
                            const rawBytes = Uint8Array.from(atob(item.data), c => c.charCodeAt(0));
                            if (rawBytes[0] === 0x1f && rawBytes[1] === 0x8b) {
                                item.data = JSON.parse(pako ? pako.ungzip(rawBytes, { to: 'string' }) : new TextDecoder('utf-8').decode(rawBytes));
                            } else {
                                item.data = JSON.parse(new TextDecoder('utf-8').decode(rawBytes));
                            }
                        } catch (_strErr) {
                            // JSON 도 base64 도 아니면 원본 문자열을 그대로 둡니다.
                        }
                    }
                }
            } catch (err) {
                console.warn("[SYNC-BG] pako decompression failed:", err);
            }
            // 🌟 [TOMBSTONE GUARD] analytic 탭에서도 commerce D1 을 백그라운드로 긁으므로
            //    syncData 와 동일한 부활 경로가 열려 있습니다. 같은 게이트를 적용합니다.
            const bgTombstones = await loadTalkTombstones();
            const bgItemTombs = await loadItemTombstones();
            let bgTombBlocked = 0;
            let bgItemTombBlocked = 0;
            const filteredResults = response.results.filter((newItem: any) => {
                if (bgTombstones.has(String(newItem.id))) {
                    bgTombBlocked++;
                    return false;
                }
                // 🌟 [ITEM TOMBSTONE CHECK] 문서 삭제도 백그라운드 재삽입을 차단합니다.
                if (bgItemTombs.has(String(newItem.id))) {
                    bgItemTombBlocked++;
                    return false;
                }
                const existingEl = document.getElementById(newItem.id);
                if (!existingEl) return true;
                let localUpdated = parseInt(existingEl.dataset.updatedAt || existingEl.dataset.createdAt || "0");
                const serverUpdated = newItem.updated_at || newItem.created_at || 0;
                return serverUpdated > localUpdated;
            });
            if (bgTombBlocked > 0) {
                console.log(`[TOMBSTONE] 🪦 [BG] 삭제된 메시지 ${bgTombBlocked}건의 백그라운드 재삽입을 차단했습니다.`);
            }
            if (bgItemTombBlocked > 0) {
                console.log(`[ITEM-TOMBSTONE] 🪦 [BG] 삭제된 문서 ${bgItemTombBlocked}건의 백그라운드 재삽입을 차단했습니다.`);
            }
            if (filteredResults.length > 0) {
                for (const r of filteredResults) {
                    if (!r) continue;
                    if (r.mode) continue;
                    r.mode = modeOfType(String(r.type || ""));
                }
                const sessionFlag = String((currentSession as any).flag || "");
                if (sessionFlag) {
                    for (const r of filteredResults) {
                        if (!r) continue;
                        if (r.flag) continue;
                        const inner = (r.data && typeof r.data === 'object') ? r.data.flag : undefined;
                        if (inner) { r.flag = inner; continue; }
                        r.flag = sessionFlag;
                    }
                }
                for (const r of filteredResults) {
                    if (!r) continue;
                    if (!r.data || typeof r.data !== 'object') continue;
                    for (const k in r) {
                        if (!Object.prototype.hasOwnProperty.call(r, k)) continue;
                        if (ENVELOPE_ROOT_KEYS.has(k)) continue;
                        const v = r[k];
                        if (v === undefined || v === null) continue;
                        if (typeof v === 'function') continue;
                        if (r.data[k] === undefined || r.data[k] === null || r.data[k] === "") {
                            r.data[k] = v;
                        }
                    }
                }
                // 🌟 [BACKOFF RESET] 백그라운드 commerce 동기화에서 변경이 있었으므로 리셋합니다.
                updateSyncBackoff(true);
                console.log(`[SYNC-BG] Commerce D1 background sync: ${filteredResults.length} item(s)`);
                await invoke("upsert_items", { items: filteredResults });
                // 🌟 [PAGE CACHE GUARD] syncData 와 동일한 규칙으로 페이지 셀렉터 캐시를 분리합니다.
                //    기존에는 node / item 체크 없이 type 만으로 필터링하여
                //    { table:'pages', type:'tracking', node:1, item:true } 문서를
                //    items 테이블에 잘못 저장했습니다.
                //    (로그: "[DEBUG] Syncing item - ID: , Type: tracking" 의 직접 원인)
                const bgNewPages = filteredResults.filter((r: any) => {
                    if (r.table === "pages" || r.table === "page") return true;
                    if (r.type === "pages" || r.type === "page") return true;
                    const d = typeof r.data === 'string' ? JSON.parse(r.data) : (r.data || r);
                    return !!d.node || !!d.item;
                });
                if (bgNewPages.length > 0 && appDb) {
                    await appDb.table("pages").bulkPut(normalizeEnvelope(bgNewPages)).catch(() => null);
                }
                const newItems = filteredResults.filter((r: any) =>
                    r.type !== "team" && r.type !== "user" && r.type !== "member"
                    && r.type !== "pages" && r.type !== "page"
                    && r.type !== "talk" && r.table !== "talks"
                    && !bgNewPages.includes(r)
                );
                if (newItems.length > 0 && appDb) {
                    await appDb.table("items").bulkPut(normalizeEnvelope(newItems)).catch(() => null);
                }
                // 현재 탭이 commerce/list 일 때만 UI 갱신
                if (currentSearchMode === "commerce" && currentTab === "list") {
                    await loadMoreDocs(false, true);
                }
                runLocalEmbeddingSync();
            }
        }
    } catch (e) {
        console.warn("[SYNC-BG] Commerce background sync failed:", e);
    } finally {
        isCommerceSyncRunning = false;
        // 🌟 [BACKOFF] 백그라운드 commerce 동기화에서 변경이 없었으면
        //    syncData 의 백오프 카운터가 이미 갱신되어 있으므로 여기서는 추가 처리 불필요.
        //    (syncData 가 filteredResults 판정으로 updateSyncBackoff 를 호출합니다)
    }
}


// [통합 락 매니저 & 프론트엔드 큐 관리자]
if (!(window as any).Dexie) {
    console.error("🚨 [ERROR] Dexie library is missing! public 폴더 안의 파일들은 반드시 절대경로(/)로 불러와야 합니다.");
}
const DexieLocal = (window as any).Dexie;

const appDb = new DexieLocal("LogisAppDB");

// 🌟 [v8 / TRADING INDEXES]
//  무역(shipping/trading) 트랙 전용 조회 축을 추가합니다.
//  스토어 구조는 v7 과 동일하고 인덱스만 추가하므로 Dexie 가 자동 백필합니다.
//  → upgrade 콜백이 필요 없습니다.
//
//  ── 왜 봉투를 안 늘리는가 ──
//   app-logis-center 의 TradeDocument 는 vessel/pol/pod/incoterms 등 55개를
//   Rust 구조체에 못 박아 두어, 새 무역 서식이 추가될 때마다
//   구조체 → LanceDB 스키마 → 프론트엔드 3곳을 동시에 고쳐야 했습니다.
//   v4 봉투 구조에서는 값이 전부 data 안에 있으므로,
//   '자주 eq/range 로 조회하는 경로' 만 여기에 한 줄씩 추가하면 끝입니다.
//
//  ⚠️ 여기 없는 무역 필드(marks_numbers, notify_party_name, place_receipt 등)는
//     executeDexiePlan 이 .filter() 풀스캔으로 처리합니다.
//     로컬 수천~수만 건 기준 수 ms 이므로 인덱스가 없어도 동작에 지장이 없습니다.
// 🌟 [SCHEMA CONTRACT]
//  Dexie 는 version(N).stores() 에 '선언되지 않은' object store 를 업그레이드 시점에 삭제합니다.
//  v8 이 items 하나만 선언한 탓에 kv_store / ts_queue / talks / users / pages 5개가
//  물리적으로 소멸했고, 그 결과 세션(kv_store)까지 함께 날아가 로그인이 풀렸습니다.
//  → 앞으로 stores() 에는 '앱이 쓰는 전 테이블' 을 항상 함께 적어야 합니다.
const ITEMS_SCHEMA = [
    // ── 봉투 (v7 그대로 유지) ──
    'id', 'type', 'flag', 'from', 'to', 'cc', 'bcc', 'ref', 'mode',
    'created_at', 'updated_at',
    '[cc+type]', '[mode+type]', '[ref+created_at]', '[mode+updated_at]',
    // ── commerce 축 (v7 그대로 유지) ──
    'data.index', 'data.no', 'data.code', 'data.tracking_number',
    'data.goods', 'data.order', 'data.tracking',
    'data.stock_keeping_unit', 'data.barcode',
    'data.status', 'data.amount', 'data.sale_price', 'data.supply_price',
    'data.quantity', 'data.weight', 'data.discount',
    'data.carrier', 'data.shipping_method',
    'data.started_at', 'data.expired_at',
    'data.title', 'data.name', 'data.sender_name', 'data.recipient_name',
    'data.embed', 'data.digest',
    '*data.tags',
    // ── 🌟 trading 축 ──
    //  ① 문서 식별 : B/L No, AWB No, PO No, Booking No 를 하나로 흡수
    'data.doc_type', 'data.doc_number', 'data.issue_date',
    //  ② 운송 : 선박/항공편 + 출발/도착 항구 (교차 조회 최다 축)
    'data.vessel', 'data.voyage_number', 'data.pol', 'data.pod',
    'data.etd', 'data.eta',
    //  ③ 계약 : 인코텀즈 / 결제조건 (Enum 성격, 카디널리티 낮지만 eq 조회 빈발)
    'data.incoterms', 'data.payment_terms', 'data.currency',
    //  ④ 화물 : 컨테이너/씰 번호 (식별자, 카디널리티 최상)
    'data.container_number', 'data.seal_number',
    'data.package_count', 'data.weight_gross', 'data.weight_net', 'data.volume',
    //  ⑤ 참조 : 인보이스/LC 상호 참조 (N:N RELAY 축)
    'data.reference_invoice', 'data.reference_lc', 'data.reference_booking',
    //  ⑤-1 🌟 [TRADING INDEX RELAY] commerce 의 data.order / data.tracking 과 동일한 역할입니다.
    //     logic::trading_index_column() 이 만드는 crc32 숫자 인덱스 컬럼으로,
    //     'BL 하나로 연결된 CI/PL 전부 조회' 를 O(log n) 으로 처리합니다.
    //     문자열 doc_number 를 그대로 쓰면 표기 흔들림(대소문자/하이픈)에 매번 어긋납니다.
    //  ⑤-1 🌟 [TRADING INDEX RELAY] ...
    //     ⚠️ logic.rs 의 trading_index_column() 은 'rel_{소문자}' 를 45종 전부에 대해
    //        생성합니다. 여기 없는 축은 인덱스 없이 .filter() 풀스캔으로 떨어집니다.
    //        (동작은 하지만 문서가 늘수록 느려집니다)
    'data.rel_bl', 'data.rel_hbl', 'data.rel_swb', 'data.rel_awb',
    'data.rel_ci', 'data.rel_cinv', 'data.rel_csi', 'data.rel_pi', 'data.rel_pl',
    'data.rel_po', 'data.rel_sc', 'data.rel_lc', 'data.rel_llc', 'data.rel_co',
    'data.rel_bc', 'data.rel_bk', 'data.rel_sr', 'data.rel_do', 'data.rel_an',
    'data.rel_sa', 'data.rel_fcr', 'data.rel_pod', 'data.rel_cm', 'data.rel_fi',
    'data.rel_wr', 'data.rel_ed', 'data.rel_id', 'data.rel_ccc', 'data.rel_cnm',
    'data.rel_el', 'data.rel_ic', 'data.rel_wc', 'data.rel_ca', 'data.rel_coa',
    'data.rel_pc', 'data.rel_fc', 'data.rel_hc', 'data.rel_cdr',
    'data.rel_ip', 'data.rel_icf', 'data.rel_lg', 'data.rel_tr',
    'data.rel_soa', 'data.rel_dn', 'data.rel_cn', 'data.rel_ti', 'data.rel_cp',
    'data.rel_be', 'data.rel_ins', 'data.rel_dgd',
    //  ⑤-2 🌟 [BASE v2 신규 축] bias.json trade_schema.base v2 에서 추가된 조회 축입니다.
    'data.reference_bl', 'data.reference_po', 'data.reference_contract',
    'data.reference_master_bl', 'data.reference_sr', 'data.reference_number',
    'data.expiry_date', 'data.place_receipt', 'data.place_delivery',
    'data.flight_number', 'data.departure_date', 'data.arrival_date',
    'data.transport_mode', 'data.freight_payment_term',
    'data.amount_subtotal', 'data.amount_tax', 'data.freight_amount',
    'data.due_date', 'data.payment_status',
    'data.package_unit', 'data.type_size', 'data.hs_code',
    //  ⑥ 복합 : 무역 문서는 '문서종류(type) + 발행일' 로 스캔하는 빈도가 압도적입니다.
    //     doc_type 은 data.* 경로라 복합 인덱스의 구성 요소로 쓸 수 없으므로,
    //     봉투 type 컬럼(BL/AWB/CI/PI/...)과 발행일을 묶습니다.
    '[type+created_at]'
].join(', ');


// 🌟 v13 : talk_tombstones 추가 — 삭제한 채팅의 '묘비'
//  ── 왜 필요한가 ──
//   Client Worker(index.ts)에는 talk 개별 삭제 엔드포인트가 없습니다.
//     DELETE 핸들러 = { type=crons → tasks 삭제 } | { 그 외 → S3 인증 해시 삭제(로그아웃) }
//   PUT 은 talk.id 를 hashId() 난수로 새로 발급하므로 기존 행을 지목할 수도 없습니다.
//   따라서 서버 행은 살아남고, 로컬에서만 지우면 다음 syncData 폴링(3초)이
//   `!existingEl && !localMap.has(id)` 조건을 통과시켜 그대로 부활시킵니다.
//   '이 id 는 내가 의도적으로 지웠다' 는 사실을 로컬에 영구 기록해야
//   재삽입을 구조적으로 차단할 수 있습니다.
//
//  ── 왜 GC 를 하지 않는가 ──
//   서버 행이 살아 있는 한 묘비를 지우는 순간 메시지가 되살아납니다.
//   서버 행의 소멸 여부를 클라이언트가 확인할 방법이 없으므로 만료를 두지 않습니다.
//   행 하나는 (42자 id + 타임스탬프) 이므로 1만 건 삭제해도 1MB 미만입니다.
//   전체 초기화(btn-reset-db)는 Dexie DB 자체를 삭제하므로 함께 정리됩니다.
//
//  ⚠️ [SCHEMA CONTRACT] Dexie 는 stores() 에 선언되지 않은 object store 를
//     업그레이드 시점에 삭제합니다. 아래에는 v12 의 전 테이블이 그대로 포함되어 있습니다.
// 🌟 v14 : trading 인덱스 축 확장 (rel_* 45종 + base v2 신규 축)
//  스토어 구조는 v13 과 동일하고 인덱스만 추가하므로 Dexie 가 자동 백필합니다.
//  → upgrade 콜백이 필요 없습니다.
//
//  ⚠️ [SCHEMA CONTRACT] Dexie 는 stores() 에 선언되지 않은 object store 를
//     업그레이드 시점에 삭제합니다. 아래에는 v13 의 전 테이블이 그대로 포함되어 있습니다.
appDb.version(14).stores({
    items: ITEMS_SCHEMA,
    kv_store: 'key',
    ts_queue: 'taskId, type',
    talks: 'id, type, role, from, to, cc, bcc, ref, task_id, status, created_at, updated_at',
    users: 'id, type, flag, from, to, cc, bcc, ref, mode, created_at, updated_at, data.is_device, data.email, data.origin',
    pages: 'id, type, flag, from, to, cc, bcc, ref, mode, created_at, updated_at, data.type, data.detail, data.origin',
    translit_cache: '++id, source_word, doc_lang, [source_word+doc_lang], created_at',
    talk_tombstones: 'id, deleted_at, ref'
});

(window as any).appDb = appDb; // db.ts 등 외부 스크립트에서 참조하기 위해 전역 노출

// --- kv 헬퍼 (반드시 appDb 선언 바로 아래, 모든 호출부보다 위에 위치) ---
// 🌟 [FAIL-SOFT] 스토어가 없거나 업그레이드 중이면 예외 대신 null / no-op 으로 흡수합니다.
//    kvGet 이 throw 하면 initSession 첫 줄에서 전체 초기화가 중단되어
//    updateAuthUI() 조차 실행되지 않고 Sign Out 버튼이 그대로 노출됩니다.
async function kvGet(key: string): Promise<any> {
    try {
        const record = await appDb.table("kv_store").get(key);
        return record ? record.value : null;
    } catch (e) {
        console.warn(`[KV] get('${key}') failed:`, e);
        return null;
    }
}
async function kvSet(key: string, value: any) {
    try {
        await appDb.table("kv_store").put({ key, value });
    } catch (e) {
        console.warn(`[KV] set('${key}') failed:`, e);
    }
}
async function kvRemove(key: string) {
    try {
        await appDb.table("kv_store").delete(key);
    } catch (e) {
        console.warn(`[KV] remove('${key}') failed:`, e);
    }
}

// =====================================================================
// 🌟 [TALK TOMBSTONE] 삭제한 채팅의 묘비
// ---------------------------------------------------------------------
//  삭제 범위 계약:
//    ① 내 PC   : LanceDB messages + Dexie talks + DOM 에서 완전 제거
//    ② 내 PC   : 묘비를 남겨 서버 폴링의 재삽입을 영구 차단
//    ③ 서버    : 행은 그대로 남습니다 (워커에 talk 삭제 엔드포인트 없음)
//    ④ 타 사용자: 각자의 로컬 원장에 그대로 남아 계속 노출됩니다
//                 (syncData 는 가산 전용이라 서버에서 사라져도 지우지 않습니다)
// =====================================================================
let talkTombstoneCache: Set<string> | null = null;

// 🌟 [ITEM TOMBSTONE] 삭제한 문서의 묘비.
//    서버 D1 에 행이 남아 있으면 syncData 폴링(3초)이 재삽입하므로,
//    '이 id 는 내가 의도적으로 지웠다' 는 사실을 로컬에 영구 기록합니다.
//    talk_tombstones 와 동일한 원리이나 대상이 items 테이블입니다.
let itemTombstoneCache: Set<string> | null = null;

async function loadItemTombstones(): Promise<Set<string>> {
    if (itemTombstoneCache) return itemTombstoneCache;
    const s = new Set<string>();
    try {
        // kv_store 에 'item_tombstones' 키 하나로 JSON 배열 저장
        const raw = await kvGet("item_tombstones");
        if (Array.isArray(raw)) {
            for (const id of raw) s.add(String(id));
        }
    } catch (e) {
        console.warn("[ITEM-TOMBSTONE] load failed:", e);
    }
    itemTombstoneCache = s;
    return s;
}

async function addItemTombstone(id: string) {
    if (!id) return;
    const cache = await loadItemTombstones();
    cache.add(String(id));
    try {
        await kvSet("item_tombstones", Array.from(cache));
    } catch (e) {
        console.warn(`[ITEM-TOMBSTONE] save('${id}') failed:`, e);
    }
}

/** 묘비 목록을 메모리에 적재합니다. 최초 1회만 Dexie 를 읽습니다. */
async function loadTalkTombstones(): Promise<Set<string>> {
    if (talkTombstoneCache) return talkTombstoneCache;
    const s = new Set<string>();
    try {
        const rows = await appDb.table("talk_tombstones").toArray();
        for (const r of rows) {
            if (r && r.id) s.add(String(r.id));
        }
        if (s.size > 0) {
            console.log(`[TOMBSTONE] 삭제 묘비 ${s.size}건 적재 완료. 해당 메시지는 서버 폴링으로 부활하지 않습니다.`);
        }
    } catch (e) {
        console.warn("[TOMBSTONE] load failed:", e);
    }
    talkTombstoneCache = s;
    return s;
}

/** 동기 조회용. loadTalkTombstones() 가 선행되어야 정확합니다. */
function isTalkTombstoned(id: string): boolean {
    if (!id) return false;
    return talkTombstoneCache ? talkTombstoneCache.has(String(id)) : false;
}

/** 묘비를 세웁니다. 메모리 캐시와 Dexie 를 동시에 갱신합니다. */
async function addTalkTombstone(id: string, refVal: string = "") {
    if (!id) return;
    const cache = await loadTalkTombstones();
    cache.add(String(id));
    try {
        await appDb.table("talk_tombstones").put({
            id: String(id),
            ref: refVal || "",
            deleted_at: Date.now()
        });
    } catch (e) {
        console.warn(`[TOMBSTONE] put('${id}') failed:`, e);
    }
}

/**
 * 🌟 [DELETE CHAT MESSAGE] 채팅 한 건을 내 기기에서 삭제합니다.
 *
 *  순서가 중요합니다. 묘비를 '가장 먼저' 세워야
 *  삭제 도중에 폴링이 끼어들어도 재삽입되지 않습니다.
 *    ① 묘비 기록          → 재삽입 영구 차단
 *    ② LanceDB messages   → get_chat_messages 조회 결과에서 제거
 *    ③ Dexie talks 캐시   → syncData 의 localMap 잔재 제거
 *    ④ DOM                → 화면 즉시 반영
 *
 *  ⚠️ 서버 호출은 하지 않습니다. index.ts 의 DELETE 는
 *     `type=crons` 가 아니면 S3 인증 해시(/hash/{hash})를 지워
 *     사용자를 강제 로그아웃시키기 때문입니다.
 */
async function deleteChatMessage(msgId: string, opts: { skipConfirm?: boolean } = {}): Promise<boolean> {
    if (!msgId) return false;

    if (!opts.skipConfirm) {
        const confirmed = await ask(
            "이 메시지를 삭제하시겠습니까?\n\n" +
            "· 내 기기에서 완전히 사라지며 다시 나타나지 않습니다.\n" +
            "· 이미 메시지를 받아간 다른 팀원의 화면에는 그대로 남습니다.",
            { title: "Delete Message", kind: "warning" }
        );
        if (!confirmed) return false;
    }

    const el = document.getElementById(msgId) as HTMLElement | null;
    const refVal = el?.dataset.ref || activeContext.ref || "";

    // ① 묘비 (가장 먼저)
    await addTalkTombstone(msgId, refVal);

    // ② LanceDB messages
    //    upsert_items 의 talk 분기가 task_id 에 자기 id 를 각인하므로
    //    delete_message(taskId) 가 정확히 이 한 행만 지웁니다.
    try {
        await invoke("delete_message", { taskId: msgId });
    } catch (e) {
        console.warn(`[CHAT] LanceDB delete_message('${msgId}') failed:`, e);
    }

    // ③ Dexie talks 캐시
    try {
        if (appDb) await appDb.table("talks").delete(msgId);
    } catch (e) { /* 캐시에 없을 수 있으므로 무시 */ }

    // ④ DOM
    if (el) el.remove();

    // 마지막 한 건을 지웠다면 안내 문구를 복원합니다.
    if (chatTalks && chatTalks.querySelectorAll('.chat-talk').length === 0) {
        if (!chatTalks.querySelector('.no-msg')) {
            chatTalks.insertAdjacentHTML(
                'beforeend',
                "<div class='no-msg' data-created-at=\"0\" style='text-align:center; padding:20px; color:#999; font-size:0.8rem;'>No messages yet.</div>"
            );
        }
    }

    console.log(`[CHAT] 🗑️ [DELETED] '${msgId}' 를 내 기기에서 삭제했습니다. (서버 행 및 타 사용자 로컬 원장은 유지)`);
    return true;
}

// 🌟 v4 : 봉투 정규화 규칙을 단 하나만 유지하기 위해 db.ts 에도 같은 함수를 공유합니다.
//  (db.ts 가 자체 enrich 복사본을 갖고 있으면 두 규칙이 어긋나 인덱스가 조용히 깨집니다)
(window as any).normalizeEnvelope = normalizeEnvelope;
(window as any).canonicalizeData = canonicalizeData;

// =====================================================================
// 🌟 [DEXIE PLAN ENGINE v4]
// ---------------------------------------------------------------------
//  Rust(build_dexie_plan)가 내려준 정밀 필터 플랜을 실제 Dexie 쿼리로 실행합니다.
//
//  ── 왜 이 엔진이 필요한가 ──
//   기존에는 convert_conditions_to_sql 이 valid_cols 7개 밖의 조건을 전부 버렸고,
//   그 손실을 메우려고 STAGE-3 이 A/FULL ~ E/TABLE-FALLBACK 5티어를 발행했습니다.
//   v4 는 조건을 하나도 버리지 않고 여기로 넘기므로, 티어 보험이 필요 없어집니다.
//
//  ── 실행 전략 ──
//   ① 인덱스가 선언된 경로  → where().equals()/.between()/.above() (O(log n))
//   ② 인덱스가 없는 경로    → .filter() 풀스캔 (로컬 수천~수만 건 = 수 ms)
//   ③ top / bottom          → 정렬 후 백분위 슬라이스
//   ④ contains/not_contains → .filter() (IndexedDB 는 substring 인덱스가 없음)
// =====================================================================

// 🌟 Dexie 스키마에 실제로 선언한 인덱스 경로 목록입니다.
//  이 집합에 없는 경로는 자동으로 .filter() 로 떨어집니다.
//
//  ⚠️ [계약] 이 집합은 appDb.version(N).stores() 의 items 선언과 '반드시' 일치해야 합니다.
//     여기에만 있고 실제 스키마에 없으면 pickDriverCondition 이 where('없는경로') 를 호출해
//     Dexie 가 SchemaError 를 던집니다. 반대로 스키마에만 있으면 인덱스가 놀 뿐이라 안전합니다.
//     따라서 스키마를 줄일 때는 반드시 이 집합을 먼저 줄이세요.
//
//  🌟 v7 : data.link / data.origin 제거 (contains 전용이라 인덱스 이득 0),
//          data.title / data.name / data.sender_name / data.recipient_name 추가.
const DEXIE_INDEXED_PATHS = new Set<string>([
    // ── 봉투 ──
    'id', 'type', 'flag', 'from', 'to', 'cc', 'bcc', 'ref', 'mode',
    'created_at', 'updated_at',
    // ── commerce 축 ──
    'data.index', 'data.no', 'data.code', 'data.tracking_number',
    'data.goods', 'data.order', 'data.tracking',
    'data.stock_keeping_unit', 'data.barcode',
    'data.status', 'data.amount', 'data.sale_price', 'data.supply_price',
    'data.quantity', 'data.weight', 'data.discount',
    'data.carrier', 'data.shipping_method',
    'data.started_at', 'data.expired_at',
    'data.title', 'data.name', 'data.sender_name', 'data.recipient_name',
    'data.embed', 'data.digest',
    // ── 🌟 trading 축 : v8 stores 선언과 1:1 로 일치해야 합니다 ──
    //    이 21개가 빠져 있으면 선언한 인덱스가 단 한 번도 쓰이지 않고
    //    모든 무역 조건이 .filter() 풀스캔으로 떨어집니다.
    'data.doc_type', 'data.doc_number', 'data.issue_date',
    'data.vessel', 'data.voyage_number', 'data.pol', 'data.pod',
    'data.etd', 'data.eta',
    'data.incoterms', 'data.payment_terms', 'data.currency',
    'data.container_number', 'data.seal_number',
    'data.package_count', 'data.weight_gross', 'data.weight_net', 'data.volume',
    'data.reference_invoice', 'data.reference_lc', 'data.reference_booking',
    // 🌟 [TRADING INDEX RELAY] ITEMS_SCHEMA 의 data.rel_* 와 1:1 로 일치해야 합니다.
    //    ⚠️ 여기에만 있고 실제 스키마에 없으면 pickDriverCondition 이
    //       where('없는경로') 를 호출해 Dexie 가 SchemaError 를 던집니다.
    //       (executeDexiePlan 의 INDEX FALLBACK 이 흡수하지만 매번 예외 비용이 듭니다)
    'data.rel_bl', 'data.rel_hbl', 'data.rel_swb', 'data.rel_awb',
    'data.rel_ci', 'data.rel_cinv', 'data.rel_csi', 'data.rel_pi', 'data.rel_pl',
    'data.rel_po', 'data.rel_sc', 'data.rel_lc', 'data.rel_llc', 'data.rel_co',
    'data.rel_bc', 'data.rel_bk', 'data.rel_sr', 'data.rel_do', 'data.rel_an',
    'data.rel_sa', 'data.rel_fcr', 'data.rel_pod', 'data.rel_cm', 'data.rel_fi',
    'data.rel_wr', 'data.rel_ed', 'data.rel_id', 'data.rel_ccc', 'data.rel_cnm',
    'data.rel_el', 'data.rel_ic', 'data.rel_wc', 'data.rel_ca', 'data.rel_coa',
    'data.rel_pc', 'data.rel_fc', 'data.rel_hc', 'data.rel_cdr',
    'data.rel_ip', 'data.rel_icf', 'data.rel_lg', 'data.rel_tr',
    'data.rel_soa', 'data.rel_dn', 'data.rel_cn', 'data.rel_ti', 'data.rel_cp',
    'data.rel_be', 'data.rel_ins', 'data.rel_dgd',
    // 🌟 [BASE v2 신규 축]
    'data.reference_bl', 'data.reference_po', 'data.reference_contract',
    'data.reference_master_bl', 'data.reference_sr', 'data.reference_number',
    'data.expiry_date', 'data.place_receipt', 'data.place_delivery',
    'data.flight_number', 'data.departure_date', 'data.arrival_date',
    'data.transport_mode', 'data.freight_payment_term',
    'data.amount_subtotal', 'data.amount_tax', 'data.freight_amount',
    'data.due_date', 'data.payment_status',
    'data.package_unit', 'data.type_size', 'data.hs_code'
]);

interface DexieCondition {
    path: string;
    op: string;              // eq | neq | gt | gte | lt | lte | contains | not_contains | top | bottom
    value?: any;
    percent?: number;
    kind?: string;           // number | string | rank
}

interface DexiePlan {
    type?: string;
    /** 🌟 v4 : 확정 도메인 + 교차 후보 도메인. LanceDB 의 IN 절과 동일한 집합입니다. */
    types?: string[];
    mode?: string;
    conditions?: DexieCondition[];
    keywords?: string[];
    alternates?: Record<string, string[]>;
    substantial?: string;
    find?: string;
}

// 🌟 중첩 경로('data.sale_price')를 안전하게 읽습니다.
function readPath(row: any, path: string): any {
    if (!row) return undefined;
    if (path.indexOf('.') === -1) return row[path];
    let cur: any = row;
    for (const seg of path.split('.')) {
        if (cur === null || cur === undefined) return undefined;
        cur = cur[seg];
    }
    return cur;
}

// 🌟 조건 하나를 '메모리상의 행'에 적용합니다.
//  인덱스 경로든 아니든 최종 검증은 전부 이 함수를 통과시켜
//  인덱스 쿼리의 오탐(타입 혼재 등)을 이중으로 막습니다.
function matchCondition(row: any, cond: DexieCondition): boolean {
    // top / bottom 은 개별 행으로 판정 불가. 정렬 단계에서 처리합니다.
    if (cond.op === 'top' || cond.op === 'bottom') return true;

    const raw = readPath(row, cond.path);

    if (cond.kind === 'number') {
        const target = typeof cond.value === 'number' ? cond.value : Number(cond.value);
        if (isNaN(target)) return true; // 비교 불가 → 조건 무시(리콜 우선)

        // 🌟 [MISSING VALUE GUARD] 값이 없는 것과 0 은 완전히 다른 사실입니다.
        //    기존에는 Number('') = 0 으로 떨어져 'sale_price lte 5000' 같은 조건을
        //    가격 필드가 아예 없는 문서까지 전부 통과시켰습니다.
        //    neq 만 예외로 통과시킵니다. (없는 값은 target 과 같지 않은 것이 맞습니다)
        const isMissing = (raw === undefined || raw === null || raw === "");
        if (isMissing) return cond.op === 'neq';

        const actual = typeof raw === 'number'
            ? raw
            : Number(String(raw).replace(/[^\d.\-]/g, ''));
        if (isNaN(actual)) return false;

        switch (cond.op) {
            case 'gt':  return actual >  target;
            case 'gte': return actual >= target;
            case 'lt':  return actual <  target;
            case 'lte': return actual <= target;
            case 'neq': return actual !== target;
            default:    return actual === target;
        }
    }

    // 문자열 계열
    const actualStr = (raw === null || raw === undefined) ? '' : String(raw);
    const targetStr = (cond.value === null || cond.value === undefined) ? '' : String(cond.value);
    if (!targetStr) return true; // 빈 조건은 무시

    const a = actualStr.toLowerCase();
    const t = targetStr.toLowerCase();

    switch (cond.op) {
        case 'contains':     return a.includes(t);
        case 'not_contains': return !a.includes(t);
        case 'neq':          return a !== t;
        case 'gt':           return a >  t;
        case 'gte':          return a >= t;
        case 'lt':           return a <  t;
        case 'lte':          return a <= t;
        default:             return a === t;
    }
}

// 🌟 조건 배열 중 '인덱스 쿼리로 후보를 좁히기에 가장 유리한 것' 하나를 고릅니다.
//  eq 가 범위보다 선택도가 높고, 식별자 경로가 상태/수치보다 선택도가 높습니다.
function pickDriverCondition(conds: DexieCondition[]): DexieCondition | null {
    const HIGH_SELECTIVITY = [
        // ── commerce 식별자 ──
        'data.tracking_number', 'data.no', 'data.code', 'data.index',
        'data.barcode', 'data.stock_keeping_unit', 'data.digest',
        // ── 🌟 trading 식별자 : 카디널리티가 상품명/항구명보다 압도적으로 높습니다 ──
        'data.doc_number', 'data.container_number', 'data.seal_number',
        'data.reference_invoice', 'data.reference_lc', 'data.reference_booking',
        // ── 🌟 trading 인덱스 참조 : crc32 숫자라 카디널리티가 최상입니다 ──
        //    DEXIE_INDEXED_PATHS 의 rel_* 전량과 동일 집합입니다.
        'data.rel_bl', 'data.rel_hbl', 'data.rel_swb', 'data.rel_awb',
        'data.rel_ci', 'data.rel_cinv', 'data.rel_csi', 'data.rel_pi', 'data.rel_pl',
        'data.rel_po', 'data.rel_sc', 'data.rel_lc', 'data.rel_llc', 'data.rel_co',
        'data.rel_bc', 'data.rel_bk', 'data.rel_sr', 'data.rel_do', 'data.rel_an',
        'data.rel_sa', 'data.rel_fcr', 'data.rel_pod', 'data.rel_cm', 'data.rel_fi',
        'data.rel_wr', 'data.rel_ed', 'data.rel_id', 'data.rel_ccc', 'data.rel_cnm',
        'data.rel_el', 'data.rel_ic', 'data.rel_wc', 'data.rel_ca', 'data.rel_coa',
        'data.rel_pc', 'data.rel_fc', 'data.rel_hc', 'data.rel_cdr',
        'data.rel_ip', 'data.rel_icf', 'data.rel_lg', 'data.rel_tr',
        'data.rel_soa', 'data.rel_dn', 'data.rel_cn', 'data.rel_ti', 'data.rel_cp',
        'data.rel_be', 'data.rel_ins', 'data.rel_dgd',
        // 🌟 [BASE v2 참조 축] 문서번호와 동급의 카디널리티를 갖습니다.
        'data.reference_bl', 'data.reference_po', 'data.reference_contract',
        'data.reference_master_bl', 'data.reference_sr'
    ];

    let best: DexieCondition | null = null;
    let bestScore = -1;

    for (const c of conds) {
        if (!DEXIE_INDEXED_PATHS.has(c.path)) continue;
        if (c.op === 'top' || c.op === 'bottom') continue;
        if (c.op === 'contains' || c.op === 'not_contains' || c.op === 'neq') continue;

        let score = 0;
        if (c.op === 'eq') score += 10;
        else score += 4; // 범위 연산자
        if (HIGH_SELECTIVITY.includes(c.path)) score += 20;

        if (score > bestScore) { bestScore = score; best = c; }
    }
    return best;
}

// 🌟 [MAIN] 플랜을 실행해 최종 문서 배열을 돌려줍니다.
//  candidateIds 가 있으면 LanceDB 리콜 후보 안에서만 필터링하고,
//  없으면 Dexie 전체를 대상으로 합니다(목록 조회 경로).
async function executeDexiePlan(
    plan: DexiePlan,
    opts: { candidateIds?: string[]; limit?: number; offset?: number } = {}
): Promise<any[]> {
    if (!appDb) return [];

    const conds: DexieCondition[] = Array.isArray(plan.conditions) ? plan.conditions : [];
    const limit = opts.limit ?? 200;
    const offset = opts.offset ?? 0;

    let rows: any[] = [];

    // ── 후보 집합이 주어진 경우 : LanceDB 리콜 결과 안에서만 정밀 필터 ──
    if (opts.candidateIds && opts.candidateIds.length > 0) {
        rows = await appDb.table('items').where('id').anyOf(opts.candidateIds).toArray();
        console.log(`[DEXIE-PLAN] 후보 ${opts.candidateIds.length}건 → Dexie 적재 ${rows.length}건`);
    } else {
        // ── 후보가 없는 경우 : 인덱스로 최대한 좁혀서 적재 ──
        const driver = pickDriverCondition(conds);

        if (driver) {
            // 🌟 [INDEX FALLBACK] DEXIE_INDEXED_PATHS 와 실제 스키마가 어긋난 세대에서는
            //    where('없는경로') 가 SchemaError 를 던집니다. 그 경우 조용히 전량 적재로 폴백해
            //    '검색이 통째로 실패' 하는 대신 '조금 느린 검색' 으로 흡수합니다.
            try {
                const coll = appDb.table('items').where(driver.path);
                if (driver.op === 'eq') {
                    rows = await coll.equals(driver.value).toArray();
                } else if (driver.op === 'gt') {
                    rows = await coll.above(driver.value).toArray();
                } else if (driver.op === 'gte') {
                    rows = await coll.aboveOrEqual(driver.value).toArray();
                } else if (driver.op === 'lt') {
                    rows = await coll.below(driver.value).toArray();
                } else if (driver.op === 'lte') {
                    rows = await coll.belowOrEqual(driver.value).toArray();
                } else {
                    rows = await appDb.table('items').toArray();
                }
                console.log(`[DEXIE-PLAN] 드라이버 인덱스 '${driver.path} ${driver.op} ${driver.value}' → ${rows.length}건 적재`);
            } catch (e) {
                console.warn(`[DEXIE-PLAN] ⚠️ 드라이버 인덱스 '${driver.path}' 조회 실패. 전량 적재로 폴백합니다.`, e);
                rows = await appDb.table('items').toArray();
            }
        } else if (plan.types && plan.types.length > 0) {
            // 🌟 types 인덱스(anyOf)로 좁힙니다. mode 보다 선택도가 높습니다.
            rows = await appDb.table('items').where('type').anyOf(plan.types).toArray();
            console.log(`[DEXIE-PLAN] 드라이버 없음. type anyOf [${plan.types.join(', ')}] 로 ${rows.length}건 적재`);
        } else if (plan.mode) {
            rows = await appDb.table('items').where('mode').equals(plan.mode).toArray();
            console.log(`[DEXIE-PLAN] 드라이버 없음. mode='${plan.mode}' 로 ${rows.length}건 적재`);
        } else {
            rows = await appDb.table('items').toArray();
            console.log(`[DEXIE-PLAN] 전체 적재 ${rows.length}건`);
        }
    }

    // ── 스코프 검증 (types / mode) ──
    //  🌟 v4 : plan.type 하나만 보면 교차 후보 도메인 결과가 전부 잘립니다.
    //     LanceDB 가 IN 절로 넓힌 만큼 Dexie 도 같은 집합을 통과시켜야 합니다.
    const allowedTypes: string[] = (plan.types && plan.types.length > 0)
        ? plan.types
        : (plan.type ? [plan.type] : []);

    if (allowedTypes.length > 0) {
        const before = rows.length;
        rows = rows.filter(r => allowedTypes.includes(r.type || ''));
        if (before !== rows.length) {
            console.log(`[DEXIE-PLAN] types 필터 [${allowedTypes.join(', ')}]: ${before} → ${rows.length}건`);
        }
    }
    if (plan.mode) {
        rows = rows.filter(r => (r.mode || 'commerce') === plan.mode);
    }

    // ── 정밀 조건 전량 적용 ──
    const rankConds = conds.filter(c => c.op === 'top' || c.op === 'bottom');
    const plainConds = conds.filter(c => c.op !== 'top' && c.op !== 'bottom');

    if (plainConds.length > 0) {
        const before = rows.length;
        rows = rows.filter(r => plainConds.every(c => matchCondition(r, c)));
        console.log(`[DEXIE-PLAN] 정밀 조건 ${plainConds.length}개 적용: ${before} → ${rows.length}건`);
        for (const c of plainConds) {
            const viaIndex = DEXIE_INDEXED_PATHS.has(c.path) ? 'index' : 'scan';
            console.log(`  ↳ ${c.path} ${c.op} ${JSON.stringify(c.value)} (${c.kind}, ${viaIndex})`);
        }
    }

    // ── top / bottom 백분위 : 정렬 후 슬라이스 ──
    for (const rc of rankConds) {
        if (rows.length === 0) break;

        // 🌟 [RANK MISSING GUARD] 값이 없는 문서는 순위 자체가 성립하지 않습니다.
        //    Number(undefined) || 0 으로 떨어지면 bottom 20% 가
        //    '해당 축을 아예 갖지 않은 문서' 로 채워집니다.
        //    (수치 시딩을 제거한 뒤에는 결손이 실제로 발생하므로 반드시 필요합니다)
        const ranked = rows.filter(r => {
            const v = readPath(r, rc.path);
            if (v === undefined || v === null || v === "") return false;
            return !isNaN(Number(v));
        });
        const skipped = rows.length - ranked.length;
        if (ranked.length === 0) {
            console.log(`[DEXIE-PLAN] ${rc.op} ${rc.path}: 값을 가진 문서가 0건이라 랭킹을 건너뜁니다.`);
            continue;
        }

        const pct = Math.max(1, Math.min(100, rc.percent ?? 20));
        const take = Math.max(1, Math.ceil(ranked.length * (pct / 100)));

        const sorted = [...ranked].sort((a, b) => {
            const av = Number(readPath(a, rc.path));
            const bv = Number(readPath(b, rc.path));
            return rc.op === 'top' ? bv - av : av - bv;
        });
        rows = sorted.slice(0, take);
        console.log(`[DEXIE-PLAN] ${rc.op} ${pct}% on ${rc.path} → ${rows.length}건 (값 결손 ${skipped}건 제외)`);
    }

    // ── keywords : 조건이 되지 못한 청크로 보조 스코어링 ──
    //  버리지 않고 '가산점' 으로만 씁니다. 여기서 잘라내면 리콜이 무너집니다.
    if (plan.keywords && plan.keywords.length > 0) {
        for (const r of rows) {
            const hay = `${r.data?.text ?? ''} ${r.data?.title ?? ''} ${r.data?.masked_text ?? ''}`.toLowerCase();
            let hit = 0;
            for (const k of plan.keywords) {
                if (k && hay.includes(k.toLowerCase())) hit++;
            }
            r.__kw_score = hit;
        }
        rows.sort((a, b) => (b.__kw_score || 0) - (a.__kw_score || 0));
    }

    const start = Math.min(offset, rows.length);
    const end = Math.min(start + limit, rows.length);
    return rows.slice(start, end);
}

// [통합 락 매니저 & 프론트엔드 큐 관리자]
class GlobalTaskManager {
    static isBusy: boolean = false;
    static currentTaskId: string | null = null;
    static currentTaskPayload: any = null; 
    static activeRefs: Set<string> = new Set();
    static queue: Array<{taskId: string, type: string, payload: any}> = [];
    static backendQueued: any[] = []; // 🌟 [CRITICAL FIX] 백엔드가 이미 관리 중인 대기열 추적용 배열 추가
    static cancelledTasks: Set<string> = new Set(); // 🌟 [CRITICAL FIX] 취소된 작업 ID 블랙리스트 추가

    // 🌟 [추가] 큐를 Dexie(IndexedDB)에 저장하여 새로고침 시에도 증발 방지
    static async saveQueue() {
        await appDb.table("ts_queue").clear();
        if (this.queue.length > 0) {
            await appDb.table("ts_queue").bulkAdd(this.queue);
        }
    }

    // 🌟 [수정] 앱 시작 시 Dexie에서 저장된 큐 복원
    static async loadQueue() {
        // 🌟 [CRITICAL FIX] 앱을 완전히 종료 후 재시작했을 때 대기열 자동 실행 방지
        // sessionStorage는 F5 새로고침 시에는 유지되지만, 앱 종료 시에는 초기화됩니다.
        if (!sessionStorage.getItem("app_running_session")) {
            sessionStorage.setItem("app_running_session", "true");
            
            // 🌟 [추가] 강제 종료 전 Dexie에 남아있던 대기열을 가져와 LanceDB에 에러(Error) 히스토리로 남깁니다.
            try {
                const leftoverTasks = await appDb.table("ts_queue").toArray();
                if (leftoverTasks && leftoverTasks.length > 0) {
                    const errorItems = leftoverTasks.map((task: any) => {
                        const now = Date.now();
                        let taskRef = "Queued Task";
                        if (task.payload) {
                            taskRef = task.payload.query || task.payload.link || task.payload.image_path || "Queued Task";
                        }
                        
                        const textMsg = `[Cancelled] ${taskRef} (App closed unexpectedly)`;

                        return {
                            id: task.taskId,
                            type: "talk",
                            role: "system_task",
                            from: "system",
                            to: "user",
                            cc: task.payload?.cc || "",
                            bcc: task.payload?.bcc || "",
                            ref: task.payload?.refId || task.payload?.ref || "",
                            status: 6, // 6: Error 상태 코드로 UI에 붉게 표기됨
                            created_at: now,
                            updated_at: now,
                            data: {
                                text: textMsg,
                                link: "",
                                origin: "https://commerce.logis.center"
                            }
                        };
                    });
                    
                    // 백엔드 LanceDB에 에러 히스토리 일괄 삽입
                    await invoke("upsert_items", { items: errorItems });
                    console.log(`[QUEUE] Recorded ${errorItems.length} leftover tasks as ERROR in LanceDB.`);
                }
            } catch (e) {
                console.error("[QUEUE] Failed to log leftover tasks to LanceDB:", e);
            }

            // 🌟 [FAIL-SOFT] 스토어 부재/업그레이드 중이어도 초기화 흐름을 끊지 않습니다.
            try {
                await appDb.table("ts_queue").clear();
                console.log("[QUEUE] App restarted. Cleared persistent Dexie queue to mark as STOPPED.");
            } catch (e) {
                console.warn("[QUEUE] ts_queue clear failed (table may be missing):", e);
            }
            this.queue = [];
            return;
        }

        try {
            const q = await appDb.table("ts_queue").toArray();
            if (q && q.length > 0) {
                this.queue = q;
                this.queue.forEach((task: any) => this.activeRefs.add(task.taskId));
                console.log(`[QUEUE] Restored ${this.queue.length} pending tasks from Dexie.`);
            } else {
                this.queue = [];
            }
        } catch(e) {
            console.error("[QUEUE] Failed to load queue from Dexie", e);
            this.queue = [];
        }
    }

    static async addToQueue(taskId: string, type: string, payload: any) {
        if (this.activeRefs.has(taskId)) return;
        this.queue.push({ taskId, type, payload });
        this.activeRefs.add(taskId);
        await this.saveQueue(); // 🌟 즉시 저장 (Dexie)
        
        // 🌟 [추가] 큐에 담기자마자 사용자에게 시각적 피드백 제공 (DB 등록 전 선행 렌더링)
        const startTime = parseInt(taskId.split('_')[1]) || Date.now();
        
        // 1. 사용자 질문 선행 렌더링 (검색인 경우)
        if (payload.query) {
            await renderMessage({
                id: `${taskId}_query`,
                role: "user",
                text: payload.query,
                status: 9,
                created_at: startTime - 100,
                updated_at: startTime - 100
            });
        }

        // 2. 시스템 대기열 말풍선 선행 렌더링
        await renderMessage({
            id: taskId,
            task_id: taskId,
            role: "system_task",
            text: payload.link || payload.image_path || "Waiting in queue...",
            status: 10, // Pending
            created_at: startTime,
            updated_at: startTime
        });

        console.log(`[QUEUE] Task ${taskId} (${type}) added. Current queue length: ${this.queue.length}`);
        await this.processNext();
    }

    // 다음 작업 실행 판단 로직
    static async processNext() {
        if (this.isBusy || this.queue.length === 0) return;

        this.isBusy = true;
        const task = this.queue.shift()!;
        await this.saveQueue(); // 🌟 큐에서 항목이 나갔으므로 갱신 (Dexie)
        
        this.currentTaskId = task.taskId;
        this.currentTaskPayload = task.payload; // 🌟 추가: 실행중인 페이로드 동시 기록
        await kvSet("sys_lock", task.taskId);

        console.log(`[QUEUE] Starting Task: ${task.taskId}`);
        
        // 🌟 [CRITICAL FIX] await로 인한 프론트엔드 프리징 및 큐 막힘 현상 원천 차단 (Fire-and-Forget)
        if (task.type === "ai_search") {
            invoke("ai_search_complex", task.payload).catch(async e => {
                console.error(`[QUEUE] Task execution failed:`, e);
                await this.release(task.taskId, task.taskId);
            });
        } else {
            emit("new-task-from-browser", task.payload).catch(async e => {
                console.error(`[QUEUE] Task execution failed:`, e);
                await this.release(task.taskId, task.taskId);
            });
        }
    }

    static async release(taskId: string, refOrQuery: string) {
        if (this.currentTaskId === taskId) {
            this.isBusy = false;
            this.currentTaskId = null;
            this.currentTaskPayload = null; 
        }
        this.activeRefs.delete(taskId);
        this.backendQueued = this.backendQueued.filter(p => p.id !== taskId && p.taskId !== taskId); // 🌟 종료된 작업은 가림막에서 제거
        
        if (await kvGet("sys_lock") === taskId) {
            await kvRemove("sys_lock");
        }
        await this.saveQueue(); // 🌟 참조 목록(activeRefs)이 변했으므로 갱신 (Dexie)
        await this.processNext();
    }

    static async forceReset() {
        this.isBusy = false;
        this.currentTaskId = null;
        this.currentTaskPayload = null;
        this.activeRefs.clear();
        this.queue = [];
        this.backendQueued = []; // 🌟 전체 초기화 반영
        // 🌟 Dexie DB 초기화 (세션/설정 키는 보존)
        try {
            await appDb.table("ts_queue").clear();
            // 🌟 [SESSION PRESERVE] kv_store 를 통째로 clear() 하면
            //    chat_session(로그인 세션), search_mode, hidden_pages,
            //    my_sync_seed, oauth_registered_sites 등 사용자 상태가
            //    전부 소멸하여 앱 재시작 시 로그인이 풀립니다.
            //    '작업 큐/락/터미널 로그' 관련 키만 선택적으로 삭제합니다.
            //
            //    ⚠️ btn-reset-db 경로는 이 함수 호출 '직후' 에
            //       appDb.delete() 로 Dexie DB 자체를 물리 삭제하므로
            //       여기서 보존해도 완전 초기화에는 영향이 없습니다.
            const PRESERVE_KEYS = new Set([
                "chat_session",
                "search_mode",
                "hidden_pages",
                "my_sync_seed",
                "oauth_registered_sites",
                "oauth_client_address",
                "item_tombstones",
                "schema_v4_notified",
                "force_cpu_mode"
            ]);
            const allKeys = await appDb.table("kv_store").toCollection().primaryKeys();
            for (const key of allKeys) {
                if (typeof key === "string" && !PRESERVE_KEYS.has(key)) {
                    await appDb.table("kv_store").delete(key);
                }
            }
            console.log("[QUEUE] Dexie DB tables cleared (session keys preserved).");
        } catch (e) {
            console.error("[QUEUE] Dexie DB clear error:", e);
        }
        // 🌟 LanceDB 전면 초기화 호출 (새로고침 전에 백엔드 초기화가 완료되도록 대기)
        try {
            await invoke("reset_lancedb");
            console.log("[QUEUE] LanceDB fully reset.");
        } catch (e) {
            console.error("[QUEUE] LanceDB reset error:", e);
        }
    }
}

// [TAG SYSTEM] Hashtag-style search state
interface SearchTag {
    id: string;
    label: string;
    type: 'domain' | 'type' | 'mode' | 'path';
    value: string;
}
let activeTags: SearchTag[] = [];

// List State
let cachedDocs: any[] = [];
let currentPage = 0;
const pageSize = 10;
let isLoading = false;
let hasMore = true;

// Chat Pagination State
let chatPage = 0;
let chatHasMore = true;
let isChatLoading = false;

// [NEW] Track first-load status for UI loaders
let isFirstNavRender = true;
let isFirstChatLoad = true;

// [NEW] Window Focus State (백그라운드 리소스 최적화용)
let isFocus = true;

// 🌟 [CRITICAL FIX] 새로고침 시 스텝 순서 꼬임 방지용 대기열
let isFetchingLogs = false;
let pendingLiveEvents: any[] = [];
const livePayloads = new Map<string, any>(); // 🌟 [CRITICAL FIX] 퍼센트(%) 지연 노출을 막기 위한 프론트엔드 초고속 캐시 메모리

// ==========================================
// [PARITY] Cloud front.js Core Utilities
// ==========================================
function isDiff(obj1: any, obj2: any): boolean {
    if (!obj1 && !obj2) return false;
    if (!obj1 || !obj2) return true;
    const keys1 = Object.keys(obj1);
    const keys2 = Object.keys(obj2);
    if (keys1.length !== keys2.length) return true;
    
    for (const key of keys1) {
        if (typeof obj1[key] === 'object' && typeof obj2[key] === 'object') {
            if (isDiff(obj1[key], obj2[key])) return true;
        } else if (obj1[key] !== obj2[key]) {
            return true;
        }
    }
    return false;
}

function safeClone(obj: any) {
    const seen = new WeakMap();
    function clone(value: any) {
        if (typeof value !== "object" || value === null) return value;
        if (seen.has(value)) return null; 
        const copy: any = Array.isArray(value) ? [] : {};
        seen.set(value, copy);
        for (const key in value) {
            copy[key] = clone(value[key]);
        }
        return copy;
    }
    return clone(obj);
}

function mergeNode(obj1: any, obj2: any) {
    const isEmpty = (value: any) => value === null || value === undefined || value === '' || value === 0;
    const merged = { ...obj1 };
    for (const key in obj2) {
        if (obj2.hasOwnProperty(key)) {
            const value2 = obj2[key];
            if (!isEmpty(value2)) {
                merged[key] = value2;
            }
        }
    }
    return merged;
}

const taskSteps = new Map<string, Map<string, number>>();
const taskTotalSteps = new Map<string, number>(); // 🌟 [CRITICAL FIX] 작업별 총 스텝 수를 기억하는 장부 추가

let selectedUuids = new Set<string>();
let currentDetailUuid: string | null = null;
let activeTaskId: string | null = null; 
// [DEPRECATED] 흩어져 있던 개별 락 변수들은 GlobalTaskManager로 대체되었습니다.
let spinnerInterval: number | null = null;
let qrSpinnerIndex = 0; 
let systemLogCount = 0;

function stepQrSpinner() {
    const el = document.getElementById("qr-auth-spinner");
    if (el) {
        qrSpinnerIndex = (qrSpinnerIndex + 1) % spinnerFrames.length;
        el.innerText = spinnerFrames[qrSpinnerIndex];
    }
}
// [NEW] Active navigation context for related logs/chat
let activeContext = {
    cc: "",
    bcc: "",
    ref: ""
};

// --- UI Elements ---
const contentPanel = document.getElementById("content-panel") as HTMLElement;
const searchInput = document.getElementById("global-search") as HTMLInputElement;
const btnSubmit = document.getElementById("btn-submit") as HTMLButtonElement; 
const btnExtract = document.getElementById("btn-extract") as HTMLButtonElement; 
const btnAutoLaunch = document.getElementById("btn-auto-launch") as HTMLButtonElement;
const settingsBtn = document.getElementById("btn-settings") as HTMLButtonElement;
const tabContents = document.querySelectorAll<HTMLElement>(".tab-content");

const navPreviewContainer = document.getElementById("nav-preview-container") as HTMLElement;
const navImgThumbnail = document.getElementById("nav-img-thumbnail") as HTMLImageElement;
const navImgClear = document.getElementById("nav-img-clear") as HTMLButtonElement;
const navUploadBtn = document.getElementById("nav-upload-btn");

const listView = document.getElementById("list-view") as HTMLElement;
const detailView = document.getElementById("detail-view") as HTMLElement;
const detailTitle = document.getElementById("detail-title") as HTMLElement;
const detailContent = document.getElementById("detail-content") as HTMLElement;
const btnDetailBack = document.getElementById("btn-detail-back") as HTMLButtonElement;
const btnListBack = document.getElementById("btn-list-back") as HTMLButtonElement;
const btnDetailDelete = document.getElementById("btn-detail-delete") as HTMLButtonElement;
const btnStopTask = document.getElementById("btn-stop-task") as HTMLButtonElement; 

// [CHANGED] Replaced table body with generic list container
const docListContainer = document.getElementById("doc-list") as HTMLElement;

const listRefreshBtn = document.getElementById("list-refresh-btn") as HTMLButtonElement;
const btnDeleteSelected = document.getElementById("btn-delete-selected") as HTMLButtonElement;
const btnSyncQr = document.getElementById("btn-sync-qr") as HTMLButtonElement;
const listScrollContainer = document.getElementById("list-scroll-container") as HTMLElement;
const headerLoading = document.getElementById("header-loading") as HTMLElement;

// 🌟 기존 loadingIndicator 대신 h2 태그를 선택합니다.
const listTitle = document.querySelector("#list-view .header-row h2") as HTMLElement;

const aiResultsArea = document.getElementById("ai-search-results") as HTMLElement;
const aiResultsTitle = document.getElementById("ai-results-title") as HTMLElement;
const aiResultsContent = document.getElementById("ai-results-content") as HTMLElement;

const chatTalks = document.querySelector('.chat-talks') as HTMLElement;
const chatForm = document.querySelector('form[name="chat-form"]') as HTMLFormElement;

// 🌟 채팅폼(submit) 이벤트 (chrome.js 방식 적용)
if (chatForm) {
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault(); // 폼 기본 동작인 새로고침 방지

        const input = chatForm.querySelector('input[name="talk"]') as HTMLInputElement;
        if (!input) return;

        const query = input.value.trim();
        if (!query) return;

        input.value = ""; // 하단 채팅창 비우기

        const now = Date.now();

        let effectiveCc = activeContext.cc;
        let effectiveBcc = activeContext.bcc;
        let effectiveRef = activeContext.ref;

        const isDefaultForced = activeTags.some(t => t.value === "logis.center" && t.type === "domain");

        if (!effectiveCc || (!isDefaultForced && activeTags.length === 0)) {
            let targetUrlStr = currentDetectedUrl || "https://commerce.logis.center/tracking";
            if (targetUrlStr.includes("localhost") || targetUrlStr.includes("127.0.0.1") || targetUrlStr === "about:blank") {
                targetUrlStr = "https://commerce.logis.center/tracking";
            }
            try {
                const urlObj = new URL(targetUrlStr.toLowerCase());
                const rootDomain = getRootDomain(urlObj.hostname);
                effectiveCc = await hashId(rootDomain);
                const link = (urlObj.pathname + urlObj.search).toLowerCase();
                effectiveRef = await hashId((currentSession.team || "") + effectiveCc + link);
            } catch (err) {}
        }

        // 1. WebRTC (모바일 기기 등) P2P 연결이 되어있다면 상대방 기기로 전송
        if (dataChannel && dataChannel.readyState === "open") {
            dataChannel.send(JSON.stringify({ 
                type: "chat_message", 
                content: query 
            }));
            console.log("[CHAT] Message sent via WebRTC");
        }

        // 🌟 [ANALYTIC LOCAL] analytic 모드에서도 로컬 LanceDB 검색(ai_search_complex)을 사용합니다.
        //    서버 Vectorize POST 를 제거하고, commerce/shipping 과 동일한 검색 큐로 합류시킵니다.
        //    parse_analytic_query 가 질의를 파싱하고, 로컬 item_chunks + Dexie 가 결과를 반환합니다.
        if (currentSearchMode === "analytic") {
            const taskId = `search_${Date.now()}`;
            const startTime = Date.now();

            // 사용자 질문 말풍선 즉시 렌더링
            await renderMessage({
                id: `${taskId}_query`,
                role: "user",
                text: query,
                status: 9,
                created_at: startTime,
                updated_at: startTime
            });

            try {
                const devicePref = getDevicePref();

                // 로컬 검색 큐에 등록 (ai_search_complex 가 mode="analytic" 으로 동작)
                await GlobalTaskManager.addToQueue(taskId, "ai_search", {
                    taskId: taskId,
                    query: query,
                    language: "korean",
                    devicePreference: devicePref,
                    searchMode: "analytic",
                    cc: activeContext.cc || "",
                    bcc: activeContext.bcc || "",
                    refId: activeContext.ref || ""
                });
            } catch (e) {
                console.error("[ANALYTIC-LOCAL] Search queue failed:", e);
            }

            setTimeout(() => {
                const scrollEl = document.getElementById("chat-scroll");
                const container = document.querySelector(".chat-container") as HTMLElement;

                if (scrollEl && container) {
                    const maxScroll = Math.max(0, scrollEl.scrollHeight - container.clientHeight);
                    currentY = maxScroll;
                    scrollEl.style.transition = "transform 0.3s ease-out";
                    updateTransform();
                    setTimeout(() => { scrollEl.style.transition = ""; }, 300);
                }
            }, 100);

            return;
        }

        // 🌟 [OPTIMISTIC LOCAL WRITE]
        //  ── 무엇이 문제였나 ──
        //   기존 구조는 `if (response.results.length > 0)` 안에서만 로컬에 저장했습니다.
        //   서버(index.ts)의 PUT 핸들러는 `if(cookies.sender)` 게이트에 막혀
        //   talks INSERT 를 한 번도 수행하지 못했고, 빈 results 를 돌려주었습니다.
        //   그래서 LanceDB talks 에 아무것도 안 들어가고
        //   get_chat_messages 가 0건 → .chat-talks 에 "No messages yet." 이 남았습니다.
        //
        //  ── 해결 ──
        //   이전 구현(content.js / chrome.js)과 동일하게, 사용자가 입력한 즉시
        //   로컬 messages 테이블에 먼저 적재하고 화면을 그립니다.
        //   서버 응답이 오면 그 행이 별도 id 로 추가되며(서버가 hashId() 로 새 id 발급),
        //   upsertChatMessages 의 중복 제거 + 시간순 정렬이 자연스럽게 합칩니다.
        //   서버가 실패해도 채팅 목록은 절대 사라지지 않습니다.
        const localTalkId = `talk_${now}_${Math.random().toString(36).slice(2, 8)}`;
        {
            let localLink = "/tracking";
            let localOrigin = "https://commerce.logis.center";
            try {
                let hrefForLink = currentDetectedUrl || "https://commerce.logis.center/tracking";
                if (hrefForLink.includes("localhost") || hrefForLink.includes("127.0.0.1") || hrefForLink === "about:blank") {
                    hrefForLink = "https://commerce.logis.center/tracking";
                }
                const u = new URL(hrefForLink.toLowerCase());
                localLink = (u.pathname + u.search).toLowerCase();
                localOrigin = u.origin;
            } catch (e) {}

            try {
                await invoke("upsert_items", {
                    items: [{
                        id: localTalkId,
                        table: "talks",
                        type: "talk",
                        from: currentSession.address || "",
                        to: currentSession.team || "",
                        cc: effectiveCc || "",
                        bcc: effectiveBcc || "",
                        ref: effectiveRef || "",
                        status: 9,
                        created_at: now,
                        updated_at: now,
                        data: {
                            text: query,
                            link: localLink,
                            origin: localOrigin
                        }
                    }]
                });
                console.log(`[CHAT] Optimistically stored local talk '${localTalkId}' (ref: ${effectiveRef})`);
            } catch (e) {
                console.warn("[CHAT] Local optimistic write failed:", e);
            }

            // 로컬 저장 직후 즉시 말풍선 렌더링 (서버 왕복을 기다리지 않습니다)
            await renderMessage({
                id: localTalkId,
                role: "user",
                text: query,
                status: 9,
                created_at: now,
                updated_at: now
            });
        }

        // 2. 클라우드플레어 Workers (서버)로 PUT 요청 전송 및 정식 응답 처리 (chrome.js 방식)
        try {
            const origin = "https://commerce.logis.center";
            const tzOffset = new Date().getTimezoneOffset() * 60 * 1000;
            const createdAt = now - tzOffset;
            
            let targetHref = currentDetectedUrl || "https://commerce.logis.center/tracking";
            if (targetHref.includes("localhost") || targetHref.includes("127.0.0.1") || targetHref === "about:blank") {
                targetHref = "https://commerce.logis.center/tracking";
            }

            // 🌟 [SENDER GATE] index.ts 의 PUT 핸들러는 아래 게이트를 통과해야만
            //    INSERT INTO talks 를 실행합니다.
            //      if(cookies.sender){ if(isAddress(from) && isAddress(to)){ ... } }
            //    cookies.sender 는 요청 최상단 세션 블록에서
            //    req.query.sender → data.sender → cookies.sender 순으로 세팅되며,
            //    이 블록은 method 와 무관하게 매 요청 실행됩니다.
            //    따라서 이 PUT 요청 자체에 sender 를 실으면 그 자리에서 게이트를 통과합니다.
            //
            // 🌟 [TO FIX] 기존에는 to 로 effectiveRef(페이지 ref 해시)를 보냈습니다.
            //    서버는 talk.to 에 그 값을 그대로 저장하는데,
            //    GET 조회 쿼리 중 하나가
            //      SELECT * FROM talks WHERE "from" = team AND "to" = address
            //    이므로 ref 를 넣으면 대화 상대 축이 어긋납니다.
            //    ref 는 서버가 자체적으로 hashId(team+cc+link) 로 재계산하므로
            //    to 에는 소속 팀 주소를 보내는 것이 계약상 올바릅니다.
            const talkSender = currentSession.email || currentSession.name || "";

            const params = new URLSearchParams({
                origin: origin,
                created_at: createdAt.toString(),
                hash: currentSession.hash,
                token: currentSession.token || "",
                href: targetHref,
                type: "talk",
                sender: talkSender,
                from: currentSession.address || "",
                to: currentSession.team || currentSession.address || "",
                text: encodeURIComponent(query)
            });
            
            const url = `${API_HOST}/?${params.toString()}`;
            
            const response = await invoke<any>("proxy_fetch", {
                url: url,
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                session_params: { hash: currentSession.hash, token: currentSession.token }
            });

            // 서버 응답 결과(결과 배열)가 온 경우 chrome.js처럼 로컬 DB에 동기화
            if (response && response.results && response.results.length > 0) {
                await invoke("upsert_items", { items: response.results });
                for (const item of response.results) {
                    if (item.table === "talks" || item.type === "talk") {
                        await appDb.table("talks").put(item);
                    }
                }
                console.log(`[CHAT] Server accepted talk. rows=${response.results.length}`);
            } else {
                // 🌟 서버가 빈 배열을 돌려주면 cookies.sender 게이트에서 탈락한 것입니다.
                //    낙관적 로컬 저장 덕분에 화면은 유지되지만 원인을 반드시 표면화합니다.
                console.warn(
                    "[CHAT] ⚠️ Server returned no talk rows. " +
                    "Check that `sender` reached the worker (cookies.sender gate) — " +
                    `sent sender='${talkSender}', from='${currentSession.address}', to='${currentSession.team}'`
                );
            }

            // 3. 서버 동기화 후 최신 메시지 렌더링
            await fetchChatHistory(false, true);

            console.log("[CHAT] Message sent to Cloudflare worker and synced");
        } catch (err) {
            console.error("[CHAT] Failed to send to Cloudflare worker:", err);
        }

        // 4. 스크롤을 맨 아래로 부드럽게 이동
        setTimeout(() => {
            const scrollEl = document.getElementById("chat-scroll");
            const container = document.querySelector(".chat-container") as HTMLElement;
            if (scrollEl && container) {
                const maxScroll = Math.max(0, scrollEl.scrollHeight - container.clientHeight);
                currentY = maxScroll;
                scrollEl.style.transition = "transform 0.3s ease-out";
                updateTransform();
                setTimeout(() => { scrollEl.style.transition = ""; }, 300);
            }
        }, 100);
    });
}

// 🌟 [DELETE DELEGATION] 삭제 버튼은 말풍선이 재렌더링될 때마다 새로 만들어지므로
//  개별 노드에 리스너를 붙이면 upsertChatMessages 의 replaceChild 시점에 유실됩니다.
//  컨테이너에 위임 리스너 하나만 두면 이후 어떤 렌더링 경로에서도 그대로 동작합니다.
if (chatTalks) {
    chatTalks.addEventListener("click", async (e) => {
        const target = e.target as HTMLElement;
        const btn = target.closest('.btn-delete-talk') as HTMLElement | null;
        if (!btn) return;

        // 🌟 태스크 말풍선의 handleTaskClick 이 함께 발화하지 않도록 반드시 차단합니다.
        e.preventDefault();
        e.stopPropagation();

        const talkId = btn.dataset.talkId;
        if (!talkId) return;

        btn.style.pointerEvents = "none";
        btn.style.opacity = "0.15";

        const ok = await deleteChatMessage(talkId);

        if (!ok) {
            // 사용자가 취소했으면 버튼을 원상복구합니다.
            btn.style.pointerEvents = "";
            btn.style.opacity = "0.35";
        }
    });
}

// --- Settings Toggle Logic ---
const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
const settingsPanel = document.getElementById("settings-panel") as HTMLElement;
const docList = document.getElementById("doc-list") as HTMLElement;
// 🌟 nav-section은 여러 개이므로 querySelectorAll로 잡습니다.
const navSections = document.querySelectorAll(".nav-section"); 

settingsToggle?.addEventListener("change", (e) => {
    const isChecked = (e.target as HTMLInputElement).checked;
    const label = document.querySelector('label[for="settings-toggle"]') as HTMLElement;
    const listRefreshBtn = document.getElementById("list-refresh-btn"); // 🌟 버튼 참조 추가
    
    if (isChecked) {
        // 설정 켜짐: 설정 패널 표시, 리스트 및 네비게이션 숨김
        if (settingsPanel) settingsPanel.style.display = "block";
        if (docList) docList.style.display = "none";
        if (listRefreshBtn) listRefreshBtn.style.display = "none"; // 🌟 새로고침 버튼 숨김
        navSections.forEach(el => (el as HTMLElement).style.display = "none");
        
        if (label) {
            label.classList.add("on")
        }
    } else {
        // 설정 꺼짐: 설정 패널 숨김, 리스트 및 네비게이션 원상복구
        if (settingsPanel) settingsPanel.style.display = "none";
        if (docList) docList.style.display = ""; 
        if (listRefreshBtn) listRefreshBtn.style.display = "flex"; // 🌟 새로고침 버튼 다시 표시
        navSections.forEach(el => (el as HTMLElement).style.display = "");
        
        if (label) {
            label.classList.remove("on");
        }

        applySearchModeUI(); 
    }
});

// --- Spinner Logic ---
const spinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

function startSpinner() {
    if (spinnerInterval) clearInterval(spinnerInterval);
    
    if (settingsBtn) {
        settingsBtn.classList.add("active-spinner-mode");
        // 🌟 [CRITICAL FIX] 글로벌 스피너가 돌 때 번개 버튼을 무조건 숨기던 코드를 제거합니다! (대기열 큐잉 허용)
        // isSearching 변수가 Part 1에서 선언되었으므로 이제 에러가 발생하지 않습니다.
        if (isSearching && btnSubmit) btnSubmit.style.display = "none";
    }
    
    let i = 0;
    spinnerInterval = window.setInterval(() => {
        const char = spinnerFrames[i % spinnerFrames.length];
        if (settingsBtn) settingsBtn.innerText = char;
        
        document.querySelectorAll('.spinner, .active-spinner').forEach(el => {
            (el as HTMLElement).innerText = char;
        });
        i++;
    }, 80);
}

function stopSpinner() {
    // 🌟 [CRITICAL FIX] 추출(Extracting) 중이거나 검색(Searching) 중이면, 
    // 백그라운드 태스크가 함부로 글로벌 스피너를 끄지 못하도록 절대 방어합니다!
    if (isExtracting || isSearching) return;

    if (spinnerInterval) {
        clearInterval(spinnerInterval);
        spinnerInterval = null;
    }
    
    if (settingsBtn) {
        settingsBtn.classList.remove("active-spinner-mode");
        settingsBtn.innerText = settingsBtn.classList.contains('active') ? "💬" : "🗨️";
    }
    
    document.querySelectorAll('.spinner, .active-spinner').forEach(el => {
        if (!el.closest('#extraction-log')) {
            el.classList.remove('active-spinner');
            (el as HTMLElement).innerText = "";
        }
    });

    // 🌟 [수정] 스피너 정지 시, 진행 중이지 않은 유효한 텍스트 입력값이 존재할 때만 검색 버튼 노출
    if (btnSubmit) {
        const currentVal = searchInput?.value.trim() || "";
        // 스피너가 멈췄다는 건 작업이 끝났다는 의미이므로, isQueryActive(currentVal)가 false가 되어 버튼이 살아납니다.
        if (currentVal !== "" && !isQueryActive(currentVal)) {
            btnSubmit.style.display = "flex";
        } else {
            btnSubmit.style.display = "none";
        }
    }

    updateExtractButtonVisibility();
}

// --- Layout & Window Logic ---
async function setWindowSize(expanded: boolean) {
    const height = expanded ? EXPANDED_HEIGHT : COLLAPSED_HEIGHT;
    await invoke("resize_window", { width: WIDGET_WIDTH, height: height });
}

function switchTab(tabName: string) {
    tabContents.forEach(c => {
        if (c.id === `tab-${tabName}`) c.classList.add("active");
        else c.classList.remove("active");
    });
    currentTab = tabName;
    
    if (tabName === "settings") {
        settingsBtn?.classList.add("active-emoji", "active");
        
        // 🌟 [CRITICAL FIX] Settings 버튼을 누르면 3초를 기다리지 않고 즉시 1회 서버 통신(인증/동기화)을 강제 실행합니다!
        if (!currentSession.email) {
            checkAuthStatus();
        }

        // 🌟 [CRITICAL FIX] 검색 중(isSearching)이거나 추출 중(isExtracting)일 때는 
        // 탭을 전환하더라도 억지로 히스토리를 리셋하여 방금 작성한 말풍선을 날려버리지 않도록 방어합니다!
        if (!isSearching && !isExtracting) {
            fetchChatHistory();
        } else {
            // 🌟 진행 중인 작업 때문에 돔을 리셋(innerHTML="")할 수는 없지만,
            // 달랑 진행 중인 말풍선 1~2개만 있고 과거 내역이 안 불러와진 상태라면 과거 내역(isHistory=true)을 끌어와서 화면을 채웁니다!
            if (chatTalks && chatTalks.children.length < 10 && chatHasMore) {
                loadMoreChat(true, true);
            } else {
                // 이미 화면이 채워져 있다면 최신 상태 변경점(status)만 조용히 동기화합니다.
                loadMoreChat(false, true);
            }
        }
        
        // 🌟 탭이 열렸으므로 폴링 타이머를 새롭게 리셋하여 주기를 맞춥니다.
        startPolling();
    } else {
        settingsBtn?.classList.remove("active-emoji", "active");
    }
    
    // 🌟 [추가] 리스트 탭으로 이동 시 검색 결과 화면이었다면 전체 리스트로 복구합니다.
    if (tabName === "list") {
        const resultH3 = document.querySelector('.nav-section.search h3');
        const isShowingSearchResult = resultH3 && resultH3.textContent?.toLowerCase().includes("search");
        
        // 🌟 [CRITICAL FIX] 검색이 진행 중(isSearching)일 때는 절대 초기화(refreshList)를 방지하여 검색 화면 및 진행 상태(Cancel 버튼)가 날아가는 것을 막습니다!
        // 또한 일반 리스트 상태일 때는 탭을 전환한다고 굳이 초기화하지 않아 무한스크롤 상태를 보존합니다.
        if (isShowingSearchResult && !isSearching) {
            if (searchInput) searchInput.value = "";
            if (resultH3) resultH3.innerHTML = `Result <strong class="count"></strong>`;
            refreshList(); 
        }
    }
    if (tabName === "automation") initBrowserDropdown();
}

function openWidget(tabName: string = "list") {
    if (!isExpanded) {
        isExpanded = true;
        contentPanel.classList.add("open");
        settingsBtn.innerText = "💬";
        setWindowSize(true);
    }
    switchTab(tabName);
}

function collapseWidget() {
    isExpanded = false;
    contentPanel.classList.remove("open");
    setWindowSize(false);
    settingsBtn?.classList.remove("active-emoji", "active");
    settingsBtn.innerText = "🗨️";
}

// --- Mouse Passthrough Logic ---
const interactiveElements = ['.pill-nav', '#content-panel'];

function setupMousePassthrough() {
    // [FIX] 기본적으로 위젯은 클릭이 가능해야 합니다. 
    // 윈도우 크기(380x80)가 이미 작기 때문에 윈도우 밖은 자동으로 클릭이 통과됩니다.
    invoke('set_ignore_cursor_events', { ignore: false }).catch(console.error);

    interactiveElements.forEach(selector => {
        const el = document.querySelector(selector);
        if (el) {
            el.addEventListener('mouseenter', () => {
                invoke('set_ignore_cursor_events', { ignore: false }).catch(console.error);
            });
        }
    });
}

// Drag Logic
const pillNav = document.querySelector('.pill-nav') as HTMLElement;
if (pillNav) {
    setupMousePassthrough(); // Initialize passthrough with the nav
    let isMouseDown = false;
    let startX = 0, startY = 0;
    const DRAG_THRESHOLD = 5;

    pillNav.addEventListener('mousedown', (e) => {
        const target = e.target as HTMLElement;
        if (!target.closest('button, input') && e.button === 0) {
             isMouseDown = true; startX = e.clientX; startY = e.clientY;
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (!isMouseDown) return;
        if (Math.abs(e.clientX - startX) > DRAG_THRESHOLD || Math.abs(e.clientY - startY) > DRAG_THRESHOLD) {
            isMouseDown = false; invoke('start_drag').catch(console.error);
        }
    });
    window.addEventListener('mouseup', () => isMouseDown = false);
    pillNav.addEventListener('dblclick', (e) => {
        const target = e.target as HTMLElement;
        if (!target.closest('button, input')) invoke("move_to_top_center").catch(console.error);
    });
}

let extractClickLock = false; 

async function updateExtractButtonVisibility() {
    if (!btnExtract || !btnAutoLaunch) return;

    // 1. 브라우저 물리 상태 체크 (동기/즉시 실행)
    // 이미지(currentImage)가 선택된 상태라면 브라우저 실행 여부와 무관하게 반환하지 않고 진행합니다.
    if (!isBrowserRunning && !isAutoLaunchLocked && !currentImage) {
        btnAutoLaunch.style.display = "flex";
        btnAutoLaunch.classList.remove("hidden");
        btnExtract.style.display = "none";
        return;
    }

    btnAutoLaunch.style.display = "none";
    btnAutoLaunch.classList.add("hidden");

    // 2. URL 유효성 및 이미지 업로드 즉시 검사
    const isInvalidUrl = !currentDetectedUrl || 
                         currentDetectedUrl === "" || 
                         currentDetectedUrl === "about:blank" || 
                         currentDetectedUrl.startsWith("chrome://") || 
                         currentDetectedUrl.startsWith("edge://");

    if (!currentImage && isInvalidUrl) {
        btnExtract.style.display = "none";
        btnExtract.classList.add("hidden");
        return;
    }

    // 3. 도메인 허용 여부 판별 (DB 조회 없이 DOM 캐시 활용)
    let isAllowedDomain = isCurrentShop;

    if (!isAllowedDomain && currentDetectedUrl) {
        try {
            const currentHostname = new URL(currentDetectedUrl.toLowerCase()).hostname;
            // DB를 비동기로 호출하지 않고 이미 그려진 내비게이션 요소에서 도메인 목록을 즉시 확인합니다.
            const pageList = document.getElementById("nav-list-pages");
            if (pageList) {
                const labels = Array.from(pageList.querySelectorAll(".logis-label")) as HTMLElement[];
                isAllowedDomain = labels.some(label => {
                    const domain = label.dataset.domain;
                    return domain && (currentHostname === domain || currentHostname.endsWith("." + domain));
                });
            }
        } catch (e) { isAllowedDomain = false; }
    }

    // 4. 도메인 및 이미지 업로드 여부로 1차 필터링
    if (!isAllowedDomain && !currentImage) {
        btnExtract.style.display = "none";
        btnExtract.classList.add("hidden");
        return; 
    }

    // 5. 더블 클릭 및 비동기 작업(Lock 확인 및 태스크 상태 질의) 처리 후 최종 가시성 결정
    // 🌟 [CRITICAL FIX] 즉시 렌더링(flex) 후 비동기로 숨기는(none) 로직 때문에 깜빡임이 발생했습니다. 
    // 비동기 검증을 먼저 await로 대기한 뒤 최종적으로 한 번만 렌더링하여 깜빡임을 원천 차단합니다.
    if (extractClickLock) {
        btnExtract.style.display = "none";
        return;
    }

    // 고아 락 해제용 로직 (버튼 가시성에는 영향 주지 않음)
    const currentLock = await kvGet("sys_lock");
    if (currentLock) {
        const lockEl = document.getElementById(currentLock);
        if (!lockEl) {
            // 🌟 [CRITICAL FIX] 5000ms라는 불확실한 시간 기반 땜질 로직을 전면 폐기하고,
            // 실제 프론트엔드/백엔드 큐(대기열)에 존재하는지 명확한 상태 기반으로 교차 검증하여 유령 락을 즉각 해제합니다.
            const isFrontendActive = GlobalTaskManager.currentTaskId === currentLock || GlobalTaskManager.queue.some(q => q.taskId === currentLock);
            const isBackendActive = GlobalTaskManager.backendQueued.some(p => p.id === currentLock || p.taskId === currentLock);
            
            if (!isFrontendActive && !isBackendActive) {
                console.log(`[LOCK] Zombie lock detected without active queue: ${currentLock}. Releasing immediately.`);
                await kvRemove("sys_lock");
            }
        }
    }

    let shouldHide = false;
    try {
        if (currentImage) {
            const imageRefHash = await hashId(currentImage); 
            const isActive = await invoke<boolean>("check_active_task", { payload: { cc: activeContext.cc || "", ref: imageRefHash } });
            // 🌟 프론트엔드 대기 큐 및 백엔드 대기 큐(backendQueued) 동시 확인
            const isQueued = GlobalTaskManager.queue.some(q => q.payload && q.payload.ref === imageRefHash) ||
                             GlobalTaskManager.backendQueued.some(p => p.ref === imageRefHash);
            
            const isCurrentExecuting = GlobalTaskManager.currentTaskId && GlobalTaskManager.currentTaskPayload && 
                GlobalTaskManager.currentTaskPayload.ref === imageRefHash;

            if (isActive || isQueued || isCurrentExecuting) shouldHide = true;
        } else if (currentDetectedUrl) {
            const urlObj = new URL(currentDetectedUrl.toLowerCase());
            const link = (urlObj.pathname + urlObj.search).toLowerCase();
            const rootDomain = getRootDomain(urlObj.hostname);
            const ccHash = await hashId(rootDomain);
            const hashedRefId = await hashId((currentSession.team || "") + ccHash + link);
            // 🌟 [CRITICAL FIX v3] 현재 URL 기반 해시를 1순위로 검사합니다.
            const currentRefToCheck = hashedRefId;
            let isActive = await invoke<boolean>("check_active_task", { payload: { cc: ccHash, ref: currentRefToCheck } });
            // 🌟 [CRITICAL FIX v3] hashedRefId로 매칭되지 않았을 때 activeContext.ref도 추가 확인합니다.
            //    네비게이션 클릭 후 추출 시 task.ref = activeContext.ref로 저장되므로
            //    URL 기반 해시만으로는 매칭이 안 되는 케이스를 커버합니다.
            //    (URL 변경 시 browser-match-found에서 activeContext.ref가 ""로 초기화되므로
            //     다른 페이지에서는 이 분기가 실행되지 않아 원래 버그가 재발하지 않습니다.)
            if (!isActive && activeContext.ref && activeContext.ref !== currentRefToCheck) {
                isActive = await invoke<boolean>("check_active_task", { payload: { cc: ccHash, ref: activeContext.ref } });
            }
            // 🌟 [CRITICAL FIX v3] 네비게이션 렌더링이 아직 activeContext.ref를 설정하지 못한
            //    레이스 컨디션 상태에서, 백엔드 ACTIVE_TASK_MEM에 같은 페이지의 진행 중 태스크가
            //    있는지 get_active_task_context로 추가 교차 검증합니다.
            //    🌟 [FIX v4] 단, 활성 태스크의 link가 현재 URL의 path와 일치할 때만 숨김 처리합니다.
            //    아직 동기화하지 않은 새 팝업 페이지로 포커싱했을 때, 다른 페이지의 태스크 때문에
            //    버튼이 잘못 숨겨지는 것을 방지합니다.
            if (!isActive && !activeContext.ref) {
                try {
                    const activeCtx = await invoke<any>("get_active_task_context");
                    if (activeCtx && activeCtx.id && (activeCtx.status === 1 || activeCtx.status === 10)) {
                        const activeLink = (activeCtx.link || "").toLowerCase();
                        if (activeLink && activeLink === link) {
                            isActive = true;
                        }
                    }
                } catch (_e2) { /* 무시 */ }
            }
            // 🌟 프론트엔드 대기 큐 및 백엔드 대기 큐(backendQueued) 동시 확인
            const isQueued = GlobalTaskManager.queue.some(q => q.payload && (q.payload.ref === currentRefToCheck || q.payload.link === link)) ||
                GlobalTaskManager.backendQueued.some(p => p.ref === currentRefToCheck || p.link === link);
            const isCurrentExecuting = GlobalTaskManager.currentTaskId && GlobalTaskManager.currentTaskPayload &&
                (GlobalTaskManager.currentTaskPayload.ref === currentRefToCheck || GlobalTaskManager.currentTaskPayload.link === link);
            if (isActive || isQueued || isCurrentExecuting) shouldHide = true;
        }
    } catch (e) {
        // 통신 에러 발생 시 노출 유지
    }

    if (shouldHide) {
        btnExtract.style.display = "none";
        btnExtract.classList.add("hidden");
    } else {
        btnExtract.style.display = "flex";
        btnExtract.innerHTML = "⚡";
        btnExtract.classList.remove("hidden");
    }
}

listen("browser-match-found", async (event: any) => {
    const payload = event.payload;
    if (payload.status === "running" || (payload.url && payload.url !== "")) {
        isBrowserRunning = true;
    } else if (payload.status === "stopped") {
        isBrowserRunning = false;
        isAutoLaunchLocked = false;
    }
    currentDetectedUrl = payload.url || "";
    isCurrentShop = payload.is_client || payload.is_admin || false;
    // 🌟 [CRITICAL FIX] URL 변경 시 이전 페이지의 activeContext.ref를 초기화합니다.
    //    이어서 호출되는 renderNavigation()이 새 페이지의 ref를 재설정하므로,
    //    이전 페이지의 stale ref가 버튼 가시성 판정을 오염시키는 것을 원천 차단합니다.
    activeContext.ref = "";
    // 🌟 [CRITICAL FIX v3] renderNavigation()이 activeContext.ref를 재설정하기 전에
    //    updateExtractButtonVisibility()를 호출하면 ref="" 상태에서 매칭 실패 → 버튼 오노출됩니다.
    //    따라서 renderNavigation()을 먼저 완료시킨 뒤, 설정된 ref 기반으로 최종 판정을 내립니다.
    await renderNavigation();
    await updateExtractButtonVisibility();
});

listen("browser-status", async (event: any) => {
    const payload = event.payload; 
    const statusStr = typeof payload === "object" ? payload.status : payload;
    
    if (typeof payload === "object" && payload.url !== undefined) {
        const prevUrl = currentDetectedUrl;
        currentDetectedUrl = payload.url || "";
        isCurrentShop = payload.is_client || payload.is_admin || false;
        // 🌟 [CRITICAL FIX v3] URL이 실제로 변경되었을 때만 버튼 가시성 재평가를 수행합니다.
        //    800ms 주기 하트비트에서 매번 재평가하면 비동기 레이스로 버튼이 깜빡입니다.
        if (prevUrl !== currentDetectedUrl) {
            await updateExtractButtonVisibility();
        }
    }

    if (statusStr === "running") {
        isBrowserRunning = true;
        isAutoLaunchLocked = true; // 실행 중엔 런처 버튼 잠금
    } else if (statusStr === "stopped") {
        isBrowserRunning = false;
        isAutoLaunchLocked = false;
        currentDetectedUrl = "";
        // 🌟 [CRITICAL FIX] 브라우저 종료 시 btnAutoLaunch를 직접 노출시킵니다.
        //    updateExtractButtonVisibility() 내부의 extractClickLock 조기 리턴이나
        //    isExtracting 상태에 의해 btnAutoLaunch 노출이 누락되는 것을 원천 차단합니다.
        if (btnAutoLaunch) {
            btnAutoLaunch.style.display = "flex";
            btnAutoLaunch.classList.remove("hidden");
        }
        if (btnExtract) {
            btnExtract.style.display = "none";
            btnExtract.classList.add("hidden");
        }
        await updateExtractButtonVisibility();
    }
});

const handleSearchInteraction = () => {
    // 🌟 [추가] 검색창 클릭/포커스 시 세팅 패널이 열려있다면 강제로 스위치를 끄고 닫아줍니다.
    const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
    if (settingsToggle && settingsToggle.checked) {
        settingsToggle.checked = false;
        settingsToggle.dispatchEvent(new Event("change")); // UI 원상복구 이벤트 트리거
    }

    const resultH3 = document.querySelector('.nav-section.search h3');
    const isShowingSearchResult = resultH3 && resultH3.textContent?.toLowerCase().includes("search");

    // 🌟 [추가] 도메인(CC)이 선택되어 있지 않을 경우, 기본 홈 링크(/tracking) 도메인으로 강제 할당합니다.
    // if (!activeContext.cc && currentSession.team) {
    //     (async () => {
    //         const defaultDomain = "logis.center";
    //         const defaultPath = "/tracking";
            
    //         // 해시값 생성
    //         const ccHash = await hashId(defaultDomain);
    //         const refHash = await hashId(currentSession.team + ccHash + defaultPath);
            
    //         // 컨텍스트에 강제 주입
    //         activeContext.cc = ccHash;
    //         activeContext.ref = refHash;
            
    //         // UI 태그 추가 (사용자에게 시각적 피드백 제공)
    //         addSearchTag(`@${defaultDomain}`, 'domain', defaultDomain);
    //         addSearchTag(`#tracking`, 'type', 'tracking');
    //         updateTagsUI();
            
    //         console.log(`[NAV] No domain selected. Defaulting to: ${defaultDomain} (${defaultPath})`);
    //     })();
    // }

    // [UI-FIX] If the panel is already expanded, don't refresh the navigation or clear the list.
    // This prevents annoying UI flickering when the user just wants to type in the search bar.
    if (isExpanded && currentTab === "list") {
        // 🌟 [수정] 검색 결과가 표시된 상태에서 검색창을 클릭했다면 다시 전체 리스트로 복구합니다.
        // 🌟 [CRITICAL FIX] 단, 현재 검색이 진행 중(isSearching)이라면 초기화하지 않고 그대로 둡니다.
        if (isShowingSearchResult && !isSearching) {
            if (searchInput) searchInput.value = "";
            if (resultH3) resultH3.innerHTML = `Result <strong class="count"></strong>`;
            refreshList();
        }
        return;
    }

    openWidget("list");
    const navOverlay = document.getElementById("nav-categories");
    if (navOverlay) {
        navOverlay.classList.remove("hidden");
        navOverlay.classList.add("visible");
        renderNavigation();
        if (listScrollContainer) listScrollContainer.scrollTo({ top: 0, behavior: 'smooth' });
    }
    
    // 🌟 [CRITICAL FIX] 검색 진행 중일 때는 하단 초기화 로직도 타지 않도록 완벽히 방어합니다.
    if (!isSearching && (!searchInput.value || isShowingSearchResult)) {
        if (searchInput) searchInput.value = "";
        if (resultH3) resultH3.innerHTML = `Result <strong class="count"></strong>`;
        if (docListContainer) docListContainer.innerHTML = "";
        cachedDocs = [];
        currentPage = 0;
        hasMore = true;
        // 🌟 [CRITICAL FIX] 빈 검색창 클릭으로 위젯을 열었을 때, 목록을 지우기만 하고 다시 불러오지 않아 빈 화면이 출력되는 현상 수정
        loadMoreDocs(true);
    }
};

searchInput?.addEventListener("focus", handleSearchInteraction);
searchInput?.addEventListener("click", handleSearchInteraction);

function hideNavigation() {
    const navOverlay = document.getElementById("nav-categories");
    if (navOverlay) {
        navOverlay.classList.add("hidden");
        navOverlay.classList.remove("visible");
    }
}

function addSearchTag(label: string, type: 'domain' | 'type' | 'mode' | 'path', value: string) {
    const id = `${type}:${value}`;
    if (activeTags.find(t => t.id === id)) return;
    activeTags.push({ id, label, type, value });
    updateTagsUI();
    if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
    searchDebounceTimer = window.setTimeout(() => loadMoreDocs(true), 300);
}

function removeSearchTag(id: string) {
    const tagToRemove = activeTags.find(t => t.id === id);
    if (tagToRemove) {
        // [FIX] Reset specific context when corresponding tags are removed
        if (tagToRemove.type === 'domain') activeContext.cc = "";
        if (tagToRemove.type === 'type') activeContext.ref = "";
        if (tagToRemove.type === 'path') activeContext.ref = "";
    }

    activeTags = activeTags.filter(t => t.id !== id);
    
    // If no more tags left, clear the entire active context
    if (activeTags.length === 0) {
        activeContext = { cc: "", bcc: "", ref: "" };
    }

    updateTagsUI();
    loadMoreDocs(true);
    
    // [FIX] Also refresh chat history to reflect cleared filters
    fetchChatHistory(true);
}

function updateTagsUI() {
    const container = document.getElementById("search-tags-container");
    if (!container) return;
    container.innerHTML = "";
    activeTags.forEach(tag => {
        const chip = document.createElement("div");
        chip.className = `search-chip ${tag.type}`;
        chip.innerHTML = `<span>${tag.label}</span><span class="remove-btn" onclick="document.dispatchEvent(new CustomEvent('remove-tag', {detail: '${tag.id}'}))">✕</span>`;
        container.appendChild(chip);
    });
}

document.addEventListener('remove-tag', (e: any) => removeSearchTag(e.detail));

// --- Tree Rendering Logic (Pages & Users) ---
// --- Original Logic Implementation from content.js ---
let navTmp: Record<string, boolean> = {};

async function renderAccordion(nodes: any[], level = 1): Promise<string> {
    let html = `<ul class="logis-branch">`;

    for (var n = 0; n < nodes.length; n++) {
        var node = nodes[n];
        var nodeId = node.id || node.uuid || `node-${level}-${n}`;
        var active = '';
        var host = '';
        var type = 'page';
        var content = '';
        var name = '';
        var desc: string[] = [];
        var _url: URL | null = null;

        // ONLY generate HTML if this node hasn't been rendered yet
        if (!navTmp[nodeId]) {
            navTmp[nodeId] = true;

            if (node.name) {
                type = node.type || "team";
                name = node.name;
                if (node.type === "team") {
                    var teamName = node.name;
                    if (node.from === currentSession.address && nodeId === node.to) {
                        teamName = "Members";
                    }
                    host = `<strong>${teamName}</strong>`;
                } else {
                    let cancelBtn = "";

                    // 🌟 [수정] 본인 계정만 (owner), 초대된 멤버나 펜딩 상태는 (member)로 표시
                    if (node.id === currentSession.address) {
                        desc.push("(owner)");
                        // 🌟 본인(Owner 또는 Self)일 경우 삭제/취소 버튼을 노출하지 않습니다.
                    } else {
                        desc.push("(member)");
                        // 🌟 타인(Member)일 경우에만 삭제/취소 버튼을 노출합니다.
                        cancelBtn = `<button class="btn-cancel-member" data-id="${nodeId}" data-name="${name}" style="background:none; border:none; color:#ef4444; font-size:0.85rem; cursor:pointer; padding:0 5px; margin-left:auto; display:flex; align-items:center; justify-content:center;" title="Remove / Cancel Invite">✕</button>`;
                    }

                    // 🌟 [수정] 버튼이 우측 끝에 붙도록 Flex 구조 적용
                    content = `<div style="display:flex; align-items:center; width:100%; gap:8px;">
                        <span>${name}${desc.length ? `<i>${desc.toString()}</i>` : ''}</span>
                        ${cancelBtn}
                    </div>`;
                }
                if (nodeId === activeContext.ref) active = "active";
            } else if (node.data || node.type) {
                type = 'page';
                const data = node.data || {};
                // 🌟 [CRITICAL FIX] DB 테이블명(pages)이 아닌 진짜 속성(goods, tracking)을 우선적으로 참조하도록 수정합니다.
                const nodeType = data.type || node.type || 'unknown';
                console.log("[DEBUG] renderAccordion Page Node:", { id: nodeId, type: nodeType, domain: node.domain, origin: data.origin, link: data.link });

                if (data.origin) {
                    _url = new URL(data.origin);
                    const domain = node.domain || _url.hostname;
                    if (!navTmp[domain] && data.item) {
                        // 🌟 [CRITICAL FIX] bb.ts 패리티 완벽 복원: CSS 파괴를 막기 위해 불필요한 div 래핑을 모두 제거하고 원본 구조 유지
                        host = `<strong>${domain}</strong>`;
                        if (API_HOST.includes(domain)) {
                            host += `<label for="membership">Edit</label>`;
                        }
                        navTmp[domain] = true;
                    }

                    // 🌟 [CRITICAL FIX] before.ts 패리티 완벽 복원: Hash ID 기반 매칭에 실패했을 경우를 대비해, 
                    // 현재 브라우저 URL의 파라미터를 분석하여 리스트(부모)와 상세(자식) 활성화 대상을 100% 정확히 찾아냅니다.
                    let isActive = nodeId === activeContext.ref;

                    if (!isActive && currentDetectedUrl && data.link) {
                        try {
                            const currentUrl = new URL(currentDetectedUrl.toLowerCase());
                            const targetUrl = new URL((data.origin + data.link).toLowerCase());

                            if (currentUrl.pathname === targetUrl.pathname) {
                                const currentParams = Object.fromEntries(currentUrl.searchParams.entries());
                                const targetParams = Object.fromEntries(targetUrl.searchParams.entries());
                                const currentKeys = Object.keys(currentParams);
                                const targetKeys = Object.keys(targetParams);

                                const isDetailMode = Object.values(currentParams).some(val => 
                                    val === "form" || val === "view" || val === "detail" || val === "update" || val === "edit" || val === "read"
                                ) || (currentKeys.length > targetKeys.length);

                                if (data.detail) {
                                    if (isDetailMode) isActive = true;
                                } else {
                                    if (!isDetailMode) {
                                        let isExactMatch = true;
                                        for (const key of targetKeys) {
                                            if (currentParams[key] !== targetParams[key]) {
                                                isExactMatch = false; break;
                                            }
                                        }
                                        if (isExactMatch) isActive = true;
                                    }
                                }
                            }
                        } catch(e) {}
                    }

                    if (isActive) {
                        active = "active";
                    }
                }
                var total = { draft: 0, count: 0 };
                const cc = node.cc || data.cc;

                // 🌟 [DEXIE COUNT] team.data.base.pages 증감 통계 대신 Dexie 인덱스로 직접 셉니다.
                //    ① 증감 방식은 서버(proxy/index.ts)와 로컬(metrics.rs)이 같은 트리를 각자 갱신하므로
                //       한쪽이 실패하면 영구히 어긋납니다.
                //    ② relay draft / DEDUP 스킵 / ORDER-INDEX FALLBACK 처럼 분기가 늘 때마다
                //       증감 지점을 같이 늘려야 합니다.
                //    ③ '[cc+type]' 복합 인덱스가 이미 선언돼 있어 range 조회 1회면 끝납니다.
                //
                //    🌟 [DRAFT 판정 계약] proxy/index.ts 와 완전히 동일합니다.
                //       updated_at == 0  → draft (리스트 스캔으로 껍데기만 존재)
                //       updated_at >  0  → count (상세 추출까지 완료)
                //    cc+type 단위 단일 통계이므로 리스트 노드와 상세 노드가 같은 total 을 공유하고,
                //    리스트 노드는 total.draft 를, 상세 노드는 total.count 를 표시합니다.
                //    (이 구조 자체가 base.pages[cc][type] 와 동일합니다)
                if (
                    nodeType !== 'team' &&
                    nodeType !== 'user' &&
                    nodeType !== 'member' &&
                    nodeType !== 'pages' &&
                    nodeType !== 'page' &&
                    cc &&
                    appDb
                ) {
                    try {
                        const rows = await appDb.table('items')
                            .where('[cc+type]')
                            .equals([cc, nodeType])
                            .toArray();

                        for (const r of rows) {
                            // 🌟 [DRAFT COUNT v2] 봉투 루트 updated_at 과 data.updated_at 둘 다 확인합니다.
                            //    normalizeEnvelope 가 data.* 로 내린 뒤에도 루트 updated_at 이
                            //    별도 경로(syncData bulkPut 이전)로 갱신될 수 있으므로 이중 판정합니다.
                            const rootUp = Number(r.updated_at ?? 0);
                            const dataUp = Number(r.data?.updated_at ?? rootUp);
                            const up = dataUp;

                            if (up > 0) total.count++;
                            else total.draft++;
                        }

                        console.log(`[NAV-COUNT] cc=${cc} type=${nodeType} | 총 ${rows.length}건 → draft ${total.draft} / count ${total.count}`);

                        if (rows.length > 0 && total.draft === 0 && total.count === rows.length) {
                            console.warn(`[NAV-COUNT] ⚠️ 전 건이 count 로 분류되었습니다. store.rs 의 updated_at=0 보존 수정이 반영되었는지 확인하세요.`);
                        }
                    } catch (e) {
                        console.warn(`[NAV-COUNT] Dexie count failed for ${nodeType}:`, e);
                    }
                }

                var recent = '';
                try {
                    const bcc = node.bcc || data.bcc;
                    if (bcc) {
                        // 🌟 [CRITICAL FIX 1] AI 검색(Select)을 타격하여 무한 스피너가 도는 현상을 원천 차단합니다.
                        const _items = await invoke<any[]>("get_all_documents", { limit: 1, offset: 0, filter: `bcc = '${bcc}'` });
                        if (_items.length && _items[0].created_at) {
                            // 🌟 bb.ts의 유저 이름 표기 로직과 동일하게 작성자 정보를 가져와 병합 표기
                            const timeStr = time2text(Number(_items[0].created_at));
                            const author = _items[0].from ? _items[0].from.substring(0,6) : "system";
                            recent = `<strong>${timeStr} - ${author}</strong>`;
                        }
                    }
                } catch (err) {}

                // 🌟 [CRITICAL FIX] bb.ts 패리티 완벽 복원: 실제 타입(nodeType)을 최우선으로 출력하여 'pages'로 노출되는 버그를 고칩니다!
                // Draft 텍스트는 아래 count 변수 조립 시 명시적으로 통합합니다.
                name = `<span>${nodeType}</span>`;
                
                var count = '';
                if (data.item) {
                    // 🌟 전처리 중인 리스트 페이지일 경우, Draft 수량과 정식 처리된 Count 수량을 함께 노출합니다.
                    count = `<span style="font-size: 0.9em; margin-left: 4px;"> Draft <u>(${total.draft || 0})</u></span>`;
                } else {
                    // 🌟 상세 페이지일 경우 기존처럼 Count만 노출합니다.
                    count = `<u>(${total.count || 0})</u>`;
                }
                
                // 🌟 [추가] 숨김 처리 상태 아이콘 및 스타일 적용
                const isHidden = hiddenPages.includes(nodeId);
                const visibilityIcon = isHidden ? "show" : "hide";
                
                // 🌟 [CRITICAL FIX] 기존 CSS 레이아웃을 파괴하지 않도록 절대 위치(absolute)를 사용하여 우측 상단에 버튼을 배치합니다.
                const visibilityBtn = `<button class="btn-toggle-visibility" data-id="${nodeId}" style="position: absolute; right: 10px; top: 1px; background: none; border: none; cursor: pointer; font-size: 10px; text-decoration: underline; color: #888; z-index: 10;">${visibilityIcon}</button>`;

                // 🌟 [CRITICAL FIX] 실수로 누락했던 visibilityBtn 변수를 content 문자열 맨 끝에 다시 포함시킵니다!
                const opacityStyle = isHidden ? 'opacity: 0.3;' : 'opacity: 1;';
                content = `<span style="${opacityStyle}">${name}\n${count}\n</span>\n${recent}\n${visibilityBtn}`;
            }

            var hasChildren = node.children && node.children.length > 0;
            const inputId = `${type}-${nodeId}`;

            html += `
                <input type="checkbox" name="${type}" id="${inputId}" ${hasChildren ? 'checked' : ''} style="display:none;" />
                <li class="logis-parent ${hasChildren ? 'has-children' : ''}" ${type}-id="${nodeId}">
                    ${host}
                    <label for="${inputId}" class="logis-label ${inputId} ${active}" 
                           data-id="${nodeId}" 
                           data-cc="${node.cc || (node.data && node.data.cc) || ''}" 
                           data-bcc="${node.bcc || (node.data && node.data.bcc) || ''}" 
                           data-ref="${node.ref || node.ref_val || (node.data && node.data.ref) || ''}"
                           data-domain="${node.domain || (_url ? _url.hostname : '')}" 
                           data-type="${node.type || (node.data && node.data.type) || ''}">${content}</label>
            `;

            if (hasChildren) {
                html += `<div class="logis-child ${inputId}">`;
                html += await renderAccordion(node.children, level + 1);
                html += `</div>`;
            }

            html += `</li>`;
        }
    }

    html += `</ul>`;
    return html;
}

async function renderNavigation() {
    const pageList = document.getElementById("nav-list-pages");
    const userList = document.getElementById("nav-list-users");
    const profileName = document.getElementById("nav-profile-name");
    const profileFavicon = document.getElementById("nav-profile-favicon");
    const btnSignin = document.getElementById("nav-signin");
    const btnSignout = document.getElementById("nav-signout");

    if (!pageList || !userList) return;

    // [FIX] Show spinner only on the very first navigation render
    if (isFirstNavRender) {
        startSpinner();
    }

    // Profile UI
    if (currentSession.email) {
        if (profileName) profileName.innerText = currentSession.email.split('@')[0];
        if (btnSignin) btnSignin.classList.add("hidden");
        if (btnSignout) btnSignout.classList.remove("hidden");
        if (profileFavicon && blockies) {
            const icon = blockies.create({ seed: currentSession.email, size: 8, scale: 4 });
            profileFavicon.innerHTML = ""; profileFavicon.appendChild(icon);
        }
    }

    try {
        navTmp = {}; // Reset for fresh render
        let _pagesRaw = await Select["pages"]({});
        
        // 🌟 [CRITICAL FIX] 백엔드(LanceDB)에서 가져온 TradeDocument는 알맹이가 json_data 문자열에 있으므로 반드시 파싱해 주어야 UI 필터링에서 증발하지 않습니다!
        let _pages = _pagesRaw.map(p => {
            if (!p.data && p.json_data && typeof p.json_data === "string") {
                try { p.data = JSON.parse(p.json_data); } catch(e) {}
            }
            return p;
        });

        // 🌟 [CRITICAL FIX] 크롬 브라우저의 현재 접속 도메인과 일치하는 페이지만 남깁니다.
        let currentDomain = "";
        console.log(`[DEBUG-NAV] 브라우저 현재 감지된 URL(currentDetectedUrl):`, currentDetectedUrl);
        
        if (currentDetectedUrl) {
            try {
                const footprint = new URL(currentDetectedUrl.toLowerCase());
                currentDomain = footprint.hostname;
                console.log(`[DEBUG-NAV] 파싱된 현재 도메인(currentDomain):`, currentDomain);
                
                // 🌟 [CRITICAL FIX] before.ts 패리티 완벽 복원: 해시 규칙 불일치로 못 찾던 문제를, 
                // URL 문자열 직접 대조 및 상세/리스트 파라미터 판별을 통해 활성 컨텍스트(activeContext)를 100% 완벽히 복원합니다.
                if (!activeContext.ref) {
                    console.log(`[DEBUG-NAV] 활성 컨텍스트(activeContext.ref)가 비어있어 URL 기반 자동 복구를 시도합니다.`);
                    const currentParams = Object.fromEntries(footprint.searchParams.entries());
                    const isDetailMode = Object.values(currentParams).some(val => 
                        val === "form" || val === "view" || val === "detail" || val === "update" || val === "edit" || val === "read"
                    );

                    let matchedPage = null;
                    
                    const localPages = await Select["pages"]({});
                    for (const p of localPages) {
                        const d = p.data || p;
                        if (d.origin && d.link) {
                            try {
                                const targetUrl = new URL((d.origin + d.link).toLowerCase());
                                if (currentDomain === targetUrl.hostname && footprint.pathname === targetUrl.pathname) {
                                    if (isDetailMode && d.detail) {
                                        matchedPage = p;
                                        break;
                                    } else if (!isDetailMode && !d.detail) {
                                        const targetParams = Object.fromEntries(targetUrl.searchParams.entries());
                                        let isExactMatch = true;
                                        for (const key of Object.keys(targetParams)) {
                                            if (currentParams[key] !== targetParams[key]) {
                                                isExactMatch = false;
                                                break;
                                            }
                                        }
                                        if (isExactMatch) {
                                            matchedPage = p;
                                        }
                                    }
                                }
                            } catch(e) {}
                        }
                    }
                    
                    if (matchedPage) {
                        activeContext.cc = matchedPage.cc || "";
                        activeContext.bcc = matchedPage.bcc || "";
                        activeContext.ref = matchedPage.id || "";
                        console.log("[NAV] Restored activeContext from exact URL match:", activeContext);
                    }
                }
            } catch(e) {}
        }

        if (currentDomain) {
            _pages = _pages.filter(p => {
                const data = p.data || p;
                return data.origin && data.origin.toLowerCase().includes(currentDomain);
            });
        }
        
        const navSection = pageList.closest('.nav-section') as HTMLElement;
        const isSettingsOpen = (document.getElementById("settings-toggle") as HTMLInputElement)?.checked;

        // 🌟 [OAUTH REGISTER BUTTON] Pages nav-section 상단에 사이트 등록 버튼을 삽입합니다.
        //    매 렌더링마다 중복 생성을 방지하기 위해 기존 버튼을 먼저 제거합니다.
        if (navSection) {
            const existingBtn = navSection.querySelector("#btn-oauth-register");
            if (existingBtn) existingBtn.remove();

            // analytic 모드에서만 등록 버튼 노출 (이 버튼의 목적이 analytic 조회이므로)
            // 🌟 [LOGIN GATE] 로그인(currentSession.email)이 되어 있어야만 버튼을 노출합니다.
            //    미로그인 상태에서 등록을 시도하면 submitOAuthRegistration 내부에서
            //    "로그인이 필요합니다." 에러가 반환되므로, 사전에 UI에서 차단합니다.
            if (currentSearchMode === "analytic" && !isSettingsOpen && currentSession.email) {
                const h3 = navSection.querySelector("h3");
                const registerBtn = document.createElement("button");
                registerBtn.id = "btn-oauth-register";
                registerBtn.style.cssText = "position: absolute; left: 5em; top: 13px; border: 0px; padding: 0px; font-size: 0.8rem; cursor: pointer; text-align: center; text-decoration: underline; background: none;";
                registerBtn.textContent = "+ 사이트 등록 (Analytic)";
                registerBtn.addEventListener("click", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    renderOAuthRegistrationForm();
                });

                // h3 바로 아래, pageList 위에 삽입
                if (h3 && h3.nextSibling) {
                    navSection.insertBefore(registerBtn, h3.nextSibling);
                } else if (h3) {
                    navSection.appendChild(registerBtn);
                } else {
                    navSection.insertBefore(registerBtn, pageList);
                }
            }
        }

        if (_pages.length === 0) {
            // 🌟 [OAUTH CLEANUP] innerHTML 교체 전에 잔존 oauth 노드를 명시적으로 제거합니다.
            //    innerHTML = "" 자체가 전부 지우지만, 비동기 레이스에서
            //    이전 라운드의 insertAdjacentHTML 이 이 할당 '이후' 에 도착하는
            //    윈도우를 원천 차단하기 위해 쿼리로도 한 번 제거합니다.
            pageList.querySelectorAll(".oauth-site-item").forEach((el: Element) => el.remove());
            pageList.innerHTML = `<div class="empty">No shared pages found for this domain.</div>`;
            // 🌟 [CRITICAL FIX] 데이터가 없더라도 Commerce/Analytic 모드이면 "비어있음" 문구가 노출되도록 통일
            if (navSection) navSection.style.display = (isSettingsOpen || currentSearchMode === "shipping") ? "none" : "block";
        } else {
            // 🌟 일치하는 데이터가 있으면 섹션을 화면에 표시하되, 세팅/Shipping 상태에 맞춰 제어합니다.
            if (navSection) navSection.style.display = (isSettingsOpen || currentSearchMode === "shipping") ? "none" : "block";
            
            // 🌟 [CRITICAL FIX] bb.ts의 페이지 트리(Branch) 조립 로직을 완벽히 복원하여 뎁스가 깨지는 현상을 해결합니다.
            const branchs: Record<string, any> = {};

            for (let p = 0; p < _pages.length; p++) {
                let _page = _pages[p];
                const data = _page.data || _page;
                if (!data.origin) continue;

                const domain = new URL(data.origin).hostname;
                _page.domain = domain;
                _page.id = _page.id || _page.uuid;

                if (data.item) {
                    branchs[`${data.origin}#${_page.type}`] = { ..._page, children: [] };
                }
                branchs[_page.id] = { ..._page, children: [] };
            }

            const temp: Record<string, any> = {};
            for (let key in branchs) {
                if (branchs.hasOwnProperty(key)) {
                    let _page = safeClone(branchs[key]);
                    const data = _page.data || _page;

                    if (!temp[_page.id]) {
                        temp[_page.id] = true;

                        let parent = branchs[`${data.origin}#${_page.type}`];

                        // 🌟 [CRITICAL FIX] before.ts 패리티 완벽 복원: 모순이 발생하는 복잡한 Splice 로직을 걷어내고 가장 간결하고 정확한 트리 조립을 수행합니다.
                        if (parent) {
                            if (data.item) {
                                let children = safeClone(parent.children);
                                branchs[`${data.origin}#${_page.type}`] = {
                                    ..._page,
                                    children: children
                                };
                            } else {
                                branchs[`${data.origin}#${_page.type}`].children.push(_page);
                            }
                        } else {
                            if (data.item) {
                                if (!branchs[`${data.origin}#${_page.type}`]) {
                                    branchs[`${data.origin}#${_page.type}`] = {
                                        ..._page,
                                        children: []
                                    };
                                }
                            } else {
                                if (!branchs[`${data.origin}#${_page.type}`]) {
                                    branchs[`${data.origin}#${_page.type}`] = { children: [] };
                                }
                                branchs[`${data.origin}#${_page.type}`].children.push(_page);
                            }
                        }
                    }
                }
            }

            const tree: any[] = [];
            for (let key in branchs) {
                if (branchs.hasOwnProperty(key)) {
                    if (key.includes('#')) {
                        tree.push(branchs[key]);
                    }
                }
            }

            // 🌟 [CRITICAL FIX] 네비게이션(Accordion)을 렌더링하기 직전에, 
            // 로컬 DB(users 테이블)에 저장된 AI의 최신 추출 통계(count)를 불러와 메모리에 덮어씌웁니다!
            try {
                const _usersForStats = await Select["users"]({});
                const teamDoc = _usersForStats.find(u => u.type === "team" || (u.data && u.data.type === "team"));
                if (teamDoc) {
                    // 🌟 [CRITICAL FIX] LanceDB(TradeDocument)와 Server(JSON)의 포맷 차이 완벽 호환
                    // TradeDocument의 경우 최신 데이터가 문자열 형태의 json_data에 들어있으므로 최우선으로 파싱합니다.
                    let teamData: any = teamDoc;
                    
                    // 🌟 [CRITICAL FIX] 백엔드에서 이중, 삼중으로 인코딩된 json_data(Matryoshka 버그)를 완벽하게 벗겨냅니다.
                    while (teamData && teamData.json_data && typeof teamData.json_data === "string") {
                        try {
                            const parsed = JSON.parse(teamData.json_data);
                            if (parsed && typeof parsed === "object") {
                                teamData = parsed;
                            } else {
                                break;
                            }
                        } catch(e) {
                            break;
                        }
                    }
                    
                    if (teamData && !teamData.base && teamData.data) {
                        teamData = typeof teamData.data === "string" ? JSON.parse(teamData.data) : teamData.data;
                    }
                    teamData = teamData || teamDoc;

                    // 🌟 [로그 추가] 검색창 클릭 및 네비게이션 렌더링 시 Dexie에서 로드된 통계를 출력합니다.
                    console.log("\n=====================================");
                    console.log("[DEBUG-UI] Dexie에서 로드된 Team 데이터:", teamDoc);
                    console.log("[DEBUG-UI] 화면에 렌더링될 Base 통계:", JSON.stringify(teamData.base, null, 2));
                    console.log("=====================================\n");

                    if (teamData.base && teamData.base.pages) {
                        (currentSession as any).pages = teamData.base.pages;
                    }
                }
            } catch(e) { 
                console.warn("[Navigation] Failed to load local stats:", e); 
            }

            // 3. Render
            // 🌟 [OAUTH CLEANUP] renderAccordion 이전에도 잔존 노드를 제거합니다.
            pageList.querySelectorAll(".oauth-site-item").forEach((el: Element) => el.remove());
            pageList.innerHTML = await renderAccordion(tree);

            // 🌟 [OAUTH SITES] 등록 사이트 렌더링은 renderOAuthSitesUI() 로 분리했습니다.
            //    이 블록이 `else` 안에만 있어서, _pages 가 0건인 analytic 화면에서는
            //    조회도 렌더링도 통째로 건너뛰었습니다. 아래 if/else 가 끝난 뒤에
            //    분기와 무관하게 한 번 호출합니다.

            // 🌟 [추가] Show/Hide 토글 버튼 이벤트 바인딩
            pageList.querySelectorAll(".btn-toggle-visibility").forEach((btn: any) => {
                btn.onclick = async (e: Event) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const targetId = btn.dataset.id;
                    if (!targetId) return;

                    if (hiddenPages.includes(targetId)) {
                        hiddenPages = hiddenPages.filter(id => id !== targetId);
                    } else {
                        hiddenPages.push(targetId);
                    }
                    await kvSet("hidden_pages", JSON.stringify(hiddenPages));
                    await renderNavigation(); // UI 즉시 갱신
                };
            });

            // 🌟 [추가] 숨겨진 항목이 있는 Host(도메인)의 Show 버튼 노출 및 일괄 해제 이벤트 바인딩
            pageList.querySelectorAll(".logis-label").forEach((label: any) => {
                const id = label.dataset.id;
                const domain = label.dataset.domain;
                // 해당 도메인에 속한 아이템 중 하나라도 숨김(hidden) 상태라면 Host의 Show 버튼을 노출합니다.
                if (hiddenPages.includes(id)) {
                    const hostShowBtn = pageList.querySelector(`.btn-show-domain-hidden[data-domain="${domain}"]`) as HTMLElement;
                    if (hostShowBtn) hostShowBtn.style.display = "inline";
                }
            });

            pageList.querySelectorAll(".btn-show-domain-hidden").forEach((btn: any) => {
                btn.onclick = async (e: Event) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const domain = btn.dataset.domain;
                    
                    // 🌟 해당 도메인을 가진 모든 라벨을 찾아 hiddenPages 배열에서 전부 제거합니다.
                    pageList.querySelectorAll(`.logis-label[data-domain="${domain}"]`).forEach((label: any) => {
                        const id = label.dataset.id;
                        if (hiddenPages.includes(id)) {
                            hiddenPages = hiddenPages.filter(hId => hId !== id);
                        }
                    });
                    
                    await kvSet("hidden_pages", JSON.stringify(hiddenPages));
                    await renderNavigation(); // UI 즉시 갱신하여 숨겨졌던 모든 항목을 표시
                };
            });

            // 4. Bind Clicks manually to labels
            pageList.querySelectorAll(".logis-label").forEach((label: any) => {
                label.onclick = async (e: Event) => {
                    const ds = label.dataset;
                    if (!ds.id) return;

                    // 🌟 [기획 반영] 초대 패널이 열려있는지 확인
                    const inviteContainer = document.getElementById("nav-cloud-invite-container");
                    const isInviteMode = inviteContainer && !inviteContainer.classList.contains("hidden");

                    if (isInviteMode) {
                        // A. 초대 모드: 클래스 토글 및 이벤트 중단
                        e.preventDefault();
                        e.stopPropagation();
                        label.classList.toggle("selected");
                        console.log(`[INVITE-MODE] Page ${ds.id} selection toggled:`, label.classList.contains("selected"));
                        return; // 필터링 로직 실행 방지
                    }

                    // B. 일반 모드: 기존 필터링 및 내비게이션 로직
                    // 1. 컨텍스트 업데이트
                    activeContext.cc = ds.cc || "";
                    activeContext.bcc = ds.bcc || "";
                    activeContext.ref = ds.ref || "";
                    
                    activeTags = activeTags.filter(t => t.type !== 'type' && t.type !== 'domain' && t.type !== 'path');
                    
                    // 2. 검색 태그 추가
                    addSearchTag(`@${ds.domain}`, 'domain', ds.domain);
                    addSearchTag(`#${ds.type}`, 'type', ds.type);
                    updateTagsUI();
                    
                    // 3. 버튼(#btn-extract) 강제 업데이트 호출
                    await updateExtractButtonVisibility();

                    // 4. UI 갱신 및 닫기
                    fetchChatHistory(true);
                    hideNavigation();
                };
            });
        }

        // 🌟 [OAUTH SITES] _pages 가 0건이든 아니든 analytic 모드에서는 항상 렌더링합니다.
        //    (analytic 화면은 currentDomain 필터 때문에 _pages 가 비는 경우가 대부분입니다)
        await renderOAuthSitesUI(pageList);

        // Users rendering (simplified parity)
        const localUserList = document.getElementById("nav-list-local-users");
        const usersRaw = await Select["users"]({});
        
        // 🌟 [CRITICAL FIX] 백엔드(LanceDB)에서 가져온 유저 데이터 역시 json_data를 파싱해 주어야 Local/Cloud 분류가 정상 작동합니다!
        const users = usersRaw.map(u => {
            if (!u.data && u.json_data && typeof u.json_data === "string") {
                try { u.data = JSON.parse(u.json_data); } catch(e) {}
            }
            return u;
        });
        
        if (userList) userList.innerHTML = "";
        if (localUserList) localUserList.innerHTML = `<div class="empty">No local Members/Devices</div>`;

        if (users.length > 0) {
            // 1. 꼬리표를 기준으로 로컬/클라우드 유저 분할
            // 🌟 [BOOL PARITY] canonicalizeData 의 BOOL_KEYS 가 is_device 를 0|1 정수로 확정합니다.
            //    (IndexedDB 는 boolean 을 유효한 키로 인정하지 않기 때문입니다)
            //    따라서 === true 비교는 항상 false 가 되어 로컬 디바이스가 전부
            //    Cloud Members 로 새어 나갔습니다. truthy 판정으로 통일합니다.
            const isDevice = (u: any) => {
                const v = u?.data?.is_device;
                return v === 1 || v === true || v === "1" || v === "true";
            };
            const localUsers = users.filter(u => isDevice(u));
            const cloudUsers = users.filter(u => !isDevice(u));

            // 2. Cloud Team Members 렌더링 (중복 Row 제거 로직 추가)
            if (cloudUsers.length > 0 && userList) {
                // 🌟 [CRITICAL FIX] bb.ts의 유저 트리(Tree) 조립 로직을 완벽히 복원하여 팀과 멤버의 구조를 맞춥니다.
                const tempUsers: Record<string, any> = {};
                const treeUsers: any[] = [];

                for (let u = 0; u < cloudUsers.length; u++) {
                    let user = cloudUsers[u];
                    tempUsers[user.id] = { ...user, children: [] };
                }

                for (let key in tempUsers) {
                    if (tempUsers.hasOwnProperty(key)) {
                        let user = tempUsers[key];
                        let parentId = user.to;

                        // 클라우드 동기화 과정에서 member 타입으로도 내려올 수 있으므로 포괄 처리
                        if (user.type === "user" || user.type === "member") { 
                            if (tempUsers[parentId]) {
                                tempUsers[parentId].children.push(tempUsers[key]);
                            } else {
                                treeUsers.push(tempUsers[key]);
                            }
                        } else if (user.type === "team") {
                            treeUsers.push(tempUsers[key]);
                        }
                    }
                }
                
                userList.innerHTML = await renderAccordion(treeUsers);

                // 🌟 [수정] 방장(Owner)인 경우에만 ADD 버튼 노출 (폼은 HTML에 정적으로 존재)
                const myTeam = cloudUsers.find(u => u.type === "team" && u.from === currentSession.address && u.id === u.to);
                const btnCloudInvite = document.getElementById("btn-cloud-invite-toggle");
                
                if (myTeam) {
                    if (btnCloudInvite) btnCloudInvite.style.display = "inline-block";
                } else {
                    if (btnCloudInvite) btnCloudInvite.style.display = "none";
                }

                // 🌟 [추가] 멤버 삭제 및 초대 취소 이벤트 위임 (Event Delegation)
                userList.onclick = async (e: Event) => {
                    const target = e.target as HTMLElement;
                    const cancelBtn = target.closest('.btn-cancel-member') as HTMLElement;
                    if (!cancelBtn) return;

                    // 라벨의 기본 클릭 이벤트(검색 컨텍스트 전환) 방지
                    e.preventDefault();
                    e.stopPropagation();

                    const targetId = cancelBtn.dataset.id;
                    const targetName = cancelBtn.dataset.name;

                    // Tauri 네이브 ask 팝업으로 확인
                    const confirmed = await ask(`정말 '${targetName}' 멤버를 삭제하거나 초대를 취소하시겠습니까?`, { 
                        title: "멤버 삭제 확인", 
                        kind: "warning" 
                    });

                    if (confirmed && targetId) {
                        try {
                            // 로컬 및 클라우드(동기화 시)에서 데이터 삭제
                            await invoke("delete_document", { uuid: targetId });
                            console.log(`[AUTH] Member/Invite removed: ${targetId}`);
                            
                            // UI 즉시 새로고침
                            await renderNavigation();
                        } catch (err) {
                            console.error("Failed to remove member:", err);
                        }
                    }
                };
            }

            // 3. Local Devices 렌더링 (단일 리스트 구조)
            if (localUsers.length > 0 && localUserList) {
                // 로컬 기기는 자식(children)이 없는 플랫한 노드로 렌더링합니다.
                const localNodes = localUsers.map(u => ({ ...u, children: [] }));
                localUserList.innerHTML = await renderAccordion(localNodes);
            }
        }

    } catch (e) { 
        console.error("Nav render error:", e); 
    } finally {
        // [FIX] Navigation rendered (or failed), stop spinner if it was the first time
        if (isFirstNavRender) {
            isFirstNavRender = false;
            stopSpinner();
        }
        // 🌟 [CRITICAL FIX v3] 네비게이션 렌더링 완료 후 DOM을 참조하는 버튼 가시성 로직을 강제 재평가하여 버튼을 복구합니다.
        //    단, activeContext.ref가 아직 비어있다면(매칭 실패 또는 레이스 컨디션),
        //    caller(browser-match-found 리스너)가 renderNavigation() 직후 별도로 호출하므로 여기서는 스킵합니다.
        if (activeContext.ref) {
            await updateExtractButtonVisibility();
        }
    }
}

// --- Invite Logic ---
async function handleTeamInvite() {
    const emailInput = document.getElementById("invite-email-input") as HTMLInputElement;
    const email = emailInput?.value.trim();

    // 🌟 이메일 형식 검증을 위한 정규식 추가
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

    if (!email || !emailRegex.test(email)) {
        alert("Please enter a valid email address (e.g., user@example.com).");
        if (emailInput) {
            emailInput.focus();
            emailInput.style.outline = "2px solid #ef4444";
        }
        return;
    }

    // 검증 성공 시 스타일 초기화
    if (emailInput) emailInput.style.outline = "none";

    const btn = document.getElementById("btn-send-invite") as HTMLButtonElement;
    const originalText = btn.innerText;
    btn.innerText = "Wait...";
    btn.disabled = true;

    try {
        const origin = "https://commerce.logis.center";
        const now = Date.now();
        const createdAt = now - timezoneOffset;

        // 🌟 [추가] Pages 영역에서 selected 클래스가 붙은 모든 라벨의 data-id 수집
        const selectedPages: string[] = [];
        const pageList = document.getElementById("nav-list-pages");
        if (pageList) {
            pageList.querySelectorAll(".logis-label.selected").forEach((label: any) => {
                if (label.dataset.id) selectedPages.push(label.dataset.id);
            });
        }
        
        let targetHref = currentDetectedUrl || "https://commerce.logis.center/tracking";
        if (targetHref.includes("localhost") || targetHref.includes("127.0.0.1") || targetHref === "about:blank") {
            targetHref = "https://commerce.logis.center/tracking";
        }

        const params = new URLSearchParams({
            origin: origin,
            created_at: createdAt.toString(),
            hash: currentSession.hash,
            token: currentSession.token || "",
            href: targetHref,
            from: currentSession.team || "",
            to: currentSession.address || "",
            email: email,
            // 🌟 수집된 페이지 ID 배열을 JSON 문자열로 변환하여 전달
            ref: JSON.stringify(selectedPages)
        });
        
        const url = `${API_HOST}/?${params.toString()}`;
        
        const response = await invoke<any>("proxy_fetch", {
            url: url,
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            session_params: { hash: currentSession.hash, token: currentSession.token }
        });

        // 서버 응답에서 hook URL을 가져오거나, 기본 mailto 훅을 사용
        let hookUrl = `${currentSession.hash}.logis.center@oauth.email`;
        if (response.results && response.results.length > 0) {
            const invite = response.results[0];
            if (invite.hook) hookUrl = invite.hook;
        }

        showInviteQr(hookUrl, email);
        emailInput.value = "";

        // 🌟 [추가] 클라우드 멤버 리스트에 '대기 중(Pending)' 상태로 즉시 렌더링되도록 로컬 DB에 임시 주입합니다.
        try {
            const pendingMember = {
                id: `pending_invite_${Date.now()}`,
                type: "user", // users 테이블로 분류되어 아코디언 메뉴에 들어갑니다.
                name: `${email.split('@')[0]} (Pending ⏳)`,
                from: currentSession.address || "0x0000000000000000000000000000000000000000",
                to: currentSession.team || "0x0000000000000000000000000000000000000000",
                cc: currentSession.team || "",
                data: { origin: "cloud", is_pending: true, email: email }
            };
            
            await invoke("upsert_items", { items: [pendingMember] });
            await renderNavigation(); // UI 즉시 갱신
        } catch (err) {
            console.warn("[INVITE] Failed to add pending member to UI:", err);
        }

    } catch (e) {
        console.error("[INVITE] Failed:", e);
        alert("Error sending invite.");
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
}

function showInviteQr(hook: string, email: string) {
    if (!chatTalks) return;
    
    // 기존 열려있던 네비게이션 숨기고 세팅(채팅) 탭 열기
    hideNavigation();
    openWidget("settings");

    const existing = document.getElementById("msg-invite-qr");
    if (existing) existing.remove();
    
    const mailtoLink = `mailto:${encodeURIComponent(hook)}`;

    const html = `
        <div class="chat-talk system" id="msg-invite-qr" data-created-at="9999999999999">
            <div class="chat-message" style="padding:15px; background: #fff; color: #000; border:0; border-radius: 8px; text-align: center;">
                <div style="font-size:0.8rem; font-weight: bold; margin-bottom: 10px; color: #333;">
                    Invite <span style="color:var(--primary);">${email}</span>
                </div>
                <div style="font-size:0.65rem; color: #666; margin-bottom: 15px; line-height: 1.4;">
                    Scan this QR code with mobile camera<br>to send an invitation email.
                </div>
                <div id="invite-qr-target" style="display: inline-block; background: #fff; padding: 10px; border-radius: 8px; border: 1px solid #eee;"></div>
                <div style="margin-top: 15px;">
                    <a href="${mailtoLink}" style="display: inline-block; padding: 8px 16px; background: var(--primary); color: #000; text-decoration: none; border-radius: 4px; font-weight: bold; font-size: 0.7rem;">Open Mail App</a>
                </div>
            </div>
        </div>`;
        
    chatTalks.insertAdjacentHTML('beforeend', html);
    
    const qrTarget = document.getElementById("invite-qr-target");
    if (qrTarget) {
        qrTarget.innerHTML = "";
        new (window as any).QRCode(qrTarget, { 
            text: mailtoLink, 
            width: 300, 
            height: 300, 
            colorDark: "#000000", 
            colorLight: "#ffffff", 
            correctLevel: (window as any).QRCode.CorrectLevel.M 
        });
        const scroll = document.getElementById("chat-scroll");
        if (scroll) scroll.scrollTop = scroll.scrollHeight;
    }
}

// --- Sync Logic ---
// main.ts 내부
async function syncData() {
    // 🌟 [ANALYTICS TRACK] analytic 모드는 console.logis.center Client Worker 와 동기화합니다.
    if (currentSearchMode === "analytic") {
        await syncAnalyticsData();
        if (currentSession.hash && currentSession.email) {
            syncCommerceInBackground();
        }
        return;
    }
    // 🌟 [ANALYTICS BACKGROUND] commerce / shipping 탭에 있어도 analytics 이벤트를 계속 받습니다.
    //    ── 왜 필요한가 ──
    //     기존 구조는 analytic 탭을 열어야만 D1 이벤트를 가져왔습니다.
    //     그 사이 쌓인 이벤트는 탭을 열기 전까지 로컬에 존재하지 않았고,
    //     Worker 의 LIMIT 1000 창 밖으로 밀려나면 영구히 유실됩니다.
    //     30초 스로틀이 걸려 있어 폴링 부하는 사실상 없습니다.
    if (currentSession.hash) {
        syncAnalyticsInBackground();
    }
    if (!currentSession.hash || !currentSession.email) return;
    
    console.log("[SYNC] 1. 서버에 최신 데이터 요청 중...");
    try {
        const origin = "https://commerce.logis.center";
        const now = Date.now();
        const createdAt = now - timezoneOffset;
        
        // 🌟 [CRITICAL FIX] front.js 패리티: 서버가 나를 정확히 인지하도록 cc, type, 실제 href 파라미터를 추가합니다.
        let targetHref = currentDetectedUrl || "https://commerce.logis.center/tracking";
        if (targetHref.includes("localhost") || targetHref.includes("127.0.0.1") || targetHref === "about:blank") {
            targetHref = "https://commerce.logis.center/tracking";
        }

        const queryParams: any = {
            origin: origin,
            created_at: createdAt.toString(),
            hash: currentSession.hash,
            token: currentSession.token || "",
            href: targetHref
        };

        // 🌟 [SENDER IMPRINT] checkAuthStatus 와 동일한 이유입니다.
        //    syncData 는 currentSession.email 이 확정된 뒤에만 실행되므로
        //    (함수 상단의 `if (!currentSession.hash || !currentSession.email) return;`)
        //    여기서 보내는 sender 는 반드시 서버 user.data 에 각인됩니다.
        //    이 값이 있어야 PUT(talks) / POST(tasks) 가 cookies.sender 게이트를 통과합니다.
        const syncSender = currentSession.email || currentSession.name || "";
        if (syncSender) queryParams.sender = syncSender;

        // 🌟 [CRITICAL FIX] chrome.js 패리티: 서버 동기화 시, 강제 지정된 사이드바 메뉴가 없다면 현재 URL의 도메인(CC)을 최우선으로 서버에 전달합니다.
        let syncEffectiveCc = activeContext.cc;
        const isDefaultForced = activeTags.some(t => t.value === "logis.center" && t.type === "domain");
        if (!syncEffectiveCc || (!isDefaultForced && activeTags.length === 0)) {
            try {
                const urlObj = new URL(targetHref.toLowerCase());
                const rootDomain = getRootDomain(urlObj.hostname);
                syncEffectiveCc = await hashId(rootDomain);
            } catch(e) {}
        }

        if (syncEffectiveCc) queryParams.cc = syncEffectiveCc;
        if (currentSearchMode && currentSearchMode !== "commerce") queryParams.type = currentSearchMode;

        const params = new URLSearchParams(queryParams);
        const url = `${API_HOST}/?${params.toString()}`;
        
        // 1. 서버 요청
        const response = await invoke<any>("proxy_fetch", {
            url: url,
            method: "GET",
            headers: { "Content-Type": "application/json" },
            // 🌟 서버가 cc를 헤더나 쿠키처럼 파싱할 수 있게 proxy_fetch 파라미터에도 주입합니다.
            session_params: { hash: currentSession.hash, token: currentSession.token, cc: activeContext.cc || "" }
        });

        stepQrSpinner();

        console.log('response',response);

        if (response.results && Array.isArray(response.results)) {
            // 🌟 [추가] 서버에서 압축된 gzip 데이터를 그대로 보낼 경우, chrome.js와 동일하게 pako로 압축 해제
            try {
                const pako = (window as any).pako;
                for (let i = 0; i < response.results.length; i++) {
                    let item = response.results[i];
                    if (item.data && typeof item.data === 'object' && !item.data.text && !item.data.title) {
                        let arrData = item.data.data || item.data;
                        let arr: Uint8Array | null = null;
                        
                        if (Array.isArray(arrData)) {
                            arr = new Uint8Array(arrData);
                        } else if (arrData.buffer) {
                            arr = new Uint8Array(arrData.buffer);
                        } else if (Object.keys(arrData).length > 0 && !isNaN(Number(Object.keys(arrData)[0]))) {
                            arr = new Uint8Array(Object.values(arrData) as number[]);
                        }

                        if (arr) {
                            try {
                                if (pako) {
                                    const decompressed = pako.ungzip(arr, { to: 'string' });
                                    item.data = JSON.parse(decompressed);
                                } else {
                                    const decompressed = new TextDecoder('utf-8').decode(arr);
                                    item.data = JSON.parse(decompressed);
                                }
                            } catch (e) {
                                try {
                                    const decompressed = new TextDecoder('utf-8').decode(arr);
                                    item.data = JSON.parse(decompressed);
                                } catch (err) {}
                            }
                        }
                    }
                }
            } catch (err) {
                console.warn("[SYNC] pako decompression failed:", err);
            }

            // 🌟 [버그 수정] DOM에 렌더링된 요소뿐만 아니라, Dexie DB에 있는 백그라운드 통계(team) 객체의 최신 시간도 대조해야 합니다.
            const localUsers = await Select["users"]({});
            const localPages = await Select["pages"]({});
            const localTalks = await appDb.table("talks").toArray(); // 🌟 채팅(talks) 데이터도 로컬 맵에 반드시 포함
            const localMap = new Map();
            [...localUsers, ...localPages, ...localTalks].forEach((item: any) => {
                // 🌟 updated_at이 0인 레거시 데이터를 대비해 created_at을 백업으로 사용
                localMap.set(item.id, item.updated_at_ts || item.updated_at || item.created_at_ts || item.created_at || 0);
            });

            // 🌟 [TOMBSTONE GUARD] 내가 삭제한 메시지는 서버에 행이 남아 있으므로
            //    매 폴링마다 다시 내려옵니다. 그때 로컬에는 DOM 도 Dexie 도 없으니
            //    바로 아래 `!existingEl && !localMap.has(id)` 조건을 통과해 재삽입됩니다.
            //    묘비 조회는 메모리 Set 이라 폴링 비용이 사실상 0 입니다.
            const tombstones = await loadTalkTombstones();
            // 🌟 [ITEM TOMBSTONE] 문서(items) 삭제도 동일하게 서버 재삽입을 차단합니다.
            const itemTombs = await loadItemTombstones();
            let tombBlocked = 0;
            let itemTombBlocked = 0;
            const filteredResults = response.results.filter((newItem: any) => {
                // 🌟 삭제 의사가 기록된 id 는 어떤 조건보다 먼저, 무조건 차단합니다.
                if (tombstones.has(String(newItem.id))) {
                    tombBlocked++;
                    return false;
                }
                // 🌟 [ITEM TOMBSTONE CHECK] talk 이 아닌 일반 문서도 차단합니다.
                if (itemTombs.has(String(newItem.id))) {
                    itemTombBlocked++;
                    return false;
                }
                const existingEl = document.getElementById(newItem.id);
                // 🌟 완전 신규 데이터 (DOM에도 없고 로컬 DB 캐시에도 없는 경우)는 조건 없이 즉시 통과
                if (!existingEl && !localMap.has(newItem.id)) {
                    return true;
                }
                let localUpdated = 0;
                if (existingEl) {
                    localUpdated = parseInt(existingEl.dataset.updatedAt || existingEl.dataset.createdAt || "0");
                } else if (localMap.has(newItem.id)) {
                    localUpdated = parseInt(localMap.get(newItem.id) || "0");
                }
                // 🌟 서버의 updated_at이 0일 수 있으므로(기존 index.ts 특성), created_at을 백업 비교값으로 활용
                const serverUpdated = newItem.updated_at || newItem.created_at || 0;
                return serverUpdated > localUpdated; // 서버 데이터가 더 최신인 경우만 포함
            });
            if (tombBlocked > 0) {
                console.log(`[TOMBSTONE] 🪦 삭제된 메시지 ${tombBlocked}건의 서버 재삽입을 차단했습니다.`);
            }
            if (itemTombBlocked > 0) {
                console.log(`[ITEM-TOMBSTONE] 🪦 삭제된 문서 ${itemTombBlocked}건의 서버 재삽입을 차단했습니다.`);
            }

            if (filteredResults.length > 0) {
                updateSyncBackoff(true);
                // 🌟 [MODE TAGGING v2] 클라우드 D1 에는 mode 컬럼이 없습니다.
                //    mode 는 '동기화 시점에 클라이언트가 확정하는 값' 이며,
                //    판정은 파일 상단의 modeOfType() 단일 함수가 전담합니다.
                //
                //    ── v1 의 결함 ──
                //     TRADING_TYPES 에 소문자 'tracking' 이 들어 있어
                //     proxy 의 Relay("tracking","order") 가 만든 commerce 배송추적 문서가
                //     전부 mode='shipping' 으로 오염되었고,
                //     loadMoreDocs 의 mode 필터에서 commerce 목록이 통째로 비었습니다.
                for (const r of filteredResults) {
                    if (!r) continue;
                    if (r.mode) continue;
                    r.mode = modeOfType(String(r.type || ""));
                }
                // 🌟 [DRAFT PRESERVE] 서버 행에 updated_at 이 없으면
                //    Rust upsert_item 이 현재 시각을 부여해 draft → count 가 됩니다.
                //    data 내부에도 없으면 여기서 0 을 명시해 draft 계약을 보존합니다.
                for (const r of filteredResults) {
                    if (!r) continue;
                    const hasRoot = r.updated_at !== undefined && r.updated_at !== null;
                    const inner = (r.data && typeof r.data === 'object') ? r.data.updated_at : undefined;
                    const hasInner = inner !== undefined && inner !== null;
                    if (!hasRoot && !hasInner) {
                        r.updated_at = 0;
                    }
                }

                // 🌟 [FLAG RECOVERY] commerce D1 의 items 테이블에는 flag 컬럼이 없습니다.
                //    (analytics D1 · LanceDB · Dexie 에는 모두 존재)
                //    그대로 두면 commerce 트랙 전 문서의 flag 가 영구 공백이 되어
                //    지역 스코프 조회가 성립하지 않습니다.
                //    세션 flag(= Client Worker 가 GeoIP 로 확정한 국가 코드)로 보강합니다.
                const sessionFlag = String((currentSession as any).flag || "");
                if (sessionFlag) {
                    let flagFilled = 0;
                    for (const r of filteredResults) {
                        if (!r) continue;
                        if (r.flag) continue;
                        // data 안에 이미 flag 가 있으면(users/team 문서) 그쪽을 신뢰합니다.
                        const inner = (r.data && typeof r.data === 'object') ? r.data.flag : undefined;
                        if (inner) { r.flag = inner; continue; }
                        r.flag = sessionFlag;
                        flagFilled++;
                    }
                    if (flagFilled > 0) {
                        console.log(`[SYNC] flag 컬럼 부재 문서 ${flagFilled}건을 세션 flag('${sessionFlag}')로 보강했습니다.`);
                    }
                }

                // 🌟 [ROOT ABSORB / RUST] Rust 의 upsert_items 는 item.data 객체의 알맹이를
                //    루트로 끌어올리는 방향만 처리합니다. 반대로 '루트에만 있는 물리 컬럼' 은
                //    data 로 내려 주지 않으므로, 여기서 미리 합쳐 보냅니다.
                //    (normalizeEnvelope 의 ROOT ABSORB 와 동일 규칙이어야 두 저장소가 일치합니다)
                for (const r of filteredResults) {
                    if (!r) continue;
                    if (!r.data || typeof r.data !== 'object') continue;
                    for (const k in r) {
                        if (!Object.prototype.hasOwnProperty.call(r, k)) continue;
                        if (ENVELOPE_ROOT_KEYS.has(k)) continue;
                        const v = r[k];
                        if (v === undefined || v === null) continue;
                        if (typeof v === 'function') continue;
                        if (r.data[k] === undefined || r.data[k] === null || r.data[k] === "") {
                            r.data[k] = v;
                        }
                    }
                }

                console.log(`[SYNC] 2. 로컬 LanceDB 최신화 중... (${filteredResults.length} / ${response.results.length} 건 변경됨)`);
                await invoke("upsert_items", { items: filteredResults });
                
                // 🌟 [누락 복구] 서버 통계를 LanceDB에 덮어썼다면, 반드시 프론트엔드 Dexie DB 에도 동기화해줘야 화면이 바뀝니다!
                const newUsers = filteredResults.filter((r: any) => r.type === "team" || r.type === "user" || r.type === "member");
                
                // 🌟 [ROUTING FIX] 서버(Client Worker)는 페이지 캐시 행에 table:'pages' 를 실어 보냅니다.
                //    기존처럼 data.node / data.item 의 '존재 여부' 로 추정하면
                //    canonicalize_data 가 node / detail 을 0 으로 시딩하는 순간 전 아이템이 참이 되어
                //    일반 상품/주문까지 Dexie pages 테이블로 오분류됩니다.
                //    따라서 서버가 명시한 table 을 1순위로 신뢰하고,
                //    없을 때만 셀렉터 마커의 '값이 truthy 인지' 를 봅니다.
                const newPages = filteredResults.filter((r: any) => {
                    if (r.table === "pages" || r.table === "page") return true;
                    if (r.type === "pages" || r.type === "page") return true;
                    const d = typeof r.data === 'string' ? JSON.parse(r.data) : (r.data || r);
                    return !!d.node || !!d.item;
                });
                
                // chrome.js 형태의 talks(채팅) 메시지 추출
                const newTalks = filteredResults.filter((r: any) => r.type === "talk" || r.table === "talks");

                const newItems = filteredResults.filter((r: any) => !newUsers.includes(r) && !newPages.includes(r) && !newTalks.includes(r));

                // 🌟 v4 : users / pages 도 봉투 정규화를 거칩니다.
                //  talks 는 스키마가 다르므로(role/task_id/status) 그대로 넣습니다.
                if (newUsers.length > 0) await appDb.table("users").bulkPut(normalizeEnvelope(newUsers));
                if (newPages.length > 0) await appDb.table("pages").bulkPut(normalizeEnvelope(newPages));
                if (newTalks.length > 0) await appDb.table("talks").bulkPut(newTalks);
                if (newItems.length > 0) await appDb.table("items").bulkPut(normalizeEnvelope(newItems));

                // 🌟 [CRITICAL FIX] 서버에서 가져온 데이터는 이미 윗줄에서 invoke("upsert_items")를 통해 Rust(LanceDB)에 
                // 일괄 저장되었습니다. 프론트엔드가 이를 다시 백엔드로 밀어넣는 병목 루프를 삭제합니다.
            } else {
                updateSyncBackoff(false);
                
                console.log(`[SYNC] 2. 변경된 데이터가 없어 DB 쓰기를 건너뜁니다.`);
            }

            // 🌟 [추가] '대기 중' 멤버 정화(Cleanup) 로직
            // 서버에서 받은 결과 중 정식 멤버(member/user)가 있는지 확인합니다.
            const realMembers = response.results.filter((item: any) => item.type === "member" || item.type === "user");
            if (realMembers.length > 0) {
                const localUsers = await Select["users"]({});
                // 로컬에 저장된 'pending_invite_'로 시작하는 가짜 데이터들을 찾습니다.
                const pendingInvites = localUsers.filter(u => u.id && u.id.startsWith("pending_invite_"));

                for (const pending of pendingInvites) {
                    const pendingEmail = pending.data?.email;
                    // 서버에서 온 정식 멤버 중 이메일(혹은 이름)이 일치하는 사람이 있는지 대조
                    const isNowMember = realMembers.some((m: any) => {
                        // 서버 데이터(m) 내부에 이메일 정보가 있거나, 이름이 이메일 아이디와 같은지 확인
                        return m.to === pending.from || (m.data && m.data.email === pendingEmail);
                    });

                    if (isNowMember) {
                        // 정식 멤버가 확인되었으므로 가짜(Pending) 데이터를 로컬 DB에서 삭제합니다.
                        await invoke("delete_document", { uuid: pending.id });
                        console.log(`[SYNC] Pending invite for ${pendingEmail} is now a real member. Placeholder removed.`);
                    }
                }
            }
            
            // 🌟 [CLOUD TASK LIFECYCLE] 서버 tasks 목록과 대조하여 클라우드 작업 완료 여부를 판정합니다.
            if (cloudPendingTasks.size > 0) {
                const serverTaskIds = new Set<string>();
                for (const r of response.results) {
                    if (!r) continue;
                    if (r.table === "tasks" && r.id) {
                        serverTaskIds.add(r.id);
                    }
                }

                for (const [localTid, meta] of Array.from(cloudPendingTasks.entries())) {
                    const stillRunning = meta.serverId ? serverTaskIds.has(meta.serverId) : false;

                    if (stillRunning) {
                        await renderProgressToUI({
                            task_id: localTid,
                            category: "Cloud Queue",
                            summary: "Processing on Logis Center...",
                            spinner: "☁️"
                        });
                        continue;
                    }

                    // 서버 등록 직후의 레이스 컨디션 방어(최소 5초 유예)
                    if (Date.now() - meta.createdAt < 5000) continue;

                    cloudPendingTasks.delete(localTid);

                    await renderProgressToUI({
                        task_id: localTid,
                        category: "Done",
                        summary: meta.kind === "search"
                            ? "Cloud AI search complete."
                            : "Cloud AI extraction complete.",
                        spinner: "✅"
                    });

                    console.log(`[CLOUD] Task ${localTid} (server: ${meta.serverId}) finished on Logis Center.`);
                }
            }

            console.log("[SYNC] 3. 로컬 DB에서 데이터 불러와 메뉴 렌더링...");
            // 🌟 [RENDER GATE] 변경된 데이터가 없으면 네비게이션/리스트 재렌더링을 건너뜁니다.
            //    기존에는 매 폴링마다 renderNavigation + loadMoreDocs 가 돌아
            //    동일 데이터를 반복 적재하고 콘솔 로그를 낭비했습니다.
            if (filteredResults.length > 0) {
                // 3. LanceDB 불러오기
                await renderNavigation();

                // 🌟 [CRITICAL FIX] 서버 데이터를 로컬 DB에 밀어넣었으니, 현재 보고 있는 탭에 맞춰 UI를 갱신합니다!
                if (currentTab === "list") {
                    await loadMoreDocs(false, true);
                } else if (currentTab === "settings") {
                    await fetchChatHistory(false, true);
                }
            } else {
                console.log("[SYNC] 3. 변경 없음 → 네비게이션/리스트 재렌더링을 건너뜁니다.");
            }

            // 🌟 [CLIENT-SIDE EMBEDDING] 클라우드는 구조화만 했으므로 임베딩은 여기서 로컬로 수행합니다.
            //    runLocalEmbeddingSync 내부의 2초 디바운스가 initSession의 4초 타이머와
            //    겹치는 중복 호출을 자동으로 병합하여 1회만 실행합니다.
            console.log("[SYNC] 4. 로컬 임베딩 파이프라인 스케줄링 (2초 디바운스 적용)...");
            runLocalEmbeddingSync();
        }
        
    } catch (e) {
        console.error("[SYNC] 동기화 실패:", e);
    } finally {
        if (!isExtracting && !isSearching) stopSpinner();
        // 🌟 [SYNC DONE → SUBMIT RESTORE] 동기화가 끝나면 검색 입력 상태를 재평가합니다.
        //    기존에는 stopSpinner() 내부에서만 조건부로 노출했지만,
        //    syncData 가 백그라운드에서 돌 때 btnSubmit 이 숨겨진 채
        //    복귀하지 않는 경로가 있었습니다.
        if (btnSubmit) {
            const currentVal = searchInput?.value.trim() || "";
            if (currentVal !== "" && !isQueryActive(currentVal)) {
                btnSubmit.style.display = "flex";
            } else {
                btnSubmit.style.display = "none";
            }
        }
    }
}

// --- 기존 State 영역 어딘가에 추가 ---
let currentSearchMode = "commerce"; // 값은 initSession에서 Dexie 비동기로 덮어씌워집니다.

// 🌟 앱 시작 시 탭 UI 초기화 함수
function applySearchModeUI() {
    document.querySelectorAll('.mode-tab').forEach(btn => {
        const el = btn as HTMLElement;
        if (el.dataset.mode === currentSearchMode) {
            el.style.color = "#000";
            el.style.fontWeight = "bold";
            el.classList.add('active');
        } else {
            el.style.color = "#999";
            el.style.fontWeight = "bold";
            el.classList.remove('active');
        }
    });
    // 🌟 [MODE LABEL] 파일 상단의 modeLabel() 단일 정의를 씁니다.
    //    저장/쿼리 계약은 여전히 mode='shipping' 이므로 DB 나 Rust 쪽 변경이 전혀 없습니다.
    if (searchInput) {
        searchInput.placeholder = `${modeLabel(currentSearchMode)} Search or Ask`;
    }
    // 🌟 [추가] Shipping 모드일 때 Pages 섹션 통째로 숨기기
    const pagesSection = document.getElementById("nav-list-pages")?.closest(".nav-section") as HTMLElement;
    const isSettingsOpen = (document.getElementById("settings-toggle") as HTMLInputElement)?.checked;
    if (pagesSection) {
        if (currentSearchMode === "shipping" || isSettingsOpen) {
            pagesSection.style.display = "none"; // Shipping이거나 세팅 패널이 열려있으면 숨김
        } else {
            pagesSection.style.display = "block"; // 🌟 명시적으로 block 처리하여 노출 보장
        }
    }

    // 🌟 [OAUTH REGISTER BUTTON GATE] analytic 모드가 아니면 사이트 등록 버튼을 제거합니다.
    //    renderNavigation() 이 호출되지 않는 탭 전환 경로(applySearchModeUI)에서
    //    이전 analytic 모드에서 생성된 버튼이 commerce/shipping 에 잔존하는 것을 차단합니다.
    const existingOAuthBtn = document.getElementById("btn-oauth-register");
    if (existingOAuthBtn && currentSearchMode !== "analytic") {
        existingOAuthBtn.remove();
    }

    // 🌟 [OAUTH SITE NODE CLEANUP] 모드 전환 시점에 페이지 목록에 남아 있는
    //    .oauth-site-item 노드를 즉시 제거합니다.
    //    기존에는 renderNavigation() 의 pageList.innerHTML 교체 시점까지
    //    노드가 잔존하다가, renderOAuthSitesUI 가 조기 탈락하면
    //    다음 renderAccordion 에서 삭제되어 '갑자기 사라짐' 이 발생했습니다.
    //    여기서 명시적으로 지우면 모드 전환 직후 화면이 깨끗하게 정리됩니다.
    if (currentSearchMode !== "analytic") {
        const pageListEl = document.getElementById("nav-list-pages");
        if (pageListEl) {
            pageListEl.querySelectorAll(".oauth-site-item").forEach((el: Element) => el.remove());
        }
    }

    // 🌟 [OAUTH REGISTER BUTTON RESTORE] analytic 모드로 재전환 시 버튼이 없으면 재생성합니다.
    //    모드 탭 클릭 경로는 applySearchModeUI() + refreshList() 만 호출하고
    //    renderNavigation() 을 호출하지 않으므로, 여기서 버튼을 복원해야 합니다.
    //    생성 조건은 renderNavigation() 내부와 완전히 동일합니다.
    //      · currentSearchMode === "analytic"
    //      · 세팅 패널이 닫혀 있어야 함 (!isSettingsOpen)
    //      · 로그인 상태 (currentSession.email)
    //    ⚠️ existingOAuthBtn 은 remove() 후에도 변수 참조가 남아 있으므로,
    //       반드시 document.getElementById() 로 DOM 존재 여부를 다시 확인합니다.
    if (currentSearchMode === "analytic" && !document.getElementById("btn-oauth-register") && !isSettingsOpen && currentSession.email) {
        if (pagesSection) {
            const h3 = pagesSection.querySelector("h3");
            const pageListEl = document.getElementById("nav-list-pages");
            const registerBtn = document.createElement("button");
            registerBtn.id = "btn-oauth-register";
            registerBtn.style.cssText = "position: absolute; left: 5em; top: 13px; border: 0px; padding: 0px; font-size: 0.8rem; cursor: pointer; text-align: center; text-decoration: underline; background: none;";
            registerBtn.textContent = "+ 사이트 등록 (Analytic)";
            registerBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                renderOAuthRegistrationForm();
            });
            if (h3 && h3.nextSibling) {
                pagesSection.insertBefore(registerBtn, h3.nextSibling);
            } else if (h3) {
                pagesSection.appendChild(registerBtn);
            } else if (pageListEl) {
                pagesSection.insertBefore(registerBtn, pageListEl);
            }
        }
    }

    // 🌟 [ANALYTIC LOGO HIDE] analytic 모드에서는 logo-section 영역을 숨깁니다.
    //    console.logis.center 기반이므로 상점 로고/브랜딩 영역이 의미가 없습니다.
    const logoSection = document.querySelector('.logo-section') as HTMLElement;
    if (logoSection) {
        logoSection.style.display = currentSearchMode === "analytic" ? "none" : "";
    }
}

// DOM 로드 후 이벤트 리스너 추가
document.querySelectorAll('.mode-tab').forEach(btn => {
    btn.addEventListener('click', async (e) => {
        const target = e.target as HTMLElement;
        const prevMode = currentSearchMode;
        currentSearchMode = target.dataset.mode || "commerce";

        // 🌟 [SCOPE RESET] 트랙마다 cc 네임스페이스가 다릅니다.
        //  ── 무엇이 문제였나 ──
        //   commerce 는 activeContext.cc = hashId(getRootDomain(host)) 즉 'cafe24.com' 해시이고,
        //   analytic 은 Worker 가 넣은 hashId(url.host) 즉 'abc.cafe24.com' 해시입니다.
        //   commerce 에서 페이지를 클릭한 뒤 analytic 으로 넘어오면
        //   loadMoreDocs 가 그 cc 를 드라이버 인덱스로 삼아 무조건 0건이 됩니다.
        //   (검색 태그도 같은 이유로 남아 있으면 안 됩니다)
        if (prevMode !== currentSearchMode) {
            activeContext = { cc: "", bcc: "", ref: "" };
            activeTags = [];
            updateTagsUI();
        }

        // 🌟 탭 클릭 시 상태 저장 및 UI 업데이트
        await kvSet("search_mode", currentSearchMode);
        applySearchModeUI();

        console.log(`[UI] Search mode changed to: ${currentSearchMode}. Refreshing list...`);
        // 🌟 [BACKOFF RESET] 사용자가 탭을 전환하면 즉시 폴링을 기본 간격으로 리셋합니다.
        syncConsecutiveNoChange = 0;
        syncCurrentIntervalMs = SYNC_BASE_INTERVAL_MS;
        await refreshList();

        await refreshList();

        // 🌟 [IMMEDIATE PULL] analytic 으로 전환했으면 폴링 주기를 기다리지 않고 즉시 1회 당겨옵니다.
        if (currentSearchMode === "analytic" && currentSession.hash) {
            lastAnalyticsSyncAt = 0; // 스로틀 해제
            syncAnalyticsData();
        }

        const _isSettingsOpen = (document.getElementById("settings-toggle") as HTMLInputElement)?.checked;
        if (currentSearchMode === "analytic" && currentSession.email && !_isSettingsOpen) {
            await renderNavigation();
        }
    });
});

// 파일이 로드될 때 즉시 UI 적용
applySearchModeUI();


// [NEW] Global Navigation Link Handler (from item2html)
document.addEventListener('nav-link', async (e: any) => {
    const targetLink = e.detail;
    console.log("[NAV] Internal Link Clicked:", targetLink);
    addSearchTag(targetLink, 'path', targetLink);
    openWidget("list");
    listView.style.display = "block";
    detailView.style.display = "none";
});

// 🌟 [추가] 현재 입력한 검색어가 이미 대기열(10)이나 진행 중(1)인지 확인하는 완벽한 헬퍼 함수
function isQueryActive(text: string): boolean {
    const query = text.trim();
    // 1. 프론트엔드 큐 배열 검사 (아직 UI에 안 그려진 찰나의 순간 방어)
    if (GlobalTaskManager.queue.some(q => q.type === "ai_search" && q.payload && q.payload.query === query)) return true;

    // 2. DOM 상태 검사 (현재 실행 중인 작업 및 대기열 포함)
    let active = false;
    const bubbles = document.querySelectorAll('.task-bubble');
    for (let i = 0; i < bubbles.length; i++) {
        const el = bubbles[i] as HTMLElement;
        const status = parseInt(el.dataset.status || "0");
        const taskId = el.id;

        // 🌟 [CRITICAL FIX] 이미 Cancel 버튼을 눌러 취소된 작업(블랙리스트)이라면 중복 검사에서 즉시 제외하여 
        // 새로운 검색어가 막히는 버그를 원천 차단합니다!
        if (GlobalTaskManager.cancelledTasks.has(taskId)) {
            continue;
        }

        // 🌟 상태가 1(Processing)이거나 10(Queued)일 때만 활성 상태로 간주
        if ((status === 1 || status === 10) && taskId.startsWith("search_")) {
            const queryEl = document.getElementById(`${taskId}_query`);
            if (queryEl) {
                const qText = queryEl.querySelector('.content')?.textContent || "";
                if (qText.trim() === query) {
                    active = true;
                    break;
                }
            }
        }
    }
    return active;
}

searchInput?.addEventListener("input", () => {
    // 🌟 [CRITICAL FIX] 입력값이 비어있지 않고, 현재 진행/대기 중인 검색어와 '다를 때만' 버튼을 노출합니다.
    if (btnSubmit) {
        const currentVal = searchInput.value.trim();
        if (currentVal !== "" && !isQueryActive(currentVal)) {
            btnSubmit.style.display = "flex";
        } else {
            btnSubmit.style.display = "none";
        }
    }

    // 🌟 [CRITICAL FIX] 추출 중(isExtracting)이거나 큐가 바쁠 때(GlobalTaskManager.isBusy)
    // 타이핑만으로 백그라운드 임베딩 로직이 몰래 실행되는 것을 원천 차단합니다!
    if (isSearching || isExtracting || GlobalTaskManager.isBusy) return; 
    if(searchDebounceTimer) clearTimeout(searchDebounceTimer);
    searchDebounceTimer = window.setTimeout(async () => {
        if (isSearching || isExtracting || GlobalTaskManager.isBusy) return; 
        await loadMoreDocs(true);
    }, 800);
});

// [신규] 검색창에서 엔터 키를 누르면 AI 검색(돋보기 버튼)을 강제로 실행하도록 연결
searchInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault(); 
        // 🌟 [CRITICAL FIX] isExtracting 검사를 삭제하여, 전처리 중에도 검색을 대기열에 넣을 수 있게 허용합니다!
        if (!isSearching) { 
            btnSubmit?.click(); 
        }
    }
});

// --- main.ts 소스 ---

btnSubmit?.addEventListener("click", async () => {
    const query = searchInput.value.trim();
    if (!query) return;

    // 🌟 [CRITICAL FIX] 다른 검색어가 진행 중이더라도 새로운 검색어를 큐에 추가할 수 있도록 허용하되,
    // 완전히 동일한 검색어가 이미 진행/대기 중일 때만 중복 실행을 방어합니다!
    if (isQueryActive(query)) {
        console.warn("[SEARCH] The exact same query is already in progress or queued.");
        return; 
    }

    if (searchDebounceTimer) {
        clearTimeout(searchDebounceTimer);
        searchDebounceTimer = null;
    }

    // 인풋창 비우기
    searchInput.value = "";

    // 🌟 [CRITICAL FIX] 검색 버튼 숨김 (번개 버튼은 독립적인 추출 대기열 노출 조건을 따르도록 강제 숨김 코드를 제거합니다)
    if (btnSubmit) btnSubmit.style.display = "none";

    // 🌟 [CRITICAL FIX] 버튼을 누른 직후, 전역 플래그를 참(true)으로 고정하여 탭 전환 등 기타 UI 이벤트에 의해 초기화되는 것을 즉시 차단합니다!
    isSearching = true;

    const taskId = `search_${Date.now()}`;
    const startTime = Date.now();
    
    // 🌟 [추가] 검색 진행 중 UI 변경 및 초기화
    const resultH3 = document.querySelector('.nav-section.search h3');
    if (resultH3) {
        resultH3.innerHTML = `searching<strong class="count" style="cursor:pointer; margin-left:10px; color:#ef4444;" id="cancel-search-btn">Cancel</strong>`;
        const cancelBtn = document.getElementById("cancel-search-btn");
        if (cancelBtn) {
            cancelBtn.addEventListener("click", async () => {
                const confirmed = await ask("정말 검색을 취소하시겠습니까?", { title: "Cancel Search", kind: "warning" });
                if (confirmed) {
                    const targetTaskId = activeTaskId || taskId;
                    if (targetTaskId) {
                        GlobalTaskManager.cancelledTasks.add(targetTaskId);

                        // 🌟 [CRITICAL FIX] 화면에 남아있는 취소된 태스크의 말풍선(DOM) 상태를 2(Stopped)로 변경하여
                        // isQueryActive 검사에서 중복 진행 중으로 오인받지 않도록 시각적/구조적으로 확실히 처리합니다!
                        const el = document.getElementById(targetTaskId);
                        if (el) {
                            el.dataset.status = "2";
                            const statusBar = el.querySelector('.status-bar');
                            if (statusBar) statusBar.innerHTML = `<span style="color:#ef4444;">❌ STOPPED</span>`;
                        }
                        const queryEl = document.getElementById(`${targetTaskId}_query`);
                        if (queryEl) queryEl.dataset.status = "2";
                        await kvRemove(`term_${targetTaskId}`);
                    }
                    activeTaskId = null;
                    GlobalTaskManager.isBusy = false;
                    GlobalTaskManager.currentTaskId = null;
                    GlobalTaskManager.currentTaskPayload = null;

                    isSearching = false; 
                    stopSpinner();
                    
                    if (btnSubmit) btnSubmit.style.display = "flex";
                    
                    try {
                        await invoke<string>("stop_current_extraction", { taskId: targetTaskId });
                        await GlobalTaskManager.release(targetTaskId, targetTaskId);
                    } catch (e) { 
                        console.error("Stop failed:", e); 
                    }
                    
                    if (resultH3) {
                        const count = document.querySelectorAll('#doc-list .logis-result').length;
                        resultH3.innerHTML = `Result <strong class="count">${count > 0 ? `(${count})` : ""}</strong>`;
                    }

                    // 🌟 [추가] 검색 취소 후 텅 빈 화면에 원래 리스트(기본값)를 다시 렌더링합니다.
                    refreshList();
                }
            });
        }
    }
    
    // 🌟 [추가] 검색 시작 전 기존 리스트 싹 비우기 (append 방지 및 뷰 클리어)
    if (docListContainer) docListContainer.innerHTML = "";
    
    // 🌟 [수정] 검색 시 설정(채팅) 탭으로 화면을 전환합니다.
    openWidget("settings");

    // 3. 사용자 질문 말풍선 즉시 렌더링
    await renderMessage({
        id: `${taskId}_query`,
        role: "user", 
        text: query,
        status: 9, 
        created_at: startTime,
        updated_at: startTime
    });

    try {
        const devicePref = getDevicePref();
        const isCloudMode = (document.getElementById("cloud-mode-toggle") as HTMLInputElement)?.checked;

        if (isCloudMode && currentSession.hash && currentSession.email) {
            // ☁️ [CLOUD SEARCH] LLM 은 서버에서 돌지만, 질의 임베딩은 반드시 로컬 모델로 만듭니다.
            renderProgressToUI({ task_id: taskId, category: "Cloud Sync", summary: "Embedding query locally...", spinner: "⠋" });

            let queryVector: number[] = [];
            try {
                queryVector = await invoke<number[]>("get_query_embedding", {
                    text: query,
                    devicePreference: devicePref
                });
                console.log(`[CLOUD SEARCH] Local query vector generated. dim = ${queryVector.length}`);
            } catch (err) {
                console.warn("[CLOUD SEARCH] Local embedding failed. Server will fall back.", err);
            }

            const origin = "https://commerce.logis.center";
            let targetHref = currentDetectedUrl || "https://commerce.logis.center/tracking";
            if (targetHref.includes("localhost") || targetHref.includes("127.0.0.1") || targetHref === "about:blank") {
                targetHref = "https://commerce.logis.center/tracking";
            }

            const urlObj = new URL(API_HOST);
            urlObj.searchParams.append("origin", origin);
            urlObj.searchParams.append("created_at", (Date.now() - timezoneOffset).toString());
            urlObj.searchParams.append("hash", currentSession.hash);
            urlObj.searchParams.append("token", currentSession.token || "");
            urlObj.searchParams.append("href", targetHref);
            urlObj.searchParams.append("from", currentSession.address || "");
            urlObj.searchParams.append("to", currentSession.team || "");

            const response = await invoke<any>("proxy_fetch", {
                url: urlObj.toString(),
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip"
                },
                body: { query: query, vector: queryVector },
                session_params: { hash: currentSession.hash, token: currentSession.token }
            });

            let serverTaskId = "";
            if (response && response.results && response.results.length > 0) {
                serverTaskId = response.results[0].id || "";
            }

            cloudPendingTasks.set(taskId, {
                serverId: serverTaskId,
                kind: "search",
                createdAt: Date.now()
            });

            renderProgressToUI({ task_id: taskId, category: "Cloud Queue", summary: "Query queued on Logis Center. Processing remotely.", spinner: "☁️" });

            isSearching = false;
            stopSpinner();
            if (btnSubmit) btnSubmit.style.display = "flex";
        } else {
            // 🌟 큐에 추가 (스피너는 백엔드가 실제 작업을 픽업하면 renderProgressToUI가 켭니다)
            await GlobalTaskManager.addToQueue(taskId, "ai_search", { 
                taskId: taskId, 
                query: query, 
                language: "korean",
                devicePreference: devicePref,
                searchMode: currentSearchMode,
                cc: activeContext.cc || "",
                bcc: activeContext.bcc || "",
                refId: activeContext.ref || ""
            });
        }
        
        // 🌟 [CRITICAL FIX] 검색을 대기열에 추가한 직후, 현재 주소가 전처리 중인지 여부를 재검사하여 번개 버튼을 확실히 숨깁니다.
        updateExtractButtonVisibility();

        // 🌟 [추가] 생성된 검색 테스크(질문) 말풍선 위치로 부드럽게 스크롤 이동
        setTimeout(() => {
            const taskEl = document.getElementById(`${taskId}_query`) || document.getElementById(taskId);
            const scrollEl = document.getElementById("chat-scroll");
            const container = document.querySelector(".chat-container") as HTMLElement;
            
            if (taskEl && scrollEl && container) {
                const maxScroll = Math.max(0, scrollEl.scrollHeight - container.clientHeight);
                let targetY = taskEl.offsetTop - (container.clientHeight / 2) + (taskEl.clientHeight / 2);
                
                if (targetY < 0) targetY = 0;
                if (targetY > maxScroll) targetY = maxScroll;
                
                currentY = targetY;
                scrollEl.style.transition = "transform 0.3s ease-out";
                updateTransform();
                
                setTimeout(() => { scrollEl.style.transition = ""; }, 300);
            }
        }, 100);

        // 🌟 [CRITICAL FIX] 여기서 isSearching = false를 하지 않습니다! 백엔드의 Done/Error 신호가 풀어줄 때까지 잠가둡니다.
    } catch(e) { 
        console.error("[SEARCH-ERROR]", e);
        if (aiResultsContent) aiResultsContent.innerHTML = "<div style='color:#ef4444;'>Error: " + e + "</div>"; 
        
        // 에러 발생 시에만 강제 해제
        isSearching = false; 
        if (btnSubmit) btnSubmit.style.display = "flex";
        stopSpinner(); 
        updateExtractButtonVisibility();
    } 
    // 🌟 [CRITICAL FIX] finally 블록을 통째로 삭제하여 isSearching이 조기 해제되어 큐가 뚫리는 치명적 버그를 차단했습니다.
});

document.addEventListener('show-doc', (e: any) => showDetail(e.detail));
document.addEventListener('view-task-log', () => { openWidget("list"); listView.style.display = "none"; detailView.style.display = "flex"; });


// 🌟 [CRITICAL FIX] 추출 버튼 더블클릭 완벽 방어 로직 적용
// 🌟 [CRITICAL FIX] 추출 버튼 더블클릭 방어 및 대기열(Queue) 다중 진입 허용
btnExtract?.addEventListener("click", async () => {
    // 1. 순수하게 더블클릭(extractClickLock)만 막고, 
    // 기존 작업이 돌아가고 있더라도 주소가 다르다면 큐에 넣을 수 있도록 조건 해제
    if (extractClickLock) {
        console.warn("[LOCK] Click locked to prevent double submission.");
        if (btnExtract) btnExtract.style.display = "none";
        return; 
    }
    
    // 2. 버튼 숨김 (isExtracting = true 는 백엔드 작업이 실제 픽업될 때 켜지도록 제외)
    extractClickLock = true;
    if (btnExtract) btnExtract.style.display = "none";

    console.log("[DEBUG] btnExtract clicked. currentDetectedUrl:", currentDetectedUrl, "currentImage:", currentImage);
    
    try {
        if (currentDetectedUrl || currentImage) {
            const logArea = document.getElementById("extraction-log");
            if (logArea) logArea.innerHTML = "";
            
            // 🌟 [CRITICAL FIX] 추출(Extract) 시 채팅창(settings) 탭으로 자동 이동합니다.
            openWidget("settings"); 

            const taskId = `task_${Date.now()}`;
            
            // 🌟 수동 renderMessage 및 startSpinner 제거: addToQueue가 대기열 UI(10번)를 예쁘게 그려줍니다.
            
            const isCloudMode = (document.getElementById("cloud-mode-toggle") as HTMLInputElement)?.checked;

            if (isCloudMode && currentSession.hash) {
                // ==========================================
                // ☁️ [SERVER MODE]
                // ==========================================
                console.log("[WIDGET] Routing task to Cloud Server...");
                let payloadBody = "";
                let format = "";

                if (currentImage) {
                    const ext = currentImage.split('.').pop()?.toLowerCase() || '';
                    const isDocument = ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'hwpx', 'txt', 'csv'].includes(ext);

                    const contents = await readFile(currentImage);
                    const blob = new Blob([contents]);
                    const base64Data = await new Promise<string>((resolve) => {
                        const reader = new FileReader();
                        reader.onloadend = () => { resolve(reader.result as string); };
                        reader.readAsDataURL(blob);
                    });
                    
                    payloadBody = base64Data;
                    // 문서일 경우 포맷 매핑 분기
                    if (isDocument) {
                        format = `application/${ext}`; // 서버에서 확장자 기반 파싱을 위해 전달
                    } else {
                        format = "image/png"; 
                    }
                } else {
                    payloadBody = await invoke<string>("extract_html_from_current_tab");
                    format = "text/html";
                }

                let extractionType = "html_extraction";
                if (currentImage) {
                    const ext = currentImage.split('.').pop()?.toLowerCase() || '';
                    const isDocument = ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'hwpx', 'txt', 'csv'].includes(ext);
                    extractionType = isDocument ? "document_extraction" : "image_extraction";
                }

                const requestData = {
                    id: taskId,
                    from: currentSession.address,
                    to: currentSession.team,
                    cc: activeContext.cc || "",
                    bcc: activeContext.bcc || "",
                    ref: activeContext.ref || "",
                    body: payloadBody,
                    link: currentDetectedUrl || "local",
                    type: extractionType
                };

                const urlObj = new URL(API_HOST);
                urlObj.searchParams.append("from", currentSession.address || "");
                urlObj.searchParams.append("to", currentSession.team || "");
                if (format.includes("image")) {
                    urlObj.searchParams.append("format", encodeURIComponent(format));
                }

                renderProgressToUI({ task_id: taskId, category: "Cloud Sync", summary: "Sending data to Logis Center...", spinner: "⠋" });

                const response = await invoke<any>("proxy_fetch", {
                    url: urlObj.toString(),
                    method: "POST",
                    headers: { 
                        "Content-Type": "application/json",
                        "Content-Encoding": "gzip" 
                    },
                    body: requestData,
                    session_params: { hash: currentSession.hash, token: currentSession.token }
                });

                console.log("[SERVER MODE] Task accepted by server:", response);

                // 🌟 [CLOUD TASK LIFECYCLE] 서버가 만든 task.id 를 기억해 두고 syncData 에서 완료를 판정합니다.
                let serverTaskId = "";
                if (response && response.results && response.results.length > 0) {
                    serverTaskId = response.results[0].id || "";
                }

                cloudPendingTasks.set(taskId, {
                    serverId: serverTaskId,
                    kind: "extract",
                    createdAt: Date.now()
                });

                renderProgressToUI({ task_id: taskId, category: "Cloud Queue", summary: "Task queued on server. Processing remotely.", spinner: "☁️" });

                // 클라우드 작업은 로컬 GPU 큐를 점유하지 않으므로 즉시 락을 해제합니다.
                isExtracting = false;
                stopSpinner();
                await GlobalTaskManager.release(taskId, taskId);
                await updateExtractButtonVisibility();
                
            } else {
                // ==========================================
                // 💻 [LOCAL MODE]
                // ==========================================
                if (currentImage) {
                    const ext = currentImage.split('.').pop()?.toLowerCase() || '';
                    const isDocument = ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'hwpx', 'txt', 'csv'].includes(ext);
                    const taskTypeStr = isDocument ? "document_extraction" : "image_extraction";
                    const logPrefix = isDocument ? "DOCUMENT" : "IMAGE";

                    console.log(`[WIDGET] Queuing LOCAL ${logPrefix} task...`);
                    const imageRefHash = await hashId(currentImage);

                    // 🚀 큐에 등록
                    await GlobalTaskManager.addToQueue(taskId, taskTypeStr, { 
                        id: taskId, type: taskTypeStr, image_path: currentImage, document_ext: ext,
                        ref: imageRefHash, 
                        cc: activeContext.cc || "",
                        bcc: activeContext.bcc || "",
                        link: `Local ${logPrefix}`,
                        device_preference: getDevicePref(), search_mode: currentSearchMode
                    });
                } else {
                    console.log("[WIDGET] Queuing LOCAL HTML/ANALYTIC task...");
                    const html = await invoke<string>("extract_html_from_current_tab");
                    
                    // 🌟 [CRITICAL FIX] 브라우저가 유휴 상태(Idle/Background)로 전환되어 currentDetectedUrl이 
                    // 빈 값이거나 about:blank로 날아갔을 경우, localhost로 엉뚱하게 매칭되는 것을 방어합니다!
                    let validUrl = currentDetectedUrl;
                    if (!validUrl || validUrl === "" || validUrl === "about:blank") {
                        const pageList = document.getElementById("nav-list-pages");
                        const activeLabel = pageList?.querySelector(".logis-label.active") as HTMLElement;
                        if (activeLabel && activeLabel.dataset.domain) {
                            validUrl = `https://${activeLabel.dataset.domain}`;
                        } else {
                            validUrl = "https://commerce.logis.center"; // 최후의 수단
                        }
                    }

                    const urlObj = new URL(validUrl.toLowerCase());
                    const rootDomain = getRootDomain(urlObj.hostname);
                    const cc = await hashId(rootDomain);
                    const rawPath = urlObj.pathname + urlObj.search;
                    const teamId = currentSession.team || "";
                    const hashedRefId = await hashId(teamId + cc + rawPath.toLowerCase());
                    
                    // 🌟 [추가] 사용자의 모드 선택에 따라 전처리 파이프라인(Task Type) 분기
                    const extractType = currentSearchMode === "analytic" ? "analytic_extraction" : "html_extraction";
                    
                    // 🚀 큐에 등록
                    await GlobalTaskManager.addToQueue(taskId, extractType, { 
                        id: taskId, type: extractType, html: html, link: rawPath, 
                        origin: urlObj.origin, // 🌟 [핵심] Rust 스케줄러가 localhost로 오판하지 않도록 origin(도메인)을 명시적으로 전달!
                        cc: activeContext.cc || cc, 
                        ref: activeContext.ref || hashedRefId, 
                        bcc: activeContext.bcc || "", 
                        from: currentSession.address, to: currentSession.team,
                        device_preference: getDevicePref(), search_mode: currentSearchMode
                    });
                }
            }
            
            if (currentImage) {
                currentImage = null;
                if (navPreviewContainer) navPreviewContainer.classList.add("hidden");
                if (navUploadBtn) navUploadBtn.classList.remove("active-emoji");
                if (searchInput) {
                    searchInput.disabled = false;
                    if (btnSubmit) {
                        const currentVal = searchInput.value.trim();
                        if (currentVal !== "" && !isQueryActive(currentVal)) {
                            btnSubmit.style.display = "flex";
                        } else {
                            btnSubmit.style.display = "none";
                        }
                    }
                }
            }
            console.log("[WIDGET] Task safely added to backend queue:", taskId);

            // 🌟 [CRITICAL FIX] 생성된 테스크 말풍선 위치로 부드럽게 스크롤 이동
            setTimeout(() => {
                const taskEl = document.getElementById(taskId);
                const scrollEl = document.getElementById("chat-scroll");
                const container = document.querySelector(".chat-container") as HTMLElement;
                
                if (taskEl && scrollEl && container) {
                    const maxScroll = Math.max(0, scrollEl.scrollHeight - container.clientHeight);
                    // 엘리먼트를 화면 중앙쯤에 오도록 Y값 계산
                    let targetY = taskEl.offsetTop - (container.clientHeight / 2) + (taskEl.clientHeight / 2);
                    
                    if (targetY < 0) targetY = 0;
                    if (targetY > maxScroll) targetY = maxScroll;
                    
                    currentY = targetY;
                    // 부드러운 스크롤 효과를 위해 transition 임시 적용
                    scrollEl.style.transition = "transform 0.3s ease-out";
                    updateTransform();
                    
                    // 이동 후 transition 제거 (원래 드래그를 위해 없는 상태 유지)
                    setTimeout(() => {
                        scrollEl.style.transition = "";
                    }, 300);
                }
            }, 100);
        }
    } catch (e) {
        console.error("[WIDGET] Extraction failed:", e);
        extractClickLock = false;
        // 다른 작업이 정상적으로 돌아가고 있을 수 있으므로 sys_lock이나 전역 스피너를 함부로 날리지 않습니다.
        updateExtractButtonVisibility();
    } finally {
        // 🌟 [CRITICAL FIX] Rust 백엔드(DB)에 작업이 완전히 등재되도록 1.5초간 여유를 줍니다.
        // 이 시간 동안은 버튼이 절대 부활하지 않으며, 1.5초 뒤 DB를 조회하여 정상적으로 큐에 등록되었다면 버튼은 계속 숨겨집니다.
        setTimeout(async () => {
            extractClickLock = false;
            await updateExtractButtonVisibility();
        }, 1500);
    }
});

// 🌟 [추가] Rust 백엔드에 성공적으로 등록되었을 때 가상 렌더링 내용을 실제 데이터로 덮어씌웁니다.
listen("task-db-registered", async (event: any) => {
    const p = event.payload;
    console.log(`[WIDGET] Task ${p.task_id} successfully registered in Backend DB.`);
    
    await renderMessage({
        id: p.task_id,
        task_id: p.task_id,
        role: "system_task",
        text: p.text,
        status: p.status,
        created_at: p.created_at,
        updated_at: Date.now()
    });
});

listen("extraction-progress", async (event: any) => { 
    const payload = event.payload;

    // 🌟 [CRITICAL FIX] 취소된 작업의 이벤트가 뒤늦게 도착하면 DOM을 파괴/재생성하지 못하도록 가장 먼저 폐기합니다.
    if (payload.task_id && GlobalTaskManager.cancelledTasks.has(payload.task_id)) {
        return;
    }

    if (payload.task_id) livePayloads.set(payload.task_id, payload);

    const summary = (payload.summary || "").toLowerCase();
    const isTerminal = payload.category === "Done" || payload.category === "Error" || summary.includes("cancelled") || summary.includes("stopped");
    
    if (isTerminal && payload.task_id) {
        console.log(`[QUEUE] Terminal state reached for ${payload.task_id}. Releasing and checking next.`);
        
        // 🌟 [CRITICAL FIX] 검색 완료(Done) 시점에 isSearching을 미리 false로 풀면, openWidget("list") 내부에서 
        // refreshList()가 발동하여 검색 결과를 전부 날려버리는 Race Condition이 발생합니다!
        // 따라서 추출 상태(isExtracting)만 먼저 풀고, 검색 상태(isSearching)는 UI 렌더링이 100% 끝난 최하단에서 풉니다.
        if (payload.task_id.startsWith("task_") || payload.task_id.startsWith("img_")) {
            isExtracting = false;
        } 

        // 🌟 큐 매니저 릴리즈 (비동기로 Dexie 업데이트 후 processNext 호출됨)
        await GlobalTaskManager.release(payload.task_id, payload.task_id);
        
        // 🌟 버튼 UI 즉시 갱신 및 스피너 중단 (검색 모드가 아닐 때만 선반영)
        if (!isSearching) {
            stopSpinner();
            updateExtractButtonVisibility();
        }
        // 🌟 [NEW] 추출 태스크 완료 시 네비게이션 카운트 및 리스트 최신화
        if ((payload.task_id.startsWith("task_") || payload.task_id.startsWith("img_")) && payload.category === "Done") {
            try {
                const freshDocs = await invoke<any[]>("get_all_documents", {
                    limit: pageSize,
                    offset: 0,
                    filter: `mode = '${currentSearchMode}'`
                });
                if (freshDocs.length > 0 && appDb) {
                    const normalized = normalizeEnvelope(freshDocs);
                    await appDb.table("items").bulkPut(normalized).catch(() => null);
                    await renderNavigation();
                    if (currentTab === "list") {
                        upsertListItems(normalized, 'prepend');
                    }
                }
            } catch (e) {
                console.warn("[SYNC] Post-extraction refresh failed:", e);
                // 폴백: 위 방식 실패 시 완전 새로고침
                if (currentTab === "list") {
                    await loadMoreDocs(true);
                }
            }
        }
        // 🌟 [추가] 에러이거나 취소된 경우 H3 복원
        if (payload.task_id.startsWith("search_") && payload.category !== "Done") {
             const resultH3 = document.querySelector('.nav-section.search h3');
             if (resultH3 && resultH3.textContent?.includes("searching")) {
                 const count = document.querySelectorAll('#doc-list .logis-result').length;
                 resultH3.innerHTML = `Result <strong class="count">${count > 0 ? `(${count})` : ""}</strong>`;
             }
             // 에러나 취소 시에는 여기서 락을 해제합니다.
             isSearching = false;
             stopSpinner();
             updateExtractButtonVisibility();
        }

        // 🌟 [추가] 검색 작업이 완료(Done)되었을 경우, 백엔드가 보내준 데이터를 결과창에 렌더링합니다.
        if (payload.task_id.startsWith("search_") && payload.category === "Done" && payload.data) {
            const response = payload.data;

            // 🌟 [ANALYTIC LOCAL RENDER] analytic 모드 검색 결과는 리스트 탭이 아닌
            //    채팅 탭에 말풍선으로 렌더링합니다.
            //    parse_analytic_query → search_items(mode='analytic') → search_chunks → Dexie Plan
            //    순서로 로컬에서 확정된 결과를 그대로 보여줍니다.
            if (currentSearchMode === "analytic") {
                const resultCount = response.results ? response.results.length : 0;

                // ① 요약 말풍선 (질의 · 기간 · 회수 건수)
                let summaryText = `Analytic 검색 완료: ${resultCount}건`;
                if (response.structured && response.structured.original_text) {
                    summaryText = `${response.structured.original_text} → ${resultCount}건`;
                }
                if (response.structured && Number(response.structured.started_at) > 0) {
                    const fmt = (ms: number) => {
                        const d = new Date(Number(ms));
                        const p = (n: number) => String(n).padStart(2, "0");
                        return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())}`;
                    };
                    const s = fmt(response.structured.started_at);
                    const e = Number(response.structured.expired_at) > 0
                        ? fmt(response.structured.expired_at)
                        : s;
                    const ti = response.structured.time_intent || "";
                    const si = response.structured.season_intent || "";
                    const tag = [ti, si].filter(Boolean).join(" / ");
                    summaryText += `\n기간: ${s} ~ ${e}${tag ? ` (${tag})` : ""}`;
                }
                await renderMessage({
                    id: `${payload.task_id}_answer`,
                    role: "system",
                    text: summaryText,
                    status: 9,
                    created_at: Date.now(),
                    updated_at: Date.now()
                });

                // 🌟 ①-1 리포트 말풍선
                //    Rust(STAGE-6)가 회수한 시맨틱 기록만 근거로 Qwen3.5 2B 가 작성한 답변입니다.
                //    근거가 없으면 백엔드가 빈 문자열을 돌려주므로 그때는 건너뜁니다.
                if (response.report && String(response.report).trim().length > 0) {
                    await renderMessage({
                        id: `${payload.task_id}_report`,
                        role: "system",
                        text: String(response.report).trim(),
                        status: 9,
                        created_at: Date.now() + 1,
                        updated_at: Date.now() + 1
                    });
                }

                // ② Dexie Plan 실행 → 상세 결과 말풍선
                if (response.dexie_plans && Array.isArray(response.dexie_plans) && response.dexie_plans.length > 0) {
                    const candidateIds = (response.results || []).map((r: any) => r.id).filter(Boolean);
                    for (const plan of response.dexie_plans) {
                        const condCount = plan.conditions ? plan.conditions.length : 0;
                        if (condCount === 0) continue;
                        try {
                            const passed = await executeDexiePlan(plan, { candidateIds, limit: 20 });
                            for (const p of passed) {
                                const d = p.data || {};
                                const actionText = d.action || "";
                                const summaryDoc = d.summary || "";
                                const crossFlow = d.cross_action_flow || "";
                                const text = d.text || "";
                                const displayText = actionText || summaryDoc || crossFlow || text;
                                if (displayText) {
                                    await renderMessage({
                                        id: `analytic_res_${p.id}`,
                                        role: "system",
                                        text: displayText,
                                        status: 9,
                                        created_at: Date.now(),
                                        updated_at: Date.now()
                                    });
                                }
                            }
                        } catch (e) {
                            console.warn("[ANALYTIC-LOCAL] Dexie plan execution failed:", e);
                        }
                    }
                }

                // ③ 결과 없음 안내
                if (resultCount === 0) {
                    await renderMessage({
                        id: `${payload.task_id}_empty`,
                        role: "system",
                        text: "해당 조건에 맞는 행동 로그를 찾지 못했습니다.",
                        status: 9,
                        created_at: Date.now(),
                        updated_at: Date.now()
                    });
                }

                // ④ 상태 정리
                isSearching = false;
                stopSpinner();
                updateExtractButtonVisibility();

                // ⑤ 채팅 스크롤 맨 아래로
                setTimeout(() => {
                    const scrollEl = document.getElementById("chat-scroll");
                    const container = document.querySelector(".chat-container") as HTMLElement;
                    if (scrollEl && container) {
                        const maxScroll = Math.max(0, scrollEl.scrollHeight - container.clientHeight);
                        currentY = maxScroll;
                        scrollEl.style.transition = "transform 0.3s ease-out";
                        updateTransform();
                        setTimeout(() => { scrollEl.style.transition = ""; }, 300);
                    }
                }, 100);
                return;
            }

            // 🌟 [CRITICAL FIX] isSearching = true 인 상태에서 탭을 전환해야 초기화(refreshList)가 방어됩니다!
            openWidget("list");

            if (listView) listView.style.display = "block";
            if (detailView) detailView.style.display = "none";
            
            // 🌟 [추가] 검색어와 카운트 H3에 업데이트
            const resultH3 = document.querySelector('.nav-section.search h3');
            if (resultH3) {
                // 🌟 [CONDITION SUMMARY] dexie_plans 에서 실제 적용된 조건을 요약해 보여줍니다.
                //  기존에는 tracking_number 만 특별 취급했지만,
                //  v4 는 모든 조건이 동등하게 플랜에 들어 있으므로 일반화합니다.
                const applied: string[] = [];
                const coveredTypes = new Set<string>();

                if (response.dexie_plans && Array.isArray(response.dexie_plans)) {
                    for (const plan of response.dexie_plans) {
                        // 🌟 v4 : 어떤 도메인을 훑었는지도 함께 보여줍니다.
                        const t = (plan.types && plan.types.length > 0) ? plan.types : (plan.type ? [plan.type] : []);
                        for (const x of t) coveredTypes.add(x);

                        if (!plan.conditions) continue;
                        for (const c of plan.conditions) {
                            const label = c.path.replace('data.', '');
                            if (c.op === 'top' || c.op === 'bottom') {
                                applied.push(`${label} ${c.op} ${c.percent ?? 20}%`);
                            } else {
                                const opSym: Record<string, string> = {
                                    eq: '=', neq: '≠', gt: '>', gte: '≥', lt: '<', lte: '≤',
                                    contains: '⊇', not_contains: '⊉'
                                };
                                applied.push(`${label} ${opSym[c.op] || c.op} ${c.value}`);
                            }
                        }
                    }
                }

                console.log(`[SEARCH-DEBUG] 커버 도메인: [${Array.from(coveredTypes).join(', ')}]`);

                let queryText = "";
                if (response.structured && response.structured.original_text) {
                    queryText = response.structured.original_text;
                }
                if (!queryText) {
                    const queryEl = document.getElementById(`${payload.task_id}_query`);
                    if (queryEl) queryText = queryEl.querySelector('.content')?.textContent?.trim() || "";
                }

                // 🌟 [MODE LABEL] 파일 상단 modeLabel() 단일 정의를 사용합니다.
                //    (v1 에서는 여기만 'Goods' 로 되어 있어 검색창 라벨과 어긋났습니다)
                let displayMode = modeLabel(currentSearchMode);

                // 🌟 카운트는 '렌더링될 문서 수' 로 확정해야 하므로 아래 updateResultCount 가 다시 갱신합니다.
                //    여기서는 조건 요약만 먼저 노출합니다.
                const condStr = applied.length > 0
                    ? ` <span style="font-size:0.85em; opacity:0.7;">[${applied.slice(0, 3).join(' · ')}${applied.length > 3 ? ` +${applied.length - 3}` : ''}]</span>`
                    : "";

                resultH3.innerHTML = `Search ${displayMode}: "${queryText}"${condStr} <strong class="count"></strong>`;
            }

            console.log(`[SEARCH-DEBUG] 백엔드에서 수신한 리콜 후보 수: ${response.results ? response.results.length : 0}`);
            console.log(`[SEARCH-DEBUG] 수신한 Dexie 플랜 수: ${response.dexie_plans ? response.dexie_plans.length : 0}`);

            // 🌟 [DEXIE PLAN EXECUTION]
            //  LanceDB 는 조건을 적용하지 않은 '리콜 후보' 만 돌려줍니다.
            //  실제 조건 필터링(가격/수량/송장번호/상태/기간/top·bottom)은 여기서 수행합니다.
            //  → 기존의 trackingEqDocs 특수 분기는 plan.conditions 의
            //    data.tracking_number eq 조건으로 일반화되어 사라집니다.
            let planFilteredIds: Set<string> | null = null;
            const planBadges = new Map<string, string>();

            if (response.dexie_plans && Array.isArray(response.dexie_plans) && response.dexie_plans.length > 0) {
                const candidateIds = (response.results || []).map((r: any) => r.id).filter(Boolean);
                const accepted = new Set<string>();

                for (const plan of response.dexie_plans) {
                    const condCount = plan.conditions ? plan.conditions.length : 0;

                    // 조건이 하나도 없는 플랜은 후보를 그대로 통과시킵니다. (리콜 우선)
                    if (condCount === 0) {
                        for (const id of candidateIds) accepted.add(id);
                        console.log(`[DEXIE-PLAN] type='${plan.type}' 조건 0개 → 후보 전량 통과`);
                        continue;
                    }

                    try {
                        const passed = await executeDexiePlan(plan, { candidateIds, limit: 500 });
                        for (const p of passed) {
                            accepted.add(p.id);
                            // 어떤 조건으로 통과했는지 배지로 남깁니다.
                            const first = plan.conditions[0];
                            if (first) {
                                planBadges.set(p.id, `🎯 ${first.path.replace('data.', '')} ${first.op}`);
                            }
                        }
                        console.log(`[DEXIE-PLAN] type='${plan.type}' 조건 ${condCount}개 → ${passed.length}건 통과`);

                        // 🌟 [PLAN RECALL v2] 조건이 있는 플랜은 '항상' Dexie 전체를 한 번 더 훑습니다.
                        //  ── 왜 조건부를 없애는가 ──
                        //   candidateIds 는 lib.rs 가 점수 상위 N건으로 자른 배열이고,
                        //   그 정렬 기준에는 도메인 조건이 아직 반영되어 있지 않습니다.
                        //   기존처럼 passed.length === 0 일 때만 구출하면,
                        //   후보 안에서 1건이라도 통과하는 순간 구출이 멈춰
                        //   순위 밖의 정답이 그대로 사라집니다.
                        //   조건 필터링은 Dexie 인덱스로 O(log n) 이므로
                        //   항상 도는 편이 안전하고, 리콜이 전송 상한과 완전히 분리됩니다.
                        {
                            const rescued = await executeDexiePlan(plan, { limit: 200 });
                            let added = 0;
                            for (const p of rescued) {
                                if (accepted.has(p.id)) continue;
                                accepted.add(p.id);
                                planBadges.set(p.id, `🛟 recall`);
                                added++;
                            }
                            if (added > 0) {
                                console.log(`[DEXIE-PLAN] 🛟 후보 밖에서 ${added}건 추가 확보 (전송 상한과 무관한 조건 리콜)`);
                            }
                        }
                    } catch (e) {
                        console.error(`[DEXIE-PLAN] 실행 실패 (type='${plan.type}'):`, e);
                        // 실패 시 조건을 포기하고 후보를 통과시킵니다. 0건보다 낫습니다.
                        for (const id of candidateIds) accepted.add(id);
                    }
                }

                planFilteredIds = accepted;
                console.log(`[DEXIE-PLAN] 최종 통과 문서 ${accepted.size}건`);
            }

            // 🌟 2. 실제 검색 결과 문서(Card)를 #doc-list 영역에 렌더링하고 카운트 갱신
            if (response.results && response.results.length > 0) {
                if (docListContainer) docListContainer.innerHTML = ""; // 기존 목록 비우기

                let docs: any[] = [];
                const seenIds = new Set<string>();

                for (const res of response.results) {
                    try {
                        // 🌟 [PLAN GATE] 플랜이 존재하면 통과한 문서만 렌더링합니다.
                        if (planFilteredIds && !planFilteredIds.has(res.id)) continue;

                        // 🌟 백엔드가 data 컬럼(JSON 문자열)을 res.text 로 보냅니다.
                        //    get_document 를 건건이 재호출하는 N+1 쿼리를 원천 차단합니다.
                        let parsedData: any = {};
                        try { parsedData = JSON.parse(res.text); } catch(e) {}

                        let fullDoc: any = {
                            id: res.id,
                            uuid: res.id,
                            type: parsedData.type || res.context_type || "unknown",
                            mode: parsedData.mode || currentSearchMode,
                            data: parsedData,
                            text: parsedData.text || "",
                            created_at: parsedData.created_at || 0,
                            updated_at: parsedData.updated_at || 0
                        };

                        if (fullDoc.data) {
                            fullDoc.data.search_score = res.score;
                            fullDoc.data.search_context = res.context_type;
                            const badge = planBadges.get(res.id);
                            if (badge) fullDoc.data.search_badge = badge;
                        }

                        seenIds.add(res.id);
                        docs.push(fullDoc);
                    } catch (e) {
                        console.error("Failed to process search result:", e);
                    }
                }

                // 🌟 [RESCUED MERGE] LanceDB 후보에는 없었지만 플랜이 건져 올린 문서를 합칩니다.
                if (planFilteredIds) {
                    const rescuedIds = Array.from(planFilteredIds).filter(id => !seenIds.has(id));
                    if (rescuedIds.length > 0 && appDb) {
                        const rescuedRows = await appDb.table('items').where('id').anyOf(rescuedIds).toArray();
                        for (const row of rescuedRows) {
                            docs.push({
                                id: row.id,
                                uuid: row.id,
                                type: row.type || "unknown",
                                mode: row.mode || currentSearchMode,
                                data: { ...(row.data || {}), search_badge: planBadges.get(row.id) || "🛟 recall" },
                                text: row.data?.text || "",
                                created_at: row.created_at || 0,
                                updated_at: row.updated_at || 0
                            });
                            seenIds.add(row.id);
                        }
                        console.log(`[SEARCH-DEBUG] 플랜 구출 문서 ${rescuedRows.length}건 병합 완료`);
                    }
                }

                console.log(`[SEARCH-DEBUG] 1차 파싱 완료. 기본 문서 수: ${docs.length}`);

                // 🌟 [N:N RELAY v4] 루트 호이스팅 컬럼(index/goods/order/...) 대신
                //  data.* 중첩 인덱스를 직접 사용합니다.
                //  canonicalize 가 식별자를 전부 String 으로 확정했으므로
                //  기존의 '숫자/문자 두 갈래로 쏘던 타입 방어 쿼리' 가 불필요해집니다.
                //  (쿼리 수가 절반으로 줄고, 타입 혼재로 절반을 놓치던 문제도 사라집니다)
                if (appDb && docs.length > 0) {
                    console.log(`[SEARCH-DEBUG] Dexie 연관 교차 검색(Relay v4) 시작...`);
                    const relayDocs = new Map<string, any>();
                    const existingIds = new Set(docs.map(d => d.id));
                    // 연관 축으로 사용할 data.* 경로 (전부 인덱스 선언되어 있음)
                    const LINK_PATHS = [
                        'data.index', 'data.no', 'data.code', 'data.tracking_number',
                        'data.goods', 'data.order', 'data.tracking',
                        'data.stock_keeping_unit', 'data.barcode',
                        // 🌟 [TRADE LINK PATHS] 무역 문서 간 연결 축
                        'data.doc_number', 'data.reference_invoice',
                        'data.reference_lc', 'data.reference_booking',
                        'data.container_number', 'data.seal_number',
                        // 🌟 [TRADE INDEX LINK] commerce 의 data.order / data.tracking 과 동일한
                        //    'index 로 서로를 가리키는' 축입니다.
                        //    문자열 doc_number 는 표기가 흔들려도 이 숫자는 절대 흔들리지 않습니다.
                        // 🌟 [FULL 45-CODE] ITEMS_SCHEMA 의 data.rel_* 전량과 1:1 로 일치해야 합니다.
                        //    기존에는 15종만 있어 나머지 30종의 무역 문서 간 연결이
                        //    N:N RELAY 교차 검색에서 통째로 누락되었습니다.
                        'data.rel_bl', 'data.rel_hbl', 'data.rel_swb', 'data.rel_awb',
                        'data.rel_ci', 'data.rel_cinv', 'data.rel_csi', 'data.rel_pi', 'data.rel_pl',
                        'data.rel_po', 'data.rel_sc', 'data.rel_lc', 'data.rel_llc', 'data.rel_co',
                        'data.rel_bc', 'data.rel_bk', 'data.rel_sr', 'data.rel_do', 'data.rel_an',
                        'data.rel_sa', 'data.rel_fcr', 'data.rel_pod', 'data.rel_cm', 'data.rel_fi',
                        'data.rel_wr', 'data.rel_ed', 'data.rel_id', 'data.rel_ccc', 'data.rel_cnm',
                        'data.rel_el', 'data.rel_ic', 'data.rel_wc', 'data.rel_ca', 'data.rel_coa',
                        'data.rel_pc', 'data.rel_fc', 'data.rel_hc', 'data.rel_cdr',
                        'data.rel_ip', 'data.rel_icf', 'data.rel_lg', 'data.rel_tr',
                        'data.rel_soa', 'data.rel_dn', 'data.rel_cn', 'data.rel_ti', 'data.rel_cp',
                        'data.rel_be', 'data.rel_ins', 'data.rel_dgd'
                    ];

                    // 🌟 하나의 값으로 모든 연관 축을 한 번에 훑는 헬퍼
                    const findLinked = async (value: string | number): Promise<any[]> => {
                        if (value === "" || value === 0 || value == null) return [];
                        let coll = appDb.table("items").where(LINK_PATHS[0]).equals(value);
                        for (let i = 1; i < LINK_PATHS.length; i++) {
                            coll = coll.or(LINK_PATHS[i]).equals(value);
                        }
                        return await coll.toArray();
                    };

                    const absorb = (match: any, relation: string) => {
                        if (!match || !match.id) return false;
                        if (existingIds.has(match.id) || relayDocs.has(match.id)) return false;
                        const dData = { ...(match.data || {}) };
                        dData.search_context = match.type;
                        dData.relation = relation;
                        relayDocs.set(match.id, { ...match, data: dData });
                        return true;
                    };

                    for (const doc of docs) {
                        const parsedData = doc.data || {};

                        // 1) 정방향 : 내 index 를 참조하는 문서들
                        const selfIndex = parsedData.index != null ? Number(parsedData.index) : 0;
                        if (selfIndex) {
                            for (const match of await findLinked(selfIndex)) {
                                if (match.id === doc.id) continue;
                                absorb(match, "forward");
                            }
                        }

                        // 2) 역방향 : 내가 참조하는 외래키로 부모 찾기
                        const refKeys = ["tracking_number", "no", "code", "goods", "order", "tracking", "stock_keeping_unit", "barcode"];
                        for (const key of refKeys) {
                            const rawRef = parsedData[key];
                            if (rawRef === undefined || rawRef === null || rawRef === "") continue;
                            const refVal: string | number = (key === "goods" || key === "order" || key === "tracking")
                                ? Number(rawRef)
                                : String(rawRef);
                            for (const match of await findLinked(refVal)) {
                                if (match.id === doc.id) continue;
                                if (!absorb(match, "backward")) continue;

                                // 3) 2-Depth 체이닝 : 부모의 index 로 다시 자식 찾기
                                const bIndex = match.data?.index ? String(match.data.index) : "";
                                if (!bIndex) continue;
                                for (const d2 of await findLinked(bIndex)) {
                                    if (d2.id === match.id || d2.id === doc.id) continue;
                                    absorb(d2, "backward_chained");
                                }
                            }
                        }
                    }

                    console.log(`[SEARCH-DEBUG] 연관 교차 검색으로 추가된 문서 수: ${relayDocs.size}`);
                    docs.push(...Array.from(relayDocs.values()));
                }
                
                console.log(`[SEARCH-DEBUG] 화면에 렌더링될 최종 문서 수(docs.length): ${docs.length}`);

                // 🌟 [TOTAL COUNT] AI 검색은 단발성 고정 셋이므로 docs.length 가 곧 전체 건수입니다.
                totalResultCount = docs.length;

                if (docs.length > 0) {
                    upsertListItems(docs, 'append');
                    hasMore = false; // 🌟 [CRITICAL FIX] AI 검색 결과는 단발성 고정 셋이므로 스크롤을 통한 전체 리스트 더 불러오기를 원천 차단합니다!
                } else {
                    if (docListContainer) docListContainer.innerHTML = `<div class="empty">No detailed documents found.</div>`;
                    hasMore = false;
                }
            } else {
                if (docListContainer) docListContainer.innerHTML = `<div class="empty">No matching data found.</div>`;
                totalResultCount = 0;
                hasMore = false; // 🌟 결과가 없을 때도 추가 불러오기 방지
            }
            
            console.log(`[SEARCH-DEBUG] 렌더링 직후 H3 DOM 카운트 동기화 시도 (updateResultCount)`);
            updateResultCount(); // 🌟 카운트 갱신 반영
            
            // 🌟 [CRITICAL FIX] 검색 결과 DOM 렌더링이 완벽하게 끝난 이 시점에 드디어 락을 해제합니다!
            isSearching = false;
            stopSpinner();
            updateExtractButtonVisibility();
            console.log(`[SEARCH-DEBUG] 글로벌 락 해제 완료 (isSearching = false)`);
        }
    }

    if (isFetchingLogs && payload.task_id === activeTaskId) {
        pendingLiveEvents.push(payload);
        return;
    }
    renderProgressToUI(payload); 
});

document.addEventListener('render-progress', (e: any) => { renderProgressToUI(e.detail); });

async function renderProgressToUI(payload: any, isRecovery: boolean = false) {
    payload.task_id = payload.task_id || activeTaskId || (document.getElementById("extraction-log")?.dataset.activeTaskId);
    const tId = payload.task_id;
    if (!tId) return;

    // 🌟 [CRITICAL FIX] 렌더링 함수 내부에서도 블랙리스트를 한 번 더 검사하여 좀비 UI 생성을 이중으로 방어합니다.
    if (GlobalTaskManager.cancelledTasks.has(tId)) return;

    const summary = (payload.summary || "").toLowerCase();
    const isTerminal = payload.category === "Done" || payload.category === "Error" || summary.includes("cancelled") || summary.includes("stopped");
    const isNotification = payload.category === "Warning" || payload.category === "Info";

    // 🌟 [CRITICAL FIX 1] 상태 입양(Adopt) 범위 확대: 
    // 백엔드에서 날아오는 'Loading Model', 'Saving', 'Handover' 등 모든 활동을 
    // '진행 중'으로 인지하여 큐가 풀리지 않도록 락을 단단히 고정합니다!
    const isPayloadRunning = payload.category && !["Pending", "Cloud Sync", "Cloud Queue"].includes(payload.category);

    if (!isRecovery && !isTerminal && isPayloadRunning) {
        if (activeTaskId !== payload.task_id || !GlobalTaskManager.isBusy) {
            console.log("[WIDGET] Adopting/Confirming running background task:", payload.task_id);
            
            activeTaskId = payload.task_id;
            await kvSet("sys_lock", activeTaskId!);
            GlobalTaskManager.isBusy = true;
            GlobalTaskManager.currentTaskId = activeTaskId;
            
            if (payload.task_id && payload.task_id.startsWith("search_")) {
                isSearching = true;
                if (btnSubmit) btnSubmit.style.display = "none";
            } else {
                isExtracting = true;
            }
            startSpinner();
        } else if (!spinnerInterval) {
            // 🌟 [CRITICAL FIX] 이미 activeTaskId가 일치하여 입양(Adoption) 블록을 건너뛰었더라도,
            // 스피너가 돌고 있지 않다면 강제로 스피너를 가동시켜 무반응(멈춤) 버그를 방지합니다.
            startSpinner();
        }
    }

    const baseCategory = payload.category ? payload.category.replace(/\s*\(.*?\)/g, "") : "general";
    const catId = baseCategory.replace(/[^a-zA-Z0-9]/g, "");
    const elementId = `progress-${catId}`;
    
    let displaySummary = payload.summary || "";
    
    // 🌟 [CRITICAL FIX] 백엔드에서 텍스트(summary)가 없는 순수 로그 이벤트를 보냈을 때,
    // 기존에 화면에 떠있던 텍스트를 보존하여 말풍선이 텅 비어버리는 현상을 원천 차단합니다!
    if (tId) {
        const existingEl = document.getElementById(tId) as HTMLElement;
        if (!displaySummary && existingEl) {
            displaySummary = existingEl.querySelector('.content')?.textContent || "";
        }
    }
    
    if (!taskSteps.has(tId)) {
        taskSteps.set(tId, new Map());
    }
    const stepMap = taskSteps.get(tId)!;

    // 🌟 [UI 심플화] 복잡한 계산식을 모두 삭제하고, 오직 'List Extraction' 단계에서만 [N/M]을 보여줍니다!
    if (!isTerminal && !isNotification) {
        let rawSummary = payload.summary || "";
        const pctMatch = rawSummary.match(/\(\d+%\)/);
        const hasDots = rawSummary.endsWith("...");
        
        if (hasDots) rawSummary = rawSummary.slice(0, -3).trim();
        if (pctMatch) rawSummary = rawSummary.replace(pctMatch[0], '').trim();

        let fractionStr = "";
        if (payload.category && payload.category.includes("List Extraction")) {
            const match = payload.category.match(/\((\d+)\/(\d+)\)/);
            if (match) {
                fractionStr = ` [${match[1]}/${match[2]}]`; // 백엔드가 준 정확한 숫자만 사용
            }
        }
        
        displaySummary = `${rawSummary}${fractionStr}${pctMatch ? ' ' + pctMatch[0] : ''}${hasDots ? '...' : ''}`;
    } else if (isNotification) {
        displaySummary = payload.summary || "";
    }

    // 🌟 [CRITICAL FIX] 대기열 상태(10)와 진행 상태(1)를 엄격히 구분합니다.
    let statusCode = 1; 
        
    if (isTerminal) {
        if (payload.category === "Done") statusCode = 9;
        else if (payload.category === "Error") statusCode = 6;
        else statusCode = 3;
    } else if (summary.includes("cancelled") || summary.includes("stopped")) {
        statusCode = 3;
    } else {
        // 🌟 [CRITICAL FIX 2] 백엔드에서 날아오는 중간 과정들이 10번(QUEUED)으로 오해받아 
        // 텍스트가 지워지고 스피너가 멈추는 버그를 원천 차단합니다. 오직 Pending 계열만 10번을 부여합니다!
        if (payload.category === "Pending" || payload.category === "Cloud Sync" || payload.category === "Cloud Queue") {
            statusCode = 10;
        } else {
            statusCode = 1;
        }
    }
    
    if (payload.task_id) {
        const existingEl = document.getElementById(payload.task_id) as HTMLElement;
        let originalCreatedAt = Date.now();
        if (existingEl) {
            originalCreatedAt = parseInt(existingEl.dataset.createdAt || "0");
        } else {
            const match = payload.task_id.match(/_(\d+)$/);
            if (match) originalCreatedAt = parseInt(match[1]);
        }

        await renderMessage({ 
            id: payload.task_id, 
            role: "system_task", 
            // 🌟 [CRITICAL FIX 1] content 대신 text 속성을 명시적으로 사용하여 텍스트 증발 방지
            text: displaySummary, 
            status: statusCode, 
            created_at: originalCreatedAt, 
            updated_at: Date.now(),
            task_id: payload.task_id
        });
    }

    // 🌟 [CRITICAL FIX] 1차 스피너 및 전역 상태 종료 처리 
    // 사용자가 현재 무슨 화면을 보고 있든, 작업이 끝났다면 무조건 전역 락을 풀고 스피너를 정지시킵니다!
    if (isTerminal) {
        // 🌟 [CRITICAL FIX] 과거의 로그(History)를 불러올 때 터미널 이벤트가 현재 진행 중인 작업을 끄는 것을 완벽 차단!
        // 오직 현재 활성화된 작업(activeTaskId)이 종료되었을 때만 글로벌 락을 해제합니다.
        if (tId === activeTaskId || !activeTaskId) {
            const currentLock = await kvGet("sys_lock");
            if (currentLock === tId || !currentLock) {
                await kvRemove("sys_lock");
            }
            
            isExtracting = false;
            isSearching = false;
            stopSpinner();
            
            if (btnExtract) { btnExtract.classList.remove("active-spinner"); btnExtract.innerText = "⚡"; }
            if (currentImage) {
                currentImage = null; 
                if (navPreviewContainer) navPreviewContainer.classList.add("hidden"); 
                if (navUploadBtn) navUploadBtn.classList.remove("active-emoji"); 
                if (searchInput) searchInput.disabled = false; 
                if (btnSubmit) btnSubmit.style.display = "flex"; 
            }
            // 🌟 [CRITICAL FIX] 작업 완료 시점에 브라우저가 이미 종료된 상태라면
            //    isAutoLaunchLocked를 강제로 false로 리셋하여 btnAutoLaunch 노출을 보장합니다.
            //    작업 진행 중 브라우저가 종료되면 isAutoLaunchLocked가 true로 남아있어
            //    updateExtractButtonVisibility()의 2번 단계에서 btnAutoLaunch가 노출되지 않는 버그를 수정합니다.
            if (!isBrowserRunning) {
                isAutoLaunchLocked = false;
            }
            updateExtractButtonVisibility(); 
        }

        // 🌟 [CRITICAL FIX] 전처리가 완료되면 자동으로 메뉴 카운트와 리스트를 리프레시!
        if (payload.category === "Done") {
            // 🌟 [버그 수정] 서버 모드이든 로컬 모드이든 무조건 로컬 LanceDB의 최신 전처리 결과를 Dexie에 먼저 덮어써야 합니다!
            Promise.all([
                invoke<any[]>("get_known_users"),
                invoke<any[]>("get_known_pages") 
            ]).then(async ([users, pages]) => {
                console.log("\n[TRACKING-1] Rust(LanceDB)에서 가져온 get_known_users 목록 수:", users ? users.length : 0);
                if (users && users.length > 0) {
                    const teamDocs = users.filter(u => u.type === "team" || (u.data && u.data.type === "team"));
                    console.log("[TRACKING-2] 그 중 'team' 타입 문서 파악:", teamDocs);
                    if (teamDocs.length > 0) {
                        // 🌟 [CRITICAL FIX] 로그 출력을 위해 json_data 문자열을 객체로 안전하게 파싱합니다.
                        let tData: any = null;
                        if (teamDocs[0].json_data && typeof teamDocs[0].json_data === "string") {
                            try { tData = JSON.parse(teamDocs[0].json_data); } catch(e) {}
                        }
                        if (!tData && teamDocs[0].data) {
                            tData = typeof teamDocs[0].data === "string" ? JSON.parse(teamDocs[0].data) : teamDocs[0].data;
                        }
                        tData = tData || teamDocs[0];
                        
                        console.log("[TRACKING-3] 화면에 반영될 최신 Base 통계:", JSON.stringify(tData.base?.pages, null, 2));
                    } else {
                        console.warn("[TRACKING-WARN] get_known_users에 'team' 문서가 포함되지 않았습니다! (Limit 제한 의심)");
                    }
                }
                
                // 🌟 [CRITICAL FIX] 프론트엔드 최신화 버그 해결! 서버 동기화(네트워크 상태)와 무관하게, 백엔드 로컬 통계가 갱신되었으므로 무조건 즉시 UI를 새로고침합니다.
                await renderNavigation();
                if (currentTab === "list") {
                    // 🌟 [로직 충돌 수정] AI 검색(ai_search) 완료 시에는 전용 리스너에서 직접 결과를 화면에 그려줍니다.
                    // 여기서 refreshList()를 호출하면 전체 리스트 불러오기(10개)가 중첩되어 20개로 늘어나는 Race Condition 버그가 발생하므로 제외합니다.
                    if (!(payload.task_id && payload.task_id.startsWith("search_"))) {
                        refreshList();
                    }
                }
                
                // 🌟 UI를 100% 최신 상태로 바꾼 뒤에 백그라운드에서 조용히 서버와 동기화를 진행합니다.
                if (currentSession.email) {
                    syncData(); 
                }
            });
        }
    }

    // 🌟 이제 현재 열려있는 Detail View가 이 Task의 것인지 확인 후 내부 로그(DOM)를 업데이트합니다.
    const extractionLog = document.getElementById("extraction-log");
    const targetContainer = document.getElementById("progress-container") || extractionLog;

    if (extractionLog && detailView.style.display !== "none") {
        if (extractionLog.dataset.activeTaskId !== tId) {
            // 현재 보고 있는 화면이 다른 Task면 여기서 DOM 업데이트 중지! (버블은 이미 위에서 업데이트됨)
            return;
        }

        if (payload.category === "Processing" && stepMap.size > 0) {
            stepMap.clear();
            if (targetContainer) targetContainer.innerHTML = "";
            await kvRemove(`term_${tId}`);
            const termArea = document.getElementById("terminal-logs");
            if (termArea) { termArea.innerHTML = ""; termArea.style.display = "none"; }
        }

        if (!stepMap.has(elementId)) {
            stepMap.set(elementId, stepMap.size + 1);
        }

        if (isTerminal) {
            if (targetContainer) {
                 const existingSpinners = targetContainer.querySelectorAll('.active-spinner');
                 existingSpinners.forEach(s => {
                     s.classList.remove('active-spinner');
                     s.innerHTML = payload.category === "Error" ? "❌" : "✅";
                     (s as HTMLElement).style.color = payload.category === "Error" ? "#ef4444" : "#4ade80";
                 });
            }
            if (btnStopTask) btnStopTask.style.display = "none";
            if (btnDetailDelete) btnDetailDelete.style.display = "flex";
        }

        let p = document.getElementById(elementId);
        if (!p) {
            if (targetContainer && !isNotification) {
                const existingSpinners = targetContainer.querySelectorAll('.active-spinner');
                existingSpinners.forEach(s => {
                    s.classList.remove('active-spinner');
                    s.innerHTML = "✅";
                    (s as HTMLElement).style.color = "#4ade80";
                });
            }

            p = document.createElement("div"); p.id = elementId;
            p.className = "progress-item";
            p.style.borderBottom = "1px solid #eee"; p.style.padding = "6px 0"; p.style.fontSize = "0.8rem";
            p.style.display = "flex"; p.style.flexDirection = "column"; 
            const row = document.createElement("div"); row.className = "progress-row"; row.style.display = "flex"; row.style.alignItems = "center";
            
            const spinnerIcon = `<span class="active-spinner" style="color:var(--primary); margin-right:8px; font-family:monospace; min-width:15px;">⠋</span>`;
            row.innerHTML = `${spinnerIcon}<span class="summary-text">${displaySummary}</span>`;
            p.appendChild(row);
            const results = document.createElement("div"); results.className = "results-container"; p.appendChild(results);
            
            if (targetContainer) targetContainer.appendChild(p);
        }
        
        const summaryEl = p.querySelector(".summary-text") as HTMLElement;
        const spinnerEl = p.querySelector(".active-spinner") as HTMLElement;

        if (summaryEl && summaryEl.textContent !== displaySummary) {
            summaryEl.textContent = displaySummary;
        }

        if (payload.category === "Done") {
            const row = p.querySelector(".progress-row");
            if (row) {
                const s = row.querySelector(".active-spinner") as HTMLElement;
                if (s) { s.classList.remove("active-spinner"); s.innerHTML = "✅"; s.style.color = "#4ade80"; }
            }
        } else if (payload.category === "Error") {
            const row = p.querySelector(".progress-row");
            if (row) { 
                const s = row.querySelector(".active-spinner") as HTMLElement;
                if (s) { s.classList.remove("active-spinner"); s.innerHTML = "❌"; s.style.color = "#ef4444"; }
                (row as HTMLElement).style.color = "#ef4444"; 
            }
        } else if (isNotification) {
            if (spinnerEl) {
                spinnerEl.classList.remove("active-spinner");
                spinnerEl.innerHTML = payload.spinner || "⚠️";
                spinnerEl.style.color = "#fbbf24"; 
            }
        } else {
            if (spinnerEl && spinnerEl.innerHTML !== "✅" && spinnerEl.innerHTML !== "❌" && spinnerEl.innerHTML !== "⚠️") {
                const newIcon = payload.spinner || "⠋";
                if (spinnerEl.innerText !== newIcon) { spinnerEl.innerText = newIcon; }
                if (newIcon === "✅" || newIcon === "✔") {
                    spinnerEl.classList.remove("active-spinner"); spinnerEl.style.color = "#4ade80";
                } else if (newIcon === "❌") {
                    spinnerEl.classList.remove("active-spinner"); spinnerEl.style.color = "#ef4444";
                } else {
                    spinnerEl.classList.add("active-spinner");
                }
            }
        }
    }
}

btnStopTask?.addEventListener("click", async () => {
    if (await ask("Stop the current extraction/search? (The record will be deleted)", { title: "Stop Task", kind: "warning" })) {
        const targetTaskId = activeTaskId; // 지우려는 대상 고정
        
        if (targetTaskId) {
            GlobalTaskManager.cancelledTasks.add(targetTaskId); // 🌟 [CRITICAL FIX] 취소 블랙리스트에 등록하여 지연 도착하는 이벤트를 완벽 차단
        }
        
        // 🌟 [CRITICAL FIX] 취소 즉시 락을 강제 해제하여 취소 후 #btn-extract 버튼이 먹통되는 현상을 완벽 방어합니다.
        activeTaskId = null;
        GlobalTaskManager.isBusy = false;
        GlobalTaskManager.currentTaskId = null;
        GlobalTaskManager.currentTaskPayload = null;

        isExtracting = false; 
        isSearching = false; 
        stopSpinner();
        
        if (btnExtract) {
            btnExtract.classList.remove("active-spinner");
            btnExtract.innerText = "⚡";
            btnExtract.style.display = "flex";
        }
        if (btnStopTask) btnStopTask.style.display = "none";

        try {
            console.log("[WIDGET] Stopping task:", targetTaskId);
            // 1. 백엔드 작업 중단
            await invoke<string>("stop_current_extraction", { taskId: targetTaskId });
            
            if (targetTaskId) {
                await kvRemove(`term_${targetTaskId}`);
                const el = document.getElementById(targetTaskId);
                if (el) el.remove();

                // 2. 큐 매니저에서 식별자 제거 및 다음 대기열 진행
                await GlobalTaskManager.release(targetTaskId, targetTaskId);
            }

            detailTitle.innerText = "Cancelled";
            detailContent.innerHTML = "<div style='color:#ef4444; padding:20px;'>Extraction stopped and deleted by user.</div>";
            
            await updateExtractButtonVisibility();
        } catch (e) { 
            console.error("Stop failed:", e); 
        }
    }
});

// --- Browser Auto ---
btnAutoLaunch?.addEventListener("click", async () => { 
    if (isBrowserRunning || isAutoLaunchLocked) return;

    try { 
        isAutoLaunchLocked = true; // 🌟 런칭 락 활성화 (무식하게 전부 무시 시작)
        isBrowserRunning = true; 
        
        if (btnAutoLaunch) {
            btnAutoLaunch.style.display = "none";
            btnAutoLaunch.classList.add("hidden");
        }
        
        console.log(`[WIDGET] UI LOCKED: Chrome Launching...`);
        await invoke("launch_best_browser", { url: "about:blank" }); 
        console.log(`[WIDGET] UI LOCKED: Waiting for Rust signal...`);
    } catch (e) { 
        console.error("Launch failed:", e); 
        isAutoLaunchLocked = false; // 🌟 에러 시에만 락 해제
        isBrowserRunning = false;
        syncBrowserStatus();
    } 
});
const autoBrowser = document.getElementById("auto-browser") as HTMLSelectElement;
const autoUrl = document.getElementById("auto-url") as HTMLInputElement;
const autoBtn = document.getElementById("auto-btn") as HTMLButtonElement;

async function initBrowserDropdown() {
    if (!autoBrowser) return;
    try {
        const browsers = await invoke<any[]>("check_available_browsers");
        autoBrowser.innerHTML = "";
        browsers.forEach(b => {
            const opt = document.createElement("option");
            opt.value = b.name; opt.text = b.name + (b.needs_driver ? " (No Driver)" : "");
            autoBrowser.appendChild(opt);
        });
    } catch (e) { console.error("Dropdown error:", e); }
}

autoBtn?.addEventListener("click", async () => {
    if (!autoBrowser || !autoUrl) return;
    try { await invoke("launch_browser", { browser: autoBrowser.value, url: autoUrl.value, script: "" }); } catch (e) { console.error("Manual launch error:", e); }
});

listen("browser-status", async (event: any) => {
    const payload = event.payload; 
    const statusStr = typeof payload === "object" ? payload.status : payload;
    
    if (statusStr === "running") {
        isBrowserRunning = true;
        // 🌟 [CRITICAL FIX] 정상 실행 신호가 오더라도 락을 해제하지 않고 앱 종료 때까지 무조건 숨김을 유지합니다.
        if (btnAutoLaunch) {
            btnAutoLaunch.style.display = "none";
            btnAutoLaunch.classList.add("hidden");
        }
    } else {
        // 🌟 [CRITICAL FIX] isAutoLaunchLocked 조건을 제거합니다.
        //    작업 진행 중(extractClickLock=true, isAutoLaunchLocked=true) 브라우저가 종료되면
        //    이 조건 때문에 btnAutoLaunch 노출 로직이 실행되지 않는 버그를 수정합니다.
        //    첫 번째 browser-status 리스너에서 이미 isAutoLaunchLocked=false로 설정하지만,
        //    이벤트 리스너 실행 순서 보장이 없으므로 여기서도 무조건 리셋합니다.
        console.log("[WIDGET] Browser stopped. Resetting UI.");
        isBrowserRunning = false;
        isAutoLaunchLocked = false;
        if (btnAutoLaunch) {
            btnAutoLaunch.style.display = "flex";
            btnAutoLaunch.classList.remove("hidden");
        }
        currentDetectedUrl = "";
    }
    await updateExtractButtonVisibility();
});

// --- List Logic (Updated for Cards) ---
listRefreshBtn?.addEventListener("click", refreshList);

btnDeleteSelected?.addEventListener("click", async () => {
    if (selectedUuids.size === 0) return;
    if (await ask(`Delete ${selectedUuids.size} documents?`, { title: "Confirm Delete", kind: "warning" })) {
        try {
            const uuids = Array.from(selectedUuids);
            await invoke("delete_documents", { uuids });
            // 🌟 [DEXIE DELETE] 다중 삭제도 Dexie 캐시에서 동기 제거합니다.
            if (appDb && uuids.length > 0) {
                try {
                    await appDb.table("items").bulkDelete(uuids);
                    await appDb.table("users").bulkDelete(uuids);
                    await appDb.table("pages").bulkDelete(uuids);
                    console.log(`[WIDGET] Dexie cache cleared for ${uuids.length} item(s)`);
                } catch (dexieErr) {
                    console.warn("[WIDGET] Dexie bulk delete failed (non-critical):", dexieErr);
                }
            }
            await refreshList();
            updateResultCount();
        } catch (e) { console.error(e); }
    }
});

btnSyncQr?.addEventListener("click", async () => {
    const qrContainer = document.getElementById("nav-qr-container");
    const navOverlay = document.getElementById("nav-categories");

    if (!qrContainer || !navOverlay) return;

    if (navOverlay.classList.contains("hidden")) {
        handleSearchInteraction();
    }

    const isHidden = qrContainer.classList.contains("hidden");
    if (isHidden) {
        qrContainer.classList.remove("hidden");
        if (btnSyncQr) btnSyncQr.innerText = "CLOSE"; // 🌟 [추가] 패널이 열리면 CLOSE로 변경
        await initSyncUI(); // [NEW] Initialize IP/Seed view
        listCurrentY = 0;
        updateListTransform(true);
    } else {
        qrContainer.classList.add("hidden");
        if (btnSyncQr) btnSyncQr.innerText = "ADD"; // 🌟 [추가] 패널이 닫히면 ADD로 원상복구
    }
});

// 🌟 [추가] Cloud Member 초대 패널 토글 로직 (Local Member와 동일한 구조)
document.getElementById("btn-cloud-invite-toggle")?.addEventListener("click", () => {
    const inviteContainer = document.getElementById("nav-cloud-invite-container");
    const btn = document.getElementById("btn-cloud-invite-toggle");
    const pageList = document.getElementById("nav-list-pages");
    
    if (!inviteContainer || !btn) return;

    if (inviteContainer.classList.contains("hidden")) {
        inviteContainer.classList.remove("hidden");
        btn.innerText = "CLOSE";
        listCurrentY = 0;
        updateListTransform(true);
    } else {
        inviteContainer.classList.add("hidden");
        btn.innerText = "ADD";
        // 🌟 [추가] 패널 닫을 때 선택된 클래스 일괄 제거 (기획 의도에 따라 생략 가능)
        if (pageList) {
            pageList.querySelectorAll(".logis-label.selected").forEach(el => el.classList.remove("selected"));
        }
    }
});

// 🌟 [추가] Cloud Member 초대 전송 이벤트 등록
document.getElementById("btn-send-invite")?.addEventListener("click", async () => {
    await handleTeamInvite();
});

// [NEW] Manual Connect Handler
document.getElementById("btn-manual-connect")?.addEventListener("click", async () => {
    const tSeed = (document.getElementById("target-seed") as HTMLInputElement).value;
    const btn = document.getElementById("btn-manual-connect") as HTMLButtonElement;
    
    if (!tSeed) {
        alert("Please enter target seed!");
        return;
    }

    // 1. 현재 PC의 전체 IP를 가져옵니다 (예: 192.168.45.115)
    const myFullIp = await invoke<string>("get_my_full_ip"); 
    const ipParts = myFullIp.split('.');
    
    if (ipParts.length !== 4) {
        alert("Could not determine local network subnet.");
        return;
    }

    // 2. 앞의 3자리만 잘라서 서브넷 베이스를 만듭니다 (예: 192.168.45)
    const baseIp = `${ipParts[0]}.${ipParts[1]}.${ipParts[2]}`; 
    const seed = parseInt(tSeed);

    console.log(`[SYNC] Auto-Scanning subnet ${baseIp}.1~254 with seed ${seed}...`);
    btn.innerText = "SCANNING...";
    btn.disabled = true;

    try {
        await startWebRtcOfferer(baseIp, seed);
    } catch (e) {
        alert("Connection failed. Device not found on this Wi-Fi network.");
    } finally {
        btn.innerText = "AUTO CONNECT";
        btn.disabled = false;
    }
});

// 🌟 [수정] 병렬 스캔 WebRTC 연결 함수 (이전 답변과 동일, 혹시 몰라 전체 첨부)
async function startWebRtcOfferer(baseIp: string, seed: number) {
    peerConn = new RTCPeerConnection({ iceServers: [] });
    dataChannel = peerConn.createDataChannel("logis-sync");
    setupDataChannel(dataChannel);
    
    const offer = await peerConn.createOffer();
    await peerConn.setLocalDescription(offer);
    
    // Wait for ICE gathering
    await new Promise<void>(resolve => {
        if (peerConn?.iceGatheringState === 'complete') resolve();
        else {
            const check = () => { if (peerConn?.iceGatheringState === 'complete') { peerConn?.removeEventListener('icegatheringstatechange', check); resolve(); } };
            peerConn?.addEventListener('icegatheringstatechange', check);
            setTimeout(resolve, 2000);
        }
    });

    const sdp = peerConn.localDescription?.sdp || "";
    
    // 🌟 병렬 연결 시도 (1.1부터 1.254까지 전부 핑을 날려서 가장 먼저 받는 놈과 연결)
    const scanPromises = [];
    for (let i = 1; i <= 254; i++) {
        const targetIp = `${baseIp}.${i}`;
        scanPromises.push(
            invoke<string>("send_signal_offer", { targetIp, seed, sdp })
                .then(answerSdp => ({ targetIp, answerSdp }))
        );
    }

    try {
        const result = await (Promise as any).any(scanPromises);
        await peerConn.setRemoteDescription({ type: 'answer', sdp: result.answerSdp });
        console.log(`[SYNC] Connected to ${result.targetIp} successfully via Auto Scan!`);
    } catch (e) {
        peerConn.close();
        throw new Error("Scan failed");
    }
}

listen("webrtc-offer", async (event) => {
    const [offerSdp, fromIp] = event.payload as [string, string];
    console.log(`[SYNC] Incoming offer from ${fromIp}`);
    
    peerConn = new RTCPeerConnection({ iceServers: [] });
    peerConn.ondatachannel = (e) => setupDataChannel(e.channel);

    await peerConn.setRemoteDescription({ type: 'offer', sdp: offerSdp });
    const answer = await peerConn.createAnswer();
    await peerConn.setLocalDescription(answer);

    // [FIXED] Send Answer back via TCP stream through the backend
    try {
        await invoke("submit_signal_answer", { targetIp: fromIp, sdp: answer.sdp });
        console.log(`[SYNC] Answer submitted for ${fromIp}`);
    } catch (e) {
        console.error("[SYNC] Failed to submit answer:", e);
    }
});

let mySyncSeed = 0; 
let isListenerStarted = false; // 🌟 [추가] 리스너 중복 실행 방지용 플래그

async function initSyncUI() {
    // 🌟 [CRITICAL FIX] 시드 번호를 기기별로 고정(Fix)하기 위해 로컬 DB에서 불러오거나 최초 1회만 생성하여 저장합니다.
    if (mySyncSeed === 0) {
        const savedSeed = await kvGet("my_sync_seed");
        if (savedSeed) {
            mySyncSeed = parseInt(savedSeed);
        } else {
            // 🌟 [수정] 4자리 난수(1000~9999) 대신 2자리 난수(10~99)를 생성합니다.
            mySyncSeed = Math.floor(10 + Math.random() * 90);
            await kvSet("my_sync_seed", mySyncSeed.toString());
        }
    }

    const mySyncSeedEl = document.getElementById("my-sync-seed");
    const ipPrefixEl = document.getElementById("ip-prefix");

    if (mySyncSeedEl) {
        mySyncSeedEl.innerText = mySyncSeed.toString();
    }
    if (ipPrefixEl) {
        const prefix = await invoke("get_local_network_prefix") as string;
        ipPrefixEl.innerText = prefix + ".";
    }
    
    try {
        // 🌟 [CRITICAL FIX] 아직 리스너가 열리지 않았을 때만 딱 한 번 실행하도록 차단합니다.
        if (!isListenerStarted) {
            await invoke("start_listener_command", { seed: mySyncSeed });
            isListenerStarted = true;
        }
    } catch (e) { console.error(e); }
}

let peerConn: RTCPeerConnection | null = null;
let dataChannel: RTCDataChannel | null = null;
let desktopStream: MediaStream | null = null;
let qrRotationInterval: number | null = null;

// 🌟 [추가] 양측의 인증(검증)이 완료된 후 실제 데이터 동기화를 시작하는 헬퍼 함수
function finalizeWebRtcConnection(guestSession: any) {
    const profileName = document.getElementById("nav-profile-name");
    if (profileName) {
        profileName.textContent = "✅ Mobile Linked (P2P)";
        profileName.style.color = "#4ade80";
    }
    document.getElementById("nav-qr-container")?.classList.add("hidden");
    syncDataToMobile();

    try {
        const guestName = (guestSession && guestSession.email) ? guestSession.email.split('@')[0] : "📱 Linked Device";
        const guestAddr = (guestSession && guestSession.address) ? guestSession.address : "0x0000000000000000000000000000000000000000";

        const mobileUser = {
            id: `mobile_${Date.now()}`,
            type: "user",
            name: guestName,
            from: guestAddr, 
            to: currentSession.team || "0x0000000000000000000000000000000000000000",    
            // 🌟 [BOOL PARITY] IndexedDB 는 boolean 을 키로 인정하지 않습니다.
            //    저장 시점부터 0|1 정수로 확정해야 data.is_device 인덱스가 실제로 동작합니다.
            data: { origin: "local", is_device: 1 } 
        };
        
        invoke("upsert_items", { items: [mobileUser] }).then(() => renderNavigation());
    } catch (e) {
        console.error("[WebRTC] Failed to add device to members:", e);
    }
}

function setupDataChannel(channel: RTCDataChannel) {
    channel.onopen = async () => {
        console.log("[WebRTC] Channel OPEN! Starting Zero-Trust Auth Handshake...");
        // 🌟 [핵심 1] 채널이 열리면 데이터를 즉시 붓지 않고, 내 세션(신분증)을 보내 통성명을 시작합니다.
        channel.send(JSON.stringify({ 
            type: "auth_request", 
            session: currentSession 
        }));
    };

    channel.onmessage = async (e) => {
        try {
            const msg = JSON.parse(e.data);
            console.log("[WebRTC] Received from Peer:", msg.type);
            
            // 🌟 [핵심 2] 상대방이 인증을 요청해옴 (내가 Host/수신자 역할일 때)
            if (msg.type === "auth_request") {
                const guest = msg.session;
                
                // a. 이미 클라우드 팀원인지 내 로컬 DB(LanceDB, 클라우드 동기화됨)에서 조회
                const users = await Select["users"]({});
                const isCloudMember = users.some(u => 
                    (u.id === guest.address || u.from === guest.address) &&
                    (u.to === currentSession.team || u.cc === currentSession.team)
                );

                if (isCloudMember) {
                    // 이미 클라우드에서 인증된 팀원이면 즉시 승인 및 동기화
                    console.log("[WebRTC] Guest is an authorized Cloud Member. Auto-approving.");
                    channel.send(JSON.stringify({ type: "auth_success" }));
                    finalizeWebRtcConnection(guest);
                } else {
                    // 🌟 [CRITICAL FIX] 시드 충돌 감지 및 0-멤버 자동 양보(Yield) 로직!
                    // 상대방(Guest)의 소속 팀과 내(Host) 소속 팀이 명확히 다른데 연결이 들어왔다면, 100% 시드 중복입니다.
                    if (guest.team && currentSession.team && guest.team !== currentSession.team) {
                        const myTeamMembers = users.filter(u => u.to === currentSession.team || u.cc === currentSession.team);
                        
                        // 내 팀에 나 혼자(1명 이하)밖에 없다면(초대한 멤버가 없다면) 내가 양보하고 시드를 바꿉니다.
                        if (myTeamMembers.length <= 1) {
                            console.warn("[WebRTC] Seed collision detected! I have no members. Auto-regenerating my seed...");
                            channel.send(JSON.stringify({ type: "auth_reject", reason: "Seed collision. Auto-yielding." }));
                            peerConn?.close();
                            
                            // 시드 강제 재생성 및 로컬 DB 영구 저장
                            // 🌟 [수정] 충돌 시 새로 부여받는 시드도 2자리 난수(10~99)로 통일합니다.
                            mySyncSeed = Math.floor(10 + Math.random() * 90);
                            await kvSet("my_sync_seed", mySyncSeed.toString());
                            
                            const mySyncSeedEl = document.getElementById("my-sync-seed");
                            if (mySyncSeedEl) mySyncSeedEl.innerText = mySyncSeed.toString();
                            
                            // 🌟 Rust 바인딩된 리스너의 시드만 초고속으로 업데이트 (10048 에러 없음!)
                            await invoke("start_listener_command", { seed: mySyncSeed });
                            
                            alert(`[Network] 동일한 와이파이 내에 시드 번호 충돌이 감지되었습니다.\n멤버가 없는 현재 PC의 시드가 새 번호(${mySyncSeed})로 자동 변경 및 양보되었습니다.`);
                            return;
                        } else {
                            // 내 팀에 멤버가 있다면, 상대방이 양보하도록 거절만 날려줍니다.
                            console.warn("[WebRTC] Seed collision detected, but I have members. Rejecting guest.");
                            channel.send(JSON.stringify({ type: "auth_reject", reason: "Wrong team. Please regenerate your seed." }));
                            peerConn?.close();
                            return;
                        }
                    }

                    // b. 충돌이 아니라 정상적인 외부 기기 연결이라면 화면에 팝업을 띄워 수동 승인 진행
                    const displayId = guest.email || guest.address || "Unknown Local Device";
                    const approved = await ask(`Incoming connection from '${displayId}'.\nAre you sure you want to approve this device and share local data?`, { title: "Peer Approval Required", kind: "warning" });
                    
                    if (approved) {
                        console.log("[WebRTC] Connection manually approved by peer.");
                        channel.send(JSON.stringify({ type: "auth_success" }));
                        finalizeWebRtcConnection(guest);
                        // [Option] 필요시 여기서 proxy_fetch를 날려 클라우드 DB에도 guest.address를 강제로 등록(PUT)시킬 수 있습니다.
                    } else {
                        console.log("[WebRTC] Connection rejected by peer.");
                        channel.send(JSON.stringify({ type: "auth_reject", reason: "Rejected by Team Member" }));
                        peerConn?.close();
                    }
                }
            } 
            // 🌟 [핵심 3] 상대방이 내 접속을 승인함 (내가 Guest/발신자 역할일 때)
            else if (msg.type === "auth_success") {
                console.log("[WebRTC] Auth Approved by Host Peer!");
                finalizeWebRtcConnection(null);
            }
            // 🌟 [핵심 4] 상대방이 내 접속을 거절함
            else if (msg.type === "auth_reject") {
                alert(`WebRTC Connection blocked: ${msg.reason}`);
                peerConn?.close();
            }
            // --- 기존 통신 로직 유지 ---
            else if (msg.type === "get_detail") {
                const doc = await invoke<any>("get_document", { uuid: msg.uuid });
                if (doc && dataChannel?.readyState === "open") {
                    dataChannel.send(JSON.stringify({
                        type: "sync_detail",
                        title: `${doc.doc_type || 'Detail'} ${doc.doc_number || ''}`,
                        content: `<div style="margin-bottom:15px;"><strong>Summary:</strong><br>${doc.text}</div><hr style="border-color:rgba(255,255,255,0.1);"><pre style="white-space: pre-wrap; font-size: 0.8rem; color:#fff; background:#000; padding:15px; border-radius:8px;">${doc.json_data}</pre>`
                    }));
                }
            } else if (msg.type === "get_session") {
                // Send current desktop session info to mobile
                if (dataChannel?.readyState === "open") {
                    dataChannel.send(JSON.stringify({ 
                        type: "sync_session", 
                        data: currentSession 
                    }));
                }
            } else if (msg.type === "get_navigation") {
                // Fetch pages and users for mobile tree
                const pages = await Select["pages"]({});
                const users = await Select["users"]({});
                if (dataChannel?.readyState === "open") {
                    dataChannel.send(JSON.stringify({ 
                        type: "sync_navigation", 
                        pages: pages,
                        users: users
                    }));
                }
            } else if (msg.type === "get_chat_history") {
                // Fetch last 20 messages for mobile
                const messages = await invoke<any[]>("get_chat_messages", { limit: 20, offset: 0 });
                if (dataChannel?.readyState === "open") {
                    dataChannel.send(JSON.stringify({ 
                        type: "sync_chat_history", 
                        messages: messages
                    }));
                }
            } else if (msg.type === "search") {
                // Perform local search for mobile
                console.log("[WebRTC] Remote Search Query:", msg.query);
                const docs = await Select["items"]({ 
                    value: msg.query || "", 
                    limit: 20, 
                    offset: 0 
                });
                if (dataChannel?.readyState === "open") {
                    dataChannel.send(JSON.stringify({ type: "sync_list", data: docs }));
                }
            } else if (msg.type === "chat_message") {
                // Echo for now, or could integrate with actual AI chat logic
                dataChannel?.send(JSON.stringify({ 
                    type: "sync_chat", 
                    data: { role: "system", content: "Hub: Received '" + msg.content + "'" } 
                }));
            } else if (msg.type === "mobile_upload") {
                console.log("[WebRTC] Receiving file from mobile:", msg.name);
                try {
                    // 1. Convert Base64 to Uint8Array
                    const binaryString = atob(msg.data);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }

                    // 2. Save to a temporary location using Tauri FS
                    // We'll use a specific name to identify mobile uploads
                    const tempPath = `mobile_upload_${Date.now()}_${msg.name}`;
                    const fullPath = await invoke<string>("save_mobile_temp_file", { 
                        filename: tempPath, 
                        data: Array.from(bytes) 
                    });

                    console.log("[WebRTC] Saved mobile upload to:", fullPath);

                    // 3. Trigger Desktop's existing Extraction Logic
                    const taskId = `task_mobile_${Date.now()}`;
                    await emit("new-task-from-browser", { 
                        id: taskId, 
                        type: "image_extraction", 
                        image_path: fullPath, 
                        ref: fullPath, 
                        link: "Mobile Upload",
                        device_preference: getDevicePref()
                    });

                    // 4. Relay progress to mobile
                    // (We'll handle this in the global progress listener below)

                } catch (err) {
                    console.error("[WebRTC] Mobile upload failed:", err);
                }
            }
        } catch (err) {
            console.error("[WebRTC] Message handle error:", err);
        }
    };
}

// --- Relay Desktop Progress to Mobile ---
listen("extraction-progress", (event: any) => {
    if (dataChannel && dataChannel.readyState === "open") {
        dataChannel.send(JSON.stringify({
            type: "extraction_progress",
            payload: event.payload
        }));
    }
});


const syncDataToMobile = () => {
    if (!dataChannel || dataChannel.readyState !== "open") return;
    console.log("[WebRTC] Syncing list to mobile...");
    const docs = Array.from(document.querySelectorAll('.logis-result')).map(el => {
        const card = el as HTMLElement;
        return {
            id: card.id, uuid: card.id,
            doc_type: card.dataset.type || "General",
            text: card.querySelector('.logis-info .value')?.textContent || "",
            created_at: parseInt(card.dataset.createdAt || "0"),
            updated_at: parseInt(card.dataset.updatedAt || "0")
        };
    });
    dataChannel.send(JSON.stringify({ type: "sync_list", data: docs }));
};

listen("app_error_alert", async (event: any) => {
    const payload = event.payload as any;
    // 🌟 Settings 탭 자동 열기 + 다운로드 시작
    if (payload.action === "open_settings") {
        // 1. 리스트 탭 열기 (설정 패널은 리스트 탭 내부에 있음)
        openWidget("list");
        // 2. Settings 패널 내 체크박스 켜기 (설정 패널 보이게)
        const toggle = document.getElementById("settings-toggle") as HTMLInputElement;
        if (toggle) {
            if (!toggle.checked) {
                toggle.checked = true;
            }
            toggle.dispatchEvent(new Event("change"));
        }
        // 3. 모델 목록 렌더링 후 다운로드 시작
        if (payload.model) {
            console.log(`[AUTO-DL] ${payload.model} 자동 다운로드 시작...`);
            try {
                await invoke("download_model", { modelName: payload.model });
                console.log(`[AUTO-DL] ${payload.model} 다운로드 명령 전송 완료`);
            } catch (e) {
                console.error(`[AUTO-DL] ${payload.model} 다운로드 실패:`, e);
            }
        }
    } else {
        // 기본 폴백: 기존 alert 동작
        alert(payload.message || "알 수 없는 오류가 발생했습니다.");
    }
});

listen("task-console-log", async (event: any) => {
    const { task_id, text } = event.payload;
    const key = `term_${task_id}`;
    
    // 🌟 localStorage -> Dexie(appDb) 로 영구 보존!
    let logs = (await kvGet(key)) || "";
    logs += text;
    await kvSet(key, logs);

    const termArea = document.getElementById("terminal-logs");
    if (termArea && termArea.dataset.activeTaskId === task_id) {
        termArea.appendChild(document.createTextNode(text));
        termArea.style.display = "block"; // 🌟 [추가] 텍스트가 도착하면 까만 박스를 보여줍니다!
        termArea.scrollTop = termArea.scrollHeight; 
    }
});

async function handleTaskClick(el: HTMLElement) {
    const taskId = el.dataset.taskId;
    const status = parseInt(el.dataset.status || "0");
    if (!taskId) return;
    
    console.log("[Chat] Task clicked:", taskId);

    if (taskId.startsWith("search_") && status !== 1) {
        openWidget("list");
        listView.style.display = "block";
        detailView.style.display = "none";
        return;
    }

    openWidget("list"); 
    listView.style.display = "none"; 
    detailView.style.display = "flex";
    
    if (status === 1) {
        if (btnStopTask) btnStopTask.style.display = "flex";
    } else {
        if (btnStopTask) btnStopTask.style.display = "none";
    }
    if (btnDetailDelete) btnDetailDelete.style.display = "none";

    detailTitle.innerText = taskId.startsWith("search_") ? "Search Progress" : "Task Progress";
    
    let logArea = document.getElementById("extraction-log");
    if (!logArea) {
        detailContent.innerHTML = `<div id="extraction-log"></div>`;
        logArea = document.getElementById("extraction-log");
    }

    if (logArea) {
        logArea.dataset.activeTaskId = taskId;
        
        const savedLogs = await kvGet(`term_${taskId}`);
        // 🌟 저장된 로그가 있을 때만 박스를 보여주고, 없으면 숨깁니다. (Connecting... 텍스트 제거)
        const displayStyle = savedLogs && savedLogs.trim() !== "" ? "block" : "none"; 
        
        logArea.innerHTML = `
            <div id="progress-container"></div>
            <div id="terminal-logs" data-active-task-id="${taskId}" style="display: ${displayStyle}; background: #0a0a0a; color: #4ade80; padding: 12px; font-family: monospace; font-size: 0.8rem; border-radius: 6px; max-height: 250px; overflow-y: auto; white-space: pre-wrap; border: 1px solid #333; box-shadow: inset 0 0 10px rgba(0,0,0,0.8); line-height: 1.4;">${savedLogs || ""}</div>
        `;
        
        const termArea = document.getElementById("terminal-logs");
        if (termArea && displayStyle === "block") termArea.scrollTop = termArea.scrollHeight;
        
        isFetchingLogs = true;
        pendingLiveEvents = [];

        invoke<any[]>("get_task_logs", { taskId: taskId }).then(async logs => {
            if (logArea!.dataset.activeTaskId !== taskId) {
                isFetchingLogs = false;
                return;
            }

            // 🌟 로컬 스토리지엔 없지만 백엔드에 로그가 남아있을 경우 복구하면서 박스를 노출합니다!
            if (!savedLogs && logs && logs.length > 0 && termArea) {
                const reconstructed = logs.map(l => `[${l.category ? l.category.toUpperCase() : 'SYSTEM'}] ${l.summary || ''}\n`).join("");
                if (reconstructed.trim() !== "") {
                    termArea.innerHTML = reconstructed;
                    termArea.style.display = "block"; // 숨겨뒀던 박스 노출!
                    await kvSet(`term_${taskId}`, reconstructed); 
                    termArea.scrollTop = termArea.scrollHeight;
                }
            }
            
            if (logs && logs.length > 0) {
                logs.forEach(payload => {
                    payload.task_id = payload.task_id || taskId; 
                    renderProgressToUI(payload, true);
                });
            } else if (status === 1) {
                const progContainer = document.getElementById("progress-container");
                if (progContainer) progContainer.insertAdjacentHTML('beforeend', `<div id="temp-spinner" style="padding: 10px; text-align: center; color: var(--primary);"><span class="spinner active-spinner">⠋</span> Generating Insights...</div>`);
            }

            if (status === 1 || status === 10) {
                const live = livePayloads.get(taskId);
                if (live) {
                    live.task_id = taskId;
                    renderProgressToUI(live, true);
                }
            }

            isFetchingLogs = false;
            pendingLiveEvents.forEach(p => renderProgressToUI(p, false));
            pendingLiveEvents = [];

        }).catch(err => {
            console.error(err);
            isFetchingLogs = false;
        });
    }
    
    activeTaskId = taskId; 
}

async function sendSignalingMessage(hash: string, payload: any) {
    try {
        await invoke("proxy_fetch", {
            url: `${API_HOST}/relay/${hash}`,
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: payload // payload is already JSON object or will be stringified
        });
    } catch (e) {
        console.error("[WebRTC] Relay send failed:", e);
    }
}

// --- WebRTC SDP Template for Compact Handshake ---
const SDP_TEMPLATE = `v=0
o=- {{sessId}} 2 IN IP4 {{ip}}
s=-
t=0 0
a=group:BUNDLE 0
a=msid-semantic: WMS
m=application 9 UDP/DTLS/SCTP webrtc-datachannel
c=IN IP4 {{ip}}
a=ice-ufrag:{{ufrag}}
a=ice-pwd:{{pwd}}
a=fingerprint:sha-256 {{fingerprint}}
a=setup:{{setup}}
a=mid:0
a=sctp-port:5000
a=max-message-size:262144`;

function extractSdp(sdp: string) {
    return {
        u: sdp.match(/a=ice-ufrag:(.*)/)?.[1] || "",
        p: sdp.match(/a=ice-pwd:(.*)/)?.[1] || "",
        f: sdp.match(/a=fingerprint:sha-256 (.*)/)?.[1] || "",
        s: sdp.match(/o=- (\d+) /)?.[1] || "0"
    };
}

function buildSdp(type: 'offer' | 'answer', ip: string, u: string, p: string, f: string, s: string) {
    return SDP_TEMPLATE
        .replace(/{{sessId}}/g, s)
        .replace(/{{ip}}/g, ip)
        .replace(/{{ufrag}}/g, u)
        .replace(/{{pwd}}/g, p)
        .replace(/{{fingerprint}}/g, f)
        .replace(/{{setup}}/g, type === 'offer' ? 'actpass' : 'active');
}

async function showPcPairingQr() {
    const qrTarget = document.getElementById("sync-qrcode");
    const pcView = document.getElementById("pc-qr-view");
    const mobileView = document.getElementById("mobile-scan-view");
    
    if (!qrTarget || !pcView || !mobileView) return;
    
    // Clear existing interval if any
    if (qrRotationInterval) {
        clearInterval(qrRotationInterval);
        qrRotationInterval = null;
    }

    pcView.classList.remove("hidden");
    mobileView.classList.add("hidden");
    stopDesktopCamera();

    qrTarget.innerHTML = "<div style='padding:20px;'><div class='spinner'></div><p>Generating P2P Offer...</p></div>";

    try {
        // 0. Get Local IP
        const myIp = await invoke<string>("get_my_full_ip");

        // 1. Initialize PeerConnection (No STUN for local only)
        peerConn = new RTCPeerConnection({ iceServers: [] });
        
        // 2. Create Data Channel (Must create before offer)
        dataChannel = peerConn.createDataChannel("logis-sync");
        setupDataChannel(dataChannel);

        // 3. Create Offer
        const offer = await peerConn.createOffer();
        await peerConn.setLocalDescription(offer);

        // 4. Wait for ICE Gathering (Essential for LAN connection)
        console.log("[WebRTC] Gathering ICE candidates (5s)...");
        await new Promise<void>(resolve => {
            if (peerConn?.iceGatheringState === 'complete') {
                resolve();
            } else {
                const check = () => {
                    if (peerConn?.iceGatheringState === 'complete') {
                        peerConn?.removeEventListener('icegatheringstatechange', check);
                        resolve();
                    }
                };
                peerConn?.addEventListener('icegatheringstatechange', check);
                setTimeout(resolve, 5000); // 5s timeout
            }
        });

        // Add 1 second stability delay
        await new Promise(r => setTimeout(r, 1000));

        // 5. Generate QR Data (Multipart/Chunked)
        const finalSdp = peerConn.localDescription?.sdp || "";
        const laptopHash = currentSession.hash;
        
        // [Relay] Also post to relay server so mobile can find us without scan next time
        sendSignalingMessage(laptopHash, { type: "offer", sdp: finalSdp });

        const parts = extractSdp(finalSdp);
        const compactOffer = { t: "offer", h: laptopHash, i: await invoke("get_my_full_ip"), u: parts.u, p: parts.p, f: parts.f, s: parts.s };
        const qrData = JSON.stringify(compactOffer);
        
        console.log(`[WebRTC] Offer Generated. Compact Length: ${qrData.length}`);

        // 6. Show Single QR
        qrTarget.innerHTML = ""; 
        const header = document.createElement("div");
        header.style.marginBottom = "10px";
        header.style.fontWeight = "bold";
        header.style.color = "var(--primary)";
        header.innerText = `Scan to Pair (P2P)`;
        qrTarget.appendChild(header);

        const qrDiv = document.createElement("div");
        qrTarget.appendChild(qrDiv);

        new (window as any).QRCode(qrDiv, {
            text: qrData,
            width: 250, height: 250, 
            colorDark: "#000000", colorLight: "#ffffff",
            correctLevel: (window as any).QRCode.CorrectLevel.M
        });
        // Clean up interval when view changes
        const cleanup = () => {
            if (qrRotationInterval) clearInterval(qrRotationInterval);
            document.getElementById("btn-switch-to-camera")?.removeEventListener("click", cleanup);
        };
        document.getElementById("btn-switch-to-camera")?.addEventListener("click", cleanup);

    } catch (e) {
        console.error("[WebRTC] Offer Generation Failed:", e);
        qrTarget.innerHTML = "<p style='color:red'>Failed to gen offer</p>";
    }
}

btnDetailDelete?.addEventListener("click", async () => {
    console.log("[WIDGET] Delete button clicked. UUID:", currentDetailUuid);
    if (!currentDetailUuid) {
        console.error("[WIDGET] No document UUID selected for deletion.");
        return;
    }
    try {
        const confirmed = await ask("Are you sure you want to delete this document?", {
            title: "Confirm Delete",
            kind: "warning"
        });
        if (confirmed) {
            console.log("[WIDGET] Deletion confirmed for:", currentDetailUuid);
            const res = await invoke<string>("delete_document", { uuid: currentDetailUuid });
            console.log("[WIDGET] Delete response:", res);
            // 🌟 [DEXIE DELETE] LanceDB 삭제 후 프론트엔드 Dexie 캐시에서도 제거합니다.
            //    이 처리가 없으면 refreshList() → loadMoreDocs() 가 Dexie 를 조회할 때
            //    해당 아이템이 여전히 존재하여 화면에 다시 렌더링됩니다.
            if (appDb && currentDetailUuid) {
                try {
                    await appDb.table("items").delete(currentDetailUuid);
                    await appDb.table("users").delete(currentDetailUuid);
                    await appDb.table("pages").delete(currentDetailUuid);
                    console.log(`[WIDGET] Dexie cache cleared for: ${currentDetailUuid}`);
                } catch (dexieErr) {
                    console.warn("[WIDGET] Dexie delete failed (non-critical):", dexieErr);
                }
            }
            // 🌟 [ITEM TOMBSTONE] 서버 D1 에 행이 남아 있으면 3초 폴링이 재삽입하므로
            //    묘비를 세워 영구 차단합니다.
            await addItemTombstone(currentDetailUuid);
            detailView.style.display = "none";
            listView.style.display = "block";
            await refreshList();
            updateResultCount();
        }
    } catch (e) {
        console.error("[WIDGET] Deletion process failed:", e);
    }
});

async function refreshList() {
    currentPage = 0; hasMore = true; cachedDocs = []; selectedUuids.clear();
    listCurrentY = 0; // Reset scroll
    if(docListContainer) docListContainer.innerHTML = "";
    await loadMoreDocs(true);
}

async function loadMoreDocs(reset: boolean = false, isSync: boolean = false) {
    // 🌟 [CRITICAL FIX] AI 검색 중이거나 검색 결과가 화면에 고정된 상태에서는
    // 백그라운드 자동 동기화(isSync)나 스크롤에 의한 일반 리스트 덮어쓰기가 난입하여 카운트가 23개 등으로 뻥튀기되는 경합(Race Condition)을 완벽 차단합니다!
    const resultH3 = document.querySelector('.nav-section.search h3');
    const isShowingSearchResult = resultH3 && resultH3.textContent?.toLowerCase().includes("search");
    
    if (isSearching || isShowingSearchResult) {
        // 단, 사용자가 검색창을 지우고 강제 초기화(reset=true)를 요청한 경우는 정상 목록을 불러와야 하므로 예외 처리합니다.
        if (reset && !isSearching) {
            isLoading = false;
        } else {
            return;
        }
    }

    if (reset) {
        currentPage = 0; hasMore = true;
        if (docListContainer) docListContainer.innerHTML = "";
        cachedDocs = [];
        listCurrentY = 0;
        // 🌟 [TOTAL COUNT] 스코프가 바뀌었으므로 총계를 초기화합니다.
        totalResultCount = -1;
        updateListTransform();
        // 🌟 [CRITICAL FIX] 검색어가 지워지는 등 새로운 초기화 요청이 들어오면, 기존에 대기 중이던 로딩 락(isLoading)을 강제로 해제하여 먹통 현상을 방지합니다.
        isLoading = false; 
    }

    if (isLoading || (!reset && !isSync && !hasMore)) {
        if (reset && !isSync) stopSpinner();
        return;
    }

    if (!isSync) startSpinner();
    isLoading = true;
    
    if (headerLoading) {
        headerLoading.style.display = "inline-block";
    }
    
    try {
        // 🌟 [LIST QUERY v4] 목록 조회는 LanceDB 를 거치지 않고 Dexie 에서 직접 처리합니다.
        //  목록에는 벡터/FTS 가 필요 없고, 필요한 건 스코프 + 타입 + 정렬 + 페이징뿐입니다.
        //  → SQL 문자열 조립이 사라지므로 DataFusion 문법 에러 클래스가 통째로 소멸합니다.
        //  → 텍스트 검색이 있을 때만 LanceDB(search_documents)를 후보 소스로 사용합니다.

        // 🌟 [READ SCOPE] 파일 상단의 TYPE_SETS 단일 정의를 그대로 씁니다.
        //  ── 왜 지역 정의를 없앴는가 ──
        //   기존에는 이 목록과 syncData 의 TRADING_TYPES 가 별도로 하드코딩되어
        //   한쪽만 고치면 '태깅한 mode' 와 '조회하는 type' 이 어긋났습니다.
        //   이제 쓰기(modeOfType)와 읽기(TYPE_SETS)가 같은 파일 상단에서 관리됩니다.
        const allowedTypes = TYPE_SETS[currentSearchMode] || TYPE_SETS.commerce;

        const textQuery = searchInput?.value.trim() || "";
        const currentOffset = isSync ? 0 : currentPage * pageSize;

        // [TIMESTAMPS] Scan UI for current range
        let latestUpdateTime = 0;
        const allCards = docListContainer.querySelectorAll('.logis-result');
        allCards.forEach(el => {
            const up = parseInt((el as HTMLElement).dataset.updatedAt || "0");
            if (up > latestUpdateTime) latestUpdateTime = up;
        });

        console.log(`[DEBUG-LIST] 🔍 문서 조회 시작 | mode=${currentSearchMode} | isSync=${isSync} | offset=${currentOffset}`);
        console.log(`[DEBUG-LIST] 활성 컨텍스트:`, JSON.stringify(activeContext));

        let docs: any[] = [];

        if (textQuery) {
            // ── 텍스트 검색 : LanceDB 로 후보를 긁고 Dexie 로 스코프 검증 ──
            //   스코프는 봉투 컬럼만 담습니다. (sanitize_scope_filter 가 어차피 걸러냄)
            let scopeSql = `mode = '${currentSearchMode}'`;
            if (activeContext.ref) scopeSql += ` AND \`ref\` = '${activeContext.ref}'`;
            else if (activeContext.bcc) scopeSql += ` AND bcc = '${activeContext.bcc}'`;
            else if (activeContext.cc) scopeSql += ` AND cc = '${activeContext.cc}'`;

            const searchResults = await invoke<any[]>("search_documents", {
                query: textQuery,
                limit: pageSize * 4,
                offset: 0,
                filter: scopeSql
            });

            const ids = searchResults.map((r: any) => r[0]).filter(Boolean);
            if (ids.length > 0 && appDb) {
                const rows = await appDb.table('items').where('id').anyOf(ids).toArray();
                // LanceDB 점수 순서를 보존합니다.
                const orderMap = new Map<string, number>();
                ids.forEach((id: string, i: number) => orderMap.set(id, i));
                rows.sort((a: any, b: any) => (orderMap.get(a.id) ?? 999) - (orderMap.get(b.id) ?? 999));
                docs = rows.filter((r: any) => allowedTypes.includes(r.type));
            }

            // Dexie 에 아직 없는 문서는 Rust 에서 직접 가져옵니다. (최초 진입 대비)
            if (docs.length === 0 && ids.length > 0) {
                for (const id of ids.slice(0, pageSize)) {
                    const fullDoc = await invoke<any>("get_document", { uuid: id });
                    if (fullDoc) docs.push(fullDoc);
                }
            }
            // 🌟 [TOTAL COUNT] slice 이전의 전체 매칭 건수를 기록합니다.
            if (!isSync) totalResultCount = docs.length;
            docs = docs.slice(currentOffset, currentOffset + pageSize);

        } else if (appDb) {
            // ── 일반 목록 : Dexie 컬렉션 체이닝 ──
            let coll: any;

            // 가장 좁은 스코프 인덱스를 드라이버로 선택합니다.
            if (activeContext.ref) {
                coll = appDb.table('items').where('ref').equals(activeContext.ref);
            } else if (activeContext.bcc) {
                coll = appDb.table('items').where('bcc').equals(activeContext.bcc);
            } else if (activeContext.cc) {
                coll = appDb.table('items').where('cc').equals(activeContext.cc);
            } else {
                coll = appDb.table('items').where('mode').equals(currentSearchMode);
            }

            let rows: any[];

            // 🌟 [COMPOUND DRIVER] 스코프(ref/bcc/cc)가 없을 때는 mode 만으로 훑지 말고
            //    선언해 둔 '[mode+type]' 복합 인덱스를 anyOf 로 펼칩니다.
            //    shipping 은 allowedTypes 가 15종이 넘어 mode 단독 스캔 대비 체감 차이가 큽니다.
            if (!activeContext.ref && !activeContext.bcc && !activeContext.cc) {
                const pairs = allowedTypes.map(t => [currentSearchMode, t]);
                rows = await appDb.table('items').where('[mode+type]').anyOf(pairs).toArray();
                console.log(`[DEBUG-LIST] 복합 인덱스 [mode+type] anyOf ${pairs.length}쌍 → ${rows.length}건 적재`);
            } else {
                rows = await coll.toArray();
                // 스코프 드라이버가 mode 가 아니었다면 mode 를 추가 검증합니다.
                rows = rows.filter((r: any) => (r.mode || 'commerce') === currentSearchMode);
                rows = rows.filter((r: any) => allowedTypes.includes(r.type));
            }

            if (isSync && latestUpdateTime > 0) {
                rows = rows.filter((r: any) => (r.updated_at || 0) > latestUpdateTime);
            }

            rows.sort((a: any, b: any) => (b.created_at || 0) - (a.created_at || 0));

            // 🌟 [TOTAL COUNT] slice 이전의 '스코프 전체 건수' 를 기록합니다.
            //    isSync(상단 당김 갱신)는 최신 델타만 가져오므로 총계를 갱신하지 않습니다.
            if (!isSync) totalResultCount = rows.length;

            console.log(`[DEBUG-LIST] Dexie 스코프 조회: ${rows.length}건 (allowedTypes=${allowedTypes.length}종)`);
            docs = isSync ? rows.slice(0, pageSize) : rows.slice(currentOffset, currentOffset + pageSize);

            // 🌟 [COLD START] Dexie 가 비어 있으면 Rust 에서 끌어와 캐시를 채웁니다.
            //  (앱 최초 실행 / DB 초기화 직후 경로)
            if (rows.length === 0 && currentPage === 0) {
                let scopeSql = `mode = '${currentSearchMode}'`;
                if (activeContext.ref) scopeSql += ` AND \`ref\` = '${activeContext.ref}'`;
                else if (activeContext.bcc) scopeSql += ` AND bcc = '${activeContext.bcc}'`;
                else if (activeContext.cc) scopeSql += ` AND cc = '${activeContext.cc}'`;

                const fromRust = await invoke<any[]>("get_all_documents", {
                    limit: pageSize * 5,
                    offset: 0,
                    filter: scopeSql
                });
                if (fromRust.length > 0) {
                    console.log(`[DEBUG-LIST] ❄️ Cold start: Rust 에서 ${fromRust.length}건 적재 후 Dexie 캐시 채움`);
                    await appDb.table("items").bulkPut(normalizeEnvelope(fromRust)).catch(() => null);
                    const coldRows = normalizeEnvelope(fromRust)
                        .filter((r: any) => allowedTypes.includes(r.type));
                    // 🌟 [TOTAL COUNT] Cold start 경로도 slice 이전 값을 총계로 씁니다.
                    if (!isSync) totalResultCount = coldRows.length;
                    docs = coldRows.slice(0, pageSize);
                }
            }
        }

        console.log(`[DEBUG-LIST] 📥 조회된 문서 개수: ${docs.length}`);
        if (docs.length === 0) {
            console.warn(`[DEBUG-LIST] ⚠️ 데이터가 없습니다. 스코프가 좁거나 해당 타입 데이터가 없습니다.`);
        }

        // 🌟 [CACHE WARM] Rust 경유로 들어온 문서를 Dexie 봉투 형태로 정규화해 저장합니다.
        if (appDb && docs.length > 0) {
            try {
                await appDb.table("items").bulkPut(normalizeEnvelope(docs));
            } catch (e) {
                console.error("[Dexie] Local cache update failed:", e);
            }
        }

        // 🌟 [CRITICAL FIX] 데이터를 불러오는 동안 사용자가 검색어를 변경했거나 지웠다면, 과거 데이터가 화면에 렌더링되어 혼선을 주는 것을 즉시 차단합니다.
        if (textQuery !== (searchInput?.value.trim() || "")) {
            return;
        }

        // 🌟 [CRITICAL FIX] 백그라운드에서 대기하던 일반 리스트 로딩이 끝났을 때,
        // 이미 AI 검색이 진행 중이거나 검색 결과가 화면에 렌더링된 상태(H3 태그가 Search로 변경됨)라면,
        // 일반 문서 5개가 검색 결과 18개 밑에 강제로 들러붙어 23개로 뻥튀기되는 경합(Race Condition)을 완벽 차단합니다!
        const currentH3 = document.querySelector('.nav-section.search h3');
        const currentlyShowingSearch = currentH3 && currentH3.textContent?.toLowerCase().includes("search");
        if ((isSearching || currentlyShowingSearch) && !textQuery && !isSync) {
            console.log(`[SEARCH-DEBUG] 일반 리스트 백그라운드 로딩이 완료되었으나, 현재 검색 결과가 활성화되어 있어 덮어쓰기를 원천 차단합니다.`);
            return;
        }

        if (!isSync && docs.length < pageSize) hasMore = false;

        if (docs.length > 0) {
            const mode = isSync ? 'prepend' : 'append';
            upsertListItems(docs, mode);
            
            // 🌟 [CRITICAL FIX] 문서가 성공적으로 추가되었으므로 페이지 카운터를 정상적으로 증가시킵니다.
            if (!isSync) {
                if (reset) currentPage = 1;
                else currentPage++;
            }
            
            if (isSync) {
                renderNavigation();
            }
        } else if (reset) {
            docListContainer.innerHTML = `<div class="empty">No documents found.</div>`;
        }
    } catch (e) { 
        console.error("[WIDGET] loadMoreDocs error:", e);
        if (reset && docListContainer) docListContainer.innerHTML = `<div style='text-align:center; padding:20px; color:#ef4444;'>Error loading data.</div>`;
    } 
    finally { 
        isLoading = false; 
        
        // 🌟 로딩 종료: Loading... 글자를 완전히 숨김 (아무것도 표시 안 됨)
        if (headerLoading) {
            headerLoading.style.display = "none";
        }
        
        if (!isSync) stopSpinner();

        // 🌟 [추가] 최초 로딩 및 일반 리스트 동기화 후 카운트 반영
        updateResultCount();
    }
}

// 🌟 [TOTAL COUNT] 화면에 렌더링된 카드 수가 아니라 '스코프/검색 전체 건수' 를 보관합니다.
//    -1 = 아직 집계되지 않음(→ DOM 카운트로 폴백)
let totalResultCount = -1;

// 🌟 [추가] 리스트 결과 개수 카운트 업데이트 헬퍼 함수
function updateResultCount() {
    const h3El = document.querySelector('.nav-section.search h3');
    if (h3El && h3El.textContent?.includes("searching")) {
        return; // 검색 중일 때는 카운트 업데이트 무시
    }
    const countEl = document.querySelector('.nav-section.search h3 strong.count');
    if (countEl) {
        const rendered = document.querySelectorAll('#doc-list .logis-result').length;
        // 🌟 페이징으로 몇 장을 그렸든 상관없이 전체 건수를 표기합니다.
        const total = totalResultCount >= 0 ? totalResultCount : rendered;
        console.log(`[COUNT] 전체 ${total}건 / 현재 렌더링 ${rendered}건`);
        countEl.textContent = total > 0 ? `(${total})` : "";
    } else {
        console.log(`[SEARCH-DEBUG] DOM 업데이트 실패: H3 카운트 요소(strong.count)를 찾을 수 없습니다.`);
    }
}

function upsertListItems(docs: any[], mode: 'prepend' | 'append') {
    if (!docListContainer) return;

    const scrollEl = document.getElementById("list-scroll");
    const prevScrollHeight = scrollEl ? scrollEl.scrollHeight : 0;
    const wasAtTop = listCurrentY <= 10; 

    const sortedBatch = [...docs].sort((a, b) => b.created_at - a.created_at);
    const processBatch = mode === 'prepend' ? [...sortedBatch].reverse() : sortedBatch;

    processBatch.forEach(doc => {
        const docId = doc.id || doc.uuid || (doc.data && (doc.data.id || doc.data.uuid)) || doc.uuid_val || doc.ref || doc.index;
        const existingEl = docListContainer.querySelector(`[id="${docId}"]`) as HTMLElement;

        // 🌟 [CRITICAL FIX] item2html은 숨겨진 checkbox와 메인 카드(div) 2개의 요소를 생성합니다.
        const html = item2html(doc, false, currentDetectedUrl);
        const temp = document.createElement('div');
        temp.innerHTML = html;
        
        // 🌟 클래스 이름(.logis-result)이 누락되거나 충돌하는 상황을 원천 차단하기 위해 
        // 부여된 ID 값을 이용해 가장 확실하게 두 요소를 뜯어옵니다.
        const newCheckbox = temp.querySelector(`input#more-${docId}`) as HTMLElement || temp.querySelector('.toggle-more') as HTMLElement;
        const newCard = temp.querySelector(`div[id="${docId}"]`) as HTMLElement || temp.querySelector('.logis-result') as HTMLElement;

        if (existingEl) {
            const cachedUpdatedAt = parseInt(existingEl.dataset.updatedAt || "0");
            if (doc.updated_at > cachedUpdatedAt) {
                console.log(`[List] Updating item ${docId}`);
                
                // 체크박스와 카드를 각각 찾아서 안전하게 교체(Replace)합니다.
                const oldCheckbox = docListContainer.querySelector(`#more-${docId}`);
                if (oldCheckbox && newCheckbox) docListContainer.replaceChild(newCheckbox, oldCheckbox);
                
                if (newCard) {
                    docListContainer.replaceChild(newCard, existingEl);
                    bindCardEvents(newCard, doc);
                }
            }
        } else {
            // 새 카드를 삽입할 때도 체크박스와 카드를 순서대로 온전히 다 넣습니다.
            if (mode === 'prepend') {
                if (newCard) docListContainer.prepend(newCard);
                if (newCheckbox) docListContainer.prepend(newCheckbox);
            } else {
                if (newCheckbox) docListContainer.appendChild(newCheckbox);
                if (newCard) docListContainer.appendChild(newCard);
            }
            if (newCard) bindCardEvents(newCard, doc);
        }
    });

    if (mode === 'prepend' && scrollEl) {
        const newScrollHeight = scrollEl.scrollHeight;
        const heightDiff = newScrollHeight - prevScrollHeight;
        if (heightDiff > 0) {
            if (wasAtTop) listCurrentY = 0;
            else listCurrentY += heightDiff;
            updateListTransform();
        }
    }
}

function bindCardEvents(el: HTMLElement, doc: any) {
    const toggleCheckbox = el.querySelector('.toggle-more') as HTMLInputElement;
    const moreContent = el.querySelector('.more-content') as HTMLElement;
    const moreLabel = el.querySelector('.more-label') as HTMLElement;
    const relateContainer = el.querySelector('.logis-relate') as HTMLElement;

    // 🌟 [PARITY] 클라우드의 Relay(관계 병합) 아코디언 토글 이벤트
    if (toggleCheckbox && moreContent && moreLabel) {
        toggleCheckbox.addEventListener('change', async () => {
            if (toggleCheckbox.checked) {
                // 아코디언 열림
                moreContent.style.display = "block";
                moreLabel.innerHTML = "fold ▲";
                
                // 🌟 열릴 때 연관된 데이터(Foreign/Primary)를 DB에서 긁어와 병합합니다!
                if (relateContainer) {
                    await loadRelatedData(doc, relateContainer);
                }
            } else {
                // 아코디언 닫힘
                moreContent.style.display = "none";
                moreLabel.innerHTML = "more ▼";
            }
        });
    }

    el.addEventListener("click", (e) => {
        const target = e.target as HTMLElement;
        
        // 아코디언 래퍼나 내부 연관 데이터 클릭 시, 메인 상세 페이지로 넘어가지 않도록 차단
        if (target.closest('.toggle-more') || target.closest('.more-label') || target.closest('.more-content') || target.closest('.logis-relate')) {
            return;
        }

        const docId = doc.id || doc.uuid || (doc.data && (doc.data.id || doc.data.uuid)) || doc.uuid_val || doc.ref || doc.index;
        if (!target.closest('a') && !target.closest('input') && !target.closest('button')) {
            if (docId) showDetail(String(docId));
        }
    });
}

// 🌟 [PARITY] 클라우드 Relay 로직의 클라이언트 사이드 이식
// 🌟 [PARITY] 클라우드 Relay 로직의 클라이언트 사이드 이식
async function loadRelatedData(doc: any, container: HTMLElement) {
    if (!container || container.dataset.loaded === "true") return;
    // 스피너 표시
    container.innerHTML = `<div style="padding:10px; text-align:center; font-size:0.8rem; color:var(--primary);"><span class="active-spinner">⠋</span> Loading related data...</div>`;
    try {
        const docId = doc.id || doc.uuid;
        const docRef = doc.ref;
        // 🌟 v5 : 연관 조회를 3단계 인덱스 기반으로 처리합니다.
        //   ① ref 인덱스        : 기존 경로 유지
        //   ② 정방향 (정방향)   : 내 문서의 rel_* 값 = 상대 문서의 data.index
        //   ③ 역방향 (역방향)   : 상대 문서의 rel_* 값 = 내 문서의 data.index
        //   세 경로 모두 Dexie 인덱스 O(log n) 입니다.
        let uniqueDocs: any[] = [];
        if (appDb) {
            // ── ① ref 인덱스 기반 (기존 경로 유지) ──
            const refTargets = [docId];
            if (docRef && docRef !== "") refTargets.push(docRef);
            const refRows = await appDb.table('items').where('ref').anyOf(refTargets).limit(20).toArray();
            for (const r of refRows) {
                if (r.id !== docId && !uniqueDocs.some(d => d.id === r.id)) {
                    uniqueDocs.push(r);
                }
            }

            // ── ② 정방향 : 내가 참조하는 문서들 ──
            // 내 문서의 rel_* 값 = 상대 문서의 data.index
            // 예: CI 문서의 data.rel_bl = 1234567890
            //     → BL 문서의 data.index = 1234567890 인 문서를 찾음
            const relKeys = Object.keys(doc.data || {}).filter(k => k.startsWith("rel_"));
            for (const relKey of relKeys) {
                const relVal = doc.data?.[relKey];
                if (relVal === undefined || relVal === null) continue;
                const relValNum = Number(relVal);
                if (isNaN(relValNum)) continue;
                try {
                    const revRows = await appDb.table('items')
                        .where('data.index')
                        .equals(relValNum)
                        .limit(5)
                        .toArray();
                    for (const r of revRows) {
                        if (r.id !== docId && !uniqueDocs.some(d => d.id === r.id)) {
                            uniqueDocs.push(r);
                        }
                    }
                } catch (_e) { /* 인덱스 없으면 무시 */ }
            }

            // ── ③ 역방향 : 나를 참조하는 문서들 ──
            // 상대 문서의 rel_* 값 = 내 문서의 data.index
            // 예: BL 문서의 data.rel_ci = 9876543210
            //     → 내 문서의 data.index = 9876543210 이므로
            //     → data.rel_ci = 내 문서의 data.index 인 문서를 찾음
            const myIndex = doc.data?.index;
            if (myIndex !== undefined && myIndex !== null) {
                const myIndexNum = Number(myIndex);
                if (!isNaN(myIndexNum)) {
                    // rel_* 컬럼 중 하나로 나를 참조하는 문서들
                    for (const relKey of relKeys) {
                        try {
                            const relRows = await appDb.table('items')
                                .where(`data.${relKey}`)
                                .equals(myIndexNum)
                                .limit(5)
                                .toArray();
                            for (const r of relRows) {
                                if (r.id !== docId && !uniqueDocs.some(d => d.id === r.id)) {
                                    uniqueDocs.push(r);
                                }
                            }
                        } catch (_e) { /* 인덱스 없으면 무시 */ }
                    }
                }
            }

            uniqueDocs = uniqueDocs.slice(0, 10);
        }
        // Dexie 가 비어 있으면 Rust 로 폴백합니다.
        if (uniqueDocs.length === 0) {
            let filterStr = `\`ref\` = '${docId}'`;
            if (docRef && docRef !== "") {
                filterStr += ` OR \`ref\` = '${docRef}'`;
            }
            const relatedDocs = await invoke<any[]>("get_all_documents", {
                limit: 10,
                offset: 0,
                filter: filterStr
            });
            uniqueDocs = relatedDocs.filter(d => (d.id || d.uuid) !== docId);
        }
        if (uniqueDocs.length > 0) {
            const relatedHtml = uniqueDocs.map(d => {
                // 🌟 하위 아이템은 무한 확장을 막기 위해 checked=true (펼쳐짐) 및 부가 정보 축소 형태로 렌더링
                return item2html(d, true, currentDetectedUrl);
            }).join("");
            // 연관 데이터 UI 주입
            container.innerHTML = `<div style="margin-top:15px; border-top:1px dashed rgba(255,255,255,0.2); padding-top:10px;">
<strong style="font-size:0.8rem; color:#aaa; margin-bottom:10px; display:block;">🔗 Related Documents</strong>
${relatedHtml}
</div>`;
            // 내부 연관 카드의 클릭 이벤트(상세 페이지 진입)도 재귀적으로 바인딩
            const newCards = container.querySelectorAll('.logis-result');
            newCards.forEach((card, idx) => {
                bindCardEvents(card as HTMLElement, uniqueDocs[idx]);
            });
        } else {
            // 연관 데이터가 없으면 깔끔하게 비움
            container.innerHTML = "";
        }
        container.dataset.loaded = "true"; // 불필요한 중복 쿼리 방지 (캐싱)
    } catch (e) {
        console.error("[Relay] Failed to load related data:", e);
        container.innerHTML = `<div style="color:#ef4444; font-size:0.7rem; padding:5px;">Failed to load related data.</div>`;
    }
}

function renderDocs(docs: any[]) {
    // This is now handled by upsertListItems for consistency
    upsertListItems(docs, 'append');
}

async function showDetail(uuid: string) {
    console.log("[WIDGET] Opening detail view for ID:", uuid);
    if (!uuid) {
        console.error("[WIDGET] Cannot open detail: ID is undefined");
        return;
    }
    currentDetailUuid = uuid;
    listView.style.display = "none";
    detailView.style.display = "flex";
    if (btnDetailDelete) btnDetailDelete.style.display = "flex";
    if (btnStopTask) btnStopTask.style.display = "none";

    detailTitle.innerText = "Loading...";
    detailContent.innerHTML = "Fetching details...";
    try {
        const doc = await invoke<any>("get_document", { uuid: uuid });
        if (doc) {
            detailTitle.innerText = `${doc.doc_type || 'Detail'} ${doc.doc_number || ''}`;
            let prettyJson = doc.json_data;
            try { prettyJson = JSON.stringify(JSON.parse(doc.json_data), null, 2); } catch(e) {}
            detailContent.innerHTML = `<div style="margin-bottom:10px;"><strong>Summary:</strong><br>${doc.text}</div><hr style="border-color:#444;"><pre style="white-space: pre-wrap; font-size: 0.8rem; color:#fff; background:#111; padding:10px;">${prettyJson}</pre>`;
        } else {
            detailContent.innerHTML = `<div class="empty">Document not found in database.</div>`;
        }
    } catch (e) { 
        console.error("[WIDGET] get_document failed:", e);
        detailContent.innerHTML = "Failed to load document details: " + e; 
    }
}

btnDetailBack?.addEventListener("click", () => { detailView.style.display = "none"; listView.style.display = "block"; });
document.getElementById("btn-settings-back")?.addEventListener("click", collapseWidget);

// 🌟 [수정] 세팅 패널이 열려있을 때는 세팅을 닫고 리스트로 복귀하며, 일반 리스트 상태일 때는 위젯을 닫습니다.
btnListBack?.addEventListener("click", () => {
    const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
    
    // 🌟 [추가] 검색 결과가 표시된 상태에서 뒤로가기 버튼 클릭 시 위젯을 닫지 않고 전체 리스트로 복구합니다.
    const resultH3 = document.querySelector('.nav-section.search h3');
    const isShowingSearchResult = resultH3 && resultH3.textContent?.toLowerCase().includes("search");
    
    // 🌟 [CRITICAL FIX] 검색이 진행 중(isSearching)일 때는 리스트 초기화를 막습니다.
    if (isShowingSearchResult && !isSearching) {
        if (searchInput) searchInput.value = "";
        if (resultH3) resultH3.innerHTML = `Result <strong class="count"></strong>`;
        refreshList();
        return; // 검색 복구만 수행하고 위젯은 닫지 않음
    } else if (isSearching) {
        // 🌟 진행 중일 때는 화면(결과, 상태)을 보존한 채로 패널만 닫거나 위젯을 축소합니다.
        if (settingsToggle && settingsToggle.checked) {
            settingsToggle.checked = false;
            settingsToggle.dispatchEvent(new Event("change"));
        } else {
            collapseWidget();
        }
        return;
    }

    if (settingsToggle && settingsToggle.checked) {
        settingsToggle.checked = false;
        settingsToggle.dispatchEvent(new Event("change")); // 세팅 패널 닫기 이벤트 트리거
    } else {
        collapseWidget(); // 기존처럼 위젯 닫기
    }
});

document.getElementById("nav-signin")?.addEventListener("click", () => openWidget("settings"));
document.getElementById("nav-signout")?.addEventListener("click", () => { document.getElementById("btn-logout")?.click(); });

async function handleImageUpload(path: string) {
    currentImage = path;
    if (navPreviewContainer && navImgThumbnail) {
        navPreviewContainer.classList.remove("hidden");
        navUploadBtn?.classList.add("active-emoji");
        
        // 🌟 [수정] 이미지 업로드 시 검색창을 막고 버튼을 숨기던 로직을 제거합니다.
        if (searchInput) {
            searchInput.disabled = false;
            if (btnSubmit) {
                const currentVal = searchInput.value.trim();
                if (currentVal !== "" && !isQueryActive(currentVal)) {
                    btnSubmit.style.display = "flex";
                } else {
                    btnSubmit.style.display = "none";
                }
            }
        }
        if (btnExtract) btnExtract.style.display = "flex";
        
        try {
            const ext = path.split('.').pop()?.toLowerCase() || '';
            const isDocument = ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'hwpx', 'txt', 'csv'].includes(ext);

            if (isDocument) {
                // 문서는 아이콘 형태나 텍스트(확장자) 표시
                navImgThumbnail.src = `./assets/doc_icon.svg`; // 문서용 기본 아이콘 (퍼블릭 폴더에 파일이 없을 경우 엑박 및 alt 텍스트 표시됨)
                navImgThumbnail.alt = `Doc: ${ext.toUpperCase()}`;
                navImgThumbnail.style.objectFit = "contain";
                navImgThumbnail.style.background = "#fff";
            } else {
                const contents = await readFile(currentImage);
                const blob = new Blob([contents]);
                const reader = new FileReader();
                reader.onloadend = () => { navImgThumbnail.src = reader.result as string; };
                reader.readAsDataURL(blob);
                navImgThumbnail.style.objectFit = "cover";
                navImgThumbnail.style.background = "transparent";
            }
        } catch (e) { 
            navImgThumbnail.src = convertFileSrc(currentImage); 
        }

        console.log("[WIDGET] File selected. Extraction button (⚡) is now visible.");
        
        // 🌟 [추가] 이미지 선택 시 설정(채팅) 탭으로 화면을 전환하고 스크롤을 맨 아래로 내립니다.
        openWidget("settings");
        setTimeout(() => {
            const scrollEl = document.getElementById("chat-scroll");
            const container = document.querySelector(".chat-container") as HTMLElement;
            if (scrollEl && container) {
                const maxScroll = Math.max(0, scrollEl.scrollHeight - container.clientHeight);
                currentY = maxScroll;
                scrollEl.style.transition = "transform 0.3s ease-out";
                updateTransform();
                setTimeout(() => { scrollEl.style.transition = ""; }, 300);
            }
        }, 100);
    }
}

navImgClear?.addEventListener("click", async () => {
    currentImage = null;
    navPreviewContainer.classList.add("hidden");
    navUploadBtn?.classList.remove("active-emoji");
    
    // 🌟 [유지] 검색창 활성화 및 조건부 검색 버튼 노출을 명시적으로 보장합니다.
    searchInput.disabled = false;
    if (btnSubmit) {
        const currentVal = searchInput.value.trim();
        if (currentVal !== "" && !isQueryActive(currentVal)) {
            btnSubmit.style.display = "flex";
        } else {
            btnSubmit.style.display = "none";
        }
    }
    
    await updateExtractButtonVisibility();
});

navUploadBtn?.addEventListener("click", async () => {
    const file = await open({ 
        multiple: false, 
        filters: [
            { name: 'Supported Files', extensions: ['png', 'jpg', 'jpeg', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'hwpx', 'txt', 'csv'] }
        ] 
    });
    if (file) await handleImageUpload(file as string);
});

const ZERO_ADDRESS = "0x0000000000000000000000000000000000000000";
const timezoneOffset = new Date().getTimezoneOffset() * 60 * 1000;

async function checkAuthStatus() {
    // 🌟 [BOOTSTRAP 허용] 기존에는 hash 가 없으면 즉시 return 했습니다.
    //    이제 hash 발급을 서버에 위임하므로, hash 가 없는 상태에서도
    //    '자격증명 없는 bootstrap 요청' 을 보내 서버가 유효한 쌍을 발급하게 합니다.
    const origin = "https://commerce.logis.center"; 
    const now = Date.now();
    const createdAt = now - timezoneOffset; 
    try {
        // 🌟 [CRITICAL FIX] Tauri의 window.location.href는 'localhost'이므로 서버가 도메인(cc)을 파악하지 못합니다.
        // 브라우저에서 감지된 URL(currentDetectedUrl)이나 기본 클라우드 주소를 전달해야 완벽히 매칭됩니다!
        let targetHref = currentDetectedUrl || "https://commerce.logis.center/tracking";
        if (targetHref.includes("localhost") || targetHref.includes("127.0.0.1") || targetHref === "about:blank") {
            targetHref = "https://commerce.logis.center/tracking";
        }

        const queryParams: Record<string, string> = { 
            origin: origin, 
            created_at: createdAt.toString(), 
            href: targetHref 
        };

        // 🌟 [PAIRED CREDENTIAL] hash 와 token 은 반드시 '함께' 보내야 합니다.
        //  ── 근거 ──
        //   워커의 세션 게이트는 다음 한 줄입니다.
        //     if(cookies.hash || (req.query.hash && req.query.token))
        //   Rust proxy_fetch 는 reqwest 에 쿠키 스토어가 없어 Cookie 헤더를 전혀 보내지 않으므로
        //   cookies.hash 는 항상 빈 값이고, (hash && token) 쌍이 유일한 통과 조건입니다.
        //   게이트를 통과해야만 S3 HEAD 가 실행되어 balance 가 정의되고,
        //   balance 가 undefined 로 남으면 아래 블록이 무조건 발동합니다.
        //     if(typeof balance == "undefined"){ ... ethers.Wallet.createRandom() ... }
        //   즉 hash 만 단독으로 보내는 순간 서버는 100% 새 hash 를 발급하고,
        //   화면의 QR 주소가 바뀝니다. 이것이 '요청할 때마다 주소가 달라지는' 원인입니다.
        //
        //  둘 중 하나라도 없으면 아예 보내지 않고 깨끗하게 재발급받습니다.
        //  (반쪽짜리 자격증명을 보내는 것과 결과가 같으면서, 서버 쪽 로그가 명확해집니다)
        const hasPairedCredential = !!(currentSession.hash && currentSession.token);
        if (hasPairedCredential) {
            queryParams.hash = currentSession.hash;
            queryParams.token = currentSession.token as string;
        } else {
            console.log("[AUTH] 🔑 자격증명 쌍이 없어 bootstrap 요청을 보냅니다. 서버가 (hash, token) 을 새로 발급합니다.");
        }

        // 🌟 [SENDER IMPRINT] Client Worker(index.ts)는 method 와 무관하게
        //    매 요청의 세션 블록에서 아래를 수행합니다.
        //      var sender = req.query.sender ? decodeURIComponent(req.query.sender) : data.sender
        //      if(sender){ data.sender = sender }
        //      if(data.sender){ cookies.sender = data.sender }
        //    그런데 신규 유저 생성 시 user_arr 에는 flag/name/title/region/page_count/favicon 만
        //    들어가고 sender 키가 아예 없어서 cookies.sender 가 영원히 undefined 였습니다.
        //    그 결과 서버의
        //      PUT  : if(cookies.sender){ ... INSERT INTO talks ... }
        //      POST : if(cookies.sender && created_at){ ... INSERT INTO tasks ... }
        //    두 경로가 통째로 죽어, 채팅이 D1 talks 에 단 한 건도 저장되지 않았습니다.
        //    여기서 sender 를 실어 보내면 user row 에 영구 각인되어
        //    이후 모든 요청이 자동으로 통과합니다.
        const senderName = currentSession.email || currentSession.name || "";
        if (senderName) queryParams.sender = senderName;

        const params = new URLSearchParams(queryParams);
        const finalUrl = `${API_HOST}/?${params.toString()}`.toLowerCase();

        // 🌟 [SESSION PARAMS GUARD] proxy_fetch 는 session_params 의 hash/token 을
        //    쿼리에 '한 번 더' append 합니다. (Rust proxy_fetch 의 DETAIL 1 블록)
        //    쌍이 온전하지 않을 때 hash 만 append 되면 위 쿼리 조립을 무력화하므로,
        //    쌍이 있을 때만 넘깁니다.
        const sessionParams = hasPairedCredential
            ? { hash: currentSession.hash, token: currentSession.token }
            : null;

        const sentHash = currentSession.hash || "";

        console.log('sessionParams',sessionParams);

        const data = await invoke<any>("proxy_fetch", { url: finalUrl, method: "GET", headers: { "Content-Type": "application/json" }, session_params: sessionParams });
        
        // [FIX] Step the spinner frame only when result arrives
        stepQrSpinner();

        let session = data.session || data; 
        if (session && session.hash) {
            const hashChanged = session.hash !== currentSession.hash;

            // 🌟 [CREDENTIAL ROTATION DETECT] 온전한 쌍을 보냈는데도 서버가 다른 hash 를 돌려줬다면
            //    S3 의 /hash/{hash} 객체가 사라졌거나 워커 try 블록이 예외로 빠진 것입니다.
            //    조용히 넘어가면 원인 추적이 불가능하므로 반드시 표면화합니다.
            if (hashChanged && hasPairedCredential) {
                console.warn(
                    `[AUTH] ⚠️ 서버가 자격증명을 거부하고 hash 를 회전시켰습니다. ` +
                    `sent='${sentHash}' → issued='${session.hash}'. ` +
                    `(S3 /hash/{hash} 객체 소실 또는 워커 세션 블록 예외 가능성)`
                );
            }

            currentSession = { ...currentSession, ...session };
            await saveSession();

            // 🌟 서버가 응답으로 돌려준 hash 이므로 이제 QR 을 그려도 되는 '살아 있는 주소' 입니다.
            isHashServerConfirmed = true;

            if (hashChanged && !currentSession.email && currentTab === "settings") performQrAuth();

            console.log('currentSession',currentSession);

            if (currentSession.email) {
                // 🌟 [TEAM MIGRATION TRIGGER] initialize_hub 내부에서 ZERO_ADDRESS → 실제 address
                //    마이그레이션이 자동으로 수행됩니다.
                await invoke("initialize_hub", { address: currentSession.address, email: currentSession.email, flag: session.flag || "kr" });
                // 🌟 [OAUTH SYNC] 로그인 직후 서버에서 기존 등록된 사이트 목록을 조회합니다.
                //    등록 여부와 무관하게 호출하며, 응답을 kv_store 에 저장합니다.
                await fetchOAuthRegisteredSites();
                updateAuthUI(); fetchChatHistory(); syncData();
            }
        }
    } catch (e) { 
        console.warn("Auth check failed:", e); 
    }
}

function updateAuthUI() {
    const authStatus = document.getElementById("auth-status-text");
    const btnLogout = document.getElementById("btn-logout");
    const btnQrAuth = document.getElementById("btn-qr-auth");
    const chatForm = document.querySelector(".chat-form") as HTMLElement;
    const cloudToggle = document.getElementById("cloud-mode-toggle") as HTMLInputElement;
    
    // 🌟 [추가] Cloud Members 섹션을 통째로 잡습니다.
    const cloudMembersSection = document.getElementById("nav-list-users")?.closest(".nav-section") as HTMLElement;

    console.log('currentSession',currentSession);

    if (currentSession.email) {
        if (authStatus) authStatus.innerText = "Authenticated";
        if (btnLogout) btnLogout.style.display = "block";
        if (btnQrAuth) btnQrAuth.style.display = "none";
        if (chatForm) chatForm.classList.remove("hidden");
        const qrMsg = document.getElementById("msg-qr-auth");
        if (qrMsg) qrMsg.remove();
        
        if (cloudToggle) {
            cloudToggle.disabled = false;
            cloudToggle.title = "Cloud AI Mode is available";
        }

        // 🌟 [수정] 로그인 성공 시 Cloud Members 영역을 표시하되, 세팅 화면이 켜져있다면 숨김을 유지합니다.
        const isSettingsOpen = (document.getElementById("settings-toggle") as HTMLInputElement)?.checked;
        if (cloudMembersSection) cloudMembersSection.style.display = isSettingsOpen ? "none" : ""; 
        
    } else {
        if (authStatus) authStatus.innerText = "Waiting for Auth...";
        if (btnLogout) btnLogout.style.display = "none";
        if (btnQrAuth) btnQrAuth.style.display = "block";
        if (chatForm) chatForm.classList.add("hidden");
        
        if (cloudToggle) {
            cloudToggle.disabled = true;
            cloudToggle.checked = false;
            cloudToggle.title = "Login required to use Cloud AI";
        }

        // 🌟 [추가] 비로그인 시 Cloud Members 영역 완전히 숨김
        if (cloudMembersSection) cloudMembersSection.style.display = "none"; 
    }
}

let authPollInterval: number | null = null;

// 🌟 [QR IDEMPOTENCY] 마지막으로 QR 캔버스를 그린 hash 값입니다.
//  performQrAuth 는 loadMoreChat 끝에서 조건 없이 호출되기 때문에
//  채팅이 갱신될 때마다 QR 노드를 파괴·재생성하고 있었습니다.
//  같은 hash 라면 다시 그릴 이유가 없으므로 이 값으로 차단합니다.
let renderedQrHash = "";

// 🌟 [SERVER-CONFIRMED HASH] 서버가 실제로 응답으로 돌려준 hash 인지 여부입니다.
//  클라이언트가 로컬에서 만든 임시 hash 는 S3 에 객체가 없어
//  스캔해도 절대 인증되지 않는 '죽은 주소' 입니다.
//  서버가 확인해 준 hash 로만 QR 을 그리기 위해 구분합니다.
let isHashServerConfirmed = false;

function stopAuthPolling() {
    if (authPollInterval) {
        clearTimeout(authPollInterval);
        authPollInterval = null;
    }
}

function startAuthPolling() {
    if (authPollInterval) clearTimeout(authPollInterval);
    const poll = async () => {
        // 인증이 완료되었으면 폴링 중단
        if (currentSession.email) {
            stopAuthPolling();
            return;
        }
        
        // 서버에 세션 인증 상태 확인 요청
        await checkAuthStatus();
        
        // 아직 인증되지 않았으면 3초 후 다시 재귀 요청
        if (!currentSession.email) {
            authPollInterval = window.setTimeout(poll, 3000);
        } else {
            stopAuthPolling();
        }
    };
    
    // 첫 요청은 3초 후 실행
    authPollInterval = window.setTimeout(poll, 3000);
}

async function performQrAuth() {
    if (!chatTalks) return;

    // 🌟 [DEAD ADDRESS GUARD] 서버가 확인해 주지 않은 hash 로는 QR 을 그리지 않습니다.
    //    그런 hash 는 S3 에 객체가 없어 스캔해도 인증 메일이 매칭되지 않는 죽은 주소이며,
    //    잠시 뒤 서버가 새 hash 를 내려주면 화면의 주소가 바뀌어 사용자를 혼란시킵니다.
    //    대신 '준비 중' 안내를 띄우고, checkAuthStatus 가 hash 를 확보하는 즉시
    //    hashChanged 분기가 이 함수를 다시 불러 실제 QR 로 교체합니다.
    if (!currentSession.hash || !isHashServerConfirmed) {
        const placeholderId = "msg-qr-auth";
        if (!document.getElementById(placeholderId)) {
            chatTalks.insertAdjacentHTML('beforeend',
                `<div class="chat-talk system" id="${placeholderId}" data-created-at="9999999999999">
                    <div class="chat-message" style="padding:0; background:#fff; color:#000; border:0;">
                        <div style="font-size:0.8rem; font-weight:bold; color:#333;">
                            <span id="qr-auth-spinner" class="active-spinner" style="margin-right:5px; font-family:monospace; color:#000; font-weight:bold;">⠋</span>Preparing secure session...
                        </div>
                    </div>
                </div>`
            );
        }
        // 서버에서 hash 를 받아와야 하므로 폴링은 반드시 가동합니다.
        startAuthPolling();
        return;
    }

    // 🌟 [IDEMPOTENT RENDER] 같은 hash 로 이미 QR 을 그려 두었다면 다시 그리지 않습니다.
    //    performQrAuth 는 loadMoreChat 끝에서 조건 없이 호출되기 때문에
    //    채팅이 갱신될 때마다 QR 노드를 remove → insert → 캔버스 재생성 하고 있었고,
    //    함수 말미의 startAuthPolling() 이 3초 타이머를 계속 리셋해
    //    인증 상태 확인이 지연되는 부작용까지 있었습니다.
    const alreadyRendered = document.getElementById("qr-code-target");
    if (alreadyRendered && renderedQrHash === currentSession.hash) {
        // 폴링이 꺼져 있을 수 있으므로 그것만 보증하고 즉시 반환합니다.
        if (!authPollInterval && !currentSession.email) startAuthPolling();
        return;
    }

    const existing = document.getElementById("msg-qr-auth");
    if (existing) existing.remove();
    const html = `<div class="chat-talk system" id="msg-qr-auth" data-created-at="9999999999999"><div class="chat-message" style="padding:0; background: #fff; color: #000; border:0;"><div style="font-size:0.8rem; font-weight: bold; margin-bottom: 15px; color: #333;"><span id="qr-auth-spinner" class="active-spinner" style="margin-right:5px; font-family:monospace; color:#000; font-weight:bold;">⠋</span>Scan the QR code</div><div id="qr-code-target" style="display: inline-block; background: #fff; border-radius: 8px;"></div></div></div>`;
    chatTalks.insertAdjacentHTML('beforeend', html);
    const qrTarget = document.getElementById("qr-code-target");
    if (qrTarget) {
        qrTarget.innerHTML = "";
        const mailtoAddr = `mailto:${encodeURIComponent(currentSession.hash + ".logis.center@oauth.email")}`;
        new (window as any).QRCode(qrTarget, { text: mailtoAddr, width: 300, height: 300, colorDark: "#000000", colorLight: "#ffffff", correctLevel: (window as any).QRCode.CorrectLevel.M });

        // 🌟 이 hash 로 그렸다는 사실을 기록해 다음 호출부터 재생성을 차단합니다.
        renderedQrHash = currentSession.hash;
        console.log(`[AUTH] 🔳 QR rendered for server-confirmed hash '${currentSession.hash}'`);

        const scroll = document.getElementById("chat-scroll");
        if (scroll) scroll.scrollTop = scroll.scrollHeight;
    }
    
    // 🌟 QR 코드 노출 후 3초 간격으로 세션 인증 상태 반복 확인 시작
    startAuthPolling();
}

// 🌟 [PARITY] Window Focus/Blur 이벤트 리스너 추가
window.addEventListener("blur", () => {
    isFocus = false;
    if (chatPollInterval) {
        clearTimeout(chatPollInterval);
        chatPollInterval = null;
        console.log("[WIDGET] Window blurred. Polling paused to save resources.");
    }
});

window.addEventListener("focus", () => {
    isFocus = true;
    
    // 🌟 [CRITICAL FIX] 크롬 브라우저를 끄고 앱 화면으로 돌아왔을 때 즉시 브라우저 생존 여부를 검사하여 
    // 브라우저 런처 버튼 노출 및 번개 버튼 상태를 원상복구합니다.
    syncBrowserStatus();
    
    // 🌟 [CRITICAL FIX] 이메일(로그인)이 없는 상태에서도 QR 인증 대기를 위해 폴링이 무조건 재개되어야 합니다!
    if (!chatPollInterval) {
        console.log("[WIDGET] Window focused. Polling resumed.");
        // 창을 다시 봤을 때 즉시 1회 최신화 (로그인 된 상태일 때만)
        if (currentSession.email) {
            fetchChatHistory(false, true); 
        }
        startPolling();
    }
});

// 🌟 [PARITY] startPolling 함수 업그레이드 (setInterval -> 재귀적 setTimeout)
function startPolling() {
    if (chatPollInterval) {
        clearTimeout(chatPollInterval);
        chatPollInterval = null;
    }
    if (!isFocus) return;

    const poll = async () => {
        if (!isFocus) return;

        // 히스토리(Settings) 창이 열려있을 때만 서버에 인증/동기화 요청을 보냅니다!
        if (currentTab === "settings" && isExpanded) {
            try {
                if (!currentSession.email) {
                    await checkAuthStatus();
                } else {
                    // 🌟 [CRITICAL FIX] 로컬 DB만 조회하던 fetchChatHistory 대신,
                    // front.js와 동일하게 실제 서버와 통신하는 syncData를 호출해야 합니다!
                    await syncData();
                }
            } catch (e) {
                console.error("[POLLING] Error during poll:", e);
            }
        } else {
            // 🌟 [ANALYTICS ALWAYS-ON] 채팅 화면이 닫혀 있거나 위젯이 접혀 있어도
            //    analytics 이벤트 수집은 계속되어야 합니다.
            //    (Worker 의 GET 은 로그인 없이 hash 만으로도 동작하지만,
            //     사용자 요구사항이 '로그인 이후에도' 이므로 hash 확보 시점부터 돌립니다)
            //    syncAnalyticsInBackground 내부의 30초 스로틀이 왕복을 억제합니다.
            if (currentSession.hash) {
                try {
                    await syncAnalyticsInBackground();
                } catch (e) {
                    console.error("[POLLING] Analytics background sync error:", e);
                }
            }
        }

        // 🌟 [ADAPTIVE POLLING] 고정 3초 대신 백오프 간격을 적용합니다.
        //    변경이 있으면 3초, 연속 변경 없음이면 4.5초 → 6.75초 → ... → 최대 30초.
        const nextInterval = computeSyncInterval();
        if (isFocus) {
            chatPollInterval = window.setTimeout(poll, nextInterval);
        }
    };

    // 첫 시작 시 현재 간격으로 대기 후 실행
    const initialInterval = computeSyncInterval();
    chatPollInterval = window.setTimeout(poll, initialInterval);
}



async function saveSession() { await kvSet("chat_session", JSON.stringify(currentSession)); }

// 🌟 [추가] Pages 숨김 처리 상태를 담을 전역 배열
let hiddenPages: string[] = [];

async function initSession() {
    // 🌟 [AUTH UI FIRST] 어떤 비동기 초기화가 실패하더라도 인증 UI 는 항상 옳은 상태여야 합니다.
    //    currentSession.email 이 비어 있는 최초 시점에 즉시 호출하면
    //    Sign Out 버튼이 숨겨지고 QR 인증 버튼이 노출됩니다.
    //    (세션 복원 후 아래에서 한 번 더 호출해 최종 상태를 확정합니다)
    updateAuthUI();

    // 🌟 [추가] Dexie에서 숨김 페이지 목록을 불러옵니다.
    const savedHiddenPages = await kvGet("hidden_pages");
    if (savedHiddenPages) {
        try { hiddenPages = JSON.parse(savedHiddenPages); } catch(e) {}
    }

    // 🌟 [TOMBSTONE PRELOAD] 삭제 묘비를 메모리에 먼저 올립니다.
    //    startPolling() 이 첫 syncData 를 쏘기 전에 캐시가 채워져야
    //    앱 재시작 직후 한 번의 폴링 동안 삭제된 메시지가 되살아나는 창이 생기지 않습니다.
    await loadTalkTombstones();

    // 🌟 [CRITICAL FIX 1] 앱 최초 실행 시, Dexie에서 묵은 터미널 찌꺼기 및 30일이 지난 오래된 검색 결과를 완벽 청소합니다!
    const allKeys = await appDb.table("kv_store").toCollection().primaryKeys();
    const nowTimeMs = Date.now();
    // 30일을 밀리초 단위로 계산 (30일 * 24시간 * 60분 * 60초 * 1000)
    const thirtyDaysMs = 30 * 24 * 60 * 60 * 1000;

    for (const key of allKeys) {
        if (typeof key === "string") {
            // 1. 기존 터미널 로그 찌꺼기는 즉시 청소
            if (key.startsWith("term_")) {
                await kvRemove(key);
            }
            // 2. 30일이 지난 과거 검색 결과 가비지 컬렉션 (자동 청소)
            else if (key.startsWith("search_res_search_")) {
                // key 포맷: search_res_search_1715610000000 -> 타임스탬프 숫자만 추출
                const timestampStr = key.replace("search_res_search_", "");
                const timestamp = parseInt(timestampStr, 10);

                // 유효한 숫자인지 확인 후, 30일이 경과했으면 로컬 DB에서 삭제
                if (!isNaN(timestamp) && (nowTimeMs - timestamp > thirtyDaysMs)) {
                    console.log(`[GC] Deleting expired search result (older than 30 days): ${key}`);
                    await kvRemove(key);
                }
            }
        }
    }

    // 🌟 [TRANSLIT CACHE GC] 90일이 지난 음차 캐시 정리.
    //    음차는 원문 값이 바뀌면 키가 달라지므로 자동 무효화되지만,
    //    삭제된 아이템의 잔재가 쌓이는 것을 방지하기 위해 주기적으로 청소합니다.
    try {
        const ninetyDaysMs = 90 * 24 * 60 * 60 * 1000;
        const cutoff = nowTimeMs - ninetyDaysMs;
        const staleCount = await appDb.table("translit_cache")
            .where("created_at")
            .below(cutoff)
            .delete();

        if (staleCount > 0) {
            console.log(`[GC] Deleted ${staleCount} stale translit cache entries (older than 90 days).`);
        }
    } catch (e) {
        console.warn("[GC] translit_cache cleanup failed:", e);
    }
    // 🌟 search_mode 도 여기서 비동기로 불러와 초기화합니다.
    const savedSearchMode = await kvGet("search_mode");
    if (savedSearchMode) {
        currentSearchMode = savedSearchMode;
        applySearchModeUI(); // UI에 즉시 반영
    }

    const saved = await kvGet("chat_session");
    if (saved) { try { currentSession = { ...currentSession, ...JSON.parse(saved) }; } catch (e) {} } 
    else { const legacy = await kvGet("device_hash"); if (legacy) currentSession.hash = legacy; }

    // 🌟 [BOOTSTRAP HASH 폐기]
    //  ── 무엇이 문제였나 ──
    //   기존에는 여기서 ethers.Wallet.createRandom() 으로 임시 hash 를 만들었습니다.
    //   그런데 이 값은
    //     ① S3 의 /hash/{hash} 객체가 존재하지 않고
    //     ② 짝이 되는 token 이 없습니다.
    //   Client Worker(index.ts)의 세션 게이트는
    //     if(cookies.hash || (req.query.hash && req.query.token))
    //   인데, Rust proxy_fetch 는 reqwest 에 쿠키 스토어가 없어 Cookie 헤더를
    //   전혀 보내지 않으므로 워커의 cookies.hash 는 항상 빈 값입니다.
    //   결국 (hash && token) 쌍이 유일한 통과 조건인데 임시 hash 는 token 이 없어
    //   반드시 게이트에서 탈락하고, 그러면
    //     if(typeof balance == "undefined"){ ... createRandom() ... }
    //   가 발동해 서버가 새 hash 를 발급합니다.
    //   즉 이 임시 hash 로 그린 QR 은 '스캔해도 절대 인증되지 않는 죽은 주소' 이고,
    //   서버 응답이 오는 순간 화면의 주소가 바뀌는 원인이었습니다.
    //
    //  ── 해결 ──
    //   hash 발급은 전적으로 서버에 위임합니다.
    //   hash 가 비어 있으면 checkAuthStatus 가 자격증명 없이 bootstrap 요청을 보내고,
    //   서버가 S3 에 PUT 까지 마친 유효한 (hash, token) 쌍을 돌려줍니다.
    //   그 값으로만 QR 을 그리므로 주소가 흔들릴 여지가 사라집니다.
    if (currentSession.hash && currentSession.token) {
        // 저장된 자격증명이 온전한 쌍이면 서버 확인 전까지는 잠정 신뢰합니다.
        isHashServerConfirmed = true;
    } else if (currentSession.hash && !currentSession.token) {
        // 🌟 [ORPHAN CREDENTIAL] hash 만 남고 token 이 유실된 상태입니다.
        //    이 hash 를 그대로 보내면 게이트에서 탈락해 서버가 매번 새 hash 를 발급합니다.
        //    쌍을 깨뜨려 버리고 깨끗한 bootstrap 을 유도하는 편이 안전합니다.
        console.warn(`[AUTH] ⚠️ hash 는 있으나 token 이 없어 세션이 성립하지 않습니다. 폐기 후 서버에서 재발급받습니다. (orphan hash: ${currentSession.hash})`);
        currentSession.hash = "";
        currentSession.token = undefined;
        isHashServerConfirmed = false;
    }

    await saveSession(); 
    currentSession.address = currentSession.address || ZERO_ADDRESS;
    currentSession.team = currentSession.team || await hashId(ZERO_ADDRESS);
    // 🌟 [LOGOUT STATE GUARD] 로그아웃 후 reload 시 currentSession 이 초기화되어
    //    address/team 이 ZERO 로 리셋됩니다. 이 상태에서 syncData 가 호출되면
    //    ZERO 기반 cc 로 서버에 요청하므로, 미로그인 시 sync 를 건너뜁니다.
    //    (initSession 하단의 syncData 호출은 currentSession.email 체크로 이미 방어됨)
    updateAuthUI(); 
    startPolling();

    try {
        console.log("[WIDGET] UI Ready handshake starting...");
        
        // 🌟 1. 새로고침 전 담아두었던 프론트엔드 대기열 먼저 복구 (Dexie 비동기 처리)
        await GlobalTaskManager.loadQueue();
        
        const data = await invoke<any>("mark_ui_ready");

        // 🌟 [Tauri Bridge -> Dexie] 초기 구동 시 백엔드 데이터를 봉투 형태로 정규화해 적재합니다.
        //  ⚠️ 기존 코드의 결함: users / pages 가 normalizeEnvelope 를 거치지 않아
        //     data 객체 없이 json_data 문자열만 들어갔고, 그래서 db.ts 의 Select 가
        //     매번 parseItemData 로 다시 파싱해야 했습니다.
        //     v4 부터 세 테이블 모두 동일하게 정규화합니다.
        try {
            if (data.users && data.users.length > 0) await appDb.table("users").bulkPut(normalizeEnvelope(data.users));
            if (data.pages && data.pages.length > 0) await appDb.table("pages").bulkPut(normalizeEnvelope(data.pages));
            if (data.items && data.items.length > 0) await appDb.table("items").bulkPut(normalizeEnvelope(data.items));
        } catch(dbErr) {
            console.error("[Dexie] Initial sync failed:", dbErr);
        }

        // 🌟 [CRITICAL FIX] 백엔드에서 실제로 실행 중인 작업이 있다면 프론트엔드 큐 매니저를 바쁨(Busy) 상태로 잠급니다!
        // 이렇게 해야 대기열에 있던 검색 작업이 새로고침 즉시 백엔드로 뚫고 들어가는 것을 막을 수 있습니다.
        const runningTask = data.tasks && data.tasks.find((t: any) => t.status === 1);
        if (runningTask) {
            GlobalTaskManager.isBusy = true;
            GlobalTaskManager.currentTaskId = runningTask.id;
            console.log(`[QUEUE] Backend is busy with ${runningTask.id}. Pausing frontend queue.`);
        }

        const currentLockId = await kvGet("sys_lock");
        if (currentLockId) {
            const isTaskStillAlive = data.tasks && data.tasks.some((t: any) => t.id === currentLockId && (t.status === 1 || t.status === 10));
            // 🌟 2. DB엔 없어도 TS Queue에 남아있는 녀석은 아직 Rust로 안 넘어간 정당한 대기열입니다.
            const isPendingInQueue = GlobalTaskManager.queue.some(q => q.taskId === currentLockId);
            if (!isTaskStillAlive && !isPendingInQueue) {
                console.log(`[LOCK] Zombie detected: ${currentLockId} is not active in Backend or Queue. Releasing.`);
                await kvRemove("sys_lock");
                // 🌟 [SESSION PRESERVE] 기존에는 여기서 forceReset() 을 호출하여
                //    kv_store.clear() → chat_session 소실 → 로그인 풀림이 발생했습니다.
                //    Zombie lock 해제는 락과 큐 상태만 정리하면 충분합니다.
                //    세션·설정·묘비 등 사용자 데이터는 건드리지 않습니다.
                GlobalTaskManager.isBusy = false;
                GlobalTaskManager.currentTaskId = null;
                GlobalTaskManager.currentTaskPayload = null;
                GlobalTaskManager.activeRefs.clear();
                GlobalTaskManager.queue = [];
                GlobalTaskManager.backendQueued = [];
                try {
                    await appDb.table("ts_queue").clear();
                } catch (e) {
                    console.warn("[LOCK] ts_queue clear failed:", e);
                }
                console.log("[LOCK] Zombie lock released. Session and settings preserved.");
            } else {
                console.log(`[LOCK] Valid task detected: ${currentLockId}. Keeping lock.`);
                if (currentLockId.startsWith("search_")) isSearching = true;
                else isExtracting = true;
                activeTaskId = currentLockId;
                startSpinner();
            }
        }

        // 🌟 3. 큐 복구 후 밀린 작업이 있다면 자동 재개
        if (GlobalTaskManager.queue.length > 0 && !GlobalTaskManager.isBusy) {
            GlobalTaskManager.processNext();
        }

        // 🌟 4. DOM 청소 시 TS Queue 생존자도 보호
        const allBubbles = chatTalks.querySelectorAll('.task-bubble');
        allBubbles.forEach(el => {
            const bubbleId = el.id;
            const bubbleStatus = parseInt((el as HTMLElement).dataset.status || "0");
            if (bubbleStatus === 1 || bubbleStatus === 10) {
                const existsInDb = data.tasks && data.tasks.some((t: any) => t.id === bubbleId);
                const existsInQueue = GlobalTaskManager.queue.some(q => q.taskId === bubbleId);
                
                if (!existsInDb && !existsInQueue) {
                    console.log(`[UI] Removing zombie bubble from DOM: ${bubbleId}`);
                    el.remove();
                    const queryEl = document.getElementById(`${bubbleId}_query`);
                    if (queryEl) queryEl.remove();
                }
            }
        });

        // 🌟 새로고침 시 DB에 살아남은 진짜 대기열 목록만 복구
        if (data.tasks && data.tasks.length > 0) {
            for (const t of data.tasks) {
                if (t.status === 10 || t.status === 1) {
                    let taskData: any = {};
                    let taskQuery = "";
                    try {
                        taskData = typeof t.data_json === 'string' ? JSON.parse(t.data_json) : t.data_json;
                        taskQuery = taskData.query || "";
                    } catch(e) {
                        console.warn("[WIDGET] Failed to parse task data for query recovery:", e);
                    }

                    // 1. 사용자 질문 말풍선 강제 복구 (100% DB 기반)
                    if (taskQuery) {
                        const userMsgId = `${t.id}_query`;
                        if (!document.getElementById(userMsgId)) {
                            await renderMessage({
                                id: userMsgId,
                                role: "user",
                                text: taskQuery,
                                status: 9,
                                // 🌟 [최종 수정] 시스템 태스크(t.created_at)보다 100ms 앞당겨 정렬 엔진의 충돌을 완벽히 회피합니다.
                                created_at: Number(t.created_at) - 100,
                                updated_at: Number(t.created_at) - 100
                            });
                        }
                    }

                    // 2. 시스템 대기열/진행 상태 말풍선 복구
                    if (!document.getElementById(t.id)) {
                        await renderMessage({
                            id: t.id,
                            task_id: t.id,
                            role: "system_task",
                            text: t.id.startsWith("search_") ? "Task Started: AI Search" : ("Task Started: " + (t.ref || "Local Source")),
                            status: t.status,
                            // 🌟 [핵심 수정] 기준 시간(t.created_at) 그대로 사용하여 질문 뒤에 오게 함
                            created_at: t.created_at,
                            updated_at: t.updated_at
                        });
                    }
                    
                    // 3. 진행 중(1)이거나 대기 중(10)인 작업에 대한 전역 상태 락 설정
                    // 🌟 [CRITICAL FIX] 검색 작업인데 프론트엔드 큐(TS Queue)에 존재하지 않는다면 실행될 가능성이 없는 유령(Ghost)입니다.
                    const isSearchGhost = t.id.startsWith("search_") && !GlobalTaskManager.queue.some(q => q.taskId === t.id);

                    if (!isSearchGhost) {
                        // 🌟 [CRITICAL FIX] 상태가 10(대기)인 작업까지 스피너를 돌리고 활성 작업으로 덮어쓰는 치명적 버그 수정!
                        // 오직 상태가 1(Processing)인 진짜 진행 중인 작업만 UI 락을 걸고 스피너를 돌립니다.
                        if (t.status === 1) {
                            await kvSet("sys_lock", t.id);
                            
                            if (t.id.startsWith("search_")) {
                                isSearching = true;
                                if (btnSubmit) btnSubmit.style.display = "none";
                            } else {
                                isExtracting = true;
                            }
                            activeTaskId = t.id;
                            startSpinner();

                            GlobalTaskManager.isBusy = true;
                            GlobalTaskManager.currentTaskId = t.id;
                            GlobalTaskManager.currentTaskPayload = taskData;
                        } else if (t.status === 10) {
                            // 대기열은 락을 걸지 않고, 오직 버튼 가림막(backendQueued) 목록에만 조용히 추가합니다.
                            taskData.taskId = t.id;
                            GlobalTaskManager.backendQueued.push(taskData);
                            GlobalTaskManager.activeRefs.add(t.id);
                        }
                    } else {
                        console.log(`[WIDGET] Ignoring ghost search task: ${t.id}`);
                    }
                }
            }
            await GlobalTaskManager.saveQueue(); // 🌟 Dexie에 복구된 전체 큐 상태를 영구 저장
            await updateExtractButtonVisibility();
        }

        // 브라우저 런처 상태 동기화
        if (btnAutoLaunch) {
            if (data.browser_status === "running") {
                isBrowserRunning = true;
                // 🌟 [CRITICAL FIX] isAutoLaunchLocked를 여기서 설정하지 않습니다.
                //    try_reconnect_existing_browser가 IS_BROWSER_LAUNCHING을 일시적으로 true로 설정하면
                //    mark_ui_ready가 "running"을 반환하고, isAutoLaunchLocked=true가 고정되어
                //    브라우저가 실제로 없음에도 btnAutoLaunch가 영원히 숨겨지는 버그를 수정합니다.
                //    isAutoLaunchLocked는 btnAutoLaunch 클릭 시에만 true로 설정되어야 합니다.
                btnAutoLaunch.style.display = "none";
                btnAutoLaunch.classList.add("hidden");
            } else {
                // 🌟 [CRITICAL FIX] isAutoLaunchLocked 조건을 제거합니다.
                //    브라우저가 stopped이면 무조건 btnAutoLaunch를 노출합니다.
                isBrowserRunning = false;
                isAutoLaunchLocked = false;
                btnAutoLaunch.style.display = "flex";
                btnAutoLaunch.classList.remove("hidden");
            }
            console.log(`[WIDGET] 🔵 [${new Date().toISOString().split('T')[1].slice(0, -1)}] UI Ready Browser Status: ${data.browser_status}`);
        }

        // 🌟 [CRITICAL FIX] 앱 새로고침 시 백엔드에서 감지 중인 브라우저 현재 URL 상태를 완벽 복구합니다.
        if (data.current_url) {
            currentDetectedUrl = data.current_url;
            isCurrentShop = data.is_client || data.is_admin;
            // 🌟 [CRITICAL FIX] URL 복구 직후 명시적으로 버튼 UI 업데이트 로직을 트리거하여 화면에 즉시 노출되도록 강제
            await updateExtractButtonVisibility();
        }

        // 🌟 [SCHEMA GENERATION CHECK v4]
        //  store.rs 의 init_all_tables 는 schema_v4 컬럼이 없으면 테이블을 통째로 drop 합니다.
        //  구버전 사용자는 앱 실행 직후 LanceDB 가 비어 있게 되므로,
        //  '데이터가 사라진 것처럼 보이는' 상황을 사용자에게 정확히 설명해야 합니다.
        //  판정: Dexie 에는 데이터가 있는데 LanceDB(mark_ui_ready)가 비어 있으면 세대 전환입니다.
        //
        //  🌟 [MULTI-TABLE RESTORE] 기존 구현은 items 만 복구했습니다.
        //     그런데 init_all_tables 는 items / users / pages 세 테이블을 '전부' drop 합니다.
        //     users 가 사라지면 팀 통계(base.pages)가 통째로 날아가 네비게이션 카운트가 0이 되고,
        //     pages 가 사라지면 셀렉터 캐시가 없어져 모든 페이지를 다시 AI 분석해야 합니다.
        //     세 테이블을 동일한 절차로 복구합니다.
        //
        //  🌟 [TABLE HINT] save_item / upsert_items 는 item.table 힌트를 1순위로 신뢰하므로,
        //     복구 페이로드에 table 을 명시해야 users / pages 가 items 로 새어 나가지 않습니다.
        try {
            const RESTORE_TABLES: Array<{ name: string; hint: string; lanceKey: string }> = [
                { name: "items", hint: "items", lanceKey: "items" },
                { name: "users", hint: "users", lanceKey: "users" },
                { name: "pages", hint: "pages", lanceKey: "pages" }
            ];

            let needsRestore = false;
            for (const t of RESTORE_TABLES) {
                const dexieCount = await appDb.table(t.name).count();
                const lanceArr = (data as any)[t.lanceKey];
                const lanceCount = (lanceArr && lanceArr.length) ? lanceArr.length : 0;
                if (dexieCount > 0 && lanceCount === 0) {
                    needsRestore = true;
                    break;
                }
            }

            const alreadyNotified = await kvGet("schema_v4_notified");
            // 🌟 [LOGIN GATE] 로그아웃 상태에서는 복원을 수행하지 않습니다.
            //    로그아웃 후 앱 재시작 시 이전 계정 데이터가 복원되어
            //    로그아웃의 의미가 사라지고, 미로그인 상태에서 서버 데이터가 섞입니다.
            //    (로그 실측: 로그아웃 직후 upsert_items 3건이 SCHEMA RESTORE 에서 발생)
            if (needsRestore && !alreadyNotified && currentSession.email) {
                await kvSet("schema_v4_notified", "true");
                console.warn("[SCHEMA] v4 generation detected. LanceDB was rebuilt; local index needs re-population.");

                for (const t of RESTORE_TABLES) {
                    const allRows = await appDb.table(t.name).limit(5000).toArray();
                    if (allRows.length === 0) continue;

                    const restorePayload = allRows.map((r: any) => ({
                        // 🌟 봉투를 루트에 펼치고 확장은 스프레드합니다.
                        //    (upsert_items 가 루트 평탄화 페이로드를 기대합니다)
                        id: r.id,
                        table: t.hint,
                        type: r.type,
                        flag: r.flag,
                        from: r.from,
                        to: r.to,
                        cc: r.cc,
                        bcc: r.bcc,
                        ref: r.ref,
                        mode: r.mode,
                        created_at: r.created_at,
                        updated_at: r.updated_at,
                        ...(r.data || {})
                    }));

                    console.log(`[SCHEMA] Restoring ${restorePayload.length} '${t.name}' document(s) into LanceDB v4...`);
                    for (let i = 0; i < restorePayload.length; i += 100) {
                        const chunk = restorePayload.slice(i, i + 100);
                        try {
                            await invoke("upsert_items", { items: chunk });
                        } catch (e) {
                            console.warn(`[SCHEMA] restore chunk failed for ${t.name}:`, e);
                        }
                    }
                }

                console.log(`[SCHEMA] ✅ Restore complete. Re-indexing will run in background.`);

                // 🌟 벡터/청크는 다시 만들어야 하므로 로컬 임베딩을 트리거합니다.
                runLocalEmbeddingSync();

                // 🌟 통계/네비게이션이 복구된 users 를 반영하도록 즉시 다시 그립니다.
                await renderNavigation();
            }
        } catch (e) {
            console.warn("[SCHEMA] Generation check skipped:", e);
        }

        // 🌟 [CRITICAL FIX] 렌더링 오염(pages 타입 노출) 해결: 필터링 없이 raw DB 아이템을 무작정 렌더링하던 코드를 삭제합니다.
        // 리스트 렌더링은 하단의 syncData -> loadMoreDocs(false, true) 파이프라인에서
        // Dexie 스코프 체이닝을 거쳐 100% 안전하게 수행됩니다.

        // 🌟 [CRITICAL FIX] Rust(LanceDB)에서 로드한 초기 데이터를 다시 Rust로 덮어쓰는(역동기화) 치명적인 병목 루프를 제거합니다.

        // 🌟 [CRITICAL FIX] 로그인 여부(네트워크 상태)와 무관하게 로컬 DB의 최신 데이터로 화면을 즉시 그려냅니다! (병목 및 렌더링 유격 해소)
        await renderNavigation();

        // 🌟 화면이 렌더링된 후 백그라운드에서 조용히 서버와 통신하여 최신 데이터를 반영합니다.
        if (currentSession.email) {
            console.log("[WIDGET] 로그인 확인됨. 서버 데이터를 백그라운드에서 동기화합니다...");
            syncData(); // await를 제거하여 UI 블로킹 방지
        }

         // 🌟 [CLIENT-SIDE EMBEDDING] 이전 세션에서 클라우드로 받아왔지만 로컬 임베딩이 안 된 아이템을 복구합니다.
         //    runLocalEmbeddingSync 내부에 2초 디바운스가 있으므로 syncData 완료 후 호출과
         //    이 4초 타이머 호출이 겹쳐도 1회만 실행됩니다.
         setTimeout(() => { runLocalEmbeddingSync(); }, 4000);

    } catch (e) { 
        console.error("[WIDGET] Handshake failed:", e); 
    }
}

document.getElementById("btn-qr-auth")?.addEventListener("click", performQrAuth);

document.getElementById("btn-logout")?.addEventListener("click", async () => {
    if (await ask("Are you sure you want to sign out?", { title: "Sign Out", kind: "warning" })) {
        // 1. 메모리상의 세션 데이터 초기화
        currentSession = { hash: "", cc: "logis.center" };
        // 2. Dexie DB 내 저장된 세션 및 터미널 로그 영구 삭제
        await kvRemove("chat_session");
        // 🌟 [LOGOUT CLEANUP] 이전 계정 기반 검색 모드 / 숨김 페이지 / 음차 캐시 정리
        await kvRemove("search_mode");
        await kvRemove("hidden_pages");
        // 🌟 [LOGOUT EMBED RESET] 이전 계정 문서의 embed 플래그를 리셋하지 않습니다.
        //    LanceDB 는 계정 무관 로컬 저장소이므로, 재로그인 시
        //    initialize_hub 의 migrate_team_identity 가 to 필드만 갱신합니다.
        //    embed/청크는 그대로 유지하여 재임베딩 비용을 방지합니다.
        // 3. sessionStorage 및 기타 상태 초기화
        sessionStorage.clear();
        // 🌟 [LOGOUT BACKEND RESET] 백엔드 모델 메모리 해제
        try {
            await invoke("unload_model");
        } catch (_e) { /* 이미 해제된 경우 무시 */ }
        // 4. 앱 강제 새로고침하여 초기 상태(새 해시 생성 등)로 복귀
        window.location.reload();
    }
});

// 🌟 [추가] Dexie DB 초기화 및 앱 리셋 버튼 로직
document.getElementById("btn-reset-db")?.addEventListener("click", async () => {
    if (await ask("정말 로컬 데이터베이스를 초기화하시겠습니까?\n모든 로컬 큐 데이터와 캐시가 삭제되며 앱이 재시작됩니다.", { title: "Initialize Local DB", kind: "warning" })) {
        try {
            // 🌟 [RESET STEP 0] 백엔드 활성 태스크 및 스케줄러 중단
            //    reset_lancedb 전에 반드시 호출해야 합니다.
            //    스케줄러 백그라운드 워커가 이미 픽업한 태스크가
            //    테이블 drop 이후에도 upsert_item 을 시도하거나,
            //    reindex_pending_embeddings 가 빈 테이블에 임베딩을
            //    계속 쓰는 것을 방지합니다.
            //    taskId: null → 전체 정리 모드 (모든 pending 태스크 폐기)
            await invoke("stop_current_extraction", { taskId: null }).catch(() => null);
            console.log("[RESET] Backend extraction stopped and cancellation token set.");

            // 🌟 [RESET STEP 1] 백엔드 모델 및 스토어 완전 언로드
            //    모델이 로드된 상태에서 reset 하면
            //    deep_purge_resources 가 미호출되어 VRAM 이 잔존하고,
            //    재시작 후 모델 재로드 시 이전 KV 캐시 참조 오류가 발생할 수 있습니다.
            await invoke("unload_model").catch(() => null);
            console.log("[RESET] Backend model and store unloaded.");

            // 🌟 [RESET STEP 2] 프론트엔드 폴링 / 스케줄링 타이머 정리
            //    syncData 폴링(3초)이 reset 직후에도 upsert_items 를 호출하면
            //    방금 비운 테이블에 데이터가 다시 채워집니다.
            //    reindex 디바운스 타이머(2초)도 동일하게 임베딩을 재실행합니다.
            if (chatPollInterval) {
                clearTimeout(chatPollInterval);
                chatPollInterval = null;
            }
            stopAuthPolling();
            isCommerceSyncRunning = false;
            if (reindexDebounceTimer) {
                clearTimeout(reindexDebounceTimer);
                reindexDebounceTimer = null;
            }
            reindexScheduled = false;
            isReindexing = false;
            console.log("[RESET] All frontend polling and scheduling timers cleared.");

            // 3. 프론트엔드 전역 상태 초기화
            await GlobalTaskManager.forceReset();
            isExtracting = false;
            isSearching = false;
            stopSpinner();
            cachedDocs = [];
            currentPage = 0;
            hasMore = true;
            selectedUuids.clear();
            activeTags = [];
            activeContext = { cc: "", bcc: "", ref: "" };
            if (docListContainer) docListContainer.innerHTML = "";
            if (chatTalks) chatTalks.innerHTML = "";
            // 4. 백엔드 LanceDB 완전 초기화 (tasks, talks, items, sales, tracking, event, users, pages 전부 drop & recreate)
            await invoke("reset_lancedb");
            console.log("[RESET] LanceDB backend reset complete.");
            // 5. 프론트엔드 Dexie DB 완전 삭제 후 재생성
            await appDb.delete();
            await appDb.open();
            console.log("[RESET] Dexie DB deleted and reopened.");
            // 🌟 v4 : 세대 전환 안내 플래그도 함께 초기화합니다.
            //    (전체 초기화 후에는 복구할 원본이 없으므로 안내가 다시 뜨면 안 됩니다)
            await kvRemove("schema_v4_notified");
            // 6. 세션 스토리지 초기화 (새로고침 후 큐 자동 재실행 방지)
            sessionStorage.clear();
            // 7. 앱 강제 새로고침
            window.location.reload();
        } catch (e) {
            console.error("DB Initialization failed:", e);
            alert("DB 초기화 중 오류가 발생했습니다: " + e);
        }
    }
});

// 🚀 모델 관리 UI 렌더링 엔진
async function updateModelStatusUI() {
    try {
        modelStatus = await invoke("check_model_status");
    } catch (e) {}

    const container = document.getElementById("model-list-container");
    if (!container) return;
    container.innerHTML = "";

    TARGET_MODELS.forEach(m => {
        const isDownloaded = modelStatus[m];
        const safeId = m.replace(/[\s\(\)]+/g, '-');
        
        // 🌟 [추가] stanza_ prefix 변환 로직
        let displayName = m;
        if (m.startsWith('stanza_')) {
            const lang = m.replace('stanza_', '');
            displayName = `Stanza ${lang.charAt(0).toUpperCase() + lang.slice(1)}`;
        }
        
        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.flexDirection = "column";
        row.style.background = "rgba(0,0,0,0.05)";
        row.style.border = "1px solid rgba(0,0,0,0.1)";
        row.style.padding = "8px";
        row.style.borderRadius = "6px";

        const topRow = document.createElement("div");
        topRow.style.display = "flex";
        topRow.style.justifyContent = "space-between";
        topRow.style.alignItems = "center";

        const nameSpan = document.createElement("span");
        // 🌟 [수정] 모델명 뒤에 / apache 2.0 고정 노출
        nameSpan.innerText = `${displayName} / apache 2.0`;
        nameSpan.style.fontSize = "0.75rem";
        nameSpan.style.fontWeight = "bold";

        const btn = document.createElement("button");
        btn.id = `btn-download-${safeId}`;
        btn.style.padding = "4px 8px";
        btn.style.fontSize = "0.65rem";
        btn.style.borderRadius = "4px";
        btn.style.border = "none";
        btn.style.cursor = "pointer";

        if (isDownloaded) {
            btn.innerText = "Downloaded";
            btn.style.background = "#6c757d";
            btn.style.color = "white";
            btn.disabled = true;
        } else {
            btn.innerText = "Download";
            btn.style.background = "#28a745";
            btn.style.color = "white";
            btn.onclick = async () => {
                btn.innerText = "Downloading...";
                btn.disabled = true;
                btn.style.background = "#6c757d";
                document.getElementById(`progress-container-${safeId}`)!.style.display = "block";
                await invoke("download_model", { modelName: m });
            };
        }

        topRow.appendChild(nameSpan);
        topRow.appendChild(btn);

        const progContainer = document.createElement("div");
        progContainer.id = `progress-container-${safeId}`;
        progContainer.style.width = "100%";
        progContainer.style.background = "rgba(0,0,0,0.1)";
        progContainer.style.marginTop = "6px";
        progContainer.style.borderRadius = "4px";
        progContainer.style.display = "none";

        const progBar = document.createElement("div");
        progBar.id = `progress-bar-${safeId}`;
        progBar.style.height = "8px";
        progBar.style.width = "0%";
        progBar.style.background = "#007bff";
        progBar.style.borderRadius = "4px";
        progBar.style.fontSize = "6px";
        progBar.style.color = "white";
        progBar.style.textAlign = "center";
        progBar.style.lineHeight = "8px";

        progContainer.appendChild(progBar);
        row.appendChild(topRow);
        row.appendChild(progContainer);
        container.appendChild(row);
    });
}

listen("download_progress", (event: any) => {
    const payload = event.payload;
    const safeId = payload.model.replace(/[\s\(\)]+/g, '-');
    const bar = document.getElementById(`progress-bar-${safeId}`);
    const btn = document.getElementById(`btn-download-${safeId}`);
    if (bar) {
        bar.style.width = `${payload.percent}%`;
        bar.innerText = `${payload.percent}%`;
    }
    if (btn) {
        btn.innerText = `Wait (${payload.percent}%)`;
    }
});

listen("download_complete", (event: any) => {
    const payload = event.payload;
    updateModelStatusUI();
});

listen("download_error", (event: any) => {
    const payload = event.payload;
    updateModelStatusUI();
    alert(`Error downloading ${payload.model}: ${payload.error}`);
});


// 🌟 [MODEL STATUS UI] 모델 상태를 화면에 렌더링하는 함수
function renderModelStatusUI(status: any) {
    const models: Array<{ key: string; label: string }> = [
        { key: "Qwen3", label: "Qwen3 (0.6B)" },
        { key: "Qwen3.5", label: "Qwen3.5 (2B)" },
        { key: "Granite", label: "Granite Embedding" },
        { key: "Embedding", label: "Embedding Model" },
        { key: "SigLIP2", label: "SigLIP2 Vision" },
    ];
    
    // 기존 컨테이너 초기화
    const container = document.getElementById("model-list-container");
    if (container) {
        container.innerHTML = "";
        
        models.forEach(({ key, label }) => {
            const isDownloaded = status[key] === true;
            const row = document.createElement("div");
            row.style.cssText = `
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            `;
            
            const labelSpan = document.createElement("span");
            labelSpan.textContent = label;
            labelSpan.style.cssText = `
                font-size: 0.75rem;
                color: ${isDownloaded ? "#4ade80" : "#999"};
            `;
            
            const safeId = key.replace(/[\s\(\)]+/g, '-');

            const statusBtn = document.createElement("button");
            statusBtn.id = `btn-download-${safeId}`;
            statusBtn.textContent = isDownloaded ? "Downloaded" : "Download";
            statusBtn.style.cssText = `
                padding: 4px 8px;
                font-size: 0.65rem;
                border-radius: 4px;
                border: none;
                cursor: ${isDownloaded ? "default" : "pointer"};
                background: ${isDownloaded ? "#6c757d" : "#28a745"};
                color: white;
            `;
            if (!isDownloaded) {
                statusBtn.onclick = () => {
                    console.log(`[AUTO-DL] ${key} 다운로드 시작...`);
                    statusBtn.innerText = "Downloading...";
                    statusBtn.disabled = true;
                    statusBtn.style.background = "#6c757d";
                    const pc = document.getElementById(`progress-container-${safeId}`);
                    if (pc) pc.style.display = "block";
                    invoke("download_model", { modelName: key }).then(() => {
                        invoke("check_model_status").then((newStatus) => {
                            renderModelStatusUI(newStatus);
                        });
                    });
                };
            }

            // 🌟 [추가] 프로그레스 바 컨테이너 및 바 생성
            const progContainer = document.createElement("div");
            progContainer.id = `progress-container-${safeId}`;
            progContainer.style.width = "100%";
            progContainer.style.background = "rgba(0,0,0,0.1)";
            progContainer.style.marginTop = "6px";
            progContainer.style.borderRadius = "4px";
            progContainer.style.display = "none";

            const progBar = document.createElement("div");
            progBar.id = `progress-bar-${safeId}`;
            progBar.style.height = "8px";
            progBar.style.width = "0%";
            progBar.style.background = "#007bff";
            progBar.style.borderRadius = "4px";
            progBar.style.fontSize = "6px";
            progBar.style.color = "white";
            progBar.style.textAlign = "center";
            progBar.style.lineHeight = "8px";

            progContainer.appendChild(progBar);

            row.appendChild(labelSpan);
            row.appendChild(statusBtn);
            row.appendChild(progContainer);
            container.appendChild(row);
        });
    }
}

document.getElementById("btn-download-all-models")?.addEventListener("click", async () => {
    const missing = TARGET_MODELS.filter(m => !modelStatus[m]);
    if (missing.length === 0) {
        alert("All models are already downloaded.");
        return;
    }
    if (await ask("Download all missing models?", { title: "Confirm Download", kind: "info" })) {
        for (const m of missing) {
            const safeId = m.replace(/[\s\(\)]+/g, '-');
            const btn = document.getElementById(`btn-download-${safeId}`) as HTMLButtonElement;
            if (btn) btn.click();
        }
    }
});

document.getElementById("btn-delete-all-models")?.addEventListener("click", async () => {
    if (await ask("Are you sure you want to delete all models? You will need to download them again for offline capabilities.", { title: "Warning", kind: "warning" })) {
        await invoke("delete_all_models");
        alert("All models deleted.");
        updateModelStatusUI();
    }
});

// 앱 렌더링 시 모델 UI 즉시 초기화
updateModelStatusUI();

settingsBtn?.addEventListener("click", () => { if (currentTab === "settings" && isExpanded) collapseWidget(); else openWidget("settings"); });
document.getElementById("nav-to-auto")?.addEventListener("click", () => switchTab("automation"));
document.getElementById("unload-btn")?.addEventListener("click", async () => {
    try {
        // 🌟 [SESSION PRESERVE] 기존에는 forceReset() 을 호출하여
        //    kv_store.clear() → chat_session 소실 → 로그인 풀림이 발생했습니다.
        //    메모리 해제는 모델/큐 상태만 정리하면 충분하며,
        //    세션·설정·묘비 등 사용자 데이터는 건드리지 않습니다.
        GlobalTaskManager.isBusy = false;
        GlobalTaskManager.currentTaskId = null;
        GlobalTaskManager.currentTaskPayload = null;
        isExtracting = false;
        isSearching = false;
        stopSpinner();
        await invoke("unload_model");
        alert("Memory cleared.");
        // 버튼 상태 복구
        await updateExtractButtonVisibility();
        if (btnSubmit && searchInput) {
            const currentVal = searchInput.value.trim();
            if (currentVal !== "" && !isQueryActive(currentVal)) {
                btnSubmit.style.display = "flex";
            } else {
                btnSubmit.style.display = "none";
            }
        }
    } catch (e) {
        console.error("[WIDGET] Unload failed:", e);
    }
});

document.getElementById("invite-email-input")?.addEventListener("input", (e) => {
    const input = e.target as HTMLInputElement;
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    const btn = document.getElementById("btn-send-invite") as HTMLButtonElement;

    if (input.value.trim() === "") {
        input.style.outline = "none";
        if (btn) btn.disabled = false;
    } else if (!emailRegex.test(input.value.trim())) {
        input.style.outline = "1px solid #ef4444";
        // 형식이 맞지 않으면 전송 버튼을 비활성화하여 오전송 방지
        if (btn) btn.style.opacity = "0.5";
    } else {
        input.style.outline = "1px solid #4ade80";
        if (btn) {
            btn.disabled = false;
            btn.style.opacity = "1";
        }
    }
});

async function syncBrowserStatus() { 
    try { 
        const res = await invoke<any>("get_browser_status"); 
        const s = res.status;

        // 🌟 [CRITICAL FIX] 새 탭(빈 주소) 이동 시에도 currentDetectedUrl을 정상적으로 덮어씌워 버튼을 비활성화합니다!
        if (res.url !== undefined) {
            const urlChanged = currentDetectedUrl !== res.url;
            currentDetectedUrl = res.url;
            isCurrentShop = res.is_client || res.is_admin;

            if (urlChanged && !activeContext.cc && currentTab === "settings") {
                fetchChatHistory(true, true);
            }
        }

        if (s === "running") {
            isBrowserRunning = true;
            // 🌟 [CRITICAL FIX] 런칭 성공 시그널이 오더라도 락을 해제하지 않고 앱 종료 때까지 무조건 숨김을 유지합니다.
            if (btnAutoLaunch) {
                btnAutoLaunch.style.display = "none";
                btnAutoLaunch.classList.add("hidden");
            }
        } else {
            // 🌟 [CRITICAL FIX] isAutoLaunchLocked 조건을 제거합니다.
            //    window focus 시 syncBrowserStatus()가 호출되는데, 작업 진행 중 브라우저가 종료된 후
            //    포커스가 돌아오면 isAutoLaunchLocked=true 상태로 이 분기에 진입하여
            //    btnAutoLaunch가 영원히 노출되지 않는 버그를 수정합니다.
            console.log("[WIDGET] Browser stopped. Resetting UI.");
            isBrowserRunning = false;
            isAutoLaunchLocked = false;
            if (btnAutoLaunch) {
                btnAutoLaunch.style.display = "flex";
                btnAutoLaunch.classList.remove("hidden");
            }
            currentDetectedUrl = ""; 
        }
        await updateExtractButtonVisibility();
    } catch (e) {
        console.warn("Status sync failed", e);
    } 
}

// --- Device Preference Logic ---
const forceCpuToggle = document.getElementById("force-cpu-toggle") as HTMLInputElement;

// --- List Scroll & Pull Engine ---
let listCurrentY = 0;
let listPullY = 0;
let listPullTimer: number | null = null;
let listPushStartTime = 0;
let listPushDir: 'top' | 'bottom' | null = null;

function updateListTransform(resetting: boolean = false) {
    const scrollEl = document.getElementById("list-scroll");
    const container = document.getElementById("list-scroll-container");
    const topLoader = document.getElementById("list-pull-top");
    const bottomLoader = document.getElementById("list-pull-bottom");
    
    if (!scrollEl || !container || !topLoader || !bottomLoader) return;

    if (resetting) scrollEl.classList.add("resetting");
    else scrollEl.classList.remove("resetting");

    let effectiveOffset = listPullY;
    if (listPullY === 0 && listPushStartTime !== 0) {
        const pushElapsed = Date.now() - listPushStartTime;
        if (pushElapsed > 50) { 
            effectiveOffset = listPushDir === 'top' ? 50 : -50;
        }
    }

    scrollEl.style.transform = `translateY(${-listCurrentY + effectiveOffset}px)`;

    const loader = effectiveOffset !== 0 ? (effectiveOffset > 0 ? topLoader : bottomLoader) : null;
    
    if (loader) {
        loader.classList.add("visible");
        const absPull = Math.abs(effectiveOffset);
        loader.style.opacity = "1";
        
        if (absPull >= PULL_THRESHOLD) (loader as HTMLElement).classList.add("ready");
        else (loader as HTMLElement).classList.remove("ready");

        const spinner = (loader as HTMLElement).querySelector('.spinner') as HTMLElement;
        if (spinner) {
            const frameIndex = Math.floor(Date.now() / 80) % spinnerFrames.length;
            spinner.innerText = spinnerFrames[frameIndex];
        }
    } else {
        [topLoader, bottomLoader].forEach(el => {
            if (el) {
                el.classList.remove("visible", "ready");
                (el as HTMLElement).style.opacity = "0";
                const s = el.querySelector('.spinner') as HTMLElement;
                if (s && !el.classList.contains("loading")) s.innerText = "";
            }
        });
    }
}

function initListPullLogic() {
    const container = document.getElementById("list-scroll-container") as HTMLElement;
    const scrollEl = document.getElementById("list-scroll") as HTMLElement;
    const topLoader = document.getElementById("list-pull-top") as HTMLElement;
    const bottomLoader = document.getElementById("list-pull-bottom") as HTMLElement;
    
    if (!container || !scrollEl || !topLoader || !bottomLoader) return;

    let loopId: number | null = null;
    let lastTouchY = 0;

    const resetPull = () => {
        listPullY = 0;
        listPushStartTime = 0;
        listPushDir = null;
        updateListTransform(true);
        setTimeout(() => {
            scrollEl.classList.remove("resetting");
            topLoader.classList.remove("loading");
            bottomLoader.classList.remove("loading");
        }, 400);
    };

    const triggerAction = async (dir: 'top' | 'bottom') => {
        if (isLoading) return;
        const loader = dir === 'top' ? topLoader : bottomLoader;
        loader.classList.add("loading");
        
        listPullY = dir === 'top' ? 40 : -40;
        listPushStartTime = 0;
        updateListTransform(true);

        if (dir === 'top') {
            // [Top Pull] Sync Updates (opposite of chat)
            console.log("[List] Syncing latest updates...");
            await loadMoreDocs(false, true); 
        } else {
            // [Bottom Pull] Load More History (opposite of chat)
            console.log("[List] Loading more history...");
            await loadMoreDocs(false, false); 
        }

        resetPull();
    };

    const startAnimationLoop = () => {
        if (loopId) return;
        const tick = () => {
            const now = Date.now();
            if (listPushStartTime !== 0 && now - listPushStartTime >= 1000 && listPullY === 0) {
                const dir = listPushDir;
                if (dir) {
                    listPullY = dir === 'top' ? TRIGGER_THRESHOLD : -TRIGGER_THRESHOLD;
                    triggerAction(dir);
                }
            }
            updateListTransform();
            if (listPullY !== 0 || listPushStartTime !== 0 || isLoading) {
                loopId = requestAnimationFrame(tick);
            } else {
                loopId = null;
            }
        };
        loopId = requestAnimationFrame(tick);
    };

    const getMaxScroll = () => Math.max(0, scrollEl.scrollHeight - container.clientHeight);

    const handleDelta = (delta: number) => {
        // 🌟 Settings 패널 상태를 확인하여 열려있다면 모든 델타 계산을 중단합니다.
        const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
        if (currentTab !== "list" || (settingsToggle && settingsToggle.checked)) return;

        const maxScroll = getMaxScroll();
        const isAtTop = listCurrentY <= 0;
        const isAtBottom = listCurrentY >= maxScroll;

        if (!isLoading && (listPullY !== 0 || (isAtTop && delta < 0) || (isAtBottom && delta > 0))) {
            const currentDir = (isAtTop && delta < 0) ? 'top' : 'bottom';
            if (listPullY === 0) {
                if (listPushDir !== currentDir) {
                    listPushDir = currentDir;
                    listPushStartTime = Date.now();
                }
                startAnimationLoop(); 
                if (Date.now() - listPushStartTime < 1000) return; 
            }

            listPullY -= delta * FRICTION;
            if (listPullY > PULL_MAX) listPullY = PULL_MAX;
            if (listPullY < -PULL_MAX) listPullY = -PULL_MAX;
            
            if ((listPullY < 0 && listCurrentY <= 0) || (listPullY > 0 && listCurrentY >= maxScroll)) {
                resetPull();
            }
            startAnimationLoop();
        } 
        else {
            listPushDir = null;
            listPushStartTime = 0;
            listCurrentY += delta;
            if (listCurrentY < 0) listCurrentY = 0;
            else if (listCurrentY > maxScroll) listCurrentY = maxScroll;
        }
        updateListTransform();
    };

    container.addEventListener('wheel', (e) => {
        // 🌟 [CRITICAL CHECK] Settings 패널이 활성화되어 있는지 체크합니다.
        const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
        const isSettingsOpen = settingsToggle && settingsToggle.checked;

        // 리스트 탭이 아니거나 Settings 패널이 열려 있으면 리스트 전용 스크롤 로직을 완전히 차단합니다.
        if (currentTab !== "list" || isSettingsOpen) return;

        e.preventDefault();
        handleDelta(e.deltaY);
        if (listPullTimer) clearTimeout(listPullTimer);
        listPullTimer = window.setTimeout(() => {
            if (Math.abs(listPullY) >= PULL_THRESHOLD) triggerAction(listPullY > 0 ? 'top' : 'bottom');
            else if (listPushStartTime === 0 && !isLoading) resetPull();
        }, 200);
    }, { passive: false });

    container.addEventListener('touchstart', (e) => {
        lastTouchY = e.touches[0].pageY;
        scrollEl.classList.remove("resetting");
    }, { passive: true });

    container.addEventListener('touchmove', (e) => {
        // 🌟 [CRITICAL CHECK] Settings 패널이 활성화되어 있는지 체크합니다.
        const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
        const isSettingsOpen = settingsToggle && settingsToggle.checked;

        // Settings가 열려있다면 리스트의 Pull-to-refresh 로직이 간섭하지 못하게 합니다.
        if (currentTab !== "list" || isSettingsOpen) return;

        const currentTouchY = e.touches[0].pageY;
        handleDelta(lastTouchY - currentTouchY);
        lastTouchY = currentTouchY;
        e.preventDefault();
    }, { passive: false });

    container.addEventListener('touchend', () => {
        if (Math.abs(listPullY) >= PULL_THRESHOLD) triggerAction(listPullY > 0 ? 'top' : 'bottom');
        else if (listPushStartTime === 0) resetPull();
    });
}

async function initDevicePreference() {
    if (!forceCpuToggle) return;

    // 1. Check GPU Availability
    try {
        const gpuInfo = await invoke<any>("check_gpu_availability");
        // 호환성을 위해 boolean이 반환될 경우와 객체가 반환될 경우를 모두 처리
        const hasGpu = typeof gpuInfo === "boolean" ? gpuInfo : gpuInfo.has_gpu;
        const vendor = typeof gpuInfo === "object" ? gpuInfo.vendor : "none";

        if (!hasGpu) {
            forceCpuToggle.disabled = true;
            forceCpuToggle.checked = true;
            const label = document.querySelector('label[for="force-cpu-toggle"]') as HTMLElement;
            if (label) label.innerText = "CPU Mode (No GPU detected)";
        } else {
            // 2. Load saved preference
            const savedPrefStr = await kvGet("force_cpu_mode");
            const savedPref = savedPrefStr === "true";
            forceCpuToggle.checked = savedPref;
        }

        // 🌟 [추가] GPU 라이선스 표기 제어 로직 (NVIDIA, AMD에 따라 표시 전환)
        const cudaLicense = document.getElementById("cuda-license");
        const rocmLicense = document.getElementById("rocm-license");
        const gpuLicenseContainer = document.getElementById("gpu-license-container");

        if (cudaLicense && rocmLicense && gpuLicenseContainer) {
            if (vendor === "nvidia") {
                cudaLicense.style.display = "block";
                rocmLicense.style.display = "none";
                gpuLicenseContainer.style.display = "block";
            } else if (vendor === "amd") {
                cudaLicense.style.display = "none";
                rocmLicense.style.display = "block";
                gpuLicenseContainer.style.display = "block";
            } else {
                gpuLicenseContainer.style.display = "none";
            }
        }
    } catch (e) {
        console.error("[WIDGET] Failed to check GPU status:", e);
    }

    // 3. Save on change
    forceCpuToggle.addEventListener("change", async () => {
        await kvSet("force_cpu_mode", forceCpuToggle.checked.toString());
        // [NOTE] The preference will be applied on the next model initialization.
        // Users can click "Free Memory" to force a reload if they want immediate effect.
    });
}

// --- Chat Virtual Scroll & Pull Engine ---
let currentY = 0; // Standard scroll position (positive)
let pullY = 0;    // Pull distance (positive for top, negative for bottom)
let pullTimer: number | null = null;
let pushStartTime = 0; // [NEW] Track hold time
let pushDir: 'top' | 'bottom' | null = null; 
const PULL_THRESHOLD = 50;
const PULL_MAX = 90;
const FRICTION = 0.3;
const TRIGGER_THRESHOLD = 50; 

function updateTransform(resetting: boolean = false) {
    const scrollEl = document.getElementById("chat-scroll");
    const container = document.querySelector(".chat-container") as HTMLElement;
    const topLoader = document.getElementById("chat-pull-top");
    const bottomLoader = document.getElementById("chat-pull-bottom");
    
    if (!scrollEl || !container || !topLoader || !bottomLoader) return;

    if (resetting) scrollEl.classList.add("resetting");
    else scrollEl.classList.remove("resetting");

    let effectiveOffset = pullY;
    if (pullY === 0 && pushStartTime !== 0) {
        const pushElapsed = Date.now() - pushStartTime;
        if (pushElapsed > 50) { 
            effectiveOffset = pushDir === 'top' ? 50 : -50; // Full 50px peek to show loader
        }
    }

    scrollEl.style.transform = `translateY(${-currentY + effectiveOffset}px)`;

    const loader = effectiveOffset !== 0 ? (effectiveOffset > 0 ? topLoader : bottomLoader) : null;
    
    if (loader) {
        loader.classList.add("visible");
        const absPull = Math.abs(effectiveOffset);
        loader.style.opacity = "1";
        
        if (absPull >= PULL_THRESHOLD) (loader as HTMLElement).classList.add("ready");
        else (loader as HTMLElement).classList.remove("ready");

        const spinner = (loader as HTMLElement).querySelector('.spinner') as HTMLElement;
        if (spinner) {
            const frameIndex = Math.floor(Date.now() / 80) % spinnerFrames.length;
            spinner.innerText = spinnerFrames[frameIndex];
        }
    } else {
        [topLoader, bottomLoader].forEach(el => {
            if (el) {
                el.classList.remove("visible", "ready");
                (el as HTMLElement).style.opacity = "0";
                const s = el.querySelector('.spinner') as HTMLElement;
                if (s && !el.classList.contains("loading")) s.innerText = "";
            }
        });
    }
}

function initChatPullLogic() {
    const container = document.querySelector(".chat-container") as HTMLElement;
    const scrollEl = document.getElementById("chat-scroll") as HTMLElement;
    const topLoader = document.getElementById("chat-pull-top") as HTMLElement;
    const bottomLoader = document.getElementById("chat-pull-bottom") as HTMLElement;
    
    if (!container || !scrollEl || !topLoader || !bottomLoader) return;

    let loopId: number | null = null;
    let lastTouchY = 0;

    const resetPull = () => {
        pullY = 0;
        pushStartTime = 0;
        pushDir = null;
        updateTransform(true);
        setTimeout(() => {
            scrollEl.classList.remove("resetting");
            topLoader.classList.remove("loading");
            bottomLoader.classList.remove("loading");
        }, 400);
    };

    const triggerAction = async (dir: 'top' | 'bottom') => {
        if (isChatLoading) return;
        const loader = dir === 'top' ? topLoader : bottomLoader;
        loader.classList.add("loading");
        
        pullY = dir === 'top' ? 40 : -40;
        pushStartTime = 0;
        updateTransform(true);

        if (dir === 'top') {
            // [Top Pull] Load Older History
            console.log("[Chat] Loading history (older than top)...");
            await loadMoreChat(true); 
        } else {
            // [Bottom Pull] Refresh/Load Latest Sync
            console.log("[Chat] Syncing latest/updated states...");
            await loadMoreChat(false); 
        }

        resetPull();
    };

    const startAnimationLoop = () => {
        if (loopId) return;
        const tick = () => {
            const now = Date.now();
            if (pushStartTime !== 0 && now - pushStartTime >= 1000 && pullY === 0) {
                const dir = pushDir;
                if (dir) {
                    pullY = dir === 'top' ? TRIGGER_THRESHOLD : -TRIGGER_THRESHOLD;
                    triggerAction(dir);
                }
            }
            updateTransform();
            if (pullY !== 0 || pushStartTime !== 0 || isChatLoading) {
                loopId = requestAnimationFrame(tick);
            } else {
                loopId = null;
            }
        };
        loopId = requestAnimationFrame(tick);
    };

    const getMaxScroll = () => Math.max(0, scrollEl.scrollHeight - container.clientHeight);

    const handleDelta = (delta: number) => {
        const maxScroll = getMaxScroll();
        const isAtTop = currentY <= 0;
        const isAtBottom = currentY >= maxScroll;

        if (!isChatLoading && (pullY !== 0 || (isAtTop && delta < 0) || (isAtBottom && delta > 0))) {
            const currentDir = (isAtTop && delta < 0) ? 'top' : 'bottom';
            if (pullY === 0) {
                if (pushDir !== currentDir) {
                    pushDir = currentDir;
                    pushStartTime = Date.now();
                }
                startAnimationLoop(); 
                if (Date.now() - pushStartTime < 1000) return; 
            }

            pullY -= delta * FRICTION;
            if (pullY > PULL_MAX) pullY = PULL_MAX;
            if (pullY < -PULL_MAX) pullY = -PULL_MAX;
            
            if ((pullY < 0 && currentY <= 0) || (pullY > 0 && currentY >= maxScroll)) {
                resetPull();
            }
            startAnimationLoop();
        } 
        else {
            pushDir = null;
            pushStartTime = 0;
            currentY += delta;
            if (currentY < 0) currentY = 0;
            else if (currentY > maxScroll) currentY = maxScroll;

            if (!isChatLoading && chatHasMore && currentY <= 50 && chatPage > 0) {
                loadMoreChat(false);
            }
        }
        updateTransform();
    };

    container.addEventListener('wheel', (e) => {
        e.preventDefault();
        handleDelta(e.deltaY);
        if (pullTimer) clearTimeout(pullTimer);
        pullTimer = window.setTimeout(() => {
            if (Math.abs(pullY) >= PULL_THRESHOLD) triggerAction(pullY > 0 ? 'top' : 'bottom');
            else if (pushStartTime === 0 && !isChatLoading) resetPull();
        }, 200);
    }, { passive: false });

    container.addEventListener('touchstart', (e) => {
        lastTouchY = e.touches[0].pageY;
        scrollEl.classList.remove("resetting");
    }, { passive: true });

    container.addEventListener('touchmove', (e) => {
        // 🌟 [CRITICAL CHECK] Settings 패널이 활성화되어 있는지 체크합니다.
        const settingsToggle = document.getElementById("settings-toggle") as HTMLInputElement;
        const isSettingsOpen = settingsToggle && settingsToggle.checked;

        // Settings가 열려있다면 리스트의 Pull-to-refresh 로직이 간섭하지 못하게 합니다.
        if (currentTab !== "list" || isSettingsOpen) return;

        const currentTouchY = e.touches[0].pageY;
        handleDelta(lastTouchY - currentTouchY);
        lastTouchY = currentTouchY;
        e.preventDefault();
    }, { passive: false });

    container.addEventListener('touchend', () => {
        if (Math.abs(pullY) >= PULL_THRESHOLD) triggerAction(pullY > 0 ? 'top' : 'bottom');
        else if (pushStartTime === 0) resetPull();
    });
}

// Call init functions
const getDevicePref = () => forceCpuToggle.checked ? "cpu" : null;
const talksScroll = document.getElementById("chat-scroll");
if (talksScroll) {
    initChatPullLogic();
}
const listScroll = document.getElementById("list-scroll");
if (listScroll) {
    initListPullLogic();
}
async function fetchChatHistory(reset: boolean = true, silent: boolean = false, shouldSnap: boolean = true) { 
    if (reset) { 
        chatPage = 0;
        chatHasMore = true;
        if (chatTalks) {
            chatTalks.innerHTML = "";
        }
    } 
    // Initial load is NOT history (isHistory = false)
    await loadMoreChat(false, silent); 
}

interface ChatMessage {
    id: string;
    role: string;
    text: string;
    updated_at: number;
    created_at: number;
    status: number;
    task_id?: string;
    content?: string | any;
    // 🌟 [OWNERSHIP] 삭제 버튼 노출 판정에 사용합니다.
    //    upsertChatMessages 는 from === currentSession.address 일 때 role 을 'user' 로 재계산하므로
    //    실제 판정은 role 로 하되, DOM 에 원본 from 을 남겨 두면 이후 감사·디버깅이 쉬워집니다.
    from?: string;
    ref?: string;
}

// 🌟 [LOCAL ECHO] 낙관적 로컬 저장 행의 id 접두사입니다.
//    서버(index.ts)는 talk.id 를 `hashId()` 로 발급하므로 반드시 '0x' 로 시작합니다.
//      var talk = { id : hashId(), ... }   // 인자 없음 = 완전 난수 지갑 주소
//    두 접두사가 절대 겹치지 않는다는 '구조적 사실' 이 승계 판정의 유일한 근거입니다.
const LOCAL_ECHO_PREFIX = "talk_";

/**
 * 🌟 [LOCAL ECHO RECONCILE]
 *  ── 무엇이 문제였나 ──
 *   채팅 전송 시 화면 반응을 위해 로컬 행을 먼저 만들어 그립니다(낙관적 저장).
 *   그런데 서버는 talk.id 를 난수로 발급하므로 클라이언트가 그 값을 미리 알 수 없고,
 *   결과적으로 같은 문장이 서로 다른 id 두 개로 존재하게 됩니다.
 *     talk_1787456151873_ll0pqw  (로컬)
 *     0x9568fca1abca47333aa634fbe42e354f74464135  (서버)
 *   upsertChatMessages 의 중복 제거는 id 일치를 전제로 하므로 둘 다 렌더링됩니다.
 *
 *  ── 해결 ──
 *   render.ts 의 isAlmostEqual 을 사용해 { role, text, id } 중 id 하나만 다른 쌍을
 *   '동일 메시지' 로 판정하고, 로컬 행을 서버 행에 승계시킵니다.
 *   승계 처리는 세 곳을 동시에 정리해야 재발하지 않습니다.
 *     ① DOM 노드 제거          → 화면 즉시 정리
 *     ② LanceDB messages 삭제  → 앱 재시작 후 부활 방지
 *     ③ Dexie talks 삭제       → 로컬 캐시 잔재 제거
 *
 *  ── 1:1 소비 ──
 *   같은 문장을 연속으로 두 번 보내면 로컬 에코도 2개, 서버 행도 2개입니다.
 *   서버 행을 created_at 오름차순으로 순회하며 '아직 승계되지 않은 가장 오래된 에코'
 *   하나만 소비하므로 개수가 어긋나지 않습니다.
 *
 *  @returns 승계 처리되어 이번 렌더링에서 제외해야 할 로컬 행 id 집합
 */
async function reconcileLocalEchoes(incoming: ChatMessage[]): Promise<Set<string>> {
    const superseded = new Set<string>();
    if (!chatTalks) return superseded;
    if (!incoming || incoming.length === 0) return superseded;

    // ── 서버가 발급한 talk 행만 승계 기준이 됩니다 ──
    const serverRows = incoming
        .filter(m => String(m.id || "").startsWith("0x"))
        .filter(m => String(m.text || "").trim().length > 0)
        .sort((a, b) => Number(a.created_at || 0) - Number(b.created_at || 0));
    if (serverRows.length === 0) return superseded;

    // ── 로컬 에코 후보 수집 : ① 이미 그려진 DOM 노드 ② 이번 배치에 함께 실려 온 DB 행 ──
    //    ②가 필요한 이유 : 앱을 껐다 켜면 로컬 행과 서버 행이 같은 조회 결과에 동시에 담겨
    //    옵니다. DOM 만 보면 그 경우를 잡지 못해 재시작마다 중복이 되살아납니다.
    type Echo = { id: string; role: string; text: string; createdAt: number; node: HTMLElement | null };
    const echoes: Echo[] = [];

    for (const node of Array.from(chatTalks.querySelectorAll('.chat-talk')) as HTMLElement[]) {
        if (!node.id.startsWith(LOCAL_ECHO_PREFIX)) continue;
        echoes.push({
            id: node.id,
            role: node.classList.contains('user') ? 'user' : 'system',
            text: node.querySelector('.content')?.textContent?.trim() || "",
            createdAt: Number(node.dataset.createdAt || 0),
            node
        });
    }
    for (const m of incoming) {
        const mid = String(m.id || "");
        if (!mid.startsWith(LOCAL_ECHO_PREFIX)) continue;
        if (echoes.some(e => e.id === mid)) continue;
        echoes.push({
            id: mid,
            role: m.role === "user" ? "user" : "system",
            text: String(m.text || "").trim(),
            createdAt: Number(m.created_at || 0),
            node: null
        });
    }
    if (echoes.length === 0) return superseded;

    echoes.sort((a, b) => a.createdAt - b.createdAt);

    for (const srv of serverRows) {
        const srvFp = {
            role: srv.role === "user" ? "user" : "system",
            text: String(srv.text || "").trim(),
            id: String(srv.id)
        };

        for (const echo of echoes) {
            if (superseded.has(echo.id)) continue;
            const echoFp = { role: echo.role, text: echo.text, id: echo.id };

            // 🌟 키 3개 중 id 하나만 다르면 동일 메시지 (diffCount === 1)
            if (!isAlmostEqual(echoFp, srvFp)) continue;

            superseded.add(echo.id);

            // ① 화면에서 제거
            if (echo.node) echo.node.remove();

            // ② LanceDB messages 에서 제거 (upsert_items 가 task_id 에 자기 id 를 각인해 둡니다)
            try {
                await invoke("delete_message", { taskId: echo.id });
            } catch (e) {
                console.warn(`[CHAT] local echo '${echo.id}' DB delete failed:`, e);
            }

            // ③ Dexie talks 캐시에서 제거
            try {
                if (appDb) await appDb.table("talks").delete(echo.id);
            } catch (e) { /* 캐시에 없을 수 있으므로 무시 */ }

            console.log(`[CHAT] ♻️ [LOCAL ECHO RECONCILE] '${echo.id}' → 서버 행 '${srv.id}' 로 승계 (중복 제거)`);
            break;
        }
    }

    return superseded;
}

async function upsertChatMessages(messages: ChatMessage[], mode: 'prepend' | 'append') {
    if (!chatTalks) return;

    // 🌟 [TOMBSTONE GATE] 렌더링 직전 최종 방어선입니다.
    //    syncData 를 거치지 않는 경로(loadMoreChat → get_chat_messages,
    //    renderMessage 직접 호출 등)가 새로 생겨도 삭제한 메시지가
    //    화면에 되살아나지 않도록 여기서 한 번 더 걸러냅니다.
    if (messages && messages.length > 0) {
        const tombs = await loadTalkTombstones();
        if (tombs.size > 0) {
            const before = messages.length;
            messages = messages.filter(m => !tombs.has(String(m.id || "")) && !tombs.has(String(m.task_id || "")));
            if (before !== messages.length) {
                console.log(`[TOMBSTONE] 🪦 [RENDER] 삭제된 메시지 ${before - messages.length}건을 렌더링 대상에서 제외했습니다.`);
            }
            if (messages.length === 0) return;
        }
    }

    // 🌟 [EMPTY NOTICE SWEEP] "No messages yet." 안내는 infoNodes 로 분류되어
    //    아래 정렬 로직에서 항상 최상단에 유지됩니다.
    //    실제 메시지가 한 건이라도 들어오는 순간 이 문구는 거짓이 되므로 즉시 제거합니다.
    if (messages && messages.length > 0) {
        const noMsgEl = chatTalks.querySelector('.no-msg');
        if (noMsgEl) noMsgEl.remove();
    }

    // 🌟 [RECONCILE FIRST] 서버 행이 도착했다면 그와 짝이 되는 로컬 에코를 먼저 승계 처리합니다.
    //    반드시 prevScrollHeight 측정 '이전' 에 수행해야 합니다.
    //    노드를 제거하면 scrollHeight 가 줄어드는데, 측정 이후에 지우면
    //    아래 스크롤 보정식(heightDiff)이 음수가 되어 화면이 튑니다.
    const supersededIds = await reconcileLocalEchoes(messages);
    if (supersededIds.size > 0) {
        // 승계된 로컬 행이 이번 배치에도 실려 있다면 렌더링 대상에서 제외합니다.
        messages = messages.filter(m => !supersededIds.has(String(m.id || "")));
        if (messages.length === 0) return;
    }

    const scrollEl = document.getElementById("chat-scroll");
    const prevScrollHeight = scrollEl ? scrollEl.scrollHeight : 0;

    for (const msg of messages) {
        let textContent = msg.text || "";
        const rawContent = msg.content || (msg as any).data;

        if (rawContent && rawContent !== "undefined") {
            try {
                let contentObj: any = rawContent;
                if (typeof rawContent === 'string') {
                    try {
                        contentObj = JSON.parse(rawContent);
                    } catch (e) {
                        contentObj = rawContent;
                    }
                }
                
                // ArrayBuffer 또는 Gzip 배열 형태의 데이터 파싱 보완
                if (contentObj && typeof contentObj === 'object' && !contentObj.text && !contentObj.title) {
                    if (Array.isArray(contentObj) || contentObj.buffer) {
                        try {
                            const arr = new Uint8Array(contentObj.data || contentObj);
                            const decompressed = (window as any).pako ? (window as any).pako.ungzip(arr, { to: 'string' }) : new TextDecoder().decode(arr);
                            contentObj = JSON.parse(decompressed);
                        } catch (err) {}
                    }
                }

                if (typeof contentObj === 'object' && contentObj !== null) {
                    textContent = contentObj.text || contentObj.title || contentObj.summary || contentObj.markdown || textContent;
                } else if (typeof contentObj === 'string') {
                    textContent = contentObj;
                }
            } catch (e) {
                if (!textContent) textContent = String(rawContent);
            }
        }

        // chrome.js 형태의 from 주소 기반 유저/시스템 role 자동 교정
        let computedRole = msg.role;
        if (msg.from && currentSession.address) {
            computedRole = (msg.from.toLowerCase() === currentSession.address.toLowerCase()) ? "user" : "system";
        }

        const displayMsg: ChatMessage = { ...msg, role: computedRole, text: textContent };
        const isTask = displayMsg.role === "system_task" || (displayMsg.role === "user" && !!displayMsg.task_id && displayMsg.task_id.startsWith("search_") && !displayMsg.id.endsWith("_query") && !displayMsg.task_id.endsWith("_query"));
        const domId = isTask ? (displayMsg.task_id || displayMsg.id) : displayMsg.id;
        
        const existingEl = chatTalks.querySelector(`[id="${domId}"]`) as HTMLElement;

        if (existingEl) {
            const cachedStatus = parseInt(existingEl.dataset.status || "0");
            
            // 🌟 [CRITICAL FIX 3] 한 번 진행 중(1)이 된 작업을 늦게 도착한 이벤트가 다시 대기(10)로 강등시키는 것을 원천 차단합니다!
            if ([1, 2, 6, 9].includes(cachedStatus) && msg.status === 10) {
                msg.status = cachedStatus; 
            }
            // 🌟 이미 종료 상태(2, 6, 9)인 메시지를 다시 진행(1)으로 되돌리는 것도 금지합니다.
            if ([2, 6, 9].includes(cachedStatus) && msg.status === 1) {
                msg.status = cachedStatus; 
            }

            const isTransitionFromVirtual = cachedStatus === 10 && displayMsg.status !== 10;
            const cachedUpdatedAt = parseInt(existingEl.dataset.updatedAt || "0");
            const cachedText = existingEl.querySelector('.content')?.textContent || "";

            // 🌟 [CRITICAL FIX 2] msg 대신 파싱이 완료된 displayMsg의 속성을 사용하여 안전하게 비교합니다.
            if (isTransitionFromVirtual || displayMsg.updated_at > cachedUpdatedAt || displayMsg.status !== cachedStatus || (displayMsg.text && cachedText !== displayMsg.text)) {
                
                // 1. 텍스트 내용 업데이트 (퍼센트 및 요약글)
                const contentEl = existingEl.querySelector('.content');
                // 🌟 [CRITICAL FIX 3] msg.text(undefined)가 아닌 displayMsg.text를 꽂아 넣어 빈칸 버그를 해결합니다!
                if (contentEl && contentEl.textContent !== displayMsg.text) {
                    contentEl.textContent = displayMsg.text;
                }

                // 2. 상태(Status) 및 아이콘 업데이트
                let finalStatus = displayMsg.status;
                
                // 🌟 [CRITICAL FIX] 좀비 방어: 현재 활성 작업(activeTaskId)이 아니더라도 큐가 돌리고 있는(currentTaskId) 정상 작업이면 STOPPED 처리를 면제합니다.
                if (finalStatus === 1 && !isSearching && !isExtracting && activeTaskId !== domId && GlobalTaskManager.currentTaskId !== domId) {
                    finalStatus = 2;
                }

                if (finalStatus !== cachedStatus) {
                    existingEl.dataset.status = finalStatus.toString();
                    
                    const currentLock = await kvGet("sys_lock");
                    if (currentLock === domId && [2, 6, 9].includes(finalStatus)) {
                        console.log(`[LOCK] Task ${domId} reached terminal state ${finalStatus}. Releasing lock.`);
                        await kvRemove("sys_lock");
                    }

                    const statusBar = existingEl.querySelector('.status-bar') as HTMLElement;
                    if (statusBar) {
                        const statusMap: any = {
                            1: { icon: "⠋", text: "PROCESSING", color: "#000" },
                            9: { icon: "✅", text: "DONE", color: "#22c55e" },
                            10: { icon: "📥", text: "QUEUED", color: "#999999" },
                            2: { icon: "❌", text: "STOPPED", color: "#ef4444" }, // 🌟 아이콘을 ❌로 변경하고 색상을 빨간색으로 고정
                            6: { icon: "❌", text: "ERROR", color: "#ef4444" }
                        };
                        // 🌟 finalStatus 변수를 참조하거나 msg.status를 직접 매핑에 사용하도록 보장합니다.
                        const s = statusMap[finalStatus] || statusMap[msg.status] || { icon: "⏳", text: "WAITING", color: "#999999" };
                        statusBar.style.color = s.color;
                        statusBar.innerHTML = `<span class="${(finalStatus === 1 || msg.status === 1) ? 'active-spinner' : ''}">${s.icon}</span> ${s.text}`;
                    }
                }
                existingEl.dataset.updatedAt = msg.updated_at.toString();
            }
        } else {
            const temp = document.createElement('div');
            temp.innerHTML = createMessageHTML(displayMsg);
            const newEl = temp.firstElementChild as HTMLElement;
            if (isTask) { newEl.onclick = () => handleTaskClick(newEl); }
            
            // 🌟 [CRITICAL FIX] chrome.js 패리티: 신규 메시지만 append/prepend 로 처리하여 DOM 중복 생성을 원천 차단합니다.
            if (mode === 'prepend') {
                chatTalks.prepend(newEl);
            } else {
                chatTalks.appendChild(newEl);
            }
        }
    }

    // 🌟 [CRITICAL FIX] DOM 정렬 로직 강화 (시간 오름차순 및 질문 우선순위 고정)
    const sortedChildren = Array.from(chatTalks.children) as HTMLElement[];
    
    // ".no-msg"나 ".chat-history-end" 같은 안내 문구는 정렬 및 중복 검사 대상에서 제외
    const messageNodes = sortedChildren.filter(node => !node.classList.contains('no-msg') && !node.classList.contains('chat-history-end'));
    const infoNodes = sortedChildren.filter(node => node.classList.contains('no-msg') || node.classList.contains('chat-history-end'));

    // 🌟 [중복 노드 제거 헬퍼] 동일한 ID가 여러 개 있을 경우 가장 최신 노드(아래쪽)만 남기고 삭제
    const uniqueIds = new Set();
    const uniqueNodes = [];
    for (let i = messageNodes.length - 1; i >= 0; i--) {
        const node = messageNodes[i];
        if (!uniqueIds.has(node.id)) {
            uniqueIds.add(node.id);
            uniqueNodes.unshift(node); // 원래 순서를 유지하기 위해 앞으로 넣음
        } else {
            // 중복된 노드는 DOM에서 즉시 파기
            node.remove();
        }
    }

    uniqueNodes.sort((a, b) => {
        const timeA = Number(a.dataset.createdAt || 0);
        const timeB = Number(b.dataset.createdAt || 0);
        
        // 1. 시간이 다르면 시간순 정렬
        if (timeA !== timeB) {
            return timeA - timeB;
        }
        
        // 2. 시간이 동일할 경우, 질문(_query)이 작업 메시지보다 항상 앞에 오도록 배치
        const aId = a.id || "";
        const bId = b.id || "";
        const aIsQuery = aId.endsWith("_query") || aId.includes("_query");
        const bIsQuery = bId.endsWith("_query") || bId.includes("_query");
        
        if (aIsQuery && !bIsQuery) return -1;
        if (!aIsQuery && bIsQuery) return 1;
        
        // 3. 그 외에는 ID 문자열 순서로 고정 정렬
        return aId.localeCompare(bId);
    });

    // 🌟 [핵심 수정] 정렬된 리스트와 현재 DOM 순서를 비교하여 필요한 노드만 재배치
    // 정보성 노드(안내 문구)는 무조건 가장 위쪽에 배치
    const finalNodes = [...infoNodes, ...uniqueNodes];
    finalNodes.forEach((node, idx) => {
        if (chatTalks.children[idx] !== node) {
            chatTalks.insertBefore(node, chatTalks.children[idx] || null);
        }
    });

    // [Scroll Maintenance]
    if (mode === 'prepend' && scrollEl) {
        const newScrollHeight = scrollEl.scrollHeight;
        const heightDiff = newScrollHeight - prevScrollHeight;
        if (heightDiff > 0) {
            currentY += heightDiff;
            updateTransform();
        }
    } else if (mode === 'append' && scrollEl) {
        const container = document.querySelector(".chat-container") as HTMLElement;
        const maxScroll = Math.max(0, scrollEl.scrollHeight - (container?.clientHeight || 0));
        
        if (prevScrollHeight === 0 || (currentY >= prevScrollHeight - (container?.clientHeight || 0) - 50)) {
            currentY = maxScroll;
            updateTransform();
        }
    }
}

function createMessageHTML(msg: ChatMessage) {
    // 🌟 상태 2번에 대한 정의를 명시적으로 추가하여 'WAITING'으로 빠지는 것을 방지합니다.
    const statusMap: Record<number, { icon: string, text: string, color: string }> = {
        9: { icon: "✅", text: "DONE", color: "#22c55e" },
        0: { icon: "✅", text: "DONE", color: "#22c55e" },
        1: { icon: "⠋", text: "PROCESSING", color: "#000" },
        6: { icon: "❌", text: "ERROR", color: "#ef4444" },
        2: { icon: "❌", text: "STOPPED", color: "#ef4444" }, // 🌟 좀비 테스크(2)를 ERROR 아이콘과 색상으로 지정
        10: { icon: "📥", text: "PENDING", color: "#999999" },
        3: { icon: "🛑", text: "STOPPED", color: "#ef4444" }
    };
    
    const currentStatus = statusMap[msg.status] || { icon: "⏳", text: "WAITING", color: "#999999" };
    
    // Task Bubble 판단 로직 (ID와 Role 기준)
    const isTaskBubble = msg.role === "system_task" || (!!msg.task_id && msg.task_id.startsWith("search_") && !msg.id.endsWith("_query"));
    const roleClass = msg.role === "user" ? "user" : "system";
    const domId = isTaskBubble ? (msg.task_id || msg.id) : msg.id;
    
    const timeStr = new Date(Number(msg.created_at)).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const bubbleClass = isTaskBubble ? 'task-bubble' : '';

    // 🌟 핵심: msg.text가 비어있지 않도록 보장하여 새로고침 시에도 내용 표시
    const displayContent = msg.text && msg.text.trim() !== "" ? msg.text : "대기 중인 작업입니다...";

    // 🌟 [DELETE AFFORDANCE] 삭제 버튼은 '내가 쓴 순수 채팅' 에만 노출합니다.
    //  · 태스크 말풍선(검색/추출 진행 상태)은 제외
    //    → delete_message(task_id) 가 그 태스크의 질문 말풍선까지 함께 지우고,
    //      중단은 이미 btn-stop-task 가 담당합니다.
    //  · 상대방 메시지는 제외
    //    → 삭제는 어차피 내 기기 로컬에만 적용되므로 남의 글을 지우는 것은
    //      권한 행사가 아니라 '내 화면 숨김' 에 불과해 오해를 부릅니다.
    //  role 은 upsertChatMessages 가 from === currentSession.address 로 재계산한 값입니다.
    const canDelete = !isTaskBubble && msg.role === 'user' && !!msg.id;
    const deleteBtn = canDelete
        ? `<button class="btn-delete-talk" data-talk-id="${domId}" title="Delete for me"
                style="background:none; border:none; color:inherit; opacity:0.35; cursor:pointer; font-size:0.75rem; line-height:1; padding:0 0 0 8px;">✕</button>`
        : "";

    return `<div id="${domId}" class="chat-talk ${roleClass} ${bubbleClass}" 
        data-task-id="${msg.task_id || msg.id}" 
        data-status="${msg.status}" 
        data-updated-at="${msg.updated_at}"
        data-created-at="${msg.created_at}"
        data-from="${msg.from || ''}"
        data-ref="${msg.ref || ''}"
        style="${isTaskBubble ? 'cursor:pointer;' : ''}">
        <div class="chat-message">
            <div style="font-size:0.8rem; opacity:0.5; margin-bottom:4px; display:flex; justify-content:space-between; align-items:center;">
                <span>${msg.role === 'user' ? '@YOU' : 'LOGIS AI'}</span>
                <span style="display:flex; align-items:center;">${timeStr}${deleteBtn}</span>
            </div>
            <div class="content">${displayContent}</div>
            ${isTaskBubble && msg.status !== 0 ? `
                <div class="status-bar" style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.1); font-size: 0.65rem; font-weight: bold; color: ${currentStatus.color};">
                    <span class="${msg.status === 1 ? 'active-spinner' : ''}">${currentStatus.icon}</span> ${currentStatus.text}
                </div>` : ""}
        </div>
    </div>`;
}

async function loadMoreChat(isHistory: boolean = false, silent: boolean = false) {
    if (isChatLoading || (isHistory && !chatHasMore)) {
        if (!silent) stopSpinner();
        return;
    }

    if (!silent) startSpinner();
    isChatLoading = true;

    try {
        // 🌟 [CRITICAL FIX] chrome.js 패리티: 서버에는 정상적으로 채팅 내역이 저장되었지만, 로컬 DB에서 꺼내올 때 URL 필터가 누락되어 빈 화면이 노출되는 버그를 해결합니다.
        let effectiveCc = activeContext.cc;
        let effectiveBcc = activeContext.bcc;
        let effectiveRef = activeContext.ref;

        const isDefaultForced = activeTags.some(t => t.value === "logis.center" && t.type === "domain");

        if (!effectiveCc || (!isDefaultForced && activeTags.length === 0)) {
            let targetUrlStr = currentDetectedUrl || "https://commerce.logis.center/tracking";
            if (targetUrlStr.includes("localhost") || targetUrlStr.includes("127.0.0.1") || targetUrlStr === "about:blank") {
                targetUrlStr = "https://commerce.logis.center/tracking";
            }
            try {
                const urlObj = new URL(targetUrlStr.toLowerCase());
                const rootDomain = getRootDomain(urlObj.hostname);
                effectiveCc = await hashId(rootDomain);
                const link = (urlObj.pathname + urlObj.search).toLowerCase();
                effectiveRef = await hashId((currentSession.team || "") + effectiveCc + link);
            } catch (err) {}
        }

        let baseFilter = "";
        if (effectiveRef) baseFilter = `ref = '${effectiveRef}'`;
        else if (effectiveBcc) baseFilter = `bcc = '${effectiveBcc}'`;
        else if (effectiveCc) baseFilter = `cc = '${effectiveCc}'`;
        
        let finalFilter = baseFilter;
        let oldestTime = 0;
        let latestUpdateTime = 0;

        const allMsgs = chatTalks.querySelectorAll('.chat-talk');
        allMsgs.forEach(el => {
            const up = parseInt((el as HTMLElement).dataset.updatedAt || "0");
            if (up > latestUpdateTime) latestUpdateTime = up;
        });

        if (isHistory) {
            const firstMsg = chatTalks.querySelector('.chat-talk:not(.chat-history-end)');
            if (firstMsg) {
                oldestTime = parseInt((firstMsg as HTMLElement).dataset.createdAt || "0");
            }
            
            if (oldestTime > 0) {
                let timeFilter = `created_at < ${oldestTime}`;
                if (latestUpdateTime > 0) {
                    timeFilter = `(${timeFilter}) OR (updated_at > ${latestUpdateTime})`;
                }
                finalFilter = baseFilter ? `${baseFilter} AND (${timeFilter})` : timeFilter;
            }
        } else if (latestUpdateTime > 0) {
            const syncFilter = `updated_at > ${latestUpdateTime}`;
            finalFilter = baseFilter ? `${baseFilter} AND ${syncFilter}` : syncFilter;
        }

        const limit = 10; 
        const offset = 0;

        let messages = await invoke<any[]>("get_chat_messages", { limit: limit, offset: offset, filter: finalFilter });
        
        // 🌟 [추가] 좀비 상태 보정 및 정렬 안정화 데이터 세탁
        messages = messages.map(m => {
            // 1. 유령 데이터 STOPPED 처리 강화: 
            // 현재 앱이 '초기화(Handshake) 완료 전'이거나, 백엔드 DB에서도 명시적으로 2인 경우만 중단 처리합니다.
            if ((m.status === 1 || m.status === 10) && !isSearching && !isExtracting) {
                // 단순히 검색/추출 중이 아니라고 해서 2로 바꾸지 않고, 
                // DB에서 넘어온 원본 status가 이미 2이거나 terminal state일 때만 UI를 고정합니다.
                return m; 
            }
            // 2. 사용자 질문의 경우 시스템 메시지와의 정렬 간격을 벌리기 위해 시간값 강제 보정
            if (m.role === "user" && m.id.endsWith("_query")) {
                return { ...m, created_at: Number(m.created_at) - 50 };
            }
            return m;
        });

        // 🌟 [CRITICAL FIX 2] 변수 스코프(Scope) 에러 해결! 
        let activeMemContext: any = null;
        try {
            activeMemContext = await invoke<any>("get_active_task_context");
        } catch (e) {}

        try {
            // 1. Rust 백엔드 DB에 저장된 활성 태스크 가져오기
            const activeTasks = await invoke<any[]>("get_active_tasks");
            
            // 🌟 2. [수정] 프론트엔드 큐 작업 병합 시, DB에서 이미 종료/중단된 ID는 제외합니다.
            const queuedTasks = GlobalTaskManager.queue.map(q => ({
                id: q.taskId,
                task_id: q.taskId,
                status: 10, // Pending
                created_at: parseInt(q.taskId.split('_')[1]) || Date.now(),
                data_json: q.payload,
                ref: q.payload.link || q.payload.image_path || "Queued Task"
            }));

            // 🌟 [핵심 로직] DB(activeTasks)에 있는 녀석이 10번이 아니라면(이미 2번 등으로 변했다면) 큐에서 부활시키지 않습니다.
            const combinedTasks = [...activeTasks];
            queuedTasks.forEach(qt => {
                const dbEquivalent = activeTasks.find(t => t.id === qt.id);
                // DB에 아예 없거나, DB에서도 여전히 Pending(10)인 경우에만 큐 정보를 신뢰합니다.
                if (!dbEquivalent) {
                    combinedTasks.push(qt);
                }
            });

            combinedTasks.forEach((t: any) => {
                let taskQuery = "";
                try {
                    const taskData = typeof t.data_json === 'string' ? JSON.parse(t.data_json) : t.data_json;
                    taskQuery = taskData.query || "";
                } catch(e) {}

                // 🌟 [CRITICAL FIX] 새로고침 시 질문 복구 로직 강화
                if (taskQuery) {
                    const userMsgId = `${t.id}_query`;
                    const userExistsInBatch = messages.some(m => m.id === userMsgId);
                    const userExistsInDom = document.getElementById(userMsgId);
                    
                    if (!userExistsInBatch && !userExistsInDom) {
                        messages.push({
                            id: userMsgId,
                            task_id: t.id,
                            role: "user",
                            text: taskQuery,
                            status: 9, 
                            // 🌟 initSession과 동일하게 100ms 시간차를 주어 정렬 순서를 물리적으로 강제합니다.
                            created_at: Number(t.created_at) - 100, 
                            updated_at: Number(t.created_at) - 100
                        });
                        console.log(`[RECOVERY] Restored missing user query for task: ${t.id}`);
                    }
                }

                const exists = messages.find(m => m.id === t.id || m.task_id === t.id);
                if (!exists) {
                    messages.push({
                        id: t.id,
                        task_id: t.id,
                        role: "system_task",
                        // 🌟 [UI 보강] DB에 아직 안 들어간 순수 대기열(status: 10) 상태임을 직관적으로 보여줍니다.
                        text: t.id.startsWith("search_") ? "Waiting in Queue: AI Search" : ("Waiting in Queue: " + (t.ref || "Local Source")),
                        status: t.status,
                        created_at: t.created_at + 1,
                        updated_at: t.updated_at + 1
                    });
                }
            });
        } catch (e) { }

        for (let m of messages) {
            if (m.status === 1 && (m.role === "system_task" || m.task_id)) {
                try {
                    const tId = m.task_id || m.id;
                    const logs = await invoke<any[]>("get_task_logs", { taskId: tId });
                    
                    let lastLog = null;
                    if (logs && logs.length > 0) {
                        lastLog = logs[logs.length - 1];
                    }
                    
                    let rawSummary = "Processing...";
                    const live = livePayloads.get(tId);
                    
                    if (live && live.summary) {
                        rawSummary = live.summary;
                    } else if (lastLog && lastLog.summary) {
                        rawSummary = lastLog.summary;
                    } else if (activeMemContext && activeMemContext.id === tId && activeMemContext.summary) {
                        rawSummary = activeMemContext.summary;
                    }

                    const pctMatch = rawSummary.match(/\(\d+%\)/);
                    const hasDots = rawSummary.endsWith("...");
                    if (hasDots) rawSummary = rawSummary.slice(0, -3).trim();
                    if (pctMatch) rawSummary = rawSummary.replace(pctMatch[0], '').trim();
                    
                    let fractionStr = "";
                    const targetCat = (live && live.category) ? live.category : (lastLog && lastLog.category ? lastLog.category : "");

                    // 🌟 [UI 심플화] 채팅방 히스토리에도 오직 List Extraction 단계에서만 [N/M]을 보여줍니다.
                    if (targetCat.includes("List Extraction")) {
                        const match = targetCat.match(/\((\d+)\/(\d+)\)/);
                        if (match) {
                            fractionStr = ` [${match[1]}/${match[2]}]`;
                        }
                    }
                    
                    m.text = `${rawSummary}${fractionStr}${pctMatch ? ' ' + pctMatch[0] : ''}${hasDots ? '...' : ''}`;
                    m.updated_at = Date.now();
                    
                } catch (e) {}
            }
        }

        const scrollEl = document.getElementById("chat-scroll") as HTMLElement;

        if (chatTalks) {
            if (messages && messages.length > 0) {
                const mode = isHistory ? 'prepend' : 'append';
                upsertChatMessages(messages, mode);
                if (isHistory && messages.length < limit) chatHasMore = false;
            } else { 
                if (isHistory) chatHasMore = false;
                // 🌟 [보강] 이미 no-msg 엘리먼트가 존재한다면 추가하지 않도록 방어합니다.
                const hasNoMsgEl = chatTalks.querySelector('.no-msg');
                if (!isHistory && chatTalks.querySelectorAll('.chat-talk').length === 0 && !hasNoMsgEl) {
                    chatTalks.insertAdjacentHTML('beforeend', "<div class='no-msg' data-created-at=\"0\" style='text-align:center; padding:20px; color:#999; font-size:0.8rem;'>No messages yet.</div>");
                }
            }

            if (isHistory && !chatHasMore && !chatTalks.querySelector('.chat-history-end')) {
                const endHtml = `<div class="chat-talk system chat-history-end" data-created-at="0" style="text-align:center; opacity:0.4; font-size:0.8rem; padding:15px 10px;">
                    <div style="border-top:1px solid rgba(255,255,255,0.05); margin-bottom:10px;"></div>
                    <span>No more older messages</span>
                </div>`;
                chatTalks.insertAdjacentHTML('afterbegin', endHtml);
            }

            // 🌟 [QR RE-RENDER GUARD] 이 호출은 채팅이 갱신될 때마다 발생합니다.
            //    performQrAuth 내부에 renderedQrHash 멱등성 검사가 들어갔으므로
            //    이미 같은 hash 로 그려져 있으면 즉시 반환하고 노드를 건드리지 않습니다.
            //    다만 이미 그려져 있고 hash 도 동일하다면 함수 호출 자체를 생략해
            //    불필요한 DOM 조회조차 하지 않도록 여기서 한 번 더 걸러 냅니다.
            if (!currentSession.email && currentTab === "settings") {
                const qrExists = !!document.getElementById("qr-code-target");
                if (!qrExists || renderedQrHash !== currentSession.hash) {
                    performQrAuth();
                }
            }
        }
    } catch (e) { 
        console.error(e); 
    } finally { 
        isChatLoading = false; 
        if (!silent) stopSpinner();
    }
}

async function renderMessage(msg: any, shouldScroll: boolean = true, isPrepend: boolean = false) {
    if (!chatTalks) return;
    // Single message upsert (Real-time is always append/newest in Slack style)
    await upsertChatMessages([msg], isPrepend ? 'prepend' : 'append');
}

// --- Initialize ---
initSession();
setWindowSize(false);
syncBrowserStatus();
initDevicePreference();

// =====================================================================
// 🌟 [TRANSLIT CACHE / DEXIE] Rust 스케줄러 ↔ 프론트엔드 Dexie 음차 캐시 통신
// ---------------------------------------------------------------------
//  Rust(scheduler.rs) 가 음차 생성 전에 캐시를 조회하거나,
//  생성 후에 캐시를 저장할 때 emit 하는 이벤트를 여기서 수신합니다.
//
//  조회 흐름:
//    Rust emit("translit-cache-query", {request_id, word, lang})
//    → 프론트 listen → Dexie 조회 → 코사인 계산(복수 후보)
//    → invoke("translit_cache_respond", {request_id, results})
//    → Rust oneshot receiver 에서 await
//
//  저장 흐름:
//    Rust emit("translit-cache-save", {word, lang, native, roman})
//    → 프론트 listen → Dexie upsert
// =====================================================================

// ── 캐시 조회 리스너 ──
//  응답 계약 (Rust query_translit_cache 와 1:1):
//    []                     → 캐시 미스 (레코드 자체가 없음)
//    [[native, roman]]      → 캐시 히트
//    [["", ""]]             → 네거티브 캐시 히트 ('음차 불가' 로 확정된 값)
//  ⚠️ 빈 문자열 레코드도 반드시 반환해야 Rust 가 LLM 재호출을 생략합니다.
listen("translit-cache-query", async (event: any) => {
    const { request_id, word, lang } = event.payload;

    const respond = async (results: Array<[string, string]>) => {
        try {
            await invoke("translit_cache_respond", { requestId: request_id, results });
        } catch (e) {
            console.warn("[TRANSLIT CACHE] respond failed:", e);
        }
    };

    try {
        const candidates = await appDb.table("translit_cache")
            .where("[source_word+doc_lang]")
            .equals([word, lang])
            .toArray();

        if (!candidates || candidates.length === 0) {
            // 🌟 [LANG AXIS DIAGNOSTIC] 같은 원문이 '다른 언어 키' 로 저장돼 있는지 확인합니다.
            //    이 경고가 뜨면 캐시가 깨진 게 아니라 doc_lang 이 흔들린 것입니다.
            //    (실측 사례: 페이지 셀렉터 캐시 히트 시 doc_lang 이 'en' 으로 고정되어
            //     'ko' 로 저장된 레코드를 영원히 찾지 못했습니다)
            try {
                const others = await appDb.table("translit_cache")
                    .where("source_word").equals(word).toArray();
                if (others && others.length > 0) {
                    const langs = Array.from(new Set(others.map((o: any) => String(o.doc_lang))));
                    console.warn(
                        `[TRANSLIT CACHE] ⚠️ MISS on lang='${lang}' but the same word exists under langs=[${langs.join(', ')}]. ` +
                        `doc_lang 확정 경로(scheduler.rs DOC LANG EARLY DETECT)를 확인하세요. word='${word}'`
                    );
                }
            } catch (_e) {
                // v12 미적용 세대에서는 source_word 단독 인덱스가 없어 실패할 수 있습니다.
                // 진단 전용 경로이므로 조용히 흡수합니다.
            }
            console.log(`[TRANSLIT CACHE] MISS word='${word}' lang='${lang}'`);
            await respond([]);
            return;
        }

        // 단일 후보(정상 경로): upsert 정책상 대부분 여기에 해당합니다.
        if (candidates.length === 1) {
            const c = candidates[0];
            const nv = c.native || "";
            const rm = c.roman || "";
            console.log(
                `[TRANSLIT CACHE] ${(!nv && !rm) ? "NEGATIVE HIT" : "HIT"} word='${word}' lang='${lang}' → native='${nv}' roman='${rm}'`
            );
            await respond([[nv, rm]]);
            return;
        }

        // 복수 후보: 원문과의 코사인 유사도로 최적 후보를 고릅니다.
        const candidateTexts = candidates.map((c: any) =>
            `${c.native || ""} ${c.roman || ""}`.trim()
        );

        try {
            const embeddings: number[][] = await invoke("get_embedding_batch_for_translit", {
                texts: candidateTexts
            });
            const queryEmb: number[] = await invoke("get_query_embedding", {
                text: word,
                devicePreference: null
            });

            let bestIdx = 0;
            let bestSim = -1;
            for (let i = 0; i < embeddings.length; i++) {
                const sim = cosineSimLocal(queryEmb, embeddings[i]);
                if (sim > bestSim) {
                    bestSim = sim;
                    bestIdx = i;
                }
            }
            const best = candidates[bestIdx];
            console.log(
                `[TRANSLIT CACHE] HIT(cosine ${bestSim.toFixed(4)}, ${candidates.length} cands) word='${word}' lang='${lang}'`
            );
            await respond([[best.native || "", best.roman || ""]]);
        } catch (embErr) {
            // 임베딩 실패 시 최신 후보 폴백
            const sorted = [...candidates].sort((a: any, b: any) => (b.created_at || 0) - (a.created_at || 0));
            console.log(
                `[TRANSLIT CACHE] HIT(latest fallback, ${candidates.length} cands) word='${word}' lang='${lang}'`
            );
            await respond([[sorted[0].native || "", sorted[0].roman || ""]]);
        }
    } catch (e) {
        console.warn("[TRANSLIT CACHE] query failed:", e);
        await respond([]);
    }
});

// ── 캐시 저장 리스너 ──
//  native / roman 이 모두 빈 문자열이어도 반드시 저장합니다(네거티브 캐시).
//  '이 원문은 이 언어에서 음차가 성립하지 않는다' 는 판정 자체가 재사용 가치가 있습니다.
listen("translit-cache-save", async (event: any) => {
    const { word, lang, native, roman } = event.payload;
    const nv = native || "";
    const rm = roman || "";
    try {
        // 기존 동일 키 삭제 후 삽입 (upsert)
        const removed = await appDb.table("translit_cache")
            .where("[source_word+doc_lang]")
            .equals([word, lang])
            .delete();

        await appDb.table("translit_cache").add({
            source_word: word,
            doc_lang: lang,
            native: nv,
            roman: rm,
            created_at: Date.now()
        });

        console.log(
            `[TRANSLIT CACHE] SAVED${(!nv && !rm) ? "(negative)" : ""} word='${word}' lang='${lang}' ` +
            `→ native='${nv}' roman='${rm}' (replaced ${removed})`
        );
    } catch (e) {
        // 🌟 저장 실패는 '다음 태스크에서 LLM 재호출' 로 직결되므로 반드시 표면화합니다.
        //    (기존에는 warn 만 찍혀 원인 추적이 불가능했습니다)
        console.error(
            `[TRANSLIT CACHE] ❌ SAVE FAILED word='${word}' lang='${lang}'. ` +
            `이 값은 다음 태스크에서 다시 LLM 으로 생성됩니다.`, e
        );
    }
});

// ── 로컬 코사인 계산 헬퍼 (프론트 전용) ──
function cosineSimLocal(a: number[], b: number[]): number {
    if (a.length !== b.length || a.length === 0) return 0;
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom === 0 ? 0 : dot / denom;
}

function stopDesktopCamera() {
    if (desktopStream) {
        desktopStream.getTracks().forEach(track => track.stop());
        desktopStream = null;
    }
}

async function startMobileScanning(video: HTMLVideoElement) {
    if (!video || !(video instanceof HTMLVideoElement)) {
        console.error('Invalid video element provided to startMobileScanning');
        return;
    }

    try {
        console.log("Starting desktop camera stream...");
        desktopStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        video.srcObject = desktopStream;
        await video.play();
        
        document.getElementById("mobile-scan-view")?.classList.remove("hidden");
        document.getElementById("pc-qr-view")?.classList.add("hidden");
    } catch (err) {
        console.error("Failed to start desktop camera:", err);
        alert("Camera start failed: " + err);
        return;
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    const receivedChunks: string[] = [];
    let expectedTotal = 0;
    
    const scanLoop = async () => {
        if (!video || video.paused || video.ended) return;
        try {
            if (video.readyState >= 2) {
                canvas.width = video.videoWidth; canvas.height = video.videoHeight;
                if (ctx && canvas.width > 0 && canvas.height > 0) {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    // @ts-ignore
                    const code = jsQR(imageData.data, imageData.width, imageData.height);
                    if (code) {
                        try {
                            const data = JSON.parse(code.data);
                            // Handle compact Answer
                            if (data.t === "answer") {
                                const sdp = buildSdp('answer', data.i, data.u, data.p, data.f, data.s);
                                const answer = new RTCSessionDescription({ type: 'answer', sdp });
                                if (peerConn) {
                                    await peerConn.setRemoteDescription(answer);
                                    stopDesktopCamera();
                                    const profileName = document.getElementById("nav-profile-name");
                                    if (profileName) {
                                        profileName.textContent = "✅ Mobile Connected";
                                        profileName.style.color = "#4ade80";
                                    }
                                    document.getElementById("nav-qr-container")?.classList.add("hidden");
                                }
                                return;
                            }
                            // Fallback for legacy chunked format
                            if (Array.isArray(data) && data.length === 3) {
                                const [idx, total, chunkStr] = data;
                                if (expectedTotal === 0) {
                                    expectedTotal = total;
                                    for(let i=0; i<total; i++) receivedChunks.push(""); 
                                }
                                if (!receivedChunks[idx]) {
                                    receivedChunks[idx] = chunkStr;
                                    const profileName = document.getElementById("nav-profile-name");
                                    if (profileName) {
                                        const count = receivedChunks.filter(c => c).length;
                                        profileName.textContent = `Scanning... ${count}/${total}`;
                                    }
                                }
                                if (receivedChunks.every(c => c !== "")) {
                                    const answer = new RTCSessionDescription({ type: 'answer', sdp: receivedChunks.join("") });
                                    if (peerConn) {
                                        await peerConn.setRemoteDescription(answer);
                                        stopDesktopCamera();
                                        const profileName = document.getElementById("nav-profile-name");
                                        if (profileName) {
                                            profileName.textContent = "✅ Mobile Connected";
                                            profileName.style.color = "#4ade80";
                                        }
                                        document.getElementById("nav-qr-container")?.classList.add("hidden");
                                    }
                                    return;
                                }
                            }
                        } catch (e) {}
                    }
                }
            }
        } catch (e) {}
        requestAnimationFrame(scanLoop);
    };
    requestAnimationFrame(scanLoop);
}

document.getElementById("btn-switch-to-camera")?.addEventListener("click", () => {
    const video = document.getElementById("desktop-camera-video") as HTMLVideoElement;
    if (video) startMobileScanning(video);
});
document.getElementById("btn-switch-to-qr")?.addEventListener("click", () => {
    const video = document.getElementById("desktop-camera-video") as HTMLVideoElement;
    if (video) stopDesktopCamera();
    showPcPairingQr();
});