// ============================================================
//  commerce/src/modes/types.ts
//  🌟 [MODE TAXONOMY] 모드(commerce / shipping / analytic) 분류의 단일 소스입니다.
//
//  ── 왜 별도 파일인가 ──
//   기존에는 main.ts 안에 modeOfType() 과 TYPE_SETS 가 함께 있었고,
//   syncData 안에는 TRADING_TYPES 라는 '두 번째 하드코딩 목록' 이 따로 있었습니다.
//   한쪽만 고치면 '쓰기(mode 태깅)' 와 '읽기(type 조회)' 가 어긋나
//   commerce 배송추적 문서가 통째로 mode='shipping' 으로 오염되는 사고가 났습니다.
//   이제 쓰기/읽기 양쪽 모두 이 파일 하나만 바라봅니다.
// ============================================================

export const TRADING_DOC_CODES = [
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
export const TRADING_DOC_TYPE_SET = new Set<string>([
    ...TRADING_DOC_CODES,
    ...TRADING_DOC_CODES.map(c => c.toLowerCase()),
    'shipping_doc', 'TRACKING', 'Unknown', 'unknown'
]);

// ── commerce 도메인 타입 : proxy/index.ts 의 Relay 대상 전량 ──
export const COMMERCE_TYPE_SET = new Set<string>([
    'sales', 'goods', 'order', 'tracking', 'event', 'coupon', 'review',
    'receiving', 'shipping',
    'member', 'team', 'user', 'users', 'pages', 'page', 'talk', 'prompt', 'ai_search'
]);

// ── analytics 행동 로그 / 관리자 Q&A ──
export const ANALYTIC_TYPE_SET = new Set<string>([
    'click', 'hover', 'change', 'report', 'touch', 'question', 'answer'
]);

export function modeOfType(t: string): 'commerce' | 'shipping' | 'analytic' {
    const s = String(t || '');
    if (ANALYTIC_TYPE_SET.has(s)) return 'analytic';
    if (COMMERCE_TYPE_SET.has(s)) return 'commerce';
    if (TRADING_DOC_TYPE_SET.has(s)) return 'shipping';
    return 'commerce';
}

export const TYPE_SETS: Record<string, string[]> = {
    shipping: [
        'tracking', 'receiving', 'shipping', 'shipping_doc', 'TRACKING',
        ...TRADING_DOC_CODES,
        ...TRADING_DOC_CODES.map(c => c.toLowerCase()),
        'Unknown', 'unknown'
    ],
    analytic: ['click', 'hover', 'change', 'report', 'touch', 'question', 'answer'],
    commerce: ['sales', 'goods', 'order', 'tracking', 'event', 'coupon', 'review',
        'receiving', 'shipping']
};

export const MODE_LABEL: Record<string, string> = {
    commerce: 'Commerce',
    shipping: 'Trading',
    analytic: 'Analytic'
};

export function modeLabel(m: string): string {
    return MODE_LABEL[m] || (m.charAt(0).toUpperCase() + m.slice(1));
}