// ============================================================
//  commerce/src/modes/runtime.ts
//  🌟 [MODE RUNTIME BRIDGE]
//   main.ts 가 소유한 전역 상태(세션 / Dexie / UI 렌더러)를
//   modes/*.ts 가 '순환 import 없이' 사용할 수 있게 중계합니다.
//
//  ── 왜 DI(주입) 인가 ──
//   modes/trading.ts 가 main.ts 를 직접 import 하면
//   main.ts → modes/trading.ts → main.ts 의 순환 참조가 생기고,
//   Vite(ESM) 의 평가 순서에 따라 appDb / currentSession 이 undefined 로 잡힙니다.
//   또한 timezoneOffset · GlobalTaskManager 는 main.ts 중·하단에서 선언되므로
//   '값' 으로 넘기면 TDZ 에 걸립니다. 그래서 전부 게터로 받습니다.
//
//   main.ts 는 부팅 직전(initSession() 호출 직전)에 bindModeRuntime() 을 1회 호출하고,
//   모드 모듈은 rt() 로만 접근합니다.
// ============================================================

export interface ModeSession {
    hash: string;
    token?: string;
    email?: string;
    team?: string;
    address?: string;
    name?: string;
    cc?: string;
    sender?: string;
    flag?: string;
}

export interface ModeContext {
    cc: string;
    bcc: string;
    ref: string;
}

export interface ModeTag {
    id: string;
    label: string;
    type: string;
    value: string;
}

export interface ModeRuntime {
    // ── 저장소 ──
    appDb: any;
    timezoneOffset: number;
    kvGet: (key: string) => Promise<any>;
    kvSet: (key: string, value: any) => Promise<void>;
    normalizeEnvelope: (docs: any[]) => any[];
    loadItemTombstones: () => Promise<Set<string>>;
    // 🌟 [PART 2] commerce 트랙의 TOMBSTONE GUARD 가 talk 묘비도 함께 검사합니다.
    loadTalkTombstones: () => Promise<Set<string>>;

    // ── 상태 게터 (전부 지연 평가) ──
    getSession: () => ModeSession;
    getContext: () => ModeContext;
    getSearchMode: () => string;
    getDetectedUrl: () => string;
    getActiveTags: () => ModeTag[];
    getCurrentTab: () => string;
    isBusy: () => boolean;
    // 🌟 [PART 2] analytic 구조화(structure_pending_analytics)가 CPU/GPU 선호를 넘깁니다.
    //    forceCpuToggle 은 main.ts 최하단에서 잡히는 DOM 참조라 반드시 게터여야 합니다.
    getDevicePref: () => string | null;
    // 🌟 [PART 2] commerce 트랙의 CLOUD TASK LIFECYCLE 판정용.
    //    Map 인스턴스 자체는 main.ts 가 소유하고, 여기서는 참조만 빌려옵니다.
    getCloudPendingTasks: () => Map<string, any>;

    // ── UI 콜백 ──
    renderNavigation: () => Promise<void>;
    loadMoreDocs: (reset?: boolean, isSync?: boolean) => Promise<void>;
    renderMessage: (msg: any) => Promise<void>;
    // 🌟 [PART 2] 클라우드 작업 말풍선 갱신 (Cloud Queue → Done)
    renderProgressToUI: (payload: any) => Promise<void>;
    // 🌟 [PART 2] settings 탭이 열려 있을 때 채팅 목록 재동기화
    fetchChatHistory: (reset?: boolean, silent?: boolean) => Promise<void>;
    runLocalEmbeddingSync: () => void;
    stopSpinner: () => void;
    stepQrSpinner: () => void;
    restoreSubmitButton: () => void;
}

let RT: ModeRuntime | null = null;

export function bindModeRuntime(runtime: ModeRuntime) {
    RT = runtime;
}

export function rt(): ModeRuntime {
    if (!RT) {
        throw new Error("[MODE] runtime 이 아직 주입되지 않았습니다. main.ts 에서 bindModeRuntime() 을 먼저 호출하세요.");
    }
    return RT;
}

// =====================================================================
// 🌟 [ADAPTIVE POLLING BACKOFF]
//  commerce / trading / analytic 세 트랙이 공유하는 단일 백오프입니다.
//  기존에는 main.ts 안의 모듈 스코프 변수였고, 세 트랙이 모두 직접 대입했습니다.
//  파일이 쪼개지면 그 변수를 공유할 수 없으므로 여기로 승격합니다.
// =====================================================================
export const SYNC_BASE_INTERVAL_MS = 3_000;   // 기본 폴링 간격
export const SYNC_BACKOFF_FACTOR = 1.5;       // 증가 배수
export const SYNC_MAX_INTERVAL_MS = 30_000;   // 최대 간격

let syncConsecutiveNoChange = 0;                    // 연속 '변경 없음' 카운터
let syncCurrentIntervalMs = SYNC_BASE_INTERVAL_MS;  // 현재 적용 중인 간격

export function computeSyncInterval(): number {
    if (syncConsecutiveNoChange === 0) return SYNC_BASE_INTERVAL_MS;
    const interval = SYNC_BASE_INTERVAL_MS * Math.pow(SYNC_BACKOFF_FACTOR, syncConsecutiveNoChange);
    return Math.min(interval, SYNC_MAX_INTERVAL_MS);
}

export function updateSyncBackoff(hasChange: boolean): void {
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

/** 백그라운드 스로틀 계산용. 기존 main.ts 의 syncCurrentIntervalMs 직접 참조를 대체합니다. */
export function getSyncIntervalMs(): number {
    return syncCurrentIntervalMs;
}

/** 사용자가 모드 탭을 전환했을 때 즉시 기본 간격으로 되돌립니다. */
export function resetSyncBackoff(): void {
    syncConsecutiveNoChange = 0;
    syncCurrentIntervalMs = SYNC_BASE_INTERVAL_MS;
}

// =====================================================================
// 🌟 [SHARED BLOB CODEC]
//  analytic(console.logis.center) 과 trading(trading.logis.center) 이
//  동일한 gzip/base64 BLOB 포맷을 쓰므로 공용 위치로 승격합니다.
//  (main.ts 의 decodeAnalyticBlob 원본과 동작이 100% 동일합니다)
// =====================================================================
export function decodeAnalyticBlob(rawData: any): any {
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

// =====================================================================
// 🌟 [SHARED URL HELPER]
//  chrome.js 와 완벽히 동일한 루트 도메인 추출 로직입니다. (필터 엇갈림 원천 차단)
//  ── 왜 여기로 옮기는가 ──
//   commerce 트랙(cc = hashId(rootDomain))과 main.ts 의 버튼 가시성 판정이
//   같은 규칙을 써야 하는데, 파일이 분리되면 모듈 스코프 함수를 공유할 수 없습니다.
//   두 벌로 복사하면 '한쪽만 도메인 규칙이 바뀌는' 사고가 재현되므로 단일 소스로 둡니다.
// =====================================================================
export const twoPartDomains = ["co.kr","co.uk","co.jp","com.cn","co.in","com.mx","co.id","com.my","com.sg","com.ph","com.vn"];

export function getRootDomain(hostname: string) {
    const host = hostname.split('.');
    const isTwoPart = twoPartDomains.some(domain => hostname.endsWith(domain));
    if (isTwoPart && host.length >= 3) {
        return host[host.length - 3] + "." + host[host.length - 2] + "." + host[host.length - 1];
    } else if (host.length >= 2) {
        return host[host.length - 2] + "." + host[host.length - 1];
    }
    return hostname;
}

// =====================================================================
// 🌟 [SHARED ENVELOPE CONTRACT]
//  봉투 루트 키 목록입니다. main.ts 의 normalizeEnvelope(ROOT ABSORB)와
//  commerce 트랙의 ROOT ABSORB / RUST 보강 루프가 반드시 같은 집합을 봐야 합니다.
//  두 규칙이 어긋나면 루트 컬럼이 data 로 내려가지 않아 인덱스가 조용히 비어 버립니다.
// =====================================================================
export const ENVELOPE_ROOT_KEYS = new Set([
    'id', 'uuid', 'type', 'doc_type', 'flag', 'from', 'to', 'cc', 'bcc',
    'ref', 'ref_val', 'mode', 'created_at', 'updated_at',
    'created_at_ts', 'updated_at_ts',
    // 아래 4개는 data 안으로 별도 승격되므로 하강 루프에서 제외합니다.
    'data', 'json_data', 'text', 'masked_text',
    // 클라우드 응답 전용 라우팅 힌트 (도메인 값이 아님)
    'table', 'digest', 'current', 'score', 'name'
]);