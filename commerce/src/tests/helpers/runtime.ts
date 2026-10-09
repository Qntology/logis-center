// Fake ModeRuntime (normally injected by main.ts via bindModeRuntime) for the modes/* sync tracks.
import { vi } from "vitest";
import type { ModeRuntime, ModeSession, ModeTag } from "../../modes/runtime";
import type { FakeDb } from "./fakedb";

export interface FakeRuntimeOptions {
  session?: Partial<ModeSession>;
  appDb?: FakeDb | null;
  timezoneOffset?: number;
  searchMode?: string;
  currentTab?: string;
  detectedUrl?: string;
  context?: { cc: string; bcc: string; ref: string };
  activeTags?: ModeTag[];
  itemTombstones?: string[];
  talkTombstones?: string[];
  busy?: boolean;
  cloudPendingTasks?: Map<string, any>;
  kv?: Record<string, unknown>;
}

export function makeRuntime(opts: FakeRuntimeOptions = {}) {
  const kv = new Map<string, unknown>(Object.entries(opts.kv ?? {}));
  const state = {
    session: { hash: "", ...opts.session } as ModeSession,
    busy: opts.busy ?? false,
    cloudPendingTasks: opts.cloudPendingTasks ?? new Map<string, any>(),
  };

  const runtime: ModeRuntime = {
    appDb: opts.appDb ?? null,
    timezoneOffset: opts.timezoneOffset ?? 0,
    kvGet: vi.fn(async (key: string) => (kv.has(key) ? kv.get(key) : null)),
    kvSet: vi.fn(async (key: string, value: unknown) => {
      kv.set(key, value);
    }),
    normalizeEnvelope: vi.fn((docs: any[]) => docs),
    loadItemTombstones: vi.fn(async () => new Set(opts.itemTombstones ?? [])),
    loadTalkTombstones: vi.fn(async () => new Set(opts.talkTombstones ?? [])),
    getSession: () => state.session,
    getContext: () => opts.context ?? { cc: "", bcc: "", ref: "" },
    getSearchMode: () => opts.searchMode ?? "commerce",
    getDetectedUrl: () => opts.detectedUrl ?? "",
    getActiveTags: () => opts.activeTags ?? [],
    getCurrentTab: () => opts.currentTab ?? "list",
    isBusy: () => state.busy,
    getDevicePref: () => null,
    getCloudPendingTasks: () => state.cloudPendingTasks,
    renderNavigation: vi.fn(async () => {}),
    loadMoreDocs: vi.fn(async () => {}),
    renderMessage: vi.fn(async () => {}),
    renderProgressToUI: vi.fn(async () => {}),
    fetchChatHistory: vi.fn(async () => {}),
    runLocalEmbeddingSync: vi.fn(),
    stopSpinner: vi.fn(),
    stepQrSpinner: vi.fn(),
    restoreSubmitButton: vi.fn(),
  };

  return { runtime, kv, state };
}

/** All `invoke` calls for one command, as [args] tuples. */
export function callsOf(mock: { mock: { calls: any[][] } }, command: string): any[] {
  return mock.mock.calls.filter(([cmd]) => cmd === command).map(([, args]) => args);
}
