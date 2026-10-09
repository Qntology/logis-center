// Minimal in-memory stand-in for the subset of the Dexie API used by src/lib and src/modes.
// Like IndexedDB: rows are keyed by `id`, index comparisons are type-strict (5 !== "5"),
// and reads/writes go through structuredClone so callers never share references with the store.

export type Row = Record<string, any>;
export type DbOp = [op: string, table: string, detail: unknown];

const readPath = (row: unknown, path: string): unknown =>
  path.split(".").reduce<any>((cur, seg) => (cur === null || cur === undefined ? undefined : cur[seg]), row);

const clone = <T>(value: T): T => structuredClone(value);

export interface FakeCollection {
  toArray(): Promise<Row[]>;
  limit(n: number): FakeCollection;
  filter(fn: (row: Row) => boolean): FakeCollection;
  modify(patch: Record<string, unknown>): Promise<number>;
}

export interface FakeTable {
  where(path: string): { equals(value: unknown): FakeCollection; anyOf(values: unknown[]): FakeCollection };
  toArray(): Promise<Row[]>;
  get(id: string): Promise<Row | undefined>;
  put(row: Row): Promise<void>;
  bulkPut(rows: Row[]): Promise<void>;
  delete(id: string): Promise<void>;
}

export interface FakeDb {
  table(name: string): FakeTable;
  /** Snapshot of the rows currently stored in `name`. */
  rows(name: string): Row[];
  /** Ids currently stored in `name`. */
  ids(name: string): string[];
  /** Chronological write log: ["put" | "bulkPut" | "delete" | "modify", table, ids/patch]. */
  log: DbOp[];
}

export function makeDb(seed: Record<string, Row[]> = {}): FakeDb {
  const tables = new Map<string, Map<string, Row>>();
  const log: DbOp[] = [];

  const store = (name: string): Map<string, Row> => {
    let t = tables.get(name);
    if (!t) {
      t = new Map((seed[name] ?? []).map((r) => [r.id, clone(r)]));
      tables.set(name, t);
    }
    return t;
  };

  const collection = (name: string, rows: Row[]): FakeCollection => ({
    toArray: async () => rows.map(clone),
    limit: (n) => collection(name, rows.slice(0, n)),
    filter: (fn) => collection(name, rows.filter((r) => fn(clone(r)))),
    modify: async (patch) => {
      for (const r of rows) {
        const live = store(name).get(r.id);
        if (!live) continue;
        for (const [key, value] of Object.entries(patch)) {
          const segs = key.split(".");
          let target: any = live;
          while (segs.length > 1) {
            const seg = segs.shift() as string;
            target[seg] ??= {};
            target = target[seg];
          }
          target[segs[0]] = value;
        }
      }
      log.push(["modify", name, { ids: rows.map((r) => r.id), patch }]);
      return rows.length;
    },
  });

  const table = (name: string): FakeTable => ({
    where: (path) => ({
      equals: (value) => collection(name, [...store(name).values()].filter((r) => readPath(r, path) === value)),
      anyOf: (values) => collection(name, [...store(name).values()].filter((r) => values.includes(readPath(r, path)))),
    }),
    toArray: async () => [...store(name).values()].map(clone),
    get: async (id) => {
      const row = store(name).get(id);
      return row ? clone(row) : undefined;
    },
    put: async (row) => {
      log.push(["put", name, row.id]);
      store(name).set(row.id, clone(row));
    },
    bulkPut: async (rows) => {
      log.push(["bulkPut", name, rows.map((r) => r.id)]);
      for (const r of rows) store(name).set(r.id, clone(r));
    },
    delete: async (id) => {
      log.push(["delete", name, id]);
      store(name).delete(id);
    },
  });

  return {
    table,
    rows: (name) => [...store(name).values()].map(clone),
    ids: (name) => [...store(name).keys()],
    log,
  };
}
