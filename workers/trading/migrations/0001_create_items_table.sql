CREATE TABLE IF NOT EXISTS items (
    "id"         TEXT PRIMARY KEY,
    "type"       TEXT,
    "flag"       TEXT,
    "from"       TEXT,
    "to"         TEXT,
    "cc"         TEXT,
    "bcc"        TEXT,
    "ref"        TEXT,
    "mode"       TEXT,
    "digest"     TEXT,
    "data"       BLOB NULL,
    "created_at" INTEGER,
    "updated_at" INTEGER
);

-- 🌟 [SYNC 축] GET 의 기본 정렬/필터.
--    WHERE "to" = ? AND "mode" = ? AND "created_at" < ? ORDER BY created_at DESC
CREATE INDEX IF NOT EXISTS idx_items_team_sync
    ON items ("to", "created_at" DESC);

-- 🌟 [거래 건 축] ?ref= 로 한 선적 건의 전 서류를 뽑을 때 씁니다.
CREATE INDEX IF NOT EXISTS idx_items_team_ref
    ON items ("to", "ref");

-- 🌟 [서식 버킷 축] ?doc_type= 필터 및 네비게이션 카운트.
CREATE INDEX IF NOT EXISTS idx_items_team_bcc
    ON items ("to", "bcc");

-- 🌟 [SKIP GUARD 축] POST 의 digest 사전 조회.
--    WHERE "to" = ?1 AND "id" IN (...) 는 PK 만으로도 되지만,
--    팀 조건이 앞에 붙으므로 복합 인덱스가 있어야 실제로 쓰입니다.
CREATE INDEX IF NOT EXISTS idx_items_team_id
    ON items ("to", "id");

-- 🌟 [서식 종류 축] doc_type 단독 필터.
CREATE INDEX IF NOT EXISTS idx_items_team_type
    ON items ("to", "type");