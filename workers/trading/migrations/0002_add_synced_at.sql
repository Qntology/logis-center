-- 🌟 [SERVER STAMP] fix-trading-delta-cursor (docs/workers-rs.md "고친 버그")
--    변경분(GET since=)을 클라이언트가 보낸 시각 max(updated_at, created_at) 으로 고르면,
--    오프라인 편집 · 처음 올리는 옛 서류 · 시계가 늦은 기기의 서류처럼 커서보다 옛 시각으로 나중에 들어온 행을
--    다른 기기가 영영 받지 못합니다. 서버가 쓸 때 찍는 시각 하나로 고르고 그 순서로 쪽을 나눕니다.
ALTER TABLE items ADD COLUMN "synced_at" INTEGER NOT NULL DEFAULT 0;

-- 이미 있는 행은 예전 기준(두 시각 중 큰 값)으로 채웁니다.
UPDATE items SET "synced_at" = max(coalesce("updated_at", 0), coalesce("created_at", 0)) WHERE "synced_at" = 0;

-- WHERE "to" = ? AND "mode" = ? AND "synced_at" > ? ORDER BY "synced_at", "id"
CREATE INDEX IF NOT EXISTS idx_items_team_synced
    ON items ("to", "mode", "synced_at", "id");
