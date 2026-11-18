//! 원본 proxy/src/index.ts 의 모든 템플릿 리터럴 (gen_templates.py 로 생성 — 손으로 고치지 말 것).
//! `T<줄>[_<n>]` 은 조각 목록이고 조각 사이의 식은 주석의 원문이다. [`crate::tpl`] 로 잇는다.
#![allow(dead_code)]

/// ${env.aws_bucket} | ${env.aws_region} | ${cookies.hash}
pub const T553: &[&str] = &["https://", ".s3.", ".amazonaws.com/hash/", ""];
/// ${cookies.address} | ${cookies.address}
pub const T621: &[&str] = &["SELECT * FROM users WHERE \"type\" = 'user' AND \"id\" = '", "' AND \"from\" = '", "' LIMIT 1"];
pub const T672: &[&str] = &["\n\t\t\t\t\t\t\t\tINSERT INTO users (\"id\", \"type\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\", \"data\", \"created_at\", \"updated_at\") \n\t\t\t\t\t\t\t\tVALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10) ON CONFLICT (id) DO UPDATE SET\n\t\t\t\t\t\t\t\t\t\"type\" = EXCLUDED.\"type\",\n\t\t\t\t\t\t\t\t\t\"from\" = EXCLUDED.\"from\",\n\t\t\t\t\t\t\t\t\t\"to\" = EXCLUDED.\"to\",\n\t\t\t\t\t\t\t\t\t\"cc\" = EXCLUDED.\"cc\",\n\t\t\t\t\t\t\t\t\t\"bcc\" = EXCLUDED.\"bcc\",\n\t\t\t\t\t\t\t\t\t\"ref\" = EXCLUDED.\"ref\",\n\t\t\t\t\t\t\t\t\t\"data\" = EXCLUDED.\"data\",\n\t\t\t\t\t\t\t\t\t\"created_at\" = EXCLUDED.\"created_at\",\n\t\t\t\t\t\t\t\t\t\"updated_at\" = EXCLUDED.\"updated_at\"\n\t\t\t\t\t\t\t"];
pub const T696: &[&str] = &["SELECT * FROM zones"];
pub const T836: &[&str] = &["\n\t\t\t\t\t\t\t\t\tINSERT INTO users (\"id\", \"type\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\", \"data\", \"created_at\", \"updated_at\") \n\t\t\t\t\t\t\t\t\tVALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)\n\t\t\t\t\t\t\t\t"];
pub const T845: &[&str] = &["\n\t\t\t\t\t\t\t\t\tINSERT INTO users (\"id\", \"type\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\", \"data\", \"created_at\", \"updated_at\") \n\t\t\t\t\t\t\t\t\tVALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10) ON CONFLICT (id) DO NOTHING\n\t\t\t\t\t\t\t\t"];
pub const T856: &[&str] = &["\n\t\t\t\t\t\t\t\t\tINSERT INTO zones (\"id\", \"pool\") \n\t\t\t\t\t\t\t\t\tVALUES (?1, ?2)\n\t\t\t\t\t\t\t\t\tON CONFLICT(id) DO UPDATE SET\n\t\t\t\t\t\t\t\t\t\t\"pool\" = EXCLUDED.\"pool\"\n\t\t\t\t\t\t\t\t"];
/// ${env.aws_bucket} | ${env.aws_region} | ${clientAddress}
pub const T874: &[&str] = &["https://", ".s3.", ".amazonaws.com/address/", ""];
/// ${cookies.hash}
pub const T908: &[&str] = &["hash=", "; Secure; HttpOnly; SameSite=None"];
/// ${cookies.token}
pub const T909: &[&str] = &["token=", "; Secure; HttpOnly; SameSite=None"];
/// ${cookies.team}
pub const T912: &[&str] = &["team=", "; Secure; HttpOnly; SameSite=None"];
/// ${env.aws_bucket} | ${env.aws_region} | ${hash}
pub const T932: &[&str] = &["https://", ".s3.", ".amazonaws.com/hash/", ""];
/// ${key}
pub const T945: &[&str] = &["x-amz-meta-", ""];
/// ${hash}
pub const T959: &[&str] = &["hash=", "; Secure; HttpOnly; SameSite=None"];
/// ${token}
pub const T960: &[&str] = &["token=", "; Secure; HttpOnly; SameSite=None"];
/// ${origin}
pub const T972: &[&str] = &["frame-ancestors 'self' ", ";"];
/// ${hashId(cookies.cc+_url.pathname.toLowerCase())} | ${created_at}
pub const T1005: &[&str] = &["SELECT * FROM pages WHERE \"id\" = '", "' AND \"created_at\" < ", " LIMIT 1"];
/// ${cookies.team} | ${created_at}
pub const T1089: &[&str] = &["SELECT * FROM users WHERE \"to\" = '", "' AND \"created_at\" < ", " ORDER BY created_at DESC LIMIT 10"];
/// ${zoneRegion}
pub const T1132: &[&str] = &["commerce_logis_", "_items"];
/// ${cookies.team} | ${itemId}
pub const T1132_1: &[&str] = &["SELECT * FROM items WHERE \"to\" = '", "' AND \"ref\" = '", "' LIMIT 1"];
/// ${zoneRegion}
pub const T1145: &[&str] = &["commerce_logis_", "_items"];
/// ${cookies.team}
pub const T1145_1: &[&str] = &["SELECT * FROM items WHERE \"type\" = 'pages' AND \"to\" = '", "' LIMIT 100"];
/// ${zoneRegion}
pub const T1257: &[&str] = &["commerce_logis_", "_items"];
/// ${cookies.cc} | ${created_at}
pub const T1257_1: &[&str] = &["SELECT * FROM items WHERE \"type\" = 'pages' AND \"cc\" = '", "' AND \"created_at\" < ", " LIMIT 100"];
/// ${zoneRegion}
pub const T1331: &[&str] = &["commerce_logis_", "_items"];
pub const T1331_1: &[&str] = &["\n\t\t\t\t\t\t\t\t\t\tINSERT INTO items (\n\t\t\t\t\t\t\t\t\t\t\t\"id\", \"type\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\", \"digest\", \"data\", \"created_at\", \"updated_at\"\n\t\t\t\t\t\t\t\t\t\t) VALUES (\n\t\t\t\t\t\t\t\t\t\t\t?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11\n\t\t\t\t\t\t\t\t\t\t) ON CONFLICT (id) DO UPDATE SET\n\t\t\t\t\t\t\t\t\t\t\t\"type\" = EXCLUDED.\"type\",\n\t\t\t\t\t\t\t\t\t\t\t\"from\" = EXCLUDED.\"from\",\n\t\t\t\t\t\t\t\t\t\t\t\"to\" = EXCLUDED.\"to\",\n\t\t\t\t\t\t\t\t\t\t\t\"cc\" = EXCLUDED.\"cc\",\n\t\t\t\t\t\t\t\t\t\t\t\"bcc\" = EXCLUDED.\"bcc\",\n\t\t\t\t\t\t\t\t\t\t\t\"ref\" = EXCLUDED.\"ref\",\n\t\t\t\t\t\t\t\t\t\t\t\"digest\" = EXCLUDED.\"digest\",\n\t\t\t\t\t\t\t\t\t\t\t\"data\" = EXCLUDED.\"data\",\n\t\t\t\t\t\t\t\t\t\t\t\"created_at\" = EXCLUDED.\"created_at\",\n\t\t\t\t\t\t\t\t\t\t\t\"updated_at\" = EXCLUDED.\"updated_at\"\n\t\t\t\t\t\t\t\t\t"];
/// ${zoneRegion}
pub const T1371: &[&str] = &["commerce_logis_", "_items"];
/// ${pageId} | ${created_at}
pub const T1374: &[&str] = &["SELECT * FROM pages WHERE \"id\" = '", "' AND \"created_at\" < ", " LIMIT 1"];
/// ${cron} | ${cookies.address} | ${now}
pub const T1429: &[&str] = &[" OR (\"id\" = '", "' AND \"from\" = '", "' AND \"created_at\" < ", ") "];
/// ${cron_condition} | ${crons.length}
pub const T1451: &[&str] = &["SELECT * FROM tasks WHERE ", " ORDER BY created_at LIMIT ", ""];
/// ${_page.data.origin} | ${_page.type}
pub const T1490: &[&str] = &["", "#", ""];
/// ${_page.data.origin} | ${_page.type}
pub const T1506: &[&str] = &["", "#", ""];
/// ${_page.data.origin} | ${_page.type}
pub const T1512: &[&str] = &["", "#", ""];
/// ${_page.data.origin} | ${_page.type}
pub const T1517: &[&str] = &["", "#", ""];
/// ${_page.data.origin} | ${_page.type}
pub const T1522: &[&str] = &["", "#", ""];
/// ${_page.data.origin} | ${_page.type}
pub const T1523: &[&str] = &["", "#", ""];
/// ${_page.data.origin} | ${_page.type}
pub const T1529: &[&str] = &["", "#", ""];
/// ${zoneRegion}
pub const T1567: &[&str] = &["commerce_logis_", "_items"];
/// ${cookies.address} | ${cookies.team} | ${cookies.bcc}
pub const T1567_1: &[&str] = &["SELECT * FROM items WHERE \"type\" = 'member' AND \"from\" = '", "' AND \"to\" = '", "' AND \"bcc\" = '", "' LIMIT 1"];
/// ${zoneRegion}
pub const T1572: &[&str] = &["commerce_logis_", "_items"];
/// ${cookies.address} | ${cookies.team} | ${cookies.cc}
pub const T1572_1: &[&str] = &["SELECT * FROM items WHERE \"type\" = 'member' AND \"from\" = '", "' AND \"to\" = '", "' AND \"bcc\" = '", "' LIMIT 1"];
/// ${vectorRegion} | ${vectorTable}
pub const T1677: &[&str] = &["", "-", ""];
/// ${zoneRegion}
pub const T1739: &[&str] = &["commerce_logis_", "_items"];
pub const T1740: &[&str] = &["SELECT * FROM items WHERE \"id\" = ?1 AND \"type\" = ?2 AND \"from\" = ?3 AND \"to\" = ?4 AND \"bcc\" = ?5 ORDER BY created_at DESC"];
/// ${zoneRegion}
pub const T1749: &[&str] = &["commerce_logis_", "_items"];
pub const T1750: &[&str] = &["UPDATE items SET to = ?, updated_at = ? WHERE id = ?"];
/// ${zoneRegion}
pub const T1771: &[&str] = &["commerce_logis_", "_items"];
/// ${memberId}
pub const T1772: &[&str] = &["SELECT * FROM items WHERE \"id\" = '", "' AND \"type\" = 'member' ORDER BY created_at DESC"];
/// ${zoneRegion}
pub const T1778: &[&str] = &["commerce_logis_", "_items"];
/// ${memberId} | ${req.query.to} | ${req.query.cc}
pub const T1779: &[&str] = &["SELECT * FROM items WHERE \"id\" = '", "' AND \"from\" = '", "' AND \"cc\" = '", "' ORDER BY created_at DESC"];
/// ${zoneRegion}
pub const T1800: &[&str] = &["commerce_logis_", "_items"];
pub const T1800_1: &[&str] = &["\n                                                            DELETE FROM items WHERE \"type\" = 'member' AND \"from\" = ?1 AND \"to\" = ?2 AND \"cc\" = ?3\n                                                        "];
/// ${zoneRegion}
pub const T1822: &[&str] = &["commerce_logis_", "_items"];
pub const T1822_1: &[&str] = &["\n                                                                    INSERT INTO items (\"id\", \"type\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\", \"data\", \"created_at\", \"updated_at\") \n                                                                    VALUES (?1, 'member', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)\n                                                                "];
/// ${zoneRegion}
pub const T1842: &[&str] = &["commerce_logis_", "_items"];
pub const T1842_1: &[&str] = &["\n                                                                INSERT INTO items (\"id\", \"type\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\", \"data\", \"created_at\", \"updated_at\") \n                                                                VALUES (?1, 'member', ?2, ?3, ?4, ?5, '', ?6, ?7, ?8)\n                                                            "];
/// ${zoneRegion}
pub const T1885: &[&str] = &["commerce_logis_", "_items"];
/// ${zoneRegion}
pub const T1921: &[&str] = &["commerce_logis_", "_talks"];
pub const T1921_1: &[&str] = &["\n\t\t\t\t\t\t\t\tINSERT INTO talks (\n\t\t\t\t\t\t\t\t\t\"id\", \"type\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\", \"data\", \"created_at\", \"updated_at\"\n\t\t\t\t\t\t\t\t) VALUES (\n\t\t\t\t\t\t\t\t\t?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10\n\t\t\t\t\t\t\t\t) ON CONFLICT (id) DO UPDATE SET\n\t\t\t\t\t\t\t\t\t\"type\" = EXCLUDED.\"type\",\n\t\t\t\t\t\t\t\t\t\"from\" = EXCLUDED.\"from\",\n\t\t\t\t\t\t\t\t\t\"to\" = EXCLUDED.\"to\",\n\t\t\t\t\t\t\t\t\t\"cc\" = EXCLUDED.\"cc\",\n\t\t\t\t\t\t\t\t\t\"bcc\" = EXCLUDED.\"bcc\",\n\t\t\t\t\t\t\t\t\t\"ref\" = EXCLUDED.\"ref\",\n\t\t\t\t\t\t\t\t\t\"data\" = EXCLUDED.\"data\",\n\t\t\t\t\t\t\t\t\t\"created_at\" = EXCLUDED.\"created_at\",\n\t\t\t\t\t\t\t\t\t\"updated_at\" = EXCLUDED.\"updated_at\"\n\t\t\t\t\t\t\t"];
/// ${zoneRegion}
pub const T1970: &[&str] = &["commerce_logis_", "_items"];
/// ${ref}
pub const T1971: &[&str] = &["DELETE FROM tasks WHERE ref = \"", "\""];
/// ${env.aws_bucket} | ${env.aws_region} | ${cookies.hash}
pub const T1976: &[&str] = &["https://", ".s3.", ".amazonaws.com/hash/", ""];
/// ${cookies.team} | ${hashId(page.type+cookies.cc.toUpperCase())} | ${hashId(cookies.team+cookies.cc+link)} | ${created_at}
pub const T2172: &[&str] = &["(\"to\" = \"", "\" AND \"bcc\" = '", "' AND \"ref\" = '", "' AND \"created_at\" < ", ")"];
/// ${zoneRegion}
pub const T2186: &[&str] = &["commerce_logis_", "_items"];
/// ${condition}
pub const T2187: &[&str] = &["SELECT * FROM items WHERE ", " ORDER BY created_at DESC LIMIT 1000"];
/// ${zoneRegion}
pub const T2326: &[&str] = &["commerce_logis_", "_items"];
/// ${cookies.address} | ${$to} | ${now}
pub const T2326_1: &[&str] = &["SELECT * FROM items WHERE \"type\" = 'invoice' AND \"from\" = '", "' AND \"to\" = '", "' AND \"created_at\" < ", " LIMIT 1"];
/// ${task.id} | ${now}
pub const T2357: &[&str] = &["SELECT * FROM tasks WHERE \"id\" = '", "' AND \"created_at\" < ", "  ORDER BY created_at LIMIT 1"];
pub const T2393: &[&str] = &["ON CONFLICT (id) DO UPDATE SET\n\t\t\t\t\t\t\t\t\t\t\t\"cc\" = EXCLUDED.\"cc\",\n\t\t\t\t\t\t\t\t\t\t\t\"bcc\" = EXCLUDED.\"bcc\",\n\t\t\t\t\t\t\t\t\t\t\t\"ref\" = EXCLUDED.\"ref\",\n\t\t\t\t\t\t\t\t\t\t\t\"task\" = EXCLUDED.\"task\",\n\t\t\t\t\t\t\t\t\t\t\t\"created_at\" = EXCLUDED.\"created_at\",\n\t\t\t\t\t\t\t\t\t\t\t\"updated_at\" = EXCLUDED.\"updated_at\"\n\t\t\t\t\t\t\t\t\t\t"];
/// ${onConflict}
pub const T2404: &[&str] = &["\n\t\t\t\t\t\t\t\t\t\tINSERT INTO tasks (\n\t\t\t\t\t\t\t\t\t\t\t\"id\", \"type\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\", \"data\", \"created_at\", \"updated_at\"\n\t\t\t\t\t\t\t\t\t\t) VALUES (\n\t\t\t\t\t\t\t\t\t\t\t?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10\n\t\t\t\t\t\t\t\t\t\t) ", "\n\t\t\t\t\t\t\t\t\t"];
/// ${created_at}
pub const T2481: &[&str] = &["AND \"created_at\" < ", ""];
/// ${zoneRegion}
pub const T2496: &[&str] = &["commerce_logis_", "_items"];
/// ${cookies.cc} | ${cookies.team} | ${condition}
pub const T2497: &[&str] = &["SELECT * FROM items WHERE \"cc\" = '", "' AND \"to\" = '", "' ", " AND \"updated_at\" > 0 ORDER BY created_at DESC LIMIT 100"];
/// ${zoneRegion}
pub const T2520: &[&str] = &["commerce_logis_", "_items"];
/// ${cookies.cc} | ${cookies.team} | ${condition}
pub const T2521: &[&str] = &["SELECT * FROM items WHERE \"cc\" != '", "' AND \"to\" = '", "' ", "  AND \"updated_at\" > 0 ORDER BY created_at DESC LIMIT 100"];
/// ${zoneRegion}
pub const T2542: &[&str] = &["commerce_logis_", "_items"];
/// ${cookies.team} | ${condition}
pub const T2543: &[&str] = &["SELECT * FROM items WHERE \"to\" = '", "' ", " AND \"updated_at\" = 0 ORDER BY created_at DESC LIMIT 100"];
/// ${cookies.address} | ${cookies.cc} | ${condition}
pub const T2566: &[&str] = &["SELECT * FROM tasks WHERE \"from\" = '", "' AND \"cc\" = '", "' ", " ORDER BY created_at DESC LIMIT 10"];
/// ${zoneRegion}
pub const T2586: &[&str] = &["commerce_logis_", "_talks"];
/// ${cookies.cc} | ${itemId} | ${conditions[page.type] ? conditions[page.type] : condition}
pub const T2587: &[&str] = &["SELECT * FROM talks WHERE \"cc\" = '", "' AND \"data\" IS NOT NULL AND \"ref\" = '", "' ", " ORDER BY created_at DESC LIMIT 10"];
/// ${zoneRegion}
pub const T2611: &[&str] = &["commerce_logis_", "_talks"];
/// ${cookies.cc} | ${itemId}
pub const T2612: &[&str] = &["SELECT COUNT(*) AS \"talks\" FROM talks WHERE \"cc\" = '", "' AND \"data\" IS NOT NULL AND \"ref\" = '", "'"];
/// ${zoneRegion}
pub const T2626: &[&str] = &["commerce_logis_", "_talks"];
/// ${cookies.team} | ${cookies.address} | ${conditions[page.type] ? conditions[page.type] : condition}
pub const T2627: &[&str] = &["SELECT * FROM talks WHERE \"from\" = '", "' AND \"to\" = '", "' ", " ORDER BY created_at DESC LIMIT 10"];
/// ${type} | ${cookies.team} | ${cookies.cc} | ${page.bcc} | ${conditions[page.type] ? conditions[page.type] : condition}
pub const T2650: &[&str] = &["SELECT * FROM ", " WHERE \"to\" = '", "' AND \"cc\" = '", "' AND \"bcc\" = '", "' ", " ORDER BY created_at DESC LIMIT 100"];
/// ${zoneRegion}
pub const T2654: &[&str] = &["commerce_logis_", "_items"];
/// ${page.type} | ${detail.digest} | ${condition}
pub const T2655: &[&str] = &["SELECT * FROM items WHERE \"type\" = '", "' AND \"digest\" = '", "' ", " ORDER BY created_at DESC LIMIT 1"];
/// ${page.type} | ${hint.id} | ${condition}
pub const T2664: &[&str] = &[" (\"type\" = '", "' AND \"id\" = '", "' ", ") "];
/// ${zoneRegion} | ${type}
pub const T2669: &[&str] = &["commerce_logis_", "_", ""];
/// ${type} | ${hints_condition} | ${condition}
pub const T2670: &[&str] = &["SELECT * FROM ", " WHERE ", " ", " ORDER BY created_at DESC LIMIT 1"];
/// ${type} | ${detail.id} | ${condition}
pub const T2687: &[&str] = &["SELECT * FROM ", " WHERE \"id\" = '", "' ", " ORDER BY created_at DESC LIMIT 1"];
/// ${zoneRegion} | ${type}
pub const T2688: &[&str] = &["commerce_logis_", "_", ""];
/// ${type} | ${detail.id} | ${condition}
pub const T2689: &[&str] = &["SELECT * FROM ", " WHERE \"id\" = '", "' ", " ORDER BY created_at DESC LIMIT 1"];
/// ${zoneRegion}
pub const T2702: &[&str] = &["commerce_logis_", "_sales"];
/// ${$item.index} | ${cookies.cc} | ${cookies.team} | ${conditions[type] ? conditions[type] : condition}
pub const T2703: &[&str] = &["SELECT * FROM sales WHERE \"goods\" = ", " AND \"cc\" = '", "' AND \"to\" = '", "' ", " ORDER BY created_at DESC LIMIT 10000"];
/// ${zoneRegion}
pub const T2724: &[&str] = &["commerce_logis_", "_tracking"];
/// ${$item.index} | ${cookies.team} | ${conditions[type] ? conditions[type] : condition}
pub const T2725: &[&str] = &["SELECT * FROM tracking WHERE \"order\" = ", " AND \"to\" = '", "' ", " ORDER BY created_at DESC LIMIT 10000"];
/// ${zoneRegion}
pub const T2745: &[&str] = &["commerce_logis_", "_tracking"];
/// ${item.index} | ${cookies.team} | ${conditions[type] ? conditions[type] : condition}
pub const T2746: &[&str] = &["SELECT * FROM tracking WHERE \"order\" = ", " AND \"to\" = '", "' ", " ORDER BY created_at DESC LIMIT 10000"];
/// ${zoneRegion} | ${type}
pub const T2770: &[&str] = &["commerce_logis_", "_", ""];
/// ${type} | ${cookies.team} | ${cookies.cc} | ${page.bcc} | ${conditions[page.type] ? conditions[page.type] : condition}
pub const T2771: &[&str] = &["SELECT * FROM ", " WHERE \"to\" = '", "' AND \"cc\" = '", "' AND \"bcc\" = '", "' ", " ORDER BY created_at DESC LIMIT 100"];
pub const T2853: &[&str] = &["[API-DEBUG] 클라이언트(Client)로 반환하는 데이터 통계"];
/// ${cookies.cc} | ${cookies.team} | ${type || 'all'}
pub const T2854: &[&str] = &["[API-DEBUG] 사용된 CC: ", ", 팀: ", ", 조회 타입: ", ""];
/// ${pages.length} | ${tasks.length} | ${talks.length} | ${items.length} | ${users.length}
pub const T2855: &[&str] = &["[API-DEBUG] 항목별 개수 - Pages: ", ", Tasks: ", ", Talks: ", ", Items: ", ", Users: ", ""];
pub const T2879: &[&str] = &["I'm a teapot!"];
