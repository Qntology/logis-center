//! 원본 proxy/src/index.ts 의 모든 템플릿 리터럴 (gen_templates.py 로 생성 — 손으로 고치지 말 것).
//! `T<줄>[_<n>]` 은 조각 목록이고 조각 사이의 식은 주석의 원문이다. [`crate::tpl`] 로 잇는다.
#![allow(dead_code)]

/// ${hash}
pub const T402: &[&str] = &["https://trading-team-cache.invalid/", ""];
/// ${to} | ${cached.team}
pub const T412: &[&str] = &["[TRADING] 🚫 (cache) 요청 팀 '", "' != 확인된 팀 '", "'. 본인 팀으로 강등합니다."];
/// ${COMMERCE_HOST} | ${probe.toString()}
pub const T427: &[&str] = &["", "/?", ""];
/// ${to}
pub const T439: &[&str] = &["[TRADING] 🚫 팀 확인 실패. 요청 팀 '", "' 을 거부합니다. (commerce 응답 없음)"];
/// ${to} | ${resolvedTeam}
pub const T452: &[&str] = &["[TRADING] 🚫 요청 팀 '", "' != 확인된 팀 '", "'. 본인 팀으로 강등합니다."];
/// ${env.aws_bucket} | ${env.aws_region} | ${cookies.hash}
pub const T528: &[&str] = &["https://", ".s3.", ".amazonaws.com/hash/", ""];
/// ${address}
pub const T556: &[&str] = &["[TRADING] ⛔ 팀 확인 실패로 이 요청은 데이터 접근이 차단됩니다. (address='", "')"];
/// ${cookies.hash}
pub const T559: &[&str] = &["hash=", "; Secure; HttpOnly; SameSite=None"];
/// ${cookies.token}
pub const T560: &[&str] = &["token=", "; Secure; HttpOnly; SameSite=None"];
/// ${cookies.team}
pub const T562: &[&str] = &["team=", "; Secure; HttpOnly; SameSite=None"];
/// ${env.aws_bucket} | ${env.aws_region} | ${hash}
pub const T576: &[&str] = &["https://", ".s3.", ".amazonaws.com/hash/", ""];
/// ${k}
pub const T589: &[&str] = &["x-amz-meta-", ""];
/// ${hash}
pub const T598: &[&str] = &["hash=", "; Secure; HttpOnly; SameSite=None"];
/// ${token}
pub const T599: &[&str] = &["token=", "; Secure; HttpOnly; SameSite=None"];
pub const T622: &[&str] = &["<!doctype html><meta charset=\"utf-8\"><title>trading.logis.center</title>"];
/// ${JSON.stringify(cookies, null, 2)}
pub const T623: &[&str] = &["<pre>", "</pre>"];
pub const T643: &[&str] = &["\"to\" = ?1 AND \"mode\" = ?2 AND \"created_at\" < ?3 AND max(\"updated_at\",\"created_at\") > ?4"];
/// ${slot}
pub const T649: &[&str] = &[" AND \"ref\" = ?", ""];
/// ${slot}
pub const T655: &[&str] = &[" AND \"type\" = ?", ""];
/// ${where} | ${limit}
pub const T668: &[&str] = &["SELECT * FROM items WHERE ", " ORDER BY created_at DESC LIMIT ", ""];
/// ${cookies.team} | ${since} | ${created_at}
pub const T721: &[&str] = &["[TRADING] ✅ GET team='", "' since=", " created_at<", " "];
/// ${out.length} | ${JSON.stringify(brief)}
pub const T722: &[&str] = &["→ ", "건 ", ""];
/// ${list.length} | ${MAX_BATCH}
pub const T792: &[&str] = &["[TRADING] ⚠️ POST ", "건 → 상한 ", "건으로 절단합니다. (잔여분은 다음 회차)"];
/// ${ph}
pub const T811: &[&str] = &["SELECT \"id\", \"digest\", \"ref\", \"updated_at\" FROM items WHERE \"to\" = ?1 AND \"id\" IN (", ")"];
/// ${id} | ${docType}
pub const T835: &[&str] = &["[TRADING] ⛔ id='", "' type='", "' 은 무역 서식이 아니라 거부합니다."];
pub const T904: &[&str] = &["\n                                INSERT INTO items (\n                                    \"id\", \"type\", \"flag\", \"from\", \"to\", \"cc\", \"bcc\", \"ref\",\n                                    \"mode\", \"digest\", \"data\", \"created_at\", \"updated_at\"\n                                ) VALUES (\n                                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13\n                                ) ON CONFLICT (id) DO UPDATE SET\n                                    \"type\"       = EXCLUDED.\"type\",\n                                    \"flag\"       = EXCLUDED.\"flag\",\n                                    \"from\"       = EXCLUDED.\"from\",\n                                    \"to\"         = EXCLUDED.\"to\",\n                                    \"cc\"         = EXCLUDED.\"cc\",\n                                    \"bcc\"        = EXCLUDED.\"bcc\",\n                                    \"ref\"        = EXCLUDED.\"ref\",\n                                    \"mode\"       = EXCLUDED.\"mode\",\n                                    \"digest\"     = EXCLUDED.\"digest\",\n                                    \"data\"       = EXCLUDED.\"data\",\n                                    \"created_at\" = EXCLUDED.\"created_at\",\n                                    \"updated_at\" = EXCLUDED.\"updated_at\"\n                            "];
/// ${cookies.team}
pub const T967: &[&str] = &["[TRADING] ✅ POST team='", "' "];
/// ${out.length} | ${skipped} | ${rejected}
pub const T968: &[&str] = &["저장 ", "건 / 스킵 ", "건 / 거부 ", "건 "];
/// ${Object.keys(hubBrief).length}
pub const T969: &[&str] = &["/ 거래 건 ", "개"];
pub const T1049: &[&str] = &["DELETE FROM items WHERE \"id\" = ?1 AND \"to\" = ?2"];
/// ${cookies.team} | ${delIds.length}
pub const T1061: &[&str] = &["[TRADING] 🗑️ DELETE team='", "' → ", "건"];
pub const T1075: &[&str] = &["I'm a teapot!"];
