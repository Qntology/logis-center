//! 🌟 [SDS TEAM HOLD] 로그인한 사용자의 원장이 앱을 다시 켜도 남는지 실제 파일로 확인합니다.
//!
//! 앱은 부팅 때 언제나 로컬 기본 팀(ZERO 주소)으로 `load` 하고, 로그인이 끝나면 `rebind_team` 으로 실제 팀을
//! 붙입니다. 예전에는 부팅 `load` 가 실제 팀의 원장을 '팀 불일치' 로 버린 뒤 곧바로 빈 원장으로 덮어써서,
//! 로그인한 사용자의 누적 통계가 다시 켤 때마다 사라졌습니다.
//!
//! SDS 는 프로세스 전역 상태라 다른 시험과 섞이지 않게 혼자 돌립니다.
//!
//! ```text
//! LOGIS_APP_DIR=<빈 임시 폴더> cargo test --test it --no-default-features --features rocm -- sds_ledger --ignored --test-threads 1
//! ```

use serde_json::Value;
use std::path::{Path, PathBuf};
use tauri_app_lib::score_dynamics as sds;
use tauri_app_lib::utils::hash::hash_id;

const ZERO_ADDR: &str = "0x0000000000000000000000000000000000000000";
const AXIS: &str = "search.recall_n";
const SCOPE: &str = "search|all|";

/// 원장 폴더(임베딩 공간에 따라 score_dynamics 또는 그 아래 spaces/<공간>)에서 이름이 같은 원장 파일을 찾습니다.
fn find_ledger(root: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).ok()?.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.file_name().map_or(false, |n| n != "runs" && n != "legacy") {
                    stack.push(p);
                }
            } else if p.file_name().map_or(false, |n| n == name) {
                return Some(p);
            }
        }
    }
    None
}

fn read(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("원장 읽기")).expect("원장 JSON")
}

/// (팀, 축 관측 수)
fn ledger_state(path: &Path) -> (String, u64) {
    let v = read(path);
    let n = v["scopes"][SCOPE]["baseline"][AXIS]["n"].as_u64().unwrap_or(0);
    (v["team"].as_str().unwrap_or("").to_string(), n)
}

fn observe(team: &str, values: &[f32]) {
    sds::enter_scope(team, sds::Track::Search, "all", "");
    for v in values {
        sds::record_baseline(AXIS, *v);
    }
    sds::leave_scope();
    println!("{}", sds::report());
    sds::flush();
}

#[test]
#[ignore = "전역 SDS 상태를 바꿉니다: LOGIS_APP_DIR 을 빈 임시 폴더로 주고 혼자 돌립니다"]
fn team_ledger_survives_reboot() {
    let Some(app_dir) = std::env::var_os("LOGIS_APP_DIR").filter(|v| !v.is_empty()).map(PathBuf::from) else {
        println!("LOGIS_APP_DIR 이 없어 건너뜁니다.");
        return;
    };
    let real_dir = dirs::data_local_dir().map(|d| d.join("logis-center"));
    assert_ne!(Some(app_dir.clone()), real_dir, "사용자의 앱 데이터 폴더에서는 돌리지 않습니다");
    let sds_dir = app_dir.join("score_dynamics");
    let _ = std::fs::remove_dir_all(&sds_dir);

    let local = hash_id(ZERO_ADDR);
    let real = hash_id("0x00000000000000000000000000000000000000a1");
    let other = hash_id("0x00000000000000000000000000000000000000b2");

    // 1) 로그인한 세션: 실제 팀으로 관측 3건과 보유 문서 1건을 저장합니다.
    sds::load(&local);
    sds::rebind_team(&real);
    observe(&real, &[10.0, 20.0, 30.0]);
    sds::presence_record("order", "doc-1", &["amount".to_string()]);
    sds::flush();
    let ledger = find_ledger(&sds_dir, "search.json").expect("search 원장");
    let presence = sds_dir.join("presence.json");
    assert_eq!(ledger_state(&ledger), (real.clone(), 3));
    assert_eq!(read(&presence)["docs"].as_object().map(|d| d.len()), Some(1));

    // 2) 앱을 다시 켬: 부팅은 로컬 기본 팀으로 load. 실제 팀의 파일을 덮어쓰지 않아야 합니다.
    sds::load(&local);
    assert_eq!(ledger_state(&ledger), (real.clone(), 3), "부팅 load 가 실제 팀 원장을 덮어썼습니다");
    assert_eq!(read(&presence)["team"].as_str(), Some(real.as_str()), "부팅 load 가 보유 현황을 덮어썼습니다");
    assert_eq!(read(&presence)["docs"].as_object().map(|d| d.len()), Some(1));

    // 3) 로그인 전 관측 1건 · 보유 문서 1건: 보류 중이라 실제 팀 파일은 그대로입니다.
    observe(&local, &[40.0]);
    sds::presence_record("order", "doc-2", &["amount".to_string()]);
    sds::flush();
    assert_eq!(ledger_state(&ledger), (real.clone(), 3), "보류 중에 실제 팀 원장을 덮어썼습니다");
    assert_eq!(read(&presence)["docs"].as_object().map(|d| d.len()), Some(1));

    // 4) 같은 팀으로 로그인: 보류한 원장을 되살리고 로그인 전 관측을 합칩니다.
    sds::rebind_team(&real);
    assert_eq!(ledger_state(&ledger), (real.clone(), 4), "로그인 뒤 원장이 되살아나지 않았습니다");
    assert_eq!(read(&presence)["docs"].as_object().map(|d| d.len()), Some(2));

    // 5) 다시 켜고 다른 팀으로 로그인: 보류한 원장은 버리고 로그인 전 관측만 그 팀으로 이관합니다.
    sds::load(&local);
    observe(&local, &[50.0]);
    sds::rebind_team(&other);
    assert_eq!(ledger_state(&ledger), (other.clone(), 1), "다른 팀에 이전 팀의 통계가 섞였습니다");
    assert_eq!(read(&presence)["docs"].as_object().map(|d| d.len()), Some(0));

    let _ = std::fs::remove_dir_all(&sds_dir);
}
