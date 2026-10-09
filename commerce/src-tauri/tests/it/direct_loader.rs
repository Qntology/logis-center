//! SSD 오프로딩 I/O 검증 (utils::direct_loader)
//!
//! Windows 는 DirectStorage, Linux 는 io_uring 경로를 탑니다.
//! 리눅스에서는 링이 실제로 올라오는지(조용한 fs 폴백이 아닌지)와
//! KV 블록 크기대 데이터의 무결성을 확인합니다.

use crate::common::{pattern, synthetic_pdfs, TempDir};
use std::fs;
use tauri_app_lib::utils::direct_loader::{io_backend, load_kv_block, save_kv_block};

#[cfg(target_os = "linux")]
#[test]
fn linux_uses_io_uring_backend_when_kernel_allows_it() {
    // 커널/컨테이너(seccomp)가 io_uring 을 허용하는지 직접 프로브한 결과와
    // direct_loader 가 선택한 경로가 일치해야 한다 (조용한 fs 폴백 감지).
    let kernel_allows = io_uring::IoUring::new(8).is_ok();
    let expected = if kernel_allows { "io_uring" } else { "fs-fallback" };
    eprintln!("io_uring probe: {kernel_allows}, direct_loader backend: {}", io_backend());
    assert_eq!(io_backend(), expected);
    // 이 저장소의 리눅스 CI/클라우드 세션처럼 io_uring 이 반드시 있어야 하는 환경에서는
    // LOGIS_REQUIRE_IO_URING=1 로 실행해 폴백 자체를 실패로 처리한다.
    if std::env::var("LOGIS_REQUIRE_IO_URING").as_deref() == Ok("1") {
        assert_eq!(io_backend(), "io_uring", "io_uring required but unavailable");
    }
}

#[test]
fn backend_name_is_known() {
    assert!(["directstorage", "io_uring", "metal-io", "fs-fallback"].contains(&io_backend()));
}

#[test]
fn roundtrip_small_and_empty_blocks() {
    let d = TempDir::new("dl_small");
    for n in [0usize, 1, 7, 511, 4096, 4097, 65_537] {
        let p = d.path().join(format!("b{n}.bin"));
        let data = pattern(n, n as u8);
        save_kv_block(&p, &data).unwrap();
        assert_eq!(fs::read(&p).unwrap(), data, "on-disk mismatch n={n}");
        assert_eq!(load_kv_block(&p).unwrap(), data, "load mismatch n={n}");
    }
}

#[test]
fn roundtrip_kv_sized_block() {
    // 실제 KV 오프로딩 블록 크기대 (수십 MB, 4K 정렬이 아닌 길이)
    let d = TempDir::new("dl_large");
    let p = d.path().join("l0.st");
    let data = pattern(48 * 1024 * 1024 + 123, 0x5a);
    save_kv_block(&p, &data).unwrap();
    assert_eq!(fs::metadata(&p).unwrap().len() as usize, data.len());
    let back = load_kv_block(&p).unwrap();
    assert_eq!(back.len(), data.len());
    assert!(back == data, "content mismatch after io_uring roundtrip");
}

#[test]
fn save_overwrites_and_truncates_previous_content() {
    let d = TempDir::new("dl_overwrite");
    let p = d.path().join("layer0_meta.json");
    save_kv_block(&p, &pattern(10_000, 1)).unwrap();
    save_kv_block(&p, br#"{"a":1}"#).unwrap();
    assert_eq!(load_kv_block(&p).unwrap(), br#"{"a":1}"#);
}

#[test]
fn missing_file_is_an_error_not_empty_data() {
    let d = TempDir::new("dl_missing");
    assert!(load_kv_block(&d.path().join("nope.bin")).is_err());
    assert!(save_kv_block(&d.path().join("no_dir/x.bin"), b"x").is_err());
}

#[test]
fn concurrent_threads_share_the_ring_safely() {
    let d = TempDir::new("dl_concurrent");
    let root = d.path().to_path_buf();
    let handles: Vec<_> = (0..8u8)
        .map(|t| {
            let root = root.clone();
            std::thread::spawn(move || {
                for i in 0..25usize {
                    let p = root.join(format!("t{t}_{i}.bin"));
                    let data = pattern(1000 + t as usize * 97 + i * 13, t);
                    save_kv_block(&p, &data).unwrap();
                    assert_eq!(load_kv_block(&p).unwrap(), data, "thread {t} iter {i}");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn synthetic_pdfs_load_identically_to_std_read() {
    let pdfs = synthetic_pdfs();
    assert!(!pdfs.is_empty(), "SyntheticData/trading/output_pdfs is missing");
    for p in pdfs {
        assert_eq!(load_kv_block(&p).unwrap(), fs::read(&p).unwrap(), "{p:?}");
    }
}
