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
    let expected = if std::env::var("LOGIS_DIRECT_LOADER").as_deref() == Ok("parallel") {
        "parallel-pread"
    } else if kernel_allows {
        "io_uring"
    } else {
        "fs-fallback"
    };
    eprintln!("io_uring probe: {kernel_allows}, direct_loader backend: {}", io_backend());
    assert_eq!(io_backend(), expected);
    // 이 저장소의 리눅스 CI/클라우드 세션처럼 io_uring 이 반드시 있어야 하는 환경에서는
    // LOGIS_REQUIRE_IO_URING=1 로 실행해 폴백 자체를 실패로 처리한다.
    if std::env::var("LOGIS_REQUIRE_IO_URING").as_deref() == Ok("1") {
        assert_eq!(io_backend(), "io_uring", "io_uring required but unavailable");
    }
}

#[cfg(target_os = "macos")]
#[test]
fn macos_uses_parallel_nocache_backend() {
    // macOS 기본은 병렬 pread/pwrite + F_NOCACHE (예전 MTLIO 경로는 컴파일되지 않았음).
    let expected = if std::env::var("LOGIS_MAC_IO").map(|v| v.eq_ignore_ascii_case("fs")).unwrap_or(false) {
        "fs-fallback"
    } else {
        "parallel-pread-nocache"
    };
    assert_eq!(io_backend(), expected);
}

#[test]
fn backend_name_is_known() {
    assert!(["directstorage", "io_uring", "parallel-pread", "parallel-pread-nocache", "fs-fallback"].contains(&io_backend()));
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

// ---------------------------------------------------------------------
//  [MAC-SSD-IO] 병렬 pread/pwrite 경로 (macOS 기본 · 리눅스 LOGIS_DIRECT_LOADER=parallel)
// ---------------------------------------------------------------------
#[cfg(unix)]
#[test]
fn parallel_io_roundtrips_odd_sizes_chunk_edges_and_truncation() {
    use tauri_app_lib::utils::direct_loader::parallel_io::{read, write, ParallelIo};
    let d = TempDir::new("dl_parallel");
    let chunk = 1 << 16;
    for nocache in [true, false] {
        // small = 0: 작은 블록도 병렬 경로로 보냅니다(경계 검사).
        let o = ParallelIo { threads: 3, chunk, nocache, small: 0 };
        for n in [0usize, 1, chunk - 1, chunk, chunk + 1, 3 * chunk, (1 << 20) + 7, 5 * (1 << 20) + 3] {
            let p = d.path().join(format!("p{n}_{nocache}.bin"));
            let data = pattern(n, (n % 251) as u8);
            write(&p, &data, &o).unwrap();
            assert_eq!(fs::metadata(&p).unwrap().len() as usize, n, "size n={n}");
            assert!(fs::read(&p).unwrap() == data, "on-disk mismatch n={n} nocache={nocache}");
            assert!(read(&p, &o).unwrap() == data, "parallel read mismatch n={n} nocache={nocache}");
        }
        // 더 긴 내용 위에 짧은 내용을 쓰면 잘려야 합니다.
        let p = d.path().join(format!("trunc_{nocache}.bin"));
        write(&p, &pattern(4 * chunk + 9, 1), &o).unwrap();
        let short = pattern(chunk + 5, 2);
        write(&p, &short, &o).unwrap();
        assert!(read(&p, &o).unwrap() == short, "truncate nocache={nocache}");
    }
    assert!(read(&d.path().join("nope.bin"), &ParallelIo { threads: 2, chunk, nocache: true, small: 0 }).is_err());
}

/// 시스템 파일 캐시 크기(MB): macOS `vm_stat` File-backed pages · 리눅스 /proc/meminfo Cached.
#[cfg(unix)]
fn file_cache_mb() -> Option<f64> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("vm_stat").output().ok()?;
        let s = String::from_utf8_lossy(&out.stdout);
        let page: f64 = s
            .lines()
            .next()
            .and_then(|l| l.split("page size of").nth(1))
            .and_then(|r| r.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .unwrap_or(16384.0);
        let pages: f64 = s
            .lines()
            .find(|l| l.starts_with("File-backed pages"))?
            .split(':')
            .nth(1)?
            .trim()
            .trim_end_matches('.')
            .parse()
            .ok()?;
        return Some(pages * page / 1e6);
    }
    #[cfg(target_os = "linux")]
    {
        let s = fs::read_to_string("/proc/meminfo").ok()?;
        let kb: f64 = s.lines().find(|l| l.starts_with("Cached:"))?.split_whitespace().nth(1)?.parse().ok()?;
        return Some(kb / 1e3);
    }
    #[allow(unreachable_code)]
    None
}

/// 측정 전용: 차가운(캐시에 없는) 블록 읽기/쓰기 속도와 파일 캐시 증가량.
/// `cargo test --test it -- direct_loader::kv_block_io_bench --ignored --nocapture` (LOGIS_IO_BENCH_MB, 기본 256)
#[cfg(unix)]
#[test]
#[ignore = "측정 전용 (디스크 수백 MB 사용)"]
fn kv_block_io_bench() {
    use std::time::Instant;
    use tauri_app_lib::utils::direct_loader::parallel_io::{read, write, ParallelIo};
    let mb: usize = std::env::var("LOGIS_IO_BENCH_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let size = mb << 20;
    let d = TempDir::new("dl_bench");
    let data = pattern(size, 0x3c);
    let cold = ParallelIo { threads: 4, chunk: 8 << 20, nocache: true, small: 1 << 20 };
    let rate = |secs: f64| size as f64 / 1e6 / secs;
    println!("[KV-IO] {} · block {} MiB · default backend {}", std::env::consts::OS, mb, io_backend());

    // ── 쓰기: std::fs::write vs 병렬 pwrite(nocache) ──
    let p = d.path().join("w_fs.bin");
    let t = Instant::now();
    fs::write(&p, &data).unwrap();
    let fs_w = t.elapsed().as_secs_f64();
    fs::remove_file(&p).unwrap();
    let p = d.path().join("w_par.bin");
    let t = Instant::now();
    write(&p, &data, &cold).unwrap();
    let par_w = t.elapsed().as_secs_f64();
    fs::remove_file(&p).unwrap();
    println!("[KV-IO] write: std::fs {:.0} MB/s · parallel 4×8MiB nocache {:.0} MB/s ({:.2}×)", rate(fs_w), rate(par_w), fs_w / par_w);

    // ── 차가운 읽기: 파일마다 nocache 로 새로 써서 캐시에 없는 상태에서 한 번 읽습니다 ──
    let cold_read = |label: &str, f: &dyn Fn(&std::path::Path) -> Vec<u8>| -> (f64, f64) {
        let p = d.path().join("r.bin");
        write(&p, &data, &cold).unwrap();
        let c0 = file_cache_mb().unwrap_or(f64::NAN);
        let t = Instant::now();
        let back = f(&p);
        let secs = t.elapsed().as_secs_f64();
        let c1 = file_cache_mb().unwrap_or(f64::NAN);
        assert!(back == data, "{label}: content mismatch");
        fs::remove_file(&p).unwrap();
        (rate(secs), c1 - c0)
    };
    let (fs_r, fs_cache) = cold_read("std::fs", &|p| fs::read(p).unwrap());
    println!("[KV-IO] cold read std::fs::read: {:.0} MB/s · file cache +{:.0} MB", fs_r, fs_cache);
    let mut best = (0f64, 0usize, 0usize);
    for threads in [1usize, 2, 4, 8] {
        for chunk_mb in [2usize, 8, 32] {
            let o = ParallelIo { threads, chunk: chunk_mb << 20, nocache: true, small: 1 << 20 };
            let (r, cache) = cold_read("parallel", &|p| read(p, &o).unwrap());
            println!("[KV-IO] cold read parallel {threads}×{chunk_mb}MiB nocache: {r:.0} MB/s ({:.2}× std::fs) · file cache +{cache:.0} MB", r / fs_r);
            if r > best.0 {
                best = (r, threads, chunk_mb);
            }
        }
    }
    let o = ParallelIo { threads: best.1, chunk: best.2 << 20, nocache: false, small: 1 << 20 };
    let (r, cache) = cold_read("parallel cached", &|p| read(p, &o).unwrap());
    println!("[KV-IO] cold read parallel {}×{}MiB cached: {r:.0} MB/s · file cache +{cache:.0} MB", best.1, best.2);
    println!(
        "[KV-IO] best: parallel {}×{}MiB nocache {:.0} MB/s = {:.2}× std::fs::read ({:.0} MB/s) · default 4×8MiB",
        best.1, best.2, best.0, best.0 / fs_r, fs_r
    );
}
