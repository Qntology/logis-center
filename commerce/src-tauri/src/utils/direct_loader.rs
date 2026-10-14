use anyhow::{Result, anyhow};
use std::path::Path;
use std::fs;
use once_cell::sync::Lazy;
use std::sync::Arc;

// --- Windows Implementation --- 
#[cfg(windows)]
mod windows_impl {
    use super::*;
    use direct_storage::*;
    use windows::core::HSTRING;
    use windows::Win32::Storage::FileSystem::{CreateFileW, WriteFile, FILE_SHARE_WRITE, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, FILE_FLAG_OVERLAPPED};
    use windows::Win32::Foundation::{HANDLE, CloseHandle, GENERIC_WRITE};
    use windows::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};
    use std::mem::ManuallyDrop;

    struct WinContext {
        factory: IDStorageFactory,
        queue: IDStorageQueue,
        status_array: IDStorageStatusArray,
    }

    unsafe impl Send for WinContext {}
    unsafe impl Sync for WinContext {}

    pub fn is_available() -> bool { CONTEXT.is_ok() }

    static CONTEXT: Lazy<Result<Arc<WinContext>>> = Lazy::new(|| {
        unsafe {
            let factory: IDStorageFactory = DStorageGetFactory().map_err(|e| {
                println!("[DirectLoader] DirectStorage factory init failed: {:?}. Falling back to fs::read/fs::write.", e);
                e
            })?;
            let queue_desc = DSTORAGE_QUEUE_DESC {
                SourceType: DSTORAGE_REQUEST_SOURCE_FILE,
                Capacity: DSTORAGE_MAX_QUEUE_CAPACITY as u16,
                Priority: DSTORAGE_PRIORITY_NORMAL,
                Name: windows::core::PCSTR::null(),
                Device: ManuallyDrop::new(None),
            };
            let queue = factory.CreateQueue(&queue_desc).map_err(|e| {
                println!("[DirectLoader] DirectStorage queue creation failed: {:?}. Falling back to fs::read/fs::write.", e);
                e
            })?;
            let status_array = factory.CreateStatusArray(1, None).map_err(|e| {
                println!("[DirectLoader] DirectStorage status array creation failed: {:?}. Falling back to fs::read/fs::write.", e);
                e
            })?;
            println!("[DirectLoader] DirectStorage context initialized successfully.");
            Ok(Arc::new(WinContext { factory, queue, status_array }))
        }
    });

    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        if let Ok(ctx) = CONTEXT.as_ref() {
            unsafe {
                if let Ok(metadata) = fs::metadata(path) {
                    let size = metadata.len() as usize;
                    let path_str = path.to_string_lossy().to_string();
                    if let Ok(file) = ctx.factory.OpenFile(&HSTRING::from(path_str)) {
                        let mut buffer = vec![0u8; size];
                        let mut request = DSTORAGE_REQUEST::default();
                        request.Options.set_SourceType(DSTORAGE_REQUEST_SOURCE_FILE);
                        request.Options.set_DestinationType(DSTORAGE_REQUEST_DESTINATION_MEMORY);
                        request.Source.File = ManuallyDrop::new(DSTORAGE_SOURCE_FILE {
                            Source: ManuallyDrop::new(Some(file)),
                            Offset: 0,
                            Size: size as u32,
                        });
                        request.Destination.Memory = DSTORAGE_DESTINATION_MEMORY {
                            Buffer: buffer.as_mut_ptr() as *mut _,
                            Size: size as u32,
                        };
                        ctx.queue.EnqueueRequest(&request);
                        ctx.queue.EnqueueStatus(&ctx.status_array, 0);
                        ctx.queue.Submit();
                        let start = std::time::Instant::now();
                        let timeout = std::time::Duration::from_secs(30);
                        while !ctx.status_array.IsComplete(0) {
                            if start.elapsed() > timeout {
                                println!("[DirectLoader] DirectStorage read timed out for {:?}. Falling back to fs::read.", path);
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_micros(100));
                        }
                        if ctx.status_array.IsComplete(0) && ctx.status_array.GetHResult(0).is_ok() {
                            return Ok(buffer);
                        }
                        if ctx.status_array.IsComplete(0) {
                            println!("[DirectLoader] DirectStorage read returned error HRESULT for {:?}. Falling back to fs::read.", path);
                        }
                    }
                }
            }
        }
        fs::read(path).map_err(|e| anyhow::anyhow!("Fallback read failed: {}", e))
    }

    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        unsafe {
            let path_wide = HSTRING::from(path.to_string_lossy().as_ref());
            let handle_res = CreateFileW(
                &path_wide,
                GENERIC_WRITE.0,
                FILE_SHARE_WRITE,
                None,
                CREATE_ALWAYS,
                FILE_FLAG_OVERLAPPED | FILE_ATTRIBUTE_NORMAL,
                Some(HANDLE::default()),
            );
            if let Ok(handle) = handle_res {
                if !handle.is_invalid() {
                    let mut overlapped = OVERLAPPED::default();
                    let mut bytes_written = 0u32;
                    let write_result = WriteFile(handle, Some(data), Some(&mut bytes_written), Some(&mut overlapped));
                    let mut transferred = 0u32;
                    let result = GetOverlappedResult(handle, &overlapped, &mut transferred, true);
                    let _ = CloseHandle(handle);
                    if result.is_ok() {
                        if transferred as usize == data.len() {
                            return Ok(());
                        }
                        println!("[DirectLoader] Overlapped write incomplete for {:?}: wrote {} of {} bytes. Falling back to fs::write.", path, transferred, data.len());
                    } else if write_result.is_err() {
                        println!("[DirectLoader] Overlapped WriteFile failed for {:?}. Falling back to fs::write.", path);
                    }
                }
            }
        }
        fs::write(path, data).map_err(|e| anyhow::anyhow!("Fallback write failed: {}", e))
    }
}

// --- Linux Implementation ---
#[cfg(target_os = "linux")]
mod linux_impl {
    use super::*;
    use io_uring::{opcode, types, IoUring};
    use std::fs::File;
    use std::os::unix::io::AsRawFd;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Mutex;

    /// 한 번의 SQE 로 요청할 최대 바이트 수.
    /// 리눅스 read/write 는 한 번에 최대 0x7ffff000 바이트만 처리하고 opcode 길이는 u32 이므로
    /// 1 GiB 단위로 쪼개 오프셋 루프를 돈다 (부분 완료도 같은 루프로 이어서 처리).
    const MAX_CHUNK: usize = 1 << 30;

    struct LinuxContext { ring: Mutex<IoUring> }
    unsafe impl Send for LinuxContext {}
    unsafe impl Sync for LinuxContext {}

    static CONTEXT: Lazy<Result<Arc<LinuxContext>>> = Lazy::new(|| {
        let ring = IoUring::new(128).map_err(|e| {
            println!("[DirectLoader] io_uring init failed: {:?}. Falling back to fs::read/fs::write.", e);
            anyhow!(e)
        })?;
        println!("[DirectLoader] io_uring context initialized successfully (128 entries).");
        Ok(Arc::new(LinuxContext { ring: Mutex::new(ring) }))
    });

    /// 복구 불가능한 링 오류 이후에는 링을 다시 쓰지 않는다 (남은 SQE/CQE 오염 방지).
    static RING_BROKEN: AtomicBool = AtomicBool::new(false);
    /// 요청마다 고유한 user_data — 이전 호출의 잔여 CQE 를 내 것으로 오인하지 않기 위함.
    static NEXT_TAG: AtomicU64 = AtomicU64::new(1);

    /// io_uring 링을 사용할 수 있는지 (커널/seccomp 차단 또는 링 고장 시 false → fs 폴백)
    pub fn is_available() -> bool {
        CONTEXT.is_ok() && !RING_BROKEN.load(Ordering::Acquire)
    }

    enum OpError {
        /// 커널이 버퍼를 더 이상 참조하지 않음 → 일반 오류로 처리해도 안전
        Done(anyhow::Error),
        /// 요청이 아직 in-flight 일 수 있음 → 버퍼를 해제하면 안 됨
        InFlight(anyhow::Error),
    }

    /// SQE 하나를 제출하고 대응하는 CQE 의 res 를 돌려준다.
    /// EINTR/EBUSY/EAGAIN 은 재시도하며, 요청이 끝나기 전에는 Ok 로 반환하지 않는다.
    fn submit_one(ring: &mut IoUring, entry: io_uring::squeue::Entry) -> std::result::Result<i32, OpError> {
        let tag = NEXT_TAG.fetch_add(1, Ordering::Relaxed);
        let entry = entry.user_data(tag);
        // push 실패(SQ full)는 아직 커널에 넘어가지 않았으므로 안전한 실패
        if unsafe { ring.submission().push(&entry) }.is_err() {
            return Err(OpError::Done(anyhow!("io_uring submission queue full")));
        }
        let mut transient_retries = 0u32;
        loop {
            // 이미 도착한 완료부터 확인 (다른 tag 의 잔여 CQE 는 버린다)
            let done = {
                let mut cq = ring.completion();
                let mut r = None;
                while let Some(cqe) = cq.next() {
                    if cqe.user_data() == tag {
                        r = Some(cqe.result());
                        break;
                    }
                }
                r
            };
            if let Some(res) = done {
                return Ok(res);
            }
            match ring.submit_and_wait(1) {
                Ok(_) => {}
                // 시그널 인터럽트: 제출은 이미 끝났을 수 있으므로 그대로 다시 기다린다
                Err(e) if e.raw_os_error() == Some(libc::EINTR) => {}
                // CQ overflow / 일시적 자원 부족: CQ 를 비운 뒤 재시도
                Err(e) if matches!(e.raw_os_error(), Some(libc::EBUSY) | Some(libc::EAGAIN)) => {
                    transient_retries += 1;
                    if transient_retries > 10_000 {
                        return Err(OpError::InFlight(anyhow!("io_uring wait kept failing: {}", e)));
                    }
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
                Err(e) => return Err(OpError::InFlight(anyhow!("io_uring wait failed: {}", e))),
            }
        }
    }

    fn uring_read(ctx: &LinuxContext, file: &File, path: &Path) -> Result<Vec<u8>> {
        let size = file.metadata()?.len() as usize;
        let mut buffer = vec![0u8; size];
        let fd = types::Fd(file.as_raw_fd());
        let mut ring = ctx.ring.lock().unwrap_or_else(|p| p.into_inner());
        let mut off = 0usize;
        while off < size {
            let len = (size - off).min(MAX_CHUNK) as u32;
            let e = opcode::Read::new(fd, unsafe { buffer.as_mut_ptr().add(off) }, len)
                .offset(off as u64)
                .build();
            match submit_one(&mut ring, e) {
                Ok(res) if res == -libc::EINTR || res == -libc::EAGAIN => continue,
                Ok(res) if res < 0 => {
                    return Err(anyhow!(
                        "io_uring read failed for {:?}: {}",
                        path,
                        std::io::Error::from_raw_os_error(-res)
                    ))
                }
                Ok(0) => {
                    // 읽는 도중 파일이 줄어든 경우 (정상적인 EOF)
                    println!("[DirectLoader] io_uring early EOF for {:?}: got {} of {} bytes.", path, off, size);
                    buffer.truncate(off);
                    break;
                }
                Ok(res) => off += res as usize,
                Err(OpError::Done(e)) => return Err(e),
                Err(OpError::InFlight(e)) => {
                    RING_BROKEN.store(true, Ordering::Release);
                    // 커널이 아직 이 버퍼에 쓸 수 있으므로 해제하지 않는다 (use-after-free 방지)
                    std::mem::forget(buffer);
                    return Err(e);
                }
            }
        }
        Ok(buffer)
    }

    fn uring_write(ctx: &LinuxContext, file: &File, path: &Path, data: &[u8]) -> Result<()> {
        let fd = types::Fd(file.as_raw_fd());
        let mut ring = ctx.ring.lock().unwrap_or_else(|p| p.into_inner());
        let mut off = 0usize;
        while off < data.len() {
            let len = (data.len() - off).min(MAX_CHUNK) as u32;
            let e = opcode::Write::new(fd, unsafe { data.as_ptr().add(off) }, len)
                .offset(off as u64)
                .build();
            match submit_one(&mut ring, e) {
                Ok(res) if res == -libc::EINTR || res == -libc::EAGAIN => continue,
                Ok(res) if res <= 0 => {
                    return Err(anyhow!(
                        "io_uring write failed for {:?} at offset {}: {}",
                        path,
                        off,
                        std::io::Error::from_raw_os_error(-res)
                    ))
                }
                Ok(res) => off += res as usize,
                Err(OpError::Done(e)) => return Err(e),
                Err(OpError::InFlight(e)) => {
                    RING_BROKEN.store(true, Ordering::Release);
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// `LOGIS_DIRECT_LOADER=parallel` 이면 io_uring 대신 macOS 와 같은 병렬 pread/pwrite 경로를 씁니다
    /// (macOS 경로의 리눅스 검증 · 비교용).
    pub fn parallel_forced() -> bool {
        static ON: Lazy<bool> = Lazy::new(|| {
            std::env::var("LOGIS_DIRECT_LOADER").map(|v| v.trim().eq_ignore_ascii_case("parallel")).unwrap_or(false)
        });
        *ON
    }

    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        if parallel_forced() {
            return super::parallel_io::read(path, &super::parallel_io::ParallelIo::from_env());
        }
        // 열기 실패(파일 없음 등)는 폴백해도 똑같이 실패하므로 바로 반환
        let file = File::open(path).map_err(|e| anyhow!("open failed for {:?}: {}", path, e))?;
        if let (Ok(ctx), true) = (CONTEXT.as_ref(), is_available()) {
            match uring_read(ctx, &file, path) {
                Ok(buf) => return Ok(buf),
                Err(e) => println!("[DirectLoader] {}. Falling back to fs::read.", e),
            }
        }
        fs::read(path).map_err(|e| anyhow!("Fallback read failed: {}", e))
    }

    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        if parallel_forced() {
            return super::parallel_io::write(path, data, &super::parallel_io::ParallelIo::from_env());
        }
        let file = File::create(path).map_err(|e| anyhow!("create failed for {:?}: {}", path, e))?;
        if let (Ok(ctx), true) = (CONTEXT.as_ref(), is_available()) {
            match uring_write(ctx, &file, path, data) {
                Ok(()) => return Ok(()),
                Err(e) => println!("[DirectLoader] {}. Falling back to fs::write.", e),
            }
        }
        drop(file);
        fs::write(path, data).map_err(|e| anyhow!("Fallback write failed: {}", e))
    }
}

// --- macOS Implementation ---
// =====================================================================
// 🌟 [MAC-SSD-IO] 병렬 pread/pwrite + 캐시 우회 (macOS 기본 · 리눅스는 LOGIS_DIRECT_LOADER=parallel)
// ---------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   예전 macOS 구현은 metal 크레이트(0.33)에 없는 MTLIO API(IOCommandQueue · new_io_command_queue)를
//   불러 macOS 빌드가 컴파일되지 않았습니다(macos-15 러너 첫 빌드). MTLIOCommandBuffer 는 MTLBuffer 로만
//   읽어 들이므로 호스트 Vec 를 돌려주는 이 함수 계약과도 맞지 않았습니다.
//  ── 해결: Apple Silicon 에 맞춘 두 가지 ──
//   ① 큐 깊이: Apple NVMe 는 요청이 여러 개 동시에 걸려 있어야 대역폭이 나옵니다. read() 한 번은 사실상
//      큐 깊이 1 이므로, 블록을 CHUNK(기본 8 MiB) 로 나눠 THREADS(기본 4) 개 스레드가 pread/pwrite 합니다
//      (리눅스 io_uring · Windows DirectStorage 가 하는 '여러 요청 동시 제출' 의 macOS 판).
//   ② 통합 메모리: 파일 캐시(UBC)는 GPU · RAM 상주 KV 와 같은 물리 메모리를 씁니다. KV 를 SSD 로 내린 이유가
//      메모리 확보인데 그 블록이 파일 캐시에 다시 남으면 이득이 사라집니다. 그래서 큰 블록은 F_NOCACHE
//      (리눅스 O_DIRECT 에 해당)로 캐시를 거치지 않고, 쓰기 전에는 F_PREALLOCATE 로 연속 공간을 잡습니다.
//   작은 블록(메타 JSON 등, SMALL 미만)은 캐시가 오히려 빠르므로 기존 fs 경로입니다.
//   읽기 버퍼는 0 으로 채우지 않고(48 MB 면 memset 만 수 ms) pread 가 직접 채운 뒤 길이를 확정합니다.
//  ── 조절 ──
//   LOGIS_MAC_IO=fs (끄기) · LOGIS_MAC_IO_THREADS · LOGIS_MAC_IO_CHUNK_MB · LOGIS_MAC_IO_NOCACHE=0
//   측정: tests/it/direct_loader.rs `kv_block_io_bench` (--ignored, docs/mac-verify.md)
// =====================================================================
#[cfg(unix)]
pub mod parallel_io {
    use super::*;
    use std::fs::{File, OpenOptions};
    use std::os::unix::io::AsRawFd;

    /// 병렬 블록 I/O 설정.
    #[derive(Debug, Clone, Copy)]
    pub struct ParallelIo {
        /// 동시에 pread/pwrite 하는 스레드 수 (= 대략의 큐 깊이).
        pub threads: usize,
        /// 요청 하나의 크기.
        pub chunk: usize,
        /// 페이지 캐시 우회 (macOS F_NOCACHE · 리눅스는 끝난 뒤 POSIX_FADV_DONTNEED).
        pub nocache: bool,
        /// 이보다 작은 블록은 단일 호출 + 캐시 경로.
        pub small: usize,
    }

    fn env_usize(key: &str) -> Option<usize> {
        std::env::var(key).ok().and_then(|v| v.trim().parse::<usize>().ok())
    }

    impl ParallelIo {
        /// 기본값 + 환경 변수 (프로세스에서 한 번 읽음).
        pub fn from_env() -> Self {
            static OPTS: Lazy<ParallelIo> = Lazy::new(|| {
                let o = ParallelIo {
                    threads: env_usize("LOGIS_MAC_IO_THREADS").filter(|v| *v >= 1).unwrap_or(4).min(64),
                    chunk: env_usize("LOGIS_MAC_IO_CHUNK_MB").filter(|v| *v >= 1).unwrap_or(8).min(1024) << 20,
                    nocache: std::env::var("LOGIS_MAC_IO_NOCACHE").map(|v| v.trim() != "0").unwrap_or(true),
                    small: 1 << 20,
                };
                println!(
                    "[DirectLoader] parallel block I/O: {} threads × {} MiB requests, nocache={} (≥ {} KiB)",
                    o.threads, o.chunk >> 20, o.nocache, o.small >> 10
                );
                o
            });
            *OPTS
        }
    }

    fn set_nocache(file: &File, on: bool) {
        if !on {
            return;
        }
        #[cfg(target_os = "macos")]
        unsafe {
            libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = file;
    }

    /// 리눅스에는 F_NOCACHE 가 없으므로 끝난 뒤 이 범위의 캐시를 내려놓게 합니다.
    fn drop_cache(file: &File, on: bool) {
        #[cfg(target_os = "linux")]
        if on {
            unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
            }
        }
        let _ = (file, on);
    }

    /// 쓰기 전에 연속 공간을 미리 잡습니다(macOS F_PREALLOCATE, 실패해도 그대로 진행).
    fn preallocate(file: &File, len: usize) {
        #[cfg(target_os = "macos")]
        unsafe {
            let mut st = libc::fstore_t {
                fst_flags: libc::F_ALLOCATECONTIG | libc::F_ALLOCATEALL,
                fst_posmode: libc::F_PEOFPOSMODE,
                fst_offset: 0,
                fst_length: len as libc::off_t,
                fst_bytesalloc: 0,
            };
            if libc::fcntl(file.as_raw_fd(), libc::F_PREALLOCATE, &mut st) == -1 {
                st.fst_flags = libc::F_ALLOCATEALL;
                libc::fcntl(file.as_raw_fd(), libc::F_PREALLOCATE, &mut st);
            }
        }
        let _ = (file, len);
    }

    /// [off, off+len) 를 끝까지 읽습니다(부분 읽기 · EINTR 이어 받기). 돌려준 값 < len 이면 EOF.
    unsafe fn pread_all(fd: i32, dst: *mut u8, len: usize, off: usize) -> std::io::Result<usize> {
        let mut done = 0usize;
        while done < len {
            let r = libc::pread(fd, dst.add(done) as *mut libc::c_void, len - done, (off + done) as libc::off_t);
            if r < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(e);
            }
            if r == 0 {
                break;
            }
            done += r as usize;
        }
        Ok(done)
    }

    fn pwrite_all(fd: i32, src: &[u8], off: usize) -> std::io::Result<()> {
        let mut done = 0usize;
        while done < src.len() {
            let r = unsafe {
                libc::pwrite(fd, src[done..].as_ptr() as *const libc::c_void, src.len() - done, (off + done) as libc::off_t)
            };
            if r < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(e);
            }
            if r == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "pwrite returned 0"));
            }
            done += r as usize;
        }
        Ok(())
    }

    /// 청크 번호를 스레드가 나눠 가지며 `job(chunk index)` 를 실행합니다. 첫 오류를 돌려줍니다.
    fn run_chunks(n_chunks: usize, threads: usize, job: &(dyn Fn(usize) -> std::io::Result<()> + Sync)) -> std::io::Result<()> {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let first_err: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::new(None);
        let worker = || loop {
            if failed.load(Ordering::Relaxed) {
                break;
            }
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_chunks {
                break;
            }
            if let Err(e) = job(i) {
                failed.store(true, Ordering::Relaxed);
                first_err.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert(e);
                break;
            }
        };
        let t = threads.min(n_chunks).max(1);
        if t == 1 {
            worker();
        } else {
            std::thread::scope(|s| {
                for _ in 1..t {
                    s.spawn(&worker);
                }
                worker();
            });
        }
        match first_err.into_inner().unwrap_or_else(|p| p.into_inner()) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// 파일 전체를 병렬로 읽습니다.
    pub fn read(path: &Path, o: &ParallelIo) -> Result<Vec<u8>> {
        let file = File::open(path).map_err(|e| anyhow!("open failed for {:?}: {}", path, e))?;
        let size = file.metadata()?.len() as usize;
        if size < o.small {
            let mut v = Vec::with_capacity(size);
            std::io::Read::read_to_end(&mut &file, &mut v)?;
            return Ok(v);
        }
        set_nocache(&file, o.nocache);
        let fd = file.as_raw_fd();
        let chunk = o.chunk.max(1 << 16);
        let n_chunks = (size + chunk - 1) / chunk;
        let mut buf: Vec<u8> = Vec::with_capacity(size);
        let base = buf.as_mut_ptr() as usize;
        // 청크마다 실제로 읽은 바이트 (파일이 도중에 줄면 앞쪽 연속 구간만 씁니다).
        let got: Vec<std::sync::atomic::AtomicUsize> = (0..n_chunks).map(|_| Default::default()).collect();
        run_chunks(n_chunks, o.threads, &|i| {
            let off = i * chunk;
            let len = chunk.min(size - off);
            // 청크 구간은 서로 겹치지 않고 모두 buf 용량 안입니다.
            let n = unsafe { pread_all(fd, (base as *mut u8).add(off), len, off)? };
            got[i].store(n, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        })
        .map_err(|e| anyhow!("parallel read failed for {:?}: {}", path, e))?;
        let mut valid = 0usize;
        for (i, g) in got.iter().enumerate() {
            let n = g.load(std::sync::atomic::Ordering::Relaxed);
            let len = chunk.min(size - i * chunk);
            valid += n;
            if n < len {
                println!("[DirectLoader] early EOF for {:?}: got {} of {} bytes.", path, valid, size);
                break;
            }
        }
        // 앞쪽 valid 바이트는 모두 pread 가 채웠습니다.
        unsafe { buf.set_len(valid) };
        drop_cache(&file, o.nocache);
        Ok(buf)
    }

    /// 파일을 data 로 바꿉니다(잘라내기 포함) — 병렬 pwrite.
    pub fn write(path: &Path, data: &[u8], o: &ParallelIo) -> Result<()> {
        if data.len() < o.small {
            return fs::write(path, data).map_err(|e| anyhow!("write failed for {:?}: {}", path, e));
        }
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| anyhow!("create failed for {:?}: {}", path, e))?;
        set_nocache(&file, o.nocache);
        preallocate(&file, data.len());
        let fd = file.as_raw_fd();
        let chunk = o.chunk.max(1 << 16);
        let n_chunks = (data.len() + chunk - 1) / chunk;
        run_chunks(n_chunks, o.threads, &|i| {
            let off = i * chunk;
            pwrite_all(fd, &data[off..(off + chunk).min(data.len())], off)
        })
        .map_err(|e| anyhow!("parallel write failed for {:?}: {}", path, e))?;
        // F_PREALLOCATE 는 크기를 바꾸지 않지만, 혹시 이전 내용이 더 길었던 경우를 위해 길이를 확정합니다.
        file.set_len(data.len() as u64)?;
        drop_cache(&file, o.nocache);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
mod macos_impl {
    use super::*;
    use super::parallel_io::{self, ParallelIo};
    /// `LOGIS_MAC_IO=fs` 이면 예전처럼 std::fs 만 씁니다.
    pub fn is_available() -> bool {
        static ON: Lazy<bool> = Lazy::new(|| {
            !std::env::var("LOGIS_MAC_IO").map(|v| v.trim().eq_ignore_ascii_case("fs")).unwrap_or(false)
        });
        *ON
    }
    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        if !is_available() {
            return fs::read(path).map_err(|e| anyhow!(e));
        }
        parallel_io::read(path, &ParallelIo::from_env())
    }
    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        if !is_available() {
            return fs::write(path, data).map_err(|e| anyhow!(e));
        }
        parallel_io::write(path, data, &ParallelIo::from_env())
    }
}

// --- Default/Fallback ---
#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
mod default_impl {
    use super::*;
    pub fn load_block(path: &Path) -> Result<Vec<u8>> { fs::read(path).map_err(|e| anyhow::anyhow!(e)) }
    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> { fs::write(path, data).map_err(|e| anyhow::anyhow!(e)) }
}

pub fn load_kv_block(path: &Path) -> Result<Vec<u8>> {
    #[cfg(windows)] { windows_impl::load_block(path) }
    #[cfg(target_os = "linux")] { linux_impl::load_block(path) }
    #[cfg(target_os = "macos")] { macos_impl::load_block(path) }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))] { default_impl::load_block(path) }
}

pub fn save_kv_block(path: &Path, data: &[u8]) -> Result<()> {
    #[cfg(windows)] { windows_impl::save_block(path, data) }
    #[cfg(target_os = "linux")] { linux_impl::save_block(path, data) }
    #[cfg(target_os = "macos")] { macos_impl::save_block(path, data) }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))] { default_impl::save_block(path, data) }
}

/// 현재 플랫폼에서 KV 블록 I/O 가 실제로 어떤 경로를 타는지 돌려줍니다.
/// ("directstorage" | "io_uring" | "metal-io" | "fs-fallback")
/// 진단 로그와 테스트(링 초기화 실패 → 조용한 폴백 감지)에 사용합니다.
pub fn io_backend() -> &'static str {
    #[cfg(windows)] { if windows_impl::is_available() { "directstorage" } else { "fs-fallback" } }
    #[cfg(target_os = "linux")] { if linux_impl::parallel_forced() { "parallel-pread" } else if linux_impl::is_available() { "io_uring" } else { "fs-fallback" } }
    #[cfg(target_os = "macos")] { if macos_impl::is_available() { "parallel-pread-nocache" } else { "fs-fallback" } }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))] { "fs-fallback" }
}
