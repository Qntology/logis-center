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

    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
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
#[cfg(target_os = "macos")]
mod macos_impl {
    use super::*;
    use metal::*;
    struct MacContext { queue: IOCommandQueue }
    unsafe impl Send for MacContext {}
    unsafe impl Sync for MacContext {}
    pub fn is_available() -> bool { CONTEXT.is_ok() }
    static CONTEXT: Lazy<Result<Arc<MacContext>>> = Lazy::new(|| {
        let device = Device::system_default().ok_or_else(|| {
            println!("[DirectLoader] No Metal device found. Falling back to fs::read/fs::write.");
            anyhow!("No Metal device found")
        })?;
        let queue = device.new_io_command_queue(&IOCommandQueueDescriptor::new()).map_err(|e| {
            println!("[DirectLoader] Metal IO command queue creation failed: {}. Falling back to fs::read/fs::write.", e);
            anyhow!(e)
        })?;
        println!("[DirectLoader] Metal IO command queue initialized successfully.");
        Ok(Arc::new(MacContext { queue }))
    });
    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        let ctx = CONTEXT.as_ref().map_err(|e| anyhow!(e))?;
        let io_handle = ctx.queue.new_io_handle(&path.to_string_lossy()).map_err(|e| anyhow!(e))?;
        let size = fs::metadata(path)?.len() as usize;
        let mut buffer = vec![0u8; size];
        let command_buffer = ctx.queue.new_io_command_buffer();
        command_buffer.load_buffer(&io_handle, 0, size, buffer.as_mut_ptr() as *mut _, 0);
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(buffer)
    }
    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> { fs::write(path, data).map_err(|e| anyhow!(e)) }
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
    #[cfg(target_os = "linux")] { if linux_impl::is_available() { "io_uring" } else { "fs-fallback" } }
    #[cfg(target_os = "macos")] { if macos_impl::is_available() { "metal-io" } else { "fs-fallback" } }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))] { "fs-fallback" }
}
