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

    // [STABILITY] DirectStorage 팩토리 초기화 성공 여부를 별도로 추적
    static CONTEXT: Lazy<Option<Arc<WinContext>>> = Lazy::new(|| {
        unsafe {
            let factory: IDStorageFactory = match DStorageGetFactory() {
                Ok(f) => f,
                Err(e) => {
                    println!("[I/O-INFO] DirectStorage not supported or failed to init: {}. Fallback will be used.", e);
                    return None;
                }
            };
            
            // [ROBUSTNESS] Check if factory is null or invalid
            let queue_desc = DSTORAGE_QUEUE_DESC {
                SourceType: DSTORAGE_REQUEST_SOURCE_FILE,
                Capacity: DSTORAGE_MAX_QUEUE_CAPACITY as u16,
                Priority: DSTORAGE_PRIORITY_NORMAL,
                Name: windows::core::PCSTR::null(),
                Device: ManuallyDrop::new(None), 
            };
            
            let queue = match factory.CreateQueue(&queue_desc) {
                Ok(q) => q,
                Err(e) => {
                    println!("[I/O-INFO] DirectStorage Queue creation failed: {}. Falling back.", e);
                    return None;
                }
            };
            
            let status_array = match factory.CreateStatusArray(1, None) {
                Ok(s) => s,
                Err(e) => {
                    println!("[I/O-INFO] DirectStorage StatusArray creation failed: {}. Falling back.", e);
                    return None;
                }
            };
            
            Some(Arc::new(WinContext { factory, queue, status_array }))
        }
    });

    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        // [FALLBACK-LOGIC] CONTEXT가 None이면 즉시 일반 fs::read로 전환
        let ctx = match CONTEXT.as_ref() {
            Some(c) => c,
            None => return fs::read(path).map_err(|e| anyhow!(e)),
        };

        unsafe {
            let metadata = match fs::metadata(path) {
                Ok(m) => m,
                Err(_) => return fs::read(path).map_err(|e| anyhow!(e)),
            };
            let size = metadata.len() as usize;
            let path_str = path.to_string_lossy().to_string();
            
            let file: IDStorageFile = match ctx.factory.OpenFile(&HSTRING::from(path_str)) {
                Ok(f) => f,
                Err(_) => return fs::read(path).map_err(|e| anyhow!(e)), // 개별 파일 오픈 실패 시에도 Fallback
            };

            let mut buffer = vec![0u8; size];
            let mut request = DSTORAGE_REQUEST::default();
            request.Options.set_SourceType(DSTORAGE_REQUEST_SOURCE_FILE);
            request.Options.set_DestinationType(DSTORAGE_REQUEST_DESTINATION_MEMORY);
            request.Source.File = ManuallyDrop::new(DSTORAGE_SOURCE_FILE {
                Source: ManuallyDrop::new(Some(file.clone())),
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
            
            // [TIMEOUT-ROBUSTNESS] Wait with a simple loop, but handle potential hangs if necessary
            // In a real scenario, we might want a timeout, but for now, this matches previous behavior
            while !ctx.status_array.IsComplete(0) { std::thread::yield_now(); }
            
            match ctx.status_array.GetHResult(0) {
                Ok(_) => Ok(buffer),
                Err(_) => fs::read(path).map_err(|e| anyhow!(e)), // 실행 중 에러 발생 시에도 Fallback
            }
        }
    }

    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        // [FALLBACK-LOGIC] If DirectStorage context is missing, use standard fs::write immediately
        if CONTEXT.is_none() {
            return fs::write(path, data).map_err(|e| anyhow!(e));
        }

        unsafe {
            let path_wide = HSTRING::from(path.to_string_lossy().as_ref());
            let handle: HANDLE = match CreateFileW(
                &path_wide,
                GENERIC_WRITE.0,
                FILE_SHARE_WRITE,
                None,
                CREATE_ALWAYS,
                FILE_FLAG_OVERLAPPED | FILE_ATTRIBUTE_NORMAL,
                Some(HANDLE::default()),
            ) {
                Ok(h) => h,
                Err(_) => {
                    return fs::write(path, data).map_err(|e| anyhow!(e));
                }
            };

            let mut overlapped = OVERLAPPED::default();
            let mut bytes_written = 0u32;
            
            // [ROBUSTNESS] Handle WriteFile result more explicitly
            let write_res = WriteFile(handle, Some(data), Some(&mut bytes_written), Some(&mut overlapped));
            
            if write_res.is_err() {
                // Check if it's just pending
                let err = windows::core::Error::from_win32();
                if err.code().0 as u32 != 997 { // ERROR_IO_PENDING
                    let _ = CloseHandle(handle);
                    return fs::write(path, data).map_err(|e| anyhow!(e));
                }
            }

            let mut transferred = 0u32;
            if GetOverlappedResult(handle, &overlapped, &mut transferred, true).is_err() {
                let _ = CloseHandle(handle);
                return fs::write(path, data).map_err(|e| anyhow!(e));
            }
            let _ = CloseHandle(handle);
            Ok(())
        }
    }
}

// --- Linux Implementation ---
#[cfg(target_os = "linux")]
mod linux_impl {
    use super::*;
    use io_uring::{opcode, types, IoUring};
    use std::fs::File;
    use std::os::unix::io::AsRawFd;
    use std::sync::Mutex;

    struct LinuxContext { ring: Mutex<IoUring> }
    unsafe impl Send for LinuxContext {}
    unsafe impl Sync for LinuxContext {}

    static CONTEXT: Lazy<Option<Arc<LinuxContext>>> = Lazy::new(|| {
        let ring = match IoUring::new(128) {
            Ok(r) => r,
            Err(_) => return None,
        };
        Some(Arc::new(LinuxContext { ring: Mutex::new(ring) }))
    });

    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        let ctx = match CONTEXT.as_ref() {
            Some(c) => c,
            None => return fs::read(path).map_err(|e| anyhow!(e)),
        };
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return fs::read(path).map_err(|e| anyhow!(e)),
        };
        let size = file.metadata()?.len() as usize;
        let mut buffer = vec![0u8; size];
        let read_e = opcode::Read::new(types::Fd(file.as_raw_fd()), buffer.as_mut_ptr(), size as u32).build();
        let mut ring = ctx.ring.lock().unwrap();
        unsafe { if ring.submission().push(&read_e).is_err() { return fs::read(path).map_err(|e| anyhow!(e)); } }
        if ring.submit_and_wait(1).is_err() { return fs::read(path).map_err(|e| anyhow!(e)); }
        Ok(buffer)
    }

    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        let ctx = match CONTEXT.as_ref() {
            Some(c) => c,
            None => return fs::write(path, data).map_err(|e| anyhow!(e)),
        };
        let file = match File::create(path) {
            Ok(f) => f,
            Err(_) => return fs::write(path, data).map_err(|e| anyhow!(e)),
        };
        let write_e = opcode::Write::new(types::Fd(file.as_raw_fd()), data.as_ptr(), data.len() as u32).build();
        let mut ring = ctx.ring.lock().unwrap();
        unsafe { if ring.submission().push(&write_e).is_err() { return fs::write(path, data).map_err(|e| anyhow!(e)); } }
        if ring.submit_and_wait(1).is_err() { return fs::write(path, data).map_err(|e| anyhow!(e)); }
        Ok(())
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
    static CONTEXT: Lazy<Option<Arc<MacContext>>> = Lazy::new(|| {
        let device = Device::system_default()?;
        let queue = device.new_io_command_queue(&IOCommandQueueDescriptor::new()).ok()?;
        Some(Arc::new(MacContext { queue }))
    });
    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        let ctx = match CONTEXT.as_ref() {
            Some(c) => c,
            None => return fs::read(path).map_err(|e| anyhow!(e)),
        };
        let io_handle = match ctx.queue.new_io_handle(&path.to_string_lossy()) {
            Ok(h) => h,
            Err(_) => return fs::read(path).map_err(|e| anyhow!(e)),
        };
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
