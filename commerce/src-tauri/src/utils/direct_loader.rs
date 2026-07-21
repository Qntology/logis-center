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

    static CONTEXT: Lazy<Result<Arc<WinContext>>> = Lazy::new(|| {
        unsafe {
            let factory: IDStorageFactory = DStorageGetFactory()?;
            let queue_desc = DSTORAGE_QUEUE_DESC {
                SourceType: DSTORAGE_REQUEST_SOURCE_FILE,
                Capacity: DSTORAGE_MAX_QUEUE_CAPACITY as u16,
                Priority: DSTORAGE_PRIORITY_NORMAL,
                Name: windows::core::PCSTR::null(),
                Device: ManuallyDrop::new(None), 
            };
            let queue = factory.CreateQueue(&queue_desc)?;
            let status_array = factory.CreateStatusArray(1, None)?;
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
                        while !ctx.status_array.IsComplete(0) { std::thread::yield_now(); }
                        if ctx.status_array.GetHResult(0).is_ok() {
                            return Ok(buffer);
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
                    let _ = WriteFile(handle, Some(data), Some(&mut bytes_written), Some(&mut overlapped));

                    let mut transferred = 0u32;
                    let result = GetOverlappedResult(handle, &overlapped, &mut transferred, true);
                    let _ = CloseHandle(handle);
                    if result.is_ok() {
                        return Ok(());
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
    use std::sync::Mutex;

    struct LinuxContext { ring: Mutex<IoUring> }
    unsafe impl Send for LinuxContext {}
    unsafe impl Sync for LinuxContext {}

    static CONTEXT: Lazy<Result<Arc<LinuxContext>>> = Lazy::new(|| {
        let ring = IoUring::new(128).map_err(|e| anyhow!(e))?;
        Ok(Arc::new(LinuxContext { ring: Mutex::new(ring) }))
    });

    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        let ctx = CONTEXT.as_ref().map_err(|e| anyhow!(e))?;
        let file = File::open(path)?;
        let size = file.metadata()?.len() as usize;
        let mut buffer = vec![0u8; size];
        let read_e = opcode::Read::new(types::Fd(file.as_raw_fd()), buffer.as_mut_ptr(), size as u32).build();
        let mut ring = ctx.ring.lock().unwrap();
        unsafe { ring.submission().push(&read_e).map_err(|e| anyhow!(e))?; }
        ring.submit_and_wait(1)?;
        Ok(buffer)
    }

    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        let ctx = CONTEXT.as_ref().map_err(|e| anyhow!(e))?;
        let file = File::create(path)?;
        let write_e = opcode::Write::new(types::Fd(file.as_raw_fd()), data.as_ptr(), data.len() as u32).build();
        let mut ring = ctx.ring.lock().unwrap();
        unsafe { ring.submission().push(&write_e).map_err(|e| anyhow!(e))?; }
        ring.submit_and_wait(1)?;
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
    static CONTEXT: Lazy<Result<Arc<MacContext>>> = Lazy::new(|| {
        let device = Device::system_default().ok_or_else(|| anyhow!("No Metal device found"))?;
        let queue = device.new_io_command_queue(&IOCommandQueueDescriptor::new()).map_err(|e| anyhow!(e))?;
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