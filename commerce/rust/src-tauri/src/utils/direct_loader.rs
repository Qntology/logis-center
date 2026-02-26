use anyhow::{Result, anyhow};
use std::path::Path;
use std::fs;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};

// --- Windows Implementation ---
#[cfg(windows)]
mod windows_impl {
    use super::*;
    use direct_storage::*;
    use windows::core::HSTRING;
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
        let ctx = CONTEXT.as_ref().map_err(|e| anyhow!("DirectStorage init error: {}", e))?;
        unsafe {
            let metadata = fs::metadata(path)?;
            let size = metadata.len() as usize;
            let path_str = path.to_string_lossy().to_string();
            let file: IDStorageFile = ctx.factory.OpenFile(&HSTRING::from(path_str))?;
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
            while !ctx.status_array.IsComplete(0) { std::thread::yield_now(); }
            ctx.status_array.GetHResult(0).map_err(|e| anyhow!(e))?;
            Ok(buffer)
        }
    }

    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        fs::write(path, data).map_err(|e| anyhow!(e))
    }
}

// --- Linux Implementation (Singleton io_uring) ---
#[cfg(target_os = "linux")]
mod linux_impl {
    use super::*;
    use io_uring::{opcode, types, IoUring};
    use std::fs::File;
    use std::os::unix::io::AsRawFd;

    struct LinuxContext {
        ring: Mutex<IoUring>,
    }

    unsafe impl Send for LinuxContext {}
    unsafe impl Sync for LinuxContext {}

    static CONTEXT: Lazy<Result<Arc<LinuxContext>>> = Lazy::new(|| {
        let ring = IoUring::new(16).map_err(|e| anyhow!("Failed to init io_uring: {}", e))?;
        Ok(Arc::new(LinuxContext { ring: Mutex::new(ring) }))
    });

    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        let ctx = CONTEXT.as_ref().map_err(|e| anyhow!("Linux IO init error: {}", e))?;
        let file = File::open(path)?;
        let size = file.metadata()?.len() as usize;
        let mut buffer = vec![0u8; size];

        let read_e = opcode::Read::new(types::Fd(file.as_raw_fd()), buffer.as_mut_ptr(), size as u32)
            .build()
            .user_data(0x42);

        let mut ring = ctx.ring.lock().unwrap();
        unsafe { ring.submission().push(&read_e).map_err(|e| anyhow!("SQ push error: {}", e))?; }
        ring.submit_and_wait(1)?;
        let cq = ring.completion().next().ok_or_else(|| anyhow!("No CQE"))?;
        if cq.result() < 0 { return Err(anyhow!("Read failed with {}", cq.result())); }
        Ok(buffer)
    }

    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        let ctx = CONTEXT.as_ref().map_err(|e| anyhow!("Linux IO init error: {}", e))?;
        let file = File::create(path)?;
        let write_e = opcode::Write::new(types::Fd(file.as_raw_fd()), data.as_ptr(), data.len() as u32)
            .build()
            .user_data(0x43);

        let mut ring = ctx.ring.lock().unwrap();
        unsafe { ring.submission().push(&write_e).map_err(|e| anyhow!("SQ push error: {}", e))?; }
        ring.submit_and_wait(1)?;
        let cq = ring.completion().next().ok_or_else(|| anyhow!("No CQE"))?;
        if cq.result() < 0 { return Err(anyhow!("Write failed with {}", cq.result())); }
        Ok(())
    }
}

// --- macOS Implementation (Singleton Metal IO) ---
#[cfg(target_os = "macos")]
mod macos_impl {
    use super::*;
    use metal::*;

    struct MacContext {
        device: Device,
        queue: IOCommandQueue,
    }

    unsafe impl Send for MacContext {}
    unsafe impl Sync for MacContext {}

    static CONTEXT: Lazy<Result<Arc<MacContext>>> = Lazy::new(|| {
        let device = Device::system_default().ok_or_else(|| anyhow!("No Metal device found"))?;
        let queue_desc = IOCommandQueueDescriptor::new();
        let queue = device.new_io_command_queue(&queue_desc).map_err(|e| anyhow!("Failed to create Metal IO Queue: {}", e))?;
        Ok(Arc::new(MacContext { device, queue }))
    });

    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        let ctx = CONTEXT.as_ref().map_err(|e| anyhow!("Mac IO init error: {}", e))?;
        let path_str = path.to_string_lossy().to_string();
        let io_handle = ctx.queue.new_io_handle(&path_str).map_err(|e| anyhow!("Failed to create Metal IO Handle: {}", e))?;
        
        let metadata = fs::metadata(path)?;
        let size = metadata.len() as usize;
        let mut buffer = vec![0u8; size];

        let command_buffer = ctx.queue.new_io_command_buffer();
        command_buffer.load_buffer(&io_handle, 0, size, buffer.as_mut_ptr() as *mut _, 0);
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        if let Some(err) = command_buffer.error() { return Err(anyhow!("Metal IO error: {:?}", err)); }
        Ok(buffer)
    }

    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        fs::write(path, data).map_err(|e| anyhow!(e))
    }
}

// --- Default/Fallback Implementation ---
#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
mod default_impl {
    use super::*;
    pub fn load_block(path: &Path) -> Result<Vec<u8>> { fs::read(path).map_err(|e| anyhow::anyhow!(e)) }
    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> { fs::write(path, data).map_err(|e| anyhow::anyhow!(e)) }
}

// --- Public Unified API ---
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
