use anyhow::{Result, anyhow};
use std::path::Path;
use std::fs;

// --- Windows Implementation ---
#[cfg(windows)]
mod windows_impl {
    use super::*;
    use direct_storage::*;
    use windows::core::HSTRING;
    use std::mem::ManuallyDrop;

    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        println!("[IO-WIN] Loading via DirectStorage: {:?}", path.file_name());
        unsafe {
            let factory: IDStorageFactory = DStorageGetFactory()?;
            let metadata = fs::metadata(path)?;
            let size = metadata.len() as usize;
            let path_str = path.to_string_lossy().to_string();
            let file: IDStorageFile = factory.OpenFile(&HSTRING::from(path_str))?;
            
            let mut buffer = vec![0u8; size];
            
            let queue_desc = DSTORAGE_QUEUE_DESC {
                SourceType: DSTORAGE_REQUEST_SOURCE_FILE,
                Capacity: DSTORAGE_MAX_QUEUE_CAPACITY as u16,
                Priority: DSTORAGE_PRIORITY_NORMAL,
                Name: windows::core::PCSTR::null(),
                Device: ManuallyDrop::new(None), 
            };
            
            let queue: IDStorageQueue = factory.CreateQueue(&queue_desc)?;
            
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
            
            queue.EnqueueRequest(&request);
            let status_array: IDStorageStatusArray = factory.CreateStatusArray(1, None)?;
            queue.EnqueueStatus(&status_array, 0);
            queue.Submit();
            
            while !status_array.IsComplete(0) {
                std::thread::yield_now();
            }
            
            let hr = status_array.GetHResult(0);
            if hr.is_err() {
                return Err(anyhow!("DirectStorage error: {:?}", hr));
            }
            Ok(buffer)
        }
    }

    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        // Windows DirectStorage currently does not support Write.
        // Falling back to standard fs::write.
        fs::write(path, data).map_err(|e| anyhow!(e))
    }
}

// --- Linux Implementation (io_uring for both Read/Write) ---
#[cfg(target_os = "linux")]
mod linux_impl {
    use super::*;
    use io_uring::{opcode, types, IoUring};
    use std::fs::File;
    use std::os::unix::io::AsRawFd;

    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        println!("[IO-LINUX] Loading via io_uring: {:?}", path.file_name());
        let file = File::open(path)?;
        let size = file.metadata()?.len() as usize;
        let mut buffer = vec![0u8; size];

        let mut ring = IoUring::new(8)?;
        let read_e = opcode::Read::new(types::Fd(file.as_raw_fd()), buffer.as_mut_ptr(), size as u32)
            .build()
            .user_data(0x42);

        unsafe {
            ring.submission().push(&read_e).map_err(|e| anyhow!("SQ push error: {}", e))?;
        }
        ring.submit_and_wait(1)?;

        let cq = ring.completion().next().ok_or_else(|| anyhow!("No CQE"))?;
        if cq.result() < 0 {
            return Err(anyhow!("Read failed with {}", cq.result()));
        }
        Ok(buffer)
    }

    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        println!("[IO-LINUX] Saving via io_uring: {:?}", path.file_name());
        let file = File::create(path)?;
        let mut ring = IoUring::new(8)?;
        let write_e = opcode::Write::new(types::Fd(file.as_raw_fd()), data.as_ptr(), data.len() as u32)
            .build()
            .user_data(0x43);

        unsafe {
            ring.submission().push(&write_e).map_err(|e| anyhow!("SQ push error: {}", e))?;
        }
        ring.submit_and_wait(1)?;

        let cq = ring.completion().next().ok_or_else(|| anyhow!("No CQE"))?;
        if cq.result() < 0 {
            return Err(anyhow!("Write failed with {}", cq.result()));
        }
        Ok(())
    }
}

// --- macOS Implementation (Metal IO for Read, Std for Write) ---
#[cfg(target_os = "macos")]
mod macos_impl {
    use super::*;
    use metal::*;

    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        println!("[IO-MAC] Loading via Metal IO: {:?}", path.file_name());
        let device = Device::system_default().ok_or_else(|| anyhow!("No Metal device found"))?;
        let queue_desc = IOCommandQueueDescriptor::new();
        let io_queue = device.new_io_command_queue(&queue_desc).map_err(|e| anyhow!("Failed to create Metal IO Queue: {}", e))?;
        
        let path_str = path.to_string_lossy().to_string();
        let io_handle = io_queue.new_io_handle(&path_str).map_err(|e| anyhow!("Failed to create Metal IO Handle: {}", e))?;
        
        let metadata = fs::metadata(path)?;
        let size = metadata.len() as usize;
        let mut buffer = vec![0u8; size];

        let command_buffer = io_queue.new_io_command_buffer();
        command_buffer.load_buffer(&io_handle, 0, size, buffer.as_mut_ptr() as *mut _, 0);
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        if let Some(err) = command_buffer.error() {
            return Err(anyhow!("Metal IO error: {:?}", err));
        }
        Ok(buffer)
    }

    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        // macOS Unified Memory makes VRAM -> RAM copy unnecessary.
        // Standard fs::write is efficient here.
        fs::write(path, data).map_err(|e| anyhow!(e))
    }
}

// --- Default/Fallback Implementation ---
#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
mod default_impl {
    use super::*;
    pub fn load_block(path: &Path) -> Result<Vec<u8>> {
        fs::read(path).map_err(|e| anyhow::anyhow!(e))
    }
    pub fn save_block(path: &Path, data: &[u8]) -> Result<()> {
        fs::write(path, data).map_err(|e| anyhow::anyhow!(e))
    }
}

// --- Public Unified API ---
pub fn load_kv_block(path: &Path) -> Result<Vec<u8>> {
    #[cfg(windows)]
    { windows_impl::load_block(path) }
    #[cfg(target_os = "linux")]
    { linux_impl::load_block(path) }
    #[cfg(target_os = "macos")]
    { macos_impl::load_block(path) }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    { default_impl::load_block(path) }
}

pub fn save_kv_block(path: &Path, data: &[u8]) -> Result<()> {
    #[cfg(windows)]
    { windows_impl::save_block(path, data) }
    #[cfg(target_os = "linux")]
    { linux_impl::save_block(path, data) }
    #[cfg(target_os = "macos")]
    { macos_impl::save_block(path, data) }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    { default_impl::save_block(path, data) }
}
