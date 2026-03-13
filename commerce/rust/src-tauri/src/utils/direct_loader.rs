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
    use windows::Win32::Storage::FileSystem::{CreateFileW, WriteFile, SetFilePointerEx, FILE_SHARE_WRITE, FILE_SHARE_READ, OPEN_EXISTING, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, FILE_FLAG_OVERLAPPED, FILE_BEGIN};
    use windows::Win32::Foundation::{HANDLE, CloseHandle, GENERIC_WRITE, GENERIC_READ};
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
            let status_array = factory.CreateStatusArray(128, None)?;
            Ok(Arc::new(WinContext { factory, queue, status_array }))
        }
    });

    pub fn load_block_from_offset(path: &Path, offset: u64, size: usize) -> Result<Vec<u8>> {
        let ctx = CONTEXT.as_ref().map_err(|e| anyhow!("DirectStorage init error: {}", e))?;
        unsafe {
            let path_str = path.to_string_lossy().to_string();
            let file: IDStorageFile = ctx.factory.OpenFile(&HSTRING::from(path_str))?;
            let mut buffer = vec![0u8; size];
            let mut request = DSTORAGE_REQUEST::default();
            request.Options.set_SourceType(DSTORAGE_REQUEST_SOURCE_FILE);
            request.Options.set_DestinationType(DSTORAGE_REQUEST_DESTINATION_MEMORY);
            request.Source.File = ManuallyDrop::new(DSTORAGE_SOURCE_FILE {
                Source: ManuallyDrop::new(Some(file.clone())),
                Offset: offset,
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

    pub fn save_block_at_offset(path: &Path, offset: u64, data: &[u8]) -> Result<()> {
        unsafe {
            let path_wide = HSTRING::from(path.to_string_lossy().as_ref());
            let handle: HANDLE = CreateFileW(
                &path_wide,
                GENERIC_WRITE.0,
                FILE_SHARE_WRITE | FILE_SHARE_READ,
                None,
                OPEN_EXISTING, // Assuming pool file already exists
                FILE_FLAG_OVERLAPPED | FILE_ATTRIBUTE_NORMAL,
                Some(HANDLE::default()),
            ).map_err(|e| anyhow!("Failed to open file: {}", e))?;

            let mut overlapped = OVERLAPPED::default();
            overlapped.Anonymous.Anonymous.Offset = (offset & 0xFFFFFFFF) as u32;
            overlapped.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
            
            let mut bytes_written = 0u32;
            let _ = WriteFile(handle, Some(data), Some(&mut bytes_written), Some(&mut overlapped));

            let mut transferred = 0u32;
            GetOverlappedResult(handle, &overlapped, &mut transferred, true).map_err(|e| anyhow!(e))?;
            let _ = CloseHandle(handle);
            Ok(())
        }
    }
}

// --- Linux Implementation (Simplified) ---
#[cfg(target_os = "linux")]
mod linux_impl {
    use super::*;
    use std::os::unix::fs::FileExt;
    use std::fs::OpenOptions;

    pub fn load_block_from_offset(path: &Path, offset: u64, size: usize) -> Result<Vec<u8>> {
        let file = fs::File::open(path)?;
        let mut buffer = vec![0u8; size];
        file.read_exact_at(&mut buffer, offset)?;
        Ok(buffer)
    }

    pub fn save_block_at_offset(path: &Path, offset: u64, data: &[u8]) -> Result<()> {
        let file = OpenOptions::new().write(true).open(path)?;
        file.write_all_at(data, offset)?;
        Ok(())
    }
}

// --- macOS Implementation (Simplified) ---
#[cfg(target_os = "macos")]
mod macos_impl {
    use super::*;
    use std::os::unix::fs::FileExt;
    use std::fs::OpenOptions;

    pub fn load_block_from_offset(path: &Path, offset: u64, size: usize) -> Result<Vec<u8>> {
        let file = fs::File::open(path)?;
        let mut buffer = vec![0u8; size];
        file.read_exact_at(&mut buffer, offset)?;
        Ok(buffer)
    }

    pub fn save_block_at_offset(path: &Path, offset: u64, data: &[u8]) -> Result<()> {
        let file = OpenOptions::new().write(true).open(path)?;
        file.write_all_at(data, offset)?;
        Ok(())
    }
}

pub fn load_kv_block_at(path: &Path, offset: u64, size: usize) -> Result<Vec<u8>> {
    #[cfg(windows)] { windows_impl::load_block_from_offset(path, offset, size) }
    #[cfg(target_os = "linux")] { linux_impl::load_block_from_offset(path, offset, size) }
    #[cfg(target_os = "macos")] { macos_impl::load_block_from_offset(path, offset, size) }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))] { 
        use std::io::{Seek, SeekFrom, Read};
        let mut file = fs::File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; size];
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }
}

pub fn save_kv_block_at(path: &Path, offset: u64, data: &[u8]) -> Result<()> {
    #[cfg(windows)] { windows_impl::save_block_at_offset(path, offset, data) }
    #[cfg(target_os = "linux")] { linux_impl::save_block_at_offset(path, offset, data) }
    #[cfg(target_os = "macos")] { macos_impl::save_block_at_offset(path, offset, data) }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))] {
        use std::io::{Seek, SeekFrom, Write};
        let mut file = fs::OpenOptions::new().write(true).open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(data)?;
        Ok(())
    }
}

// Keep original functions for compatibility or as convenience wrappers
pub fn load_kv_block(path: &Path) -> Result<Vec<u8>> {
    let size = fs::metadata(path)?.len() as usize;
    load_kv_block_at(path, 0, size)
}

pub fn save_kv_block(path: &Path, data: &[u8]) -> Result<()> {
    // This creates/overwrites the file, different from save_kv_block_at which assumes existence
    fs::write(path, data)?;
    Ok(())
}

