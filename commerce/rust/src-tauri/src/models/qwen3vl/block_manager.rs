use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use parking_lot::{Mutex, RwLock};
use memmap2::MmapMut;
use anyhow::{Result, anyhow};
use std::collections::{HashMap, VecDeque};

/// [BLOCK-LOCATION] 블록 데이터가 현재 위치한 저장소 계층
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockLocation {
    SSD = 0,    // SSD 풀 파일에만 존재
    RAM = 1,    // CPU Pinned Memory에 로드됨 (복사 대기)
    VRAM = 2,   // GPU VRAM에 로드됨 (연산 즉시 가능)
}

/// [BLOCK-STATUS] 블록의 상세 작업 상태 (외계어 방지 핵심)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockStatus {
    Free = 0,       // 미사용 블록
    Reserved = 1,   // 할당됨 (작업 대기)
    Loading = 2,    // SSD -> RAM 또는 RAM -> VRAM 이동 중 (접근 금지)
    Valid = 3,      // 데이터 정합성 완료 (연산 가능)
    Dirty = 4,      // VRAM 데이터가 변경되어 SSD와 불일치 (백업 필요)
    Evicting = 5,   // VRAM -> SSD로 내려가는 중
}

/// [PHYSICAL-BLOCK] 물리적 블록 제어 유닛
pub struct PhysicalBlock {
    pub id: usize,
    pub offset: u64,
    pub location: AtomicU8, // BlockLocation
    pub status: AtomicU8,   // BlockStatus
    pub last_access: Mutex<std::time::Instant>, // LRU 교체용
}

impl PhysicalBlock {
    pub fn set_status(&self, status: BlockStatus) {
        self.status.store(status as u8, Ordering::SeqCst);
    }

    pub fn get_status(&self) -> BlockStatus {
        unsafe { std::mem::transmute(self.status.load(Ordering::SeqCst)) }
    }

    pub fn set_location(&self, loc: BlockLocation) {
        self.location.store(loc as u8, Ordering::SeqCst);
    }

    pub fn get_location(&self) -> BlockLocation {
        unsafe { std::mem::transmute(self.location.load(Ordering::SeqCst)) }
    }
}

pub struct KVCachePool {
    pub path: PathBuf,
    pub mmap: MmapMut,
    pub block_size_bytes: usize,
    pub total_blocks: usize,
}

impl KVCachePool {
    pub fn new(path: &Path, num_blocks: usize, block_size_bytes: usize) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
        file.set_len((num_blocks * block_size_bytes) as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { path: path.to_path_buf(), mmap, block_size_bytes, total_blocks: num_blocks })
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct LogicalKey {
    pub session_id: String,
    pub block_idx: usize,
}

/// [BLOCK-MANAGER] 정교한 스케줄링 및 배리어 관리자
pub struct BlockManager {
    pub pool: Arc<RwLock<KVCachePool>>,
    pub physical_blocks: Vec<PhysicalBlock>,
    pub free_list: Mutex<VecDeque<usize>>,
    pub mapping: RwLock<HashMap<LogicalKey, usize>>,
    pub block_size_tokens: usize,
    pub pinned_buffer_pool: Mutex<VecDeque<Vec<u8>>>, // [병목 제거] Double Buffering용 Pinned 버퍼
}

impl BlockManager {
    pub fn new(pool_path: &Path, max_blocks: usize, tokens_per_block: usize, 
               layers: usize, kv_heads: usize, head_dim: usize, dtype_size: usize) -> Result<Self> {
        
        let block_size_bytes = tokens_per_block * layers * kv_heads * head_dim * dtype_size * 2;
        let pool = KVCachePool::new(pool_path, max_blocks, block_size_bytes)?;
        
        let mut physical_blocks = Vec::with_capacity(max_blocks);
        let mut free_list = VecDeque::with_capacity(max_blocks);
        for i in 0..max_blocks {
            physical_blocks.push(PhysicalBlock {
                id: i,
                offset: (i * block_size_bytes) as u64,
                location: AtomicU8::new(BlockLocation::SSD as u8),
                status: AtomicU8::new(BlockStatus::Free as u8),
                last_access: Mutex::new(std::time::Instant::now()),
            });
            free_list.push_back(i);
        }

        // Pinned 버퍼 미리 할당 (예: 4개 블록 분량으로 더블 버퍼링 지원)
        let mut pinned_buffer_pool = VecDeque::new();
        for _ in 0..4 {
            pinned_buffer_pool.push_back(vec![0u8; block_size_bytes]);
        }

        Ok(Self {
            pool: Arc::new(RwLock::new(pool)),
            physical_blocks,
            free_list: Mutex::new(free_list),
            mapping: RwLock::new(HashMap::new()),
            block_size_tokens: tokens_per_block,
            pinned_buffer_pool: Mutex::new(pinned_buffer_pool),
        })
    }

    /// [SCHEDULING] 블록 할당 및 우선순위 갱신
    pub fn allocate(&self, session_id: &str, block_idx: usize) -> Result<usize> {
        let mut free_list = self.free_list.lock();
        if let Some(phys_id) = free_list.pop_front() {
            let key = LogicalKey { session_id: session_id.to_string(), block_idx };
            self.mapping.write().insert(key, phys_id);
            let block = &self.physical_blocks[phys_id];
            block.set_status(BlockStatus::Reserved);
            *block.last_access.lock() = std::time::Instant::now();
            Ok(phys_id)
        } else {
            Err(anyhow!("KV Cache Pool Full! LRU Eviction Needed."))
        }
    }

    /// [BARRIER] 외계어 방지를 위한 동기화 배리어
    /// 데이터가 Valid 상태가 될 때까지 스레드를 대기시키거나 이벤트를 확인합니다.
    pub fn wait_for_block_valid(&self, phys_id: usize) -> Result<()> {
        let block = &self.physical_blocks[phys_id];
        let start = std::time::Instant::now();
        
        while block.get_status() == BlockStatus::Loading {
            // 타임아웃 1초 (SSD 장애 등 대비)
            if start.elapsed().as_secs() > 1 {
                return Err(anyhow!("Block {} loading timeout! Bottleneck or SSD error.", phys_id));
            }
            std::thread::yield_now(); // CPU 점유율 조절
        }

        if block.get_status() != BlockStatus::Valid {
            return Err(anyhow!("Block {} is in invalid state: {:?}", phys_id, block.get_status()));
        }
        
        Ok(())
    }

    pub fn get_physical_id(&self, session_id: &str, block_idx: usize) -> Option<usize> {
        let key = LogicalKey { session_id: session_id.to_string(), block_idx };
        self.mapping.read().get(&key).cloned()
    }

    /// [DOUBLE-BUFFERING] 비동기 로딩을 위한 버퍼 확보
    pub fn acquire_io_buffer(&self) -> Result<Vec<u8>> {
        self.pinned_buffer_pool.lock().pop_front()
            .ok_or(anyhow!("IO Pinned Buffer Pool exhausted!"))
    }

    pub fn release_io_buffer(&self, buffer: Vec<u8>) {
        self.pinned_buffer_pool.lock().push_back(buffer);
    }
}
