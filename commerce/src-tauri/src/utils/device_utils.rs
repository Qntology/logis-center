use candle_core::{Device, DType};
use anyhow::Result;
#[cfg(feature = "cuda")]
use std::process::Command;
use nvml_wrapper::Nvml;
use once_cell::sync::Lazy;
use std::sync::Mutex;
// mi_collect(release_heap) 를 쓰는 lib 대상(cdylib 포함)에도 mimalloc 정적 라이브러리를 링크합니다.
extern crate mimalloc as _;

// [FIX] 장치 번호별로 장치 객체를 캐싱하여 중복 생성 방지 (DeviceId 폭주 해결)
static DEVICE_CACHE: Lazy<Mutex<Vec<Option<Device>>>> = Lazy::new(|| Mutex::new(vec![None; 8]));

#[cfg(feature = "vulkan")]
static VULKAN_DEVICE_COUNT: Lazy<usize> =
    Lazy::new(|| candle_core::vulkan_backend::device_count().unwrap_or(0));

pub trait GpuDeviceExt {
    fn is_cuda_or_rocm(&self) -> bool;
}

impl GpuDeviceExt for Device {
    fn is_cuda_or_rocm(&self) -> bool {
        self.is_cuda() || self.is_rocm()
    }
}

pub fn new_gpu_device(id: usize) -> Option<Device> {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(id) {
        return Some(d);
    }
    #[cfg(feature = "rocm")]
    if let Ok(d) = Device::new_rocm(id) {
        ROCM_USED.store(true, std::sync::atomic::Ordering::SeqCst);
        return Some(d);
    }
    #[cfg(feature = "vulkan")]
    if let Ok(d) = Device::new_vulkan(id) {
        return Some(d);
    }
    // 🌟 [MAC-VERIFY] Metal 빌드도 같은 진입점으로 장치를 엽니다(get_gpu_device 의 Metal 분기와 같은 장치).
    //    없으면 테스트(app_embed 등)가 Metal 빌드에서 'GPU 장치가 없어 건너뜁니다' 로 조용히 빠졌습니다.
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(id) {
        return Some(d);
    }
    let _ = id;
    None
}

/// ROCm 장치를 한 번이라도 만들었는지 (종료 경로 판정용).
static ROCM_USED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// =====================================================================
// 🌟 [ROCM-EXIT] Windows ROCm 종료 시 멈춤 회피
// ---------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   AMD 검증(RX 6600, HIP SDK 7.2)에서 GPU 테스트 프로세스 68개 중 2개가 할 일을 다 끝낸 뒤
//   ExitProcess 안에서 멈췄습니다. ExitProcess 는 다른 스레드를 먼저 끝낸 다음 DLL 을 내리는데,
//   rocblas.dll 의 종료 처리기가 amdhip64 를 부르고, amdhip64 는 이미 사라진 HIP 런타임
//   스레드를 기다리며 무한히 돕니다(SwitchToThread 루프). taskkill 로도 끝나지 않고 GPU 를 쥔 채 남습니다.
//  ── 해결 ──
//   ROCm 장치를 쓴 프로세스는 종료 직전에 GPU 작업을 끝내고(동기화 + 풀 반환) 출력을 비운 뒤
//   TerminateProcess 로 끝냅니다. DLL 해제 단계를 건너뛰므로 멈출 자리가 없습니다.
//   Rust 소멸자는 std::process::exit 에서도 돌지 않으므로 잃는 정리는 C 수준 atexit 뿐입니다.
//   `LOGIS_ROCM_FAST_EXIT=0` 이면 예전처럼 정상 종료 경로를 탑니다.
// =====================================================================
/// ROCm 장치를 쓴 Windows 프로세스면 GPU 를 정리하고 즉시 프로세스를 끝냅니다(돌아오지 않음).
/// 그 밖의 빌드·장치에서는 아무것도 하지 않고 돌아옵니다.
pub fn rocm_exit_now_if_used(code: u32) {
    #[cfg(all(windows, feature = "rocm"))]
    {
        use std::io::Write;
        if !ROCM_USED.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        if std::env::var("LOGIS_ROCM_FAST_EXIT").map(|v| v.trim() == "0").unwrap_or(false) {
            println!("[ROCM-EXIT] LOGIS_ROCM_FAST_EXIT=0 → 정상 종료 경로를 탑니다.");
            return;
        }
        let devices: Vec<Device> = DEVICE_CACHE
            .lock()
            .map(|c| c.iter().flatten().filter(|d| d.is_rocm()).cloned().collect())
            .unwrap_or_default();
        for d in &devices {
            let _ = d.synchronize();
            if let Device::Rocm(r) = d {
                let _ = r.trim_memory_pool();
            }
        }
        println!("[ROCM-EXIT] GPU 정리 완료 → DLL 해제를 건너뛰고 종료합니다 (rocBLAS/HIP 종료 멈춤 회피).");
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        unsafe {
            use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
            TerminateProcess(GetCurrentProcess(), code);
        }
    }
    let _ = code;
}

// =====================================================================
// 🌟 [VRAM-WDDM] Windows ROCm 의 free VRAM 을 다른 프로세스 사용량까지 반영해 보정
// ---------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   Windows 의 hipMemGetInfo free 는 '전체 − 이 프로세스가 잡은 양' 입니다. 다른 프로세스(dwm,
//   브라우저, AMD 소프트웨어 등)가 쓰는 VRAM 이 빠져 있어, RX 6600 실측으로 장치 전체 기준보다
//   약 1GB 크게 나왔습니다. 8GB 카드에서는 KV-PLAN 이 실제로는 모자란 VRAM 을 고를 수 있습니다.
//  ── 해결 ──
//   DXGI 가 이 프로세스에 주는 로컬 메모리 예산(Budget − CurrentUsage)은 다른 프로세스 사용량을
//   반영합니다. AMD 어댑터(VendorId 0x1002)의 그 값과 HIP free 중 작은 쪽을 씁니다.
//   메모리 풀 유휴분은 이 프로세스가 이미 잡은 양이라 보정 뒤에 그대로 더합니다.
//   DXGI 조회가 실패하면 HIP 값을 그대로 씁니다. `LOGIS_VRAM_WDDM=0` 이면 보정하지 않습니다.
// =====================================================================
#[cfg(all(windows, feature = "rocm"))]
mod wddm {
    use std::sync::Mutex;
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        DXGI_QUERY_VIDEO_MEMORY_INFO,
    };

    struct Adapter(IDXGIAdapter3);
    // DXGI 어댑터 질의는 스레드에 묶이지 않습니다(free-threaded). 호출은 Mutex 로 직렬화합니다.
    unsafe impl Send for Adapter {}

    static ADAPTER: Mutex<Option<Option<Adapter>>> = Mutex::new(None);

    /// 전용 메모리가 가장 큰 AMD 어댑터.
    fn find_amd_adapter() -> Option<Adapter> {
        unsafe {
            let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
            let mut best: Option<(usize, IDXGIAdapter3)> = None;
            let mut i = 0;
            while let Ok(a) = factory.EnumAdapters1(i) {
                i += 1;
                let Ok(desc) = a.GetDesc1() else { continue };
                if desc.VendorId != 0x1002 {
                    continue;
                }
                let Ok(a3) = a.cast::<IDXGIAdapter3>() else { continue };
                if best.as_ref().map_or(true, |(m, _)| desc.DedicatedVideoMemory > *m) {
                    best = Some((desc.DedicatedVideoMemory, a3));
                }
            }
            best.map(|(_, a)| Adapter(a))
        }
    }

    /// 이 프로세스가 더 쓸 수 있는 로컬(VRAM) 메모리 = Budget − CurrentUsage.
    pub fn local_free_bytes() -> Option<u64> {
        let mut g = ADAPTER.lock().ok()?;
        if g.is_none() {
            *g = Some(find_amd_adapter());
        }
        let adapter = g.as_ref()?.as_ref()?;
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe { adapter.0.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info).ok()? };
        Some(info.Budget.saturating_sub(info.CurrentUsage))
    }
}

/// ROCm(Windows) 의 드라이버 free 를 DXGI 예산으로 보정합니다(작은 쪽). 그 밖에는 그대로.
fn clamp_free_to_os_budget(gpu_id: usize, is_rocm: bool, free: u64) -> u64 {
    #[cfg(all(windows, feature = "rocm"))]
    {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let off = *OFF.get_or_init(|| std::env::var("LOGIS_VRAM_WDDM").map(|v| v.trim() == "0").unwrap_or(false));
        if is_rocm && gpu_id == 0 && !off {
            if let Some(budget_free) = wddm::local_free_bytes() {
                return free.min(budget_free);
            }
        }
    }
    let _ = (gpu_id, is_rocm);
    free
}

// =====================================================================
// 🌟 [NVML-GLOBAL] 프로세스 전역 NVML 핸들
// ---------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   GpuMemProbe::new() 가 호출마다 Nvml::init() 을 수행했습니다.
//   nvml-wrapper 0.9 의 init 은 nvml.dll 을 LoadLibrary → nvmlInit_v2 하고,
//   Drop 에서 nvmlShutdown → FreeLibrary 까지 합니다.
//   이 PC 실측으로 질의 1회가 15~16ms(첫 회 88~214ms) 였고, 살아 있는
//   핸들로 질의하면 0.007ms 입니다. 임베딩 미적중 1건이 2~5회 질의를 하므로
//   검색 1회에 수 초가 VRAM '측정' 에만 쓰이고 있었습니다.
//
//  ── 해결 ──
//   최초 성공한 Nvml 을 OnceCell 에 두고 프로세스가 끝날 때까지 재사용합니다.
//   Nvml 은 Send + Sync 이므로 &'static 으로 공유해도 안전합니다.
//   반환하는 숫자는 기존과 완전히 같고(같은 nvmlDeviceGetMemoryInfo),
//   init/shutdown 만 사라집니다. 초기화 실패는 5초 간격으로만 재시도하여
//   NVIDIA 드라이버가 없는 PC 에서 매 질의마다 DLL 탐색을 반복하지 않습니다.
//   ROCm / Vulkan 빌드는 기존과 같이 NVML 을 쓰지 않습니다.
// =====================================================================
#[cfg(not(any(feature = "rocm", feature = "vulkan")))]
static NVML_HANDLE: once_cell::sync::OnceCell<Nvml> = once_cell::sync::OnceCell::new();
#[cfg(not(any(feature = "rocm", feature = "vulkan")))]
static NVML_RETRY_AFTER_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// 실제로 성공한 Nvml::init() 횟수. 전역 핸들이 재사용되는지 검증하는 데 씁니다.
static NVML_INIT_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 프로세스 전역 NVML 핸들. NVML 을 쓰지 않는 빌드이거나 초기화에 실패하면 None.
pub fn nvml_handle() -> Option<&'static Nvml> {
    #[cfg(any(feature = "rocm", feature = "vulkan"))]
    {
        None
    }
    #[cfg(not(any(feature = "rocm", feature = "vulkan")))]
    {
        use std::sync::atomic::Ordering;
        if let Some(n) = NVML_HANDLE.get() {
            return Some(n);
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if now_ms < NVML_RETRY_AFTER_MS.load(Ordering::Relaxed) {
            return None;
        }
        match NVML_HANDLE.get_or_try_init(|| {
            let r = Nvml::init();
            if r.is_ok() {
                NVML_INIT_COUNT.fetch_add(1, Ordering::SeqCst);
                println!("[NVML-GLOBAL] ✅ 프로세스 전역 NVML 핸들을 열었습니다. 이후 VRAM 질의는 init/shutdown 없이 재사용합니다.");
            }
            r
        }) {
            Ok(n) => Some(n),
            Err(e) => {
                NVML_RETRY_AFTER_MS.store(now_ms + 5_000, Ordering::Relaxed);
                println!("[NVML-GLOBAL] ⚠️ NVML 초기화 실패({}). 5초 뒤에 다시 시도합니다.", e);
                None
            }
        }
    }
}

/// 지금까지 성공한 Nvml::init() 횟수 (전역 핸들 재사용 검증용, 정상이면 0 또는 1).
pub fn nvml_init_count() -> u64 {
    NVML_INIT_COUNT.load(std::sync::atomic::Ordering::SeqCst)
}

pub struct GpuMemProbe {
    nvml: Option<&'static Nvml>,
}

impl GpuMemProbe {
    pub fn new() -> Self {
        Self { nvml: nvml_handle() }
    }

    pub fn device_count(&self) -> usize {
        #[cfg(feature = "rocm")]
        {
            let n = candle_rocm::device_count().unwrap_or(0);
            if n > 0 {
                return n;
            }
        }
        #[cfg(feature = "vulkan")]
        {
            let n = *VULKAN_DEVICE_COUNT;
            if n > 0 {
                return n;
            }
        }
        self.nvml
            .and_then(|n| n.device_count().ok())
            .unwrap_or(0) as usize
    }

    pub fn mem_info(&self, gpu_id: usize) -> Option<(u64, u64)> {
        // 🌟 [VRAM-SOURCE] 이미 만든 GPU 장치가 있으면 드라이버 값(+ 메모리 풀의 유휴분)을 씁니다.
        if let Some(info) = device_mem_info(gpu_id) {
            return Some(info);
        }
        #[cfg(feature = "rocm")]
        if let Ok((free, total)) = candle_rocm::mem_info(gpu_id) {
            return Some((clamp_free_to_os_budget(gpu_id, true, free as u64), total as u64));
        }
        #[cfg(feature = "vulkan")]
        if gpu_id == 0 && *VULKAN_DEVICE_COUNT > 0 {
            if let Device::Vulkan(d) = get_gpu_device(gpu_id) {
                if let Ok((free, total)) = d.mem_info() {
                    return Some((free as u64, total as u64));
                }
            }
        }
        let mem = self
            .nvml?
            .device_by_index(gpu_id as u32)
            .ok()?
            .memory_info()
            .ok()?;
        Some((mem.free, mem.total))
    }

    pub fn free_bytes(&self, gpu_id: usize) -> Option<u64> {
        self.mem_info(gpu_id).map(|(free, _)| free)
    }
}

pub fn gpu_count() -> usize {
    GpuMemProbe::new().device_count()
}

pub fn gpu_mem_info(gpu_id: usize) -> Option<(u64, u64)> {
    GpuMemProbe::new().mem_info(gpu_id)
}

// =====================================================================
// 🌟 [VRAM-SOURCE] VRAM 숫자는 장치(드라이버)에서 직접 읽습니다.
// ---------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   NVML free 는 메모리 풀이 쥐고 있는(바로 다시 쓸 수 있는) 블록까지 '사용 중' 으로 셉니다.
//   그래서 동기화 직후인지 아닌지에 따라 같은 상태가 수백 MB 씩 다르게 보였고,
//   KV-PLAN · 상주 판정 · 공존 판정이 호출마다 뒤집혔습니다.
//  ── 해결 ──
//   이미 만든 GPU 장치(DEVICE_CACHE)가 있으면 Device::mem_info (CUDA/ROCm/Vulkan 공통, NVML 재초기화 없음)
//   와 Device::effective_free_memory (= free + 풀 유휴분) 를 씁니다. 장치가 없을 때만 NVML 로 돌아갑니다.
//   장치를 새로 만들지는 않습니다(조회가 CUDA 컨텍스트를 만들면 안 되므로).
//   `LOGIS_VRAM_SOURCE=nvml` 이면 예전처럼 항상 NVML 을 씁니다.
// =====================================================================

/// `LOGIS_VRAM_SOURCE=nvml` 이면 예전처럼 NVML 값만 씁니다.
pub fn vram_source_is_nvml() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LOGIS_VRAM_SOURCE")
            .map(|v| v.trim().eq_ignore_ascii_case("nvml"))
            .unwrap_or(false)
    })
}

/// 이미 만들어 둔 GPU 장치 (없으면 None — 새로 만들지 않습니다).
pub fn cached_gpu_device(id: usize) -> Option<Device> {
    let cache = DEVICE_CACHE.lock().ok()?;
    cache.get(id).and_then(|d| d.clone()).filter(|d| !d.is_cpu())
}

/// 장치에서 읽은 `(실효 free, total)` 바이트. 실효 free = 드라이버 free + 메모리 풀 유휴분.
fn device_mem_info(gpu_id: usize) -> Option<(u64, u64)> {
    if vram_source_is_nvml() {
        return None;
    }
    let dev = cached_gpu_device(gpu_id)?;
    device_mem_info_of(&dev)
}

/// 주어진 장치의 `(실효 free, total)` 바이트. CPU 이거나 조회가 실패하면 None.
/// ROCm(Windows) 은 드라이버 free 를 DXGI 예산으로 보정한 뒤 풀 유휴분을 더합니다([VRAM-WDDM]).
pub fn device_mem_info_of(dev: &Device) -> Option<(u64, u64)> {
    let (free, total) = dev.mem_info().ok().flatten()?;
    let gpu_id = match dev.location() {
        candle_core::DeviceLocation::Rocm { gpu_id } => gpu_id,
        _ => 0,
    };
    let free = clamp_free_to_os_budget(gpu_id, dev.is_rocm(), free as u64);
    let idle = dev.pool_idle_bytes().unwrap_or(0);
    Some((free.saturating_add(idle as u64), total as u64))
}

/// 실효 free VRAM(bytes). 장치가 있으면 장치 값, 없으면 NVML. GPU 가 없으면 None.
pub fn effective_free_vram_bytes(gpu_id: usize) -> Option<u64> {
    gpu_mem_info(gpu_id).map(|(free, _)| free)
}

/// 단계 사이(같은 작업 안) 퍼지에서 메모리 풀에 남겨 둘 양. 다음 단계가 드라이버를 거치지 않고 재사용합니다.
/// `LOGIS_PHASE_POOL_KEEP_MB` 로 바꿀 수 있습니다(기본 256MB, 0 이면 매번 전부 반환).
pub fn phase_pool_keep_bytes() -> usize {
    static KEEP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *KEEP.get_or_init(|| {
        std::env::var("LOGIS_PHASE_POOL_KEEP_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(256)
            << 20
    })
}

static DEFERRED_ERROR_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 지금까지 [`gpu_checkpoint`] 가 회수한 지연 오류 수.
pub fn deferred_error_count() -> u64 {
    DEFERRED_ERROR_COUNT.load(std::sync::atomic::Ordering::SeqCst)
}

/// 🌟 [CUDA-DEFERRED] 드라이버가 Drop 등에서 돌려주지 못하고 쌓아 둔 오류를 꺼내 기록합니다.
/// 꺼낸 오류는 사라지므로, 다음 동기화가 엉뚱한 곳에서 실패하지 않습니다.
pub fn gpu_checkpoint(dev: &Device, label: &str) -> Option<String> {
    let err = dev.take_deferred_error()?;
    DEFERRED_ERROR_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    println!("[CUDA-DEFERRED] ⚠️ {} | 앞선 작업이 남긴 지연 오류를 회수했습니다: {}", label, err);
    Some(err)
}

/// 🌟 [POOL-TRIM] 장치를 현재 스레드에 묶고 → 지연 오류 회수 → 모든 스트림 동기화 → 풀을 `keep_bytes` 까지 반환.
/// `release_ws` 이면 양자화 matmul 작업 버퍼까지 반환합니다(작업 종료 시). 블로킹 함수입니다.
pub fn settle_gpu_memory(dev: &Device, keep_bytes: usize, release_ws: bool, label: &str) -> candle_core::Result<()> {
    if dev.is_cpu() {
        return Ok(());
    }
    let _ = dev.bind_to_thread();
    gpu_checkpoint(dev, label);
    let r = dev.synchronize_all();
    gpu_checkpoint(dev, label);
    if let Err(e) = &r {
        println!("[POOL-TRIM] ⚠️ {} | 장치 동기화 실패: {:?}", label, e);
    }
    if release_ws {
        if let Err(e) = dev.release_workspaces() {
            println!("[POOL-TRIM] ⚠️ {} | 작업 버퍼 반환 실패: {:?}", label, e);
        }
    }
    let t = dev.trim_memory_pool(keep_bytes);
    if let Err(e) = &t {
        println!("[POOL-TRIM] ⚠️ {} | 메모리 풀 반환 실패: {:?}", label, e);
    }
    r.and(t)
}

/// 메모리 풀을 드라이버로 돌려줍니다 (예전처럼 장치를 새로 만들었다 버리지 않습니다).
pub fn flush_gpu_memory_pool(gpu_id: usize) {
    flush_gpu_memory_pool_keep(gpu_id, 0);
}

/// [`flush_gpu_memory_pool`] 과 같되 풀에 `keep_bytes` 를 남깁니다.
pub fn flush_gpu_memory_pool_keep(gpu_id: usize, keep_bytes: usize) {
    if let Some(dev) = cached_gpu_device(gpu_id) {
        let _ = settle_gpu_memory(&dev, keep_bytes, false, "flush_gpu_memory_pool");
    }
}

/// 🌟 [HEAP-RELEASE] mimalloc 이 쥐고 있는 빈 힙 페이지를 OS 로 돌려줍니다.
/// 작업 집합 전체를 내쫓는 SetProcessWorkingSetSizeEx 와 달리 쓰던 페이지(mmap 가중치·토크나이저)는 건드리지 않습니다.
pub fn release_heap() {
    extern "C" {
        fn mi_collect(force: bool);
    }
    // SAFETY: mimalloc 은 전역 할당기(main.rs)이고 libmimalloc-sys 가 링크하는 C 함수입니다.
    unsafe { mi_collect(true) }
}

/// `LOGIS_MEM_TRIM=legacy` 이면 예전 위치들(퍼지·정착 대기·생성 루프)에서 작업 집합 전체 반환을 다시 수행합니다.
pub fn legacy_ws_trim_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LOGIS_MEM_TRIM")
            .map(|v| v.trim().eq_ignore_ascii_case("legacy"))
            .unwrap_or(false)
    })
}

/// 프로세스 작업 집합 전체 반환 (SetProcessWorkingSetSizeEx(-1,-1)). 무조건 수행합니다.
pub fn force_working_set_trim() {
    #[cfg(target_os = "windows")]
    unsafe {
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
        let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
    }
    #[cfg(target_os = "linux")]
    unsafe {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        malloc_trim(0);
    }
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
}

/// 🌟 [RAM-SAWTOOTH] 반복 경로(퍼지·생성 루프·정착 대기)의 작업 집합 반환 자리.
/// 기본은 아무것도 하지 않습니다. 작업 집합을 통째로 내쫓으면 곧바로 다시 읽어 오는 페이지 폴트가
/// 이어져 RAM 사용량이 톱니처럼 오르내리고 수 초씩 멈췄습니다. `LOGIS_MEM_TRIM=legacy` 이면 예전처럼 수행합니다.
pub fn legacy_working_set_trim() {
    if legacy_ws_trim_enabled() {
        force_working_set_trim();
    }
}

pub fn trim_idle_gpu_pool(device: &Device) {
    match device {
        #[cfg(feature = "vulkan")]
        Device::Vulkan(d) => {
            if d.pooled_bytes() >= (64 << 20) {
                let _ = d.trim_memory_pool();
            }
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(d) => {
            let _ = d.trim_memory_pool();
        }
        _ => {
            let _ = device;
        }
    }
}

pub fn get_gpu_device(id: usize) -> Device {
    {
        let cache = DEVICE_CACHE.lock().unwrap();
        if id < cache.len() {
            if let Some(dev) = &cache[id] {
                return dev.clone();
            }
        }
    }

    #[cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan"))]
    let dev = {
        println!("[CUDA/ROCm/Vulkan] 🚀 Attempting to Create Primary Context on GPU {}...", id);
        let d = new_gpu_device(id).unwrap_or(Device::Cpu);
        println!("[CUDA/ROCm/Vulkan] ✅ Primary Context Created on GPU {} ({:?}).", id, d);
        d
    };

    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), feature = "metal"))]
    let dev = {
        println!("[Metal] 🚀 Initializing Metal Context on GPU {}...", id);
        Device::new_metal(id).unwrap_or(Device::Cpu)
    };

    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), not(feature = "metal")))]
    let dev = Device::Cpu;

    {
        let mut cache = DEVICE_CACHE.lock().unwrap();
        if id < cache.len() {
            cache[id] = Some(dev.clone());
        }
    }
    dev
}

pub fn get_cuda_device(id: usize) -> Device {
    get_gpu_device(id)
}

pub fn get_best_device_info() -> (Device, usize) {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        let probe = GpuMemProbe::new();
        let count = probe.device_count();
        if count > 0 {
            let mut best_id = 0;
            let mut max_free = 0;

            println!("[GPU-CHECK] Found {} CUDA/ROCm device(s).", count);

            for i in 0..count {
                if let Some((free, _)) = probe.mem_info(i) {
                    let free_gb = free as f64 / 1e9;
                    println!("[GPU-CHECK] GPU {}: {:.2} GB Free VRAM", i, free_gb);
                    if free > max_free {
                        max_free = free;
                        best_id = i;
                    }
                }
            }

            if max_free > 0 {
                println!("[GPU-CHECK] Selecting GPU {} as the best device.", best_id);
                return (get_gpu_device(best_id), best_id);
            }
        }
        println!("[GPU-CHECK] VRAM query failed or no free VRAM. Defaulting to GPU 0.");
        return (get_gpu_device(0), 0);
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "vulkan"))]
    {
        let count = GpuMemProbe::new().device_count();
        if count == 0 {
            println!("[GPU-CHECK] No Vulkan device found. Falling back to CPU.");
            return (Device::Cpu, 0);
        }
        println!("[GPU-CHECK] Found {} Vulkan device(s). Selecting GPU 0.", count);
        return (get_gpu_device(0), 0);
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), feature = "metal"))]
    {
        return (get_gpu_device(0), 0);
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "vulkan"), not(feature = "metal")))]
    {
        (Device::Cpu, 0)
    }
}

pub fn get_best_device() -> Device {
    get_best_device_info().0
}

pub fn get_device(device: Option<&Device>) -> Device {
    match device {
        Some(d) => d.clone(),
        None => get_best_device()
    }
}

pub fn get_gpu_sm_arch() -> Result<f32> {
    #[cfg(feature = "cuda")]
    {
        let output = Command::new("nvidia-smi")
            .arg("--query-gpu=compute_cap")
            .arg("--format=csv,noheader")
            .output()
            .map_err(|e| anyhow::anyhow!(format!("Failed to execute nvidia-smi: {}", e)))?;
        if !output.status.success() {
            return Err(anyhow::anyhow!(format!(
                "nvidia-smi failed with status: {}
Error: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        let output_str = String::from_utf8_lossy(&output.stdout);
        let output_str = output_str.trim();
        
        let first_line = output_str.lines().next().unwrap_or("0.0");
        let sm_float = match first_line.parse::<f32>() {
            Ok(num) => num,
            Err(_) => {
                return Err(anyhow::anyhow!(format!(
                    "gpu sm arch: {} parse float32 error",
                    first_line
                )));
            }
        };
        Ok(sm_float)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(0.0)
    }
}

#[derive(Clone, Debug)]
pub struct DeviceConfig {
    pub device: Device,
    pub is_cpu: bool,
    pub classify_chunk_size: usize,
    pub extract_chunk_size: usize,
    pub name: String,
    pub gpu_id: usize,
}

pub fn get_optimal_device_config() -> DeviceConfig {
    let (device, gpu_id) = get_best_device_info();
    let is_cpu = device.is_cpu();
    
    let name = if device.is_cuda() {
        format!("CUDA (GPU {})", gpu_id)
    } else if device.is_rocm() {
        format!("ROCm (GPU {})", gpu_id)
    } else if device.is_vulkan() {
        let vk_name = device
            .as_vulkan_device()
            .map(|d| d.name().to_string())
            .unwrap_or_default();
        format!("Vulkan (GPU {}: {})", gpu_id, vk_name)
    } else if device.is_metal() {
        "Metal".to_string()
    } else {
        "CPU".to_string()
    };

    DeviceConfig {
        device,
        is_cpu,
        classify_chunk_size: 12_000, 
        extract_chunk_size: 12_000,
        name,
        gpu_id,
    }
}

pub fn get_dtype(dtype: Option<DType>, cfg_dtype: &str) -> DType {
    match dtype {
        Some(d) => d,
        None => {
            let is_cuda = cfg!(feature = "cuda");
            let is_rocm = cfg!(feature = "rocm");
            let is_metal = cfg!(feature = "metal");
            let is_vulkan = cfg!(feature = "vulkan");

            if is_vulkan && get_best_device().is_vulkan() {
                match cfg_dtype {
                    "float64" | "double" => DType::F64,
                    "uint8" => DType::U8,
                    "int8" | "int16" | "int32" | "int64" => DType::I64,
                    _ => DType::F32,
                }
            } else if (is_cuda || is_rocm || is_metal) && !get_best_device().is_cpu() {
                match cfg_dtype {
                    "float32" | "float" => DType::F32,
                    "float64" | "double" => DType::F64,
                    "float16" => DType::F16,
                    "bfloat16" => {
                        if is_cuda {
                            let arch = get_gpu_sm_arch();
                            match arch {
                                Err(_) => DType::F16,
                                Ok(a) => if a >= 8.0 { DType::BF16 } else { DType::F16 }
                            }
                        } else if is_rocm {
                            DType::BF16
                        } else {
                            DType::F16
                        }
                    }
                    "uint8" => DType::U8,
                    "int8" | "int16" | "int32" | "int64" => DType::I64,
                    _ => DType::F32,
                }
            } else {
                match cfg_dtype {
                    "float32" | "float" => DType::F32,
                    "float64" | "double" => DType::F64,
                    "float16" | "bfloat16" => DType::F16,
                    "uint8" => DType::U8,
                    "int8" | "int16" | "int32" | "int64" => DType::I64,
                    _ => DType::F32,
                }
            }
        }
    }
}