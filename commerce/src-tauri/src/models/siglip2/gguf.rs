//! 🌟 [SPLIT-GGUF] SigLIP2 를 타워별 GGUF 로 나눠 읽습니다.
//!
//! 내려받은 `model.safetensors` (F32, 4.5 GB) 에는 비전 · 텍스트 타워가 함께 있고, 장치에는 BF16 으로 올립니다.
//! 올릴 때마다 F32 를 BF16 으로 바꾸는 대신 처음 한 번 타워별 BF16 GGUF 를 만들어 두고, 그 뒤로는 매핑에서 장치로 바로 복사합니다.
//! Qwen3.5 의 mmproj 와 같은 구조이고, 변환은 `model::lang_gguf` 처럼 준비 표식(READY)과 임시 파일(.part)을 씁니다.
//!
//!   runtime-bf16/mmproj-siglip2-bf16.gguf   비전 타워 + logit 스칼라
//!   runtime-bf16/siglip2-text-bf16.gguf     텍스트 타워 + logit 스칼라
//!
//!  ── 지키는 것 ──
//!   · GGUF 는 BF16 이라 장치 dtype 이 BF16 일 때만 씁니다. F32 로 쓰는 Vulkan · CPU 는 원본을 읽어야 벡터가 그대로입니다.
//!   · 토큰 임베딩 표(590 MB)는 장치에도 호스트에도 올리지 않고 쓰는 행만 매핑에서 읽습니다.
//!   · 변환이나 읽기가 실패해도 safetensors 경로로 그대로 동작합니다.
//!
//!  ── 모델 전환 ──
//!   문서 한 장을 읽는 동안 비전 타워 → 텍스트 타워 → Qwen3.5 로 장치를 넘기고, 다음 장에서 다시 비전 타워가 올라옵니다.
//!   · 타워를 내려도 파일 매핑은 남겨 둡니다 ([`open_cached`]). 다시 올릴 때 머리를 다시 읽거나 페이지 폴트를 기다리지 않고 복사만 합니다.
//!   · 쓰기 전에 타워의 페이지를 배경에서 읽어 둡니다 ([`announce`]). 장치에는 올리지 않으므로 상주량은 그대로입니다.
//!   · 매핑 중인 파일은 Windows 에서 지우거나 덮어쓰지 못하므로, 변환과 모델 삭제 전에 [`close_handles`] 로 닫습니다.

use std::collections::{HashMap, HashSet};
use std::io::{BufWriter, Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::gguf_file::{Content, TensorInfo};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use half::bf16;
use memmap2::Mmap;
use once_cell::sync::Lazy;
use rayon::prelude::*;

use crate::model::lang_gguf::{pad_to, put_meta, put_str, Meta, StFile, ALIGN};

const RUNTIME_DIR: &str = "runtime-bf16";
const VISION_FILE: &str = "mmproj-siglip2-bf16.gguf";
const TEXT_FILE: &str = "siglip2-text-bf16.gguf";
const READY_FILE: &str = "READY";
const RECIPE: &str = "siglip2-bf16/1";
/// 올리지 않고 행만 읽는 토큰 임베딩 표
pub const TOKEN_TABLE: &str = "text_model.embeddings.token_embedding.weight";
/// 두 파일 모두에 넣는 스칼라. 어느 타워만 올려도 값을 읽을 수 있습니다.
const SCALARS: [&str; 2] = ["logit_scale", "logit_bias"];
const GGML_BF16: u32 = 30;
/// F32 를 BF16 으로 바꿀 때 스레드 하나가 맡는 원소 수
const CONVERT_CHUNK: usize = 1 << 16;
/// 페이지를 건드리는 간격
const PAGE: usize = 4096;
/// 미리 읽기에서 한 범위로 합치는 틈의 상한
const WARM_GAP: usize = 1 << 20;

fn runtime_dir(model_dir: &Path) -> PathBuf {
    model_dir.join(RUNTIME_DIR)
}

pub fn vision_path(model_dir: &Path) -> PathBuf {
    runtime_dir(model_dir).join(VISION_FILE)
}

pub fn text_path(model_dir: &Path) -> PathBuf {
    runtime_dir(model_dir).join(TEXT_FILE)
}

fn recipe_key(model_dir: &Path) -> Result<String> {
    let src = std::fs::metadata(model_dir.join("model.safetensors"))
        .with_context(|| format!("{:?} 에 model.safetensors 가 없습니다", model_dir))?
        .len();
    Ok(format!("{}|src={}", RECIPE, src))
}

pub fn ready(model_dir: &Path) -> bool {
    let key = match recipe_key(model_dir) {
        Ok(k) => k,
        Err(_) => return false,
    };
    let mark = std::fs::read_to_string(runtime_dir(model_dir).join(READY_FILE)).unwrap_or_default();
    mark.trim() == key && vision_path(model_dir).exists() && text_path(model_dir).exists()
}

/// 이 dtype 으로 GGUF 를 읽어도 되는가
pub fn usable(model_dir: &Path, dtype: DType) -> bool {
    dtype == DType::BF16 && ready(model_dir)
}

static CONVERT_LOCK: Mutex<()> = Mutex::new(());

/// 타워별 GGUF 를 만듭니다. 이미 있으면 아무것도 하지 않습니다.
pub fn prepare(model_dir: &Path) -> Result<()> {
    if ready(model_dir) {
        return Ok(());
    }
    let _guard = CONVERT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if ready(model_dir) {
        return Ok(());
    }
    close_handles(Some(model_dir));
    let start = std::time::Instant::now();
    let dir = runtime_dir(model_dir);
    std::fs::create_dir_all(&dir)?;
    let _ = std::fs::remove_file(dir.join(READY_FILE));
    let st = StFile::open(&model_dir.join("model.safetensors"))?;
    let mut names: Vec<String> = st.entries.keys().cloned().collect();
    names.sort();
    let tower = |prefix: &str| -> Vec<String> {
        names.iter().filter(|n| n.starts_with(prefix) || SCALARS.contains(&n.as_str())).cloned().collect()
    };
    write_tower(&st, &tower("vision_model."), &vision_path(model_dir), "siglip2-mmproj-bf16")?;
    write_tower(&st, &tower("text_model."), &text_path(model_dir), "siglip2-text-bf16")?;
    std::fs::write(dir.join(READY_FILE), recipe_key(model_dir)?)?;
    println!("[SigLIP2-GGUF] 타워별 GGUF 를 만들었습니다 ({:.1} s).", start.elapsed().as_secs_f64());
    Ok(())
}

/// 저장된 바이트를 BF16 으로 바꿉니다.
fn bf16_bytes(name: &str, raw: &[u8], dtype: DType) -> Result<Vec<u8>> {
    match dtype {
        DType::BF16 => Ok(raw.to_vec()),
        DType::F32 => {
            let mut out = vec![0u8; raw.len() / 2];
            out.par_chunks_mut(CONVERT_CHUNK * 2).zip(raw.par_chunks(CONVERT_CHUNK * 4)).for_each(|(o, r)| {
                for (o, r) in o.chunks_exact_mut(2).zip(r.chunks_exact(4)) {
                    let v = f32::from_le_bytes([r[0], r[1], r[2], r[3]]);
                    o.copy_from_slice(&bf16::from_f32(v).to_le_bytes());
                }
            });
            Ok(out)
        }
        other => bail!("{} 의 dtype {:?} 는 지원하지 않습니다", name, other),
    }
}

fn write_tower(st: &StFile, names: &[String], out: &Path, label: &str) -> Result<()> {
    let meta: Vec<(&str, Meta)> = vec![
        ("general.architecture", Meta::Str("siglip2".into())),
        ("general.name", Meta::Str(label.into())),
        ("general.alignment", Meta::U32(ALIGN as u32)),
        ("logis.recipe", Meta::Str(RECIPE.into())),
    ];
    let mut head: Vec<u8> = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(names.len() as u64).to_le_bytes());
    head.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    for (k, v) in meta.iter() {
        put_meta(&mut head, k, v)?;
    }
    let mut offset = 0u64;
    for name in names {
        let e = st.entries.get(name).ok_or_else(|| anyhow!("model.safetensors 에 {} 가 없습니다", name))?;
        put_str(&mut head, name)?;
        head.extend_from_slice(&(e.shape.len() as u32).to_le_bytes());
        for d in e.shape.iter().rev() {
            head.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        head.extend_from_slice(&GGML_BF16.to_le_bytes());
        head.extend_from_slice(&offset.to_le_bytes());
        let size = e.shape.iter().product::<usize>() as u64 * 2;
        offset = (offset + size).div_ceil(ALIGN) * ALIGN;
    }

    let part = out.with_extension("gguf.part");
    let result = (|| -> Result<()> {
        let file = std::fs::File::create(&part).with_context(|| format!("{:?} 를 만들 수 없습니다", part))?;
        let mut w = BufWriter::with_capacity(8 << 20, file);
        w.write_all(&head)?;
        pad_to(&mut w, head.len() as u64)?;
        for name in names {
            let e = &st.entries[name];
            let bytes = bf16_bytes(name, st.bytes(name)?, e.dtype)?;
            w.write_all(&bytes)?;
            pad_to(&mut w, bytes.len() as u64)?;
        }
        let file = w.into_inner().map_err(|e| anyhow!("{}", e))?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }
    if out.exists() {
        std::fs::remove_file(out)?;
    }
    std::fs::rename(&part, out)?;
    Ok(())
}

/// 프로세스에서 이미 미리 읽은 파일. 읽어 둔 파일을 다시 요청하면 오히려 시간이 듭니다.
static WARMED: Lazy<Mutex<HashSet<PathBuf>>> = Lazy::new(|| Mutex::new(HashSet::new()));
/// 열어 둔 타워 파일. 타워를 내려도 남겨 둡니다.
static HANDLES: Lazy<Mutex<HashMap<PathBuf, Arc<MappedGguf>>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// 타워 파일을 엽니다. 이미 열려 있으면 그 매핑을 씁니다.
pub fn open_cached(path: &Path) -> Result<Arc<MappedGguf>> {
    let mut handles = HANDLES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(file) = handles.get(path) {
        return Ok(Arc::clone(file));
    }
    let file = Arc::new(MappedGguf::open(path)?);
    handles.insert(path.to_path_buf(), Arc::clone(&file));
    Ok(file)
}

/// 열어 둔 매핑을 닫습니다. `model_dir` 이 없으면 전부 닫습니다. 부착된 텍스트 타워가 쥔 매핑은 타워를 내릴 때 닫힙니다.
pub fn close_handles(model_dir: Option<&Path>) {
    let inside = |path: &Path| model_dir.is_none_or(|dir| path.starts_with(runtime_dir(dir)));
    HANDLES.lock().unwrap_or_else(|e| e.into_inner()).retain(|path, _| !inside(path));
    WARMED.lock().unwrap_or_else(|e| e.into_inner()).retain(|path| !inside(path));
}

/// 곧 쓸 두 타워의 페이지를 배경에서 읽어 둡니다. 장치에는 아무것도 올리지 않습니다.
pub fn announce(model_dir: &Path) {
    let (vision, text) = (open_cached(&vision_path(model_dir)), open_cached(&text_path(model_dir)));
    match (vision, text) {
        (Ok(vision), Ok(text)) => {
            vision.warm(&[]);
            text.warm(&[TOKEN_TABLE]);
        }
        (Err(e), _) | (_, Err(e)) => println!("[SigLIP2-GGUF] 타워 파일을 미리 읽지 못했습니다: {}", e),
    }
}

/// 범위를 미리 읽어 두라고 OS 에 알립니다. 파일이 OS 캐시에 없을 때 페이지 폴트마다 디스크를 기다리지 않게 합니다.
fn prefetch(map: &Mmap, start: usize, len: usize) {
    #[cfg(target_os = "windows")]
    unsafe {
        use windows_sys::Win32::System::Memory::{PrefetchVirtualMemory, WIN32_MEMORY_RANGE_ENTRY};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let entry = WIN32_MEMORY_RANGE_ENTRY { VirtualAddress: map.as_ptr().add(start) as *mut std::ffi::c_void, NumberOfBytes: len };
        let _ = PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
    }
    #[cfg(unix)]
    {
        let _ = map.advise_range(memmap2::Advice::WillNeed, start, len);
    }
    #[cfg(not(any(target_os = "windows", unix)))]
    let _ = (map, start, len);
}

/// 매핑해 둔 타워 GGUF
pub struct MappedGguf {
    path: PathBuf,
    map: Arc<Mmap>,
    content: Content,
}

impl MappedGguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("{:?} 를 열 수 없습니다", path))?;
        let map = unsafe { Mmap::map(&file) }.with_context(|| format!("{:?} 를 매핑할 수 없습니다", path))?;
        let content = Content::read(&mut Cursor::new(&map[..])).map_err(|e| anyhow!("{:?} 의 머리를 읽을 수 없습니다: {}", path, e))?;
        Ok(Self { path: path.to_path_buf(), map: Arc::new(map), content })
    }

    fn range(&self, name: &str, info: &TensorInfo) -> Result<std::ops::Range<usize>> {
        if info.ggml_dtype != GgmlDType::BF16 {
            bail!("{} 는 BF16 이어야 합니다: {:?}", name, info.ggml_dtype);
        }
        let start = (self.content.tensor_data_offset + info.offset) as usize;
        let end = start + info.shape.elem_count() * 2;
        if end > self.map.len() {
            bail!("{:?} 가 잘렸습니다 ({})", self.path, name);
        }
        Ok(start..end)
    }

    fn elements(&self, range: std::ops::Range<usize>) -> Result<&[bf16]> {
        let (head, body, tail) = unsafe { self.map[range].align_to::<bf16>() };
        if !head.is_empty() || !tail.is_empty() {
            bail!("{:?} 의 텐서 위치가 원소 경계에 맞지 않습니다", self.path);
        }
        Ok(body)
    }

    /// 이 파일의 페이지를 배경에서 읽어 둡니다. `skip` 의 텐서는 읽지 않습니다. 파일마다 프로세스에서 한 번만 합니다.
    ///
    /// OS 에 미리 읽기를 요청한 뒤 페이지를 하나씩 건드려 이 프로세스에 매핑합니다. Windows 의 미리 읽기는 페이지를 메모리로 가져올 뿐이라,
    /// 처음 읽을 때의 소프트 폴트가 남아 있기 때문입니다.
    pub fn warm(self: &Arc<Self>, skip: &'static [&'static str]) {
        if !WARMED.lock().unwrap_or_else(|e| e.into_inner()).insert(self.path.clone()) {
            return;
        }
        let file = Arc::clone(self);
        let started = std::thread::Builder::new().name("siglip2-warm".into()).spawn(move || {
            let mut spans: Vec<(usize, usize)> = file
                .content
                .tensor_infos
                .iter()
                .filter(|(name, _)| !skip.contains(&name.as_str()))
                .filter_map(|(name, info)| file.range(name, info).ok())
                .map(|r| (r.start, r.end))
                .collect();
            spans.sort_unstable();
            let mut merged: Vec<(usize, usize)> = Vec::new();
            for (start, end) in spans {
                match merged.last_mut() {
                    Some(last) if start <= last.1 + WARM_GAP => last.1 = last.1.max(end),
                    _ => merged.push((start, end)),
                }
            }
            for (start, end) in merged {
                prefetch(&file.map, start, end - start);
                let mut sum = 0u8;
                for at in (start..end).step_by(PAGE) {
                    // 읽기가 사라지지 않게 합니다. 범위는 `range` 가 매핑 안임을 확인했습니다.
                    sum = sum.wrapping_add(unsafe { std::ptr::read_volatile(file.map.as_ptr().add(at)) });
                }
                std::hint::black_box(sum);
            }
        });
        if let Err(e) = started {
            println!("[SigLIP2-GGUF] 미리 읽기를 시작하지 못했습니다: {}", e);
        }
    }

    /// 텐서를 매핑에서 장치로 바로 복사해 VarBuilder 로 돌려줍니다. `skip` 의 텐서는 올리지 않습니다.
    pub fn var_builder(&self, skip: &[&str], device: &Device) -> Result<VarBuilder<'static>> {
        let names: Vec<&String> = self.content.tensor_infos.keys().filter(|n| !skip.contains(&n.as_str())).collect();
        let mut spans = Vec::with_capacity(names.len());
        for name in &names {
            spans.push(self.range(name, &self.content.tensor_infos[*name])?);
        }
        let mut tensors = HashMap::with_capacity(names.len());
        for (name, range) in names.iter().zip(spans) {
            let dims = self.content.tensor_infos[*name].shape.dims().to_vec();
            tensors.insert((*name).clone(), Tensor::from_slice(self.elements(range)?, dims, device)?);
        }
        Ok(VarBuilder::from_tensors(tensors, DType::BF16, device))
    }

    /// 2차원 표의 행을 매핑에서 읽는 핸들
    pub fn rows(&self, name: &str) -> Result<MappedRows> {
        let info = self.content.tensor_infos.get(name).ok_or_else(|| anyhow!("{:?} 에 {} 가 없습니다", self.path, name))?;
        let (rows, cols) = info.shape.dims2()?;
        let range = self.range(name, info)?;
        Ok(MappedRows { map: Arc::clone(&self.map), start: range.start, rows, cols })
    }
}

/// 표를 올리지 않고 쓰는 행만 읽습니다.
pub struct MappedRows {
    map: Arc<Mmap>,
    start: usize,
    rows: usize,
    cols: usize,
}

impl MappedRows {
    /// `ids` 모양 뒤에 열 차원을 붙인 BF16 텐서를 `device` 에 만듭니다.
    pub fn gather(&self, ids: &Tensor, device: &Device) -> candle_core::Result<Tensor> {
        let flat = ids.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
        let mut out: Vec<bf16> = Vec::with_capacity(flat.len() * self.cols);
        for id in flat {
            let id = id as usize;
            if id >= self.rows {
                candle_core::bail!("토큰 {} 이 표 크기 {} 를 넘습니다", id, self.rows);
            }
            let start = self.start + id * self.cols * 2;
            let row = &self.map[start..start + self.cols * 2];
            out.extend(row.chunks_exact(2).map(|b| bf16::from_le_bytes([b[0], b[1]])));
        }
        let mut dims = ids.dims().to_vec();
        dims.push(self.cols);
        Tensor::from_vec(out, dims, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("logis-siglip2-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample(rows: usize, cols: usize, seed: f32) -> Tensor {
        let v: Vec<f32> = (0..rows * cols).map(|i| ((i as f32 + seed) * 0.37).sin() * 3.0).collect();
        Tensor::from_vec(v, (rows, cols), &Device::Cpu).unwrap()
    }

    fn write_source(dir: &Path) -> HashMap<String, Tensor> {
        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("vision_model.head.probe".into(), sample(2, 8, 1.0));
        t.insert("vision_model.post_layernorm.bias".into(), sample(1, 8, 2.0).squeeze(0).unwrap());
        t.insert(TOKEN_TABLE.into(), sample(16, 8, 3.0));
        t.insert("text_model.head.weight".into(), sample(8, 8, 4.0));
        t.insert("logit_scale".into(), Tensor::new(&[4.5f32], &Device::Cpu).unwrap());
        t.insert("logit_bias".into(), Tensor::new(&[-3.25f32], &Device::Cpu).unwrap());
        candle_core::safetensors::save(&t, dir.join("model.safetensors")).unwrap();
        t
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn each_tower_gets_its_own_file_and_the_scalars_go_to_both() -> Result<()> {
        let dir = temp_dir("split");
        write_source(&dir);
        assert!(!ready(&dir));
        prepare(&dir)?;
        assert!(ready(&dir));
        assert!(usable(&dir, DType::BF16));
        assert!(!usable(&dir, DType::F32), "F32 장치는 원본을 읽습니다");

        let names = |p: PathBuf| -> Result<Vec<String>> {
            let f = MappedGguf::open(&p)?;
            let mut n: Vec<String> = f.content.tensor_infos.keys().cloned().collect();
            n.sort();
            Ok(n)
        };
        assert_eq!(names(vision_path(&dir))?, ["logit_bias", "logit_scale", "vision_model.head.probe", "vision_model.post_layernorm.bias"]);
        assert_eq!(names(text_path(&dir))?, ["logit_bias", "logit_scale", "text_model.embeddings.token_embedding.weight", "text_model.head.weight"]);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn values_match_the_source_rounded_to_bf16() -> Result<()> {
        let dir = temp_dir("values");
        let source = write_source(&dir);
        prepare(&dir)?;
        let vision = MappedGguf::open(&vision_path(&dir))?;
        let vb = vision.var_builder(&[], &Device::Cpu)?;
        let got = vb.get((2, 8), "vision_model.head.probe")?;
        assert_eq!(got.dtype(), DType::BF16);
        assert_eq!(bits(&got), bits(&source["vision_model.head.probe"].to_dtype(DType::BF16)?));
        assert_eq!(vb.get(1, "logit_bias")?.to_dtype(DType::F32)?.to_vec1::<f32>()?, vec![-3.25]);
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn mapped_rows_equal_the_rows_of_the_whole_table() -> Result<()> {
        let dir = temp_dir("rows");
        let source = write_source(&dir);
        prepare(&dir)?;
        let text = MappedGguf::open(&text_path(&dir))?;
        let vb = text.var_builder(&[TOKEN_TABLE], &Device::Cpu)?;
        assert!(vb.get((16, 8), TOKEN_TABLE).is_err(), "표는 올리지 않습니다");
        let rows = text.rows(TOKEN_TABLE)?;
        let ids = Tensor::new(&[[3u32, 0, 15], [7, 7, 1]], &Device::Cpu)?;
        let got = rows.gather(&ids, &Device::Cpu)?;
        assert_eq!(got.dims(), &[2, 3, 8]);
        let table = source[TOKEN_TABLE].to_dtype(DType::BF16)?;
        let want = table.index_select(&ids.flatten_all()?, 0)?.reshape((2, 3, 8))?;
        assert_eq!(bits(&got), bits(&want));
        assert!(rows.gather(&Tensor::new(&[16u32], &Device::Cpu)?, &Device::Cpu).is_err(), "범위 밖 토큰");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn a_new_source_file_makes_the_conversion_stale() -> Result<()> {
        let dir = temp_dir("stale");
        write_source(&dir);
        prepare(&dir)?;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("vision_model.head.probe".into(), sample(4, 8, 5.0));
        candle_core::safetensors::save(&t, dir.join("model.safetensors"))?;
        assert!(!ready(&dir), "원본 크기가 달라지면 다시 만들어야 합니다");
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn a_tower_stays_mapped_until_its_handles_are_closed() -> Result<()> {
        let dir = temp_dir("handles");
        write_source(&dir);
        prepare(&dir)?;
        let path = vision_path(&dir);
        let first = open_cached(&path)?;
        assert!(Arc::ptr_eq(&first, &open_cached(&path)?), "다시 올릴 때는 같은 매핑을 씁니다");
        drop(first);
        assert!(HANDLES.lock().unwrap().contains_key(&path), "타워를 내려도 매핑은 남습니다");
        close_handles(Some(&dir));
        assert!(!HANDLES.lock().unwrap().contains_key(&path));
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn a_conversion_can_replace_files_that_were_mapped() -> Result<()> {
        let dir = temp_dir("remap");
        write_source(&dir);
        prepare(&dir)?;
        announce(&dir);
        let before = open_cached(&vision_path(&dir))?;
        assert_eq!(before.content.tensor_infos.len(), 4);
        drop(before);
        let mut t = write_source(&dir);
        t.insert("vision_model.extra".into(), sample(8, 8, 6.0));
        candle_core::safetensors::save(&t, dir.join("model.safetensors"))?;
        // 배경 미리 읽기가 끝나기를 기다립니다. 그 스레드가 매핑을 쥐고 있을 수 있습니다.
        std::thread::sleep(std::time::Duration::from_millis(200));
        prepare(&dir)?;
        let after = open_cached(&vision_path(&dir))?;
        assert_eq!(after.content.tensor_infos.len(), 5, "새 파일을 읽습니다");
        drop(after);
        close_handles(Some(&dir));
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// 실제 모델 비교입니다. `LOGIS_SIGLIP2_DIR` 에 모델 폴더를 주고 `--ignored` 로 실행합니다. 원본은 하드 링크로만 씁니다.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore]
    fn real_towers_read_the_same_from_gguf_and_from_safetensors() -> Result<()> {
        use crate::models::siglip2::{vision_encoder as ve, Siglip2Config, Siglip2Model};
        use std::time::Instant;

        let source = PathBuf::from(std::env::var("LOGIS_SIGLIP2_DIR")?);
        let dir = temp_dir("real");
        for name in ["model.safetensors", "config.json", "tokenizer.json"] {
            std::fs::hard_link(source.join(name), dir.join(name))?;
        }
        let config = Siglip2Config::from_json(&dir.join("config.json"))?;
        let device = Device::new_rocm(0)?;
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(640, 480, |x, y| image::Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8])));
        let phrases = vec!["logis gguf 비교 문구 하나".to_string(), "logis gguf comparison phrase two".to_string()];
        let read = |label: &str| -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
            let t = Instant::now();
            let mut model = Siglip2Model::load_vision_only(&dir.join("model.safetensors"), &config, &device, DType::BF16)?;
            let vision_ms = t.elapsed().as_millis();
            let grid = ve::encode_image_and_release(&mut model, &image)?;
            let t = Instant::now();
            model.load_text_encoder(&dir)?;
            let text_ms = t.elapsed().as_millis();
            let text = ve::encode_phrases_ephemeral(&model, &phrases)?;
            model.detach_text();
            let t = Instant::now();
            model.load_vision_encoder(&dir)?;
            println!("[{label}] 비전 {vision_ms} ms · 텍스트 {text_ms} ms · 비전 재부착 {} ms", t.elapsed().as_millis());
            Ok((text, grid.patches))
        };
        let max_diff = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);

        let plain = read("safetensors")?;
        assert!(!ready(&dir));
        prepare(&dir)?;
        let first = read("gguf 첫 번째")?;
        let second = read("gguf 두 번째")?;
        for (label, got) in [("첫 번째", &first), ("두 번째", &second)] {
            let (text, patches) = (max_diff(&plain.0, &got.0), max_diff(&plain.1, &got.1));
            println!("{label}: 텍스트 벡터 최대 차이 {text:e} · 패치 벡터 최대 차이 {patches:e}");
            assert_eq!(plain.1.len(), got.1.len());
            assert!(text < 1e-3 && patches < 1e-3, "{label}: {text} / {patches}");
        }
        close_handles(Some(&dir));
        std::thread::sleep(std::time::Duration::from_millis(500));
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn warming_leaves_the_values_alone() -> Result<()> {
        let dir = temp_dir("warm");
        let source = write_source(&dir);
        prepare(&dir)?;
        let vision = open_cached(&vision_path(&dir))?;
        vision.warm(&[]);
        vision.warm(&[]);
        let got = vision.var_builder(&[], &Device::Cpu)?.get((2, 8), "vision_model.head.probe")?;
        assert_eq!(bits(&got), bits(&source["vision_model.head.probe"].to_dtype(DType::BF16)?));
        drop(vision);
        close_handles(Some(&dir));
        std::thread::sleep(std::time::Duration::from_millis(200));
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
