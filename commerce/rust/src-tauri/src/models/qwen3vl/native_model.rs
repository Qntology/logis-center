use std::sync::Arc;
use memmap2::Mmap;
use crate::models::qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig};
use crate::models::qwen3vl::native_backend::*;
use half::f16;
use safetensors::SafeTensors;
use anyhow::{Result, anyhow};

pub enum LinearVariant {
    Standard { weight: NativeTensor, bias: Option<NativeTensor> },
    BitSerial { weight_packed: NativeTensor, scales: NativeTensor, bias: Option<NativeTensor> },
}

pub struct NativeLinear {
    pub in_features: usize, pub out_features: usize, pub variant: LinearVariant, pub device_id: i32,
}

unsafe impl Send for NativeLinear {}
unsafe impl Sync for NativeLinear {}

impl NativeLinear {
    pub fn forward(&self, x: &[f16]) -> Vec<f16> {
        let m = x.len() / self.in_features;
        match &self.variant {
            LinearVariant::Standard { weight, bias } => {
                let w = weight.get_slice::<f16>(); let b = bias.as_ref().map(|t| t.get_slice::<f16>());
                native_linear_f16(x, w, b, m, self.out_features, self.in_features)
            },
            LinearVariant::BitSerial { weight_packed, scales, bias } => {
                #[cfg(feature = "cuda")]
                if self.device_id >= 0 && weight_packed.gpu_ptr.is_some() {
                    let mut res = bit_serial_matmul_gpu(x, weight_packed, scales, m, self.out_features, self.in_features, self.device_id as usize);
                    if let Some(b) = bias { let b_d = b.get_slice::<f16>(); for i in 0..m { for j in 0..self.out_features { res[i * self.out_features + j] += b_d[j]; } } }
                    return res;
                }
                let wp = weight_packed.get_slice::<u32>(); let s = scales.get_slice::<f16>();
                let mut out = bit_serial_matmul_f32_extreme(x, wp, s, m, self.out_features, self.in_features);
                if let Some(b) = bias.as_ref().map(|t| t.get_slice::<f16>()) {
                    for i in 0..m { for j in 0..self.out_features { out[i * self.out_features + j] += b[j].to_f32(); } }
                }
                out.into_iter().map(f16::from_f32).collect()
            }
        }
    }
    pub fn move_to_gpu(&mut self, device_id: i32) {
        self.device_id = device_id;
        if let LinearVariant::BitSerial { weight_packed, .. } = &mut self.variant {
            #[cfg(feature = "cuda")] weight_packed.move_to_gpu(device_id);
        }
    }
}

pub struct NativeLayer {
    pub input_layernorm: NativeTensor,
    pub post_attention_layernorm: NativeTensor,
    pub q_norm: Option<NativeTensor>, 
    pub k_norm: Option<NativeTensor>, 
    pub q_proj: NativeLinear,
    pub k_proj: NativeLinear,
    pub v_proj: NativeLinear,
    pub o_proj: NativeLinear,
    pub gate_proj: NativeLinear,
    pub up_proj: NativeLinear,
    pub down_proj: NativeLinear,
    pub device_id: i32,
    pub kv_cache: std::sync::Mutex<Option<(Vec<u32>, Vec<f16>)>>, 
    pub gpu_kv_cache: std::sync::Mutex<Option<(GpuPtr, GpuPtr, usize)>>, 
}

unsafe impl Send for NativeLayer {}
unsafe impl Sync for NativeLayer {}

impl NativeLayer {
    pub fn forward(&self, x: &[f16], config: &Qwen3VLTextConfig, seqlen_offset: usize, _idx: usize) -> Vec<f16> {
        let hidden_size = config.hidden_size; let q_len = x.len() / hidden_size;
        let head_dim = config.head_dim; let n_h = config.num_attention_heads; let n_kv = config.num_key_value_heads;

        let residual = x.to_vec();
        let x_norm = native_rms_norm_f16(x, self.input_layernorm.get_slice::<f16>(), config.rms_norm_eps as f32, hidden_size);
        
        let mut q = self.q_proj.forward(&x_norm);
        let mut k = self.k_proj.forward(&x_norm);
        let v = self.v_proj.forward(&x_norm);

        if let Some(ref qw) = self.q_norm { q = native_rms_norm_f16(&q, qw.get_slice::<f16>(), config.rms_norm_eps as f32, head_dim); }
        if let Some(ref kw) = self.k_norm { k = native_rms_norm_f16(&k, kw.get_slice::<f16>(), config.rms_norm_eps as f32, head_dim); }

        native_apply_rope_f16_with_offset(&mut q, &mut k, q_len, seqlen_offset, n_h, head_dim, config.rope_theta);

        #[cfg(feature = "cuda")]
        if self.device_id >= 0 {
            use cudarc::driver::sys::*;
            let mut gpu_cache_guard = self.gpu_kv_cache.lock().unwrap();
            let (k_ptr, v_ptr, current_len) = if let Some((kp, vp, l)) = *gpu_cache_guard { (kp, vp, l) } else {
                let mut kp: CUdeviceptr = 0; let mut vp: CUdeviceptr = 0;
                unsafe { 
                    let cuda_lib = lib();
                    cuda_lib.cuMemAlloc_v2(&mut kp, 32768 * n_kv * (head_dim/32) * 4); 
                    cuda_lib.cuMemAlloc_v2(&mut vp, 32768 * n_kv * head_dim * 4); 
                }
                (GpuPtr(kp as *mut _), GpuPtr(vp as *mut _), 0)
            };

            let k_packed = pack_f16_to_bits(&k);
            let v_f32: Vec<f32> = v.iter().map(|val| val.to_f32()).collect();
            unsafe {
                let cuda_lib = lib();
                let k_offset = current_len * n_kv * (head_dim/32) * 4;
                let v_offset = current_len * n_kv * head_dim * 4;
                cuda_lib.cuMemcpyHtoD_v2((k_ptr.0 as usize + k_offset) as CUdeviceptr, k_packed.as_ptr() as *const _, k_packed.len() * 4);
                cuda_lib.cuMemcpyHtoD_v2((v_ptr.0 as usize + v_offset) as CUdeviceptr, v_f32.as_ptr() as *const _, v_f32.len() * 4);
            }
            let new_len = current_len + q_len;
            *gpu_cache_guard = Some((k_ptr, v_ptr, new_len));

            let attn_out = native_bit_serial_attn_gpu(&q, k_ptr, v_ptr, n_h, head_dim, new_len, self.device_id as usize);
            let mut x_at = self.o_proj.forward(&attn_out);
            for i in 0..x_at.len() { x_at[i] += residual[i]; }
            let r_mlp = x_at.clone();
            let x_n_m = native_rms_norm_f16(&x_at, self.post_attention_layernorm.get_slice::<f16>(), config.rms_norm_eps as f32, hidden_size);
            let mut gate = self.gate_proj.forward(&x_n_m); let up = self.up_proj.forward(&x_n_m);
            native_silu_f16(&mut gate); for i in 0..gate.len() { gate[i] *= up[i]; }
            let mut x_m = self.down_proj.forward(&gate); for i in 0..x_m.len() { x_m[i] += r_mlp[i]; }
            return x_m;
        }

        let mut cache_guard = self.kv_cache.lock().unwrap();
        let (k_p_f, v_f_f) = if let Some((pk, pv)) = cache_guard.take() {
            let mut nk = pk; let mut nv = pv; nk.extend_from_slice(&pack_f16_to_bits(&k)); nv.extend_from_slice(&v); (nk, nv)
        } else { (pack_f16_to_bits(&k), v) };
        let t_s = v_f_f.len() / (n_kv * head_dim);
        *cache_guard = Some((k_p_f.clone(), v_f_f.clone())); drop(cache_guard);

        let attn_out = native_bit_serial_attn_f16(&q, &k_p_f, &v_f_f, hidden_size, n_h, q_len, t_s);
        let mut x_at = self.o_proj.forward(&attn_out);
        for i in 0..x_at.len() { x_at[i] += residual[i]; }
        let r_mlp = x_at.clone();
        let x_n_m = native_rms_norm_f16(&x_at, self.post_attention_layernorm.get_slice::<f16>(), config.rms_norm_eps as f32, hidden_size);
        let mut gate = self.gate_proj.forward(&x_n_m); let up = self.up_proj.forward(&x_n_m);
        native_silu_f16(&mut gate); for i in 0..gate.len() { gate[i] *= up[i]; }
        let mut x_m = self.down_proj.forward(&gate); for i in 0..x_m.len() { x_m[i] += r_mlp[i]; }
        x_m
    }

    pub fn move_to_gpu(&mut self, device_id: i32) {
        self.device_id = device_id;
        self.q_proj.move_to_gpu(device_id); self.k_proj.move_to_gpu(device_id);
        self.v_proj.move_to_gpu(device_id); self.o_proj.move_to_gpu(device_id);
        self.gate_proj.move_to_gpu(device_id); self.up_proj.move_to_gpu(device_id);
        self.down_proj.move_to_gpu(device_id);
    }

    pub fn clear_kv_cache(&self) {
        let mut cache = self.kv_cache.lock().unwrap();
        *cache = None;
        let mut gpu_cache = self.gpu_kv_cache.lock().unwrap();
        if let Some((k, v, _)) = gpu_cache.take() {
            #[cfg(feature = "cuda")]
            unsafe {
                use cudarc::driver::sys::*;
                lib().cuMemFree_v2(k.0 as CUdeviceptr);
                lib().cuMemFree_v2(v.0 as CUdeviceptr);
            }
        }
    }

    pub fn get_kv_data(&self) -> Option<(Vec<u32>, Vec<f16>)> {
        let cache = self.kv_cache.lock().unwrap();
        cache.clone()
    }

    pub fn set_kv_data(&self, k: Vec<u32>, v: Vec<f16>) {
        let mut cache = self.kv_cache.lock().unwrap();
        *cache = Some((k, v));
    }
}

pub struct NativeQwen3TextModel {
    pub config: Qwen3VLTextConfig, pub embed_tokens: NativeLinear, pub layers: Vec<NativeLayer>, pub norm: NativeTensor,
}

unsafe impl Send for NativeQwen3TextModel {}
unsafe impl Sync for NativeQwen3TextModel {}

impl NativeQwen3TextModel {
    pub fn forward(&self, input_ids: &[u32], seqlen_offset: usize) -> Vec<f16> {
        let hid = self.config.hidden_size;
        let x = match &self.embed_tokens.variant {
            LinearVariant::Standard { weight, .. } => native_embedding_lookup_f16(input_ids, weight.get_slice::<f16>(), hid),
            _ => vec![f16::ZERO; input_ids.len() * hid],
        };
        let mut cur_x = x;
        for (i, layer) in self.layers.iter().enumerate() { cur_x = layer.forward(&cur_x, &self.config, seqlen_offset, i); }
        native_rms_norm_f16(&cur_x, self.norm.get_slice::<f16>(), self.config.rms_norm_eps as f32, hid)
    }

    pub fn clear_kv_cache(&self) {
        for layer in &self.layers { layer.clear_kv_cache(); }
    }

    pub fn get_all_kv(&self) -> Vec<(Vec<u32>, Vec<f16>)> {
        self.layers.iter().filter_map(|l| l.get_kv_data()).collect()
    }
}

pub struct NativeQwen3VLModel {

    pub config: Qwen3VLConfig, 

    pub text_model: NativeQwen3TextModel, 

    pub lm_head: NativeLinear,

    pub visual: Option<NativeVisionModel>, // 비전 모델 필드 추가

}



pub struct NativeVisionModel {

    pub patch_embed: NativeLinear,

    pub blocks: Vec<NativeLayer>,

    pub merger: NativeLayer,

}



unsafe impl Send for NativeVisionModel {}

unsafe impl Sync for NativeVisionModel {}



impl NativeQwen3VLModel {

    pub fn load(config: Qwen3VLConfig, m_mmap: Arc<Mmap>, v_mmap: Option<Arc<Mmap>>, baking: bool) -> Result<Self> {

        let st = SafeTensors::deserialize(&m_mmap)?; 

        let t_c = config.text_config.as_ref().ok_or(anyhow!("Missing text_config"))?;

        

        // ... (find_key, get_t, get_l 로직 동일)



        

        // [SMART-LOADER] 실제 존재하는 텐서 키를 찾는 함수

        let find_key = |target: &str| -> Option<String> {

            let variations = vec![

                target.to_string(),

                target.replace("model.", "model.language_model."),

                format!("model.language_model.{}", target.replace("model.", ""))

            ];

            for v in variations {

                if st.tensor(&v).is_ok() { return Some(v); }

            }

            // 전체 목록에서 부분 일치 검색 (최후의 수단)

            st.names().iter().find(|&n| n.contains(target)).map(|n| n.to_string())

        };



        let get_t = |name: &str| -> Result<NativeTensor> {

            let key = find_key(name).ok_or_else(|| {

                println!("[ERROR] Tensor NOT found in file: {}", name);

                anyhow!("TensorNotFound: {}", name)

            })?;

            let v = st.tensor(&key)?;

            let off = unsafe { v.data().as_ptr().offset_from(m_mmap.as_ptr()) } as usize;

            Ok(NativeTensor::from_mmap(m_mmap.clone(), off, v.shape().to_vec(), NativeDType::F16))

        };



        let get_l = |base: &str, in_f: usize, out_f: usize| -> Result<NativeLinear> {

            let key = find_key(base).ok_or_else(|| anyhow!("LinearTensorNotFound: {}", base))?;

            

            let p_n = format!("{}.packed", key);

            let s_n = format!("{}.scales", key);

            

            if st.tensor(&p_n).is_ok() {

                let vp = st.tensor(&p_n)?; let vs = st.tensor(&s_n)?;

                let op = unsafe { vp.data().as_ptr().offset_from(m_mmap.as_ptr()) } as usize;

                let os = unsafe { vs.data().as_ptr().offset_from(m_mmap.as_ptr()) } as usize;

                Ok(NativeLinear { in_features: in_f, out_features: out_f, variant: LinearVariant::BitSerial {

                    weight_packed: NativeTensor::from_mmap(m_mmap.clone(), op, vp.shape().to_vec(), NativeDType::U32),

                    scales: NativeTensor::from_mmap(m_mmap.clone(), os, vs.shape().to_vec(), NativeDType::F16),

                    bias: None,

                }, device_id: -1 })

            } else {

                let v = st.tensor(&key)?;

                let o = unsafe { v.data().as_ptr().offset_from(m_mmap.as_ptr()) } as usize;

                Ok(NativeLinear { in_features: in_f, out_features: out_f, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(m_mmap.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: -1 })

            }

        };



        let emb = get_l("model.embed_tokens.weight", 151936, t_c.hidden_size)

            .or_else(|_| get_l("model.layers.0.input_layernorm.weight", t_c.hidden_size, t_c.hidden_size))?;

            

        let mut layers = Vec::new(); 

        let l_to_l = if baking { 1 } else { t_c.num_hidden_layers };

        

        for i in 0..l_to_l {

            let p = format!("model.layers.{}", i);

            layers.push(NativeLayer {

                input_layernorm: get_t(&format!("{}.input_layernorm.weight", p))?,

                post_attention_layernorm: get_t(&format!("{}.post_attention_layernorm.weight", p))?,

                q_norm: get_t(&format!("{}.self_attn.q_norm.weight", p)).ok(),

                k_norm: get_t(&format!("{}.self_attn.k_norm.weight", p)).ok(),

                q_proj: get_l(&format!("{}.self_attn.q_proj.weight", p), t_c.hidden_size, t_c.hidden_size)?,

                k_proj: get_l(&format!("{}.self_attn.k_proj.weight", p), t_c.hidden_size, t_c.hidden_size)?,

                v_proj: get_l(&format!("{}.self_attn.v_proj.weight", p), t_c.hidden_size, t_c.hidden_size)?,

                o_proj: get_l(&format!("{}.self_attn.o_proj.weight", p), t_c.hidden_size, t_c.hidden_size)?,

                gate_proj: get_l(&format!("{}.mlp.gate_proj.weight", p), t_c.hidden_size, t_c.intermediate_size)?,

                up_proj: get_l(&format!("{}.mlp.up_proj.weight", p), t_c.hidden_size, t_c.intermediate_size)?,

                down_proj: get_l(&format!("{}.mlp.down_proj.weight", p), t_c.intermediate_size, t_c.hidden_size)?,

                device_id: -1, kv_cache: std::sync::Mutex::new(None), gpu_kv_cache: std::sync::Mutex::new(None),

            });

        }

        let norm = get_t("model.norm.weight")?;

                let head = get_l("lm_head.weight", t_c.hidden_size, 151936)

                    .or_else(|_| get_l("model.language_model.lm_head.weight", t_c.hidden_size, 151936))

                    .unwrap_or_else(|_| {

                        println!("[WARN] LM Head not found, using identity fallback.");

                        NativeLinear { in_features: t_c.hidden_size, out_features: 1, variant: LinearVariant::Standard { weight: norm.clone(), bias: None }, device_id: -1 }

                    });

        

                // [VISION-LOAD] 비전 모델 선택적 로드

                let visual = if let Some(vm) = v_mmap {

                    println!("[MODEL] Vision data provided. Loading Vision Module...");

                    let vst = SafeTensors::deserialize(&vm)?;

                    

                    // 비전용 헬퍼 함수

                    let get_vt = |name: &str| -> Result<NativeTensor> {

                        let v = vst.tensor(name)?;

                        let off = unsafe { v.data().as_ptr().offset_from(vm.as_ptr()) } as usize;

                        Ok(NativeTensor::from_mmap(vm.clone(), off, v.shape().to_vec(), NativeDType::F16))

                    };

                    

                    let get_vl = |base: &str, in_f: usize, out_f: usize| -> Result<NativeLinear> {

                        let p_n = format!("{}.packed", base);

                        let s_n = format!("{}.scales", base);

                        if vst.tensor(&p_n).is_ok() {

                            let vp = vst.tensor(&p_n)?; let vs = vst.tensor(&s_n)?;

                            let op = unsafe { vp.data().as_ptr().offset_from(vm.as_ptr()) } as usize;

                            let os = unsafe { vs.data().as_ptr().offset_from(vm.as_ptr()) } as usize;

                            Ok(NativeLinear { in_features: in_f, out_features: out_f, variant: LinearVariant::BitSerial {

                                weight_packed: NativeTensor::from_mmap(vm.clone(), op, vp.shape().to_vec(), NativeDType::U32),

                                scales: NativeTensor::from_mmap(vm.clone(), os, vs.shape().to_vec(), NativeDType::F16),

                                bias: None,

                            }, device_id: -1 })

                        } else {

                            let v = vst.tensor(base)?;

                            let o = unsafe { v.data().as_ptr().offset_from(vm.as_ptr()) } as usize;

                            Ok(NativeLinear { in_features: in_f, out_features: out_f, variant: LinearVariant::Standard { weight: NativeTensor::from_mmap(vm.clone(), o, v.shape().to_vec(), NativeDType::F16), bias: None }, device_id: -1 })

                        }

                    };

        

                                let v_cfg = config.vision_config.as_ref().ok_or(anyhow!("Missing vision_config"))?;

        

                                let v_intermediate = v_cfg.hidden_size * 4; // Standard transformer ratio fallback

        

                                let v_out_hidden = v_cfg.out_hidden_size.unwrap_or(v_cfg.hidden_size * 2);

        

                                

        

                                let patch_embed = get_vl("visual.patch_embed.proj.weight", 1536, v_cfg.hidden_size)?;

        

                                let mut blocks = Vec::new();

        

                                let b_to_l = if baking { 1 } else { v_cfg.depth };

        

                                for i in 0..b_to_l {

        

                                    let p = format!("visual.blocks.{}", i);

        

                                    blocks.push(NativeLayer {

        

                                        input_layernorm: get_vt(&format!("{}.norm1.weight", p))?,

        

                                        post_attention_layernorm: get_vt(&format!("{}.norm2.weight", p))?,

        

                                        q_norm: None, k_norm: None,

        

                                        q_proj: get_vl(&format!("{}.attn.q_proj.weight", p), v_cfg.hidden_size, v_cfg.hidden_size)?,

        

                                        k_proj: get_vl(&format!("{}.attn.k_proj.weight", p), v_cfg.hidden_size, v_cfg.hidden_size)?,

        

                                        v_proj: get_vl(&format!("{}.attn.v_proj.weight", p), v_cfg.hidden_size, v_cfg.hidden_size)?,

        

                                        o_proj: get_vl(&format!("{}.attn.proj.weight", p), v_cfg.hidden_size, v_cfg.hidden_size)?,

        

                                        gate_proj: get_vl(&format!("{}.mlp.fc1.weight", p), v_cfg.hidden_size, v_intermediate)?,

        

                                        up_proj: get_vl(&format!("{}.mlp.fc1.weight", p), v_cfg.hidden_size, v_intermediate)?, 

        

                                        down_proj: get_vl(&format!("{}.mlp.fc2.weight", p), v_intermediate, v_cfg.hidden_size)?,

        

                                        device_id: -1, kv_cache: std::sync::Mutex::new(None), gpu_kv_cache: std::sync::Mutex::new(None),

        

                                    });

        

                                }

        

                                

        

                                let merger = NativeLayer {

        

                                    input_layernorm: get_vt("visual.merger.norm.weight")?,

        

                                    post_attention_layernorm: get_vt("visual.merger.norm.weight")?, // Placeholder

        

                                    q_norm: None, k_norm: None,

        

                                    q_proj: get_vl("visual.merger.mlp.0.weight", v_cfg.hidden_size * 4, v_cfg.hidden_size * 4)?,

        

                                    k_proj: get_vl("visual.merger.mlp.0.weight", v_cfg.hidden_size * 4, v_cfg.hidden_size * 4)?,

        

                                    v_proj: get_vl("visual.merger.mlp.0.weight", v_cfg.hidden_size * 4, v_cfg.hidden_size * 4)?,

        

                                    o_proj: get_vl("visual.merger.mlp.2.weight", v_cfg.hidden_size * 4, v_out_hidden)?,

        

                                    gate_proj: get_vl("visual.merger.mlp.0.weight", v_cfg.hidden_size * 4, v_cfg.hidden_size * 4)?,

        

                                    up_proj: get_vl("visual.merger.mlp.0.weight", v_cfg.hidden_size * 4, v_cfg.hidden_size * 4)?,

        

                                    down_proj: get_vl("visual.merger.mlp.2.weight", v_cfg.hidden_size * 4, v_out_hidden)?,

        

                                    device_id: -1, kv_cache: std::sync::Mutex::new(None), gpu_kv_cache: std::sync::Mutex::new(None),

        

                                };

        

                    

        

                    Some(NativeVisionModel { patch_embed, blocks, merger })

                } else {

                    None

                };

                    

                Ok(Self { config: config.clone(), text_model: NativeQwen3TextModel { config: t_c.clone(), embed_tokens: emb, layers, norm }, lm_head: head, visual })

            }

        


    pub fn forward(&self, i_ids: &[u32], _p_v: Option<&[f16]>, _g_t: Option<&[u32; 3]>, s_o: usize) -> Vec<f16> {
        let x = self.text_model.forward(i_ids, s_o); self.lm_head.forward(&x)
    }
    pub fn clear_kv_cache(&self) { self.text_model.clear_kv_cache(); }
    pub fn move_to_gpu(&mut self, device_id: i32) {
        for layer in &mut self.text_model.layers { layer.move_to_gpu(device_id); }
    }
}
