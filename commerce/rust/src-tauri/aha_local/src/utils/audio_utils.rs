use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{f64::consts::PI, io::Cursor};

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionParameters, ChatCompletionResponse, ChatMessage, ChatMessageContent,
    ChatMessageContentPart,
};
use anyhow::{Result, anyhow};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use candle_core::{D, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module};
use hound::{SampleFormat, WavReader};
use num::integer::gcd;

use crate::utils::get_default_save_dir;

// 重采样方法枚举
#[derive(Debug, Clone, Copy)]
pub enum ResamplingMethod {
    SincInterpHann,
    SincInterpKaiser,
}

// 零阶修正贝塞尔函数 I0
fn i0(x: f32) -> f32 {
    let mut result = 1.0;
    let mut term = 1.0;
    let half_x_sq = x * x / 4.0;

    for k in 1..50 {
        term = term * half_x_sq / (k * k) as f32;
        result += term;

        if term < 1e-12 {
            break;
        }
    }

    result
}

// 获取sinc重采样核
pub fn get_sinc_resample_kernel(
    orig_freq: i64,
    new_freq: i64,
    gcd_val: i64,
    lowpass_filter_width: i64,
    rolloff: f64,
    resampling_method: ResamplingMethod,
    beta: Option<f32>,
    device: &Device,
) -> Result<(Tensor, i64)> {
    if orig_freq <= 0 || new_freq <= 0 {
        return Err(anyhow!("Frequencies must be positive".to_string()));
    }

    if lowpass_filter_width <= 0 {
        return Err(anyhow!(
            "Low pass filter width should be positive".to_string()
        ));
    }

    let orig_freq = orig_freq / gcd_val;
    let new_freq = new_freq / gcd_val;

    let base_freq = (orig_freq.min(new_freq) as f64) * rolloff;

    let width_f = (lowpass_filter_width as f64) * (orig_freq as f64) / base_freq;
    let width = width_f.ceil() as i64;
    // 创建索引数组 [1, 1, 2*width + orig_freq]
    let idx = Tensor::arange(-width as f32, (width + orig_freq) as f32, device)?
        .affine(1.0 / orig_freq as f64, 0.0)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    // 创建时间数组 t [new_freq, 1, idx_len]
    let t = Tensor::arange_step(0.0, -new_freq as f32, -1.0, device)?
        .affine(1.0 / new_freq as f64, 0.0)?
        .unsqueeze(D::Minus1)?
        .unsqueeze(D::Minus1)?
        .broadcast_add(&idx)?
        .affine(base_freq, 0.0)?;
    let t = t.clamp(-lowpass_filter_width as f32, lowpass_filter_width as f32)?;
    // 计算窗口函数
    let window = match resampling_method {
        ResamplingMethod::SincInterpHann => {
            let window_arg = t.affine(PI / (lowpass_filter_width as f64) / 2.0, 0.0)?;
            window_arg.cos()?.sqr()?
        }
        ResamplingMethod::SincInterpKaiser => {
            let beta_val = beta.unwrap_or(14.769_656_f32);
            let i0_beta = i0(beta_val);

            let normalized_t = t.affine(1.0 / lowpass_filter_width as f64, 0.0)?;
            let arg = (1.0 - normalized_t.sqr()?)?;
            // 处理arg为负数的情况
            let sqrt_arg = arg.relu()?.sqrt()?;
            let sqrt_dims = sqrt_arg.dims();
            let sqrt_arg_vec = sqrt_arg.flatten_all()?.to_vec1::<f32>()?;

            let window_val: Vec<f32> = sqrt_arg_vec
                .iter()
                .map(|x| i0(beta_val * x) / i0_beta)
                .collect();
            Tensor::new(window_val, device)?.reshape(sqrt_dims)?
        }
    };

    // 计算sinc核
    let scale = base_freq / (orig_freq as f64);
    let t_scaled = t.affine(PI, 0.0)?;

    let t_zeros = Tensor::zeros_like(&t_scaled)?;
    let t_ones = Tensor::ones_like(&t_scaled)?;
    let mask = t_scaled.eq(&t_zeros)?;
    let sinc = mask.where_cond(&t_ones, &t_scaled.sin()?.div(&t_scaled)?)?;
    let kernels = sinc.mul(&window)?.affine(scale, 0.0)?;

    Ok((kernels, width))
}

// 应用sinc重采样核
pub fn apply_sinc_resample_kernel(
    waveform: &Tensor,
    orig_freq: i64,
    new_freq: i64,
    gcd_val: i64,
    kernel: &Tensor,
    width: i64,
) -> Result<Tensor> {
    let orig_freq = orig_freq / gcd_val;
    let new_freq = new_freq / gcd_val;

    // 获取波形形状
    let dims = waveform.dims();
    let waveform_flat = waveform.reshape(((), dims[dims.len() - 1]))?;

    let (num_wavs, length) = waveform_flat.dims2()?;
    let padded_waveform =
        waveform.pad_with_zeros(D::Minus1, width as usize, (width + orig_freq) as usize)?;

    // 添加通道维度 [batch_size, 1, padded_length]
    let waveform_3d = padded_waveform.unsqueeze(1)?;
    let config = Conv1dConfig {
        padding: 0,
        stride: orig_freq as usize,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };

    let conv1d = Conv1d::new(kernel.clone(), None, config);
    // 执行卷积
    // kernel形状: [new_freq_reduced, 1, kernel_len]
    // 输出形状: [batch_size, new_freq_reduced, output_length]
    let conv_output = conv1d.forward(&waveform_3d)?;

    // 转置并重塑 [batch_size, output_length * new_freq_reduced]
    let conv_transposed = conv_output.transpose(1, 2)?.reshape((num_wavs, ()))?;

    // 计算目标长度
    let target_length = ((new_freq as f64 * length as f64) / orig_freq as f64).ceil() as usize;

    // 截取目标长度
    let resampled_flat =
        conv_transposed.narrow(1, 0, target_length.min(conv_transposed.dim(1)?))?;
    let mut new_dims = dims.to_vec();
    let last_dim = new_dims.len() - 1;
    new_dims[last_dim] = resampled_flat.dim(1)?;
    // 恢复原始批次形状

    let resampled = resampled_flat.reshape(new_dims)?;

    Ok(resampled)
}

// 主要的重采样函数
pub fn resample(
    waveform: &Tensor,
    orig_freq: i64,
    new_freq: i64,
    lowpass_filter_width: i64,
    rolloff: f64,
    resampling_method: ResamplingMethod,
    beta: Option<f32>,
) -> Result<Tensor> {
    if orig_freq <= 0 || new_freq <= 0 {
        return Err(anyhow!("Frequencies must be positive".to_string(),));
    }

    if orig_freq == new_freq {
        return Ok(waveform.clone());
    }

    let gcd_val = gcd(orig_freq, new_freq);
    let device = waveform.device();

    let (kernel, width) = get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd_val,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
        device,
    )?;
    let t = apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd_val, &kernel, width)?;
    Ok(t)
}

// 为方便使用提供的简化版本
pub fn resample_simple(waveform: &Tensor, orig_freq: i64, new_freq: i64) -> Result<Tensor> {
    resample(
        waveform,
        orig_freq,
        new_freq,
        6,
        0.99,
        ResamplingMethod::SincInterpHann,
        None,
    )
}
pub fn load_audio_from_url(url: &str) -> Result<PathBuf> {
    tokio::task::block_in_place(|| {
        let client = reqwest::blocking::Client::new();
        let response = client.get(url).send()?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download file: {}",
                response.status()
            ));
        }
        let temp_dir = get_default_save_dir().expect("Failed to get home directory");
        let temp_dir = PathBuf::from(temp_dir);
        let temp_path = temp_dir.join("temp_audio.wav");

        let mut file = std::fs::File::create(&temp_path)?;
        let mut content = Cursor::new(response.bytes()?);
        std::io::copy(&mut content, &mut file)?;

        // Return the temp directory to keep it alive until the function ends
        Ok(temp_path)
    })
}

pub fn get_audio_path(path_str: &str) -> Result<PathBuf> {
    if path_str.starts_with("http://") || path_str.starts_with("https://") {
        // Download file from network
        load_audio_from_url(path_str)
    } else if path_str.starts_with("file://") {
        // Convert file:// URL to local path
        let path = url::Url::parse(path_str)?;
        let path = path.to_file_path();
        let path = match path {
            Ok(path) => path,
            Err(_) => {
                let mut path = path_str.to_owned();
                path = path.split_off(7);
                PathBuf::from(path)
            }
        };
        Ok(path)
    } else if path_str.starts_with("data:audio") && path_str.contains("base64,") {
        let data: Vec<&str> = path_str.split("base64,").collect();
        let data = data[1];
        let temp_dir = get_default_save_dir().expect("Failed to get home directory");
        let temp_dir = PathBuf::from(temp_dir);
        let temp_path = temp_dir.join("temp_audio.wav");
        save_audio_from_base64(data, &temp_path)?;
        Ok(temp_path)
    } else {
        Err(anyhow::anyhow!("get audio path error {}", path_str))
    }
}

pub fn load_audio(path: &str, device: Device) -> Result<(Tensor, usize)> {
    let audio_path = get_audio_path(path)?;
    let mut reader = WavReader::open(audio_path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            // 将整数样本转换为浮点数 [-1.0, 1.0]
            // println!("spec.bits_per_sample: {}", spec.bits_per_sample);
            match spec.bits_per_sample {
                8 => reader
                    .samples::<i8>()
                    .map(|s| s.map(|sample| sample as f32 / i8::MAX as f32))
                    .collect::<Result<Vec<_>, _>>()?,
                16 => reader
                    .samples::<i16>()
                    .map(|s| s.map(|sample| sample as f32 / i16::MAX as f32))
                    .collect::<Result<Vec<_>, _>>()?,
                24 => reader
                    .samples::<i32>()
                    .map(|s| s.map(|sample| sample as f32 / 8388607.0))
                    .collect::<Result<Vec<_>, _>>()?,
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported bit depth: {}",
                        spec.bits_per_sample
                    ));
                }
            }
        }
        SampleFormat::Float => {
            // 直接读取浮点数样本
            reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?
        }
    };
    let sample_rate = spec.sample_rate;
    let mut audio_tensor = Tensor::from_slice(
        &samples,
        (
            samples.len() / spec.channels as usize,
            spec.channels as usize,
        ),
        &device,
    )?
    .t()?;
    if spec.channels > 1 {
        // 对channel通道求平均， channel维度变为1
        audio_tensor = audio_tensor.mean_keepdim(0)?;
    }
    Ok((audio_tensor, sample_rate as usize))
}

pub fn load_audio_with_resample(
    path: &str,
    device: Device,
    target_sample_rate: Option<usize>,
) -> Result<Tensor> {
    let (mut audio, sr) = load_audio(path, device)?;
    if let Some(target_sample_rate) = target_sample_rate
        && target_sample_rate != sr
    {
        audio = resample_simple(&audio, sr as i64, target_sample_rate as i64)?;
    }
    Ok(audio)
}

pub fn save_wav(audio: &Tensor, save_path: &str, sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    assert_eq!(audio.dim(0)?, 1, "audio channel must be 1");
    let max = audio.abs()?.max_all()?;
    let max = max.to_scalar::<f32>()?;
    let ratio = if max > 1.0 { 32767.0 / max } else { 32767.0 };
    let audio = audio.squeeze(0)?;
    let audio_vec = audio.to_vec1::<f32>()?;
    let mut writer = hound::WavWriter::create(save_path, spec).unwrap();
    for i in audio_vec {
        let sample_i16 = (i * ratio).round() as i16;
        writer.write_sample(sample_i16).unwrap();
    }
    writer.finalize().unwrap();
    Ok(())
}

pub fn get_audio_wav_u8(audio: &Tensor, sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    assert_eq!(audio.dim(0)?, 1, "audio channel must be 1");
    let max = audio.abs()?.max_all()?;
    let max = max.to_scalar::<f32>()?;
    let ratio = if max > 1.0 { 32767.0 / max } else { 32767.0 };
    let audio = audio.squeeze(0)?;
    let audio_vec = audio.to_vec1::<f32>()?;
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
    for i in audio_vec {
        let sample_i16 = (i * ratio).round() as i16;
        writer.write_sample(sample_i16)?;
    }
    writer.finalize()?;
    let wav_buffer = cursor.into_inner();
    Ok(wav_buffer)
}

pub fn extract_audio_url(mes: &ChatCompletionParameters) -> Result<Vec<String>> {
    let mut audio_vec = Vec::new();
    for chat_mes in mes.messages.clone() {
        if let ChatMessage::User { content, .. } = chat_mes.clone()
            && let ChatMessageContent::ContentPart(part_vec) = content
        {
            for part in part_vec {
                if let ChatMessageContentPart::Audio(audio_part) = part {
                    let audio_url = audio_part.audio_url;
                    audio_vec.push(audio_url.url);
                }
            }
        }
        // if let ChatMessage::User { content, .. } = chat_mes.clone()
        //     && let ChatMessageContent::ContentPart(part_vec) = content
        // {
        //     for part in part_vec {
        //         if let ChatMessageContentPart::Text(text_part) = part {
        //             let text = text_part.text;
        //             if text.chars().count() > 0 {
        //                 ret = ret + &text + "\n"
        //             }
        //         }
        //     }
        // }
    }
    Ok(audio_vec)
}

// 从 ChatCompletionResponse 中提取音频数据
pub fn extract_audio_base64_from_response(
    response: &ChatCompletionResponse,
) -> Result<Vec<String>> {
    let mut audio_data_list = Vec::new();

    for choice in &response.choices {
        if let ChatMessage::Assistant {
            content: Some(ChatMessageContent::ContentPart(parts)),
            ..
        } = &choice.message
        {
            for part in parts.clone() {
                if let ChatMessageContentPart::Audio(audio_part) = part {
                    // if let Some(audio_data) = &audio_part.audio_url {
                    //     audio_data_list.push(audio_data.data.clone());
                    // }
                    let audio_url = audio_part.audio_url;
                    audio_data_list.push(audio_url.url);
                }
            }
        }
    }

    Ok(audio_data_list)
}

// 将 base64 音频数据解码并保存到文件
pub fn save_audio_from_base64<P: AsRef<Path>>(base64_data: &str, file_path: P) -> Result<()> {
    // 解码 base64 数据
    let data: Vec<&str> = base64_data.split("base64,").collect();
    let data = data[1];
    let decoded_data = BASE64_STANDARD.decode(data)?;

    // 创建文件并写入数据
    let mut file = File::create(file_path)?;
    file.write_all(&decoded_data)?;

    Ok(())
}

// 组合函数：从响应中提取音频并保存到文件
pub fn extract_and_save_audio_from_response(
    response: &ChatCompletionResponse,
    directory: &str,
) -> Result<Vec<String>> {
    let audio_data_list = extract_audio_base64_from_response(response)?;
    let mut saved_files = Vec::new();

    for (index, audio_data) in audio_data_list.iter().enumerate() {
        let file_path = format!("{}/audio_{}.wav", directory, index);
        save_audio_from_base64(audio_data, &file_path)?;
        saved_files.push(file_path);
    }

    Ok(saved_files)
}
