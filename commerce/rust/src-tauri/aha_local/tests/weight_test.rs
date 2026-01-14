use std::collections::HashMap;

use aha::utils::{find_type_files, get_device};
use anyhow::Result;
use candle_core::{Device, pickle::read_all_with_key, safetensors};
use candle_nn::VarBuilder;

#[test]
fn minicpm4_weight() -> Result<()> {
    let model_path = "/home/jhq/huggingface_model/OpenBMB/MiniCPM4-0.5B/";
    let model_list = find_type_files(model_path, "safetensors")?;
    let device = Device::Cpu;
    for m in model_list {
        let weights = safetensors::load(m, &device)?;
        for (key, tensor) in weights.iter() {
            println!("=== {} ===", key);
            println!("Shape: {:?}", tensor.shape());
            println!("DType: {:?}", tensor.dtype());
        }
    }
    Ok(())
}

#[test]
fn voxcpm_weight() -> Result<()> {
    let model_path = "/home/jhq/huggingface_model/openbmb/VoxCPM-0.5B/";
    let model_list = find_type_files(model_path, "pth")?;
    println!("model_list: {:?}", model_list);
    let dev = get_device(None);
    let mut dict_to_hashmap = HashMap::new();
    let mut dtype = candle_core::DType::F16;
    for m in model_list {
        let dict = read_all_with_key(m, Some("state_dict"))?;
        dtype = dict[0].1.dtype();
        for (k, v) in dict {
            println!("key: {}, tensor shape: {:?}", k, v);
            dict_to_hashmap.insert(k, v);
        }
    }
    let vb = VarBuilder::from_tensors(dict_to_hashmap, dtype, &dev);
    let contain_key = vb.contains_tensor("encoder.block.4.block.2.block.3.weight_g");
    println!(
        "contain encoder.block.4.block.2.block.3.weight_g: {}",
        contain_key
    );
    Ok(())
}

#[test]
fn voxcpm1_5_weight() -> Result<()> {
    let model_path = "/home/jhq/huggingface_model/OpenBMB/VoxCPM1.5/";
    let model_list = find_type_files(model_path, "pth")?;
    println!("model_list: {:?}", model_list);
    // let dev = get_device(None);
    let mut dict_to_hashmap = HashMap::new();
    // let mut dtype = candle_core::DType::F32;
    for m in model_list {
        let dict = read_all_with_key(m, Some("state_dict"))?;
        // dtype = dict[0].1.dtype();
        for (k, v) in dict {
            println!("key: {}, tensor shape: {:?}", k, v);
            dict_to_hashmap.insert(k, v);
        }
    }

    Ok(())
}

#[test]
fn qwen3vl_weight() -> Result<()> {
    let model_path = "/home/jhq/huggingface_model/Qwen/Qwen3-VL-4B-Instruct/";
    let model_list = find_type_files(model_path, "safetensors")?;

    let device = Device::Cpu;
    for m in &model_list {
        let weights = safetensors::load(m, &device)?;
        for (key, tensor) in weights.iter() {
            println!("=== {} === {:?}", key, tensor.shape());
        }
    }
    println!("model_list: {:?}", model_list);
    Ok(())
}

#[test]
fn deepseekocr_weight() -> Result<()> {
    let model_path = "/home/jhq/huggingface_model/deepseek-ai/DeepSeek-OCR/";
    let model_list = find_type_files(model_path, "safetensors")?;

    let device = Device::Cpu;
    for m in &model_list {
        let weights = safetensors::load(m, &device)?;
        for (key, tensor) in weights.iter() {
            if key.contains("rel_pos_h") {
                println!("=== {} === {:?}", key, tensor.shape());
            }
            // println!("=== {} === {:?}", key, tensor.shape());
        }
    }
    println!("model_list: {:?}", model_list);
    Ok(())
}

#[test]
fn hunyuanocr_weight() -> Result<()> {
    let model_path = "/home/jhq/huggingface_model/Tencent-Hunyuan/HunyuanOCR/";
    let model_list = find_type_files(model_path, "safetensors")?;

    let device = Device::Cpu;
    for m in &model_list {
        let weights = safetensors::load(m, &device)?;
        for (key, tensor) in weights.iter() {
            if key.contains(".image_") {
                println!("=== {} === {:?}", key, tensor.shape());
            }
            // println!("=== {} === {:?}", key, tensor.shape());
        }
    }
    println!("model_list: {:?}", model_list);
    Ok(())
}
