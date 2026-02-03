import torch
import numpy as np
from safetensors.torch import save_file, load_file
import os
import re
import gguf

def quantize_tensor_bit_serial(param):
    BLOCK_SIZE = 32
    original_shape = list(param.shape)
    flat_w = param.view(-1).to(torch.float32)
    pad_len = (BLOCK_SIZE - (flat_w.numel() % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_len > 0: flat_w = torch.cat([flat_w, torch.zeros(pad_len)])
    num_blocks = flat_w.numel() // BLOCK_SIZE
    blocks = flat_w.view(num_blocks, BLOCK_SIZE)
    scales = torch.mean(torch.abs(blocks), dim=1).to(torch.float16)
    binary = (blocks >= 0).to(torch.uint8)
    packed_uint32 = torch.zeros(num_blocks, dtype=torch.int32)
    for i in range(32):
        packed_uint32 |= (binary[:, i].to(torch.int32) << i)
    return packed_uint32, scales, torch.tensor(original_shape, dtype=torch.int32)

def split_and_quantize_v2(input_path, text_out_path, vision_out_path):
    print(f"[SPLIT-QUANT] Processing: {input_path}")
    tensors = load_file(input_path)
    
    text_dict = {}
    vision_dict = {}

    for name, param in tensors.items():
        # 1. 비전 텐서 분류 (model.visual. 또는 visual. 로 시작하는 경우)
        is_vision = "visual" in name
        target_dict = vision_dict if is_vision else text_dict
        
        # 2. 양자화 결정 (웨이트이고 크기가 2차원 이상인 경우)
        if "weight" in name and len(param.shape) >= 2:
            packed, scales, shape = quantize_tensor_bit_serial(param)
            target_dict.update({
                f"{name}.packed": packed,
                f"{name}.scales": scales,
                f"{name}.shape": shape,
                f"{name}.format": torch.tensor([0], dtype=torch.int8)
            })
        else:
            target_dict[name] = param.to(torch.float16)

    # 3. 각각 저장
    save_file(text_dict, text_out_path)
    print(f" -> DONE. Text model saved to {text_out_path} ({len(text_dict)} tensors)")
    
    if vision_dict:
        save_file(vision_dict, vision_out_path)
        print(f" -> DONE. Vision model saved to {vision_out_path} ({len(vision_dict)} tensors)")

def quantize_vision_gguf(input_path, output_path):
    """기존 GGUF mmproj 파일 처리용"""
    print(f"[IQ0-VISION-GGUF] Processing: {input_path}")
    reader = gguf.GGUFReader(input_path)
    quantized_dict = {}
    for tensor in reader.tensors:
        name = tensor.name
        data = torch.from_numpy(tensor.data).to(torch.float32)
        new_name = name.replace("v.", "visual.") if name.startswith("v.") else name
        if not new_name.startswith("visual.") and ("blk" in new_name or "mm" in new_name or "patch" in name):
             new_name = f"visual.{new_name}"
        if "weight" in new_name and len(data.shape) >= 2:
            packed, scales, shape = quantize_tensor_bit_serial(data)
            quantized_dict.update({ f"{new_name}.packed": packed, f"{new_name}.scales": scales, f"{new_name}.shape": shape, f"{new_name}.format": torch.tensor([0], dtype=torch.int8) })
        else:
            quantized_dict[new_name] = data.to(torch.float16)
    save_file(quantized_dict, output_path)
    print(f" -> DONE. GGUF Vision saved to {output_path}")

if __name__ == "__main__":
    # 1. 2B-VL 모델 분리 및 양자화
    split_and_quantize_v2(
        "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model.safetensors",
        "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model-BITSERIAL_ALL.safetensors",
        "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/mmproj-BITSERIAL_ALL.safetensors"
    )

    # 2. 0.6B 모델 양자화 (기존 방식 유지하되 명칭 통일)
    split_and_quantize_v2(
        "src-tauri/models/Qwen3-0.6B-Instruct-gguf/model.safetensors",
        "src-tauri/models/Qwen3-0.6B-Instruct-gguf/model-BITSERIAL_ALL.safetensors",
        "src-tauri/models/Qwen3-0.6B-Instruct-gguf/mmproj-DUMMY.safetensors" # 비전 없음
    )
