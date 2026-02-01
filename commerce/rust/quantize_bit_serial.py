import torch
import numpy as np
from safetensors.torch import save_file, load_file
import os
import gguf
import re

def quantize_to_bit_serial_full(input_path, output_path, is_gguf=False):
    print(f"[BIT-SERIAL] Processing: {input_path}")
    
    # 1. 모델 로드
    if is_gguf:
        reader = gguf.GGUFReader(input_path)
        tensors = {t.name: torch.from_numpy(t.data.copy()) for t in reader.tensors}
    else:
        tensors = load_file(input_path)
    
    quantized_dict = {}
    BLOCK_SIZE = 512 
    BIT_UNIT = 64 # u64 alignment

    for name, param in tensors.items():
        # 모든 레이어 보존
        if "weight" in name and len(param.shape) >= 2:
            # Embedding: 2-bit (입력 데이터 정밀도 유지)
            if "embed_tokens" in name or "token_embd" in name:
                min_val, max_val = param.min(), param.max()
                scale = (max_val - min_val) / 3.0
                q_param = torch.clamp(torch.round((param - min_val) / scale), 0, 3).to(torch.uint8)
                packed_shape = (q_param.shape[0], q_param.shape[1] // 4)
                packed = torch.zeros(packed_shape, dtype=torch.uint8)
                for i in range(4):
                    packed |= (q_param[:, i::4] << (i * 2))
                
                quantized_dict[f"{name}.packed"] = packed
                quantized_dict[f"{name}.scale"] = torch.tensor([scale.item()], dtype=torch.float32)
                quantized_dict[f"{name}.min"] = torch.tensor([min_val.item()], dtype=torch.float32)
                quantized_dict[f"{name}.shape"] = torch.tensor(list(param.shape), dtype=torch.int32)
                continue

            # Standard Weights: 1-bit Bit-Serial (XOR/POPCNT optimized)
            original_shape = list(param.shape)
            flat_w = param.view(-1).to(torch.float32)
            
            # 64비트 정렬을 위한 패딩
            pad_len = (BIT_UNIT - (flat_w.numel() % BIT_UNIT)) % BIT_UNIT
            if pad_len > 0:
                flat_w = torch.cat([flat_w, torch.zeros(pad_len)])
            
            binary = (flat_w >= 0).to(torch.uint8)
            binary_np = binary.numpy()
            
            # [FIX] Reshape and pack, then flatten BEFORE viewing as uint64
            # bitorder='little'는 Rust의 리틀 엔디안 읽기와 일치합니다.
            packed_bytes = np.packbits(binary_np.reshape(-1, 8), axis=1, bitorder='little').flatten()
            packed_u64 = packed_bytes.view(np.uint64)
            
            # 다시 torch 텐서로 변환 (int64로 취급하여 저장)
            packed_tensor = torch.from_numpy(packed_u64.view(np.int64))
            
            # 스케일 계산 (블록 단위)
            num_blocks = flat_w.numel() // BLOCK_SIZE
            blocks = flat_w.view(num_blocks, BLOCK_SIZE)
            scales = torch.max(torch.abs(blocks), dim=1)[0].to(torch.float16)
            
            quantized_dict[f"{name}.packed"] = packed_tensor
            quantized_dict[f"{name}.scales"] = scales
            quantized_dict[f"{name}.shape"] = torch.tensor(original_shape, dtype=torch.int32)
            quantized_dict[f"{name}.format"] = torch.tensor([1], dtype=torch.int8) 
        else:
            quantized_dict[name] = param

    save_file(quantized_dict, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f" -> DONE. Saved to {output_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    tasks = [
        # 1. 2B 비전 (Original GGUF)
        ("src-tauri/models/Qwen3-VL-2B-Instruct-gguf/mmproj-Qwen3VL-2B-Instruct-F16.gguf", 
         "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/mmproj-BITSERIAL_ALL.safetensors", True),
        
        # 2. 2B 언어 (Safetensors)
        ("src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model.safetensors", 
         "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model-BITSERIAL_ALL.safetensors", False),
        
        # 3. 0.6B 언어 (Safetensors)
        ("src-tauri/models/Qwen3-0.6B-Instruct-gguf/model.safetensors", 
         "src-tauri/models/Qwen3-0.6B-Instruct-gguf/model-BITSERIAL_ALL.safetensors", False),
    ]

    for src, tgt, is_gguf in tasks:
        if os.path.exists(src):
            quantize_to_bit_serial_full(src, tgt, is_gguf)
        else:
            print(f"[SKIP] File not found: {src}")