import torch
import numpy as np
from safetensors.torch import save_file, load_file
import os
import re

def quantize_iq0_gguf_style(input_path, output_path):
    print(f"[IQ0-GGUF-HYBRID] Processing: {input_path}")
    tensors = load_file(input_path) if not input_path.endswith(".gguf") else {}
    
    quantized_dict = {}
    BLOCK_SIZE = 32 # GGUF 표준 블록 사이즈로 정밀도 향상

    for name, param in tensors.items():
        idx_match = re.search(r'(layers|blk|blocks)\.(\d+)\.', name)
        layer_idx = int(idx_match.group(2)) if idx_match else -1
        
        # 앵커 레이어는 원본 유지 (GGUF 전략)
        if layer_idx == 0 or "patch_embed" in name:
            quantized_dict[name] = param.to(torch.float16)
            continue

        if "weight" in name and len(param.shape) >= 2:
            original_shape = list(param.shape)
            flat_w = param.view(-1).to(torch.float32)
            
            # GGUF 32-블록 패딩
            pad_len = (BLOCK_SIZE - (flat_w.numel() % BLOCK_SIZE)) % BLOCK_SIZE
            if pad_len > 0: flat_w = torch.cat([flat_w, torch.zeros(pad_len)])
            
            num_blocks = flat_w.numel() // BLOCK_SIZE
            blocks = flat_w.view(num_blocks, BLOCK_SIZE)
            
            # [IQ0-GGUF Mechanism] 블록별 스케일링 (1-bit Bit-serial)
            scales = torch.mean(torch.abs(blocks), dim=1).to(torch.float16)
            
            # Bit-serial 1-bit quantization (0-bit IQ0 style)
            # 0보다 크면 1, 작으면 0으로 패킹
            binary = (blocks >= 0).to(torch.uint8)
            
            # 32개 비트를 1개의 uint32로 패킹 (GGUF 호환 레이아웃)
            packed_uint32 = torch.zeros(num_blocks, dtype=torch.int32)
            for i in range(32):
                packed_uint32 |= (binary[:, i].to(torch.int32) << i)
            
            quantized_dict[f"{name}.packed"] = packed_uint32
            quantized_dict[f"{name}.scales"] = scales
            quantized_dict[f"{name}.shape"] = torch.tensor(original_shape, dtype=torch.int32)
            quantized_dict[f"{name}.format"] = torch.tensor([0], dtype=torch.int8) # IQ0 표시
        else:
            quantized_dict[name] = param.to(torch.float16)

    save_file(quantized_dict, output_path)
    print(f" -> DONE. IQ0-GGUF Hybrid model saved to {output_path}")

if __name__ == "__main__":
    tasks = [
        ("src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model.safetensors", 
         "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model-BITSERIAL_ALL.safetensors"),
        ("src-tauri/models/Qwen3-0.6B-Instruct-gguf/model.safetensors", 
         "src-tauri/models/Qwen3-0.6B-Instruct-gguf/model-BITSERIAL_ALL.safetensors"),
    ]
    for src, tgt in tasks:
        if os.path.exists(src): quantize_iq0_gguf_style(src, tgt)
