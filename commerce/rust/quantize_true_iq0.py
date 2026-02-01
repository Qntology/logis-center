import torch
import numpy as np
from safetensors.torch import save_file, load_file
import os
import gguf
import re

def quantize_to_anchor_bit_serial(input_path, output_path, is_gguf=False):
    print(f"[ANCHOR-BIT-SERIAL] Processing {input_path}...")
    
    quantized_dict = {}
    BLOCK_SIZE = 512 
    BIT_UNIT = 64 

    if is_gguf:
        reader = gguf.GGUFReader(input_path)
        tensors_to_process = {t.name: torch.from_numpy(t.data.copy()) for t in reader.tensors}
    else:
        tensors_to_process = load_file(input_path)

    for name, param in tensors_to_process.items():
        # [STRICT ANCHOR POLICY] 0번 레이어만 보존
        match = re.search(r'(layers|blk|deepstack|blocks)\.(\d+)\.', name)
        if match:
            layer_idx = int(match.group(2))
            if layer_idx > 0:
                continue 

        if "weight" in name and len(param.shape) >= 2:
            # 1. Embedding: 2-bit
            if "embed_tokens" in name or "token_embd" in name:
                min_val, max_val = param.min(), param.max()
                scale = (max_val - min_val) / 3.0
                q_param = torch.clamp(torch.round((param - min_val) / scale), 0, 3).to(torch.uint8)
                packed_shape = (q_param.shape[0], q_param.shape[1] // 4)
                packed = torch.zeros(packed_shape, dtype=torch.uint8)
                for i in range(4): packed |= (q_param[:, i::4] << (i * 2))
                
                quantized_dict[f"{name}.packed"] = packed
                quantized_dict[f"{name}.scale"] = torch.tensor([scale.item()], dtype=torch.float32)
                quantized_dict[f"{name}.min"] = torch.tensor([min_val.item()], dtype=torch.float32)
                quantized_dict[f"{name}.shape"] = torch.tensor(list(param.shape), dtype=torch.int32)
                continue

            # 2. Linear Weights: 1-bit Bit-Serial (512-bit aligned)
            original_shape = list(param.shape)
            flat_w = param.view(-1).to(torch.float32)
            
            # [FIX] BLOCK_SIZE (512) 단위 패딩 강제
            pad_len = (BLOCK_SIZE - (flat_w.numel() % BLOCK_SIZE)) % BLOCK_SIZE
            if pad_len > 0: flat_w = torch.cat([flat_w, torch.zeros(pad_len)])
            
            num_blocks = flat_w.numel() // BLOCK_SIZE
            blocks = flat_w.view(num_blocks, BLOCK_SIZE)
            scales = torch.max(torch.abs(blocks), dim=1)[0].to(torch.float16)
            binary = (blocks >= 0).to(torch.uint8)
            
            # u64 Packing (XOR/POPCNT Optimized)
            binary_np = binary.view(-1).numpy()
            packed_bytes = np.packbits(binary_np.reshape(-1, 8), axis=1, bitorder='little').flatten()
            packed_u64 = packed_bytes.view(np.uint64)
            packed_tensor = torch.from_numpy(packed_u64.view(np.int64))
            
            quantized_dict[f"{name}.packed"] = packed_tensor
            quantized_dict[f"{name}.scales"] = scales
            quantized_dict[f"{name}.shape"] = torch.tensor(original_shape, dtype=torch.int32)
            quantized_dict[f"{name}.format"] = torch.tensor([1], dtype=torch.int8) 
        else:
            quantized_dict[name] = param

    save_file(quantized_dict, output_path)
    print(f" -> DONE. Saved to {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    # 0.6B ANCHOR 모델 (Baking용)
    src_06b = "src-tauri/models/Qwen3-0.6B-Instruct-gguf/model.safetensors"
    tgt_06b = "src-tauri/models/Qwen3-0.6B-Instruct-gguf/Qwen3-0.6B-UD-ANCHOR_IQ0.safetensors"
    if os.path.exists(src_06b):
        quantize_to_anchor_bit_serial(src_06b, tgt_06b)

    # 2B ANCHOR 모델
    src_2b = "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model.safetensors"
    tgt_2b = "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/Qwen3VL-2B-Instruct-ANCHOR_IQ0.safetensors"
    if os.path.exists(src_2b):
        quantize_to_anchor_bit_serial(src_2b, tgt_2b)
