import torch
import numpy as np
from safetensors.torch import save_file, load_file
import os
import gguf
import re

def quantize_asymmetric_bit_serial(input_path, output_path, is_gguf=False):
    print(f"[FULL-LAYER-RECONSTRUCTION] Processing: {input_path}")
    
    if is_gguf:
        reader = gguf.GGUFReader(input_path)
        tensors = {t.name: torch.from_numpy(t.data.copy()) for t in reader.tensors}
    else:
        tensors = load_file(input_path)
    
    quantized_dict = {}
    BLOCK_SIZE = 512 
    BIT_UNIT = 64 

    processed_layers = set()

    for name, param in tensors.items():
        # [1] Anchor Layer 판정 (오직 Index 0만)
        # 텍스트: model.layers.0.xxx, blk.0.xxx
        # 비전: v.blk.0.xxx, model.visual.blocks.0.xxx
        is_anchor = False
        
        # 정확한 인덱스 추출
        idx_match = re.search(r'(layers|blk|blocks)\.(\d+)\.', name)
        layer_idx = int(idx_match.group(2)) if idx_match else -1
        
        if layer_idx == 0:
            is_anchor = True
        elif "patch_embed.proj" in name:
            is_anchor = True

        if is_anchor:
            # 앵커 레이어는 원본 유지
            quantized_dict[name] = param.to(torch.float16)
            if layer_idx != -1: processed_layers.add(f"Anchor-L{layer_idx}")
            continue

        # [2] 나머지 모든 레이어 (1~27) 압축 수행
        if "weight" in name and len(param.shape) >= 2:
            # Embedding: 2-bit
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

            # Linear Weights: 1-bit (모든 레이어 대상)
            original_shape = list(param.shape)
            flat_w = param.view(-1).to(torch.float32)
            
            pad_len = (BLOCK_SIZE - (flat_w.numel() % BLOCK_SIZE)) % BLOCK_SIZE
            if pad_len > 0: flat_w = torch.cat([flat_w, torch.zeros(pad_len)])
            
            num_blocks = flat_w.numel() // BLOCK_SIZE
            blocks = flat_w.view(num_blocks, BLOCK_SIZE)
            scales = torch.max(torch.abs(blocks), dim=1)[0].to(torch.float16)
            binary = (blocks >= 0).to(torch.uint8)
            
            # u64 Packing
            binary_np = binary.view(-1).numpy()
            packed_bytes = np.packbits(binary_np.reshape(-1, 8), axis=1, bitorder='little').flatten()
            packed_u64 = packed_bytes.view(np.uint64)
            packed_tensor = torch.from_numpy(packed_u64.view(np.int64))
            
            quantized_dict[f"{name}.packed"] = packed_tensor
            quantized_dict[f"{name}.scales"] = scales
            quantized_dict[f"{name}.shape"] = torch.tensor(original_shape, dtype=torch.int32)
            quantized_dict[f"{name}.format"] = torch.tensor([1], dtype=torch.int8)
            
            if layer_idx != -1: processed_layers.add(f"Compressed-L{layer_idx}")
        else:
            # Bias 등은 그대로 유지
            quantized_dict[name] = param

    save_file(quantized_dict, output_path)
    print(f" -> DONE. Layers processed: {sorted(list(processed_layers))}")
    print(f" -> Saved to {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    tasks = [
        ("src-tauri/models/Qwen3-VL-2B-Instruct-gguf/mmproj-Qwen3VL-2B-Instruct-F16.gguf", 
         "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/mmproj-BITSERIAL_ALL.safetensors", True),
        ("src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model.safetensors", 
         "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model-BITSERIAL_ALL.safetensors", False),
        ("src-tauri/models/Qwen3-0.6B-Instruct-gguf/model.safetensors", 
         "src-tauri/models/Qwen3-0.6B-Instruct-gguf/model-BITSERIAL_ALL.safetensors", False),
    ]
    for src, tgt, is_gguf in tasks:
        if os.path.exists(src): quantize_asymmetric_bit_serial(src, tgt, is_gguf)