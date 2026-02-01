import torch
import numpy as np
from safetensors.torch import save_file, load_file
import os
import gguf
import re

def quantize_to_anchor_iq0(input_path, output_path, is_gguf=False):
    mode_str = "GGUF-ANCHOR" if is_gguf else "SAFE-ANCHOR"
    print(f"[{mode_str}] Processing {input_path}...")
    
    quantized_dict = {}
    BLOCK_SIZE = 512 

    # 1. Load Tensors
    if is_gguf:
        reader = gguf.GGUFReader(input_path)
        tensors_to_process = {t.name: torch.from_numpy(t.data.copy()) for t in reader.tensors}
    else:
        tensors_to_process = load_file(input_path)

    for name, param in tensors_to_process.items():
        # [STRICT ANCHOR POLICY] 
        # Identify layer indices in names like 'model.layers.1.xxx', 'blk.1.xxx', 'v.blk.1.xxx'
        match = re.search(r'(layers|blk|deepstack)\.(\d+)\.', name)
        if match:
            layer_idx = int(match.group(2))
            if layer_idx > 0:
                continue # Skip all layers except the first one (0)

        # Quantization Logic
        if "weight" in name and len(param.shape) >= 2:
            # Embedding Special Case: 2-bit
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

            # Standard Weight: 1-bit with 512-block scaling
            original_shape = list(param.shape)
            flat_w = param.view(-1).to(torch.float32)
            pad_len = (BLOCK_SIZE - (flat_w.numel() % BLOCK_SIZE)) % BLOCK_SIZE
            if pad_len > 0: flat_w = torch.cat([flat_w, torch.zeros(pad_len)])
            
            num_blocks = flat_w.numel() // BLOCK_SIZE
            blocks = flat_w.view(num_blocks, BLOCK_SIZE)
            scales = torch.max(torch.abs(blocks), dim=1)[0].to(torch.float16)
            binary = (blocks >= 0)
            
            packed = torch.zeros((num_blocks, BLOCK_SIZE // 8), dtype=torch.uint8)
            for i in range(8):
                packed |= (binary[:, i::8].to(torch.uint8) << i)
            
            quantized_dict[f"{name}.packed"] = packed.view(-1)
            quantized_dict[f"{name}.scales"] = scales
            quantized_dict[f"{name}.shape"] = torch.tensor(original_shape, dtype=torch.int32)
        else:
            quantized_dict[name] = param

    save_file(quantized_dict, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f" -> DONE. {output_path} size: {size_mb:.2f} MB")

if __name__ == "__main__":
    # 1. 0.6B Language Model
    src_06b = "src-tauri/models/Qwen3-0.6B-Instruct-gguf/model.safetensors"
    tgt_06b = "src-tauri/models/Qwen3-0.6B-Instruct-gguf/Qwen3-0.6B-UD-ANCHOR_IQ0.safetensors"
    if os.path.exists(src_06b):
        quantize_to_anchor_iq0(src_06b, tgt_06b, is_gguf=False)

    # 2. 2B Language Model (Main)
    src_2b = "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model.safetensors"
    tgt_2b = "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/Qwen3VL-2B-Instruct-ANCHOR_IQ0.safetensors"
    if os.path.exists(src_2b):
        quantize_to_anchor_iq0(src_2b, tgt_2b, is_gguf=False)

    # 3. Vision mmproj
    src_mm = "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/mmproj-Qwen3VL-2B-Instruct-F16.gguf"
    tgt_mm = "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/mmproj-Qwen3VL-2B-Instruct-ANCHOR_IQ0.safetensors"
    if os.path.exists(src_mm):
        quantize_to_anchor_iq0(src_mm, tgt_mm, is_gguf=True)