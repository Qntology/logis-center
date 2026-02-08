import torch
import numpy as np
from safetensors.torch import save_file, load_file
import os
import re
import gguf

def quantize_tensor_4bit_sliced_shuffled(param):
    """
    [PRECISION-UPGRADE] 4-bit Multi-plane Quantization with Layout Shuffling
    Splits weights into 4 independent bit-planes for high-precision inference.
    """
    BLOCK_SIZE = 32
    N, K = param.shape[0], param.shape[1]
    
    # 1. Padding K to multiple of 32
    pad_k = (BLOCK_SIZE - (K % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_k > 0:
        param = torch.nn.functional.pad(param, (0, pad_k))
    K_padded = K + pad_k
    K_blocks = K_padded // BLOCK_SIZE

    # 2. [QUANTIZATION] Map to 0 ~ 15 (4-bit)
    flat_w = param.view(N, K_blocks, BLOCK_SIZE)
    # Get max absolute value for each block (Scaling)
    max_vals = torch.max(torch.abs(flat_w), dim=2, keepdim=True)[0]
    scales = max_vals.to(torch.float16)
    
    # Normalize and quantize to unsigned integer range (0-15)
    q_w = torch.clamp(
        torch.round(((flat_w / (max_vals + 1e-9)) + 1.0) / 2.0 * 15.0),
        0, 15
    ).to(torch.int32)

    # 3. [BIT-SLICING & SHUFFLING]
    pad_n = (8 - (N % 8)) % 8
    N_padded = N + pad_n
    
    shuffled_planes = []
    for b in range(4): # 4 Bit-planes
        bit_plane = (q_w >> b) & 1
        packed_rows = torch.zeros((N, K_blocks), dtype=torch.int32)
        for i in range(BLOCK_SIZE):
            packed_rows |= (bit_plane[:, :, i] << i)
            
        if pad_n > 0:
            packed_rows = torch.nn.functional.pad(packed_rows, (0, 0, 0, pad_n))
            
        shuffled_w = packed_rows.view(N_padded // 8, 8, K_blocks).permute(0, 2, 1).contiguous()
        shuffled_planes.append(shuffled_w.view(-1))

    if pad_n > 0:
        scales = torch.nn.functional.pad(scales.squeeze(-1), (0, 0, 0, pad_n)).unsqueeze(-1)
    shuffled_s = scales.view(N_padded // 8, 8, K_blocks).permute(0, 2, 1).contiguous()

    return shuffled_planes, shuffled_s.view(-1), torch.tensor([N, K], dtype=torch.int32)

def process_model_shuffled(input_path, output_dir, is_vision=False, layer_limit=None, layer_start=0):
    mode_name = "LAYER0" if layer_limit == 1 else ("L1_ALL" if layer_start > 0 else "ALL")
    suffix = f"4BIT_SLICED_{mode_name}.safetensors"
    prefix = "mmproj-" if is_vision else "model-"
    out_path = os.path.join(output_dir, f"{prefix}{suffix}")

    print(f"\n[SPLIT-PROCESS-4BIT-{mode_name}] Path: {input_path}")
    
    tensors = {}
    if input_path.endswith(".gguf"):
        reader = gguf.GGUFReader(input_path)
        for t in reader.tensors:
            name = t.name.replace("v.", "visual.") if t.name.startswith("v.") else t.name
            if not name.startswith("visual.") and ("blk" in name or "mm" in name or "patch" in name):
                name = f"visual.{name}"
            tensors[name] = torch.from_numpy(t.data).to(torch.float32)
    else:
        tensors = load_file(input_path)

    final_dict = {}
    for name, param in tensors.items():
        new_name = name
        if "layers." in name and "language_model" not in name:
            new_name = name.replace("model.layers", "model.language_model.layers")
        elif "model.embed_tokens" in name and "language_model" not in name:
            new_name = name.replace("model.embed_tokens", "model.language_model.embed_tokens")
        elif "model.norm" in name and "language_model" not in name:
            new_name = name.replace("model.norm", "model.language_model.norm")
        elif name.startswith("lm_head"):
            new_name = "model.language_model.lm_head" + name[7:]
            
        idx_match = re.search(r'(layers|blk|blocks|language_model\.layers)\.(\d+)\.', new_name)
        layer_idx = int(idx_match.group(2)) if idx_match else -1
        
        if layer_limit is not None:
            if layer_idx >= layer_limit: continue
            if layer_idx == -1 and ("norm" in new_name or "lm_head" in new_name): continue
            
        if layer_start > 0:
            if 0 <= layer_idx < layer_start: continue
            if layer_idx == -1 and "embed_tokens" in new_name: continue
        
        if is_vision != ("visual" in name): continue

        is_weight = "weight" in name and len(param.shape) == 2
        should_quantize = is_weight and "norm" not in name and "ln" not in name

        if should_quantize:
            planes, scales, shape = quantize_tensor_4bit_sliced_shuffled(param)
            for b, packed_data in enumerate(planes):
                final_dict[f"{new_name}.packed_b{b}"] = packed_data
            final_dict[f"{new_name}.scales"] = scales
            final_dict[f"{new_name}.shape"] = shape
            final_dict[f"{new_name}.format"] = torch.tensor([4], dtype=torch.int8) 
        else:
            final_dict[new_name] = param.to(torch.float16)

    save_file(final_dict, out_path)
    print(f" -> Saved 4-bit sliced model to {out_path}")

if __name__ == "__main__":
    BASE_DIR = "src-tauri/models"
    tasks = [
        ("Qwen3-VL-2B-Instruct-gguf", "model.safetensors", False, None, 1),
        ("Qwen3-VL-2B-Instruct-gguf", "model.safetensors", False, 1, 0),
        ("Qwen3-VL-2B-Instruct-gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf", True, None, 0),
        ("Qwen3-VL-2B-Instruct-gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf", True, 1, 0),
        ("Qwen3-0.6B-Instruct-gguf", "model.safetensors", False, None, 1),
        ("Qwen3-0.6B-Instruct-gguf", "model.safetensors", False, 1, 0),
    ]
    
    for m_dir, src_file, is_v, limit, start in tasks:
        p_src = os.path.join(BASE_DIR, m_dir, src_file)
        if os.path.exists(p_src):
            process_model_shuffled(p_src, os.path.join(BASE_DIR, m_dir), is_v, limit, start)
