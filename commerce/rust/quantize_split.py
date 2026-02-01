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

def process_text_layer0(input_path, output_dir):
    """
    Generate a Baking-Only model with ONLY Layer 0.
    No lm_head, no other layers.
    """
    print(f"\n[TEXT-L0] Extracting Layer 0 for Baking: {input_path}")
    tensors = load_file(input_path)
    layer0_dict = {}
    
    for name, param in tensors.items():
        idx_match = re.search(r'(layers)\.(\d+)\.', name)
        layer_idx = int(idx_match.group(2)) if idx_match else -1
        
        # Baking-Only: Remove lm_head and layers > 0
        if "lm_head" in name or "output.weight" in name: continue
        if layer_idx > 0: continue
        
        is_weight = "weight" in name and len(param.shape) >= 2
        should_quantize = is_weight and "embed" not in name and "norm" not in name

        if should_quantize:
            packed, scales, shape = quantize_tensor_bit_serial(param)
            layer0_dict.update({
                f"{name}.packed": packed,
                f"{name}.scales": scales,
                f"{name}.shape": shape,
                f"{name}.format": torch.tensor([0], dtype=torch.int8)
            })
        else:
            layer0_dict[name] = param.to(torch.float16)

    out_path = os.path.join(output_dir, f"{os.path.basename(input_path).replace('.safetensors', '')}-BITSERIAL_LAYER0.safetensors")
    save_file(layer0_dict, out_path)
    print(f" -> Saved: {out_path} ({len(layer0_dict)} tensors)")

def process_vision_layer0(input_path, output_dir):
    """
    Generate a Baking-Only Vision model with ONLY Block 0.
    """
    print(f"\n[VISION-L0] Extracting Vision Block 0 for Baking: {input_path}")
    reader = gguf.GGUFReader(input_path)
    layer0_dict = {}
    
    for tensor in reader.tensors:
        name = tensor.name
        data = torch.from_numpy(tensor.data).to(torch.float32)
        
        new_name = name.replace("v.", "visual.") if name.startswith("v.") else name
        if not new_name.startswith("visual.") and ("blk" in new_name or "mm" in new_name or "patch" in new_name):
             new_name = f"visual.{new_name}"
        
        idx_match = re.search(r'(blk|blocks)\.(\d+)\.', new_name)
        layer_idx = int(idx_match.group(2)) if idx_match else -1
        
        # Baking-Only: Remove vision blocks > 0
        if layer_idx > 0: continue 

        is_weight = "weight" in new_name and len(data.shape) >= 2
        should_quantize = is_weight and "norm" not in new_name and "ln" not in new_name and "embed" not in new_name and "patch" not in new_name
        
        if should_quantize:
            packed, scales, shape = quantize_tensor_bit_serial(data)
            layer0_dict.update({
                f"{new_name}.packed": packed,
                f"{new_name}.scales": scales,
                f"{new_name}.shape": shape,
                f"{new_name}.format": torch.tensor([0], dtype=torch.int8)
            })
        else:
            layer0_dict[new_name] = data.to(torch.float16)
            
    out_path = os.path.join(output_dir, "mmproj-BITSERIAL_LAYER0.safetensors")
    save_file(layer0_dict, out_path)
    print(f" -> Saved: {out_path} ({len(layer0_dict)} tensors)")

if __name__ == "__main__":
    BASE_DIR = "src-tauri/models"
    
    # 1. Text Layer 0 Extraction
    for m_dir in ["Qwen3-VL-2B-Instruct-gguf", "Qwen3-0.6B-Instruct-gguf"]:
        src = os.path.join(BASE_DIR, m_dir, "model.safetensors")
        if os.path.exists(src):
            process_text_layer0(src, os.path.dirname(src))

    # 2. Vision Layer 0 Extraction
    v_src = os.path.join(BASE_DIR, "Qwen3-VL-2B-Instruct-gguf/mmproj-Qwen3VL-2B-Instruct-F16.gguf")
    if os.path.exists(v_src):
        process_vision_layer0(v_src, os.path.dirname(v_src))
