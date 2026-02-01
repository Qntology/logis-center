import torch
import numpy as np
from safetensors.torch import save_file, load_file
import os
import re
import gguf

def quantize_tensor_bit_serial(param):
    """32-block 1-bit Quantization"""
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

def process_text_inference(input_path, output_dir):
    print(f"\n[TEXT-INF] Extracting Inference Layers (1~N): {input_path}")
    tensors = load_file(input_path)
    inf_dict = {}
    
    for name, param in tensors.items():
        idx_match = re.search(r'(layers)\.(\d+)\.', name)
        layer_idx = int(idx_match.group(2)) if idx_match else -1
        
        # [INFERENCE-ONLY FILTER]
        # Layer 0 and Embedding are excluded (They live in Baking Model)
        if layer_idx == 0: continue
        if "embed_tokens" in name: continue
        
        # Keep everything else (Layer 1~N, Norm, Head)
        is_weight = "weight" in name and len(param.shape) >= 2
        should_quantize = is_weight and "norm" not in name and "lm_head" not in name

        if should_quantize:
            packed, scales, shape = quantize_tensor_bit_serial(param)
            inf_dict.update({
                f"{name}.packed": packed,
                f"{name}.scales": scales,
                f"{name}.shape": shape,
                f"{name}.format": torch.tensor([0], dtype=torch.int8)
            })
        else:
            inf_dict[name] = param.to(torch.float16)

    out_path = os.path.join(output_dir, f"{os.path.basename(input_path).replace('.safetensors', '')}-BITSERIAL_INFERENCE.safetensors")
    save_file(inf_dict, out_path)
    print(f" -> Saved Inference-Only Model: {out_path} ({len(inf_dict)} tensors)")

def process_vision_inference(input_path, output_dir):
    print(f"\n[VISION-INF] Extracting Vision Inference Blocks (1~N): {input_path}")
    reader = gguf.GGUFReader(input_path)
    inf_dict = {}
    
    for tensor in reader.tensors:
        name = tensor.name
        data = torch.from_numpy(tensor.data).to(torch.float32)
        new_name = name.replace("v.", "visual.") if name.startswith("v.") else name
        if not new_name.startswith("visual.") and ("blk" in new_name or "mm" in new_name or "patch" in new_name):
             new_name = f"visual.{new_name}"
        
        idx_match = re.search(r'(blk|blocks)\.(\d+)\.', new_name)
        layer_idx = int(idx_match.group(2)) if idx_match else -1
        
        # [INFERENCE-ONLY FILTER]
        # Block 0, PatchEmbed, PosEmbed excluded
        if layer_idx == 0: continue
        if "patch_embed" in new_name or "patch_embd" in new_name: continue
        if "pos_embed" in new_name or "position_embd" in new_name: continue

        is_weight = "weight" in new_name and len(data.shape) >= 2
        should_quantize = is_weight and "norm" not in new_name and "ln" not in new_name
        
        if should_quantize:
            packed, scales, shape = quantize_tensor_bit_serial(data)
            inf_dict.update({
                f"{new_name}.packed": packed,
                f"{new_name}.scales": scales,
                f"{new_name}.shape": shape,
                f"{new_name}.format": torch.tensor([0], dtype=torch.int8)
            })
        else:
            inf_dict[new_name] = data.to(torch.float16)
            
    out_path = os.path.join(output_dir, "mmproj-BITSERIAL_INFERENCE.safetensors")
    save_file(inf_dict, out_path)
    print(f" -> Saved Vision Inference-Only Model: {out_path} ({len(inf_dict)} tensors)")

if __name__ == "__main__":
    BASE_DIR = "src-tauri/models"
    
    # 1. Text Inference Extraction
    for m_dir in ["Qwen3-VL-2B-Instruct-gguf", "Qwen3-0.6B-Instruct-gguf"]:
        src = os.path.join(BASE_DIR, m_dir, "model.safetensors")
        if os.path.exists(src):
            process_text_inference(src, os.path.dirname(src))

    # 2. Vision Inference Extraction
    v_src = os.path.join(BASE_DIR, "Qwen3-VL-2B-Instruct-gguf/mmproj-Qwen3VL-2B-Instruct-F16.gguf")
    if os.path.exists(v_src):
        process_vision_inference(v_src, os.path.dirname(v_src))
