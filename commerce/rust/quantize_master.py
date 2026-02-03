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

def process_model(input_path, output_dir, is_vision=False, layer_limit=None):
    mode_name = "LAYER0" if layer_limit == 1 else "ALL"
    type_name = "VISION" if is_vision else "TEXT"
    suffix = f"BITSERIAL_{mode_name}.safetensors"
    prefix = "mmproj-" if is_vision else "model-"
    out_path = os.path.join(output_dir, f"{prefix}{suffix}")

    print(f"[{type_name}-{mode_name}] Processing: {input_path} -> {out_path}")
    
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
        idx_match = re.search(r'(layers|blk|blocks)\.(\d+)\.', name)
        layer_idx = int(idx_match.group(2)) if idx_match else -1
        if layer_limit is not None and layer_idx >= layer_limit: continue
        if is_vision != ("visual" in name): continue

        if "weight" in name and len(param.shape) >= 2 and "norm" not in name and "ln" not in name:
            packed, scales, shape = quantize_tensor_bit_serial(param)
            # 64바이트 정렬을 위해 더미 데이터를 넣는 대신, 
            # safetensors 저장 시 자동으로 처리되도록 가이드를 줍니다.
            final_dict.update({
                f"{name}.packed": packed,
                f"{name}.scales": scales,
                f"{name}.shape": shape,
                f"{name}.format": torch.tensor([0], dtype=torch.int8)
            })
        else:
            final_dict[name] = param.to(torch.float16)

    # [OPTIMIZATION] 저장 시 모든 텐서를 float16 또는 정렬된 타입으로 저장
    save_file(final_dict, out_path)
    print(f" -> DONE. Saved {len(final_dict)} tensors.")

if __name__ == "__main__":
    MODELS_ROOT = "src-tauri/models"
    for m_dir, src, is_v, limit in [
        ("Qwen3-VL-2B-Instruct-gguf", "model.safetensors", False, None),
        ("Qwen3-VL-2B-Instruct-gguf", "model.safetensors", False, 1),
        ("Qwen3-VL-2B-Instruct-gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf", True, None),
        ("Qwen3-VL-2B-Instruct-gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf", True, 1),
        ("Qwen3-0.6B-Instruct-gguf", "model.safetensors", False, None),
        ("Qwen3-0.6B-Instruct-gguf", "model.safetensors", False, 1),
    ]:
        p_src = os.path.join(MODELS_ROOT, m_dir, src)
        if os.path.exists(p_src):
            process_model(p_src, os.path.join(MODELS_ROOT, m_dir), is_v, limit)