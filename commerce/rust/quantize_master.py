import torch
import numpy as np
from safetensors.torch import save_file, load_file
import os
import re
import gguf

def quantize_tensor_bit_serial(param):
    """32-블록 1비트 양자화 코어 로직"""
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
    """
    통합 프로세서: 텍스트/비전, 전체/레이어0 분기를 모두 처리
    """
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
        # 레이어 제한 필터 (Baking용)
        idx_match = re.search(r'(layers|blk|blocks)\.(\d+)\.', name)
        layer_idx = int(idx_match.group(2)) if idx_match else -1
        
        if layer_limit is not None and layer_idx >= layer_limit:
            continue
            
        # 텐서 분류 (텍스트 파일에는 비전 제외, 비전 파일에는 텍스트 제외)
        has_visual_prefix = "visual" in name
        if is_vision != has_visual_prefix:
            continue

        # 양자화 수행
        if "weight" in name and len(param.shape) >= 2 and "norm" not in name and "ln" not in name:
            packed, scales, shape = quantize_tensor_bit_serial(param)
            final_dict.update({
                f"{name}.packed": packed,
                f"{name}.scales": scales,
                f"{name}.shape": shape,
                f"{name}.format": torch.tensor([0], dtype=torch.int8)
            })
        else:
            final_dict[name] = param.to(torch.float16)

    save_file(final_dict, out_path)
    print(f" -> DONE. Saved {len(final_dict)} tensors.")

if __name__ == "__main__":
    MODELS_ROOT = "src-tauri/models"
    
    # --- 1. Qwen3-VL-2B-Instruct 정리 ---
    v2b_dir = os.path.join(MODELS_ROOT, "Qwen3-VL-2B-Instruct-gguf")
    v2b_src = os.path.join(v2b_dir, "model.safetensors")
    v2b_mmproj_src = os.path.join(v2b_dir, "mmproj-Qwen3VL-2B-Instruct-F16.gguf")

    # 2B 언어: Full & Layer0
    process_model(v2b_src, v2b_dir, is_vision=False, layer_limit=None)  # model-BITSERIAL_ALL
    process_model(v2b_src, v2b_dir, is_vision=False, layer_limit=1)     # model-BITSERIAL_LAYER0
    
    # 2B 비전: Full & Layer0
    # 원본 GGUF에서 추출하여 safetensors로 변환
    process_model(v2b_mmproj_src, v2b_dir, is_vision=True, layer_limit=None) # mmproj-BITSERIAL_ALL
    process_model(v2b_mmproj_src, v2b_dir, is_vision=True, layer_limit=1)    # mmproj-BITSERIAL_LAYER0

    # --- 2. Qwen3-0.6B-Instruct 정리 ---
    s06_dir = os.path.join(MODELS_ROOT, "Qwen3-0.6B-Instruct-gguf")
    s06_src = os.path.join(s06_dir, "model.safetensors")
    
    # 0.6B 언어: Full & Layer0
    process_model(s06_src, s06_dir, is_vision=False, layer_limit=None) # model-BITSERIAL_ALL
    process_model(s06_src, s06_dir, is_vision=False, layer_limit=1)    # model-BITSERIAL_LAYER0
