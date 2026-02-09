import torch
import numpy as np
from safetensors.torch import save_file, load_file
import os
import re
import gguf

def quantize_tensor_bit_serial_shuffled(param):
    """
    [OPTIMIZED] 1-bit Quantization with Layout Shuffling
    Rearranges weights into [N/8, K/32, 8] for direct AVX2 register loading.
    """
    BLOCK_SIZE = 32
    N, K = param.shape[0], param.shape[1]
    
    # 1. Padding K to multiple of 32
    pad_k = (BLOCK_SIZE - (K % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_k > 0:
        param = torch.nn.functional.pad(param, (0, pad_k))
    K_padded = K + pad_k
    K_blocks = K_padded // BLOCK_SIZE

    # 2. Basic Bit-packing [N, K_blocks]
    flat_w = param.view(N, K_blocks, BLOCK_SIZE)
    # Scales are also stored per block
    scales = torch.mean(torch.abs(flat_w), dim=2)
    
    # [2026-TERNARY-NOISE-BLOCK] Force bottom 20% energy blocks to absolute zero
    # Ensure float32 for quantile calculation
    threshold = torch.quantile(scales.to(torch.float32), 0.20)
    scales[scales < threshold] = 0.0
    scales = scales.to(torch.float16)

    binary = (flat_w >= 0).to(torch.int32)
    
    packed_rows = torch.zeros((N, K_blocks), dtype=torch.int32)
    for i in range(BLOCK_SIZE):
        packed_rows |= (binary[:, :, i] << i)

    # 3. [LAYOUT SHUFFLING] Group 8 output channels together
    pad_n = (8 - (N % 8)) % 8
    if pad_n > 0:
        packed_rows = torch.nn.functional.pad(packed_rows, (0, 0, 0, pad_n))
        scales = torch.nn.functional.pad(scales, (0, 0, 0, pad_n))
    
    N_padded = N + pad_n
    # Change layout to [N/8, K_blocks, 8]
    shuffled_w = packed_rows.view(N_padded // 8, 8, K_blocks).permute(0, 2, 1).contiguous()
    shuffled_s = scales.view(N_padded // 8, 8, K_blocks).permute(0, 2, 1).contiguous()

    return shuffled_w.view(-1), shuffled_s.view(-1), torch.tensor([N, K], dtype=torch.int32)

def quantize_tensor_4bit_sliced(param):
    """
    [UPGRADED] 4-bit Block-wise Sliced Quantization
    Provides high-precision scaling for every 32 elements.
    """
    BLOCK_SIZE = 32
    N, K = param.shape[0], param.shape[1]
    
    # 1. Padding K to multiple of 32
    pad_k = (BLOCK_SIZE - (K % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_k > 0:
        param = torch.nn.functional.pad(param, (0, pad_k))
    K_padded = K + pad_k
    K_blocks = K_padded // BLOCK_SIZE

    # 2. Block-wise Scaling (Crucial for Decimal Expressiveness)
    # Calculate scale for every 32-element block
    flat_w = param.view(N, K_blocks, BLOCK_SIZE)
    scales = (flat_w.abs().max(dim=2).values / 7.0).to(torch.float16)
    scales[scales == 0] = 1.0
    
    # 3. Quantize using block-wise scales
    # q_val will be in range [-8, 7]
    q_val = (flat_w / scales.unsqueeze(2)).round().clamp(-8, 7).to(torch.int32)
    q_val += 8 # Offset to [0, 15] for bit-plane extraction

    # 4. Bit-plane extraction & Layout Shuffling
    planes = []
    for b in range(4):
        plane_bits = (q_val >> b) & 1
        
        # Packing [N, K_blocks, BLOCK_SIZE] -> [N, K_blocks] (int32)
        packed_rows = torch.zeros((N, K_blocks), dtype=torch.int32)
        for i in range(BLOCK_SIZE):
            packed_rows |= (plane_bits[:, :, i] << i)

        # Shuffle N-dimension for AVX/CUDA efficiency
        pad_n = (8 - (N % 8)) % 8
        if b == 0: # Only once
            if pad_n > 0:
                packed_rows_padded = torch.nn.functional.pad(packed_rows, (0, 0, 0, pad_n))
                scales_padded = torch.nn.functional.pad(scales, (0, 0, 0, pad_n))
            else:
                packed_rows_padded = packed_rows
                scales_padded = scales
        else:
            packed_rows_padded = torch.nn.functional.pad(packed_rows, (0, 0, 0, pad_n)) if pad_n > 0 else packed_rows

        N_padded = N + pad_n
        shuffled_w = packed_rows_padded.view(N_padded // 8, 8, K_blocks).permute(0, 2, 1).contiguous()
        planes.append(shuffled_w.view(-1))

    # Also shuffle the scales to match the weight layout!
    N_padded = N + (8 - (N % 8)) % 8
    scales_padded = torch.nn.functional.pad(scales, (0, 0, 0, (8 - (N % 8)) % 8)) if (N % 8) != 0 else scales
    shuffled_s = scales_padded.view(N_padded // 8, 8, K_blocks).permute(0, 2, 1).contiguous()

    return torch.cat(planes), shuffled_s.view(-1), torch.tensor([N, K], dtype=torch.int32)

def process_model_shuffled(input_path, output_dir, is_vision=False, layer_limit=None, layer_start=0):
    mode_name = "LAYER0" if layer_limit == 1 else ("L1_ALL" if layer_start > 0 else "ALL")
    suffix = f"4BIT_SLICED_{mode_name}.safetensors"
    prefix = "mmproj-" if is_vision else "model-"
    out_path = os.path.join(output_dir, f"{prefix}{suffix}")

    print(f"\n[PROCESS-{mode_name}] Layout: 4-bit Sliced | Path: {input_path}")
    
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

    # [FORCE-LM-HEAD] If lm_head is missing (Weight Tying), duplicate embed_tokens as lm_head
    has_head = any("lm_head" in k for k in tensors.keys())
    if not has_head:
        embed_key = next((k for k in tensors.keys() if "embed_tokens" in k), None)
        if embed_key:
            print(f"  -> [AUTO-HEAD] Creating lm_head from {embed_key}")
            tensors["model.language_model.lm_head.weight"] = tensors[embed_key].clone()

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
        # [2026-LANGUAGE-GUARD] Keep LM Head in FP16 to preserve vocabulary nuances
        is_lm_head = "lm_head" in new_name
        should_quantize = is_weight and "norm" not in name and "ln" not in name and not is_lm_head

        if should_quantize:
            # [2026-STRATEGIC-HYBRID] Protect Intelligence with 4-bit Attention
            is_attention = any(x in new_name for x in ["self_attn", "attn"])
            is_embedding = "embed_tokens" in new_name
            
            if is_attention or is_embedding or layer_idx == 0:
                # Use 4-bit for core intelligence
                packed, scales, shape = quantize_tensor_4bit_sliced(param)
                format_type = 4
            else:
                # Use 1-bit for heavy MLP layers to save memory
                packed, scales, shape = quantize_tensor_bit_serial_shuffled(param)
                format_type = 1
                
            final_dict.update({
                f"{new_name}.packed": packed,
                f"{new_name}.scales": scales,
                f"{new_name}.shape": shape,
                f"{new_name}.format": torch.tensor([format_type], dtype=torch.int8) 
            })
        else:
            final_dict[new_name] = param.to(torch.float16)

    save_file(final_dict, out_path)
    print(f" -> Saved to {out_path}")

if __name__ == "__main__":
    BASE_DIR = "src-tauri/models"
    
    # 텍스트 및 비전 모델 리스트
    tasks = [
        # (디렉토리, 소스파일, 비전여부, layer_limit, layer_start)
        ("Qwen3-VL-2B-Instruct-gguf", "model.safetensors", False, None, 1),
        ("Qwen3-VL-2B-Instruct-gguf", "model.safetensors", False, 1, 0),
        ("Qwen3-VL-2B-Instruct-gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf", True, None, 0),
        ("Qwen3-VL-2B-Instruct-gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf", True, 1, 0), # [NEW] 비전 0번 레이어 조각 추가
        ("Qwen3-0.6B-Instruct-gguf", "model.safetensors", False, None, 1),
        ("Qwen3-0.6B-Instruct-gguf", "model.safetensors", False, 1, 0),
    ]
    
    for m_dir, src_file, is_v, limit, start in tasks:
        p_src = os.path.join(BASE_DIR, m_dir, src_file)
        if os.path.exists(p_src):
            process_model_shuffled(p_src, os.path.join(BASE_DIR, m_dir), is_v, limit, start)
        else:
            print(f"[SKIP] Source not found: {p_src}")