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
    scales = torch.mean(torch.abs(flat_w), dim=2).to(torch.float16)
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

def process_model(input_path, output_dir, is_vision=False, layer_limit=None, layer_start=0):
    mode_name = "LAYER0" if layer_limit == 1 else ("L1_ALL" if layer_start > 0 else "ALL")
    suffix = f"BITSERIAL_{mode_name}.safetensors"
    prefix = "mmproj-" if is_vision else "model-"
    out_path = os.path.join(output_dir, f"{prefix}{suffix}")

    print(f"[{'VISION' if is_vision else 'TEXT'}-{mode_name}] Shuffling Layout: {input_path}")
    
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
        # [NAMING-UNITY] 0.6B 모델의 이름을 2B 모델과 일치하도록 완벽 변환
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
        
        # [OPTIMIZATION] 중복 저장 방지를 위한 칼같은 분할
        if layer_limit is not None: # LAYER0 모드 (시작 조각)
            if layer_idx >= layer_limit: continue
            # 시작 조각에는 임베딩은 포함하고, 최종 출력층(Head/Norm)은 제외
            if layer_idx == -1 and ("norm" in new_name or "lm_head" in new_name): continue
        
        if layer_start > 0: # L1_ALL 모드 (본체)
            if 0 <= layer_idx < layer_start: continue
            # 본체에는 임베딩은 제외하고, 최종 출력층(Head/Norm)은 포함
            if layer_idx == -1 and "embed_tokens" in new_name: continue
        
        if is_vision != ("visual" in name): continue

        is_weight = "weight" in name and len(param.shape) == 2
        # [UNIFIED-QUANT] 임베딩, 패치 등을 포함한 모든 가중치 행렬을 양자화 대상으로 포함
        should_quantize = is_weight and "norm" not in name and "ln" not in name

        if should_quantize:
            packed, scales, shape = quantize_tensor_bit_serial_shuffled(param)
            final_dict.update({
                f"{new_name}.packed": packed,
                f"{new_name}.scales": scales,
                f"{new_name}.shape": shape,
                f"{new_name}.format": torch.tensor([1], dtype=torch.int8) 
            })
        else:
            final_dict[new_name] = param.to(torch.float16)

    save_file(final_dict, out_path)
    print(f" -> DONE. Shuffled model saved to {out_path}")

if __name__ == "__main__":
    MODELS_ROOT = "src-tauri/models"
    tasks = [
        # (디렉토리, 소스파일, 비전여부, layer_limit, layer_start)
        ("Qwen3-VL-2B-Instruct-gguf", "model.safetensors", False, None, 1),
        ("Qwen3-VL-2B-Instruct-gguf", "model.safetensors", False, 1, 0),
        ("Qwen3-VL-2B-Instruct-gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf", True, None, 0),
        ("Qwen3-VL-2B-Instruct-gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf", True, 1, 0), # [NEW] 비전 0번 레이어 조각 추가
        ("Qwen3-0.6B-Instruct-gguf", "model.safetensors", False, None, 1),
        ("Qwen3-0.6B-Instruct-gguf", "model.safetensors", False, 1, 0),
    ]
    for m_dir, src, is_v, limit, start in tasks:
        p_src = os.path.join(MODELS_ROOT, m_dir, src)
        if os.path.exists(p_src):
            process_model(p_src, os.path.join(MODELS_ROOT, m_dir), is_v, limit, start)
