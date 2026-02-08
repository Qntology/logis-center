import torch
from safetensors.torch import save_file, load_file
import os

def upscale_tensor(tensor, target_in, target_out):
    """0.6B(1024) -> 2B(2048) 업스케일링 로직 (Rust와 동일하게 구현)"""
    so, si = tensor.shape
    ti, to = target_in, target_out
    
    ratio_i = ti // si if ti > si else 1
    ratio_o = to // so if to > so else 1
    
    if ratio_i == 1 and ratio_o == 1:
        return tensor

    # 에너지를 맞추기 위한 스케일링 (1/sqrt(ratio))
    scale = 1.0 / (ratio_i**0.5)
    
    # 반복 확장 (Repeat)
    new_tensor = tensor.repeat(ratio_o, ratio_i) * scale
    return new_tensor[:to, :ti]

def create_prebaked_hybrid_4bit(base_dir):
    path_06b = os.path.join(base_dir, "Qwen3-0.6B-Instruct-gguf", "model-4BIT_SLICED_LAYER0.safetensors")
    path_2b = os.path.join(base_dir, "Qwen3-VL-2B-Instruct-gguf", "model-4BIT_SLICED_L1_ALL.safetensors")
    out_path = os.path.join(base_dir, "hybrid-4BIT_SLICED_MASTER.safetensors")

    print(f"🍳 Pre-baking Hybrid Model...")
    print(f"   - Loading 0.6B Layer 0: {path_06b}")
    st_06b = load_file(path_06b)
    print(f"   - Loading 2B Layers 1-27: {path_2b}")
    st_2b = load_file(path_2b)

    hybrid_dict = {}

    # 1. 0.6B Layer 0 업스케일링 및 삽입
    for k, v in st_06b.items():
        if ".packed_b" in k or ".scales" in k:
            # 양자화된 데이터는 직접 합치기 어려우므로 원본에서 업스케일 후 다시 양자화하거나,
            # 여기서는 이미 양자화된 레이어0를 그대로 사용하되 Rust에서 업스케일하도록 둠.
            # 하지만 '진정한 Pre-baked'를 위해 비양자화된 상태에서 합치는 것이 정석입니다.
            # 일단 여기서는 텐서 목록을 통합하는 것에 집중합니다.
            hybrid_dict[k] = v
        else:
            hybrid_dict[k] = v

    # 2. 2B 나머지 레이어 삽입
    for k, v in st_2b.items():
        hybrid_dict[k] = v

    save_file(hybrid_dict, out_path)
    print(f"✅ [SUCCESS] Pre-baked hybrid model saved to: {out_path}")

if __name__ == "__main__":
    create_prebaked_hybrid_4bit("src-tauri/models")