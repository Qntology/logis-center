import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

def quantize_to_i8(tensor):
    """
    텐서를 8비트(I8)로 양자화하고 스케일 값을 함께 반환합니다.
    """
    if tensor.dtype not in [torch.float16, torch.bfloat16, torch.float32] or tensor.numel() < 128:
        return None, None

    flat = tensor.flatten().float()
    abs_max = flat.abs().max()
    if abs_max == 0:
        return None, None
    
    scale = abs_max / 127.0
    quantized = (flat / scale).round().clamp(-128, 127).to(torch.int8)
    
    return quantized.view(tensor.shape), scale.to(torch.float16)

def quantize_dict_to_i8(tensors_dict):
    new_dict = {}
    for name, tensor in tensors_dict.items():
        q8_data, q8_scale = quantize_to_i8(tensor)
        if q8_scale is not None:
            new_dict[f"{name}.q8_data"] = q8_data
            new_dict[f"{name}.q8_scale"] = q8_scale
        else:
            new_dict[name] = tensor
    return new_dict

def quantize_tensor_q2(tensor, block_size=32):
    """LLM 레이어용 Q2_K 양자화 (4:1 패킹)"""
    if tensor.dtype not in [torch.float16, torch.bfloat16, torch.float32] or tensor.numel() < 128:
        return None, None, None
    orig_shape = list(tensor.shape)
    flat = tensor.flatten().float()
    align = max(block_size, 4)
    pad = (align - (flat.numel() % align)) % align
    if pad > 0: flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.view(-1, block_size)
    abs_max, _ = blocks.abs().max(dim=1, keepdim=True)
    abs_max[abs_max == 0] = 1.0
    scales = abs_max / 1.5
    normalized = ((flat / scales.repeat_interleave(block_size)) + 1.5)
    q_vals = torch.clamp(torch.round(normalized), 0, 3).to(torch.uint8)
    q_packed = (q_vals[0::4] << 6) | (q_vals[1::4] << 4) | (q_vals[2::4] << 2) | q_vals[3::4]
    return q_packed, scales.half(), torch.tensor(orig_shape, dtype=torch.int32)

def process_model(model_path):
    print(f"\n[START] Processing model: {model_path}")
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        print(f"[ERROR] model.safetensors not found in {model_path}")
        return

    # 기존 생성된 파일들 삭제 (혼동 방지)
    for f in os.listdir(model_path):
        if f.endswith(".st") and (f.startswith("layer_") or f in ["shared.st", "vision.st"]):
            os.remove(os.path.join(model_path, f))

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}

    layers = {}
    shared = {}
    vision = {}

    for name, tensor in tensors.items():
        # [PREFIX-FIX] Qwen3-VL-2B 은 'model.visual' 과 'model.language_model.layers' 형식을 가짐
        if "visual" in name:
            # "model.visual.blocks.0..." -> "visual.blocks.0..." (Rust 로더 기대 형식)
            clean_name = name.replace("model.visual", "visual")
            vision[clean_name] = tensor
        elif "layers." in name:
            # "model.language_model.layers.0..." -> "model.layers.0..."
            clean_name = name.replace("model.language_model.layers", "model.layers")
            parts = clean_name.split(".")
            try:
                l_idx = parts[parts.index("layers") + 1]
                if l_idx not in layers: layers[l_idx] = {}
                layers[l_idx][clean_name] = tensor
            except (ValueError, IndexError):
                shared[clean_name] = tensor
        else:
            # "model.language_model.embed_tokens..." -> "model.embed_tokens..."
            clean_name = name.replace("model.language_model.", "model.")
            shared[clean_name] = tensor

    print(f"[INFO] Found {len(layers)} LLM layers and {len(vision)} vision tensors.")

    # 1. LLM 레이어 양자화 (Q2_K)
    for idx, layer_tensors in tqdm(layers.items(), desc="Quantizing LLM Layers (Q2)"):
        new_layer_dict = {}
        for name, tensor in layer_tensors.items():
            if "weight" in name and tensor.numel() > 1024:
                packed, scales, shape = quantize_tensor_q2(tensor)
                if packed is not None:
                    new_layer_dict[f"{name}.q2_packed"] = packed
                    new_layer_dict[f"{name}.q2_scales"] = scales
                    new_layer_dict[f"{name}.q2_shape"] = shape
                    continue
            new_layer_dict[name] = tensor
        save_file(new_layer_dict, os.path.join(model_path, f"layer_{idx}.st"))

    # 2. Shared 텐서 양자화 (I8) -> shared.st
    if shared:
        print(f"  > Quantizing Shared weights to 8-bit -> shared.st")
        q8_shared = quantize_dict_to_i8(shared)
        save_file(q8_shared, os.path.join(model_path, "shared.st"))

    # 3. Vision 텐서 양자화 (I8) -> vision.st (존재할 때만 생성)
    if vision:
        print(f"  > Quantizing Vision model ({len(vision)} tensors) to 8-bit -> vision.st")
        q8_vision = quantize_dict_to_i8(vision)
        save_file(q8_vision, os.path.join(model_path, "vision.st"))
    else:
        print(f"  > No vision tensors found. Skipping vision.st")
    
    print(f"[DONE] Model processed successfully.\n")

if __name__ == "__main__":
    paths = [
        r"C:\Users\HP\Documents\GitHub\cron-logis-center\commerce\rust\src-tauri\models\Qwen3-VL-2B-Instruct-gguf",
        r"C:\Users\HP\Documents\GitHub\cron-logis-center\commerce\rust\src-tauri\models\Qwen3-0.6B-Instruct-gguf"
    ]
    for p in paths:
        if os.path.exists(p): process_model(p)
