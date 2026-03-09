import os
import torch
from safetensors.torch import save_file, load_file
from tqdm import tqdm

def quantize_tensor_q2(tensor, block_size=32):
    if tensor.dtype not in [torch.float16, torch.bfloat16, torch.float32] or tensor.numel() < 128:
        return tensor, None, None

    orig_shape = list(tensor.shape)
    flat = tensor.flatten().float()
    
    align = max(block_size, 4)
    pad = (align - (flat.numel() % align)) % align
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])
    
    blocks = flat.view(-1, block_size)
    abs_max, _ = blocks.abs().max(dim=1, keepdim=True)
    abs_max[abs_max == 0] = 1.0
    scales = abs_max / 1.5 
    
    normalized = (blocks / scales) + 1.5
    q_vals = torch.clamp(torch.round(normalized), 0, 3).to(torch.uint8)
    
    q_flat = q_vals.view(-1)
    q_packed = (q_flat[0::4] << 6) | (q_flat[1::4] << 4) | (q_flat[2::4] << 2) | q_flat[3::4]
    
    return q_packed, scales.half(), torch.tensor(orig_shape, dtype=torch.int32)

def run_split_and_quantize():
    # Target directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "src-tauri", "models", "Qwen3.5-0.8B-Split")
    model_path = os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors")
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return

    print(f"Loading original model: {model_path}")
    full_sd = load_file(model_path)
    
    shared_tensors = {}
    layer_tensors = {i: {} for i in range(24)}
    vision_tensors = {}
    mtp_tensors = {}

    for name, tensor in tqdm(full_sd.items(), desc="Quantizing & Splitting"):
        target_dict = shared_tensors
        
        # Mapping to split groups
        if "model.language_model.layers." in name:
            parts = name.split(".")
            layer_idx = int(parts[3])
            target_dict = layer_tensors[layer_idx]
        elif "model.visual." in name:
            target_dict = vision_tensors
        elif "mtp." in name:
            target_dict = mtp_tensors
            
        # Quantize target layers
        # Only quantize Linear weights (not biases, not norms, not embeddings)
        is_linear_weight = any(x in name for x in [".weight", "in_proj", "out_proj"]) and "norm" not in name and "embed" not in name and "A_log" not in name and "dt_bias" not in name
        
        if is_linear_weight and tensor.numel() > 1024:
            packed, scales, shape = quantize_tensor_q2(tensor)
            if scales is not None:
                # IMPORTANT: Keep original name for packed data
                target_dict[name] = packed
                target_dict[f"{name}.q2_scales"] = scales
                target_dict[f"{name}.q2_shape"] = shape
            else:
                target_dict[name] = tensor
        else:
            target_dict[name] = tensor

    # Save split files
    print("Saving split files...")
    save_file(shared_tensors, os.path.join(model_dir, "shared.st"))
    save_file(vision_tensors, os.path.join(model_dir, "vision.st"))
    save_file(mtp_tensors, os.path.join(model_dir, "mtp.st"))
    for i, tensors in layer_tensors.items():
        if tensors:
            save_file(tensors, os.path.join(model_dir, f"layer_{i}.st"))
            
    print("Successfully generated Q2 split layers while preserving original tensor names.")

if __name__ == "__main__":
    run_split_and_quantize()
