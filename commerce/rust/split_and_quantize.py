import os
import torch
from safetensors.torch import save_file, load_file
from tqdm import tqdm

def quantize_q4(tensor):
    """Simple 4-bit quantization (Symmetric) to save VRAM"""
    if tensor.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
        return tensor
    
    # [FIX] Keep small tensors (norms, etc) as-is
    if tensor.numel() < 1024:
        return tensor

    # [Q4] Scale to [-8, 7]
    scale = tensor.abs().max() / 7.0
    if scale == 0: return tensor
    
    # Quantize and cast to Int8 (standard safetensors support)
    q_tensor = (tensor / scale).round().to(torch.int8)
    # Note: We store scale to dequantize in Rust (optional, or just use BF16 for now)
    return q_tensor, scale

def split_model(input_file, output_dir, quantize=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Opening {input_file}...")
    state_dict = load_file(input_file)
    
    shared_tensors = {}
    layer_tensors = {} # layer_idx -> dict
    
    for name, tensor in tqdm(state_dict.items(), desc="Splitting"):
        if "layers." in name:
            # name format: model.layers.0.input_layernorm.weight
            parts = name.split(".")
            layer_idx = int(parts[2])
            if layer_idx not in layer_tensors:
                layer_tensors[layer_idx] = {}
            layer_tensors[layer_idx][name] = tensor
        else:
            shared_tensors[name] = tensor
            
    # Save Shared
    print("Saving shared.st...")
    save_file(shared_tensors, os.path.join(output_dir, "shared.st"))
    
    # Save Layers
    for idx, tensors in tqdm(layer_tensors.items(), desc="Saving Layers"):
        output_path = os.path.join(output_dir, f"layer_{idx}.st")
        save_file(tensors, output_path)
        
    print(f"Successfully split into {len(layer_tensors)} layers in {output_dir}")

if __name__ == "__main__":
    split_model("model.safetensors", "src-tauri/models/Qwen3-0.6B-Split")
