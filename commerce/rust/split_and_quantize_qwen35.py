import os
import torch
import numpy as np
from safetensors.torch import save_file, load_file
from tqdm import tqdm

def quantize_q4_0(tensor):
    """
    Quantize to Q4_0 (4-bit).
    Format: [scale (fp16), data (uint8, 2 elements per byte)]
    Block size: 32
    """
    if tensor.ndim < 2: return tensor # Skip small vectors like bias
    
    original_shape = tensor.shape
    # Flatten to multiples of 32
    flat = tensor.flatten().to(torch.float32)
    n_elements = flat.numel()
    padding = (32 - (n_elements % 32)) % 32
    if padding > 0:
        flat = torch.cat([flat, torch.zeros(padding)])
    
    n_blocks = flat.numel() // 32
    blocks = flat.view(n_blocks, 32)
    
    # Get max abs for each block
    abs_max, _ = torch.max(torch.abs(blocks), dim=1)
    scales = abs_max / 7.0
    
    # Quantize to -8..7
    # Note: Q4_0 usually uses a slightly different range but this is standard for candle
    qs = torch.round(blocks / scales.unsqueeze(1)).clamp(-8, 7).to(torch.int8)
    
    # Offset to 0..15 for packing
    qs_offset = (qs + 8).to(torch.uint8)
    
    # Pack 2 elements per byte
    # low nibble = el[i], high nibble = el[i+16] (following GGUF style)
    low = qs_offset[:, :16]
    high = qs_offset[:, 16:]
    packed = (low | (high << 4)).flatten()
    
    return {
        "scales": scales.to(torch.float16),
        "data": packed,
        "shape": torch.tensor(original_shape, dtype=torch.int32)
    }

def quantize_q8_0(tensor):
    """
    Quantize to Q8_0 (8-bit).
    Block size: 32
    Format: [scale (fp16), data (int8)]
    """
    if tensor.ndim < 2: return tensor
    
    original_shape = tensor.shape
    flat = tensor.flatten().to(torch.float32)
    n_elements = flat.numel()
    padding = (32 - (n_elements % 32)) % 32
    if padding > 0:
        flat = torch.cat([flat, torch.zeros(padding)])
        
    n_blocks = flat.numel() // 32
    blocks = flat.view(n_blocks, 32)
    
    abs_max, _ = torch.max(torch.abs(blocks), dim=1)
    scales = abs_max / 127.0
    
    qs = torch.round(blocks / scales.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
    
    return {
        "scales": scales.to(torch.float16),
        "data": qs.flatten(),
        "shape": torch.tensor(original_shape, dtype=torch.int32)
    }

def run_hybrid_quantization():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "src-tauri", "models", "Qwen3.5-0.8B-Split")
    
    # 1. Quantize Text Layers (Q4)
    for i in range(24):
        path = os.path.join(model_dir, f"layer_{i}.st")
        if os.path.exists(path):
            print(f"Applying actual Q4_0 packing to {path}...")
            sd = load_file(path)
            new_sd = {}
            for name, tensor in sd.items():
                if "weight" in name and tensor.ndim >= 2:
                    q = quantize_q4_0(tensor)
                    new_sd[f"{name}.scales"] = q["scales"]
                    new_sd[f"{name}.data"] = q["data"]
                    new_sd[f"{name}.shape"] = q["shape"]
                else:
                    new_sd[name] = tensor # Keep bias or small tensors as is
            save_file(new_sd, path, metadata={"quantization": "q4_0_packed"})

    # 2. Quantize Shared (Q8)
    shared_path = os.path.join(model_dir, "shared.st")
    if os.path.exists(shared_path):
        print(f"Applying actual Q8_0 packing to {shared_path}...")
        sd = load_file(shared_path)
        new_sd = {}
        for name, tensor in sd.items():
            if "weight" in name and tensor.ndim >= 2:
                q = quantize_q8_0(tensor)
                new_sd[f"{name}.scales"] = q["scales"]
                new_sd[f"{name}.data"] = q["data"]
                new_sd[f"{name}.shape"] = q["shape"]
            else:
                new_sd[name] = tensor
        save_file(new_sd, shared_path, metadata={"quantization": "q8_0_packed"})

    print("\n[SUCCESS] Actual block-wise quantization complete.")
    print("Files updated with .scales and .data packed format.")

if __name__ == "__main__":
    run_hybrid_quantization()
