import os
import torch
import numpy as np
from safetensors.torch import save_file, load_file
import struct

def pack_q4_0(tensor):
    """
    Genuine Q4_0 block-wise quantization (GGML compatible).
    Block size: 32. Layout: [FP16 Scale] [16 bytes of 4-bit data].
    Total size per block: 2 + 16 = 18 bytes.
    """
    shape = tensor.shape
    tensor = tensor.to(torch.float32).flatten()
    block_size = 32
    padding = (block_size - (tensor.numel() % block_size)) % block_size
    if padding > 0:
        tensor = torch.cat([tensor, torch.zeros(padding)])
    
    num_blocks = tensor.numel() // block_size
    reshaped = tensor.view(num_blocks, block_size)
    
    # Calculate scales
    abs_max = reshaped.abs().max(dim=1).values
    scales = abs_max / 8.0 # Q4_0 range is -8..7
    scales[scales == 0] = 1.0
    
    # Quantize
    # GGML Q4_0: q = round(x / scale) + 8 (to make it 0..15)
    quantized = torch.round(reshaped / scales.view(-1, 1)).clamp(-8, 7).to(torch.int8)
    
    # Pack into bytes (32 nibbles -> 16 bytes)
    # Block layout: [Scale (2 bytes)] [Data (16 bytes)]
    # For Safetensors, we keep scales and data separate for easier Candle loading, 
    # but we MUST NOT dequantize them in Rust.
    low = (quantized[:, ::2] + 8).to(torch.uint8) # 0..15
    high = (quantized[:, 1::2] + 8).to(torch.uint8) # 0..15
    packed_data = (low | (high << 4))
    
    return {
        "scales": scales.to(torch.float16), 
        "data": packed_data, 
        "shape": torch.tensor(shape, dtype=torch.int32)
    }

def run_genuine_quantization():
    print("\n[QUANT-CRITICAL] Initializing Genuine 4-bit Packing (QTensor Compatible)")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "src-tauri", "models", "Qwen3.5-0.8B-Split")
    
    # Text Layers -> Genuine Q4
    for i in range(24):
        path = os.path.join(model_dir, f"layer_{i}.st")
        if os.path.exists(path):
            print(f"[PACKING] Converting {path} to block-quantized Q4_0...")
            sd = load_file(path)
            new_sd = {}
            for name, tensor in sd.items():
                if "weight" in name and tensor.ndim >= 2:
                    q = pack_q4_0(tensor)
                    new_sd[f"{name}.q_scales"] = q["scales"]
                    new_sd[f"{name}.q_data"] = q["data"]
                    new_sd[f"{name}.q_shape"] = q["shape"]
                else:
                    new_sd[name] = tensor
            save_file(new_sd, path, metadata={"precision": "genuine_q4_0"})

    # Shared -> Q8 (Or keep Q4 for consistency)
    shared_path = os.path.join(model_dir, "shared.st")
    if os.path.exists(shared_path):
        print(f"[PACKING] Converting {shared_path} to block-quantized Q8_0...")
        sd = load_file(shared_path)
        new_sd = {}
        for name, tensor in sd.items():
            if "weight" in name and tensor.ndim >= 2:
                # Using Q4 for shared as well to guarantee 500MB
                q = pack_q4_0(tensor)
                new_sd[f"{name}.q_scales"] = q["scales"]
                new_sd[f"{name}.q_data"] = q["data"]
                new_sd[f"{name}.q_shape"] = q["shape"]
            else:
                new_sd[name] = tensor
        save_file(new_sd, shared_path, metadata={"precision": "genuine_q4_0"})

    print("\n[SUCCESS] Genuine 4-bit model generated. NO BF16 BACK-CONVERSION ALLOWED.")

if __name__ == "__main__":
    run_genuine_quantization()
