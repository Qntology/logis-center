import os
import torch
import numpy as np
from safetensors.torch import save_file, load_file
from tqdm import tqdm

def pack_q2_k_combined(tensor):
    """
    Packs scales and data into a single uint8 tensor to keep the original name.
    Block size: 32.
    Each block: 2 bytes (FP16 scale) + 8 bytes (32 elements * 2 bits) = 10 bytes total.
    """
    orig_shape = tensor.shape
    tensor = tensor.to(torch.float32).flatten()
    block_size = 32
    padding = (block_size - (tensor.numel() % block_size)) % block_size
    if padding > 0:
        tensor = torch.cat([tensor, torch.zeros(padding)])
    
    num_blocks = tensor.numel() // block_size
    reshaped = tensor.view(num_blocks, block_size)
    
    # Calculate scales
    abs_max = reshaped.abs().max(dim=1).values
    scales = (abs_max / 1.5).to(torch.float16)
    scales[scales == 0] = 1.0
    
    # Quantize to 0..3 (2 bits)
    # Mapping: -1.5 -> 0, -0.5 -> 1, 0.5 -> 2, 1.5 -> 3
    quantized = torch.round(reshaped / scales.view(-1, 1).to(torch.float32) + 1.5).clamp(0, 3).to(torch.uint8)
    
    # Pack data: [num_blocks, 8 bytes]
    q_reshaped = quantized.view(num_blocks, 8, 4)
    packed_data = (q_reshaped[:, :, 0] << 0) | \
                  (q_reshaped[:, :, 1] << 2) | \
                  (q_reshaped[:, :, 2] << 4) | \
                  (q_reshaped[:, :, 3] << 6)
    
    # Combine Scale (2 bytes) + Data (8 bytes) = 10 bytes per block
    scale_bytes = scales.view(torch.uint8).view(num_blocks, 2)
    combined = torch.cat([scale_bytes, packed_data], dim=1) # [num_blocks, 10]
    
    return combined, orig_shape
def run_q2_quantization():
    print("\n[QUANT-Q2] Quantizing layers and SHARED weights to save VRAM...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "src-tauri", "models", "Qwen3.5-0.8B-Split")

    # Process all .st files including shared.st
    files = [f for f in os.listdir(model_dir) if f.endswith(".st") and "vision" not in f]

    for filename in tqdm(files, desc="Processing Models"):
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            sd = load_file(path)
            new_sd = {}
            meta = {"precision": "q2_combined_v1"}

            for name, tensor in sd.items():
                # Quantize weights and embeddings
                if ("weight" in name or "embed" in name or "head" in name) and tensor.ndim >= 2:
                    combined, orig_shape = pack_q2_k_combined(tensor)
                    new_sd[name] = combined
                    meta[f"shape.{name}"] = ",".join(map(str, orig_shape))
                else:
                    new_sd[name] = tensor

            temp_path = path + ".tmp"
            save_file(new_sd, temp_path, metadata=meta)
            try:
                if os.path.exists(path): os.remove(path)
                os.rename(temp_path, path)
            except:
                print(f"\n[WARN] Failed to replace {path}. Please ensure it's not locked.")

    print("\n[SUCCESS] All layers quantized. Original tensor names preserved.")

if __name__ == "__main__":
    run_q2_quantization()
