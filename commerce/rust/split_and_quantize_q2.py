import os
import torch
from safetensors.torch import save_file, load_file
from tqdm import tqdm

def pack_q2_k_combined(tensor):
    """
    Packs scales and data into a single uint8 tensor.
    Block size: 32. 10 bytes per block (2 bytes scale + 8 bytes data).
    """
    orig_shape = tensor.shape
    tensor = tensor.to(torch.float32).flatten()
    block_size = 32
    padding = (block_size - (tensor.numel() % block_size)) % block_size
    if padding > 0:
        tensor = torch.cat([tensor, torch.zeros(padding)])
    
    num_blocks = tensor.numel() // block_size
    reshaped = tensor.view(num_blocks, block_size)
    
    abs_max = reshaped.abs().max(dim=1).values
    # Use 2.0 as max to allow symmetric mapping [-2, -1, 0, 1]
    scales = (abs_max / 2.0).to(torch.float16)
    scales[scales == 0] = 1.0
    
    # Quantize to 0..3 (2 bits)
    # New Mapping: -2.0 -> 0, -1.0 -> 1, 0.0 -> 2, 1.0 -> 3
    # This makes 0.0 weights (most common) map exactly to bit value 2.
    quantized = torch.round(reshaped / scales.view(-1, 1).to(torch.float32) + 2.0).clamp(0, 3).to(torch.uint8)
    
    # Pack 4 values (2 bits each) into 1 byte
    q_reshaped = quantized.view(num_blocks, 8, 4)
    packed_data = (q_reshaped[:, :, 0] << 0) | \
                  (q_reshaped[:, :, 1] << 2) | \
                  (q_reshaped[:, :, 2] << 4) | \
                  (q_reshaped[:, :, 3] << 6)
    
    scale_bytes = scales.view(torch.uint8).view(num_blocks, 2)
    combined = torch.cat([scale_bytes, packed_data], dim=1)
    
    return combined, orig_shape

def run_full_restoration():
    print("\n[RESTORE] Starting FULL restoration from original model.safetensors...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, "src-tauri", "models", "Qwen3-0.6B-Instruct-gguf")
    source_file = os.path.join(target_dir, "model.safetensors")
    
    if not os.path.exists(source_file):
        print(f"[ERROR] Source file not found: {source_file}")
        return

    print(f"[INFO] Loading original weights from {source_file}...")
    full_sd = load_file(source_file)
    
    # 1. Process layers 0 to 27 (Qwen3-0.6B has 28 layers)
    for i in tqdm(range(28), desc="Processing Layers"):
        layer_prefix = f"model.layers.{i}."
        layer_sd = {}
        meta = {"precision": "q2_combined_v1"}
        
        # Extract tensors for this layer
        for name, tensor in full_sd.items():
            if name.startswith(layer_prefix):
                short_name = name[len(layer_prefix):]
                
                # Quantize only weights/embeddings larger than 1D
                if ("weight" in name) and tensor.ndim >= 2:
                    combined, orig_shape = pack_q2_k_combined(tensor)
                    layer_sd[short_name] = combined
                    meta[f"shape.{short_name}"] = ",".join(map(str, orig_shape))
                else:
                    layer_sd[short_name] = tensor
        
        if layer_sd:
            out_path = os.path.join(target_dir, f"layer_{i}.st")
            save_file(layer_sd, out_path, metadata=meta)

    # 2. Process Shared tensors
    shared_sd = {}
    shared_meta = {"precision": "q2_combined_v1"}
    
    # Mapping for shared tensors to simplified names
    for name, tensor in full_sd.items():
        short_name = None
        if name == "model.embed_tokens.weight":
            short_name = "embed_tokens.weight"
        elif name == "model.norm.weight":
            short_name = "norm.weight"
        elif name == "lm_head.weight":
            short_name = "lm_head.weight"
            
        if short_name:
            # Apply quantization to shared weights (embed_tokens, lm_head) as requested
            if ("weight" in name) and tensor.ndim >= 2:
                print(f" - Quantizing shared tensor: {name} (Q2_K)")
                combined, orig_shape = pack_q2_k_combined(tensor)
                shared_sd[short_name] = combined
                shared_meta[f"shape.{short_name}"] = ",".join(map(str, orig_shape))
            else:
                # Store as original (FP16/BF16 for 1D tensors like norm)
                shared_sd[short_name] = tensor
                
    if shared_sd:
        # If lm_head.weight is missing but embed_tokens.weight exists, it's tied
        if "embed_tokens.weight" in shared_sd and "lm_head.weight" not in shared_sd:
             print(" - lm_head.weight not found, assuming tied weights.")
             # Some loaders expect lm_head.weight even if tied
             shared_sd["lm_head.weight"] = shared_sd["embed_tokens.weight"].clone()
             
        out_path = os.path.join(target_dir, "shared.st")
        save_file(shared_sd, out_path, metadata=shared_meta)
        print(f" - shared.st created ({len(shared_sd)} tensors)")

    print("\n[SUCCESS] Restoration complete. All files regenerated from original source.")

if __name__ == "__main__":
    run_full_restoration()
