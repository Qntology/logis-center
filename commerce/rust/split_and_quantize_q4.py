import os
import torch
from safetensors.torch import save_file, load_file
from tqdm import tqdm

def pack_q4_combined(tensor):
    """
    Packs scales and data into a single uint8 tensor using Q4 (4-bit) quantization.
    Block size: 32. 18 bytes per block (2 bytes scale + 16 bytes data).
    Mapping: Symmetric mapping centered at 0.
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
    # Q4 Mapping: -8..7 (centered around 0)
    scales = (abs_max / 8.0).to(torch.float16)
    scales[scales == 0] = 1.0
    
    # Quantize to 0..15 (4 bits)
    # Mapping: -8.0 -> 0, 0.0 -> 8, 7.0 -> 15
    quantized = torch.round(reshaped / scales.view(-1, 1).to(torch.float32) + 8.0).clamp(0, 15).to(torch.uint8)
    
    # Pack 2 values (4 bits each) into 1 byte
    q_reshaped = quantized.view(num_blocks, 16, 2)
    packed_data = (q_reshaped[:, :, 0] << 0) | (q_reshaped[:, :, 1] << 4)
    
    scale_bytes = scales.view(torch.uint8).view(num_blocks, 2)
    combined = torch.cat([scale_bytes, packed_data], dim=1)
    
    return combined, orig_shape

def run_q4_quantization():
    print("\n[QUANT-Q4] Starting Q4 quantization for improved context...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(base_dir, "model.safetensors-00001-of-00001.safetensors")
    target_dir = os.path.join(base_dir, "src-tauri", "models", "Qwen3.5-0.8B-Split")
    
    if not os.path.exists(source_file):
        print(f"[ERROR] Source file not found: {source_file}")
        return

    print(f"[INFO] Loading original weights from {source_file}...")
    full_sd = load_file(source_file)
    
    # 1. Process layers 0 to 23
    for i in tqdm(range(24), desc="Quantizing Layers"):
        layer_prefix = f"model.language_model.layers.{i}."
        layer_sd = {}
        meta = {"precision": "q4_combined_v1"}
        
        for name, tensor in full_sd.items():
            if name.startswith(layer_prefix):
                short_name = name[len(layer_prefix):]
                if ("weight" in name) and tensor.ndim >= 2:
                    combined, orig_shape = pack_q4_combined(tensor)
                    layer_sd[short_name] = combined
                    meta[f"shape.{short_name}"] = ",".join(map(str, orig_shape))
                else:
                    layer_sd[short_name] = tensor
        
        if layer_sd:
            out_path = os.path.join(target_dir, f"layer_{i}.st")
            save_file(layer_sd, out_path, metadata=meta)

    # 2. Process Shared (Keep FP16 for embed/head as requested)
    shared_sd = {}
    shared_meta = {"precision": "q4_combined_v1"}
    shared_prefixes = ["model.language_model.embed_tokens.", "model.language_model.norm.", "model.language_model.lm_head."]
    
    for name, tensor in full_sd.items():
        is_shared = False
        for pref in shared_prefixes:
            if name.startswith(pref):
                short_name = name[len("model.language_model."):]
                is_shared = True
                break
        if is_shared:
            shared_sd[short_name] = tensor
                
    if shared_sd:
        out_path = os.path.join(target_dir, "shared.st")
        save_file(shared_sd, out_path, metadata=shared_meta)

    print("\n[SUCCESS] Q4 Quantization complete.")

if __name__ == "__main__":
    run_q4_quantization()
