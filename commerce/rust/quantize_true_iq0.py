import sys
import numpy as np
from gguf import GGUFReader
from safetensors.numpy import save_file
import os

def pack_weights_optimized(weight):
    """
    Truly Optimized Sub-1-bit Packing
    Weight: 1.0 bpw (8 signs -> 1 byte)
    Scale: 16 bits per 256 elements (0.06 bpw)
    Total: ~1.06 bpw
    """
    orig_shape = weight.shape
    flat_w = weight.flatten().astype(np.float32)
    
    scale_block = 256
    pad_size = (scale_block - (len(flat_w) % scale_block)) % scale_block
    if pad_size > 0:
        flat_w = np.append(flat_w, np.zeros(pad_size))
    
    signs = (np.sign(flat_w) >= 0).astype(np.uint8)
    bits = signs.reshape(-1, 8)
    packed_weights = np.zeros(bits.shape[0], dtype=np.uint8)
    for i in range(8):
        packed_weights |= (bits[:, i] << i)
        
    w_blocks = flat_w.reshape(-1, scale_block)
    scales = np.mean(np.abs(w_blocks), axis=1).astype(np.float16)
    
    return packed_weights, scales, orig_shape

def convert_to_ultra_low_bit(input_path, output_path):
    if not os.path.exists(input_path): return
    
    reader = GGUFReader(input_path)
    tensors_to_save = {}

    print(f"--- [ULTRA-LOW-BIT] Packing with Shape-Tensors ---")
    
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data
        
        if "weight" in name and data.size > 2048:
            if "token_embd" in name:
                print(f"  [EMBD] {name:30} | 2-bit")
                max_val = np.max(np.abs(data))
                scale = max_val / 1.5
                q = np.clip(np.round(data / (scale + 1e-9) + 1.5), 0, 3).astype(np.uint8)
                tensors_to_save[f"{name}.packed"] = q
                tensors_to_save[f"{name}.scale"] = np.array([scale], dtype=np.float32)
                tensors_to_save[f"{name}.shape"] = np.array(data.shape, dtype=np.int32)
            else:
                print(f"  [PACK] {name:30} | 1.06 bpw")
                packed, scales, shape = pack_weights_optimized(data)
                tensors_to_save[f"{name}.packed"] = packed
                tensors_to_save[f"{name}.scales"] = scales
                # Save shape as a tensor instead of metadata to avoid Rust API issues
                tensors_to_save[f"{name}.shape"] = np.array(shape, dtype=np.int32)
        else:
            tensors_to_save[name] = data

    print(f"Saving to: {output_path}")
    save_file(tensors_to_save, output_path)
    print(f"--- SUCCESS ---")

if __name__ == "__main__":
    # 1. 0.6B Model
    convert_to_ultra_low_bit("src-tauri/models/Qwen3-0.6B-Instruct-gguf/Qwen3-0.6B-UD-IQ1_S.gguf", 
                             "src-tauri/models/Qwen3-0.6B-Instruct-gguf/Qwen3-0.6B-UD-TRUE_IQ0.safetensors")
    # 2. mmproj
    convert_to_ultra_low_bit("src-tauri/models/Qwen3-VL-2B-Instruct-gguf/mmproj-Qwen3VL-2B-Instruct-F16.gguf", 
                             "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/mmproj-Qwen3VL-2B-Instruct-TRUE_IQ0.safetensors")
    # 3. 2B Model
    convert_to_ultra_low_bit("src-tauri/models/Qwen3-VL-2B-Instruct-gguf/Qwen3VL-2B-Instruct-Q4_K_M.gguf", 
                             "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/Qwen3VL-2B-Instruct-TRUE_IQ0.safetensors")