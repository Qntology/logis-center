import sys
import numpy as np
from gguf import GGUFReader
from safetensors.numpy import save_file
import os

def pack_weights_optimized(weight):
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
    return packed_weights, scales

def convert_to_ultra_low_bit(input_path, output_path):
    if not os.path.exists(input_path): 
        print(f"File not found: {input_path}")
        return
    
    reader = GGUFReader(input_path)
    tensors_to_save = {}

    print(f"--- [0.6B ONLY] Packing: {os.path.basename(input_path)} ---")
    
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data
        # GGUF shapes are reversed compared to numpy
        logical_shape = tuple(reversed(tensor.shape.tolist()))
        
        if "weight" in name and np.prod(logical_shape) > 2048:
            total_elements = np.prod(logical_shape)
            flat_data = data.flatten().astype(np.float32)
            
            # Critical fix: Align raw data buffer with logical shape
            if flat_data.size != total_elements:
                if flat_data.size < total_elements:
                    flat_data = np.append(flat_data, np.zeros(total_elements - flat_data.size))
                else:
                    flat_data = flat_data[:total_elements]

            if "token_embd" in name:
                print(f"  [EMBD] {name:30} | 2-bit | Shape: {logical_shape}")
                max_val = np.max(np.abs(flat_data))
                scale = max_val / 1.5
                q = np.clip(np.round(flat_data / (scale + 1e-9) + 1.5), 0, 3).astype(np.uint8)
                tensors_to_save[f"{name}.packed"] = q
                tensors_to_save[f"{name}.scale"] = np.array([scale], dtype=np.float32)
                tensors_to_save[f"{name}.shape"] = np.array(logical_shape, dtype=np.int32)
            else:
                print(f"  [PACK] {name:30} | 1.06 bpw | Shape: {logical_shape}")
                packed, scales = pack_weights_optimized(flat_data)
                tensors_to_save[f"{name}.packed"] = packed
                tensors_to_save[f"{name}.scales"] = scales
                tensors_to_save[f"{name}.shape"] = np.array(logical_shape, dtype=np.int32)
        else:
            tensors_to_save[name] = data

    print(f"Saving to: {output_path}")
    save_file(tensors_to_save, output_path)
    print(f"--- SUCCESS ---")

if __name__ == "__main__":
    # ONLY process 0.6B model with native 1024 dimensions.
    convert_to_ultra_low_bit("src-tauri/models/Qwen3-0.6B-Instruct-gguf/Qwen3-0.6B-UD-IQ1_S.gguf", 
                             "src-tauri/models/Qwen3-0.6B-Instruct-gguf/Qwen3-0.6B-UD-TRUE_IQ0.safetensors")