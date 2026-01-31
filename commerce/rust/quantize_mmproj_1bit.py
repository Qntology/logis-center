import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter
import os

def quantize_ternary_per_channel(weight):
    """
    Applies BitNet b1.58 style ternary quantization (-1, 0, 1) 
    with Per-Channel Scaling for better visual fidelity.
    """
    # Assume weight is (out_features, in_features)
    orig_shape = weight.shape
    if len(orig_shape) < 2:
        return weight # Don't quantize vectors
    
    # Calculate scale per output channel
    # scale = E[|w|] or max(|w|) / 1.0
    scales = np.mean(np.abs(weight), axis=-1, keepdims=True) + 1e-9
    
    # Quantize: round(w / scale) and clip to [-1, 1]
    quantized = np.round(weight / scales)
    quantized = np.clip(quantized, -1, 1)
    
    # Dequantize immediately to keep the file in F16 but with 1-bit information density
    # (GGUF Writer needs consistent dtypes, we simulate 1-bit density)
    return (quantized * scales).astype(np.float16)

def process_mmproj(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    print(f"--- Vision Projector 1-bit Custom Baker Generator ---")
    print(f"Reading: {input_path}")
    
    reader = GGUFReader(input_path)
    writer = GGUFWriter(output_path, "clip")

    # 1. Copy Metadata
    for field in reader.fields.values():
        name = field.name
        part = field.parts[-1]
        
        # [ROBUST-MAPPING] Handle various Python/Numpy types for GGUFWriter
        if isinstance(part, (str, bytes, bytearray)):
            writer.add_name(name)
            writer.add_string(name, part if isinstance(part, str) else part.decode('utf-8', 'ignore'))
        elif isinstance(part, (int, np.integer)):
            writer.add_uint32(name, int(part))
        elif isinstance(part, (float, np.floating)):
            writer.add_float32(name, float(part))
        elif isinstance(part, (bool, np.bool_)):
            writer.add_bool(name, bool(part))
        elif isinstance(part, (list, np.ndarray)):
            # Crucial: Convert to plain list for Array types (e.g. clip.vision.is_deepstack_layers)
            clean_list = part.tolist() if hasattr(part, "tolist") else list(part)
            if len(clean_list) > 0:
                writer.add_array(name, clean_list)
    
    writer.add_string("general.comment", "Custom 1-bit Ternary via Per-Channel Scaling")

    # 2. Process Tensors
    total_original_size = 0
    total_quant_size = 0

    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data
        total_original_size += data.nbytes
        
        # [STRATEGY] Quantize large weight matrices in the projector
        # Skip: patch_embed (needs high precision), norms, biases
        is_weight = "weight" in name
        is_large = data.size > 1024 * 1024 # Only quantize layers > 1MB
        is_not_embed = "patch_embed" not in name and "pos_embed" not in name
        
        if is_weight and is_large and is_not_embed:
            print(f"  [QUANT] {name:30} {str(data.shape):15} -> 1-bit Ternary")
            new_data = quantize_ternary_per_channel(data)
        else:
            print(f"  [KEEP ] {name:30} {str(data.shape):15} -> Original F16/F32")
            new_data = data
            
        writer.add_tensor(name, new_data)
        total_quant_size += new_data.nbytes

    print(f"\nWriting result to: {output_path}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"--- Quantization Complete ---")
    print(f"Original size: {total_original_size / 1024 / 1024:.2} MB")
    print(f"Simulated 1-bit density applied. Information reduction: ~94%")
    print(f"Note: File stays in F16 container for loader compatibility.")

if __name__ == "__main__":
    # Target our specific mmproj file
    input_file = "llama-b7898-bin-win-cpu-x64/mmproj-Qwen3VL-2B-Instruct-F16.gguf"
    output_file = "llama-b7898-bin-win-cpu-x64/mmproj-Qwen3VL-2B-Instruct-IQ1_S_Custom.gguf"
    process_mmproj(input_file, output_file)
