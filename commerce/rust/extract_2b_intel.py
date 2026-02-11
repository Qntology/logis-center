import gguf
import os
import numpy as np

def extract_2b_full_intel_v5(input_path, output_path):
    if not os.path.exists(input_path): return
    print(f"Extracting 2B Intel Module v5 (RoPE List Fix): {output_path}")
    reader = gguf.GGUFReader(input_path)
    writer = gguf.GGUFWriter(output_path, "qwen2vl")

    # Metadata
    writer.add_block_count(1)
    writer.add_embedding_length(2048)
    writer.add_feed_forward_length(6144)
    writer.add_head_count(16)
    writer.add_head_count_kv(8)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_context_length(32768)
    writer.add_rope_dimension_count(128)
    writer.add_rope_freq_base(1000000.0)
    
    # Use standard list instead of numpy array
    writer.add_array("qwen2vl.rope.dimension_sections", [64, 64, 0, 0])

    # Extract ALL Essential Tensors
    for tensor in reader.tensors:
        name = tensor.name
        is_text_base = name in ["token_embd.weight", "output_norm.weight", "output.weight"]
        is_text_l0 = name.startswith("blk.0.")
        is_vision = any(x in name for x in ["vpm.", "visual.", "enc."])
        
        if is_text_base or is_text_l0 or is_vision:
            data = np.array(tensor.data).astype(np.float32).astype(np.float16)
            writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"SUCCESS: Unified v5 Module Created.")

extract_2b_full_intel_v5("./Qwen3-VL-2B-Instruct-BF16.gguf", "./Qwen3-2B-L0-VL-BF16.gguf")
