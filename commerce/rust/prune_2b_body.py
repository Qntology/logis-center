import gguf
import os

def prune_layer_0_from_quantized_model(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
    
    print(f"Pruning Layer 0 from quantized model: {input_path}")
    reader = gguf.GGUFReader(input_path)
    writer = gguf.GGUFWriter(output_path, "qwen2vl")

    # 1. Essential Metadata
    writer.add_block_count(26) # 27 layers (0-26) minus Layer 0
    writer.add_embedding_length(2048)
    writer.add_feed_forward_length(6144)
    writer.add_head_count(16)
    writer.add_head_count_kv(8)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_context_length(32768)
    writer.add_rope_dimension_count(128)
    writer.add_rope_freq_base(1000000.0)
    writer.add_array("qwen2vl.rope.dimension_sections", [64, 64, 0, 0])

    # 2. Filter Tensors
    count = 0
    skipped = 0
    for tensor in reader.tensors:
        name = tensor.name
        if name.startswith("blk.0."):
            skipped += 1
            continue
        
        # Copy raw data without re-quantizing
        writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type)
        count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print("\nSUCCESS: Pruned Body Created!")
    print(f"Path: {output_path}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"Tensors kept: {count}, Tensors removed: {skipped}")

# Execute
input_model = "src-tauri/models/Qwen3-VL-2B-Hybrid-gguf/Qwen3VL-2B-Instruct-Hybrid-Q4_K_M.gguf"
output_model = "./Qwen3-2B-Body-L1-27-Q4_K_M.gguf"
prune_layer_0_from_quantized_model(input_model, output_model)
