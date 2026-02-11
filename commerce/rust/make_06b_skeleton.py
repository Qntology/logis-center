import gguf
import os
import numpy as np

def make_06b_skeleton(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
    
    print(f"Creating 1-Layer Skeleton from {input_path}...")
    reader = gguf.GGUFReader(input_path)
    # Architecture is qwen2
    writer = gguf.GGUFWriter(output_path, "qwen2")

    # 1. Essential Metadata for 0.6B Skeleton
    writer.add_block_count(1) # FORCE 1 LAYER
    writer.add_embedding_length(1024)
    writer.add_feed_forward_length(3072)
    writer.add_head_count(8)
    writer.add_head_count_kv(8)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_context_length(32768)
    writer.add_rope_dimension_count(128)
    writer.add_rope_freq_base(1000000.0)

    # 2. Tensor Filtering Logic
    # We only keep what's needed for Layer 0 Baking
    keep_prefixes = ['token_embd', 'blk.0.', 'output_norm', 'output.weight']
    
    found_tensors = 0
    for tensor in reader.tensors:
        name = tensor.name
        if any(name.startswith(p) for p in keep_prefixes):
            # Create a zero tensor with the SAME SHAPE
            # Using float16 to keep it compact
            writer.add_tensor(name, np.zeros(tensor.shape, dtype=np.float16))
            found_tensors += 1
            if found_tensors < 5:
                print(f"  - Keeping skeleton for: {name} {tensor.shape}")

    print(f"Total skeleton tensors added: {found_tensors}")
    
    # 3. Save
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"Successfully created 1-layer skeleton: {output_path} ({size_mb:.2f} MB)")

# Clean up previous attempts if they exist to avoid confusion
if os.path.exists("./Qwen3-0.6B-Clean-L0-BF16.gguf"):
    os.remove("./Qwen3-0.6B-Clean-L0-BF16.gguf")

make_06b_skeleton("./Qwen3-0.6B-BF16.gguf", "./Qwen3-0.6B-Clean-L0-BF16.gguf")
