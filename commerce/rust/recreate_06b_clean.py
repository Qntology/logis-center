import gguf
import os
import numpy as np

def create_ultra_clean_shell(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
    
    print(f"Creating 1-Layer Ultra Clean Shell: {output_path}")
    reader = gguf.GGUFReader(input_path)
    writer = gguf.GGUFWriter(output_path, "qwen2")

    # 1. Core metadata - Force 1 layer
    writer.add_block_count(1)
    writer.add_embedding_length(1024)
    writer.add_feed_forward_length(3072)
    writer.add_head_count(8)
    writer.add_head_count_kv(8)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_context_length(32768)
    writer.add_rope_dimension_count(128)
    writer.add_rope_freq_base(1000000.0)

    # 2. Filter Tensors: Only keep Layer 0 and necessary base tensors
    # We zero out the data to keep it small, but keep the SHAPE so the loader works.
    keep_list = ['token_embd', 'blk.0.', 'output_norm', 'output.weight']
    
    for tensor in reader.tensors:
        name = tensor.name
        if any(x in name for x in keep_list):
            # We keep the shape but use zeros to save space during quantization
            # This makes the GGUF file act as a skeleton.
            writer.add_tensor(name, np.zeros(tensor.shape, dtype=np.float16))
        else:
            # Skip all other layers (blk.1 to blk.27)
            continue

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Successfully created ultra clean shell: {output_path}")

create_ultra_clean_shell("./Qwen3-0.6B-BF16.gguf", "./Qwen3-0.6B-Clean-L0-BF16.gguf")
