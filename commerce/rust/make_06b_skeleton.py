import gguf
import os
import numpy as np

def make_06b_ultra_skeleton(input_path, output_path):
    if not os.path.exists(input_path): return
    print(f"Creating Ultra-Skeleton (No Embedding): {output_path}")
    reader = gguf.GGUFReader(input_path)
    writer = gguf.GGUFWriter(output_path, "qwen2")

    # Essential Metadata
    writer.add_block_count(1)
    writer.add_embedding_length(1024)
    writer.add_feed_forward_length(3072)
    writer.add_head_count(8)
    writer.add_head_count_kv(8)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_context_length(32768)
    writer.add_rope_dimension_count(128)
    writer.add_rope_freq_base(1000000.0)

    # Tensor Filtering: REMOVE token_embd.weight
    # Only keep blk.0 and output components as placeholders
    keep_prefixes = ['blk.0.', 'output_norm', 'output.weight']
    
    count = 0
    for tensor in reader.tensors:
        name = tensor.name
        if any(name.startswith(p) for p in keep_prefixes):
            # Keep as tiny dummy zero tensors
            writer.add_tensor(name, np.zeros(tensor.shape, dtype=np.float16))
            count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"SUCCESS: Ultra-Skeleton created! Size: {size_mb:.2f} MB (Tensors: {count})")

make_06b_ultra_skeleton("./Qwen3-0.6B-BF16.gguf", "./Qwen3-0.6B-Clean-L0-BF16.gguf")
