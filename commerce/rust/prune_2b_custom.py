import gguf
import os
import numpy as np

def extract_and_slice_2b(input_path, output_path):
    if not os.path.exists(input_path): return
    print(f"Extracting 2B with full metadata for quantization: {output_path}")
    reader = gguf.GGUFReader(input_path)
    writer = gguf.GGUFWriter(output_path, "qwen2")

    target_h = 1024
    target_intermediate = 3072

    # Mandatory metadata for llama-quantize
    writer.add_block_count(1)
    writer.add_embedding_length(target_h)
    writer.add_feed_forward_length(target_intermediate)
    writer.add_head_count(8)
    writer.add_head_count_kv(8)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_context_length(32768)
    writer.add_rope_dimension_count(128)
    writer.add_rope_freq_base(1000000.0)

    for tensor in reader.tensors:
        name = tensor.name
        data = np.array(tensor.data).astype(np.float32).astype(np.float16)
        shape = data.shape
        rank = len(shape)

        if name == "token_embd.weight":
            if rank == 2:
                if shape[0] == 2048: sliced = data[:target_h, :]
                elif shape[1] == 2048: sliced = data[:, :target_h]
                else: sliced = data
            else: sliced = data
            writer.add_tensor(name, sliced)
        
        elif "blk.0." in name:
            if rank == 2:
                d0, d1 = shape
                s0 = target_h if d0 == 2048 else (target_intermediate if d0 == 6144 else d0)
                s1 = target_h if d1 == 2048 else (target_intermediate if d1 == 6144 else d1)
                if "attn_k" in name or "attn_v" in name:
                    s0 = target_h if d0 == 2048 else d0
                    s1 = d1 
                sliced = data[:s0, :s1]
            elif rank == 1:
                d0 = shape[0]
                s0 = target_h if d0 == 2048 else d0
                sliced = data[:s0]
            else: sliced = data
            writer.add_tensor(name, sliced)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

extract_and_slice_2b("./Qwen3-VL-2B-Instruct-BF16.gguf", "./Qwen3-2B-TextOnly-Experimental.gguf")
