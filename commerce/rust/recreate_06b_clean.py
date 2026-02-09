import gguf
import numpy as np
import os

original_path = "Qwen3-0.6B-BF16.gguf"
output_path = "Qwen3-0.6B-Clean-L0-BF16.gguf"

print(f"--- Recreating Clean 0.6B Model (Precise qwen3 Specs) ---")
reader = gguf.GGUFReader(original_path)
writer = gguf.GGUFWriter(output_path, "qwen3")

# 1. 필수 메타데이터 강제 주입
writer.add_uint32("qwen3.embedding_length", 1024)
writer.add_uint32("qwen3.block_count", 1)
writer.add_uint32("qwen3.context_length", 40960)
writer.add_uint32("qwen3.feed_forward_length", 3072)
writer.add_uint32("qwen3.attention.head_count", 16)
writer.add_uint32("qwen3.attention.head_count_kv", 8)
writer.add_float32("qwen3.attention.layer_norm_rms_epsilon", 1e-6) # 필수 키 추가
writer.add_float32("qwen3.rope.freq_base", 1000000.0)

# 기타 메타데이터 복사
for field in reader.fields.values():
    name = field.name
    if any(k in name for k in ["embedding_length", "block_count", "context_length", "feed_forward_length", "attention.", "rope.", "general.architecture"]):
        continue
    
    val = field.parts[-1]
    v_type = field.types[0]
    if v_type == gguf.GGUFValueType.ARRAY:
        writer.add_array(name, val.tolist() if hasattr(val, 'tolist') else list(val))
    elif v_type == gguf.GGUFValueType.UINT32: writer.add_uint32(name, int(val[0]))
    elif v_type == gguf.GGUFValueType.INT32: writer.add_int32(name, int(val[0]))
    elif v_type == gguf.GGUFValueType.FLOAT32: writer.add_float32(name, float(val[0]))
    elif v_type == gguf.GGUFValueType.BOOL: writer.add_bool(name, bool(val[0]))
    elif v_type == gguf.GGUFValueType.STRING: writer.add_string(name, bytes(val).decode('utf-8'))

# 2. 텐서 복사
print("Processing Tensors...")
for tensor in reader.tensors:
    name = tensor.name
    if "blk.0." in name or name == "token_embd.weight" or name == "output_norm.weight":
        shape = tensor.shape
        if name == "token_embd.weight" and shape[0] == 1024:
            shape = (shape[1], shape[0])
        
        data = tensor.data
        if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            data = data.view(np.uint16)
        writer.add_tensor(name, data, raw_shape=shape, raw_dtype=tensor.tensor_type)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print("Done.")