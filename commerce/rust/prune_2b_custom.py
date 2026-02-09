import gguf
import numpy as np
import os

path_2b = "Qwen3-VL-2B-Instruct-BF16.gguf"
output_path = "Qwen3-2B-VisionLive-NoEmbed-L1-27-BF16.gguf"

print(f"--- Custom Pruning (Fixed Arch): Keeping Vision, Removing Embed & Layer 0 ---")
reader = gguf.GGUFReader(path_2b)

# 호환성을 위해 qwen2 아키텍처 사용
arch_str = "qwen2"
writer = gguf.GGUFWriter(output_path, arch_str)

# 1. 메타데이터 복사 및 수정
print(f"Processing Metadata (Mapping to {arch_str})...")
for field in reader.fields.values():
    name = field.name
    if name.startswith("tensor_") or name == "general.architecture":
        continue
    
    # 키값 접두사 변경 (qwen3vl.* -> qwen2.*)
    new_name = name.replace("qwen3vl.", "qwen2.")
    
    # 레이어 개수 수정 (28 -> 27)
    if "block_count" in new_name:
        writer.add_uint32(new_name, 27)
        continue

    val = field.parts[-1]
    v_type = field.types[0]
    if v_type == gguf.GGUFValueType.ARRAY:
        writer.add_array(new_name, val.tolist() if hasattr(val, 'tolist') else list(val))
    elif v_type == gguf.GGUFValueType.UINT32: writer.add_uint32(new_name, int(val[0]))
    elif v_type == gguf.GGUFValueType.INT32: writer.add_int32(new_name, int(val[0]))
    elif v_type == gguf.GGUFValueType.FLOAT32: writer.add_float32(new_name, float(val[0]))
    elif v_type == gguf.GGUFValueType.BOOL: writer.add_bool(new_name, bool(val[0]))
    elif v_type == gguf.GGUFValueType.STRING: writer.add_string(new_name, bytes(val).decode('utf-8'))

# 2. 텐서 필터링
print("Processing Tensors...")
for tensor in reader.tensors:
    name = tensor.name
    
    if name == "token_embd.weight" or "blk.0." in name:
        continue
    
    new_name = name
    if "blk." in name:
        parts = name.split(".")
        try:
            old_idx = int(parts[1])
            new_idx = old_idx - 1
            parts[1] = str(new_idx)
            new_name = ".".join(parts)
        except:
            pass

    data = tensor.data
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        data = data.view(np.uint16)
        
    writer.add_tensor(new_name, data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)

print(f"Finalizing {output_path}...")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()