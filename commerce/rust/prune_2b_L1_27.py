import gguf
import numpy as np
import os

path_2b = "Qwen3-VL-2B-Instruct-BF16.gguf"
output_path = "Qwen3-2B-TextOnly-L1-27-BF16.gguf"

print(f"--- Pruning 2B Model: Layers 1-27 Only ---")
reader = gguf.GGUFReader(path_2b)
writer = gguf.GGUFWriter(output_path, "qwen2") 

# 1. 메타데이터 복사 및 수정
print("Processing Metadata...")
for field in reader.fields.values():
    name = field.name
    if name.startswith("tensor_") or name == "general.architecture":
        continue
    if "visual" in name:
        continue
    
    # 레이어 개수 수정 (28 -> 27)
    if "block_count" in name:
        writer.add_uint32(name, 27)
        print(f"Updated {name} to 27")
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

# 2. 텐서 필터링 및 리네이밍
print("Processing Tensors...")
for tensor in reader.tensors:
    name = tensor.name
    
    if name.startswith("visual.") or "blk.0." in name:
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

print("\nDone! Please run llama-quantize next.")