import gguf
import os
import numpy as np

original_path = "Qwen3-0.6B-BF16.gguf"
output_path = "Qwen3-0.6B-BF16-SingleLayer.gguf"

reader = gguf.GGUFReader(original_path)
writer = gguf.GGUFWriter(output_path, "qwen2")

# 1. 메타데이터 복사
for field in reader.fields.values():
    if field.name.startswith("tensor_") or field.name == "general.name": continue
    if "block_count" in field.name:
        writer.add_uint32(field.name, 1)
        continue
    val = field.parts[-1]
    v_type = field.types[0]
    if v_type == gguf.GGUFValueType.ARRAY:
        writer.add_array(field.name, val.tolist() if hasattr(val, 'tolist') else list(val))
    elif v_type == gguf.GGUFValueType.UINT32: writer.add_uint32(field.name, int(val[0]))
    elif v_type == gguf.GGUFValueType.INT32: writer.add_int32(field.name, int(val[0]))
    elif v_type == gguf.GGUFValueType.FLOAT32: writer.add_float32(field.name, float(val[0]))
    elif v_type == gguf.GGUFValueType.BOOL: writer.add_bool(field.name, bool(val[0]))
    elif v_type == gguf.GGUFValueType.STRING: writer.add_string(field.name, bytes(val).decode('utf-8'))

writer.add_string("general.name", "Qwen3-0.6B-SingleLayer")

# 2. 텐서 복사 (타입 강제 지정)
for tensor in reader.tensors:
    name = tensor.name
    if "blk." not in name or "blk.0." in name:
        print(f"Adding tensor: {name}")
        # BF16(30) 데이터는 uint16 넘파이 배열로 취급해야 오프셋이 정확함
        data = tensor.data
        if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            data = data.view(np.uint16)
        
        writer.add_tensor(name, data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)

writer.close()
print("Pruning done. Please try llama-quantize again.")