import gguf
import numpy as np
import os

# 경로 설정
path_06b = "src-tauri/models/Qwen3-0.6B-Instruct-gguf/Qwen3-0.6B-Q4_K_M.gguf"
path_2b = "Qwen3-VL-2B-Instruct-BF16.gguf"
output_path = "Qwen3-2B-TextOnly-Experimental.gguf"

# [CHECK] 0.6B 모델 텐서 및 레이어 확인
print(f"--- 0.6B Model Inspection: {path_06b} ---")
reader_06b = gguf.GGUFReader(path_06b)
blk_ids = set()
for tensor in reader_06b.tensors:
    if "blk." in tensor.name:
        # blk.N.attn_q.weight -> N 추출
        try:
            bid = int(tensor.name.split(".")[1])
            blk_ids.add(bid)
        except:
            pass

print(f"Detected Layer Indices in 0.6B: {sorted(list(blk_ids))}")
print(f"Total layers: {len(blk_ids)}")

# 임베딩 데이터 추출
embed_06b = None
for tensor in reader_06b.tensors:
    if tensor.name == "token_embd.weight":
        embed_06b = tensor.data
        break

# [PROCESS] 2B 모델 변환
print(f"\n--- Processing 2B Model: {path_2b} ---")
reader_2b = gguf.GGUFReader(path_2b)

# 아키텍처 수정: qwen2 -> qwen3 (사용자 요청 반영)
arch_str = "qwen3" 
print(f"Setting architecture to: {arch_str}")

writer = gguf.GGUFWriter(output_path, arch_str)

# 1. 메타데이터 복사 및 수정
for field in reader_2b.fields.values():
    if field.name.startswith("tensor_") or field.name == "general.architecture":
        continue
    if "visual" in field.name:
        continue
    
    # 아키텍처 이름 관련 필드가 있으면 강제로 qwen3 반영
    if field.name == "general.name":
        writer.add_string("general.name", "Qwen3-2B-TextOnly-Experimental")
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

# 2. 텐서 필터링 및 추가
print("Filtering and adding tensors...")
for tensor in reader_2b.tensors:
    # 시각 레이어 제거
    if tensor.name.startswith("visual."):
        continue
    
    # 텐서 추가
    writer.add_tensor(tensor.name, tensor.data, raw_shape=tensor.shape, raw_dtype=tensor.tensor_type)

print(f"Finalizing {output_path}...")
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

print("\nSuccess! Now run quantization:")
print(f"./llama/llama-quantize.exe {output_path} Qwen3-2B-TextOnly-Q4_K_M.gguf Q4_K_M")
