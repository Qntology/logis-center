from gguf import GGUFReader
import os

def inspect_gguf(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\n--- Inspecting: {file_path} ---")
    try:
        reader = GGUFReader(file_path)
        # 파일 크기 확인
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"File Size: {file_size:.2f} MB")
        
        # 텐서 정보 출력 (상위 5개만)
        print("Tensors:")
        for i, tensor in enumerate(reader.tensors):
            print(f"  - {tensor.name:30} | Type: {tensor.tensor_type.name:10} | Shape: {tensor.shape}")
            if i >= 4: break
        
        # 전체 텐서 타입 요약
        types = [t.tensor_type.name for t in reader.tensors]
        from collections import Counter
        print(f"\nType Summary: {dict(Counter(types))}")
        
    except Exception as e:
        print(f"Error reading GGUF: {e}")

# 사용자님이 언급한 두 파일 확인
path_q8 = r"src-tauri\models\Qwen3-VL-2B-Instruct-gguf\mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
path_iq1 = r"src-tauri\models\Qwen3-VL-2B-Instruct-gguf\mmproj-Qwen3VL-2B-Instruct-IQ1_S.gguf"

inspect_gguf(path_q8)
inspect_gguf(path_iq1)

