from gguf import GGUFReader
import sys
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
        
        # 텐서 정보 출력 (모든 텐서 출력)
        print("Tensors:")
        for i, tensor in enumerate(reader.tensors):
            print(f"  - {tensor.name:50} | Type: {tensor.tensor_type.name:10} | Shape: {tensor.shape}")
        
        # 전체 텐서 타입 요약
        types = [t.tensor_type.name for t in reader.tensors]
        from collections import Counter
        print(f"\nType Summary: {dict(Counter(types))}")
        
    except Exception as e:
        print(f"Error reading GGUF: {e}")

if len(sys.argv) > 1:
    inspect_gguf(sys.argv[1])
else:
    print("Please provide a GGUF file path as an argument.")