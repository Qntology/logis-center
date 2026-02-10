
import sys
from pathlib import Path

def inspect_gguf(file_path):
    print(f"Inspecting GGUF: {file_path}")
    try:
        import gguf
        reader = gguf.GGUFReader(file_path)
        print(f"--- Tensors Found ({len(reader.tensors)}) ---")
        tensors = sorted(reader.tensors, key=lambda x: x.name)
        for tensor in tensors:
            print(f"Name: {tensor.name:40} | Shape: {tensor.shape} | Type: {tensor.tensor_type}")
    except ImportError:
        print("Python 'gguf' package not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Pointing to the Instruct model as requested
    p = Path(r"C:\Users\HP\Documents\GitHub\cron-logis-center\commerce\rust\src-tauri\models\Qwen3-VL-2B-Instruct-gguf\Qwen3VL-2B-Instruct-Q4_K_M.gguf")
    if p.exists():
        inspect_gguf(str(p))
    else:
        print(f"File not found: {p}")
