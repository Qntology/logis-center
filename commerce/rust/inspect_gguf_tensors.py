from safetensors.torch import load_file
import os

path = "src-tauri/models/Qwen3-0.6B-Instruct-gguf/model-BITSERIAL_LAYER0.safetensors"
if os.path.exists(path):
    tensors = load_file(path)
    print("--- [TENSOR KEYS] ---")
    for k in sorted(tensors.keys()):
        print(k)
else:
    print(f"File not found: {path}")