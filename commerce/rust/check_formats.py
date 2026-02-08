from safetensors.torch import load_file
import os

files = [
    "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model-4BIT_SLICED_LAYER0.safetensors",
    "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model-4BIT_SLICED_L1_ALL.safetensors"
]

for p in files:
    if not os.path.exists(p):
        print(f"File not found: {p}")
        continue
    print(f"
--- Checking: {p} ---")
    t = load_file(p)
    for k in sorted(t.keys()):
        if "head" in k or "embed" in k:
            val = t[k]
            if k.endswith(".format"):
                print(f"{k}: {val.item()} (Format 4 expected)")
            elif k.endswith(".shape"):
                print(f"{k}: {val.tolist()}")
            elif k.endswith(".scales"):
                print(f"{k}: Tensor size {val.numel()}")
            elif k.endswith(".packed"):
                print(f"{k}: Tensor size {val.numel()}")
            else:
                print(f"{k}: Standard F16 Tensor")
