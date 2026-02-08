from safetensors.torch import load_file
import os

f1 = "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model-4BIT_SLICED_LAYER0.safetensors"
f2 = "src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model-4BIT_SLICED_L1_ALL.safetensors"

def check(path):
    if not os.path.exists(path):
        print("NOT FOUND:", path)
        return
    print("\nFILE:", path)
    t = load_file(path)
    keys = sorted([k for k in t.keys() if "embed" in k or "lm_head" in k])
    for k in keys:
        if k.endswith(".format"):
            print(f"  {k} -> Format: {t[k].item()}")
        elif k.endswith(".packed"):
            print(f"  {k} -> Packed (Exists)")
        elif k.endswith(".scales"):
            print(f"  {k} -> Scales (Size: {t[k].numel()})")

check(f1)
check(f2)