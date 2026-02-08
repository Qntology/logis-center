from safetensors.torch import load_file
import os

def export_tensor_names(model_path, output_txt):
    if not os.path.exists(model_path):
        print(f"[ERROR] File not found: {model_path}")
        return
    
    print(f"[READING] {model_path}...")
    try:
        tensors = load_file(model_path)
        names = sorted(tensors.keys())
        with open(output_txt, "w", encoding="utf-8") as f:
            for name in names:
                f.write(name + "\n")
        print(f"[SUCCESS] Saved {len(names)} names to {output_txt}")
    except Exception as e:
        print(f"[FAILED] Error reading {model_path}: {e}")

if __name__ == "__main__":
    export_tensor_names("src-tauri/models/Qwen3-0.6B-Instruct-gguf/model.safetensors", "names_0.6b.txt")
    export_tensor_names("src-tauri/models/Qwen3-VL-2B-Instruct-gguf/model.safetensors", "names_2b.txt")