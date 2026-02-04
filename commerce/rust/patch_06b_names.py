from safetensors.torch import load_file, save_file
import os
import shutil

def patch_model(path):
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found.")
        return
    
    print(f"[PATCHing] {path}...")
    tensors = load_file(path)
    new_tensors = {}
    
    for name, data in tensors.items():
        new_name = name
        if "model.layers" in name:
            new_name = name.replace("model.layers", "model.language_model.layers")
        elif "model.embed_tokens" in name:
            new_name = name.replace("model.embed_tokens", "model.language_model.embed_tokens")
        elif "model.norm" in name:
            new_name = name.replace("model.norm", "model.language_model.norm")
        elif name.startswith("lm_head"):
            new_name = "model.language_model.lm_head" + name[7:]
            
        new_tensors[new_name] = data
        
    # 임시 파일로 저장 후 이동 (파일 잠김 대비)
    tmp_path = path + ".tmp"
    save_file(new_tensors, tmp_path)
    print(f"  -> Saved to temporary file: {tmp_path}")
    
    try:
        os.remove(path)
        os.rename(tmp_path, path)
        print(f"[DONE] Successfully patched {path}")
    except Exception as e:
        print(f"[RETRY-NEEDED] Could not replace {path} directly: {e}")
        print(f"  Please close any applications using the model and manually rename {tmp_path} to {path}")

if __name__ == "__main__":
    target_dir = "src-tauri/models/Qwen3-0.6B-Instruct-gguf"
    patch_model(os.path.join(target_dir, "model-BITSERIAL_ALL.safetensors"))
    patch_model(os.path.join(target_dir, "model-BITSERIAL_LAYER0.safetensors"))
