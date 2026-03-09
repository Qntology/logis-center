import os
import torch
from safetensors.torch import save_file, load_file
from tqdm import tqdm

def run_fp16_split():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "src-tauri", "models", "Qwen3.5-0.8B-Split")
    model_path = os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors")
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return

    print(f"Loading original model (FP16/BF16): {model_path}")
    full_sd = load_file(model_path)
    
    shared_tensors = {}
    layer_tensors = {i: {} for i in range(24)}
    vision_tensors = {}
    mtp_tensors = {}

    for name, tensor in tqdm(full_sd.items(), desc="Splitting Layers"):
        # Map tensors to their respective split files
        if "model.language_model.layers." in name:
            parts = name.split(".")
            layer_idx = int(parts[3])
            layer_tensors[layer_idx][name] = tensor
        elif "model.visual." in name:
            vision_tensors[name] = tensor
        elif "mtp." in name:
            mtp_tensors[name] = tensor
        else:
            shared_tensors[name] = tensor

    # Save split files
    print("Saving original quality split files...")
    save_file(shared_tensors, os.path.join(model_dir, "shared.st"))
    save_file(vision_tensors, os.path.join(model_dir, "vision.st"))
    save_file(mtp_tensors, os.path.join(model_dir, "mtp.st"))
    for i, tensors in layer_tensors.items():
        if tensors:
            save_file(tensors, os.path.join(model_dir, f"layer_{i}.st"))
            
    print("Successfully generated FP16/BF16 split layers. No quality loss.")

if __name__ == "__main__":
    run_fp16_split()
