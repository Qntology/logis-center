import os
import torch
from safetensors.torch import save_file, load_file
from tqdm import tqdm

def split_model(input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Opening {input_file}...")
    # 단일 파일이므로 load_file 직접 호출
    state_dict = load_file(input_file)
    
    shared_tensors = {}
    vision_tensors = {}
    mtp_tensors = {}
    layer_tensors = {} # layer_idx -> dict
    
    for name, tensor in tqdm(state_dict.items(), desc="Categorizing Tensors"):
        # 1. Vision Tensors
        if "model.visual" in name:
            vision_tensors[name] = tensor
        
        # 2. Language Model Layers
        elif "model.language_model.layers." in name:
            # Name format: model.language_model.layers.14.linear_attn...
            parts = name.split(".")
            # layers 다음 숫자가 layer index (parts[3])
            try:
                layer_idx = int(parts[3])
                if layer_idx not in layer_tensors:
                    layer_tensors[layer_idx] = {}
                layer_tensors[layer_idx][name] = tensor
            except (ValueError, IndexError):
                shared_tensors[name] = tensor
                
        # 3. MTP (Multi-Token Prediction) Tensors
        elif name.startswith("mtp."):
            mtp_tensors[name] = tensor
            
        # 4. Shared / Global Tensors
        else:
            shared_tensors[name] = tensor
            
    # Save Categorized Files
    print("\n[1/4] Saving shared.st...")
    save_file(shared_tensors, os.path.join(output_dir, "shared.st"))
    
    if vision_tensors:
        print("[2/4] Saving vision.st...")
        save_file(vision_tensors, os.path.join(output_dir, "vision.st"))
    
    if mtp_tensors:
        print("[3/4] Saving mtp.st...")
        save_file(mtp_tensors, os.path.join(output_dir, "mtp.st"))
    
    print("[4/4] Saving Individual Layers...")
    for idx, tensors in tqdm(layer_tensors.items(), desc="Saving Layers"):
        output_path = os.path.join(output_dir, f"layer_{idx}.st")
        save_file(tensors, output_path)
        
    print(f"\nSuccessfully split Qwen 3.5 into {len(layer_tensors)} layers in {output_dir}")

if __name__ == "__main__":
    # 실제 파일명에 맞춰 경로 설정
    input_model = "model.safetensors-00001-of-00001.safetensors"
    output_path = "src-tauri/models/Qwen3.5-Split"
    
    if os.path.exists(input_model):
        split_model(input_model, output_path)
    else:
        print(f"Error: {input_model} not found.")
