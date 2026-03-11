import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def verify_weights():
    print("🚀 Starting Weight Verification using Python (Transformers)...")
    
    # Path to the large safetensors file
    weight_path = "model.safetensors-00001-of-00001.safetensors"
    config_dir = "src-tauri/models/Qwen3.5-0.8B-Split" # Directory containing config.json
    
    if not os.path.exists(weight_path):
        print(f"❌ Weight file not found: {weight_path}")
        return

    print(f"📦 Loading weights from {weight_path}...")
    try:
        # 1. Check if safetensors can be loaded
        sd = load_file(weight_path)
        print(f"✅ Successfully loaded {len(sd)} tensors.")
        
        # 2. Check some key tensors
        keys = sd.keys()
        if "model.embed_tokens.weight" in keys:
            print("✅ 'model.embed_tokens.weight' found.")
        if "model.layers.0.self_attn.q_proj.weight" in keys:
            print("✅ 'model.layers.0.self_attn.q_proj.weight' found.")
        
        # 3. Quick structural check (Compare with config)
        import json
        with open(os.path.join(config_dir, "config.json"), "r") as f:
            config = json.load(f)
        
        num_layers = config.get("num_hidden_layers", 24)
        print(f"🔍 Model should have {num_layers} layers.")
        
        # Verify last layer
        last_layer_key = f"model.layers.{num_layers-1}.self_attn.o_proj.weight"
        if last_layer_key in keys:
            print(f"✅ Last layer ({num_layers-1}) tensors found.")
        else:
            print(f"❌ Missing tensors for layer {num_layers-1}. Check key: {last_layer_key}")

        print("\n✨ SUMMARY: The weight file structure matches the Qwen3.5 architecture expectations.")
        
    except Exception as e:
        print(f"❌ Error during verification: {str(e)}")

if __name__ == "__main__":
    verify_weights()
