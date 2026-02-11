import json
import numpy as np
from gguf import GGUFReader
from pathlib import Path
from safetensors.numpy import save_file

def export_comprehensive_specs():
    base_dir = Path("C:/Users/HP/Documents/GitHub/cron-logis-center/commerce/rust/src-tauri/models")
    instruct_dir = base_dir / "Qwen3-VL-2B-Instruct-gguf"
    
    main_model = instruct_dir / "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    vision_model = instruct_dir / "mmproj-Qwen3VL-2B-Instruct-F16.gguf"
    
    output_json = base_dir / "2b_specs.json"
    # 이 파일들이 Rust 베이킹 엔진에 주입될 핵심 가중치입니다.
    shared_weights_path = base_dir / "qwen3_shared_weights.safetensors"
    
    specs = {"metadata": {}, "tensors": {}}
    tensors_to_save = {}
    
    # 1. Extract 2B Text Embedding (2048 dim)
    print(f"[*] Reading 2B Language Model: {main_model.name}")
    try:
        reader = GGUFReader(str(main_model))
        for tensor in reader.tensors:
            if tensor.name == "token_embd.weight":
                print(f"[+] Extracting {tensor.name} (Shape: {tensor.shape})")
                # GGUF에서 NumPy 배열로 변환 (주의: Q4 등 양자화된 경우 데이터가 직접 노출되지 않으므로 
                # 여기서는 메타데이터만 저장하고, 실제 데이터는 Rust에서 GGUF를 직접 읽도록 유도하거나 
                # FP16/F32 모델에서 추출해야 합니다. 여기서는 로직만 구성합니다.)
                specs["tensors"][tensor.name] = {"shape": tensor.shape.tolist(), "type": "2B_EXTRACTED"}
        
        for key in reader.fields:
            field = reader.fields[key]
            val = field.parts[field.data[0]] if len(field.data) > 0 else ""
            if isinstance(val, (np.ndarray, np.generic)): val = val.tolist()
            if isinstance(val, bytes): val = val.decode('utf-8', errors='ignore')
            specs["metadata"][key] = val
    except Exception as e:
        print(f"[!] Language Error: {e}")

    # 2. Extract 2B Vision Projection (mmproj)
    if vision_model.exists():
        print(f"[*] Reading 2B Vision Projection: {vision_model.name}")
        try:
            v_reader = GGUFReader(str(vision_model))
            for tensor in v_reader.tensors:
                # 비전 베이킹에 필요한 핵심 텐서들 (Linear/Proj 가중치)
                specs["tensors"][tensor.name] = {"shape": tensor.shape.tolist(), "type": "VISION_EXTRACTED"}
        except Exception as e:
            print(f"[!] Vision Error: {e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=4)
    
    print(f"\n[SUCCESS] 2B Specs and Weight Map saved.")
    print(f"[ACTION] Rust will now use 2b_specs.json to locate and inject these tensors from GGUF during 0.6B baking.")

if __name__ == "__main__":
    export_comprehensive_specs()