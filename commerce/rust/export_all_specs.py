import json
import numpy as np
from gguf import GGUFReader
from pathlib import Path

def export_comprehensive_specs():
    # Use forward slashes to avoid string termination issues
    instruct_dir = Path("C:/Users/HP/Documents/GitHub/cron-logis-center/commerce/rust/src-tauri/models/Qwen3-VL-2B-Instruct-gguf")
    output_json = Path("C:/Users/HP/Documents/GitHub/cron-logis-center/commerce/rust/src-tauri/models/2b_specs.json")
    
    main_model = instruct_dir / "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    # Try both mmproj versions
    vision_models = ["mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf"]
    
    specs = {"metadata": {}, "tensors": {}}
    
    # 1. Main Model (Language)
    print(f"[*] Reading Language: {main_model.name}")
    try:
        reader = GGUFReader(str(main_model))
        for key in reader.fields:
            field = reader.fields[key]
            val = field.parts[field.data[0]] if len(field.data) > 0 else ""
            if isinstance(val, (np.ndarray, np.generic)): val = val.tolist()
            if isinstance(val, bytes): val = val.decode('utf-8', errors='ignore')
            specs["metadata"][key] = val
        for tensor in reader.tensors:
            specs["tensors"][tensor.name] = tensor.shape.tolist()
    except Exception as e:
        print(f"[!] Language Error: {e}")

    # 2. Vision Model (mmproj)
    for v_name in vision_models:
        v_path = instruct_dir / v_name
        if v_path.exists():
            print(f"[*] Attempting Vision: {v_name}")
            try:
                v_reader = GGUFReader(str(v_path))
                for tensor in v_reader.tensors:
                    # Vision tensors usually start with 'v.' or have specific names
                    specs["tensors"][tensor.name] = tensor.shape.tolist()
                print(f"[+] Successfully added vision tensors from {v_name}")
                break # Stop if one succeeds
            except Exception as e:
                print(f"[!] Vision Error ({v_name}): {e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=4)
    print(f"\n[SUCCESS] Master specs saved to: {output_json}")

if __name__ == "__main__":
    export_comprehensive_specs()