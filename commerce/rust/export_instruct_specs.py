
import json
import numpy as np
from gguf import GGUFReader
from pathlib import Path

def export_instruct_2b_specs():
    # Use raw string for absolute path to avoid escape issues
    instruct_dir = Path(r"C:\Users\HP\Documents\GitHub\cron-logis-center\commerce\rust\src-tauri\models\Qwen3-VL-2B-Instruct-gguf")
    output_json = Path(r"C:\Users\HP\Documents\GitHub\cron-logis-center\commerce\rust\src-tauri\models\2b_specs.json")
    
    main_model_path = instruct_dir / "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    vision_model_path = instruct_dir / "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    
    specs = {"metadata": {}, "tensors": {}, "vision_metadata": {}, "vision_tensors": {}}
    
    # 1. Main Model Specs (Language)
    print(f"[*] Extracting Language specs from: {main_model_path}")
    main_reader = GGUFReader(str(main_model_path))
    for key in main_reader.fields:
        field = main_reader.fields[key]
        val = field.parts[field.data[0]] if len(field.data) > 0 else ""
        if isinstance(val, (np.ndarray, np.generic)): val = val.tolist()
        if isinstance(val, bytes): val = val.decode('utf-8', errors='ignore')
        specs["metadata"][key] = val
    for tensor in main_reader.tensors:
        specs["tensors"][tensor.name] = tensor.shape.tolist()

    # 2. Vision Model Specs (mmproj)
    if vision_model_path.exists():
        print(f"[*] Extracting Vision specs from: {vision_model_path}")
        try:
            vis_reader = GGUFReader(str(vision_model_path))
            for key in vis_reader.fields:
                field = vis_reader.fields[key]
                val = field.parts[field.data[0]] if len(field.data) > 0 else ""
                if isinstance(val, (np.ndarray, np.generic)): val = val.tolist()
                if isinstance(val, bytes): val = val.decode('utf-8', errors='ignore')
                specs["vision_metadata"][key] = val
            for tensor in vis_reader.tensors:
                specs["vision_tensors"][tensor.name] = tensor.shape.tolist()
                specs["tensors"][tensor.name] = tensor.shape.tolist()
        except Exception as e:
            print(f"[!] Error reading Vision model: {e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=4)
    print(f"[+] Comprehensive specs saved to {output_json}")

if __name__ == "__main__":
    export_instruct_2b_specs()
