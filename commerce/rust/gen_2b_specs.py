import json
from gguf import GGUFReader
from pathlib import Path

def export_2b_specs():
    # 경로를 분할하여 안전하게 생성
    base = Path(r"C:\Users\HP\Documents\GitHub\cron-logis-center\commerce\rust\src-tauri\models")
    model_path = base / "Qwen3-VL-2B-Hybrid-gguf" / "Qwen3VL-2B-Instruct-Hybrid-Q4_K_M.gguf"
    output_json = base / "2b_specs.json"
    
    if not model_path.exists():
        print(f"Error: {model_path} not found.")
        return

    reader = GGUFReader(str(model_path))
    specs = {
        "tensors": {t.name: t.shape.tolist() for t in reader.tensors},
        "metadata": {"hidden_size": 2048, "intermediate_size": 6144}
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=4)
    print(f"Saved {len(reader.tensors)} specs to {output_json}")

if __name__ == "__main__":
    export_2b_specs()