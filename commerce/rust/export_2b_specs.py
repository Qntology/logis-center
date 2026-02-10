
import json
from gguf import GGUFReader
from pathlib import Path

def export_2b_specs():
    input_path = r"C:\Users\HP\Documents\GitHub\cron-logis-center\commerceust\src-tauri\models\Qwen3-VL-2B-Hybrid-gguf\Qwen3VL-2B-Instruct-Hybrid-Q4_K_M.gguf"
    output_json = r"C:\Users\HP\Documents\GitHub\cron-logis-center\commerceust\src-tauri\models\2b_specs.json"
    
    print(f"[*] 2B 모델 설계도 분석 중: {input_path}")
    reader = GGUFReader(input_path)
    
    specs = {
        "tensors": {},
        "metadata": {}
    }
    
    # 1. 모든 텐서의 이름과 Shape 저장
    for tensor in reader.tensors:
        specs["tensors"][tensor.name] = tensor.shape.tolist()
        
    # 2. 주요 아키텍처 정보(Hidden Size 등) 추출
    # GGUF 필드에서 정보를 가져오거나 텐서 차원에서 유추
    if "blk.0.attn_norm.weight" in specs["tensors"]:
        specs["metadata"]["hidden_size"] = specs["tensors"]["blk.0.attn_norm.weight"][0]
    
    if "blk.0.ffn_up.weight" in specs["tensors"]:
        # [In, Out] 또는 [Out, In] 중 큰 쪽이 Intermediate Size
        shape = specs["tensors"]["blk.0.ffn_up.weight"]
        specs["metadata"]["intermediate_size"] = max(shape)

    print(f"[*] 추출된 핵심 스펙: {specs['metadata']}")
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=4)
    print(f"[+] 설계도 저장 완료: {output_json}")

if __name__ == "__main__":
    export_2b_specs()
