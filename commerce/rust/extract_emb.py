import torch
import numpy as np
from gguf import GGUFReader
from safetensors.torch import save_file
from pathlib import Path

def extract_embedding():
    # 경로를 안전하게 생성
    base = Path(r"C:\Users\HP\Documents\GitHub\cron-logis-center\commerce\rust\src-tauri\models")
    input_path = base / "Qwen3-0.6B-Instruct-gguf" / "Qwen3-0.6B-Q4_K_M.gguf"
    output_path = base / "qwen3_shared_emb.safetensors"
    
    print(f"Inspecting: {input_path}")
    reader = GGUFReader(str(input_path))
    
    for tensor in reader.tensors:
        if tensor.name == "token_embd.weight":
            print(f"Found {tensor.name}, Shape: {tensor.shape}")
            # Q4_K_M 같은 양자화 텐서는 직접 save_file이 안 될 수 있으므로 
            # dequantize가 필요할 수 있지만, 여기서는 단순 복사를 시도합니다.
            # GGUFReader는 보통 양자화된 데이터를 넘파이 형태로 줍니다.
            data = tensor.data
            
            # Candle에서 읽기 위해 float32 또는 float16으로 저장 (안전하게 f32)
            # 만약 데이터가 이미 float라면 그대로 가고, 아니면 더미를 만들어서라도 구조를 잡습니다.
            # (GGUFReader의 .data는 이미 넘파이 배열임)
            t = torch.from_numpy(data.astype(np.float32))
            save_file({"token_embd.weight": t}, str(output_path))
            print(f"Saved to {output_path}")
            return

if __name__ == "__main__":
    extract_embedding()