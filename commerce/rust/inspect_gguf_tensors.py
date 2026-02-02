import torch
from safetensors.torch import load_file
import numpy as np
import sys

def debug_quantized_tensors(file_path):
    print(f"\n[1] Loading Tensors from: {file_path}")
    try:
        tensors = load_file(file_path)
    except Exception as e:
        print(f"FAILED to load safetensors: {e}")
        return

    # 64k 시퀀스 시뮬레이션
    SEQ_LEN = 64970
    print(f"\n[2] Sequence Length Check: {SEQ_LEN} tokens")
    
    # 메모리 계산 (Attention Mask)
    # Mask는 보통 (SEQ_LEN, SEQ_LEN) 크기의 float16 또는 float32 텐서
    mask_memory_gb = (SEQ_LEN * SEQ_LEN * 2) / (1024**3) # f16 기준
    print(f"  -> Predicted Attention Mask VRAM: {mask_memory_gb:.2f} GB")
    
    if mask_memory_gb > 4.0:
        print("  !! ALERT !! Mask size exceeds available VRAM (3.93GB). This is likely the cause of CUDA_ERROR_INVALID_VALUE.")

    # 비트 직렬화 복원 테스트
    print("\n[3] Testing Bit-serial Dequantization Logic...")
    
    # 가장 큰 텐서 중 하나 선택 (q_proj 등)
    target_key = "model.layers.0.self_attn.q_proj.weight"
    if f"{target_key}.packed" in tensors:
        packed = tensors[f"{target_key}.packed"]
        scales = tensors[f"{target_key}.scales"]
        shape = tensors[f"{target_key}.shape"].tolist()
        
        print(f"  Target: {target_key}")
        print(f"  Packed Shape: {packed.shape}, DType: {packed.dtype}")
        print(f"  Scales Shape: {scales.shape}, DType: {scales.dtype}")
        print(f"  Original Shape: {shape}")

        # 복원 로직 시뮬레이션
        try:
            # 32비트 언패킹
            # 파이썬 int32 -> uint32 변환
            p_uint32 = packed.numpy().view(np.uint32)
            
            # 일부 블록만 테스트 (속도 위해)
            num_test_blocks = 10
            s_val = scales[0].item()
            b_val = p_uint32[0]
            
            reconstructed = []
            for bit in range(32):
                val = s_val * (1.0 if (b_val >> bit) & 1 else -1.0)
                reconstructed.append(val)
            
            print(f"  Sample Decoded Values (Block 0): {reconstructed[:5]}...")
            
            if np.isnan(reconstructed).any() or np.isinf(reconstructed).any():
                print("  !! ALERT !! Found NaN or Inf in reconstructed weights.")
            else:
                print("  -> Dequantization math looks STABLE.")
                
        except Exception as e:
            print(f"  !! ERROR during dequantization simulation: {e}")

if __name__ == "__main__":
    path = "src-tauri/models/Qwen3-0.6B-Instruct-gguf/model-BITSERIAL_LAYER0.safetensors"
    debug_quantized_tensors(path)
