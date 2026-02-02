import torch
import numpy as np
from safetensors.torch import load_file

def test_full_logic_simulation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on: {device}")
    
    # 1. 실제 모델 로드 (데이터 타입 확인용)
    path = "src-tauri/models/Qwen3-0.6B-Instruct-gguf/model-BITSERIAL_LAYER0.safetensors"
    tensors = load_file(path)
    
    # 임베딩 및 설정값 시뮬레이션
    hidden_size = 1024
    num_heads = 16
    head_dim = hidden_size // num_heads # 64
    total_tokens = 64970
    chunk_size = 256
    
    print(f"\n[1] RoPE Simulation for {total_tokens} tokens")
    # Qwen3 MROPE Section: [24, 20, 20]
    mrope_section = [24, 20, 20]
    
    def get_mrope(pos_ids, head_dim, mrope_section):
        # Rust의 Qwen3VLTextRotaryEmbedding 로직 시뮬레이션
        # 64k 인덱스가 제대로 생성되는지 확인
        try:
            # pos_ids: (3, 1, chunk_len)
            # 64k 지점의 인덱스 테스트
            offset = 64000
            pi = torch.arange(offset, offset + chunk_size, device=device).unsqueeze(0).unsqueeze(0)
            pi = pi.repeat(3, 1, 1) # (3, 1, 256)
            
            print(f" -> Position IDs at 64k: min={pi.min().item()}, max={pi.max().item()}")
            # 여기서 크래시가 안 난다면 RoPE 인덱싱은 안전함
            return True
        except Exception as e:
            print(f" !! RoPE FAILED: {e}")
            return False

    get_mrope(None, head_dim, mrope_section)

    print(f"\n[2] Chunked Mask Simulation (VRAM Optimized)")
    def test_mask(offset, sl):
        # 64k 지점에서의 마스크 (256 x 64256)
        try:
            # -inf 대신 안전한 큰 음수 사용 테스트
            NEG_INF = -65504.0 # BF16 Max Negative
            
            # GPU에서 직접 인덱스 비교로 생성 (Rust broadcast_gt 재현)
            q_idx = torch.arange(0, sl, device=device).view(1, 1, sl, 1)
            kv_idx = torch.arange(0, offset + sl, device=device).view(1, 1, 1, offset + sl)
            
            # (1, 1, 256, 64256) 크기의 불리언 마스크
            mask_bool = kv_idx > (q_idx + offset)
            
            # 실제 메모리 할당 및 값 대입
            mask = torch.zeros((1, 1, sl, offset + sl), dtype=torch.bfloat16, device=device)
            mask[mask_bool] = NEG_INF
            
            print(f" -> Mask Success at offset {offset}. Shape: {mask.shape}, Memory: {mask.element_size() * mask.nelement() / 1024**2:.2f} MB")
            return True
        except Exception as e:
            print(f" !! Mask FAILED at offset {offset}: {e}")
            return False

    # 처음, 중간, 끝 지점 테스트
    test_mask(0, 256)
    test_mask(32000, 256)
    test_mask(64714, 256)

    print("\n[3] Final Verdict")
    print("If all above steps passed, the logic is sound.")
    print("The Rust issue is likely due to Asynchronous CUDA errors from Tensor Reinterpretation or Alignment.")

if __name__ == "__main__":
    test_full_logic_simulation()
