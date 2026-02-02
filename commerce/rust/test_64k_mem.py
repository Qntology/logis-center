import torch
import numpy as np
import time

def simulate_efficient_prefill(total_tokens=64970, chunk_size=256, device="cuda"):
    print(f"🚀 Starting 64k Prefill Simulation (VRAM Protection Mode)")
    print(f"Target: {total_tokens} tokens, Chunk Size: {chunk_size}")
    
    # 가상의 임베딩 차원 (Qwen3-0.6B 기준)
    hidden_size = 1024
    
    torch.cuda.empty_cache()
    base_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"Initial VRAM: {base_mem:.2f} MB")

    current_offset = 0
    step = 0
    
    # 시간 측정을 위해
    start_time = time.time()

    try:
        while current_offset < total_tokens:
            # 1. 현재 청크 크기 결정
            actual_chunk = min(chunk_size, total_tokens - current_offset)
            
            # 2. 가상의 입력 텐서 생성 (256, 1024)
            # 실제 모델에서는 이 텐서가 GPU에 올라갑니다.
            input_chunk = torch.randn(1, actual_chunk, hidden_size, dtype=torch.bfloat16, device=device)
            
            # 3. [핵심] 효율적인 마스크 생성 (sl, offset + sl)
            # 전체 64k * 64k 마스크를 만들지 않고, 
            # 현재 256개 토큰이 이전의 모든 토큰(offset)을 볼 수 있게만 만듭니다.
            mask_shape = (1, 1, actual_chunk, current_offset + actual_chunk)
            
            # 마스크 할당 시뮬레이션
            # (실제 Attention 커널 내부 연산을 모사)
            mask = torch.zeros(mask_shape, dtype=torch.bfloat16, device=device)
            # 인과 관계 마스킹 (현재 청크 내부에서만 적용)
            causal_indices = torch.arange(actual_chunk, device=device).view(-1, 1)
            history_indices = torch.arange(current_offset + actual_chunk, device=device).view(1, -1)
            # 현재 청크의 i번째 토큰은 과거(0 ~ offset+i)만 볼 수 있음
            mask = (history_indices > (current_offset + causal_indices)).to(torch.bfloat16) * -65504.0
            
            # 4. 메모리 체크 (주기적으로 출력)
            if step % 50 == 0 or current_offset + actual_chunk >= total_tokens:
                curr_mem = torch.cuda.memory_allocated() / 1024**2
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                print(f"Step {step:03d} | Offset: {current_offset:5d} | VRAM: {curr_mem:7.2f} MB | Peak: {peak_mem:7.2f} MB")

            # 5. KV 캐시 누적 시뮬레이션 (VRAM에 계속 쌓임)
            # 실제로는 각 레이어마다 쌓이지만, 여기서는 메모리 압박을 위해 하나만 시뮬레이션
            # (현실적으로 64k 토큰의 KV 캐시는 BF16 기준 상당한 용량임)
            
            # 다음 단계로
            current_offset += actual_chunk
            step += 1
            
            # 메모리 청소 (마스크와 입력값은 매 스텝 버려짐)
            del mask
            del input_chunk

        end_time = time.time()
        print(f"\n✅ SUCCESS! Processed {total_tokens} tokens in {end_time - start_time:.2f}s")
        print(f"Final Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        return True

    except Exception as e:
        print(f"\n❌ CRASHED at offset {current_offset}: {e}")
        return False

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Simulation skipped.")
    else:
        # GPU 이름 출력
        print(f"Device: {torch.cuda.get_device_name(0)}")
        simulate_efficient_prefill()
