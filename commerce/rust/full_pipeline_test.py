import torch
import numpy as np
from safetensors.torch import save_file, load_file
import time
import os

class FullPipelineSimulation:
    def __init__(self, device="cuda"):
        self.device = device
        # 0.6B Spec
        self.s_hidden = 1024
        self.s_kv_heads = 8
        self.head_dim = 128
        # 2B Spec
        self.l_hidden = 2048
        self.l_kv_heads = 8
        
        self.total_tokens = 64970
        self.chunk_size = 256

    def simulate_pug_preprocessing(self):
        print(f"\n[PUG] Preprocessing HTML to PUG...")
        # 64k 토큰 분량의 가상 토큰 ID 생성
        tokens = torch.randint(0, 151936, (self.total_tokens,), dtype=torch.long)
        print(f" -> Generated {len(tokens)} tokens.")
        return tokens

    def phase1_baking_06b(self, tokens):
        print(f"\n[PHASE 1] 0.6B Baking (KV Cache Generation)")
        kv_cache_k = []
        kv_cache_v = []
        
        current_offset = 0
        torch.cuda.reset_peak_memory_stats()
        
        while current_offset < self.total_tokens:
            actual_chunk = min(self.chunk_size, self.total_tokens - current_offset)
            
            # Mask 생성 (메모리 최적화 방식)
            q_idx = torch.arange(actual_chunk, device=self.device).view(actual_chunk, 1)
            kv_idx = torch.arange(current_offset + actual_chunk, device=self.device).view(1, -1)
            mask = (kv_idx > (current_offset + q_idx)).to(torch.bfloat16) * -65504.0
            
            # KV 캐시 누적 시뮬레이션 (Layer 0)
            # Shape: (bs, heads, len, dim) -> (1, 8, 256, 128)
            k_chunk = torch.randn(1, self.s_kv_heads, actual_chunk, self.head_dim, dtype=torch.bfloat16, device=self.device)
            v_chunk = torch.randn(1, self.s_kv_heads, actual_chunk, self.head_dim, dtype=torch.bfloat16, device=self.device)
            
            # 실제로는 리스트에 누적하거나 텐서 결합
            # 여기서는 메모리 시뮬레이션을 위해 마지막 상태만 유지하거나 가상으로 누적
            current_offset += actual_chunk
            
            if current_offset % 10240 == 0 or current_offset == self.total_tokens:
                print(f" -> Offset: {current_offset:5d} | VRAM: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

        # 64k 토큰의 최종 KV 캐시 (가상 생성)
        full_k = torch.randn(1, self.s_kv_heads, self.total_tokens, self.head_dim, dtype=torch.bfloat16, device="cpu")
        full_v = torch.randn(1, self.s_kv_heads, self.total_tokens, self.head_dim, dtype=torch.bfloat16, device="cpu")
        
        return full_k, full_v

    def phase2_bridge_to_2b(self, k_06b, v_06b):
        print(f"\n[BRIDGE] Upscaling 0.6B KV -> 2B KV (1024 -> 2048)")
        # hidden_size가 2배이므로, KV 캐시의 head 구조는 동일하되 
        # 만약 모델 구조상 head_dim이나 heads가 다르다면 여기서 변환
        
        # Qwen3 0.6B와 2B는 둘 다 head_dim 128, heads 8로 동일함 (Hidden만 다름)
        # 따라서 Bridge 연산은 거의 Identity에 가깝거나, 
        # 만약 차원이 달랐다면 선형 보간을 수행했을 것.
        
        # 시뮬레이션: 데이터 복사 및 VRAM 로드
        start_time = time.time()
        k_2b = k_06b.to(self.device)
        v_2b = v_06b.to(self.device)
        
        print(f" -> Bridge Complete. Shape: {k_2b.shape} | VRAM: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        return k_2b, v_2b

    def run(self):
        tokens = self.simulate_pug_preprocessing()
        k_small, v_small = self.phase1_baking_06b(tokens)
        
        # Disk Bridge 시뮬레이션
        save_path = "tmp_kv_test.safetensors"
        save_file({"k": k_small, "v": v_small}, save_path)
        print(f"\n[DISK] Saved KV Cache to {save_path} ({os.path.getsize(save_path)/1024**2:.2f} MB)")
        
        # 2B 로드 시뮬레이션
        k_large, v_large = self.phase2_bridge_to_2b(k_small, v_small)
        
        print(f"\n✅ FULL PIPELINE SUCCESS")
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")

if __name__ == "__main__":
    sim = FullPipelineSimulation()
    sim.run()
