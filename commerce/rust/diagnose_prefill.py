import torch
import numpy as np
from safetensors.torch import load_file, save_file
import time
import os

class PythonQuantDiagnostic:
    def __init__(self, device="cuda"):
        self.device = device
        self.block_size = 32

    def dequantize_bit_serial(self, packed, scales, original_shape):
        """Rust의 dequantize_on_the_fly를 파이썬(Torch)으로 재현"""
        # packed는 I32로 저장되어 있을 수 있으므로 U32 패턴으로 해석
        p_uint32 = packed.view(torch.int32).cpu().numpy().view(np.uint32)
        scales_np = scales.to(torch.float32).cpu().numpy()
        
        total_el = np.prod(original_shape)
        weights = np.zeros(total_el, dtype=np.float32)
        
        print(f" -> Dequantizing {total_el} elements...")
        for b_i in range(len(p_uint32)):
            if b_i >= len(scales_np): break
            s_val = scales_np[b_i]
            b = p_uint32[b_i]
            for bit in range(32):
                idx = b_i * 32 + bit
                if idx < total_el:
                    weights[idx] = s_val * (1.0 if (b >> bit) & 1 else -1.0)
        
        return torch.from_numpy(weights).view(original_shape).to(self.device)

    def test_mask_generation(self, seq_len, offset):
        """64k 토큰 마스크 생성 시 메모리 압박 테스트"""
        print(f"\n[MASK TEST] SeqLen: {seq_len}, Offset: {offset}")
        try:
            start_mem = torch.cuda.memory_allocated() / 1024**2
            # Rust의 prepare_causal_attention_mask 로직
            # (bs, 1, query_len, total_kv_len)
            mask_shape = (1, 1, seq_len, offset + seq_len)
            print(f" -> Attempting to allocate mask of shape: {mask_shape}")
            
            # 실제 할당 시도
            mask = torch.full((seq_len, offset + seq_len), float("-inf"), device=self.device)
            mask = torch.triu(mask, diagonal=offset + 1)
            
            end_mem = torch.cuda.memory_allocated() / 1024**2
            print(f" -> Success! Mask Memory: {end_mem - start_mem:.2f} MB")
            return True
        except Exception as e:
            print(f" !! FAILED: {e}")
            return False

    def inspect_kv_cache(self, kv_path):
        """Rust가 저장한 tmp/kv/layer_0_bitkv.safetensors 읽기 테스트"""
        if not os.path.exists(kv_path):
            print(f" !! KV Path not found: {kv_path}")
            return
        
        print(f"\n[KV INSPECT] Loading: {kv_path}")
        kv = load_file(kv_path)
        for k in kv.keys():
            print(f" -> {k}: shape={kv[k].shape}, dtype={kv[k].dtype}")

def run_diagnostic():
    diag = PythonQuantDiagnostic()
    
    # 1. 텐서 로딩 및 복원 테스트
    model_path = "src-tauri/models/Qwen3-0.6B-Instruct-gguf/model-BITSERIAL_LAYER0.safetensors"
    if os.path.exists(model_path):
        tensors = load_file(model_path)
        # q_proj 테스트
        packed = tensors["model.layers.0.self_attn.q_proj.weight.packed"]
        scales = tensors["model.layers.0.self_attn.q_proj.weight.scales"]
        shape = tensors["model.layers.0.self_attn.q_proj.weight.shape"].tolist()
        
        w = diag.dequantize_bit_serial(packed, scales, shape)
        print(f" -> Reconstructed weights min/max: {w.min().item():.4f} / {w.max().item():.4f}")

    # 2. 64k 마스크 할당 테스트 (현 문제의 핵심 의심지점)
    # Rust에서 offset이 0인 상태로 64k를 한꺼번에 구우려 할 때
    diag.test_mask_generation(64970, 0)
    
    # 3. 만약 512 Chunk로 한다면?
    diag.test_mask_generation(512, 64000)

if __name__ == "__main__":
    run_diagnostic()
