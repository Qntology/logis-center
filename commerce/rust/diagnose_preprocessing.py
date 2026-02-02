import os
import time
from transformers import AutoTokenizer
import sys

def diagnose_preprocessing(task_id):
    base_path = f"src-tauri/tmp/task_data/{task_id}"
    pug_path = os.path.join(base_path, "light_pug.txt")
    
    if not os.path.exists(pug_path):
        print(f" !! PUG file not found at: {pug_path}")
        return

    # 1. 원본 데이터 읽기
    start_time = time.time()
    with open(pug_path, "r", encoding="utf-8") as f:
        pug_content = f.read()
    read_time = time.time() - start_time
    
    print(f"\n[1] PUG Data Stats")
    print(f" -> File Size: {len(pug_content)/1024:.2f} KB")
    print(f" -> Raw String Length: {len(pug_content)}")
    print(f" -> Read Time: {read_time:.4f}s")

    # 2. 토큰화 시뮬레이션 (Qwen3/Qwen2.5 토크나이저 사용)
    # 실제 모델이 사용하는 토크나이저 경로
    model_path = "src-tauri/models/Qwen3-0.6B-Instruct-gguf"
    
    print(f"\n[2] Tokenization Stress Test (using {model_path})")
    try:
        # local_files_only=True로 로컬의 tokenizer.json 사용
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        start_time = time.time()
        # 토큰화 수행
        tokens = tokenizer.encode(pug_content)
        encode_time = time.time() - start_time
        
        print(f" -> Total Tokens: {len(tokens)}")
        print(f" -> Encoding Speed: {len(tokens)/encode_time:.2f} tokens/s")
        print(f" -> Encoding Time: {encode_time:.4f}s")
        
        # 3. 데이터 무결성 검사
        print(f"\n[3] Integrity Check")
        vocab_size = tokenizer.vocab_size
        max_token_id = max(tokens)
        min_token_id = min(tokens)
        
        print(f" -> Vocab Size: {vocab_size}")
        print(f" -> Token ID Range: {min_token_id} ~ {max_token_id}")
        
        if max_token_id >= vocab_size:
            print(" !! ALERT !! Found Token IDs exceeding Vocab Size. This will cause CUDA_ERROR_INVALID_VALUE!")
        else:
            print(" -> All Token IDs are within valid range.")

        # 4. 메모리 추정
        # KV 캐시 메모리 추정 (Layer 0, BF16)
        # (batch, heads, seq_len, head_dim) -> (1, 8, 64970, 128)
        kv_mem = (1 * 8 * len(tokens) * 128 * 2) / (1024**2)
        print(f"\n[4] KV Cache Memory Estimate (Layer 0)")
        print(f" -> For {len(tokens)} tokens: {kv_mem:.2f} MB")

    except Exception as e:
        print(f" !! Tokenization FAILED: {e}")

if __name__ == "__main__":
    # 가장 최근의 태스크 ID 사용
    diagnose_preprocessing("task_1770041737513")
