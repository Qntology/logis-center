import torch
from safetensors.torch import save_file, load_file
import os

def compute_alignment():
    base_dir = "src-tauri/models"
    path_06b = os.path.join(base_dir, "Qwen3-0.6B-Instruct-gguf", "model.safetensors")
    path_2b = os.path.join(base_dir, "Qwen3-VL-2B-Instruct-gguf", "model.safetensors")
    output_path = os.path.join(base_dir, "align_matrix.safetensors")

    print(f"[ALIGN] Loading 0.6B embeddings from {path_06b}...")
    tensors_06b = load_file(path_06b)
    # 텐서명은 quantize_split.py 분석 결과에 따름
    emb_06b = tensors_06b["model.embed_tokens.weight"].to(torch.float32) # [151936, 1024]

    print(f"[ALIGN] Loading 2B embeddings from {path_2b}...")
    tensors_2b = load_file(path_2b)
    emb_2b = tensors_2b["model.language_model.embed_tokens.weight"].to(torch.float32) # [151936, 2048]

    print("[ALIGN] Computing optimal projection matrix (Least Squares)...")
    # emb_06b * W = emb_2b  =>  W = (emb_06b^T * emb_06b)^-1 * emb_06b^T * emb_2b
    # PyTorch의 lstsq를 사용하여 최적의 W를 구함
    # 연산 부하를 줄이기 위해 처음 50000개 토큰(자주 쓰이는 단어) 위주로 정렬
    X = emb_06b[:50000, :]
    Y = emb_2b[:50000, :]
    
    # solve X * W = Y
    W = torch.linalg.lstsq(X, Y).solution
    
    print(f"[ALIGN] Projection Matrix Shape: {W.shape}") # Should be [1024, 2048]
    
    # 검증: 오차 확인
    pred = X @ W
    error = torch.mean(torch.abs(pred - Y))
    print(f"[ALIGN] Mean Alignment Error: {error.item():.6f}")

    save_file({"weight": W.to(torch.float16).contiguous()}, output_path)
    print(f"[ALIGN] Saved alignment matrix to {output_path}")

if __name__ == "__main__":
    compute_alignment()
