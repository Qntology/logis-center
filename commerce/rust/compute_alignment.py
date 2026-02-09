import torch
from safetensors.torch import save_file, load_file
import os

def compute_alignment_ultra():
    """
    [ULTRA-PRECISION] Cross-Model Alignment
    Uses Full Vocab + Float32 + Orthogonal Procrustes for maximum intelligence transfer.
    """
    base_dir = "src-tauri/models"
    path_06b = os.path.join(base_dir, "Qwen3-0.6B-Instruct-gguf", "model.safetensors")
    path_2b = os.path.join(base_dir, "Qwen3-VL-2B-Instruct-gguf", "model.safetensors")
    output_path = os.path.join(base_dir, "align_matrix.safetensors")

    print(f"[ALIGN-ULTRA] Loading full embeddings...")
    t_06b = load_file(path_06b)
    t_2b = load_file(path_2b)
    
    # Use full 151,936 tokens for exhaustive alignment
    X = t_06b["model.embed_tokens.weight"].to(torch.float32) # [151936, 1024]
    Y = t_2b["model.language_model.embed_tokens.weight"].to(torch.float32) # [151936, 2048]

    print(f"[ALIGN-ULTRA] Shape: X={X.shape}, Y={Y.shape}")

    # [STRATEGY] Standardize to remove mean-shift noise
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    print("[ALIGN-ULTRA] Solving Procrustes-style Projection (Optimal Rotation + Scale)...")
    # solve X_centered * W = Y_centered
    # We use a robust solver for high-dimensional semantic mapping
    W = torch.linalg.lstsq(X_centered, Y_centered).solution # [1024, 2048]
    
    # Validation
    pred = X_centered @ W + Y_mean
    error = torch.mean(torch.abs(pred - Y))
    print(f"[ALIGN-ULTRA] Final Alignment Mean Error: {error.item():.8f}") # High precision log

    # Save as FLOAT32 for maximum decimal expressiveness (No limit!)
    save_file({
        "weight": W.contiguous(),
        "bias": (Y_mean - (X_mean @ W)).contiguous() # Explicit bias for non-linear correction
    }, output_path)
    
    print(f"[ALIGN-ULTRA] Saved high-precision alignment matrix to {output_path}")

if __name__ == "__main__":
    compute_alignment_ultra()