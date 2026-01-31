import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter
import os

def quantize_littlebit_logic(weight, target_bpw=0.1):
    """
    LittleBit Simulation: Latent Matrix Factorization (W = sign(A) * sign(B) * scale)
    Aims for extreme compression (0.1 bits per weight).
    Handles N-D tensors by folding into 2D matrix.
    """
    orig_shape = weight.shape
    if len(orig_shape) < 2:
        return weight
    
    # [FIX] Handle N-D tensors (like patch_embd with 4 dims) by folding into 2D
    # Fold everything except the last dimension into rows
    N = orig_shape[-1]
    M = weight.size // N
    w_2d = weight.reshape((M, N)).astype(np.float32)
    
    # Calculate rank 'r' to achieve target BPW
    r = int((target_bpw * M * N) / (M + N))
    r = max(1, r)
    
    # 1. Singular Value Decomposition (SVD)
    try:
        U, S, Vh = np.linalg.svd(w_2d, full_matrices=False)
    except np.linalg.LinAlgError:
        return weight
    
    # 2. Extract top 'r' components
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vhr = Vh[:r, :]
    
    # 3. Binarize Factors
    A_bin = np.sign(Ur @ np.sqrt(Sr))
    B_bin = np.sign(np.sqrt(Sr) @ Vhr)
    
    # 4. Reconstruct Approximation
    W_approx = A_bin @ B_bin
    
    # 5. Compensation Scale
    num = np.sum(w_2d * W_approx)
    den = np.sum(W_approx * W_approx) + 1e-9
    scale = num / den
    
    result = (W_approx * scale).reshape(orig_shape)
    return result.astype(np.float16)

def process_littlebit(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"--- [LittleBit] NeurIPS 2025 Sub-1-bit Baker ---")
    print(f"Target Density: 0.1 bits per weight (Simulation)")
    
    reader = GGUFReader(input_path)
    writer = GGUFWriter(output_path, "clip")

    # 1. Metadata Cloning (Excluding core fields that GGUFWriter adds automatically)
    skip_fields = {"general.architecture", "general.type", "general.name"}
    for field in reader.fields.values():
        name = field.name
        if name in skip_fields:
            continue
            
        part = field.parts[-1]
        if isinstance(part, (str, bytes, bytearray)):
            writer.add_string(name, part if isinstance(part, str) else part.decode('utf-8', 'ignore'))
        elif isinstance(part, (int, np.integer)):
            writer.add_uint32(name, int(part))
        elif isinstance(part, (float, np.floating)):
            writer.add_float32(name, float(part))
        elif isinstance(part, (bool, np.bool_)):
            writer.add_bool(name, bool(part))
        elif isinstance(part, (list, np.ndarray)):
            clean_list = part.tolist() if hasattr(part, "tolist") else list(part)
            if len(clean_list) > 0:
                writer.add_array(name, clean_list)
    
    writer.add_string("general.name", "Qwen3VL LittleBit IQ0_S")
    writer.add_string("general.comment", "IQ0_S via LittleBit (0.1 bpw Latent Factorization)")

    # 2. Tensor Processing
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data
        
        is_weight = "weight" in name
        is_large = data.size > 1024 * 256 # Process tensors > 256KB
        is_not_embed = "patch_embed" not in name and "pos_embed" not in name
        
        if is_weight and is_large and is_not_embed:
            print(f"  [LittleBit] {name:30} | Rank Factorizing...")
            new_data = quantize_littlebit_logic(data, target_bpw=0.1)
        else:
            new_data = data
            
        writer.add_tensor(name, new_data)

    print(f"\nWriting Result to: {output_path}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"--- LittleBit IQ0_S Baker Complete ---")

if __name__ == "__main__":
    input_file = "llama-b7898-bin-win-cpu-x64/mmproj-Qwen3VL-2B-Instruct-F16.gguf"
    output_file = "llama-b7898-bin-win-cpu-x64/mmproj-Qwen3VL-2B-Instruct-LittleBit_IQ0_S.gguf"
    process_littlebit(input_file, output_file)
