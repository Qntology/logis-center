import gguf
import os

def inspect_hybrid_model(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
    
    print(f"\n=== Analyzing Hybrid Model: {file_path} ===")
    size_mb = os.path.getsize(file_path) / (1024*1024)
    print(f"File Size: {size_mb:.2f} MB")
    
    reader = gguf.GGUFReader(file_path)
    
    # 1. Block count check
    if 'qwen2.block_count' in reader.fields:
        print(f"Total Block Count in Metadata: {reader.fields['qwen2.block_count'].data[0]}")
    
    # 2. Layer range check
    blk_indices = set()
    for tensor in reader.tensors:
        if 'blk.' in tensor.name:
            try:
                blk_indices.add(int(tensor.name.split('.')[1]))
            except: pass
    
    if blk_indices:
        print(f"Actual Layer indices present: {sorted(list(blk_indices))}")
        print(f"Layer Range: {min(blk_indices)} to {max(blk_indices)}")
    
    # 3. Embedding check
    has_embd = any('token_embd' in t.name for t in reader.tensors)
    print(f"Has Embedding (token_embd): {has_embd}")

possible_paths = [
    "src-tauri/models/Qwen3-VL-2B-Hybrid-gguf/Qwen3VL-2B-Instruct-Hybrid-Q4_K_M.gguf",
    "./Qwen3VL-2B-Instruct-Hybrid-Q4_K_M.gguf",
    "models/Qwen3VL-2B-Instruct-Hybrid-Q4_K_M.gguf"
]

for p in possible_paths:
    if os.path.exists(p):
        inspect_hybrid_model(p)
        break
else:
    print("Could not find the hybrid model file.")
