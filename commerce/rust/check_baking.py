import gguf
import os

def inspect_baking_models(files):
    for f in files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            continue
        
        print(f"\n--- Inspecting: {f} ---")
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"File Size: {size_mb:.2f} MB")
        
        try:
            reader = gguf.GGUFReader(f)
            # Check Layer Count
            if 'qwen2.block_count' in reader.fields:
                print(f"Block Count (Metadata): {reader.fields['qwen2.block_count'].data[0]}")
            
            # Check Tensors
            tensors = reader.tensors
            print(f"Total Tensors: {len(tensors)}")
            
            blk_indices = []
            for t in tensors:
                if 'blk.' in t.name:
                    try:
                        blk_indices.append(int(t.name.split('.')[1]))
                    except: pass
            
            if blk_indices:
                print(f"Layer Range: {min(blk_indices)} to {max(blk_indices)}")
            
            has_embd = any('token_embd' in t.name for t in tensors)
            has_l0 = any('blk.0.' in t.name for t in tensors)
            print(f"Has Embedding: {has_embd}, Has Layer 0: {has_l0}")
        except Exception as e:
            print(f"Error reading GGUF: {e}")

inspect_baking_models(["./Qwen3-0.6B-BF16.gguf", "./Qwen3-0.6B-Clean-L0-BF16.gguf", "./Qwen3-2B-TextOnly-Experimental.gguf"])
