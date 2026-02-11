import gguf
import os

def check_gguf(path, name):
    print(f"\n=== Checking {name} ({path}) ===")
    if not os.path.exists(path):
        print("File not found.")
        return
    reader = gguf.GGUFReader(path)
    print(f"Model Architecture: {reader.fields.get('general.architecture')}")
    
    # Check dimensions
    h_size = 0
    if 'qwen2.embedding_length' in reader.fields:
        h_size = reader.fields['qwen2.embedding_length'].parts[0]
    elif 'llm.embedding_length' in reader.fields:
        h_size = reader.fields['llm.embedding_length'].parts[0]
    print(f"Embedding Length (Hidden Size): {h_size}")

    # List first few tensors
    print("Top 15 Tensors:")
    for i, tensor in enumerate(reader.tensors):
        if i < 15 or 'blk.0' in tensor.name or 'token_embd' in tensor.name:
            print(f"  - {tensor.name}: {tensor.shape} ({tensor.tensor_type})")
        if i == 15: print("  ...")

check_gguf("./Qwen3-0.6B-BF16.gguf", "0.6B Model")
check_gguf("./Qwen3-VL-2B-Instruct-BF16.gguf", "2B Model")
