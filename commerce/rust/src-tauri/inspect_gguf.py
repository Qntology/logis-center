import sys
from gguf import GGUFReader

def dump_gguf_structure(file_path, output_path):
    try:
        reader = GGUFReader(file_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Structure for: " + file_path + "\n")
            f.write("-" * 50 + "\n")
            f.write("KV Metadata count: " + str(len(reader.fields)) + "\n")
            for key in reader.fields.keys():
                f.write("Meta: " + str(key) + "\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Tensor count: " + str(len(reader.tensors)) + "\n")
            f.write("-" * 50 + "\n")
            
            sorted_tensors = sorted(reader.tensors, key=lambda x: x.name)
            for tensor in sorted_tensors:
                f.write("Tensor: " + str(tensor.name) + " | Shape: " + str(tensor.shape) + " | Type: " + str(tensor.tensor_type) + "\n")
        
        print("Successfully dumped structure to " + output_path)
    except Exception as e:
        print("Error: " + str(e))

if __name__ == "__main__":
    model_path = "src-tauri/models/Qwen3.5-0.8B-gguf/Qwen3.5-0.8B.gguf"
    output_path = "src-tauri/models/Qwen3.5-0.8B-gguf/tensor_structure.txt"
    dump_gguf_structure(model_path, output_path)
