import torch
import os
from safetensors.torch import load_file, save_file

def fix_shared_st():
    model_dir = r'src-tauri\models\Qwen3.5-0.8B-Split'
    path = os.path.join(model_dir, 'shared.st')
    if not os.path.exists(path):
        print("shared.st not found")
        return

    print(f"Surgically fixing {path}...")
    sd = load_file(path)
    new_sd = {}
    for k, v in sd.items():
        if k.endswith(".data") and v.dtype == torch.int8:
            print(f"Converting {k} from int8 to uint8...")
            # We assume the previous quantization was a simple cast or a shift.
            # If it was my previous quantize_q8_0, it was torch.round(reshaped / scales).clamp(-128, 127).to(torch.int8)
            # To make it U8 (0..255), we do v.to(torch.int16) + 128 -> to(uint8)
            v_u8 = (v.to(torch.int16) + 128).to(torch.uint8)
            new_sd[k] = v_u8
        else:
            new_sd[k] = v
    
    temp_path = path + ".fix"
    save_file(new_sd, temp_path)
    os.remove(path)
    os.rename(temp_path, path)
    print("Done.")

if __name__ == "__main__":
    fix_shared_st()
