import os
import torch
import numpy as np
from safetensors.torch import save_file, load_file
from tqdm import tqdm

def quantize_tensor_q2(tensor, block_size=32):
    """
    Quantizes a tensor to 2-bit using block-wise scaling.
    Packs 4 values into one uint8.
    """
    if tensor.dtype not in [torch.float16, torch.bfloat16, torch.float32] or tensor.numel() < 128:
        return tensor, None, None

    orig_shape = list(tensor.shape)
    # Flatten and pad to block_size
    flat = tensor.flatten().float()
    # We need numel to be multiple of block_size and also multiple of 4 for packing
    align = max(block_size, 4)
    pad = (align - (flat.numel() % align)) % align
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad)])
    
    # Reshape to blocks for scaling
    blocks = flat.view(-1, block_size)
    abs_max, _ = blocks.abs().max(dim=1, keepdim=True)
    abs_max[abs_max == 0] = 1.0
    scales = abs_max / 1.5 # Maps [-1.5, 1.5] to [-1.5, 1.5] roughly
    
    # Quantize to 4 levels: 0, 1, 2, 3
    # (val / scale) + 1.5 -> [0, 3]
    normalized = (blocks / scales) + 1.5
    q_vals = torch.clamp(torch.round(normalized), 0, 3).to(torch.uint8)
    
    # Pack 4 values into one uint8
    q_flat = q_vals.view(-1)
    q_packed = (q_flat[0::4] << 6) | (q_flat[1::4] << 4) | (q_flat[2::4] << 2) | q_flat[3::4]
    
    return q_packed, scales.half(), torch.tensor(orig_shape, dtype=torch.int32)

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = [f for f in os.listdir(input_dir) if f.endswith(".st")]
    print(f"Quantizing {len(files)} files to Q2...")

    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        state_dict = load_file(input_path)
        new_state_dict = {}
        
        for name, tensor in tqdm(state_dict.items(), desc=f"Packing {file_name}", leave=False):
            # Only quantize large weights (Linear layers)
            if any(x in name for x in [".weight", "in_proj", "out_proj"]) and tensor.numel() > 1024:
                packed, scales, shape = quantize_tensor_q2(tensor)
                if scales is not None:
                    new_state_dict[f"{name}.q2_packed"] = packed
                    new_state_dict[f"{name}.q2_scales"] = scales
                    new_state_dict[f"{name}.q2_shape"] = shape
                else:
                    new_state_dict[name] = tensor
            else:
                new_state_dict[name] = tensor
                
        save_file(new_state_dict, output_path)
        print(f" Saved {file_name}")

if __name__ == "__main__":
    process_directory("src-tauri/models/Qwen3.5-Split", "src-tauri/models/Qwen3.5-Split-Q2")
