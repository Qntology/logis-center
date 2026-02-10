
from pathlib import Path

def check_magic(p):
    with open(p, "rb") as f:
        magic = f.read(4)
        print(f"File: {p.name} | Magic: {magic}")

base = Path(r"C:\Users\HP\Documents\GitHub\cron-logis-center\commerceust\src-tauri\models\Qwen3-VL-2B-Instruct-gguf")
check_magic(base / "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf")
check_magic(base / "mmproj-Qwen3VL-2B-Instruct-F16.gguf")
check_magic(base / "Qwen3VL-2B-Instruct-Q4_K_M.gguf")
