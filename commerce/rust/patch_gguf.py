
import os

file_path = r'src-tauri/models/Qwen3VL-2B-Instruct-Q4_K_M.gguf'
backup_path = file_path + '.bak'

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

print(f"Patching {file_path}...")

# Create a backup
if not os.path.exists(backup_path):
    import shutil
    shutil.copy(file_path, backup_path)
    print(f"Backup created at {backup_path}")

with open(file_path, 'rb') as f:
    content = f.read(1024 * 1024) # Read first 1MB for header

# GGUF strings are prefixed by their length (uint64)
# 'qwen3vl' is 7 chars.
old_arch = b'qwen3vl'
new_arch = b'qwen2vl'

if old_arch in content:
    print(f"Found '{old_arch.decode()}' in header. Patching all occurrences...")
    with open(file_path, 'r+b') as f:
        f_content = f.read(1024 * 1024)
        new_content = f_content.replace(old_arch, new_arch)
        
        count = f_content.count(old_arch)
        f.seek(0)
        f.write(new_content)
        print(f"Successfully patched {count} occurrences.")
else:
    print(f"'{old_arch.decode()}' not found in the first 1MB. It might already be patched or the file is different.")
