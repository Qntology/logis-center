import os
import time
import torch
import re
from transformers import AutoTokenizer
import numpy as np

class FullPipelineStressTest:
    def __init__(self, model_path="src-tauri/models/Qwen3-0.6B-Instruct-gguf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def html_to_pug_sim(self, html):
        """Rust의 parsing.rs 정규식 로직을 파이썬으로 모사"""
        print(f"\n[PUG] Starting HTML to PUG Conversion (Regex Mode)...")
        start_time = time.time()
        
        # 1. 주석 제거
        html = re.sub(r"(?s)<!--.*?-->", "", html)
        # 2. script, style 등 제거
        html = re.sub(r"(?is)<(script|style|link|noscript|iframe)\b[^>]*>.*?</(script|style|link|noscript|iframe)>", "", html)
        # 3. 연속 공백 정리
        html = re.sub(r"\s+", " ", html)
        
        # PUG 변환 시뮬레이션 (태그 추출)
        tags = re.findall(r"<([a-zA-Z0-9]+)([^>]*)>", html)
        pug_lines = []
        for tag, attrs in tags:
            class_match = re.search(r'class="([^"]*)"', attrs)
            classes = class_match.group(1).replace(" ", ".") if class_match else ""
            pug_lines.append(f"{tag}.{classes}" if classes else tag)
            
        pug_result = "\n".join(pug_lines)
        print(f" -> Simulated PUG Length: {len(pug_result)} chars")
        print(f" -> Tag Count: {len(tags)}")
        print(f" -> Time Taken: {time.time() - start_time:.4f}s")
        return pug_result

    def run_model_simulation(self, pug_text):
        print(f"\n[MODEL] Starting Model Pipeline Simulation...")
        tokens = self.tokenizer.encode(pug_text)
        print(f" -> Total Tokens: {len(tokens)}")
        
        if len(tokens) > 100000:
            print(" !! WARNING !! Token count is extremely high. Scaling down simulation...")
            tokens = tokens[:100000]

        chunk_size = 256
        offset = 0
        peak_vram = 0
        
        while offset < len(tokens):
            sl = min(chunk_size, len(tokens) - offset)
            # Mask 생성 부하 측정
            q_idx = torch.arange(sl, device=self.device).view(sl, 1)
            kv_idx = torch.arange(offset + sl, device=self.device).view(1, -1)
            # (sl, offset + sl) 마스크 연산
            mask = (kv_idx > (q_idx + offset)).to(torch.float16)
            
            peak_vram = max(peak_vram, torch.cuda.memory_allocated() / 1024**2)
            offset += sl
            if offset % 20480 == 0:
                print(f"    - Progress: {offset}/{len(tokens)} | VRAM: {peak_vram:.2f} MB")

        kv_mem = (1 * 8 * len(tokens) * 128 * 2) / (1024**2)
        print(f" -> Final KV Cache VRAM: {kv_mem:.2f} MB")
        print(f"\n✅ SIMULATION COMPLETE")
        print(f"Peak Estimated VRAM: {peak_vram + kv_mem:.2f} MB")

def get_latest_task_data():
    task_dir = "src-tauri/tmp/task_data"
    latest_task = sorted([d for d in os.listdir(task_dir) if d.startswith("task")])[-1]
    html_path = os.path.join(task_dir, latest_task, "raw_html.txt")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    stress_test = FullPipelineStressTest()
    try:
        html = get_latest_task_data()
        print(f"Input HTML Size: {len(html)/1024:.2f} KB")
        pug = stress_test.html_to_pug_sim(html)
        stress_test.run_model_simulation(pug)
    except Exception as e:
        print(f"❌ ERROR: {e}")