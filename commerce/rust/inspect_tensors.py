from safetensors import safe_open
import os
import sys

def analyze_model(model_path):
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return

    print(f"\n🔍 Analyzing: {model_path}")
    print("-" * 50)
    
    with safe_open(model_path, framework="pt") as f:
        keys = set(f.keys())
        
        # 1. 필수 텐서 체크
        essentials = ["embed_tokens", "norm", "lm_head"]
        found_essentials = [k for k in keys if any(e in k for e in essentials)]
        print(f"✅ Found {len(found_essentials)} essential tensor components (Embed/Norm/Head)")

        # 2. 비트-슬라이싱 상태 체크
        base_weights = set()
        for k in keys:
            if ".packed_b" in k:
                base_weights.add(k.split(".packed_b")[0])
            elif ".packed" in k and ".packed_b" not in k:
                base_weights.add(k.split(".packed")[0])

        if not base_weights:
            print("⚠️ No bit-sliced tensors found. This might be a standard FP16 model.")
        else:
            print(f"📈 Found {len(base_weights)} quantized linear layers.")
            
            # 4비트 완결성 전수 조사
            incomplete = []
            bit_depths = {}
            for bw in sorted(base_weights):
                planes = [k for k in keys if k.startswith(bw + ".packed_b")]
                depth = len(planes)
                bit_depths[depth] = bit_depths.get(depth, 0) + 1
                if depth < 4 and depth > 0:
                    incomplete.append(f"{bw} ({depth} bits)")
            
            for depth, count in bit_depths.items():
                print(f"   - {depth}-bit layers: {count}")
            
            if incomplete:
                print("🚨 WARNING: Incomplete bit-planes found in:")
                for inc in incomplete[:5]: print(f"     - {inc}")
                if len(incomplete) > 5: print(f"     ... and {len(incomplete)-5} more")

        # 3. 스케일/셰이프 메타데이터 체크
        has_scales = any(".scales" in k for k in keys)
        has_shapes = any(".shape" in k for k in keys)
        print(f"📐 Metadata: Scales={'OK' if has_scales else 'MISSING'}, Shapes={'OK' if has_shapes else 'MISSING'}")

if __name__ == "__main__":
    # 인자로 파일 경로를 받거나 기본 경로 조사
    files_to_check = sys.argv[1:] if len(sys.argv) > 1 else [
        r"src-tauri\models\Qwen3-0.6B-Instruct-gguf\model-4BIT_SLICED_LAYER0.safetensors",
        r"src-tauri\models\Qwen3-VL-2B-Instruct-gguf\model-4BIT_SLICED_L1_ALL.safetensors"
    ]
    
    for f in files_to_check:
        analyze_model(f)
