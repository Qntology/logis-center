fn main() {
    // Tauri 기본 빌드 설정
    tauri_build::build();

    // [NATIVE-GPU] CUDA 커널 컴파일 설정
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-rerun-if-changed=src/models/qwen3vl/native_backend.cu");
        
        let out_dir = std::env::var("OUT_DIR").unwrap();
        
        // 1. CUDA 커널 컴파일 (정적 라이브러리 .lib 생성)
        let status = std::process::Command::new("nvcc")
            .args(&[
                "-O3",
                "-lib",
                "src/models/qwen3vl/native_backend.cu",
                "-o",
                &format!("{}/native_backend.lib", out_dir),
            ])
            .status()
            .expect("Failed to run nvcc. Is CUDA Toolkit installed?");

        if status.success() {
            // 2. 직접 빌드한 native_backend.lib 경로 추가
            println!("cargo:rustc-link-search=native={}", out_dir);
            println!("cargo:rustc-link-lib=static=native_backend");
            
            // 3. 시스템 CUDA 라이브러리 경로 추가 (cudart.lib 해결)
            if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
                // 확인된 경로: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64
                println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
            } else {
                // Fallback (환경변수 없을 경우)
                println!("cargo:rustc-link-search=native=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64");
            }
            
            // CUDA 런타임 라이브러리 링크
            println!("cargo:rustc-link-lib=cudart");
        } else {
            panic!("CUDA compilation failed");
        }
    }
}