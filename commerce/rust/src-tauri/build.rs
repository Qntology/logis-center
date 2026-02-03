fn main() {
    // Tauri 기본 빌드 설정
    tauri_build::build();

    // [NATIVE-GPU] CUDA 커널 컴파일 설정
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-rerun-if-changed=src/models/qwen3vl/native_backend.cu");
        
        let out_dir = std::env::var("OUT_DIR").unwrap();
        
        // 1. CUDA 커널 컴파일 (정적 라이브러리 .lib 생성)
        let mut nvcc_cmd = std::process::Command::new("nvcc");
        
        // 환경변수에서 CCBIN 경로 가져오기 (없으면 제공된 기본값 사용)
        let ccbin = std::env::var("NVCC_CCBIN")
            .unwrap_or_else(|_| "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe".to_string());

        nvcc_cmd.args(&[
            "-O3",
            "-lib",
            "src/models/qwen3vl/native_backend.cu",
            "-o",
            &format!("{}/native_backend.lib", out_dir),
            "-ccbin", &ccbin,
            // 아키텍처 타겟 (Turing, Ampere)
            "--generate-code=arch=compute_75,code=[compute_75,sm_75]",
            "--generate-code=arch=compute_80,code=[compute_80,sm_80]",
            "--generate-code=arch=compute_86,code=[compute_86,sm_86]",
        ]);
        
        let status = nvcc_cmd.status()
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