fn main() {
    // Tauri 기본 빌드 설정
    tauri_build::build();

    // [NATIVE-GPU] CUDA 커널 컴파일 설정
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-rerun-if-changed=src/models/qwen3vl/native_backend.cu");
        
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let status = std::process::Command::new("nvcc")
            .args(&[
                "-O3",
                "-lib",
                "-ptx", // 또는 직접 컴파일 시 "-fatbin"
                "src/models/qwen3vl/native_backend.cu",
                "-o",
                &format!("{}/native_backend.lib", out_dir),
            ])
            .status()
            .expect("Failed to run nvcc. Is CUDA Toolkit installed?");

        if status.success() {
            println!("cargo:rustc-link-search=native={}", out_dir);
            println!("cargo:rustc-link-lib=static=native_backend");
        }
    }
}