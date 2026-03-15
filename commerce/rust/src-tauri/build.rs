fn main() {
    // CUDA 컴파일러(nvcc)를 사용하여 커널들을 빌드합니다.
    cc::Build::new()
        .cuda(true)
        .static_crt(true) 
        .flag("-gencode").flag("arch=compute_80,code=sm_80") 
        .flag("-O3") 
        .flag("-use_fast_math")
        // 👇 두 개의 CUDA 파일을 모두 등록합니다.
        .file("src/models/qwen3vl/cuda/paged_flash_decoding.cu")
        .file("src/models/qwen3vl/cuda/dequantize_q2.cu") 
        .compile("paged_flash_decoding");

    // 👇 두 파일 중 하나라도 변경되면 다시 빌드하도록 설정합니다.
    println!("cargo:rerun-if-changed=src/models/qwen3vl/cuda/paged_flash_decoding.cu");
    println!("cargo:rerun-if-changed=src/models/qwen3vl/cuda/dequantize_q2.cu");

    // ... (이하 Visual Studio 경로 및 DirectStorage 설정 유지) ...
    println!("cargo:rustc-env=NVCC_CCBIN=C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64\\cl.exe");

    if std::env::var("TARGET").map_or(false, |t| t.contains("windows")) {
        std::env::set_var("CFLAGS", "/Zc:preprocessor");
        std::env::set_var("CXXFLAGS", "/Zc:preprocessor");
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let lib_path = std::path::Path::new(&manifest_dir)
            .join("microsoft.direct3d.directstorage.1.3.0")
            .join("native")
            .join("lib")
            .join("x64");
        println!("cargo:rustc-link-search=native={}", lib_path.display());
        println!("cargo:rustc-link-lib=dstorage");
    }
    tauri_build::build()
}