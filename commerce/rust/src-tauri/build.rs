fn main() {
    // CUDA 컴파일러(nvcc)를 사용하여 paged_flash_decoding.cu를 빌드합니다.
    cc::Build::new()
        .cuda(true)
        .static_crt(true) // <-- [추가된 부분] Rust의 기본 C 런타임 링크 방식(/MT)과 일치시킵니다.
        .flag("-gencode").flag("arch=compute_80,code=sm_80") // Ampere 아키텍처 이상 (A100, RTX 30/40 시리즈 등 환경에 맞게 수정)
        .flag("-O3") // 최대 최적화
        .flag("-use_fast_math")
        .file("src/models/qwen3vl/cuda/paged_flash_decoding.cu") // 실제 .cu 파일 경로로 맞춰주세요.
        .compile("paged_flash_decoding");

    // Rust 컴파일러에게 파일이 변경될 때만 다시 빌드하라고 알려줍니다.
    println!("cargo:rerun-if-changed=src/models/qwen3vl/cuda/paged_flash_decoding.cu");

    println!("cargo:rustc-env=NVCC_CCBIN=C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64\\cl.exe");
    if std::env::var("TARGET").map_or(false, |t| t.contains("windows")) {
        // Force-disable fPIC and enable standard-conforming preprocessor for MSVC
        std::env::set_var("CFLAGS", "/Zc:preprocessor");
        std::env::set_var("CXXFLAGS", "/Zc:preprocessor");
        std::env::set_var("CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING", "1");

        // Link DirectStorage library
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