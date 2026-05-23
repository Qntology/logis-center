fn main() {
    println!("cargo:rerun-if-changed=src/kernels/flash_attn.cu");

    cc::Build::new()
        .cuda(true)
        .file("src/kernels/flash_attn.cu")
        .flag("-arch=sm_80") // 사용하시는 GPU 아키텍처에 맞게 변경 (예: Ampere=sm_80)
        .flag("-O3")
        .flag("-Xcompiler=/utf-8") // 한글 윈도우 MSVC 인코딩 에러(C4819) 방지
        .flag("-Xcompiler=/MT") // 추가: C++ 런타임을 MT(Static)로 강제 지정
        .static_crt(true) // 추가: cc 크레이트에서 정적 CRT를 명시적으로 사용하도록 설정
        .compile("flash_attn");

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