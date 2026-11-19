fn main() {
    println!("cargo:rustc-env=NVCC_CCBIN=C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64\\cl.exe");
    if std::env::var("TARGET").map_or(false, |t| t.contains("windows")) {
        // Force-disable fPIC and enable standard-conforming preprocessor for MSVC
        std::env::set_var("CFLAGS", "/Zc:preprocessor");
        std::env::set_var("CXXFLAGS", "/Zc:preprocessor");
        std::env::set_var("CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING", "1");

        // Link DirectStorage library
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let lib_path = std::path::Path::new(&manifest_dir)
            .join("libs");
        println!("cargo:rustc-link-search=native={}", lib_path.display());
        println!("cargo:rustc-link-lib=dstorage");
        // (통합 시험은 apps/core 으로 옮겨 tauri::Builder 를 쓰지 않으므로, 시험 exe 에 comctl32 v6 매니페스트를 넣던
        //  rustc-link-arg-tests 는 뺐습니다. 이 패키지에는 시험 대상이 없어 그 지시는 cargo 가 거부합니다.)
    }
    tauri_build::build()
}
