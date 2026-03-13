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
            .join("microsoft.direct3d.directstorage.1.3.0")
            .join("native")
            .join("lib")
            .join("x64");
        println!("cargo:rustc-link-search=native={}", lib_path.display());
        println!("cargo:rustc-link-lib=dstorage");
    }
    tauri_build::build()
}
