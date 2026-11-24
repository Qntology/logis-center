fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_path = std::path::Path::new(&manifest_dir);

    // `tauri.conf.json` bundles `dlls/*` as resources, while `dlls/` is gitignored.
    // Ensure at least one placeholder file exists so `tauri_build::build()` never fails
    // on a fresh clone or direct `cargo check/test/tauri dev` invocation.
    let dlls_dir = manifest_path.join("dlls");
    if std::fs::create_dir_all(&dlls_dir).is_ok() {
        let is_empty = std::fs::read_dir(&dlls_dir)
            .map(|mut it| it.next().is_none())
            .unwrap_or(true);
        if is_empty {
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .open(dlls_dir.join(".gitkeep"));
        }
    }

    if std::env::var("TARGET").map_or(false, |t| t.contains("windows")) {
        // Force-disable fPIC and enable standard-conforming preprocessor for MSVC
        std::env::set_var("CFLAGS", "/Zc:preprocessor");
        std::env::set_var("CXXFLAGS", "/Zc:preprocessor");
        std::env::set_var("CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING", "1");

        // Link DirectStorage library
        let lib_path = manifest_path.join("libs");
        println!("cargo:rustc-link-search=native={}", lib_path.display());
        println!("cargo:rustc-link-lib=dstorage");
        // (통합 시험은 apps/core 으로 옮겨 tauri::Builder 를 쓰지 않으므로, 시험 exe 에 comctl32 v6 매니페스트를 넣던
        //  rustc-link-arg-tests 는 뺐습니다. 이 패키지에는 시험 대상이 없어 그 지시는 cargo 가 거부합니다.)
    }
    tauri_build::build()
}
