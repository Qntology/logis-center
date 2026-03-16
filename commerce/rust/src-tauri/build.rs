use std::process::Command;
use std::env;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let msvc_path = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64\\cl.exe";

    // 컴파일할 CUDA 파일들
    let cuda_files = [
        "src/models/qwen3vl/cuda/paged_flash_decoding.cu",
        "src/models/qwen3vl/cuda/dequantize_q2.cu",
    ];

    for file in &cuda_files {
        let status = Command::new("nvcc")
            .arg("-ccbin")
            .arg(msvc_path)
            .arg("-gencode")
            .arg("arch=compute_80,code=sm_80")
            .arg("-O3")
            .arg("-c")
            .arg(file)
            .arg("-o")
            .arg(format!("{}/{}.obj", out_dir, Path::new(file).file_stem().unwrap().to_str().unwrap()))
            .status()
            .expect("Failed to execute nvcc");

        if !status.success() {
            panic!("CUDA compilation failed for {}", file);
        }
    }

    // 오브젝트 파일들을 정적 라이브러리로 묶기 (MSVC lib.exe 사용)
    let lib_exe = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64\\lib.exe";
    let status = Command::new(lib_exe)
        .arg(format!("/OUT:{}/paged_flash_decoding.lib", out_dir))
        .arg(format!("{}/paged_flash_decoding.obj", out_dir))
        .arg(format!("{}/dequantize_q2.obj", out_dir))
        .status()
        .expect("Failed to execute lib.exe");

    if !status.success() {
        panic!("Failed to create static library");
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=paged_flash_decoding");

    for file in &cuda_files {
        println!("cargo:rerun-if-changed={}", file);
    }

    println!("cargo:rustc-env=NVCC_CCBIN={}", msvc_path);

    if env::var("TARGET").map_or(false, |t| t.contains("windows")) {
        println!("cargo:rustc-link-lib=cudart");
        
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
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
