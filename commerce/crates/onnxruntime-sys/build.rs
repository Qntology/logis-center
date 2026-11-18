#![allow(dead_code)]

use std::{
    borrow::Cow,
    env, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    str::FromStr,
};

/// ONNX Runtime version
///
/// WARNING: If version is changed, bindings for all platforms will have to be re-generated.
///          To do so, run this:
///              cargo build --package onnxruntime-sys --features generate-bindings
const ORT_VERSION: &str = "1.28.0";

/// Base Url from which to download pre-built releases/
const ORT_RELEASE_BASE_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download";

/// Environment variable selecting which strategy to use for finding the library
/// Possibilities:
/// * "download": Download a pre-built library from upstream. This is the default if `ORT_STRATEGY` is not set.
/// * "system": Use installed library. Use `ORT_LIB_LOCATION` to point to proper location.
/// * "compile": Download source and compile (TODO).
const ORT_ENV_STRATEGY: &str = "ORT_STRATEGY";

/// Name of environment variable that, if present, contains the location of a pre-built library.
/// Only used if `ORT_STRATEGY=system`.
const ORT_ENV_SYSTEM_LIB_LOCATION: &str = "ORT_LIB_LOCATION";
/// Name of environment variable that, if present, controls wether to use CUDA or not.
const ORT_ENV_GPU: &str = "ORT_USE_CUDA";

/// Subdirectory (of the 'target' directory) into which to extract the prebuilt library.
const ORT_PREBUILT_EXTRACT_DIR: &str = "onnxruntime";

#[cfg(feature = "disable-sys-build-script")]
fn main() {
    println!("Build script disabled!");
}

#[cfg(not(feature = "disable-sys-build-script"))]
fn main() {
    let libort_install_dir = prepare_libort_dir();

    let include_dir = libort_install_dir.join("include");
    let lib_dir = libort_install_dir.join("lib");

    println!("Include directory: {:?}", include_dir);
    println!("Lib directory: {:?}", lib_dir);

    // Tell cargo to tell rustc to link onnxruntime shared library.
    println!("cargo:rustc-link-lib=onnxruntime");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    println!("cargo:rerun-if-env-changed={}", ORT_ENV_STRATEGY);
    println!("cargo:rerun-if-env-changed={}", ORT_ENV_GPU);
    println!("cargo:rerun-if-env-changed={}", ORT_ENV_SYSTEM_LIB_LOCATION);

    generate_bindings(&include_dir);
}

#[cfg(not(feature = "generate-bindings"))]
fn generate_bindings(_include_dir: &Path) {
    println!("Bindings not generated automatically, using committed files instead.");
    println!("Enable with the 'generate-bindings' cargo feature.");

    // NOTE: If bindings could not be be generated for Apple Sillicon M1, please uncomment the following
    // let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    // let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Unable to get TARGET_ARCH");
    // if os == "macos" && arch == "aarch64" {
    //     panic!(
    //         "OnnxRuntime {} bindings for Apple M1 are not available",
    //         ORT_VERSION
    //     );
    // }
}

#[cfg(feature = "generate-bindings")]
fn generate_bindings(include_dir: &Path) {
    let clang_args = &[
        format!("-I{}", include_dir.display()),
        format!(
            "-I{}",
            include_dir
                .join("onnxruntime")
                .join("core")
                .join("session")
                .display()
        ),
    ];

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=src/generated/bindings.rs");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        // The current working directory is 'onnxruntime-sys'
        .clang_args(clang_args)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Set `size_t` to be translated to `usize` for win32 compatibility.
        .size_t_is_usize(true)
        // Format using rustfmt
        .rustfmt_bindings(true)
        .rustified_enum("*")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to (source controlled) src/generated/<os>/<arch>/bindings.rs
    let generated_file = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src")
        .join("generated")
        .join(env::var("CARGO_CFG_TARGET_OS").unwrap())
        .join(env::var("CARGO_CFG_TARGET_ARCH").unwrap())
        .join("bindings.rs");
    println!("cargo:rerun-if-changed={:?}", generated_file);
    bindings
        .write_to_file(&generated_file)
        .expect("Couldn't write bindings!");
}

fn download<P>(source_url: &str, target_file: P)
where
    P: AsRef<Path>,
{
    let resp = ureq::get(source_url)
        .timeout(std::time::Duration::from_secs(300))
        .call()
        .unwrap_or_else(|err| panic!("ERROR: Failed to download {}: {:?}", source_url, err));

    let len = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap();
    let mut reader = resp.into_reader();
    // FIXME: Save directly to the file
    let mut buffer = vec![];
    let read_len = reader.read_to_end(&mut buffer).unwrap();
    assert_eq!(buffer.len(), len);
    assert_eq!(buffer.len(), read_len);

    let f = fs::File::create(&target_file).unwrap();
    let mut writer = io::BufWriter::new(f);
    writer.write_all(&buffer).unwrap();
}

fn extract_archive(filename: &Path, output: &Path) {
    match filename.extension().map(|e| e.to_str()) {
        Some(Some("zip")) => extract_zip(filename, output),
        Some(Some("tgz")) => extract_tgz(filename, output),
        _ => unimplemented!(),
    }
}

fn extract_tgz(filename: &Path, output: &Path) {
    let file = fs::File::open(&filename).unwrap();
    let buf = io::BufReader::new(file);
    let tar = flate2::read::GzDecoder::new(buf);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(output).unwrap();
}

fn extract_zip(filename: &Path, outpath: &Path) {
    let file = fs::File::open(&filename).unwrap();
    let buf = io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(buf).unwrap();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).unwrap();
        #[allow(deprecated)]
        let outpath = outpath.join(file.sanitized_name());
        if !(&*file.name()).ends_with('/') {
            println!(
                "File {} extracted to \"{}\" ({} bytes)",
                i,
                outpath.as_path().display(),
                file.size()
            );
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(&p).unwrap();
                }
            }
            let mut outfile = fs::File::create(&outpath).unwrap();
            io::copy(&mut file, &mut outfile).unwrap();
        }
    }
}

trait OnnxPrebuiltArchive {
    fn as_onnx_str(&self) -> Cow<str>;
}

#[derive(Debug)]
enum Architecture {
    X86,
    X86_64,
    Arm,
    Arm64,
}

impl FromStr for Architecture {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "x86" => Ok(Architecture::X86),
            "x86_64" => Ok(Architecture::X86_64),
            "arm" => Ok(Architecture::Arm),
            "aarch64" => Ok(Architecture::Arm64),
            _ => Err(format!("Unsupported architecture: {}", s)),
        }
    }
}

impl OnnxPrebuiltArchive for Architecture {
    fn as_onnx_str(&self) -> Cow<str> {
        match self {
            Architecture::X86 => Cow::from("x86"),
            Architecture::X86_64 => Cow::from("x64"),
            Architecture::Arm => Cow::from("arm"),
            Architecture::Arm64 => Cow::from("arm64"),
        }
    }
}

#[derive(Debug)]
#[allow(clippy::enum_variant_names)]
enum Os {
    Windows,
    Linux,
    MacOs,
}

impl Os {
    fn archive_extension(&self) -> &'static str {
        match self {
            Os::Windows => "zip",
            Os::Linux => "tgz",
            Os::MacOs => "tgz",
        }
    }
}

impl FromStr for Os {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "windows" => Ok(Os::Windows),
            "macos" => Ok(Os::MacOs),
            "linux" => Ok(Os::Linux),
            _ => Err(format!("Unsupported os: {}", s)),
        }
    }
}

impl OnnxPrebuiltArchive for Os {
    fn as_onnx_str(&self) -> Cow<str> {
        match self {
            Os::Windows => Cow::from("win"),
            Os::Linux => Cow::from("linux"),
            Os::MacOs => Cow::from("osx"),
        }
    }
}

#[derive(Debug)]
enum Accelerator {
    None,
    Gpu,
}

impl FromStr for Accelerator {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "1" | "yes" | "true" | "on" => Ok(Accelerator::Gpu),
            _ => Ok(Accelerator::None),
        }
    }
}

impl OnnxPrebuiltArchive for Accelerator {
    fn as_onnx_str(&self) -> Cow<str> {
        match self {
            Accelerator::None => Cow::from(""),
            Accelerator::Gpu => Cow::from("gpu"),
        }
    }
}

#[derive(Debug)]
struct Triplet {
    os: Os,
    arch: Architecture,
    accelerator: Accelerator,
}

impl OnnxPrebuiltArchive for Triplet {
    fn as_onnx_str(&self) -> Cow<str> {
        match (&self.os, &self.arch, &self.accelerator) {
            // onnxruntime-win-x86-1.28.0.zip
            // onnxruntime-win-x64-1.28.0.zip
            // onnxruntime-win-arm-1.28.0.zip
            // onnxruntime-win-arm64-1.28.0.zip
            // onnxruntime-linux-x64-1.28.0.tgz
            // onnxruntime-osx-x64-1.28.0.tgz
            // onnxruntime-osx-arm64-1.28.0.tgz (Apple Silicon; [MAC-VERIFY] 없으면 macOS arm64 빌드가 여기서 panic)
            (Os::Windows, Architecture::X86, Accelerator::None)
            | (Os::Windows, Architecture::X86_64, Accelerator::None)
            | (Os::Windows, Architecture::Arm, Accelerator::None)
            | (Os::Windows, Architecture::Arm64, Accelerator::None)
            | (Os::Linux, Architecture::X86_64, Accelerator::None)
            | (Os::MacOs, Architecture::X86_64, Accelerator::None)
            | (Os::MacOs, Architecture::Arm64, Accelerator::None) => Cow::from(format!(
                "{}-{}",
                self.os.as_onnx_str(),
                self.arch.as_onnx_str()
            )),
            // onnxruntime-win-gpu-x64-1.28.0.zip
            // Note how this one is inverted from the linux one next
            (Os::Windows, Architecture::X86_64, Accelerator::Gpu) => Cow::from(format!(
                "{}-{}-{}",
                self.os.as_onnx_str(),
                self.accelerator.as_onnx_str(),
                self.arch.as_onnx_str(),
            )),
            // onnxruntime-linux-x64-gpu-1.28.0.tgz
            // Note how this one is inverted from the windows one above
            (Os::Linux, Architecture::X86_64, Accelerator::Gpu) => Cow::from(format!(
                "{}-{}-{}",
                self.os.as_onnx_str(),
                self.arch.as_onnx_str(),
                self.accelerator.as_onnx_str(),
            )),
            _ => {
                panic!(
                    "Unsupported prebuilt triplet: {:?}, {:?}, {:?}. Please use {}=system and {}=/path/to/onnxruntime",
                    self.os, self.arch, self.accelerator, ORT_ENV_STRATEGY, ORT_ENV_SYSTEM_LIB_LOCATION
                );
            }
        }
    }
}

fn prebuilt_archive_url() -> (PathBuf, String) {
    let triplet = Triplet {
        os: env::var("CARGO_CFG_TARGET_OS")
            .expect("Unable to get TARGET_OS")
            .parse()
            .unwrap(),
        arch: env::var("CARGO_CFG_TARGET_ARCH")
            .expect("Unable to get TARGET_ARCH")
            .parse()
            .unwrap(),
        accelerator: env::var(ORT_ENV_GPU).unwrap_or_default().parse().unwrap(),
    };

    let prebuilt_archive = format!(
        "onnxruntime-{}-{}.{}",
        triplet.as_onnx_str(),
        ORT_VERSION,
        triplet.os.archive_extension()
    );
    let prebuilt_url = format!(
        "{}/v{}/{}",
        ORT_RELEASE_BASE_URL, ORT_VERSION, prebuilt_archive
    );

    (PathBuf::from(prebuilt_archive), prebuilt_url)
}

fn prepare_libort_dir_prebuilt() -> PathBuf {
    let (prebuilt_archive, prebuilt_url) = prebuilt_archive_url();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let extract_dir = out_dir.join(ORT_PREBUILT_EXTRACT_DIR);
    let downloaded_file = out_dir.join(&prebuilt_archive);

    println!("cargo:rerun-if-changed={}", downloaded_file.display());

    let install_dir = extract_dir.join(prebuilt_archive.file_stem().unwrap());

    if !downloaded_file.exists() && !install_dir.join("lib").exists() {
        println!("Creating directory {:?}", out_dir);
        fs::create_dir_all(&out_dir).unwrap();

        println!(
            "Downloading {} into {}",
            prebuilt_url,
            downloaded_file.display()
        );
        // 🌟 [NUGET FALLBACK] GitHub 배포판을 받지 못하면(2026-11-18 github.com/microsoft/onnxruntime 404) 같은 판을 NuGet
        //    Microsoft.ML.OnnxRuntime 에서 꺼내 배포판과 같은 include/ · lib/ 로 둡니다. 헤더 · 라이브러리는 배포판과 바이트 단위로
        //    같습니다(1.28.0 win-x64 확인). CPU 판만: GPU 판은 NuGet 패키지가 달라 원래대로 실패합니다.
        let fetched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| download(&prebuilt_url, &downloaded_file)));
        if let Err(cause) = fetched {
            if prebuilt_archive.to_string_lossy().contains("gpu") {
                std::panic::resume_unwind(cause);
            }
            match prepare_from_nuget(&out_dir, &install_dir) {
                Ok(()) => {
                    println!(
                        "cargo:warning=ONNX Runtime {}: {} could not be downloaded; using NuGet Microsoft.ML.OnnxRuntime {} instead",
                        ORT_VERSION, prebuilt_url, ORT_VERSION
                    );
                    return install_dir;
                }
                Err(e) => panic!(
                    "ERROR: Failed to download {} and the NuGet fallback failed too: {}",
                    prebuilt_url, e
                ),
            }
        }
    }

    if !extract_dir.exists() {
        println!("Extracting to {}...", extract_dir.display());
        extract_archive(&downloaded_file, &extract_dir);
    }

    install_dir
}

/// NuGet 패키지 주소 (`{base}/{version}/microsoft.ml.onnxruntime.{version}.nupkg`)
const ORT_NUGET_BASE_URL: &str = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime";

/// NuGet 패키지 안의 런타임 식별자 (패키지에 없는 대상은 None)
fn nuget_runtime_id() -> Option<&'static str> {
    let os = env::var("CARGO_CFG_TARGET_OS").ok()?;
    let arch = env::var("CARGO_CFG_TARGET_ARCH").ok()?;
    Some(match (os.as_str(), arch.as_str()) {
        ("windows", "x86_64") => "win-x64",
        ("windows", "aarch64") => "win-arm64",
        ("linux", "x86_64") => "linux-x64",
        ("linux", "aarch64") => "linux-arm64",
        ("macos", "aarch64") => "osx-arm64",
        _ => return None,
    })
}

/// `download` 와 같지만 실패를 돌려줍니다 (NuGet 은 Content-Length 없이 보낼 수 있어 길이를 묻지 않음).
fn try_download(source_url: &str, target_file: &Path) -> Result<(), String> {
    let resp = ureq::get(source_url)
        .timeout(std::time::Duration::from_secs(300))
        .call()
        .map_err(|err| format!("{}: {:?}", source_url, err))?;
    let mut buffer = vec![];
    resp.into_reader()
        .read_to_end(&mut buffer)
        .map_err(|err| format!("{}: {}", source_url, err))?;
    let f = fs::File::create(target_file).map_err(|err| err.to_string())?;
    io::BufWriter::new(f)
        .write_all(&buffer)
        .map_err(|err| err.to_string())
}

/// NuGet `Microsoft.ML.OnnxRuntime` 에서 `build/native/include` → `include/`, `runtimes/<rid>/native` → `lib/`.
/// 실행할 때 로더가 찾는 이름(리눅스 SONAME `libonnxruntime.so.1`, 맥 install name `libonnxruntime.1.dylib`)도 함께 둡니다.
fn prepare_from_nuget(out_dir: &Path, install_dir: &Path) -> Result<(), String> {
    let rid = nuget_runtime_id().ok_or("this target has no runtime in the NuGet package")?;
    let package = out_dir.join(format!("microsoft.ml.onnxruntime.{}.nupkg", ORT_VERSION));
    if !package.exists() {
        let url = format!("{0}/{1}/microsoft.ml.onnxruntime.{1}.nupkg", ORT_NUGET_BASE_URL, ORT_VERSION);
        println!("Downloading {} into {}", url, package.display());
        try_download(&url, &package)?;
    }
    let file = fs::File::open(&package).map_err(|err| err.to_string())?;
    let mut archive = zip::ZipArchive::new(io::BufReader::new(file)).map_err(|err| err.to_string())?;
    let include_dir = install_dir.join("include");
    let lib_dir = install_dir.join("lib");
    let native = format!("runtimes/{}/native/", rid);
    let mut libs = 0usize;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|err| err.to_string())?;
        let name = entry.name().to_string();
        if name.ends_with('/') {
            continue;
        }
        let target = if let Some(rest) = name.strip_prefix("build/native/include/") {
            include_dir.join(rest)
        } else if let Some(rest) = name.strip_prefix(native.as_str()) {
            libs += 1;
            lib_dir.join(rest)
        } else {
            continue;
        };
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        let mut out = fs::File::create(&target).map_err(|err| err.to_string())?;
        io::copy(&mut entry, &mut out).map_err(|err| err.to_string())?;
    }
    if libs == 0 {
        return Err(format!("{} has no {}", package.display(), native));
    }
    for (have, loader_name) in [
        ("libonnxruntime.so", "libonnxruntime.so.1"),
        ("libonnxruntime.dylib", "libonnxruntime.1.dylib"),
    ] {
        let (src, dst) = (lib_dir.join(have), lib_dir.join(loader_name));
        if src.exists() && !dst.exists() && fs::hard_link(&src, &dst).is_err() {
            fs::copy(&src, &dst).map_err(|err| err.to_string())?;
        }
    }
    Ok(())
}

fn prepare_libort_dir() -> PathBuf {
    let strategy = env::var(ORT_ENV_STRATEGY);
    println!(
        "strategy: {:?}",
        strategy
            .as_ref()
            .map(String::as_str)
            .unwrap_or_else(|_| "unknown")
    );
    match strategy.as_ref().map(String::as_str) {
        Ok("download") | Err(_) => prepare_libort_dir_prebuilt(),
        Ok("system") => PathBuf::from(match env::var(ORT_ENV_SYSTEM_LIB_LOCATION) {
            Ok(p) => p,
            Err(e) => {
                panic!(
                    "Could not get value of environment variable {:?}: {:?}",
                    ORT_ENV_SYSTEM_LIB_LOCATION, e
                );
            }
        }),
        Ok("compile") => unimplemented!(),
        _ => panic!("Unknown value for {:?}", ORT_ENV_STRATEGY),
    }
}
