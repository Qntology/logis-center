#!/usr/bin/env bash
# env-report.sh <label> — record the runner environment as annotations (and in the step summary).
# Runner logs cannot be read from the working session (blob storage is 403), annotations can, so
# whatever explains a build failure (compilers, SDKs, PATH, build inputs) is written here, up front.
# GitHub shows at most 10 notices per step, so the facts are packed into four annotations.
# Never fails the job. Works in Git Bash (Windows) and bash (Linux).
set -u
label="${1:-env}"
os="$(uname -s 2>/dev/null || echo unknown)"
case "$os" in MINGW*|MSYS*|CYGWIN*) win=1 ;; *) win=0 ;; esac

have() { command -v "$1" >/dev/null 2>&1; }
# line NAME CMD... : first 3 lines of the command's output (stdout+stderr), CR stripped
line() { local n="$1"; shift; local v; v="$("$@" 2>&1 | head -3 | tr -d '\r' | tr '\n' '|')" || true; printf '%s: %s\n' "$n" "${v:-<none>}"; }
present() { local n="$1" v="${2:-}"; if [ -n "$v" ]; then printf '%s: %s\n' "$n" "${v:0:160}"; else printf '%s: <unset>\n' "$n"; fi; }

emit() {
  local title="$1" body="$2" msg
  msg="${body//%/%25}"; msg="${msg//$'\r'/}"; msg="${msg//$'\n'/%0A}"
  echo "::notice title=env ${label} / ${title}::${msg}"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    { echo "#### env ${label} / ${title}"; echo '```'; printf '%s\n' "$body"; echo '```'; } >> "$GITHUB_STEP_SUMMARY"
  fi
}

# ── 1. runner ──
g1="$(
  echo "uname: $(uname -a 2>/dev/null | cut -c1-200)"
  if [ "$win" = 1 ]; then
    line "windows" powershell -NoProfile -Command '(Get-CimInstance Win32_OperatingSystem | % { $_.Caption + " " + $_.Version + " " + $_.OSArchitecture })'
    line "memory-GB" powershell -NoProfile -Command '[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)'
    echo "cores: ${NUMBER_OF_PROCESSORS:-?}"
    line "disk-free" powershell -NoProfile -Command '(Get-PSDrive -Name (Get-Location).Drive.Name | % { [math]::Round($_.Free/1GB,1).ToString() + " GB free" })'
    line "long-paths" powershell -NoProfile -Command '(Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem").LongPathsEnabled'
  else
    line "os" bash -c '. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME"'
    echo "cores: $(nproc 2>/dev/null)"
    line "memory-MB" bash -c "free -m | awk '/Mem:/{print \$2\" total, \"\$7\" available\"}'"
    line "disk-free" bash -c "df -h . | tail -1"
  fi
  echo "image: ${ImageOS:-?} ${ImageVersion:-?}"
  echo "cwd: $(pwd) (length $(pwd | wc -c))"
)"
emit "runner" "$g1"

# ── 2. toolchain ──
g2="$(
  line "rustc" rustc -vV
  line "cargo" cargo -V
  line "git" git --version
  line "python" python --version
  line "protoc" protoc --version
  if [ "$win" = 1 ]; then
    pf86="${PROGRAMFILES_X86_UNIX:-/c/Program Files (x86)}"
    vsw="$pf86/Microsoft Visual Studio/Installer/vswhere.exe"
    if [ -x "$vsw" ]; then
      line "vswhere" "$vsw" -all -products '*' -format value -property installationVersion -property installationPath
      line "vs-components" "$vsw" -all -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format value -property displayName
    else
      echo "vswhere: <not found>"
    fi
    line "cl (PATH)" where.exe cl
    line "link (PATH)" where.exe link
    have cl && line "cl banner" cl
    line "autocrlf" git config --get core.autocrlf
  else
    line "gcc" gcc --version
    line "g++" g++ --version
    line "clang" clang --version
    line "pkg-config" pkg-config --version
    line "webkit2gtk-4.1" pkg-config --modversion webkit2gtk-4.1
    line "gtk+-3.0" pkg-config --modversion gtk+-3.0
    line "libsoup-3.0" pkg-config --modversion libsoup-3.0
    line "openssl" pkg-config --modversion openssl
  fi
)"
emit "toolchain" "$g2"

# ── 3. GPU SDKs ──
g3="$(
  if have nvcc; then line "nvcc" bash -c 'nvcc --version | tail -2'; else echo "nvcc: <not in PATH>"; fi
  present "CUDA_PATH" "${CUDA_PATH:-}"
  present "CUDA_HOME" "${CUDA_HOME:-}"
  present "NVCC_CCBIN" "${NVCC_CCBIN:-}"
  present "CUDA_COMPUTE_CAP" "${CUDA_COMPUTE_CAP:-}"
  if [ "$win" = 1 ]; then
    line "cuda-dirs" bash -c 'ls "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA" 2>&1'
  else
    line "cuda-dirs" bash -c 'ls -d /usr/local/cuda* /usr/lib/cuda* 2>&1'
  fi
  if have hipcc; then line "hipcc" hipcc --version; else echo "hipcc: <not in PATH>"; fi
  present "ROCM_PATH" "${ROCM_PATH:-}"
  present "HIP_PATH" "${HIP_PATH:-}"
  present "CANDLE_ROCM_SKIP_KERNEL_BUILD" "${CANDLE_ROCM_SKIP_KERNEL_BUILD:-}"
  present "VULKAN_SDK" "${VULKAN_SDK:-}"
  present "VK_ICD_FILENAMES" "${VK_ICD_FILENAMES:-}"
  present "CANDLE_VULKAN_ALLOW_CPU" "${CANDLE_VULKAN_ALLOW_CPU:-}"
  if [ "$win" = 1 ]; then
    line "vulkan-1.dll" where.exe vulkan-1.dll
  else
    have vulkaninfo && line "vulkaninfo" bash -c 'vulkaninfo --summary 2>&1 | grep -E "deviceName|apiVersion|driverName" | head -4' || echo "vulkaninfo: <not installed>"
  fi
)"
emit "gpu-sdk" "$g3"

# ── 4. build inputs + PATH names ──
g4="$(
  root="${GITHUB_WORKSPACE:-.}/logis-center/commerce"
  [ -d "$root" ] || root="./commerce"
  [ -f "$root/dist/index.html" ] && echo "dist/index.html: present" || echo "dist/index.html: MISSING"
  if [ -d "$root/src-tauri/dlls" ] && [ -n "$(ls -A "$root/src-tauri/dlls" 2>/dev/null)" ]; then echo "src-tauri/dlls: non-empty"; else echo "src-tauri/dlls: MISSING or empty"; fi
  [ -f "$root/src-tauri/Cargo.toml" ] && echo "fork path lines: $(grep -cE '^candle-(rocm|core|nn|flash-attn) = \{ path' "$root/src-tauri/Cargo.toml")"
  present "RUSTFLAGS" "${RUSTFLAGS:-}"
  present "CARGO_ENCODED_RUSTFLAGS" "${CARGO_ENCODED_RUSTFLAGS:-}"
  present "LIB" "${LIB:-}"
  present "INCLUDE" "${INCLUDE:-}"
  echo "PATH entries (compiler / SDK related): $(echo "$PATH" | tr ':;' '\n' | grep -iE 'cuda|msvc|visual studio|vc/tools|windows kits|rocm|hip|vulkan|cargo|rustup|llvm|mingw' | head -12 | tr '\n' '|')"
)"
emit "build-inputs" "$g4"
exit 0
