# ==========================================================================================
# 안드로이드 하이브리드 빌드 스크립트 (Tauri CLI + 수동 복사)
# ==========================================================================================
$ErrorActionPreference = "Continue" 
Write-Host ">>> [하이브리드 모드] 정석 빌드와 수동 우회를 결합합니다..." -ForegroundColor Cyan

$projectRoot = "C:\Users\HP\Documents\GitHub\cron-logis-center\commerce\rust"
$mobilePath = "$projectRoot\mobile"
$appId = "com.logis.scanner.v110"
$adbPath = "C:\Users\HP\AppData\Local\Android\Sdk\platform-tools\adb.exe"

# Node.js 경로 설정
$env:PATH = "C:\Program Files\nodejs\;" + $env:PATH
$env:PATH = "C:\Users\HP\AppData\Roaming\npm\;" + $env:PATH

# 1. 웹 프론트엔드 빌드
Write-Host "[1/5] 웹 프론트엔드 빌드 중..." -ForegroundColor Yellow
cd $mobilePath
npm run build

# 2. Tauri 안드로이드 빌드 (Rust 컴파일 수행)
Write-Host "[2/5] Tauri CLI 실행 (Rust 컴파일 시도)..." -ForegroundColor Yellow
npx tauri android build --debug

# 3. 라이브러리 및 자산 수동 복사 (심볼릭 링크 에러 우회)
Write-Host "[3/5] 빌드 결과물 수동 복사 중 (Link Error 우회)..." -ForegroundColor Yellow
$jniPath = "$mobilePath\src-tauri\gen/android\app/src/main/jniLibs/arm64-v8a"
if (!(Test-Path $jniPath)) { New-Item -ItemType Directory -Path $jniPath -Force | Out-Null }
Copy-Item -Force "$mobilePath\src-tauri\target\aarch64-linux-android\debug\libmobile_app_lib.so" "$jniPath\libmobile_app_lib.so"

$targetAssets = "$mobilePath\src-tauri\gen/android\app/src/main/assets\_tauri_assets"
if (Test-Path $targetAssets) { Remove-Item -Recurse -Force $targetAssets }
New-Item -ItemType Directory -Path $targetAssets -Force | Out-Null
Copy-Item -Recurse -Force "$projectRoot\dist\mobile\*" "$targetAssets\"

# 4. Gradle 최종 패키징
Write-Host "[4/5] Gradle 최종 APK 생성 중 (이미 빌드된 Rust는 스킵)..." -ForegroundColor Yellow
cd "$mobilePath\src-tauri\gen/android"
./gradlew clean
./gradlew :app:assembleArm64Debug -x :app:rustBuildArm64Debug -x :app:rustBuildArmDebug --offline

# 5. 설치 및 실행
Write-Host "[5/5] APK 설치 및 실행 중..." -ForegroundColor Yellow
$apkFile = "$mobilePath\src-tauri\gen/android\app\build\outputs\apk\arm64\debug\app-arm64-debug.apk"
if (Test-Path $apkFile) {
    cmd /c "$adbPath uninstall $appId" 2>$null | Out-Null
    cmd /c "$adbPath install -r -t -d $apkFile"
    Write-Host ">>> [성공] 하이브리드 빌드 완료!" -ForegroundColor Green
    cmd /c "$adbPath shell monkey -p $appId -c android.intent.category.LAUNCHER 1" | Out-Null
} else {
    Write-Host "ERR: APK 생성 실패!" -ForegroundColor Red
}