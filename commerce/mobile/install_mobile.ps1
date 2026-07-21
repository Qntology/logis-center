# ==========================================================================================
# 안드로이드 하이브리드 빌드 및 자동 설정 스크립트
# ==========================================================================================
$ErrorActionPreference = "Continue" 
Write-Host ">>> [하이브리드 모드] 빌드 및 설정 자동 복구를 시작합니다..." -ForegroundColor Cyan

$projectRoot = "C:\Users\HP\Documents\GitHub\cron-logis-center\commerce\rust"
$mobilePath = "$projectRoot\mobile"
$appId = "com.logis.scanner.v111"
$adbPath = "C:\Users\HP\AppData\Local\Android\Sdk\platform-tools\adb.exe"

# Node.js 경로 설정
$env:PATH = "C:\Program Files\nodejs\;" + $env:PATH
$env:PATH = "C:\Users\HP\AppData\Roaming\npm\;" + $env:PATH

# 0. 딥 클린 (Deep Clean) - 이전 빌드 결과물 삭제
Write-Host "[0/5] 클린 작업 실행 중..." -ForegroundColor Yellow
$distPath = "$projectRoot\dist\mobile"
$androidAssetsPath = "$mobilePath\src-tauri\gen/android\app/src/main/assets\_tauri_assets"
$localPropertiesPath = "$mobilePath\src-tauri\gen/android\local.properties"
$manifestPath = "$mobilePath\src-tauri\gen/android\app/src/main/AndroidManifest.xml"

if (Test-Path $distPath) { Remove-Item -Recurse -Force $distPath }
if (Test-Path $androidAssetsPath) { Remove-Item -Recurse -Force $androidAssetsPath }

# 0.1 안드로이드 SDK 경로 설정 (local.properties)
if (!(Test-Path $localPropertiesPath)) {
    Write-Host "  - local.properties 복구 중..." -ForegroundColor Cyan
    "sdk.dir=C\:/Users/HP/AppData/Local/Android/Sdk" | Out-File -FilePath $localPropertiesPath -Encoding ascii
}

# 0.2 AndroidManifest.xml 권한 자동 복구 (카메라 권한 등)
if (Test-Path $manifestPath) {
    $manifestContent = Get-Content $manifestPath -Raw
    if ($manifestContent -notmatch "android.permission.CAMERA") {
        Write-Host "  - AndroidManifest.xml에 카메라 권한 추가 중..." -ForegroundColor Cyan
        $newPerms = @"
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-feature android:name="android.hardware.camera" android:required="false" />
    <uses-feature android:name="android.hardware.camera.autofocus" android:required="false" />
"@
        $manifestContent = $manifestContent -replace '<uses-permission android:name="android.permission.INTERNET" />', $newPerms
        $manifestContent | Out-File -FilePath $manifestPath -Encoding utf8
    }
}

# 1. 웹 프론트엔드 빌드
Write-Host "[1/5] 웹 프론트엔드 빌드 중..." -ForegroundColor Yellow
cd $mobilePath
npm run build

# 2. Tauri 안드로이드 빌드 (Rust 컴파일)
Write-Host "[2/5] Tauri CLI 실행 (Rust 컴파일 시도)..." -ForegroundColor Yellow
npx tauri android build --debug

# 3. 라이브러리 및 자산 수동 복사 (심볼릭 링크 에러 우회)
Write-Host "[3/5] 자산 동기화 중..." -ForegroundColor Yellow
$jniPath = "$mobilePath\src-tauri\gen/android\app/src/main/jniLibs/arm64-v8a"
if (!(Test-Path $jniPath)) { New-Item -ItemType Directory -Path $jniPath -Force | Out-Null }
if (Test-Path "$mobilePath\src-tauri\target\aarch64-linux-android\debug\libmobile_app_lib.so") {
    Copy-Item -Force "$mobilePath\src-tauri\target\aarch64-linux-android\debug\libmobile_app_lib.so" "$jniPath\libmobile_app_lib.so"
}

if (!(Test-Path $androidAssetsPath)) { New-Item -ItemType Directory -Path $androidAssetsPath -Force | Out-Null }
Copy-Item -Recurse -Force "$projectRoot\dist\mobile\*" "$androidAssetsPath\"

# 4. Gradle 최종 패키징
Write-Host "[4/5] Gradle 최종 APK 생성 중..." -ForegroundColor Yellow
cd "$mobilePath\src-tauri\gen/android"
./gradlew :app:assembleArm64Debug -x :app:rustBuildArm64Debug -x :app:rustBuildArmDebug

# 5. 설치 및 실행
Write-Host "[5/5] APK 설치 및 실행 중..." -ForegroundColor Yellow
$apkFile = "$mobilePath\src-tauri\gen/android\app\build\outputs\apk\arm64\debug\app-arm64-debug.apk"
if (Test-Path $apkFile) {
    # Uninstall existing to prevent version conflicts
    cmd /c "$adbPath uninstall $appId" 2>$null | Out-Null
    cmd /c "$adbPath install -r -t -d $apkFile"
    Write-Host ">>> [성공] 빌드 및 권한 복구 완료!" -ForegroundColor Green
    cmd /c "$adbPath shell monkey -p $appId -c android.intent.category.LAUNCHER 1" | Out-Null
} else {
    Write-Host "ERR: APK 생성 실패!" -ForegroundColor Red
}
