# Picks the ROCm installation used by the *_rocm.bat scripts (called from _env_rocm.bat).
#
# Two release streams ("tracks") are supported side by side, with the same rules as the
# candle-rocm-vulkan build scripts (hip-sys/build.rs):
#   legacy : AMD HIP SDK 6.x / 7.x   (C:\Program Files\AMD\ROCm\7.2, HIP 7.2 and older)
#   core   : ROCm Core SDK 10.x      (TheRock tarball / rocm-sdk wheels, HIP 7.10 and later)
# CANDLE_ROCM_TRACK=auto (default) takes the newest core installation when every GPU of this PC is
# supported by it, otherwise the newest legacy one. legacy / core force a track.
#
# Searched: `rocm-sdk path --root` (pip), C:\Program Files\AMD\ROCm\*, and the folders listed in
# ROCM_SEARCH_PATHS (separated by ';', e.g. where a ROCm 10.x tarball was extracted).
#
# Prints KEY=VALUE lines: ROOT, VER, TRACK, HIP, ARCHS, NOTE. Exit code 1 when nothing is found.

$ErrorActionPreference = 'SilentlyContinue'

# GPU targets supported by ROCm Core SDK 10.1.
$CoreArchs = @('gfx908', 'gfx90a', 'gfx942', 'gfx950', 'gfx1030', 'gfx1100', 'gfx1101', 'gfx1102', 'gfx1103',
               'gfx1150', 'gfx1151', 'gfx1152', 'gfx1153', 'gfx1200', 'gfx1201')

function Get-HipVersion([string]$Root) {
    $header = Join-Path $Root 'include\hip\hip_version.h'
    if (-not (Test-Path -LiteralPath $header)) { return $null }
    $v = @{}
    foreach ($line in Get-Content -LiteralPath $header) {
        if ($line -match '^\s*#define\s+HIP_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)') { $v[$Matches[1]] = [int]$Matches[2] }
    }
    if (-not $v.ContainsKey('MAJOR')) { return $null }
    return New-Object System.Version ([int]$v['MAJOR']), ([int]$v['MINOR']), ([int]$v['PATCH'])
}

function Get-DirNumbers([string]$Root) {
    $leaf = Split-Path -Leaf $Root
    return @([regex]::Matches($leaf, '\d+') | ForEach-Object { [int]$_.Value })
}

function Get-Track([string]$Root, $Hip) {
    if ($Hip) {
        if ($Hip.Major -gt 7 -or ($Hip.Major -eq 7 -and $Hip.Minor -ge 10)) { return 'core' }
        return 'legacy'
    }
    $leaf = (Split-Path -Leaf $Root).ToLower()
    $nums = Get-DirNumbers $Root
    if ($leaf.StartsWith('core') -or $leaf.StartsWith('runtime') -or ($nums.Count -gt 0 -and $nums[0] -ge 10)) { return 'core' }
    return 'legacy'
}

function Test-Usable([string]$Root) {
    $compiler = (Test-Path -LiteralPath (Join-Path $Root 'bin\hipcc.exe')) -or
                (Test-Path -LiteralPath (Join-Path $Root 'lib\llvm\bin\clang++.exe')) -or
                (Test-Path -LiteralPath (Join-Path $Root 'bin\clang++.exe'))
    $libs = (Test-Path -LiteralPath (Join-Path $Root 'lib\amdhip64.lib')) -and
            (Test-Path -LiteralPath (Join-Path $Root 'lib\rocblas.lib'))
    return ($compiler -and $libs)
}

function Get-GpuArchs($Installs) {
    $tools = @('lib\llvm\bin\amdgpu-arch.exe', 'lib\llvm\bin\offload-arch.exe', 'bin\amdgpu-arch.exe', 'bin\hipInfo.exe')
    foreach ($inst in $Installs) {
        foreach ($tool in $tools) {
            $exe = Join-Path $inst.Root $tool
            if (-not (Test-Path -LiteralPath $exe)) { continue }
            $savedPath = $env:PATH
            # Let the tool load the HIP runtime (amdhip64_*.dll) of the same installation.
            $env:PATH = (Join-Path $inst.Root 'bin') + ';' + $env:PATH
            $out = (& $exe 2>$null | Out-String)
            $env:PATH = $savedPath
            $archs = @([regex]::Matches($out, 'gfx[0-9a-f]+') | ForEach-Object { $_.Value } |
                       Where-Object { $_ -ne 'gfx000' } | Select-Object -Unique)
            if ($archs.Count -gt 0) { return $archs }
        }
    }
    return @()
}

$roots = @()
$sdk = Get-Command rocm-sdk -ErrorAction SilentlyContinue
if ($sdk) {
    $r = (& $sdk.Path path --root 2>$null | Select-Object -First 1)
    if ($r) { $roots += $r.Trim() }
}
$base = Join-Path $env:ProgramFiles 'AMD\ROCm'
if (Test-Path -LiteralPath $base) {
    $roots += @(Get-ChildItem -LiteralPath $base -Directory | ForEach-Object { $_.FullName })
}
if ($env:ROCM_SEARCH_PATHS) {
    $roots += @($env:ROCM_SEARCH_PATHS.Split(';') | ForEach-Object { $_.Trim().Trim('"').TrimEnd('\') } | Where-Object { $_ })
}

$installs = @()
foreach ($root in ($roots | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $root)) { continue }
    if (-not (Test-Usable $root)) { continue }
    $hip = Get-HipVersion $root
    $installs += [pscustomobject]@{
        Root  = $root
        Ver   = (Split-Path -Leaf $root)
        Hip   = $hip
        Track = (Get-Track $root $hip)
        Key   = $(if ($hip) { $hip } else { New-Object System.Version 0, 0, 0 })
        Dir   = ((Get-DirNumbers $root | ForEach-Object { '{0:D6}' -f $_ }) -join '.')
    }
}
if ($installs.Count -eq 0) { exit 1 }
$installs = @($installs | Sort-Object -Property @{ Expression = 'Key'; Descending = $true }, @{ Expression = 'Dir'; Descending = $true })

$track = ('' + $env:CANDLE_ROCM_TRACK).Trim().ToLower()
if ($track -eq 'therock') { $track = 'core' }
if (@('legacy', 'core') -notcontains $track) { $track = 'auto' }

$core = $installs | Where-Object { $_.Track -eq 'core' } | Select-Object -First 1
$legacy = $installs | Where-Object { $_.Track -eq 'legacy' } | Select-Object -First 1
$archs = @()
$note = ''
$sel = $null
if ($track -eq 'core') {
    $sel = $core
    if (-not $sel) { $note = 'CANDLE_ROCM_TRACK=core but no ROCm Core SDK 10.x was found' }
} elseif ($track -eq 'legacy') {
    $sel = $legacy
    if (-not $sel) { $note = 'CANDLE_ROCM_TRACK=legacy but no HIP SDK 7.x or older was found' }
} elseif ($core -and $legacy) {
    $archs = @(Get-GpuArchs $installs)
    $unsupported = @($archs | Where-Object { $CoreArchs -notcontains $_ })
    if ($unsupported.Count -eq 0) {
        $sel = $core
    } else {
        $sel = $legacy
        $note = "$($unsupported -join ',') is not supported by ROCm $($core.Ver), using the legacy HIP SDK $($legacy.Ver); set CANDLE_ROCM_TRACK=core to override"
    }
} elseif ($core) {
    $sel = $core
} else {
    $sel = $legacy
}

if (-not $sel) {
    if ($note) { Write-Output "NOTE=$note" }
    exit 1
}
Write-Output "ROOT=$($sel.Root)"
Write-Output "VER=$($sel.Ver)"
Write-Output "TRACK=$($sel.Track)"
Write-Output "HIP=$(if ($sel.Hip) { $sel.Hip.ToString() } else { 'unknown' })"
Write-Output "ARCHS=$($archs -join ',')"
Write-Output "NOTE=$note"
exit 0
