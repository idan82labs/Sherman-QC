<#
.SYNOPSIS
  Build, install and self-check the portable zs_rust binary (zero-shot bend lane v3) on the Windows x86-64
  station. Docs: docs/production/runlog/zeroshot_2026-10-25/v3_rust/PORTABLE.md (section "Windows station").

.DESCRIPTION
  -Mode Install   (default) build with the pinned toolchain, copy binary + models + config + golden set into
                  -InstallDir, check every sha256 against the pin record, run `selftest --golden` at 1 and N
                  threads, and write install_record.json. Any mismatch or self-test failure aborts (exit 3) and
                  leaves no runnable install (the binary is renamed *.refused).
  -Mode Startup   at every station start: re-hash the installed files against install_record.json, then run
                  `selftest --golden`. Exit 0 = the lane may run; anything else = refuse the lane.

  Build: cargo build --release --locked --target x86_64-pc-windows-msvc, RUSTFLAGS "-C target-cpu=<TargetCpu>"
  (plus /Brepro and /PDBALTPATH link args so a rebuild at another path gives the same sha256).
  -TargetCpu x86-64-v2 (default): SSE4.2, no AVX. The explicit fused multiply-adds go through libm::fma, which
    uses the CPU's FMA instruction when present (run-time detection) and gives the same bits either way. This is
    the x86 code path proven byte-identical to the arm64 reference on all 125 DEV scans and 18 dense meshes.
  -TargetCpu x86-64-v3 (AVX2 + FMA, the originally planned flag): under Rosetta its output differed from the
    reference on 1 of 122 DEV scans (sc_766be04dc456: two Z2 candidates wrongly pooled as duplicates; it moved
    the DEV OOF by one FP). The cause was traced to an AVX min/blend sequence in the compiled pooling code and
    could not be separated between Rosetta's AVX translation and the compiler without real x86 hardware
    (PORTABLE.md). Use v3 only after it reproduces the reference on the station hardware (see PORTABLE.md). Requires the MSVC linker and Windows SDK (Visual Studio Build Tools,
  licence accepted by the station owner) and rustup with the toolchain named in the pin record.

  Signing (optional, -SignThumbprint): signtool changes the file and so its sha256. The install record keeps the
  unsigned sha256 (build output, reproducible) and the signed sha256 (what runs); the startup check verifies the
  signed one. Signing never changes outputs; the self-test runs again after signing.

  The script never touches the production runner or its pins: wiring the lane into lane_cli is a separate step.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File build_install.ps1 -RepoRoot C:\ShermanQC\FullProject `
      -InstallDir C:\ShermanQC\zs_rust -Revision 3
  powershell -ExecutionPolicy Bypass -File C:\ShermanQC\zs_rust\build_install.ps1 -Mode Startup `
      -InstallDir C:\ShermanQC\zs_rust
#>
[CmdletBinding()]
param(
    [ValidateSet("Install", "Startup")] [string] $Mode = "Install",
    [string] $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path,
    [string] $InstallDir = "C:\ShermanQC\zs_rust",
    [ValidateSet(2, 3)] [int] $Revision = 3,
    [ValidateSet("x86-64-v2", "x86-64-v3")] [string] $TargetCpu = "x86-64-v2",
    [int] $Threads = 0,
    [string] $SignThumbprint = "",
    [string] $TimestampUrl = "http://timestamp.digicert.com",
    [switch] $SkipBuild
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

function Sha([string] $p) { (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLowerInvariant() }
function Refuse([string] $msg) {
    Write-Host "zs_rust station: REFUSED: $msg" -ForegroundColor Red
    exit 3
}
function PhysicalCores {
    $n = 0
    foreach ($c in (Get-CimInstance Win32_Processor)) { $n += [int]$c.NumberOfCores }
    if ($n -lt 1) { $n = [Environment]::ProcessorCount }
    return $n
}
if ($Threads -le 0) { $Threads = PhysicalCores }

$cfgName = if ($Revision -eq 3) { "config_v3r3_rs.json" } else { "config_v3_rs.json" }
$pinName = "zs_lane_v3_rs.rev$Revision.pin.json"
$relAdapter = "research\zeroshot\zs_lane_v3_rs"
$relCrate = "research\zeroshot\zs_rust"

function RunSelftest([string] $exe, [string] $root, [int] $threads, [string] $report) {
    $cfg = Join-Path $root "$relAdapter\$cfgName"
    $golden = Join-Path $root "$relCrate\golden"
    & $exe selftest --golden $golden --config $cfg --repo $root --threads $threads --report $report | Out-Null
    return $LASTEXITCODE
}

if ($Mode -eq "Startup") {
    $recP = Join-Path $InstallDir "install_record.json"
    if (-not (Test-Path $recP)) { Refuse "no install_record.json in $InstallDir" }
    $rec = Get-Content -Raw $recP | ConvertFrom-Json
    foreach ($f in $rec.files.PSObject.Properties) {
        $p = Join-Path $InstallDir $f.Name
        if (-not (Test-Path $p)) { Refuse "missing $($f.Name)" }
        if ((Sha $p) -ne $f.Value) { Refuse "sha256 differs: $($f.Name)" }
    }
    $exe = Join-Path $InstallDir "bin\zs_rust.exe"
    $rc = RunSelftest $exe $InstallDir $Threads (Join-Path $InstallDir "selftest_startup.json")
    if ($rc -ne 0) { Refuse "selftest --golden exit $rc (see selftest_startup.json)" }
    Write-Host "zs_rust station: startup self-check passed ($($rec.binary.sha256_installed))"
    exit 0
}

# ------------------------------------------------------------------------------------------------- install
$pinP = Join-Path $RepoRoot "$relAdapter\pins\$pinName"
if (-not (Test-Path $pinP)) { Refuse "pin record not found: $pinP" }
$pin = Get-Content -Raw $pinP | ConvertFrom-Json
$crate = Join-Path $RepoRoot $relCrate

# 1. toolchain and inputs equal the pin record
$rustc = (& rustc --version) -replace "^rustc (\S+).*$", '$1'
if ($rustc -ne $pin.binary.rustc) { Refuse "rustc $rustc, pin record wants $($pin.binary.rustc) (rustup toolchain install $($pin.binary.rustc))" }
if ((Sha (Join-Path $crate "Cargo.lock")) -ne $pin.binary.cargo_lock_sha256) { Refuse "Cargo.lock differs from the pin record" }
foreach ($m in $pin.models.PSObject.Properties) {
    if ((Sha (Join-Path $RepoRoot $m.Value.path)) -ne $m.Value.sha256) { Refuse "model $($m.Name) differs from the pin record" }
}
if ((Sha (Join-Path $RepoRoot $pin.config.path)) -ne $pin.config.sha256) { Refuse "config differs from the pin record" }
if ((Sha (Join-Path $crate "golden\golden.json")) -ne $pin.golden.golden_json_sha256) { Refuse "golden.json differs from the pin record" }

# 2. build (x86-64-v3, locked dependencies; the crates must be in the local cargo cache or vendored)
$target = "x86_64-pc-windows-msvc"
$built = Join-Path $crate "target\$target\release\zs_rust.exe"
if (-not $SkipBuild) {
    # /Brepro: deterministic PE timestamp; /PDBALTPATH: no absolute build path in the binary (reproducible sha256)
    $env:RUSTFLAGS = "-C target-cpu=$TargetCpu -C link-arg=/Brepro -C link-arg=/PDBALTPATH:%_PDB%"
    Push-Location $crate
    try {
        & cargo build --release --locked --target $target
        if ($LASTEXITCODE -ne 0) { Refuse "cargo build failed ($LASTEXITCODE)" }
    } finally { Pop-Location; Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue }
}
if (-not (Test-Path $built)) { Refuse "no binary at $built" }
$shaUnsigned = Sha $built
$known = $pin.binaries.PSObject.Properties | Where-Object { $_.Name -eq "$target+$TargetCpu" }
if ($known -and $known.Value.sha256) {
    if ($known.Value.sha256 -ne $shaUnsigned) { Refuse "unsigned build sha256 $shaUnsigned differs from the pinned $($known.Value.sha256)" }
} else {
    Write-Host "note: the pin record has no $target build sha256 yet; recording $shaUnsigned (first station build)"
}

# 3. the CPU runs the binary (x86-64-v2: SSE4.2 / POPCNT; x86-64-v3 also AVX2 / FMA / BMI2)
& $built version --build | Out-Null
if ($LASTEXITCODE -ne 0) { Refuse "the binary does not start on this CPU ($TargetCpu required); exit $LASTEXITCODE" }

# 4. stage the install tree (repo-relative layout, so the pinned config resolves its model paths with --repo)
$stage = "$InstallDir.staging"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path (Join-Path $stage "bin"), (Join-Path $stage "$relAdapter\models"), (Join-Path $stage "$relAdapter\pins"), (Join-Path $stage "$relCrate\golden") | Out-Null
$exe = Join-Path $stage "bin\zs_rust.exe"
Copy-Item $built $exe
Copy-Item (Join-Path $RepoRoot $pin.config.path) (Join-Path $stage "$relAdapter\$cfgName")
foreach ($m in $pin.models.PSObject.Properties) { Copy-Item (Join-Path $RepoRoot $m.Value.path) (Join-Path $stage $m.Value.path) }
Copy-Item $pinP (Join-Path $stage "$relAdapter\pins\$pinName")
Get-ChildItem (Join-Path $crate "golden") -File | Where-Object { $_.Extension -in ".json", ".ply", ".stl" } |
    ForEach-Object { Copy-Item $_.FullName (Join-Path $stage "$relCrate\golden\$($_.Name)") }
Copy-Item $PSCommandPath (Join-Path $stage "build_install.ps1")

# 5. optional code signing (changes the sha256; outputs unchanged)
$shaSigned = $null
if ($SignThumbprint) {
    & signtool sign /fd SHA256 /tr $TimestampUrl /td SHA256 /sha1 $SignThumbprint $exe
    if ($LASTEXITCODE -ne 0) { Refuse "signtool failed ($LASTEXITCODE)" }
    & signtool verify /pa $exe | Out-Null
    if ($LASTEXITCODE -ne 0) { Refuse "signature does not verify" }
    $shaSigned = Sha $exe
}

# 6. golden self-test at 1 thread and at the station's thread count: refuse on any difference
foreach ($t in @(1, $Threads)) {
    $rc = RunSelftest $exe $stage $t (Join-Path $stage "selftest_install_t$t.json")
    if ($rc -ne 0) {
        Rename-Item $exe "$exe.refused"
        Refuse "selftest --golden at $t threads: exit $rc (see $stage\selftest_install_t$t.json)"
    }
}

# 7. install record (what the startup check verifies), then move the staged tree into place
$files = [ordered]@{}
foreach ($f in Get-ChildItem -Recurse -File $stage | Where-Object { $_.Name -notlike "selftest_*" -and $_.Name -ne "build_install.ps1" }) {
    $files[$f.FullName.Substring($stage.Length + 1)] = Sha $f.FullName
}
$record = [ordered]@{
    schema = "sherman.zs_rust.station_install.v1"
    installed_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    host = $env:COMPUTERNAME
    cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
    threads = $Threads
    revision = $Revision
    pin = $pinName
    binary = [ordered]@{ target = "$target+$TargetCpu"; rustc = $rustc; sha256_unsigned = $shaUnsigned;
                         sha256_signed = $shaSigned; sha256_installed = (Sha $exe) }
    files = $files
    selftest = "passed at threads 1 and $Threads"
}
$record | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $stage "install_record.json")
if (Test-Path $InstallDir) { Rename-Item $InstallDir "$InstallDir.previous.$((Get-Date).ToString('yyyyMMddHHmmss'))" }
Rename-Item $stage $InstallDir
Write-Host "zs_rust station: installed $InstallDir (unsigned $shaUnsigned; installed $($record.binary.sha256_installed))"
Write-Host "record the station build in the pin: binaries['$target+$TargetCpu'].sha256 = $shaUnsigned"
exit 0
