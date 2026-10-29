<#
.SYNOPSIS
  Batch-export SolidWorks parts (.SLDPRT) to STEP AP214 (mm) over COM, read-only, with a manifest.
  Docs: README.md next to this file. Python side: python -m apps.production_line.recipes.cad_import.

.DESCRIPTION
  Runs on stock Windows PowerShell 5.1 with SolidWorks installed. Nothing is installed. Enum values are
  taken from the installed SolidWorks' own SolidWorks.Interop.swconst.dll (<SolidWorks dir>\api\redist),
  so no SolidWorks API number is hard-coded; the script refuses (exit 3) when that DLL cannot be found.

  For every *.SLDPRT under -Source (recursive) or listed in -ListFile:
    1. sha256 + size of the source (before).
    2. Up to date? Same source sha256 as the previous manifest row and the STEP still exists with the
       recorded sha256 -> SKIPPED_UP_TO_DATE (unless -Force).
    3. OpenDoc6 Silent|ReadOnly. Errors -> FAILED_OPEN (future version, missing file, ...). Missing
       references and rebuild errors are recorded as warnings; the export is still attempted.
    4. Active configuration (or every configuration with -AllConfigurations -> <name>__<config>.step).
    5. Document length unit forced to mm IN MEMORY ONLY (never saved), then Extension.SaveAs to
       <name>.~tmp.step (Silent|Copy) and an atomic rename to <name>.step.
    6. CloseAllDocuments(discard) - the source is never saved. sha256 of the source (after); a
       difference -> SOURCE_CHANGED.
  The user preference swStepAP is set to 214 at start and restored at the end (also restored at the next
  start if a run was killed: <LogDir>\steppref.restore.json).

  Per-file timeout: a watchdog job kills SLDWORKS.exe when one file takes longer than -TimeoutSec; the file
  is retried once in a fresh SolidWorks, then marked TIMEOUT. SolidWorks is restarted every -RestartEvery
  files to cap memory. *.SLDASM / *.SLDDRW are reported as UNSUPPORTED_TYPE. Any path containing _SEALED is
  skipped and only counted (never listed).

  Output: <Output>\manifest.json and manifest.csv (one row per source x configuration). -DryRun writes
  <LogDir>\manifest.dryrun.json/.csv, starts no SolidWorks and writes no STEP.
  Exit: 0 all OK/skipped, 1 at least one FAILED_OPEN/FAILED_EXPORT/TIMEOUT/SOURCE_CHANGED, 2 bad arguments,
  3 refused (SolidWorks / its constants not available, or SolidWorks already running without -AttachRunning).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File sldprt_to_step.ps1 -Source D:\CAD\Parts -Output D:\CAD\STEP_out
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File sldprt_to_step.ps1 -ListFile first_batch.txt -Output D:\STEP -DryRun
#>
[CmdletBinding()]
param(
    [string[]] $Source = @(),
    [string] $ListFile = "",
    [Parameter(Mandatory = $true)] [string] $Output,
    [switch] $DryRun,
    [switch] $Force,
    [switch] $AllConfigurations,
    [int] $TimeoutSec = 300,
    [string] $LogDir = "",
    [int] $RestartEvery = 25,
    [switch] $AttachRunning,
    [string] $SwConstDll = ""
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$ScriptVersion = "sldprt_to_step 1.0 (2026-10-28)"
$FailStatuses = @("FAILED_OPEN", "FAILED_EXPORT", "TIMEOUT", "SOURCE_CHANGED")

function Sha([string] $p) { (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLowerInvariant() }
function UtcNow { (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ") }
function Refuse([string] $msg, [int] $code = 3) {
    Write-Host "sldprt_to_step: REFUSED: $msg" -ForegroundColor Red
    exit $code
}

# ------------------------------------------------------------------ arguments
if ($Source.Count -eq 0 -and -not $ListFile) { Refuse "give -Source <folder>[,<folder>...] and/or -ListFile <file>" 2 }
if ($TimeoutSec -lt 10) { Refuse "-TimeoutSec must be >= 10" 2 }
if ($RestartEvery -lt 1) { Refuse "-RestartEvery must be >= 1" 2 }
$Output = [IO.Path]::GetFullPath($Output)
if (-not $LogDir) { $LogDir = Join-Path $Output "_logs" }
$LogDir = [IO.Path]::GetFullPath($LogDir)
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $Output | Out-Null }
$RunId = (Get-Date).ToString("yyyyMMdd-HHmmss")
$LogFile = Join-Path $LogDir "run-$RunId.log"
function Log([string] $msg) {
    $line = "$(UtcNow) $msg"
    Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8
    Write-Host $line
}

# ------------------------------------------------------------------ enumerate inputs
function IsSealed([string] $p) { return $p.ToUpperInvariant().Contains("_SEALED") }
function SafeLeaf([string] $p) {
    $leaf = Split-Path -Leaf ($p.TrimEnd('\', '/'))
    if (-not $leaf) { $leaf = "root" }
    return ($leaf -replace '[:\\/]', '_')
}

$items = New-Object System.Collections.ArrayList   # @{Path; Rel}
$sealedSkipped = 0
$roots = @()
foreach ($s in $Source) {
    if (-not (Test-Path -LiteralPath $s -PathType Container)) { Refuse "source folder not found: $s" 2 }
    $roots += [IO.Path]::GetFullPath($s).TrimEnd('\')
}
$listed = @()
if ($ListFile) {
    if (-not (Test-Path -LiteralPath $ListFile -PathType Leaf)) { Refuse "list file not found: $ListFile" 2 }
    foreach ($line in (Get-Content -LiteralPath $ListFile -Encoding UTF8)) {
        $t = $line.Trim().Trim('"')
        if (-not $t -or $t.StartsWith("#")) { continue }
        if (Test-Path -LiteralPath $t -PathType Container) { $roots += [IO.Path]::GetFullPath($t).TrimEnd('\') }
        elseif (Test-Path -LiteralPath $t -PathType Leaf) { $listed += [IO.Path]::GetFullPath($t) }
        else { Refuse "list file entry not found: $t" 2 }
    }
}
$multiRoot = (($roots.Count + [int]($listed.Count -gt 0)) -gt 1)
$usedLeaves = @{}
function RootPrefix([string] $root) {
    if (-not $multiRoot) { return "" }
    $leaf = SafeLeaf $root
    $key = $leaf.ToUpperInvariant()
    if ($usedLeaves.ContainsKey($key) -and $usedLeaves[$key] -ne $root) {
        $i = 2
        while ($usedLeaves.ContainsKey("$key`_$i")) { $i++ }
        $key = "$key`_$i"; $leaf = "$leaf`_$i"
    }
    $usedLeaves[$key] = $root
    return "$leaf\"
}
$exts = @(".sldprt", ".sldasm", ".slddrw")
foreach ($root in $roots) {
    $prefix = RootPrefix $root
    foreach ($f in (Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue)) {
        if ($exts -notcontains $f.Extension.ToLowerInvariant()) { continue }
        if (IsSealed $f.FullName) { $sealedSkipped++; continue }
        if ($f.Name.StartsWith("~$")) { continue }   # SolidWorks lock/temp files
        $rel = $prefix + $f.FullName.Substring($root.Length).TrimStart('\')
        [void]$items.Add(@{ Path = $f.FullName; Rel = $rel })
    }
}
foreach ($p in $listed) {
    if (IsSealed $p) { $sealedSkipped++; continue }
    if ($exts -notcontains ([IO.Path]::GetExtension($p).ToLowerInvariant())) { continue }
    $rel = (SafeLeaf (Split-Path -Parent $p)) + "\" + (Split-Path -Leaf $p)
    if ($multiRoot) { $rel = "_list\" + $rel }
    [void]$items.Add(@{ Path = $p; Rel = $rel })
}
Log "$ScriptVersion run ${RunId}: $($items.Count) candidate files, $sealedSkipped sealed paths skipped (not listed)"

# ------------------------------------------------------------------ previous manifest (up-to-date skip)
$manifestJson = Join-Path $Output "manifest.json"
$previous = @{}
if (-not $Force -and (Test-Path -LiteralPath $manifestJson)) {
    try {
        $prev = Get-Content -LiteralPath $manifestJson -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($r in $prev.files) {
            if ($r.status -eq "OK" -or $r.status -eq "SKIPPED_UP_TO_DATE") {
                $previous["$($r.source_rel)|$($r.configuration)"] = $r
            }
        }
    } catch { Log "WARNING previous manifest unreadable, nothing is skipped: $($_.Exception.Message)" }
}

function StepRelFor([string] $rel, [string] $config, [bool] $perConfig) {
    $dir = Split-Path -Parent $rel
    $stem = [IO.Path]::GetFileNameWithoutExtension($rel)
    if ($perConfig) {
        $safe = ($config -replace '[<>:"/\\|?*\x00-\x1f]', '_').Trim().TrimEnd('.')
        $stem = "$stem`__$safe"
    }
    if ($dir) { return (Join-Path $dir "$stem.step") }
    return "$stem.step"
}

function NewRow($item) {
    return [ordered]@{
        source_path = $item.Path; source_rel = $item.Rel; source_name = (Split-Path -Leaf $item.Path)
        source_size = $null; source_sha256_before = $null; source_sha256_after = $null
        configuration = $null; output_rel = $null; output_path = $null
        step_sha256 = $null; step_size = $null; status = $null; error = ""; warnings = ""
        sw_revision = $null; doc_length_unit = $null
        started_utc = (UtcNow); finished_utc = $null; duration_s = $null; attempts = 0
    }
}
function Finish($row, [datetime] $t0) {
    $row.finished_utc = UtcNow
    $row.duration_s = [math]::Round(((Get-Date) - $t0).TotalSeconds, 3)
    return $row
}

$rows = New-Object System.Collections.ArrayList

# ------------------------------------------------------------------ dry run: no SolidWorks
if ($DryRun) {
    foreach ($it in $items) {
        $t0 = Get-Date; $row = NewRow $it
        $ext = [IO.Path]::GetExtension($it.Path).ToLowerInvariant()
        $fi = Get-Item -LiteralPath $it.Path
        $row.source_size = $fi.Length
        if ($ext -ne ".sldprt") { $row.status = "UNSUPPORTED_TYPE"; $row.error = "$ext is not exported (parts only)" }
        else {
            $row.source_sha256_before = Sha $it.Path
            $row.configuration = if ($AllConfigurations) { "*" } else { "(active)" }
            $prevRow = $null
            foreach ($k in $previous.Keys) { if ($k.StartsWith("$($it.Rel)|")) { $prevRow = $previous[$k]; break } }
            if ($prevRow -and $prevRow.source_sha256_before -eq $row.source_sha256_before -and $prevRow.output_path -and
                (Test-Path -LiteralPath $prevRow.output_path)) { $row.status = "SKIPPED_UP_TO_DATE" }
            else { $row.status = "PLANNED" }
            $row.output_rel = StepRelFor $it.Rel "" $false
            $row.output_path = Join-Path $Output $row.output_rel
        }
        [void]$rows.Add((Finish $row $t0))
    }
    $doc = [ordered]@{ tool = $ScriptVersion; run_id = $RunId; dry_run = $true; output = $Output
                       sealed_paths_skipped = $sealedSkipped; files = $rows }
    $doc | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $LogDir "manifest.dryrun.json") -Encoding UTF8
    $rows | ForEach-Object { New-Object PSObject -Property $_ } |
        Export-Csv -LiteralPath (Join-Path $LogDir "manifest.dryrun.csv") -NoTypeInformation -Encoding UTF8
    $planned = @($rows | Where-Object { $_.status -eq "PLANNED" }).Count
    Log "DRY RUN: $planned to export, $(@($rows | Where-Object { $_.status -eq 'SKIPPED_UP_TO_DATE' }).Count) up to date, $(@($rows | Where-Object { $_.status -eq 'UNSUPPORTED_TYPE' }).Count) unsupported; manifest in $LogDir"
    exit 0
}

# ------------------------------------------------------------------ SolidWorks constants (from its own interop DLL)
function FindSwExe {
    $cands = @()
    $cands += Get-ChildItem -Path "C:\Program Files\SOLIDWORKS Corp*\SOLIDWORKS*\SLDWORKS.exe" -ErrorAction SilentlyContinue
    $cands += Get-ChildItem -Path "C:\Program Files\SolidWorks*\SLDWORKS.exe" -ErrorAction SilentlyContinue
    return @($cands | Select-Object -ExpandProperty FullName)
}
if (-not $SwConstDll) {
    foreach ($exe in (FindSwExe)) {
        $c = Join-Path (Split-Path -Parent $exe) "api\redist\SolidWorks.Interop.swconst.dll"
        if (Test-Path -LiteralPath $c) { $SwConstDll = $c; break }
    }
}
if (-not $SwConstDll -or -not (Test-Path -LiteralPath $SwConstDll)) {
    Refuse "SolidWorks.Interop.swconst.dll not found (pass -SwConstDll '<SolidWorks dir>\api\redist\SolidWorks.Interop.swconst.dll')"
}
Add-Type -LiteralPath $SwConstDll
function E([string] $enum, [string] $name) {
    $t = "SolidWorks.Interop.swconst.$enum" -as [type]
    if (-not $t) { return $null }
    if ([Enum]::GetNames($t) -notcontains $name) { return $null }
    return [int][Enum]::Parse($t, $name)
}
function Need([string] $enum, [string] $name) {
    $v = E $enum $name
    if ($null -eq $v) { Refuse "constant $enum.$name missing in $SwConstDll" }
    return $v
}
$swDocPART      = Need "swDocumentTypes_e" "swDocPART"
$optSilent      = Need "swOpenDocOptions_e" "swOpenDocOptions_Silent"
$optReadOnly    = Need "swOpenDocOptions_e" "swOpenDocOptions_ReadOnly"
$saveSilent     = Need "swSaveAsOptions_e" "swSaveAsOptions_Silent"
$saveCopy       = Need "swSaveAsOptions_e" "swSaveAsOptions_Copy"
$saveCurrent    = Need "swSaveAsVersion_e" "swSaveAsCurrentVersion"
$prefStepAP     = Need "swUserPreferenceIntegerValue_e" "swStepAP"
$prefUnitSystem = E "swUserPreferenceIntegerValue_e" "swUnitSystem"
$unitMMGS       = E "swUnitSystem_e" "swUnitSystem_MMGS"
$lenMM          = E "swLengthUnit_e" "swMM"
$prefOptDefault = E "swUserPreferenceOption_e" "swDetailingNoOptionSpecified"
if ($null -eq $prefOptDefault) { $prefOptDefault = 0 }

function FlagNames([string] $enum, [int] $value) {
    if ($value -eq 0) { return "" }
    $t = "SolidWorks.Interop.swconst.$enum" -as [type]
    if (-not $t) { return "$value" }
    $names = @()
    foreach ($n in [Enum]::GetNames($t)) {
        $v = [int][Enum]::Parse($t, $n)
        if ($v -ne 0 -and (($value -band $v) -eq $v)) { $names += $n }
    }
    if ($names.Count -eq 0) { return "$value" }
    return ($names -join "|") + " ($value)"
}

# ------------------------------------------------------------------ SolidWorks process + watchdog
$script:sw = $null
$script:swPid = $null
$script:swRevision = $null
$deadlineFile = Join-Path $LogDir "watchdog.deadline"
$killedFile = Join-Path $LogDir "watchdog.killed"

function SwPids { @(Get-Process -Name SLDWORKS -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id) }

$running = SwPids
if ($running.Count -gt 0 -and -not $AttachRunning) {
    Refuse "SolidWorks is already running (pid $($running -join ',')). Close it, or pass -AttachRunning (a hang then kills that session)."
}

function StartSw {
    $before = SwPids
    if ($AttachRunning -and $before.Count -gt 0) {
        $script:sw = [Runtime.InteropServices.Marshal]::GetActiveObject("SldWorks.Application")
        $script:swPid = $before[0]
    } else {
        $script:sw = New-Object -ComObject SldWorks.Application
        $deadline = (Get-Date).AddSeconds(120)
        do {
            $new = @(SwPids | Where-Object { $before -notcontains $_ })
            if ($new.Count -gt 0) { $script:swPid = $new[0]; break }
            Start-Sleep -Milliseconds 500
        } while ((Get-Date) -lt $deadline)
        if (-not $script:swPid) { $p = SwPids; if ($p.Count -gt 0) { $script:swPid = $p[0] } }
    }
    $script:sw.Visible = $false
    try { $script:sw.UserControl = $false } catch { }
    try { $script:sw.CommandInProgress = $true } catch { }
    $script:swRevision = [string]$script:sw.RevisionNumber()
    Log "SolidWorks started/attached: pid $($script:swPid), revision $($script:swRevision)"
}
function ReleaseSw {
    if ($script:sw) {
        try { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($script:sw) } catch { }
    }
    $script:sw = $null
    [GC]::Collect(); [GC]::WaitForPendingFinalizers()
}
function StopSw([bool] $kill) {
    if ($script:sw -and -not $kill) {
        try { $script:sw.CommandInProgress = $false } catch { }
        try { [void]$script:sw.CloseAllDocuments($true) } catch { }
        try { $script:sw.ExitApp() } catch { }
    }
    ReleaseSw
    if ($script:swPid) {
        try { Wait-Process -Id $script:swPid -Timeout 60 -ErrorAction Stop } catch {
            Stop-Process -Id $script:swPid -Force -ErrorAction SilentlyContinue
        }
    }
    $script:swPid = $null
}

$watchdog = Start-Job -ArgumentList $deadlineFile, $killedFile -ScriptBlock {
    param($deadlineFile, $killedFile)
    while ($true) {
        Start-Sleep -Seconds 2
        if (-not (Test-Path -LiteralPath $deadlineFile)) { continue }
        try { $parts = (Get-Content -LiteralPath $deadlineFile -Raw).Trim().Split("|") } catch { continue }
        if ($parts.Count -lt 3) { continue }
        if ([DateTime]::UtcNow.Ticks -gt [long]$parts[1]) {
            Stop-Process -Id ([int]$parts[0]) -Force -ErrorAction SilentlyContinue
            Set-Content -LiteralPath $killedFile -Value $parts[2]
            Remove-Item -LiteralPath $deadlineFile -Force -ErrorAction SilentlyContinue
        }
    }
}
function ArmWatchdog([string] $token) {
    Remove-Item -LiteralPath $killedFile -Force -ErrorAction SilentlyContinue
    $ticks = [DateTime]::UtcNow.AddSeconds($TimeoutSec).Ticks
    Set-Content -LiteralPath $deadlineFile -Value "$($script:swPid)|$ticks|$token"
}
function DisarmWatchdog { Remove-Item -LiteralPath $deadlineFile -Force -ErrorAction SilentlyContinue }
function WasKilled([string] $token) {
    if (-not (Test-Path -LiteralPath $killedFile)) { return $false }
    return ((Get-Content -LiteralPath $killedFile -Raw).Trim() -eq $token)
}

# ------------------------------------------------------------------ STEP preference (AP214), restorable
$restoreFile = Join-Path $LogDir "steppref.restore.json"
function RestoreStepPref([int] $value) {
    if ($script:sw) { [void]$script:sw.SetUserPreferenceIntegerValue($prefStepAP, $value) }
}

# ------------------------------------------------------------------ one file
function ExportOne($item, [string] $token) {
    # Returns a list of rows (one per configuration). Throws only when SolidWorks died (timeout/crash).
    $out = New-Object System.Collections.ArrayList
    $t0 = Get-Date
    $base = NewRow $item
    $base.sw_revision = $script:swRevision
    $base.source_size = (Get-Item -LiteralPath $item.Path).Length
    $base.source_sha256_before = Sha $item.Path
    $errs = 0; $warns = 0
    $doc = $null
    $doc = $script:sw.OpenDoc6($item.Path, $swDocPART, ($optSilent -bor $optReadOnly), "", [ref]$errs, [ref]$warns)
    if ($null -eq $doc) {
        $base.status = "FAILED_OPEN"
        $base.error = "OpenDoc6 returned nothing; errors: " + (FlagNames "swFileLoadError_e" $errs)
        $base.warnings = FlagNames "swFileLoadWarning_e" $warns
        [void]$script:sw.CloseAllDocuments($true)
        $base.source_sha256_after = Sha $item.Path
        [void]$out.Add((Finish $base $t0)); return ,$out
    }
    $openWarn = FlagNames "swFileLoadWarning_e" $warns
    try {
        $configs = @()
        if ($AllConfigurations) { $configs = @($doc.GetConfigurationNames()) }
        else { $configs = @([string]$doc.ConfigurationManager.ActiveConfiguration.Name) }
        try { $base.doc_length_unit = [int]$doc.LengthUnit } catch { }
        foreach ($cfg in $configs) {
            $row = [ordered]@{}; foreach ($k in $base.Keys) { $row[$k] = $base[$k] }
            $row.configuration = [string]$cfg
            $row.output_rel = StepRelFor $item.Rel ([string]$cfg) ([bool]$AllConfigurations)
            $row.output_path = Join-Path $Output $row.output_rel
            $w = @(); if ($openWarn) { $w += "open: $openWarn" }
            $tmp = $null
            try {
                if ($AllConfigurations) {
                    if (-not $doc.ShowConfiguration2([string]$cfg)) { throw "cannot activate configuration '$cfg'" }
                }
                [void]$doc.ForceRebuild3($false)
                try {
                    $ww = [int]$doc.Extension.GetWhatsWrongCount()
                    if ($ww -gt 0) { $w += "rebuild: $ww what's-wrong item(s)" }
                } catch { }
                # mm in memory only: the document is read-only and never saved.
                if ($null -ne $prefUnitSystem -and $null -ne $unitMMGS -and $null -ne $lenMM -and
                    $null -ne $row.doc_length_unit -and $row.doc_length_unit -ne $lenMM) {
                    [void]$doc.Extension.SetUserPreferenceInteger($prefUnitSystem, $prefOptDefault, $unitMMGS)
                    $w += "length unit $($row.doc_length_unit) set to mm for export"
                }
                $dir = Split-Path -Parent $row.output_path
                New-Item -ItemType Directory -Force -Path $dir | Out-Null
                $tmp = Join-Path $dir (([IO.Path]::GetFileNameWithoutExtension($row.output_path)) + ".~tmp.step")
                Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
                $se = 0; $sw2 = 0
                $ok = $doc.Extension.SaveAs($tmp, $saveCurrent, ($saveSilent -bor $saveCopy), $null, [ref]$se, [ref]$sw2)
                if (-not $ok -or -not (Test-Path -LiteralPath $tmp) -or (Get-Item -LiteralPath $tmp).Length -eq 0) {
                    throw ("SaveAs STEP failed; errors: " + (FlagNames "swFileSaveError_e" $se) +
                           "; warnings: " + (FlagNames "swFileSaveWarning_e" $sw2))
                }
                if ($sw2 -ne 0) { $w += "save: " + (FlagNames "swFileSaveWarning_e" $sw2) }
                if (Test-Path -LiteralPath $row.output_path) { [IO.File]::Replace($tmp, $row.output_path, $null) }
                else { [IO.File]::Move($tmp, $row.output_path) }
                $row.step_sha256 = Sha $row.output_path
                $row.step_size = (Get-Item -LiteralPath $row.output_path).Length
                $row.status = "OK"
            } catch {
                if (-not $script:sw -or (WasKilled $token)) { throw }
                if ($_.Exception -is [Runtime.InteropServices.COMException] -and
                    ($_.Exception.HResult -eq -2147023174 -or $_.Exception.HResult -eq -2147417848)) { throw }  # RPC server gone
                $row.status = "FAILED_EXPORT"; $row.error = $_.Exception.Message
                if ($tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
            }
            $row.warnings = ($w -join "; ")
            [void]$out.Add($row)
        }
    } finally {
        try { [void]$script:sw.CloseAllDocuments($true) } catch { }
        if ($doc) { try { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc) } catch { } }
    }
    $after = Sha $item.Path
    foreach ($r in $out) {
        $r.source_sha256_after = $after
        if ($after -ne $r.source_sha256_before) {
            $r.status = "SOURCE_CHANGED"; $r.error = "source sha256 changed while open (never saved by this tool)"
        }
        [void](Finish $r $t0)
    }
    return ,$out
}

# ------------------------------------------------------------------ main loop
$originalStepAP = $null
$exitCode = 0
try {
    StartSw
    if (Test-Path -LiteralPath $restoreFile) {
        $old = Get-Content -LiteralPath $restoreFile -Raw | ConvertFrom-Json
        RestoreStepPref ([int]$old.swStepAP)
        Log "restored swStepAP=$($old.swStepAP) left by an interrupted run"
    }
    $originalStepAP = [int]$script:sw.GetUserPreferenceIntegerValue($prefStepAP)
    @{ swStepAP = $originalStepAP; run_id = $RunId } | ConvertTo-Json | Set-Content -LiteralPath $restoreFile -Encoding UTF8
    [void]$script:sw.SetUserPreferenceIntegerValue($prefStepAP, 214)
    Log "swStepAP $originalStepAP -> 214 (restored at the end)"

    $sinceRestart = 0
    $n = 0
    foreach ($it in $items) {
        $n++
        $ext = [IO.Path]::GetExtension($it.Path).ToLowerInvariant()
        if ($ext -ne ".sldprt") {
            $t0 = Get-Date; $row = NewRow $it
            $row.source_size = (Get-Item -LiteralPath $it.Path).Length
            $row.status = "UNSUPPORTED_TYPE"; $row.error = "$ext is not exported (parts only)"
            [void]$rows.Add((Finish $row $t0)); Log "[$n/$($items.Count)] UNSUPPORTED_TYPE $($it.Rel)"; continue
        }
        # up to date?
        if (-not $Force) {
            $prevRows = @($previous.Keys | Where-Object { $_.StartsWith("$($it.Rel)|") } | ForEach-Object { $previous[$_] })
            if ($prevRows.Count -gt 0) {
                $srcSha = Sha $it.Path
                $fresh = $true
                foreach ($p in $prevRows) {
                    if ($p.source_sha256_before -ne $srcSha -or -not $p.output_path -or
                        -not (Test-Path -LiteralPath $p.output_path) -or (Sha $p.output_path) -ne $p.step_sha256) { $fresh = $false }
                }
                if ($AllConfigurations -ne ($prevRows[0].output_rel -like "*__*")) { $fresh = $false }
                if ($fresh) {
                    foreach ($p in $prevRows) {
                        $row = [ordered]@{}; foreach ($pp in $p.PSObject.Properties) { $row[$pp.Name] = $pp.Value }
                        $row.status = "SKIPPED_UP_TO_DATE"; $row.error = ""
                        $row.source_sha256_after = $srcSha
                        $row.started_utc = UtcNow; $row.finished_utc = UtcNow; $row.duration_s = 0
                        [void]$rows.Add($row)
                    }
                    Log "[$n/$($items.Count)] SKIPPED_UP_TO_DATE $($it.Rel)"; continue
                }
            }
        }
        if ($sinceRestart -ge $RestartEvery -and -not $AttachRunning) {
            Log "restarting SolidWorks after $sinceRestart files (memory cap)"
            StopSw $false; StartSw; [void]$script:sw.SetUserPreferenceIntegerValue($prefStepAP, 214); $sinceRestart = 0
        }
        $result = $null
        for ($attempt = 1; $attempt -le 2; $attempt++) {
            $token = "$RunId-$n-$attempt"
            ArmWatchdog $token
            try {
                $result = ExportOne $it $token
                DisarmWatchdog
                foreach ($r in $result) { $r.attempts = $attempt }
                break
            } catch {
                DisarmWatchdog
                $killed = WasKilled $token
                Log "[$n/$($items.Count)] attempt ${attempt}: SolidWorks $(if ($killed) { 'hung (killed after ' + $TimeoutSec + ' s)' } else { 'failed' }): $($_.Exception.Message)"
                StopSw $true
                Get-Process -Name SLDWORKS -ErrorAction SilentlyContinue | Where-Object { $_.Id -eq $script:swPid } | Stop-Process -Force
                StartSw; [void]$script:sw.SetUserPreferenceIntegerValue($prefStepAP, 214); $sinceRestart = 0
                if ($attempt -eq 2) {
                    $row = NewRow $it
                    $row.sw_revision = $script:swRevision
                    $row.source_size = (Get-Item -LiteralPath $it.Path).Length
                    $row.source_sha256_after = Sha $it.Path
                    $row.status = if ($killed) { "TIMEOUT" } else { "FAILED_EXPORT" }
                    $row.error = "SolidWorks " + $(if ($killed) { "exceeded $TimeoutSec s" } else { "crashed" }) +
                                 " twice; last: $($_.Exception.Message)"
                    $row.attempts = 2
                    $result = @((Finish $row (Get-Date)))
                }
            }
        }
        $sinceRestart++
        foreach ($r in $result) {
            [void]$rows.Add($r)
            Log "[$n/$($items.Count)] $($r.status) $($it.Rel) [$($r.configuration)] $($r.error)"
        }
    }
} finally {
    DisarmWatchdog
    if ($watchdog) { Stop-Job $watchdog -ErrorAction SilentlyContinue; Remove-Job $watchdog -Force -ErrorAction SilentlyContinue }
    if ($null -ne $originalStepAP) {
        try {
            if (-not $script:sw) { StartSw }
            RestoreStepPref $originalStepAP
            Remove-Item -LiteralPath $restoreFile -Force -ErrorAction SilentlyContinue
            Log "swStepAP restored to $originalStepAP"
        } catch { Log "WARNING could not restore swStepAP now; it is restored at the next start ($restoreFile)" }
    }
    if (-not $AttachRunning) { StopSw $false } else { ReleaseSw }

    # manifest (always, also after an interruption)
    $doc = [ordered]@{
        tool = $ScriptVersion; run_id = $RunId; dry_run = $false; output = $Output
        step_schema = "AP214"; units = "mm"; all_configurations = [bool]$AllConfigurations
        sw_revision = $script:swRevision; sealed_paths_skipped = $sealedSkipped
        created_utc = (UtcNow); files = $rows
    }
    $tmpJson = "$manifestJson.tmp"
    $doc | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $tmpJson -Encoding UTF8
    if (Test-Path -LiteralPath $manifestJson) { [IO.File]::Replace($tmpJson, $manifestJson, $null) } else { [IO.File]::Move($tmpJson, $manifestJson) }
    $csv = Join-Path $Output "manifest.csv"
    $rows | ForEach-Object { New-Object PSObject -Property $_ } | Export-Csv -LiteralPath "$csv.tmp" -NoTypeInformation -Encoding UTF8
    if (Test-Path -LiteralPath $csv) { [IO.File]::Replace("$csv.tmp", $csv, $null) } else { [IO.File]::Move("$csv.tmp", $csv) }
    Copy-Item -LiteralPath $manifestJson -Destination (Join-Path $LogDir "manifest-$RunId.json") -Force
}

$counts = $rows | Group-Object { $_.status } | ForEach-Object { "$($_.Name)=$($_.Count)" }
Log "done: $($counts -join ', '); sealed skipped=$sealedSkipped; manifest $manifestJson"
if (@($rows | Where-Object { $FailStatuses -contains $_.status }).Count -gt 0) { $exitCode = 1 }
exit $exitCode
