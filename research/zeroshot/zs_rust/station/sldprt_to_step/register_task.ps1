<#
.SYNOPSIS
  Optional: register sldprt_to_step.ps1 as a Windows Task Scheduler task (daily, or once).

.DESCRIPTION
  SolidWorks needs an interactive desktop session, so the task runs only while -User is logged on
  (LogonType Interactive). Uses the stock ScheduledTasks module; nothing is installed. Re-running replaces
  the task of the same name. Remove with: Unregister-ScheduledTask -TaskName <name> -Confirm:$false

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File register_task.ps1 -Source D:\CAD\Parts -Output D:\CAD\STEP_out -At 02:00
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string[]] $Source,
    [Parameter(Mandatory = $true)] [string] $Output,
    [string] $At = "02:00",
    [switch] $Once,
    [string] $TaskName = "ShermanQC SLDPRT to STEP",
    [string] $User = "$env:USERDOMAIN\$env:USERNAME",
    [string] $ExtraArgs = ""
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$script = Join-Path $PSScriptRoot "sldprt_to_step.ps1"
if (-not (Test-Path -LiteralPath $script)) { throw "sldprt_to_step.ps1 not found next to this file" }
$src = ($Source | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ","
$cmd = "-NoProfile -ExecutionPolicy Bypass -Command `"& '$script' -Source $src -Output '$Output' $ExtraArgs; exit `$LASTEXITCODE`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $cmd -WorkingDirectory $PSScriptRoot
$time = [datetime]::ParseExact($At, "HH:mm", $null)
$trigger = if ($Once) { New-ScheduledTaskTrigger -Once -At $time } else { New-ScheduledTaskTrigger -Daily -At $time }
$principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew `
    -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
    -Description "Export SLDPRT to STEP AP214 (read-only) with manifest; see README.md" -Force | Out-Null
Write-Host "registered '$TaskName' for $User at $At ($(if ($Once) { 'once' } else { 'daily' })): powershell.exe $cmd"
