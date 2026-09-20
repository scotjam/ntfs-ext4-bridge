# install-agent.ps1 - installs the bridge two-way sync agent
#
# Run once, elevated, inside the Windows VM (or push via WinRM):
#   .\install-agent.ps1 -ControlUrl "http://192.168.122.1:10810" -Token "<token>"
#
# Copies bridge-agent.ps1 to C:\ProgramData\BridgeAgent, writes config.json,
# and registers a SYSTEM scheduled task that starts at boot and restarts on
# failure. No services, no compiled code.

param(
    [Parameter(Mandatory=$true)][string]$ControlUrl,
    [Parameter(Mandatory=$true)][string]$Token
)

$ErrorActionPreference = 'Stop'
$ConfigDir = 'C:\ProgramData\BridgeAgent'
$TaskName = 'BridgeAgent'

New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
Copy-Item (Join-Path $PSScriptRoot 'bridge-agent.ps1') `
    (Join-Path $ConfigDir 'bridge-agent.ps1') -Force

@{ control_url = $ControlUrl; token = $Token } |
    ConvertTo-Json | Set-Content (Join-Path $ConfigDir 'config.json')

# Remove any previous registration
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false `
    -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -File "{0}"' -f `
               (Join-Path $ConfigDir 'bridge-agent.ps1'))
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' `
    -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 3650) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $TaskName -Action $action `
    -Trigger $trigger -Principal $principal -Settings $settings | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "BridgeAgent installed and started (task: $TaskName)"
Write-Host "Log: $ConfigDir\agent.log"
