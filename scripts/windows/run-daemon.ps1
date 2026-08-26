[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Debug",
    [string]$DaemonPath = "",
    [string]$DataDirectory = "",
    [string]$BindAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 3030,
    [string]$RigId = "windows-dev-rig"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function ConvertTo-TomlString {
    param([Parameter(Mandatory = $true)][string]$Value)
    return $Value.Replace("\", "\\").Replace('"', '\"')
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "This runner is for the Windows rig daemon."
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$daemonRoot = Join-Path $repositoryRoot "achernar-daemon"
$profile = if ($Configuration -eq "Release") { "release" } else { "debug" }

if ([string]::IsNullOrWhiteSpace($DaemonPath)) {
    $DaemonPath = Join-Path $daemonRoot "target\x86_64-pc-windows-msvc\$profile\achernar-daemon.exe"
}
$DaemonPath = (Resolve-Path -LiteralPath $DaemonPath).Path

if ([string]::IsNullOrWhiteSpace($DataDirectory)) {
    $DataDirectory = Join-Path $env:LOCALAPPDATA "Achernar\dev"
}
$DataDirectory = [IO.Path]::GetFullPath($DataDirectory)
$framesDirectory = Join-Path $DataDirectory "frames"
New-Item -ItemType Directory -Force -Path $DataDirectory, $framesDirectory | Out-Null

$configPath = Join-Path $DataDirectory "achernar.windows.toml"
$databasePath = Join-Path $DataDirectory "achernar.db"
$config = @"
[daemon]
port = $Port
log_level = "info"

[storage]
database_path = "$(ConvertTo-TomlString $databasePath)"
frames_dir = "$(ConvertTo-TomlString $framesDirectory)"

[telemetry]
level = "info"
json_output = false
"@
$utf8WithoutBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText($configPath, $config, $utf8WithoutBom)

$env:ACHERNAR_CONFIG = $configPath
$env:ACHERNAR_HOST = $BindAddress
$env:ACHERNAR_PORT = $Port.ToString()
$env:ACHERNAR_RIG_ID = $RigId

Write-Host "Starting Achernar rig daemon at http://${BindAddress}:$Port"
Write-Host "Data directory: $DataDirectory"
Write-Host "This process is the local hardware agent only; achernar.com remains the hosted UI."
& $DaemonPath
exit $LASTEXITCODE
