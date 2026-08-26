[CmdletBinding()]
param(
    [string]$VcpkgRoot = "",
    [ValidateSet("x64-windows")]
    [string]$Triplet = "x64-windows",
    [switch]$InstallMissing,
    [switch]$SkipDependencies
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Test-IsWindows {
    return [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
}

function Test-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Find-VisualStudio {
    $programFilesX86 = [Environment]::GetFolderPath("ProgramFilesX86")
    if ([string]::IsNullOrWhiteSpace($programFilesX86)) {
        return $null
    }

    $vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        return $null
    }

    $installation = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($installation)) {
        return $null
    }
    return $installation.Trim()
}

function Install-WingetPackage {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [string]$Override = ""
    )

    $arguments = @(
        "install", "--id", $Id, "--exact", "--source", "winget",
        "--accept-package-agreements", "--accept-source-agreements"
    )
    if (-not [string]::IsNullOrWhiteSpace($Override)) {
        $arguments += @("--override", $Override)
    }
    & winget @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "winget could not install $Id (exit $LASTEXITCODE)."
    }
}

function Resolve-VcpkgRoot {
    param(
        [string]$RequestedRoot,
        [string]$RepositoryRoot,
        [bool]$MayInstall
    )

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        $candidates += $RequestedRoot
    }
    if (-not [string]::IsNullOrWhiteSpace($env:VCPKG_ROOT)) {
        $candidates += $env:VCPKG_ROOT
    }
    if (-not [string]::IsNullOrWhiteSpace($env:VCPKG_INSTALLATION_ROOT)) {
        $candidates += $env:VCPKG_INSTALLATION_ROOT
    }
    $candidates += "C:\vcpkg"
    $candidates += (Join-Path $RepositoryRoot ".tools\vcpkg")

    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        $executable = Join-Path $candidate "vcpkg.exe"
        if (Test-Path -LiteralPath $executable -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    if (-not $MayInstall) {
        throw @"
vcpkg was not found. Set VCPKG_ROOT or run this script with -InstallMissing.
The repository-local default is .tools\vcpkg.
"@
    }

    if (-not (Test-Command "git")) {
        throw "Git is required to install vcpkg. Install Git, reopen PowerShell, and retry."
    }

    $destination = if ([string]::IsNullOrWhiteSpace($RequestedRoot)) {
        Join-Path $RepositoryRoot ".tools\vcpkg"
    } else {
        [IO.Path]::GetFullPath($RequestedRoot)
    }

    if (Test-Path -LiteralPath $destination) {
        throw "The vcpkg destination already exists but has no vcpkg.exe: $destination"
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    # The manifest pins a builtin registry commit. A shallow checkout at only
    # today's HEAD may not contain that commit after the registry advances.
    & git clone https://github.com/microsoft/vcpkg.git $destination
    if ($LASTEXITCODE -ne 0) {
        throw "Could not clone vcpkg (exit $LASTEXITCODE)."
    }
    & (Join-Path $destination "bootstrap-vcpkg.bat") -disableMetrics
    if ($LASTEXITCODE -ne 0) {
        throw "Could not bootstrap vcpkg (exit $LASTEXITCODE)."
    }
    return (Resolve-Path -LiteralPath $destination).Path
}

if (-not (Test-IsWindows)) {
    throw "This bootstrap is for a Windows x64 rig computer."
}
if (-not [Environment]::Is64BitOperatingSystem) {
    throw "Achernar's Windows test target requires 64-bit Windows."
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$daemonRoot = Join-Path $repositoryRoot "achernar-daemon"
$manifest = Join-Path $daemonRoot "vcpkg.json"
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "Missing vcpkg manifest: $manifest"
}

$missing = New-Object System.Collections.Generic.List[string]
if (-not (Test-Command "git")) { $missing.Add("Git") }
if (-not (Test-Command "rustup") -or -not (Test-Command "cargo")) { $missing.Add("Rust") }
if (-not (Test-Command "cmake")) { $missing.Add("CMake") }
if ($null -eq (Find-VisualStudio)) { $missing.Add("Visual Studio C++ Build Tools") }

if ($missing.Count -gt 0) {
    if (-not $InstallMissing) {
        throw "Missing prerequisites: $($missing -join ', '). Run again with -InstallMissing."
    }
    if (-not (Test-Command "winget")) {
        throw "winget is unavailable. Install the missing prerequisites manually: $($missing -join ', ')."
    }

    if ($missing.Contains("Git")) {
        Install-WingetPackage -Id "Git.Git"
    }
    if ($missing.Contains("Rust")) {
        Install-WingetPackage -Id "Rustlang.Rustup"
    }
    if ($missing.Contains("CMake")) {
        Install-WingetPackage -Id "Kitware.CMake"
    }
    if ($missing.Contains("Visual Studio C++ Build Tools")) {
        Install-WingetPackage `
            -Id "Microsoft.VisualStudio.2022.BuildTools" `
            -Override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    }

    Write-Host "Prerequisites were installed. Close this terminal, open a new 64-bit PowerShell, and run bootstrap.ps1 again."
    return
}

$cmakeVersionText = (& cmake --version | Select-Object -First 1)
if ($cmakeVersionText -notmatch "([0-9]+\.[0-9]+\.[0-9]+)") {
    throw "Could not determine the installed CMake version."
}
if ([version]$Matches[1] -lt [version]"3.21.0") {
    throw "CMake 3.21 or newer is required; found $($Matches[1])."
}

& rustup toolchain install stable --profile minimal
if ($LASTEXITCODE -ne 0) { throw "rustup could not install/update the stable toolchain." }
& rustup target add x86_64-pc-windows-msvc --toolchain stable
if ($LASTEXITCODE -ne 0) { throw "rustup could not install the Windows x64 MSVC target." }

$resolvedVcpkg = Resolve-VcpkgRoot `
    -RequestedRoot $VcpkgRoot `
    -RepositoryRoot $repositoryRoot `
    -MayInstall $InstallMissing.IsPresent

$env:VCPKG_ROOT = $resolvedVcpkg
$env:VCPKG_TARGET_TRIPLET = $Triplet
$env:VCPKG_DEFAULT_TRIPLET = $Triplet
$env:VCPKG_INSTALLED_DIR = Join-Path $daemonRoot "vcpkg_installed"

if (-not $SkipDependencies) {
    $vcpkg = Join-Path $resolvedVcpkg "vcpkg.exe"
    & $vcpkg install `
        "--triplet=$Triplet" `
        "--x-manifest-root=$daemonRoot" `
        "--x-install-root=$env:VCPKG_INSTALLED_DIR"
    if ($LASTEXITCODE -ne 0) {
        throw "vcpkg dependency restore failed (exit $LASTEXITCODE)."
    }
}

Write-Host "Windows daemon prerequisites are ready."
Write-Host "VCPKG_ROOT=$resolvedVcpkg"
Write-Host "VCPKG_INSTALLED_DIR=$env:VCPKG_INSTALLED_DIR"
Write-Host "Next: .\scripts\windows\build.ps1 -Configuration Debug"
