[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Debug",
    [string]$VcpkgRoot = "",
    [ValidateSet("x64-windows")]
    [string]$Triplet = "x64-windows",
    [switch]$SkipDependencies,
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-Vcpkg {
    param([string]$Requested, [string]$RepositoryRoot)

    $candidates = @($Requested, $env:VCPKG_ROOT, $env:VCPKG_INSTALLATION_ROOT, "C:\vcpkg", (Join-Path $RepositoryRoot ".tools\vcpkg"))
    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        if (Test-Path -LiteralPath (Join-Path $candidate "vcpkg.exe") -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "vcpkg was not found. Run .\scripts\windows\bootstrap.ps1 first."
}

function Import-VisualStudioEnvironment {
    $programFilesX86 = [Environment]::GetFolderPath("ProgramFilesX86")
    $vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        throw "vswhere.exe was not found. Install Visual Studio 2022 C++ Build Tools."
    }

    $installation = (& $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath).Trim()
    if ([string]::IsNullOrWhiteSpace($installation)) {
        throw "Visual Studio 2022 with the C++ x64 tools was not found."
    }

    $vsDevCmd = Join-Path $installation "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path -LiteralPath $vsDevCmd -PathType Leaf)) {
        throw "Visual Studio developer environment script is missing: $vsDevCmd"
    }

    $environmentLines = & cmd.exe /s /c "`"$vsDevCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not initialize the Visual Studio x64 build environment."
    }
    foreach ($line in $environmentLines) {
        if ($line -match "^([^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "This build script targets Windows x64/MSVC."
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$daemonRoot = Join-Path $repositoryRoot "achernar-daemon"
$target = "x86_64-pc-windows-msvc"
$resolvedVcpkg = Resolve-Vcpkg -Requested $VcpkgRoot -RepositoryRoot $repositoryRoot

Import-VisualStudioEnvironment
$env:VCPKG_ROOT = $resolvedVcpkg
$env:VCPKG_TARGET_TRIPLET = $Triplet
$env:VCPKG_DEFAULT_TRIPLET = $Triplet
$env:VCPKG_INSTALLED_DIR = Join-Path $daemonRoot "vcpkg_installed"

if (-not $SkipDependencies) {
    & (Join-Path $resolvedVcpkg "vcpkg.exe") install `
        "--triplet=$Triplet" `
        "--x-manifest-root=$daemonRoot" `
        "--x-install-root=$env:VCPKG_INSTALLED_DIR"
    if ($LASTEXITCODE -ne 0) { throw "vcpkg dependency restore failed." }
}

$cargoArguments = @("build", "--locked", "--target", $target)
if ($Configuration -eq "Release") { $cargoArguments += "--release" }

Push-Location $daemonRoot
try {
    & cargo @cargoArguments
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)." }

    if ($RunTests) {
        $testArguments = @("test", "--locked", "--all-targets", "--no-fail-fast", "--target", $target)
        if ($Configuration -eq "Release") { $testArguments += "--release" }
        & cargo @testArguments
        if ($LASTEXITCODE -ne 0) { throw "cargo test failed (exit $LASTEXITCODE)." }
    }
} finally {
    Pop-Location
}

$profile = if ($Configuration -eq "Release") { "release" } else { "debug" }
$daemon = Join-Path $daemonRoot "target\$target\$profile\achernar-daemon.exe"
if (-not (Test-Path -LiteralPath $daemon -PathType Leaf)) {
    throw "Cargo reported success but the daemon executable is missing: $daemon"
}

Write-Host "Windows daemon build ready: $daemon"
