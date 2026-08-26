# Windows rig-agent and SV105 testing

This is a development test path for the telescope-side backend. It builds and
runs `achernar-daemon.exe`; it does not package the React site or Tauri shell.
The intended product boundary remains:

```text
achernar.com (hosted UI and cloud backend)
                 |
                 | commands / telemetry / reduced-rate previews
                 v
Windows rig computer: achernar-daemon.exe
                 |
                 v
SV105, mount, focuser, and filter wheel
```

The local API smoke test deliberately bypasses the hosted website. That lets us
prove hardware acquisition first, without confusing camera failures with cloud,
authentication, or frontend failures.

## What this currently validates

The automated Windows CI job is configured to build the x64 MSVC daemon and run
the complete Rust test suite plus a simulator API test. Until that new job has
its first green run, it is an intended gate rather than proof that the port
builds. GitHub's hosted runners also have no physical camera, so an SV105 must
still be accepted on a real Windows machine.

The hardware script checks:

1. daemon health and compiled-driver reporting;
2. UVC discovery and exact device-index selection;
3. connect and the frontend-ready camera descriptor;
4. advertised formats and capability-driven controls;
5. safe exposure/gain write-readback when those controls are available;
6. stream startup, REST JPEG snapshot, and binary WebSocket JPEG delivery;
7. a persisted verification capture with byte counts and SHA-256 provenance;
8. stream stop, restoration of original controls, and disconnect.

Unsupported controls are skipped rather than treated as failures. This is
intentional: the backend must expose what the camera and Windows driver truly
support, not a fictional universal control panel.

## Windows prerequisites

- Windows 10 or 11, x64 (Windows 11 is the reference environment)
- Visual Studio 2022 Build Tools with **Desktop development with C++**
- CMake 3.21 or newer
- Git
- Rust stable with the `x86_64-pc-windows-msvc` target
- vcpkg
- the normal Windows camera driver for the SV105

Do **not** install Zadig or replace the camera driver with WinUSB. The Windows
backend uses Media Foundation and should coexist with normal Windows astronomy
software. If SharpCap already opens the SV105, that is a useful baseline.

The checked-in [`vcpkg.json`](../achernar-daemon/vcpkg.json) restores the native
OpenCV and TIFF dependencies as dynamic x64 Windows libraries. Its registry
baseline is pinned so a later vcpkg port update cannot silently change what CI
and a newly bootstrapped rig build.

## Clone and bootstrap

Open a new 64-bit PowerShell window:

```powershell
git config --global core.longpaths true
git -c core.autocrlf=false clone https://github.com/ibrahimshahid1/Achernar.tech.git C:\src\Achernar
Set-Location C:\src\Achernar

Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\bootstrap.ps1 -InstallMissing
```

If the bootstrap installs prerequisites, close PowerShell, open a fresh window,
return to the repository, and run it again. This refreshes `PATH` and the Visual
Studio installation registry before dependency restore.

The initial OpenCV build through vcpkg can take a while. Subsequent builds reuse
`achernar-daemon\vcpkg_installed`.

## Build the daemon

```powershell
.\scripts\windows\build.ps1 -Configuration Debug -RunTests
```

The expected development executable is:

```text
achernar-daemon\target\x86_64-pc-windows-msvc\debug\achernar-daemon.exe
```

This is a developer build, not the eventual signed `Achernar Rig Agent`
installer. Native DLLs needed by the daemon are placed beside the executable by
the build.

## Establish the camera baseline

Before testing Achernar:

1. connect the SV105 directly to the Windows computer;
2. confirm `SVBONY SV105` appears under **Cameras** in Device Manager;
3. allow desktop applications under **Settings > Privacy & security > Camera**;
4. open the camera in SharpCap and verify live video;
5. close SharpCap and every other application using the camera.

Windows N editions may need Microsoft's Media Feature Pack before Media
Foundation camera capture is available.

## Run the isolated SV105 acceptance test

From the repository root:

```powershell
$daemon = Resolve-Path .\achernar-daemon\target\x86_64-pc-windows-msvc\debug\achernar-daemon.exe

.\scripts\windows\smoke-api.ps1 `
  -DaemonPath $daemon `
  -Driver uvc `
  -CameraNamePattern 'SV105'
```

The script launches its own daemon on `127.0.0.1:31337`, creates a new SQLite
database and frame archive under `%TEMP%`, and disables the cloud relay for that
child process. It therefore cannot overwrite the normal rig database or send
test commands through achernar.com. On success, temporary files are removed. On
failure, the exact artifact directory and the last daemon errors are printed.

To retain successful evidence:

```powershell
.\scripts\windows\smoke-api.ps1 `
  -DaemonPath $daemon `
  -Driver uvc `
  -CameraNamePattern 'SV105' `
  -KeepArtifacts
```

If two matching cameras are connected, first inspect discovery and then pass
the scan-local index explicitly:

```powershell
.\scripts\windows\smoke-api.ps1 `
  -DaemonPath $daemon `
  -Driver uvc `
  -CameraNamePattern 'SV105' `
  -DeviceIndex 1
```

The index is valid only for that scan. Unplugging/reconnecting devices can
change enumeration order.

## Run the daemon manually

For API inspection without the managed smoke test:

```powershell
.\scripts\windows\run-daemon.ps1 -Configuration Debug -Port 3030
```

This stores development state under `%LOCALAPPDATA%\Achernar\dev`. In another
PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:3030/health
Invoke-RestMethod http://127.0.0.1:3030/api/v1/drivers
Invoke-RestMethod -Method Post http://127.0.0.1:3030/api/v1/discover
```

You can run the acceptance checks against that process without allowing the
script to start or stop it:

```powershell
.\scripts\windows\smoke-api.ps1 `
  -BaseUri http://127.0.0.1:3030 `
  -Driver uvc `
  -CameraNamePattern 'SV105'
```

The script still disconnects the camera it connects. It does not terminate a
daemon it did not launch.

## Troubleshooting

`uvc` is reported as not compiled
: Confirm this is the new Windows build and inspect the CMake section of the
  build log. Re-run bootstrap and build from a fresh PowerShell.

No matching camera is discovered
: Close SharpCap, OBS, Teams, browser camera tabs, and any other camera client.
  Check Device Manager and Windows camera privacy settings. Replug the camera,
  then retry. Do not use Zadig.

The daemon cannot find a DLL
: Re-run `build.ps1` instead of copying only `achernar-daemon.exe`. Development
  builds need the bridge and vcpkg runtime DLLs placed next to the executable.

The test fails after connecting
: Keep the artifact directory and share `daemon.stderr.log`,
  `daemon.stdout.log`, the reported format/control descriptor, and the exact
  failure line. Avoid repeatedly reconnecting a wedged camera without first
  unplugging it.

The SV105 passes SharpCap but not Achernar
: That is useful evidence: Windows and the physical camera are sound, and the
  defect is isolated to our Media Foundation backend or bridge. Preserve the
  smoke artifacts for diagnosis.

## CI boundary

The workflow in `.github/workflows/windows-daemon.yml` is intended to become the
Windows build gate and simulator test once it has run successfully. It cannot
certify SV105 hardware. A green CI run plus a passing physical smoke run are
both required before calling the Windows camera path accepted.
