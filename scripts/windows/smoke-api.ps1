[CmdletBinding()]
param(
    [string]$DaemonPath = "",
    [string]$BaseUri = "",
    [ValidateRange(1, 65535)]
    [int]$Port = 31337,
    [string]$Driver = "uvc",
    [int]$DeviceIndex = -1,
    [string]$CameraNamePattern = "SV105",
    [string]$DriverOption = "",
    [int]$FormatIndex = -1,
    [ValidateRange(1, 300)]
    [int]$StartupTimeoutSeconds = 45,
    [ValidateRange(1, 300)]
    [int]$FrameTimeoutSeconds = 30,
    [ValidateRange(0.001, 60.0)]
    [double]$CaptureExposureSeconds = 0.05,
    [string]$WorkDirectory = "",
    [switch]$SkipControlWrite,
    [switch]$SkipCapture,
    [switch]$KeepArtifacts
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function ConvertTo-TomlString {
    param([Parameter(Mandatory = $true)][string]$Value)
    return $Value.Replace("\", "\\").Replace('"', '\"')
}

function Write-JsonArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][object]$Value
    )

    $json = $Value | ConvertTo-Json -Depth 20
    [IO.File]::WriteAllText(
        (Join-Path $Directory $Name),
        $json,
        [Text.UTF8Encoding]::new($false)
    )
}

function Invoke-AchernarApi {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("Get", "Post")][string]$Method,
        [Parameter(Mandatory = $true)][string]$Uri,
        [object]$Body = $null,
        [int]$TimeoutSeconds = 75
    )

    $arguments = @{
        Method = $Method
        Uri = $Uri
        TimeoutSec = $TimeoutSeconds
    }
    if ($null -ne $Body) {
        $arguments.ContentType = "application/json"
        $arguments.Body = ($Body | ConvertTo-Json -Depth 12 -Compress)
    }

    try {
        return Invoke-RestMethod @arguments
    } catch {
        $detail = $_.Exception.Message
        if ($null -ne $_.ErrorDetails -and -not [string]::IsNullOrWhiteSpace($_.ErrorDetails.Message)) {
            $detail = $_.ErrorDetails.Message
        }
        throw "$Method $Uri failed: $detail"
    }
}

function Wait-ForHealth {
    param([string]$Uri, [int]$TimeoutSeconds, [Diagnostics.Process]$Process)

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        if ($null -ne $Process -and $Process.HasExited) {
            throw "The daemon exited before its health endpoint became ready (exit $($Process.ExitCode))."
        }
        try {
            $health = Invoke-RestMethod -Method Get -Uri $Uri -TimeoutSec 2
            if ($health -eq "OK") { return }
        } catch {
            Start-Sleep -Milliseconds 250
        }
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "The daemon health endpoint was not ready within $TimeoutSeconds seconds: $Uri"
}

function Get-StepAlignedTestValue {
    param([object]$Control)

    $minimum = [double]$Control.min
    $maximum = [double]$Control.max
    $current = [double]$Control.current
    if ($maximum -le $minimum) { return $current }

    $step = [Math]::Abs([double]$Control.step)
    if ($step -le 0.0) { $step = ($maximum - $minimum) / 20.0 }
    if ($step -le 0.0) { return $current }

    $candidate = $current + $step
    if ($candidate -gt $maximum) { $candidate = $current - $step }
    if ($candidate -lt $minimum) { $candidate = $minimum }
    $stepsFromMinimum = [Math]::Round(($candidate - $minimum) / $step)
    return [Math]::Min($maximum, [Math]::Max($minimum, $minimum + ($stepsFromMinimum * $step)))
}

function Receive-LiveFrame {
    param([string]$HttpBaseUri, [int]$TimeoutSeconds)

    $webSocketUri = $HttpBaseUri.TrimEnd('/') -replace '^http:', 'ws:' -replace '^https:', 'wss:'
    $webSocketUri = "$webSocketUri/api/v1/camera/stream"
    $socket = New-Object System.Net.WebSockets.ClientWebSocket
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    try {
        $connectCts = New-Object Threading.CancellationTokenSource
        $connectCts.CancelAfter($TimeoutSeconds * 1000)
        $socket.ConnectAsync([Uri]$webSocketUri, $connectCts.Token).GetAwaiter().GetResult()
        $connectCts.Dispose()

        while ([DateTime]::UtcNow -lt $deadline) {
            $memory = New-Object IO.MemoryStream
            do {
                $buffer = New-Object byte[] 65536
                $segment = [System.ArraySegment[byte]]::new($buffer)
                $remainingMs = [Math]::Max(1, [int]($deadline - [DateTime]::UtcNow).TotalMilliseconds)
                $receiveCts = New-Object Threading.CancellationTokenSource
                $receiveCts.CancelAfter($remainingMs)
                $result = $socket.ReceiveAsync($segment, $receiveCts.Token).GetAwaiter().GetResult()
                $receiveCts.Dispose()
                if ($result.MessageType -eq [Net.WebSockets.WebSocketMessageType]::Close) {
                    throw "The camera WebSocket closed before delivering a frame."
                }
                if ($result.Count -gt 0) { $memory.Write($buffer, 0, $result.Count) }
            } while (-not $result.EndOfMessage)

            $packet = $memory.ToArray()
            $memory.Dispose()
            if ($packet.Length -le 20) { continue }

            $width = [BitConverter]::ToUInt32($packet, 0)
            $height = [BitConverter]::ToUInt32($packet, 4)
            $format = [BitConverter]::ToUInt32($packet, 16)
            if ($width -gt 0 -and $height -gt 0 -and $format -eq 1 -and
                $packet[20] -eq 0xFF -and $packet[21] -eq 0xD8) {
                return [PSCustomObject]@{
                    width = $width
                    height = $height
                    packet_bytes = $packet.Length
                    jpeg_bytes = $packet.Length - 20
                }
            }
        }
        throw "No non-empty JPEG live-view frame arrived within $TimeoutSeconds seconds."
    } finally {
        $socket.Dispose()
    }
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "The SV105 hardware smoke test must run on Windows."
}
if (-not [string]::IsNullOrWhiteSpace($DaemonPath) -and -not [string]::IsNullOrWhiteSpace($BaseUri)) {
    throw "Specify either -DaemonPath (managed isolated daemon) or -BaseUri (already-running daemon), not both."
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$managedDaemon = -not [string]::IsNullOrWhiteSpace($DaemonPath)
if ($managedDaemon) {
    $DaemonPath = (Resolve-Path -LiteralPath $DaemonPath).Path
    $BaseUri = "http://127.0.0.1:$Port"
} elseif ([string]::IsNullOrWhiteSpace($BaseUri)) {
    $BaseUri = "http://127.0.0.1:3030"
}
$BaseUri = $BaseUri.TrimEnd('/')
$api = "$BaseUri/api/v1"

if ([string]::IsNullOrWhiteSpace($WorkDirectory)) {
    $WorkDirectory = Join-Path ([IO.Path]::GetTempPath()) ("achernar-windows-smoke-" + [Guid]::NewGuid().ToString("N"))
} else {
    $WorkDirectory = [IO.Path]::GetFullPath($WorkDirectory)
    if (Test-Path -LiteralPath $WorkDirectory) {
        if (@(Get-ChildItem -LiteralPath $WorkDirectory -Force).Count -gt 0) {
            throw "Smoke-test work directory must be new or empty: $WorkDirectory"
        }
    }
}
New-Item -ItemType Directory -Force -Path $WorkDirectory | Out-Null

$framesDirectory = Join-Path $WorkDirectory "frames"
$configPath = Join-Path $WorkDirectory "achernar.smoke.toml"
$stdoutPath = Join-Path $WorkDirectory "daemon.stdout.log"
$stderrPath = Join-Path $WorkDirectory "daemon.stderr.log"
$process = $null
$connected = $false
$streaming = $false
$success = $false
$restoreControls = New-Object System.Collections.Generic.List[object]
$savedEnvironment = @{}

try {
    if ($managedDaemon) {
        New-Item -ItemType Directory -Force -Path $framesDirectory | Out-Null
        $databasePath = Join-Path $WorkDirectory "achernar.db"
        $config = @"
[daemon]
port = $Port
log_level = "info"

[storage]
database_path = "$(ConvertTo-TomlString $databasePath)"
frames_dir = "$(ConvertTo-TomlString $framesDirectory)"
"@
        $utf8WithoutBom = New-Object Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($configPath, $config, $utf8WithoutBom)

        foreach ($name in @(
            "ACHERNAR_CONFIG", "ACHERNAR_HOST", "ACHERNAR_PORT", "ACHERNAR_RIG_ID",
            "ACHERNAR_CLOUD_HTTP", "ACHERNAR_CLOUD_WS", "ACHERNAR_RELAY_DEV_TOKEN"
        )) {
            $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
        }
        [Environment]::SetEnvironmentVariable("ACHERNAR_CONFIG", $configPath, "Process")
        [Environment]::SetEnvironmentVariable("ACHERNAR_HOST", "127.0.0.1", "Process")
        [Environment]::SetEnvironmentVariable("ACHERNAR_PORT", $Port.ToString(), "Process")
        [Environment]::SetEnvironmentVariable("ACHERNAR_RIG_ID", "windows-smoke-rig", "Process")
        [Environment]::SetEnvironmentVariable("ACHERNAR_CLOUD_HTTP", $null, "Process")
        [Environment]::SetEnvironmentVariable("ACHERNAR_CLOUD_WS", $null, "Process")
        [Environment]::SetEnvironmentVariable("ACHERNAR_RELAY_DEV_TOKEN", $null, "Process")

        $process = Start-Process -FilePath $DaemonPath `
            -WorkingDirectory (Join-Path $repositoryRoot "achernar-daemon") `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -PassThru
    }

    Wait-ForHealth -Uri "$BaseUri/health" -TimeoutSeconds $StartupTimeoutSeconds -Process $process
    Write-Host "PASS  daemon health"

    $drivers = Invoke-AchernarApi -Method Get -Uri "$api/drivers"
    Write-JsonArtifact -Directory $WorkDirectory -Name "drivers.json" -Value $drivers
    $driverCapability = @($drivers.camera | Where-Object { $_.id -eq $Driver } | Select-Object -First 1)
    if ($driverCapability.Count -eq 0) {
        throw "The daemon did not report camera driver '$Driver'."
    }
    if (-not $driverCapability[0].compiled -or -not $driverCapability[0].enabled) {
        throw "Camera driver '$Driver' is not compiled/enabled: $($driverCapability[0].reason)"
    }
    Write-Host "PASS  driver '$Driver' compiled and enabled"

    if ($Driver -eq "sim") {
        if ([string]::IsNullOrWhiteSpace($DriverOption)) {
            $DriverOption = Join-Path $repositoryRoot "OpenLiveStacker_Drivers\OpenLiveStacker\sim"
        }
        if (-not (Test-Path -LiteralPath $DriverOption -PathType Container)) {
            throw "Simulator data directory does not exist: $DriverOption"
        }
        if ($DeviceIndex -lt 0) { $DeviceIndex = 0 }
    } else {
        $discovery = Invoke-AchernarApi -Method Post -Uri "$api/discover"
        Write-JsonArtifact -Directory $WorkDirectory -Name "discovery.json" -Value $discovery
        $matchingCameras = @($discovery.cameras | Where-Object { $_.driver_id -eq $Driver })
        if (-not [string]::IsNullOrWhiteSpace($CameraNamePattern)) {
            $matchingCameras = @($matchingCameras | Where-Object { $_.name -match $CameraNamePattern })
        }
        if ($DeviceIndex -ge 0) {
            $matchingCameras = @($matchingCameras | Where-Object { [int]$_.index -eq $DeviceIndex })
        }
        if ($matchingCameras.Count -eq 0) {
            $seen = @($discovery.cameras | ForEach-Object { "$($_.driver_id):$($_.index) $($_.name)" }) -join "; "
            throw "No matching '$Driver' camera was discovered. Cameras seen: $seen"
        }
        if ($matchingCameras.Count -gt 1 -and $DeviceIndex -lt 0) {
            throw "More than one camera matched. Re-run with -DeviceIndex using the discovery index."
        }
        $DeviceIndex = [int]$matchingCameras[0].index
        Write-Host "PASS  discovered $($matchingCameras[0].name) at $Driver`:$DeviceIndex"
    }

    $connectBody = @{
        equipment_type = "camera"
        driver = $Driver
        index = $DeviceIndex
    }
    if (-not [string]::IsNullOrWhiteSpace($DriverOption)) { $connectBody.option = $DriverOption }
    $connect = Invoke-AchernarApi -Method Post -Uri "$api/connect" -Body $connectBody
    Write-JsonArtifact -Directory $WorkDirectory -Name "connect.json" -Value $connect
    $connected = $true
    if ($connect.status -ne "connected") { throw "Unexpected connect result: $($connect | ConvertTo-Json -Compress)" }
    Write-Host "PASS  connected $($connect.camera)"

    $descriptor = Invoke-AchernarApi -Method Get -Uri "$api/camera/"
    Write-JsonArtifact -Directory $WorkDirectory -Name "camera-descriptor.json" -Value $descriptor
    if (-not $descriptor.connected) { throw "Camera descriptor says the camera is disconnected." }
    $formats = @($descriptor.formats)
    if ($formats.Count -eq 0) { throw "The connected camera exposes no formats." }
    Write-Host "PASS  descriptor: $($formats.Count) format(s), $(@($descriptor.controls).Count) control(s)"

    if (-not $SkipControlWrite) {
        $writable = @($descriptor.controls | Where-Object { -not $_.read_only })
        $autoExposure = @($writable | Where-Object { $_.name -eq "auto_exposure" } | Select-Object -First 1)
        if ($autoExposure.Count -gt 0 -and [double]$autoExposure[0].current -ne 0.0) {
            $restoreControls.Add([PSCustomObject]@{ name = "auto_exposure"; value = [double]$autoExposure[0].current })
            $null = Invoke-AchernarApi -Method Post -Uri "$api/camera/parameters" -Body @{ name = "auto_exposure"; value = 0.0 }
            Write-Host "PASS  disabled auto exposure for control test"
        }

        foreach ($controlName in @("exposure", "gain")) {
            $control = @($writable | Where-Object { $_.name -eq $controlName } | Select-Object -First 1)
            if ($control.Count -eq 0) {
                Write-Host "SKIP  '$controlName' is not exposed by this camera/backend"
                continue
            }
            $testValue = Get-StepAlignedTestValue -Control $control[0]
            $restoreControls.Add([PSCustomObject]@{ name = $controlName; value = [double]$control[0].current })
            $setResult = Invoke-AchernarApi -Method Post -Uri "$api/camera/parameters" -Body @{ name = $controlName; value = $testValue }
            if ($setResult.name -ne $controlName) { throw "Unexpected control response for '$controlName'." }
            Write-Host "PASS  '$controlName' write/readback ($testValue -> $($setResult.applied_value))"
        }
    }

    if ($FormatIndex -ge 0) {
        $selectedFormat = @($formats | Where-Object { [int]$_.index -eq $FormatIndex } | Select-Object -First 1)
    } else {
        $selectedFormat = @($formats | Where-Object { $_.previewable -and $_.mode -eq "liveview" } | Select-Object -First 1)
        if ($selectedFormat.Count -eq 0) {
            $selectedFormat = @($formats | Where-Object { $_.previewable } | Select-Object -First 1)
        }
    }
    if ($selectedFormat.Count -eq 0) { throw "No previewable format matched the requested selection." }
    $FormatIndex = [int]$selectedFormat[0].index

    $start = Invoke-AchernarApi -Method Post -Uri "$api/camera/stream/start?format_index=$FormatIndex"
    Write-JsonArtifact -Directory $WorkDirectory -Name "stream-start.json" -Value $start
    $streaming = $true
    if ($start.status -ne "streaming") { throw "Unexpected stream-start result." }
    Write-Host "PASS  stream started: $($selectedFormat[0].width)x$($selectedFormat[0].height) $($selectedFormat[0].format)"

    $snapshot = $null
    $snapshotDeadline = [DateTime]::UtcNow.AddSeconds($FrameTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $snapshotDeadline) {
        try {
            $snapshot = Invoke-AchernarApi -Method Post -Uri "$api/camera/snapshot" -TimeoutSeconds 5
            if ([int]$snapshot.width -gt 0 -and [int]$snapshot.height -gt 0) { break }
        } catch {
            Start-Sleep -Milliseconds 250
        }
    }
    if ($null -eq $snapshot) { throw "No REST snapshot arrived within $FrameTimeoutSeconds seconds." }
    $jpeg = [Convert]::FromBase64String([string]$snapshot.data_base64)
    if ($jpeg.Length -lt 4 -or $jpeg[0] -ne 0xFF -or $jpeg[1] -ne 0xD8) {
        throw "Snapshot payload was not a JPEG."
    }
    Write-Host "PASS  snapshot JPEG: $($snapshot.width)x$($snapshot.height), $($jpeg.Length) bytes"
    Write-JsonArtifact -Directory $WorkDirectory -Name "snapshot-metadata.json" -Value ([PSCustomObject]@{
        width = $snapshot.width
        height = $snapshot.height
        frame_counter = $snapshot.frame_counter
        timestamp = $snapshot.timestamp
        jpeg_bytes = $jpeg.Length
    })

    $live = Receive-LiveFrame -HttpBaseUri $BaseUri -TimeoutSeconds $FrameTimeoutSeconds
    Write-JsonArtifact -Directory $WorkDirectory -Name "websocket-frame.json" -Value $live
    Write-Host "PASS  WebSocket JPEG: $($live.width)x$($live.height), $($live.jpeg_bytes) bytes"

    if (-not $SkipCapture) {
        $capture = Invoke-AchernarApi -Method Post -Uri "$api/camera/capture" -TimeoutSeconds 120 -Body @{
            exposure = $CaptureExposureSeconds
            capture_type = "verification"
        }
        if ($capture.status -ne "committed" -or @($capture.assets).Count -eq 0) {
            throw "Capture did not commit a persisted asset: $($capture | ConvertTo-Json -Depth 8 -Compress)"
        }
        Write-JsonArtifact -Directory $WorkDirectory -Name "capture.json" -Value $capture
        foreach ($asset in @($capture.assets)) {
            if ([string]$asset.sha256 -notmatch '^[0-9a-fA-F]{64}$' -or [int64]$asset.byte_length -le 0) {
                throw "Capture asset has invalid provenance fields: $($asset | ConvertTo-Json -Compress)"
            }
        }
        Write-Host "PASS  persisted capture $($capture.capture_id) with $(@($capture.assets).Count) asset(s)"
    }

    $null = Invoke-AchernarApi -Method Post -Uri "$api/camera/stream/stop"
    $streaming = $false
    Write-Host "PASS  stream stopped"

    for ($i = $restoreControls.Count - 1; $i -ge 0; $i--) {
        $item = $restoreControls[$i]
        $null = Invoke-AchernarApi -Method Post -Uri "$api/camera/parameters" -Body @{ name = $item.name; value = $item.value }
    }
    if ($restoreControls.Count -gt 0) { Write-Host "PASS  original camera controls restored" }
    $restoreControls.Clear()

    $null = Invoke-AchernarApi -Method Post -Uri "$api/disconnect" -Body @{ equipment_type = "camera" }
    $connected = $false
    Write-Host "PASS  camera disconnected"

    $success = $true
    Write-Host "SV105/daemon smoke test passed. Artifacts: $WorkDirectory"
} finally {
    if ($streaming) {
        try { $null = Invoke-AchernarApi -Method Post -Uri "$api/camera/stream/stop" -TimeoutSeconds 10 } catch { Write-Warning $_ }
    }
    if ($connected -and $restoreControls.Count -gt 0) {
        for ($i = $restoreControls.Count - 1; $i -ge 0; $i--) {
            $item = $restoreControls[$i]
            try {
                $null = Invoke-AchernarApi -Method Post -Uri "$api/camera/parameters" -TimeoutSeconds 10 -Body @{ name = $item.name; value = $item.value }
            } catch { Write-Warning "Could not restore '$($item.name)': $_" }
        }
    }
    if ($connected) {
        try { $null = Invoke-AchernarApi -Method Post -Uri "$api/disconnect" -TimeoutSeconds 10 -Body @{ equipment_type = "camera" } } catch { Write-Warning $_ }
    }
    if ($null -ne $process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
        $process.WaitForExit(5000) | Out-Null
    }
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], "Process")
    }

    if (-not $success -and (Test-Path -LiteralPath $stderrPath -PathType Leaf)) {
        Write-Host "--- daemon stderr (last 80 lines) ---"
        Get-Content -LiteralPath $stderrPath -Tail 80
    }
    if (-not $success) {
        Write-Warning "Smoke-test artifacts were preserved at $WorkDirectory"
    } elseif (-not $KeepArtifacts -and $managedDaemon) {
        $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
        $resolvedWork = [IO.Path]::GetFullPath($WorkDirectory)
        if ($resolvedWork.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
            (Split-Path -Leaf $resolvedWork).StartsWith("achernar-windows-smoke-")) {
            Remove-Item -LiteralPath $resolvedWork -Recurse -Force
        }
    }
}
