#!/usr/bin/env python3
"""Verify opt-in native preview stage timing against mocked blocking PTP calls.

No camera is opened, exposure triggered, or camera setting written. This uses
existing native regression dependencies, compiling in a temporary directory.
"""
from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import tempfile
from pathlib import Path

from run_gphoto_retention_test import OLS, REPO, native_environment, pkg_flags


def main() -> None:
    env = native_environment()
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    cflags = pkg_flags("--cflags", ["libgphoto2", "opencv4"], env)
    libs = pkg_flags("--libs", ["libgphoto2"], env)
    libs += pkg_flags("--libs-only-L", ["opencv4"], env)
    libs += ["-lopencv_imgcodecs", "-lopencv_imgproc", "-lopencv_core"]
    strip = ["-Wl,-dead_strip"] if platform.system() == "Darwin" else ["-Wl,--gc-sections"]
    bridge = REPO / "achernar-daemon" / "ols-bridge"
    with tempfile.TemporaryDirectory(prefix="achernar-preview-timing-") as tmp:
        binary = Path(tmp) / "preview-timing-test"
        subprocess.run(compiler + [
            "-std=c++17", "-pthread", "-ffunction-sections", "-fdata-sections",
            "-I" + str(OLS / "include"), "-I" + str(bridge / "include"),
            "-I" + str(bridge / "shims"),
        ] + cflags + [
            str(OLS / "test" / "test_gphoto_preview_timing.cpp"),
            str(bridge / "src" / "camera_base.cpp"),
        ] + libs + strip + ["-o", str(binary)], check=True, env=env)
        result = subprocess.run([str(binary)], capture_output=True, text=True, env=env)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    prefix = "[gphoto2-preview-timing] "
    windows = [json.loads(line[len(prefix):]) for line in result.stderr.splitlines()
               if line.startswith(prefix)]
    assert windows, "Production stream emitted no native timing window"
    assert sum(window["frames"] for window in windows) > 0
    assert max(window["max_sdk_ms"] for window in windows) >= 80, windows
    assert max(window["max_callback_ms"] for window in windows) >= 25, windows
    # The deliberate SDK/callback stalls must appear in their own stages. A
    # fast final frame racing stop_stream can still have ordinary pacing sleep.
    assert all(window["requested_fps"] == 60 for window in windows), windows
    assert all(window["width"] == 6 and window["height"] == 4 for window in windows)
    assert all(window["frames"] == window["attempts"] for window in windows)
    assert all(window["mean_payload_ms"] < window["mean_callback_ms"] for window in windows)
    print(result.stdout.strip())
    print(json.dumps({"mocked_windows": windows}, indent=2))


if __name__ == "__main__":
    main()
