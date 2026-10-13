#!/usr/bin/env python3
"""Compile and run the production gphoto driver's native regressions.

From the repository root: ``python3 tools/run_gphoto_retention_test.py``.
Requires the existing Unix native build dependencies: a C++17 compiler,
pkg-config, libgphoto2 and OpenCV 4. The runner discovers Homebrew's opencv@4
keg, honors CXX/PKG_CONFIG_PATH, and builds in a temporary directory.

No physical camera is opened or triggered. A second fixture mocks PTP events
and downloads while exercising the production capture/bridge code, including
late companions and partial transfers. The optional RAW decoder is not needed.
The control fixture mocks config/preview traffic and exercises a real preview
thread to verify serialized writes, immediate readback and native errors.
These tests do not qualify physical card retention or archive durability.
"""
from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
OLS = REPO / "OpenLiveStacker_Drivers" / "OpenLiveStacker"


def native_environment() -> dict[str, str]:
    env = os.environ.copy()
    if platform.system() != "Darwin":
        return env
    brew = shutil.which("brew")
    if not brew:
        return env
    paths = []
    for formula in ("opencv@4", "libgphoto2"):
        result = subprocess.run(
            [brew, "--prefix", formula], capture_output=True, text=True, env=env,
        )
        if result.returncode == 0:
            pkgconfig = Path(result.stdout.strip()) / "lib" / "pkgconfig"
            if pkgconfig.is_dir():
                paths.append(str(pkgconfig))
    if env.get("PKG_CONFIG_PATH"):
        paths.append(env["PKG_CONFIG_PATH"])
    if paths:
        env["PKG_CONFIG_PATH"] = os.pathsep.join(paths)
    return env


def pkg_flags(option: str, packages: list[str], env: dict[str, str]) -> list[str]:
    result = subprocess.run(
        ["pkg-config", option] + packages,
        capture_output=True, text=True, env=env,
    )
    if result.returncode:
        raise RuntimeError(
            "Native test requires libgphoto2 and OpenCV 4 development packages "
            "visible to pkg-config.\n" + result.stderr.strip()
        )
    return shlex.split(result.stdout)


def main() -> int:
    if platform.system() == "Windows":
        raise RuntimeError("The gphoto2 driver and this native test require a Unix build")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or not shutil.which(compiler[0]):
        raise RuntimeError("Set CXX to an available C++17 compiler")
    if not shutil.which("pkg-config"):
        raise RuntimeError("pkg-config is required by the native test")
    env = native_environment()
    cflags = pkg_flags("--cflags", ["libgphoto2", "opencv4"], env)
    # OpenCV's aggregate package lists many unrelated optional modules. Match
    # the driver's three components, keeping this test's linkage focused.
    libs = pkg_flags("--libs", ["libgphoto2"], env)
    libs += pkg_flags("--libs-only-L", ["opencv4"], env)
    libs += ["-lopencv_imgcodecs", "-lopencv_imgproc", "-lopencv_core"]
    bridge = REPO / "achernar-daemon" / "ols-bridge"
    # Keep only the exported error/deletion functions from the bridge. Other
    # bridge drivers are irrelevant to this regression and need no SDK linkage.
    strip_flags = ["-Wl,-dead_strip"] if platform.system() == "Darwin" else ["-Wl,--gc-sections"]
    with tempfile.TemporaryDirectory(prefix="achernar-gphoto-retention-") as tmp:
        binary = Path(tmp) / "retention-test"
        subprocess.run(
            compiler + [
                "-std=c++17", "-pthread", "-ffunction-sections", "-fdata-sections",
                "-I" + str(OLS / "include"),
                "-I" + str(bridge / "include"), "-I" + str(bridge / "shims"),
            ] + cflags + [
                str(OLS / "test" / "test_gphoto_retention.cpp"),
                str(bridge / "src" / "camera_base.cpp"),
                str(bridge / "src" / "ols_bridge.cpp"),
            ] + libs + strip_flags + ["-o", str(binary)],
            check=True, env=env,
        )
        subprocess.run([str(binary)], check=True, env=env)
        snapshot = Path(tmp) / "snapshot-test"
        subprocess.run(
            compiler + [
                "-std=c++17", "-pthread", "-ffunction-sections", "-fdata-sections",
                "-I" + str(OLS / "include"), "-I" + str(bridge / "include"),
                "-I" + str(bridge / "shims"),
            ] + cflags + [
                str(bridge / "tests" / "test_still_snapshot.cpp"),
                str(bridge / "src" / "camera_base.cpp"),
            ] + libs + strip_flags + ["-o", str(snapshot)],
            check=True, env=env,
        )
        subprocess.run([str(snapshot)], check=True, env=env)
        controls = Path(tmp) / "controls-test"
        subprocess.run(
            compiler + [
                "-std=c++17", "-pthread", "-ffunction-sections", "-fdata-sections",
                "-I" + str(OLS / "include"), "-I" + str(bridge / "include"),
                "-I" + str(bridge / "shims"),
            ] + cflags + [
                str(OLS / "test" / "test_gphoto_controls.cpp"),
                str(bridge / "src" / "camera_base.cpp"),
            ] + libs + strip_flags + ["-o", str(controls)],
            check=True, env=env,
        )
        subprocess.run([str(controls)], check=True, env=env)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"GPhoto retention test could not complete: {error}", file=sys.stderr)
        sys.exit(1)
