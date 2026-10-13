#!/usr/bin/env python3
"""Run A04 production native unpack checks without opening a camera.

Optional: --raw-fixture /absolute/path/to/existing.CR3
Requires LibRaw and OpenCV development packages. Builds only in a temp dir.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile

from run_gphoto_retention_test import native_environment, pkg_flags, OLS, REPO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-fixture", type=Path)
    args = parser.parse_args()
    env = native_environment()
    brew = shutil.which("brew")
    if platform.system() == "Darwin" and brew:
        prefix = subprocess.check_output([brew, "--prefix", "libraw"], text=True).strip()
        env["PKG_CONFIG_PATH"] = str(Path(prefix) / "lib/pkgconfig") + os.pathsep + env.get("PKG_CONFIG_PATH", "")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    bridge = REPO / "achernar-daemon/ols-bridge"
    cflags = pkg_flags("--cflags", ["libraw", "opencv4"], env)
    libs = pkg_flags("--libs", ["libraw"], env)
    if platform.system() == "Darwin":
        libs = [flag for flag in libs if flag != "-lstdc++"]
    libs += pkg_flags("--libs-only-L", ["opencv4"], env)
    libs += ["-lopencv_imgcodecs", "-lopencv_imgproc", "-lopencv_core"]
    strip = ["-Wl,-dead_strip"] if platform.system() == "Darwin" else ["-Wl,--gc-sections"]
    with tempfile.TemporaryDirectory(prefix="achernar-untouched-raw-") as temporary:
        binary = str(Path(temporary) / "raw-test")
        subprocess.run(compiler + ["-std=c++17", "-DWITH_LIBRAW", "-pthread",
            "-ffunction-sections", "-fdata-sections", "-I" + str(OLS / "include"),
            "-I" + str(bridge / "include"), "-I" + str(bridge / "shims")]
            + cflags + [str(bridge / "tests/test_untouched_raw.cpp"),
                         str(bridge / "src/camera_base.cpp")]
            + libs + strip + ["-o", binary], check=True, env=env)
        subprocess.run([binary] + ([str(args.raw_fixture.resolve())] if args.raw_fixture else []), check=True, env=env)


if __name__ == "__main__":
    main()
