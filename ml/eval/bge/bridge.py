"""Frozen bridge v1 between this harness and the daemon's ``bge_eval_bridge`` test.

Input (written here)::

    <dir>/manifest.json   {"schema": "bge-eval-bridge-v1", "cases": [...], "backends": [...],
                           "settings": {...}, "write_maps": bool, "extras": {...}}
    <dir>/inputs/<id>.f32            float32 little-endian, row-major
    <dir>/inputs/<id>.<mask>.u8      uint8 0/1, row-major (only masks that apply)

Each case also carries ``extras.sha256`` (sha256 of every input file); the
top-level ``extras.input_sha256`` repeats them by relative path. The Rust
reader ignores unknown keys (coordinator, bridge v1).

Output (written by the daemon test)::

    <out>/<id>__<backend>.json        {"case", "backend", "background", "detection",
                                       "catalog", "candidates", "quality", "trailing",
                                       "timing_ms", "error"}
    <out>/<id>__<backend>.back.f32    optional, when write_maps and the field is spatial
    <out>/<id>__<backend>.rms.f32

Run (see ``run_bridge``)::

    ACHERNAR_BGE_EVAL_MANIFEST=<dir>/manifest.json ACHERNAR_BGE_EVAL_OUT=<out> \\
      cargo test --release --manifest-path <repo>/achernar-daemon/Cargo.toml \\
      bge_eval_bridge -- --ignored --nocapture
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import BRIDGE_SCHEMA, GENERATOR_VERSION, HARNESS_VERSION

RUST_BACKENDS = ("legacy_global_clip", "sep")
MASK_KEYS = ("bad_pixel", "invalid", "edge", "native_saturation")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
UNAVAILABLE = "skipped: rust bridge unavailable"


def valid_case_id(case_id: str) -> bool:
    return bool(_ID_RE.match(case_id)) and "__" not in case_id


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_inputs(scenarios, root: Path) -> list:
    """Write pixels and masks for ``scenarios`` under ``root/inputs``; return case entries."""
    root = Path(root)
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    cases = []
    for sc in scenarios:
        if not valid_case_id(sc.id):
            raise ValueError(f"case id {sc.id!r} is not filesystem/bridge safe")
        if sc.image.ndim != 2 or sc.image.size == 0:
            raise ValueError(f"case {sc.id!r} has zero-size geometry; bridge v1 excludes it")
        rel = f"inputs/{sc.id}.f32"
        np.ascontiguousarray(sc.image, dtype="<f4").tofile(root / rel)
        sha = {rel: _sha256_file(root / rel)}
        masks = {}
        for name in MASK_KEYS:
            if name in sc.masks:
                mrel = f"inputs/{sc.id}.{name}.u8"
                np.ascontiguousarray(sc.masks[name], dtype=np.uint8).tofile(root / mrel)
                masks[name] = mrel
                sha[mrel] = _sha256_file(root / mrel)
        cases.append({
            "id": sc.id,
            "width": sc.width,
            "height": sc.height,
            "pixels": rel,
            "masks": masks,
            "saturation_adu": sc.metadata.get("saturation_adu"),
            "units": "adu",
            "extras": {"sha256": sha, "generator_version": sc.metadata.get("generator_version"),
                       "seed": sc.metadata.get("seed")},
        })
    return cases


def write_manifest(cases: list, root: Path, name: str = "manifest.json", backends=RUST_BACKENDS,
                   settings: dict | None = None, write_maps: bool = True, extras: dict | None = None) -> Path:
    root = Path(root)
    input_sha = {}
    for c in cases:
        input_sha.update(c["extras"]["sha256"])
    manifest = {
        "schema": BRIDGE_SCHEMA,
        "cases": cases,
        "backends": list(backends),
        "settings": settings or {},
        "write_maps": bool(write_maps),
        "extras": dict({"generator_version": GENERATOR_VERSION, "harness_version": HARNESS_VERSION,
                        "input_sha256": input_sha}, **(extras or {})),
    }
    path = root / name
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return path


def read_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != BRIDGE_SCHEMA:
        raise ValueError(f"unexpected bridge schema {manifest.get('schema')!r}")
    return manifest


def load_case(manifest_path: Path, case: dict, verify: bool = True):
    """Read a case's pixels and masks back; verify recorded sha256 values."""
    root = Path(manifest_path).parent
    shape = (case["height"], case["width"])
    if verify:
        for rel, sha in case.get("extras", {}).get("sha256", {}).items():
            if _sha256_file(root / rel) != sha:
                raise ValueError(f"sha256 mismatch for {rel}")
    image = np.fromfile(root / case["pixels"], dtype="<f4").reshape(shape)
    masks = {k: np.fromfile(root / v, dtype=np.uint8).reshape(shape) for k, v in case.get("masks", {}).items()}
    return image, masks


def output_stem(out_dir: Path, case_id: str, backend: str) -> Path:
    return Path(out_dir) / f"{case_id}__{backend}"


def read_output(out_dir: Path, case_id: str, backend: str, width: int, height: int) -> dict:
    """Parse one bridge output. Status: completed | failed | missing."""
    stem = output_stem(out_dir, case_id, backend)
    path = stem.with_suffix(".json")
    if not path.exists():
        return {"status": "missing", "reason": f"no bridge output {path.name}"}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {"status": "failed", "reason": f"unparseable bridge output: {exc}"}
    if payload.get("case") != case_id or payload.get("backend") != backend:
        return {"status": "failed", "reason": "bridge output case/backend does not match its file name"}
    out = {"status": "failed" if payload.get("error") else "completed", "json": payload,
           "reason": payload.get("error"), "back": None, "rms": None}
    for key, suffix in (("back", ".back.f32"), ("rms", ".rms.f32")):
        mpath = Path(str(stem) + suffix)
        if mpath.exists():
            data = np.fromfile(mpath, dtype="<f4")
            if data.size != width * height:
                out["status"] = "failed"
                out["reason"] = f"{mpath.name} has {data.size} values, expected {width * height}"
            else:
                out[key] = data.reshape(height, width)
    return out


def bridge_source_present(repo_root: Path) -> bool:
    src = Path(repo_root) / "achernar-daemon" / "src"
    if not src.is_dir():
        return False
    pattern = re.compile(r"fn\s+bge_eval_bridge\b")
    for path in src.rglob("*.rs"):
        try:
            if pattern.search(path.read_text(errors="ignore")):
                return True
        except OSError:
            continue
    return False


def _cargo_env(cargo_target_dir, manifest_path, out_dir):
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(cargo_target_dir)
    env["ACHERNAR_BGE_EVAL_MANIFEST"] = str(manifest_path)
    env["ACHERNAR_BGE_EVAL_OUT"] = str(out_dir)
    return env


def rust_revision(repo_root: Path) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {"repo": str(repo_root), "commit": git("rev-parse", "HEAD"),
            "dirty_daemon_files": git("status", "--porcelain", "--", "achernar-daemon")}


def run_bridge(manifest_path: Path, out_dir: Path, repo_root: Path, cargo_target_dir: Path,
               log_path: Path, jobs: int = 4, timeout_s: int = 5400) -> dict:
    """Run the ignored daemon test. Absent or unbuildable bridge -> skipped."""
    repo_root = Path(repo_root)
    cargo_toml = repo_root / "achernar-daemon" / "Cargo.toml"
    info = {"revision": rust_revision(repo_root), "log": str(log_path), "manifest": str(manifest_path),
            "out_dir": str(out_dir)}
    if not cargo_toml.exists():
        return dict(info, status="skipped", reason=f"{UNAVAILABLE}: {cargo_toml} not found")
    cargo = shutil.which("cargo")
    if cargo is None:
        return dict(info, status="skipped", reason=f"{UNAVAILABLE}: cargo not on PATH")
    if not bridge_source_present(repo_root):
        return dict(info, status="skipped",
                    reason=f"{UNAVAILABLE}: no `fn bge_eval_bridge` in {repo_root}/achernar-daemon/src")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [cargo, "test", "--release", "-j", str(jobs), "--manifest-path", str(cargo_toml), "bge_eval_bridge",
           "--", "--ignored", "--nocapture"]
    info["command"] = " ".join(cmd)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=_cargo_env(cargo_target_dir, manifest_path, out_dir), capture_output=True,
                              text=True, timeout=timeout_s, cwd=cargo_toml.parent)
    except subprocess.TimeoutExpired:
        return dict(info, status="failed", reason=f"bridge timed out after {timeout_s} s")
    info["wall_s"] = time.perf_counter() - t0
    Path(log_path).write_text(f"$ {info['command']}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    info["returncode"] = proc.returncode
    text = proc.stdout + proc.stderr
    if proc.returncode != 0:
        if "could not compile" in text or re.search(r"^error(\[E\d+\])?:", text, re.M):
            return dict(info, status="skipped", reason=f"{UNAVAILABLE}: build failed (see log)")
        return dict(info, status="failed", reason="bridge test failed (see log)")
    if re.search(r"running 0 tests", text) and not re.search(r"running [1-9]\d* tests?", text):
        return dict(info, status="skipped", reason=f"{UNAVAILABLE}: no test matched bge_eval_bridge")
    return dict(info, status="completed", reason=None)


def bridge_test_binary(repo_root: Path, cargo_target_dir: Path, jobs: int = 4) -> str | None:
    """Path of the release test binary containing ``bge_eval_bridge`` (already built)."""
    cargo = shutil.which("cargo")
    if cargo is None:
        return None
    cmd = [cargo, "test", "--release", "-j", str(jobs), "--no-run", "--message-format=json", "--manifest-path",
           str(Path(repo_root) / "achernar-daemon" / "Cargo.toml")]
    env = dict(os.environ, CARGO_TARGET_DIR=str(cargo_target_dir))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=5400,
                          cwd=Path(repo_root) / "achernar-daemon")
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") == "compiler-artifact" and msg.get("executable") and msg.get("profile", {}).get("test"):
            if "bin" in msg.get("target", {}).get("kind", []):
                return msg["executable"]
    return None


def run_binary_timed(binary: str, manifest_path: Path, out_dir: Path, log_path: Path, timeout_s=3600) -> dict:
    """Run the test binary directly under ``/usr/bin/time -l`` (macOS) for peak RSS."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ACHERNAR_BGE_EVAL_MANIFEST=str(manifest_path), ACHERNAR_BGE_EVAL_OUT=str(out_dir))
    cmd = ["/usr/bin/time", "-l", binary, "bge_eval_bridge", "--ignored", "--nocapture", "--test-threads=1"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    wall = time.perf_counter() - t0
    Path(log_path).write_text(f"$ {' '.join(cmd)}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    rss = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
    real = re.search(r"([\d.]+)\s+real", proc.stderr)
    return {"returncode": proc.returncode, "wall_s": wall, "time_real_s": float(real.group(1)) if real else None,
            "peak_rss_bytes": int(rss.group(1)) if rss else None, "command": " ".join(cmd), "log": str(log_path)}
