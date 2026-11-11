"""Frozen bridge v1 between this harness and the daemon's ``bge_eval_bridge`` test.

Input (written here)::

    <dir>/manifest.json   {"schema": "bge-eval-bridge-v1", "cases": [...], "backends": [...],
                           "settings": {...}, "write_maps": bool, "extras": {...}}
    <dir>/inputs/<id>.f32            float32 little-endian, row-major
    <dir>/inputs/<id>.<mask>.u8      uint8 0/1, row-major (only masks that apply)

Each case also carries ``extras.sha256`` (sha256 of every input file); the
top-level ``extras.input_sha256`` repeats them by relative path. The Rust
reader ignores unknown keys (coordinator, bridge v1).

BGE v2 (docs/bge-v2-spec.md §4): ``"settings"`` is merged into the daemon's
``BackgroundSettings`` (e.g. ``{"sep": {"noise_policy": "masked_local_v2"}}``)
and an optional ``"star_gates": {"blend_test": ..., "regroup_deblended": ...}``
sits beside it, at the top level and per case, merged the same way. Each
method id of :data:`RUST_VARIANTS` (aliases included) is itself a bridge
backend that forces its backend, noise policy, blend test and regrouping, so
one manifest lists every variant: ``<id>__<method>.json``. The plain backend
strings ``legacy_global_clip``/``sep`` run the daemon's *current defaults*
(now the v2 policies) and are never used to label a v1 row. Outputs
additionally carry ``background.refinement``,
``quality.gate_diagnostics``, ``quality.star_gates_config_sha256`` and
``quality.used_stars``; the readers below also accept them at the top level.

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
import sys
import time
from pathlib import Path

import numpy as np

from . import BRIDGE_SCHEMA, GENERATOR_VERSION, HARNESS_VERSION

RUST_BACKENDS = ("legacy_global_clip", "sep")

_G1 = {"blend_test": "peak_count_v1", "regroup_deblended": False}
_G2 = {"blend_test": "dip_annulus_v2", "regroup_deblended": True}
_N1 = {"sep": {"noise_policy": "frame_wide_v1"}}
_N2 = {"sep": {"noise_policy": "masked_local_v2"}}
#: Rust method matrix (spec §4): backend, BackgroundSettings overlay and
#: star_gates overlay. Every variant passes its policies explicitly, because
#: the daemon defaults are now the v2 policies. ``rust_legacy_global_clip``
#: and ``rust_sep`` are aliases of the v1 baselines.
RUST_VARIANTS = {
    "rust_legacy__g1": {"backend": "legacy_global_clip", "settings": {}, "star_gates": _G1},
    "rust_legacy__g2": {"backend": "legacy_global_clip", "settings": {}, "star_gates": _G2},
    "rust_sep__n1g1": {"backend": "sep", "settings": _N1, "star_gates": _G1},
    "rust_sep__n2g1": {"backend": "sep", "settings": _N2, "star_gates": _G1},
    "rust_sep__n1g2": {"backend": "sep", "settings": _N1, "star_gates": _G2},
    "rust_sep__n2g2": {"backend": "sep", "settings": _N2, "star_gates": _G2},
}
RUST_ALIASES = {"rust_legacy_global_clip": "rust_legacy__g1", "rust_sep": "rust_sep__n1g1"}


def rust_variant(method: str) -> tuple[str, dict]:
    """``(canonical id, spec)`` of a Rust method id or alias."""
    canonical = RUST_ALIASES.get(method, method)
    return canonical, RUST_VARIANTS[canonical]


def merge_overlay(base: dict | None, overlay: dict | None) -> dict:
    """Recursive JSON-object merge (``overlay`` wins), as the bridge merges."""
    out = dict(base or {})
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_overlay(out[key], value)
        else:
            out[key] = value
    return out


def _first(payload: dict, *paths):
    for path in paths:
        node = payload
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if node is not None:
            return node
    return None


def output_refinement(payload: dict):
    """``refinement`` record of a bridge output (``None`` for frame_wide_v1/legacy or old revisions)."""
    return _first(payload, ("background", "refinement"), ("refinement",))


def output_gate_diagnostics(payload: dict):
    return _first(payload, ("quality", "gate_diagnostics"), ("gate_diagnostics",))


def output_star_gates_sha256(payload: dict):
    return _first(payload, ("quality", "star_gates_config_sha256"), ("star_gates_config_sha256",))
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
                   settings: dict | None = None, write_maps: bool = True, extras: dict | None = None,
                   star_gates: dict | None = None) -> Path:
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
    if star_gates is not None:
        manifest["star_gates"] = dict(star_gates)
    path = root / name
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return path


def write_variant_manifest(cases: list, root: Path, method: str, name: str | None = None,
                           settings: dict | None = None, write_maps: bool = True, extras: dict | None = None) -> Path:
    """Manifest of one Rust method (alias allowed): the canonical method id as
    the backend (it forces the variant), plus the variant's settings and
    star_gates overlays (redundant with the id; kept so the manifest states
    the variant explicitly)."""
    canonical, spec = rust_variant(method)
    return write_manifest(cases, root, name=name or f"manifest_{canonical}.json", backends=(canonical,),
                          settings=merge_overlay(settings, spec["settings"]), write_maps=write_maps,
                          extras=dict({"method": canonical}, **(extras or {})), star_gates=spec["star_gates"])


def write_methods_manifest(cases: list, root: Path, methods, name: str = "manifest.json",
                           settings: dict | None = None, write_maps: bool = True, extras: dict | None = None) -> Path:
    """One manifest for several Rust methods: ``backends`` = their canonical
    method ids (each forces its variant); ``settings`` (e.g. a mesh size) is
    shared."""
    canonical = []
    for m in methods:
        c, _ = rust_variant(m)
        if c not in canonical:
            canonical.append(c)
    return write_manifest(cases, root, name=name, backends=canonical, settings=settings, write_maps=write_maps,
                          extras=dict({"methods": canonical}, **(extras or {})))


def backend_of(backend_or_method: str) -> str:
    """``legacy_global_clip``/``sep`` of a backend string or method id."""
    if backend_or_method in RUST_BACKENDS:
        return backend_or_method
    return rust_variant(backend_or_method)[1]["backend"]


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
    return dict(info, **classify_cargo_test(proc.returncode, proc.stdout + proc.stderr))


_RUNNING_BINARY = re.compile(r"Running unittests src/main\.rs \(([^)]+)\)")
_COMPILE_ERROR = re.compile(r"^error\[E\d+\]", re.M)


def classify_cargo_test(returncode: int, text: str) -> dict:
    """Status of one ``cargo test`` run of the bridge.

    A build failure (``skipped: rust bridge unavailable``) means no test
    binary ran: "could not compile", an ``error[E...]`` diagnostic, or no
    "Running unittests" line. Once the binary ran, a non-zero exit (panic,
    "test failed, to rerun pass", a signal) is a runtime ``failed``; cargo's
    own "error: test failed" line is not a build error.
    """
    exe = _RUNNING_BINARY.search(text)
    out = {}
    if exe:
        out["binary"] = exe.group(1)
        try:
            out["binary_sha256"] = _sha256_file(Path(exe.group(1)))
        except OSError:
            out["binary_sha256"] = None
    if returncode != 0:
        if "could not compile" in text or _COMPILE_ERROR.search(text) or not exe:
            return dict(out, status="skipped", reason=f"{UNAVAILABLE}: build failed (see log)")
        detail = []
        panic = re.search(r"panicked at ([^\n]+)\n?([^\n]*)", text)
        if panic:
            detail.append(f"panic at {panic.group(1).strip()} {panic.group(2).strip()}".strip())
        sig = re.search(r"signal: (\d+)(?:, (SIG\w+))?", text)
        if sig or returncode < 0:
            detail.append(f"terminated by signal {sig.group(2) or sig.group(1) if sig else -returncode}")
        if "test result: FAILED" in text or "test failed, to rerun pass" in text:
            detail.append("test result FAILED")
        return dict(out, status="failed",
                    reason=f"bridge test ran and failed at runtime (exit {returncode}): "
                           f"{'; '.join(detail) or 'see log'}")
    if re.search(r"running 0 tests", text) and not re.search(r"running [1-9]\d* tests?", text):
        return dict(out, status="skipped", reason=f"{UNAVAILABLE}: no test matched bge_eval_bridge")
    return dict(out, status="completed", reason=None)


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
    test_args = [binary, "bge_eval_bridge", "--ignored", "--nocapture", "--test-threads=1"]
    if sys.platform == "darwin" and os.path.exists("/usr/bin/time"):
        cmd = ["/usr/bin/time", "-l", *test_args]  # BSD time: "maximum resident set size" in bytes
    elif os.path.exists("/usr/bin/time"):
        cmd = ["/usr/bin/time", "-v", *test_args]  # GNU time: "Maximum resident set size (kbytes)"
    else:
        cmd = test_args
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    wall = time.perf_counter() - t0
    Path(log_path).write_text(f"$ {' '.join(cmd)}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    rss = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
    gnu = re.search(r"Maximum resident set size \(kbytes\): (\d+)", proc.stderr)
    peak = int(rss.group(1)) if rss else (int(gnu.group(1)) * 1024 if gnu else None)
    real = re.search(r"([\d.]+)\s+real", proc.stderr)
    return {"returncode": proc.returncode, "wall_s": wall, "time_real_s": float(real.group(1)) if real else None,
            "peak_rss_bytes": peak, "command": " ".join(cmd), "log": str(log_path)}
