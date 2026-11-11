"""Method variants (spec ``docs/bge-v2-spec.md`` section 4) behind one interface.

Python variants run through ``ml.eval.bge.python_methods`` (imported by name;
a variant the module does not implement is ``unavailable`` with the reason).
Rust variants run through ``ml.eval.bge.bridge.run_bridge`` with one bridge
manifest per variant: ``backends`` = the variant's backend, ``settings`` =
``{"sep": {"noise_policy": ...}}`` and a top-level ``star_gates`` object
(spec section 4). A bridge revision that predates v2 merges only ``settings``
and ignores unknown keys, so every Rust output is checked to echo its variant:
``background.config.noise_policy`` (absent for frame_wide_v1, whose config is
byte-identical to v1), and the top-level ``star_gates`` record, a
``star_gates_config``/``blend_test`` key, or the spec section 1.6 hash
``star_gates_config_sha256`` in ``quality``/``detection``. A baseline variant
(n1, g1) is accepted without an echo because the pre-v2 daemon is that
behaviour; any v2 variant whose output does not echo it is ``unavailable``,
never silently scored as the default.

``oracle_true_background`` (synthetic validation only) uses truth maps a
manifest frame may carry for tests; real frames have none.
"""

from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import MASK_NAMES, config_sha256

NOISE_POLICIES = {"n1": "frame_wide_v1", "n2": "masked_local_v2"}
BLEND_TESTS = {"g1": "peak_count_v1", "g2": "dip_annulus_v2"}


def _variant(kind, backend, noise, gate):
    spec = {"kind": kind, "backend": backend, "blend_test": BLEND_TESTS[gate]}
    if noise is not None:
        spec["noise_policy"] = NOISE_POLICIES[noise]
        spec["regroup_deblended"] = gate == "g2"
    spec["baseline"] = (noise in (None, "n1")) and gate == "g1"
    return spec


METHOD_SPECS = {
    "rust_legacy__g1": _variant("rust", "legacy_global_clip", None, "g1"),
    "rust_legacy__g2": _variant("rust", "legacy_global_clip", None, "g2"),
    **{f"rust_sep__{n}{g}": _variant("rust", "sep", n, g) for n in ("n1", "n2") for g in ("g1", "g2")},
    **{f"python_sep__{n}{g}": _variant("python", "sep", n, g) for n in ("n1", "n2") for g in ("g1", "g2")},
    "python_sep": {"kind": "python", "backend": "sep", "baseline": True, "noise_policy": "frame_wide_v1",
                   "policy": "harness acceptance policy (python_methods.ACCEPTANCE_POLICY), not the daemon gates",
                   "role": "method-independent reference for PSF estimation and injection placement"},
    "oracle_true_background": {"kind": "oracle", "backend": "truth", "baseline": True,
                               "note": "synthetic validation only (truth maps); not a method"},
}
ALIASES = {"rust_legacy_global_clip": "rust_legacy__g1", "rust_sep": "rust_sep__n1g1"}
DEFAULT_METHODS = ("rust_legacy__g1", "rust_legacy__g2", "rust_sep__n1g1", "rust_sep__n2g1", "rust_sep__n1g2",
                   "rust_sep__n2g2", "python_sep", "python_sep__n1g1", "python_sep__n2g2")
#: Method-independent reference (PSF stars, injection avoidance, reference levels):
#: the harness Python SEP path, which exists at every revision of python_methods.
REFERENCE_METHOD = "python_sep"


def canonical(method_id: str) -> str:
    mid = ALIASES.get(method_id, method_id)
    if mid not in METHOD_SPECS:
        raise KeyError(f"unknown method {method_id!r}; known: {sorted(METHOD_SPECS) + sorted(ALIASES)}")
    return mid


def star_gates_record(blend_test: str, regroup_deblended: bool, accept_deblended: bool = False) -> dict:
    """Spec section 1.6 canonical star-gate configuration record."""
    if blend_test == "dip_annulus_v2":
        return {"blend_test": "dip_annulus_v2", "peak_window_half": 3, "threshold_sigma": 5.0,
                "dip_fraction": 0.15, "dip_min_sigma": 4.0,
                "annulus": {"bins": 16, "bin_fraction": 0.5, "min_coverage": 0.625, "half_width": 0.25,
                            "hole_radius": 0.35, "hole_fraction": 0.6, "min_radius_px": 2.5, "radius_tol": 0.35},
                "regroup_deblended": bool(regroup_deblended), "accept_deblended": bool(accept_deblended)}
    return {"blend_test": "peak_count_v1", "peak_window_half": 3, "threshold_sigma": 5.0,
            "regroup_deblended": bool(regroup_deblended), "accept_deblended": bool(accept_deblended)}


def expected_gate_hashes(blend_test: str) -> set:
    return {config_sha256(star_gates_record(blend_test, r, a)) for r in (False, True) for a in (False, True)}


def bridge_overlays(method_id: str) -> tuple:
    """``(backend, settings, star_gates)`` for a Rust variant's bridge manifest."""
    spec = METHOD_SPECS[canonical(method_id)]
    settings = {"sep": {"noise_policy": spec["noise_policy"]}} if spec["backend"] == "sep" else {}
    gates = {"blend_test": spec["blend_test"]}
    if "regroup_deblended" in spec:
        gates["regroup_deblended"] = spec["regroup_deblended"]
    return spec["backend"], settings, gates


def method_config(method_id: str) -> dict:
    """Static description of a variant, hashed into the hold-out ledger."""
    mid = canonical(method_id)
    spec = dict(METHOD_SPECS[mid])
    out = {"method": mid, "spec": spec}
    if spec["kind"] == "rust":
        backend, settings, gates = bridge_overlays(mid)
        out["bridge"] = {"backend": backend, "settings": settings, "star_gates": gates}
    elif spec["kind"] == "python":
        try:
            from .. import python_methods as PM

            out["python"] = {"background": PM.sep_background_config_json(), "detection": PM.sep_detection_config_json(),
                             "acceptance": PM.ACCEPTANCE_POLICY}
        except Exception as exc:  # noqa: BLE001
            out["python"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    return out


def method_config_sha256(method_id: str) -> str:
    return config_sha256(method_config(method_id))


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class Case:
    """One image a method is run on (a frame, an injected copy, a dark trial)."""

    id: str
    image: np.ndarray
    masks: dict
    saturation_adu: float | None = None
    truth: dict | None = None


@dataclass
class MethodOutput:
    method: str
    status: str  # completed | failed | unavailable
    reason: str | None = None
    level_ok: bool = False
    noise_ok: bool = False
    back: object = None
    rms: object = None
    detections: list | None = None
    accepted: list | None = None
    frame_hfr: float | None = None
    level_summary: float | None = None
    rms_summary: float | None = None
    config_sha256: str | None = None
    reasons: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    #: Pixels the method itself declares unusable for noise (masked_local_v2 M_inv).
    invalid_mask: object = None

    def exclusion(self, shape):
        """Boolean map of the method's own invalid-noise pixels (all False if none)."""
        if self.invalid_mask is None:
            return np.zeros(shape, dtype=bool)
        return np.asarray(self.invalid_mask, dtype=bool)

    @property
    def usable(self) -> bool:
        return self.status == "completed" and self.level_ok and self.noise_ok

    @property
    def status_failure(self) -> bool:
        """Status failure for R12: ran, but no measured level and noise (or crashed)."""
        return self.status == "failed" or (self.status == "completed" and not (self.level_ok and self.noise_ok))

    def back_map(self, shape):
        if self.back is None:
            return None
        if np.isscalar(self.back) or np.ndim(self.back) == 0:
            return np.full(shape, float(self.back), dtype=np.float32)
        return self.back

    def rms_map(self, shape):
        if self.rms is None:
            return None
        if np.isscalar(self.rms) or np.ndim(self.rms) == 0:
            return np.full(shape, float(self.rms), dtype=np.float32)
        return self.rms


def unavailable(method, reason) -> MethodOutput:
    return MethodOutput(method=method, status="unavailable", reason=f"method_unavailable: {reason}")


def unusable_mask(image, masks) -> np.ndarray:
    bad = ~np.isfinite(image)
    for name in MASK_NAMES:
        if name in masks:
            bad |= masks[name].astype(bool)
    return bad


# --------------------------------------------------------------------------
# Python variants
# --------------------------------------------------------------------------


def _pm():
    from .. import python_methods as PM

    return PM


def _python_callable(method_id):
    """``(callable(image, masks, saturation_adu) -> (estimate, measurement), None)`` or ``(None, reason)``.

    ``python_sep`` is the harness reference (``python_sep_background`` +
    ``detect_and_measure``). A ``python_sep__nXgY`` variant runs through
    ``python_methods.run_python_method`` when the module lists it in
    ``PYTHON_SEP_VARIANTS``; otherwise through keyword arguments
    (``noise_policy=``, ``blend_test=``, ``regroup_deblended=``) when the
    functions accept them; otherwise it is unavailable."""
    from types import SimpleNamespace

    spec = METHOD_SPECS[method_id]
    try:
        PM = _pm()
    except Exception as exc:  # noqa: BLE001
        return None, f"ml.eval.bge.python_methods import failed: {type(exc).__name__}: {exc}"
    if method_id == "python_sep":
        def call(image, masks, saturation_adu=None):
            est = PM.python_sep_background(image, masks)
            return est, PM.detect_and_measure(image, masks, est)
        return call, None
    variants = getattr(PM, "PYTHON_SEP_VARIANTS", None)
    if isinstance(variants, dict) and method_id in variants and callable(getattr(PM, "run_python_method", None)):
        want = {k: spec[k] for k in ("noise_policy", "blend_test", "regroup_deblended")}
        have = {k: variants[method_id].get(k) for k in want}
        if have != want:
            return None, f"python_methods.PYTHON_SEP_VARIANTS[{method_id}] = {have} != spec {want}"

        def call(image, masks, saturation_adu=None):
            scenario = SimpleNamespace(image=image, masks=masks, metadata={"saturation_adu": saturation_adu})
            res = PM.run_python_method(method_id, scenario)
            return res["estimate"], res["measurement"]
        return call, None
    bg_kw = {"noise_policy": spec["noise_policy"]}
    det_kw = {"blend_test": spec["blend_test"], "regroup_deblended": spec["regroup_deblended"]}
    bg_params = inspect.signature(PM.python_sep_background).parameters
    det_params = inspect.signature(PM.detect_and_measure).parameters
    missing = [k for k in bg_kw if k not in bg_params] + [k for k in det_kw if k not in det_params]
    if missing:
        return None, (f"python_methods has no {method_id} (not in PYTHON_SEP_VARIANTS; python_sep_background/"
                      f"detect_and_measure lack {', '.join(missing)})")

    def call(image, masks, saturation_adu=None):
        est = PM.python_sep_background(image, masks, **bg_kw)
        meas = PM.detect_and_measure(image, masks, est, **det_kw)
        return est, meas

    return call, None


def _from_python(method_id, est, meas) -> MethodOutput:
    level_ok = est.level_status == "measured"
    noise_ok = est.noise_status == "measured"
    status = "failed" if est.status == "failed" else "completed"
    det_hash = meas.get("detection_config_sha256")
    gate_hash = meas.get("acceptance_policy_sha256") or meas.get("star_gates_config_sha256")
    out = MethodOutput(
        method=method_id, status=status, reason=est.detail, level_ok=level_ok, noise_ok=noise_ok,
        back=est.back if level_ok else None, rms=est.rms if noise_ok else None,
        detections=meas.get("candidates"), accepted=meas.get("accepted"), frame_hfr=meas.get("frame_hfr_median_px"),
        level_summary=est.level_summary, rms_summary=est.rms_summary,
        config_sha256=config_sha256({"background": est.config_sha256, "detection": det_hash, "gates": gate_hash}),
        reasons=list(est.reasons),
        diagnostics={"detection": meas.get("detection"), "rejections": meas.get("rejections"),
                     "refinement": meas.get("refinement") or getattr(est, "refinement", None),
                     "quality_status": meas.get("quality_status")},
        invalid_mask=getattr(getattr(est, "spatial_background", None), "invalid_mask", None),
    )
    return out


def run_python(method_id, case: Case) -> MethodOutput:
    call, why = _python_callable(method_id)
    if call is None:
        return unavailable(method_id, why)
    try:
        est, meas = call(case.image, case.masks, case.saturation_adu)
    except Exception as exc:  # noqa: BLE001 - MissingPrerequisite included: reported, never scored
        return MethodOutput(method=method_id, status="failed", reason=f"{type(exc).__name__}: {exc}")
    return _from_python(method_id, est, meas)


def run_oracle(case: Case) -> MethodOutput:
    mid = "oracle_true_background"
    if not case.truth or case.truth.get("background") is None or case.truth.get("noise") is None:
        return unavailable(mid, "no truth maps (real frames have none)")
    PM = _pm()
    cfg = {"method": mid, "source": "manifest truth maps (synthetic validation only)"}
    est = PM.BackgroundEstimate(method=mid, status="completed", level_status="measured", noise_status="measured",
                                reasons=[], config=cfg, config_sha256=config_sha256(cfg), spatial=True,
                                back=case.truth["background"], rms=case.truth["noise"],
                                level_summary=float(np.median(case.truth["background"])),
                                rms_summary=float(np.median(case.truth["noise"])), noise_source="truth")
    try:
        meas = PM.detect_and_measure(case.image, case.masks, est)
    except Exception as exc:  # noqa: BLE001
        return MethodOutput(method=mid, status="failed", reason=f"{type(exc).__name__}: {exc}")
    return _from_python(mid, est, meas)


# --------------------------------------------------------------------------
# Rust variants through the bridge
# --------------------------------------------------------------------------


def _find(obj, key):
    """First value of ``key`` anywhere in nested dicts/lists, or None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find(v, key)
            if found is not None:
                return found
    return None


def variant_echo(method_id, payload: dict) -> tuple:
    """``(ok, reason)``: does a bridge output carry the variant it was asked for?"""
    spec = METHOD_SPECS[canonical(method_id)]
    bg = payload.get("background") or {}
    if spec["backend"] == "sep":
        policy = (bg.get("config") or {}).get("noise_policy") or _find(bg, "noise_policy")
        if policy is None and spec["noise_policy"] != "frame_wide_v1":
            return False, f"output does not echo noise_policy {spec['noise_policy']} (daemon ignores the setting?)"
        if policy is not None and policy != spec["noise_policy"]:
            return False, f"output noise_policy {policy} != requested {spec['noise_policy']}"
    gate_area = {"quality": payload.get("quality"), "detection": payload.get("detection")}
    record = payload.get("star_gates") if isinstance(payload.get("star_gates"), dict) else None
    record = record or _find(gate_area, "star_gates_config")
    blend = (record or {}).get("blend_test") if isinstance(record, dict) else None
    blend = blend or _find(gate_area, "blend_test")
    gate_hash = payload.get("star_gates_config_sha256") or _find(gate_area, "star_gates_config_sha256")
    if blend is None and gate_hash is not None:
        blend = next((b for b in BLEND_TESTS.values() if gate_hash in expected_gate_hashes(b)), "unrecognised-hash")
    if blend is None and spec["blend_test"] != "peak_count_v1":
        return False, f"output does not echo blend_test {spec['blend_test']} (daemon ignores star_gates?)"
    if blend is not None and blend != spec["blend_test"]:
        return False, f"output blend_test {blend} != requested {spec['blend_test']}"
    return True, None


def parse_rust_output(method_id, out: dict) -> MethodOutput:
    """A ``bridge.read_output`` result as a MethodOutput (maps may be memmaps)."""
    if out.get("status") != "completed":
        return MethodOutput(method=method_id, status="failed", reason=out.get("reason") or out.get("status"))
    j = out["json"]
    ok, why = variant_echo(method_id, j)
    if not ok:
        return unavailable(method_id, why)
    bgr = j.get("background") or {}
    level_ok = bgr.get("level_status") == "measured"
    noise_ok = bgr.get("noise_status") == "measured"
    spatial = bgr.get("method") == "sep_spatial_mesh"
    if spatial:
        back = out.get("back") if level_ok else None
        rms = out.get("rms") if noise_ok else None
        if level_ok and back is None:
            return MethodOutput(method=method_id, status="failed", reason="spatial field but no .back.f32 emitted")
        if noise_ok and rms is None:
            return MethodOutput(method=method_id, status="failed", reason="spatial noise but no .rms.f32 emitted")
    else:
        back = float(bgr["level"]) if level_ok and bgr.get("level") is not None else None
        rms = float(bgr["rms"]) if noise_ok and bgr.get("rms") is not None else None
        level_ok, noise_ok = back is not None, rms is not None
    quality = j.get("quality") or {}
    detections = j.get("catalog") if j.get("catalog") is not None else j.get("candidates")
    gate_hash = _find({"q": quality, "d": j.get("detection")}, "star_gates_config_sha256")
    return MethodOutput(
        method=method_id, status="completed", level_ok=level_ok, noise_ok=noise_ok, back=back, rms=rms,
        detections=detections, accepted=quality.get("used_stars"), frame_hfr=quality.get("hfr_px"),
        level_summary=bgr.get("level"), rms_summary=bgr.get("rms"),
        config_sha256=config_sha256({"background": bgr.get("config_sha256"), "gates": gate_hash,
                                     "method": method_id}),
        reasons=list(bgr.get("reasons") or []),
        diagnostics={"quality_status": quality.get("status"), "star_rejections": quality.get("star_rejections"),
                     "refinement": bgr.get("refinement") or j.get("refinement"),
                     "backend_error": bgr.get("backend_error")},
        invalid_mask=out.get("invalid"),
    )


class RustBridgeRunner:
    """Runs Rust variants on a batch of cases via ``bridge.run_bridge``.

    Inputs are written once per batch and shared by every variant's manifest.
    Outputs are read lazily (``np.memmap`` maps) and deleted by
    :meth:`cleanup` unless ``keep_io``."""

    def __init__(self, repo_root, work_dir, cargo_target_dir, keep_io=False, run_bridge=None, read_output=None,
                 jobs: int = 4):
        from .. import bridge

        self.bridge = bridge
        self.repo_root = Path(repo_root)
        self.work_dir = Path(work_dir)
        self.cargo_target_dir = Path(cargo_target_dir)
        self.keep_io = keep_io
        self.run_bridge = run_bridge or bridge.run_bridge
        self.read_output = read_output or bridge.read_output
        self.jobs = jobs
        self.batches = 0
        self.log = []
        self._roots = []

    def _write_inputs(self, root: Path, cases) -> list:
        (root / "inputs").mkdir(parents=True, exist_ok=True)
        entries = []
        for c in cases:
            if not self.bridge.valid_case_id(c.id):
                raise ValueError(f"case id {c.id!r} is not bridge safe")
            rel = f"inputs/{c.id}.f32"
            np.ascontiguousarray(c.image, dtype="<f4").tofile(root / rel)
            masks = {}
            for name in MASK_NAMES:
                if name in c.masks and np.any(c.masks[name]):
                    mrel = f"inputs/{c.id}.{name}.u8"
                    np.ascontiguousarray(c.masks[name], dtype=np.uint8).tofile(root / mrel)
                    masks[name] = mrel
            entries.append({"id": c.id, "width": int(c.image.shape[1]), "height": int(c.image.shape[0]),
                            "pixels": rel, "masks": masks, "saturation_adu": c.saturation_adu, "units": "adu",
                            "extras": {"origin": "bge-real-qualification"}})
        return entries

    def run(self, cases, method_ids) -> dict:
        self.batches += 1
        root = self.work_dir / f"batch{self.batches:05d}"
        root.mkdir(parents=True, exist_ok=True)
        entries = self._write_inputs(root, cases)
        outputs = {}
        for mid in method_ids:
            backend, settings, gates = bridge_overlays(mid)
            manifest = {"schema": self.bridge.BRIDGE_SCHEMA, "cases": entries, "backends": [backend],
                        "settings": settings, "star_gates": gates, "write_maps": True,
                        "extras": {"qualification_method": mid}}
            mpath = root / f"{mid}.manifest.json"
            mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            out_dir = root / f"out_{mid}"
            info = self.run_bridge(mpath, out_dir, self.repo_root, self.cargo_target_dir, root / f"{mid}.log",
                                   jobs=self.jobs)
            self.log.append({"batch": self.batches, "method": mid, "status": info.get("status"),
                             "reason": info.get("reason"), "revision": info.get("revision")})
            for c in cases:
                if info.get("status") != "completed":
                    reason = info.get("reason") or info.get("status")
                    outputs[(c.id, mid)] = (unavailable(mid, reason) if info.get("status") == "skipped"
                                            else MethodOutput(method=mid, status="failed", reason=reason))
                    continue
                raw = self._read_lazy(out_dir, c, backend)
                outputs[(c.id, mid)] = parse_rust_output(mid, raw)
        self._roots.append(root)
        return outputs

    def _read_lazy(self, out_dir, case, backend) -> dict:
        stem = self.bridge.output_stem(out_dir, case.id, backend)
        h, w = case.image.shape
        maps = {}
        for key, suffix in (("back", ".back.f32"), ("rms", ".rms.f32")):
            p = Path(str(stem) + suffix)
            if p.exists() and p.stat().st_size == 4 * w * h:
                maps[key] = np.memmap(p, dtype="<f4", mode="r", shape=(h, w))
        out = self.read_output(out_dir, case.id, backend, w, h) if not maps else self._json_only(stem, case, backend)
        out.update({k: v for k, v in maps.items()})
        inv = Path(str(stem) + ".invalid.u8")  # masked_local_v2 locally invalid pixels, when any
        if inv.exists() and inv.stat().st_size == w * h:
            out["invalid"] = np.memmap(inv, dtype=np.uint8, mode="r", shape=(h, w))
        return out

    def _json_only(self, stem, case, backend) -> dict:
        path = stem.with_suffix(".json")
        if not path.exists():
            return {"status": "missing", "reason": f"no bridge output {path.name}"}
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            return {"status": "failed", "reason": f"unparseable bridge output: {exc}"}
        if payload.get("case") != case.id or payload.get("backend") != backend:
            return {"status": "failed", "reason": "bridge output case/backend does not match its file name"}
        return {"status": "failed" if payload.get("error") else "completed", "json": payload,
                "reason": payload.get("error"), "back": None, "rms": None}

    def cleanup(self):
        """Remove every batch written since the last cleanup (unless keep_io)."""
        if not self.keep_io:
            for root in self._roots:
                shutil.rmtree(root, ignore_errors=True)
        self._roots = []


class MethodSuite:
    """Runs a fixed list of variants on cases. Rust variants go through one
    bridge batch per call; Python variants run in process."""

    def __init__(self, method_ids, rust_runner: RustBridgeRunner | None = None, python_runner=None):
        self.method_ids = [canonical(m) for m in method_ids]
        self.rust_runner = rust_runner
        self.python_runner = python_runner or run_python
        self._unavailable = {}
        for mid in self.method_ids:
            spec = METHOD_SPECS[mid]
            if spec["kind"] == "rust" and rust_runner is None:
                self._unavailable[mid] = "Rust variants need --rust (bridge runner not configured)"
            elif spec["kind"] == "python":
                _call, why = _python_callable(mid)
                if why:
                    self._unavailable[mid] = why

    def availability(self) -> dict:
        return {m: ("available" if m not in self._unavailable else f"method_unavailable: {self._unavailable[m]}")
                for m in self.method_ids}

    def run(self, cases) -> dict:
        outputs = {}
        rust = [m for m in self.method_ids if METHOD_SPECS[m]["kind"] == "rust" and m not in self._unavailable]
        if rust:
            outputs.update(self.rust_runner.run(cases, rust))
        for mid in self.method_ids:
            spec = METHOD_SPECS[mid]
            for c in cases:
                if mid in self._unavailable:
                    outputs[(c.id, mid)] = unavailable(mid, self._unavailable[mid])
                elif spec["kind"] == "python":
                    outputs[(c.id, mid)] = self.python_runner(mid, c)
                elif spec["kind"] == "oracle":
                    outputs[(c.id, mid)] = run_oracle(c)
        return outputs

    def cleanup(self):
        if self.rust_runner is not None:
            self.rust_runner.cleanup()
