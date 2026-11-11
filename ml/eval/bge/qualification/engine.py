"""Per-night evaluation: run every method on every frame and derive the night metrics.

Order within a night: lights (repeat groups together, each in time order, then
ungrouped lights in time order), then darks and bias. Per light: the methods
run on the frame; the reference method's result gives the PSF and the
injection plan; the methods run on the injected copies; then per method the
pull, negated-residual detections, injection scores, repeat-group state and
focus samples are recorded. Darks/bias give s0 samples (pull robust std) and
dark + simulated-sky trials. Sweep fits and repeatability close the night.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import dark_injection as DI
from . import injection as INJ
from . import realmetrics as RM
from .manifest import load_frame, load_truth, representation
from .masks import hot_pixel_mask, source_mask
from .methods import METHOD_SPECS, REFERENCE_METHOD, Case, run_python, unusable_mask


@dataclass
class EngineConfig:
    injection: bool = True
    injection_rounds: int = 2
    injections_per_round: int = INJ.DEFAULT_INJECTIONS_PER_ROUND
    bright_rounds: int = 1
    inject_sweep_frames: bool = False
    dark_trials_per_night: int = 4
    seed: int = 20261111
    max_frames_per_night: int | None = None
    verify_sha: bool = True


def _case_id(frame_id, suffix) -> str:
    return "q" + hashlib.sha1(str(frame_id).encode()).hexdigest()[:12] + "_" + suffix


def _seed(cfg, frame, *extra):
    digest = int(str(frame.get("sha256") or hashlib.sha256(frame["id"].encode()).hexdigest())[:12], 16)
    return np.random.default_rng(np.random.SeedSequence([cfg.seed, digest, *extra]))


def _sorted_lights(lights):
    grouped, single = {}, []
    for f in lights:
        (grouped.setdefault(f["repeat_group"], []) if f.get("repeat_group") else single).append(f)
    order = []
    for g in sorted(grouped, key=lambda g: min(x.get("timestamp_utc") or "" for x in grouped[g])):
        order += sorted(grouped[g], key=lambda x: (x.get("timestamp_utc") or "", x["id"]))
    return order + sorted(single, key=lambda x: (x.get("timestamp_utc") or "", x["id"]))


class _Acc:
    """Accumulator of one method's frame-level results within a night."""

    def __init__(self):
        self.lights = 0
        self.failures = 0
        self.unavailable = []
        self.pulls, self.diff_pulls, self.negated = [], [], []
        self.injection_records, self.rounds = [], 0
        self.dark_trials = {}
        self.calibration_pulls = []
        self.groups = {}
        self.sweeps = {}

    def availability(self):
        if self.lights == 0 and self.unavailable and not (self.pulls or self.calibration_pulls):
            return self.unavailable[0]
        return "available"


class Engine:
    def __init__(self, manifest, manifest_dir, suite, config: EngineConfig | None = None, log=None):
        self.manifest = manifest
        self.base = manifest_dir
        self.suite = suite
        self.cfg = config or EngineConfig()
        self.log = log or (lambda msg: None)
        self.frame_rows = []

    # ------------------------------------------------------------------
    def _load(self, frame):
        image, masks = load_frame(frame, self.base, verify=self.cfg.verify_sha)
        return image, masks, load_truth(frame, self.base)

    def _case(self, frame, suffix, image, masks, truth=None):
        return Case(id=_case_id(frame["id"], suffix), image=image, masks=masks,
                    saturation_adu=frame.get("saturation_adu"), truth=truth)

    def _reference(self, outputs, case):
        out = outputs.get((case.id, REFERENCE_METHOD))
        if out is None or out.status == "unavailable":
            out = run_python(REFERENCE_METHOD, case)
        return out

    # ------------------------------------------------------------------
    def evaluate_night(self, night, frames) -> dict:
        lights = [f for f in frames if f["frame_type"] == "light"]
        darks = [f for f in frames if f["frame_type"] == "dark"]
        bias = [f for f in frames if f["frame_type"] == "bias"]
        if self.cfg.max_frames_per_night:
            lights = lights[: self.cfg.max_frames_per_night]
        accs = {m: _Acc() for m in self.suite.method_ids}
        hot_cache, hot_info = {}, {}

        def hot_for(shape):
            if shape not in hot_cache:
                mask, info = hot_pixel_mask((load_frame(d, self.base, self.cfg.verify_sha) for d in darks), shape) \
                    if darks else (None, {"n_darks": 0, "reason": "no darks in this night"})
                hot_cache[shape], hot_info[f"{shape[1]}x{shape[0]}"] = mask, info
            return hot_cache[shape]

        ref_levels, psfs = [], []
        prev = {}  # repeat_group -> state of the previous frame
        group_next = {}
        ordered = _sorted_lights(lights)
        for i, f in enumerate(ordered):
            g = f.get("repeat_group")
            group_next[f["id"]] = bool(g) and i + 1 < len(ordered) and ordered[i + 1].get("repeat_group") == g
        for f in ordered:
            self.log(f"night {night}: light {f['id']}")
            image, masks, truth = self._load(f)
            hot = hot_for(image.shape)
            bad = unusable_mask(image, masks) | (hot if hot is not None else False)
            orig = self._case(f, "o", image, masks, truth)
            outputs = self.suite.run([orig])
            ref = self._reference(outputs, orig)
            if ref.usable and ref.level_summary is not None:
                ref_levels.append(ref.level_summary)
            sweep = bool(f.get("focus_run_id"))
            inj = None
            if self.cfg.injection and (not sweep or self.cfg.inject_sweep_frames):
                inj = self._injection_cases(f, image, masks, truth, ref, hot)
                if inj.get("psf", {}).get("status") == "completed":
                    psfs.append(inj["psf"]["psf"])
                if inj.get("cases"):
                    outputs.update(self.suite.run(inj["cases"]))
            g = f.get("repeat_group")
            state = {"image": image, "methods": {}} if group_next.get(f["id"]) else None
            for m in self.suite.method_ids:
                acc, out = accs[m], outputs[(orig.id, m)]
                row = {"frame": f["id"], "night": night, "method": m, "frame_type": "light", "status": out.status,
                       "reason": out.reason, "status_failure": out.status_failure, "level": out.level_summary,
                       "rms": out.rms_summary, "frame_hfr_px": out.frame_hfr,
                       "n_detections": None if out.detections is None else len(out.detections),
                       "n_accepted": None if out.accepted is None else len(out.accepted),
                       "config_sha256": out.config_sha256, "representation": representation(f)}
                if out.status == "unavailable":
                    acc.unavailable.append(out.reason)
                    self.frame_rows.append(row)
                    continue
                acc.lights += 1
                acc.failures += int(out.status_failure)
                if g:
                    rec = acc.groups.setdefault(f"{night}/{g}", {"hfr": [], "level": [], "rms": []})
                    rec["hfr"].append(out.frame_hfr)
                    rec["level"].append(out.level_summary if out.usable else None)
                    rec["rms"].append(out.rms_summary if out.usable else None)
                if sweep:
                    s = acc.sweeps.setdefault(f["focus_run_id"], {"positions": [], "hfr": [], "excluded": 0,
                                                                  "focal_ratio": [], "stepsize_um": []})
                    flags = f.get("flags") or []
                    if ("moved_during_exposure" in flags or "autofocus_verification_frame" in flags
                            or f.get("focus_position") is None):
                        s["excluded"] += 1
                    else:
                        s["positions"].append(float(f["focus_position"]))
                        s["hfr"].append(out.frame_hfr)
                    if f.get("focal_ratio"):
                        s["focal_ratio"].append(f["focal_ratio"])
                    if f.get("focuser_stepsize_um"):
                        s["stepsize_um"].append(f["focuser_stepsize_um"])
                if out.usable:
                    back, rms = out.back_map(image.shape), out.rms_map(image.shape)
                    own = out.exclusion(image.shape)
                    excl = bad | own | source_mask(out.detections, image.shape)
                    pull = RM.frame_pull(image, back, rms, excl)
                    row["pull"] = {k: pull.get(k) for k in ("status", "n_pixels", "clipped_mean", "robust_std",
                                                            "frac_abs_gt3", "frac_abs_gt5")}
                    if pull["status"] == "completed":
                        acc.pulls.append(pull)
                    try:
                        neg = RM.negated_detections(image, back, rms, bad | own, out.detections, out.frame_hfr)
                    except Exception as exc:  # noqa: BLE001
                        neg = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                    row["negated"] = {k: neg.get(k) for k in ("status", "n_detections", "rate_per_mpx",
                                                              "fraction_near_bright")}
                    if neg["status"] == "completed":
                        acc.negated.append(neg)
                    p = prev.get(g) if g else None
                    if p is not None and m in p["methods"]:
                        r1, x1 = p["methods"][m]
                        dp = RM.difference_pull(p["image"], r1, x1, image, rms, excl)
                        row["difference_pull"] = {k: dp.get(k) for k in ("status", "n_pixels", "robust_std")}
                        if dp["status"] == "completed":
                            acc.diff_pulls.append(dp)
                    if state is not None:
                        state["methods"][m] = (np.array(rms, dtype=np.float32) if np.ndim(rms) else float(rms), excl)
                if inj and inj.get("cases"):
                    acc.rounds += len(inj["cases"])
                    for case, stars in zip(inj["cases"], inj["stars"]):
                        acc.injection_records += INJ.score_round(stars, case.image, outputs[(case.id, m)], out)
                self.frame_rows.append(row)
            if g:
                prev = {g: state} if state is not None else {}
            self.suite.cleanup()
        # Darks and bias: s0 samples and dark + simulated-sky trials.
        sky_level = float(np.median(ref_levels)) if ref_levels else None
        psf = DI.DEFAULT_PSF
        moffats = [p for p in psfs if p.get("model") == "moffat"]
        if moffats:
            psf = {"model": "moffat", "fwhm_px": float(np.median([p["fwhm_px"] for p in moffats])),
                   "beta": float(np.median([p["beta"] for p in moffats]))}
        self._calibration_frames(night, darks, bias, accs, hot_for, sky_level, psf)
        return self._summarise(night, frames, lights, accs, hot_info, sky_level, psf)

    def calibrate_night(self, night, frames) -> dict:
        """s0 samples only (darks and bias of a development night, no trials)."""
        darks = [f for f in frames if f["frame_type"] == "dark"]
        bias = [f for f in frames if f["frame_type"] == "bias"]
        accs = {m: _Acc() for m in self.suite.method_ids}
        hot_cache = {}

        def hot_for(shape):
            if shape not in hot_cache:
                hot_cache[shape] = hot_pixel_mask((load_frame(d, self.base, self.cfg.verify_sha) for d in darks),
                                                  shape)[0] if darks else None
            return hot_cache[shape]

        trials, self.cfg.dark_trials_per_night = self.cfg.dark_trials_per_night, 0
        try:
            self._calibration_frames(night, darks, bias, accs, hot_for, None, DI.DEFAULT_PSF)
        finally:
            self.cfg.dark_trials_per_night = trials
        return {"methods": {m: {"calibration_pulls": a.calibration_pulls} for m, a in accs.items()}}

    # ------------------------------------------------------------------
    def _injection_cases(self, f, image, masks, truth, ref, hot):
        psf = INJ.estimate_psf(image, masks, ref)
        out = {"psf": psf, "cases": [], "stars": []}
        if psf["status"] != "completed":
            return out
        try:
            distance, n_ref = INJ.avoidance_distance(image, masks, ref, hot)
        except Exception as exc:  # noqa: BLE001
            out["psf"] = dict(psf, status="not_evaluable", reason=f"avoidance extraction failed: {exc}")
            return out
        gain = f.get("gain_e_per_adu")
        bright = INJ.bright_flux_range(ref, psf["psf"], f.get("saturation_adu"))
        out["bright_flux_range"] = bright
        plan = [(r, None, self.cfg.injections_per_round) for r in range(1, self.cfg.injection_rounds + 1)]
        if bright is not None:
            plan += [(100 + r, bright, INJ.BRIGHT_ROUND_STARS) for r in range(1, self.cfg.bright_rounds + 1)]
        for r, flux_range, n in plan:
            rng = _seed(self.cfg, f, 1, r)
            stars, stars_img, info = INJ.plan_injections(rng, image, ref, psf["psf"], distance, n, gain,
                                                         f.get("saturation_adu"), flux_range=flux_range)
            if not stars:
                continue
            INJ.true_aperture_flux(stars_img, stars)
            img, msk, _noise = INJ.injected_image(image, masks, stars_img, rng, gain, f.get("saturation_adu"))
            out["cases"].append(self._case(f, f"i{r}", img, msk, truth))
            out["stars"].append(stars)
        return out

    def _calibration_frames(self, night, darks, bias, accs, hot_for, sky_level, psf):
        trials_done = 0
        previous = None
        for f in sorted(darks + bias, key=lambda x: (x["frame_type"], x.get("timestamp_utc") or "", x["id"])):
            image, masks, truth = self._load(f)
            hot = hot_for(image.shape)
            bad = unusable_mask(image, masks) | (hot if hot is not None else False)
            cases = [self._case(f, "d", image, masks, truth)]
            trial_info = []
            if f["frame_type"] == "dark" and trials_done < self.cfg.dark_trials_per_night:
                other = previous if previous is not None and previous.shape == image.shape else None
                sigma, how = DI.sigma_dark(image, masks, other)
                if np.isfinite(sigma) and sigma > 0:
                    level = (sky_level - float(np.median(image[~bad]))) if sky_level is not None else None
                    if level is None or not level > 0:
                        level = DI.SKY_SIGMA_MULTIPLE * sigma
                    rng = _seed(self.cfg, f, 2)
                    for name, sky in DI.sky_models(image.shape[1], image.shape[0], level).items():
                        timg, ttruth = DI.build_trial(image, masks, sky, psf, sigma, rng, f.get("gain_e_per_adu"))
                        tcase = self._case(f, f"t{name[:3]}", timg, masks, None)
                        cases.append(tcase)
                        trial_info.append((name, tcase, sky, ttruth, sigma, how))
                    trials_done += 1
            if f["frame_type"] == "dark":
                previous = image
            outputs = self.suite.run(cases)
            for m in self.suite.method_ids:
                out = outputs[(cases[0].id, m)]
                row = {"frame": f["id"], "night": night, "method": m, "frame_type": f["frame_type"],
                       "status": out.status, "reason": out.reason, "level": out.level_summary, "rms": out.rms_summary}
                if out.status == "unavailable":
                    accs[m].unavailable.append(out.reason)
                    self.frame_rows.append(row)
                    continue
                if out.usable:
                    excl = bad | out.exclusion(image.shape) | source_mask(out.detections, image.shape)
                    pull = RM.frame_pull(image, out.back_map(image.shape), out.rms_map(image.shape), excl)
                    if pull["status"] == "completed":
                        accs[m].calibration_pulls.append({"frame": f["id"], "frame_type": f["frame_type"],
                                                          "representation": representation(f),
                                                          "robust_std": pull["robust_std"],
                                                          "phases": pull["robust_std_by_cfa_phase_00_01_10_11"]})
                        row["pull_robust_std"] = pull["robust_std"]
                for name, tcase, sky, ttruth, sigma, how in trial_info:
                    usable = ~(unusable_mask(tcase.image, tcase.masks) | (hot if hot is not None else False))
                    score = DI.score_trial(out, outputs[(tcase.id, m)], sky, ttruth, image.shape, usable, sigma)
                    score.update({"sky_model": name, "sigma_d_source": how, "noise": ttruth["noise"],
                                  "n_stars": ttruth["n_stars"]})
                    accs[m].dark_trials[f"{f['id']}/{name}"] = score
                self.frame_rows.append(row)
            self.suite.cleanup()

    # ------------------------------------------------------------------
    def _summarise(self, night, frames, lights, accs, hot_info, sky_level, psf) -> dict:
        med = RM.finite_median
        out = {}
        for m, acc in accs.items():
            avail = acc.availability()
            metrics = {"availability": avail, "method_kind": METHOD_SPECS[m]["kind"], "n_lights": acc.lights,
                       "n_status_failures": acc.failures,
                       "status_failure_rate": (acc.failures / acc.lights) if acc.lights else None}
            metrics["pull"] = {"n_frames": len(acc.pulls),
                               "median_abs_clipped_mean": med([abs(p["clipped_mean"]) for p in acc.pulls
                                                               if p["clipped_mean"] is not None]),
                               "median_robust_std": med([p["robust_std"] for p in acc.pulls]),
                               "median_frac_abs_gt3": med([p["frac_abs_gt3"] for p in acc.pulls]),
                               "median_frac_abs_gt5": med([p["frac_abs_gt5"] for p in acc.pulls]),
                               "gaussian_reference": RM.TAIL_REFERENCE}
            metrics["diff_pull"] = {"n_pairs": len(acc.diff_pulls),
                                    "median_robust_std": med([p["robust_std"] for p in acc.diff_pulls])}
            n_det = sum(x["n_detections"] for x in acc.negated)
            mpx = sum(x["usable_mpx"] for x in acc.negated)
            near = sum(x["n_near_bright"] for x in acc.negated)
            metrics["negated"] = {"n_frames": len(acc.negated), "n_detections": n_det, "usable_mpx": mpx,
                                  "rate_per_mpx": (n_det / mpx) if mpx > 0 else None, "n_near_bright": near,
                                  "fraction_near_bright": (near / n_det) if n_det else None}
            metrics["injection"] = dict(INJ.aggregate(acc.injection_records), n_rounds=acc.rounds)
            trials = acc.dark_trials
            done = [t for t in trials.values() if t.get("status") == "completed"]
            metrics["dark_injection"] = {
                "n_trials": len(done), "trials": trials,
                "median_abs_bias_sigma": med([abs(t["bias_sigma"]) for t in done if t.get("bias_sigma") is not None]),
                "median_rmse_adu": med([t["background"]["rmse_adu"] for t in done]),
                "median_rmse_sigma": med([t["rmse_sigma"] for t in done if t.get("rmse_sigma") is not None])}
            groups = {}
            for gid, rec in acc.groups.items():
                groups[gid] = {"n_frames": len(rec["hfr"]), "hfr_cv": RM.cv(rec["hfr"]), "level_cv": RM.cv(rec["level"]),
                               "rms_cv": RM.cv(rec["rms"])}
            metrics["repeatability"] = {
                "n_groups": sum(1 for g in groups.values() if g["hfr_cv"] is not None), "groups": groups,
                "median_hfr_cv": med([g["hfr_cv"] for g in groups.values()]),
                "median_level_cv": med([g["level_cv"] for g in groups.values()]),
                "median_rms_cv": med([g["rms_cv"] for g in groups.values()])}
            sweeps, cfzs, n_frames, n_hfr = {}, [], 0, 0
            for run, s in acc.sweeps.items():
                fit = RM.fit_sweep(s["positions"], s["hfr"]) if s["positions"] else {"status": "not_evaluable",
                                                                                     "reason": "no usable sweep frame"}
                cfz = RM.cfz_steps(med(s["focal_ratio"]), med(s["stepsize_um"]))
                fit.update({"cfz_steps": cfz, "frames_excluded": s["excluded"]})
                sweeps[run] = fit
                if cfz:
                    cfzs.append(cfz)
                n_frames += len(s["positions"])
                n_hfr += sum(1 for h in s["hfr"] if h is not None)
            p0s = [f["p0"] for f in sweeps.values() if f.get("status") == "completed"]
            metrics["focus"] = {"n_sweeps": len(sweeps), "n_fits": len(p0s), "sweeps": sweeps,
                                "p0_std": float(np.std(p0s, ddof=1)) if len(p0s) >= 2 else None,
                                "cfz_steps": med(cfzs),
                                "coverage": {"n_frames": n_frames, "n_with_hfr": n_hfr,
                                             "fraction": (n_hfr / n_frames) if n_frames else None}}
            metrics["calibration_pulls"] = acc.calibration_pulls
            out[m] = metrics
        types = {}
        for f in frames:
            types[f["frame_type"]] = types.get(f["frame_type"], 0) + 1
        return {"methods": out,
                "inputs": {"frame_types": types, "lights_evaluated": len(lights),
                           "representations": sorted({representation(f) for f in frames}),
                           "hot_pixel_masks": hot_info, "sky_level_for_dark_trials_adu": sky_level,
                           "dark_trial_psf": psf}}


def s0_from_calibration(night_results: dict, methods) -> dict:
    """s0 per method from development nights' dark/bias pulls (median; >= 2 frames)."""
    out = {}
    for m in methods:
        samples = []
        for res in night_results.values():
            samples += [p for p in res["methods"].get(m, {}).get("calibration_pulls", []) if p["robust_std"]]
        reps = sorted({p["representation"] for p in samples})
        if len(samples) >= 2:
            out[m] = {"value": float(np.median([p["robust_std"] for p in samples])), "calibrated": True,
                      "n_frames": len(samples), "representations": reps,
                      "by_frame_type": {t: RM.finite_median([p["robust_std"] for p in samples if p["frame_type"] == t])
                                        for t in ("dark", "bias")}}
        else:
            out[m] = {"value": 1.0, "calibrated": False, "n_frames": len(samples), "representations": reps,
                      "flag": "s0_uncalibrated: fewer than 2 development dark/bias frames"}
    return out
