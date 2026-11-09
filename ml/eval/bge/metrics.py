"""Metrics against simulated truth. Definitions and units are in README.md.

All functions are pure numpy and take plain arrays/lists so they can be
checked on toy inputs. Pixel metrics are evaluated over *usable* pixels
(finite and outside every detector mask); a scalar background is evaluated
as a constant map.
"""

from __future__ import annotations

import math

import numpy as np

#: Block size (px) for the large-scale residual component.
RESIDUAL_BLOCK_PX = 32
#: A block counts toward the large-scale residual if >= this fraction is usable.
RESIDUAL_BLOCK_MIN_FRAC = 0.5
#: "Far from sources" = farther than this from any pixel where the noiseless
#: source image exceeds ``FOOTPRINT_SIGMA`` x the truth sky noise.
FAR_FROM_SOURCES_PX = 16
FOOTPRINT_SIGMA = 1.0
#: Matching radius to truth: max(MATCH_MIN_PX, MATCH_HFR_FRACTION * true HFR).
MATCH_MIN_PX = 2.0
MATCH_HFR_FRACTION = 0.5
#: Extended sources use a fixed radius (their barycentre is ill-defined).
MATCH_EXTENDED_PX = 10.0
#: "Bright" truth stars: noiseless peak >= this many x truth sky noise.
BRIGHT_PEAK_SNR = 20.0
#: HFR outlier: |relative error| above this.
HFR_OUTLIER_REL = 0.2


def _median(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.median(a)) if a.size else None


def _f(v):
    return None if v is None or not np.isfinite(v) else float(v)


# --------------------------------------------------------------------------
# Background and noise maps
# --------------------------------------------------------------------------


def source_masks(source_image, noise_std_sky, usable, far_px=FAR_FROM_SOURCES_PX, sigma=FOOTPRINT_SIGMA):
    """(under_sources, far_from_sources) boolean maps."""
    from scipy import ndimage

    if source_image is None or noise_std_sky is None:
        return None, None
    under = source_image > sigma * np.maximum(noise_std_sky, 1e-12)
    if not under.any():
        return under, usable.copy()
    dist = ndimage.distance_transform_edt(~under)
    return under, dist > far_px


def background_map_metrics(b_hat, b_true, usable, under=None, far=None, star_xy=None,
                           block=RESIDUAL_BLOCK_PX) -> dict:
    """Error e = B_hat - B_true over usable pixels (ADU)."""
    b_hat = np.broadcast_to(np.asarray(b_hat, dtype=np.float64), b_true.shape)
    e_map = b_hat - b_true.astype(np.float64)
    e = e_map[usable]
    if e.size == 0:
        return {"n_pixels": 0}
    med = float(np.median(e))
    out = {
        "n_pixels": int(e.size),
        "bias_median_adu": med,
        "bias_mean_adu": float(np.mean(e)),
        "mae_adu": float(np.mean(np.abs(e))),
        "rmse_adu": float(np.sqrt(np.mean(e * e))),
        "p99_abs_error_adu": float(np.percentile(np.abs(e), 99)),
        "residual_rms_about_median_adu": float(np.sqrt(np.mean((e - med) ** 2))),
    }
    h, w = b_true.shape
    blocks = []
    for y0 in range(0, h - block + 1, block):
        for x0 in range(0, w - block + 1, block):
            u = usable[y0:y0 + block, x0:x0 + block]
            if u.mean() >= RESIDUAL_BLOCK_MIN_FRAC:
                blocks.append(float(np.mean(e_map[y0:y0 + block, x0:x0 + block][u]) - med))
    out["large_scale_rms_adu"] = float(np.sqrt(np.mean(np.square(blocks)))) if blocks else None
    out["large_scale_blocks"] = len(blocks)
    yy, xx = np.nonzero(usable)
    if e.size >= 3:
        design = np.column_stack([np.ones(e.size), xx - (w - 1) / 2, yy - (h - 1) / 2])
        coef, *_ = np.linalg.lstsq(design, e_map[yy, xx], rcond=None)
        out["residual_slope_x_adu_per_px"] = float(coef[1])
        out["residual_slope_y_adu_per_px"] = float(coef[2])
    for name, region in (("under_sources", under), ("far_from_sources", far)):
        if region is not None:
            sel = region & usable
            out[f"bias_median_{name}_adu"] = _median(e_map[sel]) if sel.any() else None
            out[f"n_pixels_{name}"] = int(sel.sum())
    if star_xy:
        vals = []
        for x, y in star_xy:
            i, j = int(round(x)), int(round(y))
            if 0 <= j < h and 0 <= i < w and usable[j, i]:
                vals.append(e_map[j, i])
        if vals:
            out["bias_median_at_star_centres_adu"] = float(np.median(vals))
            out["max_abs_error_at_star_centres_adu"] = float(np.max(np.abs(vals)))
            out["n_star_centres"] = len(vals)
    return out


def noise_map_metrics(r_hat, r_true, usable) -> dict:
    """Relative RMS error R_hat/R_true - 1 over usable pixels with R_true > 0."""
    r_hat = np.broadcast_to(np.asarray(r_hat, dtype=np.float64), r_true.shape)
    sel = usable & (r_true > 0)
    if not sel.any():
        return {"n_pixels": 0}
    ratio = r_hat[sel] / r_true[sel].astype(np.float64)
    return {
        "n_pixels": int(sel.sum()),
        "rel_error_median": float(np.median(ratio - 1.0)),
        "rel_error_mean": float(np.mean(ratio - 1.0)),
        "ratio_p05": float(np.percentile(ratio, 5)),
        "ratio_p95": float(np.percentile(ratio, 95)),
        "fraction_nonpositive": float(np.mean(r_hat[sel] <= 0)),
    }


# --------------------------------------------------------------------------
# Catalog matching, completeness, false detections, blends
# --------------------------------------------------------------------------


def truth_match_radius(source) -> float:
    if source.get("kind") == "extended":
        return MATCH_EXTENDED_PX
    hfr = source.get("hfr_true_px") or 0.0
    return max(MATCH_MIN_PX, MATCH_HFR_FRACTION * hfr)


def match_greedy(det_xy, truth_xy, radii):
    """One-to-one matching, closest pairs first, pair kept if d <= radius of
    the truth object. Returns list of (det_index, truth_index, distance)."""
    det_xy = np.asarray(det_xy, dtype=np.float64).reshape(-1, 2)
    truth_xy = np.asarray(truth_xy, dtype=np.float64).reshape(-1, 2)
    if det_xy.size == 0 or truth_xy.size == 0:
        return []
    d = np.hypot(det_xy[:, None, 0] - truth_xy[None, :, 0], det_xy[:, None, 1] - truth_xy[None, :, 1])
    ok = d <= np.asarray(radii, dtype=np.float64)[None, :]
    pairs = sorted(zip(*np.nonzero(ok)), key=lambda ij: d[ij])
    used_d, used_t, out = set(), set(), []
    for i, j in pairs:
        if i in used_d or j in used_t:
            continue
        used_d.add(i)
        used_t.add(j)
        out.append((int(i), int(j), float(d[i, j])))
    return out


def detection_metrics(candidates, truth, width, height, source_image=None, noise_std_sky=None) -> dict:
    """Completeness/purity of a candidate list against the truth sources.

    ``truth`` is a list of TruthSource.to_json() dicts; ``candidates`` a list
    of dicts with x, y (and optionally xmin..ymax bounding boxes).
    """
    det_xy = [(c["x"], c["y"]) for c in candidates]
    truth_xy = [(s["x"], s["y"]) for s in truth]
    radii = [truth_match_radius(s) for s in truth]
    matches = match_greedy(det_xy, truth_xy, radii)
    matched_det = {i for i, _, _ in matches}
    matched_truth = {j for _, j, _ in matches}
    stars = [j for j, s in enumerate(truth) if s["kind"] == "star"]
    bright = []
    for j in stars:
        s = truth[j]
        sig = _noise_at(noise_std_sky, s["x"], s["y"])
        if sig and s["peak_adu"] >= BRIGHT_PEAK_SNR * sig:
            bright.append(j)
    unmatched = [i for i in range(len(candidates)) if i not in matched_det]
    false, fragments = [], []
    for i in unmatched:
        x, y = candidates[i]["x"], candidates[i]["y"]
        inside = False
        if source_image is not None and noise_std_sky is not None:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < height and 0 <= xi < width:
                inside = source_image[yi, xi] > FOOTPRINT_SIGMA * noise_std_sky[yi, xi]
        (fragments if inside else false).append(i)
    quadrants = [0, 0, 0, 0]
    for i in false:
        x, y = candidates[i]["x"], candidates[i]["y"]
        quadrants[(1 if x >= width / 2 else 0) + (2 if y >= height / 2 else 0)] += 1
    centroid = [d for i, j, d in matches if truth[j]["kind"] == "star"]
    return {
        "match_rule": f"one-to-one greedy, d <= max({MATCH_MIN_PX}, {MATCH_HFR_FRACTION}*HFR_true) px "
                      f"({MATCH_EXTENDED_PX} px for extended)",
        "n_truth": len(truth),
        "n_truth_stars": len(stars),
        "n_candidates": len(candidates),
        "n_matched": len(matches),
        "completeness": (len(matched_truth) / len(truth)) if truth else None,
        "completeness_stars": (sum(1 for j in stars if j in matched_truth) / len(stars)) if stars else None,
        "completeness_bright_stars": (sum(1 for j in bright if j in matched_truth) / len(bright)) if bright else None,
        "n_bright_stars": len(bright),
        "purity": (len(matched_det) / len(candidates)) if candidates else None,
        "false_detections": len(false),
        "false_detections_by_quadrant_tl_tr_bl_br": quadrants,
        "fragments_on_sources": len(fragments),
        "centroid_error_median_px": _median(centroid),
        "_matches": matches,
    }


def _noise_at(noise_map, x, y):
    if noise_map is None:
        return None
    h, w = noise_map.shape
    xi, yi = min(max(int(round(x)), 0), w - 1), min(max(int(round(y)), 0), h - 1)
    return float(noise_map[yi, xi])


def blend_metrics(candidates, truth, matches) -> dict:
    """Merged multi-member blend groups and split sources (counts)."""
    groups = {}
    for j, s in enumerate(truth):
        groups.setdefault(s["blend_group"], []).append(j)
    multi = {g: m for g, m in groups.items() if len(m) > 1}
    truth_to_det = {j: i for i, j, _ in matches}
    merged = 0
    for members in multi.values():
        matched = [j for j in members if j in truth_to_det]
        missing = [j for j in members if j not in truth_to_det]
        if not matched or not missing:
            continue
        hit = False
        for j in missing:
            for k in matched:
                c = candidates[truth_to_det[k]]
                if _inside_bbox(c, truth[j]["x"], truth[j]["y"]):
                    hit = True
        merged += int(hit)
    txy = np.array([[s["x"], s["y"]] for s in truth]) if truth else np.zeros((0, 2))
    split = 0
    if len(truth) and candidates:
        dxy = np.array([[c["x"], c["y"]] for c in candidates])
        nearest = np.argmin(np.hypot(dxy[:, None, 0] - txy[None, :, 0], dxy[:, None, 1] - txy[None, :, 1]), axis=1)
        for j, s in enumerate(truth):
            if s["kind"] != "star":
                continue
            r_split = max(3.0, 1.5 * (s["hfr_true_px"] or 0.0))
            near = np.hypot(dxy[:, 0] - s["x"], dxy[:, 1] - s["y"]) <= r_split
            if int(np.sum(near & (nearest == j))) >= 2:
                split += 1
    return {
        "n_multi_member_groups": len(multi),
        "merged_groups": merged,
        "merged_rule": "group with a matched member and an unmatched member inside that detection's bounding box",
        "split_sources": split,
        "split_rule": "truth star with >= 2 candidates within max(3, 1.5*HFR_true) px whose nearest truth it is",
    }


def _inside_bbox(c, x, y, pad=0.5):
    if all(k in c for k in ("xmin", "xmax", "ymin", "ymax")):
        return c["xmin"] - pad <= x <= c["xmax"] + pad and c["ymin"] - pad <= y <= c["ymax"] + pad
    return False


# --------------------------------------------------------------------------
# HFR of accepted stars
# --------------------------------------------------------------------------


def hfr_metrics(accepted, truth) -> dict:
    """Signed HFR error of accepted stars matched to truth stars (px)."""
    if accepted is None:
        return {"n_accepted": None}
    stars = [s for s in truth if s["kind"] == "star"]
    group_size = {}
    for s in truth:
        group_size[s["blend_group"]] = group_size.get(s["blend_group"], 0) + 1
    matches = match_greedy([(a["x"], a["y"]) for a in accepted], [(s["x"], s["y"]) for s in stars],
                           [truth_match_radius(s) for s in stars])
    clean, all_err = [], []
    sat_leak = blend_leak = 0
    matched_ids = []
    for i, j, _ in matches:
        s = stars[j]
        hfr = accepted[i]["hfr_px"]
        if hfr is None or s["hfr_true_px"] is None:
            continue
        err = hfr - s["hfr_true_px"]
        all_err.append((err, err / s["hfr_true_px"]))
        matched_ids.append(s["id"])
        if s["saturated"]:
            sat_leak += 1
        elif group_size.get(s["blend_group"], 1) > 1:
            blend_leak += 1
        else:
            clean.append((err, err / s["hfr_true_px"]))
    out = {
        "n_accepted": len(accepted),
        "n_matched": len(all_err),
        "n_unmatched_accepted": len(accepted) - len(matches),
        "saturated_leakage": sat_leak,
        "blend_leakage": blend_leak,
        "n_clean": len(clean),
        "matched_truth_ids": sorted(matched_ids),
    }
    for label, rows in (("clean", clean), ("all", all_err)):
        if rows:
            e = np.array([r[0] for r in rows])
            rel = np.array([r[1] for r in rows])
            out[f"{label}_bias_median_px"] = float(np.median(e))
            out[f"{label}_bias_mean_px"] = float(np.mean(e))
            out[f"{label}_mae_px"] = float(np.mean(np.abs(e)))
            out[f"{label}_rel_bias_median"] = float(np.median(rel))
            out[f"{label}_rel_dispersion"] = float(np.std(rel))
            out[f"{label}_outlier_fraction"] = float(np.mean(np.abs(rel) > HFR_OUTLIER_REL))
    return out


# --------------------------------------------------------------------------
# Focus curves
# --------------------------------------------------------------------------


def fit_parabola(positions, values) -> dict:
    p = np.asarray(positions, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(y)
    p, y = p[ok], y[ok]
    if p.size < 3:
        return {"status": "failed", "reason": f"{p.size} valid points < 3"}
    c, s = p.mean(), p.std() or 1.0
    a2, a1, a0 = np.polyfit((p - c) / s, y, 2)
    if not a2 > 0:
        return {"status": "failed", "reason": "parabola opens downward or is flat", "a2": float(a2)}
    p0 = c - s * a1 / (2.0 * a2)
    resid = y - np.polyval([a2, a1, a0], (p - c) / s)
    out = {"status": "completed", "p0": float(p0), "rms_residual": float(np.sqrt(np.mean(resid**2))),
           "n_points": int(p.size)}
    if not (p.min() <= p0 <= p.max()):
        out.update(status="failed", reason="minimum outside the sampled range")
    return out


def hyperbola(p, a, p0, b):
    return a * np.sqrt(1.0 + ((np.asarray(p, dtype=np.float64) - p0) / b) ** 2)


def fit_hyperbola(positions, values, max_iter=200) -> dict:
    """Least squares HFR(p) = a*sqrt(1 + ((p-p0)/b)^2) (Levenberg-Marquardt).

    Initialised from the exact linear fit of HFR^2 = A p^2 + B p + C, which
    the hyperbola satisfies with p0 = -B/2A, a^2 = C - B^2/4A, b = a/sqrt(A).
    """
    p = np.asarray(positions, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(y) & (y > 0)
    p, y = p[ok], y[ok]
    if p.size < 4:
        return {"status": "failed", "reason": f"{p.size} valid points < 4"}
    c, s = p.mean(), p.std() or 1.0
    x = (p - c) / s
    A, B, C = np.polyfit(x, y * y, 2)
    if A > 0:
        x0 = -B / (2 * A)
        a2 = C - B * B / (4 * A)
        a = math.sqrt(a2) if a2 > 0 else float(y.min())
        b = a / math.sqrt(A)
    else:
        x0, a, b = float(x[np.argmin(y)]), float(y.min()), 1.0
    theta = np.array([a, x0, max(b, 1e-3)])
    lam = 1e-3

    def resid(t):
        return y - hyperbola(x, *t)

    r = resid(theta)
    cost = float(r @ r)
    converged = False
    for _ in range(max_iter):
        a, x0, b = theta
        u = (x - x0) / b
        sq = np.sqrt(1 + u * u)
        jac = np.column_stack([sq, -a * u / (sq * b), -a * u * u / (sq * b)])
        g = jac.T @ r
        hmat = jac.T @ jac
        step = np.linalg.solve(hmat + lam * np.diag(np.diag(hmat) + 1e-12), g)
        trial = theta + step
        if trial[0] <= 0 or trial[2] <= 0:
            lam *= 10
            continue
        r_t = resid(trial)
        cost_t = float(r_t @ r_t)
        if cost_t < cost:
            done = abs(cost - cost_t) <= 1e-12 * max(cost, 1e-30) or np.max(np.abs(step)) < 1e-10
            theta, r, cost = trial, r_t, cost_t
            lam = max(lam / 10, 1e-12)
            if done:
                converged = True
                break
        else:
            lam *= 10
            if lam > 1e12:
                converged = True  # no further decrease possible: local minimum
                break
    a, x0, b = theta
    p0 = c + s * x0
    out = {"status": "completed", "p0": float(p0), "a_px": float(a), "b_steps": float(b * s),
           "rms_residual": float(np.sqrt(cost / p.size)), "n_points": int(p.size), "converged": converged}
    if not converged:
        out.update(status="failed", reason="did not converge")
    elif not (p.min() <= p0 <= p.max()):
        out.update(status="failed", reason="minimum outside the sampled range")
    return out


#: A fitted minimum needs at least this many valid frames on each side.
MIN_FRAMES_EACH_SIDE = 2


def focus_shift(positions, values, p0_true) -> dict:
    """Estimated minus true best-focus position for both curve models.

    A fit whose minimum has fewer than ``MIN_FRAMES_EACH_SIDE`` valid frames
    on either side is a failed sweep (the minimum is extrapolated)."""
    out = {}
    pos = np.asarray(positions, dtype=np.float64)
    ok = np.isfinite(np.asarray(values, dtype=np.float64))
    for name, fit in (("hyperbola", fit_hyperbola(positions, values)), ("parabola", fit_parabola(positions, values))):
        if fit["status"] == "completed":
            left, right = int(np.sum(ok & (pos < fit["p0"]))), int(np.sum(ok & (pos > fit["p0"])))
            fit["valid_frames_left_right"] = [left, right]
            if min(left, right) < MIN_FRAMES_EACH_SIDE:
                fit.update(status="failed", reason=f"only {left}/{right} valid frames left/right of the minimum")
            else:
                fit["p0_shift"] = fit["p0"] - p0_true
        out[name] = fit
    vals = np.asarray(values, dtype=np.float64)
    out["n_frames"] = int(vals.size)
    out["n_frames_valid"] = int(np.sum(np.isfinite(vals)))
    if np.any(np.isfinite(vals)):
        out["min_sample_position"] = float(np.asarray(positions)[int(np.nanargmin(vals))])
    return out


def jaccard(a, b) -> float | None:
    a, b = set(a), set(b)
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def strip_private(d):
    """Drop keys starting with '_' (internal match lists) for JSON output."""
    if isinstance(d, dict):
        return {k: strip_private(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [strip_private(v) for v in d]
    return d
