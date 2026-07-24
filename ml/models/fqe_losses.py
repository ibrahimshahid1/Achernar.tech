"""
fqe_losses — training objectives + robust aggregation for the FQE (Section 7.6.1).

  * heteroscedastic_nll: Gaussian mean/variance NLL in log-HFR space
    (Nix & Weigend 1994; Kendall & Gal 2017). This is the primary loss.
  * ranknet_within_run: an auxiliary pairwise ordering loss (Burges 2005) that
    sharpens *relative* focus ranking within a single run — the thing the
    autofocus curve fit actually cares about.
  * aggregate_stars: transparent robust pooling (median + between-star MAD) that
    turns per-star predictions into a frame HFR and an uncertainty estimate,
    *before* any learned pooling (Bosse 2018). The resulting frame interval still
    needs its own held-out calibration scale.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def heteroscedastic_nll(mu, log_var, target_log_hfr, clamp_log_var=(-8.0, 8.0)):
    """Negative Gaussian log-likelihood of ``target_log_hfr`` under N(mu, exp(log_var)).

        L = 0.5 * exp(-s) * (y - mu)^2 + 0.5 * s      (s = log_var)

    The ``exp(-s)`` term down-weights high-uncertainty samples; the ``+0.5*s``
    term prevents the trivial collapse to infinite variance.
    """
    lo, hi = clamp_log_var
    log_var = torch.clamp(log_var, lo, hi)
    inv_var = torch.exp(-log_var)
    return (0.5 * inv_var * (target_log_hfr - mu) ** 2 + 0.5 * log_var).mean()


def ranknet_within_run(mu, target_log_hfr, run_ids, max_pairs=4096):
    """Auxiliary RankNet loss over pairs drawn from the *same* run.

    Score = -mu (higher score = sharper = lower HFR). For a pair (i, j) the model
    probability that i is sharper than j is sigmoid(score_i - score_j) =
    sigmoid(mu_j - mu_i); the target is 1 when HFR_i < HFR_j. Cross-focus pairs
    are ignored (they belong to different sweeps).
    """
    device = mu.device
    run_ids = np.asarray(run_ids)
    idx_i, idx_j = [], []
    for run in np.unique(run_ids):
        members = np.where(run_ids == run)[0]
        if members.size < 2:
            continue
        # All ordered pairs within the run (capped downstream).
        ii, jj = np.meshgrid(members, members)
        mask = ii < jj
        idx_i.append(ii[mask])
        idx_j.append(jj[mask])
    if not idx_i:
        return torch.zeros((), device=device)

    idx_i = np.concatenate(idx_i)
    idx_j = np.concatenate(idx_j)
    if idx_i.size > max_pairs:
        sel = np.random.choice(idx_i.size, max_pairs, replace=False)
        idx_i, idx_j = idx_i[sel], idx_j[sel]

    i = torch.as_tensor(idx_i, device=device, dtype=torch.long)
    j = torch.as_tensor(idx_j, device=device, dtype=torch.long)
    logits = mu[j] - mu[i]  # >0 => model says i sharper (lower HFR)
    labels = (target_log_hfr[i] < target_log_hfr[j]).float()
    # Drop ties to avoid a meaningless 0.5 target.
    keep = target_log_hfr[i] != target_log_hfr[j]
    if keep.sum() == 0:
        return torch.zeros((), device=device)
    return F.binary_cross_entropy_with_logits(logits[keep], labels[keep])


def aggregate_stars(mu_log_hfr, log_var):
    """Pool per-star predictions for ONE frame into (hfr_px, sigma_log_hfr).

    - Frame log-HFR = median of per-star mu (robust to a few bad stars).
    - Uncertainty combines between-star scatter (MAD of mu) with the mean
      per-star aleatoric variance, shrinking with the number of stars used.

    Returns ``(None, None)`` for an empty input — never a spurious 0.
    """
    mu = np.asarray(mu_log_hfr, dtype=np.float64).ravel()
    if mu.size == 0:
        return None, None
    lv = np.asarray(log_var, dtype=np.float64).ravel()

    med = float(np.median(mu))
    between = float(np.median(np.abs(mu - med)) * 1.4826)  # robust MAD scatter
    mean_aleatoric_var = float(np.mean(np.exp(np.clip(lv, -8.0, 8.0)))) if lv.size else 0.0
    n = mu.size
    sigma_log = float(np.sqrt(between**2 + mean_aleatoric_var / max(n, 1)))
    hfr_px = float(np.exp(med))
    return hfr_px, sigma_log
