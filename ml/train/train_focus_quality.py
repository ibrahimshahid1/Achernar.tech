"""Train the Focus Quality Estimator (FQE) — Section 7.6.1.

Predicts per-star log(HFR_px) with a heteroscedastic variance head, optionally
regularized by a within-run RankNet ordering loss. Grouped (by-run) splits.

Usage:
    python -m train.train_focus_quality --synthetic --epochs 12
    python -m train.train_focus_quality --dataset focus_quality_ds_v1.npz --epochs 50
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_dataset(config: dict) -> dict:
    if config.get("synthetic") or not config.get("dataset"):
        from data_pipeline.synthetic_psf import make_synthetic_dataset

        return make_synthetic_dataset(
            n_runs=config.get("synthetic_runs", 48), seed=config.get("seed", 0)
        )
    data = np.load(config["dataset"], allow_pickle=False)
    return {k: data[k] for k in data.files}


class GroupedBatchSampler:
    """Yield batches from one autofocus run at a time.

    A row-shuffled DataLoader usually puts only one crop from a given run in a
    batch, making the within-run RankNet term silently zero. Grouped batches make
    every optimization step contain meaningful same-run pairs while shuffling
    run and crop order each epoch.
    """

    def __init__(self, run_ids, batch_size: int, seed: int = 0):
        self.run_ids = np.asarray(run_ids)
        self.batch_size = max(2, int(batch_size))
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        groups = np.unique(self.run_ids)
        rng.shuffle(groups)
        for group in groups:
            members = np.flatnonzero(self.run_ids == group)
            rng.shuffle(members)
            chunks = [
                members[start : start + self.batch_size]
                for start in range(0, len(members), self.batch_size)
            ]
            if len(chunks) > 1 and chunks[-1].size == 1:
                chunks[-2] = np.concatenate((chunks[-2], chunks[-1]))
                chunks.pop()
            for batch in chunks:
                if batch.size:
                    yield batch.tolist()

    def __len__(self):
        total = 0
        for group in np.unique(self.run_ids):
            count = int(np.sum(self.run_ids == group))
            batches = int(np.ceil(count / self.batch_size))
            if batches > 1 and count % self.batch_size == 1:
                batches -= 1
            total += batches
        return total


def evaluate(model, crops, log_hfr, hfr_px, run_id, device, batch_size=256):
    """Return per-star HFR MAE (px), 95% interval coverage, and mean within-run
    Spearman rank correlation of predicted vs. true HFR."""
    import torch
    from scipy.stats import spearmanr

    model.eval()
    mus, lvs = [], []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            xb = torch.as_tensor(crops[i : i + batch_size][:, None], device=device)
            mu, lv = model(xb)
            mus.append(mu.cpu().numpy())
            lvs.append(lv.cpu().numpy())
    mu = np.concatenate(mus)
    lv = np.clip(np.concatenate(lvs), -8.0, 8.0)

    hfr_pred = np.exp(mu)
    mae = float(np.mean(np.abs(hfr_pred - hfr_px)))
    sigma = np.sqrt(np.exp(lv))
    covered = np.abs(log_hfr - mu) <= 1.96 * sigma
    coverage = float(np.mean(covered))

    rhos = []
    for run in np.unique(run_id):
        m = run_id == run
        if m.sum() >= 4 and np.ptp(hfr_px[m]) > 0:
            rho = spearmanr(mu[m], hfr_px[m]).statistic
            if np.isfinite(rho):
                rhos.append(rho)
    rank_rho = float(np.mean(rhos)) if rhos else float("nan")
    return {"hfr_mae_px": mae, "coverage_95": coverage, "within_run_rank_rho": rank_rho}


def train(config: dict) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from data_pipeline.split_registry import create_group_splits, splits_for_dataset
    from models.focus_quality_cnn import FocusQualityEstimator
    from models.fqe_losses import heteroscedastic_nll, ranknet_within_run

    torch.manual_seed(config.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = _load_dataset(config)
    crops = ds["crops"].astype(np.float32)
    log_hfr = ds["log_hfr"].astype(np.float32)
    hfr_px = ds["hfr_px"].astype(np.float32)
    run_id = ds["run_id"].astype(np.int64)

    split_groups = ds.get("split_group", run_id)
    if config.get("dataset") and not config.get("synthetic"):
        splits = splits_for_dataset(
            config["dataset"],
            ds,
            seed=config.get("seed", 0),
            registry_path=config.get("split_registry"),
        )
    else:
        splits = create_group_splits(split_groups, seed=config.get("seed", 0))
    tr, va = splits["train"], splits["val"]

    tr_ds = TensorDataset(
        torch.as_tensor(crops[tr][:, None]),
        torch.as_tensor(log_hfr[tr]),
        torch.as_tensor(run_id[tr]),
    )
    batch_sampler = GroupedBatchSampler(
        run_id[tr], config.get("batch_size", 64), seed=config.get("seed", 0)
    )
    loader = DataLoader(tr_ds, batch_sampler=batch_sampler)

    model = FocusQualityEstimator(pretrained=config.get("pretrained", False)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.get("epochs", 12))
    lambda_rank = config.get("lambda_rank", 0.2)
    epochs = config.get("epochs", 12)
    variance_warmup_epochs = config.get("variance_warmup_epochs")
    if variance_warmup_epochs is None:
        variance_warmup_epochs = max(1, min(3, epochs // 5))

    best = {"val_mae": float("inf"), "epoch": -1, "state": None}
    for epoch in range(epochs):
        model.train()
        for xb, yb, rb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, lv = model(xb)
            if epoch < variance_warmup_epochs:
                # Teach the mean/ranking signal before the variance head is
                # allowed to explain residuals (Nix-Weigend failure mode).
                loss = torch.nn.functional.mse_loss(mu, yb)
            else:
                loss = heteroscedastic_nll(mu, lv, yb)
            if lambda_rank > 0:
                loss = loss + lambda_rank * ranknet_within_run(mu, yb, rb.numpy())
            opt.zero_grad()
            loss.backward()
            # Guard the warmup->NLL transition: once the variance head switches on,
            # inv_var = exp(-log_var) (up to exp(8) ~ 3000x) amplifies the mean-head
            # gradient and can transiently diverge (the epoch-3 MAE ~7e10 spike).
            # heteroscedastic_nll clamps log_var's *value* but not the gradient, so
            # clip the global grad norm — standard, and cheap insurance on noisy data.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()

        metrics = evaluate(model, crops[va], log_hfr[va], hfr_px[va], run_id[va], device)
        print(
            f"epoch {epoch:2d}  val HFR MAE {metrics['hfr_mae_px']:.3f}px  "
            f"cov95 {metrics['coverage_95']:.2f}  rank_rho {metrics['within_run_rank_rho']:.3f}"
        )
        if metrics["hfr_mae_px"] < best["val_mae"]:
            best = {
                "val_mae": metrics["hfr_mae_px"],
                "epoch": epoch,
                "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "metrics": metrics,
            }

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    out_path = config.get("out", "focus_quality_fqe.pt")
    torch.save(model.state_dict(), out_path)
    print(f"✓ Saved best model (epoch {best['epoch']}, val MAE {best['val_mae']:.3f}px) -> {out_path}")

    # Final test metrics on the held-out group.
    te = splits["test"]
    test_metrics = (
        evaluate(model, crops[te], log_hfr[te], hfr_px[te], run_id[te], device)
        if te.size
        else {}
    )
    return {
        "best_val": best.get("metrics", {}),
        "test": test_metrics,
        "model_path": out_path,
        "split_counts": {name: int(index.size) for name, index in splits.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-runs", type=int, default=48)
    parser.add_argument("--split-registry")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-rank", type=float, default=0.2)
    parser.add_argument("--variance-warmup-epochs", type=int)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--out", default="focus_quality_fqe.pt")
    args = parser.parse_args()
    result = train(vars(args))
    print("RESULT:", result)
