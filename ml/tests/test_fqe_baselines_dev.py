"""Explicit-partition and unavailable-rank contract of eval.baselines (generated fixtures only)."""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import fqe_sources as fs  # noqa: E402
from data_pipeline.fqe_sources import SealedPartitionError, load_source, write_fixture_source  # noqa: E402
from eval import baselines  # noqa: E402
from eval.baselines import (  # noqa: E402
    UNAVAILABLE,
    direct_hfr_measure,
    direct_hfr_measure_detailed,
    freeze_baselines,
    is_available,
    within_run_rank,
)


def _star(size=32, sigma=2.0, amp=1000.0, sky=100.0, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.indices((size, size), dtype=np.float64)
    c = (size - 1) / 2.0
    image = sky + amp * np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma**2))
    return image + rng.normal(0.0, noise, image.shape)


def _legacy_dataset():
    """Two genuine 5-row runs per split (train/val/test) with monotone HFR inside each run."""
    crops, hfr, run, group = [], [], [], []
    for run_id in range(6):
        for p in range(5):
            sigma = 1.0 + 0.4 * p
            crops.append(_star(sigma=sigma, seed=10 * run_id + p).astype(np.float32))
            hfr.append(sigma * math.sqrt(2 * math.log(2)) + 0.01 * run_id)
            run.append(run_id)
            group.append(f"run{run_id}")
    data = {"crops": np.stack(crops), "hfr_px": np.asarray(hfr, dtype=np.float32),
            "run_id": np.asarray(run), "split_group": np.asarray(group)}
    rows = np.arange(len(hfr))
    splits = {"train": rows[np.isin(run, [0, 1])], "val": rows[np.isin(run, [2, 3])],
              "test": rows[np.isin(run, [4, 5])]}
    return data, splits


@pytest.fixture
def sim(tmp_path):
    return write_fixture_source(tmp_path / "data", "fixsim", n_groups=20, rows_per_group=3, style="sim")


def test_partition_is_required_keyword_without_default():
    data, splits = _legacy_dataset()
    with pytest.raises(TypeError, match="partition"):
        freeze_baselines(data, splits=splits)
    with pytest.raises(TypeError):
        freeze_baselines(data, 0, splits, "test")  # positional partition is not accepted


def test_legacy_dict_scores_only_the_named_partition():
    data, splits = _legacy_dataset()
    val = freeze_baselines(data, splits=splits, partition="val")
    assert val["partition"] == "val" and val["n_eval"] == splits["val"].size and val["n_fit"] == splits["train"].size
    median = float(np.median(data["hfr_px"][splits["train"]].astype(np.float64)))
    assert val["train_median_hfr_px"] == pytest.approx(median)
    expected_mae = float(np.mean(np.abs(data["hfr_px"][splits["val"]].astype(np.float64) - median)))
    assert val["constant"]["hfr_mae_px"] == pytest.approx(expected_mae, rel=1e-12)
    # The explicit final evaluation of genuine local runs remains available.
    test = freeze_baselines(data, splits=splits, partition="test")
    assert test["partition"] == "test" and test["n_eval"] == splits["test"].size
    expected_test = float(np.mean(np.abs(data["hfr_px"][splits["test"]].astype(np.float64) - median)))
    assert test["constant"]["hfr_mae_px"] == pytest.approx(expected_test, rel=1e-12)
    with pytest.raises(ValueError):
        freeze_baselines(data, splits=splits, partition="train")


def test_genuine_runs_keep_numeric_rank_metrics_and_constant_is_unavailable():
    data, splits = _legacy_dataset()
    res = freeze_baselines(data, splits=splits, partition="val")
    assert res["direct_hfr"]["within_run_rho"] == pytest.approx(1.0)
    assert res["direct_hfr"]["within_run_rho_runs"] == 2
    assert is_available(res["tenengrad"]["within_run_rho"])
    # A constant has no ordering: never the old hard-coded 0.0.
    assert res["constant"]["within_run_rho"] == UNAVAILABLE
    assert res["constant"]["within_run_rho_reason"]


def test_within_run_rank_needs_four_rows_and_never_returns_nan():
    hand = within_run_rank(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(4), sign=1.0)
    assert hand["within_run_rho"] == pytest.approx(1.0) and hand["within_run_rho_runs"] == 1
    flipped = within_run_rank(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(4), sign=-1.0)
    assert flipped["within_run_rho"] == pytest.approx(-1.0)
    singletons = within_run_rank(np.arange(10.0), np.arange(10.0) + 1, np.arange(10), sign=1.0)
    assert singletons["within_run_rho"] == UNAVAILABLE and singletons["within_run_rho_runs"] == 0
    three = within_run_rank(np.arange(3.0), np.arange(3.0) + 1, np.zeros(3), sign=1.0)
    assert three["within_run_rho"] == UNAVAILABLE
    flat = within_run_rank(np.arange(4.0), np.ones(4), np.zeros(4), sign=1.0)
    assert flat["within_run_rho"] == UNAVAILABLE
    # Non-finite scores (failed measurements) are masked, not counted as rows.
    holes = within_run_rank(np.array([1.0, np.nan, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(4), sign=1.0)
    assert holes["within_run_rho"] == UNAVAILABLE
    for value in (UNAVAILABLE, None, float("nan")):
        assert not is_available(value)
    assert is_available(0.0) and is_available(np.float32(0.5))


def test_source_arrays_path_scores_val_and_fits_train_fit_only(sim):
    arrays = load_source("fixsim", partitions=("train_fit", "calibration", "val"), contract=sim.contract,
                         data_root=sim.data_root)
    assert (arrays.partition == "calibration").any()
    res = freeze_baselines(arrays, partition="val")
    fit = arrays.partition == "train_fit"
    val = arrays.partition == "val"
    assert res["n_fit"] == int(fit.sum()) and res["n_eval"] == int(val.sum())
    target = arrays.target_hfr_px.astype(np.float64)
    assert res["train_median_hfr_px"] == pytest.approx(float(np.median(target[fit])))
    assert res["constant"]["hfr_mae_px"] == pytest.approx(float(np.mean(np.abs(target[val] - np.median(target[fit])))))
    # Archive rows are singletons: every rank metric is unavailable with a reason.
    for name in ("constant", "direct_hfr", "tenengrad", "norm_variance"):
        assert res[name]["within_run_rho"] == UNAVAILABLE
        assert res[name]["within_run_rho_reason"]
    assert res["source"]["source_id"] == "fixsim" and res["runtime_eligible"] is False
    assert res["source"]["eval_membership_sha256"] == fs._membership_sha256(arrays.sample_uid[val])
    calibration = freeze_baselines(arrays, partition="calibration")
    assert calibration["n_eval"] == int((arrays.partition == "calibration").sum())


def test_source_arrays_refuse_the_sealed_test_partition(sim):
    arrays = load_source("fixsim", partitions=("train_fit", "val"), contract=sim.contract, data_root=sim.data_root)
    with pytest.raises(SealedPartitionError):
        freeze_baselines(arrays, partition="test")
    with pytest.raises(ValueError):
        freeze_baselines(arrays, partition="val", fit_partition="calibration")
    with pytest.raises(ValueError):
        freeze_baselines(arrays, splits={"train": np.arange(3)}, partition="val")


def test_direct_hfr_reasons_and_never_zero():
    measured = direct_hfr_measure_detailed(_star())
    assert measured["status"] == "measured" and measured["hfr_px"] > 0 and measured["reason"] is None
    assert direct_hfr_measure(_star()) == pytest.approx(measured["hfr_px"])
    plateau = direct_hfr_measure_detailed(np.full((32, 32), 0.5))
    assert (plateau["status"], plateau["reason"], plateau["hfr_px"]) == ("unmeasured", "bg_unresolved", None)
    blank = direct_hfr_measure_detailed(np.full((32, 32), np.nan))
    assert blank["reason"] == "bg_unresolved"
    # A normalized-looking crop with no pixel above level + 3*rms: no core.
    rng = np.random.default_rng(3)
    flat = rng.uniform(0.4, 0.6, size=(64, 64))
    no_core = direct_hfr_measure_detailed(flat)
    assert (no_core["status"], no_core["reason"], no_core["hfr_px"]) == ("unmeasured", "no_core_above_3sigma", None)
    assert math.isnan(direct_hfr_measure(flat))


def _write_local(path, data):
    """A marker-free local legacy NPZ (genuine runs, no run_key_semantics)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return path


def _write_registry(npz_path, data, splits, registry_path=None):
    """A frozen registry bound to ``npz_path`` (what load_registered_splits checks)."""
    registry_path = registry_path or npz_path.with_suffix(".splits.json")
    registry_path.write_text(json.dumps({
        "dataset_sha256": baselines._sha256_path(npz_path), "n_samples": int(len(data["hfr_px"])),
        "indices": {k: np.asarray(v).tolist() for k, v in splits.items()},
    }))
    return registry_path


def _contract_with(sim, tmp_path, *, pin=None, excluded=None):
    """The fixture contract, written to disk, optionally pinning or excluding extra digests."""
    data = json.loads(json.dumps(sim.contract.data))
    entry = data["sources"]["fixsim"]
    if pin is not None:
        entry["files"]["extra.npz"] = pin
    if excluded is not None:
        entry["excluded_files"] = {"fqe_all_references.npz": f"{excluded} (no registry; test-era rows; never opened)"}
    path = tmp_path / f"contract-{len(list(tmp_path.glob('contract-*.json')))}.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_legacy_cli_refuses_a_public_data_path(tmp_path, monkeypatch):
    root = tmp_path / "public"
    (root / "src").mkdir(parents=True)
    monkeypatch.setenv(fs.DATA_ROOT_ENV, str(root))
    with pytest.raises(SystemExit, match="public data root"):
        baselines._refuse_public_dataset(str(root / "src" / "fqe.npz"))
    data, _ = _legacy_dataset()
    local = _write_local(tmp_path / "local" / "local.npz", data)
    with pytest.raises(SystemExit, match="split registry .* public data root"):
        baselines._refuse_public_dataset(str(local), str(root / "src" / "fqe.splits.json"))
    baselines._refuse_public_dataset(str(local))  # outside, marker-free, unpinned: allowed


def test_legacy_cli_refuses_copies_of_public_npzs_outside_the_root(sim, tmp_path):
    """A byte copy or a re-saved copy carries the importers' archive marker."""
    source = sim.data_root / "fixsim" / "fqe.npz"
    copy = tmp_path / "elsewhere" / "copy.npz"
    copy.parent.mkdir()
    copy.write_bytes(source.read_bytes())
    with pytest.raises(SystemExit, match="run_key_semantics"):
        baselines._refuse_public_dataset(str(copy))
    with np.load(source, allow_pickle=False) as npz:
        resaved = {k: npz[k] for k in npz.files}
    _write_local(tmp_path / "elsewhere" / "resaved.npz", resaved)
    with pytest.raises(SystemExit, match="run_key_semantics"):
        baselines._refuse_public_dataset(str(tmp_path / "elsewhere" / "resaved.npz"))


def test_legacy_cli_refuses_pinned_and_excluded_digests_without_the_marker(sim, tmp_path):
    """The hash screen is independent of the marker: pinned files, excluded files, pinned registries."""
    data, splits = _legacy_dataset()
    local = _write_local(tmp_path / "local" / "local.npz", data)
    digest = baselines._sha256_path(local)
    baselines._refuse_public_dataset(str(local), contract_path=_contract_with(sim, tmp_path))  # unpinned: allowed
    with pytest.raises(SystemExit, match="sha256 equals the source contract's fixsim/extra.npz"):
        baselines._refuse_public_dataset(str(local), contract_path=_contract_with(sim, tmp_path, pin=digest))
    with pytest.raises(SystemExit, match=r"fqe_all_references.npz \(excluded file\)"):
        baselines._refuse_public_dataset(str(local), contract_path=_contract_with(sim, tmp_path, excluded=digest))
    # A copied public registry next to a marker-free dataset is refused by its pin.
    registry = tmp_path / "local" / "copied.splits.json"
    registry.write_bytes((sim.data_root / "fixsim" / "fqe.splits.json").read_bytes())
    with pytest.raises(SystemExit, match="split registry .* sha256 equals the source contract's fixsim/fqe.splits.json"):
        baselines._refuse_public_dataset(str(local), str(registry), contract_path=_contract_with(sim, tmp_path))
    # The tracked contract's own pins are always consulted (fixture contract optional).
    tracked = baselines._public_digests()
    assert "13d6133b42ada94e88e572b33449840e85ba8669e30f95b95c77e38fc0a6f2ca" in tracked  # fqe_all_references.npz
    assert "8b5b9fcfcc394e73871c762e3a4448220289b94502b064d479444c2c9580754d" in tracked  # stellar_sim_v1 fqe.npz


def test_legacy_cli_never_creates_a_split_registry(tmp_path):
    """Without an existing registry the legacy path refuses; with one it scores that frozen split."""
    data, splits = _legacy_dataset()
    local = _write_local(tmp_path / "local" / "local.npz", data)
    before = _snapshot(tmp_path / "local")
    with pytest.raises(SystemExit, match="no split registry"):
        baselines.main(str(local), partition="val")
    assert _snapshot(tmp_path / "local") == before  # nothing written next to the NPZ
    _write_registry(local, data, splits)
    result = baselines.main(str(local), partition="val")
    assert result["partition"] == "val" and result["n_eval"] == splits["val"].size
    explicit = _write_registry(local, data, splits, tmp_path / "frozen.json")
    (local.with_suffix(".splits.json")).unlink()
    assert baselines.main(str(local), split_registry=str(explicit), partition="val")["n_eval"] == splits["val"].size
    assert not local.with_suffix(".splits.json").exists()


def test_main_requires_a_partition():
    with pytest.raises(SystemExit, match="partition"):
        baselines.main(None, True)


def _snapshot(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*"))


def test_main_source_path_never_writes_next_to_the_npz(sim, tmp_path):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(__import__("json").dumps(sim.contract.data))
    before = _snapshot(sim.data_root)
    out = tmp_path / "baselines.json"
    result = baselines.main(sources=["fixsim"], partition="val", data_root=sim.data_root,
                            contract_path=str(contract_path), out=str(out))
    assert result["partition"] == "val" and out.is_file()
    assert _snapshot(sim.data_root) == before
    with pytest.raises(SealedPartitionError):
        baselines.main(sources=["fixsim"], partition="test", data_root=sim.data_root,
                       contract_path=str(contract_path))
