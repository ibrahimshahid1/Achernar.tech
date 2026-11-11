"""Trainer/experiment contract tests for the FQE public-data experiment.

Generated tmp-dir fixtures only (``fqe_sources.write_fixture_source``); a few
optimization steps on CPU at most. Covers: pointwise minibatches from singleton
rows, preserved grouped runs, lambda_rank refusal, opt-in test evaluation,
development commands that never reach a test evaluator, sealed predict,
unavailable calibration, resume == uninterrupted, the non-finite guard,
output-path refusal, data failures and the receipt-bound final test.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

import data_pipeline.fqe_sources as fs  # noqa: E402
import train.fqe_dev_experiment as dx  # noqa: E402
import train.train_focus_quality as tfq  # noqa: E402
from models.fqe_losses import NonFiniteLossError, require_finite  # noqa: E402
from train.calibrate_uncertainty import (  # noqa: E402
    calibrate,
    calibrate_development_arrays,
    calibrate_from_arrays,
)
from train.train_focus_quality import (  # noqa: E402
    GroupedBatchSampler,
    PointwiseBatchSampler,
    UpdateStream,
    choose_batching,
    require_rank_compatible,
)


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #
def _write_contract(tmp_path: Path, contract) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract.data))
    return path


def _arm(tmp_path: Path, fixture, *, name="arm.json", arm="synthetic_only", sources=None, **overrides) -> Path:
    contract_path = _write_contract(tmp_path, fixture.contract)
    config = {
        "schema": dx.CONFIG_SCHEMA,
        "arm": arm,
        "seed": 0,
        "batch_size": 8,
        "updates": 4,
        "warmup_updates": 2,
        "checkpoint_every": 2,
        "device": "cpu",
        "contract_path": str(contract_path),
        "data_root": str(fixture.data_root),
        "sources": sources or [fixture.source_id],
    }
    config.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def sim(tmp_path):
    return fs.write_fixture_source(tmp_path / "data", "fixsim", n_groups=20, rows_per_group=3,
                                   style="sim", role="synthetic_reference")


@pytest.fixture
def completed(tmp_path, sim):
    run = tmp_path / "run"
    record = dx.fit(_arm(tmp_path, sim), run)
    assert record["status"] == dx.STATUS_COMPLETED
    return run


def _test_keys(mapping: dict) -> list:
    """Keys that would report the test partition (the explicit false flag is not one)."""
    return [key for key in mapping if "test" in key and key != "test_reported"]


def _weights(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
def test_singleton_runs_select_pointwise_batching():
    batching, reason = choose_batching(np.arange(100))
    assert batching == "pointwise" and "no run" in reason
    assert choose_batching(np.repeat(np.arange(5), 9)) == ("grouped_run", "multi-row runs present")
    assert choose_batching(np.repeat(np.arange(5), 9), "pointwise")[0] == "pointwise"


def test_pointwise_sampler_builds_real_shuffled_minibatches_of_32():
    run_ids = np.arange(100)  # archive rows: every run id is a singleton
    sampler = PointwiseBatchSampler(run_ids.size, batch_size=32, seed=3)
    epoch0 = list(iter(sampler))
    assert [len(b) for b in epoch0] == [32, 32, 32, 4]
    assert len(sampler) == 4
    flat = np.concatenate(epoch0)
    assert sorted(flat.tolist()) == list(range(100))          # every row once per epoch
    assert not np.array_equal(flat, np.arange(100))            # shuffled
    assert len(np.unique(run_ids[epoch0[0]])) == 32            # 32 distinct "runs" per batch
    epoch1 = list(iter(sampler))
    assert not np.array_equal(np.concatenate(epoch1), flat)    # reshuffled per epoch
    again = list(iter(PointwiseBatchSampler(100, 32, seed=3)))
    assert all(np.array_equal(a, b) for a, b in zip(again, epoch0))  # seeded


def test_pointwise_sampler_merges_a_single_row_tail():
    batches = list(iter(PointwiseBatchSampler(65, 32, seed=0)))
    assert [len(b) for b in batches] == [32, 33]


def test_grouped_sampler_still_yields_one_run_per_batch():
    # The genuine-run path is unchanged (see also test_fqe_pipeline).
    run_ids = np.repeat(np.arange(5), 9)
    for batch in GroupedBatchSampler(run_ids, batch_size=4, seed=0):
        assert len(np.unique(run_ids[batch])) == 1 and len(batch) >= 2


def test_update_stream_is_epoch_free_and_a_pure_function_of_the_update():
    stream = UpdateStream(70, 32, seed=5)
    batches = [stream.rows(u) for u in range(7)]
    assert all(b.size == 32 for b in batches)
    covered = np.concatenate(batches)[:70]
    assert sorted(covered.tolist()) == list(range(70))          # first pass covers every row once
    fresh = UpdateStream(70, 32, seed=5)
    assert np.array_equal(fresh.rows(4), batches[4])            # no hidden sampler state
    assert not np.array_equal(UpdateStream(70, 32, seed=6).rows(0), batches[0])


def test_public_fixture_rows_are_pointwise_and_local_runs_are_grouped(tmp_path, sim):
    rows = fs.load_source("fixsim", contract=sim.contract, data_root=sim.data_root)
    assert rows.batching == "pointwise" and not rows.ranking_eligible.any()
    local = fs.write_fixture_source(tmp_path / "local", "fixlocal", n_groups=6, rows_per_group=5, style="local")
    runs = fs.load_source("fixlocal", contract=local.contract, data_root=local.data_root)
    assert runs.batching == "grouped_run" and runs.ranking_eligible.any()
    run_ids = np.unique(runs.group_uid, return_inverse=True)[1]
    require_rank_compatible(0.2, batching="grouped_run", run_ids=run_ids,
                            ranking_eligible=runs.ranking_eligible, roles=[runs.role])  # allowed


def test_generated_synthetic_runs_keep_grouped_batching_and_rank_loss():
    data = tfq._prepare_training_data({"synthetic": True, "synthetic_runs": 3, "seed": 0})
    assert data["batching"] == "grouped_run"
    tr = data["splits"]["train"]
    require_rank_compatible(0.2, batching=data["batching"], run_ids=data["run_id"][tr], roles=data["roles"])


# --------------------------------------------------------------------------- #
# lambda_rank refusal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("role", ["synthetic_reference", "archive_measurement_reference"])
def test_nonzero_lambda_rank_is_refused_without_ranking_pairs(role):
    with pytest.raises(ValueError) as err:
        require_rank_compatible(0.2, batching="pointwise", run_ids=np.arange(40), roles=[role])
    assert role in str(err.value) and "lambda_rank" in str(err.value)
    with pytest.raises(ValueError):
        require_rank_compatible(0.2, batching="grouped_run", run_ids=np.repeat(np.arange(4), 3),
                                ranking_eligible=np.zeros(12, dtype=bool), roles=[role])
    require_rank_compatible(0.0, batching="pointwise", run_ids=np.arange(40), roles=[role])


def test_train_refuses_rank_loss_on_contract_sources_before_training(sim, monkeypatch):
    monkeypatch.setattr(tfq, "GroupedBatchSampler", None)  # never reached
    with pytest.raises(ValueError, match="synthetic_reference"):
        tfq.train({"source_ids": ["fixsim"], "contract": sim.contract, "data_root": str(sim.data_root),
                   "lambda_rank": 0.2, "epochs": 1})


def test_dev_fit_refuses_rank_loss_and_writes_nothing(tmp_path, sim):
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="lambda_rank"):
        dx.fit(_arm(tmp_path, sim, lambda_rank=0.5), run)
    assert not run.exists()


def test_legacy_by_path_route_refuses_public_archive_npz(sim):
    with pytest.raises(ValueError, match="singleton_sample_not_focus_run"):
        tfq.train({"dataset": str(sim.directory / "fqe.npz"), "lambda_rank": 0.0, "epochs": 1})


# --------------------------------------------------------------------------- #
# Test evaluation is opt-in; rank metric is "unavailable" on singletons
# --------------------------------------------------------------------------- #
def test_train_evaluates_test_only_when_asked(tmp_path):
    base = {"synthetic": True, "synthetic_runs": 3, "epochs": 1, "batch_size": 64, "seed": 0, "device": "cpu"}
    default = tfq.train(dict(base, out=str(tmp_path / "a.pt")))
    assert default["test"] == {} and default["test_evaluated"] is False
    assert default["batching"] == "grouped_run" and default["lambda_rank"] == 0.2
    opted = tfq.train(dict(base, out=str(tmp_path / "b.pt")), evaluate_test=True)
    assert opted["test_evaluated"] is True and "hfr_mae_px" in opted["test"]


class _MeanPixelModel(torch.nn.Module):
    """Stub FQE whose mu varies with the crop (an untrained CNN is ~constant)."""

    def forward(self, x):
        mu = x.mean(dim=(1, 2, 3))
        return mu, torch.zeros_like(mu)


def test_within_run_spearman_is_unavailable_on_singleton_rows():
    model = _MeanPixelModel()
    crops = (np.arange(6, dtype=np.float32)[:, None, None] * np.ones((6, 64, 64), dtype=np.float32)) / 6.0
    hfr = np.linspace(1.0, 3.0, 6).astype(np.float32)
    metrics = tfq.evaluate(model, crops, np.log(hfr), hfr, np.arange(6), "cpu")
    assert metrics["within_run_rank_rho"] == "unavailable"
    assert ">= 4 rows" in metrics["within_run_rank_status"]
    grouped = tfq.evaluate(model, crops, np.log(hfr), hfr, np.zeros(6, dtype=np.int64), "cpu")
    assert isinstance(grouped["within_run_rank_rho"], float)


# --------------------------------------------------------------------------- #
# Development commands never reach a test evaluator
# --------------------------------------------------------------------------- #
def test_fit_predict_calibrate_never_touch_test(tmp_path, sim, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a development command reached a test evaluator")

    requested = []
    real_load = fs.load_source

    def recording_load(source_id, partitions=fs.DEV_PARTITIONS, **kwargs):
        requested.extend(partitions)
        return real_load(source_id, partitions=partitions, **kwargs)

    monkeypatch.setattr(fs, "load_final_test", forbidden)
    monkeypatch.setattr(dx, "final_test", forbidden)
    monkeypatch.setattr(tfq, "evaluate", forbidden)
    monkeypatch.setattr(fs, "load_source", recording_load)

    run = tmp_path / "run"
    record = dx.fit(_arm(tmp_path, sim), run)
    assert record["status"] == "completed" and record["validation_consulted_during_training"] is False
    manifest = dx.predict(run, partitions=("val", "calibration"))
    calibration = dx.calibrate(run)
    assert "test" not in requested and set(requested) <= set(fs.DEV_PARTITIONS)
    for result in (record, manifest, calibration):
        assert result["runtime_eligible"] is False
        assert result["data_roles"] == {"fixsim": "synthetic_reference"}
    assert all(json.loads(line)["finite"] for line in (run / "curves.jsonl").read_text().splitlines())


def test_predict_writes_per_row_outputs_and_refuses_test(tmp_path, completed):
    manifest = dx.predict(completed, partitions=("val",))
    (entry,) = manifest["outputs"]
    assert entry["status"] == "ok" and entry["rows"] > 0
    header = (completed / "predictions" / entry["file"]).read_text().splitlines()[0].split(",")
    for column in ("sample_uid", "group_uid", "source_id", "partition", "target_hfr_px", "pred_hfr_px",
                   "mu", "log_var", "sigma"):
        assert column in header
    with pytest.raises(fs.SealedPartitionError):
        dx.predict(completed, partitions=("test",), out_dir=tmp_path / "never")
    assert not (tmp_path / "never").exists()
    assert dx.main(["predict", "--run", str(completed), "--partition", "test"]) == dx.EXIT_REFUSED


def test_calibrate_is_unavailable_for_a_source_without_calibration_groups(tmp_path):
    hst = fs.write_fixture_source(tmp_path / "data", "fixhst", n_groups=8, rows_per_group=4, style="hst",
                                  role="archive_measurement_reference", calibration="unavailable")
    run = tmp_path / "run"
    dx.fit(_arm(tmp_path, hst, arm="hst_only"), run)
    result = dx.calibrate(run)
    entry = result["sources"]["fixhst"]
    assert entry["status"] == "unavailable" and entry["sigma_scale"] is None
    assert result["test_reported"] is False
    manifest = dx.predict(run, partitions=("calibration",))
    assert manifest["outputs"][0]["status"] == "unavailable"


def test_calibrate_fits_on_calibration_and_reports_val_only(completed):
    entry = dx.calibrate(completed)["sources"]["fixsim"]
    assert entry["status"] == "ok" and entry["fit_on"] == "calibration"
    assert entry["evaluated_on"] == ["val"] and entry["sigma_scale"] > 0
    assert _test_keys(entry) == []


# --------------------------------------------------------------------------- #
# Calibration module: development mode never reports test
# --------------------------------------------------------------------------- #
def test_development_calibration_refuses_test_and_keeps_legacy_contract():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 50)
    ones = np.ones_like(y)
    with pytest.raises(ValueError, match="sealed"):
        calibrate_development_arrays(y * 0, ones, y, evaluations={"test": {"mu": y, "sigma": ones, "y": y}})
    with pytest.raises(ValueError, match="sealed"):
        calibrate_development_arrays(y * 0, ones, y, fit_partition="test")
    record = calibrate_development_arrays(y * 0, ones, y, evaluations={"val": {"mu": y * 0, "sigma": ones, "y": y}})
    assert record["status"] == "ok" and record["test_reported"] is False
    assert _test_keys(record) == []
    empty = calibrate_development_arrays(np.zeros(0), np.zeros(0), np.zeros(0))
    assert empty["status"] == "unavailable" and empty["sigma_scale"] is None
    legacy = calibrate_from_arrays(y * 0, ones, y, y * 0, ones, y, dataset="fixture")  # unchanged signature
    assert "coverage_95_test_after" in legacy


def test_calibrate_cli_function_development_mode(tmp_path):
    from models.focus_quality_cnn import FocusQualityEstimator

    torch.manual_seed(0)
    model_path = tmp_path / "model.pt"
    torch.save(FocusQualityEstimator().state_dict(), model_path)
    out = tmp_path / "calibration.json"
    record = calibrate(str(model_path), None, str(out), True, 0, None, 3, fit_partition="calibration")
    assert record["status"] == "unavailable" and "calibration" in record["reason"]
    record = calibrate(str(model_path), None, str(out), True, 0, None, 3, development_only=True)
    assert record["status"] == "ok" and record["fit_on"] == "val" and record["evaluated_on"] == []
    assert _test_keys(json.loads(out.read_text())) == []


# --------------------------------------------------------------------------- #
# Resume, non-finite guard, output guard, data failures
# --------------------------------------------------------------------------- #
def test_resume_equals_uninterrupted_run_on_cpu(tmp_path, sim):
    config = _arm(tmp_path, sim, updates=6, checkpoint_every=3)
    dx.fit(config, tmp_path / "full")
    with pytest.raises(dx.TrialInterrupted):
        dx.fit(config, tmp_path / "part", stop_after=4)
    record = json.loads((tmp_path / "part" / "run.json").read_text())
    assert record["status"] == "interrupted" and record["updates_completed"] == 3
    with pytest.raises(dx.ExperimentRefusal, match="--resume"):
        dx.fit(config, tmp_path / "part")
    resumed = dx.fit(config, tmp_path / "part", resume=True)
    assert resumed["status"] == "completed" and resumed["resumes"][0]["from_update"] == 3
    full, part = _weights(tmp_path / "full" / "final" / "model_state.pt"), _weights(
        tmp_path / "part" / "final" / "model_state.pt")
    for key in full:
        assert torch.allclose(full[key].double(), part[key].double(), rtol=0, atol=1e-6), key
    assert (tmp_path / "full" / "curves.jsonl").read_text() == (tmp_path / "part" / "curves.jsonl").read_text()
    assert (tmp_path / "part" / "curves.discarded.jsonl").exists()  # replayed update 4 kept as evidence
    lineage = resumed["checkpoints"]
    assert lineage[0]["parent_sha256"] is None and lineage[1]["parent_sha256"] == lineage[0]["sha256"]


def test_resume_refuses_a_changed_config(tmp_path, sim):
    config = _arm(tmp_path, sim, updates=4)
    with pytest.raises(dx.TrialInterrupted):
        dx.fit(config, tmp_path / "run", stop_after=2)
    changed = _arm(tmp_path, sim, name="changed.json", updates=4, lr=5e-4)
    with pytest.raises(dx.ExperimentRefusal, match="config"):
        dx.fit(changed, tmp_path / "run", resume=True)


def test_nonfinite_loss_fails_the_trial_without_a_checkpoint(tmp_path, sim, monkeypatch):
    real = tfq.fqe_objective
    calls = {"n": 0}

    def poisoned(mu, log_var, target, **kwargs):
        calls["n"] += 1
        loss = real(mu, log_var, target, **kwargs)
        return loss * float("nan") if calls["n"] >= 3 else loss

    monkeypatch.setattr(tfq, "fqe_objective", poisoned)
    run = tmp_path / "run"
    with pytest.raises(NonFiniteLossError):
        dx.fit(_arm(tmp_path, sim), run)
    record = json.loads((run / "run.json").read_text())
    assert record["status"] == "failed_nonfinite" and record["last_good_checkpoint"].endswith("000002.pt")
    assert not (run / "final" / "model_state.pt").exists()
    last = json.loads((run / "curves.jsonl").read_text().splitlines()[-1])
    assert last["finite"] is False and last["update"] == 3
    for checkpoint in record["checkpoints"]:
        state = torch.load(run / checkpoint["path"], map_location="cpu", weights_only=True)
        assert tfq.state_is_finite(state["model"])
    with pytest.raises(dx.ExperimentRefusal, match="failed_nonfinite"):
        dx.fit(_arm(tmp_path, sim), run, resume=True)


def test_legacy_train_aborts_on_nonfinite_loss_and_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(tfq, "fqe_objective", lambda mu, lv, y, **kw: (mu * 0).mean() * float("nan"))
    out = tmp_path / "never.pt"
    with pytest.raises(NonFiniteLossError):
        tfq.train({"synthetic": True, "synthetic_runs": 3, "epochs": 1, "out": str(out), "device": "cpu"})
    assert not out.exists()


def test_require_finite_on_tensors_and_arrays():
    assert require_finite(torch.tensor(1.0)) is not None
    with pytest.raises(NonFiniteLossError):
        require_finite(torch.tensor([1.0, float("inf")]))
    with pytest.raises(NonFiniteLossError):
        require_finite(np.array([np.nan]))


def test_outputs_under_models_or_the_data_root_are_refused(tmp_path, sim):
    target = dx.REPO_ROOT / "models" / "focus_quality" / "__fqe_dev_refusal_probe__"
    assert not target.exists()
    with pytest.raises(dx.ExperimentRefusal, match="production model registry"):
        dx.fit(_arm(tmp_path, sim), target)
    assert not target.exists()
    with pytest.raises(dx.ExperimentRefusal, match="data root"):
        dx.refuse_forbidden_output(sim.data_root / "runs", data_root=sim.data_root)
    with pytest.raises(dx.ExperimentRefusal):
        dx.refuse_forbidden_output(dx.ML_ROOT / "data" / "public" / "x")


def _same_directory(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def test_case_variant_of_the_model_registry_is_refused():
    # Pure guard calls only: a regression here must not be able to write into <repo>/models.
    models = dx.REPO_ROOT / "models"
    variant = Path(str(models).swapcase())
    if not _same_directory(variant, models):
        pytest.skip("case-sensitive filesystem: the swapped-case path is a different directory")
    for spelling in (variant / "focus_quality" / "__fqe_dev_refusal_probe__",
                     models.parent / "MODELS" / "focus_quality" / "__fqe_dev_refusal_probe__",
                     Path(str(models.parent).lower()) / "models" / "__fqe_dev_refusal_probe__"):
        with pytest.raises(dx.ExperimentRefusal, match="production model registry"):
            dx.refuse_forbidden_output(spelling)
    assert not (models / "focus_quality" / "__fqe_dev_refusal_probe__").exists()


def test_data_root_aliases_are_refused_by_identity(tmp_path):
    root = tmp_path / "PublicRoot"
    root.mkdir()
    variant = tmp_path / "publicroot"
    if not _same_directory(variant, root):
        pytest.skip("case-sensitive filesystem")
    with pytest.raises(dx.ExperimentRefusal, match="configured data root"):
        dx.refuse_forbidden_output(variant / "run", data_root=root)
    # Unicode-normalization alias (APFS is normalization-insensitive): the
    # case-folded string match cannot see it; only the inode comparison can.
    import unicodedata

    nfc = tmp_path / unicodedata.normalize("NFC", "café")
    nfc.mkdir()
    nfd = tmp_path / unicodedata.normalize("NFD", "café")
    if str(nfd) == str(nfc) or not _same_directory(nfd, nfc):
        pytest.skip("filesystem is normalization-sensitive")
    with pytest.raises(dx.ExperimentRefusal, match="configured data root"):
        dx.refuse_forbidden_output(nfd / "run", data_root=nfc)
    # A sibling that merely shares a prefix is not inside the root.
    assert dx.refuse_forbidden_output(tmp_path / "PublicRootSibling" / "run", data_root=root)


class _SpyNpz:
    """Records which members of a lazily opened NpzFile are read."""

    def __init__(self, inner, log):
        self._inner, self._log = inner, log

    @property
    def files(self):
        return self._inner.files

    def __contains__(self, key):
        return key in self._inner.files

    def __iter__(self):
        return iter(self._inner.files)

    def __getitem__(self, key):
        self._log.append(key)
        return self._inner[key]

    def get(self, key, default=None):
        return self[key] if key in self._inner.files else default

    def close(self):
        self._inner.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._inner.close()


def _archive_npz_without_registry(tmp_path: Path) -> Path:
    directory = tmp_path / "archive_copy"
    directory.mkdir()
    path = directory / "archive.npz"
    n = 12
    np.savez(path, crops=np.zeros((n, 64, 64), np.float32), hfr_px=np.ones(n, np.float32),
             log_hfr=np.zeros(n, np.float32), run_id=np.arange(n), split=np.array(["test"] * n),
             split_group=np.array([f"g{i}" for i in range(n)]),
             run_key_semantics=np.array(["singleton_sample_not_focus_run"] * n))
    return path


def test_by_path_archive_refusal_reads_only_the_semantics_member(tmp_path, monkeypatch):
    from train import calibrate_uncertainty as cu

    path = _archive_npz_without_registry(tmp_path)
    before = sorted(p.name for p in path.parent.iterdir())
    assert before == ["archive.npz"]  # no adjacent .splits.json: a regression would create one
    log = []
    real_load = np.load
    monkeypatch.setattr(np, "load", lambda *a, **k: _SpyNpz(real_load(*a, **k), log))
    with pytest.raises(ValueError, match="singleton_sample_not_focus_run"):
        tfq.train({"dataset": str(path), "lambda_rank": 0.0, "epochs": 1, "device": "cpu"})
    with pytest.raises(ValueError, match="singleton_sample_not_focus_run"):
        cu.calibrate(str(tmp_path / "unused.pt"), str(path), str(tmp_path / "cal.json"), False)
    assert set(log) == {"run_key_semantics"}, log
    assert sorted(p.name for p in path.parent.iterdir()) == before
    assert not (tmp_path / "cal.json").exists()


def test_by_path_route_refuses_a_pinned_public_file_by_hash(tmp_path, monkeypatch):
    path = tmp_path / "copy.npz"
    n = 6
    np.savez(path, crops=np.zeros((n, 64, 64), np.float32), hfr_px=np.ones(n, np.float32),
             log_hfr=np.zeros(n, np.float32), run_id=np.arange(n))
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    class _Pins:
        sources = {"pubsrc": {"role": "synthetic_reference", "files": {"fqe.npz": digest}},
                   "localsrc": {"role": "local_verified_run", "files": {"x.npz": "0" * 64}}}

    monkeypatch.setattr(fs, "load_contract", lambda path=None: _Pins())
    log = []
    real_load = np.load
    monkeypatch.setattr(np, "load", lambda *a, **k: _SpyNpz(real_load(*a, **k), log))
    with pytest.raises(ValueError, match="pinned contract file of public source 'pubsrc'"):
        tfq.train({"dataset": str(path), "lambda_rank": 0.0, "epochs": 1, "device": "cpu"})
    assert log == [] and sorted(p.name for p in tmp_path.iterdir()) == ["copy.npz"]


def test_missing_data_root_is_a_recorded_data_unavailable_trial(tmp_path, sim):
    config = _arm(tmp_path, sim)
    run = tmp_path / "run"
    with pytest.raises(fs.DataUnavailable):
        dx.fit(config, run, data_root=tmp_path / "not-mounted")
    assert json.loads((run / "run.json").read_text())["status"] == "data_unavailable"
    assert not (tmp_path / "not-mounted").exists()
    record = dx.fit(config, run)  # data back: the trial starts, recording the earlier attempt
    assert record["status"] == "completed" and record["earlier_attempts"][0]["status"] == "data_unavailable"


def test_hash_mismatch_and_malformed_arrays_are_rejected_trials(tmp_path, sim):
    config = _arm(tmp_path, sim)
    with (sim.directory / "records.jsonl").open("a") as fh:
        fh.write("\n")
    with pytest.raises(fs.HashMismatch):
        dx.fit(config, tmp_path / "run")
    assert json.loads((tmp_path / "run" / "run.json").read_text())["status"] == "data_rejected"
    bad = fs.write_fixture_source(tmp_path / "bad", "fixbad", n_groups=10, rows_per_group=3,
                                  role="synthetic_reference", nonfinite_rows=tuple(range(18)))
    assert dx.main(["fit", "--config", str(_arm(tmp_path / "bad", bad)), "--out", str(tmp_path / "bad_run")]) \
        == dx.EXIT_REFUSED
    assert json.loads((tmp_path / "bad_run" / "run.json").read_text())["status"] == "data_rejected"


# --------------------------------------------------------------------------- #
# Config contract and the mixed arm
# --------------------------------------------------------------------------- #
def test_config_defaults_mirror_contract_v1_and_reject_contradictions():
    config = dx.resolve_config({"schema": dx.CONFIG_SCHEMA, "arm": "mixed", "seed": 2})
    assert (config["batch_size"], config["updates"], config["warmup_updates"]) == (32, 500, 50)
    assert config["lambda_rank"] == 0.0 and config["lr"] == tfq.ADAMW_LR
    assert config["weight_decay"] == tfq.ADAMW_WEIGHT_DECAY and config["grad_clip_norm"] == tfq.GRAD_CLIP_NORM
    assert config["sources"] == ["hst_acs_corpus_v1", "stellar_sim_v1"]
    assert config["sampling"] == "source_balanced_with_replacement"
    for bad in ({"arm": "everything", "seed": 0}, {"arm": "mixed", "seed": 0, "epochs": 3},
                {"arm": "hst_only", "seed": 0, "sources": ["stellar_sim_v1"]},
                {"arm": "synthetic_only", "seed": -1}):
        with pytest.raises(dx.ExperimentRefusal):
            dx.resolve_config(dict(bad, schema=dx.CONFIG_SCHEMA))


def test_mixed_arm_draws_source_balanced_batches_with_namespaced_identity(tmp_path):
    a = fs.write_fixture_source(tmp_path / "data", "fixsim", n_groups=12, rows_per_group=3, style="sim",
                                role="synthetic_reference", target_definition_id="fixture:sim_target")
    b = fs.write_fixture_source(tmp_path / "data", "fixhst", n_groups=8, rows_per_group=4, style="hst",
                                role="archive_measurement_reference", target_definition_id="fixture:hst_target",
                                base_contract=a.contract)
    # Raw keys and integer run ids overlap between the fixtures; only namespacing separates them.
    undeclared = _arm(tmp_path, b, name="undeclared.json", arm="mixed", sources=["fixsim", "fixhst"])
    with pytest.raises(dx.ExperimentRefusal, match="common reference"):
        dx.fit(undeclared, tmp_path / "undeclared")
    assert not (tmp_path / "undeclared").exists()
    declared = b.contract.with_mixture("fixture_mixed", ["fixsim", "fixhst"], "fixture common reference")
    b.contract = declared
    config = _arm(tmp_path, b, name="mixed.json", arm="mixed", sources=["fixsim", "fixhst"])
    record = dx.fit(config, tmp_path / "mixed")
    assert record["status"] == "completed" and record["sampling"] == "source_balanced_with_replacement"
    rows = [json.loads(line)["rows"] for line in (tmp_path / "mixed" / "curves.jsonl").read_text().splitlines()]
    assert all(r == {"fixhst": 4, "fixsim": 4} for r in rows)
    assert record["data_roles"] == {"fixhst": "archive_measurement_reference", "fixsim": "synthetic_reference"}


# --------------------------------------------------------------------------- #
# Final test: explicit, receipt-bound, once
# --------------------------------------------------------------------------- #
def _receipt(tmp_path: Path, run: Path, source_id: str, **overrides) -> Path:
    record = json.loads((run / "run.json").read_text())
    receipt = {
        "candidate_id": "fixture-candidate",
        "candidate_sha256": record["final_model_sha256"],
        "protocol_id": "fixture-protocol",
        "protocol_sha256": record["config_sha256"],
        "authorized_by": "test fixture",
        "acknowledgement": fs.FINAL_TEST_ACKNOWLEDGEMENT.format(source_id=source_id),
    }
    receipt.update(overrides)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    return path


def test_final_test_requires_flag_and_frozen_receipt_and_runs_once(tmp_path, completed):
    receipt = _receipt(tmp_path, completed, "fixsim")
    with pytest.raises(fs.SealedPartitionError, match="acknowledge"):
        dx.final_test(completed, source_id="fixsim", receipt_path=receipt, acknowledge=False)
    assert dx.main(["final-test", "--run", str(completed), "--source", "fixsim",
                    "--receipt", str(receipt)]) == dx.EXIT_REFUSED
    wrong = _receipt(tmp_path, completed, "fixsim", candidate_sha256="0" * 64)
    with pytest.raises(fs.SealedPartitionError, match="candidate_sha256"):
        dx.final_test(completed, source_id="fixsim", receipt_path=wrong, acknowledge=True)
    assert not (completed / "final_test").exists()

    receipt = _receipt(tmp_path, completed, "fixsim")
    result = dx.final_test(completed, source_id="fixsim", receipt_path=receipt, acknowledge=True)
    assert result["runtime_eligible"] is False and result["metrics"]["rows"] > 0
    out = completed / "final_test" / "fixsim"
    assert (out / "evaluation_receipt.json").exists() and (out / "final_test_receipt__fixsim.json").exists()
    with pytest.raises(fs.SealedPartitionError, match="once"):
        dx.final_test(completed, source_id="fixsim", receipt_path=receipt, acknowledge=True)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a device that is unavailable here")
def test_final_test_setup_errors_do_not_consume_the_sealed_test(tmp_path, completed):
    receipt = _receipt(tmp_path, completed, "fixsim")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        dx.final_test(completed, source_id="fixsim", receipt_path=receipt, acknowledge=True, device="cuda")
    out = completed / "final_test" / "fixsim"
    assert not (out / "final_test_receipt__fixsim.json").exists()
    result = dx.final_test(completed, source_id="fixsim", receipt_path=receipt, acknowledge=True, device="cpu")
    assert result["status"] == "completed" and result["metrics"]["rows"] > 0


def test_final_test_failure_after_loading_writes_a_failed_receipt(tmp_path, completed, monkeypatch):
    from train import calibrate_uncertainty as cu

    real = cu.predict_mu_log_var
    calls = []

    def flaky(model, crops, device, *args, **kwargs):
        calls.append(len(crops))
        if len(calls) > 1:  # the dummy probe passes; the sealed rows fail (e.g. MPS OOM)
            raise MemoryError("simulated device failure")
        return real(model, crops, device, *args, **kwargs)

    monkeypatch.setattr(cu, "predict_mu_log_var", flaky)
    receipt = _receipt(tmp_path, completed, "fixsim")
    with pytest.raises(MemoryError):
        dx.final_test(completed, source_id="fixsim", receipt_path=receipt, acknowledge=True)
    assert calls[0] == 1
    out = completed / "final_test" / "fixsim"
    failed = json.loads((out / "evaluation_receipt.json").read_text())
    assert failed["status"] == "failed" and "simulated device failure" in failed["error"]
    assert failed["runtime_eligible"] is False and "metrics" not in failed
    assert json.loads((out / "final_test_receipt__fixsim.json").read_text())["status"] == "loaded"


def test_final_test_and_development_nll_share_one_convention():
    from train.calibrate_uncertainty import NLL_CONVENTION, gaussian_nll

    rng = np.random.default_rng(3)
    target = np.exp(rng.normal(1.0, 0.2, 40))
    mu = np.log(target) + rng.normal(0, 0.1, 40)
    log_var = rng.normal(-3.0, 0.5, 40)
    metrics = dx._point_metrics(target, mu, log_var)
    sigma = np.exp(0.5 * np.clip(log_var, -8, 8))
    assert metrics["raw_nll"] == pytest.approx(gaussian_nll(mu, sigma, np.log(target)), rel=1e-12)
    dev = calibrate_development_arrays(mu, sigma, np.log(target),
                                       evaluations={"val": {"mu": mu, "sigma": sigma, "y": np.log(target)}})
    assert dev["evaluation"]["val"]["nll_before"] == pytest.approx(metrics["raw_nll"], rel=1e-12)
    assert metrics["nll_convention"] == dev["nll_convention"] == NLL_CONVENTION


def test_tracked_contract_keeps_the_final_test_disabled():
    contract = fs.load_contract()
    for source_id in ("stellar_sim_v1", "hst_acs_corpus_v1"):
        assert contract.source(source_id)["final_test"]["enabled"] is False
