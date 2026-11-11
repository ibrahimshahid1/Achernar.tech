"""eval.fqe_dev_report: M4 metrics, F-010 selection, ONNX parity and F-006 gate evidence (fixtures only).

Generated tmp-dir sources (``fqe_sources.write_fixture_source``), a few CPU
optimizer updates per trial, hand-written classical rows in the
``fqe_benchmark`` row format (plus one real ``fqe_benchmark`` run on the
fixtures). No public data, no test partition, nothing under ``models/``.
"""

import copy
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

import data_pipeline.fqe_sources as fs  # noqa: E402
import eval.fqe_benchmark as fb  # noqa: E402
import eval.fqe_dev_report as rep  # noqa: E402
import pipelines.rebuild_fqe as rebuild  # noqa: E402
import train.fqe_dev_experiment as dx  # noqa: E402
from data_pipeline.fqe_sources import SealedPartitionError  # noqa: E402

BUDGET = {"batch_size": 8, "updates": 2, "warmup_updates": 1, "lambda_rank": 0.0, "lr": 1e-3,
          "weight_decay": 1e-4, "grad_clip_norm": 1.0, "lr_schedule": "cosine"}
SIM, HST = "fixsim", "fixhst"
RUST_G1 = "native_rust__rust_legacy__g1__sim_observation64__pixel_flags"


# --------------------------------------------------------------------------- #
# Fixture world: two sources, four tiny trials (synthetic_only x 3 seeds, hst_only x 1)
# --------------------------------------------------------------------------- #
def _arm_config(path: Path, world: dict, arm: str, seed: int, **overrides) -> Path:
    config = {"schema": dx.CONFIG_SCHEMA, "arm": arm, "seed": seed, "batch_size": BUDGET["batch_size"],
              "updates": BUDGET["updates"], "warmup_updates": BUDGET["warmup_updates"], "checkpoint_every": 2,
              "device": "cpu", "contract_path": str(world["contract_file"]), "data_root": str(world["root"]),
              "sources": [SIM] if arm == "synthetic_only" else [HST], "eval_sources": [HST, SIM]}
    config.update(overrides)
    path.write_text(json.dumps(config))
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("fqe_dev_report")
    root = base / "data"
    sim = fs.write_fixture_source(root, SIM, n_groups=20, rows_per_group=3, style="sim",
                                  role="synthetic_reference", target_definition_id="fixture:sim_target")
    hst = fs.write_fixture_source(root, HST, n_groups=12, rows_per_group=4, style="hst",
                                  role="archive_measurement_reference", calibration="unavailable",
                                  target_definition_id="fixture:hst_target", base_contract=sim.contract)
    contract_file = base / "contract.json"
    declared = hst.contract.with_mixture("fixture_mixed", [SIM, HST], "fixture common reference")
    contract_file.write_text(json.dumps(declared.data))
    contract = fs.load_contract(contract_file)
    w = {"base": base, "root": root, "contract_file": contract_file, "contract": contract, "trials": {}}
    for arm, seed in (("synthetic_only", 0), ("synthetic_only", 1), ("synthetic_only", 2), ("hst_only", 0)):
        run = base / "trials" / f"{arm}-s{seed}"
        dx.fit(_arm_config(base / f"{arm}-s{seed}.json", w, arm, seed), run)
        dx.predict(run, partitions=("val", "calibration"))
        dx.calibrate(run)
        w["trials"][f"{arm}-s{seed}"] = run
    return w


def _arrays(world, source_id):
    return fs.load_source(source_id, partitions=("train_fit", "val"), contract=world["contract"],
                          data_root=world["root"])


def write_classical(world, out: Path, *, gate_offset: float, gate_measured: float = 0.75,
                    row_hook=None, config_hook=None) -> Path:
    """A classical (M2-format) run over the fixture val populations, rows built with fqe_benchmark.make_row."""
    out.mkdir(parents=True)
    rows, sources = [], {}
    arms = {"fixsim_only": SIM, "fixhst_only": HST}
    medians = {arm: float(np.median(_arrays(world, sid).target_hfr_px[_arrays(world, sid).partition_indices("train_fit")]))
               for arm, sid in arms.items()}
    for source_id in (SIM, HST):
        arrays = _arrays(world, source_id)
        val = arrays.partition_indices("val")
        constants = [fb.ComparatorSpec(comparator_id=f"constant__{arm}", family="constant",
                                       method="train_fit_target_median", processing_mode="training_fit_constant",
                                       input_definition=fb.INPUT_NONE, measurement_role="fitted_baseline", arm=arm)
                     for arm in arms]
        n_measured = int(round(gate_measured * val.size))
        for k, i in enumerate(val.tolist()):
            target = float(arrays.target_hfr_px[i])
            ok = k < n_measured
            rows.append(fb.make_row(fb.NORMALIZED_SPEC, arrays, i, prediction=target + gate_offset if ok else None,
                                    status="measured" if ok else "unmeasured",
                                    reason=None if ok else "no_core_above_3sigma", timing_ms=0.3))
            for spec in constants:
                rows.append(fb.make_row(spec, arrays, i, prediction=medians[spec.arm], status="measured",
                                        timing_ms=0.0))
            if source_id == SIM:
                rust = fb.ComparatorSpec(comparator_id=RUST_G1, family="native_rust", method="rust_legacy__g1",
                                         processing_mode="native_linear_64x64_observation",
                                         input_definition=fb.INPUT_SIM_NATIVE, measurement_role="deployable_classical",
                                         context="sim_observation64", mask_policy="pixel_flags")
                ok = k != 0
                rows.append(fb.make_row(rust, arrays, i, prediction=0.9 * target if ok else None,
                                        status="measured" if ok else "unmeasured",
                                        reason=None if ok else "gate_rejected"))
        ident = {key: arrays.identity.get(key) for key in ("role", "contract_sha256", "files", "target_definition_id")}
        ident["eval_population"] = {"partition": "val", "rows": int(val.size),
                                    "groups": len(set(arrays.group_uid[val].tolist())),
                                    "membership_sha256": rep._membership_sha256(arrays.sample_uid[val])}
        sources[source_id] = ident
    config = {"schema": fb.SCHEMA, "partition": "val", "runtime_eligible": False, "sources": sources,
              "arms": {"fixsim_only": {"sources": [SIM]}, "fixhst_only": {"sources": [HST]}}}
    if config_hook:
        config_hook(config)
    if row_hook:
        rows = row_hook(rows)
    (out / "config.json").write_text(json.dumps(fb._clean(config)))
    (out / "rows.jsonl").write_text("".join(json.dumps(fb._clean(r)) + "\n" for r in rows))
    return out


def _report(world, classical: Path, out: Path, trials=None, **kwargs) -> dict:
    trials = trials if trials is not None else list(world["trials"].values())
    return rep.build_report(trials, classical, out, contract=world["contract"], budget=kwargs.pop("budget", BUDGET),
                            **kwargs)


def _copy_trial(world, name: str, dest: Path) -> Path:
    shutil.copytree(world["trials"][name], dest)
    for path in dest.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o644)
    return dest


def _export(run: Path, dest: Path) -> Path:
    from export.export_quality_onnx import export

    onnx_path = dest / f"{run.name}.onnx"
    export(str(run / "final" / "model_state.pt"), str(onnx_path))
    return onnx_path


# --------------------------------------------------------------------------- #
# Metric math on hand-computed rows
# --------------------------------------------------------------------------- #
def _csv_row(uid, group, target, mu, log_var, status="ok"):
    return {"sample_uid": uid, "group_uid": group, "frame_uid": uid, "target_hfr_px": repr(target),
            "log_target": repr(math.log(target)), "mu": repr(mu), "log_var": repr(log_var), "status": status,
            "secondary_measured_hfr_px": repr(target)}


def test_candidate_rows_and_point_metrics_hand_values():
    trial = {"trial_id": "synthetic_only-s0", "comparator_id": "fqe__synthetic_only__s0", "arm": "synthetic_only",
             "seed": 0}
    csv_rows = [_csv_row("s:a", "s:g1", 1.0, math.log(1.5), 0.0), _csv_row("s:b", "s:g1", 2.0, math.log(2.0), 0.0),
                _csv_row("s:c", "s:g2", 4.0, float("nan"), 0.0, status="nonfinite_prediction"),
                _csv_row("s:d", "s:g2", 2.0, math.log(1.0), 0.0)]
    reference = {r["sample_uid"]: {"target": float(r["target_hfr_px"]), "strata": {"star_snr": 30.0}}
                 for r in csv_rows}
    rows = rep.candidate_rows(trial, "s", csv_rows, reference)
    assert [r["status"] for r in rows] == ["measured", "measured", "unmeasured", "measured"]
    assert rows[2]["prediction"] is None and rows[2]["reason"] == "nonfinite_prediction"
    s = fb.summarize_rows(rows)
    # errors on measured rows: +0.5, 0, -1 (px); the NaN row only lowers coverage.
    assert s["coverage"] == pytest.approx(0.75) and s["measured"] == 3
    m = s["metrics"]
    assert m["hfr_mae_px"] == pytest.approx(0.5)
    assert m["signed_bias_px"] == pytest.approx(-1.0 / 6.0)
    assert m["log_hfr_mae"] == pytest.approx((math.log(1.5) + math.log(2.0)) / 3.0)
    assert m["median_relative_error"] == pytest.approx(0.5)
    groups = rep.group_summary(rows, level="family")
    assert groups["n_groups"] == 2 and groups["per_group"]["s:g1"]["hfr_mae_px"] == pytest.approx(0.25)
    assert groups["per_group"]["s:g2"]["hfr_mae_px"] == pytest.approx(1.0)
    assert groups["per_group"]["s:g2"]["coverage"] == pytest.approx(0.5)
    assert groups["group_mean_hfr_mae_px"] == pytest.approx(0.625)
    # a target that disagrees with the classical population is a data-identity error
    with pytest.raises(rep.ReportRefusal, match="target"):
        rep.candidate_rows(trial, "s", csv_rows, dict(reference, **{"s:a": {"target": 1.1}}))


def test_raw_uncertainty_hand_values_and_nll_convention():
    rows = [{"mu": 0.0, "log_var": 0.0, "log_target": 0.5},             # sigma 1, z 0.5
            {"mu": 0.0, "log_var": 2 * math.log(2.0), "log_target": 3.0},  # sigma 2, z 1.5
            {"mu": float("nan"), "log_var": 0.0, "log_target": 1.0}]
    out = rep.raw_uncertainty(rows)
    assert out["rows"] == 3 and out["finite_rows"] == 2 and out["calibrated"] is False
    assert out["nll_convention"] == rep.NLL_CONVENTION
    assert out["nll"] == pytest.approx((0.5 * 0.25 + 0.0 + 0.5 * 2.25 + math.log(2.0)) / 2.0)
    assert out["coverage_68"] == pytest.approx(0.5)
    assert out["coverage_95"] == pytest.approx(1.0)


def test_spread_keeps_seeds_and_describes():
    out = rep.spread({0: 0.3, 1: 0.1, 2: float("nan")})
    assert out["per_seed"]["0"] == 0.3 and out["n"] == 2
    assert (out["min"], out["median"], out["max"]) == (0.1, pytest.approx(0.2), 0.3)
    assert rep.spread({0: None})["median"] == rep.UNAVAILABLE


# --------------------------------------------------------------------------- #
# Report on fixture trials
# --------------------------------------------------------------------------- #
def test_report_end_to_end_on_fixture_trials(world, tmp_path):
    classical = write_classical(world, tmp_path / "m2", gate_offset=50.0)
    report = _report(world, classical, tmp_path / "report")
    out = tmp_path / "report"
    assert (out / "report.json").is_file() and (out / "report.md").is_file()
    assert (out / "candidate_rows.jsonl").is_file()
    on_disk = json.loads((out / "report.json").read_text())
    assert on_disk["test_partition_used"] is False and on_disk["runtime_eligible"] is False
    assert on_disk["sources"][SIM]["role"] == "synthetic_reference"
    assert "repeat one field" in on_disk["sources"][HST]["interpretation"]
    trial = next(t for t in on_disk["trials"] if t["trial_id"] == "synthetic_only-s0")
    assert trial["status"] == "completed" and trial["training_finite"] is True
    sim_block = trial["sources"][SIM]

    # Independent recomputation from the predictions CSV.
    csv_rows = rep._read_csv(world["trials"]["synthetic_only-s0"] / "predictions" / f"{SIM}__val.csv")
    pred = np.exp([float(r["mu"]) for r in csv_rows])
    target = np.array([float(r["target_hfr_px"]) for r in csv_rows])
    summary = sim_block["summary"]["target"]
    assert summary["population"] == len(csv_rows) and summary["coverage"] == 1.0
    assert summary["metrics"]["hfr_mae_px"] == pytest.approx(float(np.mean(np.abs(pred - target))))
    assert summary["metrics"]["signed_bias_px"] == pytest.approx(float(np.mean(pred - target)))
    assert "measured_hfr_px" in sim_block["summary"]
    assert sim_block["sweep_rank_metrics"].startswith("unavailable")
    assert "width_bin" in sim_block["strata"] and "star_snr_bin" in sim_block["strata"]
    assert sim_block["groups"]["level"] == "family" and sim_block["groups"]["n_groups"] == 3

    # Paired on the common valid rows, full-population coverage for both methods.
    paired = sim_block["paired"][rep.GATE_COMPARATOR]
    n_val = len(csv_rows)
    n_measured = int(round(0.75 * n_val))
    assert paired["population"] == n_val and paired["common_valid"] == n_measured
    per = paired["per_comparator"]
    assert per["fqe__synthetic_only__s0"]["coverage_full_population"] == 1.0
    assert per[rep.GATE_COMPARATOR]["coverage_full_population"] == pytest.approx(n_measured / n_val)
    assert per[rep.GATE_COMPARATOR]["common_valid_metrics"]["hfr_mae_px"] == pytest.approx(50.0)
    assert set(sim_block["paired"]) == {rep.GATE_COMPARATOR, "constant__fixsim_only", "constant__fixhst_only", RUST_G1}

    # Uncertainty: raw on val; calibrated only for the synthetic source, equal to the trial's calibrate output.
    cal = json.loads((world["trials"]["synthetic_only-s0"] / "calibration" / "calibration.json").read_text())
    calibrated = sim_block["uncertainty"]["calibrated"]
    assert calibrated["status"] == "ok"
    assert calibrated["coverage_95_after"] == cal["sources"][SIM]["evaluation"]["val"]["coverage_95_after"]
    assert sim_block["uncertainty"]["raw"]["nll"] == pytest.approx(cal["sources"][SIM]["evaluation"]["val"]["nll_before"])
    assert trial["sources"][HST]["uncertainty"]["calibrated"]["status"] == "not_reported"
    assert trial["sources"][HST]["groups"]["level"] == "exposure (date)"

    # Arm summary keeps every seed plus min/median/max.
    arm = on_disk["arms"]["synthetic_only"]
    assert arm["seeds"] == [0, 1, 2] and arm["classical_arm"] == "fixsim_only"
    mae = arm["sources"][SIM]["metrics"]["hfr_mae_px"]
    assert set(mae["per_seed"]) == {"0", "1", "2"} and mae["min"] <= mae["median"] <= mae["max"]
    assert "per_group_hfr_mae_px" in arm["sources"][SIM]["groups"]
    md = (out / "report.md").read_text()
    assert "Synthetic latent truth: fixsim" in md and "HST measurement-reference agreement: fixhst" in md
    assert report["candidate_rows"]["rows"] == 4 * (n_val + trial["sources"][HST]["summary"]["target"]["population"])


def test_report_reads_a_real_fixture_benchmark_run(world, tmp_path):
    result = fb.run_benchmark([SIM, HST], out_dir=tmp_path / "m2", methods=["constant", "normalized_input", "sharpness"],
                              contract=world["contract"], data_root=world["root"], run_rust=False)
    assert result["config"]["partition"] == "val"
    report = _report(world, tmp_path / "m2", tmp_path / "report")
    block = next(t for t in report["trials"] if t["trial_id"] == "hst_only-s0")["sources"][HST]
    assert "sharpness__tenengrad__fixhst_only" in block["paired"] and rep.GATE_COMPARATOR in block["paired"]
    assert report["arms"]["hst_only"]["classical_arm"] == "fixhst_only"


# --------------------------------------------------------------------------- #
# Refusals: test partitions, identity, budget
# --------------------------------------------------------------------------- #
def test_report_refuses_test_partitions_in_classical_inputs(world, tmp_path):
    def test_row(rows):
        rows[0] = dict(rows[0], partition="test")
        return rows

    bad_rows = write_classical(world, tmp_path / "m2_rows", gate_offset=1.0, row_hook=test_row)
    with pytest.raises(SealedPartitionError):
        _report(world, bad_rows, tmp_path / "r1")
    bad_config = write_classical(world, tmp_path / "m2_cfg", gate_offset=1.0,
                                 config_hook=lambda c: c.update(partition="test"))
    with pytest.raises(SealedPartitionError):
        _report(world, bad_config, tmp_path / "r2")
    assert not (tmp_path / "r1").exists() and not (tmp_path / "r2").exists()
    code = rep.main(["report", "--trials", str(world["trials"]["synthetic_only-s0"]), "--classical", str(bad_rows),
                     "--out", str(tmp_path / "r3"), "--contract", str(world["contract_file"]),
                     "--budget-json", json.dumps(BUDGET)])
    assert code == rep.EXIT_REFUSED


def test_report_refuses_test_partitions_in_trial_predictions(world, tmp_path):
    classical = write_classical(world, tmp_path / "m2", gate_offset=1.0)
    # manifest names the test partition
    t1 = _copy_trial(world, "synthetic_only-s0", tmp_path / "t1")
    manifest = json.loads((t1 / "predictions" / "manifest.json").read_text())
    manifest["partitions"] = ["val", "test"]
    (t1 / "predictions" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SealedPartitionError):
        _report(world, classical, tmp_path / "r1", trials=[t1])
    # a test prediction file sits next to the development files
    t2 = _copy_trial(world, "synthetic_only-s0", tmp_path / "t2")
    shutil.copy(t2 / "predictions" / f"{SIM}__val.csv", t2 / "predictions" / f"{SIM}__test.csv")
    with pytest.raises(SealedPartitionError):
        _report(world, classical, tmp_path / "r2", trials=[t2])
    # a row labelled test inside a val file (manifest hash updated so only the row check can catch it)
    t3 = _copy_trial(world, "synthetic_only-s0", tmp_path / "t3")
    path = t3 / "predictions" / f"{SIM}__val.csv"
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace(",val,", ",test,", 1)
    path.write_text("\n".join(lines) + "\n")
    manifest = json.loads((t3 / "predictions" / "manifest.json").read_text())
    for entry in manifest["outputs"]:
        if entry.get("file") == path.name:
            entry["sha256"] = rep._sha256_file(path)
    (t3 / "predictions" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SealedPartitionError):
        _report(world, classical, tmp_path / "r3", trials=[t3])


def test_report_refuses_identity_mismatches(world, tmp_path):
    classical = write_classical(world, tmp_path / "m2", gate_offset=1.0)
    # a different frozen contract (one more declared source: another canonical sha256). The classical run is
    # loaded before any trial, so this is refused at the classical run's identity.
    other = world["contract"].with_source("extra", world["contract"].source(SIM))
    with pytest.raises(rep.ReportRefusal, match="classical run .* read under contract"):
        rep.build_report([world["trials"]["synthetic_only-s0"]], classical, tmp_path / "r1", contract=other,
                         budget=BUDGET)
    # the classical run is valid but the trial's config names another contract (config.sha256 rewritten, so
    # only the trial identity check can catch it)
    moved = _copy_trial(world, "synthetic_only-s0", tmp_path / "moved")
    config = json.loads((moved / "config.json").read_text())
    config["contract_sha256"] = "e" * 64
    raw = json.dumps(config).encode("utf-8")
    (moved / "config.json").write_bytes(raw)
    (moved / "config.sha256").write_text(rep._sha256_bytes(raw) + "\n")
    with pytest.raises(rep.ReportRefusal, match=f"config contract_sha256 {'e' * 64} is not the frozen contract"):
        _report(world, classical, tmp_path / "r1b", trials=[moved])
    assert not (tmp_path / "r1b").exists()
    # re-pinned training data recorded by the trial
    t = _copy_trial(world, "synthetic_only-s0", tmp_path / "t")
    identity = json.loads((t / "dataset_identity.json").read_text())
    name = sorted(identity["sources"][SIM]["files"])[0]
    identity["sources"][SIM]["files"][name] = "f" * 64
    (t / "dataset_identity.json").write_text(json.dumps(identity))
    with pytest.raises(rep.ReportRefusal, match="pinned file"):
        _report(world, classical, tmp_path / "r2", trials=[t])
    # a classical run read under other pins
    def repin(config):
        config["sources"][SIM]["files"] = dict(config["sources"][SIM]["files"], **{name: "0" * 64})

    bad = write_classical(world, tmp_path / "m2_bad", gate_offset=1.0, config_hook=repin)
    with pytest.raises(rep.ReportRefusal, match="pinned file"):
        _report(world, bad, tmp_path / "r3")
    # the same arm/seed twice
    duplicate = _copy_trial(world, "synthetic_only-s0", tmp_path / "dup")
    with pytest.raises(rep.ReportRefusal, match="given twice"):
        _report(world, classical, tmp_path / "r4", trials=[world["trials"]["synthetic_only-s0"], duplicate])
    # val predictions on another population than the classical one (a row dropped, manifest re-hashed):
    # refused after loading, and no partial report directory is left behind
    short = _copy_trial(world, "synthetic_only-s0", tmp_path / "short")
    path = short / "predictions" / f"{SIM}__val.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    manifest = json.loads((short / "predictions" / "manifest.json").read_text())
    for entry in manifest["outputs"]:
        if entry.get("file") == path.name:
            entry["sha256"] = rep._sha256_file(path)
    (short / "predictions" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(rep.ReportRefusal, match="membership"):
        _report(world, classical, tmp_path / "r5", trials=[short])
    assert not (tmp_path / "r5").exists()


def test_report_refuses_budget_mismatch_and_frozen_matrix_overrides(world, tmp_path):
    classical = write_classical(world, tmp_path / "m2", gate_offset=1.0)
    with pytest.raises(rep.ReportRefusal, match="budget differs"):
        _report(world, classical, tmp_path / "r1", budget=dict(BUDGET, updates=3))
    with pytest.raises(rep.ReportRefusal, match="budget differs"):
        _report(world, classical, tmp_path / "r2", budget=dict(BUDGET, lr=5e-4))
    # The tracked contract's matrix is frozen: no overrides, and its budget is contract-v1's.
    tracked = fs.load_contract()
    with pytest.raises(rep.ReportRefusal, match="frozen"):
        rep.resolve_matrix(tracked, budget=BUDGET)
    frozen = rep.resolve_matrix(tracked)
    assert frozen["budget"]["updates"] == 500 and frozen["budget"]["batch_size"] == 32
    assert frozen["seeds"] == [0, 1, 2] and frozen["tracked_contract"] is True
    code = rep.main(["report", "--trials", str(world["trials"]["synthetic_only-s0"]), "--classical", str(classical),
                     "--out", str(tmp_path / "r3"), "--budget-json", json.dumps(BUDGET)])
    assert code == rep.EXIT_REFUSED


def test_matrix_configs_bind_each_trial_to_the_file_it_was_fit_from(world, tmp_path):
    classical = write_classical(world, tmp_path / "m2", gate_offset=1.0)
    syn = [world["trials"][f"synthetic_only-s{s}"] for s in (0, 1, 2)]
    configs = tmp_path / "arms"
    configs.mkdir()
    for s in (0, 1, 2):
        shutil.copy(world["base"] / f"synthetic_only-s{s}.json", configs)
    kwargs = dict(contract=world["contract"], budget=BUDGET, arms=["synthetic_only"], seeds=[0, 1, 2])
    report = rep.build_report(syn, classical, tmp_path / "r0", matrix_configs=configs, **kwargs)
    assert set(report["matrix"]["arm_configs"]) == {"synthetic_only-s0", "synthetic_only-s1", "synthetic_only-s2"}
    assert all(t["matrix_config"]["verified"] is True for t in report["trials"])
    assert "arm configs verified" in (tmp_path / "r0" / "report.md").read_text()
    # Same budget, different file (an extra key): the trial was not fit from this matrix config.
    raw = json.loads((configs / "synthetic_only-s1.json").read_text())
    (configs / "synthetic_only-s1.json").write_text(json.dumps(dict(raw, checkpoint_every=1)))
    with pytest.raises(rep.ReportRefusal, match="not the matrix config"):
        rep.build_report(syn, classical, tmp_path / "r1", matrix_configs=configs, **kwargs)
    # A matrix config whose own budget differs from the matrix budget.
    (configs / "synthetic_only-s1.json").write_text(json.dumps(dict(raw, updates=3)))
    with pytest.raises(rep.ReportRefusal, match="budget differs"):
        rep.build_report(syn, classical, tmp_path / "r2", matrix_configs=configs, **kwargs)
    # Every arm x seed of the matrix needs its config file.
    (configs / "synthetic_only-s1.json").write_text(json.dumps(raw))
    (configs / "synthetic_only-s2.json").unlink()
    with pytest.raises(rep.ReportRefusal, match="missing"):
        rep.build_report(syn, classical, tmp_path / "r3", matrix_configs=configs, **kwargs)
    assert not any((tmp_path / name).exists() for name in ("r1", "r2", "r3"))


def test_matrix_contract_must_be_the_frozen_experiment_contract(world, tmp_path):
    contract = world["contract"].with_source(SIM, world["contract"].source(SIM))
    data = copy.deepcopy(contract.data)
    v1 = {"training": {"batch_size": 8, "updates": 2, "mean_only_warmup_updates": 1, "lambda_rank": 0,
                       "arms": ["synthetic_only", "hst_only", "mixed"], "seeds": [0, 1, 2]}}
    path = tmp_path / "contract-v1.json"
    path.write_text(json.dumps(v1))
    data["derived_from"] = {"experiment_contract_sha256": rep._sha256_file(path)}
    bound = fs._contract_from_data(data)
    matrix = rep.resolve_matrix(bound, budget=BUDGET, matrix_contract=path)
    assert matrix["matrix_contract"]["sha256"] == rep._sha256_file(path)
    with pytest.raises(rep.ReportRefusal, match="disagrees"):
        rep.resolve_matrix(bound, budget=dict(BUDGET, updates=4), matrix_contract=path)
    path.write_text(json.dumps(v1) + " ")
    with pytest.raises(rep.ReportRefusal, match="not the frozen experiment contract"):
        rep.resolve_matrix(bound, budget=BUDGET, matrix_contract=path)


# --------------------------------------------------------------------------- #
# F-010 selection
# --------------------------------------------------------------------------- #
def _trial(arm, seed, sim_log, hst_log, *, status="completed", training_finite=True, finite=(True, True)):
    def block(value, ok):
        return {"predictions_finite": ok, "summary": {"target": {"metrics": {"log_hfr_mae": value}}}}

    return {"arm": arm, "seed": seed, "trial_id": f"{arm}-s{seed}", "status": status,
            "training_finite": training_finite, "model_sha256": f"{arm}{seed}".ljust(64, "0"),
            "config_sha256": f"c{arm}{seed}".ljust(64, "0"), "run_dir": f"/runs/{arm}-s{seed}",
            "comparator_id": f"fqe__{arm}__s{seed}",
            "sources": {"sim": block(sim_log, finite[0]), "hst": block(hst_log, finite[1])}}


def _synthetic_report(trials):
    return {"schema": rep.REPORT_SCHEMA, "test_partition_used": False, "contract": {"sha256": "c" * 64},
            "matrix": {"arms": ["synthetic_only", "hst_only", "mixed"], "seeds": [0, 1, 2]},
            "sources": {"sim": {"role": "synthetic_reference"}, "hst": {"role": "archive_measurement_reference"}},
            "trials": trials}


def test_f010_lowest_mean_arm_and_median_seed_with_ineligible_arms():
    trials = [
        _trial("synthetic_only", 0, 0.10, 0.30),  # 0.20
        _trial("synthetic_only", 1, 0.20, 0.40),  # 0.30
        _trial("synthetic_only", 2, 0.05, 0.15),  # 0.10   arm mean 0.20
        _trial("hst_only", 0, 0.20, 0.10),        # 0.15
        _trial("hst_only", 1, 0.30, 0.04),        # 0.17 (median)
        _trial("hst_only", 2, 0.12, 0.26),        # 0.19   arm mean 0.17
        _trial("mixed", 0, 0.01, 0.01),
        _trial("mixed", 1, 0.01, 0.01),
        _trial("mixed", 2, 0.01, float("nan"), finite=(True, False)),  # best scores, but not finite
    ]
    out = rep.select_candidate(_synthetic_report(trials), report_sha256="r" * 64)
    assert out["status"] == "selected" and out["selected_arm"] == "hst_only"
    cand = out["candidate"]
    assert cand["seed"] == 1 and cand["trial_id"] == "hst_only-s1"  # median, not the best seed (s0)
    assert cand["score"] == pytest.approx(0.17) and cand["arm_score"] == pytest.approx(0.17)
    ranked = [(e["arm"], e["eligible"], e.get("rank")) for e in out["ranking"]]
    assert ranked == [("hst_only", True, 1), ("synthetic_only", True, 2), ("mixed", False, None)]
    mixed = out["ranking"][2]
    assert any("non-finite predictions on hst" in r for r in mixed["reasons"])
    syn = out["ranking"][1]
    assert syn["score"] == pytest.approx(0.20)
    assert [s["score"] for s in syn["seeds"]] == [pytest.approx(0.20), pytest.approx(0.30), pytest.approx(0.10)]
    assert out["report_sha256"] == "r" * 64 and "F-010" in out["rule"]


def test_f010_missing_failed_or_nonfinite_seeds_make_an_arm_ineligible():
    trials = [
        _trial("synthetic_only", 0, 0.1, 0.1), _trial("synthetic_only", 1, 0.1, 0.1),  # seed 2 missing
        _trial("hst_only", 0, 0.1, 0.1), _trial("hst_only", 1, 0.1, 0.1),
        _trial("hst_only", 2, None, None, status="failed_nonfinite", training_finite=False),
        _trial("mixed", 0, 0.1, 0.1), _trial("mixed", 1, 0.1, 0.1),
        _trial("mixed", 2, 0.1, 0.1, training_finite=False),
    ]
    out = rep.select_candidate(_synthetic_report(trials))
    assert out["status"] == "no_eligible_arm" and out["candidate"] is None and out["selected_arm"] is None
    reasons = {e["arm"]: " ".join(e["reasons"]) for e in out["ranking"]}
    assert "trial not reported" in reasons["synthetic_only"]
    assert "status 'failed_nonfinite'" in reasons["hst_only"]
    assert "training not finite" in reasons["mixed"]


def test_f010_seed_ties_break_by_seed_and_are_recorded():
    trials = [_trial("synthetic_only", s, 0.2, 0.2) for s in (2, 0, 1)]
    trials += [_trial("hst_only", s, 0.5, 0.5) for s in (0, 1, 2)]
    out = rep.select_candidate(_synthetic_report(trials))
    assert out["candidate"]["seed"] == 1 and out["selected_arm"] == "synthetic_only"
    assert any("seed score tie" in note for note in out["notes"])
    with pytest.raises(rep.ReportRefusal, match="exactly one"):
        rep.select_candidate(dict(_synthetic_report(trials), sources={"sim": {"role": "synthetic_reference"}}))


def test_select_cli_writes_the_ranking_and_fails_without_an_eligible_arm(tmp_path):
    trials = [_trial("synthetic_only", s, 0.1, 0.1, training_finite=(s != 1)) for s in (0, 1, 2)]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_synthetic_report(trials)))
    assert rep.main(["select", "--report", str(path)]) == rep.EXIT_NO_CANDIDATE
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["status"] == "no_eligible_arm" and selection["candidate"] is None
    assert [e["arm"] for e in selection["ranking"]] == ["synthetic_only", "hst_only", "mixed"]
    assert all(e["eligible"] is False for e in selection["ranking"])
    path.write_text(json.dumps(dict(_synthetic_report(trials), test_partition_used=True)))
    assert rep.main(["select", "--report", str(path), "--out", str(tmp_path / "s2.json")]) == rep.EXIT_REFUSED


def test_select_on_a_fixture_report(world, tmp_path):
    classical = write_classical(world, tmp_path / "m2", gate_offset=50.0)
    _report(world, classical, tmp_path / "report")
    selection = rep.select_file(tmp_path / "report" / "report.json")
    assert Path(selection["_path"]) == (tmp_path / "report" / "selection.json").resolve()
    report = json.loads((tmp_path / "report" / "report.json").read_text())
    scores = {}
    for t in report["trials"]:
        if t["arm"] == "synthetic_only":
            logs = [t["sources"][s]["summary"]["target"]["metrics"]["log_hfr_mae"] for s in (SIM, HST)]
            scores[t["seed"]] = sum(logs) / 2.0
    median_seed = sorted(scores, key=lambda s: (scores[s], s))[1]
    assert selection["selected_arm"] == "synthetic_only" and selection["candidate"]["seed"] == median_seed
    by_arm = {e["arm"]: e for e in selection["ranking"]}
    assert by_arm["hst_only"]["eligible"] is False and by_arm["mixed"]["eligible"] is False
    assert selection["report_sha256"] == rep._sha256_file(tmp_path / "report" / "report.json")


# --------------------------------------------------------------------------- #
# ONNX parity on a tiny exported model
# --------------------------------------------------------------------------- #
def test_parity_on_tiny_exported_model(world, tmp_path):
    run = world["trials"]["synthetic_only-s0"]
    onnx_path = _export(run, tmp_path)
    result = rep.parity_file(run, onnx_path, tmp_path / "parity.json")
    assert result["passed"] is True and result["failures"] == []
    assert result["max_abs_delta_mu"] <= 1e-4 and result["max_abs_delta_log_var"] <= 1e-4
    assert set(result["fixtures"]) == {"random_uniform_seed0", "ramp"}
    assert set(result["dev_crops"]) == {SIM, HST}
    n_val = len(rep._read_csv(run / "predictions" / f"{SIM}__val.csv"))
    assert result["dev_crops"][SIM]["rows"] == n_val and result["dev_crops"][SIM]["partition"] == "val"
    assert result["test_partition_used"] is False
    # Another seed's graph does not compute these weights: parity fails (computed, not assumed).
    other = _export(world["trials"]["synthetic_only-s1"], tmp_path)
    bad = rep.onnx_parity(run, other)
    assert bad["passed"] is False and bad["max_abs_delta_mu"] > 1e-4
    ok, failures = rep.parity_passed(dict(result, dev_crops={}))
    assert ok is False and "no development crops compared" in failures
    assert rep.main(["parity", "--trial", str(run), "--onnx", str(other), "--out",
                     str(tmp_path / "bad.json")]) == rep.EXIT_GATE_FAILED
    # A recorded number above tolerance fails the recomputed verdict, whatever 'passed' says.
    tampered = copy.deepcopy(result)
    tampered["dev_crops"][SIM]["max_abs_delta_log_var"] = 2e-4
    ok, failures = rep.parity_passed(tampered)
    assert ok is False and any("max_abs_delta_log_var" in f for f in failures)
    # --out is optional: <onnx stem>.parity.json next to the ONNX file.
    assert rep.main(["parity", "--trial", str(run), "--onnx", str(onnx_path)]) == rep.EXIT_OK
    written = json.loads((tmp_path / f"{onnx_path.stem}.parity.json").read_text())
    assert written["passed"] is True and written["onnx_sha256"] == rep._sha256_file(onnx_path)


# --------------------------------------------------------------------------- #
# F-006 evidence, checked by rebuild_fqe's validator
# --------------------------------------------------------------------------- #
def _pipeline(world, tmp_path, *, gate_offset):
    classical = write_classical(world, tmp_path / "m2", gate_offset=gate_offset)
    _report(world, classical, tmp_path / "report")
    report_path = tmp_path / "report" / "report.json"
    selection = rep.select_file(report_path)
    run = Path(selection["candidate"]["run_dir"])
    onnx_path = _export(run, tmp_path)
    rep.parity_file(run, onnx_path, tmp_path / "parity.json")
    return report_path, Path(selection["_path"]), tmp_path / "parity.json", run


def _validate(world, run: Path, evidence_path: Path) -> dict:
    role_info = rebuild.resolve_data_role(rebuild.build_parser().parse_args(["--from-run", str(run)]))
    return rebuild.verify_prototype_gate(role_info, evidence_path, contract=world["contract"])


def test_passing_evidence_is_accepted_by_the_rebuild_validator(world, tmp_path):
    report, selection, parity, run = _pipeline(world, tmp_path, gate_offset=50.0)
    code = rep.main(["evidence", "--selection", str(selection), "--parity", str(parity), "--report", str(report),
                     "--out", str(tmp_path / "gate.json"), "--contract", str(world["contract_file"])])
    assert code == rep.EXIT_OK
    evidence = json.loads((tmp_path / "gate.json").read_text())
    assert evidence["schema"] == rebuild.PROTOTYPE_GATE_SCHEMA and evidence["gate"]["passed"] is True
    assert evidence["candidate_outputs_finite"] is True and evidence["test_partition_used"] is False
    assert all(check["passed"] for check in evidence["gate"]["checks"].values())
    info = evidence["informational_comparators"]["comparators"]
    assert set(info) == {RUST_G1, "constant__fixsim_only", "constant__fixhst_only"}
    assert all(entry["gating"] is False for entry in info.values())
    assert info["constant__fixsim_only"]["same_training_sources_as_candidate"] is True
    assert info["constant__fixhst_only"]["same_training_sources_as_candidate"] is False
    assert info[RUST_G1]["same_training_sources_as_candidate"] is None
    # Rust legacy g1 (0.9 x target on all but one row) is informational only, computed on its own common rows.
    rust = info[RUST_G1]
    assert rust["common_valid"] == rust["paired_comparison"]["common_valid"] == \
        evidence["population"]["rows"] - 1
    assert rust["comparator_coverage"] == pytest.approx((evidence["population"]["rows"] - 1)
                                                        / evidence["population"]["rows"])
    assert isinstance(rust["candidate_no_worse_on_common_rows"], bool)
    verified = _validate(world, run, tmp_path / "gate.json")
    assert verified["status"] == "passed" and verified["population"] == {"source_id": SIM, "partition": "val"}
    assert verified["candidate"]["hfr_mae_px"] == evidence["gate"]["checks"]["mae_no_worse"]["candidate_mae_px"]
    assert verified["classical"]["coverage"] == pytest.approx(0.75, abs=0.1)


def test_failing_evidence_is_computed_and_rejected_by_the_rebuild_validator(world, tmp_path):
    report, selection, parity, run = _pipeline(world, tmp_path, gate_offset=0.0)  # classical MAE 0 on common rows
    code = rep.main(["evidence", "--selection", str(selection), "--parity", str(parity), "--report", str(report),
                     "--out", str(tmp_path / "gate.json"), "--contract", str(world["contract_file"])])
    assert code == rep.EXIT_GATE_FAILED
    evidence = json.loads((tmp_path / "gate.json").read_text())
    # A failed gate is recorded under a schema the installer does not accept.
    assert evidence["schema"] == rep.FAILED_EVIDENCE_SCHEMA != rebuild.PROTOTYPE_GATE_SCHEMA
    assert evidence["gate"]["passed"] is False
    assert evidence["gate"]["checks"]["mae_no_worse"]["passed"] is False
    assert evidence["gate"]["checks"]["onnx_parity"]["passed"] is True
    with pytest.raises(rebuild.RebuildRefusal, match="must have schema"):
        _validate(world, run, tmp_path / "gate.json")
    # With the accepted schema forged in, the validator still refuses it: it now
    # requires gate.passed and every gate check (coordinator change to rebuild_fqe).
    (tmp_path / "forged.json").write_text(json.dumps(dict(evidence, schema=rebuild.PROTOTYPE_GATE_SCHEMA)))
    with pytest.raises(rebuild.RebuildRefusal, match="gate.passed"):
        _validate(world, run, tmp_path / "forged.json")
    # And with the verdict forged too, its own MAE/coverage recheck still refuses it.
    forged_gate = dict(evidence["gate"], passed=True,
                       checks={k: dict(v, passed=True) for k, v in evidence["gate"]["checks"].items()})
    (tmp_path / "forged2.json").write_text(json.dumps(dict(evidence, schema=rebuild.PROTOTYPE_GATE_SCHEMA,
                                                           gate=forged_gate)))
    with pytest.raises(rebuild.RebuildRefusal, match="evidence gate is not met: candidate MAE"):
        _validate(world, run, tmp_path / "forged2.json")


def test_evidence_gate_fails_on_parity_and_refuses_unbound_inputs(world, tmp_path):
    report, selection, parity, run = _pipeline(world, tmp_path, gate_offset=50.0)
    # parity measured against another seed's graph: the gate fails on parity. MAE and coverage pass; the
    # failed-record schema keeps it out, and so does the installer's gate.checks requirement.
    other_dir = tmp_path / "other_onnx"
    other_dir.mkdir()
    donor = "synthetic_only-s0" if run.name != "synthetic_only-s0" else "synthetic_only-s1"
    rep.parity_file(run, _export(world["trials"][donor], other_dir), tmp_path / "bad_parity.json")
    evidence = rep.build_evidence(selection, tmp_path / "bad_parity.json", report, contract=world["contract"])
    assert evidence["gate"]["passed"] is False and evidence["gate"]["checks"]["onnx_parity"]["passed"] is False
    assert evidence["gate"]["checks"]["mae_no_worse"]["passed"] is True
    assert evidence["gate"]["checks"]["coverage_no_lower"]["passed"] is True
    assert rep.main(["evidence", "--selection", str(selection), "--parity", str(tmp_path / "bad_parity.json"),
                     "--report", str(report), "--out", str(tmp_path / "bad_gate.json"),
                     "--contract", str(world["contract_file"])]) == rep.EXIT_GATE_FAILED
    written = json.loads((tmp_path / "bad_gate.json").read_text())
    assert written["schema"] == rep.FAILED_EVIDENCE_SCHEMA and written["gate"]["passed"] is False
    with pytest.raises(rebuild.RebuildRefusal, match="must have schema"):
        _validate(world, run, tmp_path / "bad_gate.json")
    # parity computed on other dev crops than the report's val population
    moved = json.loads(parity.read_text())
    moved["dev_crops"][SIM]["membership_sha256"] = "0" * 64
    (tmp_path / "moved_parity.json").write_text(json.dumps(moved))
    with pytest.raises(rep.ReportRefusal, match="not the report's val population"):
        rep.build_evidence(selection, tmp_path / "moved_parity.json", report, contract=world["contract"])
    moved["dev_crops"][SIM]["partition"] = "test"
    (tmp_path / "moved_parity.json").write_text(json.dumps(moved))
    with pytest.raises(SealedPartitionError):
        rep.build_evidence(selection, tmp_path / "moved_parity.json", report, contract=world["contract"])
    # a selection made from another report
    edited = tmp_path / "edited_report.json"
    edited.write_text(report.read_text() + "\n")
    with pytest.raises(rep.ReportRefusal, match="different report"):
        rep.build_evidence(selection, parity, edited, contract=world["contract"])
    # parity of other weights
    other_run = world["trials"]["hst_only-s0"]
    rep.parity_file(other_run, _export(other_run, other_dir), tmp_path / "hst_parity.json")
    with pytest.raises(rep.ReportRefusal, match="different weights"):
        rep.build_evidence(selection, tmp_path / "hst_parity.json", report, contract=world["contract"])
    # no eligible candidate: nothing is exported
    data = json.loads(selection.read_text())
    data["candidate"] = None
    empty = tmp_path / "empty_selection.json"
    empty.write_text(json.dumps(data))
    with pytest.raises(rep.ReportRefusal, match="nothing is exported"):
        rep.build_evidence(empty, parity, report, contract=world["contract"])
    # a report built under another contract than the one given
    with pytest.raises(rep.ReportRefusal, match="contract"):
        rep.build_evidence(selection, parity, report, contract=fs.load_contract())


def _edited_selection(selection: Path, dest: Path, trial_id: str) -> Path:
    """A hand edit of selection.json: the candidate replaced by another trial's ranking entry."""
    data = json.loads(selection.read_text())
    entry = next(s for arm in data["ranking"] for s in arm["seeds"] if s["trial_id"] == trial_id)
    candidate = dict(data["candidate"])
    candidate.update({k: entry[k] for k in ("trial_id", "seed", "score", "log_hfr_mae_sim", "log_hfr_mae_hst",
                                            "model_sha256", "config_sha256", "run_dir", "comparator_id")})
    candidate["arm"] = trial_id.rsplit("-s", 1)[0]
    data.update(candidate=candidate, selected_arm=candidate["arm"])
    dest.write_text(json.dumps(data))
    return dest


def test_evidence_refuses_a_selection_that_is_not_the_f010_outcome(world, tmp_path):
    report, selection, parity, run = _pipeline(world, tmp_path, gate_offset=50.0)
    chosen = json.loads(selection.read_text())["candidate"]
    assert chosen["arm"] == "synthetic_only" and run.name == chosen["trial_id"]
    by_arm = {e["arm"]: e for e in json.loads(selection.read_text())["ranking"]}
    assert by_arm["hst_only"]["eligible"] is False  # seeds 1 and 2 are not reported
    non_median = next(f"synthetic_only-s{s}" for s in (0, 1, 2) if s != chosen["seed"])
    for trial_id in (non_median, "hst_only-s0"):
        dest = tmp_path / trial_id
        dest.mkdir()
        edited = _edited_selection(selection, dest / "selection.json", trial_id)
        # parity bound to the edited candidate's own weights, so only the F-010 binding can refuse it
        trial_run = world["trials"][trial_id]
        assert rep.parity_file(trial_run, _export(trial_run, dest), dest / "parity.json")["passed"] is True
        with pytest.raises(rep.ReportRefusal, match="not the F-010 outcome of this report"):
            rep.build_evidence(edited, dest / "parity.json", report, contract=world["contract"])
        assert rep.main(["evidence", "--selection", str(edited), "--parity", str(dest / "parity.json"),
                         "--report", str(report), "--out", str(dest / "gate.json"),
                         "--contract", str(world["contract_file"])]) == rep.EXIT_REFUSED
        assert not (dest / "gate.json").exists()
    # the F-010 candidate kept but the ranking edited: not this report's F-010 outcome either
    data = json.loads(selection.read_text())
    data["ranking"][0]["score"] = 0.0
    (tmp_path / "ranking_edit.json").write_text(json.dumps(data))
    with pytest.raises(rep.ReportRefusal, match="not the F-010 outcome of this report.*ranking"):
        rep.build_evidence(tmp_path / "ranking_edit.json", parity, report, contract=world["contract"])
    # the untouched selection still produces accepted evidence
    assert rep.build_evidence(selection, parity, report, contract=world["contract"])["gate"]["passed"] is True


def test_outputs_refuse_the_production_model_registry(world, tmp_path):
    classical = write_classical(world, tmp_path / "m2", gate_offset=1.0)
    with pytest.raises(dx.ExperimentRefusal):
        _report(world, classical, rep.ML_ROOT.parent / "models" / "fqe_dev_report_test")
    assert not (rep.ML_ROOT.parent / "models" / "fqe_dev_report_test").exists()
