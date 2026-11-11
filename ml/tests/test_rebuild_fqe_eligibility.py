"""rebuild_fqe eligibility follows the data role, never "an NPZ was supplied".

Generated tmp-dir fixtures only; every bundle is written under a tmp
``--models-root`` (never the repository's ``models/``). The frozen classical
baselines are stubbed in end-to-end runs: this file tests eligibility and test
policy, not ``eval.baselines``.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

import data_pipeline.fqe_sources as fs  # noqa: E402
import pipelines.rebuild_fqe as rebuild  # noqa: E402
import train.fqe_dev_experiment as dx  # noqa: E402
from pipelines.rebuild_fqe import RebuildRefusal, derive_eligibility, resolve_data_role  # noqa: E402

NOTE = "fixture prototype; development evidence reviewed by the coordinator"


def _args(*argv):
    return rebuild.build_parser().parse_args(list(argv))


def _stub_baselines(monkeypatch) -> list:
    """Replace eval.baselines.freeze_baselines (and its labels.psf_measure work)
    with a stub that still binds every call to the real signature and records
    the explicit evaluation partition."""
    import inspect

    import eval.baselines as baselines

    real = inspect.signature(baselines.freeze_baselines)
    calls = []

    def stub(*args, **kwargs):
        real.bind(*args, **kwargs)
        assert "partition" in kwargs, "freeze_baselines has no default partition"
        calls.append(kwargs["partition"])
        return {"train_median_hfr_px": 1.0, "constant": {"hfr_mae_px": 1.0, "within_run_rho": "unavailable"},
                "direct_hfr": {"hfr_mae_px": float("nan"), "within_run_rho": float("nan")},
                "tenengrad": {"within_run_rho": "unavailable"}, "norm_variance": {"within_run_rho": float("nan")}}

    monkeypatch.setattr(baselines, "freeze_baselines", stub)
    return calls


@pytest.fixture
def sim(tmp_path):
    return fs.write_fixture_source(tmp_path / "data", "fixsim", n_groups=12, rows_per_group=3,
                                   style="sim", role="synthetic_reference")


@pytest.fixture
def contract_file(tmp_path, sim):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(sim.contract.data))
    return path


@pytest.fixture
def dev_run(tmp_path, sim, contract_file):
    config = {"schema": dx.CONFIG_SCHEMA, "arm": "synthetic_only", "seed": 0, "batch_size": 8, "updates": 2,
              "warmup_updates": 1, "checkpoint_every": 2, "device": "cpu", "contract_path": str(contract_file),
              "data_root": str(sim.data_root), "sources": ["fixsim"]}
    (tmp_path / "arm.json").write_text(json.dumps(config))
    run = tmp_path / "run"
    dx.fit(tmp_path / "arm.json", run)
    dx.calibrate(run)
    return run


def _metadata(result) -> dict:
    return json.loads((Path(result["out_dir"]) / "metadata.json").read_text())


@pytest.fixture
def sealed(monkeypatch):
    """Public-role rebuilds must never reach a test evaluator or the baselines."""

    def forbidden(*args, **kwargs):
        raise AssertionError("public-role rebuild reached a test evaluator")

    import eval.baselines as baselines

    monkeypatch.setattr(fs, "load_final_test", forbidden)
    monkeypatch.setattr(baselines, "freeze_baselines", forbidden)


# --------------------------------------------------------------------------- #
# Pure eligibility rules
# --------------------------------------------------------------------------- #
def test_only_local_verified_multi_row_runs_are_eligible():
    local = {"roles": ["local_verified_run"], "multi_row_runs": True, "batching": "grouped_run"}
    assert derive_eligibility(local) == {"real_multi_run_data": True, "runtime_eligible": True, "prototype": None}
    single = dict(local, multi_row_runs=False)
    assert derive_eligibility(single)["runtime_eligible"] is False
    for roles in (["fixture_only"], ["synthetic_reference"], ["archive_measurement_reference"],
                  ["archive_measurement_reference", "synthetic_reference"]):
        info = {"roles": roles, "multi_row_runs": True, "batching": "pointwise"}
        assert derive_eligibility(info) == {"real_multi_run_data": False, "runtime_eligible": False,
                                            "prototype": None}


def test_prototype_override_is_explicit_noted_public_from_run_and_gated():
    public = {"origin": rebuild.ORIGIN_RUN, "roles": ["archive_measurement_reference", "synthetic_reference"],
              "multi_row_runs": False, "source_ids": ["hst_acs_corpus_v1", "stellar_sim_v1"],
              "contract_sha256": "c" * 64}
    gate = {"status": "passed", "evidence_sha256": "e" * 64}
    with pytest.raises(RebuildRefusal, match="prototype-note"):
        derive_eligibility(public, prototype_runtime_eligible=True, prototype_note="  ", prototype_gate=gate)
    with pytest.raises(RebuildRefusal, match="verified F-006 evidence gate"):
        derive_eligibility(public, prototype_runtime_eligible=True, prototype_note=NOTE)
    with pytest.raises(RebuildRefusal, match="verified F-006 evidence gate"):
        derive_eligibility(public, prototype_runtime_eligible=True, prototype_note=NOTE,
                           prototype_gate={"status": "failed"})
    with pytest.raises(RebuildRefusal, match="only with --from-run"):
        derive_eligibility(dict(public, origin="contract_sources"), prototype_runtime_eligible=True,
                           prototype_note=NOTE, prototype_gate=gate)
    result = derive_eligibility(public, prototype_runtime_eligible=True, prototype_note=NOTE, prototype_gate=gate)
    assert result["runtime_eligible"] is True and result["real_multi_run_data"] is False
    block = result["prototype"]
    assert "F-006" in block["authorization"] and "not qualified for R50/Z114" in block["domain"]
    assert block["ready_for_promotion_review"] is False and block["contract_sha256"] == "c" * 64
    assert block["gate"]["evidence_sha256"] == "e" * 64
    for roles in (["fixture_only"], ["local_verified_run"]):
        with pytest.raises(RebuildRefusal, match="public contract roles"):
            derive_eligibility({"origin": rebuild.ORIGIN_RUN, "roles": roles, "multi_row_runs": True},
                               prototype_runtime_eligible=True, prototype_note=NOTE, prototype_gate=gate)


def test_lambda_rank_defaults_to_zero_for_pointwise_sources():
    assert rebuild.resolve_lambda_rank(None, {"batching": "pointwise"}) == 0.0
    assert rebuild.resolve_lambda_rank(None, {"batching": "grouped_run"}) == 0.2
    assert rebuild.resolve_lambda_rank(0.5, {"batching": "pointwise"}) == 0.5  # train() then refuses it


# --------------------------------------------------------------------------- #
# Role resolution
# --------------------------------------------------------------------------- #
def test_role_resolution_per_input(tmp_path, sim, contract_file, dev_run):
    synthetic = resolve_data_role(_args("--synthetic"))
    assert synthetic["roles"] == ["fixture_only"] and synthetic["origin"] == "generated_synthetic_runs"
    sources = resolve_data_role(_args("--source-id", "fixsim", "--contract", str(contract_file)))
    assert sources["roles"] == ["synthetic_reference"] and sources["evaluate_test"] is False
    assert sources["batching"] == "pointwise"
    run = resolve_data_role(_args("--from-run", str(dev_run)))
    assert run["roles"] == ["synthetic_reference"] and run["evaluate_test"] is False
    local = fs.write_fixture_source(tmp_path / "local", "fixlocal", n_groups=6, rows_per_group=4, style="local")
    local_info = resolve_data_role(_args("--dataset", str(local.files["focus_quality_ds.npz"])))
    assert local_info["roles"] == ["local_verified_run"] and local_info["multi_row_runs"] is True
    assert local_info["evaluate_test"] is True
    with pytest.raises(RebuildRefusal, match="exactly one"):
        resolve_data_role(_args("--synthetic", "--source-id", "fixsim"))


def test_public_archive_and_unqualified_npz_are_refused_by_path(tmp_path, sim):
    with pytest.raises(RebuildRefusal, match="--source-id"):
        resolve_data_role(_args("--dataset", str(sim.directory / "fqe.npz")))
    legacy = tmp_path / "legacy.npz"
    n = 12
    np.savez_compressed(legacy, crops=np.zeros((n, 64, 64), np.float32), hfr_px=np.ones(n, np.float32),
                        log_hfr=np.zeros(n, np.float32), run_id=np.repeat(np.arange(4), 3),
                        split_group=np.repeat(np.array(["a", "b", "c", "d"]), 3))
    with pytest.raises(RebuildRefusal, match="not a qualified"):
        resolve_data_role(_args("--dataset", str(legacy)))


# --------------------------------------------------------------------------- #
# End to end (tmp models root)
# --------------------------------------------------------------------------- #
def test_from_run_public_candidate_is_not_runtime_eligible(tmp_path, dev_run, sealed):
    result = rebuild.main(_args("--from-run", str(dev_run), "--models-root", str(tmp_path / "models")))
    info = _metadata(result)["training_info"]
    assert info["runtime_eligible"] is False and info["promotion_status"] == "shadow"
    assert info["promotion_gate"]["real_multi_run_data"] is False
    assert info["promotion_gate"]["ready_for_promotion_review"] is False
    provenance = info["data_provenance"]
    assert provenance["data_roles"] == ["synthetic_reference"] and provenance["test_evaluated"] is False
    assert info["candidate"]["model_sha256"] == json.loads((dev_run / "run.json").read_text())["final_model_sha256"]
    calibration = json.loads((Path(result["out_dir"]) / "calibration.json").read_text())
    assert calibration["sigma_scale"] is None and calibration["frame_sigma_scale"] is None
    assert calibration["model_sha256"] == _metadata(result)["sha256"]
    assert "prototype" not in info


def _comparison(candidate_mae: float, classical_mae: float, *, candidate_measured: int = 12,
                classical_measured: int = 10) -> dict:
    """A real ``eval.fqe_benchmark.paired_comparison`` over generated rows."""
    from eval.fqe_benchmark import paired_comparison

    n, rows = 12, {"fqe_candidate": [], rebuild.PROTOTYPE_CLASSICAL_COMPARATOR: []}
    for i in range(n):
        target = 2.0 + 0.1 * i
        for cid, mae, measured in (("fqe_candidate", candidate_mae, candidate_measured),
                                   (rebuild.PROTOTYPE_CLASSICAL_COMPARATOR, classical_mae, classical_measured)):
            ok = i < measured
            rows[cid].append({"sample_uid": f"fixsim:{i}", "target": target, "status": "measured" if ok else
                              "unmeasured", "prediction": target + mae if ok else None})
    return paired_comparison(rows, ["fqe_candidate", rebuild.PROTOTYPE_CLASSICAL_COMPARATOR])


def _evidence(tmp_path: Path, run: Path, contract_sha: str, *, name="gate.json", candidate_mae=0.2,
              classical_mae=0.3, comparison=None, **overrides) -> Path:
    record = json.loads((run / "run.json").read_text())
    evidence = {
        "schema": rebuild.PROTOTYPE_GATE_SCHEMA,
        "candidate_model_sha256": record["final_model_sha256"],
        "run_config_sha256": record["config_sha256"],
        "contract_sha256": contract_sha,
        "population": {"source_id": "fixsim", "partition": "val"},
        "test_partition_used": False,
        "candidate_outputs_finite": True,
        "candidate_comparator": "fqe_candidate",
        "classical_comparator": rebuild.PROTOTYPE_CLASSICAL_COMPARATOR,
        "paired_comparison": comparison or _comparison(candidate_mae, classical_mae),
    }
    evidence.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(evidence))
    return path


@pytest.fixture
def tracked_fixture(monkeypatch, contract_file):
    """Stand in for the tracked contract with the fixture contract the run used
    (the real tracked contract pins public data that tests never read)."""
    contract = fs.load_contract(contract_file)
    monkeypatch.setattr(rebuild, "_tracked_contract", lambda: contract)
    return contract


def _prototype_args(tmp_path, dev_run, evidence, *extra):
    argv = ["--from-run", str(dev_run), "--models-root", str(tmp_path / "models"), "--prototype-runtime-eligible",
            "--prototype-note", NOTE]
    if evidence is not None:
        argv += ["--prototype-gate-evidence", str(evidence)]
    return _args(*argv, *extra)


def test_prototype_override_writes_shadow_provenance_and_never_promotion(tmp_path, dev_run, sealed,
                                                                           tracked_fixture):
    evidence = _evidence(tmp_path, dev_run, tracked_fixture.sha256)
    result = rebuild.main(_prototype_args(tmp_path, dev_run, evidence))
    info = _metadata(result)["training_info"]
    assert info["runtime_eligible"] is True and info["promotion_status"] == "shadow"
    assert info["status"] == "trained"
    assert info["promotion_gate"]["ready_for_promotion_review"] is False
    assert info["promotion_gate"]["real_multi_run_data"] is False
    prototype = info["prototype"]
    assert prototype["note"] == NOTE and "F-006" in prototype["authorization"]
    assert prototype["data_roles"] == ["synthetic_reference"] and "R50/Z114" in prototype["domain"]
    assert prototype["contract_sha256"] == tracked_fixture.sha256 and prototype["model_sha256"]
    gate = prototype["gate"]
    import hashlib

    assert gate["status"] == "passed" and gate["evidence_sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert gate["candidate"]["hfr_mae_px"] <= gate["classical"]["hfr_mae_px"]
    assert gate["onnx_parity_dev_crops"]["passed"] is True and gate["onnx_parity_dev_crops"]["rows"] > 0
    assert gate["onnx_parity_dev_crops"]["max_abs_delta_mu"] <= 1e-4
    assert gate["onnx_parity_fixtures"]["passed"] is True and gate["test_partition_used"] is False


def test_prototype_override_refuses_a_run_trained_under_a_fixture_contract(tmp_path, dev_run, contract_file,
                                                                            sealed):
    # The real tracked contract: the fixture-contract run cannot match it.
    contract_sha = fs.load_contract(contract_file).sha256
    evidence = _evidence(tmp_path, dev_run, contract_sha)
    with pytest.raises(RebuildRefusal, match="not the tracked contract"):
        rebuild.main(_prototype_args(tmp_path, dev_run, evidence))
    assert not (tmp_path / "models").exists()


def test_prototype_override_refuses_source_id_training(tmp_path, sim, contract_file, sealed, tracked_fixture):
    with pytest.raises(RebuildRefusal, match="only with --from-run"):
        rebuild.main(_args("--source-id", "fixsim", "--contract", str(contract_file), "--data-root",
                           str(sim.data_root), "--models-root", str(tmp_path / "models"),
                           "--prototype-runtime-eligible", "--prototype-note", NOTE))
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize(
    "case, match",
    [
        ("missing", "requires --prototype-gate-evidence"),
        ("worse_mae", "evidence gate is not met: candidate MAE"),
        ("lower_coverage", "evidence gate is not met: candidate coverage"),
        ("other_weights", "candidate_model_sha256"),
        ("test_used", "test_partition_used"),
        ("finite_unstated", "candidate_outputs_finite"),
        ("unknown_population", "not in the tracked contract"),
        ("test_population", "val partition"),
        ("other_comparator", "normalized_input__direct_hfr"),
        ("no_common_rows", "no common valid rows"),
        ("wrong_schema", "schema"),
    ],
)
def test_prototype_override_refuses_failed_or_unbound_evidence(tmp_path, dev_run, sealed, tracked_fixture,
                                                                 case, match):
    sha = tracked_fixture.sha256
    if case == "missing":
        evidence = None
    elif case == "worse_mae":
        evidence = _evidence(tmp_path, dev_run, sha, candidate_mae=0.4, classical_mae=0.3)
    elif case == "lower_coverage":
        evidence = _evidence(tmp_path, dev_run, sha,
                             comparison=_comparison(0.2, 0.3, candidate_measured=10, classical_measured=12))
    elif case == "other_weights":
        evidence = _evidence(tmp_path, dev_run, sha, candidate_model_sha256="0" * 64)
    elif case == "test_used":
        evidence = _evidence(tmp_path, dev_run, sha, test_partition_used=True)
    elif case == "finite_unstated":
        evidence = _evidence(tmp_path, dev_run, sha, candidate_outputs_finite=None)
    elif case == "unknown_population":
        evidence = _evidence(tmp_path, dev_run, sha, population={"source_id": "stellar_sim_v1", "partition": "val"})
    elif case == "test_population":
        evidence = _evidence(tmp_path, dev_run, sha, population={"source_id": "fixsim", "partition": "test"})
    elif case == "other_comparator":
        evidence = _evidence(tmp_path, dev_run, sha, classical_comparator="constant__synthetic_only")
    elif case == "no_common_rows":
        evidence = _evidence(tmp_path, dev_run, sha,
                             comparison=_comparison(0.2, 0.3, candidate_measured=12, classical_measured=0))
    else:
        evidence = _evidence(tmp_path, dev_run, sha, schema="something-else")
    with pytest.raises(RebuildRefusal, match=match):
        rebuild.main(_prototype_args(tmp_path, dev_run, evidence))
    assert not (tmp_path / "models").exists()


def test_prototype_override_refuses_a_run_whose_recorded_pins_differ(tmp_path, dev_run, sealed, tracked_fixture):
    identity_path = dev_run / "dataset_identity.json"
    identity = json.loads(identity_path.read_text())
    name = sorted(identity["sources"]["fixsim"]["files"])[0]
    identity["sources"]["fixsim"]["files"][name] = "f" * 64
    os.chmod(identity_path, 0o644)
    identity_path.write_text(json.dumps(identity))
    evidence = _evidence(tmp_path, dev_run, tracked_fixture.sha256)
    with pytest.raises(RebuildRefusal, match="recorded file sha256s"):
        rebuild.main(_prototype_args(tmp_path, dev_run, evidence))
    assert not (tmp_path / "models").exists()


def test_prototype_dev_crop_parity_failure_writes_a_non_eligible_bundle(tmp_path, dev_run, sealed,
                                                                          tracked_fixture, monkeypatch):
    evidence = _evidence(tmp_path, dev_run, tracked_fixture.sha256)
    monkeypatch.setattr(rebuild, "check_dev_crop_parity",
                        lambda *a, **k: {"source_id": "fixsim", "partition": "val", "rows": 3,
                                         "max_abs_delta_mu": 1e-2, "tolerance": 1e-4, "finite": True,
                                         "passed": False})
    with pytest.raises(RebuildRefusal, match="dev-crop ONNX parity failed"):
        rebuild.main(_prototype_args(tmp_path, dev_run, evidence))
    metadata = json.loads((tmp_path / "models" / "focus_quality" / "1.1.0" / "metadata.json").read_text())
    info = metadata["training_info"]
    assert info["runtime_eligible"] is False and "prototype" not in info
    assert info["prototype_refused"]["gate"]["onnx_parity_dev_crops"]["passed"] is False


def test_source_id_rebuild_is_pointwise_rank_free_and_sealed(tmp_path, sim, contract_file, sealed):
    result = rebuild.main(_args("--source-id", "fixsim", "--contract", str(contract_file), "--data-root",
                                str(sim.data_root), "--epochs", "1", "--batch-size", "16", "--device", "cpu",
                                "--models-root", str(tmp_path / "models")))
    info = _metadata(result)["training_info"]
    assert info["runtime_eligible"] is False
    assert info["data_provenance"]["batching"] == "pointwise" and info["data_provenance"]["lambda_rank"] == 0.0
    assert info["test"] == {} and info["baselines"]["status"] == "not_run"
    assert info["loss"].startswith("heteroscedastic_nll + 0.0")


def test_source_id_rebuild_refuses_override_for_fixture_role(tmp_path):
    fixture = fs.write_fixture_source(tmp_path / "data", "fixonly", n_groups=6, rows_per_group=2)
    contract = tmp_path / "c.json"
    contract.write_text(json.dumps(fixture.contract.data))
    with pytest.raises(RebuildRefusal, match="public contract roles"):
        rebuild.main(_args("--source-id", "fixonly", "--contract", str(contract), "--models-root",
                           str(tmp_path / "models"), "--prototype-runtime-eligible", "--prototype-note", NOTE))
    assert not (tmp_path / "models").exists()


def test_local_verified_runs_keep_the_genuine_path(tmp_path, monkeypatch):
    calls = _stub_baselines(monkeypatch)
    local = fs.write_fixture_source(tmp_path / "local", "fixlocal", n_groups=6, rows_per_group=4, style="local")
    result = rebuild.main(_args("--dataset", str(local.files["focus_quality_ds.npz"]), "--epochs", "1",
                                "--batch-size", "8", "--device", "cpu", "--models-root", str(tmp_path / "models")))
    info = _metadata(result)["training_info"]
    assert info["runtime_eligible"] is True and info["promotion_gate"]["real_multi_run_data"] is True
    assert info["data_provenance"]["data_roles"] == ["local_verified_run"]
    assert info["data_provenance"]["test_evaluated"] is True and "hfr_mae_px" in info["test"]
    assert info["data_provenance"]["batching"] == "grouped_run" and info["data_provenance"]["lambda_rank"] == 0.2
    calibration = json.loads((Path(result["out_dir"]) / "calibration.json").read_text())
    assert "coverage_95_test_after" in calibration  # legacy genuine-run calibration (fit val, report test)
    assert calls == ["test"]  # explicit final evaluation of genuine local runs


def test_generated_synthetic_stays_explicit_path_only(tmp_path, monkeypatch):
    _stub_baselines(monkeypatch)
    with pytest.raises(RebuildRefusal):
        rebuild.main(_args("--synthetic", "--models-root", str(tmp_path / "m0"),
                           "--prototype-runtime-eligible", "--prototype-note", NOTE))
    result = rebuild.main(_args("--synthetic", "--synthetic-runs", "3", "--epochs", "1", "--device", "cpu",
                                "--models-root", str(tmp_path / "models")))
    info = _metadata(result)["training_info"]
    assert info["runtime_eligible"] is False and info["promotion_gate"]["real_multi_run_data"] is False
    assert info["data_provenance"]["data_roles"] == ["fixture_only"]


def test_pinned_public_source_recognises_excluded_digest_and_fails_closed(tmp_path, monkeypatch):
    """A byte copy of an excluded public file (e.g. the test-era all-references
    NPZ) is recognised by its digest; an unreadable contract refuses."""
    import json as _json

    from data_pipeline import fqe_sources
    from train import train_focus_quality as tfq

    contract = fqe_sources.load_contract()
    excluded = None
    for source_id, entry in contract.sources.items():
        for note in (entry.get("excluded_files") or {}).values():
            import re as _re

            found = _re.findall(r"\b[0-9a-f]{64}\b", str(note))
            if found:
                excluded = (source_id, found[0])
                break
        if excluded:
            break
    assert excluded is not None, "the tracked contract should record an excluded-file digest"
    assert tfq.pinned_public_source(tmp_path / "copy.npz", sha256=excluded[1]) == excluded[0]

    def _broken(*args, **kwargs):
        raise fqe_sources.ContractViolation("unreadable")

    monkeypatch.setattr(fqe_sources, "load_contract", _broken)
    with pytest.raises(ValueError):
        tfq.pinned_public_source(tmp_path / "x.npz", sha256="0" * 64)
