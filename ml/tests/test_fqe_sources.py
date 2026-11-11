"""Contract tests for the source-aware FQE loader (generated tmp-dir fixtures only)."""

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_pipeline.fqe_sources as fs  # noqa: E402
from data_pipeline import split_registry  # noqa: E402
from data_pipeline.fqe_sources import (  # noqa: E402
    CalibrationUnavailable,
    ContractViolation,
    DataUnavailable,
    HashMismatch,
    SealedPartitionError,
    calibration_groups,
    load_contract,
    load_final_test,
    load_source,
    make_mixture,
    verify_source,
    write_fixture_source,
)

VALUE_FIELDS = {"crops", "hfr_px", "log_hfr", "reference_hfr_px", "measured_hfr_px", "star_snr"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _repin(fx):
    """Re-pin every fixture file after a deliberate edit; returns the new contract."""
    entry = dict(fx.entry)
    entry["files"] = {name: _sha(fx.directory / name) for name in entry["files"]}
    fx.entry = entry
    fx.contract = fx.contract.with_source(fx.source_id, entry)
    return fx.contract


def _rewrite_npz(fx, mutate):
    """Rewrite the fixture NPZ through ``mutate(arrays)`` and rebind registry/manifest/pins."""
    npz_path = fx.directory / "fqe.npz"
    with np.load(npz_path, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    mutate(arrays)
    np.savez_compressed(npz_path, **arrays)
    registry_path = fx.directory / "fqe.splits.json"
    registry = json.loads(registry_path.read_text())
    registry["dataset_sha256"] = _sha(npz_path)
    registry_path.write_text(json.dumps(registry))
    manifest_path = fx.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["sha256"] = _sha(npz_path)
    manifest_path.write_text(json.dumps(manifest))
    return _repin(fx)


def _snapshot(directory: Path):
    return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in Path(directory).iterdir())


def _receipt(source_id, **overrides):
    receipt = {
        "candidate_id": "fqe-candidate-fixture",
        "candidate_sha256": "a" * 64,
        "protocol_id": "fqe-final-protocol-fixture",
        "protocol_sha256": "b" * 64,
        "authorized_by": "fixture test",
        "acknowledgement": fs.FINAL_TEST_ACKNOWLEDGEMENT.format(source_id=source_id),
    }
    receipt.update(overrides)
    return receipt


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    """Every test records final-test consumption in its own tmp ledger."""
    path = tmp_path / "ledger"
    monkeypatch.setenv(fs.FINAL_TEST_LEDGER_ENV, str(path))
    return path


@pytest.fixture
def sim(tmp_path):
    return write_fixture_source(tmp_path / "data", "fix_sim", n_groups=20, rows_per_group=3, style="sim")


@pytest.fixture
def hst(tmp_path):
    return write_fixture_source(tmp_path / "data", "fix_hst", n_groups=8, rows_per_group=5, style="hst",
                                calibration="unavailable", nonstrict_rows=(1, 2))


# --------------------------------------------------------------------------- #
# Tracked contract
# --------------------------------------------------------------------------- #
def test_tracked_contract_mirrors_frozen_rules_without_machine_paths():
    contract = load_contract()
    assert contract.file_sha256 == _sha(fs.DEFAULT_CONTRACT_PATH)
    sources = contract.sources
    assert sources["stellar_sim_v1"]["role"] == "synthetic_reference"
    assert sources["hst_acs_corpus_v1"]["role"] == "archive_measurement_reference"
    assert sources["stellar_fits_pilot"]["role"] == "diagnostic_sealed"
    assert sources["stellar_sim_v1"]["target_field"] == "reference_hfr_px"
    assert sources["stellar_sim_v1"]["calibration"]["expected_groups"] == 41
    assert sources["hst_acs_corpus_v1"]["calibration"]["rule"] == "unavailable"
    assert sources["hst_acs_corpus_v1"]["tier_filter"]["equals"] == "strict_unmasked_aperture"
    for source_id in ("stellar_sim_v1", "hst_acs_corpus_v1"):
        assert sources[source_id]["final_test"]["enabled"] is False
        assert sources[source_id]["batching"] == "pointwise"
        assert sources[source_id]["ranking_eligible"] is False
    assert contract.mixture_for(["stellar_sim_v1", "hst_acs_corpus_v1"])["common_reference_policy"]
    text = fs.DEFAULT_CONTRACT_PATH.read_text()
    assert "/Users/" not in text and "/Volumes/" not in text

    def strings(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                yield from strings(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from strings(value)
        elif isinstance(obj, str):
            yield obj

    assert not any(s.startswith("/") for s in strings(contract.data))


def test_contract_rejects_absolute_paths_and_bad_pins(sim):
    with pytest.raises(ContractViolation):
        sim.contract.with_source("bad", dict(sim.entry, directory="/abs/path"))
    with pytest.raises(ContractViolation):
        sim.contract.with_source("bad", dict(sim.entry, files=dict(sim.entry["files"], **{"fqe.npz": "xyz"})))
    with pytest.raises(ContractViolation):
        sim.contract.with_source("bad", dict(sim.entry, batching="grouped_run"))


# --------------------------------------------------------------------------- #
# Development loading
# --------------------------------------------------------------------------- #
def test_development_load_returns_namespaced_dev_rows_only(sim):
    arrays = load_source("fix_sim", contract=sim.contract, data_root=sim.data_root)
    assert set(arrays.partition.tolist()) <= {"train_fit", "calibration", "val"}
    assert {"train_fit", "calibration", "val"} == set(arrays.partition.tolist())
    assert arrays.crops.shape == (len(arrays), 64, 64) and arrays.crops.dtype == np.float32
    assert arrays.target_hfr_px.dtype == np.float32 and arrays.log_target.dtype == np.float32
    np.testing.assert_allclose(arrays.log_target, np.log(arrays.target_hfr_px), atol=1e-6)
    assert arrays.secondary["measured_hfr_px"].shape == (len(arrays),)
    assert arrays.batching == "pointwise" and not arrays.ranking_eligible.any()
    for name in ("sample_uid", "group_uid", "frame_uid"):
        values = getattr(arrays, name)
        assert all(v.startswith("fix_sim:") for v in values.tolist())
    assert len(set(arrays.sample_uid.tolist())) == len(arrays)
    test_groups = {g for g, split in sim.group_split.items() if split == "test"}
    sealed = arrays.identity["sealed_test"]
    assert sealed["rows"] == 3 * len(test_groups) and sealed["groups"] == len(test_groups)
    assert sealed["arrays_returned"] is False
    assert len(arrays) + sealed["rows"] == sim.n_rows
    assert arrays.identity["files"] == sim.entry["files"]
    assert arrays.identity["registry_sha256"] == sim.entry["files"]["fqe.splits.json"]
    assert arrays.identity["runtime_eligible"] is False
    assert set(arrays.strata) >= {"profile", "motion", "night_id"}
    assert arrays.joins["array_index"].dtype == np.int64
    np.testing.assert_array_equal(arrays.joins["array_index"], 2 * arrays.npz_row)
    only_val = load_source("fix_sim", partitions=("val",), contract=sim.contract, data_root=sim.data_root)
    assert set(only_val.partition.tolist()) == {"val"}


@pytest.mark.parametrize("partitions", [("test",), ("val", "test"), "test", ("train_fit", "test_sealed")])
def test_requesting_test_is_sealed(sim, partitions):
    with pytest.raises(SealedPartitionError):
        load_source("fix_sim", partitions=partitions, contract=sim.contract, data_root=sim.data_root)


def test_unknown_partition_is_rejected(sim):
    with pytest.raises(ValueError):
        load_source("fix_sim", partitions=("train",), contract=sim.contract, data_root=sim.data_root)


def test_sealed_test_values_are_never_decoded(sim, monkeypatch):
    test_rows = set(np.flatnonzero(np.repeat([sim.group_split[g] for g in range(20)], 3) == "test").tolist())
    decoded, full_reads = [], []
    original_rows, original_full = fs._NpzReader.read_rows, fs._NpzReader.read_full

    def spy_rows(self, key, rows):
        decoded.append((key, set(np.asarray(rows).tolist())))
        return original_rows(self, key, rows)

    def spy_full(self, key):
        full_reads.append(key)
        return original_full(self, key)

    monkeypatch.setattr(fs._NpzReader, "read_rows", spy_rows)
    monkeypatch.setattr(fs._NpzReader, "read_full", spy_full)
    load_source("fix_sim", contract=sim.contract, data_root=sim.data_root)
    verify_source("fix_sim", contract=sim.contract, data_root=sim.data_root)
    assert decoded and all(not (rows & test_rows) for _, rows in decoded)
    assert not (set(full_reads) & VALUE_FIELDS)


def test_nonfinite_values_in_sealed_rows_are_not_even_inspected(tmp_path):
    splits = ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    fx = write_fixture_source(tmp_path / "data", "fix_seal", 10, 2, group_splits=splits, nonfinite_rows=(18,))
    arrays = load_source("fix_seal", contract=fx.contract, data_root=fx.data_root)
    assert np.isfinite(arrays.crops).all()


def test_diagnostic_sealed_pilot_never_loads(tmp_path):
    contract = load_contract()
    with pytest.raises(SealedPartitionError):
        load_source("stellar_fits_pilot", contract=contract, data_root=tmp_path / "absent")
    with pytest.raises(SealedPartitionError):
        verify_source("stellar_fits_pilot", contract=contract, data_root=tmp_path / "absent")
    with pytest.raises(SealedPartitionError):
        calibration_groups("stellar_fits_pilot", [], contract)
    with pytest.raises(SealedPartitionError):
        load_final_test("stellar_fits_pilot", _receipt("stellar_fits_pilot"), receipt_dir=tmp_path / "out",
                        contract=contract, data_root=tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_unknown_source_is_a_contract_violation(sim):
    with pytest.raises(ContractViolation):
        load_source("no_such_source", contract=sim.contract, data_root=sim.data_root)


# --------------------------------------------------------------------------- #
# Mount, files and hashes
# --------------------------------------------------------------------------- #
def test_missing_drive_and_files_are_data_unavailable(sim, tmp_path, monkeypatch):
    with pytest.raises(DataUnavailable):
        load_source("fix_sim", contract=sim.contract, data_root=tmp_path / "not-mounted")
    link = tmp_path / "public"
    link.symlink_to(tmp_path / "volume-absent" / "public")
    with pytest.raises(DataUnavailable, match="mounted"):
        load_source("fix_sim", contract=sim.contract, data_root=link)
    assert not (tmp_path / "volume-absent").exists()
    monkeypatch.setenv(fs.DATA_ROOT_ENV, str(tmp_path / "env-absent"))
    with pytest.raises(DataUnavailable):
        load_source("fix_sim", contract=sim.contract)
    monkeypatch.setenv(fs.DATA_ROOT_ENV, str(sim.data_root))
    assert len(load_source("fix_sim", contract=sim.contract)) > 0
    (sim.directory / "records.jsonl").unlink()
    with pytest.raises(DataUnavailable, match="records.jsonl"):
        load_source("fix_sim", contract=sim.contract, data_root=sim.data_root)


def test_missing_source_directory_is_data_unavailable(sim, tmp_path):
    empty = tmp_path / "empty-root"
    empty.mkdir()
    with pytest.raises(DataUnavailable):
        verify_source("fix_sim", contract=sim.contract, data_root=empty)


def test_hash_mismatch_on_any_pinned_file(sim):
    with (sim.directory / "records.jsonl").open("a") as fh:
        fh.write("\n")
    with pytest.raises(HashMismatch, match="records.jsonl"):
        load_source("fix_sim", contract=sim.contract, data_root=sim.data_root)
    _repin(sim)
    bad = sim.contract.with_source("fix_sim", dict(sim.entry, files=dict(sim.entry["files"], **{"fqe.npz": "0" * 64})))
    with pytest.raises(HashMismatch, match="fqe.npz"):
        verify_source("fix_sim", contract=bad, data_root=sim.data_root)


def test_registry_bound_to_another_dataset_is_rejected(sim):
    registry_path = sim.directory / "fqe.splits.json"
    registry = json.loads(registry_path.read_text())
    registry["dataset_sha256"] = "1" * 64
    registry_path.write_text(json.dumps(registry))
    _repin(sim)
    with pytest.raises(HashMismatch, match="registry"):
        load_source("fix_sim", contract=sim.contract, data_root=sim.data_root)


# --------------------------------------------------------------------------- #
# Malformed arrays and broken joins
# --------------------------------------------------------------------------- #
def test_wrong_crop_geometry_is_a_contract_violation(tmp_path):
    fx = write_fixture_source(tmp_path / "data", "fix_geo", 6, 2, crop_shape=(63, 64))
    with pytest.raises(ContractViolation, match="64, 64"):
        load_source("fix_geo", contract=fx.contract, data_root=fx.data_root)


def test_nonfinite_dev_crop_is_a_contract_violation(tmp_path):
    fx = write_fixture_source(tmp_path / "data", "fix_nan", 6, 2, nonfinite_rows=(0,))
    with pytest.raises(ContractViolation, match="non-finite"):
        load_source("fix_nan", contract=fx.contract, data_root=fx.data_root)


@pytest.mark.parametrize("mutation, message", [
    (lambda a: a.__setitem__("star_snr", a["star_snr"][:-1]), "shape"),
    (lambda a: a["reference_hfr_px"].__setitem__(0, np.nan), "non-finite"),
    (lambda a: a["reference_hfr_px"].__setitem__(0, -1.0), "non-positive"),
    (lambda a: a["hfr_px"].__setitem__(0, a["hfr_px"][0] * 2), "differs from target"),
    (lambda a: a["log_hfr"].__setitem__(0, 0.0), "log"),
    (lambda a: a["ranking_eligible"].__setitem__(0, True), "ranking_eligible"),
    (lambda a: a["run_key"].__setitem__(1, a["run_key"][0]), "unique"),
    (lambda a: a["crops"].__setitem__((0, 0, 0), 3.0), "range"),
])
def test_malformed_arrays_are_contract_violations(sim, mutation, message):
    _rewrite_npz(sim, mutation)
    with pytest.raises(ContractViolation, match=message):
        load_source("fix_sim", contract=sim.contract, data_root=sim.data_root)


def test_broken_record_join_is_a_contract_violation(sim):
    path = sim.directory / "records.jsonl"
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["frame_id"] = "some-other-frame"
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    _repin(sim)
    with pytest.raises(ContractViolation, match="bad join"):
        load_source("fix_sim", contract=sim.contract, data_root=sim.data_root)


def test_registry_and_split_column_disagreement(tmp_path):
    splits = ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    fx = write_fixture_source(tmp_path / "data", "fix_col", 10, 2, group_splits=splits,
                              split_column_overrides={0: "validation", 1: "validation"})
    with pytest.raises(ContractViolation, match="disagree"):
        load_source("fix_col", contract=fx.contract, data_root=fx.data_root)


def test_registry_val_key_matches_validation_column(sim):
    column = np.load(sim.directory / "fqe.npz")["split"]
    assert "validation" in set(column.tolist()) and "val" not in set(column.tolist())
    registry = json.loads((sim.directory / "fqe.splits.json").read_text())
    assert set(registry["indices"]) == {"train", "val", "test"}
    assert len(load_source("fix_sim", partitions=("val",), contract=sim.contract, data_root=sim.data_root)) > 0


def test_registry_group_leak_is_rejected(tmp_path):
    fx = write_fixture_source(tmp_path / "data", "fix_leak", 10, 2,
                              group_splits=["train"] * 6 + ["val"] * 2 + ["test"] * 2)
    registry_path = fx.directory / "fqe.splits.json"
    registry = json.loads(registry_path.read_text())
    moved = registry["indices"]["train"].pop(0)        # row 0 of group 0 (train)
    registry["indices"]["val"].append(moved)           # group 0 now straddles train/val
    registry["counts"] = {k: len(v) for k, v in registry["indices"].items()}
    registry.pop("groups")
    registry_path.write_text(json.dumps(registry))
    _repin(fx)
    with pytest.raises(ContractViolation):
        load_source("fix_leak", contract=fx.contract, data_root=fx.data_root)


# --------------------------------------------------------------------------- #
# Derivative isolation
# --------------------------------------------------------------------------- #
def test_shared_ancestor_across_registry_partitions_is_rejected(tmp_path):
    splits = ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    fx = write_fixture_source(tmp_path / "data", "fix_anc", 10, 2, group_splits=splits,
                              ancestor_of_group={6: 0})  # val group 6 descends from train group 0
    with pytest.raises(ContractViolation, match="ancestor"):
        load_source("fix_anc", contract=fx.contract, data_root=fx.data_root)
    fx = write_fixture_source(tmp_path / "data2", "fix_anc", 10, 2, group_splits=splits,
                              ancestor_of_group={8: 7})  # test group 8 descends from val group 7
    with pytest.raises(ContractViolation, match="ancestor"):
        verify_source("fix_anc", contract=fx.contract, data_root=fx.data_root)


def test_shared_ancestor_across_train_fit_and_calibration_is_rejected(tmp_path):
    splits = ["train"] * 20 + ["val"] * 3 + ["test"] * 3
    probe = write_fixture_source(tmp_path / "probe", "fix_cal", 26, 1, group_splits=splits)
    train_uids = [f"fix_cal:family{g:05d}" for g in range(20)]
    chosen = calibration_groups("fix_cal", train_uids, probe.contract)
    cal = sorted(int(u.split("family")[1]) for u in chosen)
    fit = sorted(set(range(20)) - set(cal))
    fx = write_fixture_source(tmp_path / "data", "fix_cal", 26, 1, group_splits=splits,
                              ancestor_of_group={fit[0]: cal[0]})
    with pytest.raises(ContractViolation, match="ancestor"):
        load_source("fix_cal", contract=fx.contract, data_root=fx.data_root)


def test_partitions_are_group_and_ancestor_disjoint(sim):
    arrays = load_source("fix_sim", contract=sim.contract, data_root=sim.data_root)
    for keys in (arrays.group_uid, arrays.frame_uid, arrays.ancestor_uid):
        owner = {}
        for key, part in zip(keys.tolist(), arrays.partition.tolist()):
            assert owner.setdefault(key, part) == part


# --------------------------------------------------------------------------- #
# Calibration allocation
# --------------------------------------------------------------------------- #
def test_sim_calibration_allocation_is_deterministic_and_identity_only():
    contract = load_contract()
    uids = [f"stellar_sim_v1:achernar-analytic-prf-v1:seed42:family{i:05d}" for i in range(271)]
    chosen = calibration_groups("stellar_sim_v1", uids, contract)
    assert len(chosen) == 41
    shuffled = list(np.random.default_rng(3).permutation(uids))
    assert calibration_groups("stellar_sim_v1", shuffled + uids[:5], contract) == chosen
    expected = sorted(uids, key=lambda u: hashlib.sha256(u.encode("utf-8")).hexdigest())[:41]
    assert chosen == set(expected)
    with pytest.raises(ContractViolation):
        calibration_groups("stellar_sim_v1", [u.split(":", 1)[1] for u in uids], contract)


def test_hst_calibration_is_unavailable():
    contract = load_contract()
    uids = [f"hst_acs_corpus_v1:HST-ACS-date:200{i}-01-01" for i in range(6)]
    with pytest.raises(CalibrationUnavailable):
        calibration_groups("hst_acs_corpus_v1", uids, contract)


def test_too_few_groups_make_calibration_unavailable(sim):
    with pytest.raises(CalibrationUnavailable):
        calibration_groups("fix_sim", ["fix_sim:family00000", "fix_sim:family00001"], sim.contract)


def test_loaded_calibration_partition_matches_rule(sim):
    arrays = load_source("fix_sim", contract=sim.contract, data_root=sim.data_root)
    train_groups = {f"fix_sim:family{g:05d}" for g, s in sim.group_split.items() if s == "train"}
    chosen = calibration_groups("fix_sim", train_groups, sim.contract)
    cal_groups = set(arrays.group_uid[arrays.partition == "calibration"].tolist())
    fit_groups = set(arrays.group_uid[arrays.partition == "train_fit"].tolist())
    assert cal_groups == chosen and cal_groups.isdisjoint(fit_groups)
    assert cal_groups | fit_groups == train_groups
    assert arrays.identity["calibration"]["available"] is True
    assert arrays.identity["calibration"]["groups"] == len(chosen)


def test_hst_style_tier_filter_and_unavailable_calibration(hst):
    arrays = load_source("fix_hst", contract=hst.contract, data_root=hst.data_root)
    assert "calibration" not in set(arrays.partition.tolist())
    assert arrays.identity["calibration"]["available"] is False
    counts = arrays.identity["partition_counts"]
    excluded = sum(c["rows_excluded_by_tier"] for c in counts.values())
    assert excluded == 2
    assert counts["calibration"]["rows"] == 0
    train_rows = sum(1 for g, s in hst.group_split.items() if s == "train") * 5
    assert counts["train_fit"]["rows"] == train_rows - 2
    assert all(int(r) not in (1, 2) for r in arrays.npz_row.tolist())
    assert arrays.joins["crop_origin_xy"].shape == (len(arrays), 2)
    # frame_id names the exposure: frames repeat while sample identities stay unique
    assert len(set(arrays.frame_uid.tolist())) < len(arrays) == len(set(arrays.sample_uid.tolist()))


def test_hst_training_eligible_must_follow_rule(hst):
    _rewrite_npz(hst, lambda a: a["training_eligible"].__setitem__(1, True))  # non-strict train row
    with pytest.raises(ContractViolation, match="training_eligible"):
        load_source("fix_hst", contract=hst.contract, data_root=hst.data_root)


# --------------------------------------------------------------------------- #
# Mixtures and identity namespacing
# --------------------------------------------------------------------------- #
def test_colliding_integer_run_ids_are_namespaced_apart(tmp_path):
    a = write_fixture_source(tmp_path / "data", "fix_a", 10, 3, run_id_start=0)
    b = write_fixture_source(tmp_path / "data", "fix_b", 10, 3, run_id_start=0, base_contract=a.contract)
    raw_a = np.load(a.directory / "fqe.npz")
    raw_b = np.load(b.directory / "fqe.npz")
    assert set(raw_a["run_id"].tolist()) == set(raw_b["run_id"].tolist())          # ints collide
    assert set(raw_a["run_key"].tolist()) == set(raw_b["run_key"].tolist())        # raw keys collide
    sa = load_source("fix_a", contract=b.contract, data_root=b.data_root)
    sb = load_source("fix_b", contract=b.contract, data_root=b.data_root)
    plan = make_mixture([sb, sa], "train_fit", contract=b.contract)
    assert plan.source_ids == ("fix_a", "fix_b")
    assert set(sa.sample_uid.tolist()).isdisjoint(sb.sample_uid.tolist())
    assert set(sa.group_uid.tolist()).isdisjoint(sb.group_uid.tolist())
    for source_id, arrays in (("fix_a", sa), ("fix_b", sb)):
        assert set(arrays.partition[plan.indices[source_id]].tolist()) == {"train_fit"}
        assert np.isclose(plan.weights[source_id].sum(), 0.5)
    positions, rows, weights = plan.flat()
    assert np.isclose(weights.sum(), 1.0) and positions.size == rows.size
    batch = plan.sample_batch(np.random.default_rng(0), 32)
    assert {k: v.size for k, v in batch.items()} == {"fix_a": 16, "fix_b": 16}
    odd = plan.sample_batch(np.random.default_rng(0), 33)
    assert sorted(v.size for v in odd.values()) == [16, 17]
    assert "run_id" not in sa.secondary and "run_id" not in sa.strata


def test_mixture_collision_and_namespacing_assertions(tmp_path):
    a = write_fixture_source(tmp_path / "data", "fix_a", 10, 3)
    b = write_fixture_source(tmp_path / "data", "fix_b", 10, 3, base_contract=a.contract)
    sa = load_source("fix_a", contract=b.contract, data_root=b.data_root)
    sb = load_source("fix_b", contract=b.contract, data_root=b.data_root)
    with pytest.raises(ContractViolation, match="duplicate"):
        make_mixture([sa, sa], contract=b.contract)
    raw = replace(sb, sample_uid=np.asarray([u.split(":", 1)[1] for u in sb.sample_uid.tolist()]))
    with pytest.raises(ContractViolation, match="namespaced"):
        make_mixture([sa, raw], contract=b.contract)
    stolen = replace(sb, group_uid=np.asarray([u.replace("fix_b:", "fix_a:") for u in sb.group_uid.tolist()]))
    with pytest.raises(ContractViolation):
        make_mixture([sa, stolen], contract=b.contract)
    dup = replace(sb, sample_uid=np.asarray([sb.sample_uid[0]] * len(sb)))
    with pytest.raises(ContractViolation, match="duplicate sample_uid"):
        make_mixture([sa, dup], contract=b.contract)
    with pytest.raises(SealedPartitionError):
        make_mixture([sa, sb], "test", contract=b.contract)


def test_mixture_refuses_undeclared_target_definitions_and_role_mixes(tmp_path):
    a = write_fixture_source(tmp_path / "data", "fix_a", 10, 3, target_definition_id="fixture:def_a")
    b = write_fixture_source(tmp_path / "data", "fix_b", 10, 3, style="hst", target_definition_id="fixture:def_b",
                             base_contract=a.contract)
    sa = load_source("fix_a", contract=b.contract, data_root=b.data_root)
    sb = load_source("fix_b", contract=b.contract, data_root=b.data_root)
    with pytest.raises(ContractViolation, match="common reference"):
        make_mixture([sa, sb], contract=b.contract)
    declared = b.contract.with_mixture("fixture_mix", ["fix_a", "fix_b"], "fixture common reference policy")
    sa2 = load_source("fix_a", contract=declared, data_root=b.data_root)
    sb2 = load_source("fix_b", contract=declared, data_root=b.data_root)
    plan = make_mixture([sa2, sb2], contract=declared)
    assert plan.common_reference_policy == "fixture common reference policy"
    assert plan.mixture_id == "fixture_mix"
    with pytest.raises(ContractViolation, match="different source contracts"):
        make_mixture([sa, sb2], contract=declared)
    real_role = write_fixture_source(tmp_path / "data3", "fix_real", 10, 3, role="synthetic_reference")
    sr = load_source("fix_real", contract=real_role.contract, data_root=real_role.data_root)
    with pytest.raises(ContractViolation, match="fixture_only"):
        make_mixture([sr, sa], contract=b.contract)


def test_split_registry_never_writes_next_to_npz(sim, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("split creation must never be called by the loader")

    monkeypatch.setattr(split_registry, "splits_for_dataset", forbidden)
    monkeypatch.setattr(split_registry, "create_splits", forbidden)
    assert not hasattr(fs, "splits_for_dataset") and not hasattr(fs, "create_splits")
    before_root, before_dir = _snapshot(sim.data_root), _snapshot(sim.directory)
    load_source("fix_sim", contract=sim.contract, data_root=sim.data_root)
    verify_source("fix_sim", contract=sim.contract, data_root=sim.data_root)
    load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=tmp_path / "out", contract=sim.contract,
                    data_root=sim.data_root)
    assert _snapshot(sim.directory) == before_dir
    assert _snapshot(sim.data_root) == before_root
    assert not list(sim.directory.glob("*.splits.json.*"))


# --------------------------------------------------------------------------- #
# Final test entry point
# --------------------------------------------------------------------------- #
def test_final_test_is_disabled_by_the_tracked_contract(tmp_path, ledger):
    contract = load_contract()
    for source_id in ("stellar_sim_v1", "hst_acs_corpus_v1"):
        with pytest.raises(SealedPartitionError, match="disabled"):
            load_final_test(source_id, _receipt(source_id), receipt_dir=tmp_path / "out", contract=contract,
                            data_root=tmp_path / "absent")
    assert not (tmp_path / "out").exists() and not ledger.exists()


@pytest.mark.parametrize("overrides", [
    {"acknowledgement": "yes"},
    {"candidate_sha256": "not-a-hash"},
    {"protocol_sha256": ""},
    {"authorized_by": ""},
])
def test_final_test_requires_a_complete_receipt(sim, tmp_path, ledger, overrides):
    with pytest.raises(SealedPartitionError):
        load_final_test("fix_sim", _receipt("fix_sim", **overrides), receipt_dir=tmp_path / "out",
                        contract=sim.contract, data_root=sim.data_root)
    assert not (tmp_path / "out").exists() and not ledger.exists()


def test_final_test_records_receipt_once_and_returns_only_test(sim, tmp_path, ledger, monkeypatch):
    out = tmp_path / "out"
    receipt_path = out / "final_test_receipt__fix_sim.json"
    original = fs._NpzReader.read_rows

    def receipt_first(self, key, rows):
        assert receipt_path.exists(), "receipt must exist before sealed values are decoded"
        return original(self, key, rows)

    monkeypatch.setattr(fs._NpzReader, "read_rows", receipt_first)
    arrays = load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=out, contract=sim.contract,
                             data_root=sim.data_root)
    assert set(arrays.partition.tolist()) == {"test"}
    assert arrays.identity["mode"] == "final_test"
    assert len(arrays) == sum(1 for s in sim.group_split.values() if s == "test") * 3
    record = json.loads(receipt_path.read_text())
    assert record["status"] == "loaded" and record["rows_returned"] == len(arrays)
    assert record["receipt"]["candidate_sha256"] == "a" * 64
    assert record["contract_sha256"] == sim.contract.sha256
    ledger_entry = ledger.resolve() / f"final_test__{sim.entry['files']['fqe.npz']}.json"
    assert record["ledger_path"] == str(ledger_entry) == arrays.identity["final_test_ledger"]
    assert json.loads(ledger_entry.read_text())["status"] == "loaded"
    with pytest.raises(SealedPartitionError, match="already"):
        load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=out, contract=sim.contract,
                        data_root=sim.data_root)
    with pytest.raises(SealedPartitionError):
        make_mixture([arrays], "train_fit", contract=sim.contract)
    with pytest.raises(ContractViolation, match="data root"):
        load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=sim.data_root / "receipts",
                        contract=sim.contract, data_root=sim.data_root)
    dev = load_source("fix_sim", contract=sim.contract, data_root=sim.data_root)
    assert set(dev.sample_uid.tolist()).isdisjoint(arrays.sample_uid.tolist())


def test_final_test_refuses_the_production_model_registry(sim):
    with pytest.raises(ContractViolation, match="model registry"):
        load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=fs.REPO_ROOT / "models" / "never",
                        contract=sim.contract, data_root=sim.data_root)
    assert not (fs.REPO_ROOT / "models" / "never").exists()


def test_final_test_disabled_on_fixture_when_contract_says_so(tmp_path):
    fx = write_fixture_source(tmp_path / "data", "fix_off", 6, 2, final_test_enabled=False)
    with pytest.raises(SealedPartitionError, match="disabled"):
        load_final_test("fix_off", _receipt("fix_off"), receipt_dir=tmp_path / "out", contract=fx.contract,
                        data_root=fx.data_root)


def test_final_test_is_consumed_once_across_receipt_dirs_and_source_ids(sim, tmp_path, ledger):
    first = load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=tmp_path / "oA", contract=sim.contract,
                            data_root=sim.data_root)
    assert len(first) > 0
    entry_path = ledger / f"final_test__{sim.entry['files']['fqe.npz']}.json"
    entry = json.loads(entry_path.read_text())
    assert entry["status"] == "loaded" and entry["source_id"] == "fix_sim"
    assert entry["rows_returned"] == len(first) and entry["npz_sha256"] == sim.entry["files"]["fqe.npz"]
    assert entry["receipt_path"] == str((tmp_path / "oA" / "final_test_receipt__fix_sim.json").resolve())
    with pytest.raises(SealedPartitionError, match="already consumed"):
        load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=tmp_path / "oB", contract=sim.contract,
                        data_root=sim.data_root)
    assert not (tmp_path / "oB").exists()
    # The same pinned NPZ under another source id is the same sealed test.
    alias = sim.contract.with_source("fix_alias", dict(sim.entry))
    with pytest.raises(SealedPartitionError, match="consumed once"):
        load_final_test("fix_alias", _receipt("fix_alias"), receipt_dir=tmp_path / "oC", contract=alias,
                        data_root=sim.data_root)
    assert not (tmp_path / "oC").exists()
    assert sorted(p.name for p in ledger.iterdir()) == [entry_path.name]


def test_final_test_reads_no_sealed_value_before_the_receipt(sim, tmp_path, monkeypatch):
    out = tmp_path / "out"
    receipt_path = out / "final_test_receipt__fix_sim.json"
    test_rows = np.flatnonzero(np.repeat([sim.group_split[g] for g in range(20)], 3) == "test")
    calls = {"join": [], "dotted_test": [], "rows": []}
    original_join, original_dotted, original_rows = fs._join_records, fs._dotted, fs._NpzReader.read_rows

    def spy_join(source_id, entry, directory, n, cols, sealed_mask):
        calls["join"].append((receipt_path.exists(), bool(np.asarray(sealed_mask)[test_rows].all())))
        return original_join(source_id, entry, directory, n, cols, sealed_mask)

    def spy_dotted(record, path):
        if record.get("split") == "test":
            calls["dotted_test"].append(receipt_path.exists())
        return original_dotted(record, path)

    def spy_rows(self, key, rows):
        calls["rows"].append((key, receipt_path.exists()))
        return original_rows(self, key, rows)

    monkeypatch.setattr(fs, "_join_records", spy_join)
    monkeypatch.setattr(fs, "_dotted", spy_dotted)
    monkeypatch.setattr(fs._NpzReader, "read_rows", spy_rows)
    arrays = load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=out, contract=sim.contract,
                             data_root=sim.data_root)
    assert len(arrays) == len(test_rows)
    # Before the receipt: one identity-only join in which every test row is sealed (keeps no strata/joins).
    assert [sealed for exists, sealed in calls["join"] if not exists] == [True]
    assert (True, False) in calls["join"]                       # test strata joined only after the receipt
    assert calls["dotted_test"] and all(calls["dotted_test"])
    assert calls["rows"] and all(exists for _, exists in calls["rows"])
    for value in calls.values():
        value.clear()
    for receipt_dir in (out, tmp_path / "elsewhere"):
        with pytest.raises(SealedPartitionError, match="already"):
            load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=receipt_dir, contract=sim.contract,
                            data_root=sim.data_root)
    assert calls == {"join": [], "dotted_test": [], "rows": []}  # refusals read nothing


@pytest.mark.parametrize("final", [{"enabled": "false"}, {"enabled": "true"}, {"enabled": 1}, {"enabled": None},
                                   "enabled"])
def test_final_test_switch_must_be_a_json_boolean(sim, final):
    with pytest.raises(ContractViolation, match="final_test"):
        sim.contract.with_source("fix_sim", dict(sim.entry, final_test=final))


def test_final_test_gate_requires_identity_true_even_on_an_unvalidated_contract(sim, tmp_path, ledger):
    data = json.loads(json.dumps(sim.contract.data))
    for value in ("false", "true", 1):
        data["sources"]["fix_sim"]["final_test"] = {"enabled": value}
        unvalidated = fs.SourcesContract(data=data, sha256="0" * 64)
        with pytest.raises(SealedPartitionError, match="disabled"):
            load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=tmp_path / "out", contract=unvalidated,
                            data_root=sim.data_root)
    assert not (tmp_path / "out").exists() and not ledger.exists()


def test_final_test_ledger_sits_beside_a_contract_file_and_is_required(sim, tmp_path, monkeypatch):
    monkeypatch.delenv(fs.FINAL_TEST_LEDGER_ENV)
    with pytest.raises(SealedPartitionError, match="ledger"):
        load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=tmp_path / "out", contract=sim.contract,
                        data_root=sim.data_root)
    assert not (tmp_path / "out").exists()
    contract_dir = tmp_path / "contracts"
    contract_dir.mkdir()
    contract_path = contract_dir / "fixture_contract.json"
    contract_path.write_text(json.dumps(sim.contract.data))
    contract = load_contract(contract_path)
    assert contract.sha256 == sim.contract.sha256
    assert fs.final_test_ledger_dir(contract) == contract_dir.resolve() / fs.FINAL_TEST_LEDGER_DIRNAME
    load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=tmp_path / "out", contract=contract,
                    data_root=sim.data_root)
    entry = contract_dir / fs.FINAL_TEST_LEDGER_DIRNAME / f"final_test__{sim.entry['files']['fqe.npz']}.json"
    assert json.loads(entry.read_text())["status"] == "loaded"
    with pytest.raises(SealedPartitionError, match="once"):
        load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=tmp_path / "elsewhere", contract=contract,
                        data_root=sim.data_root)
    # The tracked contract's default ledger is beside it (computed only, never created here).
    tracked = fs.final_test_ledger_dir(load_contract())
    assert tracked == fs.DEFAULT_CONTRACT_PATH.resolve().parent / fs.FINAL_TEST_LEDGER_DIRNAME


def _protected_standins(tmp_path, monkeypatch):
    """Stand-ins for ml/data (links to a fake volume) and <repo>/models; nothing real is at risk."""
    volume = tmp_path / "volume"
    (volume / "public").mkdir(parents=True)
    (volume / "raws").mkdir()
    data_dir = tmp_path / "repo" / "ml" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "public").symlink_to(volume / "public")
    (data_dir / "raws").symlink_to(volume / "raws")
    models = tmp_path / "repo" / "models"
    monkeypatch.setattr(fs, "DEFAULT_DATA_ROOT", data_dir / "public")
    monkeypatch.setattr(fs, "REPO_DATA_DIR", data_dir)
    monkeypatch.setattr(fs, "MODEL_REGISTRY_ROOT", models)
    return volume, data_dir, models


def test_receipt_dir_and_ledger_refuse_every_data_root_and_the_model_registry(sim, tmp_path, monkeypatch, ledger):
    volume, data_dir, models = _protected_standins(tmp_path, monkeypatch)
    for receipt_dir in (data_dir / "public" / "receipts", volume / "public" / "receipts", volume / "raws" / "r",
                        data_dir / "receipts", models / "focus_quality"):
        with pytest.raises(ContractViolation, match="lies inside"):
            load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=receipt_dir, contract=sim.contract,
                            data_root=sim.data_root)
    monkeypatch.setenv(fs.FINAL_TEST_LEDGER_ENV, str(volume / "public" / "ledger"))
    with pytest.raises(ContractViolation, match="ledger"):
        load_final_test("fix_sim", _receipt("fix_sim"), receipt_dir=tmp_path / "out", contract=sim.contract,
                        data_root=sim.data_root)
    assert not any((volume / "public").iterdir()) and not any((volume / "raws").iterdir())
    assert sorted(p.name for p in data_dir.iterdir()) == ["public", "raws"]
    assert not models.exists() and not (tmp_path / "out").exists() and not ledger.exists()


def test_fixture_writer_refuses_data_roots_the_model_registry_and_overwrites(tmp_path, monkeypatch):
    volume, data_dir, models = _protected_standins(tmp_path, monkeypatch)
    env_root = tmp_path / "env_public"
    env_root.mkdir()
    monkeypatch.setenv(fs.DATA_ROOT_ENV, str(env_root))
    targets = (data_dir / "public", volume / "public", volume / "raws" / "nested", data_dir / "scratch",
               models / "focus_quality", env_root, env_root / "nested")
    for target in targets:
        for style, source_id in (("sim", "stellar_sim_v1"), ("hst", "hst_acs_corpus_v1"), ("local", "local_x")):
            with pytest.raises(ContractViolation, match="lies inside"):
                write_fixture_source(target, source_id, 6, 2, style=style)
    assert not any((volume / "public").iterdir()) and not any((volume / "raws").iterdir())
    assert not any(env_root.iterdir()) and not models.exists()
    assert sorted(p.name for p in data_dir.iterdir()) == ["public", "raws"]
    with pytest.raises(ValueError):
        write_fixture_source(tmp_path / "ok", "../escape", 6, 2)
    assert not (tmp_path / "escape").exists()

    fx = write_fixture_source(tmp_path / "ok", "fix_once", 6, 2)
    before = {name: _sha(path) for name, path in fx.files.items()}
    with pytest.raises(FileExistsError, match="never overwrites"):
        write_fixture_source(tmp_path / "ok", "fix_once", 6, 2, seed=9)
    assert {name: _sha(path) for name, path in fx.files.items()} == before
    local = write_fixture_source(tmp_path / "ok", "loc_once", 6, 2, style="local")
    before = {name: _sha(path) for name, path in local.files.items()}
    with pytest.raises(FileExistsError, match="never overwrites"):
        write_fixture_source(tmp_path / "ok", "loc_once", 6, 2, style="local", seed=9)
    assert {name: _sha(path) for name, path in local.files.items()} == before


# --------------------------------------------------------------------------- #
# Local verified runs (delegated to load_qualified_dataset)
# --------------------------------------------------------------------------- #
def test_local_runs_delegate_to_qualified_loader(tmp_path, monkeypatch):
    from data_pipeline import build_focus_quality_dataset as builder

    fx = write_fixture_source(tmp_path / "local", "local_r50", 12, 5, style="local")
    calls = []
    original = builder.load_qualified_dataset

    def spy(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(builder, "load_qualified_dataset", spy)
    arrays = load_source("local_r50", contract=fx.contract, data_root=fx.data_root)
    assert len(calls) == 1
    assert arrays.role == "local_verified_run" and arrays.batching == "grouped_run"
    assert arrays.ranking_eligible.all()
    assert arrays.identity["real_multi_run_data"] is True
    assert arrays.identity["runtime_eligible"] is False
    assert arrays.identity["delegated_loader"].endswith("load_qualified_dataset")
    assert "test" not in set(arrays.partition.tolist())
    assert all(u.startswith("local_r50:") for u in arrays.group_uid.tolist())
    assert len(set(arrays.sample_uid.tolist())) == len(arrays)
    assert arrays.secondary["position"].shape == (len(arrays),)
    with pytest.raises(SealedPartitionError):
        load_source("local_r50", partitions=("test",), contract=fx.contract, data_root=fx.data_root)
    with pytest.raises(ContractViolation, match="explicit data_root"):
        load_source("local_r50", contract=fx.contract)


def test_local_legacy_labels_are_refused_by_the_delegate(tmp_path):
    from data_pipeline.build_focus_quality_dataset import LegacyUnverifiedLabels

    fx = write_fixture_source(tmp_path / "local", "local_r50", 6, 3, style="local")
    manifest = fx.directory / "focus_quality_ds.manifest.json"
    manifest.write_text(json.dumps({"label_method": "legacy"}))
    entry = dict(fx.entry, files={name: _sha(fx.directory / name) for name in fx.entry["files"]})
    contract = fx.contract.with_source("local_r50", entry)
    with pytest.raises(LegacyUnverifiedLabels):
        load_source("local_r50", contract=contract, data_root=fx.data_root)


def test_local_entry_must_pin_the_qualified_manifest(tmp_path):
    fx = write_fixture_source(tmp_path / "local", "local_r50", 6, 3, style="local")
    assert "focus_quality_ds.manifest.json" in fx.entry["files"]
    files = {name: digest for name, digest in fx.entry["files"].items() if name != "focus_quality_ds.manifest.json"}
    with pytest.raises(ContractViolation, match="manifest"):
        fx.contract.with_source("local_r50", dict(fx.entry, files=files))


def test_local_ranking_eligibility_ignores_test_rows(tmp_path, monkeypatch):
    from data_pipeline import build_focus_quality_dataset as builder

    fx = write_fixture_source(tmp_path / "local", "local_r50", 12, 5, style="local")
    test_runs = sorted(f"run-{g:04d}" for g, split in fx.group_split.items() if split == "test")
    assert test_runs
    original = builder.load_qualified_dataset

    def single_position_dev_runs(path):
        dataset, report = original(path)
        dataset = dict(dataset)
        position = np.asarray(dataset["position"]).copy()
        position[~np.isin(np.asarray(dataset["split_group"]).astype(str), test_runs)] = 1000
        dataset["position"] = position
        return dataset, report

    monkeypatch.setattr(builder, "load_qualified_dataset", single_position_dev_runs)
    arrays = load_source("local_r50", contract=fx.contract, data_root=fx.data_root)
    # Only the (multi-position) test runs could be ranked; they must not make the dev rows eligible.
    assert len(arrays) > 0 and not arrays.ranking_eligible.any()
    assert arrays.identity["real_multi_run_data"] is False
    assert arrays.identity["sealed_test"]["rows"] == 5 * len(test_runs)


def test_local_and_public_roles_do_not_mix(tmp_path):
    loc = write_fixture_source(tmp_path / "local", "local_r50", 12, 5, style="local")
    pub = write_fixture_source(tmp_path / "local", "fix_pub", 10, 3, role="synthetic_reference",
                               base_contract=loc.contract)
    sl = load_source("local_r50", contract=pub.contract, data_root=pub.data_root)
    sp = load_source("fix_pub", contract=pub.contract, data_root=pub.data_root)
    with pytest.raises(ContractViolation):
        make_mixture([sl, sp], contract=pub.contract)


def test_protected_roots_refuse_case_variants(tmp_path):
    """APFS is case-insensitive: a differently-cased spelling of a protected
    root (e.g. <repo>/MODELS) must be refused like the canonical one."""
    from data_pipeline import fqe_sources as fs

    roots = [root for root, _ in fs._protected_roots(None)]
    models = next(root for root in roots if Path(root).name == "models")
    variant = Path(models).parent / Path(models).name.upper() / "x"
    with pytest.raises(fs.ContractViolation):
        fs._refuse_protected(variant, "fixture directory")
    # An unrelated tmp directory is still allowed.
    assert fs._refuse_protected(tmp_path / "ok", "fixture directory") == (tmp_path / "ok").resolve()
