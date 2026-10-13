"""Import a pinned 16-exposure ACS/WFC feasibility corpus, preserving originals.

The dates are a deliberate multi-year calibration-field selection, not a
representative sample of HST or of Achernar cameras. No autofocus runs exist.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .import_public_imaging import prepare_hst, _fqe_arrays, _write_npz, _frozen_registry
from .public_common import sha256_file, write_json, write_jsonl

SELECTION = Path(__file__).with_name("hst_acs_selection_v1.json")


def exposure_split(date):
    # Fixed temporal holdouts. A later archive expansion must version this plan.
    if date < "2011-01-01":
        return "train"
    if date < "2014-01-01":
        return "validation"
    return "test"


def prepare(root, max_sources=2000):
    output = Path(root) / "hst_acs_corpus_v1"
    output.mkdir(parents=True, exist_ok=True)
    (output/'importer_source.py').write_bytes(Path(__file__).read_bytes())
    selection = json.loads(SELECTION.read_text())
    write_json(output / "source_selection.json", selection)
    arrays, records, sources = [], [], []
    for item in selection:
        product = item["product"]
        folder = output / "exposures" / product
        if (folder / "manifest.json").exists():
            manifest = json.loads((folder / "manifest.json").read_text())
            original = folder / manifest["source"]["path"]
            if original.stat().st_size != item["bytes"] or sha256_file(original) != manifest["source"]["sha256"]:
                raise ValueError("Cached HST original changed: " + product)
        else:
            manifest = prepare_hst(output / "exposures", product=product,
                                   dataset_name=product, expected_size=item["bytes"],
                                   max_sources=max_sources)
        headers = json.loads((folder / "headers.json").read_text())["primary"]
        date = headers["DATE-OBS"]
        split = exposure_split(date)
        rows = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
        accepted = sorted((r for r in rows if r["fqe_quality_accepted"]), key=lambda r:r["fqe_index"])
        for row in accepted:
            row.update(split=split, split_group="HST-ACS-date:"+date,
                       training_eligible=(split == "train" and row['label_quality_tier']=='strict_unmasked_aperture'),
                       source_path="exposures/"+product+"/"+row["source_path"],
                       task=["FQE_feasibility_measurement_reference"])
            row['filter_name']=';'.join(str(headers[k]) for k in ('FILTER1','FILTER2') if headers[k] and not str(headers[k]).startswith('CLEAR'))
            row["all_reference_index"] = len(records)
            row["fqe_index"] = None
            records.append(row)
        with np.load(folder / "fqe_diagnostic.npz", allow_pickle=False) as data:
            arrays.extend(data["crops"])
        sources.append({"product":product, "date":date, "split":split,
                        "source_sha256":manifest["source"]["sha256"],
                        "source_bytes":item["bytes"], "accepted":len(accepted),
                        "detected":manifest["source_count_detected"],
                        "inspected":manifest["source_count_inspected"],
                        "filter1":headers["FILTER1"], "filter2":headers["FILTER2"]})
        print(json.dumps(sources[-1]), flush=True)
    values = [r["hfr_px"] for r in records]
    all_data = _fqe_arrays(arrays, values, values, records)
    all_data['reference_quality_tier']=np.asarray([r['label_quality_tier'] for r in records])
    all_data['training_eligible']=np.asarray([r['training_eligible'] for r in records])
    all_artifact=_write_npz(output/'fqe_all_references.npz',all_data)
    strict=np.asarray([r['label_quality_tier']=='strict_unmasked_aperture' for r in records])
    strict_rows=[r for r,keep in zip(records,strict) if keep]
    for i,row in enumerate(strict_rows):row['fqe_index']=i
    data={key:array[strict] for key,array in all_data.items()}
    artifact = _write_npz(output / "fqe.npz", data)
    registry = _frozen_registry(output / "fqe.npz", data, 42)
    registry.update(strategy="fixed_temporal_date_groups", seed=None)
    write_json(output / "fqe.splits.json", registry)
    write_jsonl(output / "records.jsonl", records)
    manifest = {"dataset_id":"hst_acs_corpus_v1", "status":"prepared_feasibility_corpus",
        "source_selection_sha256":sha256_file(SELECTION), "sources":sources,
        "importer_source_sha256":sha256_file(output/'importer_source.py'),
        "exposures":len(sources), "original_bytes":sum(s["source_bytes"] for s in sources),
        "accepted_crops":len(records), "split_counts":registry["counts"],
        "strict_fqe_crops":len(strict_rows),
        "all_reference_split_counts":dict(Counter(r['split'] for r in records)),
        "quality_tiers":dict(Counter(r["label_quality_tier"] for r in records)),
        "artifacts":[artifact,all_artifact], "default_dataset_policy":"fqe.npz contains strict unmasked apertures only; experimental masked-wing references retained separately in fqe_all_references.npz",
        "training_config_required":{"lambda_rank":0,"split_registry":"fqe.splits.json"},
        "native_units":"calibrated electrons, SCI1 only; complete FLT originals retain all HDUs",
        "license":"public-domain HST mission data; retain STScI NASA ESA acknowledgement",
        "selection":"NGC104 aliases, ACS/WFC, exposure10-120s; one FLT per selected year from first100 public MAST query results; deliberately selected not random",
        "split_policy":"before2011 train;2011-2013 validation;2014on test; actual FITS DATE-OBS groups; fixed after input quality inspection before any model training",
        "limitations":["Same calibration field/instrument; temporal holdout is not an unseen-field or unseen-rig test.",
          "SEP HFR measurement references are not independent physical truth or motor offset labels.",
          "Masked-wing references are explicitly experimental; core/saturation gates and input imputation are recorded per exposure.",
          "Only SCI1 processed; SCI2 retained in original files. Full-frame crowding and no CTE/PAM correction limit transfer.",
          "No positions, direction, run ranking, tracking cause, or R50 sensor calibration inferred."]}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("ml/data/public"))
    parser.add_argument("--max-sources", type=int, default=2000)
    args=parser.parse_args()
    result=prepare(args.root,args.max_sources)
    print(json.dumps({k:result[k] for k in ("exposures","accepted_crops","split_counts")}))


if __name__ == "__main__":
    main()
