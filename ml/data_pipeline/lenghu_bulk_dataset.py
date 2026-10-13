"""Lazy paired TAR access and leakage exclusions for Lenghu teacher logits."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timedelta
import io
import json
import re
from pathlib import Path
import sqlite3
import zipfile

import numpy as np
from PIL import Image

from .public_common import write_json,sha256_file


REVIEWED_UNPAIRED = {("image", ".jpg", "source/data/image/sky_image_76.tar", 37366272, 76242):
    "6865fcab28d2c5befff320360b581602f7cca2ea6edef1f0635d5d7db694ff0f"}


def quarantine_unpaired(root, rows, reviewed=None):
    """Preserve a reviewed producer orphan; reject every unknown mismatch."""
    reviewed = REVIEWED_UNPAIRED if reviewed is None else reviewed
    ledger = []
    for row in rows:
        key = tuple(row)
        if key not in reviewed:
            raise ValueError("Unreviewed unpaired member in complete cloud release")
        kind, source_id, tar, offset, size = key
        digest = hashlib.sha256(read_member(root, {"tar": tar, "offset": offset, "bytes": size})).hexdigest()
        if digest != reviewed[key]:
            raise ValueError("Reviewed orphan payload changed")
        ledger.append({"kind": kind, "source_id": source_id, "tar": tar, "offset": offset,
                       "bytes": size, "sha256": digest, "training_eligible": False,
                       "reason": "No timestamp, geometry identity or paired teacher logits; original TAR member preserved"})
    return ledger


def filename_timestamp(source_id):
    # The complete producer map includes 39 names ending in _1. Retain the
    # full identifier for pairing; only the date grouping uses the base time.
    match=re.fullmatch(r'(\d{4}(?:-\d{2}){5})(?:_\d+)?',source_id)
    if not match:raise ValueError('Unrecognized source filename timestamp')
    return datetime.strptime(match.group(1),'%Y-%m-%d-%H-%M-%S')


def prepare_geometry(root, connection):
    """Preserve producer content masks and exact per-frame lookup; no pickle."""
    archive_path = root / "source/data/bkg_mask"
    if not archive_path.exists():
        raise ValueError("Producer geometry archive missing; fetch auxiliary source files first")
    for receipt in json.loads((root/'auxiliary_receipts.json').read_text()):
        if sha256_file(root/receipt['path'])!=receipt['sha256']:raise ValueError('Auxiliary source content changed')
    connection.execute("CREATE TABLE IF NOT EXISTS geometry (sample_id TEXT PRIMARY KEY, mask_id TEXT)")
    masks = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.filename.startswith("bkg_mask/mask_mat/mask_") and not member.is_dir():
                if member.file_size > 100_000:raise ValueError("Unexpected geometry size")
                mask=np.load(io.BytesIO(archive.read(member)),allow_pickle=False)
                if mask.shape!=(64,64) or mask.dtype!=np.uint8 or not set(np.unique(mask))<={0,1}:
                    raise ValueError("Geometry mask contract changed")
                masks[Path(member.filename).name.removesuffix('.npy')]=mask
        if connection.execute("SELECT count(*) FROM geometry").fetchone()[0] == 0:
            member=archive.getinfo("bkg_mask/bkg_map.txt")
            if member.file_size>40_000_000:raise ValueError("Unexpected geometry map size")
            entries=[]
            for line in archive.read(member).decode().splitlines():
                source,mask_id=line.split(',')
                key=Path(mask_id).stem
                if key not in masks:raise ValueError("Missing referenced geometry mask")
                entries.append((Path(source).stem.removesuffix('_logits'),key))
            connection.executemany("INSERT INTO geometry VALUES (?,?)",entries)
    temporary=root/'geometry_masks.partial.npz'
    np.savez_compressed(temporary,**masks)
    temporary.replace(root/'geometry_masks.npz')
    with zipfile.ZipFile(root/'source/data/calibration.zip') as archive:
        calibrations={Path(n).name:json.loads(archive.read(n)) for n in archive.namelist() if n.endswith('.json')}
    write_json(root/'calibration_records.json',calibrations)
    return {"masks":len(masks),"mapped_frames":connection.execute("SELECT count(*) FROM geometry").fetchone()[0],
            "shape":[64,64],"values":{"0":"outside producer content region","1":"inside producer content region"},
            "coordinate_calibration_applied":False,
            "calibration_note":"Producer coefficients preserved; special camera variants require their original mapping before azimuth/altitude use."}


def read_member(root, record):
    with (Path(root) / record["tar"]).open("rb") as stream:
        stream.seek(record["offset"])
        value = stream.read(record["bytes"])
    if len(value) != record["bytes"]:
        raise ValueError("Truncated archive member")
    return value


def numeric_logits(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as source:
        classes = source["classes"]
        grid = source["logits_grid"]
        if not np.array_equal(classes, [0, 1, 2]) or grid.shape != (64, 64, 3):
            raise ValueError("Teacher class/grid contract changed")
        if grid.dtype != np.float32 or not np.isfinite(grid).all():
            raise ValueError("Invalid teacher logits")
        # class_names is object-typed in the producer payload: never load it.
        return grid.copy()


def excluded_dates(human_root):
    dates = set()
    with (Path(human_root) / "records.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] in {"validation", "test"}:
                date = datetime.fromisoformat(row["source_timestamp"]).date()
                dates.update(str(date + timedelta(days=offset)) for offset in (-1, 0, 1))
    return dates


def prepare_pairs(root, human_root=None):
    root = Path(root)
    human_root = Path(human_root) if human_root else root.parent / "lenghu_sky8_human"
    receipts = json.loads((root / "download_receipts.json").read_text())
    exclusions = excluded_dates(human_root)
    database = root / "pairs.sqlite3"
    with sqlite3.connect(database) as connection:
        geometry = prepare_geometry(root, connection)
        connection.execute("CREATE TABLE IF NOT EXISTS members (kind TEXT, source_id TEXT, tar TEXT, offset INTEGER, bytes INTEGER, PRIMARY KEY(kind,source_id))")
        connection.execute("CREATE TABLE IF NOT EXISTS indexed (path TEXT PRIMARY KEY, sha256 TEXT)")
        connection.execute("CREATE TABLE IF NOT EXISTS pairs (sample_id TEXT PRIMARY KEY, image_tar TEXT, image_offset INTEGER, image_bytes INTEGER, logit_tar TEXT, logit_offset INTEGER, logit_bytes INTEGER, split_group TEXT, training_eligible INTEGER, exclusion TEXT)")
        checked = []
        for key, receipt in sorted(receipts.items()):
            done = connection.execute("SELECT sha256 FROM indexed WHERE path=?", (key,)).fetchone()
            if done:
                if done[0] != receipt["sha256"]:
                    raise ValueError("Previously indexed source content changed")
                continue
            kind = "image" if "/image/" in key else "logits"
            entries = []
            with (root / receipt["index"]).open() as stream:
                for line in stream:
                    row = json.loads(line)
                    source_id = row["source_id"].removesuffix("_logits")
                    entries.append((kind, source_id, receipt["path"], row["offset"], row["bytes"]))
            if kind == "logits":
                for entry in (entries[0], entries[-1]):
                    grid = numeric_logits(read_member(root, {"tar": entry[2], "offset": entry[3], "bytes": entry[4]}))
                    checked.append({"tar": receipt["path"], "source_id": entry[1], "shape": list(grid.shape)})
            connection.executemany("INSERT INTO members VALUES (?,?,?,?,?)", entries)
            connection.execute("INSERT INTO indexed VALUES (?,?)", (key, receipt["sha256"]))
        joined = connection.execute("SELECT i.source_id,i.tar,i.offset,i.bytes,l.tar,l.offset,l.bytes FROM members i JOIN members l ON i.source_id=l.source_id WHERE i.kind='image' AND l.kind='logits' AND i.source_id NOT IN (SELECT sample_id FROM pairs)")
        pending = []
        for row in joined:
            stamp = filename_timestamp(row[0])
            date = str(stamp.date())
            eligible = date not in exclusions
            pending.append((*row, "lenghu-filename-date:" + date, int(eligible),
                            None if eligible else "human_holdout_date_or_adjacent_date"))
        connection.executemany("INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?,?,?)", pending)
        counts = {"paired": connection.execute("SELECT count(*) FROM pairs").fetchone()[0],
                  "training_candidates": connection.execute("SELECT count(*) FROM pairs WHERE training_eligible=1").fetchone()[0],
                  "excluded_for_human_holdouts": connection.execute("SELECT count(*) FROM pairs WHERE training_eligible=0").fetchone()[0],
                  "indexed_shards": connection.execute("SELECT count(*) FROM indexed").fetchone()[0]}
        unpaired = connection.execute("SELECT kind,source_id,tar,offset,bytes FROM members WHERE source_id NOT IN (SELECT sample_id FROM pairs)").fetchall()
    expected_pairs = sum(r["member_count"] for k,r in receipts.items() if "/image/" in k)
    expected_logits = sum(r["member_count"] for k,r in receipts.items() if "/logits/" in k)
    complete = len(receipts) == 200
    quarantine = quarantine_unpaired(root, unpaired) if complete else []
    if complete and (counts["paired"] != 429620 or expected_pairs != 429620 + len(quarantine)
                     or expected_logits != 429620 or any(row["kind"] != "image" for row in quarantine)):
        raise ValueError("Complete release failed the one-image/one-logit population contract")
    counts["quarantined_unpaired_images"] = len(quarantine)
    write_json(root / "unpaired_member_quarantine.json", quarantine)
    result = {"dataset_id": "lenghu_sky8_bulk", "status": "complete_paired_teacher_corpus" if complete else "partial_paired_teacher_corpus",
        "download_complete":complete, "geometry":geometry, "downloaded_image_members":expected_pairs,
        "downloaded_logit_members":expected_logits,
        "counts": counts, "database": "pairs.sqlite3", "class_order": ["sky", "cloud", "contamination"],
        "quarantine": {"ledger": "unpaired_member_quarantine.json", "count": len(quarantine),
                       "original_members_preserved": True, "unknown_orphans_rejected": True},
        "class_order_evidence": "Source class_names strings inspected using pickletools opcode parsing without executing pickle; numeric classes [0,1,2].",
        "object_arrays_loaded": False, "sampled_numeric_checks_this_refresh": checked,
        "human_records": str(human_root / "records.jsonl"), "human_excluded_calendar_dates": sorted(exclusions),
        "contract": {"image": [3, 512, 512], "image_dtype": "float32 /255",
                     "teacher_logits": [3, 64, 64], "teacher_dtype": "float32 raw logits",
                     "supervision": "pseudo-label distillation; separate from human targets"},
        "limitations": ["All candidates are for training only; use human/local held-out data for evaluation.",
                        "Exclusion includes +/-1 filename date around every human holdout; source timezone/nighttime remains unverified.",
                        "Every loaded tensor checked; import-time tensor inspection samples first/last of each new logit shard.",
                        "Teacher confidence, class-specific reliability and local transfer not established.",
                        "This refresh includes only completed checksum-verified shards; rerun after bulk download."]}
    result['filename_variants']='Numeric suffixes retained as distinct producer IDs; grouped by base timestamp date, never claimed as independent captures.'
    write_json(root / "adapter_manifest.json", result)
    return result


class LenghuTeacherDataset:
    def __init__(self, root, limit=None):
        self.root = Path(root)
        with np.load(self.root/'geometry_masks.npz',allow_pickle=False) as source:
            self.geometry={k:source[k].astype(bool) for k in source.files}
        with sqlite3.connect(self.root / "pairs.sqlite3") as connection:
            sql = "SELECT p.sample_id,image_tar,image_offset,image_bytes,logit_tar,logit_offset,logit_bytes,g.mask_id FROM pairs p LEFT JOIN geometry g ON p.sample_id=g.sample_id WHERE training_eligible=1 ORDER BY p.sample_id"
            if limit is not None:
                if int(limit) < 1:
                    raise ValueError("limit must be positive")
                sql += " LIMIT " + str(int(limit))
            self.records = connection.execute(sql).fetchall()
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        row = self.records[index]
        payload = read_member(self.root, {"tar": row[1], "offset": row[2], "bytes": row[3]})
        with Image.open(io.BytesIO(payload)) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
        if image.shape != (3, 512, 512):
            raise ValueError("Teacher image dimensions changed")
        logits = numeric_logits(read_member(self.root, {"tar": row[4], "offset": row[5], "bytes": row[6]}))
        if row[7] not in self.geometry:raise ValueError("No producer content mask for frame")
        return {"image": image, "teacher_logits": logits.transpose(2, 0, 1),
                "content_mask":self.geometry[row[7]].copy(), "sample_id": row[0],"geometry_id":row[7]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="ml/data/public/lenghu_sky8_bulk")
    parser.add_argument("--human-root")
    args = parser.parse_args()
    result = prepare_pairs(args.root, args.human_root)
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
