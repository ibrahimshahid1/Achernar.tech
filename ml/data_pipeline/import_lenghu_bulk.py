"""Download pinned Lenghu image/logit TARs without duplicating them by unpacking.

The human masks remain a separate supervised dataset. Bulk supervision is
teacher-generated. TAR indexes support lazy image/logit loading later.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time
import urllib.request

from .public_common import download_file, sha256_file, write_json, write_jsonl

REPO = "ruiyicheng/LenghuSky-8"
REVISION = "30f97ef28201a4eb4ec639c20f19c3b166c56bcb"


def download_auxiliary(root, inventory):
    path=root/'auxiliary_receipts.json'
    previous={r['path']:r for r in json.loads(path.read_text())} if path.exists() else {}
    receipts=[]
    for item in inventory['files']:
        if item.get('path') not in {'README.md','data/bkg_mask','data/calibration.zip','data/interrupt.csv'}:continue
        relative='source/'+item['path']
        expected=item.get('lfs',{}).get('oid') or previous.get(relative,{}).get('sha256')
        receipt=download_file(f"https://huggingface.co/datasets/{REPO}/resolve/{inventory['revision']}/{item['path']}",
            root/relative,expected_sha256=expected,expected_size=item['size'],max_bytes=item['size'])
        receipt['path']=relative;receipts.append(receipt)
    if len(receipts)!=4:raise ValueError('Missing geometry/provenance source files')
    write_json(path,receipts)
    return receipts


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Achernar-public-data-intake/1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read(5_000_001)
    if len(payload) > 5_000_000:
        raise ValueError("Unexpectedly large metadata response")
    return json.loads(payload)


def index_tar(path, destination):
    """Index byte offsets, retaining source member names; never extract paths."""
    rows = []
    with tarfile.open(path, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe archive member path")
            rows.append({"member": member.name, "offset": member.offset_data,
                         "bytes": member.size, "source_id": name.stem})
    if not rows:
        raise ValueError("Empty source TAR")
    write_jsonl(destination, rows)
    return len(rows)


def prepare(root, *, workers=4, limit_pairs=None, reserve_gb=20, max_download_gb=85):
    root = Path(root) / "lenghu_sky8_bulk"
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "source_inventory.json"
    if snapshot.exists():
        inventory = json.loads(snapshot.read_text())
    else:
        revision = REVISION
        files = fetch_json(f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}?recursive=true&expand=false&limit=1000")
        inventory = {"repository": REPO, "revision": revision, "files": files}
        write_json(snapshot, inventory)
    revision = inventory["revision"]
    if revision!=REVISION:raise ValueError("Unexpected cached source revision")
    download_auxiliary(root,inventory)
    files = [item for item in inventory["files"] if item.get("type") == "file" and
             (item["path"].startswith("data/image/") or item["path"].startswith("data/logits/")) and item["path"].endswith(".tar")]
    def shard_number(item):
        return int(Path(item["path"]).stem.rsplit("_", 1)[-1])
    if limit_pairs is not None:
        files = [item for item in files if shard_number(item) < limit_pairs]
    files.sort(key=lambda item: (shard_number(item), item["path"]))
    if not files or len(files) % 2:
        raise ValueError("Image/logit shard inventory incomplete")
    expected = sum(item["size"] for item in files)
    if expected > max_download_gb * 1e9:
        raise ValueError(f"Inventory {expected} bytes exceeds explicit download budget")
    receipt_path = root / "download_receipts.json"
    completed = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    lock, reserved = threading.Lock(), [0]
    progress_path = root / "progress.json"

    def progress():
        active = [completed[item["path"]] for item in files if item["path"] in completed]
        write_json(progress_path, {"revision": revision, "expected_files": len(files),
            "completed_files": len(active), "expected_bytes": expected,
            "completed_bytes": sum(item["bytes"] for item in active),
            "updated_at": datetime.now(timezone.utc).isoformat(), "status": "downloading"})

    def one(item):
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe inventory path")
        destination = root / "source" / relative
        digest = item.get("lfs", {}).get("oid")
        if digest is None:
            raise ValueError("No pinned content hash for large shard")
        size = item["size"]
        need = 0 if destination.exists() else size
        with lock:
            if shutil.disk_usage(root).free - reserved[0] - need < reserve_gb * 1e9:
                raise OSError("Disk reserve reached; resume with --root on the external drive")
            reserved[0] += need
        try:
            for attempt in range(3):
                try:
                    receipt = download_file(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{item['path']}",
                                            destination, expected_sha256=digest, expected_size=size, max_bytes=size)
                    break
                except (OSError, TimeoutError) as error:
                    if attempt == 2:raise
                    print(json.dumps({"retry":attempt+1,"path":item['path'],"error":str(error)}),flush=True)
                    time.sleep(2**attempt)
            index_path = root / "indexes" / f"{destination.stem}.jsonl"
            count = index_tar(destination, index_path)
            receipt.update(path=str(destination.relative_to(root)), index=str(index_path.relative_to(root)),
                           member_count=count)
            with lock:
                completed[item["path"]] = receipt
                write_json(receipt_path, completed)
                progress()
                print(json.dumps({"completed": len([f for f in files if f["path"] in completed]),
                                  "total": len(files), "file": item["path"], "bytes": size}), flush=True)
            return receipt
        finally:
            with lock:
                reserved[0] -= need

    progress()
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(one, item): item for item in files}
        for future in as_completed(jobs):
            try:
                future.result()
            except Exception as error:
                failures.append({"path": jobs[future]["path"], "error": str(error)})
                print(json.dumps(failures[-1]), flush=True)
    write_json(root / "failures.json", failures)
    sources = [completed[item["path"]] for item in files if item["path"] in completed]
    manifest = {"schema_version": 1, "dataset_id": "lenghu_sky8_bulk", "source_revision": revision,
        "task": "cloud_segmentation_teacher_supervision", "status": "downloaded_indexed" if not failures else "incomplete",
        "expected_files": len(files), "completed_files": len(sources), "sources": sources,
        "expected_bytes": expected, "completed_bytes": sum(source["bytes"] for source in sources),
        "storage": "original TAR plus byte-offset indexes; no duplicate full extraction",
        "contract": {"image": "RGB JPEG512x512", "logits_grid": [64, 64, 3],
                     "logits_up": [512, 512, 3], "dtype": "float32 logits; not calibrated probabilities",
                     "class_order": ["sky","cloud","contamination"],
                     "object_arrays": "never load with pickle"},
        "training_eligible": False,
        "license": {"dataset_card": "Apache-2.0", "teacher_backbone_downloaded": False},
        "limitations": ["Bulk logits are model-derived, not human labels.",
                        "Human holdout originals and adjacent same-night sequences must be excluded before pseudo-label training.",
                        "Source logits contain object metadata; the adapter reads only numeric arrays and validates pairing separately.",
                        "Does not establish local R50 transparency classes or transfer performance."]}
    write_json(root / "manifest.json", manifest)
    with lock:
        progress()
    final_progress = json.loads(progress_path.read_text())
    final_progress["status"] = manifest["status"]
    write_json(progress_path, final_progress)
    if failures:
        raise RuntimeError(f"{len(failures)} shards incomplete; successful downloads retained for resume")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="ml/data/public")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--reserve-gb", type=float, default=20)
    parser.add_argument("--max-download-gb", type=float, default=85)
    args = parser.parse_args()
    result = prepare(args.root, workers=args.workers, limit_pairs=args.limit_pairs,
                     reserve_gb=args.reserve_gb, max_download_gb=args.max_download_gb)
    print(json.dumps({key: result[key] for key in ("dataset_id", "completed_files", "completed_bytes", "status")}))


if __name__ == "__main__":
    main()
