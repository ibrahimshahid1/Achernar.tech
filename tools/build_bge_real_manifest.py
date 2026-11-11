#!/usr/bin/env python3
"""Build a bge-real-frames-v2 manifest for the background-estimator real-frame qualification.

Three sources, all read-only (metadata, file names, sha256; no pixels decoded):

    # (a) a directory of daemon capture bundles / declared analysis FITS + a sidecar log
    python3 tools/build_bge_real_manifest.py dir --frames ~/bge-nights/ --sidecar ~/bge-nights/log.csv \\
        --out ~/bge-nights/manifest.json

    # (b) rows written by tools/import_session.py into an Achernar DB
    python3 tools/build_bge_real_manifest.py import-db --db achernar-daemon/achernar.db \\
        --capture-root ~/achernar-frames --sidecar log.csv --out manifest.json

    # (c) the daemon's own captures and autofocus sweeps (DB + capture archive)
    python3 tools/build_bge_real_manifest.py daemon-db --db achernar-daemon/achernar.db \\
        [--capture-root <frames_dir>/captures] --sidecar log.jsonl --out manifest.json

    # an empty sidecar CSV with every column
    python3 tools/build_bge_real_manifest.py template > log.csv

``--set key=value`` adds sidecar defaults (e.g. ``--set site_utc_offset=-05:00
--set rig_id=z114``). Nights are split by ``ml/eval/bge/qualification/split-rule-v1.json``.
The manifest lists every frame it could not use under ``excluded`` with the reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.eval.bge.qualification import DEFAULT_SPLIT_RULE  # noqa: E402
from ml.eval.bge.qualification import builders as B  # noqa: E402
from ml.eval.bge.qualification.sidecar import TEMPLATE_CSV, SidecarLog  # noqa: E402


def _sidecar(args) -> SidecarLog:
    log = SidecarLog.read(args.sidecar) if args.sidecar else SidecarLog.empty()
    defaults = {}
    for item in args.set or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set {item!r}: expected key=value")
        defaults[key.strip()] = value.strip()
    return log.with_defaults(defaults) if defaults else log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dir", "import-db", "daemon-db"):
        p = sub.add_parser(name)
        p.add_argument("--sidecar", type=Path)
        p.add_argument("--set", action="append", help="sidecar default key=value (repeatable)")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--split-rule", type=Path, default=DEFAULT_SPLIT_RULE)
        if name == "dir":
            p.add_argument("--frames", type=Path, required=True)
            p.add_argument("--no-verify", action="store_true", help="do not hash bundle files against descriptors")
        else:
            p.add_argument("--db", type=Path, required=True)
            p.add_argument("--capture-root", type=Path, default=None)
            p.add_argument("--verify", action="store_true", help="re-hash every analysis file (slow)")
        if name == "daemon-db":
            p.add_argument("--include-imports", action="store_true", help="also rows from tools/import_session.py")
    sub.add_parser("template")
    args = ap.parse_args(argv)
    if args.cmd == "template":
        sys.stdout.write(TEMPLATE_CSV)
        return 0
    sidecar = _sidecar(args)
    if args.cmd == "dir":
        manifest = B.build_from_directory(args.frames, sidecar, verify=not args.no_verify, rule_path=args.split_rule)
    else:
        root = args.capture_root or B.default_capture_root(args.db)
        if args.cmd == "import-db":
            manifest = B.build_from_import_session_db(args.db, root, sidecar, verify=args.verify,
                                                      rule_path=args.split_rule)
        else:
            manifest = B.build_from_daemon_db(args.db, root, sidecar, include_imports=args.include_imports,
                                              verify=args.verify, rule_path=args.split_rule)
    B.write_manifest(manifest, args.out)
    counts = {}
    for f in manifest["frames"]:
        counts[(f["night_id"], f["split"], f["frame_type"])] = counts.get((f["night_id"], f["split"], f["frame_type"]), 0) + 1
    print(f"wrote {args.out}: {len(manifest['frames'])} frames, {len(manifest['excluded'])} excluded")
    for (night, split, ftype), n in sorted(counts.items()):
        print(f"  {night} {split:<11} {ftype:<5} {n}")
    reasons = {}
    for e in manifest["excluded"]:
        key = e["reason"].split(":")[0].split(" (")[0]
        reasons[key] = reasons.get(key, 0) + 1
    for reason, n in sorted(reasons.items()):
        print(f"  excluded {n} x {reason}")
    unused = manifest["builder"].get("unused_sidecar_rows") or []
    if unused:
        print(f"  ! {len(unused)} sidecar row(s) matched no frame: {json.dumps(unused[:3])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
