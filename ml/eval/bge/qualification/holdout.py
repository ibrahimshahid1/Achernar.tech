"""One-shot hold-out guard and ledger (read-only git).

A hold-out run is refused unless:

1. the thresholds file and the split rule are tracked in git and identical to
   HEAD (no staged or unstaged change);
2. the qualification code and the method code it runs are clean vs HEAD
   (``CODE_PATHS``; the ledger itself is exempt);
3. the manifest's split assignment matches the split rule;
4. no earlier ledger entry has the same (git HEAD, thresholds sha256,
   manifest sha256, method config hashes), and no earlier entry consumed any
   of the same hold-out nights, unless ``--acknowledge-repeat`` (recorded).

The ledger ``holdout_ledger.jsonl`` is append-only JSON lines: a ``started``
entry is written before any hold-out pixel is read (a crash still consumes the
hold-out) and a ``completed`` entry with the result's sha256 afterwards.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

from . import LEDGER_SCHEMA, REPO_ROOT, config_sha256

CODE_PATHS = ("ml/eval/bge", "ml/labels/psf_measure.py")
RUST_CODE_PATHS = ("achernar-daemon/src/vision", "achernar-daemon/src/processing")


class HoldoutRefusal(RuntimeError):
    pass


def git(args, repo=REPO_ROOT):
    """``(returncode, stdout)`` of a read-only git command."""
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    return proc.returncode, proc.stdout


def _rel(path, repo) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(Path(repo).resolve()).as_posix()
    except ValueError as exc:
        raise HoldoutRefusal(f"{path} is outside the repository {repo}; it cannot be tracked") from exc


def check_tracked_unchanged(path, repo=REPO_ROOT, git_fn=git) -> None:
    rel = _rel(path, repo)
    code, _ = git_fn(["ls-files", "--error-unmatch", rel], repo)
    if code != 0:
        raise HoldoutRefusal(f"{rel} is not tracked in git: commit it before the hold-out run")
    code, out = git_fn(["status", "--porcelain", "--", rel], repo)
    if code != 0 or out.strip():
        raise HoldoutRefusal(f"{rel} differs from HEAD (staged or unstaged change): {out.strip()}")


def check_code_clean(paths, ledger_path, repo=REPO_ROOT, git_fn=git) -> None:
    try:
        ledger_rel = _rel(ledger_path, repo)
    except HoldoutRefusal:
        ledger_rel = None  # a ledger outside the repository cannot appear in git status
    code, out = git_fn(["status", "--porcelain", "--", *paths], repo)
    if code != 0:
        raise HoldoutRefusal("git status failed: cannot establish that the code is frozen")
    dirty = [line for line in out.splitlines()
             if line.strip() and not (ledger_rel and line[3:].strip().endswith(ledger_rel))]
    if dirty:
        raise HoldoutRefusal("code is not frozen (uncommitted changes): " + "; ".join(dirty[:10]))


def head(repo=REPO_ROOT, git_fn=git) -> str:
    code, out = git_fn(["rev-parse", "HEAD"], repo)
    if code != 0 or not out.strip():
        raise HoldoutRefusal("cannot read git HEAD")
    return out.strip()


def read_ledger(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise HoldoutRefusal(f"ledger {path}:{n} is not valid JSON ({exc}); refusing to append") from exc
    return entries


def run_key(git_head, thresholds_sha, manifest_sha, config_hashes) -> str:
    return config_sha256({"git_head": git_head, "thresholds_sha256": thresholds_sha, "manifest_sha256": manifest_sha,
                          "method_config_sha256": dict(sorted(config_hashes.items()))})


def guard(*, thresholds_path, split_rule_path, ledger_path, manifest_sha, config_hashes, holdout_nights,
          split_problems, acknowledge_repeat=False, include_rust=False, repo=REPO_ROOT, git_fn=git,
          command=None, now=None) -> dict:
    """Run every check; on success append the ``started`` entry and return it."""
    from . import sha256_file

    if split_problems:
        raise HoldoutRefusal("manifest split assignment does not match the split rule: " + "; ".join(split_problems[:5]))
    if not holdout_nights:
        raise HoldoutRefusal("the manifest has no hold-out night")
    check_tracked_unchanged(thresholds_path, repo, git_fn)
    check_tracked_unchanged(split_rule_path, repo, git_fn)
    check_code_clean(CODE_PATHS + (RUST_CODE_PATHS if include_rust else ()), ledger_path, repo, git_fn)
    git_head = head(repo, git_fn)
    th_sha = sha256_file(thresholds_path)
    key = run_key(git_head, th_sha, manifest_sha, config_hashes)
    entries = read_ledger(ledger_path)
    same = [e for e in entries if e.get("run_key") == key and e.get("event") == "started"]
    overlap = [e for e in entries if e.get("event") == "started" and e.get("run_key") != key
               and set(e.get("holdout_nights") or []) & set(holdout_nights)]
    if (same or overlap) and not acknowledge_repeat:
        what = []
        if same:
            what.append(f"{len(same)} earlier run(s) with the same HEAD/thresholds/manifest/config (first {same[0]['utc']})")
        if overlap:
            what.append(f"{len(overlap)} earlier run(s) consumed hold-out night(s) "
                        f"{sorted(set().union(*[set(e['holdout_nights']) for e in overlap]) & set(holdout_nights))}")
        raise HoldoutRefusal("hold-out already consumed: " + "; ".join(what) +
                             ". A repeat is not a one-shot hold-out; pass --acknowledge-repeat to run it anyway "
                             "(recorded in the ledger and the result)")
    entry = {
        "schema": LEDGER_SCHEMA, "event": "started",
        "utc": (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds"),
        "git_head": git_head, "thresholds_path": _rel(thresholds_path, repo), "thresholds_sha256": th_sha,
        "split_rule_sha256": sha256_file(split_rule_path), "manifest_sha256": manifest_sha,
        "method_config_sha256": dict(sorted(config_hashes.items())), "holdout_nights": sorted(holdout_nights),
        "run_key": key, "repeat": bool(same or overlap), "acknowledge_repeat": bool(acknowledge_repeat),
        "previous_same_key": len(same), "previous_overlapping": len(overlap), "command": command,
    }
    append(ledger_path, entry)
    return entry


def append(ledger_path, entry: dict) -> None:
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def complete(ledger_path, started: dict, result_sha256: str, summary: dict, now=None) -> dict:
    entry = {"schema": LEDGER_SCHEMA, "event": "completed",
             "utc": (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds"),
             "run_key": started["run_key"], "started_utc": started["utc"], "result_sha256": result_sha256,
             "criteria": summary}
    append(ledger_path, entry)
    return entry
