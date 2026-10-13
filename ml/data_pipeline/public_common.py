"""Small, dependency-light primitives for reproducible public-data intake."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def download_file(url: str, destination: str | Path, expected_sha256: str | None = None,
                  max_bytes: int = 1024**3, expected_size: int | None = None) -> dict:
    """Bounded atomic HTTPS download; never reuse a partial or mismatched file."""
    if not url.startswith("https://"):
        raise ValueError("Public intake requires HTTPS")
    if max_bytes <= 0 or (expected_size is not None and expected_size > max_bytes):
        raise ValueError("Invalid download size bound")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        size = destination.stat().st_size
        digest = sha256_file(destination)
        if (0 < size <= max_bytes and (expected_size is None or size == expected_size)
                and (expected_sha256 is None or digest == expected_sha256)):
            return {"path": str(destination), "url": url, "sha256": digest, "bytes": size}
        raise ValueError(f"Existing download does not match its contract: {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "Achernar-public-data-intake/1"})
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".partial",
                                            dir=destination.parent)
    try:
        count = 0
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(request, timeout=90) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > max_bytes:
                raise ValueError("Remote file exceeds download bound")
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                count += len(chunk)
                if count > max_bytes:
                    raise ValueError("Remote stream exceeds download bound")
                output.write(chunk)
        if count == 0 or (expected_size is not None and count != expected_size):
            raise ValueError("Remote file has unexpected size")
        digest = sha256_file(temporary)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("Remote SHA-256 mismatch")
        Path(temporary).replace(destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"path": str(destination), "url": url, "sha256": digest, "bytes": count}


def grouped_split(group_ids, seed: int = 42) -> dict[str, str]:
    """Stable group-hash partition; adding a group never moves an existing one."""
    result = {}
    for group in sorted(set(map(str, group_ids))):
        value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
        result[group] = "train" if value < 0.70 else "validation" if value < 0.85 else "test"
    return result
