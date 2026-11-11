"""Sidecar log ``bge-real-sidecar-v1``: per-frame metadata the camera/daemon does not record.

Two encodings, same columns:

* CSV with a header row; lines starting with ``#`` are comments.
* JSONL, one object per line; a line ``{"defaults": {...}}`` sets values for
  every frame in the file (several such lines merge in order and apply to the
  whole file, not only to the rows after them); ``#`` lines are comments.

A row applies to frames by exactly one selector:

* ``file``: path relative to the frames directory, or a bare file/bundle name;
* ``frame_id`` / ``capture_id``: the daemon's ids (daemon or import_session DB);
* ``t_start`` + ``t_end``: every frame whose timestamp lies in the closed UTC
  window (a "from 21:10 to 21:40 the Barlow was in" row).

Precedence for one frame: defaults < window rows (file order) < its own
row(s). Unknown columns are an error (a misspelt ``repeat_group`` would
otherwise vanish silently). Values the DB records authoritatively are never
overridden by the sidecar; conflicts are reported by the builder.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from . import FRAME_TYPES
from .split import parse_utc

SELECTORS = ("file", "frame_id", "capture_id", "t_start", "t_end")
FLOAT_FIELDS = ("iso", "gain", "exposure_s", "sensor_temp_c", "ambient_temp_c", "focus_position",
                "focuser_stepsize_um", "focal_ratio", "pointing_ra_deg", "pointing_dec_deg", "pointing_alt_deg",
                "pointing_az_deg", "saturation_adu", "gain_e_per_adu")
INT_FIELDS = ("width", "height", "hdu")
TEXT_FIELDS = ("frame_type", "night_id", "rig_id", "camera", "optical_train", "timestamp_utc", "focus_run_id",
               "sweep_direction", "repeat_group", "target", "notes", "analysis_representation", "analysis_bundle",
               "mask_bad_pixel", "mask_invalid", "mask_edge", "mask_native_saturation", "site_utc_offset",
               "exclude_reason")
BOOL_FIELDS = ("exclude",)
ALL_FIELDS = SELECTORS + FLOAT_FIELDS + INT_FIELDS + TEXT_FIELDS + BOOL_FIELDS
SWEEP_DIRECTIONS = ("increasing", "decreasing", "mixed")
#: Columns a builder should be told about per frame (documented field checklist).
REQUIRED_FOR_PROTOCOL = ("frame_type", "timestamp_utc", "rig_id", "camera", "optical_train", "iso", "exposure_s")


class SidecarError(ValueError):
    pass


def _coerce(key, value, where):
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if key in FLOAT_FIELDS:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise SidecarError(f"{where}: {key}={value!r} is not a number") from exc
    if key in INT_FIELDS:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise SidecarError(f"{where}: {key}={value!r} is not an integer") from exc
    if key in BOOL_FIELDS:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "y"):
            return True
        if text in ("0", "false", "no", "n"):
            return False
        raise SidecarError(f"{where}: {key}={value!r} is not a boolean")
    text = str(value).strip()
    if key == "frame_type":
        text = text.lower()
        if text not in FRAME_TYPES:
            raise SidecarError(f"{where}: frame_type {text!r} not in {FRAME_TYPES}")
    if key == "sweep_direction":
        text = text.lower()
        text = {"in": "decreasing", "out": "increasing", "inward": "decreasing", "outward": "increasing"}.get(text, text)
        if text not in SWEEP_DIRECTIONS:
            raise SidecarError(f"{where}: sweep_direction {text!r} not in {SWEEP_DIRECTIONS}")
    if key in ("timestamp_utc", "t_start", "t_end") and parse_utc(text) is None:
        raise SidecarError(f"{where}: {key}={text!r} must be ISO-8601 with a UTC offset (e.g. 2026-11-15T02:13:05Z)")
    return text


def _clean_row(raw: dict, where: str) -> dict:
    unknown = set(raw) - set(ALL_FIELDS)
    if unknown:
        raise SidecarError(f"{where}: unknown column(s) {sorted(unknown)}; allowed: {list(ALL_FIELDS)}")
    return {k: v for k, v in ((k, _coerce(k, raw[k], where)) for k in raw) if v is not None}


class SidecarLog:
    def __init__(self, rows=None, defaults=None, source=None):
        self.source = source
        self.defaults = dict(defaults or {})
        self.frame_rows, self.window_rows = [], []
        self.used = set()
        for index, row in enumerate(rows or []):
            selectors = [k for k in ("file", "frame_id", "capture_id") if k in row]
            window = "t_start" in row or "t_end" in row
            where = f"{source or 'sidecar'} row {index + 1}"
            if window and not ("t_start" in row and "t_end" in row):
                raise SidecarError(f"{where}: a window row needs both t_start and t_end")
            if len(selectors) + int(window) != 1:
                raise SidecarError(f"{where}: exactly one selector of file | frame_id | capture_id | t_start+t_end")
            (self.window_rows if window else self.frame_rows).append((index, row))

    @classmethod
    def read(cls, path) -> "SidecarLog":
        path = Path(path)
        text = path.read_text()
        rows, defaults = [], {}
        if path.suffix.lower() in (".jsonl", ".ndjson", ".json"):
            for n, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SidecarError(f"{path}:{n}: {exc}") from exc
                if set(obj) == {"defaults"}:
                    defaults.update(_clean_row(obj["defaults"], f"{path}:{n} defaults"))
                else:
                    rows.append(_clean_row(obj, f"{path}:{n}"))
        else:
            lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
            reader = csv.DictReader(io.StringIO("\n".join(lines)))
            for n, raw in enumerate(reader, 2):
                rows.append(_clean_row({k.strip(): v for k, v in raw.items() if k is not None}, f"{path}:{n}"))
        return cls(rows, defaults, source=str(path))

    @classmethod
    def empty(cls, defaults=None) -> "SidecarLog":
        return cls([], {k: v for k, v in (defaults or {}).items()}, source=None)

    def with_defaults(self, extra: dict) -> "SidecarLog":
        self.defaults.update(_clean_row(extra, "command-line --set"))
        return self

    def tags_for(self, file=None, frame_id=None, capture_id=None, timestamp=None) -> dict:
        tags = dict(self.defaults)
        t = parse_utc(timestamp) if timestamp else None
        if t is not None:
            for index, row in self.window_rows:
                if parse_utc(row["t_start"]) <= t <= parse_utc(row["t_end"]):
                    tags.update({k: v for k, v in row.items() if k not in SELECTORS})
                    self.used.add(index)
        names = set()
        if file:
            p = Path(str(file))
            names = {str(file), p.as_posix(), p.name, p.stem}
        for index, row in self.frame_rows:
            hit = (("file" in row and (row["file"] in names or Path(row["file"]).as_posix() in names))
                   or ("frame_id" in row and frame_id is not None and row["frame_id"] == str(frame_id))
                   or ("capture_id" in row and capture_id is not None and row["capture_id"] == str(capture_id)))
            if hit:
                tags.update({k: v for k, v in row.items() if k not in SELECTORS})
                self.used.add(index)
        return tags

    def mask_paths(self, frames_dir) -> set:
        """Relative paths named in mask_* columns (these files are masks, not frames)."""
        out = set()
        rows = [self.defaults] + [r for _i, r in self.frame_rows] + [r for _i, r in self.window_rows]
        for row in rows:
            for name in ("mask_bad_pixel", "mask_invalid", "mask_edge", "mask_native_saturation"):
                if row.get(name):
                    p = (Path(frames_dir) / row[name]).resolve()
                    root = Path(frames_dir).resolve()
                    out.add(p.relative_to(root).as_posix() if p.is_relative_to(root) else str(p))
        return out

    def unused_rows(self) -> list:
        """Per-frame rows that matched no frame (likely filename typos)."""
        return [row for index, row in self.frame_rows if index not in self.used]


def pointing_from(tags: dict) -> dict | None:
    out = {k[len("pointing_"):]: tags[k] for k in ("pointing_ra_deg", "pointing_dec_deg", "pointing_alt_deg",
                                                   "pointing_az_deg") if k in tags}
    if "target" in tags:
        out["target"] = tags["target"]
    return out or None


TEMPLATE_CSV = (
    "# bge-real-sidecar-v1 field log. One row per frame (file=...), or a window row (t_start,t_end) that tags\n"
    "# every frame in that UTC interval. Leave a cell empty when unknown; never guess.\n"
    "file,frame_id,capture_id,t_start,t_end,frame_type,timestamp_utc,rig_id,camera,optical_train,iso,exposure_s,"
    "sensor_temp_c,ambient_temp_c,focus_position,focuser_stepsize_um,focal_ratio,focus_run_id,sweep_direction,"
    "repeat_group,pointing_ra_deg,pointing_dec_deg,pointing_alt_deg,pointing_az_deg,target,notes\n"
)
