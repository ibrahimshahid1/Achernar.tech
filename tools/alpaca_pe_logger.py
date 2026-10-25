# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Log a mount's RA drive while it tracks at sidereal rate (periodic-error bench run).

Works with ANY ASCOM Alpaca telescope (a commercial mount, ASCOM Remote, or a DIY
Alpaca controller). It connects, sets sidereal tracking, and then only reads:
no guiding, no rate changes, no slews.

    uv run tools/alpaca_pe_logger.py --url http://achernar-mount.local:11111 --duration 4200 --csv PE_LOG.csv

(on macOS prefix with `caffeinate -i` so the laptop does not sleep).

Outputs:
  <stem>_radec.csv   every device: host_unix_s,ra_hours,dec_deg,sidereal_time_hours,tracking,slewing
  PATH (+ PATH.json) only when the device offers the custom Action "achernar:telemetry":
                     the frozen step-counter CSV that analyze_pe_frames.py --pe-log reads,
                     header host_unix_s,device_us,ra_usteps,dec_usteps,ra_rate_usteps_s,dec_rate_usteps_s,state

Connecting energises the motors on mounts that enable their drivers on connect.
Exit status: 0 done, 1 setup failed, 2 bad arguments/refused, 3 aborted by a check, 130 Ctrl-C.
Standard library only; runs with plain python3 >= 3.9 too.
"""
from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import socket
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CSV_HEADER = "host_unix_s,device_us,ra_usteps,dec_usteps,ra_rate_usteps_s,dec_rate_usteps_s,state"
RADEC_HEADER = "host_unix_s,ra_hours,dec_deg,sidereal_time_hours,tracking,slewing"
SIDECAR_KEYS = (
    "motor_full_steps",
    "microsteps",
    "ra_ratio",
    "dec_ratio",
    "ra_invert",
    "sidereal_rate_usteps_s",
    "firmware_version",
    "start_host_unix_s",
    "port",
)
TELEMETRY_ACTION = "achernar:telemetry"
DISCOVERY_PORT = 32227
NOT_IMPLEMENTED = 0x400

# pe_log_health (analyze_pe_frames.py) acceptance rules, mirrored exactly.
HEALTH_MAX_RESIDUAL_USTEPS = 1.5
HEALTH_MAX_GAP_S = 3.0
HEALTH_MAX_RATE_PPM = 50.0


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- Alpaca client


class AlpacaError(Exception):
    """An Alpaca call failed: transport/HTTP problem (number None) or an ASCOM ErrorNumber."""

    def __init__(self, message: str, number: Optional[int] = None) -> None:
        super().__init__(message)
        self.number = number


class AlpacaClient:
    """Minimal ASCOM Alpaca Telescope client (one HTTP request per connection)."""

    def __init__(self, base_url: str, device: int = 0, client_id: int = 1, timeout: float = 2.0) -> None:
        u = urllib.parse.urlsplit(base_url)
        if u.scheme != "http" or not u.hostname:
            raise ValueError(f"expected http://host:port, got {base_url!r}")
        self.host = u.hostname
        self.port = u.port or 80
        self.device = device
        self.client_id = client_id
        self.timeout = timeout
        self._ctid = 0

    @property
    def hostport(self) -> str:
        return f"{self.host}:{self.port}"

    def _ids(self) -> List[Tuple[str, str]]:
        self._ctid = self._ctid % 4294967295 + 1
        return [("ClientID", str(self.client_id)), ("ClientTransactionID", str(self._ctid))]

    def _request(self, method: str, path: str, body: Optional[str] = None) -> dict:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            headers = {}
            data = None
            if body is not None:
                data = body.encode()
                headers["Content-Type"] = "application/x-www-form-urlencoded"
            conn.request(method, path, body=data, headers=headers)
            r = conn.getresponse()
            text = r.read().decode("utf-8", "replace")
        except (OSError, http.client.HTTPException) as e:
            raise AlpacaError(f"{method} {path}: {e}") from e
        finally:
            conn.close()
        if r.status != 200:
            raise AlpacaError(f"{method} {path}: HTTP {r.status} {text[:200]!r}")
        try:
            return json.loads(text)
        except ValueError as e:
            raise AlpacaError(f"{method} {path}: response is not JSON") from e

    @staticmethod
    def _check(member: str, reply: dict) -> dict:
        num = reply.get("ErrorNumber", 0)
        if num:
            raise AlpacaError(f"{member}: ASCOM error {num:#x} {reply.get('ErrorMessage', '')}", int(num))
        return reply

    def management(self, what: str) -> dict:
        return self._check(what, self._request("GET", f"/management/{what}"))

    def get(self, member: str, **params: str) -> object:
        q = urllib.parse.urlencode(list(params.items()) + self._ids())
        reply = self._request("GET", f"/api/v1/telescope/{self.device}/{member}?{q}")
        return self._check(member, reply).get("Value")

    def put(self, member: str, **params: str) -> dict:
        body = urllib.parse.urlencode(list(params.items()) + self._ids())
        return self._check(member, self._request("PUT", f"/api/v1/telescope/{self.device}/{member}", body))

    def action(self, name: str, parameters: str = "") -> object:
        return self.put("action", Action=name, Parameters=parameters).get("Value")

    def telemetry(self) -> dict:
        v = self.action(TELEMETRY_ACTION)
        if not isinstance(v, str):
            raise AlpacaError(f"{TELEMETRY_ACTION}: Value is not a string")
        try:
            tm = json.loads(v)
        except ValueError as e:
            raise AlpacaError(f"{TELEMETRY_ACTION}: Value is not JSON") from e
        if not isinstance(tm, dict):
            raise AlpacaError(f"{TELEMETRY_ACTION}: Value is not a JSON object")
        return tm


def discover(timeout: float = 1.0) -> Optional[str]:
    """ASCOM Alpaca UDP discovery (IPv4 broadcast); returns http://ip:port of the first responder."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.settimeout(0.2)
        s.sendto(b"alpacadiscovery1", ("255.255.255.255", DISCOVERY_PORT))
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                data, (ip, _) = s.recvfrom(1024)
            except socket.timeout:
                continue
            try:
                port = int(json.loads(data)["AlpacaPort"])
            except (ValueError, KeyError, TypeError):
                continue
            return f"http://{ip}:{port}"
    return None


# ----------------------------------------------------------------------------- arguments


def _bounded(lo: float, hi: float, kind=float):
    def conv(text: str):
        try:
            v = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}")
        if not lo <= v <= hi:
            raise argparse.ArgumentTypeError(f"must be in {lo}..{hi}")
        return v

    return conv


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Log RA drive telemetry from any ASCOM Alpaca telescope while it tracks at sidereal rate.",
        epilog="Connecting energises the motors on mounts that enable their drivers on connect.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="Alpaca server base URL, e.g. http://achernar-mount.local:11111")
    src.add_argument("--discover", action="store_true", help="find the server by Alpaca UDP discovery (1 s)")
    p.add_argument("--device", type=_bounded(0, 2147483647, int), default=None,
                   help="Telescope device number (default: the first telescope in configureddevices)")
    p.add_argument("--duration", type=_bounded(1, 86400), required=True, help="seconds to log (1..86400)")
    p.add_argument("--csv", type=Path, required=True, help="output CSV path (frozen telemetry format)")
    p.add_argument("--interval", type=_bounded(0.1, 1.0), default=1.0, help="sample interval, s (0.1..1.0)")
    p.add_argument("--settle-s", type=_bounded(0, 600), default=2.0, help="wait after steady tracking, s")
    p.add_argument("--timeout", type=_bounded(0.1, 30), default=2.0, help="HTTP timeout, s")
    p.add_argument("--client-id", type=_bounded(1, 4294967295, int), default=os.getpid() % 65535 + 1)
    p.add_argument("--force", action="store_true", help="overwrite existing output files")
    p.add_argument("--stop-tracking-at-end", action="store_true", help="PUT Tracking=false when done")
    p.add_argument("--disconnect-at-end", action="store_true", help="PUT Connected=false when done")
    return p.parse_args(argv)


def output_paths(csv_path: Path) -> Tuple[Path, Path, Path]:
    csv_path = Path(csv_path)
    return csv_path, Path(str(csv_path) + ".json"), csv_path.with_name(csv_path.stem + "_radec.csv")


# ----------------------------------------------------------------------------- summary


def _fit_line(t: List[float], y: List[float]) -> Tuple[float, float]:
    n = len(t)
    mt = sum(t) / n
    my = sum(y) / n
    sxx = sum((a - mt) ** 2 for a in t)
    if sxx == 0:
        return 0.0, my
    slope = sum((a - mt) * (b - my) for a, b in zip(t, y)) / sxx
    return slope, my - slope * mt


def summarize(csv_path) -> dict:
    """Health summary of a frozen PE CSV, using exactly the pe_log_health rules
    (analyze_pe_frames.py): residual <= 1.5 usteps, gap <= 3 s, one unique rate
    (rounded to 9 decimals), and |fit|/sidereal - 1 < 50 ppm when the sidecar exists."""
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"status": "EMPTY", "rows": 0}
    dev = [float(r["device_us"]) / 1e6 for r in rows]
    steps = [float(r["ra_usteps"]) for r in rows]
    rates = [float(r["ra_rate_usteps_s"]) for r in rows]
    side = Path(str(csv_path) + ".json")
    meta = json.loads(side.read_text()) if side.exists() else {}
    t = [d - dev[0] for d in dev]
    slope, icpt = _fit_line(t, steps)
    resid = max(abs(s - (slope * x + icpt)) for x, s in zip(t, steps))
    expected = meta.get("sidereal_rate_usteps_s")
    duration = dev[-1] - dev[0]
    out = {
        "rows": len(rows),
        "duration_s": duration,
        "max_gap_s": max((b - a for a, b in zip(dev, dev[1:])), default=0.0),
        "states": sorted({r["state"] for r in rows}),
        "rate_values_usteps_s": sorted({round(x, 9) for x in rates})[:5],
        "fitted_rate_usteps_s": slope,
        "expected_sidereal_usteps_s": expected,
        "rate_error_ppm": (abs(slope) / expected - 1) * 1e6 if expected else None,
        "max_step_deviation_from_line_usteps": resid,
        "ppm_resolution": (2e6 / (duration * expected)) if expected and duration > 0 else None,
    }
    ok = (
        resid <= HEALTH_MAX_RESIDUAL_USTEPS
        and out["max_gap_s"] <= HEALTH_MAX_GAP_S
        and len(out["rate_values_usteps_s"]) == 1
    )
    if expected:
        ok = ok and abs(out["rate_error_ppm"]) < HEALTH_MAX_RATE_PPM
    out["status"] = "CONSTANT_RATE_OK" if ok else "CHECK_LOG"
    return out


def print_summary(s: dict) -> None:
    if s.get("status") == "EMPTY":
        log("summary: no telemetry rows")
        return
    log(f"summary: rows={s['rows']} duration={s['duration_s']:.1f}s")
    log(f"  fitted rate {s['fitted_rate_usteps_s']:.9f} usteps/s, sidereal {s['expected_sidereal_usteps_s']}")
    if s["rate_error_ppm"] is not None:
        log(f"  rate error {s['rate_error_ppm']:+.2f} ppm (resolution {s['ppm_resolution']:.1f} ppm)")
    log(f"  max residual {s['max_step_deviation_from_line_usteps']:.3f} usteps, max gap {s['max_gap_s']:.3f} s")
    log(f"  rates seen {s['rate_values_usteps_s']} states {s['states']}")
    log(f"status: {s['status']}")
    if s.get("ppm_resolution") is not None and s["ppm_resolution"] > HEALTH_MAX_RATE_PPM:
        log("NOTE: run too short for the 50 ppm check")


# ----------------------------------------------------------------------------- run


class _Abort(Exception):
    pass


def _resolve(args: argparse.Namespace) -> Tuple[str, int]:
    if args.discover:
        base = discover(1.0)
        if not base:
            raise _Abort("no Alpaca server answered discovery on UDP 32227")
        log(f"discovered Alpaca server {base}")
    else:
        base = args.url.rstrip("/")
    mc = AlpacaClient(base, 0, args.client_id, args.timeout)
    device = args.device
    try:
        devs = mc.management("v1/configureddevices").get("Value") or []
        scopes = [d for d in devs if str(d.get("DeviceType", "")).lower() == "telescope"]
        if device is None:
            if not scopes:
                raise _Abort(f"{base} lists no Telescope device")
            device = int(scopes[0]["DeviceNumber"])
            log(f"using telescope {device}: {scopes[0].get('DeviceName', '?')}")
        elif scopes and device not in [int(d["DeviceNumber"]) for d in scopes]:
            raise _Abort(f"{base} has no Telescope device {device}")
    except AlpacaError as e:
        if device is None:
            raise _Abort(f"cannot list devices on {base}: {e}")
        log(f"warning: configureddevices failed ({e}); using --device {device}")
    return base, device


def _setup(cl: AlpacaClient) -> bool:
    """Connect, detect telemetry, set sidereal tracking. Returns telemetry mode."""
    if cl.get("connected") is not True:
        log("connecting (Connected=true): on many mounts this energises the motors")
        cl.put("connected", Connected="true")
        if cl.get("connected") is not True:
            raise _Abort("the device did not report Connected=true")
    try:
        actions = cl.get("supportedactions") or []
    except AlpacaError as e:
        log(f"warning: supportedactions failed ({e}); assuming no telemetry")
        actions = []
    telem = any(isinstance(a, str) and a.lower() == TELEMETRY_ACTION for a in actions)
    try:
        cl.put("trackingrate", TrackingRate="0")
    except AlpacaError as e:
        if e.number != NOT_IMPLEMENTED:
            raise _Abort(f"cannot select sidereal tracking: {e}")
        try:
            current = cl.get("trackingrate")
        except AlpacaError:
            current = None
        if current not in (0, None):
            raise _Abort(f"TrackingRate cannot be set and is {current}, not sidereal")
        log("warning: TrackingRate is not settable; assuming sidereal")
    try:
        cl.put("tracking", Tracking="true")
    except AlpacaError as e:
        raise _Abort(f"cannot start tracking: {e}")
    return telem


def _wait_steady(cl: AlpacaClient, settle_s: float) -> dict:
    end = time.monotonic() + 10.0
    tm = None
    while True:
        try:
            tm = cl.telemetry()
            if tm.get("state") == "TRACK" and tm.get("ra_rate_usteps_s") == tm.get("sidereal_rate_usteps_s"):
                break
        except AlpacaError as e:
            log(f"telemetry: {e}")
        if time.monotonic() > end:
            raise _Abort(f"tracking did not reach a steady sidereal rate within 10 s (last {tm})")
        time.sleep(0.2)
    settle_end = time.monotonic() + settle_s
    while time.monotonic() < settle_end:
        time.sleep(min(0.2, max(0.0, settle_end - time.monotonic())))
        try:
            tm = cl.telemetry()
        except AlpacaError as e:
            log(f"telemetry: {e}")
    return tm


def _csv_row(host: float, tm: dict) -> str:
    return (
        f"{host:.6f},{int(tm['device_us'])},{int(tm['ra_usteps'])},{int(tm['dec_usteps'])},"
        f"{float(tm['ra_rate_usteps_s']):.9f},{float(tm['dec_rate_usteps_s']):.9f},{tm['state']}"
    )


def _sidecar(tm: dict, start: float, port: str) -> dict:
    return {
        "motor_full_steps": int(tm["motor_full_steps"]),
        "microsteps": int(tm["microsteps"]),
        "ra_ratio": tm["ra_ratio"],
        "dec_ratio": tm["dec_ratio"],
        "ra_invert": bool(tm["ra_invert"]),
        "sidereal_rate_usteps_s": float(tm["sidereal_rate_usteps_s"]),
        "firmware_version": str(tm["firmware_version"]),
        "start_host_unix_s": float(start),
        "port": port,
    }


def _b(v: object) -> str:
    return "true" if v is True else "false" if v is False else ""


def _end_actions(cl: AlpacaClient, args: argparse.Namespace) -> None:
    if args.stop_tracking_at_end:
        try:
            cl.put("tracking", Tracking="false")
            log("tracking stopped")
        except AlpacaError as e:
            log(f"warning: could not stop tracking: {e}")
    if args.disconnect_at_end:
        try:
            cl.put("connected", Connected="false")
            log("disconnected")
        except AlpacaError as e:
            log(f"warning: could not disconnect: {e}")


def run(args: argparse.Namespace) -> int:
    csv_path, side_path, radec_path = output_paths(args.csv)
    existing = [p for p in (csv_path, side_path, radec_path) if p.exists()]
    if existing and not args.force:
        log("refusing to overwrite " + ", ".join(str(p) for p in existing) + " (use --force)")
        return 2
    try:
        base, device = _resolve(args)
        cl = AlpacaClient(base, device, args.client_id, args.timeout)
        telem = _setup(cl)
    except _Abort as e:
        log(f"error: {e}")
        return 1
    except AlpacaError as e:
        log(f"error: {e}")
        return 1
    except ValueError as e:
        log(f"error: {e}")
        return 2
    if telem:
        log(f"device offers {TELEMETRY_ACTION}: writing {csv_path} and {side_path.name}")
    else:
        log(f"device has no {TELEMETRY_ACTION} action: writing RA/Dec only to {radec_path}")
        for p in (csv_path, side_path):
            if p.exists():
                log(f"warning: {p} is from an earlier run and is not rewritten")

    try:
        if telem:
            _wait_steady(cl, args.settle_s)
        elif args.settle_s > 0:
            time.sleep(args.settle_s)
    except _Abort as e:
        log(f"error: {e}")
        _end_actions(cl, args)
        return 1
    except KeyboardInterrupt:
        log("interrupted")
        _end_actions(cl, args)
        return 130

    rc = 0
    f_csv = f_radec = None
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        f_radec = open(radec_path, "w", newline="")
        f_radec.write(RADEC_HEADER + "\n")
        f_radec.flush()
        if telem:
            f_csv = open(csv_path, "w", newline="")
            f_csv.write(CSV_HEADER + "\n")
            f_csv.flush()
        rows = 0
        sidecar_done = False
        not_track = slewing_n = failed = 0
        t0 = time.monotonic()
        t_end = t0 + float(args.duration)
        last_progress = t0
        k = 0
        log(f"logging for {args.duration:g} s every {args.interval:g} s")
        while True:
            slot = t0 + k * args.interval
            if slot > t_end:
                break
            now = time.monotonic()
            if slot < now - 1e-3:  # late: skip the missed slots, never bunch them
                k = int((now - t0) / args.interval) + 1
                continue
            time.sleep(max(0.0, slot - now))
            k += 1
            ok = True
            tm = None
            if telem:
                try:
                    h0 = time.time()
                    tm = cl.telemetry()
                    h1 = time.time()
                    host = (h0 + h1) / 2.0
                    line = _csv_row(host, tm)
                except (AlpacaError, KeyError, TypeError, ValueError) as e:
                    log(f"sample failed: telemetry: {e}")
                    ok = False
                if ok:
                    f_csv.write(line + "\n")
                    f_csv.flush()
                    rows += 1
                    if not sidecar_done:
                        with open(side_path, "w") as fs:
                            json.dump(_sidecar(tm, host, cl.hostport), fs, indent=2)
                            fs.write("\n")
                        sidecar_done = True
                    not_track = 0 if tm.get("state") == "TRACK" else not_track + 1
            t_ra = time.time()
            vals: Dict[str, object] = {}
            for member in ("rightascension", "declination", "siderealtime", "tracking", "slewing"):
                try:
                    vals[member] = cl.get(member)
                except AlpacaError as e:
                    vals[member] = None
                    if member in ("rightascension", "declination"):
                        log(f"sample: {member}: {e}")
            ra, dec = vals["rightascension"], vals["declination"]
            if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
                lst = vals["siderealtime"]
                lst_s = f"{float(lst):.9f}" if isinstance(lst, (int, float)) else ""
                f_radec.write(
                    f"{t_ra:.6f},{float(ra):.9f},{float(dec):.9f},{lst_s},"
                    f"{_b(vals['tracking'])},{_b(vals['slewing'])}\n"
                )
                f_radec.flush()
                if not telem:
                    rows += 1
            elif not telem:
                ok = False
            failed = 0 if ok else failed + 1
            slewing_n = slewing_n + 1 if vals["slewing"] is True else 0
            if vals["tracking"] is False:
                raise _Abort("the mount stopped tracking")
            if not_track >= 3:
                raise _Abort(f"telemetry state is {tm.get('state') if tm else '?'} for 3 samples, not TRACK")
            if slewing_n >= 3:
                raise _Abort("the mount reports Slewing for 3 samples")
            if failed >= 10:
                raise _Abort("10 consecutive samples failed")
            if time.monotonic() - last_progress >= 60:
                last_progress = time.monotonic()
                log(f"{last_progress - t0:.0f} s, {rows} rows")
        log(f"done: {rows} rows")
    except _Abort as e:
        log(f"ABORT: {e}")
        rc = 3
    except KeyboardInterrupt:
        log("interrupted")
        rc = 130
    finally:
        for f in (f_csv, f_radec):
            if f:
                f.close()
    _end_actions(cl, args)
    if telem and csv_path.exists():
        print_summary(summarize(csv_path))
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:  # during setup, before any output file exists
        log("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
