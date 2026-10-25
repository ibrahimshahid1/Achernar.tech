"""Tests for alpaca_pe_logger.py against an in-process fake Alpaca telescope.

The fake is a linear step-counter model behind http.server, with the custom
telemetry Action switchable on and off, so both the frozen-CSV path and the
plain-Alpaca (RA/Dec only) path are exercised without hardware.
"""
import contextlib
import csv
import io
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alpaca_pe_logger as pel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "mount_cad_esp_32_firmware" / "achernar-mount-esp32" / "host_sim" / "build" / "achernar-mount-sim"
SIDEREAL = 12.478516210
FROZEN = "host_unix_s,device_us,ra_usteps,dec_usteps,ra_rate_usteps_s,dec_rate_usteps_s,state"


class FakeMount:
    """Linear counter model: after tracking starts (and an optional ramp of
    `steady_after` s) ra_usteps = floor(SIDEREAL * t)."""

    def __init__(self, telemetry=True, steady_after=0.0, stop_tracking_after=None, trackingrate_ni=False):
        self.telemetry = telemetry
        self.steady_after = steady_after
        self.stop_tracking_after = stop_tracking_after
        self.trackingrate_ni = trackingrate_ni
        self.lock = threading.Lock()
        self.t0 = time.monotonic()
        self.connected = False
        self.tracking = False
        self.t_track = None
        self.trackingrate = 2
        self.puts = []  # (member, params)
        self.stid = 0

    def _now(self):
        return time.monotonic() - self.t0

    def tick(self):
        if self.tracking and self.stop_tracking_after is not None:
            if self._now() - self.t_track >= self.stop_tracking_after:
                self.tracking = False

    def steady_since(self):
        return None if self.t_track is None else self.t_track + self.steady_after

    def telem(self):
        now = self._now()
        steady = self.steady_since()
        if not self.tracking or steady is None:
            state, rate, pos = "IDLE", 0.0, 0
        elif now < steady:
            state, rate, pos = "STOPPING", SIDEREAL * 0.5, 0
        else:
            state, rate, pos = "TRACK", SIDEREAL, int(math.floor(SIDEREAL * (now - steady)))
        return {
            "device_us": int(now * 1e6), "ra_usteps": pos, "dec_usteps": 0,
            "ra_rate_usteps_s": rate, "dec_rate_usteps_s": 0.0, "state": state,
            "microsteps": 64, "motor_full_steps": 200, "ra_ratio": 84, "dec_ratio": 64,
            "ra_invert": False, "sidereal_rate_usteps_s": SIDEREAL, "firmware_version": "9.9.9",
            "extra_key_ignored": 1,
        }

    def handle(self, method, path, params):
        with self.lock:
            self.tick()
            self.stid += 1
            out = {"ClientTransactionID": int(params.get("ClientTransactionID", "0") or 0),
                   "ServerTransactionID": self.stid, "ErrorNumber": 0, "ErrorMessage": ""}
            if path == "/management/v1/configureddevices":
                out["Value"] = [{"DeviceName": "Fake Scope", "DeviceType": "Telescope", "DeviceNumber": 0,
                                 "UniqueID": "fake"}]
                return 200, out
            prefix = "/api/v1/telescope/0/"
            if not path.startswith(prefix):
                return 400, None
            m = path[len(prefix):]
            if method == "PUT":
                self.puts.append((m, dict(params)))
            if m == "connected":
                if method == "PUT":
                    self.connected = params["Connected"] == "true"
                else:
                    out["Value"] = self.connected
                return 200, out
            if not self.connected:
                out.update(ErrorNumber=0x407, ErrorMessage="Not connected")
                if method == "GET":
                    out["Value"] = 0
                return 200, out
            if method == "GET":
                now = self._now()
                values = {
                    "supportedactions": ["achernar:telemetry"] if self.telemetry else [],
                    "trackingrate": self.trackingrate,
                    "tracking": self.tracking,
                    "slewing": False,
                    "rightascension": 5.5,
                    "declination": 20.0,
                    "siderealtime": (1.0 + now * 1.0027379 / 3600) % 24,
                }
                if m not in values:
                    return 400, None
                out["Value"] = values[m]
                return 200, out
            if m == "trackingrate":
                if self.trackingrate_ni:
                    out.update(ErrorNumber=0x400, ErrorMessage="Not implemented")
                else:
                    self.trackingrate = int(params["TrackingRate"])
                return 200, out
            if m == "tracking":
                on = params["Tracking"] == "true"
                if on and not self.tracking:
                    self.t_track = self._now()
                self.tracking = on
                return 200, out
            if m == "action":
                if not self.telemetry or params.get("Action") != "achernar:telemetry":
                    out.update(ErrorNumber=0x40C, ErrorMessage="Action not implemented")
                else:
                    out["Value"] = json.dumps(self.telem(), separators=(",", ":"))
                return 200, out
            return 400, None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _go(self, method):
        u = urllib.parse.urlsplit(self.path)
        if method == "PUT":
            n = int(self.headers.get("Content-Length", "0"))
            params = dict(urllib.parse.parse_qsl(self.rfile.read(n).decode(), keep_blank_values=True))
        else:
            params = dict(urllib.parse.parse_qsl(u.query, keep_blank_values=True))
        status, out = self.server.mount.handle(method, u.path, params)
        body = json.dumps(out).encode() if out is not None else b"bad request"
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._go("GET")

    def do_PUT(self):
        self._go("PUT")


@contextlib.contextmanager
def fake_server(mount):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.mount = mount
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def run_logger(*argv):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = pel.run(pel.parse_args(list(argv)))
    return rc, err.getvalue()


def write_csv(path, rows, sidereal=SIDEREAL):
    with open(path, "w") as f:
        f.write(FROZEN + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with open(str(path) + ".json", "w") as f:
        json.dump({"sidereal_rate_usteps_s": sidereal}, f)


class LoggerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.csv = self.dir / "PE_LOG.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, mount, *extra, duration="2", interval="0.2"):
        with fake_server(mount) as url:
            rc, err = run_logger("--url", url, "--duration", duration, "--interval", interval, "--settle-s", "0",
                                 "--csv", str(self.csv), *extra)
        return rc, err, url

    def test_frozen_header_and_row_format(self):
        rc, err, _ = self._run(FakeMount())
        self.assertEqual(rc, 0, err)
        lines = self.csv.read_text().split("\n")
        self.assertEqual(lines[0], FROZEN)
        self.assertEqual(lines[-1], "", "LF line endings, final newline")
        self.assertNotIn("\r", self.csv.read_text())
        rows = [ln for ln in lines[1:] if ln]
        self.assertGreaterEqual(len(rows), 9)
        for ln in rows:
            f = ln.split(",")
            self.assertEqual(len(f), 7)
            self.assertRegex(f[0], r"^\d+\.\d{6}$")
            self.assertRegex(f[1], r"^\d+$")
            self.assertRegex(f[2], r"^-?\d+$")
            self.assertEqual(f[4], "12.478516210")
            self.assertEqual(f[5], "0.000000000")
            self.assertEqual(f[6], "TRACK")
        self.assertIn("status: ", err)
        self.assertIn("NOTE: run too short for the 50 ppm check", err)
        # one RA/Dec row per sample as well
        radec = (self.dir / "PE_LOG_radec.csv").read_text().split("\n")
        self.assertEqual(radec[0], "host_unix_s,ra_hours,dec_deg,sidereal_time_hours,tracking,slewing")
        self.assertRegex(radec[1], r"^\d+\.\d{6},5\.500000000,20\.000000000,\d+\.\d{9},true,false$")

    def test_sidecar_keys_exact_order_and_types(self):
        rc, err, url = self._run(FakeMount())
        self.assertEqual(rc, 0, err)
        side = Path(str(self.csv) + ".json")
        text = side.read_text()
        self.assertTrue(text.endswith("}\n"))
        meta = json.loads(text)
        self.assertEqual(list(meta), list(pel.SIDECAR_KEYS))
        self.assertEqual(list(meta), ["motor_full_steps", "microsteps", "ra_ratio", "dec_ratio", "ra_invert",
                                      "sidereal_rate_usteps_s", "firmware_version", "start_host_unix_s", "port"])
        types = [int, int, (int, float), (int, float), bool, float, str, float, str]
        for key, ty in zip(meta, types):
            self.assertIsInstance(meta[key], ty, key)
        self.assertNotIsInstance(meta["motor_full_steps"], bool)
        self.assertEqual(meta["sidereal_rate_usteps_s"], SIDEREAL)
        self.assertEqual(meta["firmware_version"], "9.9.9")
        self.assertEqual(meta["port"], url[len("http://"):])
        first = self.csv.read_text().split("\n")[1].split(",")[0]
        self.assertEqual(f"{meta['start_host_unix_s']:.6f}", first)

    def test_radec_only_for_plain_device(self):
        mount = FakeMount(telemetry=False)
        rc, err, _ = self._run(mount)
        self.assertEqual(rc, 0, err)
        self.assertFalse(self.csv.exists())
        self.assertFalse(Path(str(self.csv) + ".json").exists())
        radec = (self.dir / "PE_LOG_radec.csv").read_text().splitlines()
        self.assertEqual(radec[0], pel.RADEC_HEADER)
        self.assertGreaterEqual(len(radec) - 1, 9)
        self.assertIn("no achernar:telemetry action", err)
        self.assertNotIn("action", [m for m, _ in mount.puts])

    def test_refuses_existing_without_force(self):
        self.csv.write_text("keep me\n")
        rc, err, _ = self._run(FakeMount())
        self.assertEqual(rc, 2)
        self.assertIn("refusing", err)
        self.assertEqual(self.csv.read_text(), "keep me\n")
        (self.dir / "PE_LOG_radec.csv").write_text("x\n")
        rc, err, _ = self._run(FakeMount(), "--force")
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.csv.read_text().startswith(FROZEN))

    def test_put_sequence_is_connect_trackingrate_tracking_then_actions_only(self):
        mount = FakeMount()
        rc, err, _ = self._run(mount)
        self.assertEqual(rc, 0, err)
        members = [m for m, _ in mount.puts]
        self.assertEqual(members[:3], ["connected", "trackingrate", "tracking"])
        self.assertEqual(mount.puts[0][1]["Connected"], "true")
        self.assertEqual(mount.puts[1][1]["TrackingRate"], "0")
        self.assertEqual(mount.puts[2][1]["Tracking"], "true")
        self.assertTrue(all(m == "action" for m in members[3:]))
        self.assertTrue(all(p["Action"] == "achernar:telemetry" for m, p in mount.puts[3:]))
        self.assertTrue(all("ClientID" in p and "ClientTransactionID" in p for _, p in mount.puts))
        # optional end PUTs only when asked
        mount2 = FakeMount()
        self.csv.unlink()
        rc, err, _ = self._run(mount2, "--force", "--stop-tracking-at-end", "--disconnect-at-end")
        self.assertEqual(rc, 0, err)
        tail = [(m, p.get("Tracking", p.get("Connected"))) for m, p in mount2.puts if m != "action"][-2:]
        self.assertEqual(tail, [("tracking", "false"), ("connected", "false")])

    def test_trackingrate_not_implemented_is_tolerated(self):
        mount = FakeMount(trackingrate_ni=True)
        mount.trackingrate = 0
        rc, err, _ = self._run(mount, duration="1")
        self.assertEqual(rc, 0, err)
        self.assertIn("TrackingRate is not settable", err)

    def test_waits_for_steady_rate(self):
        mount = FakeMount(steady_after=1.5)
        rc, err, _ = self._run(mount)
        self.assertEqual(rc, 0, err)
        with self.csv.open() as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(rows)
        self.assertEqual({r["ra_rate_usteps_s"] for r in rows}, {"12.478516210"})
        self.assertEqual({r["state"] for r in rows}, {"TRACK"})
        steady_us = (mount.t_track + 1.5) * 1e6
        self.assertGreaterEqual(int(rows[0]["device_us"]), steady_us)
        # and it gives up after 10 s if the rate never settles
        never = FakeMount(steady_after=1e9)
        self.csv.unlink(missing_ok=True)
        t0 = time.monotonic()
        rc, err, _ = self._run(never, "--force")
        self.assertEqual(rc, 1)
        self.assertGreater(time.monotonic() - t0, 9.5)
        self.assertIn("steady sidereal rate", err)

    def test_aborts_when_tracking_stops(self):
        mount = FakeMount(stop_tracking_after=1.0)
        rc, err, _ = self._run(mount, duration="5")
        self.assertEqual(rc, 3, err)
        self.assertIn("ABORT", err)
        rows = self.csv.read_text().splitlines()
        self.assertGreaterEqual(len(rows), 2)

    def test_summary_matches_pe_log_health_rules(self):
        def rows(n=200, dt=1.0, rate=SIDEREAL, jitter=0.0, gap_at=None, rate_col=None):
            out = []
            t = 0.0
            for i in range(n):
                if gap_at is not None and i == gap_at:
                    t += 4.0
                pos = math.floor(rate * t + (jitter if i == n // 2 else 0.0))
                rc = rate_col(i) if rate_col else SIDEREAL
                out.append((f"{1e9 + t:.6f}", int(t * 1e6), pos, 0, f"{rc:.9f}", "0.000000000", "TRACK"))
                t += dt
            return out

        cases = {
            "good": (rows(), "CONSTANT_RATE_OK"),
            "residual": (rows(jitter=3.0), "CHECK_LOG"),
            "gap": (rows(gap_at=50), "CHECK_LOG"),
            "two_rates": (rows(rate_col=lambda i: SIDEREAL if i < 100 else SIDEREAL + 1e-9), "CHECK_LOG"),
        }
        # a 100 ppm fast drive over a long run fails the 50 ppm rule
        cases["ppm"] = (rows(n=2401, dt=2.5, rate=SIDEREAL * (1 + 100e-6)), "CHECK_LOG")
        cases["ppm_ok"] = (rows(n=2401, dt=2.5, rate=SIDEREAL * (1 + 20e-6)), "CONSTANT_RATE_OK")
        for name, (data, want) in cases.items():
            with self.subTest(case=name):
                p = self.dir / f"{name}.csv"
                write_csv(p, data)
                s = pel.summarize(p)
                self.assertEqual(s["status"], want, s)
                self.assertEqual(s["rows"], len(data))
        s = pel.summarize(self.dir / "ppm.csv")
        self.assertAlmostEqual(s["rate_error_ppm"], 100.0, delta=1.0)
        self.assertAlmostEqual(s["ppm_resolution"], 2e6 / (6000 * SIDEREAL), places=6)
        # without a sidecar the ppm rule is skipped, exactly like pe_log_health
        os.remove(str(self.dir / "ppm.csv") + ".json")
        self.assertEqual(pel.summarize(self.dir / "ppm.csv")["status"], "CONSTANT_RATE_OK")
        self.assertIsNone(pel.summarize(self.dir / "ppm.csv")["rate_error_ppm"])

    @unittest.skipUnless(SIM.exists(), "host simulator not built (make -C mount_cad_esp_32_firmware/achernar-mount-esp32/host_sim)")
    def test_against_simulator(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        proc = subprocess.Popen([str(SIM), "--alpaca-port", str(port), "--discovery-port", "0", "--lx200-port", "0",
                                 "--microsteps", "64"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            self.assertTrue(proc.stdout.readline().startswith("achernar-mount-sim "))
            rc, err = run_logger("--url", f"http://127.0.0.1:{port}", "--duration", "5", "--interval", "1.0",
                                 "--settle-s", "0", "--csv", str(self.csv))
            self.assertEqual(rc, 0, err)
        finally:
            proc.terminate()
            proc.wait(5)
            proc.stdout.close()
        with self.csv.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0]), FROZEN.split(","))
        self.assertGreaterEqual(len(rows), 5)
        self.assertEqual({r["ra_rate_usteps_s"] for r in rows}, {"12.478516210"})
        meta = json.loads(Path(str(self.csv) + ".json").read_text())
        self.assertEqual(list(meta), list(pel.SIDECAR_KEYS))
        self.assertEqual((meta["microsteps"], meta["ra_ratio"], meta["motor_full_steps"]), (64, 84, 200))
        self.assertIsInstance(meta["sidereal_rate_usteps_s"], float)
        self.assertEqual(meta["port"], f"127.0.0.1:{port}")
        radec = (self.dir / "PE_LOG_radec.csv").read_text().splitlines()
        self.assertEqual(radec[0], pel.RADEC_HEADER)
        self.assertGreaterEqual(len(radec) - 1, 5)


if __name__ == "__main__":
    unittest.main()
