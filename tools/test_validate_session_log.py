import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_session_log import validate  # noqa: E402

TEMPLATES = Path(__file__).resolve().parent / "templates"


def good_cloud_log() -> dict:
    labels = [
        {"t": f"2026-10-08T21:{m:02d}:00-04:00", "sky_class": c}
        for m, c in ((0, 0), (10, 1), (20, 2))
    ]
    return {
        "session": {
            "night_id": "2026-10-08", "rig_id": "R4", "camera": "Canon EOS R50",
            "camera_utc_offset": "-04:00", "site_lat": 36.07, "site_lon": -79.81,
            "saturation_adu": None, "exposure_s": 20,
        },
        "environment": labels,
        "runs": [{"run_id": "coc-1", "run_type": "cloud_block", "status": "completed"}],
        "disturbances": [
            {"t": "2026-10-08T21:05:00-04:00", "kind": "headlights", "affected_run_id": "coc-1"}
        ],
    }


class TestValidate(unittest.TestCase):
    def test_complete_log_has_no_errors(self):
        errors, warnings = validate(good_cloud_log())
        self.assertEqual(errors, [])
        # Saturation is unknown until BP-2; that is a warning, not a guess.
        self.assertTrue(any("saturation_adu" in w for w in warnings))

    def test_unfilled_templates_are_rejected(self):
        for path in TEMPLATES.glob("*.json"):
            errors, _ = validate(json.loads(path.read_text()))
            self.assertTrue(errors, f"{path.name} validated while still a template")

    def test_naive_label_time_is_an_error(self):
        log = good_cloud_log()
        log["environment"][0]["t"] = "2026-10-08T21:00:00"
        errors, _ = validate(log)
        self.assertTrue(any("UTC offset" in e for e in errors))

    def test_cloud_block_needs_labels(self):
        log = good_cloud_log()
        for e in log["environment"]:
            e["sky_class"] = None
        errors, _ = validate(log)
        self.assertTrue(any("no sky_class labels" in e for e in errors))

    def test_disturbance_must_name_a_real_run(self):
        log = good_cloud_log()
        log["disturbances"][0]["affected_run_id"] = "nope"
        errors, _ = validate(log)
        self.assertTrue(any("matches no run" in e for e in errors))

    def test_bad_offset_and_sky_class(self):
        log = good_cloud_log()
        log["session"]["camera_utc_offset"] = "EDT"
        log["environment"][1]["sky_class"] = 7
        errors, _ = validate(log)
        self.assertEqual(len(errors), 2)


if __name__ == "__main__":
    unittest.main()
