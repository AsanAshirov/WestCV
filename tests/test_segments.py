import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from westcv.segments import drop_short, merge, runs, sanitize  # noqa: E402

CLASSES = ["stopped_vehicle", "jaywalking"]


def test_runs_end_one_sample_after_last_true():
    t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    got = runs(t, [0, 1, 1, 0, 1, 1], dt=0.1)
    assert [pytest.approx(s) for s in got] == [(0.1, 0.3), (0.4, 0.6)]


def test_merge_gap_and_overlap():
    assert merge([(5, 6), (0, 2), (1, 3), (3.5, 4)], gap=0.5) == [(0, 4), (5, 6)]
    assert merge([(0, 1), (1.2, 2)]) == [(0, 1), (1.2, 2)]


def test_drop_short():
    assert drop_short([(0, 0.4), (1, 2)], 0.5) == [(1, 2)]


def test_sanitize_clamps_rounds_merges_and_filters():
    events = [
        [-1.0, 2.0, "jaywalking"],           # start clamped to 0
        [1.5, 3.0, "jaywalking"],            # overlaps the first -> merged
        [10.0, 10.0004, "jaywalking"],       # collapses after rounding -> dropped
        [5.0, float("nan"), "jaywalking"],   # not finite -> dropped
        [5.0, 6.0, "fire_smoke"],            # not in CLASSES -> dropped
        [99.0, 150.0, "stopped_vehicle"],    # clipped to floor(duration, 3)
        ["a", 2.0, "stopped_vehicle"],       # not a number -> dropped
    ]
    assert sanitize(events, CLASSES, duration=127.6275) == [
        [0.0, 3.0, "jaywalking"],
        [99.0, 127.627, "stopped_vehicle"],
    ]


def test_sanitized_output_passes_official_validation(tmp_path):
    events = sanitize([[0.0001, 0.0499, "jaywalking"], [3.3333333, 5.55555, "jaywalking"],
                       [120.0, 1e9, "stopped_vehicle"]], CLASSES, duration=127.6275)
    pred = {"team": "t", "videos": {"C3905.MP4": {"events": events}}}
    gt = {"C3905.MP4": {"duration": 127.6275, "fps": 29.97, "events": []}}
    (tmp_path / "p.json").write_text(json.dumps(pred))
    (tmp_path / "g.json").write_text(json.dumps(gt))
    r = subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--pred", str(tmp_path / "p.json"),
                        "--gt", str(tmp_path / "g.json")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
