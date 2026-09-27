"""Part B risk on synthetic tracks: a crossing course to contact raises it, ordinary following does not."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from westcv.risk import RiskModel  # noqa: E402

FPS, STRIDE = 30.0, 6


def box(x, y, w=60, h=40, cls=2):
    return [x - w / 2, y - h, x + w / 2, y, 0.9, cls]


def run(paths, seconds=3.0):
    """paths: functions t -> (x, y) of each car; returns the risk after every processed frame."""
    m, out = RiskModel(FPS, STRIDE), []
    for k in range(int(seconds * FPS / STRIDE)):
        t = k * STRIDE / FPS
        out.append(m.update(t, np.array([box(*p(t)) for p in paths], np.float32)))
    return np.array(out)


def test_crossing_course_raises_an_alarm():
    east = lambda t: (100 + 150 * t, 400)   # noqa: E731  meets the other car at (400, 400) at t = 2 s
    south = lambda t: (400, 100 + 150 * t)  # noqa: E731
    risk = run([east, south])
    assert risk.max() >= 0.5
    assert risk[: int(0.5 * FPS / STRIDE)].max() < 0.5  # not before the course is established


def test_following_in_one_lane_is_not_a_conflict():
    lead = lambda t: (300 + 150 * t, 400)    # noqa: E731
    follow = lambda t: (180 + 150 * t, 400)  # noqa: E731  same speed, two box widths behind
    assert run([lead, follow]).max() < 0.5
