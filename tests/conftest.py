import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trafficwatch.config import load_config  # noqa: E402
from trafficwatch.tracks import Track, compute_kinematics  # noqa: E402

FRAME = (1280, 720)
DT = 0.1


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def make_track(tid, sc, t, foot, size=(80, 50), cfg=None):
    """Track from foot positions (bottom-centre of the box)."""
    t = np.asarray(t, float)
    foot = np.asarray(foot, float)
    w, h = size
    box = np.column_stack([foot[:, 0] - w / 2, foot[:, 1] - h, foot[:, 0] + w / 2, foot[:, 1]])
    tr = Track(tid, sc, t, box, np.full(len(t), 0.9))
    return compute_kinematics(tr, (cfg or load_config())["tracks"])


def drive(t0, t1, x0, x1, y, dt=DT):
    """Constant-speed drive along y between times t0..t1 (inclusive)."""
    t = np.round(np.arange(t0, t1 + 1e-9, dt), 3)
    x = np.interp(t, [t0, t1], [x0, x1])
    return t, np.column_stack([x, np.full(len(t), y)])
