"""Signal heads are read the same on a downscaled copy of this camera as on the 4K original.

Needs the local proxies (DataSets/, not in git) and the dev signal cache; skipped without them.
The detector is replaced by an empty one: only decoding, registration and signals run.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from westcv import pipeline as P  # noqa: E402
from westcv.registration import Reference  # noqa: E402

STEM = "C3905"
CACHE = ROOT / f"cache/{STEM}_signals.npz"


def _pipeline_without_detector():
    pipe = P.Pipeline.__new__(P.Pipeline)
    pipe.root = ROOT
    pipe.det = lambda frames: [np.zeros((0, 6), np.float32) for _ in frames]
    pipe.batch = 4
    pipe.ref = Reference.load(ROOT / "scene/reference_sift.npz")
    pipe.geometry = json.loads((ROOT / "scene/geometry.json").read_text(encoding="utf-8"))
    pipe.heads_cfg = json.loads((ROOT / "scene/signal_heads.json").read_text(encoding="utf-8"))["heads"]
    pipe.templates = dict(np.load(ROOT / "scene/signal_templates.npz"))
    return pipe


@pytest.mark.parametrize("res", ["720p", "1080p"])
def test_signals_on_downscaled_video_match_4k(res):
    video = ROOT / f"DataSets/proxies/{STEM}_{res}.mp4"
    if not video.exists() or not CACHE.exists():
        pytest.skip("local proxies / signal cache not available")
    pipe = _pipeline_without_detector()
    got = {}
    signals = pipe._signals

    def spy(*args):
        got["sig"] = signals(*args)
        return got["sig"]

    pipe._signals = spy
    pipe(str(video), ["red_light"])
    sig = got["sig"]
    ref = np.load(CACHE)
    # the proxies decode frames 0, 3, 6, .. and the 4K original frames 2, 5, 8, ..: nearest sample
    j = np.abs(ref["t"][None, :] - sig.t[:, None]).argmin(1)
    ok = np.abs(ref["t"][j] - sig.t) < 0.1
    assert ok.mean() > 0.95
    for name, on in (("veh_median", sig.veh_on), ("ped_left", sig.ped_on)):
        agree = (on[ok] == ref[f"{name}_on"][j[ok]]).all(1).mean()
        assert agree >= 0.95, f"{name} lamp agreement {agree:.3f} at {res}"
