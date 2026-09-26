"""Part B on a scripted head-on conflict: the alarm must start before contact, not after."""
import numpy as np

from trafficwatch import risk
from trafficwatch.detector import VEHICLE


class ScriptedDetector:
    """Two cars driving at each other; boxes touch at t = 2.375 s."""

    def __init__(self):
        self.t = 0.0

    def __call__(self, frames):
        xa = 100 + 80 * self.t
        xb = 480 - 80 * self.t
        return [np.array([[xa, 180, xa + 60, 220, 0.9, VEHICLE],
                          [xb, 180, xb + 60, 220, 0.9, VEHICLE]], np.float32)]


def test_alarm_starts_before_contact(monkeypatch):
    det = ScriptedDetector()
    monkeypatch.setattr(risk, "_DETECTOR", det)
    est = risk.RiskEstimator()
    fps = 30.0
    est.reset({"video_id": "x.mp4", "fps": fps, "width": 640, "height": 360, "n_frames": 150})
    frame = np.zeros((360, 640, 3), np.uint8)
    scores = []
    for k in range(150):
        det.t = k / fps
        scores.append(est.step(frame, k / fps))
    scores = np.array(scores)
    times = np.arange(150) / fps
    alarm = times[scores >= 0.5]
    assert len(alarm) > 0
    assert 0.8 <= alarm[0] < 2.375                 # before contact
    assert scores[times > 3.5].max() < 0.5         # reset after contact
    assert scores[times < 0.8].max() < 0.5         # no alarm while tracks are young
    assert all(isinstance(s, float) for s in scores)
