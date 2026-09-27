"""
solution.py — WestCV submission (WIUT Hackathon 2026, CV track).

detect_events(video_path) -> [[start_sec, end_sec, label], ...]    Part A
RiskEstimator().reset(meta); .step(frame, t_sec) -> float           Part B

Part A: YOLO26m (1280 px) detections on every 3rd frame -> ByteTrack -> scene registration against the
reference view of this intersection -> signal timeline read from the heads facing the camera ->
per-class rules on tracks, geometry and signals (src/westcv/). Models are loaded and warmed up
here, at import time, which the harness does not count against a video's time budget.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")

import numpy as np  # noqa: E402
import torch  # noqa: E402

np.random.seed(0)  # nothing samples at random; fixed anyway
torch.manual_seed(0)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from westcv.pipeline import Pipeline  # noqa: E402
from westcv.risk import RiskRunner  # noqa: E402

# Official class ids. Classes we never predict are removed (allowed by the task FAQ):
# a class that is predicted but absent from the test set would add a zero to the mean.
CLASSES: list[str] = [
    "red_light",           # crossing stop line 4 on red (signal head facing the camera)
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "stop_line",           # stopped past stop line 4 on red
    "congestion",          # many vehicles standing still on the carriageway at once
    "solid_line_crossing", # driving over the solid divider of approach 4
]

RISK_HORIZON_SEC = 5.0
# the harness budget for Part A + Part B, x video duration; WESTCV_TIME_FACTOR mirrors run_submission.py
# --time-factor for development runs on slow machines (the brakes below scale with it)
TIME_FACTOR = float(os.environ.get("WESTCV_TIME_FACTOR", "3.0"))
TIME_SAFETY = 0.25     # x duration kept free for the harness's own decoding speed varying

_PIPELINE = Pipeline(ROOT, "weights/yolo26m.pt")
_PIPELINE.det.warmup((360, RiskRunner.WIDTH, 3), imgsz=RiskRunner.WIDTH)  # Part B's input size
_STARTED: dict[str, float] = {}  # video file name -> when detect_events began (the harness timer)


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection."""
    _STARTED[Path(video_path).name] = time.perf_counter()
    events = _PIPELINE(video_path, CLASSES)
    print(f"[westcv] {Path(video_path).name} timing {_PIPELINE.last_timing}", file=sys.stderr, flush=True)
    return events


class RiskEstimator:
    """Part B — causal accident anticipation (src/westcv/risk.py).

    Every 6th frame the Part A detector (already loaded) runs on the received frame at 640 px, an online
    tracker follows the road users, and the risk rises when two of them are on a course to contact
    within about a second (time to collision in box units, closing speed, ordinary lane following and
    image-plane occlusions excluded) or a vehicle brakes hard next to another road user: the cues
    the task names. It only uses frames it has received. There is no accident in our samples, so the
    alarm threshold is set from the false-alarm rate on them; its hit rate cannot be measured here.
    Only if the video would otherwise overrun its time budget does detection stop (the risk then fades)."""

    def reset(self, meta: dict) -> None:
        duration = meta["n_frames"] / meta["fps"]
        started = _STARTED.get(meta["video_id"], time.perf_counter() - 1.5 * duration)
        deadline = started + (TIME_FACTOR - TIME_SAFETY) * duration
        self.runner = RiskRunner(_PIPELINE.det, meta, deadline)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self.runner.step(frame, t_sec)
