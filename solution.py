"""
solution.py — team Antigradient, WIUT Hackathon 2026, Computer Vision track.

The organizers' harness (run_submission.py) imports this module and calls:

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

All logic lives in src/trafficwatch (pipeline.py for Part A, risk.py for Part B).
Models are loaded and warmed up here, at import time, before the harness starts
its per-video timer. See README.md and docs/RUNBOOK_RU.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trafficwatch import env  # noqa: E402,F401  (offline mode before torch/ultralytics)

# isort: split
from trafficwatch import pipeline, risk  # noqa: E402

# Official class ids (14). Classes we never predict stay in the list: that is allowed
# and costs nothing; only predicted or ground-truth classes enter Score A.
CLASSES: list[str] = [
    "accident",            # collision between road users / with a fixed object
    "near_miss",           # sharp braking or swerving to avoid a collision, no contact
    "red_light",           # crossing the stop line on red
    "wrong_way",           # driving against the traffic direction / in the oncoming lane
    "illegal_u_turn",      # U-turn where prohibited
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "illegal_turn",        # turn from the wrong lane or in a prohibited direction
    "solid_line_crossing", # lane change / manoeuvre across a solid marking
    "stop_line",           # stopped past the stop line on red
    "congestion",          # standstill / crawling traffic across all lanes of a direction
    "road_obstacle",       # debris, animal or fallen object on the carriageway
    "fire_smoke",          # visible fire or smoke from a vehicle or on the road
]

try:
    pipeline.warmup()
    risk.warmup()
except Exception as exc:  # e.g. missing weights: log it, never crash the harness at import
    pipeline.log(f"warmup failed: {exc!r}")


def detect_events(video_path: str) -> list[list]:
    """Part A: [[start_sec, end_sec, label], ...] for one .mp4."""
    return pipeline.detect_events(video_path)


RiskEstimator = risk.RiskEstimator
