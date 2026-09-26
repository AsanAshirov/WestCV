"""Part A: video -> perception -> tracks -> scene maps -> rules -> events.

Models are loaded by `warmup()` when solution.py is imported, before the harness
starts its per-video timer. Part A gets a time budget computed from a quick
measurement of how long the harness's own frame loop (Part B) will take:
    T_A = (3 - part_b_factor * h - margin) * duration,  h = harness cost / duration.
When the budget runs out, the pass stops and events are built from what was seen.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import env
from .config import load_config, resolve
from .detector import Detector
from .perception import Perception, perceive, track
from .postprocess import finalize
from .rules import RULES
from .rules.common import RuleContext
from .scene import SceneMaps, build_maps, load_scene
from .tracks import Track, build_tracks
from .video import VideoMeta, read_meta

CFG = load_config()
env.seed_everything(CFG["seed"])
SCENE = load_scene(resolve(CFG["scene"]["file"]))
RUN_CLOCK: dict[str, float] = {}  # video_id -> start of detect_events; Part B reads only this time
PART_B_OFF: set[str] = set()      # videos where the budget is too tight for Part B (timing only, no content)
_DETECTOR: Detector | None = None


def log(msg: str) -> None:
    print(f"[trafficwatch] {msg}", file=sys.stderr, flush=True)


def get_detector() -> Detector:
    global _DETECTOR
    if _DETECTOR is None:
        d = CFG["detector"]
        _DETECTOR = Detector(resolve(d["weights"]), d["imgsz"], d["conf"], d["iou"], d["half"])
    return _DETECTOR


def warmup() -> None:
    get_detector().warmup(720, 1280)


@dataclass
class Analysis:
    perception: Perception
    tracks: list[Track]
    maps: SceneMaps
    intervals: dict[str, list[tuple[float, float]]]
    events: list[list]


def analyze(perception: Perception, cfg: dict = CFG) -> Analysis:
    """Everything after detection. Cheap: used on cached perceptions while tuning."""
    obs = track(perception, cfg["tracker"])
    tracks = build_tracks(obs, cfg["tracks"])
    maps = build_maps(tracks, (perception.frame_w, perception.frame_h), SCENE, cfg["scene"])
    ctx = RuleContext(tracks, maps, perception.duration, perception.dt, cfg["tracks"])
    intervals = {label: RULES[label](ctx, c) for label, c in cfg["classes"].items() if c["enabled"]}
    events = finalize(intervals, perception.duration, cfg["classes"])
    return Analysis(perception, tracks, maps, intervals, events)


def harness_cost(meta: VideoMeta, n_probe: int) -> float:
    """Seconds the harness needs to cv2.read() the whole video, as a fraction of its duration."""
    cap = cv2.VideoCapture(meta.path)
    t0, n = time.perf_counter(), 0
    while n < n_probe and cap.read()[0]:
        n += 1
    cap.release()
    per_frame = (time.perf_counter() - t0) / max(n, 1)
    return per_frame * meta.fps


def part_a_deadline(meta: VideoMeta, t0: float) -> float:
    """If the harness loop alone leaves Part A less than `part_b_off_below` x duration,
    Part B is switched off for this video: its step() then costs nothing and Part A
    gets that time (Part B is worth at most 0.18 of the score, an empty video far more)."""
    b = CFG["budget"]
    h = harness_cost(meta, b["probe_frames"]) if meta.n_frames > 3 * b["probe_frames"] else b["default_h"]
    factor = 3.0 - b["part_b_factor"] * h - b["margin"]
    if factor < b["part_b_off_below"]:
        PART_B_OFF.add(meta.video_id)
        factor = 3.0 - b["harness_only_factor"] * h - b["margin"]
    factor = float(np.clip(factor, b["min_factor"], b["max_factor"]))
    log(f"{meta.video_id}: harness cost h={h:.2f}x, Part A budget {factor:.2f}x = {factor * meta.duration:.0f}s"
        + (" (Part B off)" if meta.video_id in PART_B_OFF else ""))
    return t0 + factor * meta.duration


def detect_events(video_path: str) -> list[list]:
    t0 = time.perf_counter()
    meta = read_meta(video_path)
    RUN_CLOCK[meta.video_id] = t0
    perception = perceive(meta, get_detector(), CFG, deadline=part_a_deadline(meta, t0))
    if not perception.complete:
        log(f"{meta.video_id}: time budget reached at {perception.times[-1]:.0f}s of {meta.duration:.0f}s")
    result = analyze(perception)
    log(f"{meta.video_id}: {len(result.tracks)} tracks, {len(result.events)} events, "
        f"{time.perf_counter() - t0:.1f}s")
    return result.events
