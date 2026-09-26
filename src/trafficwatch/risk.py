"""Part B: P(accident starts within 5 s) from the frames received so far.

Causal by construction: own detector + tracker on the frames passed to step(),
nothing from Part A (only the start time of the video run, to respect the budget).
Signal: time-to-collision between pairs of road users, extrapolated at constant
velocity in the image. Two channels, as the metric rewards:
  - ranking channel in [0, 0.4999]: grows as a conflict becomes imminent (AP);
  - alarm channel >= 0.5: only for a consistent conflict with TTC below
    `alarm_ttc_s`, at most `max_alarm_s` long, the pair muted afterwards (F1_alarm).
"""
from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from . import pipeline
from .config import resolve
from .detector import PERSON, TWO_WHEELER, VEHICLE, Detector
from .tracker import ByteTracker

RCFG = pipeline.CFG["risk"]
_DETECTOR: Detector | None = None


def warmup() -> None:
    global _DETECTOR
    if not RCFG["enabled"]:
        return
    import torch

    if RCFG["require_cuda"] and not torch.cuda.is_available():
        pipeline.log("Part B disabled: no CUDA device (it would not fit the time budget on CPU)")
        return
    _DETECTOR = Detector(resolve(RCFG["weights"]), RCFG["imgsz"], RCFG["conf"], half=True)
    _DETECTOR.warmup(360, RCFG["width"])


def _strip(box: np.ndarray, fraction: float) -> np.ndarray:
    """Lower part of the box: a proxy of the object's footprint on the road."""
    return np.array([box[0], box[3] - fraction * (box[3] - box[1]), box[2], box[3]])


def time_to_collision(a: np.ndarray, va: np.ndarray, b: np.ndarray, vb: np.ndarray,
                      horizon: float, step: float, fraction: float) -> float | None:
    """First time (s) the two footprints overlap under constant velocity; None if they
    already overlap or never meet within the horizon."""
    sa, sb = _strip(a, fraction), _strip(b, fraction)
    taus = np.arange(0.0, horizon + 1e-9, step)
    da = np.outer(taus, va)
    db = np.outer(taus, vb)
    x1 = np.maximum(sa[0] + da[:, 0], sb[0] + db[:, 0])
    x2 = np.minimum(sa[2] + da[:, 0], sb[2] + db[:, 0])
    y1 = np.maximum(sa[1] + da[:, 1], sb[1] + db[:, 1])
    y2 = np.minimum(sa[3] + da[:, 1], sb[3] + db[:, 1])
    hit = (x2 > x1) & (y2 > y1)
    if hit[0] or not hit.any():
        return None
    return float(taus[np.argmax(hit)])


class RiskEstimator:
    def reset(self, meta: dict) -> None:
        self.fps = float(meta.get("fps") or 25.0)
        self.n_frames = int(meta.get("n_frames") or 0)
        self.stride = max(1, int(round(self.fps / RCFG["fps"])))
        width = min(RCFG["width"], int(meta.get("width") or RCFG["width"]))
        height = int(round((meta.get("height") or 1) * width / max(meta.get("width") or 1, 1) / 2)) * 2
        self.size = (width, max(height, 2))
        self.buf = np.empty((self.size[1], self.size[0], 3), np.uint8)
        self.tracker = ByteTracker(**pipeline.CFG["tracker"])
        self.first_seen: dict[int, float] = {}
        self.history: dict[tuple[int, int], deque] = {}
        self.muted: dict[tuple[int, int], float] = {}
        self.alarm_pair: tuple[int, int] | None = None
        self.alarm_start = 0.0
        self.k, self.last = -1, 0.0
        self.enabled = _DETECTOR is not None and meta.get("video_id") not in pipeline.PART_B_OFF
        started = pipeline.RUN_CLOCK.get(meta.get("video_id"), time.perf_counter())
        self.deadline = started + 3.0 * self.n_frames / self.fps - RCFG["deadline_margin_s"]
        self.t_first: float | None = None
        self.error_logged = False

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self.k += 1
        now = time.perf_counter()
        if self.t_first is None:
            self.t_first = now
        elif self.enabled and RCFG["budget_guard"] and self.k % 32 == 0:
            rate = (now - self.t_first) / self.k  # includes the harness's own decoding
            if now + rate * (self.n_frames - self.k) > self.deadline:
                self.enabled = False
                pipeline.log(f"Part B: switched off at t={t_sec:.1f}s to stay inside the time budget")
        if not self.enabled:
            self.last = min(self.last, 0.4999) * RCFG["decay"]
            return float(self.last)
        if self.k % self.stride:
            return float(self.last)
        try:
            cv2.resize(frame, self.size, dst=self.buf, interpolation=cv2.INTER_AREA)
            obs = self.tracker.update(float(t_sec), _DETECTOR([self.buf])[0])
            self.last = self._score(obs, float(t_sec))
        except Exception as exc:
            if not self.error_logged:
                pipeline.log(f"Part B step failed: {exc!r}")
                self.error_logged = True
            self.last = 0.0 if self.last >= 0.5 else 0.5 * self.last
        return float(min(1.0, max(0.0, self.last)))

    def _score(self, obs: np.ndarray, t: float) -> float:
        c = RCFG
        for tid in obs[:, 0].astype(int):
            self.first_seen.setdefault(tid, t)
        rows = [r for r in obs if int(r[6]) in (VEHICLE, TWO_WHEELER, PERSON)
                and t - self.first_seen[int(r[0])] >= c["min_age_s"]]
        pair_risk: dict[tuple[int, int], tuple[float, float]] = {}
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if VEHICLE not in (int(a[6]), int(b[6])):
                    continue
                key = (int(min(a[0], b[0])), int(max(a[0], b[0])))
                risk, ttc = self._pair(a, b, key, t)
                if risk > 0:
                    pair_risk[key] = (risk, ttc)
        top = sorted(pair_risk.values(), reverse=True)[:3]
        r = 1.0 - float(np.prod([1.0 - p for p, _ in top])) if top else 0.0

        if self.alarm_pair is not None:
            p = pair_risk.get(self.alarm_pair, (0.0, np.inf))[0]
            if t - self.alarm_start > c["max_alarm_s"] or p < c["release_r"]:
                self.muted[self.alarm_pair] = t + c["mute_s"]
                self.alarm_pair = None
        if self.alarm_pair is None:
            for key, (p, ttc) in sorted(pair_risk.items(), key=lambda kv: -kv[1][0]):
                if p >= c["alarm_min_r"] and ttc <= c["alarm_ttc_s"] and self.muted.get(key, -1) < t:
                    self.alarm_pair, self.alarm_start = key, t
                    break
        return 0.5 + 0.4999 * r if self.alarm_pair is not None else min(0.4999 * r, 0.4999)

    def _pair(self, a: np.ndarray, b: np.ndarray, key: tuple[int, int], t: float) -> tuple[float, float]:
        c = RCFG
        box_a, box_b = a[1:5], b[1:5]
        size = max(np.sqrt((box_a[2] - box_a[0]) * (box_a[3] - box_a[1])),
                   np.sqrt((box_b[2] - box_b[0]) * (box_b[3] - box_b[1])), 1.0)
        foot_a = np.array([(box_a[0] + box_a[2]) / 2, box_a[3]])
        foot_b = np.array([(box_b[0] + box_b[2]) / 2, box_b[3]])
        gap = foot_b - foot_a
        if np.linalg.norm(gap) > c["max_pair_dist_bs"] * size:
            return 0.0, np.inf
        va, vb = self.tracker.velocity(int(a[0])), self.tracker.velocity(int(b[0]))
        if va is None or vb is None:
            return 0.0, np.inf
        closing = float(-(vb - va) @ gap / max(np.linalg.norm(gap), 1e-6)) / size  # BS/s
        ttc = time_to_collision(box_a, va, box_b, vb, c["horizon_s"], 0.1, c["footprint_fraction"])
        hist = self.history.setdefault(key, deque(maxlen=4))
        hist.append(ttc)
        if ttc is None or closing < c["min_closing_bs"]:
            return 0.0, np.inf
        recent = list(hist)[-3:]
        consistent = (len(recent) == 3 and all(x is not None for x in recent)
                      and recent[2] <= recent[1] + 0.05 and recent[1] <= recent[0] + 0.05)  # TTC keeps falling
        risk = float(np.clip((c["ttc_hi"] - ttc) / (c["ttc_hi"] - c["ttc_lo"]), 0.0, 1.0))
        risk *= 1.0 if consistent else c["inconsistent_weight"]
        if self.muted.get(key, -1) >= t:
            risk *= 0.1
        return risk, ttc
