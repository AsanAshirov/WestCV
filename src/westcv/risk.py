"""Part B: causal accident risk from road-user tracks.

Only frames already seen are used: every `stride`-th frame the detector runs on a subsampled view,
an online tracker (ByteTrack) updates the road users, and pairwise kinematics in the image plane
(units: box sizes and seconds, so the camera scale does not matter) give
  ttc    the shortest time to contact over pairs on a collision course that close faster than
         CLOSING_MIN; pairs that follow each other in one lane only count when they close fast (rear-end).
         Pairs whose boxes already overlap are occlusions in this oblique view, not conflicts
  decel  the strongest recent deceleration of a vehicle that was moving fast, next to another user
The risk is a logistic of the shortest time to contact (0.5 at `TTC_ALARM` s) and of hard braking,
held with a short decay so that one conflict gives one alarm.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from .tracking import _Dets, tracker_args

ROAD_USERS = (0, 1, 2, 3, 5, 7)   # person, bicycle, car, motorcycle, bus, truck (COCO)
HIST_SEC = 2.0
VEL_SEC = 0.6
MOVING = 0.3                      # sizes / s
CONTACT = 0.6                     # contact when centres are closer than this many mean box sizes
FOLLOW_COS = 0.9                  # |cos| between headings above which two moving users share a lane
REAR_END_CLOSING = 1.5            # sizes / s: a follower only counts when closing faster than this
CLOSING_MIN = 1.0                 # sizes / s: slower approaches (creeping in queues, passing) are not conflicts
TTC_ALARM = 0.8                   # s: risk 0.5 (the alarm threshold) at this time to contact
DECEL_ALARM = 4.0                 # sizes / s^2: risk 0.5 at this braking next to another user
HOLD = 0.8                        # per processed frame: risk decays to this fraction of its last value


class RiskTracker:
    def __init__(self, fps: float, stride: int):
        self.dt = stride / fps
        args = tracker_args(fps / stride, buffer_sec=1.5)
        args.match_thresh = 0.95  # IoU >= 0.05: at 5 updates/s a fast car moves most of its box length
        self.tracker = BYTETracker(args)
        self.hist: dict[int, deque] = {}

    def update(self, t: float, dets: np.ndarray) -> dict:
        """dets: (N, 6) x1 y1 x2 y2 conf cls, in any fixed pixel scale."""
        d = dets[np.isin(dets[:, 5], ROAD_USERS)] if len(dets) else np.zeros((0, 6), np.float32)
        res = self.tracker.update(_Dets(d))
        alive = set()
        for x1, y1, x2, y2, tid, conf, cls, _ in (res if len(res) else []):
            tid = int(tid)
            alive.add(tid)
            h = self.hist.setdefault(tid, deque(maxlen=int(HIST_SEC / self.dt) + 2))
            w, hh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            h.append((t, (x1 + x2) / 2, y2 - 0.25 * hh, np.sqrt(w * hh), int(cls), w, hh))
        for tid in [k for k in self.hist if k not in alive and self.hist[k][-1][0] < t - 1.0]:
            del self.hist[tid]
        return self._features(t)

    @staticmethod
    def _state(h):
        """Position, velocity (px/s), box (w, h), speed (sizes/s), speed about VEL_SEC earlier, class."""
        t1, x1, y1, s1, c, w1, h1 = h[-1]
        old = [e for e in h if e[0] <= t1 - VEL_SEC]
        if not old:
            return None
        t0, x0, y0, s0 = old[-1][:4]
        v = np.array([x1 - x0, y1 - y0]) / (t1 - t0)
        prev = [e for e in h if e[0] <= t0 - VEL_SEC]
        v_prev = None
        if prev:
            tp, xp, yp = prev[-1][:3]
            v_prev = np.linalg.norm([x0 - xp, y0 - yp]) / (t0 - tp) / s0
        return np.array([x1, y1]), v, np.array([w1, h1]), np.linalg.norm(v) / s1, v_prev, c

    def _features(self, t):
        states = [s for s in (self._state(h) for h in self.hist.values() if h[-1][0] >= t - 1e-6) if s is not None]
        ttc, decel = 9.0, 0.0
        for i in range(len(states)):
            pi, vi, si, spi, vpi, ci = states[i]
            for j in range(i + 1, len(states)):
                pj, vj, sj, spj, vpj, cj = states[j]
                if ci == 0 and cj == 0:
                    continue  # two pedestrians
                if max(spi, spj) < MOVING:
                    continue  # both standing: queues and parked cars are not conflicts
                # per-axis box units: in this oblique view neighbouring lanes are close in y, and a car
                # passing a parked one must not look like a course to contact
                scale = 0.5 * (si + sj)
                dp, dv = (pj - pi) / scale, (vj - vi) / scale
                dist = np.linalg.norm(dp)
                if dist < 3.0:  # hard braking next to another road user
                    for sp, vp, c in ((spi, vpi, ci), (spj, vpj, cj)):
                        if c != 0 and vp is not None and vp > 1.0:
                            decel = max(decel, (vp - sp) / (2 * VEL_SEC))
                closing = -(dp @ dv) / (dist + 1e-6)  # box units per second
                if closing < CLOSING_MIN:
                    continue
                if min(spi, spj) >= MOVING:
                    cos = vi @ vj / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-9)
                    if cos > FOLLOW_COS and closing < REAR_END_CLOSING:
                        continue  # same lane, same direction: ordinary following
                a = dv @ dv
                b = 2 * (dp @ dv)
                c = dp @ dp - CONTACT ** 2
                disc = b * b - 4 * a * c
                if c <= 0 or disc < 0 or a <= 1e-9:
                    continue  # boxes already overlapping in the image: occlusion, not a course to contact
                tt = (-b - np.sqrt(disc)) / (2 * a)
                if tt >= 0:
                    ttc = min(ttc, tt)
        return {"ttc": ttc, "decel": decel, "n": len(states)}


def risk_from_features(f: dict) -> float:
    s = 1.0 / (1.0 + np.exp(4.0 * (f["ttc"] - TTC_ALARM)))
    return float(max(s, 1.0 / (1.0 + np.exp(-2.0 * (f["decel"] - DECEL_ALARM)))))


class RiskModel:
    """Per-video causal risk: feed (t, detections) of every `stride`-th frame, read the score."""

    def __init__(self, fps: float, stride: int):
        self.tracker = RiskTracker(fps, stride)
        self.score = 0.0

    def update(self, t: float, dets: np.ndarray) -> float:
        self.score = max(risk_from_features(self.tracker.update(t, dets)), HOLD * self.score)
        return self.score


class RiskRunner:
    """Part B inside the harness: step(frame, t) for every frame, detection on every `stride`-th.

    Detection runs on the frame subsampled to about WIDTH px (4K[::6, ::6] = 640 x 360), a few
    milliseconds on a T4, so Part B adds ~0.1x of the duration to the harness's own decoding and the
    curve does not depend on timing. Only as an emergency brake: if the projected finish (measured
    decode and detection costs) passes `deadline`, detection stops for the rest of the video and the
    risk fades, because an overrun would zero both parts of the video.
    """

    WIDTH = 640
    WARMUP = 5          # detections before the cost estimate is trusted (the first ones are slow)

    def __init__(self, detector, meta: dict, deadline: float, stride: int = 6):
        self.det, self.stride, self.deadline = detector, stride, deadline
        self.fps, self.n = float(meta["fps"]), int(meta["n_frames"])
        self.sub = max(1, round(meta["width"] / self.WIDTH))  # subsampling, no resize cost
        self.model = RiskModel(self.fps, stride)
        self.last_call = None
        self.t_read = None        # seconds per harness decode (moving average)
        self.t_proc: deque = deque(maxlen=25)
        self.stopped = False

    def _overrun(self, now: float, idx: int) -> bool:
        if len(self.t_proc) < self.WARMUP or self.t_read is None:
            return False
        left = self.n - idx
        return now + left * self.t_read + left / self.stride * float(np.median(self.t_proc)) > self.deadline

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        import time
        now = time.perf_counter()
        if self.last_call is not None:  # the time since the last step returned: the harness decoding
            gap = now - self.last_call
            self.t_read = gap if self.t_read is None else 0.9 * self.t_read + 0.1 * gap
        idx = int(round(t_sec * self.fps))
        if idx % self.stride == 0:
            self.stopped = self.stopped or self._overrun(now, idx)
            if self.stopped:
                self.model.score *= HOLD
            else:
                img = np.ascontiguousarray(frame[::self.sub, ::self.sub])
                self.model.update(t_sec, self.det([img], imgsz=self.WIDTH)[0])
                self.t_proc.append(time.perf_counter() - now)
        self.last_call = time.perf_counter()
        return self.model.score
