"""ByteTrack-style multi-object tracker for a fixed camera.

Two-stage association (confident detections first, then low-confidence ones for
tracks that are still alive), constant-velocity prediction, lifetimes in seconds.
Written here instead of taken from a library so that buffers are in seconds and
IoU is never multiplied by the score (the Ultralytics `fuse_score` pitfall that
drops fast cars at 10 Hz), and so that Part A and Part B share one implementation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import box_iou


class ByteTracker:
    def __init__(self, high_thresh: float = 0.4, low_thresh: float = 0.15, new_thresh: float = 0.5,
                 match_iou: float = 0.2, low_match_iou: float = 0.4, lost_sec: float = 3.0,
                 vel_alpha: float = 0.5):
        self.high, self.low, self.new = high_thresh, low_thresh, new_thresh
        self.match_iou, self.low_match_iou = match_iou, low_match_iou
        self.lost_sec, self.vel_alpha = lost_sec, vel_alpha
        self.next_id = 1
        self.ids = np.zeros(0, np.int64)
        self.boxes = np.zeros((0, 4), np.float32)
        self.vel = np.zeros((0, 2), np.float32)  # box-center velocity, px/s
        self.sc = np.zeros(0, np.int64)
        self.last_t = np.zeros(0, np.float64)
        self.hits = np.zeros(0, np.int64)

    def _predict(self, t: float) -> np.ndarray:
        dt = np.clip(t - self.last_t, 0.0, self.lost_sec)[:, None]
        shift = self.vel * dt
        return self.boxes + np.hstack([shift, shift]).astype(np.float32)

    def _assign(self, pred: np.ndarray, tracks: np.ndarray, dets: np.ndarray, min_iou: float):
        """Hungarian matching of track indices to detection rows; different superclasses never match."""
        if len(tracks) == 0 or len(dets) == 0:
            return []
        iou = box_iou(pred[tracks], dets[:, :4])
        iou[self.sc[tracks][:, None] != dets[None, :, 5].astype(np.int64)] = 0.0
        rows, cols = linear_sum_assignment(-iou)
        return [(tracks[r], c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]

    def update(self, t: float, dets: np.ndarray) -> np.ndarray:
        """dets: (N, 6) x1, y1, x2, y2, conf, sc. Returns (K, 7): id, x1, y1, x2, y2, conf, sc
        for tracks observed at time t."""
        pred = self._predict(t)
        alive = np.arange(len(self.ids))
        high = dets[dets[:, 4] >= self.high]
        low = dets[(dets[:, 4] >= self.low) & (dets[:, 4] < self.high)]

        matches = self._assign(pred, alive, high, self.match_iou)
        used_tracks = {ti for ti, _ in matches}
        used_high = {di for _, di in matches}
        recent = np.array([ti for ti in alive if ti not in used_tracks and t - self.last_t[ti] <= 0.5],
                          dtype=np.int64)
        low_matches = self._assign(pred, recent, low, self.low_match_iou)

        out = []
        for ti, det in [(ti, high[di]) for ti, di in matches] + [(ti, low[di]) for ti, di in low_matches]:
            dt = t - self.last_t[ti]
            if dt > 0:
                raw = (_center(det[:4]) - _center(self.boxes[ti])) / dt
                a = self.vel_alpha if self.hits[ti] > 1 else 1.0
                self.vel[ti] = (1 - a) * self.vel[ti] + a * raw
            self.boxes[ti] = det[:4]
            self.last_t[ti] = t
            self.hits[ti] += 1
            out.append([self.ids[ti], *det[:5], self.sc[ti]])

        fresh = [d for i, d in enumerate(high) if i not in used_high and d[4] >= self.new]
        for det in fresh:
            self.ids = np.append(self.ids, self.next_id)
            self.boxes = np.vstack([self.boxes, det[None, :4]]).astype(np.float32)
            self.vel = np.vstack([self.vel, np.zeros((1, 2), np.float32)])
            self.sc = np.append(self.sc, int(det[5]))
            self.last_t = np.append(self.last_t, t)
            self.hits = np.append(self.hits, 1)
            out.append([self.next_id, *det[:5], det[5]])
            self.next_id += 1

        keep = (t - self.last_t) <= self.lost_sec
        self.ids, self.boxes, self.vel = self.ids[keep], self.boxes[keep], self.vel[keep]
        self.sc, self.last_t, self.hits = self.sc[keep], self.last_t[keep], self.hits[keep]
        return np.array(out, np.float64).reshape(-1, 7)

    def velocity(self, track_id: int) -> np.ndarray | None:
        idx = np.where(self.ids == track_id)[0]
        return self.vel[idx[0]].copy() if len(idx) else None


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.float32)
