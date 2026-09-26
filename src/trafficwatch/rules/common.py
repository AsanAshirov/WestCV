"""Shared context and helpers for the rule modules."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene import SceneMaps
from ..tracks import Track


@dataclass
class RuleContext:
    tracks: list[Track]
    maps: SceneMaps
    duration: float
    dt: float                      # analysis step, s
    track_cfg: dict
    by_time: dict[float, list[tuple[Track, int]]] = field(default_factory=dict)

    def __post_init__(self):
        for tr in self.tracks:
            for i, t in enumerate(tr.t):
                self.by_time.setdefault(round(float(t), 3), []).append((tr, i))

    def others_at(self, t: float, exclude: Track) -> list[tuple[Track, int]]:
        return [(o, i) for o, i in self.by_time.get(round(float(t), 3), []) if o is not exclude]


def runs(mask: np.ndarray, t: np.ndarray, max_gap_s: float, min_dur_s: float, dt: float) -> list[tuple[float, float]]:
    """Segments where `mask` holds, gaps up to `max_gap_s` filled; each segment ends one
    analysis step after its last true sample."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    segments, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if t[i] - t[prev] > max_gap_s + 1.5 * dt:
            segments.append((start, prev))
            start = i
        prev = i
    segments.append((start, prev))
    out = []
    for a, b in segments:
        s, e = float(t[a]), float(t[b]) + dt
        if e - s >= min_dur_s:
            out.append((s, e))
    return out


def inside(point: np.ndarray, box: np.ndarray) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
