"""Event segments: from per-sample flags to the list the harness and evaluate.py accept.

A single malformed event makes evaluate.py reject the whole predictions file, and the
harness drops (not merges) a later overlapping segment of the same class, so every event
list leaves Part A through `sanitize`.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

MIN_EVENT_SEC = 0.05  # shorter events can collapse to [x, x] after rounding to 3 decimals

Segment = tuple[float, float]


def runs(times: Sequence[float], flags: Sequence[bool], dt: float) -> list[Segment]:
    """Contiguous True runs of samples at `times`; a run ends `dt` after its last sample."""
    out: list[Segment] = []
    start = prev = None
    for t, f in zip(times, flags):
        if f and start is None:
            start = t
        elif not f and start is not None:
            out.append((start, prev + dt))
            start = None
        prev = t
    if start is not None:
        out.append((start, prev + dt))
    return out


def merge(segments: Iterable[Segment], gap: float = 0.0) -> list[Segment]:
    """Union of segments; segments closer than `gap` seconds are joined."""
    out: list[list[float]] = []
    for s, e in sorted(segments):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def drop_short(segments: Iterable[Segment], min_dur: float) -> list[Segment]:
    return [(s, e) for s, e in segments if e - s >= min_dur]


def sanitize(events: Iterable[Sequence], classes: Sequence[str], duration: float) -> list[list]:
    """Valid, rounded, per-class non-overlapping, sorted [start, end, label] list."""
    end_cap = math.floor(duration * 1000) / 1000
    by_label: dict[str, list[Segment]] = {}
    for ev in events:
        if len(ev) != 3 or ev[2] not in classes:
            continue
        try:
            s, e = float(ev[0]), float(ev[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(s) and math.isfinite(e)):
            continue
        s, e = max(0.0, s), min(e, end_cap)
        if e - s >= MIN_EVENT_SEC:
            by_label.setdefault(ev[2], []).append((s, e))
    out = []
    for label, segs in by_label.items():
        for s, e in merge(segs):
            s, e = round(s, 3), min(round(e, 3), end_cap)
            if e > s:
                out.append([s, e, label])
    return sorted(out, key=lambda x: (x[0], x[1], x[2]))
