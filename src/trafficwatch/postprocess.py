"""Per-class intervals -> final event list in the exact format the harness accepts."""
from __future__ import annotations

import math


def merge_intervals(intervals: list[tuple[float, float]], gap_s: float) -> list[tuple[float, float]]:
    """Union of overlapping intervals; also joins intervals separated by <= gap_s."""
    merged: list[list[float]] = []
    for s, e in sorted(intervals):
        if merged and s - merged[-1][1] <= gap_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def finalize(intervals: dict[str, list[tuple[float, float]]], duration: float, classes_cfg: dict) -> list[list]:
    events = []
    for label, items in intervals.items():
        c = classes_cfg[label]
        for s, e in merge_intervals(items, c["merge_gap_s"]):
            s = max(0.0, s - c.get("pad_start_s", 0.0))
            e = min(duration, e + c.get("pad_end_s", 0.0))
            if c.get("max_duration_s"):
                e = min(e, s + c["max_duration_s"])
            if e - s >= c["min_duration_s"]:
                events.append([s, e, label])
    return sanitize(events, duration)


def sanitize(events: list[list], duration: float, min_len: float = 0.05) -> list[list]:
    """Python floats, 3 decimals, 0 <= start < end <= floor(duration, 3), length >= min_len,
    no same-class overlap. The harness would silently drop anything else, and an
    event shorter than 1 ms after rounding makes evaluate.py reject the whole file."""
    cap = math.floor(duration * 1000) / 1000
    clean = []
    for s, e, label in events:
        s, e = float(s), float(e)
        if not (math.isfinite(s) and math.isfinite(e)):
            continue
        s, e = max(0.0, s), min(e, cap)
        if e - s < min_len:
            continue
        s, e = round(s, 3), min(round(e, 3), cap)
        if s < e:
            clean.append([s, e, str(label)])
    clean.sort(key=lambda x: (x[2], x[0]))
    out: list[list] = []
    for s, e, label in clean:
        if out and out[-1][2] == label and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, label])
    return sorted(out)
