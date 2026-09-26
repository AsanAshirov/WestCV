"""congestion: traffic at a standstill or crawling across a direction of travel.

Directions are the main peaks of the flow field; each road cell belongs to the
direction its traffic normally follows. Per second and per direction: enough
vehicles present, median speed at a crawl, most of them slow. Directions are merged
into one timeline (simultaneous events of one class are one segment).
`min_duration_s` separates a jam from an ordinary red-light queue.
"""
from __future__ import annotations

import numpy as np

from ..detector import VEHICLE
from .common import RuleContext, runs


def detect(ctx: RuleContext, cfg: dict) -> list[tuple[float, float]]:
    groups = _direction_groups(ctx.maps, cfg)
    if not groups:
        return []
    n_bins = int(np.ceil(ctx.duration / cfg["bin_s"])) + 1
    per_bin = [[{} for _ in range(n_bins)] for _ in groups]  # group -> bin -> tid -> [speeds]
    fgw, fgh = ctx.maps.flow_grid
    for tr in ctx.tracks:
        if tr.sc != VEHICLE:
            continue
        ix, iy = ctx.maps.cell(tr.foot, ctx.maps.flow_grid)
        bins = (tr.t / cfg["bin_s"]).astype(int)
        for g, cells in enumerate(groups):
            member = cells[iy, ix]
            for b, v in zip(bins[member], tr.speed[member]):
                per_bin[g][b].setdefault(tr.tid, []).append(v)
    t_bins = np.arange(n_bins) * cfg["bin_s"]
    out = []
    for g in range(len(groups)):
        jam = np.zeros(n_bins, bool)
        for b, tracks in enumerate(per_bin[g]):
            if len(tracks) < cfg["min_vehicles"]:
                continue
            speeds = np.array([np.median(v) for v in tracks.values()])
            jam[b] = (np.median(speeds) <= cfg["crawl_speed"]
                      and np.mean(speeds <= cfg["slow_speed"]) >= cfg["min_slow_fraction"])
        out += runs(jam, t_bins, cfg["gap_s"], cfg["min_duration_s"], cfg["bin_s"])
    return out


def _direction_groups(maps, cfg: dict) -> list[np.ndarray]:
    """Boolean flow-grid masks, one per main direction of travel."""
    s, n = maps.flow_sum, maps.flow_count
    norm = np.linalg.norm(s, axis=2)
    ok = (n >= cfg["flow_min_count"]) & (norm >= cfg["flow_min_coherence"] * np.maximum(n, 1e-9))
    if not ok.any():
        return []
    ang = np.arctan2(s[..., 1], s[..., 0])
    hist, edges = np.histogram(ang[ok], bins=36, range=(-np.pi, np.pi), weights=n[ok])
    peaks, total = [], hist.sum()
    work = hist.astype(float).copy()
    while len(peaks) < cfg["max_directions"]:
        k = int(np.argmax(work))
        if work[k] < cfg["min_direction_share"] * total:
            break
        peaks.append((edges[k] + edges[k + 1]) / 2)
        for d in range(-6, 7):  # suppress +-60 degrees around the peak
            work[(k + d) % 36] = 0
    groups = []
    for p in peaks:
        diff = np.abs(np.angle(np.exp(1j * (ang - p))))
        groups.append(ok & (diff <= np.deg2rad(cfg["direction_tolerance_deg"])))
    return groups
