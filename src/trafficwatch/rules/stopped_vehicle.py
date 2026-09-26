"""stopped_vehicle: stationary on the carriageway >= 10 s, not in a queue at a signal.

Start = the moment the vehicle stops (not +10 s: that would cost IoU), end = it moves
again or its track ends. Queue vs stopped: while it stands, does same-direction
traffic keep flowing past it? If yes, it is a stopped vehicle. If nothing passes and
other vehicles stand next to it, it is a queue, unless the stop lasts longer than
`queue_max_s` (longer than any red light).
"""
from __future__ import annotations

import numpy as np

from ..detector import VEHICLE
from ..tracks import stationary_mask
from .common import RuleContext, runs


def detect(ctx: RuleContext, cfg: dict) -> list[tuple[float, float]]:
    out = []
    maps = ctx.maps
    for tr in ctx.tracks:
        if tr.sc != VEHICLE:
            continue
        still = stationary_mask(tr, ctx.track_cfg)
        on_road = maps.lookup(maps.road, tr.foot) & ~maps.lookup(maps.offroad, tr.foot)
        for s, e in runs(still, tr.t, cfg["gap_s"], cfg["min_stop_s"], ctx.dt):
            sel = (tr.t >= s) & (tr.t < e)
            if on_road[sel].mean() < cfg["min_on_road"]:
                continue
            if cfg["ignore_whole_video"] and s <= cfg["edge_s"] and e >= ctx.duration - cfg["edge_s"]:
                continue  # parked for the whole clip
            flow, queued = _surroundings(ctx, tr, sel, cfg)
            if flow < cfg["min_flow_fraction"] and queued >= cfg["queue_fraction"] and e - s < cfg["queue_max_s"]:
                continue
            out.append((s, e))
    return out


def _surroundings(ctx: RuleContext, tr, sel: np.ndarray, cfg: dict) -> tuple[float, float]:
    """Fractions of sampled moments with same-direction traffic moving past / with
    other vehicles standing close by."""
    idx = np.flatnonzero(sel)
    idx = idx[:: max(1, int(round(cfg["probe_every_s"] / ctx.dt)))]
    mid = idx[len(idx) // 2]
    ref = ctx.maps.flow_at(tr.foot[mid:mid + 1], exclude=tr)[0]
    tcfg = ctx.track_cfg
    flowing = queued = 0
    for i in idx:
        radius = cfg["neighbour_radius_bs"] * tr.bs[i]
        moving_same = standing = False
        for other, j in ctx.others_at(tr.t[i], tr):
            if other.sc != VEHICLE or np.linalg.norm(other.foot[j] - tr.foot[i]) > radius:
                continue
            if other.speed[j] >= tcfg["move_speed"]:
                h = other.heading[j]
                if np.isnan(ref[0]) or (not np.isnan(h[0]) and float(h @ ref) > cfg["same_direction_dot"]):
                    moving_same = True
            elif other.speed[j] < tcfg["stop_speed"]:
                standing = True
        flowing += moving_same
        queued += standing and not moving_same
    n = max(len(idx), 1)
    return flowing / n, queued / n
