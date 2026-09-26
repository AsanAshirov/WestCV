"""wrong_way: a vehicle moving against the dominant direction of the lane it is in.

The direction of every lane is learned from all tracks of the same video (the flow
field), with the tested vehicle's own votes removed. Start/end = first/last sample of
the opposite-direction run; a vehicle leaving the frame ends it.
"""
from __future__ import annotations

import numpy as np

from ..detector import TWO_WHEELER, VEHICLE
from .common import RuleContext, runs


def detect(ctx: RuleContext, cfg: dict) -> list[tuple[float, float]]:
    out = []
    move = ctx.track_cfg["move_speed"]
    for tr in ctx.tracks:
        if tr.sc not in (VEHICLE, TWO_WHEELER) or tr.speed.max() < move:
            continue
        ref = ctx.maps.flow_at(tr.foot, exclude=tr, min_count=cfg["flow_min_count"],
                               min_coherence=cfg["flow_min_coherence"])
        dot = np.einsum("ij,ij->i", np.nan_to_num(tr.heading), np.nan_to_num(ref))
        valid = ~np.isnan(tr.heading[:, 0]) & ~np.isnan(ref[:, 0])
        opposite = valid & (dot < -cfg["opposite_dot"]) & (tr.speed >= move)
        for s, e in runs(opposite, tr.t, cfg["gap_s"], cfg["min_duration_s"], ctx.dt):
            sel = (tr.t >= s) & (tr.t < e)
            if opposite[sel].mean() < cfg["min_opposite_fraction"]:
                continue
            path = np.linalg.norm(np.diff(tr.foot[sel], axis=0), axis=1) / tr.bs[sel][1:]
            if path.sum() >= cfg["min_path_bs"]:
                out.append((s, e))
    return out
