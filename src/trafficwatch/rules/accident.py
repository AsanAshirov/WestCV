"""accident: collision between road users. Precision first.

A candidate is an abrupt stop (speed falls from >= `pre_speed` to <= `stop_ratio` of
it within `max_decel_s`) while the vehicle touches another road user at the same
depth in the image (bottom edges level, boxes overlapping). Queues brake gradually
and stop behind each other (different depth), so they are not candidates.
Start = first touching sample, end = when the last participant stops or leaves.
"""
from __future__ import annotations

import numpy as np

from ..detector import VEHICLE, box_iou
from ..tracks import stationary_mask
from .common import RuleContext


def detect(ctx: RuleContext, cfg: dict) -> list[tuple[float, float]]:
    out = []
    for tr in ctx.tracks:
        if tr.sc != VEHICLE or len(tr.t) < 3:
            continue
        for i_stop in _abrupt_stops(tr, cfg):
            hit = _partner(ctx, tr, i_stop, cfg)
            if hit is None:
                continue
            other, t_contact = hit
            end = max(_settle_time(tr, t_contact, ctx, cfg), _settle_time(other, t_contact, ctx, cfg))
            out.append((t_contact, max(end, t_contact + cfg["min_duration_s"])))
    return out


def _abrupt_stops(tr, cfg) -> list[int]:
    """Indices where the vehicle has just come to a halt after an abrupt deceleration."""
    found, last_t = [], -np.inf
    for i in range(len(tr.t)):
        if tr.speed[i] > cfg["pre_speed"] * cfg["stop_ratio"] or tr.t[i] - last_t < cfg["max_decel_s"]:
            continue
        window = (tr.t >= tr.t[i] - cfg["max_decel_s"]) & (tr.t < tr.t[i])
        if window.any() and tr.speed[window].max() >= cfg["pre_speed"]:
            found.append(i)
            last_t = tr.t[i]
    return found


def _partner(ctx: RuleContext, tr, i_stop: int, cfg: dict):
    """Another road user touching `tr` around the stop; returns (track, first contact time)."""
    t_stop = tr.t[i_stop]
    for i in np.flatnonzero((tr.t >= t_stop - cfg["max_decel_s"] - 0.5) & (tr.t <= t_stop + 0.5)):
        a = tr.box[i]
        for other, j in ctx.others_at(tr.t[i], tr):
            b = other.box[j]
            level = abs(a[3] - b[3]) <= cfg["depth_tolerance"] * max(a[3] - a[1], b[3] - b[1])
            if level and box_iou(a[None], b[None])[0, 0] >= cfg["contact_iou"] and _closing(tr, other, tr.t[i], cfg):
                return other, float(tr.t[i])
    return None


def _closing(a, b, t: float, cfg: dict) -> bool:
    """The two were approaching each other before contact (side-by-side traffic is not)."""
    t0 = t - cfg["closing_window_s"]
    if t0 < max(a.t[0], b.t[0]):
        return False

    def foot(tr, when):
        return np.array([np.interp(when, tr.t, tr.foot[:, 0]), np.interp(when, tr.t, tr.foot[:, 1])])

    before = np.linalg.norm(foot(a, t0) - foot(b, t0))
    now = np.linalg.norm(foot(a, t) - foot(b, t))
    return before - now >= cfg["min_closing_bs"] * np.interp(t, a.t, a.bs)


def _settle_time(tr, t_from: float, ctx: RuleContext, cfg: dict) -> float:
    """First moment after t_from at which the participant is stationary, or its last sample."""
    still = stationary_mask(tr, ctx.track_cfg)
    idx = np.flatnonzero((tr.t >= t_from) & still)
    return float(tr.t[idx[0]]) if len(idx) else float(tr.t[-1]) + ctx.dt
