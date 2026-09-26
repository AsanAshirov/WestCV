"""jaywalking: a pedestrian on the carriageway outside a crossing.

Needs crosswalk polygons in configs/scene.json: without them every pedestrian on a
crossing would be reported, so the class switches itself off (`require_crosswalks`).
Riders (a `person` box on a motorcycle or bicycle) and people inside vehicles are
excluded. Start/end = first/last sample with the feet inside the road.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..detector import PERSON, TWO_WHEELER, VEHICLE
from .common import RuleContext, inside, runs


def detect(ctx: RuleContext, cfg: dict) -> list[tuple[float, float]]:
    maps = ctx.maps
    if cfg["require_crosswalks"] and not maps.has_crosswalks:
        return []
    road = maps.road.astype(np.uint8)
    if cfg["edge_margin_cells"]:
        k = np.ones((2 * cfg["edge_margin_cells"] + 1,) * 2, np.uint8)
        road = cv2.erode(road, k)
    walkable = road.astype(bool) & ~maps.crosswalk & ~maps.offroad
    out = []
    for tr in ctx.tracks:
        if tr.sc != PERSON:
            continue
        on_road = maps.lookup(walkable, tr.foot)
        if not on_road.any():
            continue
        carried = np.array([_carried(ctx, tr, i) for i in range(len(tr.t))])
        if carried.mean() > cfg["max_carried_fraction"]:
            continue  # a rider or passenger, not a pedestrian
        out += runs(on_road & ~carried, tr.t, cfg["gap_s"], cfg["min_duration_s"], ctx.dt)
    return out


def _carried(ctx: RuleContext, tr, i: int) -> bool:
    foot, box = tr.foot[i], tr.box[i]
    centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    for other, j in ctx.others_at(tr.t[i], tr):
        if other.sc == TWO_WHEELER and (inside(foot, other.box[j]) or inside(centre, other.box[j])):
            return True
        if other.sc == VEHICLE and inside(centre, other.box[j]) and inside(foot, other.box[j]):
            return True
    return False
