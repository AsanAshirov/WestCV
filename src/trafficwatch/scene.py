"""Scene layout: optional hand-drawn polygons plus maps learned from the video itself.

- configs/scene.json (from tools/annotation/labelme_to_scene.py, coordinates 0-1):
  `crosswalk`, `carriageway`, `sidewalk`, `island`, `parking` polygons. All optional.
- Learned per video (Part A may look at the whole video): the carriageway as the
  area where moving vehicles drive, and a flow field of the dominant direction of
  travel in each grid cell. The per-video learning keeps working if the tripod
  moved between recording sessions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import TWO_WHEELER, VEHICLE
from .tracks import Track


@dataclass
class Scene:
    polygons: dict[str, list[np.ndarray]]  # label -> list of (k, 2) arrays in [0, 1]

    def mask(self, labels: tuple[str, ...], grid: tuple[int, int], dilate_cells: int = 0) -> np.ndarray | None:
        polys = [p for lab in labels for p in self.polygons.get(lab, [])]
        if not polys:
            return None
        gw, gh = grid
        img = np.zeros((gh, gw), np.uint8)
        for p in polys:
            cv2.fillPoly(img, [np.round(p * [gw, gh]).astype(np.int32)], 1)
        if dilate_cells:
            img = cv2.dilate(img, np.ones((2 * dilate_cells + 1,) * 2, np.uint8))
        return img.astype(bool)


def load_scene(path: str | Path) -> Scene | None:
    path = Path(path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    polygons: dict[str, list[np.ndarray]] = {}
    for label, shapes in data.get("shapes", {}).items():
        for s in shapes:
            if s.get("type") in ("polygon", "rectangle") and len(s["points"]) >= 3:
                polygons.setdefault(label, []).append(np.asarray(s["points"], np.float64))
    return Scene(polygons)


@dataclass
class SceneMaps:
    frame_wh: tuple[int, int]
    grid: tuple[int, int]            # (gw, gh) of the masks
    road: np.ndarray                 # (gh, gw) bool
    crosswalk: np.ndarray            # (gh, gw) bool
    offroad: np.ndarray              # (gh, gw) bool: sidewalk / island / parking
    has_crosswalks: bool
    flow_grid: tuple[int, int]
    flow_sum: np.ndarray             # (fgh, fgw, 2) sum of unit headings of moving vehicles
    flow_count: np.ndarray           # (fgh, fgw)

    def cell(self, xy: np.ndarray, grid: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        gw, gh = grid or self.grid
        w, h = self.frame_wh
        xy = np.atleast_2d(xy)
        ix = np.clip((xy[:, 0] / w * gw).astype(int), 0, gw - 1)
        iy = np.clip((xy[:, 1] / h * gh).astype(int), 0, gh - 1)
        return ix, iy

    def lookup(self, grid_mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
        ix, iy = self.cell(xy)
        return grid_mask[iy, ix]

    def flow_at(self, xy: np.ndarray, exclude: Track | None = None, min_count: int = 1,
                min_coherence: float = 0.0) -> np.ndarray:
        """Dominant unit heading per point (NaN where the flow is weak or mixed).
        `exclude` removes that track's own contribution (so a wrong-way car does not vote)."""
        ix, iy = self.cell(xy, self.flow_grid)
        s = self.flow_sum[iy, ix].copy()
        n = self.flow_count[iy, ix].astype(np.float64)
        if exclude is not None:
            own_s, own_n = _flow_accumulate([exclude], self)
            s -= own_s[iy, ix]
            n -= own_n[iy, ix]
        norm = np.linalg.norm(s, axis=1)
        ok = (n >= min_count) & (norm >= min_coherence * np.maximum(n, 1e-9))
        out = s / np.maximum(norm, 1e-9)[:, None]
        out[~ok] = np.nan
        return out


def _flow_accumulate(tracks: list[Track], maps: SceneMaps):
    fgw, fgh = maps.flow_grid
    s = np.zeros((fgh, fgw, 2))
    n = np.zeros((fgh, fgw))
    for tr in tracks:
        ok = ~np.isnan(tr.heading[:, 0])
        if not ok.any():
            continue
        ix, iy = maps.cell(tr.foot[ok], maps.flow_grid)
        np.add.at(s, (iy, ix), tr.heading[ok])
        np.add.at(n, (iy, ix), 1)
    return s, n


def build_maps(tracks: list[Track], frame_wh: tuple[int, int], scene: Scene | None, cfg: dict) -> SceneMaps:
    grid = tuple(cfg["grid"])
    gw, gh = grid
    w, h = frame_wh
    movers = [tr for tr in tracks if tr.sc in (VEHICLE, TWO_WHEELER)]

    # Carriageway = cells covered by the lower part of moving vehicles' boxes, counted
    # once per track, kept where at least `road_min_tracks` different vehicles drove.
    votes = np.zeros((gh, gw), np.int32)
    for tr in movers:
        moving = tr.speed >= cfg["road_min_speed"]
        if not moving.any():
            continue
        seen = np.zeros((gh, gw), np.uint8)
        for x1, y1, x2, y2 in tr.box[moving]:
            ya = y2 - cfg["road_box_fraction"] * (y2 - y1)
            c0, c1 = int(x1 / w * gw), int(x2 / w * gw) + 1  # +1: include the cell of the edge itself
            r0, r1 = int(ya / h * gh), int(y2 / h * gh) + 1
            seen[max(r0, 0):min(r1, gh), max(c0, 0):min(c1, gw)] = 1
        votes += seen
    road = (votes >= cfg["road_min_tracks"]).astype(np.uint8)
    k = np.ones((2 * cfg["road_close_cells"] + 1,) * 2, np.uint8)
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, k).astype(bool)

    drawn_road = scene.mask(("carriageway",), grid) if scene else None
    if drawn_road is not None:
        road = drawn_road if cfg["road_source"] == "scene" else (road | drawn_road)
    crosswalk = scene.mask(("crosswalk",), grid, cfg["crosswalk_margin_cells"]) if scene else None
    offroad = scene.mask(("sidewalk", "island", "parking"), grid) if scene else None
    empty = np.zeros((gh, gw), bool)
    maps = SceneMaps(
        frame_wh=frame_wh, grid=grid, road=road,
        crosswalk=crosswalk if crosswalk is not None else empty,
        offroad=offroad if offroad is not None else empty,
        has_crosswalks=crosswalk is not None,
        flow_grid=tuple(cfg["flow_grid"]),
        flow_sum=np.zeros((cfg["flow_grid"][1], cfg["flow_grid"][0], 2)),
        flow_count=np.zeros((cfg["flow_grid"][1], cfg["flow_grid"][0])),
    )
    maps.flow_sum, maps.flow_count = _flow_accumulate(movers, maps)
    return maps
