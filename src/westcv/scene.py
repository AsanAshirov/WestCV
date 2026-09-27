"""Per-video scene context for the event rules.

Holds the tracks (split at ID switches), the scene geometry of this intersection mapped into
the video by registration, the carriageway mask, per-sample filters (people inside vehicles,
riders of two-wheelers) and the signal timeline of the heads that face the camera.

Units: pixels of the original video; times in seconds (frame index / fps).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

GRID = 8  # carriageway mask cell, original pixels
PERSON, VEHICLE = 0, 1  # track id % 3
TWO_WHEELERS = (1, 3)
BIG_VEHICLES = (2, 5, 7)


# ---------------------------------------------------------------- small geometry helpers
def warp_poly(H, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)


def inside(poly, xy, margin=0.0):
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(xy[0]), float(xy[1])), True) >= -margin


def signed_dist(poly, xy):
    """> 0 inside the polygon, < 0 outside (pixels)."""
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(xy[0]), float(xy[1])), True)


def foot(r):
    return ((r[2] + r[4]) / 2, r[5])


def feet(tr):
    return np.stack([(tr[:, 2] + tr[:, 4]) / 2, tr[:, 5]], axis=1)


def sizes(tr):
    return np.sqrt(((tr[:, 4] - tr[:, 2]) * (tr[:, 5] - tr[:, 3])).clip(1))


def speeds(tr, fps, win=1.0):
    """Ground-point speed in box sizes per second over +-win/2 s. Returns t, points, sizes, speed."""
    t = tr[:, 0] / fps
    p = feet(tr)
    s = sizes(tr)
    j0 = np.searchsorted(t, t - win / 2)
    j1 = np.minimum(len(tr) - 1, np.searchsorted(t, t + win / 2))
    dt = t[j1] - t[j0]
    v = np.where(dt > 0, np.linalg.norm(p[j1] - p[j0], axis=1) / np.where(dt > 0, dt, 1) / s, 0.0)
    return t, p, s, v


def runs_from_flags(t, flags, gap, min_len, dt):
    """Runs of True samples; gaps up to `gap` s bridged; each run ends one sample (dt) after its last True."""
    idx = np.flatnonzero(flags)
    if not len(idx):
        return []
    segs = []
    start = prev = t[idx[0]]
    for i in idx[1:]:
        if t[i] - prev > gap + 1e-9:
            segs.append((start, prev + dt))
            start = t[i]
        prev = t[i]
    segs.append((start, prev + dt))
    return [(a, b) for a, b in segs if b - a >= min_len]


# ---------------------------------------------------------------- signals
@dataclass
class Signals:
    t: np.ndarray            # sample times (s)
    veh_on: np.ndarray       # (T, 3) head 7 lamps red, yellow, green
    ped_on: np.ndarray       # (T, 2) head 8 lamps red, green

    def __post_init__(self):
        self.veh_green = self._flash_fill(self.veh_on[:, 2])
        self.veh_red = self.veh_on[:, 0].astype(bool) & ~self.veh_on[:, 1].astype(bool)
        self.ped_green = self._flash_fill(self.ped_on[:, 1])
        g = self.veh_green.astype(int)
        self.veh_green_onsets = self.t[1:][np.diff(g) == 1]

    def _flash_fill(self, on, gap=1.3):
        """Green including its flashing end: off-gaps shorter than `gap` s are filled."""
        on = on.astype(bool).copy()
        idx = np.flatnonzero(on)
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 1 and self.t[b] - self.t[a] <= gap:
                on[a:b] = True
        return on

    def index(self, t):
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))

    def veh_red_for(self, t):
        """Seconds head 7 has shown plain red at time t (0 if it is not red)."""
        i = self.index(t)
        if not self.veh_red[i]:
            return 0.0
        j = i
        while j > 0 and self.veh_red[j - 1]:
            j -= 1
        return float(self.t[i] - self.t[j])


# ---------------------------------------------------------------- scene
class Scene:
    def __init__(self, rows: np.ndarray, dets: np.ndarray, meta: dict, H: np.ndarray | None, signals: Signals | None,
                 geometry: dict | None):
        """With H and geometry: this intersection, geometry mapped from the reference view. With
        H=None (another camera, or a view that does not register): no zebras, lines or signals are
        known and the carriageway is learnt from where vehicles drive (`self.auto` is True)."""
        self.meta = meta
        self.fps = float(meta["fps"])
        self.dt = 3 / self.fps  # tracker sample spacing (every 3rd frame)
        self.H = H
        self.sig = signals
        self.auto = H is None or geometry is None
        g = geometry if not self.auto else {"crosswalks": {}, "islands": {}, "median": {}, "sidewalks": {}, "stop_lines": {}}
        self.crosswalks = {k: warp_poly(H, v) for k, v in g["crosswalks"].items()}
        self.islands = [warp_poly(H, v) for v in g["islands"].values()] + [warp_poly(H, v) for v in g["median"].values()]
        self.sidewalks = [warp_poly(H, v) for v in g["sidewalks"].values()]
        self.stop4 = warp_poly(H, g["stop_lines"]["4"]) if "4" in g["stop_lines"] else None
        self.solid_lines = {k: warp_poly(H, v) for k, v in g.get("solid_lines", {}).items()}
        self.dets = dets
        self.tracks = self._split_teleports(rows)
        self.rows = np.concatenate(list(self.tracks.values())) if self.tracks else np.zeros((0, 8), np.float32)
        self._people_filters(dets)
        if self.auto:
            self._road_mask_from_traffic()
        else:
            self._road_mask()

    # -- construction helpers
    @staticmethod
    def _split_teleports(rows):
        """Cut a track where its ground point jumps > 2 box heights between samples (ID switch).
        The later parts get id + 3000000 * k (id % 3, the object group, is kept)."""
        out = {}
        if not len(rows):
            return out
        order = np.lexsort((rows[:, 0], rows[:, 1]))
        rows = rows[order]
        ids, starts = np.unique(rows[:, 1], return_index=True)
        ends = list(starts[1:]) + [len(rows)]
        for tid, a0, b0 in zip(ids.astype(int), starts, ends):
            tr = rows[a0:b0]
            p = feet(tr)
            h = (tr[:, 5] - tr[:, 3]).clip(1)
            jump = np.linalg.norm(np.diff(p, axis=0), axis=1) > 2 * np.minimum(h[1:], h[:-1])
            cuts = [0] + (np.flatnonzero(jump) + 1).tolist() + [len(tr)]
            for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
                part = tr[a:b].copy()
                nid = int(tid) + 3000000 * k
                part[:, 1] = nid
                out[nid] = part
        return out

    STANDING_EXEMPT = True  # see _people_filters; a switch for the dev comparison

    def _people_filters(self, dets):
        """occupant: (frame, person id) inside a car/bus/truck box (>60% of the person box), unless the
        person's feet reach the vehicle's bottom edge and the person is much shorter than the vehicle:
        that is somebody standing in front of a bus or car, not sitting in it.
        riders: person tracks with a two-wheeler detection under them in >= 20% of samples."""
        rows = self.rows
        ppl = rows[rows[:, 1] % 3 == PERSON]
        big = dets[np.isin(dets[:, 6], BIG_VEHICLES) & (dets[:, 5] >= 0.25)]
        two = dets[np.isin(dets[:, 6], TWO_WHEELERS)]
        tracked_veh = rows[rows[:, 1] % 3 == VEHICLE]
        vboxes = np.concatenate([big[:, :5], tracked_veh[:, [0, 2, 3, 4, 5]]]) if len(big) or len(tracked_veh) else np.zeros((0, 5))
        occ, rider_hits = set(), {}
        vb_f = self._by_frame(vboxes)
        tw_f = self._by_frame(two[:, :5])
        for f, grp in self._by_frame(ppl).items():
            x1, y1, x2, y2 = grp[:, 2], grp[:, 3], grp[:, 4], grp[:, 5]
            vb = vb_f.get(f)
            if vb is not None:
                area = np.maximum(1.0, (x2 - x1) * (y2 - y1))[:, None]
                ix = np.clip(np.minimum(x2[:, None], vb[None, :, 3]) - np.maximum(x1[:, None], vb[None, :, 1]), 0, None)
                iy = np.clip(np.minimum(y2[:, None], vb[None, :, 4]) - np.maximum(y1[:, None], vb[None, :, 2]), 0, None)
                vh = (vb[:, 4] - vb[:, 2])[None, :]
                standing = (vb[None, :, 4] - y2[:, None] < 0.05 * vh) & ((y2 - y1)[:, None] < 0.6 * vh)
                inside = (ix * iy / area > 0.6) & ~(standing & self.STANDING_EXEMPT)
                for pid in grp[inside.any(axis=1), 1]:
                    occ.add((f, int(pid)))
            tw = tw_f.get(f)
            if tw is not None:
                cx, cy = (tw[:, 1] + tw[:, 3]) / 2, (tw[:, 2] + tw[:, 4]) / 2
                w, h = (x2 - x1)[:, None], (y2 - y1)[:, None]
                hit = ((cx[None] > x1[:, None] - 0.3 * w) & (cx[None] < x2[:, None] + 0.3 * w)
                       & (cy[None] > ((y1 + y2) / 2)[:, None]) & (cy[None] < y2[:, None] + 0.4 * h)).any(axis=1)
                for pid in grp[hit, 1]:
                    rider_hits[int(pid)] = rider_hits.get(int(pid), 0) + 1
        self.occupant = occ
        self.riders = {tid for tid, n in rider_hits.items() if n >= 0.2 * len(self.tracks[tid])}

    @staticmethod
    def _by_frame(arr):
        if not len(arr):
            return {}
        arr = arr[np.argsort(arr[:, 0], kind="stable")]
        fr, starts = np.unique(arr[:, 0].astype(int), return_index=True)
        ends = list(starts[1:]) + [len(arr)]
        return {int(f): arr[a:b] for f, a, b in zip(fr, starts, ends)}

    def _road_mask(self):
        """Carriageway = frame minus sidewalks, islands and the median."""
        w, h = self.meta["width"] // GRID, self.meta["height"] // GRID
        road = np.ones((h, w), np.uint8)
        for poly in self.sidewalks + self.islands:
            cv2.fillPoly(road, [(poly / GRID).round().astype(np.int32)], 0)
        self.road = road
        self.road_depth = cv2.distanceTransform(road, cv2.DIST_L2, 3) * GRID

    def _road_mask_from_traffic(self, min_vehicles=3, speed=0.5):
        """Carriageway = cells where at least `min_vehicles` different vehicles have driven (moving),
        closed morphologically. Parking bays and sidewalks stay out: vehicles do not drive there."""
        w, h = self.meta["width"] // GRID, self.meta["height"] // GRID
        acc = np.zeros((h, w), np.float32)
        for tid, tr in self.tracks.items():
            if tid % 3 != VEHICLE or len(tr) < 10:
                continue
            t, p, s, v = speeds(tr, self.fps)
            seen = np.zeros((h, w), np.uint8)
            for r, (x, y), vv in zip(tr, p, v):
                if vv >= speed:
                    rad = max(1, int(min(0.15 * (r[4] - r[2]), 24) / GRID))
                    cv2.circle(seen, (int(x // GRID), int((y - 0.05 * (r[5] - r[3])) // GRID)), rad, 1, -1)
            acc += seen
        road = (acc >= min_vehicles).astype(np.uint8)
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        self.road = road
        self.road_depth = cv2.distanceTransform(road, cv2.DIST_L2, 3) * GRID

    # -- queries
    def road_depth_at(self, x, y):
        i, j = int(y // GRID), int(x // GRID)
        if 0 <= i < self.road.shape[0] and 0 <= j < self.road.shape[1]:
            return float(self.road_depth[i, j])
        return 0.0

    def road_depths(self, pts):
        i = np.clip((pts[:, 1] // GRID).astype(int), 0, self.road.shape[0] - 1)
        j = np.clip((pts[:, 0] // GRID).astype(int), 0, self.road.shape[1] - 1)
        ok = (pts[:, 1] >= 0) & (pts[:, 0] >= 0) & (pts[:, 1] < self.meta["height"]) & (pts[:, 0] < self.meta["width"])
        return np.where(ok, self.road_depth[i, j], 0.0)

    def on_road(self, x, y):
        return self.road_depth_at(x, y) > 0

    # -- construction from the development caches
    @classmethod
    def from_cache(cls, stem: str, root: str | Path, tag: str = ""):
        from .registration import Reference, register
        root = Path(root)
        z = np.load(root / f"cache/{stem}{tag}_tracks.npz")
        meta = json.loads(str(z["meta"]))
        dets = np.load(root / f"cache/{stem}{tag}_det.npz")["dets"]
        ref = Reference.load(root / "scene/reference_sift.npz")
        H, _ = register(ref, cv2.imread(str(root / f"DataSets/proxies/ref/{stem}_median.png")), meta["width"])
        s = np.load(root / f"cache/{stem}_signals.npz")
        sig = Signals(s["t"], s["veh_median_on"], s["ped_left_on"])
        geo = json.loads((root / "scene/geometry.json").read_text(encoding="utf-8"))
        sc = cls(z["tracks"], dets, meta, H, sig, geo)
        sc.stem = stem
        return sc
