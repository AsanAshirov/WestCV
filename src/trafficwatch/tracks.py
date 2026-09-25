"""Offline track repair and kinematics (Part A only; uses past and future samples).

Kinematic quantities are in box-size units: BS = sqrt(w*h) of the object's box,
smoothed over time. A car's BS is roughly 2-3 m, so 1 BS/s ~ 9 km/h. This keeps
thresholds meaningful without a camera calibration; a homography can replace it later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    tid: int
    sc: int                      # superclass, see detector.SUPER_NAMES
    t: np.ndarray                # (n,) seconds
    box: np.ndarray              # (n, 4) xyxy in analysis-frame pixels
    conf: np.ndarray             # (n,)
    foot: np.ndarray = field(init=False)     # (n, 2) bottom-centre, smoothed
    bs: np.ndarray = field(init=False)       # (n,) box size, px
    vel: np.ndarray = field(init=False)      # (n, 2) foot velocity, px/s
    speed: np.ndarray = field(init=False)    # (n,) BS/s
    heading: np.ndarray = field(init=False)  # (n, 2) unit vector, NaN when not moving

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    def at(self, t: float) -> int | None:
        """Index of the sample at time t (tolerance 1 ms) or None."""
        i = int(np.searchsorted(self.t, t - 1e-3))
        return i if i < len(self.t) and abs(self.t[i] - t) < 1e-3 else None


def _interp2(t_query, t, xy):
    return np.column_stack([np.interp(t_query, t, xy[:, 0]), np.interp(t_query, t, xy[:, 1])])


def _rolling(values: np.ndarray, t: np.ndarray, half_s: float, fn) -> np.ndarray:
    out = np.empty(len(values), np.float64)
    lo = np.searchsorted(t, t - half_s, side="left")
    hi = np.searchsorted(t, t + half_s, side="right")
    for i, (a, b) in enumerate(zip(lo, hi)):
        out[i] = fn(values[a:b], axis=0)
    return out


def compute_kinematics(track: Track, cfg: dict) -> Track:
    t, box = track.t, track.box
    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    track.bs = np.maximum(_rolling(np.sqrt(np.maximum(w * h, 1.0)), t, cfg["size_window_s"], np.median), 1.0)
    raw_foot = np.column_stack([(box[:, 0] + box[:, 2]) / 2, box[:, 3]])
    half = cfg["smooth_window_s"]
    track.foot = np.column_stack([_rolling(raw_foot[:, k], t, half, np.median) for k in (0, 1)])
    dv = cfg["velocity_window_s"]
    t0, t1 = np.clip(t - dv, t[0], t[-1]), np.clip(t + dv, t[0], t[-1])
    span = np.maximum(t1 - t0, 1e-6)
    track.vel = (_interp2(t1, t, track.foot) - _interp2(t0, t, track.foot)) / span[:, None]
    track.vel[(t1 - t0) < 1e-6] = 0.0
    track.speed = np.linalg.norm(track.vel, axis=1) / track.bs
    heading = track.vel / np.maximum(np.linalg.norm(track.vel, axis=1), 1e-9)[:, None]
    heading[track.speed < cfg["heading_min_speed"]] = np.nan
    track.heading = heading
    return track


def displacement(track: Track, window_s: float) -> np.ndarray:
    """Displacement of the foot over [t - w/2, t + w/2], in BS."""
    t = track.t
    a = _interp2(np.clip(t - window_s / 2, t[0], t[-1]), t, track.foot)
    b = _interp2(np.clip(t + window_s / 2, t[0], t[-1]), t, track.foot)
    return np.linalg.norm(b - a, axis=1) / track.bs


def stationary_mask(track: Track, cfg: dict) -> np.ndarray:
    return (track.speed < cfg["stop_speed"]) & (displacement(track, cfg["stop_window_s"]) < cfg["stop_disp"])


def build_tracks(obs: np.ndarray, cfg: dict) -> list[Track]:
    """obs rows: t, track_id, x1, y1, x2, y2, conf, sc -> repaired tracks with kinematics."""
    tracks = []
    for tid in np.unique(obs[:, 1]).astype(int):
        rows = obs[obs[:, 1] == tid]
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        sc = int(np.bincount(rows[:, 7].astype(int)).argmax())
        tracks.append(Track(tid, sc, rows[:, 0].copy(), rows[:, 2:6].astype(np.float64), rows[:, 6].copy()))
    tracks = _split_on_jumps(tracks, cfg)
    tracks = [compute_kinematics(tr, cfg) for tr in tracks if len(tr.t) >= cfg["min_samples"]]
    tracks = _stitch_stationary(tracks, cfg)
    return [tr for tr in tracks if tr.end - tr.start >= cfg["min_duration_s"]]


def _split_on_jumps(tracks: list[Track], cfg: dict) -> list[Track]:
    """Cut a track where the box centre jumps implausibly: usually an ID switch."""
    out, next_id = [], max((tr.tid for tr in tracks), default=0) + 1
    for tr in tracks:
        c = (tr.box[:, :2] + tr.box[:, 2:]) / 2
        size = np.sqrt(np.maximum((tr.box[:, 2] - tr.box[:, 0]) * (tr.box[:, 3] - tr.box[:, 1]), 1.0))
        step = np.linalg.norm(np.diff(c, axis=0), axis=1) / size[1:]
        dt = np.maximum(np.diff(tr.t), 1e-6)
        cuts = np.where((step > cfg["jump_bs"]) & (dt < cfg["jump_max_dt_s"]))[0] + 1
        pieces = np.split(np.arange(len(tr.t)), cuts)
        for k, idx in enumerate(pieces):
            tid = tr.tid if k == 0 else next_id
            next_id += k > 0
            out.append(Track(tid, tr.sc, tr.t[idx], tr.box[idx], tr.conf[idx]))
    return out


def _stitch_stationary(tracks: list[Track], cfg: dict) -> list[Track]:
    """Join a track that ends standing still with a later one that starts at the same
    place (a parked car hidden by a passing bus gets a new ID)."""
    tracks = sorted(tracks, key=lambda tr: tr.start)
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(tracks):
            if a.speed[-1] >= cfg["stop_speed"]:
                continue
            for j in range(i + 1, len(tracks)):
                b = tracks[j]
                gap = b.start - a.end
                if gap <= 0 or b.sc != a.sc:
                    continue
                if gap > cfg["stitch_max_gap_s"]:
                    break
                dist = np.linalg.norm(b.foot[0] - a.foot[-1]) / a.bs[-1]
                if dist < cfg["stitch_max_dist_bs"] and b.speed[0] < cfg["stop_speed"]:
                    joined = Track(a.tid, a.sc, np.concatenate([a.t, b.t]),
                                   np.vstack([a.box, b.box]), np.concatenate([a.conf, b.conf]))
                    tracks[i] = compute_kinematics(joined, cfg)
                    del tracks[j]
                    merged = True
                    break
            if merged:
                break
    return tracks
