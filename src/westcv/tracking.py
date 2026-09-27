"""Multi-object tracking on detections (ByteTrack from Ultralytics, no ReID, no camera motion).

Pedestrians and vehicles are tracked separately: association is by IoU only, and a person
next to a car would otherwise be able to take over the car's track.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

PERSON = (0,)
VEHICLE = (1, 2, 3, 5, 7)  # bicycle, car, motorcycle, bus, truck
ANIMAL = (15, 16, 17, 18, 19)
GROUPS = {"person": PERSON, "vehicle": VEHICLE, "animal": ANIMAL}


class _Dets:
    """The minimal 'results' interface BYTETracker reads: conf, cls, xywh, boolean indexing."""

    def __init__(self, a: np.ndarray):
        self.a = a

    conf = property(lambda self: self.a[:, 4])
    cls = property(lambda self: self.a[:, 5])
    xyxy = property(lambda self: self.a[:, :4])

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.a[:, :4].T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __getitem__(self, mask) -> "_Dets":
        return _Dets(self.a[mask])

    def __len__(self) -> int:
        return len(self.a)


def tracker_args(updates_per_sec: float, buffer_sec: float = 3.0) -> SimpleNamespace:
    return SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.3,
                           track_buffer=max(1, round(buffer_sec * updates_per_sec)),
                           match_thresh=0.8, fuse_score=False)  # fuse_score off at stride 3 (report §4.2)


def track(frames: np.ndarray, dets: np.ndarray, fps: float) -> np.ndarray:
    """Tracks for cached detections.

    frames: processed frame indices; dets: rows (frame, x1, y1, x2, y2, conf, cls).
    Returns rows (frame, track_id, x1, y1, x2, y2, conf, cls); ids are unique across groups.
    """
    step = float(np.median(np.diff(frames))) if len(frames) > 1 else 1.0
    args = tracker_args(fps / step)
    out = []
    for g, (name, classes) in enumerate(GROUPS.items()):
        trk = BYTETracker(args)
        sel = dets[np.isin(dets[:, 6], classes)]
        order = np.searchsorted(sel[:, 0], frames, side="left"), np.searchsorted(sel[:, 0], frames, side="right")
        for f, lo, hi in zip(frames, *order):
            res = trk.update(_Dets(sel[lo:hi, 1:7]))
            if len(res):  # x1 y1 x2 y2 id score cls idx
                ids = res[:, 4] * len(GROUPS) + g
                out.append(np.column_stack([np.full(len(res), f), ids, res[:, :4], res[:, 5], res[:, 6]]))
    tracks = np.concatenate(out) if out else np.zeros((0, 8), np.float32)
    return tracks[np.lexsort((tracks[:, 1], tracks[:, 0]))].astype(np.float32)
