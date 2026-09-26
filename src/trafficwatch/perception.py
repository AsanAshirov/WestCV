"""One decode pass + batched detection -> Perception; tracking of a Perception.

A Perception can be cached to .npz (tools/cache_perception.py) so that tracking and
rules can be re-run on the sample videos in seconds while tuning thresholds.
"""
from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detector import Detector
from .tracker import ByteTracker
from .video import VideoMeta, analysis_size, iter_frames


@dataclass
class Perception:
    video_id: str
    fps: float
    n_frames: int
    frame_w: int                 # analysis frame size (detections are in these pixels)
    frame_h: int
    stride: int
    times: np.ndarray            # (F,) seconds of analysed frames
    dets: np.ndarray             # (M, 7): k (index into times), x1, y1, x2, y2, conf, sc
    complete: bool               # False if the time budget stopped the pass early
    brightness: np.ndarray | None = None  # (F,) mean grey level of each analysed frame (EDA)

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def dt(self) -> float:
        return self.stride / self.fps

    def save(self, path: str | Path) -> None:
        extra = {} if self.brightness is None else {"brightness": self.brightness}
        np.savez_compressed(path, times=self.times, dets=self.dets, **extra, meta=np.array([
            self.video_id, self.fps, self.n_frames, self.frame_w, self.frame_h, self.stride, self.complete], object))

    @classmethod
    def load(cls, path: str | Path) -> Perception:
        z = np.load(path, allow_pickle=True)
        vid, fps, n, w, h, stride, complete = z["meta"].tolist()
        brightness = z["brightness"] if "brightness" in z.files else None
        return cls(vid, float(fps), int(n), int(w), int(h), int(stride), z["times"], z["dets"], bool(complete),
                   brightness)


def perceive(meta: VideoMeta, detector: Detector, cfg: dict, deadline: float | None = None,
             on_progress: Callable[[float], None] | None = None,
             on_frame: Callable[[float, np.ndarray], None] | None = None) -> Perception:
    """`on_frame(t, frame)` sees every analysed frame in order (the demo uses it to run
    Part B and keep small preview frames without decoding the video twice)."""
    dcfg = cfg["decode"]
    stride = max(1, int(round(meta.fps / dcfg["analysis_fps"])))
    w, h = analysis_size(meta, dcfg["width"])
    batch_size = cfg["detector"]["batch"]
    times, rows, frames, idxs, brightness = [], [], [], [], []
    complete = True

    def flush():
        for k_idx, det in zip(idxs, detector(frames)):
            k = len(times)
            times.append(k_idx / meta.fps)
            if len(det):
                rows.append(np.column_stack([np.full(len(det), k, np.float32), det]))
        frames.clear()
        idxs.clear()

    source = iter_frames(meta, stride, w, dcfg["backend"], tuple(dcfg.get("ffmpeg_input_args") or ()))
    batch_start = time.perf_counter()
    with contextlib.closing(source):
        for idx, frame in source:
            brightness.append(float(frame[::4, ::4].mean()))
            if on_frame:
                on_frame(idx / meta.fps, frame)
            frames.append(frame)
            idxs.append(idx)
            if len(frames) == batch_size:
                flush()
                if on_progress and meta.n_frames:
                    on_progress(min(1.0, idx / meta.n_frames))
                now = time.perf_counter()
                # stop if the next batch (decode + detect, as long as this one) would end past the deadline
                if deadline is not None and now + (now - batch_start) > deadline:
                    complete = False
                    break
                batch_start = now
    if frames:
        flush()
    dets = np.vstack(rows) if rows else np.zeros((0, 7), np.float32)
    return Perception(meta.video_id, meta.fps, meta.n_frames, w, h, stride,
                      np.asarray(times, np.float64), dets, complete,
                      np.asarray(brightness[:len(times)], np.float32))


def track(perception: Perception, tracker_cfg: dict) -> np.ndarray:
    """-> observations (N, 8): t, track_id, x1, y1, x2, y2, conf, sc."""
    tracker = ByteTracker(**tracker_cfg)
    k_col = perception.dets[:, 0].astype(int)
    bounds = np.searchsorted(k_col, np.arange(len(perception.times) + 1))
    out = []
    for k, t in enumerate(perception.times):
        res = tracker.update(float(t), perception.dets[bounds[k]:bounds[k + 1], 1:])
        if len(res):
            out.append(np.column_stack([np.full(len(res), t), res]))
    return np.vstack(out) if out else np.zeros((0, 8))
