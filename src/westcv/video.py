"""Video access for Part A and the offline tools.

`probe` reads fps and frame count exactly like run_submission.py (OpenCV), so our time axis
t = frame_index / fps is the harness's and the metric's.

`iter_frames` decodes with PyAV and lets the decoder drop non-reference frames. The camera
writes long-GOP H.264 (IBBPBBP..., B frames not referenced), so this returns exactly every
3rd frame at about half the cost of a full decode. Streams without such frames are decoded
fully and thinned by index, so the output is always about every `stride`-th frame.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import av
import cv2
import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class VideoMeta:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps


def probe(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    try:
        return VideoMeta(
            path=str(path),
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(path: str | Path, width: int = 1920, stride: int = 3,
                skip_nonref: bool = True) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame scaled to `width`) for about every `stride`-th frame."""
    meta = probe(path)
    height = round(meta.height * width / meta.width / 2) * 2
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if skip_nonref:
            stream.codec_context.skip_frame = "NONREF"
        start = stream.start_time or 0
        last = -stride
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            idx = round(float((frame.pts - start) * stream.time_base) * meta.fps)
            if idx - last < stride or idx >= meta.n_frames:
                continue
            last = idx
            yield idx, frame.to_ndarray(width=width, height=height, format="bgr24", interpolation="AREA")


def prefetch(items: Iterable[T], size: int = 8) -> Iterator[T]:
    """Run `items` in a background thread so decoding overlaps with GPU inference."""
    q: queue.Queue = queue.Queue(maxsize=size)
    done = object()
    error: list[BaseException] = []

    def worker():
        try:
            for item in items:
                q.put(item)
        except BaseException as e:  # re-raised in the consumer thread
            error.append(e)
        finally:
            q.put(done)

    threading.Thread(target=worker, daemon=True).start()
    while (item := q.get()) is not done:
        yield item
    if error:
        raise error[0]
