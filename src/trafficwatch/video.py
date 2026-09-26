"""Video metadata and a single sequential decode pass.

Time base everywhere: t = frame_index / CAP_PROP_FPS, exactly as the harness
computes it. The ffmpeg backend scales frames inside ffmpeg (much cheaper than
converting full 4K frames to BGR and resizing them in Python) and runs in its own
process, so decoding overlaps with GPU inference.
"""
from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def video_id(self) -> str:
        return Path(self.path).name


def read_meta(path: str | Path) -> VideoMeta:
    """Same fields and fallbacks as run_submission.video_meta."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    meta = VideoMeta(
        path=str(path),
        fps=float(fps),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return meta


def analysis_size(meta: VideoMeta, width: int) -> tuple[int, int]:
    width = min(width, meta.width)
    height = int(round(meta.height * width / meta.width / 2)) * 2
    return width, height


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def iter_frames(
    meta: VideoMeta,
    stride: int,
    width: int,
    backend: str = "auto",
    ffmpeg_input_args: tuple[str, ...] = (),
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame resized to `width`) for every `stride`-th frame."""
    size = analysis_size(meta, width)
    exe = ffmpeg_exe() if backend in ("auto", "ffmpeg") else None
    if backend == "ffmpeg" and exe is None:
        raise RuntimeError("ffmpeg backend requested but no ffmpeg binary found (pip install imageio-ffmpeg)")
    if exe is not None:
        yield from _iter_ffmpeg(exe, meta, stride, size, ffmpeg_input_args)
    else:
        yield from _iter_cv2(meta, stride, size)


def _iter_ffmpeg(exe, meta, stride, size, input_args):
    w, h = size
    vf = f"select='not(mod(n\\,{stride}))',scale={w}:{h}:flags=area"
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "0", *input_args,
           "-i", meta.path, "-map", "0:v:0", "-an", "-vf", vf, "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    nbytes = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=nbytes * 4)
    k = 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield k * stride, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    if k == 0:
        raise RuntimeError(f"ffmpeg produced no frames for {meta.path}")


def _iter_cv2(meta, stride, size):
    cap = cv2.VideoCapture(meta.path, cv2.CAP_FFMPEG)
    idx = 0
    try:
        while cap.grab():  # grab every frame (long-GOP), convert only the ones we use
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield idx, cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            idx += 1
    finally:
        cap.release()
