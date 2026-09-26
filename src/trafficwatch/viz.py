"""Drawing shared by tools/render_review.py, the demo app and the site builder.

Videos are written as H.264 yuv420p through the bundled ffmpeg so that browsers can
play them (OpenCV's mp4v output does not play in browsers).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .scene import SceneMaps
from .tracks import Track
from .video import ffmpeg_exe

TRACK_COLORS = [(60, 180, 255), (255, 160, 60), (80, 220, 80), (200, 80, 220)]  # BGR per superclass
EVENT_COLORS = {  # hex, one per official class, shared by the demo and the site
    "accident": "#d62728", "near_miss": "#ff7f0e", "red_light": "#e377c2", "wrong_way": "#9467bd",
    "illegal_u_turn": "#8c564b", "stopped_vehicle": "#1f77b4", "jaywalking": "#2ca02c",
    "failure_to_yield": "#bcbd22", "illegal_turn": "#7f7f7f", "solid_line_crossing": "#17becf",
    "stop_line": "#aec7e8", "congestion": "#ffbb78", "road_obstacle": "#98df8a", "fire_smoke": "#ff9896",
}
BAR_H = 70


class H264Writer:
    def __init__(self, path: str | Path, size: tuple[int, int], fps: float, crf: int = 26):
        exe = ffmpeg_exe()
        if exe is None:
            raise RuntimeError("no ffmpeg binary (pip install imageio-ffmpeg)")
        w, h = size
        self.size = size
        self.proc = subprocess.Popen(
            [exe, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode the video")


def tracks_by_time(tracks: list[Track]) -> dict[float, list[tuple[Track, int]]]:
    out: dict[float, list[tuple[Track, int]]] = {}
    for tr in tracks:
        for i, t in enumerate(tr.t):
            out.setdefault(round(float(t), 3), []).append((tr, i))
    return out


def scene_overlay(maps: SceneMaps, size: tuple[int, int]) -> np.ndarray:
    """Learned carriageway in green, drawn crossings in yellow."""
    w, h = size
    overlay = np.zeros((h, w, 3), np.uint8)
    road = cv2.resize(maps.road.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    cross = cv2.resize(maps.crosswalk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay[road] = (0, 90, 0)
    overlay[cross] = (0, 150, 150)
    return overlay


def draw_frame(frame: np.ndarray, items: list[tuple[Track, int]], events: list[list], t: float,
               scale: float = 1.0, overlay: np.ndarray | None = None) -> np.ndarray:
    """Boxes with track id and speed (BS/s), active events on top. `scale` maps
    analysis-frame pixels (where tracks live) to this frame."""
    img = cv2.addWeighted(frame, 1.0, overlay, 0.35, 0) if overlay is not None else frame.copy()
    for tr, i in items:
        x1, y1, x2, y2 = (tr.box[i] * scale).astype(int)
        color = TRACK_COLORS[tr.sc]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{tr.tid} {tr.speed[i]:.1f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    active = [label for s, e, label in events if s <= t < e]
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, f"t={t:7.2f}s  {'  '.join(active)}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (80, 80, 255) if active else (255, 255, 255), 2, cv2.LINE_AA)
    return img


def timeline_bar(width: int, t: float, duration: float, pred: list[list], gt: list[list] | None = None) -> np.ndarray:
    """One row per class: prediction filled, ground truth outlined, cursor at t."""
    gt = gt or []
    labels = sorted({e[2] for e in pred} | {e[2] for e in gt})
    bar = np.full((BAR_H, width, 3), 30, np.uint8)
    rh = (BAR_H - 6) // max(len(labels), 1)
    x0 = 110
    for r, label in enumerate(labels):
        y = 3 + r * rh
        color = _bgr(EVENT_COLORS.get(label, "#cccccc"))
        cv2.putText(bar, label[:15], (2, y + rh - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1)
        for events, fill in ((gt, 1), (pred, -1)):
            for s, e, lab in events:
                if lab == label:
                    a = x0 + int(s / duration * (width - x0))
                    b = x0 + int(e / duration * (width - x0))
                    cv2.rectangle(bar, (a, y + 1), (max(b, a + 1), y + rh - 1), color, fill)
    x = x0 + int(t / max(duration, 1e-6) * (width - x0))
    cv2.line(bar, (x, 0), (x, BAR_H), (255, 255, 255), 1)
    return bar


def _bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return b, g, r
