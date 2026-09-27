"""Part A pipeline: video -> events.

One decode pass (PyAV, non-reference frames dropped: every 3rd frame, scaled to WIDTH)
feeds the detector; on the side it keeps a few frames for the background median (scene
registration) and small crops around the expected signal heads. After the pass: registration,
signal timeline, tracking, scene, rules, sanitising.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from .detector import Detector
from .registration import Reference, register
from .rules import detect
from .scene import Scene, Signals
from .segments import sanitize
from .signals import lamp_scores, lamp_states, locate, Head
from .tracking import track
from .video import iter_frames, prefetch, probe

WIDTH = 1280          # decode / detector width (1280 matches 1920 on the dev labels at ~half the cost)
N_BG = 25             # frames kept for the background median
HEAD_PAD = 80         # px (at WIDTH) around a head's reference position: camera shifts up to ~100 px at 4K
DECODE_BRAKE = 1.35 * float(os.environ.get("WESTCV_TIME_FACTOR", "3.0")) / 3.0
                      # x duration: stop decoding here and use what we have. The harness then needs
                      # ~1.4x on a Kaggle T4 (4 vCPU) to decode every 4K frame for Part B, of a 3x budget


class Pipeline:
    def __init__(self, root: Path, weights: str, imgsz: int = 1280, batch: int = 4):
        self.root = Path(root)
        self.det = Detector(str(self.root / weights), imgsz=imgsz)
        self.batch = batch
        self.ref = Reference.load(self.root / "scene/reference_sift.npz")
        self.geometry = json.loads((self.root / "scene/geometry.json").read_text(encoding="utf-8"))
        self.heads_cfg = json.loads((self.root / "scene/signal_heads.json").read_text(encoding="utf-8"))["heads"]
        self.templates = dict(np.load(self.root / "scene/signal_templates.npz"))
        self.det.warmup((round(2160 * WIDTH / 3840), WIDTH, 3))
        self.last_timing: dict = {}

    def __call__(self, video_path: str, classes) -> list[list]:
        t0 = time.perf_counter()
        meta = probe(video_path)
        scale = WIDTH / meta.width             # this video's pixels -> working pixels
        ref_s = WIDTH / self.ref.size[0]       # reference (4K) pixels -> working pixels, any input resolution
        n_proc = max(1, meta.n_frames // 3)
        bg_every = max(1, n_proc // N_BG)
        rois = {name: self._roi(h["box"], ref_s) for name, h in self.heads_cfg.items()}
        frames, rows, bg, crops = [], [], [], {name: [] for name in rois}
        buf = []

        def flush():
            for (idx, _), d in zip(buf, self.det([f for _, f in buf])):
                d[:, :4] /= scale
                rows.append(np.concatenate([np.full((len(d), 1), idx, np.float32), d], axis=1))
                frames.append(idx)
            buf.clear()

        k = 0
        self.braked = False
        for idx, img in prefetch(iter_frames(video_path, width=WIDTH)):
            if k % 50 == 0 and time.perf_counter() - t0 > DECODE_BRAKE * meta.duration:
                self.braked = True  # a slow machine: better a partial video than a zeroed one
                break
            if k % bg_every == 0 and len(bg) < N_BG:
                bg.append(img)
            for name, (x1, y1, x2, y2) in rois.items():
                crops[name].append(img[y1:y2, x1:x2].copy())
            buf.append((idx, img))
            if len(buf) == self.batch:
                flush()
            k += 1
        if buf:
            flush()
        t_decode = time.perf_counter() - t0

        dets = np.concatenate(rows) if rows else np.zeros((0, 7), np.float32)
        frames = np.array(frames)
        tm = {"decode_detect": time.perf_counter() - t0}
        t = time.perf_counter()
        background = np.median(np.stack(bg), axis=0).astype(np.uint8) if bg else None
        H, inliers = register(self.ref, background, meta.width) if background is not None else (None, 0)
        tm["registration"] = time.perf_counter() - t
        t = time.perf_counter()
        # another camera (or a view that does not register): no geometry or signals, auto carriageway
        sig = None
        if H is not None:
            try:
                sig = self._signals(H, scale, ref_s, rois, crops, frames, meta.fps)
            except Exception:  # no signal timeline: only red_light and stop_line are lost
                print(f"[westcv] signals failed: {traceback.format_exc()}", file=sys.stderr)
        tm["signals"] = time.perf_counter() - t
        t = time.perf_counter()
        tracks = track(frames, dets, meta.fps)
        tm["tracking"] = time.perf_counter() - t
        t = time.perf_counter()
        m = {"fps": meta.fps, "n_frames": meta.n_frames, "width": meta.width, "height": meta.height}
        sc = Scene(tracks, dets, m, H, sig, self.geometry if H is not None else None)
        tm["scene"] = time.perf_counter() - t
        t = time.perf_counter()
        events = detect(sc, classes)
        tm["rules"] = time.perf_counter() - t
        tm["total"] = time.perf_counter() - t0
        self.last_timing = {k: round(v, 1) for k, v in tm.items()} | {"inliers": inliers, "frames": len(frames), "braked": self.braked, "auto_scene": sc.auto,
                                                                     "x_duration": round(tm["total"] / meta.duration, 2)}
        return sanitize(events, classes, meta.duration)

    @staticmethod
    def _roi(box, scale):
        x1, y1, x2, y2 = (np.array(box) * scale).round().astype(int)
        return (max(0, x1 - HEAD_PAD), max(0, y1 - HEAD_PAD), x2 + HEAD_PAD, y2 + HEAD_PAD)

    def _signals(self, H, scale, ref_s, rois, crops, frames, fps) -> Signals:
        on = {}
        for name, cfg in self.heads_cfg.items():
            x1, y1, x2, y2 = cfg["box"]
            c = cv2.perspectiveTransform(np.float32([[x1, y1], [x2, y2]]).reshape(-1, 1, 2), H).reshape(-1, 2) * scale
            ox, oy = rois[name][:2]
            box = np.array([c[0, 0] - ox, c[0, 1] - oy, c[1, 0] - ox, c[1, 1] - oy])
            tpl = cv2.resize(self.templates[name].astype(np.float32), None, fx=ref_s, fy=ref_s, interpolation=cv2.INTER_AREA)
            head = Head(name, box, cfg["lamps"], tpl)
            scores = np.array([lamp_scores(cr, locate(head, cr, ref_s), head.lamps) for cr in crops[name]])
            on[name] = lamp_states(scores)
        return Signals(frames / fps, on["veh_median"], on["ped_left"])
