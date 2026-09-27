"""Traffic-signal state from pixels.

Signal heads that face the camera are boxed once on the reference frame
(scene/signal_heads.json) and mapped into each video by registration. The camera shakes
for the first seconds of a recording, so every sample the head is re-located by matching
the housing's grey template (cut from the reference frame, scene/signal_templates.npz)
in a small window before the lamps are read.

A lamp's score is the mean of the brightest saturated pixels in its colour band, taken
from the central part of its cell (the head box also covers background such as the tail
lights of cars behind the pole). On/off thresholds are set per video and per lamp from
the score distribution: day and dusk differ by a factor of ~3 in brightness.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HUE = {"red": ((0, 12), (165, 180)), "yellow": ((13, 38),), "green": ((40, 100),)}
SEARCH = 40          # px at full resolution: shake at the start of a recording is ~25 px
TOP_K = 8
MARGIN = 16          # reference px of context around the head in the template
MIN_ON_DELTA = 25.0  # an 'on' lamp is at least this much brighter than 'off' (score units)
ABS_ON = 45.0        # fallback threshold for a lamp that does not switch within the clip. At the
                     # pipeline's 1280 px on daytime dev clips: 'on' p1 ~54-60, 'off' p99.9 <= ~33,
                     # per-video Otsu splits 35-45. Re-measure if the working width or TOP_K changes


@dataclass
class Head:
    name: str
    box: np.ndarray      # x1, y1, x2, y2 in this video's pixels at the working scale
    lamps: list[str]     # colours top to bottom
    template: np.ndarray | None = None


def _grey(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def make_templates(reference_bgr: np.ndarray, heads_path: str | Path, out: str | Path) -> None:
    """Grey crops of each head (+ MARGIN) from the reference frame, saved once."""
    cfg = json.loads(Path(heads_path).read_text(encoding="utf-8"))["heads"]
    crops = {}
    for name, h in cfg.items():
        x1, y1, x2, y2 = h["box"]
        crops[name] = _grey(reference_bgr[y1 - MARGIN:y2 + MARGIN, x1 - MARGIN:x2 + MARGIN]).astype(np.uint8)
    np.savez_compressed(out, **crops)


def load_heads(path: str | Path, templates: str | Path, H: np.ndarray, scale: float) -> list[Head]:
    """Heads mapped from reference pixels into the video (homography H), then scaled."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))["heads"]
    tpl = np.load(templates)
    heads = []
    for name, h in cfg.items():
        x1, y1, x2, y2 = h["box"]
        corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        c = cv2.perspectiveTransform(corners, H).reshape(-1, 2) * scale
        t = tpl[name].astype(np.float32)
        t = cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        heads.append(Head(name, np.array([*c.min(0), *c.max(0)]), h["lamps"], t))
    return heads


def locate(head: Head, frame: np.ndarray, scale: float) -> np.ndarray:
    """Box of the head in this frame: template search around the expected position."""
    x1, y1, x2, y2 = head.box.round().astype(int)
    m, r = int(round(MARGIN * scale)), int(SEARCH * scale)
    X1, Y1 = max(0, x1 - m - r), max(0, y1 - m - r)
    win = _grey(frame[Y1:y2 + m + r, X1:x2 + m + r])
    if win.shape[0] < head.template.shape[0] or win.shape[1] < head.template.shape[1]:
        return head.box
    res = cv2.matchTemplate(win, head.template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (dx, dy) = cv2.minMaxLoc(res)
    if score < 0.5:
        return head.box
    return np.array([X1 + dx + m, Y1 + dy + m, X1 + dx + m + (x2 - x1), Y1 + dy + m + (y2 - y1)], np.float64)


def lamp_scores(frame: np.ndarray, box: np.ndarray, lamps: list[str]) -> np.ndarray:
    x1, y1, x2, y2 = box.round().astype(int)
    w = x2 - x1
    crop = frame[y1:y2, x1 + w // 5:x2 - w // 5]
    if crop.size == 0:
        return np.zeros(len(lamps))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv.shape[0]
    out = []
    for i, colour in enumerate(lamps):
        cell = hsv[i * h // len(lamps):(i + 1) * h // len(lamps)].reshape(-1, 3).astype(np.float32)
        hue = cell[:, 0]
        band = np.zeros(len(cell), bool)
        for lo, hi in HUE[colour]:
            band |= (hue >= lo) & (hue <= hi)
        v = cell[band, 2] * cell[band, 1] / 255.0
        out.append(float(np.sort(v)[-TOP_K:].mean()) if len(v) >= TOP_K else 0.0)
    return np.array(out)


def otsu_threshold(x: np.ndarray) -> tuple[float, float]:
    """Threshold between 'off' and 'on' and the separation (difference of class means / spread)."""
    xs = np.sort(x)
    n = len(xs)
    if n < 10 or xs[-1] - xs[0] < 1e-6:
        return float("inf"), 0.0
    c = np.cumsum(xs)
    k = np.arange(1, n)
    m0, m1 = c[:-1] / k, (c[-1] - c[:-1]) / (n - k)
    between = k * (n - k) * (m1 - m0) ** 2
    i = int(np.argmax(between))
    if m1[i] - m0[i] < MIN_ON_DELTA:
        return float("inf"), 0.0
    return float((xs[i] + xs[i + 1]) / 2), float((m1[i] - m0[i]) / (xs.std() + 1e-6))


def lamp_states(scores: np.ndarray, min_separation: float = 1.5) -> np.ndarray:
    """(T, L) bool: lamp on. A lamp whose scores are not bimodal did not switch in this clip (a clip
    shorter than one red phase, ~40 s, can be all red): it is read with the fixed ABS_ON threshold."""
    on = np.zeros(scores.shape, bool)
    for j in range(scores.shape[1]):
        thr, sep = otsu_threshold(scores[:, j])
        on[:, j] = scores[:, j] > (thr if sep >= min_separation else ABS_ON)
    return on
