"""Per-video scene registration.

All scene geometry (lanes, stop lines, crossings, signal heads) is drawn once on the
reference frame (median of sample C3896). The camera was re-aimed between recordings:
by up to ~100 px at 4K and ~1 degree even within one session, so every video is mapped
onto the reference with a homography: SIFT on CLAHE-equalised grey images, ratio test,
RANSAC. Works across day and evening lighting (hundreds of inliers).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WORK_WIDTH = 1920       # features are computed at this width, coordinates stored in full-res pixels
MIN_INLIERS = 60


@dataclass
class Reference:
    points: np.ndarray       # (N, 2) float32, reference full-resolution pixels
    descriptors: np.ndarray  # (N, 128) float32
    size: tuple[int, int]    # (width, height) of the reference frame

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, points=self.points, descriptors=self.descriptors.astype(np.uint8),
                            size=np.array(self.size))

    @classmethod
    def load(cls, path: str | Path) -> "Reference":
        z = np.load(path)
        return cls(z["points"], z["descriptors"].astype(np.float32), tuple(int(v) for v in z["size"]))


def _features(image_bgr: np.ndarray, full_width: int, n: int = 8000):
    s = WORK_WIDTH / image_bgr.shape[1]
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if s != 1:
        grey = cv2.resize(grey, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    grey = cv2.createCLAHE(3.0, (8, 8)).apply(grey)
    kp, desc = cv2.SIFT_create(nfeatures=n).detectAndCompute(grey, None)
    to_full = full_width / grey.shape[1]
    return np.float32([k.pt for k in kp]) * to_full, desc


def build_reference(image_bgr: np.ndarray) -> Reference:
    h, w = image_bgr.shape[:2]
    pts, desc = _features(image_bgr, w)
    return Reference(pts, desc, (w, h))


def register(ref: Reference, image_bgr: np.ndarray, full_width: int) -> tuple[np.ndarray | None, int]:
    """Homography mapping reference pixels to this video's full-resolution pixels, and inlier count.

    `image_bgr` may be downscaled; `full_width` is the video's real width. Returns (None, n)
    when registration is not trustworthy, so fragile classes can be switched off.
    """
    pts, desc = _features(image_bgr, full_width)
    if desc is None or len(desc) < MIN_INLIERS:
        return None, 0
    pairs = cv2.BFMatcher().knnMatch(ref.descriptors, desc, k=2)
    good = [m for m, n2 in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n2.distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    src = ref.points[[m.queryIdx for m in good]]
    dst = pts[[m.trainIdx for m in good]]
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0 * full_width / WORK_WIDTH)
    n = int(inliers.sum()) if inliers is not None else 0
    return (H, n) if H is not None and n >= MIN_INLIERS else (None, n)


def warp_points(H: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
