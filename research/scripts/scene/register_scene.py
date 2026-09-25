"""Per-video scene registration prototype (src/scene/register.py in the repo).

Reference = median background plate of ONE sample video (the frame scene.yaml was drawn on),
stored in the repo as weights/scene/ref_plate_960.png + ref_mask_960.png.

H convention: H maps REFERENCE pixel coords -> CURRENT-video pixel coords at WORK scale (960x540).
At 4K: H4k = S @ H @ inv(S), S = diag(4, 4, 1).  Warp every scene element with H4k.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

WORK_W, WORK_H = 960, 540

# ---- quality gates (work-scale px unless stated) ---------------------------------------
IDENTITY_MAX_4K = 4.0       # max corner displacement below this -> keep identity (no warp)
PLAUSIBLE_MAX_4K = 240.0    # above this the camera was re-framed -> accept only with STRONG evidence
STRONG_INLIERS, STRONG_ECC = 150, 0.90
MIN_INLIERS = 40
MIN_INLIER_RATIO = 0.25
MAX_RMSE = 1.5              # RANSAC inlier reprojection RMSE, work px (= 6 px at 4K)
MIN_ECC = 0.60              # ECC correlation coefficient on masked, CLAHE-normalised plates
MAX_FEAT_ECC_DISAGREE_4K = 12.0
MAX_SCALE_DEV = 0.06        # |sqrt(det(A)) - 1|


@dataclass
class Registration:
    H: np.ndarray                      # ref -> cur, work scale
    status: str                        # identity | warped | fallback
    max_disp_4k: float
    ecc: float
    inliers: int
    rmse: float
    ms: float
    reasons: list = field(default_factory=list)

    def H4k(self, scale: float = 4.0) -> np.ndarray:
        S = np.diag([scale, scale, 1.0])
        return S @ self.H @ np.linalg.inv(S)

    def warp_points(self, pts_4k: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_4k, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H4k()).reshape(-1, 2)


def to_work(frame_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    return cv2.resize(g, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)


def median_plate(stack: list[np.ndarray]) -> np.ndarray:
    a = np.stack(stack, 0)
    return np.median(a, axis=0).astype(np.uint8)


def _normalise(img: np.ndarray) -> np.ndarray:
    # CLAHE removes most auto-exposure / white-balance differences before matching
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def _corner_disp_4k(H: np.ndarray, pts_work: np.ndarray) -> float:
    q = cv2.perspectiveTransform(pts_work.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.max(np.linalg.norm(q - pts_work, axis=1)) * (3840 / WORK_W))


_AKAZE = None


def _features_H(ref: np.ndarray, cur: np.ndarray, mask: np.ndarray):
    global _AKAZE
    if _AKAZE is None:
        _AKAZE = cv2.AKAZE_create(threshold=0.0005)
    k1, d1 = _AKAZE.detectAndCompute(ref, mask)
    k2, d2 = _AKAZE.detectAndCompute(cur, mask)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, 0, 0, 99.0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < 0.8 * b.distance]
    if len(good) < 8:
        return None, 0, len(good), 99.0
    p1 = np.float32([k1[g.queryIdx].pt for g in good])
    p2 = np.float32([k2[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0, maxIters=5000, confidence=0.999)
    if H is None:
        return None, 0, len(good), 99.0
    inl = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(p1[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((proj - p2[inl]) ** 2, 1)))) if inl.any() else 99.0
    return H, int(inl.sum()), len(good), rmse


def register_plate(ref_plate: np.ndarray, ref_mask: np.ndarray, cur_plate: np.ndarray,
                   key_pts_work: np.ndarray | None = None) -> Registration:
    """ref_plate/cur_plate: uint8 gray WORK_WxWORK_H. ref_mask: 255 = stable static structure."""
    t0 = time.perf_counter()
    reasons = []
    ref_n, cur_n = _normalise(ref_plate), _normalise(cur_plate)
    corners = np.float64([[0, 0], [WORK_W - 1, 0], [WORK_W - 1, WORK_H - 1], [0, WORK_H - 1]])
    pts = corners if key_pts_work is None else np.vstack([corners, key_pts_work])

    Hf, n_inl, n_good, rmse = _features_H(ref_n, cur_n, ref_mask)
    H0 = Hf if Hf is not None else np.eye(3)
    ecc, He = 0.0, None
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
        # ECC at half work scale first (speed), then full work scale
        S = np.diag([0.5, 0.5, 1.0])
        Hs = (S @ H0 @ np.linalg.inv(S)).astype(np.float32)
        r2, c2 = cv2.pyrDown(ref_n), cv2.pyrDown(cur_n)
        m2 = cv2.resize(ref_mask, (r2.shape[1], r2.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, Hs = cv2.findTransformECC(r2, c2, Hs, cv2.MOTION_HOMOGRAPHY, crit, m2, 5)
        He = (np.linalg.inv(S) @ Hs.astype(np.float64) @ S).astype(np.float32)
        ecc, He = cv2.findTransformECC(ref_n, cur_n, He, cv2.MOTION_HOMOGRAPHY,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5), ref_mask, 5)
        He = He.astype(np.float64)
        He /= He[2, 2]
    except cv2.error as e:  # ECC diverged
        reasons.append(f"ecc_fail:{str(e)[:40]}")

    H = He if He is not None else H0
    disp = _corner_disp_4k(H, pts)
    A = H[:2, :2]
    scale_dev = abs(np.sqrt(abs(np.linalg.det(A))) - 1.0)

    if Hf is None or n_inl < MIN_INLIERS:
        reasons.append(f"inliers={n_inl}<{MIN_INLIERS}")
    elif n_inl / max(n_good, 1) < MIN_INLIER_RATIO:
        reasons.append(f"inlier_ratio={n_inl / max(n_good, 1):.2f}")
    if rmse > MAX_RMSE:
        reasons.append(f"rmse={rmse:.2f}")
    if He is None or ecc < MIN_ECC:
        reasons.append(f"ecc={ecc:.3f}")
    if Hf is not None and He is not None:
        dis = _corner_disp_4k(np.linalg.inv(Hf) @ He, pts)
        if dis > MAX_FEAT_ECC_DISAGREE_4K:
            reasons.append(f"feat_vs_ecc={dis:.1f}px4k")
    if disp > PLAUSIBLE_MAX_4K and not (n_inl >= STRONG_INLIERS and ecc >= STRONG_ECC):
        reasons.append(f"disp={disp:.0f}px4k>plausible without strong evidence")
    if scale_dev > MAX_SCALE_DEV and not (n_inl >= STRONG_INLIERS and ecc >= STRONG_ECC):
        reasons.append(f"scale_dev={scale_dev:.3f} without strong evidence")

    ms = (time.perf_counter() - t0) * 1000
    if reasons:
        return Registration(np.eye(3), "fallback", disp, ecc, n_inl, rmse, ms, reasons)
    if disp < IDENTITY_MAX_4K:
        return Registration(np.eye(3), "identity", disp, ecc, n_inl, rmse, ms, reasons)
    return Registration(H, "warped", disp, ecc, n_inl, rmse, ms, reasons)


def jitter_track(plate: np.ndarray, frames_work: list[np.ndarray]) -> np.ndarray:
    """Per-sample translation (4K px) of each frame vs the video's own plate (phase correlation)."""
    win = cv2.createHanningWindow((WORK_W, WORK_H), cv2.CV_32F)
    p = plate.astype(np.float32)
    out = []
    for f in frames_work:
        (dx, dy), _ = cv2.phaseCorrelate(p, f.astype(np.float32), win)
        out.append((dx * 4, dy * 4))
    return np.array(out)


class CausalRegistrar:
    """For RiskEstimator.step: registers from frames already received only.

    t < t_init           : identity (risk features that depend on geometry are down-weighted)
    t = t_init (~3 s)    : median of n_init frames sampled every `every` frames -> register_plate
    every recheck_s      : phase-correlation drift check on one frame; if |shift| > 8 px(4K), re-register
    """

    def __init__(self, ref_plate, ref_mask, fps=29.97, every_s=0.4, n_init=8, recheck_s=30.0):
        self.ref_plate, self.ref_mask = ref_plate, ref_mask
        self.every = max(1, int(round(every_s * fps)))
        self.n_init, self.recheck = n_init, int(round(recheck_s * fps))
        self.buf, self.reg, self.i = [], None, 0
        self.cost_ms = 0.0

    def step(self, frame_bgr) -> Registration | None:
        i = self.i
        self.i += 1
        if self.reg is None and i % self.every == 0:
            t0 = time.perf_counter()
            self.buf.append(to_work(frame_bgr))
            if len(self.buf) >= self.n_init:
                self.reg = register_plate(self.ref_plate, self.ref_mask, median_plate(self.buf))
                self.buf = []
            self.cost_ms += (time.perf_counter() - t0) * 1000
        return self.reg


def find_step(ts: np.ndarray, j: np.ndarray, min_step_4k: float = 8.0, min_seg_s: float = 6.0):
    """Single change-point in the jitter track: least-squares piecewise-constant split (location),
    median difference of the two segments (robust size). Returns (t_split or None, size_4k)."""
    best_sse, k_best = np.inf, None
    for k in range(1, len(ts)):
        if ts[k] - ts[0] < min_seg_s or ts[-1] - ts[k] < min_seg_s:
            continue
        sse = ((j[:k] - j[:k].mean(0)) ** 2).sum() + ((j[k:] - j[k:].mean(0)) ** 2).sum()
        if sse < best_sse:
            best_sse, k_best = sse, k
    if k_best is None:
        return None, 0.0
    d = float(np.linalg.norm(np.median(j[:k_best], 0) - np.median(j[k_best:], 0)))
    return (float(ts[k_best]), d) if d >= min_step_4k else (None, d)
