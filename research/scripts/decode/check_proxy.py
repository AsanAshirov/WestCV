#!/usr/bin/env python3
"""check_proxy.py - prove that a proxy transcode is frame-exact w.r.t. its original.

Usage:
    python check_proxy.py ORIGINAL.MP4 PROXY.mp4 [--thumb 64x36] [--seeks 20] [--max-frames N] [--json out.json]

Checks (exit code 0 only if all hard checks pass):
  1. ffprobe (if on PATH): packet count (no decode), r_frame_rate / avg_frame_rate == 30000/1001,
     CFR timestamps (every pts delta == one frame), identical first pts.
  2. cv2: CAP_PROP_FRAME_COUNT equal, CAP_PROP_FPS equal (the harness uses t = idx / CAP_PROP_FPS and
     duration = CAP_PROP_FRAME_COUNT / CAP_PROP_FPS, so both must match for labels to transfer).
  3. Full decode of both files: decoded frame counts equal each other and CAP_PROP_FRAME_COUNT.
  4. Per-frame alignment: grayscale thumbnails (INTER_AREA). For every frame i with motion, the
     proxy frame i must be closer to original frame i than proxy frames i-1 and i+1 are, and the
     best global lag in [-15, 15] must be 0.
  5. Random-access seek (CAP_PROP_POS_FRAMES) on the proxy returns the same frame as sequential decode.
     (Reported for the original too, as information for Part A random access.)
Labels made on the proxy as frame_idx / fps seconds then map 1:1 onto the original.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import cv2
import numpy as np

NTSC30 = Fraction(30000, 1001)


def ffprobe_info(path: str) -> dict | None:
    if not shutil.which("ffprobe"):
        return None
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,"
         "time_base,nb_frames,nb_read_packets,start_time,duration",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    pts = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts",
         "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.split()
    pts = np.sort(np.array([int(p) for p in pts if p.strip().lstrip("-").isdigit()], dtype=np.int64))
    tb = Fraction(st["time_base"])
    d = np.diff(pts)
    frame_ticks = (1 / (Fraction(st["avg_frame_rate"]) * tb)) if st["avg_frame_rate"] != "0/0" else None
    st["pts_first_sec"] = float(pts[0] * tb) if len(pts) else None
    st["pts_delta_unique"] = sorted(set(int(x) for x in d))[:5]
    st["cfr"] = bool(len(d) and frame_ticks is not None and frame_ticks.denominator == 1
                     and np.all(d == int(frame_ticks)))
    return st


def thumbs(path: str, size: tuple[int, int], max_frames: int | None):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {path}")
    meta = {"cap_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "cap_fps": cap.get(cv2.CAP_PROP_FPS),
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    th, col, msec, t0 = [], [], [], time.perf_counter()
    while max_frames is None or len(th) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        msec.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        th.append(cv2.resize(g, size, interpolation=cv2.INTER_AREA))
        col.append(cv2.resize(fr, (16, 9), interpolation=cv2.INTER_AREA))  # colour check
    cap.release()
    meta["decoded"] = len(th)
    meta["decode_fps"] = round(len(th) / max(1e-9, time.perf_counter() - t0), 1)
    meta["colour_thumbs"] = np.stack(col).astype(np.float32)
    return np.stack(th).astype(np.float32), np.array(msec), meta


def seek_test(path: str, ref: np.ndarray, size, n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(ref), size=min(n, len(ref)), replace=False).tolist())
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    bad = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            bad.append((i, "read failed"))
            continue
        t = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32)
        dists = np.abs(ref - t).mean(axis=(1, 2))
        j = int(dists.argmin())
        if j != i and dists[j] < dists[i] - 1e-3:
            bad.append((i, f"got frame {j}"))
    cap.release()
    return {"tested": len(idx), "wrong": len(bad), "examples": bad[:5]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("proxy")
    ap.add_argument("--thumb", default="64x36")
    ap.add_argument("--seeks", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--abs-tol", type=float, default=3.0, help="max mean |diff| (gray levels) for frame i vs i")
    ap.add_argument("--colour-tol", type=float, default=3.0, help="max mean |diff| per B,G,R channel over all frames")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    size = tuple(int(x) for x in a.thumb.split("x"))
    rep, ok = {"original": a.original, "proxy": a.proxy}, True

    def fail(msg):
        nonlocal ok
        ok = False
        print("FAIL:", msg)

    # 1. ffprobe
    fo, fp = ffprobe_info(a.original), ffprobe_info(a.proxy)
    rep["ffprobe"] = {"original": fo, "proxy": fp}
    if fo and fp:
        for k in ("nb_read_packets",):
            if fo[k] != fp[k]:
                fail(f"packet count differs: {fo[k]} vs {fp[k]}")
        for s in (fo, fp):
            if Fraction(s["r_frame_rate"]) != NTSC30 or Fraction(s["avg_frame_rate"]) != NTSC30:
                fail(f"frame rate is {s['r_frame_rate']} / {s['avg_frame_rate']}, expected 30000/1001")
        if not fp["cfr"]:
            fail(f"proxy timestamps are not CFR: deltas {fp['pts_delta_unique']}")
        if abs((fo["pts_first_sec"] or 0) - (fp["pts_first_sec"] or 0)) > 0.0005:
            print(f"WARN: first pts differs {fo['pts_first_sec']} vs {fp['pts_first_sec']} (harness ignores pts)")
    else:
        print("WARN: ffprobe not found; skipping container checks")

    # 2-3. cv2 metadata + full decode (both files in parallel; cv2 releases the GIL)
    with ThreadPoolExecutor(2) as ex:
        jo = ex.submit(thumbs, a.original, size, a.max_frames)
        jp = ex.submit(thumbs, a.proxy, size, a.max_frames)
        To, mo, meta_o = jo.result()
        Tp, mp, meta_p = jp.result()
    Co, Cp = meta_o.pop("colour_thumbs"), meta_p.pop("colour_thumbs")
    rep["cv2"] = {"original": meta_o, "proxy": meta_p}
    if meta_o["cap_frame_count"] != meta_p["cap_frame_count"]:
        fail(f"CAP_PROP_FRAME_COUNT {meta_o['cap_frame_count']} vs {meta_p['cap_frame_count']}")
    if abs(meta_o["cap_fps"] - meta_p["cap_fps"]) > 1e-6:
        fail(f"CAP_PROP_FPS {meta_o['cap_fps']} vs {meta_p['cap_fps']}")
    if abs(meta_p["cap_fps"] - float(NTSC30)) > 1e-4:
        fail(f"proxy CAP_PROP_FPS {meta_p['cap_fps']} != 29.97003")
    if meta_o["decoded"] != meta_p["decoded"]:
        fail(f"decoded frames {meta_o['decoded']} vs {meta_p['decoded']}")
    if a.max_frames is None and meta_p["decoded"] != meta_p["cap_frame_count"]:
        fail(f"proxy decoded {meta_p['decoded']} != CAP_PROP_FRAME_COUNT {meta_p['cap_frame_count']}")
    n = min(len(To), len(Tp))
    To, Tp = To[:n], Tp[:n]
    fps = meta_p["cap_fps"]
    harness_t = np.arange(n) / fps * 1000.0
    rep["pos_msec_minus_idx_over_fps_ms"] = {
        "original_max_abs": float(np.abs(mo[:n] - harness_t).max()) if n else None,
        "proxy_max_abs": float(np.abs(mp[:n] - harness_t).max()) if n else None}

    # 3b. colour: catches wrong YUV->RGB matrix / range (e.g. forcing bt709 tags on an untagged source)
    m = min(len(Co), len(Cp))
    ch = np.abs(Co[:m] - Cp[:m]).mean(axis=(0, 1, 2))
    rep["colour_mean_abs_diff_BGR"] = [round(float(x), 2) for x in ch]
    if ch.max() > a.colour_tol:
        fail(f"colour differs: mean |dB,dG,dR| = {rep['colour_mean_abs_diff_BGR']} > {a.colour_tol} "
             "(check color_space/color_primaries/color_trc/color_range tags with ffprobe)")

    # 4. per-frame alignment
    d0 = np.abs(To - Tp).mean(axis=(1, 2))
    lags = {}
    for L in range(-15, 16):
        a0, b0 = max(0, L), max(0, -L)
        m = n - abs(L)
        lags[L] = float(np.abs(To[b0:b0 + m] - Tp[a0:a0 + m]).mean())
    best_lag = min(lags, key=lags.get)
    motion = np.full(n, np.nan)
    motion[1:] = np.abs(np.diff(To, axis=0)).mean(axis=(1, 2))
    dm = np.full(n, np.inf)
    dp = np.full(n, np.inf)
    dm[1:] = np.abs(To[1:] - Tp[:-1]).mean(axis=(1, 2))
    dp[:-1] = np.abs(To[:-1] - Tp[1:]).mean(axis=(1, 2))
    informative = (np.nan_to_num(motion, nan=0) > 0.5)  # frames where the scene changed measurably
    misaligned = informative & ~((d0 < dm) & (d0 < dp))
    rep["alignment"] = {"frames": int(n), "best_global_lag": best_lag, "lag_mad": lags,
                        "mean_abs_diff_same_index": float(d0.mean()), "max_abs_diff_same_index": float(d0.max()),
                        "informative_frames": int(informative.sum()), "misaligned_frames": int(misaligned.sum()),
                        "first_misaligned": np.flatnonzero(misaligned)[:10].tolist()}
    if best_lag != 0:
        fail(f"global lag {best_lag} frames (proxy frame i == original frame i{-best_lag:+d})")
    if misaligned.any():
        fail(f"{int(misaligned.sum())} informative frames closer to a neighbour than to the same index")
    if d0.max() > a.abs_tol:
        fail(f"max per-frame thumbnail difference {d0.max():.2f} > {a.abs_tol}")

    # 5. seek accuracy
    if a.seeks:
        rep["seek_proxy"] = seek_test(a.proxy, Tp, size, a.seeks)
        rep["seek_original_info"] = seek_test(a.original, To, size, a.seeks)
        if rep["seek_proxy"]["wrong"]:
            fail(f"proxy CAP_PROP_POS_FRAMES seeks wrong: {rep['seek_proxy']['examples']}")

    print(json.dumps({k: v for k, v in rep.items() if k != "ffprobe"}, indent=1, default=str))
    if fo and fp:
        print("ffprobe original:", {k: fo[k] for k in ("codec_name", "profile", "pix_fmt", "r_frame_rate", "nb_read_packets", "cfr")})
        print("ffprobe proxy   :", {k: fp[k] for k in ("codec_name", "profile", "pix_fmt", "r_frame_rate", "nb_read_packets", "cfr")})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1, default=str)
    print("RESULT:", "PASS - labels as idx/fps transfer 1:1" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
