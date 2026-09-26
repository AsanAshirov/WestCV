"""Measure the time budget on THIS machine for one video (first N seconds).

    python tools/bench_decode.py DataSets/C3905.MP4 --seconds 20

Prints, as fractions of video duration:
  h        - the harness's own Part B loop (cv2.read of every full frame): fixed cost;
  decode   - our Part A decode pass (per backend) at the configured analysis fps;
and the Part A budget the pipeline will give itself. Run it on the evaluation-like
machine (Kaggle T4) as well as on the laptop.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficwatch.config import load_config  # noqa: E402
from trafficwatch.video import analysis_size, ffmpeg_exe, iter_frames, read_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    cfg = load_config()
    meta = read_meta(args.video)
    n = min(meta.n_frames, int(args.seconds * meta.fps))
    print(f"{meta.video_id}: {meta.width}x{meta.height} @ {meta.fps:.3f} fps, {meta.duration:.1f}s; probing {n} frames")

    cap = cv2.VideoCapture(meta.path)
    t0 = time.perf_counter()
    for _ in range(n):
        if not cap.read()[0]:
            break
    h = (time.perf_counter() - t0) / n * meta.fps
    cap.release()
    print(f"  harness cv2.read h = {h:.2f}x duration")

    stride = max(1, int(round(meta.fps / cfg["decode"]["analysis_fps"])))
    w, _ = analysis_size(meta, cfg["decode"]["width"])
    backends = ["cv2"] + (["ffmpeg"] if ffmpeg_exe() else [])
    for backend in backends:
        t0 = time.perf_counter()
        source = iter_frames(meta, stride, w, backend, tuple(cfg["decode"].get("ffmpeg_input_args") or ()))
        with contextlib.closing(source):
            for idx, _ in source:
                if idx >= n:
                    break
        f = (time.perf_counter() - t0) / n * meta.fps
        print(f"  Part A decode [{backend}] stride {stride}, width {w}: {f:.2f}x duration")

    b = cfg["budget"]
    factor = 3.0 - b["part_b_factor"] * h - b["margin"]
    print(f"  => Part A budget {factor:.2f}x (clipped to [{b['min_factor']}, {b['max_factor']}]); "
          f"Part A must decode + detect within it")


if __name__ == "__main__":
    main()
