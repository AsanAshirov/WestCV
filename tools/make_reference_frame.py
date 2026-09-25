"""Median frame of a video (no moving traffic) for drawing the scene in labelme.

    python tools/make_reference_frame.py DataSets/C3896.MP4 --out geometry/ref_C3896.png

Then:
    labelme geometry/ref_C3896.png --labels configs/scene_labels.txt --validate-label exact --output geometry/
    python tools/annotation/labelme_to_scene.py geometry/ref_C3896.json configs/scene.json
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficwatch.video import iter_frames, read_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-s", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--max-frames", type=int, default=90)
    args = ap.parse_args()
    meta = read_meta(args.video)
    stride = max(1, int(round(args.every_s * meta.fps)))
    frames = []
    source = iter_frames(meta, stride, args.width)
    with contextlib.closing(source):
        for _, frame in source:
            frames.append(frame)
            if len(frames) >= args.max_frames:
                break
    median = np.median(np.stack(frames), axis=0).astype(np.uint8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, median)
    print(f"{len(frames)} frames -> {args.out} ({median.shape[1]}x{median.shape[0]})")


if __name__ == "__main__":
    main()
