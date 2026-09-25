"""Review video: tracks, learned road mask, crossings, detected (and true) events.

    python tools/render_review.py DataSets/C3896.MP4 --cache cache --out review
    python tools/render_review.py DataSets/C3896.MP4 --cache cache --out review --gt dev_labels/dev_gt.json

Writes review/<video>_review.mp4 (analysis resolution, 10 fps) and <video>_events.csv.
Use it to (1) check what the pipeline sees, (2) label the dev set quickly: watch the
review video and write/correct rows in dev_labels/labels.csv.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficwatch import pipeline  # noqa: E402
from trafficwatch.perception import Perception  # noqa: E402
from trafficwatch.video import iter_frames, read_meta  # noqa: E402

COLORS = [(60, 180, 255), (255, 160, 60), (80, 220, 80), (200, 80, 220)]  # BGR per superclass
BAR_H = 70


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="review")
    ap.add_argument("--gt", default=None)
    args = ap.parse_args()
    meta = read_meta(args.video)
    per = Perception.load(Path(args.cache) / f"{meta.video_id}.npz")
    res = pipeline.analyze(per)
    gt = json.loads(Path(args.gt).read_text())[meta.video_id]["events"] if args.gt else []
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{meta.video_id}_events.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "start", "end", "label"])
        for s, e, label in res.events:
            w.writerow([meta.video_id, s, e, label])

    by_time: dict[float, list] = {}
    for tr in res.tracks:
        for i, t in enumerate(tr.t):
            by_time.setdefault(round(float(t), 3), []).append((tr, i))
    W, H = per.frame_w, per.frame_h
    overlay = np.zeros((H, W, 3), np.uint8)
    road = cv2.resize(res.maps.road.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    cross = cv2.resize(res.maps.crosswalk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay[road] = (0, 90, 0)
    overlay[cross] = (120, 120, 0)
    labels = sorted({e[2] for e in res.events} | {e[2] for e in gt})
    writer = cv2.VideoWriter(str(out_dir / f"{meta.video_id}_review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             meta.fps / per.stride, (W, H + BAR_H))
    source = iter_frames(meta, per.stride, W, pipeline.CFG["decode"]["backend"])
    with contextlib.closing(source):
        for idx, frame in source:
            t = idx / meta.fps
            img = cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)
            for tr, i in by_time.get(round(t, 3), []):
                x1, y1, x2, y2 = tr.box[i].astype(int)
                color = COLORS[tr.sc]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{tr.tid} {tr.speed[i]:.1f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            active = [lab for s, e, lab in res.events if s <= t < e]
            cv2.putText(img, f"t={t:7.2f}s  {' '.join(active)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if active else (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(np.vstack([img, _timeline(W, t, meta.duration, labels, res.events, gt)]))
    writer.release()
    print(f"{meta.video_id}: {len(res.events)} events -> {out_dir}")


def _timeline(width: int, t: float, duration: float, labels: list[str], pred: list, gt: list) -> np.ndarray:
    bar = np.full((BAR_H, width, 3), 30, np.uint8)
    rows = max(len(labels), 1)
    rh = (BAR_H - 6) // rows
    for r, label in enumerate(labels):
        y = 3 + r * rh
        cv2.putText(bar, label[:12], (2, y + rh - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        for events, color, fill in ((gt, (0, 200, 255), 1), (pred, (80, 80, 255), -1)):
            for s, e, lab in events:
                if lab == label:
                    x0 = 100 + int(s / duration * (width - 100))
                    x1 = 100 + int(e / duration * (width - 100))
                    cv2.rectangle(bar, (x0, y + 1), (max(x1, x0 + 1), y + rh - 1), color, fill)
    x = 100 + int(t / duration * (width - 100))
    cv2.line(bar, (x, 0), (x, BAR_H), (255, 255, 255), 1)
    return bar


if __name__ == "__main__":
    main()
