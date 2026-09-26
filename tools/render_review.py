"""Review video: tracks, learned road mask, crossings, detected (and true) events.

    python tools/render_review.py DataSets/C3896.MP4 --cache cache --out review
    python tools/render_review.py DataSets/C3896.MP4 --cache cache --out review --gt dev_labels/dev_gt.json

Writes review/<video>_review.mp4 (H.264, analysis resolution and frame rate) and
<video>_events.csv. Use it to (1) check what the pipeline sees, (2) label the dev set
quickly: watch the video and write/correct rows in dev_labels/labels.csv. The same
video is what the site shows as the annotated sample.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficwatch import pipeline, viz  # noqa: E402
from trafficwatch.perception import Perception  # noqa: E402
from trafficwatch.video import iter_frames, read_meta  # noqa: E402


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

    size = (per.frame_w, per.frame_h)
    by_time = viz.tracks_by_time(res.tracks)
    overlay = viz.scene_overlay(res.maps, size)
    writer = viz.H264Writer(out_dir / f"{meta.video_id}_review.mp4", (size[0], size[1] + viz.BAR_H),
                            meta.fps / per.stride)
    source = iter_frames(meta, per.stride, per.frame_w, pipeline.CFG["decode"]["backend"])
    with contextlib.closing(source):
        for idx, frame in source:
            t = idx / meta.fps
            img = viz.draw_frame(frame, by_time.get(round(t, 3), []), res.events, t, overlay=overlay)
            writer.write(np.vstack([img, viz.timeline_bar(size[0], t, meta.duration, res.events, gt)]))
    writer.close()
    print(f"{meta.video_id}: {len(res.events)} events -> {out_dir}")


if __name__ == "__main__":
    main()
