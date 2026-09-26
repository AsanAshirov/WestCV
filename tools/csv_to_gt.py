"""Dev labels written by hand (CSV) -> ground_truth.json in the organizers' format.

    python tools/csv_to_gt.py dev_labels/labels.csv --videos DataSets --out dev_labels/dev_gt.json

CSV columns: video,start,end,label[,note]. Times in seconds (62.4) or m:ss.s (1:02.4).
Every video in --videos gets an entry (a video with no rows = no events). fps and
duration are read exactly like the harness does.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trafficwatch.postprocess import sanitize  # noqa: E402
from trafficwatch.video import read_meta  # noqa: E402

sys.path.insert(0, str(ROOT))
from evaluate import OFFICIAL_CLASSES  # noqa: E402


def parse_time(text: str) -> float:
    parts = text.strip().split(":")
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + float(p)
    return seconds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gt = {}
    for path in sorted(p for p in Path(args.videos).iterdir() if p.suffix in (".mp4", ".MP4")):
        meta = read_meta(path)
        gt[path.name] = {"duration": meta.duration, "fps": round(meta.fps, 2), "events": []}
    with open(args.csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vid, label = row["video"].strip(), row["label"].strip()
            if vid not in gt:
                sys.exit(f"unknown video {vid!r} (not in {args.videos})")
            if label not in OFFICIAL_CLASSES:
                sys.exit(f"unknown label {label!r} in row {row}")
            gt[vid]["events"].append([parse_time(row["start"]), parse_time(row["end"]), label])
    for entry in gt.values():
        entry["events"] = sanitize(entry["events"], entry["duration"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(gt, indent=1))
    print({k: len(v["events"]) for k, v in gt.items()}, "->", args.out)


if __name__ == "__main__":
    main()
