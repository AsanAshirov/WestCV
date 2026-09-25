"""Tracking + rules on cached perceptions -> predictions json (+ official score).

    python tools/run_rules.py --cache cache --out dev_pred.json
    python tools/run_rules.py --cache cache --out dev_pred.json --gt dev_labels/dev_gt.json
    python tools/run_rules.py --cache cache --config configs/my_try.yaml --gt dev_labels/dev_gt.json

Edit thresholds in configs/pipeline.yaml (or a copy passed with --config), re-run,
compare F1 per class and per tIoU in the evaluate.py output.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="dev_pred.json")
    ap.add_argument("--config", default=None, help="alternative pipeline.yaml")
    ap.add_argument("--gt", default=None, help="ground truth json -> run evaluate.py")
    args = ap.parse_args()
    if args.config:
        import os
        os.environ["TW_CONFIG"] = str(Path(args.config).resolve())
    from trafficwatch import pipeline
    from trafficwatch.perception import Perception

    videos = {}
    for npz in sorted(Path(args.cache).glob("*.npz")):
        per = Perception.load(npz)
        res = pipeline.analyze(per)
        videos[per.video_id] = {"events": res.events, "risk": []}
        counts = {}
        for _, _, label in res.events:
            counts[label] = counts.get(label, 0) + 1
        print(f"{per.video_id}: {len(res.tracks)} tracks, events {counts or '{}'}")
    Path(args.out).write_text(json.dumps({"team": "Antigradient", "videos": videos}, indent=1))
    print(f"wrote {args.out}")
    if args.gt:
        subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--pred", args.out, "--gt", args.gt,
                        "--per-video"], check=False)


if __name__ == "__main__":
    main()
