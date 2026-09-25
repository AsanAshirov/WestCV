"""Run decode + detection once per video and cache it (no time budget).

    python tools/cache_perception.py --videos DataSets --out cache

Afterwards tools/run_rules.py re-runs tracking and rules from the cache in seconds.
The cache key is the video name plus the detector settings; delete the cache after
changing decode/detector settings.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficwatch import pipeline  # noqa: E402
from trafficwatch.perception import perceive  # noqa: E402
from trafficwatch.video import read_meta  # noqa: E402

VIDEO_EXTS = {".mp4", ".MP4"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder with videos or one video")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    src, out = Path(args.videos), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix in VIDEO_EXTS)
    detector = pipeline.get_detector()
    for path in videos:
        target = out / f"{path.name}.npz"
        if target.exists() and not args.force:
            print(f"skip {path.name} (cached)")
            continue
        meta = read_meta(path)
        t0 = time.perf_counter()
        per = perceive(meta, detector, pipeline.CFG,
                       on_progress=lambda f: print(f"\r  {path.name}: {f:5.1%}", end="", flush=True))
        per.save(target)
        took = time.perf_counter() - t0
        print(f"\r{path.name}: {len(per.times)} frames, {len(per.dets)} detections, "
              f"{took:.0f}s = {took / meta.duration:.2f}x duration -> {target}")


if __name__ == "__main__":
    main()
