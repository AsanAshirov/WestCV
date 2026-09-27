# build_tracks.py — tracks from cached detections (CPU, seconds per video).
#
#   python tools/build_tracks.py cache/C3905_det.npz [...]   -> cache/C3905_tracks.npz
#
# tracks: rows (frame, track_id, x1, y1, x2, y2, conf, COCO class), original video pixels;
# track_id % 3 = group (0 person, 1 vehicle, 2 animal), see westcv.tracking.
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from westcv.tracking import track  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detections", nargs="+")
    args = ap.parse_args()
    for path in map(Path, args.detections):
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        t = time.perf_counter()
        tracks = track(z["frames"], z["dets"], meta["fps"])
        out = path.with_name(path.name.replace("_det.npz", "_tracks.npz"))
        np.savez_compressed(out, tracks=tracks, meta=json.dumps(meta))
        ids = np.unique(tracks[:, 1]).astype(int)
        print(f"{out.name}: {len(ids)} tracks (person {(ids % 3 == 0).sum()}, vehicle {(ids % 3 == 1).sum()}, "
              f"animal {(ids % 3 == 2).sum()}), {time.perf_counter() - t:.0f}s")


if __name__ == "__main__":
    main()
