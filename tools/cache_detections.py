# cache_detections.py — run the detector once over the videos, save raw detections.
# Rules and the tracker are then iterated on the cache in seconds, without a GPU.
#
#   python tools/cache_detections.py DataSets/C3905.MP4 [...] --out cache
#
# cache/<stem>_det.npz:
#   frames  (M,)   processed frame indices (about every 3rd, see westcv.video.iter_frames)
#   dets    (K, 7) frame index, x1, y1, x2, y2 (ORIGINAL video pixels), confidence, COCO class
#   meta    json:  fps, n_frames, width, height, weights, imgsz, conf, decode/detect seconds
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from westcv.detector import Detector  # noqa: E402
from westcv.video import iter_frames, prefetch, probe  # noqa: E402


def cache_video(path: Path, det: Detector, out: Path, width: int, batch: int, tag: str = "") -> None:
    meta = probe(path)
    scale = meta.width / width
    frames, rows, buf = [], [], []
    t0 = time.perf_counter()
    t_det = 0.0

    def flush():
        nonlocal t_det
        t = time.perf_counter()
        for (idx, _), d in zip(buf, det([f for _, f in buf])):
            d[:, :4] *= scale
            rows.append(np.concatenate([np.full((len(d), 1), idx, np.float32), d], axis=1))
            frames.append(idx)
        t_det += time.perf_counter() - t
        buf.clear()

    for item in prefetch(iter_frames(path, width=width)):
        buf.append(item)
        if len(buf) == batch:
            flush()
        if not buf and len(frames) % 600 == 0:
            print(f"  {path.name}: frame {frames[-1]}/{meta.n_frames}, {time.perf_counter() - t0:.0f}s", flush=True)
    if buf:
        flush()
    total = time.perf_counter() - t0
    info = {**meta.__dict__, "duration": meta.duration, "weights": det.model.ckpt_path, "imgsz": det.imgsz,
            "conf": det.conf, "decode_width": width, "quantize": det.quantize,
            "wall_sec": round(total, 1), "detect_sec": round(t_det, 1)}
    np.savez_compressed(out / f"{path.stem}{tag}_det.npz", frames=np.array(frames, np.int32),
                        dets=np.concatenate(rows) if rows else np.zeros((0, 7), np.float32),
                        meta=json.dumps(info))
    print(f"{path.name}: {len(frames)} frames, {sum(len(r) for r in rows)} detections, "
          f"{total:.0f}s wall ({total / meta.duration:.2f}x video), detector {t_det:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--weights", default="weights/yolo26m.pt")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--width", type=int, default=1920, help="decode width")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--tag", default="", help="suffix of the cache file, e.g. _w1280")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det = Detector(args.weights, imgsz=args.imgsz)
    det.warmup()
    for v in map(Path, args.videos):
        cache_video(v, det, out, args.width, args.batch, args.tag)


if __name__ == "__main__":
    main()
