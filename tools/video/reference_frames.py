# reference_frames.py — full-resolution reference images for drawing the scene geometry.
#
#   python tools/video/reference_frames.py DataSets/C3896.MP4 DataSets/C3905.MP4 --out DataSets/proxies/ref
#
# For each video writes <stem>_median.png (per-pixel temporal median of --n evenly spaced
# frames: moving traffic disappears, road markings stay) and <stem>_f<idx>.png (one real
# frame, to check lights and signs). Frames are decoded from the ORIGINAL with OpenCV, the
# same way the harness decodes them.
import argparse
from pathlib import Path

import cv2
import numpy as np


def sample_frames(path: Path, n: int) -> tuple[list[np.ndarray], list[int]]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, n + 2, dtype=int)[1:-1].tolist()
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames, idxs


def strip_median(frames: list[np.ndarray], strip: int = 270) -> np.ndarray:
    h = frames[0].shape[0]
    out = np.empty_like(frames[0])
    for y in range(0, h, strip):  # strips keep peak memory at a fraction of the full stack
        out[y:y + strip] = np.median(np.stack([f[y:y + strip] for f in frames]), axis=0).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=25, help="frames in the median")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in map(Path, args.videos):
        frames, idxs = sample_frames(v, args.n)
        cv2.imwrite(str(out / f"{v.stem}_median.png"), strip_median(frames))
        mid = len(frames) // 2
        cv2.imwrite(str(out / f"{v.stem}_f{idxs[mid]}.png"), frames[mid])
        print(f"{v.name}: median of {len(frames)} frames, sample frame {idxs[mid]}")


if __name__ == "__main__":
    main()
