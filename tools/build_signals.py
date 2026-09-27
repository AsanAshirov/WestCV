# build_signals.py — per-video lamp states of the signal heads that face the camera.
#
#   python tools/build_signals.py C3905 [...]   -> cache/<stem>_signals.npz
#
# Reads the frame-exact 1080p proxy (every 3rd frame), registers the video to the reference
# frame, re-locates each head by template matching and scores its lamps (westcv.signals).
# npz: t (sec), for each head: <head>_scores (T, L), <head>_on (T, L) bool, and lamps order.
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from westcv.registration import Reference, register  # noqa: E402
from westcv.signals import lamp_scores, lamp_states, load_heads, locate  # noqa: E402

SCALE = 0.5  # proxy 1920 / original 3840


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    args = ap.parse_args()
    ref = Reference.load(ROOT / "scene/reference_sift.npz")
    for stem in args.stems:
        t0 = time.perf_counter()
        H, n = register(ref, cv2.imread(str(ROOT / f"DataSets/proxies/ref/{stem}_median.png")), 3840)
        heads = load_heads(ROOT / "scene/signal_heads.json", ROOT / "scene/signal_templates.npz", H, SCALE)
        cap = cv2.VideoCapture(str(ROOT / f"DataSets/proxies/{stem}_1080p.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        times, scores, i = [], {h.name: [] for h in heads}, 0
        while cap.grab():
            if i % 3 == 2:
                _, frame = cap.retrieve()
                times.append(i / fps)
                for h in heads:
                    scores[h.name].append(lamp_scores(frame, locate(h, frame, SCALE), h.lamps))
            i += 1
        out = {"t": np.array(times), "fps": fps, "registration_inliers": n}
        for h in heads:
            sc = np.array(scores[h.name])
            out[f"{h.name}_scores"], out[f"{h.name}_on"] = sc, lamp_states(sc)
            out[f"{h.name}_lamps"] = np.array(h.lamps)
        np.savez_compressed(ROOT / f"cache/{stem}_signals.npz", **out)
        print(f"{stem}: {len(times)} samples, inliers {n}, {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
