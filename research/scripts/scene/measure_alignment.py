"""EDA/registration audit for the sample videos (run once the .MP4 files are local).

    python measure_alignment.py --videos samples --ref C3896.MP4 --out reg_out [--lamp x0,y0,x1,y1 ...] [--mask mask_960.png]

Per video (one sequential decode pass; retrieve() only on sampled frames):
  * plate_<vid>.png           median of one frame every --plate-every s (960x540 gray)
  * luma_<vid>.csv            t, mean luma (full frame), p5/p95 luma, lamp ROI mean B,G,R,V per ROI (1 Hz)
  * jitter_<vid>.csv          t, dx, dy (4K px) phase-correlation shift of each 1 Hz sample vs the video's own plate
Summary (summary.json + stdout):
  * inter-video H (ref -> video), max displacement at 4K over image corners + scene key points, ECC, inliers, status
  * intra-video jitter: median / p95 / max |shift| (4K px), first-2 s and last-2 s max (REC-button nudge), step test
  * exposure: largest 1-s luma jump and slow drift (first vs last 10 s)
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

import register_scene as R


def scan(path: Path, plate_every: float, lamps: list[tuple[int, int, int, int]]):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_1hz = max(1, int(round(fps)))
    step_plate = max(1, int(round(plate_every * fps)))
    samples, plate_stack, luma_rows = [], [], []
    i = 0
    t0 = time.perf_counter()
    while True:
        need = (i % step_1hz == 0) or (i % step_plate == 0)
        if not cap.grab():
            break
        if need:
            ok, fr = cap.retrieve()
            if not ok:
                break
            g = R.to_work(fr)
            t = i / fps
            if i % step_plate == 0:
                plate_stack.append(g)
            if i % step_1hz == 0:
                samples.append((t, g))
                row = [round(t, 2), float(g.mean()), float(np.percentile(g, 5)), float(np.percentile(g, 95))]
                s = W / 3840.0  # lamp ROIs are given in 4K coords
                for (x0, y0, x1, y1) in lamps:
                    roi = fr[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)]
                    b, gg, r = roi.reshape(-1, 3).mean(0) if roi.size else (0, 0, 0)
                    v = float(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 2].max()) if roi.size else 0.0
                    row += [float(b), float(gg), float(r), v]
                luma_rows.append(row)
        i += 1
    cap.release()
    return dict(fps=fps, n=n, width=W, samples=samples, plate=R.median_plate(plate_stack),
                luma=luma_rows, decode_s=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--ref", default=None, help="file name of the reference video (scene.yaml drawn on it)")
    ap.add_argument("--out", default="reg_out")
    ap.add_argument("--plate-every", type=float, default=2.0)
    ap.add_argument("--lamp", action="append", default=[], help="x0,y0,x1,y1 in 4K px; repeatable")
    ap.add_argument("--mask", default=None, help="960x540 png, 255 = stable structure (default: all but top 20 pct)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lamps = [tuple(int(v) for v in s.split(",")) for s in a.lamp]
    vids = sorted(p for p in Path(a.videos).iterdir() if p.suffix.lower() == ".mp4")
    ref_name = a.ref or vids[0].name
    if a.mask:
        mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
    else:
        mask = np.full((R.WORK_H, R.WORK_W), 255, np.uint8)
        mask[: int(0.2 * R.WORK_H)] = 0          # sky / tree tops usually unstable
    scans = {}
    for p in vids:
        scans[p.name] = s = scan(p, a.plate_every, lamps)
        cv2.imwrite(str(out / f"plate_{p.stem}.png"), s["plate"])
        with open(out / f"luma_{p.stem}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "luma_mean", "luma_p5", "luma_p95"] + [f"lamp{k}_{c}" for k in range(len(lamps)) for c in "BGRV"])
            w.writerows(s["luma"])
        print(f"{p.name}: {s['n']} frames @ {s['fps']:.3f} fps, width {s['width']}, scan {s['decode_s']:.0f} s")

    ref_plate = scans[ref_name]["plate"]
    summary = {"reference": ref_name, "videos": {}}
    for name, s in scans.items():
        reg = R.register_plate(ref_plate, mask, s["plate"])
        j = R.jitter_track(s["plate"], [g for _, g in s["samples"]])
        mag = np.linalg.norm(j, axis=1)
        ts = np.array([t for t, _ in s["samples"]])
        with open(out / f"jitter_{Path(name).stem}.csv", "w", newline="") as f:
            csv.writer(f).writerows([["t", "dx_4k", "dy_4k"]] + [[round(t, 2), round(x, 2), round(y, 2)] for t, (x, y) in zip(ts, j)])
        third = max(1, len(j) // 3)
        step = float(np.linalg.norm(np.median(j[:third], 0) - np.median(j[-third:], 0)))
        lum = np.array([r[1] for r in s["luma"]])
        dl = np.abs(np.diff(lum)) / np.maximum(lum[:-1], 1) if len(lum) > 1 else np.zeros(1)
        summary["videos"][name] = dict(
            status=reg.status, max_disp_4k=round(reg.max_disp_4k, 1), ecc=round(reg.ecc, 3), inliers=reg.inliers,
            rmse_work=round(reg.rmse, 2), reasons=reg.reasons, H_ref_to_video_4k=np.round(reg.H4k(), 6).tolist(),
            jitter_median_4k=round(float(np.median(mag)), 2), jitter_p95_4k=round(float(np.percentile(mag, 95)), 2),
            jitter_max_4k=round(float(mag.max()), 2),
            jitter_first2s_max_4k=round(float(mag[ts < 2.0].max()), 2) if (ts < 2.0).any() else None,
            jitter_last2s_max_4k=round(float(mag[ts > ts.max() - 2.0].max()), 2),
            drift_first_vs_last_third_4k=round(step, 2),
            luma_max_1s_jump_pct=round(float(dl.max() * 100), 1), luma_first_vs_last10s_pct=round(
                float((lum[-10:].mean() - lum[:10].mean()) / max(lum[:10].mean(), 1) * 100), 1),
        )
        v = summary["videos"][name]
        print(f"{name:12s} {v['status']:8s} disp {v['max_disp_4k']:7.1f}px4k ecc {v['ecc']:.3f} inl {v['inliers']:4d} | "
              f"jitter med {v['jitter_median_4k']:.1f} p95 {v['jitter_p95_4k']:.1f} max {v['jitter_max_4k']:.1f} "
              f"(first2s {v['jitter_first2s_max_4k']}, last2s {v['jitter_last2s_max_4k']}) drift {v['drift_first_vs_last_third_4k']:.1f} | "
              f"luma jump {v['luma_max_1s_jump_pct']}%/s drift {v['luma_first_vs_last10s_pct']}% {';'.join(reg.reasons)}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
