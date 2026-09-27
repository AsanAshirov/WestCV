# render_review.py — review video: what the model sees and decides, next to the team's labels.
#
#   python tools/render_review.py C3905 [--tag _w1280] [--start 0 --end 60] [--out renders/review_C3905.mp4]
#
# Top: the 720p proxy with scene geometry (zebras white, islands pink, sidewalks green, stop line 4
# red, solid line yellow), people (orange; red = on the carriageway outside a zebra), vehicles (blue),
# the state of signal heads 7 and 8, and which classes the model / the labels have active now.
# Bottom: timeline per class — labels (upper bar, green) vs model (lower bar, orange), with a cursor.
# Written at 10 fps (the tracker's frames), about 1 min of encoding per minute of video.
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from westcv import rules  # noqa: E402
from westcv.scene import Scene, feet, inside  # noqa: E402

W, H = 1280, 720
PANEL_ROW = 18
CLASS_ORDER = ["jaywalking", "failure_to_yield", "stop_line", "red_light", "stopped_vehicle", "congestion",
               "solid_line_crossing", "illegal_turn", "wrong_way", "near_miss", "accident"]
SHORT = {"jaywalking": "jaywalking", "failure_to_yield": "failure_yield", "stop_line": "stop_line",
         "red_light": "red_light", "stopped_vehicle": "stopped_veh", "congestion": "congestion",
         "solid_line_crossing": "solid_line", "illegal_turn": "illegal_turn", "wrong_way": "wrong_way",
         "near_miss": "near_miss", "accident": "accident"}


def load_gt(stem):
    src = json.loads((ROOT / "labels/dev/sources.json").read_text(encoding="utf-8"))
    if stem not in src:
        return {}, set()
    s = src[stem]
    ev = next(iter(json.loads((ROOT / s["file"]).read_text(encoding="utf-8")).values()))["events"]
    covered = set(CLASS_ORDER) if s["classes"] == "all" else set(s["classes"])
    gt = {}
    for a, b, c in ev:
        gt.setdefault(c, []).append((a, b))
    return gt, covered


def person_flags(sc):
    """(frame, track) -> True when the person stands on the carriageway outside every zebra."""
    out = {}
    for tid, tr in sc.tracks.items():
        if tid % 3 != 0 or tid in sc.riders:
            continue
        p = feet(tr)
        h = (tr[:, 5] - tr[:, 3]).clip(1)
        depth = sc.road_depths(p)
        for r, q, hh, d in zip(tr, p, h, depth):
            if (int(r[0]), tid) in sc.occupant:
                continue
            out[(int(r[0]), tid)] = d > 0 and not any(inside(z, q, 0.2 * hh) for z in sc.crosswalks.values())
    return out


def timeline(classes, gt, pred, covered, dur, width):
    img = np.full((PANEL_ROW * len(classes) + 26, width, 3), 28, np.uint8)
    x0 = 110
    scale = (width - x0 - 10) / dur
    for i, c in enumerate(classes):
        y = 20 + i * PANEL_ROW
        cv2.putText(img, SHORT[c], (4, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y), (width - 10, y + PANEL_ROW - 3), (50, 50, 50), -1)
        if c not in covered:
            cv2.putText(img, "not labelled", (x0 + 4, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        for a, b in gt.get(c, []):
            cv2.rectangle(img, (int(x0 + a * scale), y + 1), (max(int(x0 + b * scale), int(x0 + a * scale) + 1), y + 7), (80, 200, 80), -1)
        for a, b in pred.get(c, []):
            cv2.rectangle(img, (int(x0 + a * scale), y + 8), (max(int(x0 + b * scale), int(x0 + a * scale) + 1), y + 14), (0, 150, 255), -1)
    for s in range(0, int(dur) + 1, 30):
        x = int(x0 + s * scale)
        cv2.line(img, (x, 14), (x, img.shape[0] - 4), (70, 70, 70), 1)
        cv2.putText(img, f"{s}s", (x + 2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1)
    return img, x0, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--tag", default="_w1280")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--out")
    args = ap.parse_args()
    sc = Scene.from_cache(args.stem, ROOT, args.tag)
    dur = sc.meta["n_frames"] / sc.fps
    pred = {c: f(sc) for c, f in rules.RULES.items()}
    gt, covered = load_gt(args.stem)
    classes = [c for c in CLASS_ORDER if c in pred or c in gt]
    panel, x0, tscale = timeline(classes, gt, pred, covered, dur, W)
    flags = person_flags(sc)
    s = W / sc.meta["width"]
    by_frame = {}
    for tid, tr in sc.tracks.items():
        for r in tr:
            by_frame.setdefault(int(r[0]), []).append((tid, r))
    out = Path(args.out or ROOT / f"renders/review_{args.stem}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                            f"{W}x{H + panel.shape[0]}", "-r", f"{sc.fps / 3:.4f}", "-i", "-", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "veryfast", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(ROOT / f"DataSets/proxies/{args.stem}_720p.mp4"))
    f0 = int(args.start * sc.fps)
    f1 = int((args.end or dur) * sc.fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    geo = [(z, (255, 255, 255), True) for z in sc.crosswalks.values()] + [(z, (200, 90, 255), True) for z in sc.islands] + \
          [(z, (0, 170, 0), True) for z in sc.sidewalks] + [(sc.stop4, (40, 40, 255), False)] + \
          [(z, (0, 220, 255), False) for z in sc.solid_lines.values()]
    n = 0
    for f in range(f0, f1):
        ok, img = cap.read()
        if not ok:
            break
        if f not in by_frame:
            continue
        t = f / sc.fps
        for poly, col, closed in geo:
            cv2.polylines(img, [(poly * s).astype(np.int32)], closed, col, 1, cv2.LINE_AA)
        for tid, r in by_frame[f]:
            x1, y1, x2, y2 = (r[2:6] * s).astype(int)
            if tid % 3 == 0:
                if (f, tid) in sc.occupant or tid in sc.riders:
                    continue
                col = (0, 0, 255) if flags.get((f, tid)) else (0, 165, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if col == (0, 0, 255) else 1)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 140, 40), 1)
        i = sc.sig.index(t)
        veh = "RED" if sc.sig.veh_red[i] else ("GREEN" if sc.sig.veh_green[i] else "YELLOW/-")
        ped = "GREEN" if sc.sig.ped_green[i] else "RED"
        act_p = [SHORT[c] for c in classes if any(a <= t < b for a, b in pred.get(c, []))]
        act_g = [SHORT[c] for c in classes if any(a <= t < b for a, b in gt.get(c, []))]
        cv2.rectangle(img, (0, 0), (W, 64), (0, 0, 0), -1)
        cv2.putText(img, f"{args.stem}  t={t:6.1f}s  frame {f}   head7 (cars, stop line 4): {veh}   head8 (people): {ped}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "MODEL:  " + (", ".join(act_p) or "-"), (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 170, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "LABELS: " + (", ".join(act_g) or "-"), (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 90), 1, cv2.LINE_AA)
        pan = panel.copy()
        x = int(x0 + t * tscale)
        cv2.line(pan, (x, 14), (x, pan.shape[0] - 2), (255, 255, 255), 1)
        enc.stdin.write(np.vstack([img, pan]).tobytes())
        n += 1
    enc.stdin.close()
    enc.wait()
    print(f"{out}: {n} frames")


if __name__ == "__main__":
    main()
