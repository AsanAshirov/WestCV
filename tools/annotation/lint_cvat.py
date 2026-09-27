# lint_cvat.py — find annotation slips in CVAT exports before they become ground truth.
#
#   python tools/annotation/lint_cvat.py labels/cvat/*/*.xml
#
# Run before cvat_to_gt.py. Everything flagged is for a human to look at in CVAT (frame
# numbers are CVAT frames), nothing is changed. Checks:
#   OPEN      a moving-actor track never switched outside: it runs to the end of the video. If its
#             box stopped changing long before the end, O was probably forgotten.
#   SHORT     a visible run shorter than 0.5 s (a stray click or a one-frame blip).
#   REAPPEAR  the same track visible again less than 1 s after it was switched outside.
#   LONG      a run much longer than such an event normally lasts.
#   SHAPE     a box drawn as a single shape, not a track (exported as visible on frame f, outside on
#             f + 1); cvat_to_gt turns it into a one-frame event.
#   CHAIN     one-frame tracks on consecutive frames (CVAT Propagate): they add up to one event,
#             shown once for a check that this was meant.
# stopped_vehicle, congestion, road_obstacle and fire_smoke legitimately stay open to the end of a clip.
import argparse
import glob
import sys
from pathlib import Path

from cvat_to_gt import CLASSES, FPS_DEFAULT, load_xml, meta_value, visible_runs

STATIC = ("stopped_vehicle", "congestion", "road_obstacle", "fire_smoke")
FROZEN_S = 5.0     # an open track whose last keyframe is this long before the end is suspicious
SHORT_S = 0.5
REAPPEAR_S = 1.0
LONG_S = {"solid_line_crossing": 10, "red_light": 15, "failure_to_yield": 15, "stop_line": 90,
          "illegal_turn": 30, "illegal_u_turn": 30, "wrong_way": 60, "jaywalking": 60, "near_miss": 15}


def lint(path: Path, fps: float) -> list[str]:
    root = load_xml(path)
    stop = int(meta_value(root, "stop_frame") or int(meta_value(root, "size") or 1) - 1)
    t = lambda f: f"{f / fps:.1f}s (frame {f})"  # noqa: E731
    out, singles = [], []
    for tr in root.findall("track"):
        label, tid = tr.get("label"), tr.get("id")
        if label not in CLASSES:
            continue
        shapes = [el for el in tr if el.get("frame") is not None]
        keys = sorted(int(s.get("frame")) for s in shapes if s.get("keyframe") == "1" and s.get("outside") != "1")
        runs = visible_runs(shapes, stop)
        for i, (f0, f1, _) in enumerate(runs):
            dur = (f1 + 1 - f0) / fps
            where = f"track {tid} {label} {t(f0)} - {t(f1)}"
            if f1 == stop and label not in STATIC:
                last = max((k for k in keys if k <= f1), default=f0)
                frozen = (stop - last) / fps
                hint = f", box unchanged for the last {frozen:.0f} s: O forgotten?" if frozen >= FROZEN_S else ""
                out.append(f"OPEN     {where}: never switched outside{hint}")
            if f0 == f1:
                shape = len(shapes) <= 2 and all(s.get("keyframe") == "1" for s in shapes)
                singles.append((label, f0, tid, shape))
            elif dur < SHORT_S:
                out.append(f"SHORT    {where}: visible {dur:.2f} s")
            if dur > LONG_S.get(label, 1e9):
                out.append(f"LONG     {where}: {dur:.0f} s")
            if i and (f0 - runs[i - 1][1] - 1) / fps < REAPPEAR_S:
                out.append(f"REAPPEAR {where}: back {(f0 - runs[i - 1][1] - 1) / fps:.2f} s after outside")
    chains = []  # one-frame runs of a class on consecutive frames
    for label, f, tid, shape in sorted(singles):
        if chains and chains[-1][0] == label and f <= chains[-1][2] + 1:
            chains[-1][2], chains[-1][4] = f, chains[-1][4] + 1
        else:
            chains.append([label, f, f, tid, 1, shape])
    for label, f0, f1, tid, n, shape in chains:
        if n == 1:
            kind = "SHAPE   " if shape else "SHORT   "
            out.append(f"{kind} track {tid} {label} {t(f0)}: visible one frame")
        else:
            out.append(f"CHAIN    {label} {t(f0)} - {t(f1)}: {n} one-frame tracks in a row (Propagate?)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+", help="CVAT for video 1.1 annotations.xml or export .zip")
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    total = 0
    paths = [Path(p) for a in args.exports for p in (sorted(glob.glob(a)) or [a])]  # PowerShell/cmd
    for path in paths:
        issues = lint(path, args.fps)
        total += len(issues)
        print(f"== {path}: {len(issues)} to check")
        for line in issues:
            print("  " + line)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
