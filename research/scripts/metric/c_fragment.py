"""(c) Fragmentation / merging / duplicates, and optimal gap-merge + min-duration post-processing."""
import numpy as np
from common import EV, TAUS, f1_mean_tau
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("rs", os.path.join(os.path.dirname(__file__), "..", "kit", "wiut_cv_scripts", "run_submission.py"))

def counts(gt, pred):
    return {t: EV.match_segments(gt, pred, t) for t in TAUS}

def show(name, gt, pred):
    c = counts(gt, pred)
    f = [EV.prf(*c[t])["f1"] for t in TAUS]
    print(f"  {name:55s} " + "  ".join(f"t{t}:{c[t][0]}/{c[t][1]}/{c[t][2]}" for t in TAUS) + f"   meanF1={np.mean(f):.3f}")

print("TP/FP/FN per tau for one class, one video")
show("perfect: GT[0,10] pred[0,10]", [(0, 10)], [(0, 10)])
show("split 50/50 with 0.5 s hole: pred[0,5],[5.5,10]", [(0, 10)], [(0, 5), (5.5, 10)])
show("split 70/30: pred[0,7],[7.5,10]", [(0, 10)], [(0, 7), (7.5, 10)])
show("split in 3: [0,3.3],[3.5,6.6],[6.8,10]", [(0, 10)], [(0, 3.3), (3.5, 6.6), (6.8, 10)])
show("2 GT merged: GT[0,5],[7,12] pred[0,12]", [(0, 5), (7, 12)], [(0, 12)])
show("2 GT merged: GT[0,20],[22,40] pred[0,40]", [(0, 20), (22, 40)], [(0, 40)])
show("2 GT, correct: GT[0,5],[7,12] pred[0,5],[7,12]", [(0, 5), (7, 12)], [(0, 5), (7, 12)])
show("dup non-overlapping blip: GT[0,10] pred[0,10],[10.5,11]", [(0, 10)], [(0, 10), (10.5, 11)])
show("GT 3s accident, pred 60s window", [(20, 23)], [(0, 60)])

# harness behaviour on overlapping same-class output (exact code from run_submission.clean_events)
src = open(os.path.join(os.path.dirname(__file__), "..", "kit", "wiut_cv_scripts", "run_submission.py")).read()
ns = {}; exec(src.split("def run_risk")[0].replace("import cv2", ""), ns)
kept, probs = ns["clean_events"]([[0, 3, "jaywalking"], [2, 10, "jaywalking"], [5, 8, "congestion"], [8, 9, "congestion"],
                                 [290, 305, "fire_smoke"], [301, 302, "red_light"], [4, 4, "near_miss"]], EV.OFFICIAL_CLASSES, 300.0)
print("\nHarness clean_events on overlapping/edge output (duration 300 s):")
print("  kept:", kept)
for p in probs: print("  drop:", p)
show("effect: GT jaywalk[0,10]; we emitted [0,3]+[2,10] -> harness keeps [0,3]", [(0, 10)], [(0, 3)])

# ------------------ gap-merge / min-duration simulation ----------------------
rng = np.random.default_rng(0)
FPS = 25; STRIDE = 3; dt = STRIDE / FPS

def make_video(dur, regime):
    t = np.arange(0, dur, dt)
    gts = []; cur = rng.uniform(2, 20)
    while True:
        L = rng.uniform(2, 6) if regime == "short" else rng.uniform(20, 120)
        if cur + L > dur - 1: break
        gts.append((cur, cur + L))
        cur += L + (rng.exponential(25) + 3 if regime == "short" else rng.exponential(90) + 10)
    flag = np.zeros(len(t), bool)
    for s, e in gts:
        # onset/offset jitter + dropouts (2-state Markov: mean on-run 2 s short / 6 s long, mean off-run 0.4 s / 1.5 s)
        s2 = s + rng.normal(0.2, 0.25 if regime == "short" else 1.5); e2 = e + rng.normal(-0.2, 0.25 if regime == "short" else 1.5)
        on_mean, off_mean = (2.0, 0.4) if regime == "short" else (6.0, 1.5)
        idx = np.nonzero((t >= s2) & (t < e2))[0]
        state = True
        for i in idx:
            flag[i] = state
            if state and rng.random() < dt / on_mean: state = False
            elif not state and rng.random() < dt / off_mean: state = True
    # spurious blips: 1.5/min, length exp(0.4 s) short-regime, exp(3 s) long-regime
    nb = rng.poisson(1.5 * dur / 60)
    for _ in range(nb):
        c = rng.uniform(0, dur); L = rng.exponential(0.4 if regime == "short" else 3.0)
        flag[(t >= c) & (t < c + L)] = True
    return t, flag, gts

def to_segments(t, flag, gap, mind):
    d = np.diff(np.r_[0, flag.astype(int), 0]); st = np.nonzero(d == 1)[0]; en = np.nonzero(d == -1)[0]
    segs = [[t[a], t[b - 1] + dt] for a, b in zip(st, en)]
    out = []
    for s, e in segs:
        if out and s - out[-1][1] <= gap: out[-1][1] = e
        else: out.append([s, e])
    return [(s, e) for s, e in out if e - s >= mind]

for regime, gaps, minds in [("short", [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0], [0, 0.25, 0.5, 1.0, 1.5, 2.0]),
                            ("long", [0, 1, 2, 4, 6, 10, 20], [0, 1, 2, 5, 10, 15])]:
    vids = [make_video(300, regime) for _ in range(40)]
    print(f"\nRegime '{regime}' ({sum(len(v[2]) for v in vids)} GT events in 40x5min videos): tau-mean F1 vs gap-merge g (rows) and min-duration m (cols)")
    print("   g\\m   " + "".join(f"{m:>7}" for m in minds))
    best = (-1, None)
    for g in gaps:
        row = []
        for m in minds:
            f, _ = f1_mean_tau([v[2] for v in vids], [to_segments(v[0], v[1], g, m) for v in vids])
            row.append(f)
            if f > best[0]: best = (f, (g, m))
        print(f"   {g:<6}" + "".join(f"{x:>7.3f}" for x in row))
    print(f"   best: g={best[1][0]} m={best[1][1]} F1={best[0]:.3f}")
