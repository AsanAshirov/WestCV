"""Verify the vectorised Part-B scorer against the official evaluate.py on random cases,
and the AP helper against sklearn."""
import numpy as np
from common import EV, score_b, ap_sklearn
from sklearn.metrics import average_precision_score

rng = np.random.default_rng(0)
maxdiff = 0.0
for trial in range(40):
    gt, pred, vids = {}, {"videos": {}}, []
    for k in range(rng.integers(1, 4)):
        fps = 25.0; dur = float(rng.integers(30, 90)); n = int(dur * fps)
        t = np.arange(n) / fps
        acc, nm, evs = [], [], []
        for _ in range(rng.integers(0, 3)):
            s = float(rng.uniform(6, dur - 8)); e = s + float(rng.uniform(1, 6))
            if all(e < a or s > b for a, b in acc):
                acc.append((round(s, 2), round(e, 2)))
        for _ in range(rng.integers(0, 3)):
            s = float(rng.uniform(3, dur - 4)); nm.append((round(s, 2), round(s + rng.uniform(1, 3), 2)))
        # risk: noise + bumps + quantisation to make ties
        sc = np.clip(rng.normal(0.15, 0.12, n), 0, 1)
        for s, e in acc:
            sc[(t > s - rng.uniform(0.5, 9)) & (t < e)] += rng.uniform(0.2, 0.8)
        for _ in range(rng.integers(0, 6)):
            c = rng.uniform(0, dur); sc[(t > c) & (t < c + rng.uniform(0.2, 3))] += 0.5
        sc = np.round(np.clip(sc, 0, 1) * 20) / 20
        name = f"v{k}.mp4"
        gt[name] = {"duration": dur, "fps": fps,
                    "events": [[s, e, "accident"] for s, e in acc] + [[s, e, "near_miss"] for s, e in nm]}
        pred["videos"][name] = {"events": [], "risk": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(t, sc)]}
        vids.append(dict(t=t, score=sc, acc=acc, nm=nm))
    off = EV.evaluate_part_b(gt, pred["videos"])
    mine = score_b(vids)
    if off is None:
        assert mine is None; continue
    for k1, k2 in [("score_b", "score_b"), ("ap", "ap"), ("f1_alarm", "f1"), ("mtta_sec", "mtta")]:
        maxdiff = max(maxdiff, abs(off[k1] - mine[k2]))
print("max |official - vectorised| over 40 random multi-video cases:", maxdiff)

# AP helper vs sklearn
for _ in range(20):
    s = np.round(rng.random(5000), 2); l = (rng.random(5000) < 0.05).astype(int)
    assert abs(ap_sklearn(s, l) - average_precision_score(l, s)) < 1e-12
    assert abs(EV.average_precision(list(s), list(l)) - average_precision_score(l, s)) < 1e-9
print("official AP == sklearn.average_precision_score (with ties): OK")
