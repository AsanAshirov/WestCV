# usage: python disagree.py gt_A.json gt_B.json [thr=0.5]
import json, sys

def tiou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0

A = json.load(open(sys.argv[1], encoding="utf-8"))
B = json.load(open(sys.argv[2], encoding="utf-8"))
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
for name, X, Y in (("A->B", A, B), ("B->A", B, A)):
    for vid, v in X.items():
        for ev in v["events"]:
            same = [e2 for e2 in Y.get(vid, {}).get("events", []) if e2[2] == ev[2]]
            best = max((tiou(ev, e2) for e2 in same), default=0.0)
            if best < thr:
                print(name, vid, ev, "best tIoU", round(best, 2))
