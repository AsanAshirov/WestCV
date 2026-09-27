# tune_timeline.py — two-stage tuning of a class detector against the team labels.
#
# Stage 1 (per-sample decision): which pedestrian samples count as "jaywalking now". Scored as the
#   F1 of the class timeline on a 0.1 s grid (union over people vs union of labelled segments) —
#   thousands of samples instead of ~30 segments, so the thresholds are stable.
# Stage 2 (segmentation): how the timeline is cut into events (bridged gap, minimum length),
#   scored with the official segment metric (evaluate.py match_segments), leave-one-video-out.
#
#   python tools/eval/tune_timeline.py
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools/eval"))
from dev_eval import load_gt  # noqa: E402
from evaluate import TIOU_THRESHOLDS, match_segments, prf  # noqa: E402
from westcv.scene import Scene, feet, signed_dist, speeds  # noqa: E402
from westcv.segments import merge  # noqa: E402

TAG = "_w1280"
STEP = 0.1


def person_samples(sc):
    """Rows: t, depth/h, zebra signed distance/h (>0 inside), speed (heights/s), track id."""
    rows = []
    for tid, tr in sc.tracks.items():
        if tid % 3 != 0 or len(tr) < 5 or tid in sc.riders:
            continue
        keep = np.array([(int(r[0]), tid) not in sc.occupant for r in tr]) & (tr[:, 5] < sc.meta["height"] - 8)
        tr = tr[keep]
        if len(tr) < 5:
            continue
        t, p, s, v = speeds(tr, sc.fps)
        h = (tr[:, 5] - tr[:, 3]).clip(1)
        depth = sc.road_depths(p) / h
        dz = np.array([max(signed_dist(z, q) for z in sc.crosswalks.values()) for q in p]) / h
        vh = v * s / h
        rows.append(np.column_stack([t, depth, dz, vh, np.full(len(t), tid)]))
    return np.concatenate(rows) if rows else np.zeros((0, 5))


def grid(segs, dur):
    g = np.zeros(int(np.ceil(dur / STEP)) + 1, bool)
    for a, b in segs:
        g[int(a / STEP):int(np.ceil(b / STEP))] = True
    return g


def timeline(rows, dur, depth_min, zebra_margin, min_speed, dt):
    m = (rows[:, 1] > depth_min) & (rows[:, 2] < -zebra_margin) & (rows[:, 3] >= min_speed)
    g = np.zeros(int(np.ceil(dur / STEP)) + 1, bool)
    for t in rows[m, 0]:
        g[int(t / STEP):int(np.ceil((t + dt) / STEP))] = True
    return g


def segments(g, gap, min_seg):
    idx = np.flatnonzero(g)
    if not len(idx):
        return []
    segs, a, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if (i - prev) * STEP > gap:
            segs.append((a * STEP, (prev + 1) * STEP))
            a = i
        prev = i
    segs.append((a * STEP, (prev + 1) * STEP))
    return [(x, y) for x, y in segs if y - x >= min_seg]


def seg_score(pairs):
    f = []
    for thr in TIOU_THRESHOLDS:
        tp = fp = fn = 0
        for g, p in pairs:
            a, b, c = match_segments(g, p, thr)
            tp, fp, fn = tp + a, fp + b, fn + c
        f.append(prf(tp, fp, fn)["f1"])
    return float(np.mean(f)), f


def frame_f1(pairs):
    tp = sum((g & p).sum() for g, p in pairs)
    fp = sum((~g & p).sum() for g, p in pairs)
    fn = sum((g & ~p).sum() for g, p in pairs)
    return prf(int(tp), int(fp), int(fn))


def main(label="jaywalking"):
    gt = load_gt()
    vids = [v for v in gt if label in gt[v]["covered"]]
    data = {}
    for v in vids:
        sc = Scene.from_cache(v, ROOT, TAG)
        dur = sc.meta["n_frames"] / sc.fps
        segs = [(a, b) for a, b, lab in gt[v]["events"] if lab == label]
        data[v] = (person_samples(sc), dur, segs, grid(segs, dur), sc.dt)
    # human ceiling on the doubly labelled video
    src = json.loads((ROOT / "labels/dev/sources.json").read_text(encoding="utf-8"))
    for v, s in src.items():
        if "second_opinion" in s and v in data:
            other = next(iter(json.loads((ROOT / s["second_opinion"]["file"]).read_text(encoding="utf-8")).values()))["events"]
            o = [(a, b) for a, b, lab in other if lab == label]
            print(f"human vs human on {v}: frame F1 {frame_f1([(data[v][3], grid(o, data[v][1]))])['f1']:.3f}, "
                  f"segment F1 {seg_score([(data[v][2], o)])[0]:.3f}")
    # stage 1
    res = []
    for dmin, zm, ms in itertools.product([0.0, 0.1, 0.2, 0.4], [-0.3, -0.1, 0.0, 0.1, 0.2, 0.4], [0.0, 0.1, 0.2, 0.3]):
        tl = {v: timeline(d[0], d[1], dmin, zm, ms, d[4]) for v, d in data.items()}
        r = frame_f1([(data[v][3], tl[v]) for v in vids])
        res.append((r["f1"], r["precision"], r["recall"], dmin, zm, ms))
    res.sort(reverse=True)
    print("stage 1 (frame F1, precision, recall, depth_min, zebra_margin, min_speed):")
    for r in res[:5]:
        print("   %.3f P %.3f R %.3f  depth>%.1f  outside zebra by %.1f h  speed>=%.1f" % r)
    best = res[0][3:]
    tl = {v: timeline(data[v][0], data[v][1], *best, data[v][4]) for v in vids}
    for v in vids:
        r = frame_f1([(data[v][3], tl[v])])
        print(f"   {v}: frame F1 {r['f1']:.3f} (P {r['precision']:.2f} R {r['recall']:.2f})")
    # stage 2 with leave-one-video-out
    grid2 = list(itertools.product([0.5, 1.0, 2.0, 3.0, 5.0, 8.0], [0.3, 1.0, 2.0, 3.0]))
    table = {prm: {v: (data[v][2], segments(tl[v], *prm)) for v in vids} for prm in grid2}
    b_all = max(grid2, key=lambda prm: seg_score(list(table[prm].values()))[0])
    print("stage 2 best on all (gap, min_seg):", b_all, "segment F1 %.3f" % seg_score(list(table[b_all].values()))[0])
    lovo = []
    for hold in vids:
        b = max(grid2, key=lambda prm: seg_score([table[prm][v] for v in vids if v != hold])[0])
        lovo.append(table[b][hold])
        print(f"   hold {hold}: {b} -> {seg_score([table[b][hold]])[0]:.3f}")
    print("LOVO segment F1 %.3f" % seg_score(lovo)[0], [round(x, 3) for x in seg_score(lovo)[1]])
    out = {"label": label, "depth_min": best[0], "zebra_margin": best[1], "min_speed": best[2], "gap": b_all[0], "min_seg": b_all[1]}
    (ROOT / f"labels/dev/{label}_timeline_params.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
