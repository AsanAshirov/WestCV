"""Shared helpers: import the OFFICIAL evaluate.py and a vectorised Part-B scorer
that reproduces it exactly (verified in verify.py)."""
import sys, os
import numpy as np

KIT = os.path.join(os.path.dirname(__file__), "..", "kit", "wiut_cv_scripts")
sys.path.insert(0, os.path.abspath(KIT))
import evaluate as EV  # official metric, unmodified

H, W, THETA, GAP = EV.H, EV.W, EV.THETA, EV.MERGE_GAP
TAUS = EV.TIOU_THRESHOLDS


def f1_mean_tau(gt, pred):
    """mean over tau of F1 for ONE class, pooled over videos.
    gt, pred: list (per video) of list of (s,e)."""
    out = []
    for t in TAUS:
        TP = FP = FN = 0
        for g, p in zip(gt, pred):
            tp, fp, fn = EV.match_segments(g, p, t)
            TP += tp; FP += fp; FN += fn
        out.append(EV.prf(TP, FP, FN)["f1"])
    return float(np.mean(out)), out


# ---------------- vectorised Part B (same semantics as evaluate.py) ----------------
def frame_labels(t, acc, nm):
    """+1 pos, 0 neg, -1 ignored (vectorised version of EV.frame_label)."""
    lab = np.zeros(len(t), dtype=np.int8)
    ign_acc = np.zeros(len(t), bool)
    pos = np.zeros(len(t), bool)
    ign_nm = np.zeros(len(t), bool)
    for s, e in acc:
        ign_acc |= (t >= s) & (t <= e)
        pos |= (t >= s - H) & (t < s)
    for s, e in nm:
        ign_nm |= (t >= s - H) & (t <= e)
    lab[:] = 0
    lab[ign_nm] = -1
    lab[pos] = 1          # positive window beats near-miss ignore (order in EV.frame_label)
    lab[ign_acc] = -1     # inside accident beats everything
    return lab


def ap_sklearn(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    n_pos = labels.sum()
    if n_pos == 0 or len(scores) == 0:
        return 0.0
    o = np.argsort(-scores, kind="mergesort")
    s, l = scores[o], labels[o]
    tp = np.cumsum(l); fp = np.cumsum(1 - l)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]   # end of each tie group
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp); rec = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def alarm_starts(t, sc, theta=THETA, gap=GAP):
    on = sc >= theta
    if not on.any():
        return []
    d = np.diff(np.r_[0, on.astype(np.int8), 0])
    st = np.nonzero(d == 1)[0]; en = np.nonzero(d == -1)[0] - 1
    merged = []
    for a, b in zip(t[st], t[en]):
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [m[0] for m in merged]


def score_b(videos, return_parts=True):
    """videos: list of dict(t=np.array, score=np.array, acc=[(s,e)], nm=[(s,e)]).
    Scores are rounded to 4 decimals like run_submission.py."""
    all_s, all_l = [], []
    n_acc = n_alarm = n_match = 0; tta = 0.0
    for v in videos:
        t = np.round(v["t"], 4); sc = np.round(np.clip(v["score"], 0, 1), 4)
        acc = sorted(v["acc"]); nm = v["nm"]
        lab = frame_labels(t, acc, nm)
        m = lab >= 0
        all_s.append(sc[m]); all_l.append(lab[m])
        starts = alarm_starts(t, sc)
        starts = [a for a in starts if frame_labels(np.array([a]), acc, nm)[0] >= 0]
        n_alarm += len(starts)
        un = list(starts)
        for s, _ in acc:
            n_acc += 1
            c = [a for a in un if s - W <= a < s]
            if c:
                a = min(c); un.remove(a); n_match += 1; tta += s - a
    if n_acc == 0:
        return None
    S = np.concatenate(all_s); L = np.concatenate(all_l)
    apr = ap_sklearn(S, L); r = L.mean()
    ap = max(0.0, (apr - r) / (1 - r)) if r < 1 else 0.0
    P = n_match / n_alarm if n_alarm else 0.0
    R = n_match / n_acc
    f1 = 2 * P * R / (P + R) if P + R else 0.0
    mtta = tta / n_acc
    sb = 0.4 * ap + 0.4 * f1 + 0.2 * min(1.0, mtta / W)
    return dict(score_b=sb, ap=ap, ap_raw=apr, r=r, P=P, R=R, f1=f1, mtta=mtta,
                n_alarms=n_alarm, n_match=n_match, n_acc=n_acc)
