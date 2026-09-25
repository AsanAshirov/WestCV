"""(a) Macro-F1 dilution: should we ever emit class c?"""
import numpy as np
from common import EV

# --- 1. exact check with the official scorer --------------------------------
gt = {"v.mp4": {"duration": 300, "fps": 25, "events": [[10, 15, "accident"], [50, 80, "congestion"],
                                                         [100, 104, "jaywalking"]]}}
base = {"v.mp4": {"events": [[10, 15, "accident"], [50, 70, "congestion"]]}}
spur = {"v.mp4": {"events": base["v.mp4"]["events"] + [[200, 205, "fire_smoke"]]}}
spur3 = {"v.mp4": {"events": base["v.mp4"]["events"] + [[200, 205, "fire_smoke"], [220, 225, "fire_smoke"],
                                                          [240, 245, "fire_smoke"]]}}
jw = {"v.mp4": {"events": base["v.mp4"]["events"] + [[101, 104, "jaywalking"], [150, 152, "jaywalking"]]}}
for name, p in [("base (3 GT classes, jaywalking not predicted)", base), ("+1 spurious fire_smoke FP", spur),
                ("+3 spurious fire_smoke FPs", spur3), ("+jaywalking 1TP+1FP", jw)]:
    a = EV.evaluate_part_a(gt, p)
    print(f"{name:48s} C={a['classes']}  ScoreA={a['score_a']:.4f}")

# --- 2. closed form -----------------------------------------------------------
# N other classes are scored (present); their mean F1 = Sbar.
# present (prob p): emitting c adds f/(N+1)     (not emitting adds 0/(N+1))
# absent (prob 1-p): emitting >=1 FP (prob q) turns Sbar into Sbar*N/(N+1): loss Sbar/(N+1)
# EV = [p f - (1-p) q Sbar] / (N+1)   ->  emit iff p/(1-p) > q Sbar / f,  p* = qSbar/(f+qSbar)
print("\nBreak-even presence probability p* = q*Sbar/(f+q*Sbar)  (independent of |C|)")
print("   f\\(q*Sbar) " + "".join(f"{x:>7.2f}" for x in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]))
for f in [0.1, 0.2, 0.3, 0.5, 0.7]:
    print(f"   f={f:<5}    " + "".join(f"{x/(f+x):>7.2f}" for x in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]))

print("\nExpected change in Score_A (x100 points) from enabling class c; Sbar=0.40, q=1 (it WILL fire if absent)")
for f in [0.2, 0.5]:
    print(f"  f={f}:   p=" + "".join(f"{p:>7}" for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]))
    for C in [4, 6, 8, 10]:
        row = [(p * f - (1 - p) * 1.0 * 0.40) / C * 100 for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]]
        print(f"    |C|={C:<3}   " + "".join(f"{x:>7.2f}" for x in row))
print("\nSame, but precision-first thresholds: q=0.2 (P(any FP over whole test set | absent)=20%), f reduced by 25%")
for f in [0.2, 0.5]:
    f2 = 0.75 * f
    print(f"  f={f}->{f2:.3f}: p=" + "".join(f"{p:>7}" for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]))
    for C in [4, 6, 8, 10]:
        row = [(p * f2 - (1 - p) * 0.2 * 0.40) / C * 100 for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]]
        print(f"    |C|={C:<3}   " + "".join(f"{x:>7.2f}" for x in row))

# --- 3. q from FP rate: q = 1 - exp(-lambda * T) --------------------------------
print("\nq = P(>=1 FP | class absent) = 1-exp(-lambda*T)   (lambda = FPs per hour, T = test hours)")
print("  lambda/h:  " + "".join(f"{l:>7}" for l in [0.1, 0.25, 0.5, 1, 2, 5]))
for T in [0.5, 1, 2]:
    print(f"  T={T:<4}h    " + "".join(f"{1-np.exp(-l*T):>7.2f}" for l in [0.1, 0.25, 0.5, 1, 2, 5]))

# --- 4. threshold choice for one class under presence uncertainty -------------
# toy detector: n_true GT events if present (n ~ 1+Poisson(2)); each detected as a candidate w.p. 0.85
# with confidence ~ Beta(5,2) and localisation IoU ~ Beta(6,2); false candidates Poisson(lam_fc*T) with conf ~ Beta(2,5)
rng = np.random.default_rng(1)
def sim_class(thr, present, reps=4000, T=1.0, lam_fc=6.0):
    f1s = []; anyfp = 0
    for _ in range(reps):
        n = 1 + rng.poisson(2) if present else 0
        det = rng.random(n) < 0.85
        conf = rng.beta(5, 2, n); iou = rng.beta(6, 2, n)
        kept_true = det & (conf >= thr)
        nfc = rng.poisson(lam_fc * T); fcconf = rng.beta(2, 5, nfc); nfp = int((fcconf >= thr).sum())
        if not present:
            anyfp += nfp > 0; continue
        f = []
        for tau in (0.3, 0.5, 0.7):
            tp = int((kept_true & (iou >= tau)).sum())
            fp = nfp + int((kept_true & (iou < tau)).sum())
            fn = n - tp
            f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        f1s.append(np.mean(f))
    return (np.mean(f1s) if present else anyfp / reps)

print("\nToy detector (6 false candidates/h before thresholding, 1 test hour): best threshold vs presence prob p")
thrs = np.round(np.arange(0.3, 0.96, 0.05), 2)
F = {th: sim_class(th, True) for th in thrs}
Q = {th: sim_class(th, False) for th in thrs}
print("  thr    E[F1|present]  q=P(FP|absent)")
for th in thrs:
    print(f"  {th:.2f}   {F[th]:.3f}          {Q[th]:.3f}")
Sbar = 0.40
for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    ev = {th: p * F[th] - (1 - p) * Q[th] * Sbar for th in thrs}
    best = max(ev, key=ev.get)
    print(f"  p={p}: best thr={best:.2f}  EV*|C|={ev[best]:+.3f}   (thr=0.5 gives {ev[0.5]:+.3f}; never emit = 0)")
