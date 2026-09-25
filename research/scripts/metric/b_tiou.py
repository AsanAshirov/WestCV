"""(b) temporal-IoU sensitivity."""
import numpy as np
from common import EV, TAUS

Ls = [2, 3, 5, 10, 30, 120]
cases = {
    "shift (both ends move same way by d)": lambda t: (1 - t) / (1 + t),       # IoU=(L-d)/(L+d)
    "symmetric widen (d on each side)":     lambda t: (1 - t) / (2 * t),       # IoU=L/(L+2d)
    "symmetric shrink (d on each side)":    lambda t: (1 - t) / 2,             # IoU=(L-2d)/L
    "one end late/extended by d":           lambda t: (1 - t) / t,             # IoU=L/(L+d)
    "one end early/truncated by d":         lambda t: (1 - t),                 # IoU=(L-d)/L
}
print("Max tolerable error d (seconds) for a GT segment of length L to still match at tau")
for name, fn in cases.items():
    print(f"\n  {name}:  d_max = L * {', '.join(f'{fn(t):.3f}@{t}' for t in TAUS)}")
    print("    L(s) " + "".join(f"{'t=' + str(t):>9}" for t in TAUS))
    for L in Ls:
        print(f"    {L:<5}" + "".join(f"{L*fn(t):>9.2f}" for t in TAUS))
# sanity with official tiou
assert abs(EV.tiou((0, 5), (1, 6)) - 4 / 6) < 1e-12

# per-instance credit = (#tau passed)/3 as a function of IoU
print("\nPer-matched-instance credit in the tau-average: IoU<0.3 ->0 (and counts FP+FN!), [0.3,0.5)->1/3, [0.5,0.7)->2/3, >=0.7 ->1")

# Gaussian boundary noise: both ends independently N(0, sigma)
rng = np.random.default_rng(0)
print("\nExpected tau-averaged TP credit for ONE GT with both boundaries ~ N(0, sigma) independently (1e5 draws)")
sig = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
print("    L(s) " + "".join(f"{'s=' + str(s):>8}" for s in sig))
for L in Ls:
    row = []
    for s in sig:
        a = rng.normal(0, s, 100000); b = L + rng.normal(0, s, 100000)
        ok = b > a
        inter = np.clip(np.minimum(b, L) - np.maximum(a, 0), 0, None)
        union = (L) + (b - a) - inter
        iou = np.where(ok, inter / union, 0)
        row.append(np.mean([(iou >= t).mean() for t in TAUS]))
    print(f"    {L:<5}" + "".join(f"{x:>8.2f}" for x in row))

# frame-stride quantisation: boundaries snap to sampled frames (worst case error = stride/fps per end)
print("\nWorst-case IoU from boundary quantisation only (stride k frames at 25 fps, each end off by up to k/25 s, widening)")
print("    L(s) " + "".join(f"{'k=' + str(k):>8}" for k in [1, 2, 3, 5, 10, 25]))
for L in Ls:
    print(f"    {L:<5}" + "".join(f"{L/(L+2*k/25):>8.3f}" for k in [1, 2, 3, 5, 10, 25]))

# when the END is uncertain (start anchored): what predicted length maximises tau-averaged credit?
print("\nStart known, true length L ~ LogNormal(median m, sigma_log); best predicted length / m")
for sl in [0.2, 0.4, 0.6, 0.8]:
    Lt = np.exp(rng.normal(0, sl, 200000))
    best, bk = -1, None
    for k in np.arange(0.5, 2.01, 0.05):
        iou = np.minimum(Lt, k) / np.maximum(Lt, k)
        v = np.mean([(iou >= t).mean() for t in TAUS])
        if v > best: best, bk = v, k
    iou1 = np.minimum(Lt, 1) / np.maximum(Lt, 1)
    v1 = np.mean([(iou1 >= t).mean() for t in TAUS])
    print(f"  sigma_log={sl}: best k={bk:.2f} credit={best:.3f} (k=1 credit={v1:.3f})")

# double penalty: a badly localised prediction vs no prediction (class with 4 GT, 3 perfect TPs)
def f1(tp, fp, fn): return 2 * tp / (2 * tp + fp + fn)
print("\nDouble penalty: class with 4 GT, 3 perfectly matched; 4th GT either not predicted or predicted with IoU=0.4")
print(f"  not predicted: F1 per tau = {f1(3,0,1):.3f} ,{f1(3,0,1):.3f}, {f1(3,0,1):.3f} -> mean {f1(3,0,1):.3f}")
m = np.mean([f1(4, 0, 0), f1(3, 1, 1), f1(3, 1, 1)])
print(f"  IoU 0.4 pred:  F1 per tau = {f1(4,0,0):.3f}, {f1(3,1,1):.3f}, {f1(3,1,1):.3f} -> mean {m:.3f}")
m2 = np.mean([f1(3, 1, 1)] * 3)
print(f"  IoU 0.2 pred:  mean {m2:.3f}  (worse than not predicting)")
