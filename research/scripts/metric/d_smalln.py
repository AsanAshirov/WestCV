"""(d) variance of pooled per-class F1 with few test instances."""
import numpy as np
rng = np.random.default_rng(0)

def f1(tp, fp, fn): return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

print("Attainable tau-mean F1 values are coarse. n GT, k TP (all taus), j FP:")
for n in [1, 2, 3]:
    print(f"  n={n}: " + ", ".join(f"TP{k}/FP{j}={f1(k, j, n-k):.2f}" for k in range(1, n + 1) for j in range(0, 3)))

# detector: each GT detected w.p. d, localisation IoU ~ Beta(6,2) (mean .75), FPs ~ Poisson(mu) over test set
print("\nMean +- sd (and 10-90% range) of tau-mean F1_c across hypothetical test sets")
print("  detector: recall d=0.7, IoU~Beta(6,2), FP ~ Poisson(1.0) over the whole test set")
for n in [1, 2, 3, 5, 10, 20]:
    vals = []
    for _ in range(20000):
        det = rng.random(n) < 0.7; iou = rng.beta(6, 2, n); fp0 = rng.poisson(1.0)
        vals.append(np.mean([f1((det & (iou >= t)).sum(), fp0 + (det & (iou < t)).sum(), n - (det & (iou >= t)).sum())
                             for t in (0.3, 0.5, 0.7)]))
    v = np.array(vals)
    print(f"  n={n:<3} mean={v.mean():.3f} sd={v.std():.3f}  p10={np.percentile(v,10):.2f} p90={np.percentile(v,90):.2f}  P(F1=0)={np.mean(v==0):.2f}")

print("\nWeight of a single instance: with |C| classes, one class with n=1 GT moves Score_A by up to 1/|C|:")
for C in [4, 6, 8, 10, 14]:
    print(f"  |C|={C:<3} one rare event = {100/C:.1f} Score_A points = {0.42*100/C:.2f} elimination points")

# dev-set size: how many labelled events per class to distinguish two post-processing settings?
print("\nDev-set noise: sd of a class F1 estimate ~ sqrt(F(1-F)/n_eff); with F=0.6:")
for n in [3, 5, 10, 20, 40]:
    print(f"  n={n:<3} sd ~ {np.sqrt(0.6*0.4/n):.3f}")
