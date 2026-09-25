"""marginal-candidate rule, F1_alarm expectation, elimination weights, harness edge cases."""
import numpy as np

# ---- marginal candidate: add iff E[tau-credit] > F1/2 -----------------------------------
def f1(tp, g, p): return 2 * tp / (g + p)
print("Adding one candidate to a class with G=10 GT, P=10 preds, TP=6 at every tau (F1=0.60):")
for credit_taus in [0, 1, 2, 3]:
    for prob in [0.2, 0.3, 0.5, 0.8]:
        base = f1(6, 10, 10)
        new = np.mean([prob * f1(7 if i < credit_taus else 6, 10, 11) + (1 - prob) * f1(6, 10, 11) for i in range(3)])
        print(f"  P(real)={prob:.1f}, passes {credit_taus}/3 taus if real: expected credit={prob*credit_taus/3:.2f}  dF1={new-base:+.4f}")
print("  -> break-even expected credit ~ F1/2 = 0.30 (Lipton et al. 2014 result, applied per tau)")

# ---- F1_alarm as function of recall and false alarms --------------------------------------
print("\nE[F1_alarm] ~ 2m/(n+m+k): recall 0.6, k = false alarms over the whole test set")
print("  n_acc " + "".join(f"{'k=' + str(k):>7}" for k in [0, 1, 2, 3, 5, 10, 20]))
for n in [1, 2, 3, 5, 10]:
    m = 0.6 * n
    print(f"  {n:<5} " + "".join(f"{2*m/(n+m+k):>7.2f}" for k in [0, 1, 2, 3, 5, 10, 20]))

# ---- elimination weights -------------------------------------------------------------------
print("\nElimination = 0.6*(0.7A+0.3B) + 0.25*Web + 0.15*Code")
items = [("Score_A (per 0.10)", 0.42 * 0.10), ("Score_B (per 0.10)", 0.18 * 0.10),
         ("Web: live demo (30%)", 0.25 * 0.30), ("Web: sample-video viz (20%)", 0.25 * 0.20),
         ("Web: EDA (15%)", 0.25 * 0.15), ("Web: approach&report (15%)", 0.25 * 0.15),
         ("Web: team&portfolio (10%)", 0.25 * 0.10), ("Web: design/UX/extras (10%)", 0.25 * 0.10),
         ("Code: runs as submitted (40%)", 0.15 * 0.40), ("Code: reproducibility (25%)", 0.15 * 0.25),
         ("Code: structure (20%)", 0.15 * 0.20), ("Code: engineering (15%)", 0.15 * 0.15)]
for n, v in items: print(f"  {n:32s} {v:.4f}   (= {v/0.042*0.10:.3f} of Score_A)")
for A, B, Wb, C in [(0.25, 0.10, 0.8, 0.85), (0.40, 0.20, 0.8, 0.85), (0.40, 0.0, 0.9, 0.9), (0.30, 0.15, 0.5, 0.7)]:
    el = 0.6 * (0.7 * A + 0.3 * B) + 0.25 * Wb + 0.15 * C
    fail = 0.25 * Wb + 0.15 * (C - 0.40)
    print(f"  A={A} B={B} Web={Wb} Code={C}: elimination={el:.3f}; if package fails to run: {fail:.3f} (loss {el-fail:.3f})")

# ---- harness score coercion ------------------------------------------------------------------
for x in [float("nan"), 1.7, -0.2, 0.49996, np.float32(0.3), np.array([0.7])]:
    try: last = min(1.0, max(0.0, float(x)))
    except Exception: last = 0.0
    print(f"  step returned {x!r:>22} -> recorded {round(last, 4)}")
try:
    float(None)
except Exception as e:
    print("  step returned None -> float() raises -> recorded 0.0 (caught)")
