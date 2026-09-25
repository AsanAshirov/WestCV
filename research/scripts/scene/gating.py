import math
Ts = 1103.5 / 3600          # sample footage hours (340.3+317.8+317.8+127.6 s)
print(f"sample hours Ts={Ts:.4f}")
Tt = {"T=0.284h (6 gap clips only)": 1023.1 / 3600, "T=0.55h (old assumption)": 0.55, "T=1.0h": 1.0}
print("\n(a) q = 1-exp(-lambda*T) for a known rate lambda [events/h]")
print("lambda  " + "  ".join(f"{k[:9]:>9s}" for k in Tt))
for lam in [0.5, 1, 2, 3, 5, 10, 20]:
    print(f"{lam:6.1f}  " + "  ".join(f"{1 - math.exp(-lam * T):9.2f}" for T in Tt.values()))
print("\n(b) posterior-predictive q given n events counted in your OWN labels of the 18.4 min samples (Jeffreys Gamma(n+.5, Ts))")
print("n_obs   " + "  ".join(f"{k[:9]:>9s}" for k in Tt))
for n in [0, 1, 2, 3, 5, 10]:
    print(f"{n:6d}  " + "  ".join(f"{1 - (Ts / (Ts + T)) ** (n + 0.5):9.2f}" for T in Tt.values()))
print("\n(c) break-even: enable class iff q > q* = p*S/(f + p*S)   (S=mean F1 of other classes, f=F1 if present, p=P(>=1 FP | absent))")
for S, f in [(0.4, 0.3), (0.4, 0.5), (0.3, 0.3)]:
    for rfp in [0.5, 2, 5]:
        row = []
        for T in Tt.values():
            p = 1 - math.exp(-rfp * T)
            qs = p * S / (f + p * S)
            lam = -math.log(1 - qs) / T
            row.append(f"q*={qs:.2f} lam*={lam:4.1f}/h")
        print(f"S={S} f={f} FP-rate={rfp}/h: " + " | ".join(row))
print("\n(d) Part B alarm F1 with 1 accident matched and F false alarms; FA-rate budget r=F/T")
T = 1023.1 / 3600
for F in [0, 1, 2, 3, 5]:
    P = 1 / (1 + F)
    print(f"F={F}: P={P:.2f} F1_alarm={2 * P / (P + 1):.3f}; max FA rate for T=0.284h: {F / T:5.1f}/h, for T=0.55h: {F / 0.55:4.1f}/h")
n_frames = 1023.1 * 30000 / 1001
print(f"\npositives per accident = 5 s * 29.97 = {5 * 29.97:.0f} frames; test frames ~ {n_frames:.0f}; positive rate r ~ {5 * 29.97 / n_frames * 100:.2f}% per accident")
