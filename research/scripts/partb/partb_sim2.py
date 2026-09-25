from partb_sim import *
hours = NV*DUR/3600
for br in [0.3, 1.0, 3.0]:
    print(f"== benign interactions {br}/min; test set = {hours:.2f} h, {N_ACC} accidents")
    best=None
    for c in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        m, s = run(dict(c_on=c, two_band=False), gen_kw=dict(benign_per_min=br), seeds=range(8))
        fp = m[6] - m[5]*N_ACC
        print(fmt(f"c_on={c}", m) + f"  FP/h={fp/hours:.1f}  FP/acc={fp/N_ACC:.2f}")
