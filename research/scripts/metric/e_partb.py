"""(e) Part B: mechanics demos + strategy simulation on a toy hazard world, scored with the exact metric."""
import numpy as np
from scipy.signal import lfilter
from common import score_b, EV

FPS = 25.0

def vid(dur, acc, nm, fn):
    t = np.arange(int(dur * FPS)) / FPS
    return dict(t=t, score=fn(t), acc=acc, nm=nm)

def box(t, a, b, v=0.9):
    return np.where((t >= a) & (t < b), v, 0.02)

# ---------------------------------------------------------------------------------
#  Toy world: latent hazard a TTC module could compute (NOT a model of real data;
#  used only to expose the metric's trade-offs)
# ---------------------------------------------------------------------------------
def world(rng, n_videos=12, dur=300.0, n_acc=3, n_nm=6, benign_per_h=40, bg_offset_sd=0.0, lead=(1.0, 5.0),
          p_unforeseeable=0.25):
    vids = []
    slots = rng.choice(n_videos, n_acc, replace=True)
    nm_slots = rng.choice(n_videos, n_nm, replace=True)
    for v in range(n_videos):
        t = np.arange(int(dur * FPS)) / FPS
        off = abs(rng.normal(0, bg_offset_sd)) if bg_offset_sd else 0.0
        h = 0.05 + off + 0.03 * lfilter([0.05], [1, -0.95], rng.normal(0, 1, len(t)))
        def bump(onset, peak_t, end_t, amp, decay):
            rise = np.clip((t - onset) / max(peak_t - onset, 1e-3), 0, 1)
            fall = np.where(t > end_t, np.exp(-(t - end_t) / decay), 1.0)
            return amp * rise * fall * (t >= onset)
        # benign conflicts
        for _ in range(rng.poisson(benign_per_h * dur / 3600)):
            c = rng.uniform(0, dur); r = rng.uniform(0.3, 1.5); p = rng.uniform(0.3, 2.0)
            h = np.maximum(h, bump(c, c + r, c + r + p, rng.beta(2, 4), 1.0))
        acc, nm = [], []
        for _ in range((slots == v).sum()):
            for _try in range(50):
                s = rng.uniform(15, dur - 20); e = s + rng.uniform(3, 12)
                if all(e + 15 < a or s > b + 15 for a, b in acc + nm): break
            acc.append((round(s, 2), round(e, 2)))
            L = rng.uniform(*lead)
            amp = rng.uniform(0.1, 0.4) if rng.random() < p_unforeseeable else rng.uniform(0.6, 1.0)
            h = np.maximum(h, bump(s - L, s, s + 2, amp, 2.0))
        for _ in range((nm_slots == v).sum()):
            for _try in range(50):
                s = rng.uniform(10, dur - 10); e = s + rng.uniform(1.5, 4)
                if all(e + 12 < a or s > b + 12 for a, b in acc + nm): break
            nm.append((round(s, 2), round(e, 2)))
            L = rng.uniform(0.5, 2.5)
            h = np.maximum(h, bump(s - L, s + 0.5, e, rng.uniform(0.5, 0.95), 1.0))
        # measurement noise + tracker glitches (short spikes)
        obs = h + rng.normal(0, 0.04, len(t))
        g = np.nonzero(rng.random(len(t)) < 0.002)[0]
        for i in g: obs[i:i + rng.integers(1, 4)] += 0.45
        vids.append(dict(t=t, h=np.clip(obs, 0, 1), acc=acc, nm=nm))
    return vids

def ema(x, tau_s):
    a = 1 - np.exp(-1 / (FPS * tau_s))
    return lfilter([a], [1, -(1 - a)], x, zi=[x[0] * (1 - a)])[0]

def hysteresis(e, t_on, t_off, min_on_s=0.4, hold_s=3.0):
    on = np.zeros(len(e), bool); state = False; cnt = 0; since = 0
    need = int(min_on_s * FPS); hold = int(hold_s * FPS)
    for i, x in enumerate(e):
        if not state:
            cnt = cnt + 1 if x >= t_on else 0
            if cnt >= need: state, since = True, 0
        else:
            since += 1
            if x < t_off and since >= hold: state, cnt = False, 0
        on[i] = state
    return on

def strat(name, v, t_on=0.6):
    h = v["h"]
    if name == "raw":        return h
    if name == "ema":        return ema(h, 0.3)
    if name == "ap_only":    return 0.49 * ema(h, 0.3)
    if name == "const1":     return np.ones_like(h)
    e = ema(h, 0.3)
    if name == "early":      # extrapolate 1 s ahead with the smoothed slope
        e = np.clip(e + 1.0 * FPS * np.r_[0, np.diff(ema(e, 0.5))], 0, 1)
    on = hysteresis(e, t_on, t_on - 0.2)
    base = 0.49 * np.clip(ema(h, 0.3), 0, 1)                 # ranking info below theta
    return np.where(on, 0.5 + 0.49 * np.clip(e, 0, 1), base)

def run(strategy, reps=30, t_on=0.6, **kw):
    out = []
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        vs = world(rng, **kw)
        res = score_b([dict(t=v["t"], score=strat(strategy, v, t_on), acc=v["acc"], nm=v["nm"]) for v in vs])
        if res: out.append(res)
    k = ["ap", "P", "R", "f1", "mtta", "score_b", "n_alarms"]
    return {x: np.mean([o[x] for o in out]) for x in k} | {"sd_b": np.std([o["score_b"] for o in out])}


if __name__ == "__main__":
    print("=== Mechanics (single 60 s video, accident at s=30 lasting to e=36; near_miss [45,47]) ===")
    A, NM = [(30.0, 36.0)], [(45.0, 47.0)]
    demos = {
        "ideal: score 0.9 on [25,30) only":                       lambda t: box(t, 25, 30),
        "ramp starts s-10 (0.5->0.9 over [20,30))":               lambda t: np.where((t >= 20) & (t < 30), 0.5 + 0.4 * (t - 20) / 10, 0.02),
        "flat 0.9 on [20,30) (early, not ranked)":                lambda t: box(t, 20, 30),
        "late: 0.9 on [29,30)":                                   lambda t: box(t, 29, 30),
        "alarm starts AT contact s (ignored) [30,33)":            lambda t: box(t, 30, 33),
        "alarm held through accident [27,36] then off":           lambda t: box(t, 27, 36.04),
        "held 6 s past accident end [27,42)":                     lambda t: box(t, 27, 42),
        "merge trap: blip [18.0,18.5) + true [20.3,30)":          lambda t: np.maximum(box(t, 18, 18.5), box(t, 20.3, 30)),
        "no trap: blip [17.0,17.5) + true [20.3,30) (gap>2s)":    lambda t: np.maximum(box(t, 17, 17.5), box(t, 20.3, 30)),
        "2 alarms in window: [21,22) + [25,30)":                  lambda t: np.maximum(box(t, 21, 22), box(t, 25, 30)),
        "true [26,30) + alarm on near-miss [43,47)":              lambda t: np.maximum(box(t, 26, 30), box(t, 43, 47)),
        "true [26,30) + alarm 6 s before near-miss [39,40)":      lambda t: np.maximum(box(t, 26, 30), box(t, 39, 40)),
        "true [26,30) + 1 FP alarm elsewhere [5,6)":              lambda t: np.maximum(box(t, 26, 30), box(t, 5, 6)),
        "constant 1.0":                                           lambda t: np.ones_like(t),
        "score 0.4999 on [25,30) (rounds to 0.5!)":               lambda t: box(t, 25, 30, 0.49996),
        "score 0.49 on [25,30), AP only":                         lambda t: box(t, 25, 30, 0.49),
    }
    print(f"  {'case':55s} {'AP':>6} {'P':>5} {'R':>5} {'F1':>5} {'mTTA':>5} {'ScoreB':>7}")
    for name, fn in demos.items():
        r = score_b([vid(60, A, NM, fn)])
        print(f"  {name:55s} {r['ap']:6.3f} {r['P']:5.2f} {r['R']:5.2f} {r['f1']:5.2f} {r['mtta']:5.2f} {r['score_b']:7.3f}")

    print("\n=== F1_alarm lookup: n accidents, m matched, k false alarms ===")
    for n in [1, 2, 3, 5, 10]:
        row = []
        for m, k in [(n, 0), (n, 1), (n, 2), (n, 4), (max(n - 1, 0), 0), (max(n - 1, 0), 2)]:
            P = m / (m + k) if m + k else 0; R = m / n
            row.append(f"m{m}k{k}={2*P*R/(P+R) if P+R else 0:.2f}")
        print(f"  n={n:<3} " + "  ".join(row))

    hdr = f"  {'strategy':34s} {'AP':>6} {'P':>5} {'R':>5} {'F1':>5} {'mTTA':>5} {'alarms':>6} {'ScoreB':>7} {'sd':>5}"
    for label, kw in [("FEW accidents (3 acc, 6 near-miss, 12x5 min)", dict(n_acc=3, n_nm=6)),
                      ("MANY accidents (10 acc, 10 near-miss, 12x5 min)", dict(n_acc=10, n_nm=10))]:
        print(f"\n=== Strategy comparison, {label}; 30 random test sets ===\n" + hdr)
        for s, t_on in [("const1", 0), ("raw", 0), ("ema", 0), ("ap_only", 0), ("hyst", 0.5), ("hyst", 0.6), ("hyst", 0.7),
                        ("hyst", 0.8), ("early", 0.6), ("early", 0.7), ("early", 0.8)]:
            r = run(s, t_on=t_on, **kw)
            nm = s if s not in ("hyst", "early") else f"{s} T_on={t_on}"
            print(f"  {nm:34s} {r['ap']:6.3f} {r['P']:5.2f} {r['R']:5.2f} {r['f1']:5.2f} {r['mtta']:5.2f} {r['n_alarms']:6.1f} {r['score_b']:7.3f} {r['sd_b']:5.3f}")

    print("\n=== AP vs how early the signal rises (perfectly separable background, flat score on last L s before s) ===")
    for L in [0.5, 1, 2, 3, 4, 5, 7]:
        r = score_b([vid(300, [(100.0, 106.0), (200.0, 205.0)], [], lambda t, L=L: np.where(((t >= 100 - L) & (t < 100)) | ((t >= 200 - L) & (t < 200)), 0.9, 0.02))])
        print(f"  lead L={L:<4}s  AP={r['ap']:.3f}  F1={r['f1']:.2f}  mTTA={r['mtta']:.1f}  ScoreB={r['score_b']:.3f}")
    print("  same but graded pre-alarm: 0.3 on [s-5, s-L), 0.9 on [s-L, s):")
    for L in [0.5, 1, 2, 3]:
        fn = lambda t, L=L: np.select([((t >= 100 - L) & (t < 100)) | ((t >= 200 - L) & (t < 200)),
                                       ((t >= 95) & (t < 100)) | ((t >= 195) & (t < 200))], [0.9, 0.3], 0.02)
        r = score_b([vid(300, [(100.0, 106.0), (200.0, 205.0)], [], fn)])
        print(f"  lead L={L:<4}s  AP={r['ap']:.3f}  ScoreB={r['score_b']:.3f}")

    print("\n=== Pooled-AP calibration: same detector, per-video background offset |N(0,sd)| added (FEW scenario, hyst 0.7) ===")
    for sd in [0.0, 0.05, 0.1, 0.2]:
        r = run("hyst", t_on=0.7, bg_offset_sd=sd, n_acc=3, n_nm=6)
        print(f"  offset sd={sd:<5} AP={r['ap']:.3f} P={r['P']:.2f} F1={r['f1']:.2f} ScoreB={r['score_b']:.3f}")

    print("\n=== Sensitivity to benign-conflict rate (FEW scenario) ===")
    for bph in [10, 40, 120]:
        for s, t_on in [("hyst", 0.6), ("hyst", 0.7), ("hyst", 0.8)]:
            r = run(s, t_on=t_on, benign_per_h=bph, n_acc=3, n_nm=6)
            print(f"  benign/h={bph:<4} {s} T_on={t_on}: AP={r['ap']:.3f} P={r['P']:.2f} R={r['R']:.2f} F1={r['f1']:.2f} mTTA={r['mtta']:.2f} ScoreB={r['score_b']:.3f}")
