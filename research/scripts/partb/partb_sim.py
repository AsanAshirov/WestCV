"""
Synthetic study of Score_B alarm/score-shaping policies using the OFFICIAL evaluate.py
(from the starter kit). The latent-risk generator is a toy model: numbers are
illustrative, not predictions of real performance.
"""
import sys, math, itertools
import numpy as np
sys.path.insert(0, r"C:\Users\Cicada\AppData\Local\Temp\claude\C--Users-Cicada-Desktop-WestHack\ced7e0d9-be88-47bc-ac13-90898f4ae082\scratchpad\kit\wiut_cv_scripts")
import evaluate as ev

FPS = 25.0
DUR = 300.0
NV = 16          # videos per simulated test set (80 min total)
N_ACC = 8        # accidents per test set
N_NM = 16        # annotated near misses per test set
BASE = -6.0


def gen_testset(rng, benign_per_min=1.0, p_vis=0.7, lead_median=3.0, acc_mu=2.0,
                nm_mu=1.5, ben_mu=-1.0, ben_sd=1.2, aftermath=True):
    n = int(DUR * FPS)
    t = np.arange(n) / FPS
    gt, lat, contacts = {}, {}, {}
    vids = [f"v{i:02d}.mp4" for i in range(NV)]
    acc_v = rng.choice(NV, N_ACC, replace=True)
    nm_v = rng.choice(NV, N_NM, replace=True)
    for vi, vid in enumerate(vids):
        z = np.full(n, BASE)
        # AR(1) jitter
        noise = np.zeros(n); a = math.exp(-1 / (0.5 * FPS)); e = rng.normal(0, 0.3 * math.sqrt(1 - a * a), n)
        for k in range(1, n):
            noise[k] = a * noise[k - 1] + e[k]
        events, cts = [], []
        occupied = []

        def free_slot(lo, hi, span):
            for _ in range(100):
                s = rng.uniform(lo, hi)
                if all(abs(s - o) > span for o in occupied):
                    occupied.append(s); return s
            return None

        def episode(onset, peak_t, amp, tau_after, hold_until=None):
            m = (t >= onset) & (t <= peak_t)
            if peak_t > onset:
                z[m] = np.maximum(z[m], -3 + (amp + 3) * (t[m] - onset) / (peak_t - onset) + noise[m])
            end_hold = peak_t if hold_until is None else hold_until
            m2 = (t > peak_t) & (t <= end_hold)
            z[m2] = np.maximum(z[m2], amp + noise[m2])
            m3 = t > end_hold
            z[m3] = np.maximum(z[m3], BASE + (amp - BASE) * np.exp(-(t[m3] - end_hold) / tau_after) + noise[m3])

        for _ in range(int((acc_v == vi).sum())):
            s = free_slot(15, DUR - 20, 30)
            if s is None: continue
            e_ = s + rng.uniform(2, 6)
            events.append([round(s, 2), round(e_, 2), "accident"])
            if rng.random() < p_vis:
                L = float(np.clip(rng.lognormal(math.log(lead_median), 0.6), 0.4, 9.5))
                episode(s - L, s, rng.normal(acc_mu, 1.0), 1.5, hold_until=s + 1.0)
            cts.append(s + rng.uniform(0.0, 1.0))
            if aftermath:  # vehicles manoeuvring around the wreck after the annotated end
                for _k in range(rng.poisson(1.5)):
                    ta = e_ + rng.uniform(1, 25)
                    episode(ta - rng.uniform(1, 3), ta, rng.normal(ben_mu + 0.8, ben_sd), 1.0)
        for _ in range(int((nm_v == vi).sum())):
            s = free_slot(10, DUR - 10, 15)
            if s is None: continue
            e_ = s + rng.uniform(1.5, 4)
            events.append([round(s, 2), round(e_, 2), "near_miss"])
            L = float(np.clip(rng.lognormal(math.log(2.5), 0.6), 0.4, 9.5))
            episode(s - L, s + 0.5, rng.normal(nm_mu, 1.0), 1.0)
        nb = rng.poisson(benign_per_min * DUR / 60)
        for _ in range(nb):
            tp_ = rng.uniform(2, DUR - 2)
            episode(tp_ - rng.uniform(1, 4), tp_, rng.normal(ben_mu, ben_sd), 1.0)
        gt[vid] = {"duration": DUR, "fps": FPS, "events": events}
        lat[vid] = z
        contacts[vid] = cts
    return gt, lat, contacts, t


def sig(x):
    return 1 / (1 + np.exp(-x))


def policy_curve(z, t, contacts, c_on=1.0, c_off=None, min_on=0.0, max_on=None,
                 release=0.0, suppress=0.0, p_contact=0.0, rng=None, two_band=True):
    """Map latent z to a score curve.
    c_on: latent level where alarm switches on (score crosses 0.5)
    c_off: hysteresis off-level (default = c_on)
    min_on: minimum alarm duration (s); release: z must stay below c_off this long before off
    max_on: cap alarm duration (s) then force off (re-arm needs z to rise again)
    suppress: seconds of suppression after a detected contact (detected with prob p_contact)
    two_band: score in [0.5,1] iff alarm on, [0,0.5) otherwise (ranking by z inside bands)
    """
    c_off = c_on if c_off is None else c_off
    z = z.copy()
    if suppress > 0 and rng is not None:
        for ct in contacts:
            if rng.random() < p_contact:
                m = (t >= ct + 0.5) & (t < ct + 0.5 + suppress)
                z[m] = BASE
    if not two_band and min_on == 0 and max_on is None and c_off == c_on and release == 0:
        return sig(z - c_on)
    n = len(z); on = np.zeros(n, bool)
    state, t_on, t_below, blocked = False, 0.0, None, False
    for k in range(n):
        tk = t[k]
        if not state:
            if blocked and z[k] < c_off:
                blocked = False
            if z[k] >= c_on and not blocked:
                state, t_on, t_below = True, tk, None
        else:
            if z[k] < c_off:
                t_below = tk if t_below is None else t_below
            else:
                t_below = None
            dur = tk - t_on
            if max_on is not None and dur >= max_on:
                state, blocked = False, True
            elif dur >= min_on and t_below is not None and tk - t_below >= release:
                state = False
        on[k] = state
    r = sig(z - c_on)  # monotone rank signal
    score = np.where(on, 0.5 + 0.4999 * r, 0.4999 * r)
    return score


def run(policy_kw, gen_kw=None, seeds=range(12)):
    gen_kw = gen_kw or {}
    res = []
    for sd in seeds:
        rng = np.random.default_rng(1000 + sd)
        gt, lat, contacts, t = gen_testset(rng, **gen_kw)
        preds = {}
        prng = np.random.default_rng(5000 + sd)
        for vid in gt:
            sc = policy_curve(lat[vid], t, contacts[vid], rng=prng, **policy_kw)
            sc = np.round(sc, 4)  # harness rounds to 4 decimals
            preds[vid] = {"events": [], "risk": [[round(float(a), 4), float(b)] for a, b in zip(t, sc)]}
        b = ev.evaluate_part_b(gt, preds)
        res.append((b["score_b"], b["ap"], b["f1_alarm"], b["mtta_sec"], b["alarm_precision"], b["alarm_recall"], b["n_alarms"]))
    r = np.array(res)
    return r.mean(0), r.std(0)


def fmt(name, m):
    return f"{name:<48} ScoreB={m[0]:.3f} AP={m[1]:.3f} F1={m[2]:.3f} mTTA={m[3]:.2f}s P={m[4]:.2f} R={m[5]:.2f} alarms={m[6]:.1f}"


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("thr", "all"):
        print("== 1. alarm threshold sweep (plain sigmoid, no hysteresis)")
        for c in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            m, s = run(dict(c_on=c, two_band=False))
            print(fmt(f"sigmoid, c_on={c}", m))
    if which in ("hyst", "all"):
        print("== 2. hysteresis / hold / cap at c_on=1.0")
        for kw in [dict(c_on=1.0, two_band=False),
                   dict(c_on=1.0),
                   dict(c_on=1.0, c_off=0.0, release=0.5),
                   dict(c_on=1.0, c_off=0.0, release=1.0, min_on=1.0),
                   dict(c_on=1.0, c_off=-1.0, release=1.0, min_on=2.0),
                   dict(c_on=1.0, c_off=0.0, release=1.0, min_on=1.0, max_on=6.0),
                   dict(c_on=1.0, c_off=0.0, release=1.0, min_on=1.0, max_on=3.0),
                   dict(c_on=1.0, c_off=0.0, release=1.0, min_on=1.0, suppress=30, p_contact=0.7),
                   ]:
            m, s = run(kw)
            print(fmt(str(kw), m))
