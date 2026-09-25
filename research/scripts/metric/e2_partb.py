"""(e) part 2: cleaner strategy grid on the same toy world (worlds generated once per rep, reused)."""
import numpy as np
from scipy.signal import lfilter
from common import score_b
from e_partb import world, ema, FPS

def monotone_map(e, T):
    """ranking-preserving map: e<T -> [0,0.49], e>=T -> [0.5,1]. AP invariant to T."""
    e = np.clip(e, 0, 1)
    return np.where(e < T, 0.49 * e / T, 0.5 + 0.5 * (e - T) / max(1 - T, 1e-6))

def with_contact_reset(sc, v, rng, p_detect=0.7, delay=0.5):
    """after a (causally) detected contact at s+delay, force score to 0.02 until e (+1 s)."""
    sc = sc.copy(); t = v["t"]
    for s, e in v["acc"]:
        if rng.random() < p_detect:
            sc[(t >= s + delay) & (t <= e + 1.0)] = 0.02
    return sc

def max_alarm_len(sc, t, max_len=8.0, gap=2.1):
    """force a >=2 s dip after an alarm has been continuously on for max_len seconds."""
    sc = sc.copy(); on_since = None; mute_until = -1
    for i in range(len(sc)):
        if t[i] < mute_until:
            sc[i] = min(sc[i], 0.49); on_since = None; continue
        if sc[i] >= 0.5:
            if on_since is None: on_since = t[i]
            elif t[i] - on_since > max_len: mute_until = t[i] + gap; sc[i] = 0.49; on_since = None
        else:
            on_since = None
    return sc

STRATS = {}
for tau in [0.0, 0.2, 0.4, 0.8]:
    for T in [0.4, 0.5, 0.6, 0.7]:
        STRATS[f"EMA{tau:.1f}s  T={T}"] = (lambda v, rng, tau=tau, T=T:
            monotone_map(v["h"] if tau == 0 else ema(v["h"], tau), T))
for k in [0.5, 1.0, 2.0]:
    for T in [0.5, 0.6, 0.7]:
        def f(v, rng, k=k, T=T):
            e = ema(v["h"], 0.2); slope = np.r_[0, np.diff(ema(e, 0.3))] * FPS
            return monotone_map(np.clip(e + k * np.clip(slope, 0, None), 0, 1), T)
        STRATS[f"EMA0.2s+{k}s-extrap T={T}"] = f
STRATS["EMA0.2s T=0.5 +contact-reset"] = lambda v, rng: with_contact_reset(monotone_map(ema(v["h"], 0.2), 0.5), v, rng)
STRATS["EMA0.2s T=0.6 +contact-reset"] = lambda v, rng: with_contact_reset(monotone_map(ema(v["h"], 0.2), 0.6), v, rng)

def evaluate_all(reps=20, **kw):
    res = {k: [] for k in STRATS}
    for r in range(reps):
        vs = world(np.random.default_rng(1000 + r), **kw)
        for name, fn in STRATS.items():
            rng = np.random.default_rng(r)
            out = score_b([dict(t=v["t"], score=fn(v, rng), acc=v["acc"], nm=v["nm"]) for v in vs])
            if out: res[name].append(out)
    return res

if __name__ == "__main__":
    for label, kw in [("FEW: 3 accidents, 6 near-misses, 12x5 min", dict(n_acc=3, n_nm=6)),
                      ("MANY: 10 accidents, 10 near-misses, 12x5 min", dict(n_acc=10, n_nm=10))]:
        res = evaluate_all(**kw)
        print(f"\n=== {label}; 20 random test sets; mean (sd of ScoreB) ===")
        print(f"  {'strategy':34s} {'AP':>6} {'P':>5} {'R':>5} {'F1':>5} {'mTTA':>5} {'alarms':>6} {'ScoreB':>7} {'sd':>5}")
        for name, o in res.items():
            m = {x: np.mean([q[x] for q in o]) for x in ["ap", "P", "R", "f1", "mtta", "n_alarms", "score_b"]}
            print(f"  {name:34s} {m['ap']:6.3f} {m['P']:5.2f} {m['R']:5.2f} {m['f1']:5.2f} {m['mtta']:5.2f} {m['n_alarms']:6.1f} {m['score_b']:7.3f} {np.std([q['score_b'] for q in o]):5.3f}")
