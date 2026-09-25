"""budget.py - per-video wall-time model for the WIUT CV harness (Part A + Part B share ONE 3.0x deadline).

    python budget.py                               # both profiles, machine g4dn2x_est (central estimate)
    python budget.py --machine wsl_laptop          # measured dev-laptop numbers (Linux wheels, 8 vCPU)
    python budget.py --machine g4dn2x_pess --dur 340
    python budget.py --csv sweep_target.csv        # decode costs measured by dbench.py ON THE TARGET (slowdown=1)

Units: ms per SOURCE frame (decode side) or ms per processed image (model side). The model is linear in the
duration, so results print as 'x duration' (harness limit 3.0x, planning target 2.4x = 0.6x margin).

Provenance
  MEASURED  dbench.py in WSL2 Ubuntu 22.04 (8 vCPU of an i5-12450H), opencv-python-headless 4.13.0.92 (FFmpeg 8.0.1),
            av 17.1.0; clips synth_cabac.mp4 (H.264 High 4:2:2 10-bit 148 Mbps 29.97p, P-only GOP 15),
            dlx/origB.mp4 (same format, IBBP GOP 30, 137 Mbps) and c1080.mp4 (1080p25 4:2:0, 4 Mbps).
            Best-of-N wall times; the laptop was shared with other jobs, so treat them as +-25 %.
            cpu_micro.py (Windows, same laptop) for resize / tracker costs.
  ESTIMATE  CPU_SLOWDOWN scales laptop numbers to other CPUs (PassMark ratios, see report).
            GPU costs = Ultralytics YOLO26 T4-TensorRT10 table x PT_OVER_TRT (PyTorch FP16) - replace with
            bench_t4_gpu.py output. VLM / clip-classifier costs are rough guesses.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass, replace

# ---------------------------------------------------------------- decode side, ms per source frame (MEASURED)
PROFILES = {
    # harness_read : cv2.VideoCapture(path).read() on EVERY frame = run_submission.run_risk (fixed cost)
    # dec          : PyAV thread_type AUTO, decode only        dec_slf: + skip_loop_filter=all (-10..17 %)
    # dec_nonref   : skip_frame=NONREF + skip_loop_filter (IBBP pyramid: 69 % of frames out, -35 % CPU)
    # ref[W]       : EXTRA per used frame for frame.reformat(W, H, 'bgr24') (one swscale call)
    # gray_small   : EXTRA per frame for 480x270 gray (static-scene engine)
    # step_resize[W]: cv2.resize (+ letterbox pad) of the full-res BGR frame that step() receives
    "4k2997_422_10": dict(fps=29.97, harness_read=31.0, dec=20.5, dec_slf=18.6, dec_nonref=13.5,
                          ref={640: 4.0, 960: 5.0, 1280: 6.0}, gray_small=3.0,
                          step_resize={640: 1.5, 960: 2.5, 1280: 4.0}),
    "1080p25_420": dict(fps=25.0, harness_read=4.1, dec=1.8, dec_slf=1.6, dec_nonref=1.2,
                        ref={640: 1.0, 960: 1.2, 1280: 1.5}, gray_small=0.5,
                        step_resize={640: 0.6, 960: 1.0, 1280: 1.5}),
}
CPU_SLOWDOWN = {                 # wall-time multiplier vs the laptop's 8-vCPU WSL slice (ESTIMATES except 1.0)
    "wsl_laptop": 1.0,           # measured
    "modern8c_est": 0.8,         # 8 physical Zen4 / Golden Cove class cores
    "g4dn2x_est": 1.9,           # AWS g4dn.2xlarge = 1xT4, 8 vCPU = 4 cores Xeon P-8259L, 32 GiB (central)
    "g4dn2x_pess": 2.3,          # PassMark whole-chip ratio (worst plausible)
    "g4dn2x_opt": 1.5,           # single-thread ratio minus HT benefit (best plausible)
}

# ---------------------------------------------------------------- model side, ms per image on a T4 (ESTIMATE)
TRT10_640 = {"n": 1.7, "s": 2.5, "m": 4.7}   # Ultralytics README: YOLO26 detect, T4 TensorRT10 FP16, b1, 640
PT_OVER_TRT = 1.8                             # PyTorch FP16 per image at batch 8 vs TensorRT FP16 (guess)
B1_FLOOR = 5.0                                # per-call floor at batch 1 (Python + launches, guess)
AREA = {640: 640 * 384 / 640 ** 2, 960: 960 * 544 / 640 ** 2, 1280: 1280 * 736 / 640 ** 2}  # 16:9 letterbox


def gpu_ms(model: str, size: int, batch: int) -> float:
    per = TRT10_640[model] * PT_OVER_TRT * AREA[size]
    return max(per, B1_FLOOR) if batch == 1 else per + 0.3


TRACK_MS, RULES_MS, BG_MS = 1.2, 0.5, 2.0   # ByteTrack-like 40 tracks (measured 1.18 ms), rules, bg model


@dataclass(frozen=True)
class Cfg:
    a_model: str = "s"; a_size: int = 1280; a_stride: int = 3; a_slf: bool = True; a_nonref: bool = False
    b_model: str = "n"; b_size: int = 640; b_stride: int = 3; b_overlap: bool = True; b_abort: bool = False
    refine_windows: int = 20; static_5hz: bool = True; fire_1hz: bool = False; clip_cls: int = 0; vlm_q: int = 0


def cost(profile: str, k: float, c: Cfg, dur: float = 300.0) -> dict:
    p = PROFILES[profile]; F = p["fps"]; N = F * dur
    dec = (p["dec_nonref"] if c.a_nonref else p["dec_slf"] if c.a_slf else p["dec"]) * k
    # Part A pass 1: decode thread -> queue -> batched GPU (CPU and GPU overlap; 10 % pipeline loss)
    nA = N / c.a_stride
    per_used = (p["ref"][c.a_size] + TRACK_MS + RULES_MS) * k
    A1 = 1.10 * max(N * dec + nA * per_used, nA * gpu_ms(c.a_model, c.a_size, 8))
    # stride-1 densification windows, 8 s each: inline if every frame is decoded anyway, re-decode if NONREF
    nW = c.refine_windows * 8.0 * F * (1.0 - 1.0 / c.a_stride)
    redecode = c.refine_windows * 8.5 * F * p["dec_slf"] * k if c.a_nonref else 0.0
    A2 = 1.10 * max(redecode + nW * per_used, nW * gpu_ms(c.a_model, c.a_size, 8))
    S = 5 * dur * (p["gray_small"] + BG_MS) * k if c.static_5hz else 0.0
    FIRE = dur * (p["ref"][640] * k + gpu_ms("n", 640, 1)) if c.fire_1hz else 0.0
    CLS = c.clip_cls * (4.5 * F * p["dec_slf"] * k + 2.0 * F * p["ref"][640] * k + 20.0)  # re-read 4 s windows
    VLM = c.vlm_q * 3000.0                                                              # ~3 s / query (guess)
    A = A1 + A2 + S + FIRE + CLS + VLM + 2000.0 + 60 * p["harness_read"] * k           # + open/seek + 60-frame probe
    # Part B: harness decodes every frame at full resolution; step() works on every b_stride-th frame
    H = N * p["harness_read"] * k
    if c.b_abort:                       # RiskEstimator raises on its first step -> harness stops decoding
        B = 1000.0
    else:
        nB = N / c.b_stride
        step = p["step_resize"][c.b_size] * k + gpu_ms(c.b_model, c.b_size, 1) + (TRACK_MS + RULES_MS) * k
        B = (max(H, nB * step) + 0.3 * nB * p["step_resize"][c.b_size] * k) if c.b_overlap else H + nB * step
        B += N * 0.02                   # harness bookkeeping per frame
    return dict(A=A / 1000 / dur, B=B / 1000 / dur, H=H / 1000 / dur, A1=A1 / 1000 / dur,
                opt=(A2 + S + FIRE + CLS + VLM) / 1000 / dur, x=(A + B) / 1000 / dur)


RICH_A = {("n", 640): 1, ("s", 640): 2, ("n", 960): 2.2, ("s", 960): 3.3, ("m", 640): 3.3,
          ("s", 1280): 4.3, ("m", 960): 4.5, ("m", 1280): 5.5}


def richness(c: Cfg) -> float:
    """Editable utility: Part B is worth 30 % of M, so losing it (b_abort) costs the most."""
    if c.b_abort:
        b = -6.0
    else:
        b = {2: 1.5, 3: 1.1, 4: 0.7, 6: 0.3}[c.b_stride] + {("n", 640): 0.3, ("s", 640): 0.6, ("s", 960): 0.9}[(c.b_model, c.b_size)]
    return (RICH_A[(c.a_model, c.a_size)] + {2: 1.6, 3: 1.2, 4: 0.6, 6: 0.2}[c.a_stride] - (0.4 if c.a_nonref else 0)
            + b + {0: 0, 10: 0.6, 20: 1.0, 40: 1.4}[c.refine_windows] + (1.5 if c.static_5hz else 0)
            + (0.3 if c.fire_1hz else 0) + {0: 0, 30: 0.6}[c.clip_cls] + {0: 0, 10: 0.8, 20: 1.2}[c.vlm_q])


def search(profile: str, k: float, target: float, dur: float):
    best = None
    for (am, asz), ast, nonref, (bm, bsz), bst, abort, rw, fire, cls, vq in itertools.product(
            RICH_A, (2, 3, 4, 6), (False, True), (("n", 640), ("s", 640), ("s", 960)), (2, 3, 4, 6),
            (False, True), (0, 10, 20, 40), (False, True), (0, 30), (0, 10, 20)):
        if nonref and ast < 3:
            continue                      # NONREF already thins to ~every 2nd-3rd frame
        c = Cfg(am, asz, ast, True, nonref, bm, bsz, bst, True, abort, rw, True, fire, cls, vq)
        r = cost(profile, k, c, dur)
        if r["x"] <= target and (best is None or (richness(c), -r["x"]) > (richness(best[0]), -best[1]["x"])):
            best = (c, r)
    return best


def part_a_hard_stop(dur: float, part_b_pred_sec: float, margin_sec: float | None = None) -> float:
    """T_A = 3*dur - 1.2*PartB_pred - margin. PartB_pred = n_frames * t_read (60-frame cv2 probe on THIS file at
    the start of detect_events) + (n_frames / b_stride) * t_step (timed at import on a bundled clip)."""
    margin = max(8.0, 0.05 * dur) if margin_sec is None else margin_sec
    return 3.0 * dur - 1.2 * part_b_pred_sec - margin


def describe(c: Cfg) -> str:
    a = f"A {c.a_model}@{c.a_size} stride {c.a_stride}{' NONREF' if c.a_nonref else ''}{' +slf' if c.a_slf else ''}"
    a += f", {c.refine_windows} dense windows, static5Hz={c.static_5hz}, fire={c.fire_1hz}, clipcls={c.clip_cls}, vlm={c.vlm_q}"
    b = "B ABORTED (raise in step)" if c.b_abort else f"B {c.b_model}@{c.b_size} stride {c.b_stride}{' overlapped' if c.b_overlap else ''}"
    return f"{a} | {b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="g4dn2x_est", choices=list(CPU_SLOWDOWN))
    ap.add_argument("--target", type=float, default=2.4)
    ap.add_argument("--dur", type=float, default=300.0)
    ap.add_argument("--csv", help="dbench.py CSV from the target machine (best fps per path replaces 4K costs)")
    args = ap.parse_args()
    k = CPU_SLOWDOWN[args.machine]
    if args.csv:
        best = {}
        for row in csv.reader(open(args.csv)):
            if len(row) >= 7 and row[4] != "ERR":
                best[row[3]] = max(best.get(row[3], 0.0), float(row[6]))
        p = PROFILES["4k2997_422_10"]
        p["harness_read"] = 1000 / best["cv_read_default(harness)"]
        p["dec"] = 1000 / best["pyav_AUTO_decode_only"]
        p["dec_slf"] = 1000 / best.get("pyav_AUTO_skiploopfilter_decode", 1000 / p["dec"])
        k = 1.0

    for prof in PROFILES:
        D = args.dur
        print(f"\n=== {prof} | machine {args.machine} (CPU x{k}) | {D:.0f} s clip | budget {3 * D:.0f} s (3.0x), plan <= {args.target}x")
        rows = [
            ("Part B floor: harness cv2 read of every frame", None, "H"),
            ("Part B: n@640 stride 2, overlapped", Cfg(b_stride=2), "B"),
            ("Part B: n@640 stride 3, overlapped", Cfg(b_stride=3), "B"),
            ("Part B: n@640 stride 3, synchronous", Cfg(b_stride=3, b_overlap=False), "B"),
            ("Part B: s@960 stride 3, overlapped", Cfg(b_model="s", b_size=960, b_stride=3), "B"),
            ("Part A: decode+slf only (floor)", Cfg(a_model="n", a_size=640, a_stride=6, refine_windows=0, static_5hz=False), "A1"),
            ("Part A: NONREF+slf decode floor", Cfg(a_model="n", a_size=640, a_stride=6, a_nonref=True, refine_windows=0, static_5hz=False), "A1"),
            ("Part A: s@1280 stride 2", Cfg(a_stride=2, refine_windows=0, static_5hz=False), "A"),
            ("Part A: s@1280 stride 3", Cfg(a_stride=3, refine_windows=0, static_5hz=False), "A"),
            ("Part A: s@960 stride 3 NONREF", Cfg(a_size=960, a_stride=3, a_nonref=True, refine_windows=0, static_5hz=False), "A"),
            ("  + 20 dense windows x 8 s (inline)", Cfg(a_stride=3, refine_windows=20, static_5hz=False), "opt"),
            ("  + static-scene engine 5 Hz", Cfg(a_stride=3, refine_windows=0, static_5hz=True), "opt"),
            ("  + fire 1 Hz + 30 clip-cls + 10 VLM q", Cfg(a_stride=3, refine_windows=0, static_5hz=False, fire_1hz=True, clip_cls=30, vlm_q=10), "opt"),
        ]
        for name, c, key in rows:
            r = cost(prof, k, c or Cfg(), D)
            print(f"  {name:<48} {r[key]:5.2f}x  ({r[key] * D:5.0f} s)")
        best = search(prof, k, args.target, D)
        if best is None:
            print("  NOTHING fits the target: run the emergency tier (A n@640 NONREF stride 6, B aborted) and ask organizers")
        else:
            c, r = best
            print(f"  RICHEST <= {args.target}x : {describe(c)}")
            print(f"     A {r['A']:.2f}x + B {r['B']:.2f}x = {r['x']:.2f}x = {r['x'] * D:.0f} s of {3 * D:.0f} s")
        bp = cost(prof, k, Cfg(), D)["B"] * D
        print(f"  Part A hard stop (B n@640 s3 predicted {bp:.0f} s): T_A = 3*{D:.0f} - 1.2*{bp:.0f} - {max(8.0, 0.05 * D):.0f}"
              f" = {part_a_hard_stop(D, bp):.0f} s")


if __name__ == "__main__":
    main()
