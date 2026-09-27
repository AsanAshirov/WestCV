"""Event rules: scene context -> per-class segments [(start, end)].

Every rule is written in scene units (box heights / sizes, seconds) so that it does not depend
on this camera's pixel scale. Simultaneous events of one class are merged into one segment,
as the organizers label them.
"""
from __future__ import annotations

import numpy as np

from .scene import PERSON, VEHICLE, BIG_VEHICLES, Scene, feet, inside, runs_from_flags, signed_dist, sizes, speeds
from .segments import merge


# ---------------------------------------------------------------- jaywalking
def jaywalking(sc: Scene, depth_min=0.15, zebra_margin=0.1, gap=0.5, min_seg=3.0, step=0.1):
    """Class timeline: at time t somebody is jaywalking if a pedestrian (not a rider, not inside a
    vehicle, feet visible) stands on the carriageway deeper than depth_min heights and outside every
    zebra by more than zebra_margin heights. Timeline gaps > gap s split events; events shorter than
    min_seg s are dropped. Tuned on the team labels (tools/eval/tune_timeline.py, renders/eval):
    on the 0.1 s timeline the model is as close to the labels as two humans are to each other (~0.88);
    segment F1 is 0.47 over the 4 dev videos, two humans agree at 0.54-0.60 on the doubly labelled
    ones: the remaining gap is in where events are cut."""
    dur = sc.meta["n_frames"] / sc.fps
    g = np.zeros(int(np.ceil(dur / step)) + 1, bool)
    for tid, tr in sc.tracks.items():
        if tid % 3 != PERSON or len(tr) < 5 or tid in sc.riders:
            continue
        keep = np.array([(int(r[0]), tid) not in sc.occupant for r in tr]) & (tr[:, 5] < sc.meta["height"] - 8)
        tr = tr[keep]
        if len(tr) < 5:
            continue
        p = feet(tr)
        h = (tr[:, 5] - tr[:, 3]).clip(1)
        on_road = sc.road_depths(p) > depth_min * h
        for i in np.flatnonzero(on_road):
            if any(signed_dist(z, p[i]) >= -zebra_margin * h[i] for z in sc.crosswalks.values()):
                continue
            t = tr[i, 0] / sc.fps
            g[int(t / step):int(np.ceil((t + sc.dt) / step))] = True
    idx = np.flatnonzero(g)
    if not len(idx):
        return []
    segs, a, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if (i - prev) * step > gap:
            segs.append((a * step, (prev + 1) * step))
            a = i
        prev = i
    segs.append((a * step, (prev + 1) * step))
    return [(x, min(y, dur)) for x, y in segs if y - x >= min_seg]


# ---------------------------------------------------------------- approach 4: stop line 4, head 7
def _approach4(sc: Scene):
    a4, b4 = sc.stop4
    n = np.array([-(b4[1] - a4[1]), b4[0] - a4[0]], np.float64)
    n /= np.linalg.norm(n)
    if np.dot(sc.crosswalks["1"].mean(0) - a4, n) < 0:
        n = -n
    far = float(((sc.crosswalks["1"] - a4) @ n).max())
    return a4, b4, n, far


def stop_line(sc: Scene, margin=0.15, min_stop=1.0, speed_thr=0.08, along_max=1.15):
    """Vehicle of approach 4 standing with its front past stop line 4 (not through zebra 1) while
    head 7 is red. Start = the stop (or the red onset if it stopped earlier), end = green onset.
    along_max > 1: the drawn line stops short of the median nose, the lane next to it is included."""
    a4, b4, n, far = _approach4(sc)
    ups = sc.sig.veh_green_onsets
    segs = []
    for tid, tr in sc.tracks.items():
        if tid % 3 != VEHICLE or len(tr) < 10:
            continue
        t, p, s, v = speeds(tr, sc.fps)
        d = (p - a4) @ n
        along = ((p - a4) @ (b4 - a4)) / np.dot(b4 - a4, b4 - a4)
        hgt = tr[:, 5] - tr[:, 3]
        flags = (d > margin * hgt) & (d < far) & (along > -0.05) & (along < along_max) & (v < speed_thr)
        for a, b in runs_from_flags(t, flags, 0.5, min_stop, sc.dt):
            ts = sc.sig.t[(sc.sig.t >= a) & (sc.sig.t <= b)]
            red = [x for x in ts if sc.sig.veh_red_for(x) > 0]
            if not red:
                continue
            start = max(a, red[0])
            nxt = ups[ups > start]
            end = float(nxt[0]) if len(nxt) else float(sc.meta["n_frames"] / sc.fps)
            if end - start >= min_stop:
                segs.append((start, end))
    return merge(segs)


def red_light(sc: Scene, min_red=0.5, pre_green=1.0):
    """Vehicle front crosses stop line 4 after head 7 has shown red for > min_red s (and the green
    is not about to start); end = the front leaves zebra 1. A vehicle that stops past the line
    before entering is stop_line, not red_light."""
    a4, b4, n, far = _approach4(sc)
    ups = sc.sig.veh_green_onsets
    segs = []
    for tid, tr in sc.tracks.items():
        if tid % 3 != VEHICLE or len(tr) < 5:
            continue
        t, p, s, v = speeds(tr, sc.fps)
        d = (p - a4) @ n
        along = ((p - a4) @ (b4 - a4)) / np.dot(b4 - a4, b4 - a4)
        for i in range(1, len(tr)):
            if not (d[i - 1] < 0 <= d[i] and -0.05 <= along[i] <= 1.05):
                continue
            tc = t[i - 1] + (t[i] - t[i - 1]) * (-d[i - 1]) / max(1e-6, d[i] - d[i - 1])
            nxt = ups[ups > tc]
            if sc.sig.veh_red_for(tc) <= min_red or (len(nxt) and nxt[0] - tc < pre_green):
                break
            after = np.flatnonzero((d > far) & (np.arange(len(d)) >= i))
            if len(after) and (v[i:after[0]] < 0.08).sum() * sc.dt <= 1.0:
                segs.append((tc, t[after[0]] + sc.dt))
            break
    return merge(segs)


# ---------------------------------------------------------------- failure to yield
def failure_to_yield(sc: Scene, near=3.0, min_hits=5, min_speed=0.6, ahead_min=-0.2, lateral=0.55, zebras=None,
                     return_detail=False):
    """A moving car/bus/truck on a zebra while a walking pedestrian (not a rider) is on the same
    zebra in front of it (within `near` vehicle sizes ahead and `lateral` sizes to the side of its
    path). Start = the vehicle enters the zebra, end = it leaves."""
    peds = {}
    for tid, tr in sc.tracks.items():
        if tid % 3 != PERSON or tid in sc.riders or len(tr) < 3:
            continue
        t, p, s, v = speeds(tr, sc.fps)
        for r, (x, y) in zip(tr, p):
            if (int(r[0]), tid) in sc.occupant:
                continue
            for k, poly in sc.crosswalks.items():
                if inside(poly, (x, y)):
                    peds.setdefault(int(r[0]), []).append((x, y, k, tid))
    segs, detail = [], []
    for tid, tr in sc.tracks.items():
        if tid % 3 != VEHICLE or len(tr) < 5 or np.median(tr[:, 7]) not in BIG_VEHICLES:
            continue
        t, p, s, v = speeds(tr, sc.fps)
        vel = np.gradient(p, t, axis=0) if len(t) > 1 else np.zeros_like(p)
        for k, poly in sc.crosswalks.items():
            if zebras is not None and k not in zebras:
                continue
            flags = np.zeros(len(tr), bool)
            hits, who = 0, set()
            for i, r in enumerate(tr):
                x1, y1, x2, y2 = r[2:6]
                yb = y2 - 0.1 * (y2 - y1)
                if not any(inside(poly, q) for q in (((x1 + x2) / 2, yb), (x1 + 0.2 * (x2 - x1), yb), (x2 - 0.2 * (x2 - x1), yb))):
                    continue
                flags[i] = True
                if v[i] < min_speed:
                    continue
                u = vel[i] / (np.linalg.norm(vel[i]) + 1e-6)
                n = np.array([-u[1], u[0]])
                c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                for px, py, pk, pid in peds.get(int(r[0]), []):
                    if pk != k or (x1 <= px <= x2 and y1 <= py <= y2):
                        continue
                    q = (np.array([px, py]) - c) / s[i]
                    if ahead_min <= q @ u <= near and abs(q @ n) <= lateral:
                        hits += 1
                        who.add(pid)
            if hits >= min_hits:
                for a, b in runs_from_flags(t, flags, 0.5, 0.3, sc.dt):
                    segs.append((a, b))
                    detail.append((a, b, tid, k, sorted(who)))
    return (merge(segs), detail) if return_detail else merge(segs)


# ---------------------------------------------------------------- stopped vehicle
def static_vehicles(sc: Scene, iou_min=0.6, max_gap=10.0, min_conf=0.3):
    """Clusters of vehicle detections that keep the same box (IoU >= iou_min) over time, surviving
    track breaks and occlusions shorter than max_gap s. Returns [(box, times)]."""
    d = sc.dets[np.isin(sc.dets[:, 6], BIG_VEHICLES) & (sc.dets[:, 5] >= min_conf)]
    d = d[np.argsort(d[:, 0], kind="stable")]
    frames, starts = np.unique(d[:, 0].astype(int), return_index=True)
    ends = list(starts[1:]) + [len(d)]
    boxes, last, times = [], [], []  # per cluster: current box, last time seen, times
    gap_f = max_gap * sc.fps
    for f, a, b in zip(frames, starts, ends):
        cur = d[a:b, 1:5]
        if boxes:
            B = np.array(boxes)
            active = np.array(last) >= f - gap_f
            ix = np.clip(np.minimum(cur[:, None, 2], B[None, :, 2]) - np.maximum(cur[:, None, 0], B[None, :, 0]), 0, None)
            iy = np.clip(np.minimum(cur[:, None, 3], B[None, :, 3]) - np.maximum(cur[:, None, 1], B[None, :, 1]), 0, None)
            inter = ix * iy
            area_c = (cur[:, 2] - cur[:, 0]) * (cur[:, 3] - cur[:, 1])
            area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
            iou = inter / np.maximum(1.0, area_c[:, None] + area_b[None, :] - inter)
            iou[:, ~active] = 0
        else:
            iou = np.zeros((len(cur), 0))
        used = set()
        for i in np.argsort(-(iou.max(axis=1) if iou.shape[1] else np.zeros(len(cur)))):
            j = int(np.argmax(iou[i])) if iou.shape[1] else -1
            if j >= 0 and iou[i, j] >= iou_min and j not in used:
                used.add(j)
                boxes[j] = 0.9 * np.asarray(boxes[j]) + 0.1 * cur[i]
                last[j] = f
                times[j].append(f / sc.fps)
            else:
                boxes.append(cur[i].copy())
                last.append(f)
                times.append([f / sc.fps])
    return [(np.asarray(bx), np.array(ts)) for bx, ts in zip(boxes, times) if len(ts) >= 10]


def stopped_vehicle(sc: Scene, min_stop=10.0, max_gap=10.0, min_density=0.4, gap=8.0, together_dist=5.0):
    """A vehicle standing on the carriageway >= min_stop s (static detection cluster, robust to
    track breaks), not a queue: at the moment it leaves, fewer than two neighbours leave too.
    Vehicles standing since the start of the video count. Class timeline = union, gaps < gap s."""
    stays = []
    for box, ts in static_vehicles(sc, max_gap=max_gap):
        breaks = np.flatnonzero(np.diff(ts) > max_gap)
        for a, b in zip(np.r_[0, breaks + 1], np.r_[breaks, len(ts) - 1]):
            t0, t1 = ts[a], ts[b] + sc.dt
            if t1 - t0 >= min_stop and (b - a + 1) * sc.dt / (t1 - t0) >= min_density:
                stays.append((t0, t1, box))
    dur = sc.meta["n_frames"] / sc.fps
    segs = []
    for t0, t1, box in stays:
        fx, fy = (box[0] + box[2]) / 2, box[3]
        if not sc.on_road(fx, fy):
            continue
        s = np.sqrt((box[2] - box[0]) * (box[3] - box[1]))
        if t1 < dur - 1.0:
            together = sum(1 for u0, u1, ob in stays if ob is not box and abs(u1 - t1) < 4.0
                           and np.hypot((ob[0] + ob[2]) / 2 - fx, ob[3] - fy) < together_dist * s)
            if together >= 2:
                continue  # left together with its neighbours: a queue released by a signal
        segs.append((t0, t1))
    return merge(segs, gap)


# ---------------------------------------------------------------- congestion
def congestion(sc: Scene, min_count=8, win=5, min_dur=30.0, gap=5.0, speed_thr=0.05):
    """Standstill: at least min_count vehicles stand still on the carriageway at once (median over
    win seconds) for >= min_dur s. Tuned on the team labels (C3896, C3897)."""
    dur = sc.meta["n_frames"] / sc.fps
    T = np.arange(0.0, dur, 1.0)
    cnt = np.zeros(len(T))
    for tid, tr in sc.tracks.items():
        if tid % 3 != VEHICLE or len(tr) < 5:
            continue
        t, p, s, v = speeds(tr, sc.fps)
        still = (v < speed_thr) & (sc.road_depths(p) > 0)
        cnt[np.unique(np.minimum(t[still].astype(int), len(T) - 1))] += 1
    h = win // 2
    padded = np.pad(cnt, h, mode="edge")
    smooth = np.array([np.median(padded[i:i + win]) for i in range(len(T))])
    segs = runs_from_flags(T, smooth >= min_count, 0.0, 0.0, 1.0)
    return [(a, min(b, dur)) for a, b in merge(segs, gap) if b - a >= min_dur]


# ---------------------------------------------------------------- solid line crossing
def solid_line_crossing(sc: Scene, thr=0.2, min_dur=1.5, min_speed=0.3, gap=0.5, extend=0.1, side=0.2, win=1.0,
                        lines=None):
    """A moving vehicle crosses a painted solid line: its ground centre is within `thr` vehicle widths
    of the line (the body over the paint) on the solid segment for >= min_dur s, and it is on the
    other side of the line afterwards (median signed distance in the `win` s before and after the run
    changes sign by more than `side` widths). Riding along the line without changing side is not a
    crossing (task definition). On the dev labels this keeps F1 (0.12 -> 0.15) with 8 instead of 57
    predictions."""
    segs = []
    for name, (a, b) in sc.solid_lines.items():
        if lines is not None and name not in lines:
            continue
        d = b - a
        L = np.linalg.norm(d)
        u, n = d / L, np.array([-d[1], d[0]]) / L
        for tid, tr in sc.tracks.items():
            if tid % 3 != VEHICLE or len(tr) < 10:
                continue
            t, p, s, v = speeds(tr, sc.fps)
            c = np.stack([(tr[:, 2] + tr[:, 4]) / 2, tr[:, 5] - 0.3 * (tr[:, 5] - tr[:, 3])], axis=1)
            w = (tr[:, 4] - tr[:, 2]).clip(1)
            along = (c - a) @ u / L
            sd = (c - a) @ n / w
            flags = (np.abs(sd) < thr) & (along > -extend) & (along < 1 + extend) & (v > min_speed)
            for r0, r1 in runs_from_flags(t, flags, gap, min_dur, sc.dt):
                pre, post = sd[(t >= r0 - win) & (t < r0 + 0.3)], sd[(t > r1 - 0.3) & (t <= r1 + win)]
                if len(pre) and len(post):
                    before, after = np.median(pre), np.median(post)
                    if before * after < 0 and abs(after - before) > side:
                        segs.append((r0, r1))
    return merge(segs)


RULES = {
    "jaywalking": jaywalking,
    "stop_line": stop_line,
    "red_light": red_light,
    "failure_to_yield": failure_to_yield,
    "stopped_vehicle": stopped_vehicle,
    "congestion": congestion,
    "solid_line_crossing": solid_line_crossing,
}


# classes that only need road users and the carriageway: they also run on an unknown camera
AUTO_OK = ("stopped_vehicle", "congestion")
# classes that read the signal timeline: off when the signal heads could not be read
NEEDS_SIGNALS = ("stop_line", "red_light")


def detect(sc: Scene, classes) -> list[list]:
    """All enabled rules; a rule that fails loses only its own class, not the video. On an unknown
    camera (sc.auto) only the classes that need no zebras, lines or signals are run; without a signal
    timeline the signal classes are skipped."""
    import sys
    import traceback
    events = []
    for c in classes:
        if c not in RULES or (sc.auto and c not in AUTO_OK) or (sc.sig is None and c in NEEDS_SIGNALS):
            continue
        try:
            events += [[a, b, c] for a, b in RULES[c](sc)]
        except Exception:
            print(f"[westcv] rule {c} failed: {traceback.format_exc()}", file=sys.stderr)
    return events
