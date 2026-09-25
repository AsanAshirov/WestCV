<!-- source: deep-research workflow wf_294a4b01-fc2, agent research:metric -->

# Scoring mathematics and the strategy it implies (WIUT CV Track)

## 0. TL;DR: the ten decisions that matter most

1. **The package must run.** If it does not, M is zeroed and "Runs as submitted" is lost. In realistic scenarios that costs 0.18 to 0.26 elimination points, which is more than any modelling gain. A working live demo is worth 0.075, the same as +0.18 Score A. The mathematics says reliability and the website come first.
2. **Enable a class only when p·f > (1−p)·q·S̄.** Here p = P(class is in the test set), f = expected F1 if it is there, q = P(at least one false positive over the whole test set if it is absent), and S̄ = mean F1 of your other classes. The **number of classes |C| cancels out**, so the break-even point does not depend on how many classes there are. For rare or uncertain classes, use precision-first thresholds so that q ≤ 0.1–0.2.
3. **Emit a candidate segment only when P(real) × E[fraction of the 3 τ it passes] > F1_c/2.** A badly localised segment (IoU < 0.5) often **lowers** the score, because it counts as a false positive and still leaves the ground truth unmatched.
4. **Boundaries decide F1@0.7.** For a 3 s event, a 0.53 s shift already fails τ = 0.7. For a 120 s event, 21 s is tolerated. Rules with confirmation windows (stopped ≥ 10 s, congestion) must **back-date the start** to the onset. Part A is offline, so this is allowed.
5. **Post-processing is worth more than model tweaks.** In a toy simulation, gap-merge plus min-duration took the τ-mean F1 from 0.28 to 0.93 for short events and from 0.03 to 0.93 for long ones. Always merge overlapping same-class segments yourself. The harness **drops the later-starting overlapping segment entirely**; it does not merge them.
6. **Per-class F1 is extremely noisy.** A class with 1 test instance has sd ≈ 0.36 and a 30 % chance of F1 = 0, and it still weighs 1/|C| of Score A. Do not over-tune on 4 sample videos.
7. **Part B is dominated by F1_alarm, not AP or mTTA.** In realistic regimes AP is about 0.03–0.15 and mTTA/W about 0.03–0.08. Alarm precision is the lever: count false alarms per hour.
8. **Part B shaping:** use two channels. A ranking channel (≤ 0.49) is always on, for AP. An alarm channel (≥ 0.5) is only for confident, imminent conflicts. Do not add confirmation delays of more than ~0.2 s. Keep a single alarm per conflict, drop the score right after contact, and cap alarm age at ~8 s. Never return 0.49996: the harness rounds it to 0.5.
9. **Guard the time budget inside the code.** Over budget means both the events **and** the risk curve for that video are emptied. Part B runs after Part A, so a slow Part A starves Part B.
10. **Return Python floats from `step`.** Under NumPy 2.4, `float(np.array([x]))` raises. The harness catches it and silently records 0.0 for that frame.

---

## 1. Source of truth: the exact metric is available

The organizers' Drive folder ("CV": Videos.pdf, the task PDF, `wiut_cv_scripts.zip`) was downloaded into the shared scratchpad by another agent. It is at `...\scratchpad\kit\wiut_cv_scripts\` and contains `evaluate.py`, `run_submission.py`, `solution.py`, `README.md` and `examples/`. The PDF says `evaluate.py` "is the exact implementation". I used it directly and also wrote a vectorised copy (`scratchpad\metric\common.py`). On 40 random multi-video cases the copy matches the official Score B to 2.2e-16. The official AP equals `sklearn.metrics.average_precision_score`, including tie handling.

**Details in the code that the PDF does not spell out:**

| Detail | Where | Consequence |
|---|---|---|
| Classes scored = GT classes ∪ predicted classes | `evaluate_part_a` | Confirms the dilution rule |
| `M = Score_A` if the test set has **no accidents** | `evaluate()` | Part B only counts if accidents exist |
| Format check allows `end ≤ duration + 0.5 s`; overlap test is `s2 < e1`, so touching segments are allowed | `validate` | [5,8] and [8,9] of the same class are both valid |
| Harness coerces types with `float()`/`str()`, clamps `end` to `duration = CAP_PROP_FRAME_COUNT/fps`, rounds times to 3 decimals, and drops a segment if `start ≥ end` after clamping | `clean_events` | Ends past the video end are clamped, not dropped |
| Same-class overlap: sort by (label, start); a later segment with `s < last_end` is **dropped** | `clean_events` | You must union overlapping segments yourself |
| Risk recorded as `round(t,4), round(clip(score),4)`; NaN → 0; a non-scalar or `None` return → 0.0 | `run_risk` | 0.49996 → 0.5, which starts an alarm |
| Part A runs first, then Part B decodes and streams every frame. The deadline is checked every 100 frames. Over budget → `{"events": [], "risk": []}` | `main` | A timeout kills Part A output too |
| Exception in `detect_events` → events []. Exception in `step` → risk [] only, Part A kept | `main` | A video with empty risk adds no AP frames, but its accidents still count as FN / TTA = 0 |
| Frame label order: inside an accident [s,e] (inclusive) → ignored; else in [s−5, s) → positive (this **beats** the near-miss ignore); else in [s−5, e] of a near_miss → ignored; else negative | `frame_label` | Alarms on near-misses are free, but not on the positive window |
| Alarm = run start. Merge if `next_start − prev_last_above_θ_frame < 2.0`. Alarms whose start frame is ignored are dropped **before** matching. For each accident, in time order, the earliest unmatched alarm in [s−10, s) matches | `alarm_starts`, `evaluate_part_b` | See §3 |

---

## 2. Part A: event detection

### 2.1 Macro-F1 dilution: should we ever predict class c?

Let N be the number of other scored classes and S̄ their mean F1.

- **If c is present (probability p):** c is in C whether or not you predict it. Predicting it adds f/(N+1); not predicting it adds 0. **Predicting a present class can never lower Score A.**
- **If c is absent:** with probability q you emit at least one false positive. That adds a zero-scoring class, so the score goes from S̄·N/N to S̄·N/(N+1), a loss of S̄/(N+1). One false positive costs as much as a hundred: once F1_c = 0 it cannot go lower.

**EV = [p·f − (1−p)·q·S̄]/(N+1). Predict iff p/(1−p) > q·S̄/f, so p\* = q·S̄/(f + q·S̄).** |C| only scales the stakes; it does not move the break-even point. Checked with the official scorer: Score A 0.5556 → 0.4167 with one spurious fire_smoke false positive, and the same with three.

Break-even p\* (columns = q·S̄):

| f \ q·S̄ | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|
| 0.1 | 0.33 | 0.50 | 0.67 | 0.75 | 0.80 |
| 0.2 | 0.20 | 0.33 | 0.50 | 0.60 | 0.67 |
| 0.3 | 0.14 | 0.25 | 0.40 | 0.50 | 0.57 |
| 0.5 | 0.09 | 0.17 | 0.29 | 0.37 | 0.44 |

Expected ΔScore_A in points (×100), S̄ = 0.40. Left block: a trigger-happy detector (q = 1). Right block: a precision-first detector (q = 0.2, with f reduced by 25 % as the price of the stricter threshold).

| | f=0.2,q=1: p=0.1 | 0.3 | 0.5 | 0.9 | f=0.5,q=1: p=0.1 | 0.3 | 0.5 | 0.9 | **f=0.375,q=0.2**: p=0.1 | 0.3 | 0.5 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| \|C\|=4 | −8.5 | −5.5 | −2.5 | +3.5 | −7.8 | −3.3 | +1.3 | +10.3 | −0.9 | +1.4 | +3.7 | +8.2 |
| \|C\|=6 | −5.7 | −3.7 | −1.7 | +2.3 | −5.2 | −2.2 | +0.8 | +6.8 | −0.6 | +0.9 | +2.5 | +5.5 |
| \|C\|=8 | −4.3 | −2.8 | −1.3 | +1.8 | −3.9 | −1.6 | +0.6 | +5.1 | −0.4 | +0.7 | +1.8 | +4.1 |
| \|C\|=10 | −3.4 | −2.2 | −1.0 | +1.4 | −3.1 | −1.3 | +0.5 | +4.1 | −0.4 | +0.6 | +1.5 | +3.3 |

**Converting a false-positive rate into q:** q = 1 − exp(−λT), where λ is false events per hour and T is test hours. With T = 1 h: λ = 0.1 → q = 0.10, λ = 0.5 → q = 0.39, λ = 1 → q = 0.63, λ = 2 → q = 0.86. Measure λ on your own labelled sample videos. They presumably contain mostly normal traffic, so they are a good false-positive estimator.

**Choosing the threshold under presence uncertainty** (toy detector: 6 false candidates/h before thresholding, recall 0.85, IoU ~ Beta(6,2)). The best confidence threshold rises as p falls: p = 0.9 → 0.55, p = 0.5 → 0.60, p = 0.3 → 0.65, p = 0.1 → 0.75. At p = 0.1 the "default" threshold of 0.5 has negative EV (−0.112·|C|⁻¹), while 0.75 is positive (+0.032). **Rule: per-class thresholds, set higher for classes you are less sure appear.**

**Practical class tiers (conditional on camera.md):**
- **Tier 1: enable, tune for F1.** stopped_vehicle, congestion, jaywalking, wrong_way (needs lane directions), accident. These are high-p, trajectory-rule-friendly classes.
- **Tier 2: enable only if the scene supports the rule.** red_light and stop_line need a visible signal (or a reliable inferred phase) and a stop line. failure_to_yield needs a crossing. solid_line_crossing needs mapped solid lines. illegal_turn and illegal_u_turn need prohibitions in camera.md. near_miss. If the prerequisite is missing, f ≈ 0, and **leaving the class off (or removing it from CLASSES as a kill switch) is strictly better.**
- **Tier 3: rare, uncertain p.** fire_smoke and road_obstacle. Enable only with q ≤ 0.1, meaning zero false positives on all sample footage including night and glare. Require a long persistence (≥ 3–5 s) and a stationary location.

### 2.2 The marginal-candidate rule

F1 = 2TP/(G+P). Adding one prediction multiplies the denominator by (G+P+1)/(G+P). If it matches, it adds 2(G+P−TP)/[(G+P)(G+P+1)]; if not, it subtracts 2TP/[(G+P)(G+P+1)]. Averaged over τ, you should emit a candidate iff **P(real) × (τ passed / 3) > F1_c/2**. This is the Lipton–Elkan–Naryanaswamy result (optimal threshold = F1\*/2) applied per τ.

A numeric check at F1 = 0.60: a candidate that is surely real but passes only 1 of 3 τ (IoU 0.3–0.5) is net negative (ΔF1 −0.003 at P = 0.8). One passing 3 of 3 breaks even at P(real) = 0.30. The same holds with a perfect class state: 4 GT with 3 perfect matches gives 0.857 if the 4th is not predicted and 0.833 if it is predicted at IoU 0.4. **Poorly localised detections should be fixed or dropped, not emitted.**

### 2.3 Temporal-IoU sensitivity

Maximum error d still passing τ, for a ground truth of length L:

| Error type | formula d_max | τ=0.3 | τ=0.5 | τ=0.7 |
|---|---|---|---|---|
| shift (both ends same direction) | L(1−τ)/(1+τ) | 0.538L | 0.333L | 0.176L |
| widen d each side | L(1−τ)/(2τ) | 1.167L | 0.5L | 0.214L |
| shrink d each side | L(1−τ)/2 | 0.35L | 0.25L | 0.15L |
| one end extended | L(1−τ)/τ | 2.33L | 1.0L | 0.43L |
| one end truncated | L(1−τ) | 0.7L | 0.5L | 0.3L |

In seconds, for a shift: L = 2 → 1.08 / 0.67 / **0.35**; L = 3 → 1.62 / 1.00 / **0.53**; L = 5 → 2.69 / 1.67 / **0.88**; L = 10 → 5.4 / 3.3 / 1.8; L = 30 → 16 / 10 / 5.3; L = 120 → 65 / 40 / 21.

Expected τ-averaged credit when both boundaries have independent N(0, σ) error:

| L \ σ | 0.25 s | 0.5 s | 1 s | 2 s | 4 s | 8 s |
|---|---|---|---|---|---|---|
| 2 s | 0.97 | 0.77 | 0.45 | 0.20 | 0.07 | 0.02 |
| 3 s | 1.00 | 0.91 | 0.64 | 0.34 | 0.14 | 0.04 |
| 5 s | 1.00 | 0.99 | 0.85 | 0.56 | 0.27 | 0.10 |
| 10 s | 1.00 | 1.00 | 0.99 | 0.85 | 0.55 | 0.27 |
| 30 s | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.74 |

**Implications**
- **Short events** (accident, near_miss, red_light, failure_to_yield, solid_line_crossing: typically 1–5 s) need boundary errors of ≤ 0.3–0.5 s. Anchor boundaries to geometric events rather than classifier-probability plateaus: front bbox edge crossing the stop line, first box contact, entering and leaving the crossing polygon. The organizers' example ground truth uses whole-second boundaries (it is only an example). If real labels are that coarse, annotator noise alone (±0.5 s) caps F1@0.7 for 2–3 s events. Nothing can be done about that except to avoid adding your own error on top.
- **Long events** (congestion, stopped_vehicle, fire_smoke, road_obstacle) tolerate seconds of error. For them the risk is fragmentation (§2.4) and a late start from confirmation windows. Reporting "stopped" at the moment the 10 s criterion is met instead of when the vehicle stopped gives IoU = (L−10)/L, which is 0.33 for L = 15 s. **Always back-date the start to the onset.**
- **Frame stride barely matters for IoU.** The worst-case IoU from quantisation alone at stride 3 (0.12 s) is 0.89 for L = 2 s and 0.95 for L = 5 s. Stride 5 still gives 0.83 for 2 s events. Stride is a runtime decision, not an accuracy one, down to about 5 for short classes.
- When only the end is uncertain (start anchored), predict the **median** plausible length. Simulation shows the optimum scale factor is 1.00 for σ_log from 0.2 to 0.8.
- **The 3-τ average rewards precision in steps.** A matched instance earns 0 (IoU < 0.3, and it becomes a false positive plus a false negative), 1/3 (0.3–0.5), 2/3 (0.5–0.7) or 1 (≥ 0.7). Moving typical IoU from 0.6 to 0.75 is worth +1/3 of that class's recall-weighted F1.

### 2.4 Fragmentation, merging and duplicates

These cases were run through the official matcher. Entries are TP/FP/FN at τ = 0.3 / 0.5 / 0.7.

| Case | 0.3 | 0.5 | 0.7 | mean F1 |
|---|---|---|---|---|
| perfect GT[0,10] | 1/0/0 | 1/0/0 | 1/0/0 | 1.000 |
| split 50/50 with 0.5 s hole | 1/1/0 | 1/1/0 | 0/2/1 | 0.444 |
| split 70/30 | 1/1/0 | 1/1/0 | 1/1/0 | 0.667 |
| split in 3 | 1/2/0 | 0/3/1 | 0/3/1 | 0.167 |
| 2 GT [0,5],[7,12] merged into [0,12] | 1/0/1 | 0/1/2 | 0/1/2 | 0.222 |
| 2 GT [0,20],[22,40] merged | 1/0/1 | 1/0/1 | 0/1/2 | 0.444 |
| extra 0.5 s blip next to a perfect match | 1/1/0 | 1/1/0 | 1/1/0 | 0.667 |
| 3 s accident reported as a 60 s window | 0/1/1 | 0/1/1 | 0/1/1 | 0.000 |
| emit [0,3]+[2,10] (overlap) → harness keeps only [0,3] | 1/0/0 | 0/1/1 | 0/1/1 | 0.333 |

**Gap-merge / min-duration simulation** (toy model: per-frame flags with Markov dropouts, onset jitter and spurious blips at 1.5/min; 40×5 min videos; stride 3):

- Short events (2–6 s, dropouts averaging 0.4 s): raw flags give F1 0.28. The best setting is **g = 1–2 s, m = 1–1.5 s, giving F1 0.93**. g = 5 s over-merges neighbours (0.78).
- Long events (20–120 s, dropouts averaging 1.5 s): raw flags give 0.03. The best is **g = 6–10 s, m = 10–15 s, giving 0.93**. g = 20 s drops back to 0.77.
- Rule of thumb: **g ≈ 3–5 × the typical dropout length, and always below the typical gap between separate same-class events. m ≈ ⅓ to ½ of the shortest plausible event.** Tune both on your dev labels at F1@0.7.

**The "simultaneous same-class events are one segment" convention.** For per-object classes (jaywalking, red_light, stopped_vehicle, failure_to_yield, wrong_way), build per-track intervals, then take the **union per class**. Merge overlapping and touching intervals, then apply the class gap-merge. Separate per-object segments that overlap would be dropped by the harness (the earlier-starting one is kept), and a split union costs as shown in the table above. The unknown is whether annotators also merge non-overlapping events separated by short gaps. **Ask in the channel**, and label your dev set the same way.

Suggested starting post-processing values (to tune on dev labels):

| class | union | gap g | min m | boundary anchor |
|---|---|---|---|---|
| accident | yes | 1.0 | 1.0 | first contact → all involved tracks < v_stop for ≥ 1 s, or out of frame |
| near_miss | yes | 0.5–1 | 0.8 | onset of hard decel/swerve → pair separation |
| red_light | yes | 0.5 | 0.5 | front edge over stop line → exits junction polygon |
| wrong_way | yes | 2 | 2 | enters opposing lane → returns or exits |
| illegal_(u_)turn | yes | 1 | 1.5 | heading-change onset → heading settles |
| stopped_vehicle | yes | 3–5 | 10 (by definition) | back-dated stop → first motion |
| jaywalking | yes | 1 | 1 | foot point enters carriageway → leaves |
| failure_to_yield | yes | 0.5 | 0.5 | enters crossing polygon → leaves |
| solid_line_crossing | yes | 0.5 | 0.5 | wheel/box edge crosses line → fully in new lane |
| stop_line | yes | 2 | 3 | stop past line on red → green |
| congestion | – | 10 | 15–20 | back-dated queue stop → clears |
| road_obstacle / fire_smoke | – | 5 | 3–5 | first appearance → removal / clear |

Also add a **sanity cap**: more than about 10 events per class per 5 minutes usually means a broken rule (flicker, camera shake), and because false positives are pooled, one bad video poisons that class for all videos.

### 2.5 Pooling and small counts

Attainable τ-mean F1 values are coarse: n = 1 gives {0, 0.5, 0.67, 1}; n = 2 gives {0.4, 0.5, 0.67, 0.8, 1, …}. Monte Carlo with recall 0.7, IoU ~ Beta(6,2) and Poisson(1) false positives per test set:

| n GT in test | mean F1 | sd | p10–p90 | P(F1=0) |
|---|---|---|---|---|
| 1 | 0.45 | 0.36 | 0.00–1.00 | 0.30 |
| 2 | 0.55 | 0.26 | 0.17–0.83 | 0.09 |
| 3 | 0.59 | 0.21 | 0.33–0.86 | 0.03 |
| 5 | 0.63 | 0.16 | 0.42–0.82 | 0.00 |
| 10 | 0.67 | 0.11 | 0.52–0.81 | 0.00 |
| 20 | 0.69 | 0.08 | 0.59–0.79 | 0.00 |

The mean also rises with n, because one pooled false positive hurts small classes more. **One rare event can be worth 1/|C| of Score A**: 25 points at |C| = 4, 10 points at |C| = 10, which is 4.2–10.5 elimination points. With 3–10 dev events per class, the dev F1 has sd ≈ 0.15–0.28, so differences smaller than about 0.1 on the dev set are noise. Prefer changes that are justified by the mechanism (boundary definition, precision) and verified on several videos, rather than tiny threshold sweeps.

---

## 3. Part B: accident anticipation

### 3.1 Mechanics, shown on one 60 s video

Setup: accident s = 30, e = 36; near_miss [45, 47]; exact metric.

| Risk curve | AP | P | R | F1 | mTTA | Score B |
|---|---|---|---|---|---|---|
| 0.9 on [25,30) only (ideal) | 1.00 | 1 | 1 | 1 | 5.0 | 0.900 |
| monotone ramp 0.5→0.9 over [20,30) | 1.00 | 1 | 1 | 1 | 10.0 | **1.000** |
| flat 0.9 over [20,30) | 0.44 | 1 | 1 | 1 | 10.0 | 0.776 |
| late: 0.9 on [29,30) | 0.20 | 1 | 1 | 1 | 1.0 | 0.500 |
| alarm starts at contact s | 0 | 0 | 0 | 0 | 0 | 0.000 |
| held through accident, off at e | 0.60 | 1 | 1 | 1 | 3.0 | 0.700 |
| held 6 s past e | 0.22 | 1 | 1 | 1 | 3.0 | 0.547 |
| **merge trap**: blip [18,18.5) + true [20.3,30) (gap 1.8 s) | 0.43 | 0 | 0 | 0 | 0 | 0.172 |
| same, gap 2.8 s (no merge) | 0.43 | .5 | 1 | .67 | 9.7 | 0.632 |
| two alarms in the window | 0.81 | .5 | 1 | .67 | 9.0 | 0.772 |
| + alarm on the near-miss [43,47) (free) | 0.80 | 1 | 1 | 1 | 4.0 | 0.800 |
| + alarm starting 6 s before the near-miss | 0.62 | .5 | 1 | .67 | 4.0 | 0.595 |
| 0.49996 on [25,30) (rounds to 0.5) | 1.00 | 1 | 1 | 1 | 5.0 | 0.900 |
| 0.49 on [25,30) (AP only) | 1.00 | 0 | 0 | 0 | 0 | 0.400 |
| sustained alarm from s−15 | 0.27 | 0 | 0 | 0 | 0 | 0.107 |
| same + age cap (dip 2.1 s at s−7) | 0.32 | .5 | 1 | .67 | 4.9 | 0.490 |
| 0.45 until imminent, 0.8 from s−3 | 0.71 | 1 | 1 | 1 | 3.0 | 0.743 |

**What these show:**
1. AP is rank-only, so early warning costs nothing **if scores keep rising toward s**. A flat early alarm ties the negatives in [s−10, s−5) with the positives.
2. **Latency is fatal**: an alarm that starts at or after contact lands on ignored frames and is dropped.
3. **High scores after e are negatives.** Once contact is detected, drop the score: the target is "an accident *starts* within 5 s", and a new start is unlikely.
4. **The merge trap:** any run above 0.5 within 2 s before the real alarm pulls its start earlier. If that start is before s−10, the alarm stops matching.
5. **Sustained alarms fail.** A run that started more than 10 s before s cannot match. Cap alarm age at about 8 s and force a dip of ≥ 2.1 s, or better, stay below 0.5 until the conflict is imminent.
6. **Alarms that start in [ns−5, ne] of a near-miss are free**, and so are the frames there for AP. Time-to-collision (TTC) alarms on near-misses cost nothing if they fire no more than 5 s before the evasive action.
7. **Every alarm in a video with no accident is a false positive.**

**F1_alarm with few accidents** (≈ 2m/(n+m+k) at recall 0.6, where k = false alarms over the whole test set):

| n accidents | k=0 | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| 1 | 0.75 | 0.46 | 0.33 | 0.26 | 0.18 | 0.10 | 0.06 |
| 3 | 0.75 | 0.62 | 0.53 | 0.46 | 0.37 | 0.24 | 0.15 |
| 10 | 0.75 | 0.71 | 0.67 | 0.63 | 0.57 | 0.46 | 0.33 |

With 1–3 accidents, **each false alarm costs 0.05–0.3 F1 (×0.4 in Score B)**. Target a false-alarm budget of about 1–2 per test hour.

### 3.2 Strategy simulation on a toy hazard world

This is an illustration of trade-offs, not a performance forecast. The toy world:
- 12×5 min videos.
- Background: 40 benign conflicts/h with amplitude ~ Beta(2,4), plus tracker-glitch spikes.
- Accident precursors rising over 1–5 s to a peak at s; 25 % of accidents are unforeseeable.
- Near-misses with similar bumps.

Scores use a monotone map in which the hazard e crosses 0.5 exactly at T. This keeps AP invariant to T and isolates the alarm trade-off. Mean of 20 test sets:

| Strategy | FEW (3 acc): AP / P / R / F1 / mTTA / alarms / **B** (sd) | MANY (10 acc): AP / F1 / **B** (sd) |
|---|---|---|
| constant 1.0 | 0 / 0 / 0 / 0 / 0 / 12 / **0.000** | 0 / 0 / **0.000** |
| raw per-frame hazard, T=0.5 | .056 / .02 / .83 / .04 / 2.2 / 130 / **0.081** (.03) | .152 / .12 / **0.158** |
| raw, T=0.7 | .056 / .24 / .62 / .34 / 0.4 / 7 / **0.166** (.10) | .152 / .63 / **0.322** |
| EMA 0.2 s, T=0.5 | .049 / .20 / .72 / .31 / 0.6 / 11 / **0.157** (.06) | .136 / .61 / **0.314** |
| **EMA 0.2 s, T=0.6** | .049 / .28 / .58 / .37 / 0.3 / 6 / **0.175** (.10) | .136 / .66 / **0.330** |
| EMA 0.8 s, T=0.6 | .028 / .34 / .20 / .24 / 0.1 / 2 / **0.107** | .079 / .38 / **0.184** |
| hysteresis (0.4 s confirm, 3 s hold), T=0.7 | .043 / .19 / .12 / .14 / 0.03 / 1.4 / **0.075** | .096 / .26 / **0.146** |
| EMA + 1 s slope extrapolation, T=0.6 | .050 / .02 / .85 / .04 / 2.3 / 116 / **0.082** | .156 / .14 / **0.164** |
| AP-only (never ≥ 0.5) | .053 / – / – / 0 / 0 / 0 / **0.021** | .115 / 0 / **0.046** |

**Takeaways**
- (i) **Sharp spikes** (raw or extrapolated with noisy derivatives) give high recall and mTTA but more than 100 alarms/h, so F1 ≈ 0.05. Spikes separated by more than 2 s are not merged.
- (ii) **Light smoothing (~0.2 s) plus a threshold** is the best in every setting. Heavier smoothing, multi-frame confirmation and long holds delay the start past s, where it is discarded, and they also lower AP.
- (iii) **Early ramps** only help if the predictor is smooth (a Kalman-based TTC, not frame differences), and the mTTA gain is small: going from 0.3 s to 2 s of mTTA is worth only +0.034 Score B at full recall.
- The components in the best FEW configuration: 0.4·AP ≈ 0.02, **0.4·F1 ≈ 0.15**, 0.2·mTTA/W ≈ 0.006. **F1_alarm is the lever.**
- With few accidents the Score B standard deviation is about 0.10, so luck dominates. Do not sink days into Part B tuning.
- A pooled-AP calibration check (per-video background offset with sd 0.2) moved AP from 0.043 to 0.028. Score on **physically meaningful, video-independent scales** (TTC in seconds, closing speed normalised by perspective), not per-video ranks.
- The F1\*/2 rule applies to alarms too. The alarm threshold should correspond to a calibrated P(an accident follows within ~5–10 s) of about **0.15–0.3**, not 0.5.

### 3.3 Recommended RiskEstimator output policy (pseudo-code)

```
every k-th frame (k=2-3; else return last score):
  causal detect+track (online tracker, Kalman velocities); only tracks with age >= 0.5 s count
  for each road-user pair: TTC (constant velocity, extents), closing speed, decel of each, conflict flags
  h = max over pairs of hazard(TTC, closing speed, decel, flags); e = EMA(h, tau=0.2 s)
  rank = 0.49 * soft_hazard   # graded "pre-alarm" cues over the whole 5 s horizon (fast approach toward a
                              # conflict zone, wrong-way track, pedestrian entering road), strictly <= 0.49
  if contact_detected(pair): mute(pair, 10 s); score = 0.05; alarm off     # accident frames are ignored anyway
  elif not alarm and e >= T_on (set for <= 1-2 false alarms / hour on sample footage) and closing speed > v_min:
       alarm on (no extra confirmation delay)
  elif alarm and (conflict resolved: TTC > 4 s or pair separating) -> alarm off
  if alarm and alarm_age > 8 s and no escalation: force 0.45 for 2.1 s (re-arm)
  re-trigger on the same pair within 10 s only if hazard exceeds the previous alarm peak
  score = 0.5 + 0.5*e if alarm else min(rank, 0.4999 -> use 0.49)
  return float(score)   # python float; never an array; never NaN
```

---

## 4. What each deliverable is worth, and how to split effort

Elimination = 0.42·A + 0.18·B + 0.25·Web + 0.15·Code, and M → 0 if the package does not run.

| Item | Elimination points | Equivalent Score A |
|---|---|---|
| +0.10 Score A | 0.042 | 0.10 |
| +0.10 Score B | 0.018 | 0.043 |
| Web: live demo (30 %) | 0.075 | **0.18** |
| Web: sample-video visualisations (20 %) | 0.050 | 0.12 |
| Web: EDA / approach & report (15 % each) | 0.0375 each | 0.09 each |
| Web: team / design & extras (10 % each) | 0.025 each | 0.06 each |
| Code: runs as submitted (40 %) | 0.060 | 0.14 |
| Code: reproducibility / structure / engineering | 0.0375 / 0.030 / 0.0225 | 0.09 / 0.07 / 0.05 |

Scenario checks:
- A = 0.40, B = 0.20, Web = 0.8, Code = 0.85 gives **0.532**. If the package fails, it gives **0.268**, a loss of 0.264.
- A = 0.25, B = 0.10 gives 0.451. If the package fails, 0.268.
- Skipping Part B entirely (B = 0) costs 0.02–0.05.

**Allocation for three people:**
- **Everyone, first ~15 % of the time:** label the sample videos under the official conventions; build a walking skeleton (harness passes, `--validate-only` clean, Docker/requirements tested on a clean VM, website shell with a stub demo).
- **P1 (Part A, about 45 % of total value):** detector and tracker, scene geometry from camera.md, Tier-1 rules, segment post-processing, back-dating, per-class thresholds.
- **P2 (evaluation, Part B and verifiers):** dev-set scoring loop with `evaluate.py`; false-positive-per-hour accounting per class; learned accident/near-miss verifier; causal TTC estimator. Part B should take about 30 % of P2.
- **P3 (website, about 25 % of value, plus Code, 15 %):** live demo with a 2-minute cap on CPU inference and progress display; auto-generated annotated sample videos, timelines and risk curves from the *same* pipeline code; EDA; README with licences and seeds; timing guard; determinism test (run twice and diff predictions).

---

## 5. Format and edge cases (checked against the harness code)

- Return `[[float, float, str], ...]` with 0 ≤ s < e.
  - If an event runs to the end, set `end = duration`. The harness clamps to `n_frames/fps` anyway, and validation allows +0.5 s.
  - If an event is ongoing at the first frame, `start = 0`.
- An empty list is valid. So is `None`: it is logged and treated as [].
- Union same-class segments yourself. Touching segments are fine. An overlap loses the later one.
- NumPy scalars are coerced. A 1-element array makes the harness `float()` fail: the event is dropped, or the risk frame is recorded as 0.
- **Keep non-alarm scores ≤ 0.4999** to allow for the 4-decimal rounding. Clip to [0,1] yourself and handle NaN.
- **Crashes:** a crash in detect_events empties the events for that video. A crash in step empties its risk (accidents become false negatives). Wrap both internally with try/except and return the last good state.
- **Budget** = 3 × `n_frames/fps`, covering Part A plus the harness's full-resolution decode for Part B. Aim for ≤ 1.5× total on T4-class hardware, and add an internal watchdog that raises the stride when behind. The sample files are large (one is 5.8 GB), so benchmark decode early.
- **Determinism:** fix seeds and use a deterministic tracker. Check the diff of two runs on the samples against `predictions_samples.json`.
- **Causality:** do not feed Part A outputs (computed offline) into RiskEstimator, and do not open the file in it. Part A **may** use a causal risk curve.

---

## 6. Prioritised metric-driven design decisions

1. Clean-machine run plus an internal time watchdog (≤ 1.5× duration); Part B budgeted separately.
2. Your own dev labels following the official conventions; evaluate with `evaluate.py --per-video`; track false positives per hour per class.
3. Per-class enable switches and thresholds from p·f vs (1−p)·q·S̄; precision-first settings for Tier 2 and 3.
4. Segment builder: per-track intervals → per-class union → gap-merge → min-duration → back-date onset → clamp.
5. Geometric boundary anchors for short classes; target ≤ 0.3–0.5 s error.
6. Candidate emission rule: expected τ-credit > F1_c/2; drop poorly localised candidates.
7. Part B: two-channel score, ~0.2 s smoothing, no confirmation lag, a false-alarm budget of ≤ 1–2/h, contact reset, 8 s age cap, one alarm per conflict, scores ≤ 0.49 when not alarmed.
8. A graded sub-threshold "pre-alarm" over the full 5 s horizon, to lift AP cheaply.
9. Globally calibrated physical features (TTC in seconds), not per-video normalisation.
10. Do not chase mTTA; do not over-tune on n < 10 dev events.

---

## Appendix: condensed code (full scripts in `scratchpad\metric\`)

```python
# EV rule (a)
def enable(p, f, q, Sbar): return p*f > (1-p)*q*Sbar          # |C| cancels
q = lambda lam_per_h, T_h: 1 - math.exp(-lam_per_h*T_h)

# Part B exact (vectorised; matches evaluate.py to 2e-16)
def frame_labels(t, acc, nm):
    lab = np.zeros(len(t), np.int8); ign_acc = pos = ign_nm = np.zeros(len(t), bool)
    for s,e in acc: ign_acc = ign_acc | ((t>=s)&(t<=e)); pos = pos | ((t>=s-5)&(t<s))
    for s,e in nm:  ign_nm = ign_nm | ((t>=s-5)&(t<=e))
    lab[ign_nm] = -1; lab[pos] = 1; lab[ign_acc] = -1; return lab
def ap_sklearn(s, l):
    o = np.argsort(-s, kind="mergesort"); s, l = s[o], l[o]; tp, fp = np.cumsum(l), np.cumsum(1-l)
    k = np.r_[np.nonzero(np.diff(s))[0], len(s)-1]; P = tp[k]/(tp[k]+fp[k]); R = tp[k]/l.sum()
    return float(np.sum(np.diff(np.r_[0, R])*P))
def alarm_starts(t, sc):
    on = sc >= 0.5; d = np.diff(np.r_[0, on.astype(int), 0]); st, en = np.nonzero(d==1)[0], np.nonzero(d==-1)[0]-1
    m = []
    for a, b in zip(t[st], t[en]):
        if m and a - m[-1][1] < 2.0: m[-1][1] = b
        else: m.append([a, b])
    return [x[0] for x in m]
# score: round t,score to 4 dp; AP pooled over labels>=0, normalised (AP-r)/(1-r);
# drop alarm starts with label -1; per accident (sorted) earliest unmatched start in [s-10,s) matches;
# B = .4*AP + .4*F1 + .2*min(1, mTTA/10)

# ranking-preserving alarm map (AP invariant to T)
def monotone_map(e, T): e = np.clip(e,0,1); return np.where(e<T, 0.49*e/T, 0.5+0.5*(e-T)/(1-T))

# segment builder
def to_segments(t, flag, dt, gap, mind):
    d = np.diff(np.r_[0, flag.astype(int), 0]); st, en = np.nonzero(d==1)[0], np.nonzero(d==-1)[0]
    out = []
    for s, e in ((t[a], t[b-1]+dt) for a, b in zip(st, en)):
        if out and s - out[-1][1] <= gap: out[-1][1] = e
        else: out.append([s, e])
    return [(s, e) for s, e in out if e - s >= mind]
```

Scripts: `common.py` (exact scorer and import of the official `evaluate.py`), `verify.py`, `a_dilution.py`, `b_tiou.py`, `c_fragment.py`, `d_smalln.py`, `e_partb.py`, `e2_partb.py`, `fg_misc.py`. All are in `C:\Users\Cicada\AppData\Local\Temp\claude\C--Users-Cicada-Desktop-WestHack\ced7e0d9-be88-47bc-ac13-90898f4ae082\scratchpad\metric\`.

Sources: [Lipton, Elkan, Naryanaswamy 2014, arXiv:1402.1892](https://arxiv.org/abs/1402.1892); [sklearn average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html); starter kit `evaluate.py` and `run_submission.py` (organizers' Drive `wiut_cv_scripts.zip`).


## Top recommendations
- Treat 'runs on a clean machine within budget' as priority zero: an internal time watchdog (target <=1.5x duration total, adaptive stride), try/except inside detect_events and step returning last good state, Python-float outputs, and a clean-VM test of both commands; failure loses ~0.18-0.26 elimination points.
- Enable each class only if p*f > (1-p)*q*Sbar (|C| cancels); use per-class thresholds, precision-first (q<=0.1-0.2) for uncertain/rare classes (fire_smoke, road_obstacle, signal-dependent classes), and remove classes whose scene prerequisites (visible signal, crossing, solid lines, prohibitions) are absent in camera.md.
- Build segments as per-track intervals -> per-class union (harness drops, not merges, overlaps) -> gap-merge (~1-2 s short classes, ~6-10 s long classes) -> min-duration -> back-date onset for confirmation-based rules (stopped_vehicle, congestion) -> clamp to duration; tune g/m on own dev labels at F1@0.7.
- Anchor short-event boundaries (accident, near_miss, red_light, failure_to_yield, solid_line_crossing) to geometric moments (contact, stop-line crossing, polygon entry/exit) to keep errors <=0.3-0.5 s; emit a candidate only if P(real) x expected tau-credit > F1_c/2.
- Part B: two-channel output (ranking <=0.49 for AP over the full 5 s horizon; alarm >=0.5 only for confident imminent conflicts), ~0.2 s smoothing and no confirmation lag, false-alarm budget <=1-2 per test hour, drop score after detected contact, cap alarm age ~8 s, one alarm per conflict, never return 0.4999x; do not chase mTTA.
- Split effort roughly: Part A ~45% of engineering, website (live demo first, then auto-generated sample-video visualisations) ~25-30%, packaging/reproducibility ~15%, Part B ~10-15%; everyone labels sample videos in the first ~15% of the time.
- Do not over-tune on small dev counts (sd of class F1 ~0.15-0.28 with 3-10 events); prefer mechanism-justified changes validated across all sample videos.


## Open questions
- Do organizer annotations merge same-class events that are close but not simultaneous (short gaps), and at what boundary granularity are they labelled (the example GT uses whole seconds)? Ask in the hackathon channel.
- How many test videos/hours and how many accidents/near-misses are in the hidden test set? This sets q (FP exposure) and the false-alarm budget for Part B.
- Is the Drive copy of wiut_cv_scripts.zip the final version of evaluate.py/run_submission.py (organizers may update before the deadline, including hardware)?
- Would caching strictly-causal per-frame tracker output from detect_events for reuse in RiskEstimator be considered a rule violation? The safe assumption is yes: keep Part B self-contained.
- What are the sample videos' resolution/bitrate (one file is 5.8 GB)? Full-resolution decode inside the harness's Part B loop counts against the 3x budget.
- Does camera.md indicate a visible traffic signal, crossings, solid lines and turn prohibitions? This decides which Tier-2 classes are worth enabling.


## Key claims (as submitted for verification)
- The starter kit's evaluate.py (from the organizers' Drive folder, wiut_cv_scripts.zip) is the exact metric; a vectorised re-implementation matched its Score B to 2.2e-16 on 40 random multi-video cases — C:\Users\Cicada\AppData\Local\Temp\claude\C--Users-Cicada-Desktop-WestHack\ced7e0d9-be88-47bc-ac13-90898f4ae082\scratchpad\kit\wiut_cv_scripts\evaluate.py ; verify.py
- Score A class set = classes in ground truth OR in predictions; a predicted-but-absent class scores 0 and enlarges the denominator (e.g. 0.5556 -> 0.4167 with one spurious fire_smoke FP, same with three) — evaluate.py evaluate_part_a; a_dilution.py
- Break-even rule for enabling class c: predict iff p*f > (1-p)*q*Sbar; the number of classes |C| cancels — reasoning (derivation in report), verified numerically with official scorer
- For calibrated probabilities the F1-optimal decision threshold equals half the optimal F1 (Lipton, Elkan, Naryanaswamy 2014) — https://arxiv.org/abs/1402.1892
- evaluate.py's average_precision equals sklearn.metrics.average_precision_score including tie groups — evaluate.py docstring + local check vs sklearn 1.8.0; https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- run_submission.py clamps event end to duration = CAP_PROP_FRAME_COUNT/fps, rounds times to 3 decimals, and drops (does not merge) a same-class segment that overlaps an earlier-starting one; touching segments are allowed — run_submission.py clean_events; evaluate.py validate (s2 < e1)
- The harness records risk as round(clip(score),4): a returned 0.49996 becomes 0.5 and starts an alarm; NaN becomes 0 — run_submission.py run_risk; e_partb.py demo; fg_misc.py
- Under NumPy 2.4, float(np.array([x])) raises TypeError, so a step() returning a 1-element array is silently recorded as 0.0 by the harness — local test with numpy 2.4.2 + run_submission.py try/except
- If a video exceeds 3x duration, both its events and its risk curve are replaced by empty lists; Part B runs after Part A in the same budget; a crash in step() only empties the risk curve — run_submission.py main
- Model score M = Score_A when the test set contains no accidents (Part B not scored) — evaluate.py evaluate()
- Frame labelling order: inside accident [s,e] ignored; else [s-5,s) positive (overrides near-miss ignore); else [s-5,e] of near_miss ignored; alarm starts on ignored frames are dropped before matching; runs merge if next start minus previous last above-threshold frame < 2.0 s — evaluate.py frame_label, alarm_starts, evaluate_part_b
- Temporal-IoU tolerance for a pure shift is d <= L(1-tau)/(1+tau): 0.35 s for a 2 s event and 0.53 s for a 3 s event at tau=0.7, vs 21 s for a 120 s event — reasoning (closed form), b_tiou.py
- A matched prediction with IoU in [0.3,0.5) can lower class F1 (e.g. 4 GT, 3 perfect: 0.857 without it vs 0.833 with it) — b_tiou.py using official F1 definition
- In a toy simulation, gap-merge + min-duration raised tau-mean F1 from 0.28 to 0.93 (short events, g~1-2 s, m~1-1.5 s) and from 0.03 to 0.93 (long events, g~6-10 s, m~10-15 s) — c_fragment.py (toy model, illustrative)
- With 1 test instance of a class, tau-mean F1 has sd ~0.36 and P(F1=0)~0.30 for a recall-0.7 detector with ~1 pooled FP — d_smalln.py (simulation)
- In the toy Part B world, F1_alarm dominates Score B (0.4*F1 ~0.15 vs 0.4*AP ~0.02 and 0.2*mTTA/W ~0.006); raw per-frame spikes produce ~130 alarms/h and Score B ~0.08, light 0.2 s EMA with threshold gives ~0.175 (3 accidents) / 0.33 (10 accidents) — e2_partb.py (toy model, illustrative)
- A working live demo is worth 0.075 elimination points (= +0.18 Score A); a non-running package loses ~0.18-0.26 elimination points in realistic scenarios — spec weights arithmetic (fg_misc.py)
