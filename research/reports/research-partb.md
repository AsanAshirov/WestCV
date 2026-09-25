<!-- source: deep-research workflow wf_294a4b01-fc2, agent research:partb -->

# Part B (RiskEstimator): how to maximise Score_B

## 0. Summary

1. **Read the official code first.** Another agent downloaded the organizers' public starter kit to `C:\Users\Cicada\AppData\Local\Temp\claude\C--Users-Cicada-Desktop-WestHack\ced7e0d9-be88-47bc-ac13-90898f4ae082\scratchpad\kit\wiut_cv_scripts\` (it contains `evaluate.py`, `run_submission.py`, `solution.py` and the examples). Several details that matter for Part B are only in that code (section 1).
2. **Only three levers matter, in this order: alarm precision and recall, then AP, and only then lead time.** One false alarm costs about as much Score_B as 5 s of extra lead time on one accident. One extra matched accident is worth about 3 false alarms. mTTA is a small term.
3. **Put the 0.5 threshold at the F1-optimal operating point, not at "probability 0.5".** Lipton et al. show that for calibrated scores the F1-optimal threshold is F1*/2, which comes to about 0.2–0.3 here. In practice, set the threshold with a false-alarm budget: about 0.3–0.5 false alarms per expected test accident, or roughly 1–2 per hour of accident-free footage.
4. **Keep alarms short and use the free rides.** The metric already merges runs less than 2 s apart, so extra hysteresis or latching mostly costs AP. Frames inside an accident, and from 5 s before a near miss to its end, are ignored. Alarms that start there are dropped rather than counted as false. Suppress the risk after a detected contact.
5. **Features: surrogate safety measures from an online detector and tracker in ground coordinates.** Use TTC (capsule model, 5 s horizon, must be decreasing consistently), DRAC, predicted PET/T2, closing speed, hard braking and swerving, plus causal versions of the Part A violation flags (red-light runner, wrong-way, pedestrian on the carriageway). Start with hand-set logistic weights. Fit them later on collision-injected tracks and on public CCTV clips with an annotated impact frame.
6. **Engineering.** Load the model once at module level. Wrap `step` in try/except. Pick the stride deterministically from `meta`. Do not use wall-clock-dependent async code. Run Part B's perception independently of `detect_events`. The FAQ says Part B may not use Part A's output.

---

## 1. What the official code does (verified in the starter kit)

Findings from `evaluate.py` and `run_submission.py` that change the design:

| Fact (from code) | Consequence |
|---|---|
| `run_submission.py` builds a new `sol.RiskEstimator()` for **every video**, then calls `reset` and `step` for every frame (`--risk-stride` defaults to 1; the official run uses 1). | Loading a model in `__init__` or `reset` costs time on every video. Keep the model in a lazily created module-level singleton. Put all per-video state (tracker, buffers) in `reset`. |
| Part A runs first, then Part B. A single deadline covers both (3× duration). Part B checks it every 100 frames. If the video goes over, **both events and risk are thrown away**. | A slow Part B wipes out that video's Part A score. Part B needs a hard budget. |
| An exception raised in `step` aborts `run_risk`, and the whole risk curve for that video becomes `[]`. | Put the body of `step` in try/except and return the last score on error. |
| Scores are clamped to [0,1] and **rounded to 4 decimals**. `t = idx/fps`. NaN ends up as 0. | Keep meaningful rank differences above 1e-4. Don't compress low risks into values like 1e-6. |
| `frame_label`: frames in [s, e] of an accident are ignored (both ends inclusive). Positive if s−5 ≤ t < s. Frames in [s−5, e] of a near miss are ignored, but an accident positive overrides that. | Post-contact frames and near-miss windows cost nothing, for AP and for alarms. |
| `alarm_starts`: a run is a maximal stretch of frames with score ≥ 0.5. The next run is merged if its start minus the previous run's last on-frame is < 2.0 s. | The metric already debounces flicker for 2 s. A new, separate alarm needs ≥ 2.0 s below 0.5. |
| Alarms whose start is on an ignored frame are dropped. For each accident in time order, the earliest unmatched alarm in [s−10, s) matches. Any other alarm in that window counts as a **false alarm**. | A second alarm before the same crash is a false alarm. Flicker after contact is free. |
| AP is the sklearn-style step AP over tie groups, pooled across videos, then chance-normalised. | Only the ranking matters, and it must be comparable across videos. |
| `M = Score_A` if the hidden test set has **no accidents**. | Part B may not count at all. It is still worth a moderate investment. |
| `F1 = 2PR/(P+R)`. With m matched alarms, A alarms and N accidents, this equals **2m/(A+N)**. | This closed form drives the decision analysis in section 2. |

## 2. How Score_B is built, and the decision rules that follow

### 2.1 Alarm F1 and lead time, marginal values

F1_alarm = 2m/(A+N). The table below assumes recall 0.5 and false alarms = 0.5·N (computed in the scratchpad):

| N accidents in test | +1 false alarm | +1 matched alarm (TTA 2 s) | +1 s lead on one accident | F1-optimal match probability q* = F1/2 |
|---|---|---|---|---|
| 3 | −0.029 | +0.085 | +0.007 | 0.29 |
| 5 | −0.018 | +0.070 | +0.004 | 0.22 |
| 10 | −0.0095 | +0.033 | +0.002 | 0.25 |
| 20 | −0.0049 | +0.017 | +0.001 | 0.25 |

Decision rules:
- **Raise an alarm on a candidate episode if its probability of matching an accident within the next 10 s is above about F1/2 ≈ 0.25.** This is the Lipton–Elkan–Narayanaswamy result applied per alarm candidate.
- **Extra lead is only worth it if it adds fewer than Δτ/5 false alarms** per accident gaining Δτ seconds (at N = 10). In the toy runs, mTTA at the optimum was 0.3–0.5 s, which is worth at most about 0.01 of Score_B.
- F1 as a function of false alarms per accident: at recall 0.5, 0.25 FP/acc gives 0.57, 0.5 gives 0.50, 1 gives 0.40, 2 gives 0.29 and 4 gives 0.18. **The false-alarm budget has to be set against the accident density of the test set, which is unknown.** Zero alarms gives F1 = 0 and mTTA = 0, so always emit some alarms.

### 2.2 AP: only the top of the ranking counts

The positive rate r is tiny. For example, 10 accidents × 125 frames out of 150k frames gives r ≈ 0.8%. A positive frame retrieved at a precision close to r adds about r × (its share of positives) to AP, which is essentially nothing. So:
- A context prior for the tail (for example traffic density) is not worth building.
- What matters is how many positive frames are ranked above almost all negatives, and how few negatives score high.
- High-scoring negatives come from: benign interactions such as queue following, turning gaps and tracker noise; the [s−10, s−5) part of a true pre-crash ramp; the aftermath after e (vehicles swerving around a wreck, stopped cars); evidence that starts more than 5 s before a near miss.
- **Make the score rise monotonically as predicted time-to-contact falls.** Then, within one event, the negatives in [s−10, s−5) rank below the positives in [s−5, s).
- **Do not normalise scores per video**, for example by z-scoring within a clip. Pooled AP compares absolute values across videos, including videos with no accidents.

### 2.3 Free rides and traps

Free:
- Any score in [s, e] of an accident. Alarms starting there are dropped. A post-contact detector cannot hurt Score_B.
- Any score in [s_nm−5, e_nm] of an annotated near miss. Evasive-action alarms caused by near misses are dropped, not counted as false.

Traps:
1. **Late alarms and annotation jitter.** An alarm that starts 0.1 s before the annotated contact falls into [s, e] if the annotator marked contact 0.3 s earlier. It is then dropped and the accident is missed. Aim for alarm onset ≥ 0.5–1 s before contact. Triggering at TTC ≈ 1.5–2.5 s, minus about 0.3–0.5 s of estimation lag, gives about 1–2 s.
2. **An early false alarm merges with the true one.** A run ending at s−10.5 followed by a real alarm at s−9 is less than 2 s apart. The two merge, the start is s−12, and the result is a false alarm plus a missed accident. Short alarms make this less likely.
3. **Long alarms.** An alarm that has been on for more than 10 s before a crash cannot match, and it stops a new alarm from starting inside the window. Prevent persistent benign alarms with habituation (section 4.1), not with forced re-arming.
4. **Second alarm before the same crash.** An on/off/on sequence with a gap of ≥ 2 s before s gives one true alarm plus one false alarm. Bridging short gaps of ≤ 3 s for the same pair is worth about +0.01 (one avoided false alarm) and costs roughly 0.003 of AP (50 negative frames near the top). Keep a small release delay.

### 2.4 Toy simulation (official `evaluate_part_b`, synthetic latent-risk traces)

Script: `...\scratchpad\partb_sim.py`. It is a toy generator: 16 videos × 5 min, 8 accidents (70% visible, lead lognormal with median 3 s), 16 near misses, and benign interaction episodes. The numbers are qualitative only.

| Policy (1 benign interaction per minute) | Score_B | AP | F1 | mTTA | FP/h |
|---|---|---|---|---|---|
| threshold c=0.0 | 0.221 | 0.165 | 0.34 | 0.93 s | 15.4 |
| c=1.0 | 0.290 | 0.165 | 0.53 | 0.50 s | 4.5 |
| **c=1.5 (best)** | **0.323** | 0.165 | **0.63** | 0.32 s | 1.6 |
| c=2.5 | 0.256 | 0.164 | 0.47 | 0.10 s | 0.2 |

Other runs:
- With 0.3, 1 and 3 benign interactions per minute, the best threshold always fell at about **0.27–0.36 false alarms per accident**, with precision around 0.6–0.7.
- Hysteresis plus a 1 s minimum-on plus a 1 s release lowered AP from 0.163 to 0.145 and raised F1 by only 0.005, a small net loss.
- **Post-contact suppression** (30 s, contact detected 70% of the time) raised precision from 0.44 to 0.51 and Score_B from 0.263 to 0.283.
- AP does not depend on the threshold, but it falls as the benign-conflict rate rises (0.22, 0.17, 0.15).

## 3. Causal features

### 3.1 Perception loop

- **Detector:** the same detector family as Part A. For reference, YOLO11n/s run at 1.5/2.5 ms and YOLO26n/s at 1.7/2.5 ms on a T4 with TensorRT 10 (Ultralytics docs; YOLO26 is AGPL-3.0 and NMS-free). PyTorch FP16 without TensorRT will be slower; plan for ≤ 20 ms per processed frame end to end (my estimate).
- **Tracker:** ByteTrack (MIT) is online and deterministic. With a static camera, no camera-motion compensation is needed.
- **Ground plane:** a homography from 4 or more road-plane points taken from camera.md (lane width, stop line, crosswalk). A useful point: **TTC and PET are unaffected by a global scale error**, because distance and speed scale together. DRAC and deceleration scale linearly. So the projective part of the homography (removing perspective) matters more than getting the absolute scale right. As a scale sanity check, the median car length should come out near 4.5 m.
- **Position:** the bottom-centre of the box. For each class use a capsule footprint: car 4.5×1.8 m, bus 12×2.5 m, truck 8×2.5 m, motorcycle 2×0.8 m, bicycle 1.8×0.6 m, pedestrian a 0.5 m disc.

### 3.2 Kinematics noise (computed in `kin_noise.py`)

Causal least-squares fit evaluated at the newest sample, per 1 m of i.i.d. position noise:

| Rate | Window | Velocity (linear fit) | Acceleration (quadratic fit) |
|---|---|---|---|
| 12.5 Hz (stride 2) | 1.0 s | 0.93 /s | 6.98 /s² |
| 12.5 Hz | 1.5 s | 0.48 /s | 2.36 /s² |
| 12.5 Hz | 2.0 s | 0.33 /s | 1.22 /s² |
| 8.33 Hz (stride 3) | 1.5 s | 0.62 /s | 3.10 /s² |

With about 0.3 m of ground-plane jitter (an estimate), velocity over a 1 s window has σ ≈ 0.28 m/s, which is fine. Acceleration needs a window of at least 1.5 s (σ ≈ 0.7 m/s² at 12.5 Hz) before a −3.5 m/s² braking threshold is about 5σ. Hard-braking detection therefore has about 0.5–0.75 s of built-in latency.
- **Stride:** stride 3 is only 15–30% noisier than stride 2. Use stride 2 when the budget allows (better association at speed), and stride 3 for inputs larger than 1080p.
- **ID switches:** if the implied acceleration exceeds 8 m/s², reset that track's kinematic buffer.

### 3.3 Pairwise conflict measures

Candidate pairs: within 40 m, at least one road user moving faster than 1.5 m/s, and both tracks at least 0.8 s old.

| Measure | Definition (causal) | Initial thresholds / literature |
|---|---|---|
| **TTC** (constant velocity) | First τ ∈ [0, 5] s (0.1 s steps) at which the two footprint capsules overlap (+0.3 m margin) | 1.5 s is the classic critical value; SSAM uses max TTC 1.5 s by default (FHWA-HRT-08-051). Use 3.0 s for "elevated". |
| **TTC consistency** | Share of the last 0.8 s in which TTC fell at about 1 s per s (±0.4) | Separates a real collision course from car-following, where TTC hovers. This is the main false-alarm filter. |
| **MTTC** | TTC with measured acceleration, clamped to [−8, +3] m/s², no reversing | Ozbay et al. 2008 (TRR 2083) |
| **Closest approach** | d_min and t_min over 5 s | Adds risk when d_min < 1.5 m, even if TTC is infinite (near-miss geometry) |
| **Predicted PET / T2 / TAdv** | For crossing paths: predicted arrival and exit times at the conflict point | Laureshyn et al. 2010; SSAM max PET 5 s. Use ≤ 2 s elevated, ≤ 1 s critical. |
| **DRAC** | Rear-end: Δv²/(2·gap). Crossing: v²/(2·distance to conflict point) for the second arrival | Archer (2005) threshold of 3.35 m/s², as cited by Fazekas et al. 2017 |
| **Closing speed** | −d‖Δp‖/dt | Severity proxy; above 8 m/s counts as high |

### 3.4 Single-agent precursors (lead time comes from these)

- **Hard braking:** longitudinal acceleration ≤ −3.5 m/s² for ≥ 0.4 s.
- **Swerving:** yaw rate > 0.35 rad/s at speed > 5 m/s, or lateral acceleration > 3 m/s² that the lane geometry does not explain.
- **Red-light runner:** signal is red (from a signal-lamp colour ROI if visible, otherwise inferred from cross-flow) and the deceleration needed to stop at the line exceeds 4 m/s².
- **Wrong way:** heading more than 120° from the lane direction for ≥ 1 s.
- **Pedestrian on carriageway outside a crossing** with a vehicle TTC < 4 s.
- **Loss of lane discipline or erratic heading:** heading variance over 2 s.
- **Speed outlier:** above the scene's 99th percentile.

These are the causal counterparts of Part A's classes. Build them once as a shared library of causal violation flags. Part A adds hindsight on top; Part B uses them as they are.

## 4. Combining features into one risk score, and calibration

### 4.1 Initial hand-set model (before any data)

Per pair logit:
```
u = -6.0 + 2.5*clip((3.0-TTC)/1.5, 0, 2)         # 0 at 3 s, +2.5 at 1.5 s, +5 at 0 s
         + 1.5*clip((DRAC-2.0)/1.35, 0, 2)        # +1.5 at 3.35 m/s^2
         + 1.0*clip((2.0-PETpred)/1.0, 0, 1.5)
         + 1.0*clip((1.5-dmin)/1.5, 0, 1)
         + 1.0*clip((v_close-3)/5, 0, 1)
         + 1.0*ttc_consistent + 1.0*violation_flag + 0.5*vru_involved + 0.5*evasive_now
         - 1.5*[both speeds < 2 m/s]              # slow queue
         - 1.0*min(1, conflict_age/6 s)           # habituation: persistent pairs are benign
```

Frame risk and latent:
- R = 1 − Π(1 − σ(u)) over the **top 3 pairs only**. Using all pairs saturates the result in dense traffic.
- z = logit(R).

With these weights, an alarm needs roughly TTC ≈ 1.5 s, DRAC ≥ 3.35 m/s², a consistent approach and a closing speed of about 8 m/s. The final offset is set by the false-alarm budget (section 5).

### 4.2 Two outputs combined into one score

Use two outputs:
- a **ranking head** r ≈ P(accident within 5 s), which drives AP;
- an **alarm decision** a, which drives F1 and TTA and carries habituation, a small hysteresis and the false-alarm budget.

Emit `score = 0.5 + 0.4999*r` when a is on, and `0.4999*r` when it is off. That score is ≥ 0.5 exactly when the alarm is on, it ranks by r inside each band, and it survives the 4-decimal rounding.

Keep a single-model mapping `σ(z − z_on)` if you do not need separate heads. With no hysteresis the two options are identical (toy results).

### 4.3 Learning and calibrating with no accident labels in the samples

Sources of positives, in priority order:

1. **Collision injection in the target scene (trajectory level).** Track all sample videos and cache the tracks. For pairs whose paths cross or share a lane, time-shift one trajectory so the two reach the conflict point together; that moment is s. For rear-ends, add a synthetic hard stop for the leader while the follower reacts 1.5 s late. Also include cases with a sudden heading change 1–2 s before impact, so that constant-velocity TTC is not the only way to succeed. Pixels are not needed, because the risk core consumes tracks. After contact, freeze both users, which lets you test the aftermath suppression. Label frames with the exact metric rules.
2. **Public CCTV clips with an impact frame.** The ACCIDENT benchmark (arXiv 2604.09819, CVPR-W 2026) has 2,027 real CCTV clips annotated with the impact frame ("first visible contact", the same convention as this task). Clips are 1–114 s (median 26.8 s) at 4–50 fps; there are 507 IID-train clips, plus 2,211 synthetic CARLA clips with tracklets. The paper states that annotations are CC BY 4.0, code is Apache-2.0, and only clips whose upstream licence allows redistribution were included. Verify the licence on the Kaggle page before use. With no homography for these clips, compute **image-plane, scale-free features**: TTC and PET in seconds, distances in box-lengths, acceleration relative to speed. Train on the same features in the target scene. CADP and TAD are secondary sources; check their licences, and list every dataset in the README.
3. **Near misses as proxy positives.** AdaLEA (Suzuki et al., CVPR 2018) used a near-miss database for anticipation training. Near misses you annotate in the samples are ignored by Part B scoring, so leave them out of the training negatives.

Models:
- Logistic regression with 8–12 features and L2, or LightGBM with monotone constraints (risk decreasing in TTC and PET, increasing in DRAC and closing speed), `deterministic=True` and a fixed thread count.
- Train on the **5 s** labels for r. For the alarm head, the target is "an accident starts within 10 s", with episode-level evaluation.
- Isotonic calibration is optional and only useful for honest website plots. The score only needs to be monotone for AP and correctly placed at 0.5 for F1.

## 5. Alarm policy (initial values)

- **Threshold offset:** choose it so that false alarms on the annotated samples (outside near-miss windows) are about **1–2 per hour**, or about 0.3–0.5 × the expected number of test accidents. If the samples contain accidents, estimate accident density from them.
- **Hysteresis:** switch on at z ≥ z_on and off below z_on − 0.5 after a **0.5 s release**. No minimum-on time; the metric's 2 s merge does the debouncing.
- **Bridge short gaps:** for the *same pair*, bridge gaps of ≤ 3 s before any contact is detected.
- **Maximum alarm duration: 8 s.** After that, turn off and re-arm only once z has dropped below z_off and risen again. An alarm older than 10 s cannot match anything anyway.
- **Contact suppression:** a detected contact is capsule overlap plus closing speed above 2 m/s at the previous step plus both users decelerating hard or stopping within 1.5 s. After one, apply u −= 6 to pairs involving those tracks or within 15 m, for 30 s.
- **Warm-up:** return 0 until tracks are at least 0.8 s old. Never let the score depend on how far into the video the frame is.

## 6. Literature and how much of it transfers

Dashcam anticipation:
- DSA-RNN (Chan et al., ACCV 2016): dynamic spatial attention over detections, LSTM and an exponential anticipation loss.
- AdaLEA (Suzuki et al., CVPR 2018): adaptive early-anticipation loss with a near-miss incident database.
- UString (Bao et al., ACM MM 2020): relational graph plus Bayesian uncertainty; introduced the CCD dataset. It reports 72.22% AP and 3.53 s mTTA on DAD.
- DRIVE (Bao et al., ICCV 2021): reinforcement learning with visual explanation.
- GSC (Wang et al., IEEE T-IV 2023): graph plus spatio-temporal continuity.
- CRASH (Liao et al., ACM MM 2024): object-aware and context-aware attention.
- "Accident Anticipation via Temporal Occurrence Prediction" (NeurIPS 2025) criticises frame-level binary labels. It predicts accident scores at several future horizons and reports recall and TTA **under false-alarm-rate constraints**, which is conceptually the same as this task's alarm term.
- The Nexar challenge (2025) scores mAP at time-to-event 0.5, 1.0 and 1.5 s.

These models do not transfer to a fixed camera: they rely on ego-motion and a first-person view, and their AP is usually per video rather than pooled per frame. What does transfer is the pairwise, object-centric reasoning, time-weighted losses, and evaluation under a false-alarm budget.

Fixed camera:
- CADP (Shah et al., AVSS 2018) forecasting on CCTV with Faster R-CNN and an LSTM reported **47.25% AP and 1.684 s TTA**.
- The ACCIDENT benchmark shows that heuristic detectors score about 0.26–0.29 against 0.98 for humans on temporal localisation. Low-quality CCTV is hard, which argues for explicit kinematics.
- The traffic-conflict literature fits this setting best: FHWA SSAM (TTC 1.5 s, PET 5 s), Saunier & Sayed 2008 (collision probability from predicted motion, fixed-camera video), Laureshyn et al. 2010 (TAdv/T2), and Archer 2005 (DRAC 3.35 m/s²).

## 7. Compute, determinism and the grey zone

**Budget (estimate, 5-minute clip, 7,500 frames).**
- The harness decodes every frame for Part B at native resolution. On the dev laptop, OpenCV `cap.read` ran at 85 fps for 1080p and **22 fps for 4K**. The machine may have been loaded by other processes; the eval machine has 8 cores.
- Resizing to 640 px costs about 1 ms (measured).
- At 1080p: decode 30–90 s, plus 3,750 detector calls × about 12–20 ms (45–75 s), plus tracking and features at about 2 ms per call. Part B totals about **0.3–0.6× duration**.
- At 4K, decoding alone is about 1.1× duration, which leaves Part A only about 1.3× before a safety margin. **Measure on the real samples on day 1.**
- Target Part B ≤ 0.6× and Part A + Part B ≤ 2.0× duration.
- Stride comes from `meta` only: 2 for ≤ 1080p, 3 above.
- Optional speed-up: **fixed-lag pipelining.** A worker runs detection on frame t, and `step(t+2)` blocks until frame t's result is ready. This overlaps GPU work with harness decoding and stays deterministic. Never let the returned score depend on whether a thread happened to finish.
- There are no batches; batch size is 1. Load the model once and reuse the same model object as Part A; sharing weights is not sharing outputs.

**Determinism.**
- Seed everything, set `cudnn.benchmark=False` and `cudnn.deterministic=True`, and use `torch.use_deterministic_algorithms(True)` if possible (PyTorch reproducibility notes).
- ByteTrack is deterministic given identical detections.
- Base any emergency stride increase on the estimator's own elapsed time only. It should never trigger in normal runs; log it if it does.
- Run the samples twice and diff the risk arrays.

**Grey zone: reusing Part A's detections.** `solution.py` says "do not reuse Part A results that were computed with access to future frames". The PDF FAQ is stricter: "Part A CAN use Part B's risk curve; reverse NOT allowed."
- Technically, a module-level cache filled by `detect_events` is readable from `step`, because both run in the same process with A first.
- Even detections that are strictly per frame are "Part A output", and breaking a rule means disqualification.
- **Recommendation:** Part B runs its own detector and tracker inside `step`. The recompute costs only about 0.2–0.4× duration at 1080p.
- Question for organizers: "May RiskEstimator.step(t) read per-frame detections for frame t that detect_events cached, when each entry depends on frame t only?"
- The allowed direction is useful. `detect_events` can run the same `RiskEstimator` code causally during its own pass, then add hindsight (did contact happen, did both users stop) to classify accident vs near_miss and to place the segment boundaries.

## 8. Architecture and pseudo-code

```python
_MODEL = None
def _model():                       # loaded once per process, shared read-only with Part A
    global _MODEL
    if _MODEL is None: _MODEL = Detector("weights/det.pt", imgsz=640, half=True, conf=0.25)
    return _MODEL

class RiskEstimator:
    def reset(self, meta):
        self.fps = meta.get("fps") or 25.0
        self.stride = 2 if meta["width"]*meta["height"] <= 1920*1080 else 3
        self.k, self.last = -1, 0.0
        self.tracker = ByteTrack(frame_rate=self.fps/self.stride, track_thresh=0.35,
                                 match_thresh=0.8, track_buffer=int(1.5*self.fps/self.stride))
        self.kin = Kinematics(scene=SCENE, win_v=1.0, win_a=1.5, min_age=0.8)
        self.shaper = AlarmShaper(z_on=Z_ON, dz_off=0.5, release=0.5, max_on=8.0)
        self.supp = ContactSuppressor(radius_m=15, dur_s=30)
    def step(self, frame, t_sec):
        self.k += 1
        if self.k % self.stride: return self.last
        try:
            tracks = self.tracker.update(_model()(frame))
            st = self.kin.update(tracks, t_sec, signal=SCENE.read_signal(frame))
            P = candidate_pairs(st, max_d=40)
            F = pair_features(P, horizon=5.0, dt=0.1)      # TTC, consistency, MTTC, dmin, PETpred, DRAC, vclose
            u = logit_model(F, context_flags(st, P))       # hand-set, later learned weights
            u = self.supp.apply(u, P, st, t_sec)
            z = logit(noisy_or(sigmoid(topk(u, 3))))
            self.last = self.shaper(z, t_sec)              # two-band score, 4-decimal safe
        except Exception as e:
            log_once(e)                                    # never raise: one exception kills the whole curve
        return self.last
```

Code structure: `src/perception.py`, `src/kinematics.py`, `src/conflicts.py` (pure numpy, unit-tested), `src/risk_model.py` and `src/alarm.py`. The risk core takes **track lists**, so cached tracks can be replayed offline for parameter sweeps in seconds instead of GPU hours.

## 9. Validation protocol

1. **Dev labels:** annotate all samples with the official conventions. Be exhaustive about `near_miss`, because it defines the free windows.
2. **Negative calibration:** run the harness on the samples. Measure false alarms per hour outside near-miss windows and the distribution of high-score negatives. Set Z_ON from the false-alarm budget. Use leave-one-video-out to see how stable the threshold is.
3. **Positive checks:**
   - Collision-injected tracks: recall of alarms at lead ≥ 0.5 s, TTA histogram, AP.
   - ACCIDENT impact-frame clips in image-plane mode: recall and lead only, since domain shift makes their AP not comparable.
4. **Composite estimate:** splice positive-episode risk curves into the sample negatives at an assumed accident density (for example 3 and 6 per hour). Score with the **official** `evaluate_part_b` (the harness in `partb_sim.py` shows how) and bootstrap over videos.
5. **Ablations:** stride 1/2/3; n vs s detector; TTC-only vs full features; with and without consistency, habituation and suppression. These double as website extra credit.
6. **Runtime and determinism:** read `part_b_sec` and `total_sec` from the harness log in predictions.json, and diff two runs.
7. **Website plot:** for each sample, show the risk curve with the θ = 0.5 line, alarm starts, shaded positive, ignored and near-miss windows, and the TTA of each matched alarm.

## 10. Cross-cutting implications

- Part A's accident and near_miss detectors should reuse the same conflict engine plus hindsight.
- Violation flags are one shared library.
- Test resolution affects the whole time budget. At 4K, Part A needs NVDEC or light decoding.
- Near-miss annotation quality on the dev set drives both Part A F1 and the Part B free windows.
- Part B is worth up to 0.18 of the elimination score (0.6 × 0.3), and is not scored at all if the test set has no accidents. Give it about one person-week, not more.


## Top recommendations
- Build Part B as its own causal loop inside step: module-level detector singleton shared with Part A, ByteTrack, ground-plane kinematics and pairwise conflict measures. Never read any data produced by detect_events; the FAQ says 'reverse NOT allowed'.
- Core features: capsule TTC over a 5 s horizon that must be falling consistently (about -1 s per s), DRAC (3.35 m/s^2 critical), predicted PET/T2, closest approach, closing speed, hard braking (<= -3.5 m/s^2 over a window of at least 1.5 s), and causal violation flags (red-light runner, wrong way, pedestrian on carriageway). Combine them with a hand-set logistic model plus noisy-OR over the top 3 pairs.
- Set the 0.5 threshold with a false-alarm budget (about 0.3-0.5 false alarms per expected test accident, about 1-2 per hour of samples outside near-miss windows), not at 'probability 0.5'. The F1-optimal threshold is about F1/2 (0.2-0.3), and one false alarm costs as much as about 5 s of lead on one accident.
- Keep alarms short: release of 0.5 s or less, let the metric's 2 s merge do the debouncing, bridge gaps of up to 3 s only for the same pair, cap alarms at 8 s, and apply 30 s / 15 m suppression after a detected contact. Use habituation to damp persistent slow-traffic conflicts. Aim for alarm onset at least 0.5-1 s before contact so annotation jitter does not drop the alarm.
- Engineering safety: wrap step in try/except (one exception erases the whole video's curve), choose the stride from meta only (2 for <=1080p, 3 above), no wall-clock-dependent async, fix seeds and cuDNN settings, and keep Part B at or below 0.6x duration. Measure decode time at the real test resolution on day 1, since 4K decoding alone takes about 1.1x.
- Validate with the official evaluate.py: exhaustive near-miss labels on the samples, collision injection at the track level in the target scene, and ACCIDENT impact-frame clips (image-plane scale-free features). Tune offline by replaying cached tracks and report the ablations on the website.


## Open questions
- May RiskEstimator.step(t) read per-frame detections for frame t that detect_events cached (strictly per-frame, no tracker smoothing), or is any data flow from Part A to Part B forbidden? The FAQ says the reverse is not allowed; ask in the hackathon channel.
- How many accidents does the hidden test set contain, and at what density per hour? This sets the false-alarm budget. If there are none, Part B is not scored (M = Score_A).
- Are near misses annotated exhaustively in the test ground truth? Their [s-5, e] windows are free for Part B, so the answer changes how aggressive the alarms can be.
- What are the test resolution and codec? The harness decodes every frame for Part B; at 4K that alone takes about 1.1x duration on the dev laptop.
- Is the traffic signal visible in frame (camera.md)? This decides whether red-light-runner risk can use a colour ROI or must infer the phase from flows.
- How precisely is accident start annotated (the exact contact frame)? Late alarms (less than 0.5 s before s) are vulnerable to jitter.
- License of the ACCIDENT dataset clips: the arXiv abstract page and the paper text differ (CC BY-NC-ND 4.0 vs annotations CC BY 4.0). Verify on the Kaggle and GitHub pages before training on it.


## Key claims (as submitted for verification)
- Official evaluate.py: F1_alarm uses alarms = runs of score>=0.5 merged when next start minus previous run's last on-frame < 2.0 s; alarms starting on ignored frames are dropped; earliest unmatched alarm in [s-10, s) matches; other alarms count as false; AP is sklearn-style step AP over tie groups, pooled, chance-normalised; M = Score_A if the test set has no accidents — Starter kit evaluate.py (downloaded to scratchpad\kit\wiut_cv_scripts\evaluate.py)
- run_submission.py creates a new RiskEstimator() per video, calls step on every frame (official stride 1), clamps and rounds scores to 4 decimals, uses t = idx/fps; an exception in step empties that video's whole risk curve; going over the 3x budget empties both events and risk for the video — Starter kit run_submission.py (scratchpad\kit\wiut_cv_scripts\run_submission.py)
- With m matched alarms, A alarms and N accidents, F1_alarm = 2m/(A+N); at N=10 one false alarm costs about 0.0095 Score_B while one extra matched alarm adds about 0.033, and 1 s of extra lead on one accident adds 0.002 — reasoning (derived from the evaluate.py formula; computed in scratchpad)
- For calibrated probability outputs, the F1-optimal decision threshold equals half the optimal F1 score — https://arxiv.org/abs/1402.1892
- FHWA SSAM uses default conflict-identification thresholds of TTC 1.5 s and PET 5.0 s — https://www.fhwa.dot.gov/publications/research/safety/08051/04.cfm
- Archer (2005) proposed a DRAC critical threshold of 3.35 m/s^2, widely adopted in later research — https://onlinelibrary.wiley.com/doi/10.1155/2017/8376572
- Modified TTC (MTTC), which includes relative acceleration, was proposed by Ozbay, Yang, Bartin and Mudigonda (TRR 2083, 2008) — https://journals.sagepub.com/doi/10.3141/2083-12
- Saunier and Sayed (2008) proposed a probabilistic framework computing collision probability from motion prediction of road users tracked in video — https://doi.org/10.3141/2083-11
- Laureshyn et al. (2010) introduced Time Advantage and T2 (predicted arrival time of the second road user; equals TTC on a collision course) — https://www.tandfonline.com/doi/full/10.1080/01441647.2018.1442888
- CADP (CCTV) accident forecasting with Faster R-CNN plus Accident LSTM achieved 47.25% AP and 1.684 s average time-to-accident — https://arxiv.org/abs/1809.05782
- UString reports 72.22% AP and 3.53 s mTTA on the dashcam DAD dataset — https://github.com/Cogito2012/UString
- ACCIDENT benchmark: 2,027 real CCTV clips (plus 2,211 synthetic CARLA clips) annotated with the impact frame (first visible contact), clips 1-114 s at 4-50 fps, 507 IID-train clips; annotations CC BY 4.0, code Apache-2.0; heuristic baselines about 0.26-0.29 vs human 0.979 — https://arxiv.org/html/2604.09819v1
- Accident Anticipation via Temporal Occurrence Prediction (NeurIPS 2025) predicts accident scores at multiple future horizons and evaluates recall/TTA under false-alarm-rate constraints — https://arxiv.org/abs/2510.22260
- YOLO11n/s run at 1.5/2.5 ms and YOLO26n/s at 1.7/2.5 ms on a T4 with TensorRT10; YOLO26 is AGPL-3.0 and NMS-free end-to-end — https://docs.ultralytics.com/models/yolo11/ ; https://docs.ultralytics.com/models/yolo26
- ByteTrack is MIT-licensed — https://github.com/FoundationVision/ByteTrack/blob/main/LICENSE
- Nexar dashcam challenge scores mean AP at time-to-event 500, 1000 and 1500 ms — https://arxiv.org/html/2503.03848
- OpenCV cap.read decoded a 1080p25 test clip at about 85 fps and a 4K clip at about 22 fps on the local dev laptop (12 logical CPUs, possibly loaded); resizing to 640 px costs about 1 ms — local measurement in scratchpad (t1080.mp4, t2160.mp4)
- Causal quadratic LSQ acceleration at 12.5 Hz has noise std of 2.36 m/s^2 per 1 m position noise over a 1.5 s window (6.98 over 1.0 s), so hard-braking detection needs a window of at least 1.5 s — reasoning (kin_noise.py computation in scratchpad)
- In a toy simulation scored with the official evaluate_part_b, the best alarm threshold consistently fell at about 0.27-0.36 false alarms per accident; hold/latching slightly reduced AP; post-contact suppression raised Score_B by about 0.02 — reasoning (partb_sim.py in scratchpad; synthetic, qualitative only)
- PyTorch reproducibility: seed with torch.manual_seed, set cudnn.benchmark=False and cudnn.deterministic=True, and use torch.use_deterministic_algorithms(True) — https://docs.pytorch.org/docs/2.14/notes/randomness.html
