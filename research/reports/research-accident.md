<!-- source: deep-research workflow wf_294a4b01-fc2, agent research:accident -->

# Detecting `accident` and `near_miss` from a fixed CCTV camera: research report

Dimension: Part A detection of the two classes where the organizers say learned models help most. Date: 2026-09-25. Items marked **(est.)** are my own estimates, with the reasoning shown. Everything else is cited in `key_claims`.

---

## 0. TL;DR (what to build)

1. **Base both classes on the trajectory pipeline, not on a model that looks at pixels.** Use the shared detector and tracker, auto-scale the image to metres, and analyse road-user pairs (distance between footprints, closing speed, TTC/PET, abrupt Δv and heading change, what happens after impact). This is the only approach that can place START/END to within about 0.3 s. That precision is needed at IoU 0.7, because accident segments (first contact until everything is at rest) are probably only 2–6 s long.
2. **Add a small learned verifier that looks at crops.** Train X3D-S/M (Apache-2.0, 3.8M params, 3–7 GFLOPs) or VideoMAEv2-S (CC-BY-NC) on 16-frame crops around the candidate pair. Positives come from public fixed-camera accident clips. Hard negatives come from our own sample videos: every candidate the rules fire on in the samples that is not an accident. The verifier removes false positives from occlusion, queues and buses. It does not set boundaries.
3. **Treat a VLM as an optional ablation, not a core component.** Four hard facts push this way. The weights cap is 5 GB total, and a 2B VLM in fp16 already takes 4.3–4.7 GB. The T4 has no bf16. There are reports of fp16 overflow in the Qwen-VL family, and vLLM does not support Qwen3-VL on Turing. Published CCTV results show VLM impact-time errors of about 1–4 s with a late bias. Use a VLM offline for triage and pseudo-labels. Ship one at runtime only if it adds at least 0.05 accident F1 on the proxy dev set.
4. **near_miss is rules-only:** a conflict (TTC ≤ 1.5 s or PET ≤ 1.0 s) plus an evasive action (Δv ≤ −3 m/s within 1 s, or a lateral swerve ≥ 0.8 m) plus no contact. No CCTV dataset has temporal near-miss labels, so a learned model has nothing to train on.
5. **Set the operating point with a formula.** Emit a candidate when its calibrated probability of being matched is greater than F1*/2 (Lipton et al.). Before the proxy dev set exists, require two independent kinds of evidence.
6. **Dev set.** Annotate every sample video with our own labels. The samples are also our false-positives-per-hour test on the target camera. Build a proxy accident set of 60–100 public fixed-CCTV clips from other cameras, relabelled with the hackathon conventions, and split it by camera.

---

## 1. Why these two classes need special treatment

### 1.1 How much they are worth
Score_A is a macro average over the classes present in the test set. Each class is worth 0.7/|C| of M, which is 0.42/|C| of the elimination score. With |C| ≈ 10, accident plus near_miss together are worth about 8.4% of elimination at most. Part B is worth 0.3 × 0.6 = 18% of elimination and depends on the same pair/TTC machinery. **The pair-interaction module is therefore the highest-leverage code in the project.** It serves accident, near_miss, Part B, and partly failure_to_yield and jaywalking. The VLM is the lowest-leverage, highest-risk piece.

Both classes are almost certainly in C. Part B is scored only on accidents, and its ignore rule refers to near_miss GT. So an absent class cannot dilute the score here, and plain per-class F1 is the right thing to optimise.

### 1.2 Tolerance to boundary error (computed: scratchpad `iou_sim.py`)
Maximum symmetric shift d that still gives IoU ≥ τ for a segment of length L: d ≤ L(1−τ)/(1+τ).

| L (s) | τ=0.3 | τ=0.5 | τ=0.7 |
|---|---|---|---|
| 2 | 1.08 s | 0.67 s | **0.35 s** |
| 3 | 1.62 | 1.00 | **0.53** |
| 5 | 2.69 | 1.67 | **0.88** |

Expected mean-over-τ match probability for a 3 s event, with Gaussian noise σ on each boundary: σ=0.25 s → 0.997; σ=0.5 s → 0.91; σ=1.0 s → 0.67; σ=1.5 s → 0.50.
- A predicted segment that is **too long costs less than one that is too short**. With one wrong boundary at τ=0.7: shorter by up to 0.3·L is tolerated, longer by up to 0.43·L. Padding each side by about σ/2 is near-optimal: 0.25 s when σ≈0.5 s, 0.5 s when σ≈1 s.
- VLM-grade timing (MAE 1–2 s) limits a 3 s event to about 50–65% expected matching. Kinematic timing at about 0.3 s gives about 99%. **Boundaries must come from trajectories.**

### 1.3 Start and end conventions to reproduce
- accident: START = first frame where contact is visible. END = all involved objects have stopped or left the frame. The ACCIDENT benchmark's human annotators agree on impact time almost perfectly (temporal score 0.979 at σ=1 s), so START is well defined. END is the fuzzier boundary: learn a bias offset from our own labels.
- near_miss: START = onset of evasive action (braking or swerve). END = road users clear of each other.
- Public labels use different conventions. A3D and DoTA start at "accident inevitable", which is earlier than contact. Their END in A3D is "participants recover or fully stop", which is close to ours. **Relabel any public clip used to tune boundaries.**

---

## 2. Dataset survey (the fixed CCTV domain is what matters)

| Dataset | View | Size | Temporal annotation | Licence / access 2026 | Use for us |
|---|---|---|---|---|---|
| **ACCIDENT** (CVPRW / CVPR 2026 challenge) | **Fixed CCTV** (real) + CARLA CCTV-style synthetic | 2,027 real + 2,211 synthetic clips; 1–114 s (median 26.8 s); 4–50 fps; 314p–3840p; ~66% poor quality | Impact time (first contact, mean of 3–5 annotators), impact point, 5 collision types. Synthetic also has boxes, masks, tracklets. Real-clip labels were withheld as the challenge test set | Kaggle `picekl/accident`. Annotations CC BY 4.0, code Apache-2.0. Real clips come from sources licensed for redistribution (CC-BY, open-government). Research use | **#1**: largest real fixed-camera accident pool. Use real clips as positives and as the proxy dev set (self-annotate START/END). Use synthetic tracklets to fit the kinematic model |
| **TAD** (Lv et al., TIP 2021) | Surveillance | 500 videos, 25 h (250 abnormal); 7 anomaly types incl. vehicle accidents, illegal turns, retrograde, pedestrian on road, road spills | Train: video-level only. Test (100): frame-level | Google Drive (frames only), via `ktr-hubrt/WSAL`. Licence not stated | **#2**: accident positives and many normal CCTV negatives. Also useful to other class owners |
| **CADP** (Shah et al. 2018) | CCTV (YouTube) | 1,416 segments; 205 with spatio-temporal annotation; avg 366 frames | Spatio-temporal (VATIC) for 205 | Project page; non-commercial research only | **#3**: extra positives. Relabel boundaries |
| UCF-Crime (RoadAccidents class) | Surveillance | 1,900 videos total, about 150 RoadAccidents (commonly cited, not verified) | Temporal annotations for the test split only (140 anomalous + 150 normal) | CRCV UCF, research use | Positives and negatives. Many videos are compilations or low quality |
| TU-DAT (Temple U., 2025) | CCTV + news + BeamNG simulation | ~210 real accident videos + ~65 simulated; 17,255 accident keyframes; 24–30 fps | Keyframes, plus trajectories and collision-type metadata | Free download (GitHub). Licence not stated on the site | Positives |
| TADS (2024) | Surveillance | 20 long videos → 966 accident clips, 259,891 frames, plus eye-tracking | Accident start/end positions | Baidu Cloud only (hard to reach). No licence stated | Positives with start/end, if downloadable |
| AI City 2021 Track 4 | Freeway CCTV (Iowa DOT) | 100 train + 150 test, ~15 min each, 30 fps, 800×410 | Anomaly start time (crashes, stalled vehicles) | Still available after email approval. Terms not stated | Far-field freeway domain. Mostly useful for stopped_vehicle. Weak for intersections |
| TUM Traffic Accid3nD (2025) | Roadside cameras + LiDAR (highway) | 111,945 labelled frames; 2D/3D boxes, masks, track IDs; 25 Hz | Few accident scenarios (the ACCIDENT paper lists 12 clips) | CC BY 4.0 | Precise real tracks for validating kinematic features |
| IITH_Accident | City CCTV (Hyderabad) | 127,138 normal + 863 accident frames; 30 fps | Frame labels | Google Sites. Licence unclear | Positives (mixed traffic similar to South and Central Asia) |
| CrashSight (2026) | Roadside CCTV | 250 crash videos, 13K QA pairs | Phases (pre / crash / post) as QA | CC BY 4.0 (paper) | VLM evaluation. Little direct use |
| Roboflow `tammy/cctv-accident` and similar | CCTV stills | ~1.5k images | Boxes around wrecks, no timing | CC BY 4.0 | Optional "post-crash state" detector |
| DoTA | **Dashcam (domain gap)** | 4,677 clips, 1280×720 | anomaly_start/end + category + boxes | MIT; YouTube download script; ~55 GB | Pretraining only |
| CCD | Dashcam | 1,500 accident + 3,000 normal (BDD100K), 5 s / 50 frames | Per-frame binary labels, accident in the last 2 s | GitHub | Pretraining only |
| DAD | Dashcam (Taiwan) | 1,750 clips (620 accident), 100 frames @20 fps | Accident at frame 90 | Research | Pretraining only |
| A3D | Dashcam | 1,500 clips, 128,175 frames @10 fps | Start (inevitable) / end (recovered or stopped) | GitHub | Pretraining only |
| MM-AU | Dashcam | 11,727 videos, 2.19M frames | Accident windows + text | GitHub / HF | Pretraining only |
| Nexar (2025) | Dashcam | 1,500 (750 positive, **collisions and near-misses mixed**), 1280×720 @30, ~40 s | time_of_event, time_of_alert | Nexar Open Data License (attribution; no resale or re-identification) | Only dashcam near-miss source. Domain gap is too large for CCTV |
| TAU-106K (ICLR 2025) | Mixed | 106K clips/images; avg clip 10.3 s; accident avg 3 s | Temporal + spatial grounding | See repo | VLM fine-tuning, which we cannot ship at scale anyway |
| TNAD | Drone / fisheye at intersections | 106 videos, >75 min | Near-accident temporal + spatial | Availability unclear | The only CCTV-like near-miss set. Try to obtain it |

**Best 2–3 for this camera:** (1) **ACCIDENT real + synthetic**, (2) **TAD**, (3) **CADP**, with UCF-Crime RoadAccidents and TU-DAT as filler. Dashcam sets (DoTA, CCD, DAD, A3D, MM-AU, Nexar) have ego-motion and a different viewpoint. At most they are generic video pretraining. Do not train the final verifier on them.

**Data rules to respect.** Do not collect footage from the target camera by other means. Before adding a public clip, check that it is not the same view as the samples. List every dataset and its licence in the README. On the website, show only our samples and our own renders. Show only CC-BY clips from other sources, with attribution; never frames from research-only sets.

---

## 3. Approaches

### 3a. Trajectory and kinematic heuristics (recommended core)

**Metric scale without calibration.** The organizers say speed "cannot be measured without calibration". We can still get approximate metres. Option 1: a homography from 4+ points on known markings, with lanes about 3.5 m wide (from camera.md and the samples). Option 2, which needs no manual work and also runs on proxy clips from other cameras: fit metres-per-pixel as a function of image row y from the median car box width, taking a car as about 1.8 m wide and 4.5 m long. Use the bottom-centre of each box as its ground point. About 20% scale error only shifts thresholds a little, and the rules below are mostly ratios.

**Noise budget, computed with a Savitzky–Golay (order 2) analysis in `accel_noise.py`.** With 0.2 m position noise at 25 fps:
- 0.4 s window: σ_v ≈ 0.48 m/s, **σ_a ≈ 8.5 m/s²**, which is useless.
- 1.0 s window: σ_v ≈ 0.14 m/s, σ_a ≈ 1.1 m/s².
- 1.4 s window: σ_a ≈ 0.46 m/s².

At 12.5 fps the noise is roughly 1.3–2× worse. Consequences:
- **Never threshold instantaneous acceleration.** Threshold the velocity change over a 1 s window (Δv₁ₛ), and find onsets with a change-point (hinge) fit on v(t). A two-segment piecewise-linear fit places the onset of a 3–5 m/s² brake to within about 0.1–0.2 s (est.).
- Part A may use future frames, so run an offline RTS or Kalman smoother, or a centred SG filter.
- **Re-track candidate windows at native 25 fps.** A 2026 CCTV accident pipeline found that tracking on sub-sampled frames swaps IDs around the collision.

**Accident signals (thresholds are starting values to tune on the dev set):**
1. *Contact geometry.* Model each footprint as an oriented rectangle of typical size (car 4.5×1.8 m, bus 12×2.5, truck 8×2.5, motorcycle 2×0.8, pedestrian 0.5×0.5), with heading taken from the track. Contact candidate: gap ≤ 0.5 m, with closing speed ≥ 1.5 m/s during the previous 1 s. **Guard against occlusion:** in perspective views, overlapping image boxes usually mean one vehicle is behind the other. Require that the bottom edges are depth-consistent (|Δy_bottom| ≤ 0.15 × box height) or that the ground-plane gap is small. If segmentation masks are available, require mask contact.
2. *Kinematic shock* within [−0.3, +0.8] s of contact, on at least one participant:
   - Δv₁ₛ ≤ −max(2.5 m/s, 0.35·v_before), which is about 0.25–0.5 g sustained over 1 s. The impact itself is sharper.
   - or heading change ≥ 25° in 0.5 s at v ≥ 3 m/s, outside learned turning zones;
   - or lateral jump ≥ 0.8 m against a constant-velocity prediction;
   - or momentum transfer: a slow or stationary participant suddenly moves in the striker's direction.
3. *After impact:* the involved vehicles are stationary (v < 0.5 m/s) for ≥ 2–3 s, outside normal stop zones and not in a queue (no lead vehicle within 8 m).
4. *Vulnerable road users:* a pedestrian or two-wheeler box's h/w ratio drops by more than 40% within 0.5 s (a fall), a rider and bike separate, or a person's track ends under or behind a vehicle.
5. *Single vehicle against a fixed object:* Δv ≥ 5 m/s within ≤0.5 s, no other road user within 3 m, near the road edge, median or a pole. Possibly a rollover, shown as a sudden change in box aspect ratio.
6. *Track pathology:* a track ends away from the frame border and known occluders, or two IDs merge into one blob, within 2 s of signals 1 or 2.

**Pros:** frame-accurate boundaries, runs on CPU, explainable (good for the report and website), transfers across cameras, and shares code with Part B's TTC.
**Cons:** depends on tracker quality at night, in rain and under occlusion. The published pixel-only zero-shot baseline (frame-difference z-score, τ=1.5) scored only 0.25 on ACCIDENT, so trajectories plus a learned verifier are needed.
**Cost:** negligible beyond detection and tracking.

### 3b. Clip-level video classifiers (use as a verifier, plus a cheap safety net)

| Model | Cost | Licence | Notes |
|---|---|---|---|
| **X3D-S / X3D-M** (PyTorchVideo) | 2.96 / 6.72 GFLOPs per view, 3.79M params; K400 73.3% / 75.9% | Apache-2.0 | Best cost/benefit. Milliseconds per clip on T4 (est.). Trains on a Kaggle T4 in a few hours (est.) |
| **VideoMAEv2 ViT-S / ViT-B distilled** | ~22M / 86M params; K400 83.7% / 86.6% | Repo MIT; **HF weights CC-BY-NC-4.0** | Stronger features. Non-commercial is acceptable for a hackathon, but list it. The CVPR 2026 SynCrash entry used VideoMAEv2-g (too big for us) |
| MViT-B 16×4 | 70.8 GFLOPs | Apache-2.0 (PyTorchVideo) | Heavier. No clear gain for binary impact detection |
| SlowFast R50 8×8 | 65.7 GFLOPs | Apache-2.0 | Heavier |
| UniFormerV2, InternVideo2-S/B | ViT-B scale | Apache-2.0 repos | Good, but more integration work |
| TimeSformer, VideoMAE v1 | ViT-B | CC-BY-NC (VideoMAE weights) | No advantage |

**How to use them:**
- **Crop, don't downsample the whole frame.** A CCTV crash covers a small part of a 1080p frame; resizing the whole frame to 224² destroys it. Crop 1.5–2× the union box of the candidate pair, resize to 160–224, and feed 16 frames at 12.5 fps (1.28 s).
- Use a three-way head: *normal / impact / aftermath*. The impact class peaks at first contact, and the aftermath class confirms stopped or damaged vehicles.
- **Positives:** windows [s−0.5, s+1.5] around impact in ACCIDENT real, CADP, TAD and UCF-Crime clips, with s from our own relabelling.
- **Hard negatives from the target domain:** every rule candidate in the sample videos that is not an accident (occlusion passes, queue stops, buses at stops, turning conflicts). These are the most valuable training data we have.
- Train with focal loss or balanced sampling. Calibrate with temperature scaling on the proxy dev set.
- **Safety net:** run the same model on the full frame plus 2×2 tiles with a 1 s stride. That is about 1,500 inferences for 5 minutes of video, roughly 10 s on T4 (est.). It catches crashes the tracker misses. A hit becomes a candidate, never an event on its own.
- **Boundaries:** peaks of the classifier score are too smooth for IoU 0.7. Always hand the peak to the kinematic refinement in 3a.

### 3c. Open-weights VLMs as a verifier: constraints and verdict

**Hard constraints from the spec and the hardware:**
- **5 GB total weights.** Qwen3-VL-2B-Instruct is 4.26 GB (Apache-2.0). Qwen3.5-2B is 4.55 GB (Apache-2.0; hybrid Gated-DeltaNet; *thinking on by default*, so disable it with `enable_thinking=False`). InternVL3.5-2B is 4.71 GB (Apache-2.0). Qwen3-VL-4B is 8.89 GB, which is over the cap: it needs 4-bit AWQ/GPTQ or a saved bitsandbytes-nf4 checkpoint (~2.5–3 GB, est.). FP8 variants do not help, because T4 has no FP8. Gemma 4 E2B-it (Apache-2.0, April 2026) is 10.2 GB, also over the cap. Qwen2.5-VL-3B uses the *Qwen Research Licence* (non-commercial). It is legal here but has no advantage.
- **Turing has no bf16.** Qwen-VL models were trained in bf16, and fp16 overflow producing gibberish, NaN or device asserts has been reported for Qwen2-VL and Qwen2.5-VL on T4. vLLM has no attention backend for Qwen3-VL on Turing: the issue was closed as not planned, and `--enforce-eager` runs at under 10 tok/s. In practice this means HF transformers with SDPA in fp16, a NaN guard, and possibly keeping the vision tower in fp32.
- **Measured T4 reference point:** Gemma-4-E2B under vLLM 0.29 prefilled 4,096 tokens in about 7.2 s. Budget roughly 1–4 s per verifier query of about 1k visual tokens (est.): 8 frames at 448² with a temporal patch of 2 is about 4 × 196 = 784 tokens for Qwen3-VL, the answer is a single token, and the score is the logit for "yes".
- **Evidence on how precise VLM timing is on CCTV:**
  - On ACCIDENT, Molmo-7B zero-shot scored 0.343 temporal, against 0.266 for optical flow and 0.979 for humans.
  - Qwen3-VL-32B coarse-to-fine combined with YOLO11x and BoT-SORT scored 0.549, but needed **40–90 s per clip on an H100**.
  - A hosted Qwen3-VL-Plus pipeline showed a **+1.55 s late bias**, with MAE growing from 0.94 s on clips ≤10 s to more than 4 s on clips ≥20 s. It tends to pick wreckage frames instead of the moment of impact.

**How to use one if we try it:**
- Candidate crop, 8 frames at 0.5 s spacing over [t_c−1.5, t_c+2.5].
- Draw a red box and a blue box on the two participants (visual prompt).
- Prompt: "Fixed traffic camera, frames 0.5 s apart. Does the road user in the RED box physically touch the road user in the BLUE box? Answer yes or no."
- Score = softmax(logit_yes, logit_no). Use greedy decoding and a fixed seed; logit scoring is deterministic up to float noise.
- Stack the score with the kinematic and clip scores in a logistic regression.
- Capacity (est.): 20–40 queries per 5-minute video within budget.

**Verdict: not worth making core.** It is worth one day as an ablation, and only if the detector and classifier weights total ≤0.7 GB. Ship it only if accident F1 (averaged over τ) improves by at least 0.05 on the proxy dev set with zero NaNs or crashes. Otherwise keep it out of the runtime and use larger open VLMs offline (for example Qwen3-VL-8B/32B on cloud GPUs) to triage candidates and pre-label proxy clips. Commit those labels and scripts so the work is reproducible. Do **not** use closed APIs for labelling; the "everything reproducible" rule makes that a grey area.

### 3d. Anomaly-detection approaches
- **Normality atlas from the samples (recommended; cheap and target-domain).** Build it from tracks in all sample videos: per grid cell, the direction histogram, speed distribution, and *stop-zone probability* (stop lines, bus stops, parking, queues). A stop in a cell where vehicles never stop is a strong accident or stopped_vehicle cue. Vehicles stopping in stop zones are hard negatives. The same atlas serves wrong_way and illegal_turn (cross-cutting).
- A GMM or KDE likelihood of (position, velocity, heading) gives a trajectory negative-log-likelihood (NLL) feature.
- Weakly supervised MIL models (RTFM or UR-DMU style, trained on UCF-Crime or TAD) are class-agnostic, and their boundaries are poor. At most, feed them in as a candidate-generator feature. Frame-prediction autoencoders are not worth the time.

---

## 4. near_miss specifics
Definition: sharp braking or swerving to avoid a collision, with no contact. Required logic:
1. **Conflict.** For pairs whose footprint discs (radius = half width + 0.3 m) would meet under constant velocity: min TTC ≤ 1.5 s. For crossing paths: PET ≤ 1.0 s. These are standard surrogate-safety cut-offs; a Rutgers CAIT video study used TTC 1.5 s, DST 3 m/s² and PET 1 s. Also include vehicle–pedestrian pairs, which will co-occur with jaywalking or failure_to_yield events. That overlap is allowed because the classes differ.
2. **Evasive action by at least one party** in [t_minTTC − 2 s, t_minTTC + 0.5 s]:
   - Braking: Δv over 1 s ≤ −3 m/s (≥0.3 g) with v_before ≥ 4 m/s. Use 0.5 g as a "high-confidence" tier; this is the 100-Car study's rapid-manoeuvre braking threshold, and its lateral threshold is 0.4 g.
   - Swerve: lateral offset from the lane-direction path (from the atlas) ≥ 0.8 m within 1.5 s, not explained by a lane change that started earlier.
   - A pedestrian stopping or stepping back sharply also counts.
3. **No contact:** the minimum footprint gap stays above 0.3 m, there is no post-impact shock, and no accident candidate exists on the same pair. Contact means accident only; near_miss is suppressed.
4. **Boundaries.**
   - START = hinge (change-point) of v(t) or of the lateral offset in [t_peak − 2 s, t_peak]. Fall back to the first time |a| exceeds 25% of its peak.
   - END = first time after closest approach when gap ≥ max(2 m, 1.5 × min gap), the gap is increasing and TTC > 3 s, or one party has left the conflict zone.
   - Pad each side by about 0.25 s.
5. **Precision first:** require conflict **and** evasive action. Braking at a red light and queue compression look like evasive braking, so exclude stops in stop zones when the lead vehicle is itself decelerating smoothly.
6. **No learned model.** No CCTV near-miss set with temporal labels is available: TNAD is uncertain, and Nexar is dashcam with near-misses mixed into collisions. Organizer labels here will be subjective. Expect lower F1 at τ=0.7 and prioritise τ=0.3/0.5 matches.

---

## 5. Recommended pipeline

```
Stage 0 (offline, per camera, from samples + camera.md):
  road mask, lane directions, stop-zone map, turning zones, m/px(y) auto-scale,
  normality atlas; stored as small JSON/NPZ in repo
Stage 1 (shared): detector+tracker @12.5 fps on the whole video (other workstream)
Stage 2 candidates (recall-first, cheap):
  C1 pair contact geometry | C2 kinematic shock | C3 abrupt halt outside stop zones
  C4 track pathology | C5 tiled clip-classifier hits | C6 causal risk peak (Part B code)
  merge within 2 s / same participants; expected 5-40 per 5 min (est.)
Stage 3 local refine: re-detect+track @25 fps, larger input (960-1280), window [t_c-2 s, t_c+6 s]
Stage 4 score: f_kin (GBDT/logistic on ~15 features) + f_clip (X3D/VideoMAEv2-S crop, 3-way)
               [+ f_vlm optional] -> calibrated logistic stack -> p_match
Stage 5 boundaries:
  accident START = first 25-fps frame with gap<=0.2 m or mask contact in [t_c-1, t_c+0.5], else t_shock-0.1 s
  accident END   = min(all involved v<0.5 m/s sustained 1 s, last involved leaves frame); fallback:
                   optical-flow magnitude in the union ROI < thr for 1 s; + learned bias; clip to duration
  near_miss: section 4
Stage 6 post: merge same-class segments with gap<1 s (pile-ups -> one segment, as annotators do);
  drop <0.5 s; pad ~sigma/2; enforce no same-class overlap; sanity cap <=3 accidents/video unless p>0.9
```

Features for f_kin: min gap, closing speed, Δv of both participants, max heading change, lateral jump, post-stop duration, stop-zone prior, queue flag, track-loss flag, mask-contact frames, min TTC in the preceding 3 s, causal risk peak, and participant classes.

Training data for f_kin:
- ACCIDENT synthetic tracklets (2,211 clips). Kinematics transfer from simulation to real far better than pixels do.
- TUM Accid3nD real tracks.
- Our own tracker run on the real proxy positives.
- Negatives: all close-pair interactions in the sample videos.

**Operating point.**
- Use calibrated p_match, which includes the chance that the boundaries match. Emit an event when p_match > F1*/2, where F1* is the best F1 reached on the proxy dev set (Lipton et al. rule).
- Until calibration exists: accident = (f_kin ≥ 0.5 and f_clip ≥ 0.3), or f_kin ≥ 0.85, or (f_clip ≥ 0.8 and contact geometry present).
- Always require two different evidence families.

**Time budget (est.)** for a 5-min 1080p25 video, where the budget is 900 s:
- Base perception: 60–90 s (other workstream).
- Part B stream: 100–200 s.
- Local 25 fps refinement: 20 windows × 8 s × 25 fps = 4,000 frames at ~15 ms, about 60 s.
- Crop classifier: under 5 s. Tiled safety net: ~10 s.
- VLM (optional): 20 × 1.5–4 s, about 30–80 s.
- Total about 0.3–0.5× the budget, which leaves margin for the "engineering judgement" rubric.

Add a deadline guard: skip the VLM and tiling once elapsed time exceeds 1.5× the video duration. Wrap every optional stage in try/except; a crash in `detect_events` empties the whole video.

**Cross-class coupling.** After a crash:
- Vehicles stationary for ≥10 s produce a legitimate **stopped_vehicle** event.
- Debris can produce **road_obstacle**; smoke produces **fire_smoke**.
- A wrong-way vehicle that crashes produces two events.

Share accident candidates with those modules; do not suppress them.

---

## 6. Dev-set plan (accidents may be missing from the samples)
1. **Sample videos (target camera).**
   - Two people independently annotate all 14 classes using the organizers' START/END conventions (CVAT or Label Studio video timeline), then adjudicate. Measure the IoU between annotators for near_miss to learn the realistic ceiling.
   - Convert to `ground_truth.json` with duration and fps, and score with `evaluate.py --gt`.
   - The samples are the **false-positives-per-hour test** for accident and near_miss. Manually review every alarm.
2. **Proxy accident set (other cameras).**
   - 60–100 real fixed-CCTV accident clips: ACCIDENT real first, prioritising elevated urban intersection views, then TAD, CADP and UCF-Crime.
   - Relabel START (first contact) and END (all involved at rest) at frame level; label near-misses where visible.
   - Add ≥2 h of normal CCTV traffic from TAD normal videos and the samples.
   - **Split by camera or source**, not by clip: tune on one half, report on the other.
   - The accident module must run without camera.md (auto-scale plus the generic atlas) so these clips can be evaluated.
3. **Near-miss dev.** Mine candidate conflicts (TTC < 2 s) from the samples and the proxy normal videos, and label those that meet the organizers' definition. This is likely the only near_miss data available.
4. **Robustness.** Degrade the samples (downscale, JPEG q=30, blur, gamma for night, rain overlay) and check that the FP rate does not explode.
5. **Projecting to test.** Expected precision ≈ R·N_acc / (R·N_acc + FP_rate·T), where R is recall on the proxy set and FP_rate comes from the samples. Report both numbers on the website as honest error analysis.
6. **Proxy pitfalls.** Many ACCIDENT clips are poor quality (~66%) and very short. Do not let proxy tuning push START/END biases away from what we see on the target-camera samples.

---

## 7. Relation to Part B
- **Allowed:** Part A may use Part B's causal risk curve. In `detect_events`, run the same causal TTC module (or instantiate `RiskEstimator` internally) and use "risk ≥ 0.5 in the 5 s before t_c" as a feature (C6).
- **Forbidden:** Part B cannot use Part A's accident times, or any Part A output computed with future frames.
  - Grey area: caching per-frame detections from Part A and replaying them in `step()`. Each detection depends only on its own frame, but judges may see the reuse as a violation. Keep B self-contained, or cache only strictly causal outputs, and document it in the README.
- **Offline use is fine:** Part A's START estimator can label proxy clips, and those labels can tune Part B's calibration and thresholds, because the labels never enter runtime.
- **Part B scoring interaction:** frames in [s−5 s, e] of GT near-misses and all frames inside accidents are ignored, and alarms that start in ignored frames are discarded. So TTC risk spikes on real near-misses cost nothing. Risk spikes on unlabelled mild conflicts cost precision, so the calibration from near-miss work transfers directly.

---

## 8. Build order (assuming about 2–3 weeks; deadline unknown)
1. Days 1–3: annotation guideline; annotate the samples; stop-zone and normality atlas; auto-scale.
2. Days 3–6: pair module (gaps, TTC/PET, Δv hinge fits); rule-only accident and near_miss; local 25 fps refinement; first `evaluate.py` numbers on the samples and the proxy set.
3. Days 6–10: download ACCIDENT, TAD and CADP; relabel 60–100 clips; mine hard negatives from the samples; train X3D-S/M on crops (Kaggle T4); stacking and calibration.
4. Days 10–12: optional VLM ablation (Qwen3-VL-2B fp16 with NaN guard), kept only if it clears +0.05 F1 within the time budget.
5. Continuous: runtime logs per stage, determinism check (two runs give identical JSON), and failure-case gallery for the website.



## Top recommendations
- Build accident and near_miss on pairwise trajectory analysis: footprint gap, closing speed, TTC/PET, Delta-v over 1 s windows, hinge-fit onsets. Re-track each candidate window at native 25 fps. This is the only way to get START/END to about 0.3 s, which IoU 0.7 needs on 2-6 s segments.
- Estimate metres-per-pixel from car box widths, or build a lane-width homography, so thresholds are in m and m/s. Also build a stop-zone and normality atlas from the sample videos to reject queue and bus stops.
- Train a small crop-based verifier (X3D-S/M, Apache-2.0, or VideoMAEv2-S, CC-BY-NC) with normal/impact/aftermath outputs. Use ACCIDENT real, TAD and CADP clips as positives and rule candidates mined from the sample videos as hard negatives. Use it to verify candidates and as a tiled safety net, never to set boundaries.
- Keep the VLM out of the core path: the 5 GB weights cap leaves room for at most a ~2B model in fp16, the T4 has no bf16, vLLM cannot serve Qwen3-VL on Turing, and VLM timing errors are about 1-4 s. Test Qwen3-VL-2B as a yes/no verifier only if it adds at least 0.05 F1 within budget; otherwise use VLMs offline for pseudo-labelling.
- Flag a near_miss only when there is a conflict (TTC <= 1.5 s or PET <= 1.0 s), an evasive action (Delta-v <= -3 m/s over 1 s, or a swerve of at least 0.8 m) and no contact. START is the hinge onset; END is when the gap is at least 2 m and growing. Suppress near_miss when an accident is found on the same pair.
- Set thresholds with the p_match > F1*/2 rule on a calibrated score. Until calibration exists, require two independent evidence families. Pad segments by about half the boundary uncertainty (predicting too long is cheaper than too short), and merge pile-ups into one segment.
- Dev set: annotate every sample video (it is also the false-positives-per-hour test). Build a camera-split proxy set of 60-100 relabelled public fixed-CCTV accident clips (ACCIDENT real first) plus at least 2 h of normal traffic, and score with evaluate.py.
- Share the causal TTC module with Part B and use its risk curve as a Part A feature (allowed). Never feed Part A outputs into RiskEstimator, and document any caching in the README.


## Open questions
- How many accidents and near-misses are in the hidden test set, and are the test videos curated to contain events? This sets the achievable precision and the per-video sanity cap.
- Do the sample videos contain any accident or near-miss? If not, accident tuning depends entirely on proxy clips from other cameras.
- Scene specifics from camera.md: resolution, fps, camera height and angle, day/night, whether markings allow a homography. These decide the scale method and footprint accuracy.
- Does the organizers' accident END mean 'at rest' only (short segments) or does it include the aftermath? This must be checked against their example ground_truth.json and starter kit.
- Have ACCIDENT real-clip labels been released since the CVPR 2026 challenge ended? If not, the team must relabel them, and the dataset terms should be checked for use outside the challenge.
- Does run_submission.py call detect_events and RiskEstimator in the same process, and would judges accept caching strictly causal per-frame detections from Part A for Part B?
- Do Qwen3-VL-2B or InternVL3.5-2B in fp16 on a T4 give stable, NaN-free, deterministic logits, and at what real latency per 8-frame query? This must be measured on Kaggle or Colab T4.
- Does near_miss annotation include vehicle-pedestrian conflicts (which overlap with jaywalking or failure_to_yield), and how consistent are the organizers' near_miss boundaries?
- Is TNAD (intersection near-accident dataset) actually downloadable in 2026? It is the only CCTV-like near-miss set with temporal labels.


## Key claims (as submitted for verification)
- ACCIDENT benchmark: 2,027 real fixed-CCTV clips + 2,211 CARLA synthetic clips; ground truth = impact (first-contact) time averaged over 3-5 annotators; clips 1-114 s (median 26.8 s), 4-50 fps; annotations CC BY 4.0, code Apache-2.0; real clips from redistribution-licensed sources; zero-shot temporal scores naive 0.190, optical flow 0.266, Molmo-7B 0.343, human 0.979 — https://arxiv.org/html/2604.09819v1
- ACCIDENT data is on Kaggle (picekl/accident); for the CVPR 2026 challenge only the synthetic development split was labelled and the real CCTV clips were the test set, with manual annotation prohibited in the challenge — https://accidentbench.github.io/ ; https://arxiv.org/html/2608.08867v1
- A Qwen3-VL-32B + YOLO11x/BoT-SORT coarse-to-fine pipeline reached temporal score 0.549 vs 0.343 best baseline, took 40-90 s per clip on an H100, and found that tracking on sub-sampled frames swaps IDs — https://arxiv.org/html/2608.08867v1
- A two-pass VLM pipeline (hosted Qwen3-VL-Plus) on ACCIDENT had a +1.55 s late bias, and temporal MAE grew from 0.94 s (clips <=10 s) to >4 s (clips >=20 s) — https://arxiv.org/html/2605.01512
- The spec caps total weights at 5 GB; Qwen3-VL-2B-Instruct is 4.26 GB, Qwen3-VL-4B-Instruct 8.89 GB, Qwen3.5-2B 4.55 GB, InternVL3.5-2B 4.71 GB, Gemma-4-E2B-it 10.2 GB — spec p.7 ; https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/main ; https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/main ; https://huggingface.co/Qwen/Qwen3.5-2B/tree/main ; https://huggingface.co/OpenGVLab/InternVL3_5-2B/tree/main ; https://huggingface.co/google/gemma-4-E2B-it/tree/main
- vLLM has no supported attention backend for Qwen3-VL on Turing (T4); issue closed as not planned; enforce-eager workaround <10 tok/s — https://github.com/vllm-project/vllm/issues/29743
- Qwen3-VL 2B/4B/8B and Qwen3.5 small models are Apache-2.0; Qwen2.5-VL-3B-Instruct uses the non-commercial Qwen Research License; Qwen3.5 has thinking mode on by default — https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct ; https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE ; https://huggingface.co/Qwen/Qwen3.5-4B
- FP16 inference of Qwen2-VL has produced errors/gibberish (T4 lacks bf16) — https://github.com/huggingface/transformers/issues/33294
- TAD: 500 surveillance videos, 25 h, 7 anomaly types incl. vehicle accidents; 400 train with video-level labels, 100 test with frame-level labels; frames on Google Drive — https://arxiv.org/html/2008.08944 ; https://github.com/ktr-hubrt/WSAL
- CADP: 1,416 CCTV video segments from YouTube, 205 with full spatio-temporal annotations; non-commercial research use — https://ankitshah009.github.io/accident_forecasting_traffic_camera
- DoTA is dashcam (4,677 clips, 1280x720), MIT licence, anomaly_start/end annotations, downloaded via a YouTube script — https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly
- Nexar collision dataset: 1,500 dashcam videos (750 positives, collisions and near-misses treated equally), 1280x720@30fps, time_of_event/time_of_alert, Nexar Open Data License — https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction
- VideoMAE and VideoMAEv2 HF weights are CC-BY-NC-4.0; VideoMAEv2 distilled ViT-S/ViT-B reach 83.7%/86.6% Kinetics-400 top-1 — https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics ; https://huggingface.co/OpenGVLab/VideoMAEv2-Base ; https://github.com/OpenGVLab/VideoMAEv2
- X3D-S: 2.96 GFLOPs, 73.33% K400; X3D-M: 6.72 GFLOPs, 75.94%; 3.79M params; PyTorchVideo is Apache-2.0 — https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html ; https://github.com/facebookresearch/pytorchvideo
- For calibrated probabilities, the F1-optimal decision threshold is half the optimal F1 — https://arxiv.org/abs/1402.1892
- Surrogate-safety thresholds used in CCTV conflict analysis: TTC 1.5 s, DST 3 m/s2, PET 1 s; 100-Car rapid manoeuvre = braking >0.5 g or lateral >0.4 g — https://cait.rutgers.edu/wp-content/uploads/2021/04/cait-utc-reg-53-final.pdf ; https://www.nhtsa.gov/sites/nhtsa.gov/files/analyses20of20rear-end20crashes20and20near-crashes20dot20hs2081020846.pdf
- With a 1 s Savitzky-Golay window and 0.2 m position noise at 25 fps, acceleration noise is about 1.1 m/s2, vs about 8.5 m/s2 with a 0.4 s window; at IoU 0.7 a 3 s segment tolerates only about 0.53 s symmetric shift — reasoning (scratchpad simulations accel_noise.py, iou_sim.py)
- AI City 2021 Track 4: 100 train / 150 test freeway videos (~15 min, 30 fps, 800x410; crashes and stalled vehicles); still downloadable after email approval — https://arxiv.org/html/2105.03827 ; https://www.aicitychallenge.org/ai-city-challenge-dataset-access/
