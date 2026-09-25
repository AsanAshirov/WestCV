<!-- source: deep-research workflow wf_294a4b01-fc2, agent research:rare -->

# Hard classes: fire_smoke, road_obstacle, jaywalking, congestion, stopped_vehicle (and pedestrians)

Scope: these five classes cannot be done with a simple "one trajectory crosses a line" rule. Each needs extra machinery: a background model, an extra detector, a scene-level aggregate, or a verifier. Accident, near_miss and the line/lane violation classes belong to other researchers and come up here only where they interact with these five.

Two kinds of number appear below. Measured numbers come from my scratchpad benchmarks on the team's dev CPU (Python 3.14, OpenCV 4.13, numpy 2.4). Published numbers are cited. Anything else is marked as an estimate.

---

## 0. Summary

| Class | Recommended core method | Extra models | Default enabled? | Expected F1 when the class is present (estimate) | Main risk |
|---|---|---|---|---|---|
| stopped_vehicle | Vehicle tracks + perspective-normalised speed, a stationary run of 10 s or more, back-dated start, and a static-scene map to survive occlusion. Exclude queues, parking bays and congestion. | none | **Yes** | 0.5–0.75 | Definitions (bus stop, parking, queue) and ID switches under occlusion |
| jaywalking | Person tracks from a higher-resolution road crop; foot point inside (carriageway minus crossings), with a distance margin and hysteresis; riders removed | high-res or tiled person detection | **Yes**, if the samples show pedestrians near the road | 0.4–0.7 by day, less at night | Short events are very sensitive to IoU 0.7; recall on small or night pedestrians |
| congestion | Per direction: occupancy plus median speed on every lane, hysteresis, a long minimum duration, and a signal-aware filter | optional DIS optical flow | **Yes, conservatively**, once the red-queue ambiguity is settled | 0.3–0.6 | Whether a normal red-light queue counts as congestion |
| road_obstacle | (a) COCO animals on the carriageway; (b) static-blob detector using a dual background, confirmed by an open-vocabulary detector or a VLM | YOLOE-26s (AGPL) or OWLv2 / Florence-2; optional small VLM | **Only with high-precision gating** | 0.2–0.5 | Shadows, rain and lens drops produce false static blobs |
| fire_smoke | YOLO fire/smoke detector fine-tuned on D-Fire (CC0) plus scene negatives, then persistence, location and dynamics filters, then optional VLM verification | fire/smoke YOLO; optional VLM | **Only with high-precision gating** (0 false positives on every sample video) | 0.5–0.8 | Night lights, exhaust vapour, fog, glare |

Structural decisions that apply to all five classes:
1. **The class output is the union of per-object intervals.** The FAQ says two simultaneous events of one class become one segment, so each class is a single binary timeline.
2. **Part A is offline.** Use centred (non-causal) smoothing, and back-date every start and end to the first sample of the run that triggered it. Never report the time at which the event was *confirmed*.
3. **Rare classes are enabled only if they clear the dilution rule in §1.2.**
4. **One shared "static-scene engine" (§1.5)** feeds stopped_vehicle, road_obstacle and congestion. It costs about 10–15 ms of CPU per processed frame at 480×270 (measured).

---

## 1. Foundations shared by all five classes

### 1.1 From per-object signals to segments

```
def segments(t, x, on, off, min_len, max_gap, duration):
    # x: per-sample score/indicator, already centred-smoothed (offline is allowed in Part A)
    # hysteresis: open when x>=on, keep open while x>=off
    # start/end are the first/last samples of the run, NOT the confirmation time
    # merge runs separated by < max_gap; drop runs shorter than min_len
    # clamp to [0, duration]; if the run touches the last 1-2 s -> end = duration
```

- Get `duration` by counting decoded frames, n_frames/fps. `CAP_PROP_FRAME_COUNT` can be wrong. Clamp `end <= duration` so validation cannot reject an event.
- Per-class defaults: gap merge / minimum length. stopped_vehicle 2 s / 10 s. jaywalking 1 s / 1.2 s. congestion 15 s / 20 s. road_obstacle 5 s / 3 s. fire_smoke 10 s / 5 s.

### 1.2 When a class earns its place in macro-F1

The spec says a class you predict that never occurs in the test set is added to C and scores 0. Let:
- p = P(class present in the hidden test)
- f = expected F1 on the class when present
- q = P(at least one false-positive segment anywhere in the test set when the class is absent)
- F̄ = mean F1 of the other classes

Enabling gains f/(K+1) when present and loses F̄/(K+1)·q when absent. K cancels, so:

**Enable iff p·f > (1−p)·q·F̄.**

The table gives the minimum f that justifies enabling, with F̄ = 0.45 (computed in the scratchpad):

| p \ q | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|
| 0.1 | 0.20 | 0.41 | 0.81 | 1.62 (never) |
| 0.2 | 0.09 | 0.18 | 0.36 | 0.72 |
| 0.3 | 0.05 | 0.11 | 0.21 | 0.42 |
| 0.5 | 0.02 | 0.05 | 0.09 | 0.18 |

Implications:
- q is a whole-test-set probability. With a per-hour false-positive rate λ and T hours of test video, q = 1 − e^(−λT). For T = 2 h: λ = 0.05/h gives q ≈ 0.10, and λ = 0.3/h gives q ≈ 0.45.
- **Rare classes therefore need fewer than about 0.05 FP segments per hour.** In practice that means zero FPs over every sample video, plus an external negative bank.
- The rule cuts both ways. A rare class that is present with one or two GT events and gets found with no FPs scores F1 = 1.0, and it weighs as much as `accident` in the macro average.
- Example: 6 present classes at mean F1 0.45 give Score_A = 0.45. One spurious extra class drops Score_A to 0.386, which is −0.045 on M.

### 1.3 How much boundary error temporal IoU tolerates

Maximum error that still passes τ = 0.7, where L is the GT length:

| L | shift (both ends late) | both ends inward | both ends outward |
|---|---|---|---|
| 5 s | 0.9 s | 0.8 s | 1.1 s |
| 8 s | 1.4 s | 1.2 s | 1.7 s |
| 20 s | 3.5 s | 3.0 s | 4.3 s |
| 120 s | 21 s | 18 s | 26 s |

Consequences:
- A 2 s confirmation lag on a 5 s jaywalk gives IoU 0.60, which fails τ = 0.7. The same lag on a 30 s congestion gives 0.93.
- **Back-dating matters most for jaywalking.** It matters least for congestion and fire_smoke, which are long events that often run to the end of the video.

### 1.4 Scene configuration (hard-coding from camera.md is allowed)

One `scene.yaml` in normalised coordinates holds these polygons: `carriageway`, `crossings[]`, `sidewalks`, `refuge_islands`, `parking_bays` / `bus_stop`, `lanes[]` (each with a direction vector), `stop_lines[]`, `queue_zones[]` (upstream of each stop line), `sky_mask`, and `signal_roi` if the signal is visible.

Perspective scale: for a flat road, pixel scale is linear in the image row, s(y) = k·(y − y_h) px/m. Fit it from two lane-width measurements, assuming a lane is about 3.5 m wide.

Worked example: lanes 120 px wide at y = 900 and 40 px at y = 400 give k = 0.16 and y_h = 150, so 34 px/m near the camera and 11 px/m far away.

Every threshold below is in metres or m/s through s(y), so one set of parameters works both near and far.

### 1.5 The static-scene engine (dual background)

The idea is Porikli's dual foreground: a short-term model absorbs stopped objects quickly, and a long-term model does not. Pixels that are foreground in the long-term model but not in the short-term model are "static new objects" (Porikli et al. 2008). The AI City 2021 anomaly-track winner used the same principle for stalled vehicles on highway CCTV. They ran MOG background modelling, detected vehicles on the background frames, and backtracked in time to find the start (arXiv 2105.03827).

**Measured MOG2 behaviour.** A synthetic stopped block, processed at f fps, with OpenCV defaults backgroundRatio 0.9, varThreshold 16 and shadows off, is absorbed into the background after

**T_abs ≈ ln(0.9) / ln(1 − α) / f ≈ 0.105 / (α·f) seconds.**

- The measured value was within 5% of this formula for α from 0.02 to 0.0002. For example, α = 0.002 at 5 fps is predicted at 10.5 s and measured at 11.0 s.
- KNN absorbed about 2.2× more slowly and did not follow the formula, so **use MOG2 for predictable timing.**

Recipe, working at 480×270 and 5 fps:

```
short = cv2.createBackgroundSubtractorMOG2(500, 16, False); alpha_s = 0.105/(3.0*5) ≈ 0.007   # absorbs a stopped object in ~3 s
B_long: own float image, init = per-pixel median of 60 frames sampled over the first 2 min (or the whole video — offline is allowed)
        update B_long += a_l*(F - B_long) ONLY where not shortFG and not confirmed-static; a_l = 1/(60 s * 5 fps)
        -> follows lighting drift, never absorbs a stopped object (the region is frozen)
longFG  = max_c |F - B_long| >= 28  OR  |grad(F) - grad(B_long)| large   (gradients resist cloud shadows)
static  = longFG & ~shortFG ; open 3x3, close 7x7 ; & carriageway mask ; connected components
global change guard: if >40 % of carriageway flips within 1 s (auto-exposure, IR switch, cloud) -> re-init B_long, mute static events for 10 s
```

Measured cost on the dev CPU, per call:

| Resolution | MOG2 | KNN | DIS optical flow, ultrafast preset | Farneback |
|---|---|---|---|---|
| 480×270 | 5.0 ms | 4.4 ms | 3.1 ms | 55 ms |
| 640×360 | 5.5 ms | 5.8 ms | 9.4 ms | 78 ms |
| 1920×1080 | 53 ms | 47 ms | 226 ms | 1,227 ms |

- **Always downscale before background modelling.** Use DIS rather than Farneback wherever flow is needed.
- A per-pixel median of 31 frames at 480×270 took 247 ms. That is fine once per window, but too slow to run per frame.

---

## 2. stopped_vehicle

**Definition used:** stationary on the carriageway for 10 s or more, not in a queue at a signal. Start is when the vehicle stops. End is when it moves or is removed.

**Method:**

```
for each vehicle track (detector+tracker at ≥5 fps; foot point = bottom-centre of box):
    v(t)   = |p(t+1s) − p(t−1s)| / 2 s / s(y)         # centred, m/s
    still  = v < 0.3 m/s  AND  displacement over 5 s < 0.7 m
    runs   = maximal still-runs; bridge gaps ≤ 3 s if a new/lost track re-appears with box IoU ≥ 0.5 (ID-switch stitching)
    keep run if len ≥ 10 s, foot point in carriageway (not parking_bays), and queued() is false for ≥ 50 % of the run
    start  = first sample of run (the moment it stopped; = 0.0 if stopped at video start)
    end    = first sample of sustained motion (v > 0.5 m/s for ≥ 1 s) or of disappearance; touches end → duration
queued(p,t) = (p in queue_zone AND (signal red OR signal unknown)) AND (vehicle ahead in lane within 1.5 car-lengths is also still,
              OR p is the head of queue at the stop line)
              OR congestion_active(direction(p), t)
```

**Occlusion.** Passing traffic, buses and trucks hide stopped cars. That breaks the track; ByteTrack in Ultralytics keeps a lost track for only `track_buffer: 30` frames by default. Handle it this way:
- When a still-track is lost, create a **ghost box**.
- Keep the ghost alive while at least 40% of its box is `static`, or at least 60% is `longFG` (an occluder turns shortFG on but longFG stays on), or the detector re-finds a vehicle with IoU ≥ 0.5.
- End the ghost when longFG coverage stays below 0.2 for 1 s or more. That also detects "removed" (towed) vehicles.

**Details that decide IoU:**
- Report from the stop moment, not stop + 10 s. For a 25 s stop, reporting [10, 25] against GT [0, 25] gives IoU 0.6, which fails τ = 0.7.
- A vehicle with fewer than 10 s of visible stillness at either edge of the video is not reported.

**Interactions with other classes:**
- After an `accident`, the crashed cars usually stay put, so a stopped_vehicle segment normally follows it. Different classes may overlap, so emit both.
- Suppress vehicles in a `stop_line` situation (past the stop line on red) and vehicles inside an active `congestion` segment.

**Ambiguities to settle with dev labels or an organiser question:**
- Buses at a stop and taxis dropping off: the literal definition says yes.
- Cars in a curb-side parking lane: are they on the "carriageway"?
- Recommended default: exclude marked parking bays; include bus stops and drop-offs of 10 s or more.

**Expected performance (estimate):** precision 0.6–0.8 and F1 0.5–0.75. Boundaries are easy because stops are crisp. Enable by default, since p is high on an urban camera.

---

## 3. jaywalking and the shared pedestrian module

This module also feeds failure_to_yield (another researcher's class), Part B ("pedestrians entering the roadway") and pedestrian accidents.

### 3.1 Detecting small people

**Why default settings fail.** In a 1080p CCTV frame, a pedestrian on the far side is often only 25–60 px tall. At imgsz 640, full-frame downscaling divides that by 3, which leaves 8–20 px. COCO-trained detectors lose most of their recall at that size.

**Options:**
- **Default: crop the road region, then infer at imgsz 1280.** For example, the bottom 60% of the frame (about 1920×650) scaled to 1280 wide is a 0.67× scale instead of 0.33×.
  - Cost estimate: roughly 4× the 640 cost. Ultralytics lists YOLO26s at 2.5 ms with TensorRT on a T4, so about 10 ms per frame. At 8 fps that is about 80 ms of GPU time per second of video, well inside the 3× budget.
- **SAHI tiling (MIT licence)** of the far strip only. SAHI reports +5 to +7 AP on small-object benchmarks without fine-tuning, and 12.7–14.5 AP with slicing-aided fine-tuning. It costs one extra forward pass per tile.
- **The `yolo26-p2.yaml` small-object head** needs training from scratch, because Ultralytics released no P2 weights. Skip it unless there is time.
- Validate by hand-labelling about 200 person boxes across the samples and measuring recall by pixel-height bin (under 20, 20–40, over 40 px).

### 3.2 Rules

```
riders out:  person box P with bicycle/motorcycle box B where IoU(P,B) ≥ 0.15 and P.bottom ∈ [B.top+0.25·B.h, B.bottom+0.15·B.h]
             in ≥ 50 % of track frames  → rider (drop); also median speed > 3.5 m/s for ≥ 2 s → rider/cyclist
foot point:  bottom-centre; if the box is truncated by an occluding vehicle (h/w < 0.6 × track median), use the Kalman-predicted foot
ROAD      := carriageway − dilate(crossings, 1.0 m) − refuge_islands − sidewalks
d(t)      := signed distance (m) of foot inside ROAD
enter     := d ≥ max(0.4 m, 0.1·person_height_m)  for ≥ 0.6 s      (curb margin kills sidewalk jitter)
exit      := d < 0 for ≥ 0.6 s
interval  := [first sample of enter-run, first sample of exit-run]  (back-dated)
keep      := len ≥ 1.2 s, track mean conf ≥ 0.35 (night: ≥ 0.25 but track length ≥ 2 s)
class     := union over persons, merge gaps ≤ 1.0 s
```

**Edge cases to settle on the dev labels:**
- A person getting out of a stopped car: the literal definition says yes. They usually reach the kerb within 2 s, and a `keep` threshold of about 2 s removes most of them if the labels exclude them.
- Traffic police or road workers standing in the intersection for minutes. Recommend a zone exclusion for the intersection box, or the hi-vis colour of the person crop, if the labels exclude them.
- Pedestrians weaving between queued cars: track gaps up to 1 s are interpolated.

**Timing matters here:** jaywalks last about 5–11 s (7–14 m of road at about 1.3 m/s, an estimate). Per §1.3, boundaries must be within about ±1 s.

**Expected performance (estimate):** F1 0.5–0.7 by day and 0.3–0.5 at night. Enable by default if pedestrians are visible.

---

## 4. congestion

**Definition:** traffic at a standstill or crawling across **all lanes of a direction**. Start is when the queue stops moving; end is when it clears.

**Per-direction features, every 0.5 s:**
- `occ_d`: area of the union of vehicle footprints (bottom 40% of each box) inside the direction's ROI, divided by the ROI area. Fall back to longFG coverage when the detector undercounts in dense occlusion.
- `v_d`: median track speed in m/s. With fewer than 3 tracks, use the median DIS flow magnitude over shortFG pixels, converted through s(y). DIS costs about 3–10 ms at 480×270 on the dev CPU.
- A lane is jammed when its occupancy is 0.30 or more and its speed is 2.0 m/s (about 7 km/h) or less.
- `jam_d` is true when at least 80% of the direction's lanes are jammed, which means all lanes for roads with 4 lanes or fewer.
- Apply a 5 s centred majority vote, then hysteresis. The clear condition is occupancy below 0.15, or speed above 4 m/s, for 10 s or more.

**The key ambiguity: is every red-light queue "congestion"?** stopped_vehicle explicitly excludes signal queues; congestion does not. If annotators labelled every red phase, congestion would appear every cycle. Two hypotheses:
- **H1 (strict, recommended default for a signalised approach).** Keep a jam only if at least one of these holds:
  - it lasts through a green phase, with the queue failing to discharge for 10 s or more while the signal is green (signal state comes from the red_light module);
  - the queue spills back to the upstream edge of the ROI or into the intersection;
  - with no signal visible, it lasts at least 1.5× the longest red phase measured in EDA from queue build and discharge cycles (typical cycles are 60–120 s, an estimate). With no signal at all, use 45 s.
- **H2 (loose).** Any all-lane standstill of 20 s or more.
- Settle it by asking in the hackathon channel and by labelling the samples both ways. Report H1 as default and switch if told otherwise.

**Boundaries:**
- Start is the first sample of the jammed run, back-dated.
- End is the first sample of the clearing run.
- The union over directions is one class timeline.
- Events are long, so IoU is forgiving (§1.3).

**Suppression:** while congestion is active in a direction, turn off stopped_vehicle in that direction's lanes.

**Expected performance (estimate):** F1 0.3–0.6, driven mostly by the definition. Enable with H1 thresholds; p is high for a busy urban road at rush hour.

---

## 5. road_obstacle (debris, animal, fallen object)

Three paths, from cheapest and most precise to most expensive:

**A. Animals (cheap, precise).**
- COCO classes 15–19 (cat, dog, horse, sheep, cow) from the main detector: confidence 0.4 or more, track 1 s or longer, foot point on the carriageway for 1 s or more.
- The interval runs from entering the carriageway to leaving it.
- Birds (class 14) are excluded because they fly off.
- Stray dogs crossing are the most plausible real obstacle event on an urban camera.

**B. Static blobs from the §1.5 engine.**
- Component area 0.05–6 m², using s(y)².
- At least 70% of pixels inside the carriageway.
- Persistent for 3 s or more.
- Not explained by vehicle or person boxes: IoU below 0.1 for at least 80% of its life.
- A texture check separates objects from shadows: the gradient-energy ratio against B_long must be above 1.2 (added edges) or below 0.8.
- Timing:
  - Start: back-date to the first sample where at least 50% of the blob differs from B_long. This catches the moment an object falls from a truck.
  - End: the first 2 s window in which the region matches B_long again ("removed"), or the video duration.

**C. Verification (required for B).**
- Crop 2–3× the blob, resize to 320–640 px, and run an open-vocabulary detector only on these crops. That is a few calls per video.
- Candidate models:

| Model | Licence | Notes |
|---|---|---|
| **YOLOE** (THU-MIG, ICCV 2025) | AGPL-3.0 | YOLOE-11-S runs about 301 FPS on T4 with TensorRT at LVIS AP 27.5. YOLOE-26 variants are in Ultralytics. Text prompts are fixed at export, and the first `set_classes()` downloads the text encoder, so **bake the vocabulary into the exported weights before the offline run.** |
| **OWLv2** base | Apache-2.0 | Third-party figure: about 50–80 ms on T4. |
| **OmDet-Turbo** | Apache-2.0 | 100 FPS reported with TensorRT and a language cache. |
| **Florence-2** base / large | MIT | 0.46 / 1.55 GB. |
| **Grounding DINO tiny** | Apache-2.0 | Slow: roughly 100–200 ms per image or more. |
| **Grounding DINO 1.5/1.6 Pro, DINO-X** | API-only | Forbidden. |
| **YOLO-World** | GPL-3.0 | Fine too. |
| **SAM 3** (Meta, Nov 2025, SAM License) | SAM License | Text-promptable, but heavy (30 ms per image on an H200). Use it at most offline for auto-labelling, not in inference. |

- Prompts: "tire", "cardboard box", "plastic bag", "wooden plank", "rock", "tree branch", "mattress", "ladder", "sack", "bucket", "dog", "cat". Accept at confidence 0.2 or more on the crop, or a VLM yes/no (§6.4).
- Traffic cones and barriers are placed deliberately. Exclude them unless the dev labels include them.

**Learned road-anomaly datasets are not worth it here.** SMIYC RoadAnomaly21 has 100 test images; RoadObstacle21 has 327 test plus 30 validation images, under CC BY 4.0. Lost and Found is non-commercial, and Fishyscapes is similar. All are ego-vehicle views at road level, a large domain gap from an elevated fixed CCTV view, and they need Cityscapes-style segmenters. The fixed camera makes background modelling far stronger than learned anomaly segmentation.

**Known misses:**
- An object present for the whole video is inside B_long and cannot be detected by path B. Accept this.
- Rain drops on the lens make static blurry blobs anywhere. The gradient ratio below 0.8, the location on the carriageway and the verifier reject most of them.

**Expected performance (estimate):** path A plus B plus C gives precision 0.5–0.8 and F1 0.2–0.5. **Enable path A by default. Enable B+C only if it gives zero false positives on all sample videos plus 1–2 h of external traffic video.**

---

## 6. fire_smoke

### 6.1 Detector

**Data: D-Fire.** 21,527 images: 1,164 fire-only, 5,867 smoke-only, 4,658 fire and smoke, 9,838 negatives. Licensed **CC0 1.0** per the repository LICENSE file. The repo also links surveillance videos, useful for testing detection and false positives.

**Existing weights:**

| Weights | Licence | Notes |
|---|---|---|
| `pedbrgs/Fire-Detection` | MIT | YOLOv5 s/l trained on D-Fire, with temporal filters AVT (area variation, recommended outdoors) and TPT. |
| Pyronear YOLO11s models | Apache-2.0 | Wildfire smoke plumes on horizon cameras, 1024 px input. They use a "sequential detection" requirement of 5 consecutive frames. Wrong domain for vehicle fires, but a good source for the persistence idea. |
| Community HF checkpoints (e.g. leeyunjai/yolo11-firedetect) | often none stated | Avoid: no licence or training data stated. |

**FASDD:** over 120k images, FASDD_CV about 95k. The ESSD preprint was withdrawn and I could not confirm the dataset licence. Verify on Science Data Bank before using it.

**Recommended training:**
- Fine-tune YOLO26s or YOLO11s at 640 on D-Fire. Add **scene negatives**: 1–3k frames from the sample videos in all lighting conditions, as background images with empty labels. Ultralytics' general guidance is 0–10% background images; for this camera, 10–15% is justified because false positives are the dominant risk.
- Ultralytics-trained weights carry AGPL-3.0, which is fine for a public repo. List D-Fire (CC0) in the README.
- Training time estimate: 21.5k images for 50 epochs on a Kaggle T4 is about 4–6 h. Run it at 2 fps on full frames, which costs about 5 ms of GPU per video-second.

### 6.2 Confirmation filters

These run only on candidates, so they are cheap:
1. **Persistence.** Cluster boxes over time (IoU above 0.2). Confirm when the cluster is present in at least 7 of 10 samples, which is 5 s. Thresholds: smoke confidence 0.35, fire confidence 0.45.
2. **Location.** The box must overlap the carriageway dilated by 3 m, or a vehicle box. Reject if it covers more than 50% sky mask, or more than 25% of the frame (fog or haze).
3. **Not a rigid part of a moving vehicle.** Reject if the box moves with one vehicle track for 3 s or more while that vehicle moves faster than 2 m/s. This removes headlights, tail lights, and cold-weather exhaust vapour, which is short-lived and trails a moving car. Real vehicle fires are almost always on stopped vehicles, so a stopped_vehicle track nearby is a strong positive cue.
4. **Dynamics.**
   - Fire: flames flicker at about 10 Hz (Töreyin et al.). Probe 2 s at the full 25 fps inside the box. Require high-frequency energy of V-channel intensity in the 3–12 Hz band, plus area variation (coefficient of variation ≥ 0.1, the AVT idea).
   - Smoke: non-rigid DIS flow, where flow direction is inconsistent inside the box and mostly upward, and background edges get blurred (gradient-energy ratio against B_long below 0.8).
5. **Colour sanity for fire.** R > G > B. Reject blue-dominant flashing emergency lights and street-lamp blobs that do not change for minutes.

**Boundaries:**
- Start: back-date along the low-threshold (0.15) detector trace to the first sample of continuous presence.
- End: the last presence sample. If that is within 5 s of the video end, end = duration, which the spec explicitly allows.
- Minimum length 5 s; merge gaps up to 10 s.

### 6.3 Expected performance and default

If present, fire events are long, salient and often run to the end of the video, so f is about 0.6–0.8 (estimate). p is low, perhaps 0.1–0.2 (estimate), because this is a real CCTV camera. Per §1.2, with p = 0.15 the class is worth enabling only if q ≤ about 0.1, which means a per-hour false-positive rate of 0.05 or less. **Ship it enabled only after this audit passes:** 0 false positives on every sample video, plus D-Fire's non-fire surveillance clips and night traffic clips.

### 6.4 Optional VLM verifier for rare-class candidates

Use it for fire_smoke, road_obstacle and possibly accident. Ask 3 keyframes of each candidate a yes/no question, for example "Is there smoke or fire coming from a vehicle or on the road?", and accept at 2 of 3.

| Model | Licence | Weights size (HF) |
|---|---|---|
| Florence-2-large | MIT | 1.55 GB safetensors |
| InternVL3.5-1B | Apache-2.0 | 2.12 GB |
| Qwen3-VL-2B-Instruct | Apache-2.0 | 4.26 GB |

- Qwen3-VL-2B nearly fills the 5 GB weight budget on its own, so quantise it or pick a smaller model.
- The T4 is a Turing GPU without native bf16, so run fp16 and check for overflow.
- Budget: at most about 20 queries per video, about 1–2 s each (estimate). This is Part A only, and it is allowed because open weights are used offline.

---

## 7. Night, rain, low light (all classes)

- **Global changes: IR/monochrome switch, auto-exposure, clouds.** The §1.5 guard re-initialises B_long and mutes static events for 10 s. In monochrome IR mode, disable the colour checks for fire; keep the detector plus dynamics.
- **Headlight glare and wet-road reflections:**
  - They cause false fire boxes (filter 3 and the colour check) and false static blobs (reflections move with cars, so they fail the persistence test).
  - Street-light reflections on wet asphalt are static, but B_long is built from the same video, so they belong to the background. **Never use a fixed reference image from another day.**
- **Rain:**
  - Downscale before background subtraction, which suppresses streaks.
  - Raise varThreshold from 16 to about 25–36 when the global shortFG rate is above 3% of the carriageway on average.
  - Morphological opening.
  - Reject lens-drop blobs with low gradient energy.
- **Pedestrians at night:**
  - Recall drops sharply (estimate: by 30–50%). Compensate with a lower confidence (0.25), a longer track requirement (2 s), and optionally gamma or CLAHE on the luminance channel. Keep it only if dev recall improves.
  - Train or fine-tune with night augmentation if time allows.
- **Congestion at night:** use track-based speed; background-subtraction occupancy is unreliable under headlights.
- **Winter:** exhaust vapour is a common smoke false positive (filter 3); snow changes the road appearance slowly (the selective B_long update follows it). Both matter if the camera is in a place with cold winters, such as Tashkent for WIUT (an assumption).

---

## 8. Compute budget for these modules

Per second of video. T4 figures are estimates built on published per-inference latencies; CPU figures are measured on the dev CPU.

| Module | Rate | Cost |
|---|---|---|
| Person / vehicle detection at 1280 on a road crop | 8 fps | about 80 ms GPU |
| Fire/smoke YOLO at 640 | 2 fps | about 5 ms GPU |
| Dual background (MOG2 plus B_long) at 480×270 | 5 fps | about 30–50 ms CPU |
| DIS flow at 480×270 (congestion fallback, smoke check) | 2–5 fps | about 10–40 ms CPU |
| Open-vocabulary verification on crops | candidates only | under 1 s per video |
| VLM verification | at most about 20 per video | about 20–40 s per video |

All of this fits well inside the 3× wall-clock limit alongside decoding and the main tracker. Measure early on a T4 in Kaggle or Colab. Engineering judgement is 15% of the code score.

---

## 9. Test plan

1. **Dev labels.** Label every sample video for these five classes, strictly following the start/end conventions. Mark ambiguous cases (bus stop, parking, red queue, worker, person exiting a car) with a flag so both conventions can be scored.
2. **Per-class scoring** with `evaluate.py --gt dev.json`. Report F1 at τ = 0.3, 0.5 and 0.7 separately; a gap at 0.7 means a boundary problem.
3. **False-positive audit for rare classes.** Run fire_smoke and road_obstacle over all samples plus 1–2 h of external negative traffic video. Include night and rain, and check each dataset's licence. The requirement is 0 segments.
4. **Injection tests for classes missing from the samples.** These are dev-only and never shipped:
   - Paste dog, tyre and box cut-outs onto the road for 10–60 s. Measure detection, start error and end error.
   - Alpha-blend smoke clips (D-Fire surveillance videos) onto a stopped car.
   - Simulate occlusion by pasting a passing bus over a stopped car.
5. **Ablations to publish on the website** (extra credit):
   - Person recall at 640 vs 1280 vs SAHI.
   - Stopped_vehicle F1 with and without ghost tracking.
   - Congestion H1 vs H2.
   - Fire false-positive rate with and without each filter.
6. **Determinism.** Fix seeds. MOG2 and the median are deterministic on identical frame sequences, so decode the same frames in both runs by using a fixed sampling stride.

## 10. Cross-cutting notes for other researchers

- **Part B** may not reuse Part A outputs. Recompute causal versions of these signals inside `RiskEstimator`, with no back-dating and no centred smoothing:
  - a pedestrian entering the ROAD polygon (d crosses 0 while moving toward the lanes);
  - a stopped vehicle or obstacle in a live lane with approaching traffic (time-to-collision against the static box).
- **The pedestrian module is shared** with failure_to_yield and accident (vehicle–pedestrian contact).
- **Static-engine heatmaps make good EDA figures** for the website: motion heatmap, stop heatmap and pedestrian foot-point heatmap.
- **stopped_vehicle and congestion depend on signal state** from the red_light and stop_line modules.



## Top recommendations
- Build one shared static-scene engine at 480x270 and 5 fps. Short-term MOG2 with alpha = 0.105/(T_abs*f), where T_abs is about 3 s. A long-term background that is a selectively updated running average, initialised from a per-video median and never updated on static or moving pixels. It feeds stopped_vehicle (keeps a stopped car alive under occlusion), road_obstacle (static blobs) and congestion (occupancy fallback).
- Treat every class as the union of per-object intervals. Back-date starts and ends to the first sample of the triggering run, using offline centred smoothing. Report stopped_vehicle from the stop moment, not stop + 10 s, and jaywalking with boundary error of about 1 s or less, because IoU 0.7 tolerates only about 0.9 s shift on a 5 s event.
- Use the dilution rule to decide which rare classes to ship: enable iff p*f > (1-p)*q*Fbar. Ship stopped_vehicle, jaywalking and congestion by default. Ship fire_smoke, and road_obstacle's static-blob path, only after an audit shows 0 false-positive segments on every sample video plus 1-2 h of external night and rain traffic video. The per-hour false-positive rate must be about 0.05 or less.
- fire_smoke: fine-tune YOLO26s or YOLO11s on D-Fire (CC0) with 10-15% scene-negative frames from the samples. Confirm with 7-of-10 persistence over 5 s, a location check (carriageway or vehicle, not sky), rejection of boxes rigidly attached to moving vehicles (lights, exhaust), flicker and area-variation or non-rigid-flow checks, and optionally a small open-weights VLM on 3 keyframes. If smoke persists to within 5 s of the end, set end = duration.
- Pedestrians: detect on a road-region crop at imgsz 1280, with SAHI on the far strip only if dev recall under 20 px is poor. Remove riders by person-over-bike geometry and speed above 3.5 m/s. Count a person on the road only when the foot point is at least max(0.4 m, 0.1 of body height) inside (carriageway minus crossings dilated 1 m), with 0.6 s hysteresis, working in metres through a fitted perspective scale s(y) = k*(y - y_h).
- Congestion: per direction, require all lanes to have occupancy of 0.30 or more and speed of 2 m/s or less. Default to the strict hypothesis H1: a signal queue counts only if it fails to discharge during green, spills back, or lasts more than 1.5x the red phase. Ask the organisers in the channel whether ordinary red-light queues are labelled, and keep H2 (any all-lane standstill of 20 s or more) as a switch.
- road_obstacle: ship the cheap, precise path by default (COCO dog, cat, horse, sheep, cow on the carriageway for 1 s or more). Gate the static-blob path behind an open-vocabulary crop verifier (YOLOE with the vocabulary baked in at export, or OWLv2 / Florence-2) or a VLM yes/no. Do not use the road-anomaly segmentation datasets (SMIYC, Lost and Found): they are ego-vehicle views with a large domain gap, and Lost and Found is non-commercial.
- Before any offline run, bake every text prompt and weight into the package. YOLOE's set_classes() downloads a text encoder on first use. Keep VLM choices under the 5 GB weight budget: Florence-2-large 1.55 GB or InternVL3.5-1B 2.12 GB rather than Qwen3-VL-2B at 4.26 GB. Run in fp16, since the T4 has no native bf16.
- Test plan: hand-label the samples with flags for ambiguous conventions (bus stop, parking bay, red queue, worker, person exiting a car). Report F1 at each tau separately to separate boundary errors from detection errors. Run injection tests (paste animals, debris and smoke into sample videos) for classes absent from the samples. Publish the ablations on the website.


## Open questions
- Does the ground truth label ordinary red-light queues as congestion, or only jams that fail to discharge or spill back? This decides H1 vs H2; ask in the hackathon channel.
- Are buses at a bus stop, taxis dropping off, and cars in a curb-side parking lane labelled as stopped_vehicle? Is a parking lane part of the carriageway?
- Do annotators label people getting out of stopped cars, road workers or traffic police in the roadway as jaywalking?
- Do deliberately placed traffic cones and barriers count as road_obstacle, and does a dog crossing the road count?
- Scene facts are still unknown until samples/camera.md arrive: resolution and far-side pedestrian pixel height, whether the traffic signal is visible, whether the camera switches to IR monochrome at night, and whether there is a parking lane or bus stop. Most thresholds above depend on these.
- How long is the hidden test set in total? The per-hour false-positive budget for rare classes depends on it: q = 1 - exp(-lambda*T).
- The FASDD dataset licence could not be verified (the preprint was withdrawn). Check Science Data Bank before using it.
- The T4 speeds of the open-vocabulary models (OWLv2, Florence-2, Grounding DINO tiny) come only from third-party blogs. Benchmark them on a Kaggle or Colab T4 before committing.
- MOG2 absorption timing was measured on synthetic frames only. Re-validate the alpha-to-seconds mapping and the global-change guard on real sample footage, including night and rain if the samples have them.


## Key claims (as submitted for verification)
- D-Fire dataset is released under Creative Commons Zero v1.0 (CC0) and contains 1,164 fire-only, 5,867 smoke-only, 4,658 fire+smoke and 9,838 negative images (21,527 total), with 14,692 fire and 11,865 smoke boxes; repo also links surveillance videos — https://github.com/gaia-solutions-on-demand/DFireDataset (LICENSE and README)
- pedbrgs/Fire-Detection is MIT-licensed and provides YOLOv5 s/l models trained on D-Fire plus temporal false-positive filters AVT (area variation, recommended outdoors) and TPT (temporal persistence) — https://github.com/pedbrgs/Fire-Detection
- Pyronear YOLO11s smoke detectors are Apache-2.0, target early wildfire smoke, use 1024x1024 input and a sequential-detection option (nb_consecutive_frames=5) — https://huggingface.co/pyronear/yolo11s_sensitive-detector_v1.0.0
- FASDD has >120k images (FASDD_CV ~95,314); the ESSD preprint was withdrawn and the dataset licence was not confirmed — https://essd.copernicus.org/preprints/essd-2023-73/
- YOLOE is AGPL-3.0; YOLOE-11-S runs ~301 FPS on T4 with TensorRT at LVIS AP 27.5 (YOLOE-v8-S 305.8 FPS) — https://github.com/THU-MIG/yoloe
- In Ultralytics YOLOE, the first set_classes() call downloads a text encoder (needs network); exported models bake the prompt vocabulary into the weights and cannot take new prompts — https://docs.ultralytics.com/models/yoloe/
- OWLv2, Grounding-DINO-tiny and OmDet-Turbo checkpoints are Apache-2.0; Florence-2 base/large are MIT; YOLOE HF weights are AGPL-3.0 — HuggingFace model API tags (huggingface.co/api/models/google/owlv2-base-patch16-ensemble, IDEA-Research/grounding-dino-tiny, omlab/omdet-turbo-swin-tiny-hf, microsoft/Florence-2-large, jameslahm/yoloe)
- OmDet-Turbo-Base reaches 100.2 FPS with TensorRT and language cache (53.4 AP COCO zero-shot) — https://arxiv.org/abs/2403.06892
- YOLO-World is GPL-3.0 and reports 35.4 AP on LVIS at 52 FPS on V100 — https://github.com/AILab-CVC/YOLO-World
- YOLO26 T4 TensorRT10 latency: n 1.7 ms, s 2.5 ms, m 4.7 ms, l 6.2 ms, x 11.8 ms (COCO mAP 40.1-56.9); AGPL-3.0; yolo26-p2.yaml exists but no P2 weights released — https://docs.ultralytics.com/models/yolo26
- SAHI (MIT) sliced inference raises AP by 6.8/5.1/5.3 for FCOS/VFNet/TOOD without fine-tuning, and 12.7/13.4/14.5 cumulatively with slicing-aided fine-tuning — https://arxiv.org/abs/2202.06934
- OpenCV MOG2 (backgroundRatio 0.9, varThreshold 16) absorbs a newly stopped object after ~ln(0.9)/ln(1-alpha)/f ≈ 0.105/(alpha*f) seconds (measured within ~5% for alpha 0.02-0.0002); KNN absorbs ~2.2x slower — reasoning + scratchpad measurement (bench_bg.py, OpenCV 4.13, synthetic scene)
- On the dev CPU at 480x270: MOG2 ~5 ms, KNN ~4.4 ms, DIS ultrafast ~3 ms, Farneback ~55 ms per call; at 1080p Farneback ~1.2 s — scratchpad measurement (bench_bg.py)
- AI City Challenge 2021 Track 4 winner used MOG background modelling plus vehicle detection on background frames, temporal backtracking for start time, and trajectory-based road masks; F1 0.9524 — https://arxiv.org/html/2105.03827
- Dual-foreground (long-term vs short-term background) method detects temporally static regions such as abandoned objects or illegally parked vehicles — https://link.springer.com/article/10.1155/2008/197875
- SMIYC: RoadAnomaly21 has 100 test + 10 val images; RoadObstacle21 has 327 eval + 30 val images, obstacle track under CC BY 4.0; ego-vehicle street scenes — https://ar5iv.labs.arxiv.org/html/2104.14812
- Turbulent flames flicker at around 10 Hz, a basis for video fire verification — https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/TOREYIN2/node5.html
- Ultralytics recommends about 0-10% background (no-object) images to reduce false positives — https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/
- Qwen3-VL-2B-Instruct is Apache-2.0 with 4.26 GB of safetensors; InternVL3.5-1B is Apache-2.0 at 2.12 GB; Florence-2-large safetensors are 1.55 GB — https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct ; https://huggingface.co/OpenGVLab/InternVL3_5-1B ; https://huggingface.co/microsoft/Florence-2-large
- Ultralytics ByteTrack default track_buffer is 30 frames; COCO ids 14-19 are bird, cat, dog, horse, sheep, cow — https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/bytetrack.yaml ; https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml
- Enabling class c is worth it iff p*f > (1-p)*q*Fbar, where q is the probability of at least one FP over the whole test set (derived from the spec rule that a predicted-but-absent class is added to C with F1 = 0) — spec + reasoning (decide.py)
