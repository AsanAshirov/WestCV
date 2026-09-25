<!-- source: deep-research workflow wf_294a4b01-fc2, agent research:rules -->

# Rule engine on trajectories and scene geometry: one design per class

Scope: this covers the rule layer of Part A. It turns tracked detections plus the `camera.md` scene layout into `[start, end, label]` segments. Detector and tracker choice, learned accident models and the Part B estimator belong to other dimensions and appear here only where they affect the rules. All thresholds are **initial values to tune on your own dev labels**. Where a number comes from reasoning and not a source, it is marked *(estimate)*.

---

## 0. Two metric facts that shape every rule

### 0.1 Boundary tolerance is tight for short events

These tolerances were computed from the Part A temporal-IoU definition (scratchpad script). "Shift" means the prediction has the same length but is displaced. "Pad" means the prediction is symmetrically too long on each side.

| GT duration | max shift for IoU ≥ 0.3 / 0.5 / 0.7 | max pad per side for 0.3 / 0.5 / 0.7 | late start (end exact), 0.5 / 0.7 |
|---|---|---|---|
| 1.5 s | 0.81 / 0.50 / 0.26 s | 1.75 / 0.75 / 0.32 s | 0.75 / 0.45 s |
| 2 s | 1.08 / 0.67 / 0.35 s | 2.33 / 1.00 / 0.43 s | 1.00 / 0.60 s |
| 5 s | 2.69 / 1.67 / 0.88 s | 5.83 / 2.50 / 1.07 s | 2.50 / 1.50 s |
| 10 s | 5.38 / 3.33 / 1.76 s | 11.7 / 5.0 / 2.14 s | 5.0 / 3.0 s |
| 30 s | 16.2 / 10.0 / 5.3 s | 35 / 15 / 6.4 s | 15 / 9 s |

What this means in practice:
- `solid_line_crossing`, `failure_to_yield` and `red_light` events last about 1–3 s. At τ = 0.7 they need boundaries within roughly 0.3 s, so you need exact geometric crossing times, not "the frame the rule fired".
- `stopped_vehicle` must look back. Example: the GT stop lasts 20 s, and the rule reports the start at confirmation time, 10 s after the stop. IoU is then 0.50, which fails τ = 0.7 and barely passes 0.5. For a 15 s GT stop, IoU is 0.33.
- Score_A averages F1 over τ ∈ {0.3, 0.5, 0.7}. A detection with IoU 0.6 therefore earns about 2/3 of full credit. Boundary precision is worth one third of every class's score.

### 0.2 Class gating: predicting an absent class is expensive

If k classes are in the test set, one predicted segment of a class that never occurs multiplies Score_A by k/(k+1). For example, with |C| = 6 and Score_A = 0.30 the result drops to 0.257.

Decision rule (reasoning). Let:
- p = P(class present in the test set)
- F = expected F1 of our rule for that class if it is present
- q = P(the rule fires at least once on the whole hidden test set | class absent)
- S = the Score_A we would otherwise have

Enable class c iff **p·F/k > (1−p)·q·S/(k+1)**.

With q = 1, the break-even p is about 0.30–0.64 (k = 6–8, S = 0.3–0.4, F = 0.2–0.6). The design goal for rare classes is therefore not recall. It is **q ≈ 0**: several independent corroborating signals, conservative thresholds, and zero false positives across all sample minutes. Once q ≈ 0, enabling a rare class is almost free.

Note on near misses: Part B ignores frames around near_miss events. That implies near_miss is annotated in the hidden GT, so treat it as likely present.

---

## 1. Shared infrastructure

### 1.1 Data flow (two-pass, cache everything)

```
Pass 1 (GPU, once per video):
  decode (stride 1 or 2) -> detector -> online tracker (ByteTrack/BoT-SORT)
  per-frame extras: signal-lamp ROI brightness, downsampled frame every 2 s (for median backgrounds),
                    lane-zone dense-flow magnitude at 2-5 Hz (backup for congestion)
  cache -> tracks.npz: frame, t, id, cls, conf, x1,y1,x2,y2 (+ optional mask footprint)
Pass 2 (CPU, seconds):
  track stitching -> contact points -> BEV -> smoothing/kinematics -> lane assignment
  scene state (signal phase, queues, per-lane speed/occupancy at 1 Hz)
  14 class modules -> candidate segments (class, t0, t1, conf, actors, evidence)
  per-class post-processing -> class gating -> list[list]
```

- Budget: A and B together get 3× video duration, i.e. about 120 ms of wall time per 25 fps frame. The rule layer (pass 2) costs well under 1% of that. Pairwise TTC for N ≤ 50 objects is about 1.2k pairs per step, which is trivial in numpy.
- Caching pass-1 output lets you grid-search thresholds against dev labels with `evaluate.py` in seconds. This is the main productivity lever.
- Cross-cut, Part B: Part A must not feed Part B, but everything below except the offline smoothing and look-ahead has a causal twin (Kalman or one-euro filter, TTC) that B can recompute itself.

### 1.2 Scene configuration (`scene.yaml`, hand-made from `camera.md` and a median frame)

Build a **median background frame** (median of 1 frame per 2 s over a sample video). It shows clean markings without cars. Click polygons on it with a small OpenCV tool, CVAT or labelme, at native resolution. Suggested schema:

```yaml
image_size: [W, H]
homography_pts: {img: [[u,v],...], world_m: [[x,y],...]}   # >= 4, ideally 8-12 points
carriageway: [poly, ...]
non_carriageway: {sidewalk: [...], island: [...], parking_bay: [...], bus_stop: [...]}
lanes:
  - id: N1, poly: [...], approach: N, dir: [dx,dy] (BEV unit vector) or centreline polyline,
    allowed: [through, left], turn_pocket: false
separators:
  - {id: s1, polyline: [...], type: solid|dashed|double_solid|solid_dashed, crossable_from: left|right|none}
stop_lines: [{approach: N, polyline: [...], signal_group: G1}]
crossings: [{id: X1, poly: [...], signalized: true, curb_zones: [poly_a, poly_b]}]   # curb zones about 2 m deep
intersection_box: poly
gates: {entry: {N1: seg, ...}, exit: {E: seg, ...}}          # origin-destination lines
movements_prohibited: [{from: N, move: left}]                # from signs 3.18.x
u_turn: {prohibited: [poly], allowed: [poly]}                # sign 3.19, double solid line, openings
right_on_red: {N: false}                                    # true only if sign 5.42 is visible
signals: [{group: G1, lamps: {red: roi, yellow: roi, green: roi, arrow_right: roi}}]
static_ignore: [poly]     # timestamp overlay, trees, sky, permanent clutter
```

Also derive a **data-driven flow field** from all sample tracks: a BEV grid of 1 m cells holding a heading histogram and the dominant direction. Use it to validate the hand-drawn lane directions and as a second vote for wrong_way.

### 1.3 Ground-contact point and vehicle footprint

- **Default:** the bbox bottom-centre (x_c, y2). It is the ground point of the footprint edge **nearest the camera**:
  - For vehicles approaching the camera it is roughly the front bumper.
  - For receding vehicles it is the rear.
  - For crossing traffic it is the near side.
- **Footprint model in BEV:** an oriented rectangle with length L and width W by class *(estimates)*:

  | Class | L | W |
  |---|---|---|
  | car | 4.5 m | 1.8 m |
  | van | 5.5 m | 2.0 m |
  | truck | 9 m | 2.5 m |
  | bus | 12 m | 2.5 m |
  | motorcycle | 2.0 m | 0.8 m |
  | bicycle | 1.8 m | 0.6 m |

  The heading h comes from smoothed velocity. Let u be the unit vector from the vehicle toward the camera's ground projection (BEV of the image bottom-centre). Then:
  `centre ≈ p_c − u·(L/2·|h·u| + W/2·|h⊥·u|)`, `front = centre + (L/2)·h`, `rear = centre − (L/2)·h`.
  This gives the **front point** that red_light ("front crosses stop line"), failure_to_yield and stop_line need. At 15 m/s, getting the front wrong by 4.5 m shifts the time by 0.3 s, which is a whole IoU-0.7 budget for a 2 s event.
- **Better, if affordable:** a YOLO-seg mask. Take the lowest 10% of mask rows as the bottom contour and project it to BEV.
- **Per-lane bias correction:** for each lane, take the median lateral offset of contact points relative to the lane centreline, over all sample tracks, and subtract it. This removes the systematic shift toward the camera side for tall vehicles. It is essential for solid_line_crossing.
- **Reliability flags:**
  - bbox within 3 px of an image border counts as truncated. Do not use its position; interpolate instead.
  - If the box overlaps another box by IoU > 0.3 and its y2 is smaller (it is behind), mark it as occluded and down-weight it.

### 1.4 Approximate ground-plane homography (no formal calibration)

- **Point pairs:** use 8–12 road-plane points: lane-marking corners, stop-line ends, crossing-stripe corners. Get metric positions from:
  - **Lane width:** measure it if possible; otherwise assume 3.5 m.
  - **Dashed-line module:** Uzbekistan follows Soviet/GOST-style markings and the Vienna Convention. In GOST R 51256, line 1.5 has dashes of 1–3 m with gaps three times the dash, and approach line 1.6 has dashes three times the gap. Verify against what you see.
- **Fit:** `cv2.findHomography(img, world, cv2.RANSAC, 3.0)`, the same approach as Roboflow supervision's `ViewTransformer`, which uses `cv2.getPerspectiveTransform`.
- **Sanity checks:**
  - Lane separators map to parallel lines.
  - Lane widths agree within 10%.
  - The median free-flow car speed comes out plausible (30–70 km/h).
- **Scale map:** compute pixels-per-metre ppm(u,v) from the Jacobian of H⁻¹. Define a **"kinematics-valid zone"** where ppm ≥ 8. Simulated with 1.5 px of white jitter (assumed) and Savitzky–Golay (SG) smoothing:

  | Setting | accel noise σ |
  |---|---|
  | 1.0 s SG window, ppm 20 | 0.4–0.5 m/s² |
  | 1.0 s SG window, ppm 8 | 1.0–1.3 m/s² |
  | 1.0 s SG window, ppm 4 | 2.0–2.7 m/s² |
  | 1.5 s SG window, ppm 8 | 0.3–0.5 m/s² |

  A hard-braking threshold of 3.4 m/s² is therefore only usable where ppm ≥ 8 with a window of at least 1.5 s. Real detector jitter is correlated, so these numbers are optimistic. Braking-based rules (near_miss, accident) belong in the near and mid field only.
- **Fallback without H:** normalise image displacement by bbox height per second, times a class height (car 1.5 m, bus or truck 3 m, person 1.7 m). This gives rough m/s. It is good enough for stopped/moving decisions and not for braking.

### 1.5 Tracking robustness: stitching, occlusion, class smoothing

- **Online tracker:** raise Ultralytics' default ByteTrack `track_buffer` from 30 frames to about 75 frames at 25 fps (3 s). Keep the two-stage low threshold (default `track_low_thresh` 0.1).
- **Offline stitching (Part A only):** for each tracklet end (t_e, p_e, v_e) and later start (t_s, p_s), link them if all of these hold:
  - 0 < Δt ≤ 4 s, or ≤ 30 s if the tracklet was stationary and |p_s − p_e| ≤ 1.5 m (detector flicker on parked cars, at night)
  - |p_e + v_e·Δt − p_s| ≤ 2 m + 0.5·|v_e|·Δt
  - same class group
  - bbox area ratio in [0.5, 2]
  - optional: HSV histogram Bhattacharyya distance < 0.4

  Solve the linking greedily or with the Hungarian algorithm on cost.
- **ID-switch guard:** mark a track as "suspect" in any 1 s window with a BEV jump > 3 m in 0.2 s or an area-ratio jump > 1.8. Rules that rely on heading reversal (U-turn, wrong_way) or Δv spikes (accident, near_miss) must reject suspect windows.
- **Class smoothing:**
  - Assign each track its majority class over its life (car/truck/bus flicker).
  - A **person is a rider**, not a pedestrian, if its box overlaps a bicycle/motorcycle box with IoU > 0.2 for more than 50% of its frames.
  - Merge duplicate tracks with IoU > 0.7 for more than 1 s.
- **Minimum length:** 0.5 s for a track to exist, 1 s for it to feed rules.

### 1.6 Smoothing and kinematics

- **Offline (A):**
  1. Hampel filter (window 7 samples, 3σ) on BEV x, y.
  2. Linear interpolation of gaps ≤ 1 s. Do not bridge longer gaps; split the track instead.
  3. Savitzky–Golay with polyorder 2, window 1.0 s for position and velocity and 1.5 s for acceleration. First-order SG with a 1 s window is standard for video-derived NGSIM trajectories.

  The centred window has zero lag. In simulation, a 5 m/s² brake step starting at t = 4.00 s produced an SG(1.5 s) acceleration that crossed −1.0 m/s² at 3.76 s and −3.4 m/s² at 4.16 s. So **onset = last crossing of a low threshold (1 m/s²) before the peak** is accurate to about ±0.25 s.
- **Online (B, cross-cut):** a constant-velocity Kalman filter or a one-euro filter, with cutoff f_c = f_cmin + β·|speed|. Start with f_cmin ≈ 0.5–1 Hz and a small β, then tune.
- **Heading:** θ = atan2(v) only when speed > 1.0 m/s; otherwise hold the last reliable θ. Take yaw rate from unwrapped θ with SG.

  Warning: image y points down. Compute turn direction in a right-handed BEV (y up), where a left turn is counter-clockwise, i.e. positive Δθ, in right-hand traffic.
- **Speed vocabulary (BEV):**

  | Term | Definition |
  |---|---|
  | stationary | speed < 0.5 m/s and displacement < 1 m over 2 s |
  | crawling | < 2 m/s (7 km/h) |
  | moving | > 3 m/s |
  | hard braking | a ≤ −3.4 m/s² (AASHTO design deceleration; DRAC conflict threshold 3.35 m/s²) |
  | swerve | yaw rate ≥ 20°/s at speed ≥ 4 m/s, or lateral accel ≥ 2–2.5 m/s² *(estimates)* |

### 1.7 Lane assignment with hysteresis and separator crossing

- `lane_raw(t)` is the lane polygon that contains the corrected footprint centre; inside the box it is "box", and off-road it is None.
- `lane(t)` only commits to a new lane after it has persisted for ≥ 0.5 s **and** the centre is ≥ 0.5 m from the separator.
- For every separator k, compute the signed BEV distance d_k(t) of the footprint centre. A crossing is a sign change where |d| ≥ 0.3 m on both sides for ≥ 0.4 s each. This is the primitive behind solid_line_crossing, wrong_way start/end and illegal_turn lane-at-stop-line.

### 1.8 Signal phase estimation (shared by red_light, stop_line, stopped_vehicle, congestion, failure_to_yield)

**Case A: the signal head is visible.**
1. For each lamp ROI, compute per frame the mean V (HSV) weighted by saturation.
2. Normalise per lamp by its 5th and 95th percentile over the whole video. A lamp is on if its normalised value is > 0.5.
3. Decode the state with Viterbi over an HMM with the legal order green → (flashing green) → yellow → red → (red+yellow) → green, and minimum dwell times (yellow ≥ 2 s). Treat flashing green as green.

Offline decoding gives about 0.04 s transition accuracy. If only the cross street's head is visible, infer the complement. Uzbekistan's traffic rules (SDA) also give a green arrow in the additional section the same meaning for its direction, so decode arrows separately.

**Case B: the signal is not visible.** Infer the phase from flows:
- n_a(t) = stop-line crossings per second for approach a.
- q_a(t) = a stationary vehicle within 3 m of stop line a.
- The approach is green from queue-head departure to the last discharge before a new queue head stops.
- Estimate the cycle length by autocorrelating n_a(t) over the whole video (fixed-time controllers are periodic), then refine each boundary locally.
- Green onset ≈ queue-head departure minus about 1–1.5 s *(estimate; HCM's default start-up lost time is 2.0 s summed over the first vehicles)*.

In case B, rules that depend on red become **high-precision, low-recall only**.

**Local rules (Uzbekistan SDA):**
- Yellow also prohibits movement, with an exception for vehicles that cannot stop safely (clause 42). The class, however, is "red" only.
- The SDA requires stopping before the stop line with the front not past it. This is the stop_line class.
- Right turn on red is **not** general. In Tashkent it is allowed only at intersections with the special sign (5.42, "yield to all, then turn right"). Set `right_on_red` from `camera.md`.

### 1.9 Generic helpers (pseudo-code)

```python
runs(mask, t, fill_gap, min_dur)          # boolean series -> [(t0,t1)], filling holes < fill_gap
onset_back(sig, t_peak, lo)               # last t <= t_peak with sig(t) < lo  (e.g. decel < 1 m/s^2)
offset_fwd(sig, t_peak, lo, hold)         # first t >= t_peak with sig < lo sustained for `hold` s
cross_time(d, t, level)                   # sub-frame linear interpolation of d(t) == level
```

Always interpolate crossing times between samples (sub-frame accuracy costs nothing).

---

## 2. Per-class designs

Legend: (a) geometry, (b) signals, (c) logic, (d) start/end per annotator convention, (e) false positives (FP) and mitigations, (f) interactions, (g) difficulty (1–5), frequency, recommendation. Frequencies are estimates for an urban Tashkent-type CCTV view.

### 2.1 stopped_vehicle
- **(a)** Carriageway; exclusion polygons (parking bays, bus/taxi stops, off-road); stop lines and queue zones; intersection box.
- **(b)** Stitched vehicle tracks; BEV speed and displacement; lane; signal phase; leader vehicle in the same lane.
- **(c)**
```python
stat = (speed < 0.5) & (disp_2s < 1.0)
for (t0, t1) in runs(stat, fill_gap=1.0, min_dur=10.0):
    if not on_carriageway(track, t0..t1) or in_exclusion(...): continue
    if frac(in_signal_queue(track, t) for t in run) > 0.5: continue
    if frac(congestion_active(dir(lane), t)) > 0.5: continue
    if yield_wait(track, t0..t1): continue      # in box/turn pocket, heading across oncoming flow, oncoming vehicle within 3 s
    emit(t_stop, t_move)

in_signal_queue(v, t) = approach_lane(v) and (phase in {red, yellow} or t - green_onset < 8 s)
                        and (dist_front_to_stopline < 3 m or (leader within 12 m is stationary and in_signal_queue(leader, t)))
```
  Without signal info, a vehicle counts as in the queue if a chain of stationary vehicles connects it to the stop line **and** the queue head departs within 120 s.
- **(d)** Start is the moment the vehicle stops: the downward crossing of speed 1.0 m/s just before the run (look-back). This is the key fix; see §0.1. End is the first time displacement from the stop position exceeds 1.0 m with speed > 0.5 m/s held for 1 s. If the track ends while stationary, check the median-background difference in its box: if the region reverts to road, end = disappearance (removed); if the car is still visible, the tracker failed, so extend. Stopped at video start → start = 0.0; still stopped at video end → end = duration.
- **(e)** FPs:
  - detector flicker splitting the stop (long-gap stitching)
  - buses at stops (exclusion polygon; or require ≥ 30 s there)
  - left-turners waiting in the box (`yield_wait`)
  - vehicles yielding at a crossing
  - parked cars in lanes that are really parking lanes (config)
  - vehicles in jams
- **(f)** Exclude vehicles that are a stop_line or queue case. Crashed vehicles standing ≥ 10 s may be labelled both accident and stopped_vehicle (ask the organizers or check dev). Two stopped vehicles at once → one union segment.
- **(g)** Difficulty 2, frequency high. **Enable by default.**

### 2.2 congestion
- **(a)** Lanes grouped by direction; per-direction monitoring zone inside the kinematics-valid area.
- **(b)** At 1 Hz, per lane l: occupancy O_l (fraction of lane mask covered by footprints; with MOG2 foreground ratio as a backup when detection saturates) and median speed v_l. As a backup, median dense optical flow in the lane mask.
- **(c)**
```python
jam_d(t) = all(v_l < 2.0 for l in lanes(d)) and all(O_l > 0.3 for l in lanes(d))
jam_d = median_filter(jam_d, 5 s)
for (t0, t1) in runs(jam_d, fill_gap=10, min_dur=30):
    if signal known and run lies inside one red phase of d + 10 s: continue   # normal red queue
    if signal unknown and (t1 - t0) < max_red_est + 15 s: continue
    emit(start, end)
```
- **(d)** Start is when the queue stops moving: use the first raw (unsmoothed) threshold crossing of median v_d below 2 m/s that lies within 10 s before the smoothed run start. End is when the queue clears: v_d > 4 m/s (hysteresis) held for 5 s, or O_d < 0.2. Clip to the duration.
- **(e)** FPs: normal red queues (phase check); night detection loss (flow and foreground backup); detector undercounting in dense occlusion (use pixel occupancy, not counts).
- **(f)** Suppresses stopped_vehicle inside it. Down-weight near_miss inside congestion (stop-and-go braking). stop_line can still co-occur.
- **(g)** Difficulty 3; frequency depends on the scene (rush hour). **Enable** if samples show dense traffic. The rule is hard to trigger falsely, so q is low.

### 2.3 wrong_way
- **(a)** Lane polygons with direction; data-driven flow field; intersection box excluded.
- **(b)** Alignment a(t) = h(t)·dir(lane(t)); speed; path length.
- **(c)**
```python
opp = (lane not in {box, None}) & (speed > 1.5) & (align < -0.5) & (flowfield_align < -0.5)
for run in runs(opp, fill_gap=0.5, min_dur=1.5):
    if path_len(run) >= 8 m and not suspect(run): emit(...)
```
  This one test covers both "against traffic" and "driving in the oncoming lane".
- **(d)** Start is when the vehicle enters the opposing lane: back-project to the zero crossing of the separator signed distance d_k, or the lane-polygon entry. If the vehicle enters the frame already wrong-way, use its first detection. End is the crossing back into a correct lane, or the last detection (left the frame).
- **(e)** FPs:
  - ID switches between opposing vehicles (suspect-window guard)
  - reversing and parking manoeuvres (speed ≤ 2 m/s and path < 8 m → skip)
  - turning inside the box (excluded)
  - lane polygons drawn badly at the median; use a 0.5 m inward buffer
- **(f)** During a U-turn arc, align ≈ 0, so no wrong_way fires. It fires only if the vehicle continues in the wrong lane afterwards. Overtaking over a solid centre line gives both solid_line_crossing and wrong_way. Turning into a one-way street against its direction gives illegal_turn and wrong_way.
- **(g)** Difficulty 2, frequency low to moderate. **Enable by default** with the 1.5 s / 8 m gate.

### 2.4 jaywalking
- **(a)** Carriageway; crossings with a 1.5 m buffer; islands and medians (not carriageway); curb polylines.
- **(b)** Pedestrian tracks (riders excluded); feet point = bbox bottom-centre; expected person height h_exp(y), from a regression of pedestrian bbox height against y.
- **(c)**
```python
on_road = in_carriageway(feet) & ~in(crossing ⊕ 1.5 m) & ~in(island) & ~rider & ~truncated & (h_bbox > 0.7 * h_exp(y2))
for run in runs(on_road, fill_gap=1.0, min_dur=1.0):
    if max_depth_from_curb(run) >= 0.8 m and not static_for(> 60 s): emit(...)
```
- **(d)** Start is the step onto the road: sub-frame zero crossing of the signed distance to the curb polyline, from the smoothed track. End is leaving the carriageway (reaching sidewalk, island or crossing) or the frame. Several pedestrians at once → union.
- **(e)** FPs:
  - feet hidden behind cars, which puts bottom-centre on the road (height check)
  - riders
  - people getting in or out of cars at the curb; decide on dev labels whether annotators count them
  - traffic police and road workers (ask the organizers)
  - static mannequins or advertisements
- **(f)** Mutually exclusive by location with failure_to_yield pedestrians. Can co-occur with near_miss or accident.
- **(g)** Difficulty 2, frequency high. **Enable by default.**

### 2.5 failure_to_yield
- **(a)** Crossing polygons (optionally split per lane strip); curb waiting zones about 2 m deep; `signalized` flag.
- **(b)** Vehicle front-entry and rear-exit times; pedestrians on the crossing or stepping onto it (velocity toward the road > 0.5 m/s); lateral distance between pedestrian and vehicle path.
- **(c)**
```python
for v crossing crossing X: t_in = cross_time(front enters X); t_out = cross_time(rear leaves X)
    if max_speed(v, t_in..t_out) < 1.0: continue
    peds = [p for p in peds if exists t in [t_in - 0.5, t_out]:
                (p in X and lat_dist(p, path_v) <= 6 m) or (p in curb_zone(X) and v_toward_road(p) > 0.5)]
    if X.signalized and veh_green(t_in) and all(p crossing on ped-red): policy_from_dev()
    if peds: emit(t_in, t_out)
```
- **(d)** Start = front enters crossing; end = vehicle leaves crossing, i.e. the **rear** exits (front exit + L/speed). Typical length 0.5–2 s, so use the precise front/rear model (§1.3) and learn a per-class bias on dev.
- **(e)** FPs: pedestrians waiting who are not crossing (require motion toward the road or presence on the stripes); pedestrian on the far half beyond an island; vehicles more than 6 m laterally away.
- **(f)** Several vehicles in overlapping intervals → union. Often co-occurs with near_miss. Can co-occur with red_light on signalized crossings.
- **(g)** Difficulty 3; frequency high wherever a zebra crossing is in view. **Enable if a crossing is in the frame.**

### 2.6 red_light
- **(a)** Stop lines with signal-group mapping; intersection box and exit gates; the `right_on_red` flag.
- **(b)** Front-crossing time t_c at the stop line; signal state at t_c (the arrow group for the vehicle's movement, if present); movement type.
- **(c)**
```python
for v with front crossing stopline(a) at t_c, speed > 1 m/s, heading within 45° of dir(a):
    s = phase(group(a, movement(v)), t_c)
    if s == RED and t_c - red_onset >= 0.3 s:
        if movement(v) == right and right_on_red[a]: continue
        if stops_before_box(v) and waits_until_green(v): continue     # -> stop_line instead
        emit(t_c, t_leave)
```
  When the phase is inferred (§1.8, case B), additionally require the conflicting approach to be actively discharging (≥ 2 crossings within ±3 s of t_c) and the own queue to stay stopped behind v.
- **(d)** Start = the front crosses the stop line (sub-frame interpolation; front model for receding vehicles). End = the footprint centre leaves the intersection box or the exit gate, or the frame.
- **(e)** FPs:
  - signal glare, or a truck occluding the head (HMM smoothing plus a lamp-visibility check)
  - wrong mapping between approach and signal
  - vehicles already in the box at red onset (handled by using t_c)
  - legal right turns on red
  - emergency vehicles
- **(f)** Platoon of red-runners → union of overlapping intervals. With stop_line: a vehicle that crosses on red and then stops before the box is stop_line only; one that continues into the box on red is red_light. Co-occurs with accident and failure_to_yield.
- **(g)** Difficulty 3 if the signal is visible, 4–5 if inferred; frequency moderate. **Enable only if the signal is visible**, or if phase inference validates on dev.

### 2.7 stop_line
- **(a)** Stop lines; the crossing behind them; box entry line; signal.
- **(b)** Stationary run; front overshoot past the stop line d_over; phase.
- **(c)**
```python
for stationary run R (>= 1.0 s) of v in approach a during RED:
    d_over = signed_dist(front(v), stopline(a))       # + = past the line
    if 0.3 m < d_over and front not past box entry:
        emit(t_stop, green_onset(a, after=R.t0))
```
  Several vehicles in the same red → one segment, from the first stop to green.
- **(d)** Start = the vehicle stops (speed falls below 1.0 m/s, look-back). End = **the signal turns green**, not the vehicle's departure. Without a visible signal, use queue-head departure minus about 1.5 s *(estimate; calibrate the bias on dev)*.
- **(e)** FPs:
  - far-field stop-line error (ppm too low for 0.3 m accuracy); restrict to near/mid field
  - front-length prior error (±0.7 m) for receding vehicles
  - left-turners waiting inside the box (they entered it, so not this class)
- **(f)** Exclusive with red_light for the same vehicle and phase. Excluded from stopped_vehicle. Can co-occur with congestion.
- **(g)** Difficulty 3 (visible signal) or 4 (inferred); frequency moderate to high at signals. **Enable when signal information exists** and the stop line is in the near/mid field.

### 2.8 illegal_u_turn
- **(a)** U-turn prohibited and allowed polygons (sign 3.19 exists in Uzbekistan; double solid lines; openings); lane directions.
- **(b)** Unwrapped θ; lane direction before and after; apex position.
- **(c)**
```python
for window [t0, t1], t1 - t0 <= 20 s, not suspect:
    if abs(theta(t1) - theta(t0)) >= 150° and dot(dir(lane_before), dir(lane_after)) < -0.7:
        apex = point of max chord distance
        if apex in u_prohibited and apex not in u_allowed: emit(t_turn_start, t_turn_end)
```
- **(d)** Start = the vehicle starts turning: back from the apex to the last time |yaw rate| < 5°/s with heading within 15° of the incoming lane direction. End = heading within 15–20° of the outgoing lane direction and |yaw rate| < 5°/s.
- **(e)** FPs: ID switches between opposing vehicles (suspect guard); three-point turns and reversing (velocity flips with near-zero displacement, so require continuous speed > 1 m/s through the arc); vehicles leaving parking.
- **(f)** Never also emit illegal_turn. Emit wrong_way only if the wrong-lane driving continues. solid_line_crossing during a U-turn may be double-labelled; ask the organizers.
- **(g)** Difficulty 3; frequency depends on the scene. **Enable only if `camera.md` shows a prohibition in view.**

### 2.9 illegal_turn
- **(a)** Entry gates per lane at the stop line; exit gates per leg; allowed movements per lane (lane arrows); prohibited movements (3.18.x); one-way exits.
- **(b)** Origin-destination pair; lane at the stop line (mode over the last 1.0 s before crossing); Δθ sign in right-handed BEV.
- **(c)**
```python
mv = movement(entry_approach, exit_leg)   # or from Δθ: |Δθ| < 30° through; 60-120° left/right by sign; > 150° u
if mv in {left, right} and (mv not in allowed[lane_at_stopline] or (approach, mv) in prohibited)
   and lateral_margin_inside_lane >= 0.5 m:
    emit(t_turn_start, t_turn_end)
```
- **(d)** Start = heading deviates more than 10–15° from the approach direction, backed to the yaw-rate onset (> 5°/s). End = heading within 15° of the exit-leg direction with yaw rate < 5°/s, or crossing the exit gate.
- **(e)** FPs: lane misassignment in perspective (margin rule and bias correction); tracks broken through the box (stitching); U-turns misclassified as turns.
- **(f)** Exclusive with illegal_u_turn. Turning on red is red_light, not illegal_turn, unless it is also from the wrong lane. Can be paired with wrong_way or solid_line_crossing.
- **(g)** Difficulty 4, frequency low to moderate. **Enable only if lane-level restrictions exist** and lanes are resolvable.

### 2.10 solid_line_crossing
- **(a)** Solid separators, including the solid sections before stop lines and double-solid centre lines. Dashed lines are excluded. For solid-dashed lines, only crossing from the solid side counts.
- **(b)** Signed BEV distance d(t) of the bias-corrected footprint centre to the separator; half-width w/2 by class (0.9 m car, 1.25 m bus or truck); lateral speed.
- **(c)**
```python
for track, solid separator k with longitudinal overlap:
    sign change of d with |d| >= w/2 + 0.2 m held >= 1.0 s on the new side
    and total lateral excursion >= 0.6 m and crossing point not in intersection box and not suspect
    start = cross_time(|d| == w/2, approach side)    # wheel touches the line
    end   = cross_time(|d| == w/2, new side)         # fully in new lane
    emit(start, end)
```
- **(d)** Start = the wheel crosses the line; end = fully in the new lane. That is about 1.8 m of lateral travel at about 1 m/s, so roughly 1–3 s. Interpolate sub-frame and calibrate the per-class bias on dev.
- **(e)** FPs: perspective lateral bias for tall vehicles (per-lane correction); vehicles straddling the line (only complete crossings count); jitter near the line (0.6 m excursion gate); worn or occluded markings; motorcycles filtering between lanes (check dev).
- **(f)** Co-occurs with wrong_way (centre line) and possibly with U-turns and turns. Simultaneous crossings → union.
- **(g)** Difficulty 3; frequency high wherever solid lines are in view. **Enable if the scene has solid lines.**

### 2.11 accident
- **(a)** Carriageway; kinematics-valid zone; fixed-object polygons (poles, barriers, islands).
- **(b)** BEV footprint gap g_ij; image mask or bbox contact at similar depth; Δv over 0.5 s (shock); yaw spike; sudden stop; post-impact immobility; pedestrian fall (w/h goes from < 0.6 to > 1); a learned clip score (other dimension) and the Part B risk curve (A may use B).
- **(c)**
```python
for pair (i, j), first t* with g_ij <= 0.3 m or (mask/bbox contact and |y2_i - y2_j| < 0.15*h):
    ev  = 2*[max(shock_i, shock_j) >= 3 m/s or yaw_spike >= 60°/s]
    ev += 1*[closing_speed(t* - 0.5) >= 2 m/s]
    ev += 1*[both stationary within 5 s and >= 3 s]
    ev += 2*[pedestrian fall] + 2*[learned_score >= θ_L]
    if ev >= 3 and no suspect window in [t* - 1, t* + 2]: emit
# single vehicle: footprint enters a fixed-object polygon, or v > 5 m/s -> < 1 m/s within 1 s with no leader or signal to explain it
```
- **(d)** Start = the first frame with visible contact: t where g reaches 0 (interpolated), snapped to the shock onset if that is within ±0.5 s. End = the time all involved objects stop or leave: the maximum over participants of (speed < 0.3 m/s held 2 s, or last detection). Cap at 30 s. Typical GT length 2–10 s *(estimate)*, so every 0.9 s of shift costs the τ = 0.7 match on a 5 s event.
- **(e)** FPs:
  - image-space overlap of vehicles at different depths (BEV check)
  - close stop-and-go gaps in queues (shock requirement)
  - ID switches (suspect guard)
  - people bending into cars (fall check needs a nearby vehicle and speed)
- **(f)** Exclusive with near_miss. Possibly followed by stopped_vehicle, fire_smoke, congestion, and drivers walking on the road (jaywalking?). Ask the organizers.
- **(g)** Difficulty 5. Rare in real footage but almost certainly in the test set, because Part B scores only accidents. **Enable**, fused with a learned model.

### 2.12 near_miss
- **(a)** Kinematics-valid zone; conflict areas (box, crossings, merges).
- **(b)** Pairwise TTC (constant-velocity, footprint circles), PET, DRAC = Δv²/(2·gap); per-track deceleration and yaw; minimum gap.
- **(c)**
```python
conflict = (min TTC <= 1.5 s) or (PET <= 1.0 s) or (DRAC >= 3.35)   # SSAM default TTC 1.5 s; PET 1.0 s is stricter than SSAM's 5 s screen
evasive  = (decel >= 3.4 for >= 0.3 s) or (yaw_rate >= 20°/s and speed >= 4) or (pedestrian |Δv| >= 1 m/s within 0.5 s)
if conflict and evasive and min_gap > 0.3 m and rel_speed >= 3 m/s and in_valid_zone: emit
```
- **(d)** Start = onset of evasive action: `onset_back(decel, t_peak, 1.0 m/s²)`, or yaw rate < 5°/s. End = the users are clear: the first t after the closest approach with gap > 3 m and range-rate > 0 (or TTC > 3 s), or one leaves the frame.
- **(e)** FPs: braking into a red queue (3.4 m/s² plus the TTC requirement); far-field acceleration noise (zone gate); slow turning near pedestrians (relative-speed gate); congestion stop-and-go (suppress).
- **(f)** Exclusive with accident. Co-occurs with failure_to_yield and jaywalking. It affects only Part A; Part B ignores frames around it.
- **(g)** Difficulty 5; frequency moderate; likely present in the GT. **Enable at high precision.**

### 2.13 road_obstacle
- **(a)** Carriageway; static-ignore mask (manholes, painted symbols, pole shadows).
- **(b)** COCO animal detections; dual-background static foreground (Porikli: a short-term model absorbs the object while the long-term model still flags it); vehicles swerving around the same point.
- **(c)**
```python
A) animal track (conf >= 0.4) with feet on carriageway >= 1.0 s -> segment
B) LT = median(frames over 120 s, every 2 s); ST = median(last 10 s)
   blob = |ST - LT| (Lab, globally brightness-normalised) > τ, area 0.1-6 m² in BEV, in carriageway,
          not under any vehicle/person box (dilated 20%), persists >= 10 s, detector finds no vehicle in crop,
          AND (>= 2 vehicles swerve >= 1 m around it, or persists >= 30 s)
```
- **(d)** Start = the obstacle appears: backtrack frame by frame over the blob area to the first frame where at least 50% of it differs from the pre-event background and stays different. End = removed (area < 30% held 2 s), or the video end.
- **(e)** FPs: moving shadows or clouds, wet patches, headlights at night, auto-exposure changes, blowing bags, people, cones placed by workers (ambiguous).
- **(f)** Stationary vehicles are stopped_vehicle, not obstacles. Accident debris is ambiguous.
- **(g)** Difficulty 4, frequency rare. **Enable the animal branch at high precision. Keep branch B off** unless samples show debris and all sample minutes give 0 false positives.

### 2.14 fire_smoke
- **(a)** Carriageway plus a 5 m margin; ignore sky, chimneys and known static lights.
- **(b)** A fire/smoke detector fine-tuned on D-Fire (about 21.5k images, YOLO format, CC0 1.0), run at 2 Hz; fire flicker (temporal variance); smoke growth and upward flow.
- **(c)**
```python
p(t) = max conf of fire/smoke boxes in ROI
active = (mean_3s(p) >= 0.5) & (box present in >= 60% of frames) & (box centre drift < 2 m over 5 s)   # not exhaust moving with a car
emit runs >= 3 s (fire also needs flicker; smoke also needs area growth)
```
- **(d)** Start = the first visible smoke: re-run the detector at full fps backwards on the ROI with a threshold of 0.2 until it drops. End = the last frame above 0.2, or the video end.
- **(e)** FPs: cold-weather exhaust, dust, fog, steam, tail lights and sodium lamps at night, sunset glare.
- **(f)** Often follows an accident; may co-occur with stopped_vehicle.
- **(g)** Difficulty 3, frequency very rare. **Enable only with q ≈ 0**: detector, a physical cue and ≥ 5 s persistence together.

---

## 3. Segment post-processing (per class)

Pipeline:
1. Candidate intervals per actor.
2. Boundary refinement (per-class rules above, sub-frame crossings).
3. **Bias calibration:** start += b_s[c], end += b_e[c], using the median (pred − GT) over matched dev pairs.
4. Gap merge.
5. Drop too-short segments; cut or flag too-long ones.
6. **Union same-class overlaps.** Annotators merge simultaneous same-class events, and the harness would otherwise drop the later overlapping one.
7. Clip to [0, duration], taking duration from the frame count and fps. Events still running at the end get end = duration; events running from the start get start = 0.
8. Enforce start < end and round to 0.01 s.
9. Class gating and a per-class confidence threshold.

| class | gap merge | min dur | max dur | typical GT dur *(estimate)* |
|---|---|---|---|---|
| accident | 1.0 s | 1.0 | 30 | 2–10 s |
| near_miss | 1.0 | 0.8 | 10 | 1–5 |
| red_light | 0.5 | 0.8 | 15 | 2–6 |
| wrong_way | 2.0 | 1.5 | – | 3–30 |
| illegal_u_turn | 1.0 | 2.0 | 30 | 4–12 |
| stopped_vehicle | 3.0 | 10.0 | – | 10–600 |
| jaywalking | 2.0 | 1.0 | – | 3–20 |
| failure_to_yield | 0.5 | 0.4 | 10 | 0.5–3 |
| illegal_turn | 1.0 | 1.5 | 20 | 3–8 |
| solid_line_crossing | 0.5 | 0.5 | 8 | 1–3 |
| stop_line | 2.0 | 1.0 | 150 | 5–60 |
| congestion | 10 | 30 | – | 60+ |
| road_obstacle | 5 | 5 | – | 10+ |
| fire_smoke | 5 | 3 | – | 10+ |

Keep each gap-merge value below the typical gap between separate events of that class, or you will fuse distinct GT events and lose true positives.

Tune each class independently. Per-class F1 does not depend on other classes, so use coordinate descent over 2–4 physically meaningful parameters per class. Prefer a parameter plateau over a single peak, because the dev set is small.

## 4. Where Part A's offline nature helps

Offline, multi-pass processing is not available to Part B.

- **Zero-lag, bidirectional smoothing:** centred SG windows and forward-backward Kalman/RTS smoothing. Onsets from threshold crossings are unbiased to within about ±0.25 s.
- **Track stitching** across long gaps using future tracklets.
- **Look-ahead confirmation plus look-back boundaries:**
  - stopped_vehicle (confirm 10 s, start at the stop)
  - congestion (confirm 30 s, start at the queue stop)
  - accident (end when everything has stopped)
  - near_miss (onset from the later peak)
  - wrong_way, illegal_turn and illegal_u_turn (classify the full origin-destination path first, then place the boundaries)
- **Future signal state:** stop_line's end is the future green onset. Whole-video cycle fitting and Viterbi decoding feed red_light and stop_line.
- **Backward search for first appearance:** road_obstacle and fire_smoke (re-scan at full fps with lower thresholds).
- **Whole-video statistics:** the flow field, person-height regression h_exp(y), per-lane bias correction, lamp normalisation percentiles, and median backgrounds.
- **Second pass at full frame rate** only around candidate windows, for failure_to_yield, red_light, solid_line_crossing and accident. This buys boundary precision cheaply.

## 5. Summary table and implementation order

| class | geometry needed | key signal | difficulty | priority |
|---|---|---|---|---|
| stopped_vehicle | carriageway, exclusions, stop lines | BEV speed run ≥ 10 s, look-back start | 2 | P1, default on |
| wrong_way | lane directions, flow field | heading·lane dir < −0.5, 1.5 s / 8 m | 2 | P1, default on |
| jaywalking | carriageway, crossings ± 1.5 m, islands | feet on road outside crossing | 2 | P1, default on |
| congestion | lanes per direction | all-lane speed < 2 m/s, occupancy, 30 s | 3 | P2, on if dense traffic |
| solid_line_crossing | solid separators by type | signed distance ± w/2 | 3 | P2, on if solid lines |
| failure_to_yield | crossings, curb zones | pedestrian on crossing while front→rear pass | 3 | P2, on if crossing |
| red_light | stop lines, signal ROI/groups, box | front crossing at RED | 3/5 | P3, on if signal visible |
| stop_line | stop lines, signal | front past line, stationary, until green | 3/4 | P3, on if signal info |
| accident | valid zone, fixed objects | contact + shock + immobility (+ learned) | 5 | P2 (with learned model), on |
| near_miss | valid zone | TTC ≤ 1.5 / DRAC ≥ 3.35 + evasive | 5 | P3, high precision |
| illegal_turn | gates, lane movements | origin-destination movement ∉ allowed[lane] | 4 | P4, only if restrictions |
| illegal_u_turn | U-turn zones | Δθ ≥ 150° in prohibited zone | 3 | P4, only if prohibited |
| road_obstacle | carriageway, ignore mask | animals; dual-background static blob | 4 | P5, animal branch only |
| fire_smoke | ROI | D-Fire detector + persistence | 3 | P5, q ≈ 0 gate only |

**Order:**
1. Scene-config tool, homography and flow field, with a pass-1 cache.
2. Label all samples with the annotator conventions; split the work by class across 3 people.
3. Stitching, smoothing and lane hysteresis.
4. stopped_vehicle, wrong_way, jaywalking.
5. Post-processing, bias calibration and the `evaluate.py` tuning loop.
6. congestion, solid_line_crossing, failure_to_yield.
7. Signal phase, then red_light and stop_line.
8. accident and near_miss (fused with the learned model; share TTC code with Part B but recompute it causally).
9. illegal_turn and illegal_u_turn.
10. High-precision gates for road_obstacle and fire_smoke.
11. A final gating review using the §0.2 decision rule.

## 6. Questions for the organizers (ask in the channel)

1. Is a crashed vehicle standing ≥ 10 s also labelled stopped_vehicle? Are drivers walking on the road after a crash labelled jaywalking?
2. A vehicle crosses the stop line on red and stops before the box: is it stop_line only? And if it then proceeds on red?
3. A U-turn or turn across a solid line: double-labelled with solid_line_crossing?
4. A pedestrian crossing against their red at a signalized crossing while a car passes on green: is that failure_to_yield?
5. Do traffic police and road workers count as jaywalking? Are riders excluded?
6. What minimum duration does annotated congestion have?
7. Is a right turn on red under sign 5.42 labelled red_light?



## Top recommendations
- Build the scene config (polygons, lane directions, separator types, stop lines, crossings, signal ROIs, homography points) on a median background frame first; every rule depends on it.
- Cache detector+tracker output per video and tune each class's 2-4 thresholds against your own dev labels with evaluate.py; per-class F1 is independent, so tune class by class and prefer plateaus.
- Make boundaries geometric and sub-frame: front/rear footprint model for stop-line and crossing events, signed-distance crossings for solid lines, look-back to the stop moment for stopped_vehicle/congestion, onset back-projection (1 m/s^2) for near_miss.
- Add offline track stitching (gap <= 4 s, or <= 30 s for stationary objects) and an ID-switch 'suspect' guard; most wrong_way/U-turn/accident false positives come from ID switches.
- Restrict kinematic rules (braking, near_miss, accident shocks) to the zone where ppm >= 8 and use SG windows >= 1.5 s for acceleration; far-field acceleration noise exceeds the 3.4 m/s^2 threshold.
- Gate classes with the rule: enable if p*F/k > (1-p)*q*S/(k+1); for rare classes (fire_smoke, road_obstacle, illegal_u_turn) design for q≈0 (multi-signal corroboration, zero FPs on all sample minutes).
- Enable by default: stopped_vehicle, wrong_way, jaywalking, accident; enable conditionally on scene: congestion, solid_line_crossing, failure_to_yield, red_light and stop_line (need signal info), near_miss at high precision; illegal_turn/illegal_u_turn only if restrictions are visible.
- Always union overlapping same-class segments yourself and learn a per-class start/end bias from dev matches; the harness would otherwise drop later overlapping segments.
- Ask the organizers the listed ambiguity questions (crash aftermath as stopped_vehicle, stop_line vs red_light, U-turn across solid line, right-turn-on-red sign) before finalising rules.


## Open questions
- Scene specifics are unknown until samples/camera.md arrive: is the signal head visible, are there zebra crossings, solid lines, turn restrictions, U-turn prohibitions in view? These decide which classes to enable.
- Do annotators label crashed vehicles standing >= 10 s as stopped_vehicle in addition to accident, and drivers on the road after a crash as jaywalking?
- Is a vehicle that crosses the stop line on red and stops before the box labelled stop_line only, and what if it then proceeds on red?
- Are U-turns/turns across a solid line double-labelled as solid_line_crossing?
- Is failure_to_yield labelled when pedestrians cross against their red at a signalized crossing?
- What is the minimum annotated duration for congestion, and is a long single-phase red queue ever labelled congestion?
- Is a right turn on red under sign 5.42 labelled red_light?
- Actual detector bbox jitter on this camera (the SG noise analysis assumed 1.5 px white noise) and the real pixels-per-metre range need to be measured on samples.
- Lane width and marking dimensions in Uzbek (O'zDSt) standards were not verified; measure from the scene or rely on GOST-style dash modules.


## Key claims (as submitted for verification)
- FHWA SSAM uses default screening thresholds TTC <= 1.5 s and PET <= 5 s for conflicts — https://highways.dot.gov/sites/fhwa.dot.gov/files/FHWA-HRT-08-049.pdf
- A DRAC value above 3.35 m/s^2 is a commonly used traffic-conflict threshold (Archer 2005 / FHWA) — https://www.sciencedirect.com/science/article/abs/pii/S0001457521000828
- AASHTO uses a deceleration of 3.4 m/s^2 (11.2 ft/s^2) for stopping sight distance; about 90% of drivers brake harder in emergencies — https://en.wikipedia.org/wiki/Stopping_sight_distance
- Ultralytics ByteTrack defaults: track_high_thresh 0.25, track_low_thresh 0.1, new_track_thresh 0.25, track_buffer 30 frames, match_thresh 0.8 — https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/bytetrack.yaml
- OpenCV 4.13 createBackgroundSubtractorMOG2 defaults: history=500, varThreshold=16, detectShadows=True — reasoning (verified locally with OpenCV 4.13.0 in scratchpad)
- The 1-euro filter is a first-order low-pass filter with adaptive cutoff f_c = f_cmin + beta*|speed| (Casiez, Roussel, Vogel, CHI 2012) — https://gery.casiez.net/publications/CHI2012-casiez.pdf
- A first-order Savitzky-Golay filter with a ~1 s window is used to smooth video-derived NGSIM trajectories and compute velocities — https://github.com/Rim-El-Ballouli/NGSIM-US-101-trajectory-dataset-smoothing
- Simulated with 1.5 px white jitter: SG(1.0 s) acceleration noise sigma is ~0.4-0.5 m/s^2 at 20 px/m but ~2-2.7 m/s^2 at 4 px/m, so braking thresholds are usable only in near/mid field — reasoning (scratchpad simulation sg_noise.py; jitter magnitude assumed)
- Temporal IoU >= 0.7 on a 5 s event tolerates at most ~0.88 s pure shift; a 20 s stop reported 10 s late gets IoU 0.5 — spec (metric definition) + reasoning (iou_tol.py)
- Predicting a class absent from the test set multiplies Score_A by k/(k+1), where k = number of classes present — spec (Score_A definition: predicted absent class is added to C with F1 0)
- Porikli's dual-background method flags static objects as pixels that are foreground in a long-term model but absorbed into a short-term model — https://www.researchgate.net/publication/26512796_Robust_Abandoned_Object_Detection_Using_Dual_Foregrounds
- The D-Fire dataset has >21,000 images (fire and smoke boxes, YOLO format) and is released under CC0 1.0 — https://github.com/gaiasd/DFireDataset
- Uzbekistan acceded to the Vienna Convention on Road Signs and Signals in 1995; sign 3.19 'U-turn prohibited' and 5.42 'turn right on red' exist — https://en.wikipedia.org/wiki/Road_signs_in_Uzbekistan
- In Tashkent, right turn on red is permitted only at intersections with a special sign ('yield to all, then right'), not generally — https://www.gazeta.uz/ru/2018/09/27/right-turn/
- Uzbekistan SDA: drivers must stop before the stop line with the vehicle's front not beyond it; yellow prohibits movement (with exceptions) — https://uzpdd.uz/pddtext.php?id=7
- HCM default start-up lost time is 2.0 s at signalized intersections — https://mctrans.ce.ufl.edu/calibrating-driver-behavior-at-signalized-intersections/
- GOST R 51256 line 1.5 has dashes 1-3 m with gaps three times the dash; line 1.6 approach line has dashes three times the gap — http://docs.cntd.ru/document/1200158480
- Roboflow supervision speed estimation uses cv2.getPerspectiveTransform between a source image polygon and a metric target rectangle — https://blog.roboflow.com/estimate-speed-computer-vision/
