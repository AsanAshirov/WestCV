<!-- source: deep-research workflow wf_294a4b01-fc2, agent gap:3 -->

# Dimension 3: Per-video camera registration, consumer-camera artifacts, and what the clip numbers say about the hidden test set

## 0. Bottom line

1. **The premise needs correcting.** The samples do not come from one midday session. They come from two sessions on 2026-09-18 (camera clock, +06:00):
   - C3896 and C3897 were recorded at 12:18–12:30.
   - C3902 and C3905 were recorded at 17:58–18:24.
   
   The camera is in Rec-Run timecode, so the timecode gaps give the exact amount of footage recorded under the missing clip numbers:
   - C3898–C3901 together hold **370.4 s**, in 4 clips (about 93 s each on average).
   - C3903–C3904 together hold **652.7 s**, in 2 clips.
   - The total is **1023 s, so T ≈ 0.28 h**, not 0.55 h.
2. The sessions are 5.5 h apart, so there were probably **two tripod set-ups** (midday and evening). Registration between the two sessions is therefore the expected case. C3903 and C3904 fall between C3902 and C3905, which bracket them within 19 minutes, so they probably share the evening framing.
3. **Lighting matters more than night.** If the camera clock shows true Tashkent time, the evening clips were shot at a sun elevation of 4.7° down to 0° (sunset is about 18:28). About 64% of the likely test minutes (C3903 and C3904) would then be low-sun or sunset footage. Prioritise that. De-prioritise true night, IR and rain unless the organisers confirm clips from C3906 onward.
4. **I did not measure real pixel offsets.** Google Drive returned "Quota exceeded" pages for any range of 32 KB or more on C3902 and C3905. I stopped at the metadata tails and did not work around the quota.
   - The measurement tool is written and smoke-tested. Run it once the files are local; the command is in §3.
   - The registration module was validated on synthetic 4K scenes. It gets ≤1.3 px (4K) error for moves of up to 134 px, a 3% zoom, dusk gain, moved shadows and 60% static congestion. It correctly rejects a different scene. Cost is about 0.8–1.1 s per video on the dev laptop while other jobs were loading it.
5. **The format differs from the spec.** The samples are 3840×2160, **29.97p drop-frame**, H.264 High 4:2:2 10-bit ("AVC140_3840_2160_H422P@L51"), with rec709 gamma. They are not 25 fps. All four files total 20.27 GB. Download them early; the Drive quota is already biting.

---

## 1. What the metadata proves

Source: the Sony NonRealTimeMeta XML at the end of each file. C3896 and C3897 were fetched as 64 KB tails; C3902 and C3905 as 4 KB tails, which is all Drive allowed. I decoded the LTC values as BCD `FF SS MM HH` with bit 0x40 marking drop-frame. For every clip, the decoded timecode span matches the frame count exactly.

| Clip | Frames | Duration | Timecode (DF) | Camera clock | Sun elev.* |
|---|---|---|---|---|---|
| C3896 | 10200 | 340.3 s | 11:03:06;18 → 11:08:46;27 | 12:18:24–12:24:05 | 50.5° |
| C3897 | 9525 | 317.8 s | 11:08:46;28 → 11:14:04;22 | 12:24:47–12:30:05 | 50.4° |
| C3902 | 9525 | 317.8 s | 11:20:15;03 → 11:25:32;27 | 17:58:18–18:03:36 | 4.7°→3.6° |
| C3905 | 3825 | 127.6 s | 11:36:25;18 → 11:38:33;06 | 18:22:21–18:24:29 | 0.2°→−0.1° |

\*Computed with the NOAA solar equations for Tashkent (41.30 N, 69.24 E, UTC+5), assuming the displayed clock is local time. Sunset is 18:28 and civil dusk 18:56. If the true instant is instead UTC+6, subtract one hour; the evening elevations then become 11–16°, which is still low sun.

- **Rec-Run is proven.** C3897 starts on the frame right after C3896's last frame, even though 42 s of wall-clock time passed between them. The timecode therefore advances only while recording. Deleted clips would still have consumed timecode, so the gaps measure everything that was recorded.
- **Camera settings:** Sony ILCE-6700; XAVC S 4K 4:2:2 10-bit; about 147 Mbit/s including LPCM audio; `CaptureGammaEquation=rec709` with rec709 primaries.
  - This is a standard picture profile, not S-Log or HLG, so COCO-pretrained detectors need no LUT.
  - A KLV "_RecStart" marker sits at frame 0.
- **Per-frame metadata tracks are present:** ImagerControlInformation, LensControlInformation, DistortionCorrection, Gyroscope and Accelerometer.
  - The XML does *not* contain the stabilisation mode, focal length, zoom or exposure mode. Those values live in the per-frame `rtmd` timed-metadata track.
  - ExifTool lists `rtmd` among its QuickTime Stream tags, which it extracts with ExtractEmbedded. Once the files are local, run `exiftool -ee -G3 -a -u C3896.MP4 > C3896_rtmd.txt` and look for per-frame exposure time, ISO/gain, F-number, white balance and gyro values.
  - Gyroflow lists the a6700 as recording gyro data.
  - Use gyro data for EDA only: a "camera stability" card is good website material. Do not depend on it at inference, because the test files may be re-muxed.
- **A curiosity with low weight:** C3897 and C3902 are both exactly 9525 frames and exactly 5,838,719,827 bytes, yet they have different UMIDs and dates. This could be fixed-length stops such as an app timer. If so, the gaps could hide one 9525-frame clip plus short fragments: 11100 = 9525 + 1575 frames.

## 2. What the clip numbers say about the hidden test set

The spec says the test set is "other videos from the same camera and angle". On this card, the candidates are:

- **C3898–C3901:** 6.2 min in total, recorded somewhere between 12:30 and 17:58. That is 4 short clips of about 1.5 min on average. Short clips suggest opportunistic or staged captures of events, which fits the spec's "test videos contain events not in the samples".
- **C3903–C3904:** 10.9 min in total, recorded between 18:04 and 18:22. That is 2 clips of about 5.4 min each, at sun elevation 3.6° down to 0.2° if the clock is local time.
- **Unknown:** clips from C3906 onward (after 18:24:29, so after sunset if the clock is local), or clips from other days.

Planning assumptions:
- Baseline test length T = 0.284 h across about 6 videos.
- The test is probably in the same two lighting regimes as the samples.
- There are probably ≥2 tripod set-ups.
- Dusk footage is a real possibility; night only if the organisers confirm it.

## 3. Measured offsets: status and how to get them

**Real offsets are not measured yet.** Once the MP4s are local, run:

```
python measure_alignment.py --videos samples --ref C3896.MP4 --out reg_out \
       --plate-every 2 [--lamp x0,y0,x1,y1] [--mask stable_mask_960.png]
```

For each video it does one sequential decode pass, calling `grab()` on every frame and `retrieve()` only on sampled frames. It writes:
- a median plate (960×540 grey);
- `luma_*.csv`: 1 Hz mean, p5 and p95 luma, plus lamp-ROI B, G, R and max V;
- `jitter_*.csv`: 1 Hz phase-correlation shift against the video's own plate, in 4K px;
- `summary.json`: the reference→video homography at 4K, max displacement, ECC, inliers, jitter median/p95/max, first-2 s and last-2 s max (the REC-button nudge), a first-vs-last-third drift step, the largest 1-s luma jump, and the luma drift.

Read the results with these rules:

| Measured | Action |
|---|---|
| Inter-video displacement < 4 px (4K) | Identity. One scene.yaml is enough. |
| 4–240 px, gates pass | Warp scene.yaml per video. This is the expected result between midday and evening. |
| > 240 px or zoom > 6% | Accept only if inliers ≥ 150 and ECC ≥ 0.90. Otherwise draw a second scene.yaml for that session. |
| Intra-video p95 jitter > 6 px | Per-frame translation compensation of track coordinates (phase correlation every 5th frame at 480×270, interpolated). |
| Step between thirds > 8 px | Piecewise H(t): least-squares change-point split, then one plate and one registration per segment. |
| First or last 2 s jitter > 8 px | Mask the first and last 1.5 s from line-crossing rules (REC press or stop). |
| Luma jump > 10% in 1 s | Re-baseline the background model and lamp statistics at that time. |

The smoke test on synthetic clips shows the tool works:
- A 13.4 px mid-clip bump at t = 15 s was found at 15.0 s, with size 13.5 px.
- A 10.7 px REC-press shake was flagged in the first 2 s.
- A 20% exposure step was reported as a 19.7% jump.
- A plain whole-clip registration of the bumped clip correctly **refused** (the feature estimate and the ECC estimate disagreed by 13.9 px), and per-segment registration then gave 0.1–0.2 px error.

## 4. The `register_scene(video_path)` spec

The prototype is `register_scene.py`.

**Reference assets** go in `weights/scene/` as scene facts (allowed):
- `ref_plate_960.png`: median plate of the video that scene.yaml was drawn on;
- `ref_mask_960.png`: 255 on stable structure (lane markings, curbs, building edges, poles), 0 on sky, trees and the sun-glare area;
- optionally, one plate per sample for multi-reference matching.

**Convention.** H maps reference pixels to current-video pixels at the 960×540 working scale. At 4K, `H4k = S·H·S⁻¹` with `S = diag(4,4,1)`.

**Algorithm (Part A, may use the whole video):**
1. **Plate.** During the main detection pass, keep one downscaled grey frame every ~6.7 s (48 frames for a 5.3-min clip) and take the median. This costs about 0.35 s and needs no extra decoding. Using 160 frames (one every 2 s) costs about 1.1–1.5 s and is worth it only for EDA.
2. **Normalise.** Apply CLAHE (clip 2.0, 8×8 tiles) to both plates. This absorbs most auto-exposure and white-balance differences.
3. **Features.**
   - Detect and match: AKAZE (threshold 5e-4) within the mask, BF-Hamming matching, ratio test 0.8.
   - Fit: `findHomography` with RANSAC, threshold 2.0 work-px, 5000 iterations, confidence 0.999.
4. **Refine.** `findTransformECC(MOTION_HOMOGRAPHY)` seeded with the feature homography. Run 80 iterations at 480×270, then 40 at 960×540, with the mask and a Gaussian filter of 5.
5. **Hard gates (all must pass):**
   - inliers ≥ 40;
   - inlier ratio ≥ 0.25;
   - RANSAC RMSE ≤ 1.5 work-px (6 px at 4K);
   - ECC ≥ 0.60;
   - feature and ECC estimates agree within 12 px (4K) at the corners and key points.
6. **Soft gates (can be overridden when inliers ≥ 150 and ECC ≥ 0.90):**
   - displacement ≤ 240 px (4K);
   - |√det A − 1| ≤ 0.06.
7. **Decision:**
   - If displacement < 4 px, use identity, to avoid injecting noise.
   - Otherwise warp.
   - Finally run the step detector on 0.2 Hz phase correlation. If a step is found, compute a piecewise H(t).

**Warping each scene.yaml element:**
- Polygons and polylines (lanes, carriageway, crossings, stop lines, solid lines, no-U-turn zones): `cv2.perspectiveTransform(pts, H4k)`.
- Lane direction vectors: transform two points along each lane.
- Raster masks: `warpPerspective` with nearest-neighbour interpolation.
- Ground-plane homography (image→metres): `G_cur = G_ref · H4k⁻¹`.
- **Lamp ROIs:** warp the corners, then do a local NCC template match of the reference lamp-housing crop within ±40 px (4K). Lamps are the most fragile element: by estimate a 30 cm lens at about 60 m is only about 20 px wide at 4K.

**Tuning.** Recalibrate the thresholds on the real plates. Set `MIN_ECC` to the minimum ECC over true sample pairs minus 0.1.

**Synthetic validation** (`synth_test.py`, 4K procedural intersection, 160 frames each; errors are max over 10 scene key points including the lamp):

| Case | True displacement (4K) | Status | Error (4K px) | ECC | Inliers |
|---|---|---|---|---|---|
| Same tripod | 0 | identity | 0.0 | 1.000 | 1215 |
| Nudge | 7.8 | warped | 0.3 | 0.999 | 790 |
| Bump 40 px + 0.3° | 43.9 | warped | 0.3 | 0.998 | 828 |
| Zoom 3% + 100 px | 134 | warped | 0.3 | 0.998 | 621 |
| Dusk: gain 0.45, γ 1.3 | 30.4 | warped | 0.4 | 0.973 | 658 |
| 60% static congestion | 30.4 | warped | 0.9 | 0.815 | 618 |
| Moved shadows, gain 0.6 | 30.4 | warped | 1.1 | 0.879 | 636 |
| Gain 0.25, noise 6, shadows | 41.9 | warped | 1.3 | 0.919 | 463 |
| Wrong scene | n/a | **fallback** (ratio 0.18, disagreement 105 px) | n/a | 0.653 | 55 |

The 500 px + 10% zoom case was refused under the original hard scale gate (ECC 0.997, 346 inliers). I then changed the rule so strong evidence can override it; I did not re-run that case.

**Cost, measured on the 12-thread dev laptop while other jobs were loading it** (eval-box numbers will differ):
- 48-frame median: 0.33–0.36 s;
- registration: 0.46–0.77 s;
- phase correlation at 480×270: about 6 ms per call;
- 4K BGR → 960×540 grey: 5.6–12 ms per frame.

**Per video: about 1–1.5 s**, which is under 0.5% of the 3×-duration budget (about 950 s for a 317.8 s clip).

## 5. Causal variant for RiskEstimator (`CausalRegistrar`)

This is legal: it uses only frames the estimator has received, plus reference plates made from the samples, which are scene facts. **Do not pass Part A's H to it.** Part A's H was computed from future frames, and the harness creates a fresh `RiskEstimator()` for every video anyway.

- Until the registration is ready, use identity and down-weight geometry-dependent risk terms (stop line, lamp).
- Take one frame every 0.4 s (every 12 frames at 29.97 fps).
- Once n_init frames are collected, take their median and call `register_plate`, using the same gates as Part A.
- Every 30 s, check for drift with phase correlation of one frame against the reference plate warped into current coordinates. If the shift is more than 8 px (4K), re-register.

Synthetic results for a 40 px bump:

| n_init | Ready after | Error (4K px), normal / congestion | Cost |
|---|---|---|---|
| 1 | 0 s | 1.3 / 2.3 | 2.3–3.2 s |
| **4 (recommended)** | **1.2 s** | **1.0 / 0.9** | **0.9–1.1 s** |
| 8 | 2.8 s | 1.0 / 1.0 | 0.5–1.6 s |

With n_init = 1, ECC on a single frame containing vehicles converges slowly.

Total Part B overhead is about 1 s per video, but it lands mostly in a single `step()` call. That is fine offline.

## 6. Fallback policy

Per-class misalignment tolerance, in 4K px. These are estimates: a lane of 3.5 m spans about 150–300 px in the near field and about 50 px in the far field.

| Element or class | Tolerance | If registration fails |
|---|---|---|
| Lamp ROI (red_light, stop_line) | 5–8 | **Disable** red_light and stop_line |
| Solid line (solid_line_crossing) | 8–10 | **Disable** |
| Stop-line position (red_light start, stop_line) | 10–15 | Disable (with the lamp-ROI classes above) |
| Crossing polygon (failure_to_yield) | 20–30 | **Disable** failure_to_yield. Jaywalking uses the self-derived road mask below. |
| Lane polygons (illegal_turn, illegal_u_turn) | 30–50 | **Disable** |
| wrong_way | Needs directions, not exact lines | Keep, but switch to a **self-calibrated flow field**: majority direction per 64-px cell from this video's own tracks (Part A may use the whole video) |
| Carriageway (stopped_vehicle, congestion, road_obstacle, jaywalking) | ~50 | Keep, using a road mask from the video's own motion heatmap |
| accident, near_miss, fire_smoke | Geometry-free | Keep |

Refinement to the fallback: first compute the identity ECC between the reference and current plates. If it is ≥ 0.90, the camera has not moved, so keep everything. Disabling fragile classes on failure is the right trade-off: a class you predict that never occurs adds a zero-F1 class to C.

## 7. Consumer-camera artifacts to plan for

- **REC-button nudges.** The operator touches the camera to start and stop every clip. Check the first and last 2 s of jitter.
- **IBIS or Active SteadyShot on a tripod.** This can cause slow drift or per-frame micro-warps; jitter p95 and the step test quantify it.
- **DistortionCorrection is on.** Its residual is second-order for moves of tens of px.
- **Auto exposure and auto white balance at sunset.** Expect a continuous ISO ramp and warmer colour. So:
  - classify lamps by *relative* measures: lamp max-V over housing median, and the hue of saturated pixels;
  - learn the lamp's per-video hue clusters in Part A; keep running estimates in Part B;
  - do not use fixed absolute thresholds;
  - raise the background-model learning rate after luma jumps of more than 10% per second;
  - at low sun, handle long vehicle shadows (a problem for background subtraction), glare, and headlights.

## 8. Questions for the hackathon channel (ready to paste)

1. "Were the hidden test videos recorded in the same sessions and tripod placement as the samples (18 Sep; samples C3896/C3897 midday, C3902/C3905 evening)? Or on other days, or with the camera re-mounted or re-zoomed?"
2. "Roughly how many test videos are there, and what is their total duration? Are they full camera clips or cut segments?"
3. "Does the test set include footage after sunset (night, street lights only) or in rain?"
4. "Will the test files be the original camera MP4s (3840×2160, 29.97 fps, XAVC S 4:2:2 10-bit) like the samples, or re-encoded? The task text mentions 25 fps."

## 9. Updated planning numbers

**Class gating.** Probability that a class appears at least once in a test of length T, `q = 1 − e^(−λT)`:

| λ (events/h) | T = 0.284 h | T = 0.55 h |
|---|---|---|
| 1 | 0.25 | 0.42 |
| 2 | 0.43 | 0.67 |
| 3 | 0.57 | 0.81 |
| 5 | 0.76 | 0.94 |
| 10 | 0.94 | 1.00 |

Posterior-predictive q given n events in your own labels of the 18.4 min of samples (Jeffreys prior):

| n | T = 0.284 h |
|---|---|
| 0 | 0.28 |
| 1 | 0.63 |
| 2 | 0.81 |
| 3 | 0.90 |

For n = 0, raise q for the classes the organisers probably added, given their "test contains events not in samples" statement.

**Decision rule.** Enable a class if `q > q* = pS/(f + pS)`, where:
- S = mean F1 of the other classes;
- f = expected F1 on the class if it is present;
- p = 1 − e^(−r_fp·T), with r_fp the false-positive rate in events per hour.

Example: with S = 0.4, f = 0.3 and r_fp = 2/h, q* = 0.37, so enable when λ ≥ 1.6/h. At T = 0.55 h the threshold was λ ≥ 1.2/h. The shorter test raises the bar only slightly. The false-positive rate of the rule is the real lever: at r_fp = 0.5/h, λ* is 0.6/h.

**Part B.**
- `evaluate.py` sets M = Score_A when the ground truth has no accidents. In 17 minutes at one intersection, an accident is unlikely unless it was staged.
- If there is exactly 1 accident, F1_alarm drops from 1.0 to 0.667 with one false alarm and to 0.5 with two.
- The false-alarm budget for at most 1 false alarm is **≤ 3.5/h** at T = 0.284 h (it was 1.8/h at 0.55 h). Aim for ≤ 1.8/h so the expected count stays ≤ 0.5.
- Each accident gives about 150 positive frames against about 30,700 test frames, a positive rate of about 0.49%.
- Keep Part B cheap and conservative, for example TTC-based with a high θ-calibration.

**De-prioritise:**
- IR or night-vision handling: this is a mirrorless camera with no IR illuminator.
- Rain and wet-road handling: everything was recorded on the same day as the samples.
- Street-light-only night detection: only needed if clips from C3906 onward are confirmed.
- Fog and snow.
- Generalising to other viewpoints.

**Promote:**
- Low-sun and sunset robustness: shadows, glare, headlights, auto-exposure and white-balance ramps.
- Per-video registration.
- Multi-reference plates (midday and evening).
- Relative lamp classification.

## 10. Files (scratchpad)

All are in `C:\Users\Cicada\AppData\Local\Temp\claude\C--Users-Cicada-Desktop-WestHack\ced7e0d9-be88-47bc-ac13-90898f4ae082\scratchpad\reg\`:
- `register_scene.py`: registration module and `CausalRegistrar`
- `measure_alignment.py`: the audit tool from §3
- `synth_test.py`: synthetic validation
- `tc_gaps.py`: timecode decoder and gap calculation
- `gating.py`: gating and Part B budget tables
- `sun.py`: sun elevation and sunset calculation
- `tail_C3896.bin`, `tail_C3897.bin`, `tail4k_C3902.bin`, `tail4k_C3905.bin`: raw XML tails



## Key claims (as submitted for verification)
- All four sample clips are Sony ILCE-6700 files: XAVC S 3840x2160, 29.97p (drop-frame TC), codec AVC140_3840_2160_H422P@L51 (H.264 4:2:2), CaptureGammaEquation=rec709, with Gyroscope, Accelerometer, DistortionCorrection, Lens and Imager per-frame tables flagged — NonRealTimeMeta XML read from the end of C3896/C3897/C3902/C3905.MP4 (Google Drive links in C:\Users\Cicada\Desktop\WestHack\Videos.pdf); saved as scratchpad\reg\tail*.bin
- Recording times (camera clock +06:00): C3896 12:18:24-12:24:05, C3897 12:24:47-12:30:05, C3902 17:58:18-18:03:36, C3905 18:22:21-18:24:29 on 2026-09-18 — NonRealTimeMeta CreationDate/lastUpdate in the file tails
- Clip lengths are 10200, 9525, 9525 and 3825 frames (340.3, 317.8, 317.8, 127.6 s), and each decoded DF timecode span matches the frame count exactly — NonRealTimeMeta Duration and LtcChangeTable; decoded by scratchpad\reg\tc_gaps.py
- The camera timecode is Rec-Run: C3897 starts on the frame after C3896's last TC frame (11:08:46;27 -> ;28) despite a 42 s wall-clock gap — LtcChangeTable values 67460811 / 68460811 plus CreationDate/lastUpdate (reasoning)
- Footage recorded in the clip-number gaps: C3898-C3901 = 11100 frames = 370.4 s; C3903-C3904 = 19560 frames = 652.7 s; total about 1023 s (0.284 h) — reasoning: drop-frame TC arithmetic in scratchpad\reg\tc_gaps.py on the metadata tails
- Sunset in Tashkent on 2026-09-18 is about 18:28 (UTC+5); sun elevation is about 4.7 deg at 17:58 and 0.2 deg at 18:22 local time — reasoning: NOAA solar-position equations in scratchpad\reg\sun.py (timeanddate.com returned HTTP 403)
- Gyroflow lists the Sony a6700 as supported with gyro data and an automatic lens profile, and says footage can be recorded with IBIS enabled (Standard mode recommended; Dynamic Active not supported) — https://docs.gyroflow.xyz/app/getting-started/supported-cameras/sony
- ExifTool lists Sony 'rtmd' as a QuickTime Stream (timed metadata) tag; stream tags are extracted when the ExtractEmbedded (-ee) option is used — https://exiftool.org/TagNames/QuickTime.html
- evaluate.py sets model score M = Score_A when the ground truth contains no accidents (Part B not scored) — starter kit evaluate.py (docstring line 36 and evaluate_part_b returns None when n_accidents == 0)
- run_submission.py creates a fresh RiskEstimator() per video and passes t = idx/fps with fps from CAP_PROP_FPS; the time budget is 3x duration for Part A + B — starter kit run_submission.py (run_risk / main)
- On synthetic 4K scenes the prototype registration (CLAHE + AKAZE/RANSAC + ECC homography) recovers moves up to 134 px and 3% zoom with <=1.3 px 4K error under dusk gain, moved shadows and 60% static congestion, and rejects a different scene — reasoning/measurement: scratchpad\reg\synth_test.py output
- Registration cost is about 0.33-0.36 s for a 48-frame median plate plus 0.46-0.77 s for registration at 960x540 on the dev laptop while other jobs were loading it; 4K-to-960 grey resize is 5.6-12 ms per frame — measured locally (scratchpad\reg benchmarks)
- Causal registration with 4 frames spaced 0.4 s is ready after 1.2 s with about 1 px 4K error in synthetic tests — measured: synth_test.py CausalRegistrar runs
- With T = 0.284 h, q = 1 - exp(-lambda*T) is 0.25/0.43/0.57/0.76 for lambda = 1/2/3/5 per hour; the Part B false-alarm budget for at most 1 false alarm is about 3.5 per hour — reasoning: scratchpad\reg\gating.py
- The four sample files total 20,267,812,894 bytes (6.24 + 5.84 + 5.84 + 2.35 GB); Drive returned 'Quota exceeded' HTML for ranges of 32 KB or more on C3902/C3905 at the time of writing — Google Drive Content-Range headers and responses observed on 2026-09-25
