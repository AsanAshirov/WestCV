<!-- source: deep-research workflow wf_294a4b01-fc2, agent research:stack -->

# Perception stack: detector, tracker, traffic-light state, T4 throughput and licences

*Research date: 2026-09-25. Numbers taken from vendor pages are quoted with their links. Anything marked **(est.)** is my own estimate, and the reasoning is given next to it. Anything marked **(measured locally)** comes from benchmarks I ran on the team's laptop (i5-12450H, GTX 1650) using synthetic H.264 clips. Those local runs happened while the CPU was already about 90% busy with other work, so treat them as order-of-magnitude figures, not exact ones.*

---

## 0. Recommendation in one table

| Layer | Primary choice | Backup | Key reason |
|---|---|---|---|
| Detector, Part A (offline, accuracy) | **Ultralytics YOLO26m**. Fine-tune it on pseudo-labelled sample frames. Run it at **1280×736 rectangular** input, **PyTorch FP16**, at stride 1–2. | **RF-DETR-S/M** (Apache-2.0) | Best tooling (train, track, seg, export, SAHI) in one package. NMS-free output. Small-object label assignment (STAL). 53.1 COCO AP at 4.7 ms on T4. |
| Detector, Part B (causal, cheap) | **YOLO26s at 960×544**, stride 2–3. Use YOLO26n at 640 if time is short. | RF-DETR-N | Part B runs at batch size 1 inside `step()`, so it needs a small model. |
| Tracker | **ByteTrack** without ReID and without GMC. Use the Ultralytics built-in or Roboflow `trackers` (Apache-2.0). Part A adds **offline tracklet stitching and smoothing**. | OC-SORT (same libraries) | The camera is fixed, and IoU plus a Kalman filter works well at 8–12 fps. Part A is offline, so we can repair broken IDs using future frames. |
| Traffic-light state | **Hand-set per-lamp ROIs** taken from camera.md. Brightness and hue features, **calibrated per video**. **Temporal max-pooling** against LED flicker. **HMM/Viterbi** decoding offline in Part A, a causal filter in Part B. | A tiny CNN on 32×96 crops | The camera is fixed, so the ROI never moves. A CNN is only worth it if glare or night footage defeats HSV. |
| Signal not visible | Infer the phase from **queue and flow at each approach's stop line**. Offline, also use cycle periodicity. | — | See §4.4. |
| Decoding | **OpenCV `grab()` for every frame, `retrieve()` only for frames we process.** Use PyAV for accurate seeks. | NVDEC (torchcodec 0.16 / PyNvVideoCodec), only if the video is 4K | The BGR conversion costs more than the decoding itself (§5.1). |
| Inference backend | **PyTorch FP16** with `cudnn.benchmark=False`. TensorRT is **not** in the default path. | CPU fallback: YOLO26n through ONNX Runtime CPU at stride 5 | Portable across "T4-class" GPUs, deterministic, and fast enough (§5–6). |
| Segmentation | **Not in the main loop.** Optionally run YOLO26s-seg inside stride-1 refinement windows. | RF-DETR-Seg (Apache-2.0) | Adds 30–40% GPU time. It does little for "contact" detection, which is a kinematics question (§7). |
| Extra classes | Fire/smoke: YOLO26n **fine-tuned on D-Fire (CC0)**, run at 1–2 fps. Obstacles: a **static-foreground background model** plus **YOLOE-26 with prompts baked in**, run at 0.5–1 fps inside the road ROI. Animals come from the COCO classes. | — | COCO has no class for debris, smoke or fire. |

Package versions at the time of writing:
- `ultralytics` 8.4.162 (PyPI, 2026-09-24).
- `rfdetr` 1.11.0 (2026-09-24).
- `torchcodec` 0.16.0 (2026-08-13, needs torch ≥ 2.11).
- `av` (PyAV) 18.1.0 (2026-08-12, **needs Python ≥ 3.11**).
- `decord` 0.6.0. Its last release was **2021-06-14**; it is unmaintained, so do not use it.
- `onnxruntime-gpu` ≥ 1.29 is built for CUDA 13 / cuDNN 9.

Pin exact versions once they work on a Kaggle T4.

---

## 1. Real-time detectors for a fixed CCTV view (2025–2026)

### 1.1 Comparison (COCO val AP50:95; latency on T4, TensorRT FP16, batch 1, from vendor tables)

| Family (size) | COCO AP | T4 TRT FP16 latency | Input | Code/weights licence | Notes |
|---|---|---|---|---|---|
| **YOLO26** n / s / m / l / x | 40.9 / 48.6 / 53.1 / 55.0 / 57.5 | 1.7 / 2.5 / 4.7 / 6.2 / 11.8 ms | 640 | **AGPL-3.0** or Enterprise | Released Jan 2026. NMS-free end-to-end (`nms=False`, up to 300 detections). DFL removed. ProgLoss and STAL (small-target label assignment). Same scales for seg, pose, OBB and YOLOE-26. [docs](https://docs.ultralytics.com/models/yolo26), [arXiv 2606.03748](https://arxiv.org/abs/2606.03748) |
| YOLO11 n / s / m / l / x | 39.5 / 47.0 / 51.5 / 53.4 / 54.7 | 1.5 / 2.5 / 4.7 / 6.2 / 11.3 ms | 640 | AGPL-3.0 | Same speed as YOLO26 with about 1.5–2.8 AP less; YOLO26 replaces it. [docs](https://docs.ultralytics.com/models/yolo11/) |
| YOLO12 s / m | ~48.0 / ~51.9 | ~2.6 / ~5.7 ms | 640 | AGPL-3.0 | Attention-based. No advantage over YOLO26. |
| YOLOv10 | — | — | 640 | AGPL-3.0 | Superseded by YOLO26's end-to-end head. [licence](https://github.com/THU-MIG/yolov10/blob/main/LICENSE) |
| **RF-DETR** N / S / M / L | 48.4 / 53.0 / 54.7 / 56.5 | 2.3 / 3.5 / 4.4 / 6.8 ms | 384 / 512 / 576 / 704 | **Apache-2.0** (the XL and 2XL sizes are PML 1.0) | DINOv2 backbone, ICLR 2026. Seg variants are Apache-2.0 at every size. [repo](https://github.com/roboflow/rf-detr) |
| D-FINE N / S / M / L / X | 42.8 / 48.5 / 52.3 / 54.0 / 55.8 | 2.12 / 3.49 / 5.62 / 8.07 / 12.89 ms | 640 | Apache-2.0 | The Objects365-pretrained checkpoints may carry Objects365 terms. [repo](https://github.com/Peterande/D-FINE) |
| RT-DETRv4 S / M / L / X | 49.8 / 53.7 / 55.4 / 57.0 | 3.66 / 5.91 / 8.07 / 12.90 ms | 640 | Apache-2.0 | The vision foundation model is only used during training. [repo](https://github.com/RT-DETRs/RT-DETRv4) |
| DEIMv2 N / S / M / L / X | 43.0 / 50.9 / 53.0 / 56.0 / 57.8 | 2.32 / 5.78 / 8.80 / 10.47 / 13.75 ms | 640 | **Custom "DEIMv2 License"** (commercial use by inquiry) plus the DINOv3 licence | Avoid: the licence is ambiguous for a public repo. [repo](https://github.com/Intellindust-AI-Lab/DEIMv2) |
| YOLOX | older generation, lower AP | — | 640 | Apache-2.0 | Only worth considering for the licence. |

These latencies are not strictly comparable. RF-DETR reports "total latency" including its NMS, at its own native resolution (384–704 px). Ultralytics reports pure model latency; YOLO26 has no NMS anyway. Independent reproductions are close: one user measured YOLO26n at 1.85 ms on a T4 with TensorRT 10, against the documented 1.7 ms ([issue #23689](https://github.com/ultralytics/ultralytics/issues/23689)).

### 1.2 What the licences mean for this hackathon
- **AGPL-3.0 (Ultralytics, YOLOv10/12, BoxMOT) is not a blocker.** The repo must be public anyway. Put an AGPL-3.0 `LICENSE` in the repo and credit Ultralytics in the README. Ultralytics treats fine-tuned weights as covered by the AGPL, which is fine here. The live-demo website runs the model as a network service, so AGPL §13 requires offering users the source. A visible link to the repo satisfies this, and the site must include that link anyway (section 7, "Links").
- Apache-2.0 and MIT code (RF-DETR, `trackers`, `supervision`, `SAHI`) can be combined into an AGPL repo without problems.
- Avoid DEIMv2 (custom licence) and RF-DETR XL/2XL (PML 1.0), unless the team reads those licences and is comfortable with them.
- Non-commercial datasets (§2.3) are acceptable for a university hackathon. They must be listed in the README.

### 1.3 Recommendation
- **Primary: YOLO26.** Use **m** in Part A and **s** (or n) in Part B.
  - Reasons: one package covers training, tracking (`model.track`), segmentation, YOLOE open-vocabulary, SAHI and export. Rectangular inference (1280×736) is supported. The NMS-free head makes post-processing cheap and deterministic. STAL helps with far-away small objects.
  - This choice carries the lowest engineering risk for a 3-person team.
- **Backup: RF-DETR-S/M** with `trackers` (Apache-2.0) and `supervision` (MIT). This gives a fully permissive stack.
  - It is more accurate per millisecond on COCO (S: 53.0 AP at 3.5 ms; YOLO26s: 48.6 AP at 2.5 ms).
  - Its published speeds are at 512–576 px square, though. At 1280×720 on 1080p footage, the ViT backbone costs several times more **(est.)**.
  - Switch to it only if the dev-set recall on small or night objects is clearly better.
- **Input size.** What matters is how many pixels the smallest object gets after resizing (§5.4). On 1080p footage:

  | Input size | Pedestrian 20×50 px in source | Car 60×40 px in source |
  |---|---|---|
  | 640 | 7×17 (too small) | 20×13 |
  | 960 | 10×25 | 30×20 |
  | 1280 | 13×33 | 40×27 |

  Use **1280 for Part A** and **960 for Part B**.
- Scaling YOLO26 by FLOPs gives these TensorRT latency estimates **(est.)**:

  | Model | 960×544 | 1280×736 |
  |---|---|---|
  | YOLO26s | ≈ 3.2 ms | ≈ 5.8 ms |
  | YOLO26m | ≈ 6.0 ms | ≈ 10.8 ms |

  PyTorch FP16 at batch 8 is typically 1.2–2× slower than TensorRT **(est.)**.

---

## 2. Classes, gaps, fine-tuning data, pseudo-labelling

### 2.1 COCO coverage
Group the COCO classes into super-classes for the rules, and **track class-agnostically**. Take a majority vote over each track's detections for its class, so that car↔truck flicker cannot split a track.

| Super-class | COCO classes |
|---|---|
| `vehicle` | car, bus, truck |
| `two_wheeler` | motorcycle, bicycle |
| `person` | person |
| `animal` | dog, cat, horse, sheep, cow, bird, … |
| `traffic_light` | traffic light (useful only for EDA; the state comes from fixed ROIs) |

**Perception trap for `jaywalking`:** COCO labels riders as `person`. Drop a person box when its bottom 30% overlaps a two-wheeler box (IoU > 0.3, or its foot point lies inside the two-wheeler box). Otherwise every motorcyclist becomes a "pedestrian on the roadway".

### 2.2 What COCO does not cover
- **Debris, fallen objects, cones** (`road_obstacle`):
  - Build a static-foreground detector with two background models (MOG2/median, one fast and one slow, on 1–2 fps downscaled frames). A blob that stays in the carriageway ROI for ≥ 3 s and does not belong to any active or stopped track becomes an obstacle candidate.
  - Confirm candidates with **YOLOE-26** open-vocabulary text prompts (e.g. "box", "tire", "rock", "traffic cone", "branch", "bag", "debris"). YOLOE-26x gets 40.6 AP on LVIS with text prompts ([YOLO26 paper](https://arxiv.org/abs/2606.03748)).
  - **Bake the prompts in offline** (`set_classes`, then save). Otherwise Ultralytics tries to download a text encoder at run time, and there is no internet during evaluation.
- **Fire/smoke:** fine-tune YOLO26n on **D-Fire** (21,527 images, YOLO format, **CC0**; [repo](https://github.com/gaia-solutions-on-demand/DFireDataset)). Run it at 1–2 fps inside a road/vehicle ROI, and require the detection to persist ≥ 2 s. The cost is negligible.
- **Overturned or deformed crash vehicles** are often missed by COCO models. When a vehicle track vanishes in the middle of the road, the static-foreground mask is the fallback for the "objects stop moving" end boundary.

### 2.3 Traffic-surveillance datasets (all require a README entry)

| Dataset | Content | Licence (verified where linked) | Use |
|---|---|---|---|
| COCO 2017 | 80 classes | Annotations CC BY 4.0; images under Flickr terms ([HF card](https://huggingface.co/datasets/shunk031/MSCOCO)) | Pretraining (already in the weights) |
| UA-DETRAC | 100 sequences, 140k frames, 8,250 vehicles, 1.21M boxes, fixed roadside cameras ([paper](https://arxiv.org/abs/1511.04136)) | Commonly cited as CC BY-NC-SA 3.0; **could not verify** because the official site is offline. Check the Kaggle mirror. | Best domain match for vehicles |
| VisDrone | Drone view, tiny objects | CC BY-NC-SA 3.0, academic use ([HF](https://huggingface.co/datasets/Voxel51/VisDrone2019-DET)) | Only if the camera is very high or top-down |
| BDD100K | Dashcam footage, includes traffic-light colour attributes | Berkeley licence: education, research, non-profit ([license](https://doc.bdd100k.com/license.html)) | Traffic-light crop classifier pretraining |
| MIO-TCD | Traffic-camera crops and boxes | No licence stated ([TIB](https://service.tib.eu/ldmservice/dataset/miovision-traf-c-camera-dataset--mio-tcd-)) | Avoid, or ask |
| AI City Challenge | Tracks vary; 2025 Track 1 is CC BY 4.0 | Older tracks carry a dataset licence agreement ([page](https://www.aicitychallenge.org/ai-city-challenge-dataset-access/)) | Anomaly/stall tracks. Read the terms first. |
| D-Fire | Fire and smoke | CC0 | Fire/smoke model |

### 2.4 Is fine-tuning worth it? Yes, mostly through self-training on the sample videos
The test videos come from the **same camera and angle**, so in-domain pseudo-labels are the best data we can get. Adapting to the background is an advantage here, not a risk.

Recipe:
1. Sample frames every 0.5–1 s from every sample video, making sure day, night and rush hour are all represented.
2. Use a **teacher**: YOLO26x at native 1920 px with TTA (flip plus 2 scales). Optionally ensemble it with RF-DETR-L on 2 tiles, merging with WBF.
   - Keep boxes with confidence ≥ 0.5.
   - Drop frames that have many boxes in the 0.25–0.5 range, because Ultralytics has no ignore regions.
3. **Temporal consistency filter.** Keep a box only if it matches a box in the ±1 neighbouring processed frame (IoU > 0.5). Fill single-frame gaps by interpolation.
4. **Hand-correct ~300 stratified frames** in CVAT or Label Studio. Hold out one entire sample video. This is the detector dev set.
5. **Fine-tune** YOLO26m and YOLO26s from their COCO weights:
   - Train on the merged super-classes plus traffic light, at imgsz 1280, for 30–50 epochs.
   - Augment with 10% grayscale (for night IR footage) and HSV jitter.
   - Mix in 20–30% UA-DETRAC frames to avoid collapsing onto the samples.
   - Use `seed=0, deterministic=True`.
   - Expected cost: a few hours on a Kaggle T4 **(est.)**. Kaggle gives 2×T4 16 GB at about 30 GPU-h/week ([Kaggle docs](https://www.kaggle.com/docs/efficient-gpu-usage)).
6. Judge the fine-tune by **event F1 on the team's dev labels**, not by mAP alone.

---

## 3. Tracking

### 3.1 Options and licences

| Tracker | Library (licence) | ReID | MOT17 HOTA (library's own benchmark) | Verdict |
|---|---|---|---|---|
| **ByteTrack** | Ultralytics (AGPL), [`trackers`](https://github.com/roboflow/trackers) (Apache-2.0), BoxMOT (AGPL) | no | 60.1 (`trackers`) / 67.7 (BoxMOT ablation split) | **Primary.** Its second association on low-confidence detections recovers occluded cars in queues. |
| **OC-SORT** | `trackers`, BoxMOT | no | 61.9 / 66.4 | **Backup.** Its observation-centric re-update helps after occlusion and non-linear motion (swerves, crashes). |
| BoT-SORT | Ultralytics, `trackers`, BoxMOT | optional | 63.7 / 69.7 | With GMC and ReID turned off it is essentially ByteTrack with a better Kalman state. Fine to use as `botsort.yaml` with `gmc_method: none`. |
| Deep OC-SORT, StrongSORT | BoxMOT (AGPL) | required | 68.0 / 68.1 | They need a vehicle ReID model and cost more per box. Not worth it with a fixed camera plus offline stitching. |
| McByte | `trackers` | — | 64.1 | Heavier; not needed. |

Sources: [`trackers` README](https://github.com/roboflow/trackers), [BoxMOT README](https://github.com/mikel-brostrom/boxmot). BoxMOT is **AGPL-3.0**; `trackers` is Apache-2.0, supports Python ≥ 3.10 and works directly with `supervision.Detections`.

### 3.2 Behaviour at stride 2–3
Consider raw box IoU between two consecutive processed frames under pure longitudinal motion, with no motion prediction: IoU = (L − d)/(L + d), where L is the object length and d the distance moved **(computed)**:

| Object | Speed | Stride 1 | Stride 2 | Stride 3 | Stride 5 |
|---|---|---|---|---|---|
| Car, 4.5 m | 30 km/h | 0.86 | 0.74 | 0.64 | 0.46 |
| Car, 4.5 m | 60 km/h | 0.74 | 0.54 | **0.38** | 0.15 |
| Car, 4.5 m | 80 km/h | 0.67 | 0.43 | 0.26 | **0.01** |
| Motorcycle, 2 m | 60 km/h | 0.50 | 0.20 | **0.00** | 0.00 |
| Pedestrian | 5 km/h | 0.80 | 0.64 | 0.50 | 0.29 |

Once a track has 2–3 updates, the Kalman prediction removes most of this. But raw IoU is what matters at **track birth** and during **abrupt manoeuvres**: collisions, hard braking, swerves. Those are exactly the moments that decide `accident` and `near_miss` boundaries.

- **Use stride ≤ 2 (12.5 fps) for Part A vehicles.** Stride 3 is acceptable in Part B.
- **Stride 5 breaks association** for fast two-wheelers.
- For accident and near-miss candidates, **re-run stride 1** on the ±3 s window (§5.5).

### 3.3 Parameters (Ultralytics YAML names; `trackers` has equivalents)
Ultralytics defaults are `track_high_thresh 0.25, track_low_thresh 0.1, new_track_thresh 0.25, track_buffer 30, match_thresh 0.8, fuse_score True, gmc_method sparseOptFlow, with_reid False` ([botsort.yaml](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/botsort.yaml)).

The current `byte_tracker.py` sets `max_frames_lost = track_buffer`. That means **the buffer counts processed frames, not seconds**. In `trackers`, by contrast, `lost_track_buffer` is given in 30-fps units and scaled by `frame_rate`, so set `frame_rate = 25/stride` there.

Recommended starting point, to be tuned on dev labels:
- Detector confidence floor: **0.10**. Keep low-score boxes so ByteTrack's second association can use them.
- `track_high_thresh` **0.40**, `track_low_thresh` **0.10**, `new_track_thresh` **0.50**. A higher birth threshold means fewer ghost tracks.
- `match_thresh` **0.8** (IoU ≥ 0.2). The second association uses IoU 0.5, which is hard-coded.
- `track_buffer` = **ceil(3 s × fps_eff)**: 38 at stride 2, 25 at stride 3.
- `gmc_method: none`. The camera is fixed, and sparse optical flow on every update costs CPU for nothing.
- `with_reid: False`.
- Rules only consume tracks that are ≥ 0.5 s long and have a mean score ≥ 0.4.
- Run a class-agnostic dedup at IoU 0.7 before tracking. An end-to-end head can occasionally output car+truck duplicates for one object.

### 3.4 Part A can repair tracks offline (Part B cannot)
Part A is allowed to use future frames, so after online tracking:

1. **Stitch tracklets.** Link A→B when all of these hold:
   - 0 < t_B,start − t_A,end ≤ 2 s;
   - same super-class;
   - ‖p_B − (p_A + v_A·Δt)‖ ≤ max(0.5·box diagonal, ~1.5 m in ground units);
   - optional: HSV-histogram similarity ≥ 0.7. This is a cheap, deterministic appearance cue.

   Solve the linking with Hungarian matching.
2. **Stationary objects get a long gap.** For stopped vehicles and obstacles, link across gaps up to **20 s** when p_B ≈ p_A within 0.3 box diagonals. A bus passing in front should not reset the 10-s `stopped_vehicle` timer.
3. **Split tracks at implausible jumps**: acceleration above ~8 m/s² or heading change above ~90° in 0.2 s, outside candidate crash windows. Splitting is safer than merging.
4. **Smooth** positions with a Savitzky–Golay filter or an RTS smoother (0.5–1 s window) before computing speed, heading and TTC.

**How ID errors hurt the rules:**
- An ID swap between opposite-direction vehicles creates false `wrong_way`, `solid_line_crossing` or `illegal_u_turn` events.
- A track break creates false negatives for `stopped_vehicle` and `congestion` (the timer resets) and splits `jaywalking` segments. Split segments fail IoU 0.7.
- Swaps during a crash smear the accident boundaries.

Mitigations: require evidence to persist ≥ 1 s, check physical plausibility, apply the stitching above, and merge same-class segments separated by < 1–2 s in post-processing.

---

## 4. Traffic-light state (fixed camera)

The signal state drives `red_light`, `stop_line` (its **end** boundary is "signal turns green"), the exclusion of signal queues from `stopped_vehicle` and `congestion`, and possibly `failure_to_yield` if a pedestrian head is visible.

### 4.1 Signal visible: algorithm (≈0.05 ms per frame, CPU)
1. **Config.** For each signal head, record the housing box and the centres and radius of the three lamps (R, Y, G) in `scene.json`. Draw them once on a sample frame, using camera.md as a guide.
2. **Features per lamp i, per processed frame.** Stride 2 (12.5 Hz, 80 ms resolution) is enough.
   - `v_i` = 90th percentile of V in the lamp disk.
   - `s_i` = median S in a ring around the disk.
   - `h_i` = circular median hue of pixels with S > 60 and V > 120.
   - Gates (OpenCV hue scale 0–180): red H ∈ [0,10] ∪ [160,180]; amber H ∈ [10,35]; green H ∈ [40,95]. LED greens are often cyan.
3. **Self-calibration.**
   - Part A: run 2-means on each lamp's `v_i` over the whole video to get an on-cluster and an off-cluster; the threshold is their midpoint. If the clusters are separated by < 30 V-units, the lamp is "unreliable" (glare or bloom); fall back to ranking lamps by relative brightness.
   - Part B: use thresholds learned on the samples for the first 10 s. After that, use (P10 + P90)/2 over a rolling 120-s history.
4. **Position beats colour.** The lit lamp is argmax_i (v_i − baseline_i), provided the margin exceeds δ. This keeps working when a night IR mode turns the image grayscale, because red is always the top (or left) lamp. Hue is only a confirmation.
5. **LED flicker at 25 fps.**
   - LED drivers switch at mains-related frequencies (100 Hz on 50 Hz mains) or at PWM frequencies. With short daylight exposures, the camera samples the LED at different phases, so an ON lamp can look OFF for a run of frames. When the frequencies are near multiples of each other, this drifts slowly as a beat ([ENTTEC](https://support.enttec.com/pixel/pixel-general-knowledge/why-do-leds-flicker-on-camera), [DiffuFlicker](https://link.springer.com/chapter/10.1007/978-3-032-31654-7_13)).
   - The effect is **asymmetric**: an ON lamp can look OFF, but an OFF lamp almost never looks ON (glare aside). So apply a temporal **max-pool** of `on_i` over w = 3–5 processed frames (0.25–0.4 s) before decoding the state. Shift the resulting transitions back by w/2.
   - Measure the dropout lengths in EDA (plot `v_i` per frame). This is also a good EDA chart for the website.
6. **Decoding the state.**
   - States: {G, G-flash, Y, R, R+Y}, with minimum durations: G ≥ 5 s, Y 2–4 s, R ≥ 5 s, R+Y 1–2 s **(est.; fit these on the samples)**.
   - Part A: Viterbi decoding over the whole video.
   - Part B: a causal forward filter or hysteresis.
   - Post-Soviet signals often have flashing green and red+amber. **Check on the samples.** Treat red+amber as "red" for `red_light`, since the prohibition still applies, then confirm against how the dev labels come out.
   - Some Tashkent intersections have countdown displays and adaptive control ([Zamin.uz](https://zamin.uz/en/society/198778-new-smart-pedestrian-traffic-lights-being-tested-in-tashkent.html), [yuz.uz](https://yuz.uz/en/news/umne-svetofor-pomogut-sokratit-probki-i-izmenit-upravlenie-transportom-v-tashkente)), so do not assume a fixed cycle.
7. **Glare and night bloom.**
   - All lamps bright with low saturation means glare: output "unknown" and hold the last state, using the cycle prior.
   - Bloom (a saturated white core): classify by the hue of the surrounding ring.
8. **Small CNN option.** Auto-label about 3k crops (32×96) with steps 1–6, correct them by hand, and train a 4-class MobileNetV3-small or a 3-layer CNN. The cost is trivial. Only do this if HSV fails on the dev set.
9. **Which approach does the head control?** The visible head usually controls one approach. For the conflicting approaches, red = the complement of green, plus an all-red margin of about 2 s **(est.)**.

### 4.2 Signal not visible (or not visible for some approaches)
Infer each approach's phase from what vehicles do.

- For each approach a, compute:
  - `flow_a(t)`: stop-line crossings in the last 5 s;
  - `queue_a(t)`: stationary vehicles (speed below ~1 m/s) within the Q m upstream of the line.
- Phase rules:
  - **Red_a** evidence: `queue_a ≥ 1` for ≥ 3 s, and `flow_a = 0` while a conflicting approach has `flow > 0`.
  - **Green onset_a** ≈ the time the first queued vehicle starts moving, minus the start-up reaction time (≈ 1–2 s, a traffic-engineering convention; calibrate it on samples where the signal is visible, if any). This gives the `stop_line` end boundary.
  - **`red_light` candidate**: a vehicle of approach a crosses the stop line while other lanes of approach a hold stationary vehicles at the line, or while a conflicting flow is crossing.
- Part A only: estimate the cycle length from the autocorrelation of `flow_a` (fixed-time plans usually run 60–150 s, **est.**). Use it as an HMM prior when the evidence is ambiguous. Disable it if the autocorrelation peak is weak (adaptive control).
- A visible **pedestrian** signal head, if any, directly gives the vehicle red for the approach it crosses.

---

## 5. Throughput on T4 + 8 CPU cores

### 5.1 Decoding. The bottleneck is YUV→BGR conversion, not decoding.
**Measured locally** with synthetic 1080p25 (4 Mbps) and 4K25 (16 Mbps) H.264 clips, on an i5-12450H under heavy background load. Ranges show run-to-run noise.

| Path | 1080p | 4K |
|---|---|---|
| ffmpeg CLI, CPU, decode only (multi-threaded) | ~560–800 fps | ~244 fps |
| NVDEC via ffmpeg `-hwaccel cuda` (GTX 1650, same 4th-gen Turing NVDEC as the T4) | 580–700 fps | 194 fps |
| PyAV 18.1, decode only (`thread_type='AUTO'`) | 240–1010 fps | 170–260 fps |
| PyAV, decode plus `to_ndarray('bgr24')` on every frame | 52–211 fps | 29–81 fps |
| OpenCV 4.13 `cap.read()` on every frame (what the harness probably does) | 56–130 fps | **27–28 fps** |
| OpenCV `grab()` every frame, `retrieve()` every 3rd | 137–317 fps (over all frames) | 37–82 fps |
| ffmpeg → pipe → numpy (Windows) | 20–35 fps | 2 fps. **Do not use pipes.** |

Conclusions:
- **Convert to BGR only the frames you process.** Use `grab()` for the skipped ones.
- The harness decodes **every** frame for Part B, and that cost counts against our budget.
  - At 1080p this is ~8 ms/frame ≈ **0.2× real time**.
  - At **4K it is ~36 ms/frame ≈ 0.9× real time** before any of our code runs.
- The T4 has **2 NVDEC engines** ([NVIDIA matrix](https://developer.nvidia.com/video-encode-decode-support-matrix)). If the footage is 4K, decode Part A on the GPU (torchcodec 0.16 CUDA / PyNvVideoCodec) and resize there.
- Do not use `decord`: its last release was in 2021. Also note that PyAV 18 needs Python ≥ 3.11.

### 5.2 Budget model (`budget.py`, per video, fps = 25; all per-frame costs are **est.**)
Per-frame cost assumptions:
- decode only: 2 ms (1080p) / 6 ms (4K);
- BGR conversion plus resize: 4 / 14 ms;
- harness `read()`: 8 / 36 ms;
- tracker plus rules: 1.5 ms;
- YOLO26s@1280×736 FP16 at batch 8: ≈ 7 ms/img;
- YOLO26m@1280×736: ≈ 14 ms/img;
- YOLO26s@960 at batch 1: ≈ 12 ms;
- YOLO26n@640 at batch 1: ≈ 8 ms;
- fixed overhead: 20 s.

Part A pipelines CPU and GPU work (max of the two); Part B runs synchronously.

| Scenario | 5-min video | ×duration | 10-min video | ×duration |
|---|---|---|---|---|
| 1080p. A: YOLO26s@1280, stride 2. B: YOLO26n@640, stride 5 | 133 s | 0.44× | 246 s | 0.41× |
| **1080p. A: YOLO26m@1280, stride 1. B: YOLO26s@960, stride 3** | 224 s | 0.75× | 428 s | 0.71× |
| 1080p. A: YOLO26s TRT, stride 2, plus a 15% stride-1 YOLO26m-seg refine pass. B: YOLO26s@960, stride 2 | 192 s | 0.64× | 364 s | 0.61× |
| 4K. A: YOLO26s@1280, stride 2. B: YOLO26n@640, stride 5 | 411 s | 1.37× | 802 s | 1.34× |
| 4K. A: YOLO26m, stride 1. B: YOLO26s@960, stride 3 | 490 s | 1.63× | 961 s | 1.60× |
| CPU-only fallback, 1080p. A: YOLO26n ONNX-CPU (~45 ms), stride 5. B: same model, stride 10 | 208 s | 0.69× | 395 s | 0.66× |

**Sensitivity check:** with GPU costs ×2 and CPU costs ×1.5 (an unknown "T4-class" machine):
- the bold 1080p row takes **1.28×**;
- the cheap 1080p row takes 0.62×;
- the 4K row takes **2.0×**.

**Rules:**
- Target ≤ 1.0× duration on a Kaggle T4, which gives ≥ 3× safety.
- Part A should spend ≤ 1.5× of the duration, because Part B's harness decode is fixed.
- The example ground truth has `duration: 600.0`, so plan for 10-minute clips.

### 5.3 Batching and threads
- Part A: a producer thread decodes (OpenCV and PyAV release the GIL) into a bounded queue. The consumer builds **fixed-size batches of 8** and pads the last batch, because different batch sizes can choose different kernels and change outputs slightly.
- Upload once with pinned memory, and letterbox on the GPU if the CPU becomes the bottleneck.
- Set `cv2.setNumThreads(4)` and `torch.set_num_threads(4)` to avoid oversubscribing the 8 cores.
- Part B: batch size 1 is unavoidable. Keep the model small and skip frames.

### 5.4 Input size, ROI crop, tiling
- **Crop to the carriageway's bounding rectangle** from camera.md before resizing. If the road covers 60% of the frame, that is roughly a 1.6× pixel gain for free.
- Choose the input size so the **5th-percentile far-field pedestrian is ≥ 12 px tall** after scaling. Measure this in EDA with the teacher model.
- For 4K footage, or far-field objects that are too small, use **2 fixed tiles** (a far-field strip at higher scale plus the full frame at 1280) and merge with NMS or WBF. This roughly doubles GPU time.
- Prefer fixed, batched tiles over the generic SAHI library slicer. SAHI (MIT) is fine for the offline teacher, but its per-slice Python overhead is high.

### 5.5 Per-frame processing plan
```
# Part A (detect_events): offline, may use future frames
cfg = load_scene('scene.json')                 # road polygons, stop lines, lamp ROIs, homography
for batch in decoder(path, stride=s_A, batch=8):     # grab() all frames, retrieve() every s_A-th; keep PTS
    tl_feats += lamp_features(batch.full_res)        # ~0.05 ms per frame
    x = gpu_letterbox(crop(batch, cfg.road_rect), (736, 1280))
    dets = yolo26m(x, half=True, conf=0.10)          # NMS-free; then agnostic dedup at IoU 0.7
    for f in batch: tracks.update(superclass(dets[f]))  # ByteTrack, online
    bg.update_every(12 frames)                       # static-foreground model (obstacle, stopped, smoke candidates)
    firesmoke.every(12 frames)                       # YOLO26n-D-Fire
tracks = smooth(stitch(tracks))                      # offline repair (sec. 3.4)
tl = viterbi(tl_feats)                               # sec. 4
cands = rules(tracks, tl, cfg)                       # other workstreams
refine(cands, stride=1, window=+-3s, model=yolo26m[-seg])   # PyAV seek to preceding keyframe; bounded to 15% of frames
# Part B (RiskEstimator.step): strictly causal
reset(meta): k = 2 if meta.width <= 1920 else 3 (deterministic, from metadata only); tracker.reset()
step(frame, t): lamp_features(frame) every frame; if i % k: return last
                dets = yolo26s(resize(frame, 960)); tracks = bytetrack.update(dets); return risk(tracks, tl)
```
- **Emergency brake.** Part A tracks wall-clock time. If elapsed time exceeds 1.8× duration, it stops refinement and returns what it has. This only triggers in cases that would otherwise score zero, so it does not threaten determinism in normal runs.
- Otherwise, **never choose strides based on measured time**. Doing so makes two runs produce different predictions.

---

## 6. Inference backend, the TensorRT caveat, determinism

**TensorRT facts** ([NVIDIA engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)):
- An engine only runs on the device type it was built for.
- Hardware-compatibility mode `kAMPERE_PLUS` excludes **Turing (T4, sm_75)**.
- Tactic selection is timing-based, so rebuilding can pick different kernels. NVIDIA recommends locking clocks and using timing caches for reproducible builds ([best practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)).

| Option | Portability | Speed | Determinism | Verdict |
|---|---|---|---|---|
| **PyTorch FP16** (Ultralytics `.pt`, `half=True`) | Any CUDA GPU; nothing to build | Baseline (~1.2–2× slower than TRT, **est.**) | Good with `cudnn.benchmark=False`, `cudnn.deterministic=True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and fixed batch size | **Default** |
| ONNX Runtime CUDA EP | Needs a matching CUDA/cuDNN runtime (≥ 1.29 → CUDA 13) | ≈ PyTorch | `cudnn_conv_algo_search` **defaults to EXHAUSTIVE**; set it to `HEURISTIC` or `DEFAULT` ([docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)) | Only as the CPU fallback (CPU EP) |
| Pre-built T4 engine plus fallback to PyTorch | Breaks if the GPU, TRT or driver differ | Fastest | Deterministic when the engine is reused | Optional, if speed is ever short |
| Build the engine at run time | Portable | Build takes minutes and counts against video 1 **(est.)** | **Trap:** run 1 builds (or switches mid-run) while run 2 uses the cached engine, so the outputs differ | Avoid. The one exception is building inside `weights/download.sh`, and only if the organisers confirm it runs on the evaluation machine. |

**Determinism checklist:**
- Seeds (`random`, `numpy`, `torch`); `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Fixed batch size with padding.
- Frames processed in order; no time-adaptive striding.
- Hysteresis on every threshold.
- Deterministic Hungarian matching (scipy).
- Test by running twice on Kaggle and diffing `predictions.json`.

**CUDA and driver risk.**
- CUDA 13 drops pre-Turing GPUs but keeps sm_75 ([PyTorch RFC](https://github.com/pytorch/pytorch/issues/190385)).
- The organisers' driver version is unknown. CUDA 12.x wheels (cu126/cu128) are the safer bet for older drivers. Put `--extra-index-url https://download.pytorch.org/whl/cu126` in `requirements.txt`. Verify the minimum driver versions on NVIDIA's compatibility table.
- Always keep the CPU fallback path, so that "CUDA unavailable" does not become "model score 0".

---

## 7. Instance segmentation: not worth it in the main loop

- **Cost.** GPU time goes up by 25–42% at the same size ([seg table](https://docs.ultralytics.com/tasks/segment/)), plus CPU mask processing:

  | Size | Detection | Segmentation |
  |---|---|---|
  | n | 1.7 ms | 2.1 ms |
  | s | 2.5 ms | 3.3 ms |
  | m | 4.7 ms | 6.7 ms |

- **Accident "contact".** In a perspective view, overlapping masks are just as ambiguous as overlapping boxes, because a nearer vehicle occludes a farther one. The start of a collision is better detected by combining:
  - footprint distance on the ground plane (homography) below ~1 m;
  - a simultaneous speed or heading discontinuity of both tracks (|Δv| > 3 m/s within 0.5 s, **est.**);
  - both vehicles stationary afterwards.
- **"Wheel crosses line".** Box **bottom corners** (not bottom-centre) projected to the ground are a good proxy for the start and end of `solid_line_crossing`:
  - start = the near bottom corner crosses the line;
  - end = the far corner crosses it.

  Box-centre conventions can be off by roughly (half vehicle width)/(lateral speed) ≈ 0.5–1 s **(est.)**, which is enough to fail IoU 0.7. Masks improve the footprint for vehicles seen diagonally.
- **Verdict.** Run YOLO26s-seg (or RF-DETR-Seg-S, Apache-2.0) only inside the stride-1 refinement windows, and only if dev labels show boundary errors above ~0.5 s for `accident`, `near_miss` or `solid_line_crossing`.

---

## 8. Offline and packaging traps in the perception stack
- Ultralytics can **download at run time**: fonts for plotting, the YOLOE text encoder, and the `lap` package for tracking through AutoUpdate. It also expects weights under a known path. Guard against this:
  - put `lap` in `requirements.txt`;
  - use cv2 for drawing (not Ultralytics plotting);
  - bake YOLOE prompts into the saved model;
  - pass explicit local weight paths;
  - **test the whole run with networking disabled** (e.g. `docker run --network none`).
- RF-DETR downloads its DINOv2 backbone and pretrained weights unless you give it a local checkpoint.
- Python version: PyAV 18 needs ≥ 3.11 and the organisers promise "3.10+". Either rely on OpenCV for decoding (abi3 wheels), pin an older PyAV conditionally, or ship a Dockerfile.
- The local development GTX 1650 is sm_75 (the same architecture as the T4, so kernels are identical) but it has **no tensor cores**. Its FP16 timings say nothing about the T4. Benchmark on Kaggle or Colab T4s.
- Load models once, at import or on the first call, then warm up with 3 dummy batches. Whether model load counts toward video 1's time limit is an open question.

---

## 9. Cross-cutting implications for the other workstreams
- **Part B reuse of Part A.** Per-frame detections, or an **online** tracker state computed only from frames ≤ t, are arguably "not computed with future frames", so caching them from Part A may be legal. It is still a grey area. **Ask in the hackathon channel.** Until then, keep Part B self-sufficient; it costs only ~0.1–0.3× duration.
- **EDA for the website comes from the perception outputs:** per-class counts over time, motion heatmaps from track footprints, a lane-direction field from smoothed velocities, lamp-brightness and flicker plots, and object-size histograms (which justify the 1280 input).
- **Timestamps.** Use container PTS minus the first PTS, so times are "seconds from first frame" and match the harness's `t_sec`.
- **Live demo on CPU.** The CPU fallback path (YOLO26n ONNX, stride 5) processes a 2-minute upload in roughly 0.5–1× real time on a typical server CPU **(est.)**. Show a progress bar.

## 10. First 48 hours after the samples arrive
1. `ffprobe` every sample: codec, resolution, fps, GOP length, VFR or not, duration. Record whether it is 4K and whether the scene is day or night.
2. Kaggle T4:
   - time the harness-style `cv2.read()` loop;
   - time YOLO26s and YOLO26m FP16 at 960 and 1280, at batch 1 and batch 8;
   - fill in the budget table with real numbers.
3. Draw `scene.json`: road polygon, lanes, stop lines, crossings, lamp ROIs. Plot per-lamp V over time.
4. Run the teacher (YOLO26x @1920 with TTA) over the samples, look at object-size percentiles, and pick the input size.
5. Start labelling the dev set (events, plus ~300 corrected detection frames), then fine-tune YOLO26s and YOLO26m.



## Top recommendations
- Use Ultralytics YOLO26m at 1280x736 rectangular input, PyTorch FP16, stride 1-2 for Part A, and YOLO26s at 960 with stride 2-3 inside RiskEstimator.step. Keep RF-DETR-S/M (Apache-2.0) as the backup. License the repo AGPL-3.0 and link it from the demo site.
- Fine-tune YOLO26s and YOLO26m on pseudo-labels from the sample videos: a YOLO26x@1920+TTA teacher, a temporal-consistency filter and about 300 hand-corrected frames (one video held out). Train on merged super-classes (vehicle, two_wheeler, person, animal). Optionally mix in 20-30% UA-DETRAC frames.
- Use ByteTrack without ReID and without GMC (gmc_method none). Set the confidence floor to 0.10, high/new thresholds to 0.4/0.5, and track_buffer to about 3 s of processed frames. In Part A, add offline tracklet stitching (gaps up to 2 s; up to 20 s for stationary objects) plus RTS or Savitzky-Golay smoothing. Re-run stride 1 on +-3 s windows around accident and near-miss candidates.
- Decode with OpenCV: grab() every frame and retrieve() only the frames you process, because BGR conversion dominates the cost. Plan for the harness decoding every frame for Part B: about 0.2x real time at 1080p and about 0.9x at 4K. If the footage is 4K, decode Part A on NVDEC (torchcodec or PyNvVideoCodec).
- Budget target: at most 1.0x video duration measured on a Kaggle T4 (the model estimates 0.4-0.75x at 1080p), with Part A at most 1.5x. Never pick strides from wall-clock time. Add an emergency brake at 1.8x that returns partial Part A results.
- Leave TensorRT out of the default path. PyTorch FP16 with cudnn.benchmark=False, deterministic flags and a fixed padded batch size is portable across 'T4-class' GPUs and reproducible. Keep an ONNX-CPU YOLO26n stride-5 fallback so a CUDA or driver problem does not zero the score.
- Traffic lights: hand-set per-lamp ROIs; per-video 2-means brightness calibration; classify by which lamp position is lit, with hue only as confirmation; temporal max-pool over 3-5 samples against LED flicker; Viterbi decoding with minimum phase durations offline, a causal filter in Part B. If the signal is not visible, infer phase from stop-line queue and flow plus cycle autocorrelation.
- Do not run instance segmentation in the main loop. Use box bottom corners projected to the ground plane for line crossings, and kinematic discontinuities plus ground-plane proximity for collision start. Try YOLO26s-seg only in refinement windows if dev labels show boundary errors above 0.5 s.
- Cover the classes COCO lacks: a YOLO26n fine-tuned on D-Fire (CC0) at 1-2 fps for fire_smoke; a static-foreground background model plus YOLOE-26 with prompts baked in offline for road_obstacle. Suppress persons riding two-wheelers so they do not trigger jaywalking.
- Test the full pipeline with networking disabled (Ultralytics can auto-download fonts, lap and the YOLOE text encoder). Pin versions that work on Python 3.10-3.13 (PyAV 18 needs 3.11+). Prefer CUDA 12.x PyTorch wheels via --extra-index-url in requirements.txt for broader driver compatibility.


## Open questions
- How does run_submission.py decode frames (cv2.VideoCapture read() or something else), and does it call detect_events before streaming frames to RiskEstimator for the same video? This sets the fixed Part B decode cost and whether any caching from Part A is possible.
- Is it allowed for RiskEstimator to read per-frame detections, or online tracker state computed only from frames <= t, that Part A cached? The rule bans 'Part A output computed with future frames'; causal per-frame outputs are a grey area. Ask in the hackathon channel.
- Is model loading and warm-up time counted in the first video's 3x budget? And is weights/download.sh run on the evaluation machine itself? That decides whether a TensorRT engine could be built there.
- What are the resolution, codec, GOP and bitrate of the sample videos (1080p vs 4K), and is there night or IR footage? This changes decode cost by about 4x and the input-size and tiling decision.
- Does camera.md say the signal head is visible, and for which approaches? Is there a visible pedestrian signal or countdown display? Is the signal fixed-time or adaptive?
- What NVIDIA driver, CUDA runtime and Python version does the evaluation machine have ('Python 3.10+' and 'T4-class, may update')? This decides between cu12.x and cu13 wheels and whether PyAV 18 can be used.
- The UA-DETRAC licence could not be verified from the official site (it is commonly cited as CC BY-NC-SA 3.0). Check it on the mirror before listing it in the README.
- My T4 latency estimates for PyTorch FP16 at 1280x736 (7 ms/img for YOLO26s, 14 ms/img for YOLO26m at batch 8) are extrapolated from the TensorRT tables and FLOP scaling, not measured. Measure them on a Kaggle T4 in the first days.
- How do the organisers' annotators treat red+amber and flashing green for red_light and stop_line boundaries? This can only be settled from the class conventions and the team's own dev labels.


## Key claims (as submitted for verification)
- YOLO26 (Ultralytics, released January 2026) reaches 40.9/48.6/53.1/55.0/57.5 COCO mAP50-95 for n/s/m/l/x at 1.7/2.5/4.7/6.2/11.8 ms T4 TensorRT10 FP16 latency, and supports NMS-free end-to-end inference. — https://docs.ultralytics.com/models/yolo26
- Ultralytics YOLO code and models are licensed AGPL-3.0 (or a commercial Enterprise licence); YOLO27 is listed as 'coming soon'; the latest ultralytics PyPI release is 8.4.162 (2026-09-24). — https://github.com/ultralytics/ultralytics ; https://pypi.org/project/ultralytics/
- RF-DETR N/S/M/L are Apache-2.0 with COCO AP 48.4/53.0/54.7/56.5 at 2.3/3.5/4.4/6.8 ms on a T4 (TensorRT FP16), at native resolutions 384-704 px; the XL and 2XL sizes are under PML 1.0; rfdetr 1.11.0 was released 2026-09-24. — https://github.com/roboflow/rf-detr ; https://pypi.org/project/rfdetr/
- D-FINE and RT-DETRv4 are Apache-2.0 (D-FINE-S 48.5 AP at 3.49 ms; RT-DETRv4-S 49.8 AP at 3.66 ms on T4), whereas DEIMv2 ships under a custom 'DEIMv2 License' with commercial use by inquiry. — https://github.com/Peterande/D-FINE ; https://github.com/RT-DETRs/RT-DETRv4 ; https://github.com/Intellindust-AI-Lab/DEIMv2
- YOLO26-seg n/s/m take 2.1/3.3/6.7 ms on a T4 (TensorRT) versus 1.7/2.5/4.7 ms for detection only, i.e. roughly 25-42% more GPU time. — https://docs.ultralytics.com/tasks/segment/
- Roboflow 'trackers' (Apache-2.0) implements SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU and McByte (MOT17 HOTA 60.1 for ByteTrack, 61.9 for OC-SORT, 63.7 for BoT-SORT); BoxMOT is AGPL-3.0. — https://github.com/roboflow/trackers ; https://github.com/mikel-brostrom/boxmot
- Ultralytics botsort.yaml defaults are track_high_thresh 0.25, track_low_thresh 0.1, new_track_thresh 0.25, track_buffer 30, match_thresh 0.8, gmc_method sparseOptFlow and with_reid False; the current byte_tracker sets max_frames_lost = track_buffer, so the buffer counts processed frames, not seconds. — https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/botsort.yaml ; https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/trackers/byte_tracker.py
- By default a TensorRT engine only runs on the device type it was built on, and the kAMPERE_PLUS hardware-compatibility mode does not cover Turing (T4, sm_75). — https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html
- TensorRT tactic selection is timing-based and can differ between builds; NVIDIA recommends locking GPU clocks and using timing caches to make builds more deterministic. — https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html
- The ONNX Runtime CUDA EP defaults cudnn_conv_algo_search to EXHAUSTIVE; onnxruntime-gpu 1.29+ targets CUDA 13.0 and cuDNN 9. — https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- The Tesla T4 has 2 NVDEC engines (4th-gen Turing); the GTX 1650 has 1. — https://developer.nvidia.com/video-encode-decode-support-matrix
- On an i5-12450H, OpenCV 4.13 cap.read() decodes 4K25 H.264 at only about 27-28 fps, while ffmpeg decode-only reaches about 244 fps; YUV-to-BGR conversion, not decoding, is the bottleneck (1080p cap.read() 56-130 fps under load). — local measurement (scratchpad dec_bench*.py, synthetic x264 clips)
- PyAV 18.1.0 (2026-08-12) requires Python >= 3.11; decord's last release was 0.6.0 on 2021-06-14; torchcodec 0.16.0 (2026-08-13) requires torch >= 2.11 and is BSD-3. — https://pypi.org/pypi/av/json ; https://pypi.org/pypi/decord/json ; https://pypi.org/project/torchcodec/
- The D-Fire fire/smoke dataset has 21,527 images with YOLO-format boxes and is released under CC0. — https://github.com/gaia-solutions-on-demand/DFireDataset
- VisDrone is CC BY-NC-SA 3.0 (academic use); BDD100K allows education, research and non-profit use only; COCO annotations are CC BY 4.0 while its images fall under Flickr terms. — https://huggingface.co/datasets/Voxel51/VisDrone2019-DET ; https://doc.bdd100k.com/license.html ; https://huggingface.co/datasets/shunk031/MSCOCO
- CUDA 13 drops Maxwell, Pascal and Volta but keeps Turing (sm_75), so current PyTorch CUDA-13 wheels still run on the T4. — https://github.com/pytorch/pytorch/issues/190385
- SAHI and Roboflow supervision are both MIT-licensed. — https://raw.githubusercontent.com/obss/sahi/main/LICENSE ; https://raw.githubusercontent.com/roboflow/supervision/develop/LICENSE.md
- Kaggle offers free 2x T4 (16 GB) GPUs with a weekly quota of about 30 GPU-hours, so the team can benchmark on the target GPU class. — https://www.kaggle.com/docs/efficient-gpu-usage
