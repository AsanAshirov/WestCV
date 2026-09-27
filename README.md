# ANTIGRADIENT — WIUT Hackathon 2026, Computer Vision Track

Traffic event detection (Part A) and accident anticipation (Part B) for the fixed camera over the
signalised intersection in the task's sample videos. Detector + tracker + scene registration + signal
reading from pixels + per-class rules; a causal time-to-collision risk for Part B.

## Install and run

Requirements: Python 3.10–3.14, an NVIDIA GPU with driver ≥ 525 (CUDA 12.6 build of PyTorch), about 4 GB
to download and 5–7 GB of disk (mostly PyTorch). On Linux, OpenCV needs the system libraries `libgl1` and
`libglib2.0-0` (present on standard Ubuntu GPU images; otherwise `sudo apt-get install -y libgl1 libglib2.0-0`).

```bash
pip install -r requirements.txt          # pulls torch/torchvision cu126 from download.pytorch.org
bash weights/download.sh                 # YOLO26m COCO weights (44 MB), checked by sha256
python run_submission.py --videos <folder with .mp4/.MP4> --out predictions.json --team westcv
python evaluate.py --pred predictions.json --validate-only
```

After `download.sh` everything runs offline (`YOLO_OFFLINE=1`). Models are loaded and warmed up when
`solution.py` is imported, before the harness starts a video's timer.
`predictions_samples.json` is the output of the command above on the four sample videos (Kaggle T4).

## Approach

```
video ─► PyAV decode (non-reference frames dropped = every 3rd frame, 1280 px)
      ├► YOLO26m (COCO, no fine-tuning) ─► ByteTrack ─► tracks of people and vehicles
      ├► median background ─► SIFT + RANSAC homography onto the reference view
      │        └► scene geometry (zebras, islands, sidewalks, stop line, solid lines) mapped into this video
      └► crops of the two signal heads facing the camera ─► lamp states (HSV, per-video threshold, fixed fallback)
                                   ▼
              per-class rules in scene units (box heights, seconds) ─► merged segments
```

| Part | Learned or rule-based | Where |
|---|---|---|
| Road-user detection | learned: Ultralytics YOLO26m, COCO-pretrained, used as is | `src/westcv/detector.py` |
| Tracking | algorithmic: ByteTrack (Ultralytics implementation) | `src/westcv/tracking.py` |
| Scene registration | algorithmic: SIFT, ratio test, RANSAC homography | `src/westcv/registration.py` |
| Signal state | rule-based: lamp colour scores, per-video Otsu threshold; a fixed threshold (set on the samples) for a lamp that does not switch within the clip | `src/westcv/signals.py` |
| Events | rule-based; thresholds tuned on our own labels of the 4 samples | `src/westcv/rules.py`, `scene.py` |
| Part B risk | rule-based on online tracks: time to collision, hard braking | `src/westcv/risk.py` |

Classes we emit: `red_light`, `stop_line`, `jaywalking`, `failure_to_yield`, `stopped_vehicle`,
`congestion`, `solid_line_crossing`. The others never occur or are not reliably separable on the samples;
emitting a class that is absent from the test set adds a zero to the macro-F1, so they are left out.

Scene geometry (`scene/geometry.json`) and the signal-head boxes (`scene/signal_heads.json`) are drawn
once on the reference frame (median of sample C3896). The camera moved by up to ~100 px between
recordings and shakes in the first seconds, so every video is registered first. If a video does not
register (another camera), only `stopped_vehicle` and `congestion` run, on a carriageway learnt from
vehicle trajectories.

**Part B** is causal: it only sees the frames the harness passes to `step`. On every 6th frame the
Part A detector (already loaded) runs on the frame subsampled to 640 px, ByteTrack follows the road
users online, and the risk rises when two of them are on a course to contact within about a second
(time to collision in per-axis box units; closing slower than 1 box/s, ordinary lane following and
boxes that already overlap in this oblique view are not conflicts) or when a vehicle brakes hard next
to another road user. The samples contain no accident, so how early it warns before a real crash could
not be measured; the filters were chosen from the false alarms on the samples.

**Runtime.** On a Kaggle T4 with 4 vCPUs, Part A takes 1.0–1.2× the video's duration (the first read of
a file is slower) and the harness's own Part B decode of every 4K frame about 1.4×; our Part B detection
adds about 0.1× (limit 3× for both). Part A stops decoding at 1.35× duration and the rules run on what
was decoded. Part B stops detecting only if the projected finish would pass 2.75× duration.

## Data and licences

| Item | Licence | Use |
|---|---|---|
| YOLO26m weights (Ultralytics) | AGPL-3.0 | detector at inference |
| COCO (via the pretrained weights) | CC BY 4.0 (annotations) | not used directly |
| Task sample videos | organizers' | our labels (`labels/`) for tuning rule thresholds; not redistributed |

No other datasets and no other footage of this camera are used. Open-source code used: Ultralytics
(YOLO26, ByteTrack; AGPL-3.0), OpenCV, PyAV, NumPy, PyTorch. The code was written with the help of an
AI coding assistant (Claude Code); no hosted model is called at inference.

## Determinism

Seeds are fixed (`solution.py`). The only randomised step is RANSAC in scene registration
(`cv2.findHomography`), which is repeatable because OpenCV seeds its generator with a fixed value. Two
runs on the same machine give identical `predictions.json` (checked). By design, two safety brakes depend
on wall-clock time: Part A stops decoding at 1.35× the duration, and Part B stops detecting when it
would overrun. Neither triggers on a T4, but on a much slower machine outputs can then differ between
runs. The detector runs in FP16 on T4/RTX GPUs and in FP32 on GTX 16xx cards (FP16 is broken there),
which can shift single detections between GPU types.

## Evaluation on the samples

We labelled all four sample videos in a self-hosted CVAT (event = track from its first frame to the
frame it is switched off; conventions in `docs/annotation_guide.md`); `tools/annotation/cvat_to_gt.py`
converts the exports to `ground_truth.json` format (`labels/`). `tools/eval/dev_eval.py` scores the
rules against these labels with the official `evaluate.py` matching:

| Class | F1@0.3 | F1@0.5 | F1@0.7 | Mean | TP/FP/FN @0.5 |
|---|---|---|---|---|---|
| jaywalking | 0.667 | 0.444 | 0.286 | 0.466 | 14/18/17 |
| stop_line | 0.545 | 0.545 | 0.364 | 0.485 | 3/4/1 |
| red_light | 0.667 | 0.667 | 0.000 | 0.444 | 1/0/1 |
| failure_to_yield | 0.320 | 0.320 | 0.240 | 0.293 | 4/12/5 |
| stopped_vehicle | 0.889 | 0.889 | 0.667 | 0.815 | 4/0/1 |
| congestion | 0.421 | 0.421 | 0.316 | 0.386 | 4/6/5 |
| solid_line_crossing | 0.222 | 0.111 | 0.111 | 0.148 | 1/7/9 |

Score A over the 9 labelled classes (including `illegal_turn` / `illegal_u_turn`, which we do not emit)
is 0.338. The numbers are in-sample (thresholds were tuned on the same four clips) and follow one
labeller's conventions, so they overstate the test score. Two people labelling the same video agree at
segment F1 0.54–0.60 on jaywalking and 0.80 on failure_to_yield, which bounds what any model can reach
against human labels. `predictions_samples.json` holds our output on the four samples.

## Repository

| Path | What it is |
|---|---|
| `solution.py` | Submission interface (`detect_events`, `RiskEstimator`) |
| `src/westcv/` | The pipeline: video, detector, tracking, registration, signals, scene, rules, segments |
| `scene/` | Reference-view geometry, signal heads and templates, reference SIFT features |
| `weights/download.sh` | Fetches the detector weights |
| `tests/` | Unit tests (`pip install pytest && pytest tests`) |
| `run_submission.py`, `evaluate.py` | Organizers' harness and metric, unmodified |
| `labels/` | Team labels of the samples (CVAT exports converted to `ground_truth.json` format) |
| `tools/` | Team tools: CVAT setup and converters, label lint, evaluation and tuning, review renders, benchmarks |
| `docs/` | Annotation guide and illustrated handbook for the labellers (Russian) |
| `examples/` | The starter kit's example `ground_truth.json` and `predictions.json` |

## Team

| Member | Role | Did |
|---|---|---|
| Azam Khodjimetov | Team lead, web platform | Team website, live demo, project coordination |
| Diyora Fatakhova | Data and annotation lead | Labelling of the sample videos in CVAT, EDA |
| Asan Ashirov | ML and computer vision | Pipeline, rules, signal reading, Part B, evaluation, submission package |
