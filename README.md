# WestCV — team Antigradient, WIUT Hackathon 2026, Computer Vision track

Traffic event detection (Part A) and causal accident anticipation (Part B) for a fixed road camera.

## Install and run

```bash
pip install -r requirements.txt          # Python 3.10-3.13; CUDA 12.6 wheels (NVIDIA driver >= 525)
bash weights/download.sh                 # optional: weights are committed; this re-fetches and verifies SHA-256
python run_submission.py --videos /data/test --out predictions.json --team Antigradient
python evaluate.py --pred predictions.json --validate-only
```

No internet is needed at run time. All weights are in `weights/` (YOLO26m 44 MB, YOLO26n 5.5 MB), and `src/trafficwatch/env.py` forces offline mode before any model library is imported. Models are loaded and warmed up when `solution.py` is imported, before the harness starts its per-video timer.

## Approach

```
video ─► one sequential CPU decode (ffmpeg, scaled to 1280 px, ~10 fps) ─► YOLO26m (COCO) ─► ByteTrack
      ─► track repair + kinematics (speed in box-size units) ─► per-video scene maps ─► rules per class
      ─► segment post-processing ─► events
frames from the harness ─► YOLO26n @ 640, ~10 fps ─► ByteTrack ─► pairwise time-to-collision ─► risk
```

| Component | Learned or rule-based |
|---|---|
| Object detection (vehicles, two-wheelers, people, animals) | Learned: pretrained YOLO26m / YOLO26n, COCO weights, not fine-tuned |
| Tracking | Algorithmic: ByteTrack-style two-stage IoU association, own implementation (`tracker.py`) |
| Carriageway mask, direction of each lane | Learned without labels from each video: where and in which direction moving vehicles drive (`scene.py`) |
| Pedestrian crossings | Hand-drawn once on a median frame of the sample videos (`configs/scene.json`) |
| `stopped_vehicle` | Rule: stationary ≥ 10 s on the carriageway while same-direction traffic keeps flowing (else: signal queue) |
| `jaywalking` | Rule: a pedestrian's feet on the carriageway outside crossings; riders and passengers excluded |
| `wrong_way` | Rule: sustained heading opposite to the lane's learned direction |
| `congestion` | Rule: per direction of travel, ≥ 4 vehicles crawling for ≥ 60 s |
| `accident` | Rule, precision first: abrupt stop while touching another road user at the same depth, approaching before contact |
| Part B risk | Rule: TTC between tracked pairs; ranking channel < 0.5, alarm ≥ 0.5 only for a consistent conflict with TTC ≤ 1 s |

Other classes are not predicted: a class predicted but absent from the test set scores 0 and lowers the macro average.

**Time budget.** Part A measures how long the harness's own frame loop (Part B) will take on the current machine and limits itself to `(3 − 1.25·h − 0.25) × duration`. Part B watches the deadline and switches itself off rather than overrun. On CPU-only machines Part B returns 0.

**Determinism.** Fixed seeds (`seed: 0`), deterministic cuDNN, sequential decoding, fixed batch size, sorted iteration. Decisions never depend on wall-clock time, except the budget brake, which logs when it fires.

## Repository layout

| Path | What it is |
|---|---|
| `solution.py` | The interface: `CLASSES`, `detect_events`, `RiskEstimator` |
| `src/trafficwatch/` | Pipeline: `video`, `detector`, `tracker`, `perception`, `tracks`, `scene`, `rules/`, `postprocess`, `pipeline` (Part A), `risk` (Part B) |
| `configs/pipeline.yaml` | All thresholds, in seconds and box-size units |
| `configs/scene.json` | Hand-drawn crossings (from labelme via `tools/annotation/labelme_to_scene.py`) |
| `weights/` | Model weights, `download.sh`, `SHA256SUMS` |
| `tools/` | Development tools: decode benchmark, perception cache, rule tuning, review video, labels → ground truth |
| `tests/` | Unit tests and synthetic-trajectory tests for every rule and for Part B |
| `run_submission.py`, `evaluate.py` | Organizers' harness and metric, unchanged |
| `docs/RUNBOOK_RU.md` | Step-by-step team runbook (Russian) |
| `docs/WestCV_plan_RU.pdf` | Analysis and plan (Russian) |

## Data and licences

| Item | Licence | Use |
|---|---|---|
| Ultralytics YOLO26m / YOLO26n weights and library | AGPL-3.0 | Detection. The repository is AGPL-3.0 as a consequence (`LICENSE`) |
| COCO 2017 (through the pretrained weights) | Annotations CC BY 4.0; images under Flickr terms | Not used directly |
| Sample videos from the organizers + our own labels | Organizers' terms | Threshold tuning and review; videos are not in the repository |
| imageio-ffmpeg (bundles an FFmpeg binary) | BSD-2 (package), GPL build of FFmpeg | Video decoding as an external process |
| OpenCV, NumPy, SciPy, PyYAML, lap | Apache-2.0 / BSD / BSD / MIT / BSD | Runtime |

No other datasets were used for training.

## Team

| Member | Role | Contributions |
|---|---|---|
| _to fill_ | | |
