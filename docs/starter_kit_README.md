# WIUT Hackathon 2026 — Computer Vision track: starter kit

Traffic events from a fixed road camera: **detect** them as time segments
(`[start_sec, end_sec, label]`) and, as a bonus, **anticipate** accidents with a
causal risk score. Three files; read the task description for the rules.

```
solution.py          <- the ONLY file you implement (CLASSES, detect_events, RiskEstimator)
run_submission.py    <- organizers' harness: folder of videos -> predictions.json   (do not modify)
evaluate.py          <- format check + the official metric                          (do not modify)
examples/            <- ground_truth.json and predictions.json in the exact format
requirements.txt     <- numpy + opencv for the harness; add your own deps to YOUR repo
```

## Quickstart

```bash
pip install -r requirements.txt
# 1. implement solution.py
# 2. label the sample videos yourselves -> my_labels.json (same shape as examples/ground_truth.json)
python run_submission.py --videos samples --out predictions_samples.json --team <your-team>
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
python evaluate.py --pred predictions_samples.json --validate-only        # format check without labels
```

## The interface (`solution.py`)

```python
CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]

def detect_events(video_path: str) -> list[list]:
    """Part A: [[start_sec, end_sec, label], ...]; label in CLASSES; same-class segments don't overlap."""

class RiskEstimator:
    def reset(self, meta: dict) -> None: ...            # meta: video_id, fps, width, height, n_frames
    def step(self, frame: np.ndarray, t_sec: float) -> float: ...   # BGR uint8 frame -> P(accident within 5 s)
```

`step` is called for **every frame in order** by the harness; it must not open the
video itself. Skipping frames internally and returning the last score is fine.
You may remove ids from `CLASSES`; never add.

## What we run (offline, one GPU, no internet)

```bash
pip install -r requirements.txt            # or: docker build -t team .
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --gt ground_truth.json
```

Time budget per video: **3 × its duration** for Part A + Part B together; a video
over budget or a crash scores as empty. Events with a bad label, bad times, or a
same-class overlap are dropped by the harness and listed in its log. Weights
≤ 5 GB, shipped in the repo or fetched once by `weights/download.sh` before the
offline run.

## predictions.json

```json
{
  "team": "your-team-name",
  "videos": {
    "test_001.mp4": {
      "events": [[12.4, 18.9, "accident"], [40.0, 43.5, "red_light"]],
      "risk":   [[0.00, 0.01], [0.04, 0.01], [0.08, 0.02]]
    },
    "test_002.mp4": {"events": [], "risk": []}
  }
}
```

`risk` is written by the harness (one `[t_sec, score]` per frame). Keys are file
names. Every test video must be present, even with `"events": []`.
Ground truth: `{"test_001.mp4": {"duration": 600.0, "fps": 25.0, "events": [[12.0, 19.0, "accident"]]}}`.

## Metric (exact code in `evaluate.py`)

**Part A.** Per class `c` and per tIoU threshold τ ∈ {0.3, 0.5, 0.7}: greedy
one-to-one matching by descending IoU; TP/FP/FN pooled over all videos; `F1_c(τ)`.
`Score_A = mean_c mean_τ F1_c(τ)`. Classes = those in the ground truth or in your
predictions (a class you predict that never occurs scores 0).

**Part B** (`accident` only; H = 5 s, W = 10 s, θ = 0.5). Frames in `[s−H, s)`
before an accident start `s` are positive; frames inside accidents and around
near-misses are ignored; the rest negative. `AP` = average precision over frames,
chance-normalised (`max(0, (AP_raw − r)/(1 − r))`, `r` = positive rate, so a
constant score gets 0). Alarms = runs of score ≥ θ (runs < 2 s apart merged),
alarm time = run start; an alarm in `[s−W, s)` of an unmatched accident matches it
→ `F1_alarm`; `mTTA` = mean of `s − alarm_time` (0 if unmatched).
`Score_B = 0.4·AP + 0.4·F1_alarm + 0.2·mTTA/W`.

**Model score** `M = 0.7·Score_A + 0.3·Score_B` (M = Score_A if the test set has no
accidents). Elimination score = 0.6·M + 0.25·Website + 0.15·Code.

## Tips

- Label the sample videos yourselves with the conventions from the task
  description and run `evaluate.py` against them. Without a dev set you are guessing.
- Detector + tracker → trajectories; most classes are rules on trajectories plus
  the scene layout. Learned models help most for `accident` / `near_miss`.
- Post-process segments: merge fragments, drop sub-second blips, then check F1@0.7.
- For Part B, time-to-collision from tracks is a strong simple signal; calibrate
  so that 0.5 means "probably within 5 s". A flat 1.0 scores ≈ 0.
- Print your runtime early; sampling every 2nd–5th frame is usually enough.
