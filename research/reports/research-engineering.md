<!-- source: deep-research workflow wf_294a4b01-fc2, agent research:engineering -->

# Engineering, packaging, reproducibility, dev set, compliance and risks

Scope: how to make the submission install and run on a clean offline T4 machine, stay deterministic and inside the time budget, how to build a dev set, what the licence rules mean, the main risks, and a plan for three people. All versions were checked on 2026-09-25.

---

## 0. Decisions at a glance

1. **Ship a pip `requirements.txt` as the primary path and a tested `Dockerfile` as the backup.**
   - Pin `torch==2.14.0+cu126` and `torchvision==0.29.0+cu126` from the PyTorch cu126 index. These CUDA 12.6 wheels need an NVIDIA driver ≥525. The CUDA 13 wheels, which PyPI serves by default since PyTorch 2.11, need a driver ≥580. The organizers' driver version is unknown, so cu126 is the safer choice.
   - Use `ultralytics-opencv-headless` and `opencv-python-headless`, never `opencv-python`. The normal OpenCV package needs `libGL.so.1`, which headless servers often lack.
   - Add `lap` to the requirements. Without it, Ultralytics' ByteTrack tries to pip-install it at runtime, which fails offline.
2. **Force offline mode in code** (environment variables set before any imports) and **rehearse offline** with a socket-blocking `sitecustomize.py` and `docker run --network none`.
3. **Store small weights in git** (under 50 MiB each). Put anything larger in **GitHub Release assets**: up to 2 GiB per file, no bandwidth limit. Check every file with SHA-256. Do not use Git LFS: its free bandwidth quota is charged to the repo owner and LFS is switched off when the quota runs out.
4. **Make determinism structural.**
   - Turn off cuDNN benchmark mode and enable deterministic algorithms with `warn_only=True`.
   - Decode frames sequentially only.
   - Sort anything you iterate over.
   - Never let wall-clock time decide what the model outputs, except in an emergency cutoff that should never fire.
5. **Plan compute from the video's metadata (a deterministic cost model), with a real-time emergency cutoff as the last line of defence.**
   - Aim for about 1.0–1.2× video duration.
   - Stop starting optional stages after 1.8×.
   - Return whatever you have at about 2.4×. The hard limit is 3×.
6. **`detect_events` and `step` must never raise or return bad values.**
   - Wrap each stage in its own try/except.
   - Clean every output: plain Python floats only (`np.float32` makes `json.dump` crash), no NaN, values clamped, same-class segments merged, labels checked against the official list.
7. **Build a dev set on day 1–2.** Each sample video gets labelled independently by two people, the disagreements are settled, and agreement is measured with the official `evaluate.py`. Tune thresholds with leave-one-video-out.
8. **Have a runnable, validated submission by day 1** (the default template plus packaging), tag it, then improve from there.

---

## 1. Clean-machine install (Code: "Runs as submitted" = 40%; a package that doesn't run zeroes the model score)

### 1.1 Current PyTorch and CUDA facts (checked 2026-09-25)
- **Latest release:** PyTorch 2.14.0, published 2026-09-02, with Python 3.10–3.14 classifiers.
- **Python 3.10:** 2.14 is the last release with Python 3.10 wheels. 2.15 goes GA on 2026-10-28 with Python 3.11 as the minimum.
- **Turing support:** 2.15 drops the CUDA 12.6 and 13.0 builds and the Maxwell, Pascal and Volta GPUs. **Turing (sm_75, the T4) stays supported** through CUDA 13.2 and 13.4. So sm_75 is safe on every current release.
- **Default wheels:** since 2.11, the default PyPI CUDA wheels are CUDA 13.0. Driver requirements: CUDA 13.x needs ≥580; CUDA 12.x works from ≥525 through minor-version compatibility.
- **Wheel check:** `uv pip compile` cross-resolved `torch 2.14.0+cu126`, `torchvision 0.29.0+cu126`, `numpy`, `scipy`, `lap`, `ultralytics-opencv-headless` and `opencv-python-headless 4.13.0.92` for Linux x86_64 on Python 3.10 and 3.12. It pulled only the headless OpenCV plus the `nvidia-*-cu12` 12.6 runtime wheels. `pip --dry-run` also found `torch 2.14.0+cu126` and `torchvision 0.29.0+cu126` wheels for manylinux x86_64 on cp310, cp312 and cp313.
- **Python-version traps:**
  - NumPy 2.3+ has no Python 3.10 wheels, and NumPy 2.5.x needs Python ≥3.12.
  - SciPy 1.16 needs Python ≥3.11.
  - Pinning your local Python 3.14 versions (numpy 2.4) would break `pip install` if the organizers use Python 3.10.

### 1.2 Recommended `requirements.txt` (top-level pins, resolution-tested)
```text
# Tested on Ubuntu x86_64, Python 3.10-3.13, NVIDIA driver >= 525 (CUDA 12.6 libs ship inside the wheels)
--extra-index-url https://download.pytorch.org/whl/cu126
torch==2.14.0+cu126
torchvision==0.29.0+cu126
ultralytics-opencv-headless==8.4.162   # NOT "ultralytics": that one pulls opencv-python -> libGL.so.1 errors
opencv-python-headless==4.13.0.92
numpy==2.2.6 ; python_version < "3.11"
numpy==2.3.5 ; python_version >= "3.11"
scipy==1.15.3 ; python_version < "3.11"
scipy==1.16.3 ; python_version >= "3.11"
lap==0.5.13                            # otherwise Ultralytics' tracker auto-pip-installs it at runtime
PyYAML==6.0.3
```
Rules for this file:
- **Pin the local version label (`+cu126`) explicitly.** A bare `torch==2.14.0` also matches `2.14.0+cu126` under PEP 440, and local versions sort higher. Explicit pinning removes the ambiguity. (Checked with `packaging`.)
- **No package that needs compiling.** No `flash-attn`, no `pycocotools` from source, no git+https dependencies.
- **Never install both `ultralytics` and `ultralytics-opencv-headless`.** Both provide the `ultralytics` module.
- **Avoid `onnxruntime-gpu` unless you need it.** From 1.27 its PyPI builds use CUDA 13.0 and would clash with torch cu126. If you need ONNX Runtime, pair it with torch cu130 and require a driver ≥580.
- **Transitive dependencies still float.** Pinning them per Python version is fragile. Once the organizers confirm the exact Python version, freeze the full environment from the tested machine into `constraints.txt` and add `-c constraints.txt` to `requirements.txt`.
- **Check the file for other Python versions without installing:**
  ```
  uv pip compile requirements.txt --python-version 3.10 --python-platform x86_64-manylinux_2_28 --index-strategy unsafe-best-match
  ```
  `pip --dry-run --python-version` is **not enough**: pip evaluates environment markers against the interpreter it runs on. Confirmed locally: it tried numpy 2.3.5 for a 3.10 target and dropped the Linux-only `nvidia-*` dependencies.

### 1.3 Dockerfile (backup path, also the reference environment)
```dockerfile
FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
ENV PIP_NO_CACHE_DIR=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    YOLO_OFFLINE=1 YOLO_AUTOINSTALL=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN bash weights/download.sh          # weights are baked into the image at build time (the build has internet)
```
- **Image size:** about 3.97 GB compressed (Docker Hub, 2026-09-02).
- **Host requirements:** the host needs the NVIDIA driver and nvidia-container-toolkit. The run needs `--gpus all`. Without it, Docker silently runs on CPU. Handle that case (section 4).
- **A leaner alternative:** `python:3.12-slim` plus the same requirements. The pip torch wheels bring their own CUDA libraries; only the driver is injected by the toolkit.
- **Exact run command for the README:**
  ```
  docker run --rm --gpus all -v /data/test:/data/test -v $PWD/out:/out team python run_submission.py --videos /data/test --out /out/predictions.json
  ```

**Which path is less risky?**
- pip: the risk is in Python version, driver and system libraries. We mitigate it with cu126, headless OpenCV, markers and tests.
- Docker: the risk is that the toolkit or the `--gpus` flag is missing on their side.

Ship both. Mark pip as primary unless the organizers say they prefer Docker, and test both on every release candidate.

### 1.4 Rehearsal matrix (run before each tag)
| Test | Where | What it proves |
|---|---|---|
| `pip install -r requirements.txt` in fresh `python:3.10-slim` and `python:3.12-slim` containers (CPU) | local Docker Desktop, or GitHub Actions | the install resolves; no compiling; no libGL problem |
| Fresh clone at the tag → venv → install → `download.sh` → run on the samples with network blocked | WSL2 Ubuntu + GTX 1650, `docker run --gpus all --network none` | truly offline; Linux paths are case-sensitive; no CRLF breakage |
| The same on a T4 | Kaggle or Colab (fresh venv) | wall-clock time against the 3× budget; FP16 behaviour on a real T4 |
| Two runs, then `compare_predictions.py` | T4 | determinism |
| `evaluate.py --validate-only` | anywhere | output format |

Windows traps to guard against:
- Add a `.gitattributes` with `*.sh text eol=lf`. A `download.sh` saved with CRLF fails on Linux with `$'\r': command not found`.
- Run `git update-index --chmod=+x weights/download.sh`.
- Linux paths are case-sensitive; Windows paths are not.
- Always pass `cv2.CAP_FFMPEG` explicitly so the video backend is the same on every OS.

---

## 2. Offline operation

### 2.1 What silently reaches the internet
- **Ultralytics:**
  - `YOLO("yolo26s.pt")` downloads the weights by name if that relative path doesn't exist. **Always pass an absolute path built from `Path(__file__)`**, because the harness's working directory is unknown.
  - The `is_online()` check does DNS lookups; `YOLO_OFFLINE=1` makes it return False immediately.
  - `check_requirements()` pip-installs missing packages when online and `YOLO_AUTOINSTALL` is on. That is how `lap` gets installed by `trackers/utils/matching.py`.
  - `check_font()` downloads Arial.ttf when plotting labels. Only render with your own OpenCV drawing code, or ship the font.
  - The `sync` setting sends analytics and crash reports to Sentry. Run `settings.update({"sync": False})`.
- **Hugging Face:** set `HF_HUB_OFFLINE=1` (no HTTP; cached files only), `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1` / `DO_NOT_TRACK=1`, and set `HF_HOME` to point inside `weights/`. These variables are **read when `huggingface_hub` is imported**, so set them first.
- **torch.hub and timm pretrained weights:** set `TORCH_HOME` to `weights/torch` and never pass `pretrained=True` at inference time. Load state dicts from files.

### 2.2 `src/<pkg>/env.py`: the first import in `solution.py`
```python
import os, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for k, v in {"YOLO_OFFLINE": "1", "YOLO_AUTOINSTALL": "false", "YOLO_VERBOSE": "false",
             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
             "DO_NOT_TRACK": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "MPLBACKEND": "Agg",
             "TOKENIZERS_PARALLELISM": "false",
             "YOLO_CONFIG_DIR": os.path.join(tempfile.gettempdir(), "tw_yolo"),  # home dir may be read-only
             "HF_HOME": str(ROOT / "weights" / "hf"), "TORCH_HOME": str(ROOT / "weights" / "torch")}.items():
    os.environ.setdefault(k, v)
```
Note: setting `PYTHONHASHSEED` from inside Python has **no effect on the running interpreter**. Ultralytics' `init_seeds` does this; it only affects child processes. Rely on sorting instead (section 3).

### 2.3 Weights hosting and `download.sh`
- GitHub blocks files over 100 MiB and warns above 50 MiB. Release assets: under 2 GiB each, up to 1000 per release, no limit on total size or bandwidth.
- Git LFS free tier: 10 GiB storage and 10 GiB bandwidth per month. Clones by other people are charged to the owner, and **LFS is disabled when the quota runs out**. Judges cloning the repo could break it, so do not use it.
- The 5 GB cap is roughly **2.5 B parameters in FP16** (5×10⁹ bytes ÷ 2 bytes per parameter) across all models. Any VLM must be ≤2 B parameters in FP16 or quantized (a cross-cutting constraint for the model agents).

```bash
#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")"
BASE="https://github.com/<org>/<repo>/releases/download/weights-v1"
MIRROR="https://huggingface.co/<org>/<repo>/resolve/main"
while read -r sha name; do
  [ -z "${sha:-}" ] && continue
  if [ -f "$name" ] && echo "$sha  $name" | sha256sum -c --status; then echo "ok $name"; continue; fi
  curl -fL --retry 5 --retry-delay 3 -o "$name.part" "$BASE/$name" || curl -fL --retry 5 -o "$name.part" "$MIRROR/$name"
  mv "$name.part" "$name"; echo "$sha  $name" | sha256sum -c
done < SHA256SUMS
```
- The script is idempotent, safe to run again, checked against SHA-256, and has a mirror.
- At import time, check that each weight file exists (a hash check of about 100 MB takes well under a second). If one is missing, log a clear error and fall back to empty predictions instead of crashing.

### 2.4 Offline rehearsal helper (tested locally)
Put `tools/netguard/sitecustomize.py` on `PYTHONPATH`. It replaces `socket.socket.connect`, `connect_ex`, `getaddrinfo` and `create_connection` with functions that print a stack trace and raise. Any hidden network call then fails loudly and shows where it came from. Tested: `urllib` is blocked with `NETGUARD: network access attempted`. Combine it with `docker run --network none` for the full rehearsal.

---

## 3. Determinism (a rule, and 25% of the Code score)

Checklist:
- **Seeds:** `random.seed`, `np.random.seed` and `torch.manual_seed` (which also seeds CUDA), all set to a fixed number.
- **Torch flags:** `torch.backends.cudnn.benchmark=False`, `cudnn.deterministic=True`, `torch.use_deterministic_algorithms(True, warn_only=True)`. Use `warn_only`: some ops have no deterministic kernel, and a hard error would crash inference.
- **`CUBLAS_WORKSPACE_CONFIG=:4096:8`:** set it before CUDA initialises.
- **Documented limit:** PyTorch does not promise identical results across releases, platforms, or CPU vs GPU. Generate `predictions_samples.json` on a T4 with the exact pinned stack, and say in the README which GPU, driver and torch version produced it.
- **Frame access:** sequential decode only. No `CAP_PROP_POS_FRAMES` seeking in the main pass. Seeking can land on keyframes for some files. The local test on a clean file matched, but don't depend on it.
- **Threading:** one decode thread feeding a first-in-first-out queue is fine. Do **not** process Part B asynchronously ("return the latest available score"): the scores would then depend on timing.
- **Batching:** keep a fixed batch size. Varying batch composition can change the cuDNN algorithm choice and the numerical results.
- **Hash-order randomness:** iteration over sets or dicts keyed by strings changes between processes. Use `sorted()` wherever order affects greedy matching, merging or tie-breaking.
- **Time-dependent logic:** the watchdog must not change the output in normal runs (section 4).
- **ONNX Runtime (if used):** set `cudnn_conv_algo_search` to `DEFAULT` or `HEURISTIC`. `EXHAUSTIVE` gives run-to-run differences of about 1e-6.
- **TensorRT:** engines are tied to the GPU and TensorRT version and take minutes to build. Avoid them unless the organizers confirm that `download.sh` runs on the evaluation GPU and may build the engine there.
- **`torch.compile`:** avoid it at evaluation. The compile time counts against the budget and adds another source of variation.
- **Rounding:** round output times to 3 decimals and scores to 4.
- **Check it:** run twice, then run `tools/compare_predictions.py run1.json run2.json` (written and tested: tolerance 1e-3 s on events, 1e-4 on risk; exit code 1 on any difference). Run this in CI on a short clip and on the T4 before each tag.

---

## 4. Time budget and crash safety

### 4.1 Measured decode costs (dev laptop, i5-12450H, 12 threads; synthetic 1080p H.264 clip; laptop timings vary by up to ±50%)
- OpenCV `read()` of every frame (with BGR conversion): **52–65 fps**. One cold first run gave only 31 fps.
- OpenCV `grab()` only: **180–330 fps**. `grab()` for every frame plus `retrieve()` for every 3rd: **143–196 fps**.
- PyAV with `thread_type="AUTO"`, converting every frame to BGR: **129–151 fps**; converting every 3rd frame: **237–289 fps**.
- Lesson: the YUV→BGR conversion and copy is the expensive part. **Never `retrieve()` or convert frames you won't use.**
- OpenCV 4.13 pip on Windows **could not encode H.264** (the openh264 DLL is missing) and fell back to `mp4v`, which browsers generally don't play. Render website and demo videos through ffmpeg with `libx264 -pix_fmt yuv420p -movflags +faststart`.

### 4.2 Budget math (estimate; replace with T4 measurements in week 1)
Worked example: a 10-minute video (15,000 frames at 25 fps) has a 1,800 s budget. Each cost below is a fraction of the video's duration:
- **Harness decoding every frame for Part B:** about 25/55 ≈ **0.45×**, if it uses `cv2.read` at 1080p. This is not under our control.
- **Our Part A decode at stride 3 (PyAV):** about **0.1×**.
- **Detector:** YOLO26s runs at 2.5 ms on a T4 with TensorRT (official). PyTorch FP16 eager with pre- and post-processing is estimated at about 8–10 ms.
  - Part A at stride 2: 12.5 frames per video-second × 10 ms ≈ **0.125×**.
  - Part B at stride 3: ≈ **0.08×**.
- **Tracking and rules:** about **0.05×**.
- **Nominal total: about 0.8×,** giving about a 3.7× safety margin.
- **A VLM checking about 20 candidate clips** at 3–8 s each adds **0.03–0.09×** on a 10-minute video, or about 0.2–0.5× on a 1-minute clip, and T4 VLM speed is itself only an estimate. Always cap the number of candidates.

### 4.3 Budget policy
```python
class Budget:
    def __init__(self, duration_s, factor=3.0):
        self.t0, self.limit = time.perf_counter(), duration_s * factor
    def frac(self):  # fraction of the hard limit used so far
        return (time.perf_counter() - self.t0) / self.limit
# Plan chosen from metadata only (deterministic): stride, input size and optional stages come from
# n_frames, width x height and a cost table measured once on a T4.
plan = make_plan(meta, cost_table)          # e.g. stride=2, imgsz=640, vlm_max_clips=min(20, ...)
# Emergency cutoffs (should never fire in normal runs; log loudly if they do):
#  - Part A: no optional stage starts after 0.6 of the limit (1.8x duration); after 0.8 (2.4x),
#    stop and return the events found so far.
#  - Part B step(): if cumulative step time > 0.5 * t_sec, double the stride (and log it).
```
- **Model loading:** load models once, into module-level singletons shared by Part A and Part B. The harness may build a new `RiskEstimator` for each video, so never load weights in `__init__`. Load eagerly at import inside a try/except, and fall back to lazy loading. We don't know whether import time counts toward the first video's budget; ask the organizers.
- **Threads:** call `torch.set_num_threads(4)` and control OpenCV/PyAV decode threads so the 8 cores aren't oversubscribed.
- **No GPU:** if `torch.cuda.is_available()` is False (for example, Docker started without `--gpus`), switch to a CPU plan: larger stride, the nano model, no VLM. The run is slower but not empty.

### 4.4 Never raise, never return junk
- Each stage (decode, detect, track, per-class rules, optional models) runs in its own try/except and adds its results to a shared list of partial results.
- Catch `torch.cuda.OutOfMemoryError`: call `empty_cache()` and retry once with half the batch size.
- **Read `run_submission.py` first.** If it enforces the time limit with an exception (for example via `signal.alarm`), your broad `except Exception` could swallow it; re-raise it.
- A sanitizer was written and tested. It:
  - casts to Python `float`,
  - drops NaN, infinity and unknown labels,
  - clamps to `[0, duration]`,
  - drops segments shorter than a minimum length,
  - merges overlapping same-class segments (the FAQ convention),
  - rounds and sorts the output.

  In the test, `json.dumps(np.float32)` raised `TypeError` and `json.dumps(nan)` produced invalid `NaN`. Both are real ways to crash the harness or have `evaluate.py` reject the file.
- `step()` always returns `min(1, max(0, float(x)))` and 0.0 on any error. It keeps no frames and runs synchronously at a fixed stride.
- Fuzz-test these inputs: a 1-frame video, a 0.5 s video, a corrupt file, a portrait or odd resolution, 30 fps and variable frame rate, a 10-minute clip.

---

## 5. Local development setup
- **Local hardware:** GTX 1650, driver 592.82, compute capability 7.5, the **same sm_75 architecture as the T4**, so kernel compatibility tests carry over. It has **no Tensor Cores** (it uses dedicated FP16 units instead), so FP16 speed is not representative. Measure timing only on a real T4.
- **Python environment:** don't develop on Python 3.14. `uv` 0.10.6 and CPython 3.12.12 are already installed:
  ```
  uv venv --python 3.12 .venv && uv pip install -r requirements.txt --index-strategy unsafe-best-match
  ```
  The cu126 wheels also exist for `win_amd64`. For Linux parity, use the existing WSL2 Ubuntu 22.04 distribution or Docker Desktop (29.4.3 is installed) with `--gpus all`. Install the NVIDIA driver on Windows only, never inside WSL.
- **With 4 GB of VRAM:** inference with n/s/m detectors and small fine-tunes at batch 8–16 are feasible. Do larger training in the cloud.
- **Cloud GPUs:**
  - Kaggle: about **30 GPU-hours per week per account** (P100 16 GB or 2×T4). Three teammates means about 90 h per week.
  - Colab free: up to 12 h per session, usage limits change and a GPU is not guaranteed.
  - Use Kaggle's T4 as the timing reference.
- **Development cache:** save per-video detections and tracks to `.npz`, keyed by video hash and a hash of the detector config. Rule changes can then be re-run in seconds. This cache is for development only, never for the submission.

---

## 6. Dev-set annotation workflow

### 6.1 Tools
| Tool | Pros | Cons |
|---|---|---|
| **VIA 3** (VGG Image Annotator), BSD-2 licence, a single offline HTML file | temporal segments with attributes; exports "Only Temporal Segments as CSV" | fiddly frame-accurate seeking |
| **Label Studio**, `<TimelineLabels>` | multi-user; exports frame ranges `{"ranges":[{"start":s,"end":e}],"timelinelabels":[...]}` | needs a local server; check whether frames are 0- or 1-indexed on a known clip |
| **mpv + CSV** (`--osd-fractions`; `.`/`,` step frame by frame) | fastest and frame-accurate; no setup | manual CSV typing |
| CVAT | strong for boxes and tracks | less suited to temporal segments |

**Recommendation:** use VIA 3, or mpv + CSV, and store labels as `dev_labels/<annotator>.csv` with columns `video,start_sec,end_sec,label,notes`. Convert them with `tools/labels_to_gt.py`. It writes the official `{"file.mp4": {"duration", "fps", "events"}}` JSON, taking duration and fps from ffprobe/PyAV, and runs the same sanitizer as the submission.

### 6.2 Annotation guide (`docs/annotation_guide.md`)
- Copy each class's official start and end convention word for word, then add team-level details. Examples:
  - "accident start = the first frame where contact is visible, stepping frame by frame"
  - "stopped_vehicle: label only if stopped ≥10 s; start = the moment of stopping; exclude signal queues"
  - "congestion: one segment per direction"
- Times are `frame_index / fps`, with frame 0 = 0.0 s.
- Record open questions (for example, congestion vs signal queue) and ask the organizers (section 10).

### 6.3 Double annotation and agreement
- For each sample, annotator A and annotator B work independently, without seeing model output. After that blind pass, a candidate list from the detector and tracker may be checked, to catch missed events.
- **Agreement:** score B as predictions against A as ground truth with the official `evaluate.py`. Target F1 ≥0.8 at τ=0.5 per class. Anything below that is settled in a joint review, and the guide is updated.
- **Boundary tolerance** is the key reason boundary conventions matter. The largest boundary error that still counts as a match at IoU τ, for a ground-truth segment of length L:

| L | τ=0.3 shift / both ends wide | τ=0.5 | τ=0.7 |
|---|---|---|---|
| 2 s | 1.08 / 2.33 s | 0.67 / 1.00 s | **0.35 / 0.43 s** |
| 5 s | 2.69 / 5.83 s | 1.67 / 2.50 s | **0.88 / 1.07 s** |
| 10 s | 5.38 / 11.7 s | 3.33 / 5.00 s | **1.76 / 2.14 s** |

Formulas: a shift of the whole segment gives IoU = (L−d)/(L+d); both ends too wide by d gives L/(L+2d).

Consequences:
- A stride of 3 at 25 fps adds up to 0.12 s of error per boundary. That is a third of the tolerance for a 2 s near-miss at τ=0.7.
- So refine boundaries: take the coarse pass at stride 2–3, then re-decode ±1–2 s around each candidate boundary at stride 1. Random access is allowed in Part A.
- Annotator disagreement of 0.3 s already caps the achievable F1@0.7 on short events.

### 6.4 Avoid overfitting a few samples
- **Leave-one-video-out:** tune on the other videos, predict the held-out one, pool all held-out predictions, then compute one pooled score (the official metric also pools true positives, false positives and false negatives).
- Keep at most about 3 tunable parameters per class, each on a coarse grid of about 5 values. Pick the middle of the flat optimum, not the single best point.
- Express parameters in physical units (seconds, fractions of lane width).
- Test videos contain events the samples don't have, so add **synthetic-trajectory unit tests** for every class: a hand-built track that should produce exactly one segment with the expected start and end.
- **Cross-cutting:** a class you predict that never occurs in the test set **enters C with F1 = 0**. Keep a per-class `enabled` switch in `configs/pipeline.yaml`, and ship a class only if its dev precision is high or its synthetic tests are strong.

---

## 7. Compliance and licences
- **Ultralytics is AGPL-3.0.** Trained models and connected code are covered too. A public repo satisfies this: add `LICENSE` (AGPL-3.0). Because the demo serves the model over a network, AGPL §13 applies: link the repository source from the demo page. Permissive alternatives exist if you want to avoid AGPL (the models agent covers these).
- **Datasets:** non-commercial research licences suit a non-commercial hackathon, but:
  - list every dataset, its URL and its licence in a README table;
  - never redistribute dataset media in the repo or on the website (BDD100K, for example, prohibits redistribution);
  - the DoTA repository is MIT-licensed, but its videos come from YouTube, so treat them as research-only.
- **Closed APIs:** allowed for writing code, the website and the report; never at inference. Using a closed model to pre-label the sample videos is a grey area. Ask first, and assume "no" until answered.
- **Footage:** don't collect footage of the same camera from other sources.
- **Website privacy:** show only what the videos already show. No face or plate crops, no plate OCR, no re-identification galleries.
- **Part B causality:** keep it a separate module that reads only `step` arguments. Add a test that monkeypatches `cv2.VideoCapture` and `open` during streaming to prove the video file is never opened.

---

## 8. Repository structure (Code: 20% structure + 15% engineering judgement)
```
solution.py              # thin: CLASSES, detect_events(), RiskEstimator -> delegate to src/trafficwatch
run_submission.py, evaluate.py   # unchanged (byte-identical; add a CI check against the starter-kit hash)
requirements.txt, Dockerfile, .gitattributes, LICENSE, README.md
weights/  download.sh  SHA256SUMS  (small *.pt in git)
configs/  scene.yaml (lanes, directions, stop lines, crossings, signal ROI from camera.md) · pipeline.yaml (strides, thresholds, enabled classes)
src/trafficwatch/
  env.py  runtime/{budget.py, safety.py, logging.py}
  io/video.py            # sequential decoder, metadata, exact duration
  perception/{detector.py, tracker.py}
  scene/{geometry.py, signal_state.py}
  features/kinematics.py # speed, heading, acceleration, TTC
  rules/<one module per class>.py   # common interface: tracks+scene -> candidate segments
  events/postprocess.py  # merge, minimum duration, boundary refinement, class gating
  risk/estimator.py      # causal Part B only
  viz/{render.py, timeline.py}      # shared with the website demo
tools/  labels_to_gt.py  compare_predictions.py  netguard/  benchmark.py  rehearsal.sh
tests/  test_rules_synthetic.py  test_sanitize.py  test_causality.py  test_format.py
training/  (scripts + configs for every trained weight; dataset download instructions)
notebooks/ (EDA only; never the only copy of any logic)
dev_labels/  predictions_samples.json
```
- **CI (GitHub Actions, free for public repos):** lint with ruff, run the unit tests, and run a CPU smoke test on a 5 s synthetic clip under netguard, followed by `--validate-only`.
- **Logging:** log per-video stage timings as JSON lines to stderr or the temp directory. Never write into the test folder or the repo. A failed log write must not crash anything.
- **`predictions_samples.json`:** regenerate it from the **tagged commit** on a T4, and state the generating environment in the README.

---

## 9. Risk register (top 12)
| # | Failure mode | Likelihood / impact | Mitigation |
|---|---|---|---|
| 1 | Install fails: CUDA/driver mismatch, no wheel for their Python, libGL, a package that compiles | M / **fatal** | cu126 wheels; headless OpenCV; markers for 3.10; test in `python:3.10-slim` and `3.12-slim`; Docker backup; ask for driver and Python versions |
| 2 | Hidden internet access: weights by name, font, lap auto-install, HF metadata | M / fatal or hang | `env.py`; absolute paths; `lap` pinned; netguard + `--network none` rehearsal |
| 3 | Over the 3× budget: VLM, 10-minute clips, CPU fallback, slow decode | M / video counts as empty | cost-model plan; watchdog; no retrieving unused frames; tested on a 10-minute clip on a T4 |
| 4 | Exception, NaN or `np.float32` in the output | M / video counts as empty or file rejected | per-stage try/except; sanitizer; fuzz tests |
| 5 | Non-determinism | M / Reproducibility score, possible rule breach | flags and seeds; sorted iteration; no time-based decisions; double-run diff in CI |
| 6 | Predicting classes absent from the test set (class enters C with 0) | H / −1/\|C\| of Score A each | per-class kill switches gated by dev precision |
| 7 | Boundaries don't match annotator conventions (τ=0.7 misses) | H / large part of Score A | guide mirrors conventions; dense boundary refinement; per-class start/end offsets calibrated on dev |
| 8 | Overfitting to a few samples | H / weak on hidden test | leave-one-video-out; physical parameters; synthetic tests; public data for learned parts |
| 9 | Part B causality violation | L / **disqualification** | separate module; causality test; ask before sharing Part A caches |
| 10 | Weights download fails or exceeds 5 GB | L-M / fatal | weights in git where small; Release assets (≤2 GiB, no bandwidth cap) + mirror + SHA-256; no LFS |
| 11 | Demo down, sleeping, or crashing on the judges' upload | M / up to 30% of the Website score | size and length checks; ffmpeg transcoding; H.264 output; progress bar; HF free CPU tier sleeps after 48 h idle, so keep it warm or host elsewhere; precomputed examples as fallback |
| 12 | VLM on T4: no bf16, no FlashAttention-2, cap-sized weights | M / NaNs or OOM, budget | FP16 with SDPA attention; ≤2 B parameters or quantized; strict candidate cap; test for NaN |

---

## 10. Questions for the hackathon channel
1. Evaluation environment: OS and distribution, **NVIDIA driver version** (`nvidia-smi`), **exact Python version** used for `pip install`, is the install a fresh venv, which pip version? Is Docker with nvidia-container-toolkit available, and what exact `docker run` command will you use (`--gpus`, volume mounts, `--shm-size`)?
2. Time budget: is it measured per video from the start of `detect_events` to the last `step()`? Does importing `solution.py` or loading models count, and for which video? Does `run_submission.py` kill the process on timeout, or finish and then discard the result?
3. Is `weights/download.sh` run **on the evaluation machine (same GPU)**? May it do other one-time setup there, such as building a TensorRT engine? Is there internet during `pip install` / `docker build`?
4. Order of calls: is `detect_events(video)` always called before the frames are streamed? May `RiskEstimator` reuse **strictly per-frame, causal** detections that Part A computed for frames ≤ t, or must it recompute from the frames it receives?
5. Test videos: same codec, resolution and fps as the samples? Maximum clip length? Any variable-frame-rate files? Is the ground-truth `duration` equal to `n_frames/fps` or the container duration? Is `t_sec` equal to `frame_idx/fps` or the PTS?
6. stopped_vehicle: does the annotated start fall at the moment of stopping, before the 10 s have passed? Is a vehicle left stopped after an accident also labelled stopped_vehicle?
7. Is a queue at a red light ever labelled congestion? Is congestion one segment per direction?
8. Is there a minimum annotated event duration? Are sub-second events (for example a brief solid-line touch) labelled?
9. If `step()` crashes part-way through a video, is the whole video's risk curve discarded, or only the remaining frames?
10. May a closed model help annotate our own dev labels? May we publish our dev labels on the website?
11. Is AGPL-3.0 code (Ultralytics) acceptable in the submission?
12. Website: what size and length will the judges' test upload be? Is a free host that wakes within about 1 minute acceptable?
13. The deadline, its time zone, and the cut-off date for any hardware or spec updates ("organizers may update").
14. Is the repo directory writable at runtime? Is `$HOME` writable? Is `/tmp` available? Are 8 CPU cores dedicated to us?

---

## 11. Plan for three people (D0 = the day the starter kit and samples arrive)
Roles:
- **P1, perception and rules:** detector, tracker, `scene.yaml`, rule classes, post-processing.
- **P2, accident, near-miss and Part B + evaluation:** annotation tooling, the metric loop, leave-one-video-out, learned accident and near-miss parts, the causal risk estimator and its calibration.
- **P3, DevOps, EDA and website:** requirements, Docker, CI, `download.sh`, rehearsals, EDA, website and demo, README and report.

All three annotate (every sample labelled twice).

| Milestone | Scope | Exit criterion |
|---|---|---|
| **M0 (D0–D1): walking skeleton** | repo layout, the default `solution.py`, requirements and Docker, `env.py`, sanitizer, CI; website online with the Team section | fresh clone runs offline on a Kaggle T4; `--validate-only` passes; **tag v0.1** |
| **M1 (D1–D3): data** | annotation guide; double labels and settled dev ground truth; cached tracks for all samples; T4 cost table; EDA plots | agreement F1 reported; runtime per video measured |
| **M2 (D3–D6): first real scores** | 4–6 high-precision rule classes (e.g. stopped_vehicle, wrong_way, congestion, jaywalking, plus red_light if the signal is visible); TTC baseline for Part B; demo MVP (upload → timeline) | leave-one-video-out Score A/B logged; **tag v0.2 = a submission that scores** |
| **M3 (D6–D10): improve** | accident and near-miss models; boundary refinement; per-class gating; calibrating Part B scores so that 0.5 means "probably within 5 s"; sample renders, ablations, failure cases on the website | every change A/B-tested on leave-one-video-out; no regression in time or determinism |
| **M4 (D10–D12): harden** | full offline rehearsal on T4 and in Docker; double-run diff; fuzz tests; regenerate `predictions_samples.json` from the release candidate; README and report | the checklist in §1.4 is all green |
| **M5 (last 24–48 h): freeze** | bug fixes only; final tag; website links point to the tag | a fresh clone at the tag reproduces `predictions_samples.json` |

If the deadline turns out to be under a week, compress the plan to M0 on D0, M1 and M2 by D3, M3 on D4–D5, and M4/M5 on D6. Never skip M0 or M4.

---

## 12. Implications for the other dimensions
- **Models:**
  - The 5 GB weight cap is about 2.5 B FP16 parameters in total.
  - The T4 has no bf16 support and no FlashAttention-2; vLLM's Qwen3-VL backend also has a Turing issue (#29743).
  - Prefer the PyTorch FP16 path. Use TensorRT only if `download.sh` can build the engine on the evaluation GPU.
- **Part B:** process synchronously at a fixed stride, run its own causal perception unless the organizers approve sharing, and keep a singleton model shared with Part A.
- **Website:** encode with ffmpeg/libx264, because OpenCV pip wheels couldn't encode H.264 in our test. Reuse `src/` so the demo and the submission share one code path. HF Spaces' free CPU tier (2 vCPU, 16 GB) sleeps after 48 h without activity.
- **Strategy:** per-class kill switches and boundary refinement are cheap, high-value levers for Score A under the macro-F1 averaged over τ ∈ {0.3, 0.5, 0.7}.



## Top recommendations
- Ship requirements.txt pinned to torch==2.14.0+cu126 / torchvision==0.29.0+cu126 via --extra-index-url (works on driver >=525), with ultralytics-opencv-headless, opencv-python-headless, lap, and numpy/scipy markers for Python 3.10. Also ship a tested Dockerfile (pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime) that bakes the weights in at build time.
- Before any other import, set YOLO_OFFLINE, YOLO_AUTOINSTALL=false, HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, HF_HUB_DISABLE_TELEMETRY, CUBLAS_WORKSPACE_CONFIG and HF_HOME/TORCH_HOME (pointing inside weights/). Load weights only by absolute path. Rehearse offline with the socket-blocking sitecustomize netguard plus docker run --network none.
- Keep weights small and in git (<50 MiB each). Put larger files in GitHub Release assets (<2 GiB each, no bandwidth cap) with a SHA-256-checked, idempotent download.sh and an HF mirror. Do not use Git LFS.
- Make determinism structural: fixed seeds; cudnn.benchmark=False; deterministic algorithms with warn_only=True; sequential decoding; fixed batch size; sorted iteration; a synchronous Part B; no wall-clock-dependent decisions except an emergency cutoff. Verify with a double-run diff (compare_predictions.py) in CI and on a T4.
- Plan compute from video metadata with a T4-measured cost table, aiming for about 1.0–1.2x duration. Stop starting optional stages at 1.8x and return partial results at 2.4x. Never retrieve or convert unused frames: BGR conversion was the dominant decode cost in local benchmarks.
- Wrap every stage in try/except and pass every output through a sanitizer: Python floats only, no NaN, clamped to [0, duration], same-class merging, valid labels, step scores clipped to [0,1]. Re-raise the harness's timeout exception if it uses one.
- Build a dev set in the first 2–3 days: annotation guide that mirrors the official conventions; independent double labels (VIA 3 or mpv+CSV) converted to ground_truth.json; agreement measured with the official evaluate.py; tuning by leave-one-video-out; synthetic-trajectory unit tests for classes that don't appear in the samples.
- Refine boundaries with a dense ±1–2 s re-decode around each candidate boundary, because τ=0.7 allows only ~0.35 s of error on a 2 s event. Add per-class enable switches, because a falsely predicted absent class enters C with F1=0.
- Tag a runnable, validated (default-template) submission on day 1, then iterate on milestones M0–M5. Split roles as P1 perception/rules, P2 accident + Part B + evaluation, P3 DevOps/EDA/website.
- Ask the organizers now for the driver and Python versions, how the time budget is measured (including model loading), whether download.sh runs on the evaluation GPU, whether Part B may reuse causal per-frame Part A detections, and the stopped_vehicle/congestion labelling conventions.


## Open questions
- What NVIDIA driver version and exact Python version does the evaluation machine use, is pip run in a fresh venv, and is nvidia-container-toolkit available if we submit a Dockerfile?
- How is the 3x time budget measured: does it include importing solution.py and loading models, the harness's own frame decoding, and does run_submission.py kill the process or just discard results over the limit?
- Is weights/download.sh run on the evaluation GPU machine, and may it do other one-time setup such as building a TensorRT engine?
- May RiskEstimator reuse strictly causal per-frame detections computed in detect_events for frames <= t, or must Part B recompute everything from the frames it receives?
- How is the ground-truth duration computed (n_frames/fps or container duration), is t_sec PTS-based or frame_idx/fps, and can test videos have a variable frame rate or a different codec?
- Labelling conventions: is the stopped_vehicle start annotated at the moment of stopping (before 10 s have passed), is a red-light queue ever labelled congestion, is congestion one segment per direction, is a minimum event duration used?
- Is AGPL-3.0 code (Ultralytics) acceptable, and may a closed model assist with annotating our own dev labels?
- What are the deadline, its time zone, and the last date for hardware or spec changes? What size and length will the judges' demo test upload be?
- How many sample videos will there be and how long are they? Is the traffic signal visible (camera.md)? This decides whether the red_light and stop_line classes are feasible.
- Starter-kit details not yet seen: how run_submission.py enforces the timeout (signals or exceptions), and whether evaluate.py requires risk arrays for label-only (agreement) scoring.


## Key claims (as submitted for verification)
- PyTorch 2.14.0 is the latest release on PyPI (published 2026-09-02) with Python 3.10–3.14 classifiers. — https://pypi.org/project/torch/
- PyTorch 2.14 is the last release with Python 3.10 (cp310) wheels; 2.15 raises the minimum to Python 3.11 and goes GA on 2026-10-28. — https://dev-discuss.pytorch.org/t/notice-python-3-10-support-is-being-removed-from-pytorch-2-15-2-14-is-the-last-release-with-3-10-wheels/3440
- PyTorch 2.15 will drop the CUDA 12.6 and 13.0 builds and the Maxwell, Pascal and Volta architectures, but Turing (sm_75, T4) stays supported through CUDA 13.2 and 13.4. — https://github.com/pytorch/pytorch/issues/190385
- torch==2.14.0+cu126 and torchvision==0.29.0+cu126 wheels exist on the PyTorch cu126 index and resolve for manylinux x86_64 on cp310, cp312 and cp313 (checked with pip --dry-run and uv pip compile). — https://download.pytorch.org/whl/cu126/torch/ (plus local resolution test)
- CUDA 13.x applications need NVIDIA driver >= 580; CUDA 12.x works from driver >= 525 through minor-version compatibility. — https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- Since PyTorch 2.11, the CUDA wheels published on PyPI are CUDA 13.0 by default. — https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325
- onnxruntime-gpu PyPI packages from version 1.27 are built with CUDA 13.0 by default (versions up to 1.26 used CUDA 12.8). — https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- Ultralytics reads YOLO_OFFLINE (skips the online check), YOLO_AUTOINSTALL (default True) and YOLO_CONFIG_DIR; check_requirements pip-installs missing packages when online, and the ByteTrack matching module calls check_requirements('lap>=0.5.12') if lap is missing. — https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/utils/__init__.py ; https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/trackers/utils/matching.py
- The ultralytics package depends on opencv-python (not headless); an official ultralytics-opencv-headless package (8.4.162, 2026-09-24) swaps in opencv-python-headless. — https://raw.githubusercontent.com/ultralytics/ultralytics/main/pyproject.toml ; https://pypi.org/project/ultralytics-opencv-headless/
- Ultralytics YOLO (including YOLO26) is AGPL-3.0, and this also covers trained models and connected code unless an Enterprise licence is bought. — https://www.ultralytics.com/license
- GitHub blocks files larger than 100 MiB in repositories; release assets must each be under 2 GiB, up to 1000 per release, with no limit on total size or bandwidth. — https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github ; https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
- Git LFS on GitHub Free includes 10 GiB storage and 10 GiB bandwidth per month, clones count against the repo owner, and LFS is disabled until the next month when the quota is exceeded. — https://docs.github.com/en/billing/concepts/product-billing/git-lfs
- PyTorch does not guarantee reproducibility across releases, platforms or CPU vs GPU; determinism needs cudnn.benchmark=False, cudnn.deterministic, torch.use_deterministic_algorithms (warn_only available) and CUBLAS_WORKSPACE_CONFIG=:4096:8. — https://docs.pytorch.org/docs/2.14/notes/randomness.html
- HF_HUB_OFFLINE=1 stops all HTTP calls to the Hub (cached files only), and huggingface_hub reads its environment variables at import time. — https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
- The Tesla T4 and the GTX 1650 family are both compute capability 7.5 (Turing); the local nvidia-smi reports compute_cap 7.5 and driver 592.82 for the team's GTX 1650. — https://developer.nvidia.com/cuda-gpus (plus local nvidia-smi)
- The GTX 1650 (TU117) has no Tensor Cores; it has dedicated FP16 units instead, so its FP16 speed is not representative of a T4. — https://www.techpowerup.com/254827/nvidia-geforce-gtx-1650-released-tu117-896-cores-4-gb-gddr5-usd-150
- FlashAttention-2 supports Ampere/Ada/Hopper, not Turing (T4); bf16 needs Ampere or newer. — https://pypi.org/project/flash-attn/
- Kaggle gives about 30 GPU-hours per week (P100 or 2xT4); free Colab sessions run at most 12 h with dynamic limits and no guaranteed GPU. — https://www.kaggle.com/docs/efficient-gpu-usage ; https://research.google.com/colaboratory/faq.html
- VIA 3 is BSD-2-licensed, runs offline as a single HTML file and supports video temporal segments with CSV/JSON export; Label Studio's TimelineLabels exports frame ranges. — https://www.robots.ox.ac.uk/~vgg/software/via/ ; https://labelstud.io/tags/timelinelabels
- Local measurement on a synthetic 1080p H.264 clip (i5-12450H): OpenCV full read ~52–65 fps, grab-only ~180–330 fps, PyAV threaded ~130–150 fps with BGR conversion. OpenCV 4.13 pip on Windows could not open an H.264 encoder and fell back to mp4v. — reasoning (local benchmark in scratchpad)
