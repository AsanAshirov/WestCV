<!-- source: deep-research workflow wf_294a4b01-fc2, agent gap:1 -->

# Runtime budget for the real sample format (4K 29.97p, H.264 High 4:2:2 10-bit, ~140 Mbps)

## 0. Summary

- **At 29.97 fps the 3× limit works out to 100 ms per source frame for Part A and Part B together.** The 2.4× planning target is 80 ms per frame, not the 120 or 96 ms you get at 25 fps. For C3896 (10,200 frames, 340 s) that means a 1,021 s hard limit and a 817 s plan.
- **Decoding is the whole problem. The GPU sits mostly idle at 4K.** On the dev laptop under Linux (the OpenCV and PyAV wheels the evaluation will actually use, 8 vCPU):
  - the harness's `cv2.read()` path runs at 30–36 fps, i.e. **0.84–1.0× of the video's duration**;
  - PyAV threaded decode alone runs at 47–50 fps, i.e. **0.60–0.64×**.
- **Scaled to an AWS g4dn.2xlarge, the machine that exactly matches the spec,** the harness decode alone becomes about **1.4–2.1× (central estimate 1.77×)** and Part A's decode floor about 0.9–1.5×. On that class of machine, running Part A and Part B together at full quality does not fit in 2.4×. It may not fit in 3.0× either.
- **NVDEC is not a way out.** Turing decodes H.264 only up to High profile (8-bit 4:2:0). I confirmed the rejection on the laptop's own Turing GPU.
- **Five levers remain:**
  1. Pin the Linux OpenCV wheel, which has threaded colour conversion.
  2. In Part A, use PyAV with `thread_type='AUTO'` and `skip_loop_filter`, and convert-plus-downscale in one call with `frame.reformat()`.
  3. In Part A, use `skip_frame=NONREF` if the real GOP has non-reference B-frames.
  4. Choose a tier per video from a 60-frame timing probe of the harness path.
  5. Keep a **"sacrifice Part B" valve**: `RiskEstimator.reset()` raises when the probe says Part A plus Part B cannot fit. The harness then records an empty risk curve but **keeps Part A's events**. That is better than going over time, which empties both.
- **Ask the organizers now (§7).** The eval CPU and the test-file format decide which tier you ship.

## 1. Format facts (step 1)

| file | status | codec / fps / frames | source |
|---|---|---|---|
| C3896.MP4 | confirmed | `AVC140_3840_2160_H422P@L51`, captureFps 29.97p, 10200 frames (340.3 s), LPCM16 2 channels, Sony ILCE-6700 | Sony XML at the end of the file (`tail_C3896.bin`) |
| C3897.MP4 | confirmed | same format, 9525 frames (317.8 s) | `tail_C3897.bin` |
| C3902.MP4 | **not confirmed** | Drive shows 5.4G, which at the same byte rate is ≈318 s / ≈9,530 frames (estimate) | byte-range retry with fresh `uuid` and cookies still returns **"Google Drive – Quota exceeded"** (HTML, 2,009 bytes) |
| C3905.MP4 | **not confirmed** | Drive shows 2.2G ⇒ ≈129 s / ≈3,880 frames (estimate) | same quota error |

Size check: C3896 is 5.8G for 340.3 s and C3897 is 5.4G for 317.8 s, both ≈18.3 MB/s ≈146 Mbps including audio. The two unknown files match that byte rate, so the same format is likely but unproven.

**To do once the quota resets:** `ffprobe -v error -show_entries stream=codec_name,profile,pix_fmt,r_frame_rate,nb_frames -show_entries frame=pict_type -read_intervals %+#64 C3902.MP4`.

The GOP structure matters as much as the format, because the NONREF trick in §3 depends on it.

Two practical warnings:
- **All four samples total ≈18.8 GiB, but the laptop's C: drive has 11 GB free.** You need an external disk or a cloud machine.
- **The website demo cannot accept raw camera files.** Two minutes of this format is about 2.1 GB. Cap uploads by size and transcode server-side.

## 2. Measured cost table (step 2)

Full data: `scratchpad\cost_table.csv` (69 rows: machine, CPU, clip, path, frames, runs, best and worst fps, × realtime, ms per source frame, CPU-ms per frame). Raw logs are `sweep_all.csv` (Windows), `sweep_wsl.csv` and `sweep_wsl2.csv` (Linux). The benchmark script is `dbench.py`.

**Setup:**
- **Linux measurements:** WSL2 Ubuntu 22.04, 8 vCPU of an i5-12450H, Python 3.10.12, `opencv-python-headless==4.13.0.92` (bundles FFmpeg 8.0.1 / swscale 9.1), `av==17.1.0`.
- **Clips:**
  - `synth_cabac.mp4`: High 4:2:2, yuv422p10le, 148 Mbps, P-only GOP.
  - `dlx/origB.mp4`: same format, IBBP GOP 30, 137 Mbps.
  - `c1080.mp4`: 1080p25 4:2:0, 4 Mbps.

**Noise caveat:** the laptop was shared with other jobs, so wall times vary by ±25%. I report the best of 2–3 runs, plus CPU time, which is robust to contention.

| path (4K 422 10-bit unless noted) | best fps | ms per source frame | CPU-ms per frame |
|---|---|---|---|
| **harness `cv2.VideoCapture(p).read()` (Linux wheel)** | 29.9 (P-only) / 35.6 (IBBP) | **33.4 / 28.1** | 175 / 158 |
| harness `read()` (Windows wheel: FFmpeg 4.4 prebuilt, single-thread swscale) | 19.9 | 50.3 | – |
| harness `read()`, taskset to 4 CPUs | 23.2 | 43.1 | 135 |
| `cv2` `grab()` only | 45.9 | 21.8 | 124 |
| `grab()` + `retrieve()` on every 2nd / 3rd frame | 34.3 / 37.1 | 29.2 / 27.0 | – |
| `CAP_PROP_N_THREADS` 4 / 8 (default = CPU count) | 24.6 / 32.7 | 40.7 / 30.6 | – |
| PyAV decode only, `thread_type` AUTO (FRAME is identical) | 47.1 / 50.0 | 21.2 / 20.0 | 124 / 125 |
| PyAV decode, **default SLICE threading** | 10.1 | 99 | – |
| PyAV + `skip_loop_filter=all` | 54.0 / 52.8 | 18.5 / 18.9 | 102 / 105 |
| PyAV `skip_frame=NONREF` (IBBP; 249 of 360 frames output) | 54.7 (source rate) | 18.3 | **82 (−35%)** |
| NONREF + skip_loop_filter + `reformat(1280×720, bgr24)` on every output frame | – | 22.2 | 90 |
| PyAV `skip_frame=NONKEY` (I-frames only) | – | I/O-bound here | **7** |
| PyAV `to_ndarray('bgr24')` at full resolution, every frame | 35.0 | 28.6 | 152 |
| PyAV `reformat(1280×720, bgr24)` on every 2nd frame | 40.4 | 24.8 | 134 |
| + skip_loop_filter | 47.5 | 21.1 | – |
| **1080p25 4:2:0:** harness `read()` / PyAV decode / reformat 1280 on every 2nd frame | 244 / 570 / 294 | 4.1 / 1.8 / 3.4 | 12 / 6 / 10 |

CPU-side costs of each processed frame (`cpu_micro.py`, Windows laptop):

| operation on a 4K frame | ms |
|---|---|
| resize to 1280×720, INTER_LINEAR / INTER_AREA | 2.1 / 3.9 |
| resize to 960×540 / 640×360 | 1.0 / 0.4 |
| letterbox to 1280 | 4.1 |
| HWC→CHW transpose on the CPU | 7.0 (do it on the GPU) |
| **copying a 4K BGR frame** | **9.1 (never copy step()'s frame)** |
| ByteTrack-like update, 20 / 40 / 80 tracks | 0.7 / 1.2 / 2.8 |

What these numbers mean:
1. **The harness cost depends on the OpenCV build.** OpenCV 4.13 converts colour with threaded swscale only when libswscale ≥ 6.4.100. The Linux wheel qualifies; the Windows wheel (FFmpeg 4.4) does not, so it is about 1.5× slower. **Pin `opencv-python-headless==4.13.0.92` in requirements.txt.** The kit's `>=4.8` would install 5.0.0.93 today, which is untested.
2. **`OPENCV_FFMPEG_CAPTURE_OPTIONS` cannot speed up the harness.** It only reaches `avformat_open_input` (the demuxer). Its `avdiscard` key would drop frames and break the harness's `idx/fps` timestamps. Do not touch it.
3. **In PyAV, set `stream.thread_type = "AUTO"`.** The default (SLICE, in both 16.1 and 17.1) was 5× slower on this stream.
4. **In long-GOP H.264, "stride" does not reduce decode work.** Every frame is decoded either way; stride only saves conversion and model time.

**Not done here: Kaggle T4, g4dn, and the `-hwaccel` check on a T4 itself.** I have no access to them. The commands for step 2 are in §6.

## 3. NVDEC (step 2g)

- **Primary source.** NVIDIA's NVDEC programming guide (SDK 13.0) lists Turing H.264 support as "Baseline, Main, High profile up to Level 5.1". Only Blackwell adds "High 10 and High 422".
- **Local test on a Turing GPU** (GTX 1650, TU117, the same NVDEC generation as the T4, driver 592.82):
  - `-hwaccel cuda` prints "Failed setup for format cuda: hwaccel initialisation returned error" and silently falls back to CPU decoding;
  - `-c:v h264_cuvid` prints "Codec h264_cuvid is not supported with this chroma format";
  - the 1080p 4:2:0 control clip decodes on NVDEC without problems.
- **Conclusion:** torchcodec, PyNvVideoCodec and DALI cannot help with this format on a T4.
- **If the organizers transcode the test set to 4:2:0 8-bit,** NVDEC works again for Part A. The harness path stays on the CPU regardless.

## 4. Scaling to the evaluation machine (estimate)

- **Target:** the spec's "1× T4 16 GB, 8 CPU cores, 32 GB" is exactly an AWS **g4dn.2xlarge**: 8 vCPU = **4 physical cores × 2 threads** of a Xeon P-8259L, 32 GiB.
- **Benchmark ratios (PassMark):** the whole 8259CL chip scores 31,403 (24 cores / 48 threads, single-thread 1,948), which is about 5,200 for a 4-core slice. The i5-12450H scores 15,862, and the 8-vCPU WSL slice is roughly 75% of that. Single-thread is about 3,300 vs 1,948.
- **Multipliers used in `budget.py`:**
  - **1.5×** (optimistic, `g4dn2x_opt`)
  - **1.9×** (central, `g4dn2x_est`)
  - **2.3×** (pessimistic, `g4dn2x_pess`)
  - **0.8×** for a modern 8-physical-core CPU (`modern8c_est`)

  All four are estimates. Replace them with a real measurement (§6).

## 5. budget.py (step 4) and the plan

**File:** `scratchpad\budget.py`. It replaces the earlier 25-fps model that was in the same file.

- **Inputs:** two profiles (`4k2997_422_10`, `1080p25_420`) with the measured decode costs above.
- **GPU costs:** Ultralytics' YOLO26 T4-TensorRT10 figures (n/s/m = 1.7 / 2.5 / 4.7 ms at 640) × 1.8 for PyTorch FP16, scaled by input area. These are estimates to replace with `bench_t4_gpu.py` output.
- **What it models:**
  - Part A: decode → queue → batched GPU (1.1 overlap factor), with stride-1 dense windows done inline (no second decode), a 5 Hz static-scene engine, and optional fire detector / clip classifier / VLM;
  - Part B: harness decode plus `step()` work, synchronous or overlapped (see §5.2).
- **What it prints:** the richest configuration under the target (`--target 2.4` by default) and `T_A = 3·dur − 1.2·PartB_pred − max(8 s, 0.05·dur)`.
- **Measured-input mode:** `--csv target.csv` loads a `dbench.py` CSV measured on the real machine.

Results for a 300 s clip, as multiples of the video's duration:

| scenario | harness decode (fixed) | Part B, n@640 stride 3 | Part A decode floor, skip_loop_filter / NONREF | Part A, s@1280 stride 2 / 3 | richest config within 2.4× | T_A hard stop |
|---|---|---|---|---|---|---|
| laptop, Linux (measured) | 0.93 | 0.93 | 0.64 / 0.48 | 0.75 / – | A m@1280 s2 + windows + all optional stages; B s@960 s2 → 2.38× | 549 s |
| modern 8-core (0.8×) | 0.74 | 0.75 | 0.52 / 0.38 | 0.60 | same → 2.15× | 616 s |
| g4dn optimistic (1.5×) | 1.39 | 1.40 | 0.97 / 0.71 | 1.13 | A m@1280 s3 NONREF; B s@960 s2 → 2.37× | 381 s |
| **g4dn central (1.9×)** | **1.77** | **1.77** | **1.22 / 0.91** | 1.42 / 1.34 | **nothing with Part B fits.** Best is Part B aborted, A m@1280 s2 → 2.39×. Keeping Part B needs about 2.75× (A m@640 s6 NONREF) | **246 s**, which is below the Part A floor |
| g4dn pessimistic (2.3×) | 2.14 | 2.15 | 1.48 / 1.10 | 1.72 | Part B aborted only | 112 s |
| 1080p25, g4dn central | 0.19 | 0.20 | 0.11 / 0.09 | 0.18 | A m@1280 s2 + 40 windows + all optional stages; B s@960 s2 → 1.05× | 814 s |

What these results mean:
- **Stages that re-read the video are expensive at 4K.** The "30 clip-classifier windows" option costs 0.4–0.5× by itself because it re-decodes 4.5 s windows. Collect crops and short clips during the single pass (a ring buffer around triggers) instead.
- **At 4K the GPU is 15–25% busy even with YOLO26m@1280 at stride 2** (≈20 ms per image in batches of 8, vs ≈35–45 ms of CPU per source frame). A bigger model is almost free. Larger input sizes and tiling are not, because their cost is CPU swscale time.
- **Part A can afford a RiskEstimator-style risk score internally, at ≈0.005×,** as long as it reuses Part A's own tracks (TTC is only arithmetic on tracks). It cannot afford a second decode or full-resolution BGR frames (+1× or more).

### 5.1 Tier choice per video (at the start of `detect_events`)

1. **Probe the harness path.** Open `cv2.VideoCapture(path)` exactly as the harness does, discard one frame, then time 60 `read()` calls. This gives `t_read` and `h = t_read·fps` (the harness cost as a fraction of duration). It costs about 2–4 s, and it is measured on this machine and this file.
2. **Choose the tier:**
   - **T0, h ≤ 0.8 (rich):**
     - Part A: full decode with skip_loop_filter, **YOLO26m@1280 (or s@1280) on the road-ROI crop at stride 2 (15 Hz)**, inline stride-1 windows (≤40 × 8 s), static engine at 5 Hz, fire detector at 1 Hz.
     - Part B: s@960 at stride 2, overlapped.
   - **T1, 0.8 < h ≤ 1.2 (standard):**
     - Part A: s@1280 or m@960 at stride 3 (10 Hz), 20 windows, static engine.
     - Part B: n@640 at stride 3, overlapped.
   - **T2, 1.2 < h ≤ 1.55 (lean):**
     - Part A: NONREF + skip_loop_filter (or plain skip_loop_filter if the real GOP has no non-reference B-frames), s@960/1280 on the decoded frames (about stride 2–3), static engine on the same frames, no re-reads.
     - Part B: n@640 at stride 3–4.
   - **T3, h > 1.55, or h > 1.4 if NONREF saves nothing on the real GOP (sacrifice Part B):**
     - set `_CLOCK[vid]["abort_b"] = True`. `RiskEstimator.reset()` raises, `run_submission.py` catches it, logs it, sets `risk = []` and **keeps the events**.
     - Part A gets up to 2.4× total (m@1280 at stride 2).

   **Where 1.55 comes from:** Part A's minimum is ≈0.435·h + 0.15 (NONREF decode plus models). The time left after Part B is 2.95 − 1.224·h. These cross at h ≈ 1.69. I took 10% below that.
3. **Set the Part A hard stop:** `T_A = 3·dur − 1.2·(n·1.02·t_read + (n/stride_B)·t_step) − max(8, 0.05·dur)`. The Part A frame loop stops pulling frames after `t0 + T_A`. Segments still open at that point are closed at the cutoff time.
4. **Guard Part B while it runs.** Every 32 calls, `step()` projects `now + remaining_frames × EMA(dt)`. If that exceeds `deadline − 5 s`, it raises. The deadline is `t0 + 3·n_frames/fps`, with `t0` handed over from `detect_events` through a module-level dict keyed by `meta["video_id"]`. Only timing crosses over, never video content.
5. **Determinism.** The tier depends on timing, so round `h` to 0.05, put the thresholds far from the eval machine's actual `h` once it is known, log the chosen tier, and add an environment override such as `WIUT_TIER=1` for reproducibility runs.

### 5.2 Part B mechanics

- **Resize synchronously inside `step()`.** Resize the 4K BGR frame with `cv2.resize` (INTER_LINEAR, 2 ms) into a preallocated buffer; do not copy the frame first (a copy costs 9 ms).
- **Run the detector with a fixed lag of L = 3 processed frames.** A worker thread runs detector, tracker and TTC. `step(t)` returns the score for frame t − L·stride, waiting for it if necessary.
  - This stays causal and deterministic, unlike "return whatever is ready".
  - It overlaps GPU work with the harness decode, because OpenCV's bindings release the GIL (ERRWRAP2 / PyAllowThreads) and so do torch CUDA calls.
  - It costs about 0.1–0.3 s of TTA, which is negligible against the 5 s horizon.
- **Load everything at import time:** models, CUDA context, cuDNN warm-up, and optionally a TensorRT engine build. `load_solution()` runs before any per-video timer starts, so this time is free.

### 5.3 Stride, input size, crop and tiling

**4K profile:**
- **Part A:** use `frame.reformat(1280, 720, 'bgr24')` straight from the decoded frame (one swscale call, +4–6 ms per used frame, instead of about +10 ms for full-resolution BGR). Then crop to the carriageway bounding box from camera.md, or reformat the crop via an `av.filter` graph `crop,scale` (not benchmarked yet).
- **Model input:** 1280 on the long side. A 60 px car at 4K becomes about 20 px, which is fine.
- **Tiling:** add only a single far-field tile at 5 Hz, and only if distant pedestrians fall below ~16 px at 1280. Do not use 2×2 tiles: they cost about 4× the CPU swscale time.
- **Static-scene classes** (stopped_vehicle, road_obstacle, congestion, fire_smoke): run them on 480×270 grey frames at 5 Hz from the same decode. Or use I-frames only (`NONKEY`, 7 CPU-ms per frame), if a later pass is ever needed.

**1080p25 profile:**
- decode is cheap, so the GPU becomes the limit;
- Part A: m@1280 at stride 2, or s@1280 at stride 1, no tiling (1080p is close to 1280 already);
- Part B: s@960 at stride 2;
- a VLM on 10–20 candidate windows fits (≈3 s per query, a guess).

## 6. Commands for Kaggle or cloud (steps 2–3, not run here)

```
lscpu | egrep 'Model name|^CPU\(s\)|Thread|Core|MHz'; nproc; nvidia-smi --query-gpu=name,driver_version --format=csv
pip install -q opencv-python-headless==4.13.0.92 av==17.1.0 ultralytics==8.4.162
ffmpeg -ss 0 -t 60 -i C3896.MP4 -c copy C3896_60s.mp4          # stream copy keeps the exact format
python dbench.py C3896_60s.mp4 --machine kaggle-t4 --reps 2 > sweep_t4.csv
taskset -c 0-7 python dbench.py ... ; ffmpeg -hide_banner -v verbose -hwaccel cuda -i C3896_60s.mp4 -frames:v 60 -f null - 2>&1 | grep -iE "Failed|hwaccel"
python bench_t4_gpu.py > gpu_t4.csv    # YOLO26n/s/m x 640/960/1280 x batch 1/8, incl. 4K resize, crop-letterbox, pinned H2D, ByteTrack
python budget.py --csv sweep_t4.csv
```

Notes:
- PyAV 17.1.0 is the last release with cp310 wheels; 18.x requires Python ≥3.11. Use 17.1.0 if the organizers' Python is 3.10.
- Kaggle's vCPU count was not verified here, so log `nproc`. A g4dn.2xlarge spot instance is the faithful proxy for the eval machine.

## 7. Question for the hackathon channel (step 5)

> Hi! Question about the runtime limit (3× video duration for Part A + Part B). The sample files (C3896/C3897) are raw Sony ILCE-6700 recordings: 3840×2160, **29.97 fps, H.264 High 4:2:2 10-bit, ~140 Mbps** (per the XML at the end of the files). The task text says clips are 25 fps. (1) Will the hidden test videos be these raw camera files, or transcoded (e.g. 25 fps / 4:2:0 8-bit / lower resolution)? (2) What exact CPU does the evaluation machine have (model, physical cores vs vCPUs; is it e.g. an AWS g4dn.2xlarge)? Which OS/Python version? (3) Have you timed `run_submission.py`'s own `cv2.read()` loop on these files on that machine? On our 8-vCPU Linux test it alone takes 0.85–1.0× of the video duration, and on a 4-core Xeon it could approach 2×, before any model runs. (4) If `RiskEstimator` raises an exception to stay inside the time budget, are that video's Part A events still scored (as `run_submission.py` does: it only empties `risk`), or is the whole video scored empty (as the PDF wording suggests)? (5) Is there an overall time limit on import and model loading before the first video? Thanks!

## 8. What is measured and what is estimated

- **Measured:**
  - all decode, resize and tracker numbers on the laptop (with ±25% noise from shared use);
  - the NVDEC rejection on a Turing GPU;
  - the OpenCV, FFmpeg and PyAV behaviours.
- **Estimated:**
  - the CPU multipliers for the eval machine;
  - all T4 GPU timings;
  - VLM and clip-classifier costs;
  - the NONREF saving on the real Sony GOP (IBBP pyramid gave −35% CPU; a plain IBBP GOP would give more; a GOP without B-frames gives nothing);
  - the durations of C3902 and C3905.



## Key claims (as submitted for verification)
- C3896 and C3897 are 3840x2160 H.264 High 4:2:2 (AVC140_3840_2160_H422P@L51), captureFps 29.97p, Sony ILCE-6700, LPCM16 stereo, 10200 and 9525 frames. — Sony NonRealTimeMeta XML at file end: scratchpad/tail_C3896.bin, tail_C3897.bin
- C3902/C3905 tail range requests still return 'Google Drive - Quota exceeded'; Drive lists them as 5.4G and 2.2G (C3896 5.8G, C3897 5.4G), implying ~318 s and ~129 s at the same byte rate (estimate). — curl of https://drive.usercontent.google.com/download?id=... (this session) + reasoning
- On Turing, NVDEC decodes H.264 only as Baseline/Main/High up to Level 5.1; High 10 and High 4:2:2 H.264 decode are added only on Blackwell. — https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html
- On a Turing GPU (GTX 1650, driver 592.82) ffmpeg -hwaccel cuda fails on the 4:2:2 10-bit clip ('Failed setup for format cuda') and h264_cuvid reports 'not supported with this chroma format', while a 1080p 4:2:0 clip decodes fine. — local test in this session (scratchpad/synth_cabac.mp4, c1080.mp4)
- AWS g4dn.2xlarge = 1x NVIDIA T4, 8 vCPU (4 cores x 2 threads), Intel Xeon P-8259L, 32 GiB memory, matching the spec's evaluation machine. — https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html
- PassMark: Xeon Platinum 8259CL CPU Mark 31,403 (24C/48T), single-thread 1,948; Core i5-12450H CPU Mark 15,862. — https://www.cpubenchmark.net/cpu.php?cpu=Intel+Xeon+Platinum+8259CL+%40+2.50GHz&id=3671 ; https://www.cpubenchmark.net/cpu_lookup.php?cpu=Intel+Core+i5-12450H
- opencv-python-headless 4.13.0.92 Linux wheels bundle FFmpeg 8.0.1 (tag 94 / 4.14.0.94 bundles 8.1.2); the Windows 4.13 wheel uses a prebuilt FFmpeg 4.4 (avcodec 58.134, swscale 5.9). — https://raw.githubusercontent.com/opencv/opencv-python/92/docker/manylinux_2_28/Dockerfile_x86_64 ; local cv2.getBuildInformation()
- OpenCV 4.13.0's FFmpeg backend sets swscale 'threads' = requestedThreads (default cv::getNumberOfCPUs) when libswscale >= 6.4.100, decoder thread_count = min(nCPUs,16), and OPENCV_FFMPEG_CAPTURE_OPTIONS is passed only to avformat_open_input (avcodec_open2 gets NULL). — https://raw.githubusercontent.com/opencv/opencv/4.13.0/modules/videoio/src/cap_ffmpeg_impl.hpp
- Measured on the replica with Linux wheels (8 vCPU of i5-12450H): harness cv2 read() 29.9-35.6 fps (0.84-1.0x duration at 29.97p), PyAV AUTO decode 47-50 fps, +skip_loop_filter 53-54 fps; Windows wheel read() 19.9 fps. — scratchpad/cost_table.csv, sweep_wsl.csv, sweep_wsl2.csv, sweep_all.csv (dbench.py)
- PyAV 16.1 and 17.1 default to thread_type SLICE, which decoded this stream ~5x slower (10 fps) than AUTO/FRAME (52 fps). — local measurement + stream.thread_type printout in this session
- skip_frame=NONREF on an IBBP (b-pyramid) replica outputs 249/360 frames and cuts decode CPU from ~125 to ~82 ms per source frame (-35%); NONKEY needs ~7 CPU-ms per source frame. — scratchpad/sweep_wsl2.csv (dlx/origB.mp4)
- PyAV 17.1.0 is the last release with cp310 wheels; 18.0.0/18.1.0 ship cp311/cp314 only (requires_python >=3.11). — https://pypi.org/pypi/av/json
- Unpinned opencv-python-headless>=4.8 currently resolves to 5.0.0.93. — https://pypi.org/pypi/opencv-python-headless/json
- YOLO26 detect T4 TensorRT10 FP16 latency at 640: n 1.7 ms, s 2.5 ms, m 4.7 ms. — https://github.com/ultralytics/ultralytics (README table; copy in scratchpad/ul_readme.md)
- OpenCV Python bindings release the GIL during wrapped calls (ERRWRAP2 uses PyAllowThreads), so a worker thread can overlap with the harness's cap.read(). — https://raw.githubusercontent.com/opencv/opencv/4.x/modules/python/src2/cv2_util.hpp
- In run_submission.py an exception from RiskEstimator empties only the risk curve and keeps the events, whereas exceeding the time budget empties both events and risk for that video. — scratchpad/kit/wiut_cv_scripts/run_submission.py (main(), run_risk())
- At 29.97 fps the 3x budget equals 100.1 ms per source frame for Part A+B (80.1 ms at the 2.4x planning target); C3896 has a 1,021 s budget. — reasoning (3000/29.97; 10200/29.97*3)
