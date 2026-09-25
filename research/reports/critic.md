<!-- source: deep-research workflow wf_294a4b01-fc2, agent critic -->

# Completeness critic

## Gaps
### Runtime budget for the real sample format: 4K, 29.97p, 10-bit 4:2:2 H.264 at ~140 Mbps. It may consume most of the 3x budget before any model runs.
Every report budgets for 25 fps (the spec's wording and the example GT) and, apart from sensitivity rows, for 1080p. The actual sample files say otherwise. The Sony XML at the end of C3896.MP4 and C3897.MP4 (scratchpad tail_C3896.bin / tail_C3897.bin) reads videoCodec="AVC140_3840_2160_H422P@L51", captureFps="29.97p", 3840x2160, camera ILCE-6700, LPCM stereo audio, with 10200 and 9525 frames (about 340 s and 318 s). The spec says the samples share resolution and fps with the test set. I benchmarked a replica (scratchpad synth_cabac.mp4: H.264 High 4:2:2, yuv422p10le, 148 Mbps, 29.97p) on the dev laptop (i5-12450H, 12 threads) with OpenCV 4.13. cap.read(), the exact harness Part B path, runs at 22.2–22.8 fps, which is about 1.33x the video's duration for the harness decode alone. grab()-only runs at 59.5–62.1 fps, so Part A's own sequential pass costs about 0.5x even before any YUV to BGR conversion. That is roughly 1.8x of the 3x budget gone before any neural network runs, and an unknown 8-core evaluation CPU could be slower. Going over budget empties both the events and the risk curve for that video. The stack report's 4K escape route (NVDEC via torchcodec/PyNvVideoCodec) almost certainly does not apply: Turing NVDEC decodes H.264 only as 4:2:0 8-bit. The per-frame budgets are also wrong at 29.97 fps: 100 ms per frame, not the 120 ms in the rules report. This single fact can zero the model score, and it shapes stride, input size, which optional stages fit, Part A's cutoff, and whether Part A can afford to run RiskEstimator internally.

Goal: a measured, end-to-end per-video cost model for the real format, plus a plan chosen from the video metadata. Inputs: the harness at C:\Users\Cicada\AppData\Local\Temp\claude\C--Users-Cicada-Desktop-WestHack\ced7e0d9-be88-47bc-ac13-90898f4ae082\scratchpad\kit\wiut_cv_scripts\run_submission.py (Part A runs first, then run_risk calls cap.read() on every frame; one deadline covers both; over budget empties everything) and the replica clip scratchpad\synth_cabac.mp4.

Steps:
(1) Confirm the format of C3902 and C3905. Their earlier tail downloads hit Drive's 'Quota exceeded'. Retry with a byte-range request for the last 64 KB, or run ffprobe after a full download, and record codec, profile, pix_fmt, fps, frame count and audio.
(2) On a Kaggle T4 notebook, log lscpu and nproc. Ideally also use a generic 8-vCPU cloud VM. Install opencv-python-headless 4.13.0.92 and benchmark:
  (a) cv2.VideoCapture(path, cv2.CAP_FFMPEG).read() over the whole file, exactly as run_risk does;
  (b) grab() only;
  (c) grab() plus retrieve() every 2nd and 3rd frame;
  (d) CAP_PROP_N_THREADS at 0, 4 and 8;
  (e) PyAV 17.1.0 (the last cp310 wheels) with thread_type AUTO and FRAME: decode-only, frame.to_ndarray('bgr24'), and frame.reformat(width=1280, height=720, format='bgr24'), which converts and downscales in one swscale call;
  (f) codec_context skip_loop_filter / skip_frame options;
  (g) ffmpeg -hwaccel cuda -i clip -f null - on the T4, to confirm that NVDEC rejects High 4:2:2 10-bit. Cite NVIDIA's Video Codec SDK decode support matrix.
(3) Measure on the T4: Ultralytics YOLO26n/s/m in PyTorch FP16 at 640, 960 and 1280 (batch 1 and 8), including 4K to input resize, road-crop letterbox and host-to-device copy, plus ByteTrack update cost.
(4) Write budget.py with two profiles, 4K29.97-422-10bit and 1080p25-420. For each, give per-video wall time for: harness decode (fixed), Part B at stride 2 and 3, Part A decode plus detection at stride 2 and 3, stride-1 refinement windows (N windows x 8 s), the static-scene engine at 5 Hz, and optional stages (fire detector, crop classifier, VLM). Output the richest configuration that stays at or below 2.4x total with 0.6x margin, and the Part A hard-stop formula T_A = 3*dur - 1.2*measured_PartB - margin.
(5) Draft the hackathon-channel question. Will the test clips be the raw camera files (4K 29.97p 4:2:2 10-bit, about 140 Mbps) or transcoded to 25 fps as the task text says? What CPU does the evaluation machine have? Have the organizers timed run_submission.py's own decode on it?

Deliverables: a CSV cost table (machine, CPU, path, fps); budget.py; a recommended stride, input size, crop and tiling plan for each profile; and the question text.

### Data logistics and browser/tooling compatibility: about 23 GB of samples the browser cannot play, annotation proxies, and demo uploads of 1–2 GB
There are four sample clips of about 5.3–5.7 min each at ~140 Mbps, roughly 5.5–6 GB per file. Google Drive already returned 'Quota exceeded' for the team's download attempts (scratchpad tail_C3902.bin is the quota HTML page). Mainstream browsers generally do not decode H.264 High 4:2:2 10-bit, which breaks several planned tools:
- the recommended browser-based annotators (VIA 3, Label Studio);
- the website Player's createObjectURL playback of the visitor's own file;
- EDA thumbnails.
The dev set is step zero for every workstream, so this blocks everything. The live demo is also exposed. It is worth 7.5 elimination points and must not crash on the judges' upload, yet a 2-minute clip from this camera is about 2.1 GB, ten times the delivery report's 200 MB cap. Moving roughly 23 GB into Kaggle or Colab for T4 work is non-trivial from a home connection. No report plans for any of this.

Deliver a runbook plus a verification script covering five things.

(1) Reliable download of large public Google Drive files despite the quota. Options: add a shortcut to your own Drive and use rclone copy with --drive-shared-with-me; gdown with --fuzzy and cookies; the Drive API with your own OAuth client. Transfer them into a Kaggle Dataset (check per-file and total size limits and the kaggle CLI resumable upload) and into Colab (Drive mount). File IDs are in scratchpad\vids.txt.

(2) A frame-exact proxy transcode for annotation and development. Example: ffmpeg -i C3896.MP4 -map 0:v:0 -vf scale=1920:-2,format=yuv420p -c:v libx264 -crf 18 -g 30 -bf 0 -fps_mode passthrough -movflags +faststart out.mp4. Also produce a 720p web proxy. Write a Python check that:
- frame counts match between original and proxy under cv2 CAP_PROP_FRAME_COUNT and by actual decode;
- the frame rate stays 30000/1001;
- frame i of the proxy corresponds to frame i of the original (compare downscaled grayscale images).
This way labels made on the proxy in idx/fps seconds map exactly onto the originals. Also test seek accuracy of VIA 3, Label Studio TimelineLabels and mpv on the proxy.

(3) A browser support table (Chrome, Edge, Firefox, Safari, iOS Safari, Android Chrome) for H.264 High 4:2:2, High 10 and HEVC, with sources.

(4) Demo upload design for original-format clips of 0.5–2 GB. Check:
- Gradio 6 max_file_size and chunked-upload behaviour on HF Spaces (the ephemeral disk is 50 GB);
- Modal's 4 GiB request limit;
- a presigned direct-to-bucket upload (Cloudflare R2/S3) followed by a job;
- a server-side trim to the first 120 s (ffmpeg -t 120 -c copy) plus an H.264 4:2:0 preview transcode for playback.
Measure the backend time for a 2-minute 4K 10-bit clip on 2 and 8 vCPUs, where decode dominates. Pick and justify the stated size and length cap (the spec says 'state size/length accepted — 2 minutes is enough'), and write the user-facing message with a one-line ffmpeg command for files over the cap.

(5) Storage and media budget for the website when every sample is transcoded to 720p H.264.

Output: the runbook, the check script, the compatibility table and the demo upload specification with concrete limits.

### Per-video camera registration and consumer-camera artifacts, plus what the clip numbering implies about the hidden test set
The 'fixed CCTV' is a Sony ILCE-6700 mirrorless camera. Its files log gyroscope data, with rec709 gamma. Everything geometric is hard-coded from one reference frame:
- stop lines, crossings, solid lines and lane polygons;
- signal-lamp ROIs;
- homography points.
A small tripod bump, re-framing, zoom change, wind shake or in-body stabilisation drift between or within clips shifts that geometry by tens of 4K pixels. That silently breaks red_light, stop_line, solid_line_crossing, wrong_way, failure_to_yield and the lamp ROIs. Auto-exposure and white-balance changes also affect lamp thresholds and background models. No report plans per-video registration; the delivery report has only a demo camera-match check and an EDA stability card.

The metadata also shows the samples were recorded back to back on 2026-09-18: C3896 at 12:18–12:24 and C3897 at 12:24–12:30 (camera clock, +06:00). The unlisted clip numbers C3898–C3901, C3903 and C3904 are therefore plausibly the hidden test set: about 6 clips of about 5.5 min each, roughly 30–35 min in total, recorded in the same midday session. If that holds:
- the test length is T ≈ 0.55 h, which sets q for class gating and the Part B false-alarm budget;
- accidents are probably few (Part B may be unscored);
- the night, IR and rain work in several reports is low priority.

Once the samples are local (or using the 1080p proxies):

(1) Build median background plates per video, one frame every 2 s. Measure inter-video misalignment against a reference plate. Use ORB or AKAZE features with RANSAC homography, and also cv2.findTransformECC with MOTION_HOMOGRAPHY on downscaled grayscale, with the road region masked in and vehicles masked out. Measure intra-video jitter as a phase-correlation shift per second against the plate. Report max and median shift in 4K pixels.

(2) Specify and prototype register_scene(video_path):
- median plate, then a homography to the reference, then warp every scene.yaml element (polygons, polylines, lamp ROIs, homography image points);
- quality gates: inlier count and reprojection error;
- fallback on failure: the unwarped config, with geometry-fragile classes disabled.
Also write a causal variant for RiskEstimator that registers from the first N received frames. That is legal, because it uses only frames the estimator has received. Benchmark the cost (target under 1 s per video).

(3) Parse the rest of the Sony NonRealTimeMeta or exiftool output for stabilisation, focal length or zoom, and exposure mode (the tail bytes are in the scratchpad). Plot mean luma and lamp-ROI brightness per second to detect auto-exposure steps.

(4) Draft the channel questions:
- Are the test clips from the same recording session and tripod placement as the samples (e.g. C3898–C3904)?
- How many test videos are there, and how long are they in total?
- Is there any night footage?

(5) Recompute the class-gating q = 1 − exp(−λT) and the Part B false-alarm budget for T ≈ 0.5 h. List which night/rain/IR work items to deprioritise.

Output: measured offsets, the registration-module spec with thresholds, the fallback policy, the organizer question text, and updated planning assumptions.

### Using the files' audio track as an extra evidence source for accident and near_miss (Part A only)
The camera files carry 2-channel LPCM16 audio (Sony metadata). Part A gets the video path and 'may read any way', and only its output list is scored, so audio is allowed. For accident, an impact sound would give a start anchor accurate to about 0.1–0.3 s once sound travel time is corrected. That is the precision IoU 0.7 needs on 2–6 s accident segments, and it is a second, independent evidence family, which the accident and gating analyses require to reach q≈0 without a VLM. Horn and tyre-screech onsets could corroborate near_miss, and a siren could help classify the aftermath. Decoding the audio costs almost nothing next to 4K video decoding. No report mentions audio. The risks: the test files could be re-encoded without audio, urban noise could produce many false onsets, and Part B cannot use audio because it only receives frames (that is fine).

(1) Check the audio streams on the samples with ffprobe, and draft a channel question asking whether the test files keep their audio track.

(2) Survey open-weights audio event taggers usable offline on CPU or T4. For each, give the exact code and weights licence, the size, and the CPU latency per minute of audio. Candidates: PANNs CNN14/MobileNetV2 (check the Zenodo weights licence), YAMNet (Apache-2.0; check PyTorch ports), EfficientAT, AST and BEATs. List the AudioSet classes that matter: Smash/crash, Skidding, Tire squeal, Vehicle horn/car horn/honking, Glass/shatter, Siren, Explosion.

(3) Build a zero-training baseline: a spectral-flux or short-time-energy onset detector on 16 kHz mono against a rolling median. Measure false onsets per hour on all sample audio, and tagger hits per hour per class.

(4) Find public road audio with crash timing to measure detection and timing error. Check the licence and availability of the MIVIA road audio events dataset (car crashes and tyre skidding), and any ACCIDENT/TAD clips that still have audio.

(5) Specify the fusion rule:
- An audio impact onset within ±1 s of a kinematic contact candidate snaps the accident start to (audio onset − distance/343 m/s), with distance taken from the homography.
- Audio alone never emits an event.
- Horn or screech within ±2 s of a TTC conflict raises near_miss confidence.
Estimate the added Part A cost.

Output: a go/no-go recommendation, backed by measured false positives per hour on the samples, the chosen model and its licence, and an integration spec with thresholds.


## Contradictions
- **Task spec text vs the actual sample files (and every report's budget assumptions)**: The spec says typical clips are 'several minutes long at 25 fps', and the example GT has fps 25. The spec also says the samples have the same resolution and fps as the test set. The Sony metadata in C3896 and C3897 shows 3840x2160 at 29.97p, H.264 High 4:2:2 10-bit at about 140 Mbps. Several budgets were built on 25 fps and mostly 1080p: the rules report's 120 ms per frame, the stack report's 1080p rows, the rare report's '1080p throughout', and the engineering and partb decode estimates.
  - Resolution: Treat 4K/29.97/4:2:2-10bit as the baseline until the organizers say the test clips are transcoded. Express every threshold, buffer and window in seconds, never in frames. Choose the plan from meta (fps, width, height) with both profiles supported. Ask in the channel right away. See gap 1.
- **Team-situation context ('only the PDF') vs the files on disk; engineering report vs the harness code**: wiut_cv_scripts.zip and Videos.pdf (4 Drive links) are on the Desktop, and the kit is unpacked in scratchpad\kit. The engineering report still treats harness behaviour as unknown and lists channel questions the code already answers. The code shows: import/load_solution is untimed; there is no signal or kill, only an after-the-fact budget check plus a Part B check every 100 frames; t = idx/fps; duration = CAP_PROP_FRAME_COUNT/fps; a new RiskEstimator is built per video; an exception inside step() empties that video's whole curve; NaN and numpy scalars are coerced.
  - Resolution: Base all engineering decisions on run_submission.py and evaluate.py. Drop the organizer questions the code already answers.
- **Stack (NVDEC/torchcodec for 4K Part A) vs partb (NVDEC at 4K) vs engineering (PyAV stride decode) vs engineering-verifier (use the same decoder as the harness)**: Four different decode paths are proposed. At the real format, T4 NVDEC most likely cannot decode H.264 4:2:2 10-bit. torchcodec also needs a system FFmpeg and has CPU-only wheels on Windows. PyAV 18 needs Python 3.11 or newer. A different decoder in Part A can also produce slightly different pixels and frame counts from the harness.
  - Resolution: Decode on the CPU only. Use OpenCV CAP_FFMPEG, the same backend as the harness, for the sequential Part A pass: grab() every frame and retrieve() only the frames you process. Use PyAV (pinned to 17.1.0 for cp310) only if a benchmark shows reformat-with-downscale is clearly faster and a parity check against cv2 passes. Otherwise use PyAV only for random-access refinement windows. Drop the NVDEC path.
- **Time-budget splits: metric (≤1.5x total) vs stack (Part A ≤1.5x, total ≤1.0x on Kaggle T4, brake at 1.8x) vs engineering (stop optional stages at 1.8x, return partial at 2.4x) vs accident (skip VLM after 1.5x) vs partb (Part B ≤0.6x, A+B ≤2.0x)**: There is no single end-to-end budget. The verifiers showed that 2.4x leaves Part B no room, and that the 1.8x brake 'only triggers when it would otherwise score zero' is false. None of these budgets includes the harness's full-resolution 4K decode, which is about 1.33x on the dev laptop at 29.97p.
  - Resolution: Use one formula: Part A hard stop T_A = 3·dur − 1.2·(measured Part B cost including harness decode) − 0.2·dur. At 4K this is about 1.0–1.2x. Order Part A stages by value so a cutoff drops refinement, tiling and VLM first. Compute everything from meta plus a cost table measured on a T4 (gap 1).
- **Part B perception settings: stack (YOLO26s@960, stride 2–3) vs partb (imgsz 640; stride 2 up to 1080p, 3 above) vs metric (k = 2–3) vs delivery demo (YOLO26n@640)**: The recommendations disagree. The stack verifier also showed that with Ultralytics' default fuse_score and the hard-coded 0.7 cost for unconfirmed tracks, stride 3 cannot confirm new tracks for fast vehicles. Partb's own rule would pick stride 3 for the 4K samples.
  - Resolution: At 29.97 fps, stride 3 gives about 10 Hz, similar to stride 2.5 at 25 fps. Use it only with tracker fixes: fuse_score=False, or Roboflow `trackers` ByteTrack with frame_rate scaling, or a lower birth threshold. Validate association on fast vehicles in the samples. Set imgsz from measured far-field object sizes at 4K; a road crop at 960 is the likely choice. The demo can use a smaller configuration if the page says so.
- **Part A stride: stack (≤2 for vehicles plus stride-1 refinement) vs metric ('stride barely matters up to 5') vs rare (≥5 fps tracking) vs delivery (3) vs engineering (2 plus stride-1 refinement)**: The metric report treats stride as a runtime-only choice. The stack and rules reports show stride drives tracker association, contact detection and TTC.
  - Resolution: Stride is a tracking decision, not an IoU decision. Run the main pass at about 12–15 Hz (stride 2 at 29.97p). Re-decode stride-1 refinement windows around candidates for short classes (accident, near_miss, red_light, failure_to_yield, solid_line_crossing). Run the static-scene engine at 5 Hz.
- **Tracker buffer settings: rules (track_buffer 75 '= 3 s') vs stack (ceil(3 s·fps_eff), 38 at stride 2) vs partb pseudo-code (int(1.5·fps/stride)) vs rare (default 30)**: Each library counts the buffer differently. Ultralytics counts processed updates, so rules' 75 is about 6 s at stride 2. FoundationVision and trackers rescale by frame_rate/30, which turns partb's intended 1.5 s into about 0.6 s. The values were also derived for 25 fps.
  - Resolution: Keep one tracker-config module. Specify buffers in seconds and convert them per library. Recompute for 29.97 fps and the chosen stride.
- **Tracklet stitching: stack (gap ≤2 s; ≤20 s if stationary within 0.3 box diagonals) vs rules (≤4 s; ≤30 s stationary within 1.5 m) vs rare (ghost boxes kept alive by long-term foreground)**: There are three overlapping mechanisms with different parameters.
  - Resolution: Build one Part A stitching module. For moving tracks: gap ≤2–4 s with a constant-velocity gate. For stationary tracks: ≤30 s, plus the rare report's ghost-box and long-term-foreground logic for occlusion. Tune against ID-switch counts on dev clips.
- **Braking and evasive-action criteria: rules (decel ≥3.4 m/s² for ≥0.3 s with SG windows ≥1.5 s) vs accident (Δv over 1 s ≤ −3 m/s, hinge-fit onset) vs partb (≤ −3.5 m/s² for ≥0.4 s with ≥1.5 s window) vs stack (|Δv| > 3 m/s within 0.5 s)**: The rules verifier showed the rules criterion is self-inconsistent: after SG(1.5 s) smoothing, a real 0.5 s brake peaks below threshold. The four modules would disagree on the same event.
  - Resolution: Use one shared kinematics library with a speed-drop criterion: Δv over 1 s ≤ −3 m/s with v_before ≥4 m/s. Part A finds the offline onset with a hinge fit; Part B uses a Kalman velocity. Gate everything to a kinematics-valid zone measured by pixels-per-metre along the direction of travel.
- **near_miss conflict thresholds: rules (TTC ≤1.5 or PET ≤1.0 or DRAC ≥3.35) vs accident (TTC ≤1.5 or PET ≤1.0) vs partb (PET ≤2 elevated, ≤1 critical)**: The thresholds differ. PET 1.0 s is not supported by the cited CAIT study, which uses 1.5 s. DRAC 3.35 is misattributed to FHWA.
  - Resolution: Build one conflict module shared by Part A and Part B, with separate causal and offline modes. Start at TTC ≤1.5 s or PET ≤1.5 s, and require an evasive action and no contact. Tune on the samples.
- **Accident report ('both classes almost certainly in C; an absent class cannot dilute') and rules ('near_miss likely present') vs metric/rare gating rule**: Part B's near-miss ignore rule is generic metric machinery. It does not show that near_miss occurs in the test set. A spurious near_miss adds a zero-F1 class to C.
  - Resolution: Gate near_miss by p·f > (1−p)·q·S̄ at a high-precision operating point. Enable accident, since Part B makes accidents likely, but still run it precision-first.
- **Congestion vs signal queues: stack (signal queues excluded) vs rules (drop runs inside one red phase +10 s; min 30 s) vs rare (strict H1 by default, H2 as a switch) vs spec (no exclusion)**: The reports use three different definitions. The spec excludes signal queues only for stopped_vehicle.
  - Resolution: Ask the organizers. Until they answer, use H1 by default and keep H2 behind a switch. The team's own dev labels cannot settle this, because they encode the team's convention, so label the samples both ways.
- **Congestion scope: rules ('all lanes') vs rare ('≥80% of lanes') vs engineering annotation guide ('one segment per direction')**: The spec says 'across all lanes of a direction'. The FAQ says same-class events that happen at the same time form one segment, and the harness drops later overlapping segments of the same class.
  - Resolution: Require every lane of a direction to be jammed, with occlusion tolerance per lane. Take the union across directions into a single congestion timeline. Fix the annotation guide to match.
- **stopped_vehicle exclusions: rules/rare (suppress inside congestion and yield-waits) and stack (exclude signal queues) vs spec ('not in a queue at a signal'); crash aftermath: rules ('ask') vs accident/rare ('emit both')**: The extra suppressions go beyond the spec. The reports disagree on whether a crashed vehicle that stays put gets its own stopped_vehicle segment.
  - Resolution: Emit stopped_vehicle for crash aftermath, since classes may overlap and the literal definition is met. Keep suppression inside congestion behind a switch until the organizers answer. Exclude signal queues as the spec says.
- **Rules (vehicle that crosses on red and stops before the box = stop_line only) vs rules-verifier (the red_light definition is also met)**: A double label is possible under the literal definitions.
  - Resolution: Default to stop_line only. The stop_line definition describes exactly this case, and red_light's end, 'leaves the intersection', implies the vehicle entered it. Ask the organizers, and label the dev set consistently.
- **Stack and accident (reusing Part A per-frame detections in Part B is a 'grey area, ask') and engineering (question 4) vs partb/metric (prohibited) vs spec FAQ**: The FAQ says 'Part A CAN use Part B's risk curve; reverse NOT allowed', and breaking a rule disqualifies the team.
  - Resolution: Treat it as prohibited. Part B runs its own causal perception inside step(). Shared weights and code are fine; shared outputs are not. Do not plan around getting an exception.
- **Accident report (run RiskEstimator inside detect_events as feature C6) and rules ('A may use B') vs the budget**: Running Part B's causal pass inside Part A pays for decode and detection a second time. At 4K that decode alone is about 0.5–1.3x.
  - Resolution: Do not instantiate RiskEstimator inside Part A unless the budget clearly allows it. Compute the same TTC and conflict features from Part A's own tracks instead. Part A may use any information, including future frames.
- **Partb (lazy module-level singleton) vs engineering (eager load at import with lazy fallback) vs stack ('at import or first call')**: load_solution() runs before any per-video timer. RiskEstimator() is constructed inside each video's timed window. Lazy loading therefore charges model load and CUDA initialisation to the first video.
  - Resolution: Load eagerly at import and run warm-up passes there. Keep module-level singletons that every RiskEstimator instance reuses. Put per-video state only in reset().
- **Stack ('container PTS minus first PTS') and delivery ('PTS-based t_sec, never frame index × fps') vs harness/engineering (t = idx/fps; duration = CAP_PROP_FRAME_COUNT/fps) vs rare (count decoded frames for duration)**: The reports propose different time bases. Any mismatch shifts every boundary.
  - Resolution: Use t = idx/fps and duration = CAP_PROP_FRAME_COUNT/fps everywhere: Part A, labels, website. PTS is only for EDA checks for variable frame rate or dropped frames.
- **Metric (a watchdog that raises stride when behind) and engineering (double Part B stride if cumulative step time > 0.5·t_sec) vs stack/partb (stride from metadata only; never from wall-clock)**: Adapting to wall-clock time conflicts with the rule that two runs must produce the same predictions.json.
  - Resolution: Plan from metadata plus the measured cost table. Wall-clock triggers exist only as emergencies that should never fire given the measured margin, and they log loudly if they do. If one fires, the lost determinism is still better than an empty video.
- **Partb (on exception in step, return the last score) vs partb-verifier (a frozen score ≥0.5 becomes one long unmatched alarm) vs engineering (return 0.0)**: The error-path behaviour differs, and one version can create a very long alarm.
  - Resolution: On an exception, return a decaying score or 0.0, log once, and never hold a value ≥0.5. Use use_deterministic_algorithms(warn_only=True) so an op without a deterministic kernel cannot raise mid-video.
- **Metric (alarm age cap ~8 s with a forced ≥2.1 s dip and re-arm; 'one alarm per conflict'; ~0.2 s EMA; mute a pair for 10 s after contact) vs partb (8 s cap then re-arm only after z drops; habituation; 0.5 s release; bridge gaps ≤3 s for the same pair; 30 s / 15 m suppression after contact)**: The forced re-arm creates a second, false alarm, so the metric report contradicts itself. The verifiers found that the 30 s / 15 m suppression silences chain collisions and relies on an error-prone contact detector. Both 'free ride' claims ignore that the metric merges runs before it filters out ignored alarm starts.
  - Resolution: Keep alarms short. Cap alarm age at 8 s without a forced re-fire. Suppress per pair for ≤10 s after contact. Never let a run that starts inside a post-contact or near-miss window come within 2 s of another pre-crash alarm. Validate every policy with the official evaluate_part_b.
- **Organizers' tip ('calibrate so 0.5 means probably within 5 s') vs metric (alarm threshold ≈ calibrated P of 0.15–0.3) vs partb (F1/2 ≈ 0.25 match probability within 10 s; false-alarm budget of 0.3–0.5 per accident and 1–2 per hour)**: The semantics of the 0.5 crossing differ. Partb's two budgets together imply 3–4 accidents per hour, which the likely ~30-minute test set probably will not have.
  - Resolution: Set the 0.5 crossing from a false-alarm budget tied to the expected number of test accidents, and document this as a deliberate departure from the tip. Keep the score monotone in imminence so AP benefits.
- **Metric TL;DR 3 ('poorly localised IoU <0.5 segments often lower the score; drop them') vs metric §2.2's own rule; accident report ('p_match > F1*/2')**: By the report's own formula, a surely-real candidate at IoU 0.3–0.5 raises F1 whenever F1_c < 0.67. The accident rule ignores that F1* and the match probability both depend on τ.
  - Resolution: Emit when Σ_τ P(match at τ)/3 > F1_c/2, and apply the class-dilution gate separately. Fix boundaries before dropping candidates.
- **Ground-plane scale: rules (8–12-point homography, RANSAC 3.0) vs accident (m/px(y) from car box widths) vs rare (linear row scale s(y)) vs partb ('scale doesn't matter for TTC') vs stack (metric thresholds, no calibration step)**: Verifiers found problems with each. RANSAC's threshold is in destination units, so 3.0 means 3 m. The rare report's k is off by 3.5x, and a row scale cannot convert along-road motion. Partb's own capsules and thresholds are metric. The stack report gives no calibration at all.
  - Resolution: Fit one shared ground-plane homography by least squares (method=0) on at least 8 hand-clicked road points. Check residuals in metres, and sanity-check with the car-length and lane-width distributions. Warp it per video through registration (gap 3). Use car-width auto-scale only for proxy clips from other cameras.
- **Rare (jaywalking starts at the first sample with d ≥ 0.4 m) vs rules (zero-crossing of the signed curb distance) vs spec ('steps onto road')**: Rare's rule starts about 0.3 s late, which costs IoU 0.7 on 5 s events.
  - Resolution: Detect with the margin and hysteresis, then back-date the start and end to the interpolated zero-crossings.
- **Rare (road_obstacle animal branch enabled by default; min_len 3 s but animals need ≥1 s) vs rules (min duration 5 s) vs rare's own zero-FP gating rule**: This is inconsistent inside one report and across reports. Small false 'dog', 'horse' or 'cow' boxes are common on CCTV.
  - Resolution: Put the animal branch behind the same false-positive audit. Use per-branch minimum durations: about 1 s for animals and 3–5 s for static blobs.
- **Rare and rules ('0 FPs on all samples plus 1–2 h external' establishes q≈0 / λ ≤0.05 per hour) vs verifiers (rule of three)**: Zero events in T hours only bounds λ below 3/T. With about 22 minutes of samples, the audit cannot certify the target rate.
  - Resolution: Treat q as uncertain. Enable fire_smoke and the static-blob obstacle path only with evidence from several independent signals and an external negative bank. By default keep them off or at very high precision. Recompute with the test length T ≈ 0.55 h (gap 3).
- **VLM verifier: accident (optional Qwen3-VL-2B; vLLM cannot serve Qwen3-VL on Turing) vs rare (Florence-2-large for yes/no; InternVL3.5-1B) vs engineering (≤2B FP16; VLM cost 0.03–0.09x)**: Verifiers found the vLLM limitation is probably outdated, Florence-2 has no free-form VQA, and the engineering cost arithmetic is about 3x too low: 20 queries is 1.0–2.7x the duration of a 1-minute clip.
  - Resolution: Allow at most one optional shared verifier: InternVL3.5-1B or Qwen3-VL-2B through transformers in fp16 with SDPA, scoring greedy logits. Cap the number of queries in proportion to duration. Keep it only if an ablation shows a gain. Given the 4K decode budget, default it to off.
- **Stack (fire detector YOLO26n on D-Fire at 1–2 fps) vs rare (YOLO26s or YOLO11s at 640, 2 fps, 10–15% scene negatives)**: The two reports pick different model sizes.
  - Resolution: Start with YOLO26n plus scene negatives, and move to s only if recall on injection tests requires it. Apply the same persistence and location filters either way.
- **Stack (ONNX Runtime CPU fallback with YOLO26n) vs engineering (avoid onnxruntime-gpu) vs verifiers (onnxruntime ≥1.24 has no cp310 wheels; ≥1.27 is CUDA 13)**: The fallback would not install on the Python 3.10 baseline the spec allows.
  - Resolution: Build the submission's CPU fallback on PyTorch CPU with the same .pt weights, adding no dependency. Use ONNX or OpenVINO only inside the demo Space.
- **Stack ('cu126/cu128 wheels via --extra-index-url') vs engineering (torch==2.14.0+cu126 pinned)**: There is no cu128 build of torch 2.14, and an unpinned mixed index is fragile.
  - Resolution: Follow the engineering pins: torch==2.14.0+cu126 and torchvision==0.29.0+cu126, headless OpenCV, lap, and numpy/scipy markers for 3.10. Add a Python 3.10 constraints file and a Dockerfile as backup.
- **Delivery (GitHub Pages plus HF PRO CPU Upgrade, ZeroGPU or Modal; free CPU Spaces are paywalled) vs engineering risk register ('HF free CPU tier, keep warm')**: The engineering assumption is out of date. The delivery plan also has its own errors: ZeroGPU admission is charged at the requested duration against the visitor's quota, so replica failover does not help, and a 20 s failover timeout contradicts a 1–2 minute wake-up.
  - Resolution: Follow the delivery plan with the verifier fixes: small or dynamic @spaces.GPU duration, CPU fallback as the main path when quota runs out, and failover driven by status_callback. Add upload handling for 1–2 GB original-format clips (gap 2).
- **Delivery (use PRO 'protected' Space visibility during development) vs AGPL-3.0 §13 (the YOLO26 weights served over the network)**: A demo that serves AGPL code while its source is private conflicts with the licence.
  - Resolution: Keep the demo's serving code in the public repo and link it from the demo page. Protected visibility is fine only if the source remains available to users.
- **Dataset and model licences: accident report (ACCIDENT annotations CC BY 4.0; real-clip labels withheld; TUM Accid3nD CC BY 4.0; VideoMAEv2-S CC-BY-NC) vs verifiers**: Verifiers found: Kaggle lists ACCIDENT as CC BY-NC-SA 4.0, and metadata-real.csv carries impact times for all 2,027 clips; Accid3nD is CC BY-NC-SA 4.0; the VideoMAEv2 distilled checkpoints are Apache-2.0; UA-DETRAC, TAD and TU-DAT have terms that are unverified or unstated.
  - Resolution: Use the verified licences in the README table and list unstated terms as such. Do not relabel ACCIDENT start times; use the published labels. The end boundary still needs the team's own labels.
- **Team plans: metric (P1 Part A; P2 evaluation, Part B and verifiers; P3 website and code) vs engineering (P1 perception and rules; P2 accident, Part B and evaluation; P3 DevOps, EDA and website; milestones M0–M5) vs delivery (one member about 50% on web and demo; another on EDA and dev labels) vs accident (2–3 week build order) vs partb (one person-week)**: There are five partial plans with overlapping roles, and the deadline is unknown. None includes the day-0 data and decode work that the real format requires.
  - Resolution: Adopt the engineering roles and milestones, using the metric report's value weights. P1: perception, geometry, registration and rules. P2: dev labels, evaluate loop, accident/near_miss and Part B, capped at about 1 person-week. P3: packaging, website, demo, data logistics and proxies. Everyone annotates. Add a D0 task to download the samples, make proxies and benchmark decode on a T4. Compress per engineering §11 if the deadline is less than a week away.
- **Class default-on sets: metric Tier 1 (stopped_vehicle, congestion, jaywalking, wrong_way, accident) vs rules (stopped_vehicle, wrong_way, jaywalking, accident; congestion conditional) vs rare (stopped_vehicle, jaywalking, congestion; plus animals)**: The starting sets are slightly different.
  - Resolution: Start with stopped_vehicle, jaywalking, wrong_way, congestion (H1) and accident. Enable signal, crossing and line classes only if camera.md supports them. Make every final decision with the gating rule, using false positives per hour measured on the samples.
- **Rules post-processing (clip, then round to 0.01 s) vs engineering (round to 3 dp) vs harness (clamp, then validity check, then round to 3 dp)**: Clipping before rounding can push an end past the duration. An event shorter than 1 ms survives the harness as [x, x], and evaluate.py then rejects the whole file.
  - Resolution: In the sanitizer: enforce end − start ≥ the class minimum duration (always ≥0.05 s), round to 3 dp, then clamp end to floor(duration, 3). Run evaluate.py --validate-only on predictions_samples.json in CI.
- **Rare, partb and accident (substantial night, IR, rain and glare work) vs sample metadata (recorded 2026-09-18 between about 12:18 and 12:30 camera time, back to back, rec709 daylight profile)**: If the test clips are the missing numbers from the same session (C3898–C3904), night and weather robustness is unlikely to matter. The effort is better spent on budget, registration and boundaries.
  - Resolution: Push night, IR and rain work to the end of the plan unless the organizers confirm night test footage exists (gap 3). Keep only cheap robustness such as the global-change guard.
