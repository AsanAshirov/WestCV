<!-- source: deep-research workflow wf_294a4b01-fc2, agent gap:4 -->

# Audio track as extra evidence for `accident` / `near_miss` (Part A only)

## 0. Verdict

**Conditional GO, in a narrow role.** Use a zero-training impact-onset detector on the audio track for two jobs:
1. Snap the start of an accident that the visual pipeline has already proposed to the impact sound, after correcting for sound travel time.
2. Give that visual accident candidate a small confidence bonus.

Optionally, PANNs Cnn14_16k can add a skid/horn bonus to `near_miss`. Audio must never create an event, a class-based ("tagger") crash detector is **NO-GO**, and Part B cannot use audio at all.

Conditions before shipping:
- (a) the organizers confirm the test files keep their audio track, or the code turns audio off cleanly when it is missing;
- (b) the false-onset rate measured on the real samples roughly matches the numbers below;
- (c) an A/B test on the team's own dev labels shows no loss at IoU 0.7.

The work is small (about one person-day) and adds less than about 2 s per video-minute.

Numbers behind the verdict. All were measured by me; the sample audio itself could not be downloaded (see §1).
- **False onsets per hour on 105.9 min of real intersection audio** (15 CC0 Freesound field recordings):
  - "strong" tier: 32.9/h, so P(a random ±1 s window contains one) = 0.013;
  - "weak" tier: 120.7/h, so P = 0.055;
  - without the flatness gate, the strong tier is 75.9/h (P = 0.039).
- **Crashes injected into that audio** (14 CC0 crash clips × 10 positions × 6 levels = 840 trials):
  - weak tier finds the impact within ±1 s in 52% / 83% / 91% of trials at event-to-background ratio (EBR) 0 / +5 / +10 dB;
  - of the onsets found, 84–86% are within 0.1 s of the reference impact at ≥ +5 dB, and the median error is 0.00–0.01 s.
- **AudioSet taggers cannot detect crashes.** PANNs Cnn14_16k found only 6–19% of injected crashes at p ≥ 0.1 (EBR 0 to +10 dB).
  - Clean crash sound effects are labelled "Breaking", "Burst/pop", "Gunshot" or "Explosion", never "Smash, crash".
  - Skids work much better: 33–53% recall at p ≥ 0.2 at +5/+10 dB, with **0 background hits/h**.

## 1. What the sample files contain, and the channel question

The samples are 4 Google Drive links (listed in `Videos.pdf`). By HTTP range requests I read the top-level atoms, the `moov` atom and the Sony XML of file 1 (`C3896.MP4`, 6,241,380,481 bytes). File 2 is `C3897.MP4`, 5,838,719,827 bytes.

What the metadata shows:
- **Camera:** Sony ILCE-6700.
- **Video:** `AVC140_3840_2160_H422P@L51`, 3840×2160, **29.97p**, 10,200 frames = 340.34 s. This contradicts the spec's "25 fps"; the other agents should know.
- **Recording time:** created 2026‑09‑18 12:18 (+06:00), so daytime footage.
- **Audio:** track `twos` (16‑bit **big-endian** LPCM), 2 channels, 48 kHz. The XML says `AudioFormat numOfChannel="2"` with `audioCodec="LPCM16"`.
- **Audio layout:** 680 chunks of 24,024 samples (0.5005 s = 15 video frames) interleaved every ~9.1 MB. Audio is therefore about 65 MB of a 6.2 GB file.
- There is also a Sony `rtmd` real-time metadata track.

**The waveform could not be extracted.** Drive's "download quota exceeded" state flipped on and off. Ranges up to 64 KB worked at first; then the allowed size shrank (8 KB, then 4 KB); then every request was refused (662 refused requests over 27 min). As a result, **step (3) "false onsets on sample audio" is still open.**

To finish it once the files are local:
```
ffprobe -v error -select_streams a -show_entries stream=index,codec_name,codec_tag_string,sample_rate,channels,duration -of compact samples/*.MP4
python sample_audio_stats.py samples/   # onset rates per tier, P(±1 s hit), tag hits/h, transient-level percentiles
```
For partial downloads, `moov.py` and `bgfetch.py` pull only the audio chunks, retrying through the quota.

**Draft channel question:**
> The sample MP4s (Sony XAVC, e.g. C3896.MP4) contain a 2‑channel 48 kHz LPCM audio track and are 29.97 fps. (1) Will the hidden test videos be delivered in the same form (original container, audio track kept), or trimmed/re‑encoded (e.g. to 25 fps as the PDF says) and possibly with audio removed? (2) We plan to use the audio track only inside `detect_events` (Part A) as a secondary cue; `RiskEstimator` stays frames-only. Please confirm this is allowed.

Robustness requirement: the code must run correctly with no audio stream, a silent stream, or a re-encoded AAC stream. AAC smears transients by only tens of ms (my estimate), which is harmless at this timing scale.

## 2. Open-weights audio taggers

All latencies are measured on my i5‑12450H CPU with PyTorch 2.12, and are per minute of audio.

| Model | Code licence | Weights licence / hosting | Size | CPU latency | Notes |
|---|---|---|---|---|---|
| **PANNs Cnn14_16k** (mAP 0.438) | MIT | **CC‑BY‑4.0**, Zenodo 3987831 | 358.7 MB | 5‑s windows, 1‑s hop: **5.9 s/min** (8 threads, including resampling). 2‑s windows, 1‑s hop: 2.3 s/min (8 thr), 8.8 s/min (1 thr) | Best of those tested. 16 kHz input. Needs README attribution. `torchlibrosa` declares no licence, so use a torchaudio front-end (done in `tagger.py`). |
| PANNs MobileNetV2 (0.383) | MIT | CC‑BY‑4.0, Zenodo | 20.8 MB | 0.44 s/min (8 thr), 1.0 s/min (1 thr) | Timed with random weights; accuracy not tested. |
| **EfficientAT mn10_as** (47.1 mAP, 4.88 M params) / mn04_as (43.2 mAP, 0.98 M) | MIT | GitHub release v0.0.1 (weights under the repo's MIT licence) | 19.7 MB / 4.1 MB | mn10 with 5‑s windows: 2.4 s/min (8 thr) | **Bug found (tested on mn10):** inputs shorter than about 4 s give saturated nonsense (every class at 1.0) on both CPU and CUDA. Use windows of 5 s or more. In our tests it was weaker than PANNs on skids and horns. |
| YAMNet | Apache‑2.0 (tensorflow/models `/research`; Kaggle model page "Apache 2.0") | `storage.googleapis.com/audioset/yamnet.h5` | 3.7 M params | not measured (TensorFlow) | 16 kHz, 0.96‑s patches, 0.48‑s hop, 521 classes. PyTorch port `w-hc/torch_audioset` (MIT). |
| AST (MIT/ast‑finetuned‑audioset‑10‑10‑0.4593) | BSD‑3 | BSD‑3 on Hugging Face | 346 MB | Equivalent ViT‑B encoder (1,214 tokens): **2.6 s per 10.24‑s window** on CPU, about 15.6 s/min without overlap | Too slow on CPU; possible on T4 for candidate windows only. |
| BEATs (iter3+, AS2M fine-tuned) | MIT (microsoft/unilm) | OneDrive links; re-host inside `weights/` | ~90 M params | not measured (ViT‑B-class, similar to AST) | No advantage for this use. |

AudioSet class indices that matter (0-based, from `class_labels_indices.csv`):
- **Impact:** Smash, crash 469; Breaking 470; Bang 466; Thump, thud 460; Slam 358; Crushing 478; Glass 441; Shatter 443; Explosion 426. Also "Burst, pop" and "Gunshot, gunfire", which clean crash clips often trigger.
- **Skid:** Skidding 312; Tire squeal 313; Squeal 485.
- **Horn:** Vehicle horn, car horn, honking 308; Air horn, truck horn 318; Toot 309; Honk 107 (this is the goose sound, ignore); Car alarm 310.
- **Siren:** Siren 396; Police car 323; Ambulance 324; Fire engine 325; Emergency vehicle 322.
- **Other:** Air brake 317.

**Tagger hits per hour on 106 min of real intersection audio** (PANNs, 5‑s windows, p ≥ 0.3; mn10 in brackets):
- **0/h:** all impact classes, Skidding, Tire squeal, Siren (mn10 also 0 for all of these);
- Vehicle horn 9.6 (5.1); Toot 43.6 (4.0); Air brake 5.7 (0); Car alarm 1.1 (0).

Horns are genuinely frequent in traffic, so a horn is weak evidence of a near miss.

## 3. Zero-training onset baseline

Implemented in `onset.py`; 0.126 s CPU per audio-minute.

**Pipeline:**
1. Decode to 16 kHz mono.
2. 4th-order Butterworth high-pass at 250 Hz, to remove engine rumble and wind.
3. STFT with 512-sample windows, hop 160 (10 ms), Hann window; keep the 250–7000 Hz band; compress as `L = log1p(100·|X|)`.
4. Spectral flux (SuperFlux style): lag of 2 frames, 3‑bin maximum filter across frequency, half-wave rectified sum.
5. Normalize: `z = (SF − med_20s)/(1.4826·MAD_20s)`.
6. Energy jump: `dE = max E[t, t+150 ms] − median E[t−2 s, t−0.2 s]` (band energy in dB).
7. Peak picking: local maximum within ±50 ms, 0.5 s refractory period.
8. **Flatness gate:** spectral flatness over the 100 ms after the onset. Impacts are broadband: median 0.197 on detected injected crashes. Background onsets (horns, whistles) are tonal: median 0.016.

**Background false onsets** (105.9 min, 15 recordings; Bangkok, Havana, Montreal, Gaza, Hamburg, Albi and others):

| Tier | onsets/h | P(hit in random ±1 s window) |
|---|---|---|
| z>6, dE>8 (no gate) | 188.6 | 0.086 |
| **weak:** z>6, dE>8, flat ≥ 0.02 | **120.7** | **0.055** |
| z>8, dE>12 (no gate) | 75.9 | 0.039 |
| **strong:** z>8, dE>12, flat ≥ 0.02 | **32.9** | **0.013** |
| z>10, dE>15, flat ≥ 0.02 | 14.2 | 0.006 |

False onsets cluster in a few recordings: 47 of 124 strong-tier (ungated) hits come from one recording with a police whistle, and 20 from one with heavy honking. Expect scene-dependent rates, so measure on the samples.

**Crash injection.** EBR is measured in the 250–7000 Hz band: event power over the 0.5 s after impact, divided by mean background power over 30 s. For scale, ordinary traffic transients in 0.5‑s windows reach +2.7 / +8.2 / +12.9 / +15.0 dB above the 30‑s mean at the 90th / 99th / 99.9th percentile and the maximum. EBR +5 dB is therefore within the loud tail of ordinary traffic; detection relies on the sharp rise, not on loudness alone.

| Tier | EBR 0 | +5 | +10 | Share of found onsets with abs. error ≤0.1 s (at +5) | Median error |
|---|---|---|---|---|---|
| weak | 0.52 | 0.83 | 0.91 | 0.84 | +0.00 s |
| strong | 0.19 | 0.54 | 0.80 | 0.83 | +0.01 s |

At EBR ≤ −5 dB recall is ≤ 0.23, so a quiet or distant crash will be missed. About 15% of the onsets found lock onto a later peak within ±1 s (errors of 0.2–0.4 s).

What EBR a real collision in this scene produces is **unknown**. The public accident video sets are silent (§4), so there is no real reference. My reasoning-based expectation is that a contact audible on the camera mic is at least as loud as a passing truck, i.e. at least 0 dB, but this must be checked.

**Likelihood ratios** (my calculation from the numbers above): a strong-tier onset near the visual candidate has LR ≈ 0.54/0.013 ≈ 40 at +5 dB. Missing audio is weak evidence against: LR ≈ 0.47 at +5 dB, and unknown EBR makes it weaker still. So a hit gives a bonus, and a miss gives **no penalty**.

**IoU effect** (illustrative simulation; the error models are **assumptions**). Accident length uniform on 2–6 s, end error sd 0.5 s, weak-tier snapping at EBR +5. P(IoU ≥ 0.7) changes as follows:

| Visual start error (sd) | Visual only | With audio snap |
|---|---|---|
| 0.3 s | 0.916 | 0.942 |
| 0.5 s | 0.845 | 0.930 |
| 0.8 s | 0.719 | 0.908 |

The gain grows with how uncertain the visual contact frame is (occlusion, bounding-box overlap before real contact).

## 4. Public road audio with crash timing

- **MIVIA Road Audio Events:** 200 crashes (326 s) and 200 tyre skids (522 s) over about 2,732 s of real road background. Recorded with an Axis P8221 module and a T83 microphone, 32 kHz, 16‑bit WAV, 4 folds. **Registration is required and no licence is stated.** Use it for dev measurement only; do not ship it or train shipped weights on it until the terms are clear. If access arrives, rerun `inject.py`/`analyze.py` on it; it gives exact event timings.
- **TAD** (500 surveillance videos): only extracted frames are distributed, so no audio. **DoTA:** frame archives (raw videos only via YouTube dashcam downloads). **CCD:** 5‑s MP4 clips at 10 fps; audio not mentioned, so assume none. None of these provides crash audio with timing. I did not verify UCF‑Crime's audio.
- **What I used instead:** CC0 Freesound recordings — 15 intersection ambiences (106 min), 14 crash clips, 7 skid clips and 8 horn clips, mixed at controlled EBR. This is a synthetic test and is labelled as such.

## 5. Integration spec (Part A, `src/audio_evidence.py`)

**Loading:**
- Use `ffmpeg -v error -i video -map 0:a:0 -ac 1 -ar 16000 -f f32le pipe:`, with the binary from `imageio-ffmpeg` (BSD‑2) or PyAV (BSD‑3).
- Measured on a 60‑s, 148 Mbps 4K file with `twos` audio: **0.3–0.5 s/min** with the ffmpeg CLI and 1.3 s/min with PyAV. Video is never decoded; the demuxer seeks from one audio chunk to the next.
- Turn audio off (and log it) if: there is no stream; decoding fails; RMS < −60 dBFS; or the audio duration differs from the video duration by more than 1 s.

**Calibration (once, on the samples):**
- Sound delay τ = d/343 + δ_AV, where 343 m/s is the speed of sound at about 20 °C.
- d = √(dx² + dy² + h²): distance from the contact point (ground-plane homography) to the camera's ground position, which is hard-coded from `camera.md` or a satellite view (allowed scene facts). h is the camera height; an error of a few metres in h changes τ by only a few ms. For d = 15–80 m, d/343 = 44–233 ms, i.e. 1–7 frames.
- δ_AV (camera audio/video offset): cross-correlate the audio band envelope (100 ms smoothing, 10‑s median removed) with a predicted loudness curve `Σ_i 1/max(d_i(t), 5 m)²` over tracked vehicles, searching lags in [−1, +1] s. The peak lag ≈ δ_AV plus a small near-field delay.

**Accident start snap** (for each visual accident candidate with contact time t_v and distance d):
```python
tau = d/343 + dAV
C = [o for o in onsets if o.z > 6 and o.dE > 8 and o.flat >= 0.02 and abs(o.t - tau - t_v) <= 1.0]
if C:
    o = max(C, key=lambda o: o.z * exp(-abs(o.t - tau - t_v) / 0.5))
    strong = o.z > 8 and o.dE > 12
    if strong or abs(o.t - tau - t_v) <= 0.5:
        start = clamp(floor((o.t - tau) * fps) / fps, 0, end - 1/fps)
        conf += 0.15 if strong else 0.07          # tune on dev labels
# optional: PANNs 5-s window [t_v-2, t_v+3], impact family (Breaking/Shatter/Glass/Bang/Explosion/
# Burst,pop/Gunshot/Smash,crash) p >= 0.1 -> conf += 0.05  (0 background hits/h at 0.2; bg 1.1 windows/h at 0.1)
# no audio / no onset -> no change (never a penalty); audio alone -> never an event
```

**Near-miss support** (for a TTC conflict [t0, t1]; PANNs 5‑s windows only on candidates):
- max p(Skidding, Tire squeal) ≥ 0.2 in windows overlapping [t0−1, t1+1]: **+0.10** (measured 0 background windows/h; 33% / 53% recall at +5 / +10 dB).
- The same at ≥ 0.1: **+0.05** (2.3 background windows/h).
- p(Vehicle horn, Air horn) ≥ 0.3 within [t0−2, t1+2]: **+0.03 only**, because horns are common (vehicle_horn 9.6 hits/h).
- Optional: move the near_miss start ("onset of evasive action") to the first skid-positive window only if it is within 1 s of the visual onset. Validate on dev labels first; the tagger's time resolution is only 1 s.

**Siren** (Siren or Emergency-vehicle family ≥ 0.3; 0 background hits/h) is useful only for the website dashboard ("response arrived"). It is not scored.

**Added Part A cost:**

| Component | Cost per video-minute |
|---|---|
| Decode | 0.3–1.3 s |
| Onset detector | 0.13 s |
| PANNs on candidate windows only | about 0.1 s per candidate on CPU (my estimate from 5.9 s per 60 windows) |
| **Total** | **≲ 2 s, about 1% of the 180 s/min budget (3 × duration)** |

Full-coverage PANNs on CPU would add +5.9 s/min (3%). On a T4 it is negligible (not measured). Weights: Cnn14_16k is 358.7 MB (CC‑BY‑4.0: credit Kong et al. in the README), well within 5 GB. The onset detector needs no weights.

## 6. Risks and cross-cutting notes

- **Test audio stripped or re-encoded:** ask in the channel. The code must degrade silently to vision-only, and the dev A/B must also be run with audio turned off.
- **Background rates come from other microphones:** these were field recorders, not the a6700 at this site. Rerun `sample_audio_stats.py` on the samples. If the strong-tier rate there is above ~70/h, tighten to z > 10 and dE > 15 (14.2/h here).
- **Correlated false positives:** a passing truck can cause both a visual overlap false positive and a loud sound. Keep the audio bonus small (≤ 0.15) so audio cannot confirm a weak visual candidate on its own.
- **Part B:** audio is forbidden there (the estimator only receives frames). Part A may use Part B's risk curve, never the reverse.
- **Website:** an audio-waveform and onset lane on the event timeline is a cheap extra-credit visualization. Do not publish speech content from the audio.
- **Other agents:** the sample files are 29.97 fps, 4K, 10-bit 4:2:2, about 140 Mbps, about 6 GB per 5.7 min. This matters for the video decode budget and for 25-fps assumptions.

**Scripts** (all in `C:\Users\Cicada\AppData\Local\Temp\claude\C--Users-Cicada-Desktop-WestHack\ced7e0d9-be88-47bc-ac13-90898f4ae082\scratchpad\audio\`):
- core: `onset.py`, `tagger.py` (EfficientAT and PANNs wrappers);
- experiments: `inject.py`, `analyze.py`, `analyze2*.py`, `err_sim.py`, `bg_full.py`;
- for the samples: `sample_audio_stats.py`;
- MP4 audio extraction by range requests: `moov.py`, `bgfetch.py`;
- results: `inject_*.json`, `bg_onsets_full.json`.


## Key claims (as submitted for verification)
- Sample file C3896.MP4 (Sony ILCE-6700) is 3840x2160 29.97p AVC 4:2:2 10-bit (AVC140), 340.34 s, with a 2-channel 48 kHz 16-bit big-endian LPCM ('twos') audio track interleaved in 0.5005 s chunks — measured: moov atom + Sony NonRealTimeMeta XML read by HTTP range requests from https://drive.google.com/file/d/1kR9jODA2Wotw4gwkvpRKdqFADNJNc1nS (Videos.pdf link)
- PANNs pretrained weights on Zenodo record 3987831 are CC-BY-4.0; Cnn14_16k_mAP=0.438.pth is 358.7 MB, MobileNetV2 20.8 MB — https://zenodo.org/records/3987831
- PANNs code repository (audioset_tagging_cnn) is MIT licensed; Cnn14_16k uses 16 kHz, window 512, hop 160, 64 mel bins — https://github.com/qiuqiangkong/audioset_tagging_cnn
- EfficientAT is MIT licensed; mn10_as has 4.88M params and 47.1 mAP, mn04_as 0.983M params and 43.2 mAP; weights in GitHub release v0.0.1 (mn10_as_mAP_471.pt, 19.7 MB) — https://github.com/fschmid56/EfficientAT and https://github.com/fschmid56/EfficientAT/releases/tag/v0.0.1
- EfficientAT mn10_as produces saturated all-class outputs for inputs shorter than ~4 s (tested 1-3 s) on both CPU and CUDA; 5 s windows work — measured locally (scratchpad/audio/tagger.py tests)
- YAMNet is Apache 2.0 (Kaggle model instances), 3.7M weights, 16 kHz input, 0.96 s patches with 0.48 s hop, 521 classes; weights at storage.googleapis.com/audioset/yamnet.h5 — https://www.kaggle.com/models/google/yamnet and https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- AST AudioSet checkpoint MIT/ast-finetuned-audioset-10-10-0.4593 is BSD-3-Clause, 346 MB safetensors, 1024-frame x 128-mel input, 12 layers — https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
- MIVIA Road Audio Events: 200 car crashes and 200 tyre skids over ~2,732 s road background, 32 kHz 16-bit WAV; download requires registration and no licence is stated — https://mivia.unisa.it/datasets/audio-analysis/mivia-road-audio-events-data-set/
- TAD distributes extracted frames (no audio); CCD distributes 50-frame 10 fps MP4 clips; DoTA distributes frame archives — https://github.com/ktr-hubrt/WSAL ; https://github.com/Cogito2012/CarCrashDataset ; https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly
- Zero-training onset detector false-onset rate on 105.9 min of real CC0 intersection recordings: 32.9/h (strong tier with flatness gate, P(hit in random +-1 s)=0.013), 120.7/h (weak tier, P=0.055), 75.9/h strong tier without flatness gate — measured locally on 15 Freesound CC0 recordings (e.g. https://freesound.org/people/kyles/sounds/407331/)
- Injected crash detection (840 trials): weak tier finds impact within +-1 s in 52/83/91% at EBR 0/+5/+10 dB; 84-86% of found onsets within 0.1 s at >=+5 dB; median error ~0.00-0.01 s — measured locally (inject.py/err_sim.py with CC0 Freesound crash clips)
- AudioSet taggers barely detect injected crashes: PANNs Cnn14_16k impact-family recall at p>=0.1 is 6/10/19% at EBR 0/+5/+10 dB; skids 33/53% at p>=0.2 at +5/+10 dB with 0 background hits/h — measured locally (analyze2_panns.py)
- CPU latency per minute of audio on i5-12450H (8 threads): onset detector 0.13 s; PANNs Cnn14_16k 2.3 s (2 s windows) / 5.9 s (5 s windows, 1 s hop); PANNs MobileNetV2 0.44 s; EfficientAT mn10 2.4 s (5 s windows); ViT-B AST-equivalent 2.6 s per 10.24 s window — measured locally (bench.py, bench_eat.py, run_tag_bg.py)
- Audio-only extraction from a 60 s, ~148 Mbps 4K MP4/MOV with twos audio takes 0.3-0.5 s via ffmpeg CLI and ~1.3 s via PyAV 16.1 — measured locally (decode_bench.py)
- Google Drive sample links hit 'download quota exceeded' repeatedly; range requests >4-16 KB were refused, so sample audio could not be extracted in this session — measured (bgfetch.log: 662 refused requests)
- Sound propagates at ~343 m/s (20 C), so a 15-80 m contact distance delays the impact sound by 44-233 ms (1-7 frames at 29.97 fps) — reasoning
- Using audio only inside detect_events is consistent with the spec (Part A may read the file any way; RiskEstimator must not open the video) — spec
