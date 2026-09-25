<!-- source: deep-research workflow wf_294a4b01-fc2, agent gap:2 -->

# Data logistics, browser compatibility and the demo upload path: runbook, check script, codec table, upload spec

## 0. Summary. Parts of the planning premise are wrong.

**Metadata from the real files.** I read the MP4 tails already in the scratchpad (`tail_C3896.bin`, `tail_C3897.bin`, which contain the Sony XML and the moov box):
- Camera: Sony ILCE-6700. Codec field: `AVC140_3840_2160_H422P@L51`, i.e. H.264 High 4:2:2 at Level 5.1, 140 Mbps.
- Frame rate: 29.97p, i.e. 30000/1001. It is not 25 fps.
- Audio: LPCM16 stereo (`twos`). The moov box sits at the end of the file.
- Keyframes fall on frames 1, 31, 61, …, so the GOP is 30 frames.
- C3896 has 10,200 frames (340.3 s). C3897 has 9,525 frames (317.8 s).
- All four files together are about 1,318 s, which is ≈23.3 GB at 141.5 Mbps (video plus LPCM).

**Browsers.** Chrome 153 and Edge 153 on Windows 11 do decode H.264 High 4:2:2 10-bit and High 10. I tested this in headless mode on this laptop: 1080p and the 4K test clip both played and passed a pixel check. They decode in software (`powerEfficient:false`), and 4K is too slow to use:
- 40 of 244 frames were dropped over 8 s.
- A seek takes 0.5–1.3 s in Chrome and up to 3.2 s in Edge.

So proxies are still required for annotation and for the website, but only for speed and cross-browser reach. The files are not unplayable. Firefox on Windows cannot play them, because the Windows Media Foundation H.264 decoder only handles 4:2:0.

**Local machine:** 11 GB free on C:, 8 GB RAM (measured). It cannot hold the 23 GB of originals. Keep the originals in the cloud and only the proxies locally.

**GPU decode is not available.** The NVIDIA support matrix lists the T4 (Turing) as unable to decode H.264 4:2:2. Blackwell (6th-generation NVDEC) is the first generation that can. The GTX 1650 is older than Turing, so it cannot either. Decoding is always on the CPU.

---

## 1. Runbook: getting the files past the Drive quota

The "Quota exceeded" page (`tail_C3902.bin`, 2,009 bytes) is a per-file limit on downloads of a public file. The robust fix is a **server-side copy into your own Drive**. The copy is a new file with its own quota. Google's Colab FAQ recommends exactly this for popular shared files: "Copy the file using drive.google.com and don't share it widely".

**Storage limit.** A free account has 15 GB shared across Drive, Gmail and Photos, and only files you own count against it. Four copies (≈23.3 GB) do not fit in one account. Either:
- have each teammate copy at most two files, or
- copy, move the file on, delete the copy, empty the trash, and repeat.

| ID | File | Frames / duration |
|---|---|---|
| 1kR9jODA2Wotw4gwkvpRKdqFADNJNc1nS | C3896 | 10,200 / 340.3 s (Drive reports 5.8G) |
| 1hp8DYeqtYHSwfM6qAo9FPSRHlpMFrIN_ | C3897 | 9,525 / 317.8 s |
| 10cHEReCWzO3u-Vk1CnNgHAx6egGy5MwJ | C3902 | unknown (~5.3–5.7 min) |
| 1aJ-QsAZVYJtLKHiRvKKeBq1D3GWNobRd | C3905 | unknown |

**Option A (recommended, entirely in the cloud).** Run this in a Colab cell:
```python
from google.colab import auth; auth.authenticate_user()
from googleapiclient.discovery import build; from googleapiclient.http import MediaIoBaseDownload
svc = build('drive', 'v3')
IDS = {'C3896':'1kR9jODA2Wotw4gwkvpRKdqFADNJNc1nS','C3897':'1hp8DYeqtYHSwfM6qAo9FPSRHlpMFrIN_'}  # 2 per free account
fold = svc.files().create(body={'name':'wiut','mimeType':'application/vnd.google-apps.folder'},fields='id').execute()['id']
for n,f in IDS.items():
    src = svc.files().get(fileId=f, fields='size,md5Checksum').execute()          # metadata is not quota-limited
    cp  = svc.files().copy(fileId=f, body={'name':n+'.MP4','parents':[fold]}, fields='id,md5Checksum').execute()
    with open(f'/content/{n}.MP4','wb') as fh:
        d = MediaIoBaseDownload(fh, svc.files().get_media(fileId=cp['id']), chunksize=256*2**20); done=False
        while not done: _, done = d.next_chunk()
    assert src['md5Checksum'] == cp['md5Checksum']   # then: !md5sum /content/{n}.MP4
```
Two notes on this cell:
- I could not open the Colab example notebook to confirm the scopes. If `auth.authenticate_user()` returns 403 "insufficient scopes", use the manual route instead: in the Drive UI choose ⋮ → Make a copy, then `drive.mount()` and `cp`.
- Do **not** read the original file through a shortcut on the mounted Drive. The Colab FAQ says quota errors appear there as `Input/output error`.

**Option B: rclone v1.75.1.**
1. `rclone config create gd drive scope drive`. On a headless VM, run `rclone authorize "drive"` on the laptop instead.
2. `rclone backend copyid gd: <ID> gd:wiut/`. Per the rclone docs, if the destination is a drive backend, it attempts a server-side copy.
3. `rclone md5sum gd:wiut`.
4. `rclone copy gd:wiut /content/samples -P --multi-thread-streams 8`.

Downloading straight from a shortcut still fetches the target file and hits its quota. So always use `copyid` first.

**Option C: gdown 6.4.0 (released 17 Sep 2026).** The locally installed 6.1.0 lacks the needed flag, so run `pip install -U gdown`. Then:
`gdown <ID> -O C3896.MP4 --cookies-from-browser chrome --continue`

The README says Google "lets a signed-in account through" when a file is throttled. It also says Drive drops connections after about 1 hour, so `--continue` is needed. If the output is a 2 KB HTML file, fall back to option A or B.

**Into Kaggle without using the home connection.** From Colab:
1. Put `dataset-metadata.json` next to the files.
2. Run `kaggle datasets create -p /content/ds -r skip`. Datasets are private by default; keep them private, because this is organizer data.

The Kaggle CLI 2.2.2 source (`ResumableUploadContext`) stores upload state under `%TEMP%/.kaggle/uploads`. Re-running the same command resumes an interrupted upload for up to about 6 days.

Kaggle limits:
- 200 GB per dataset and 200 GB of private datasets in total.
- At most 50 top-level files.
- A notebook's `/kaggle/working` holds 20 GB (sessions run 12 h; T4×2 with 4 CPUs and 29 GB RAM). All four originals will not fit as one notebook output. Create the originals dataset directly from Colab with the CLI.

**What to store where:**

| Where | Contents | Size |
|---|---|---|
| Private Kaggle dataset `wiut-orig` | Four originals | ≈23.3 GB |
| Private Kaggle dataset `wiut-proxy` | 1080p and 720p proxies | estimate ≤2.5 GB |
| Laptop | Proxies only, plus at most one original for timing tests | fits in the 11 GB free |

---

## 2. Proxies that keep every frame, and the check script

**Commands (tested).** Do not add `-colorspace`, `-color_primaries` or `-color_trc` by hand. ffmpeg copies the source tags. In my test, forcing bt709 onto an untagged source shifted OpenCV's decoded BGR by a mean of 6.5 (B) and 11.8 (R) levels. That would break red-light colour thresholds.
```
# 1080p proxy for development and annotation: keyframe every 30 frames, no B-frames, timestamps passed through
ffmpeg -i C3896.MP4 -map 0:v:0 -an -sn -dn -vf "scale=1920:-2:flags=bicubic,format=yuv420p" -c:v libx264 -preset medium -crf 18 -g 30 -keyint_min 30 -sc_threshold 0 -bf 0 -fps_mode passthrough -video_track_timescale 30000 -movflags +faststart C3896_1080p.mp4
# 720p web proxy (plays everywhere: High@4.0 4:2:0 8-bit, capped at 2.5 Mbps)
ffmpeg -i C3896.MP4 -map 0:v:0 -an -sn -dn -vf "scale=1280:-2:flags=bicubic,format=yuv420p" -c:v libx264 -preset slow -crf 23 -maxrate 2.5M -bufsize 5M -profile:v high -level:v 4.0 -g 30 -keyint_min 30 -sc_threshold 0 -bf 2 -fps_mode passthrough -video_track_timescale 30000 -movflags +faststart C3896_720p.mp4
# EDA thumbnails, decoded from keyframes only; thumbnail k (1-based) is frame 30*(k-1)
ffmpeg -skip_frame nokey -i C3896.MP4 -map 0:v:0 -vf scale=320:-2 -fps_mode passthrough -q:v 4 th/C3896_%05d.jpg
```
**Speed.** The 1080p proxy took 18.3 s for a 12 s 4K clip on 12 threads, about 1.5× realtime. That is about 8.5 min per sample on the laptop.

**Check script:** `scratchpad\dlx\check_proxy.py`
Usage: `python check_proxy.py ORIG.MP4 PROXY.mp4 [--seeks 20] [--json r.json]`. It exits with 0 only if every check passes. It checks:
- Packet count from ffprobe (no decode needed).
- `r_frame_rate` and `avg_frame_rate` both exactly 30000/1001, and constant frame rate in the proxy.
- `CAP_PROP_FRAME_COUNT` and `CAP_PROP_FPS` equal between the two files. The harness uses `t = idx/CAP_PROP_FPS`, so these must match.
- Frame counts after a full decode of both files.
- Per-frame alignment, using 64×36 grayscale thumbnails:
  - the best overall lag in ±15 frames is 0;
  - every frame with motion is closer to the same index than to either neighbour;
  - the mean difference is at most 3 grey levels.
- Mean colour difference per B/G/R channel at most 3 levels.
- Random seeks with `CAP_PROP_POS_FRAMES`.

**Test results:**
- Test originals: a synthetic 4K yuv422p10le clip at 137 Mbps with an IBBP GOP of 30 (`origB.mp4`), and a 450-frame 4K 4:2:2 10-bit clip with a barcode of the frame index burned in (`bc_orig.mp4`).
- The good 1080p and 720p proxies PASS: lag 0, 0 misaligned frames, 0 wrong seeks, grey difference ≤1.3.
- The barcode decoded to its own index in 450/450 frames in the original and both proxies.
- Four bad proxies all FAIL as expected:
  - `-r 25` fails with 377 frames against 450;
  - `-ss 0.2` fails with a lag of 6 frames;
  - forcing bt709 tags fails the colour check (mean up to 11.8 levels).

**Seek accuracy in the browser (Chrome and Edge 153, barcode clips):**
- Setting `currentTime = i/fps` exactly shows frame **i−1 in 16 of 20 seeks**. Seeking to `(i+0.5)/fps`, or to `i/fps + 1 ms`, is 20/20 correct.
- Stepping forward by +1 frame is 20/20 correct.
- During playback, `requestVideoFrameCallback` `mediaTime·fps` rounded matched the displayed frame 90/90.
- Median seek time: 9–16 ms on the 1080p proxy, 10–16 ms on 720p, 0.5–1.3 s on the 4K 10-bit original.

**Annotation tools:**
- **Label Studio.** The Video tag's `frameRate` **defaults to 24**. Set `frameRate="$fps"` with fps = 29.97002997. Frames are 1-based: the source code's `goToFrame` seeks to (f−1)/fps plus 2 ms. A TimelineLabels range `{start:s, end:e}` converts to `[(s−1)/fps, e/fps]`.
- **VIA 3.** It stores times in seconds rounded to ms (`toFixed(3)`) and does not use fps; its edge-step key moves 1/50 s. On export, snap to the frame grid: `round(t·fps)/fps`.
- **mpv.** Plays the originals natively. `.` and `,` step one frame; use hr-seek. I did not test mpv here because it is not installed.

A one-frame error (33 ms) is harmless for tIoU. The real danger is a systematic error:
- Assuming 25 fps stretches times by 1.2×.
- Assuming exactly 30 fps drifts 0.34 s by the 340 s mark. For a 3 s event that alone lowers IoU to 0.80.

---

## 3. Browser support table

The Chrome and Edge 153 columns come from tests on this laptop (headless new mode; i5-12450H with Intel iGPU and GTX 1650; HEVC Video Extension installed). The other columns come from the cited sources. The test page is `dlx\bt\index.html`. Host it on the website to fill in the cells marked unknown, from a phone and a Mac.

| Codec | Chrome (Win) | Edge (Win) | Firefox | Safari (macOS) | iOS Safari | Android Chrome |
|---|---|---|---|---|---|---|
| H.264 High 4:2:0 8-bit | Yes | Yes | Yes, via OS codec (MDN) | Yes | Yes | Yes |
| H.264 High 10 | **Yes, software**¹ | **Yes, software**¹ | Windows: **No**, the WMF decoder is limited to Baseline/Main/High 4:2:0 or mono (MS Learn); Linux: uses the system FFmpeg (not verified) | Unknown (probably no) | Unknown (probably no) | Depends on the device (probably no) |
| H.264 High 4:2:2 10-bit (the Sony files) | **Yes, software**; 4K drops ~16% of frames and seeks take 0.5–1.3 s | **Yes**; seeks take up to 3.2 s | Windows: **No** (WMF 4:2:0 only) | Unknown | Unknown | Probably no |
| HEVC Main / Main10 | Yes with a hardware decoder (tested; MDN: Chrome 107+) | Yes (tested; needs HEVC extensions) | Windows 134+, macOS 136+, where supported (MDN) | Yes (11+) | Yes | Depends on hardware |
| HEVC 4:2:2 10-bit (RExt) | **No** here: `canPlayType` says "probably", then decoder initialisation fails | **No** (`canPlayType` returns "") | Unknown | Unknown | Unknown | Unknown |

¹ `canPlayType` returns "probably" for `avc1.6E0033` and "maybe" for `avc1.7A0033`. MediaCapabilities reports `smooth:true` even though 4K 4:2:2 in fact drops frames, so **do not trust capability APIs**. Instead, create a `<video>` from `URL.createObjectURL(file)` and wait up to 5 s for `loadeddata`, treating `error` or a timeout as a failure.

**Rule:** never depend on the browser to play the visitor's original. Always play the server's 720p H.264 preview. A local instant preview is an optional extra.

---

## 4. Demo upload design: limits and backend timings

**Measured on the 12.012 s 4K 4:2:2 10-bit 137 Mbps clip.** Background load from other jobs was 20–90%, so I took the best of several runs. The mask 0x3 means one hyperthreaded P-core; 0xFF means four P-cores.

| Step | 2 logical CPUs | 8 logical CPUs | Per 120 s of video, 8 CPUs |
|---|---|---|---|
| ffmpeg decode only | 24.9 s (2.1× realtime) | 9.5 s (0.8×) | ~95 s |
| decode + 720p x264 veryfast preview | 31.5 s | 12.1 s | ~121 s |
| decode + 960×540 BGR frames | 30.3 s | 14.5 s | ~145 s |
| `-t 120 -c copy` trim | 1.0 s | 1.4 s | ~12–17 s (disk-bound) |
| cv2 `read()` (the harness path) | 9.4 fps | 15.4 fps (21.8 fps on 12 threads at low load) | — |

**Estimate for a 2-minute original.** Using one decode pass split into analysis frames and the 720p preview, a stride-3 nano/small detector running alongside, and cloud vCPUs about 1.2–1.5× slower than these laptop P-cores:
- **8 vCPU: about 2.5–4 min.**
- **2 vCPU: about 7–10 min**, too slow for a demo.
- A 1080p H.264 upload of the same length: under 1 min.

**Hosting.** Hugging Face now requires a paid plan to create Gradio or Docker Spaces; a free account can host up to two ZeroGPU Spaces. The options:
- **CPU Upgrade: 8 vCPU / 32 GB / 50 GB disk, $0.03 per hour.** Pick this.
- **T4-small: 4 vCPU, $0.40 per hour.** Not worth it. NVDEC cannot decode 4:2:2 H.264, so the GPU does not help with decoding, which dominates.
- **ZeroGPU:** unauthenticated visitors get 2 min of GPU per day, it is Gradio-only, and decoding is still on the CPU.

**Upload path:**
1. **Cloudflare R2 direct from the browser (primary).**
   - The backend signs a multipart upload of 64 MiB parts, which the browser PUTs 4 at a time with 3 retries each. The upload ID is kept in `localStorage` so an interrupted upload can resume.
   - R2 limits: single PUT ≤5 GiB, ≤10,000 parts. Presigned URLs work for GET/HEAD/PUT/DELETE, not form POST. They last at most 7 days and only work on the S3 endpoint, not a custom domain.
   - The bucket needs CORS with the site as origin, PUT/GET methods, and **`ExposeHeaders: ETag`** (needed to complete the multipart upload).
   - Never route the file through a Worker: Worker request bodies are limited to 100 MB on the Free and Pro plans.
   - Storage: 10 GB free, egress free. Add a lifecycle rule to delete uploads after 1 day.
2. **Gradio 6.28 on the Space (fallback).** Use `launch(max_file_size="2.5gb")`. From the source code: the upload is one multipart POST, streamed to a temp file on disk; above the cap it returns HTTP 413; it cannot resume.
3. **Modal.** Its request body limit is 4 GiB, which fits, but its 150 s HTTP timeout (with 303 redirects) makes long uploads risky. Treat it as secondary.

**Backend job:**
1. `ffprobe` validation: the file has a video stream that ffmpeg can decode, lasts 5 s or more, and is between 640×360 and 4096×2304.
2. If it is longer than 120 s: `ffmpeg -t 120 -i in -map 0:v:0 -c copy clip.mp4`.
3. One decode pass, split into analysis frames and the 720p H.264 preview.
4. Outputs `events.json`, `risk.json`, `preview.mp4` and thumbnails.
5. The page polls `/jobs/{id}` every 2 s and shows the stage, % of frames done and an ETA.
6. One job runs at a time, with at most 3 queued. Each job peaks at about 4.7 GB of disk (upload 2.5 + trimmed copy 2.1 + outputs), and job folders are deleted after 1 h.

Also ship 3 preloaded sample clips with cached results. The judges can then see the demo instantly even if their upload connection is slow.

**Limits and why:**
- **≤2.5 GB and first 120 s analysed.** 120 s × 141.5 Mbps is 2.12 GB, so 2.5 GB leaves about 18% headroom.
- Check `File.size` in the browser before uploading. A full 6 GB original is then refused instantly instead of after a 16-minute upload.
- Upload times for 2.1 GB: about 14, 5.6 and 2.8 min at 20, 50 and 100 Mbps upstream. That is why the page offers a trim command.

**Message shown to users:**
> Upload a clip from this camera (.mp4/.mov; H.264 or HEVC, including Sony 4K 4:2:2 10-bit). Maximum 2.5 GB; only the first 2 minutes are analysed. 4K originals are about 1 GB per minute, so for larger files cut the first 2 minutes without re-encoding: `ffmpeg -i input.MP4 -t 120 -map 0:v:0 -c copy clip.mp4`. Processing takes about 3–4 min for a 2-minute 4K clip and under a minute for 1080p; progress is shown below.

---

## 5. Storage and media budget for the website

Total sample length is about 1,318 s.

| Asset | Size |
|---|---|
| Four 720p proxies at ≤2.5 Mbps | **≤412 MB** (about 247 MB at 1.5 Mbps) |
| Pre-rendered annotated copies (optional) | +≤500 MB. Better: draw boxes on a canvas over the clean proxy from JSON, synced with rVFC `mediaTime` (tested frame-exact). |
| 40 event clips × 8 s | ~100 MB |
| Scrubbing sprites (1/s, 160×90) | ~5 MB |
| Tracks at 10 Hz plus risk curves (gzip) | ~5–15 MB |
| EDA images | ~10 MB |
| **Total** | **~0.55 GB** (~1.05 GB if annotated renders are also hosted) |

**Where to host.** Put the media in R2 (range requests for seeking, free egress) or on a Hugging Face dataset repository. Keep the site itself on Vercel, Cloudflare Pages or GitHub Pages. The static hosts have limits that rule them out for the video:
- GitHub Pages: the site may be at most 1 GB, with a soft limit of 100 GB bandwidth per month.
- GitHub rejects files over 100 MiB, and a 340 s 720p file at 2.5 Mbps is about 101 MiB.
- Cloudflare Pages: at most 25 MiB per file.

A viewer who watches all four proxies uses about 0.4 GB.

---

## 6. Findings that affect other workstreams

1. **Frame rate is 30000/1001, not 25.** Always use `CAP_PROP_FPS`. Hard-coding 25 stretches every time by 1.2×.
2. **Part B decoding is expensive at 4K.** The harness's `cap.read()` on these 4K 10-bit 4:2:2 files runs at only 15–22 fps; `grab()` runs at 57 fps. Converting to BGR costs about 28 ms per frame. So the Part B loop alone takes about **1.4–2× the video duration** on 8–12 threads. That leaves only about 1–1.5× for Part A within the 3× budget. In Part A, call `grab()` for every frame and `retrieve()` only every k-th frame, and never decode twice.
3. **No GPU decoding anywhere.** Neither the T4 nor the GTX 1650 can decode these files.
4. **Colour must match between tuning and submission.** Tune colour thresholds (for example the red light) on frames decoded exactly as the harness decodes them. `check_proxy.py` runs a colour check for this.
5. **Store scene polygons in normalised 0–1 coordinates.** The demo must handle both 4K and 1080p uploads.

Test fixtures and scripts are in `scratchpad\dlx\`: `check_proxy.py`, `origB.mp4`, `bc_orig.mp4`, `bench_demo.ps1`, `bench3.ps1`, and `bt\index.html` / `bt\seek2.html`.



## Key claims (as submitted for verification)
- The sample videos are Sony ILCE-6700 XAVC 'AVC140_3840_2160_H422P@L51' (H.264 High 4:2:2 L5.1, 140 Mbps) at 29.97p with LPCM16 stereo, keyframes every 30 frames; C3896 = 10,200 frames (340.3 s), C3897 = 9,525 frames — local evidence: Sony NonRealTimeMeta XML and stss/stts boxes in scratchpad tail_C3896.bin / tail_C3897.bin
- Chrome 153 and Edge 153 on Windows decode H.264 High 10 and High 4:2:2 10-bit in software (played and pixel-verified); 4K 137 Mbps 4:2:2 playback dropped 40/244 frames in 8 s and seeks took 0.5-1.3 s (Chrome) / up to 3.2 s (Edge) — reasoning: local headless tests, scratchpad\dlx\bt\index.html and seek2.html
- In Chrome/Edge 153, setting currentTime = i/fps exactly displays frame i-1 in 16/20 seeks; (i+0.5)/fps or i/fps+1 ms is 20/20 correct — reasoning: local barcode test scratchpad\dlx\bt\seek2.html
- The Media Foundation H.264 decoder supports only Baseline/Main/High up to level 5.1 with 4:2:0 or monochrome chroma — https://learn.microsoft.com/en-us/windows/win32/medfound/h-264-video-decoder
- Firefox AVC support depends on the operating system's codecs; HEVC is supported in Chrome 107+ with hardware and Firefox 134+ on Windows — https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/Video_codecs
- NVIDIA Tesla T4 (Turing) NVDEC does not support H.264 4:2:2 decode; Blackwell (6th gen NVDEC) is the first generation that does — https://developer.nvidia.com/video-encode-decode-support-matrix
- rclone 'backend copyid' copies files by ID and attempts a server-side copy when the destination is a drive backend — https://rclone.org/drive/
- Colab FAQ: quota errors from popular shared files appear as Input/output error; recommended workaround is to copy the file using drive.google.com — https://research.google.com/colaboratory/faq.html
- A free Google account has 15 GB shared across Drive/Gmail/Photos and only files you own count toward it — https://support.google.com/drive/answer/6374270?hl=en
- Kaggle datasets: 200 GB per dataset, 200 GB private quota, max 50 top-level files; notebooks save 20 GB in /kaggle/working and run up to 12 h — https://r.jina.ai/https://www.kaggle.com/docs/datasets and https://r.jina.ai/https://www.kaggle.com/docs/notebooks
- Kaggle CLI 2.2.2 performs resumable uploads, keeping state in the temp dir with a 6-day expiry — local source: site-packages/kaggle/api/kaggle_api_extended.py (ResumableUploadContext, RESUMABLE_UPLOAD_EXPIRY_SECONDS)
- HF Spaces: CPU Basic 2 vCPU/16 GB/50 GB free; CPU Upgrade 8 vCPU/32 GB/50 GB at $0.03/h; T4 small 4 vCPU at $0.40/h; creating Gradio/Docker Spaces now requires a paid plan — https://huggingface.co/docs/hub/spaces-gpus and https://huggingface.co/docs/hub/spaces-overview
- ZeroGPU: unauthenticated visitors get 2 min of GPU per day, Gradio SDK only; free accounts can host 2 ZeroGPU Spaces — https://huggingface.co/docs/hub/spaces-zerogpu
- Gradio 6.28.0 max_file_size (launch param) is enforced server-side with HTTP 413; uploads are a single streamed multipart POST to a temp file; default is no limit — local wheel gradio-6.28.0 routes.py/route_utils.py; https://www.gradio.app/guides/file-access
- Modal web function request bodies can be up to 4 GiB; HTTP timeout is 150 s with 303 redirects — https://modal.com/docs/guide/webhooks and https://modal.com/docs/guide/webhook-timeouts
- Cloudflare R2: single-part upload up to 5 GiB, max 10,000 parts; presigned URLs support GET/HEAD/PUT/DELETE (not POST), expire in at most 7 days, work only on the S3 endpoint; free 10 GB-month and free egress; Workers request body limit 100 MB on Free/Pro — https://developers.cloudflare.com/r2/platform/limits/ ; https://developers.cloudflare.com/r2/api/s3/presigned-urls/ ; https://developers.cloudflare.com/r2/pricing/ ; https://developers.cloudflare.com/workers/platform/limits/
- GitHub Pages sites are limited to 1 GB with a 100 GB/month soft bandwidth limit; GitHub blocks files over 100 MiB; Cloudflare Pages limits each asset to 25 MiB — https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits ; https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github ; https://developers.cloudflare.com/pages/platform/limits/
- Label Studio Video tag frameRate defaults to 24; frames are 1-based and goToFrame seeks to (f-1)/fps plus 2 ms — https://labelstud.io/tags/video ; https://raw.githubusercontent.com/HumanSignal/label-studio/develop/web/libs/editor/src/components/VideoCanvas/VideoCanvas.tsx
- Decoding the 4K 4:2:2 10-bit test clip: ffmpeg decode-only 2.1x realtime on 2 logical CPUs and 0.8x on 8; cv2 read() 9.4 fps (2 CPUs) / 15.4-21.8 fps (8-12 threads), grab() 57 fps — reasoning: local benchmarks scratchpad\dlx\bench_demo.ps1, bench3.ps1, cv2read.py (background load 20-90%)
- Forcing bt709 colour tags on a proxy of an untagged source shifts OpenCV-decoded BGR by 6.5 (B) / 11.8 (R) levels on average; letting ffmpeg propagate tags gives <1 level — reasoning: local test with scratchpad\dlx\check_proxy.py on proxy1080.mp4 vs proxy1080b.mp4
