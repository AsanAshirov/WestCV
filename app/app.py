"""Live demo: upload a road video -> events, timeline, annotated playback, risk curve.

Runs the same code as the submission (src/trafficwatch) with configs/demo.yaml: a
smaller detector and a lower frame rate, so a 2-minute clip finishes on a CPU.
Everything happens in one decoding pass: Part B and the preview frames are fed from
the same frames as detection.

Locally:            python app/app.py        -> http://127.0.0.1:7860
Hugging Face Space: bash tools/build_space.sh, then push dist/space to the Space repo.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src" / "trafficwatch").is_dir() else HERE.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TW_CONFIG", str(ROOT / "configs" / "demo.yaml"))

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from trafficwatch import env  # noqa: E402,F401

# isort: split
from trafficwatch import pipeline, risk, viz  # noqa: E402
from trafficwatch.perception import perceive  # noqa: E402
from trafficwatch.video import analysis_size, ffmpeg_exe, read_meta  # noqa: E402

MAX_SECONDS = 120
PREVIEW_WIDTH = 854
WORK_PREFIX = "tw_demo_"

pipeline.warmup()
risk.warmup()


def _cleanup_old_workdirs(max_age_s: float = 3600) -> None:
    for d in Path(tempfile.gettempdir()).glob(f"{WORK_PREFIX}*"):
        if time.time() - d.stat().st_mtime > max_age_s:
            shutil.rmtree(d, ignore_errors=True)


def _trim(path: str, workdir: Path) -> tuple[str, bool]:
    """Keep the first MAX_SECONDS without re-encoding."""
    if read_meta(path).duration <= MAX_SECONDS + 0.5:
        return path, False
    out = workdir / "clip.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", path, "-t", str(MAX_SECONDS),
                    "-map", "0:v:0", "-c", "copy", str(out)], check=True)
    return str(out), True


def _render(res, previews: list, out_path: Path) -> None:
    per = res.perception
    first = cv2.imdecode(previews[0][1], cv2.IMREAD_COLOR)
    ph, pw = first.shape[:2]
    scale = pw / per.frame_w
    overlay = viz.scene_overlay(res.maps, (pw, ph))
    by_time = viz.tracks_by_time(res.tracks)
    writer = viz.H264Writer(out_path, (pw, ph + viz.BAR_H), per.fps / per.stride)
    for t, jpg in previews:
        frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        img = viz.draw_frame(frame, by_time.get(round(t, 3), []), res.events, t, scale, overlay)
        writer.write(np.vstack([img, viz.timeline_bar(pw, t, per.duration, res.events)]))
    writer.close()


def _timeline_figure(events: list[list], duration: float) -> go.Figure:
    fig = go.Figure()
    for s, e, label in events:
        fig.add_trace(go.Bar(base=[s], x=[e - s], y=[label], orientation="h", showlegend=False,
                             marker_color=viz.EVENT_COLORS.get(label, "#888"),
                             hovertemplate=f"{label}<br>{s:.1f}–{e:.1f} s<extra></extra>"))
    labels = sorted({e[2] for e in events})
    fig.update_layout(title="Detected events", barmode="overlay", height=120 + 36 * max(len(labels), 1),
                      xaxis=dict(title="time, s", range=[0, duration]), margin=dict(l=10, r=10, t=40, b=40))
    if not events:
        fig.add_annotation(text="no events detected", x=duration / 2, y=0, showarrow=False)
    return fig


def _risk_figure(curve: list[tuple[float, float]], enabled: bool) -> go.Figure:
    fig = go.Figure()
    if curve:
        t, s = zip(*curve)
        fig.add_trace(go.Scatter(x=t, y=s, mode="lines", name="risk", line=dict(color="#d62728")))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#555", annotation_text="alarm threshold")
    title = "Accident risk (causal, next 5 s)" + ("" if enabled else " — Part B disabled")
    fig.update_layout(title=title, height=260, yaxis=dict(range=[0, 1], title="P(accident within 5 s)"),
                      xaxis=dict(title="time, s"), margin=dict(l=10, r=10, t=40, b=40), showlegend=False)
    return fig


def analyze(file: str | None, progress=gr.Progress()):
    if not file:
        raise gr.Error("Please upload a video file.")
    _cleanup_old_workdirs()
    workdir = Path(tempfile.mkdtemp(prefix=WORK_PREFIX))
    try:
        progress(0.0, desc="Checking the file")
        try:
            path, trimmed = _trim(file, workdir)
            meta = read_meta(path)
        except Exception:
            raise gr.Error("This file could not be read as a video. Upload an .mp4 or .mov (H.264 or HEVC).") from None
        if meta.duration < 2 or meta.n_frames < 10:
            raise gr.Error("The video is shorter than 2 seconds.")

        cfg = pipeline.CFG
        stride = max(1, int(round(meta.fps / cfg["decode"]["analysis_fps"])))
        aw, ah = analysis_size(meta, cfg["decode"]["width"])
        pw = min(PREVIEW_WIDTH, aw)
        ph = int(round(ah * pw / aw / 2)) * 2
        estimator = risk.RiskEstimator()
        estimator.reset({"video_id": meta.video_id, "fps": meta.fps / stride, "width": aw, "height": ah,
                         "n_frames": math.ceil(meta.n_frames / stride)})
        previews, curve = [], []

        def on_frame(t: float, frame: np.ndarray) -> None:
            curve.append((t, estimator.step(frame, t)))
            small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            previews.append((t, cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]))

        t0 = time.perf_counter()
        perception = perceive(meta, pipeline.get_detector(), cfg, on_frame=on_frame,
                              on_progress=lambda f: progress(0.05 + 0.75 * f, desc="Detecting and tracking"))
        progress(0.82, desc="Applying the event rules")
        res = pipeline.analyze(perception)
        progress(0.86, desc="Rendering the annotated video")
        video_path = workdir / "annotated.mp4"
        _render(res, previews, video_path)
        took = time.perf_counter() - t0

        events_path = workdir / "events.json"
        events_path.write_text(json.dumps({meta.video_id: {"duration": round(meta.duration, 3),
                                                           "events": res.events}}, indent=1))
        table = [[s, e, label, round(e - s, 2)] for s, e, label in res.events]
        counts = {}
        for _, _, label in res.events:
            counts[label] = counts.get(label, 0) + 1
        summary = (
            f"**{len(res.events)} events** in {meta.duration:.0f} s of video "
            f"({meta.width}×{meta.height}, {meta.fps:.2f} fps): "
            + (", ".join(f"{k} × {v}" for k, v in sorted(counts.items())) or "none")
            + f". {len(res.tracks)} tracked road users. Processed in {took:.0f} s on "
            + ("GPU" if pipeline.get_detector().device.startswith("cuda") else "CPU") + "."
            + (f" Only the first {MAX_SECONDS} s were analysed." if trimmed else "")
        )
        return (str(video_path), _timeline_figure(res.events, meta.duration),
                _risk_figure(curve, estimator.enabled), table, str(events_path), summary)
    except gr.Error:
        raise
    except Exception as exc:  # never show a stack trace to a visitor
        pipeline.log(f"demo failed: {exc!r}")
        raise gr.Error(f"Processing failed: {type(exc).__name__}. Try a shorter or re-encoded clip.") from None


INTRO = f"""
# Traffic event detection — live demo (team Antigradient)
Upload a video from a fixed road camera. The model returns every traffic event it finds as a time
segment with a class, an annotated playback, and a causal accident-risk curve.

- **Accepted:** `.mp4` / `.mov`, H.264 or HEVC (4K 4:2:2 10-bit works), up to 2.5 GB.
  **The first {MAX_SECONDS} s are analysed.**
- **Runs on CPU:** expect roughly 1–2× the clip length. Progress is shown while it runs.
- Same code as our submission, with a smaller detector and 5 frames per second.
  Classes: stopped vehicle, jaywalking (needs our crossing map, i.e. the competition camera),
  wrong-way driving, congestion, accident.
- Large file? Cut 2 minutes first: `ffmpeg -i input.MP4 -t 120 -map 0:v:0 -c copy clip.mp4`
"""

with gr.Blocks(title="Antigradient — traffic events demo", delete_cache=(3600, 3600)) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        upload = gr.File(label="Video", file_types=[".mp4", ".mov", ".MP4", ".MOV"], type="filepath")
        run = gr.Button("Analyse", variant="primary", scale=0)
    summary = gr.Markdown()
    video = gr.Video(label="Annotated playback (boxes: track id and speed; bottom bar: event timeline)")
    timeline = gr.Plot(label="Timeline")
    risk_plot = gr.Plot(label="Risk")
    table = gr.Dataframe(headers=["start, s", "end, s", "class", "duration, s"], label="Events")
    events_file = gr.File(label="events.json (competition format)")
    run.click(analyze, inputs=upload, outputs=[video, timeline, risk_plot, table, events_file, summary],
              api_name="analyze")

if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch(max_file_size="2560mb")
