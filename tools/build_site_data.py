"""Data for the static site (site/): EDA and results for every sample video.

    python tools/build_site_data.py --cache cache --pred predictions_samples.json \
        [--gt dev_labels/dev_gt.json] [--videos DataSets] [--video-urls site/videos.json]

Writes site/data/site.json and per-video images (motion heatmap, direction field).
Inputs come from the Kaggle/T4 run (kaggle_out/cache, kaggle_out/predictions_samples.json).
Background images are the median frames in geometry/ref_<video>.png when present.
`--video-urls` maps a video name to the URL of its annotated review video (hosted
outside GitHub, e.g. a Hugging Face dataset), so the page can play it and seek on click.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trafficwatch import pipeline, viz  # noqa: E402
from trafficwatch.detector import PERSON, SUPER_NAMES, TWO_WHEELER, VEHICLE  # noqa: E402
from trafficwatch.perception import Perception  # noqa: E402
from trafficwatch.video import ffmpeg_exe, read_meta  # noqa: E402

OUT = ROOT / "site" / "data"
IMG_W = 960
THUMB_W = 640
MAX_THUMBS = 60  # per video


def _background(video_id: str, size: tuple[int, int]) -> np.ndarray:
    ref = ROOT / "geometry" / f"ref_{Path(video_id).stem}.png"
    img = cv2.imread(str(ref)) if ref.exists() else None
    if img is None:
        return np.full((size[1], size[0], 3), 40, np.uint8)
    return (cv2.resize(img, size, interpolation=cv2.INTER_AREA) * 0.6).astype(np.uint8)


def heatmap_image(res, size) -> np.ndarray:
    """Where road users' feet were, vehicles and people together, log-scaled."""
    w, h = size
    per = res.perception
    acc = np.zeros((h, w), np.float32)
    for tr in res.tracks:
        pts = (tr.foot * [w / per.frame_w, h / per.frame_h]).astype(int)
        ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        np.add.at(acc, (pts[ok, 1], pts[ok, 0]), 1.0)
    acc = cv2.GaussianBlur(acc, (0, 0), 6)
    norm = np.log1p(acc) / max(np.log1p(acc).max(), 1e-6)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = np.clip(norm * 1.5, 0, 0.85)[..., None]
    return (_background(per.video_id, size) * (1 - alpha) + color * alpha).astype(np.uint8)


def flow_image(res, size, cells_across: int = 32) -> np.ndarray:
    """Learned direction of travel, pooled to about `cells_across` arrows per row; colour = direction."""
    w, h = size
    img = _background(res.perception.video_id, size)
    maps = res.maps
    k = max(1, maps.flow_grid[0] // cells_across)
    gh, gw = maps.flow_count.shape[0] // k, maps.flow_count.shape[1] // k
    n = maps.flow_count[:gh * k, :gw * k].reshape(gh, k, gw, k).sum(axis=(1, 3))
    s = maps.flow_sum[:gh * k, :gw * k].reshape(gh, k, gw, k, 2).sum(axis=(1, 3))
    cw, ch = w / gw, h / gh
    for iy in range(gh):
        for ix in range(gw):
            norm = np.linalg.norm(s[iy, ix])
            if n[iy, ix] < 10 or norm / n[iy, ix] < 0.6:  # few samples or no dominant direction
                continue
            d = s[iy, ix] / norm
            c = np.array([(ix + 0.5) * cw, (iy + 0.5) * ch])
            hue = int((np.degrees(np.arctan2(d[1], d[0])) % 360) / 2)
            bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            half = d * 0.4 * min(cw, ch)
            cv2.arrowedLine(img, tuple((c - half).astype(int)), tuple((c + half).astype(int)), bgr, 2,
                            cv2.LINE_AA, tipLength=0.35)
    return img


def stream_info(video: Path) -> dict:
    """Codec, pixel format and bitrate of the first video stream, as ffmpeg reports them."""
    err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(video)], capture_output=True, text=True).stderr
    m = re.search(r"Video: (\w+)(?: \(([^)]*)\))?.*?, (\w+)(?:\([^)]*\))?, \d+x\d+", err)
    kbps = re.search(r"Duration: .*bitrate: (\d+) kb/s", err)
    return {"codec": m.group(1) if m else None, "profile": m.group(2) if m else None,
            "pix_fmt": m.group(3) if m else None, "mbps": round(int(kbps.group(1)) / 1000, 1) if kbps else None,
            "size_gb": round(video.stat().st_size / 1e9, 2)}


def grab_frame(video: Path, t: float, width: int) -> np.ndarray | None:
    """One frame near t, scaled to `width` (fast seek, then decode)."""
    meta = read_meta(video)
    height = int(round(meta.height * width / meta.width / 2)) * 2
    out = subprocess.run([ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                          "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                         capture_output=True).stdout
    if len(out) != width * height * 3:
        return None
    return np.frombuffer(out, np.uint8).reshape(height, width, 3)


def event_thumbnails(video: Path, res, events: list[list], stem: str) -> list[str | None]:
    """A frame from the middle of each event with the tracks drawn, for the site's gallery."""
    per = res.perception
    by_time = viz.tracks_by_time(res.tracks)
    paths: list[str | None] = []
    for k, (s, e, _) in enumerate(events):
        if k >= MAX_THUMBS:
            paths.append(None)
            continue
        t = float(per.times[np.argmin(np.abs(per.times - (s + e) / 2))])
        frame = grab_frame(video, t, per.frame_w)
        if frame is None:
            paths.append(None)
            continue
        img = viz.draw_frame(frame, by_time.get(round(t, 3), []), events, t)
        img = cv2.resize(img, (THUMB_W, int(round(img.shape[0] * THUMB_W / img.shape[1]))),
                         interpolation=cv2.INTER_AREA)
        name = f"{stem}_ev{k:03d}.jpg"
        cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        paths.append(f"data/{name}")
    return paths


def counts_over_time(res, bin_s: float = 10.0) -> dict:
    duration = res.perception.duration
    n_bins = int(np.ceil(duration / bin_s))
    out = {"t": [round(i * bin_s, 1) for i in range(n_bins)]}
    for sc in (VEHICLE, TWO_WHEELER, PERSON):
        ids = [set() for _ in range(n_bins)]
        for tr in res.tracks:
            if tr.sc == sc:
                for b in np.unique((tr.t / bin_s).astype(int)):
                    if b < n_bins:
                        ids[b].add(tr.tid)
        out[SUPER_NAMES[sc]] = [len(x) for x in ids]
    return out


def speed_histogram(res) -> dict:
    speeds = np.concatenate([tr.speed for tr in res.tracks if tr.sc == VEHICLE] or [np.zeros(0)])
    edges = np.arange(0, 6.01, 0.25)
    hist, _ = np.histogram(np.clip(speeds, 0, 5.99), bins=edges)
    return {"edges": edges[:-1].round(2).tolist(), "counts": hist.tolist()}


def downsample_risk(curve: list, step_s: float = 0.5) -> list:
    if not curve:
        return []
    arr = np.asarray(curve, float)
    bins = (arr[:, 0] / step_s).astype(int)
    out = []
    for b in np.unique(bins):
        sel = arr[bins == b]
        out.append([round(b * step_s, 2), round(float(sel[:, 1].max()), 4)])
    return out


def evaluate_report(pred: Path, gt: Path) -> dict | None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        subprocess.run([sys.executable, str(ROOT / "evaluate.py"), "--pred", str(pred), "--gt", str(gt),
                        "--json", str(out)], check=False, capture_output=True)
        return json.loads(out.read_text()) if out.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--pred", default="predictions_samples.json")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--videos", default=None, help="folder with the original videos (for resolution)")
    ap.add_argument("--video-urls", default=None, help="json {video: url of the annotated video}")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pred = json.loads(Path(args.pred).read_text()) if Path(args.pred).exists() else {"videos": {}, "log": {}}
    gt = json.loads(Path(args.gt).read_text()) if args.gt else {}
    urls = json.loads(Path(args.video_urls).read_text()) if args.video_urls else {}
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    site = {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "commit": commit,
            "classes": viz.EVENT_COLORS, "videos": [],
            "config": {k: pipeline.CFG[k] for k in ("decode", "detector", "classes", "risk", "budget")}}
    for npz in sorted(Path(args.cache).glob("*.npz")):
        per = Perception.load(npz)
        res = pipeline.analyze(per)
        vid = per.video_id
        size = (IMG_W, int(round(IMG_W * per.frame_h / per.frame_w)))
        stem = Path(vid).stem
        cv2.imwrite(str(OUT / f"{stem}_heat.jpg"), heatmap_image(res, size), [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(OUT / f"{stem}_flow.jpg"), flow_image(res, size), [cv2.IMWRITE_JPEG_QUALITY, 85])
        entry = pred["videos"].get(vid, {})
        events = entry.get("events", res.events)
        source = Path(args.videos) / vid if args.videos else None
        original = read_meta(source) if source and source.exists() else None
        thumbs = event_thumbnails(source, res, events, stem) if original else [None] * len(events)
        step = max(1, int(round(1.0 / per.dt)))
        brightness = None if per.brightness is None else {
            "t": per.times[::step].round(1).tolist(), "v": per.brightness[::step].round(1).tolist()}
        site["videos"].append({
            "id": vid,
            "duration": round(per.duration, 3),
            "fps": round(per.fps, 3),
            "resolution": [original.width, original.height] if original else None,
            "stream": stream_info(source) if original else None,
            "analysed_fps": round(per.fps / per.stride, 2),
            "n_tracks": {SUPER_NAMES[sc]: sum(tr.sc == sc for tr in res.tracks)
                         for sc in (VEHICLE, TWO_WHEELER, PERSON)},
            "events": events,
            "thumbs": thumbs,
            "gt": gt.get(vid, {}).get("events", []),
            "risk": downsample_risk(entry.get("risk", [])),
            "timing": pred.get("log", {}).get(vid),
            "counts": counts_over_time(res),
            "brightness": brightness,
            "speed_hist": speed_histogram(res),
            "heatmap": f"data/{stem}_heat.jpg",
            "flow": f"data/{stem}_flow.jpg",
            "video_url": urls.get(vid),
            "has_crosswalks": bool(res.maps.has_crosswalks),
        })
        print(f"{vid}: {len(site['videos'][-1]['events'])} events")
    if args.gt and Path(args.pred).exists():
        site["metrics"] = evaluate_report(Path(args.pred), Path(args.gt))
    (OUT / "site.json").write_text(json.dumps(site))
    print(f"-> {OUT / 'site.json'}")


if __name__ == "__main__":
    main()
