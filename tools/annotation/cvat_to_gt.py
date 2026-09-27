# cvat_to_gt.py — CVAT "CVAT for video 1.1" export -> ground_truth.json (or predictions.json)
#
# Event = CVAT track. Its first visible frame is the start frame, the frame where the
# track is switched "outside" (key O) is the first frame after the event:
#     start_sec = first_visible / fps,   end_sec = (last_visible + 1) / fps
# A track that is never switched outside runs to the end of the video (end = duration).
# Same-class segments that overlap are merged into one (task FAQ: "two events of the
# same class at once -> one segment covering both").
#
# usage:
#   python tools/annotation/cvat_to_gt.py labels/cvat/A/*.xml --out labels/gt_A.json
#   python tools/annotation/cvat_to_gt.py labels/cvat/B/*.xml --out labels/pred_B.json --as-pred --team B
#   options: --videos DataSets (read fps/frame count like the harness and cross-check),
#            --congestion H1|H2, --drop-uncertain
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]
FPS_DEFAULT = 30000 / 1001
PROXY_SUFFIX = re.compile(r"_(\d{3,4}p|proxy)$", re.I)


def load_xml(path: Path) -> ET.Element:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith("annotations.xml"))
            return ET.fromstring(zf.read(name))
    return ET.parse(path).getroot()


def meta_value(root: ET.Element, tag: str) -> str | None:
    """First <tag> anywhere under <meta> (CVAT puts <source> under meta or meta/task)."""
    meta = root.find("meta")
    for el in meta.iter(tag) if meta is not None else ():
        if el.text is not None:
            return el.text.strip()
    return None


def probe_video(videos_dir: Path, stem: str) -> tuple[str, float, int] | None:
    """File name, fps and frame count exactly as run_submission.py reads them."""
    for p in sorted(videos_dir.iterdir()):
        if p.suffix in (".mp4", ".MP4") and p.stem.lower() == stem.lower():
            import cv2
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return p.name, float(fps), n
    return None


def attr_map(shape: ET.Element) -> dict[str, str]:
    return {a.get("name"): (a.text or "").strip() for a in shape.findall("attribute")}


def visible_runs(shapes: list[ET.Element], stop_frame: int) -> list[tuple[int, int, dict]]:
    """[(first_frame, last_frame, attrs)] for every contiguous visible part of a track.
    Works for exports with every frame listed and for keyframes-only exports."""
    shapes = sorted(shapes, key=lambda s: int(s.get("frame")))
    runs, start, attrs = [], None, {}
    for i, sh in enumerate(shapes):
        f, outside = int(sh.get("frame")), sh.get("outside") == "1"
        if not outside and start is None:
            start, attrs = f, attr_map(sh)
        elif outside and start is not None:
            runs.append((start, f - 1, attrs))
            start = None
    if start is not None:
        runs.append((start, stop_frame, attrs))
    return runs


def is_true(v: str | None) -> bool:
    return str(v).lower() in ("true", "1", "yes")


def merge_same_class(events: list[list]) -> list[list]:
    out = []
    for s, e, lab in sorted(events, key=lambda x: (x[2], x[0], x[1])):
        if out and out[-1][2] == lab and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, lab])
    return sorted(out, key=lambda x: (x[0], x[1], x[2]))


def convert(root: ET.Element, src: Path, args) -> tuple[str, dict, list[str]]:
    notes = []
    size = int(meta_value(root, "size") or 0)
    start_frame = int(meta_value(root, "start_frame") or 0)
    stop_frame = int(meta_value(root, "stop_frame") or size - 1)
    frame_filter = meta_value(root, "frame_filter") or ""
    if start_frame != 0 or frame_filter not in ("", "step=1"):
        sys.exit(f"{src}: task must keep every frame from 0 (start_frame={start_frame}, "
                 f"frame_filter={frame_filter!r}); re-create the task with default frame settings")

    task_name = root.findtext("meta/task/name")  # not meta.iter("name"): label names live there too
    source = meta_value(root, "source") or task_name or src.stem
    stem = PROXY_SUFFIX.sub("", Path(args.name or source).stem)
    fps, n_frames, vid = args.fps, stop_frame + 1, stem + args.ext
    if args.videos:
        probed = probe_video(Path(args.videos), stem)
        if probed is None:
            sys.exit(f"{src}: no video {stem}.mp4/.MP4 in {args.videos}")
        vid, fps, n_frames = probed
        if n_frames != stop_frame + 1:
            sys.exit(f"{src}: CVAT task has {stop_frame + 1} frames, original {vid} has {n_frames}; "
                     "the proxy is not frame-exact")
    duration = n_frames / fps

    events = []
    for tr in root.findall("track"):
        label = tr.get("label")
        if label not in CLASSES:
            notes.append(f"skip track {tr.get('id')}: label {label!r} is not an event class")
            continue
        shapes = [el for el in tr if el.get("frame") is not None]
        track_attrs = attr_map(tr)  # immutable attributes may sit on the track itself
        for f0, f1, shape_attrs in visible_runs(shapes, stop_frame):
            attrs = {**track_attrs, **shape_attrs}
            s, e = f0 / fps, min((f1 + 1) / fps, duration)
            tag = f"{vid} {label} [{s:.2f}, {e:.2f}] frames {f0}-{f1}"
            if attrs.get("note"):
                tag += f"  note: {attrs['note']}"
            if is_true(attrs.get("uncertain")):
                notes.append("UNCERTAIN " + tag)
                if args.drop_uncertain:
                    continue
            if label == "congestion" and is_true(attrs.get("h2_only")):
                notes.append("H2-only " + tag)
                if args.congestion == "H1":
                    continue
            events.append([s, e, label])
    if root.findall("image") or root.findall("tag"):
        notes.append(f"{src}: frame shapes/tags are ignored, events must be tracks")

    events = [[round(s, 3), round(e, 3), lab] for s, e, lab in merge_same_class(events)]
    events = [ev for ev in events if ev[1] > ev[0]]
    return vid, {"duration": round(duration, 4), "fps": round(fps, 5), "events": events}, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+", help="annotations.xml or CVAT export .zip, one per task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", help="folder with the original videos (exact fps / frame count)")
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT, help="used when --videos is not given")
    ap.add_argument("--ext", default=".MP4", help="extension of the original file name (GT key)")
    ap.add_argument("--name", help="override the video name (only with a single export)")
    ap.add_argument("--congestion", choices=("H1", "H2"), default="H1",
                    help="H1 drops congestion tracks marked h2_only")
    ap.add_argument("--drop-uncertain", action="store_true")
    ap.add_argument("--as-pred", action="store_true", help="write predictions.json format")
    ap.add_argument("--team", default="annotator")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # notes may be in Russian; Windows console defaults to cp1252
    if args.name and len(args.exports) > 1:
        sys.exit("--name works with a single export only")

    gt = {}
    for path in map(Path, args.exports):
        vid, entry, notes = convert(load_xml(path), path, args)
        if vid in gt:
            sys.exit(f"{path}: video {vid} appears in two exports")
        gt[vid] = entry
        for n in notes:
            print("  " + n)
        counts = {}
        for *_, lab in entry["events"]:
            counts[lab] = counts.get(lab, 0) + 1
        print(f"{vid}: {len(entry['events'])} events {counts}")

    out = {"team": args.team, "videos": {k: {"events": v["events"]} for k, v in gt.items()}} \
        if args.as_pred else gt
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
