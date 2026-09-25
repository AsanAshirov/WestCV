"""Label Studio JSON export (TimelineLabels) -> ground_truth.json (organizers' format).

Task data expected when importing tasks into Label Studio:
  {"video": "/data/local-files/?d=proxies/CLIP01_p1080.mp4",
   "orig": "CLIP01.MP4", "fps": 29.97002997, "duration": 340.307}
(fps/duration taken from ffprobe of the ORIGINAL file.)

TimelineLabels ranges are 1-based inclusive frame numbers
(label-studio-ml-backend utils/converter.py uses start-1 .. end).
"""
import json, sys, argparse
from collections import defaultdict

FPS_DEFAULT = 30000 / 1001


def frames_to_sec(start_f, end_f, fps):
    # frame k (1-based) is displayed during [(k-1)/fps, k/fps)
    return round((start_f - 1) / fps, 3), round(end_f / fps, 3)


def convert(export, by_annotator=False, only_annotator=None):
    out = defaultdict(dict)  # annotator -> {video: entry}
    for task in export:
        d = task["data"]
        vid = d.get("orig") or d["video"].split("/")[-1].split("?d=")[-1]
        fps = float(d.get("fps", FPS_DEFAULT))
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            who = str(ann.get("completed_by", "?")) if by_annotator else "all"
            if only_annotator and who != only_annotator:
                continue
            entry = out[who].setdefault(vid, {"duration": float(d.get("duration", 0)),
                                             "fps": round(fps, 2), "events": []})
            for r in ann.get("result", []):
                if r.get("type") != "timelinelabels":
                    continue
                label = r["value"]["timelinelabels"][0]
                for rng in r["value"]["ranges"]:
                    s, e = frames_to_sec(rng["start"], rng["end"], fps)
                    entry["events"].append([s, e, label])
    for who in out:
        for v in out[who].values():
            v["events"].sort()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export_json")
    ap.add_argument("--out", default="ground_truth.json")
    ap.add_argument("--by-annotator", action="store_true",
                    help="write one GT file per annotator (for agreement checks)")
    a = ap.parse_args()
    export = json.load(open(a.export_json, encoding="utf-8"))
    res = convert(export, by_annotator=a.by_annotator)
    for who, gt in res.items():
        path = a.out if who == "all" else a.out.replace(".json", f"_ann{who}.json")
        json.dump(gt, open(path, "w", encoding="utf-8"), indent=1)
        print("wrote", path, sum(len(v["events"]) for v in gt.values()), "events")
