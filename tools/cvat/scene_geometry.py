# scene_geometry.py — scene geometry <-> CVAT "WestCV scene" task (frame C3896_median).
#
#   python tools/cvat/scene_geometry.py push   # scene/geometry.json + signal heads -> CVAT (replaces the frame's shapes)
#   python tools/cvat/scene_geometry.py pull   # CVAT -> scene/geometry.json (+ scene/signal_heads.json boxes)
#
# Label mapping (CVAT -> geometry.json):
#   crosswalk polygon (attr id)      -> crosswalks[id]
#   island polygon                   -> islands[n]      (the long median divider is an island too)
#   sidewalk polygon                 -> sidewalks[n]
#   stop_line polyline (attr approach)-> stop_lines[approach]   (first and last point)
#   solid_line polyline              -> solid_lines[n]  (first and last point)
#   traffic_light rectangle (attr controls = veh_median / ped_left) -> signal_heads.json boxes
# The median is kept under "median" on push and read back as an island named "median".
import argparse
import getpass
import json
import os
import sys

from setup_tasks import REPO, Cvat

FRAME_NAME = "ref/C3896_median.png"
GEO = REPO / "scene/geometry.json"
HEADS = REPO / "scene/signal_heads.json"


def flat(pts):
    return [float(v) for p in pts for v in p]


def pairs(points):
    return [[round(points[i], 1), round(points[i + 1], 1)] for i in range(0, len(points), 2)]


def context(cv):
    task = cv.find("tasks", "scene · reference frames")
    frames = cv.get(f"tasks/{task['id']}/data/meta")["frames"]
    frame = next(i for i, f in enumerate(frames) if f["name"] == FRAME_NAME)
    job = cv.get("jobs", task_id=task["id"])["results"][0]["id"]
    labels = {l["name"]: l for l in cv.get("labels", task_id=task["id"], page_size=50)["results"]}
    return frame, job, labels


def attr(labels, label, name):
    return next(a["id"] for a in labels[label]["attributes"] if a["name"] == name)


def push(cv):
    frame, job, labels = context(cv)
    g = json.loads(GEO.read_text(encoding="utf-8"))
    heads = json.loads(HEADS.read_text(encoding="utf-8"))["heads"]
    shapes = []

    def add(kind, label, pts, attrs=()):
        shapes.append({"type": kind, "frame": frame, "label_id": labels[label]["id"], "points": flat(pts),
                       "occluded": False, "outside": False, "z_order": 0, "rotation": 0, "group": 0, "source": "manual",
                       "attributes": [{"spec_id": attr(labels, label, n), "value": str(v)} for n, v in attrs]})
    for k, pts in g["crosswalks"].items():
        add("polygon", "crosswalk", pts, [("id", k)])
    for pts in g["islands"].values():
        add("polygon", "island", pts)
    for pts in g.get("median", {}).values():
        add("polygon", "island", pts)
    for pts in g["sidewalks"].values():
        add("polygon", "sidewalk", pts)
    for k, pts in g["stop_lines"].items():
        add("polyline", "stop_line", pts, [("approach", k)])
    for pts in g.get("solid_lines", {}).values():
        add("polyline", "solid_line", pts)
    for name, h in heads.items():
        x1, y1, x2, y2 = h["box"]
        add("rectangle", "traffic_light", [[x1, y1], [x2, y2]],
            [("kind", "pedestrian" if name.startswith("ped") else "vehicle"), ("controls", name)])
    ann = cv.get(f"jobs/{job}/annotations")
    keep = [s for s in ann["shapes"] if s["frame"] != frame]
    r = cv.s.put(f"{cv.host}/api/jobs/{job}/annotations", json={"version": ann["version"], "tags": ann["tags"],
                                                                 "shapes": keep + shapes, "tracks": ann["tracks"]})
    r.raise_for_status()
    print(f"pushed {len(shapes)} shapes to job {job}, frame {frame} ({FRAME_NAME})")


def pull(cv):
    frame, job, labels = context(cv)
    names = {l["id"]: n for n, l in labels.items()}
    ann = cv.get(f"jobs/{job}/annotations")
    g = json.loads(GEO.read_text(encoding="utf-8"))
    heads = json.loads(HEADS.read_text(encoding="utf-8"))
    new = {"crosswalks": {}, "islands": {}, "sidewalks": {}, "stop_lines": {}, "solid_lines": {}}
    median_area = 0.0
    for s in ann["shapes"]:
        if s["frame"] != frame:
            continue
        label = names[s["label_id"]]
        a = {next(x["name"] for x in labels[label]["attributes"] if x["id"] == v["spec_id"]): v["value"] for v in s["attributes"]}
        pts = pairs(s["points"])
        if label == "crosswalk":
            new["crosswalks"][a.get("id") or str(len(new["crosswalks"]) + 1)] = pts
        elif label == "island":
            new["islands"][str(len(new["islands"]))] = pts
        elif label == "sidewalk":
            new["sidewalks"][str(len(new["sidewalks"]))] = pts
        elif label == "stop_line":
            new["stop_lines"][a.get("approach") or str(len(new["stop_lines"]) + 1)] = [pts[0], pts[-1]]
        elif label == "solid_line":
            new["solid_lines"][f"s{len(new['solid_lines'])}"] = [pts[0], pts[-1]]
        elif label == "traffic_light" and a.get("controls") in heads["heads"]:
            (x1, y1), (x2, y2) = pts[0], pts[-1]
            heads["heads"][a["controls"]]["box"] = [round(min(x1, x2)), round(min(y1, y2)), round(max(x1, x2)), round(max(y1, y2))]
    if "4" not in new["stop_lines"] or "1" not in new["crosswalks"]:
        sys.exit("the rules need stop_line with approach=4 and crosswalk with id=1; nothing written")
    g.update({k: v for k, v in new.items() if v})
    g["median"] = {}
    g["reference"] = g.get("reference", "") + " | edited in CVAT"
    GEO.write_text(json.dumps(g, indent=1), encoding="utf-8")
    HEADS.write_text(json.dumps(heads, indent=1), encoding="utf-8")
    print("geometry.json:", {k: len(v) for k, v in new.items()})
    import cv2
    sys.path.insert(0, str(REPO / "src"))
    from westcv.signals import make_templates
    make_templates(cv2.imread(str(REPO / "DataSets/proxies/ref/C3896_median.png")), HEADS, REPO / "scene/signal_templates.npz")
    print("signal templates rebuilt from the edited head boxes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("push", "pull"))
    ap.add_argument("--host", default="http://localhost:8080")
    ap.add_argument("--user", default="admin")
    args = ap.parse_args()
    pw = os.environ.get("CVAT_PASSWORD") or getpass.getpass(f"CVAT password for {args.user}: ")
    cv = Cvat(args.host, args.user, pw)
    push(cv) if args.action == "push" else pull(cv)


if __name__ == "__main__":
    main()
