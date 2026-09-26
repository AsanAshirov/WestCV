# usage: python labelme_to_scene.py geometry/ref_CLIP01.json scene_CLIP01.json
import json
import sys
from collections import defaultdict

REQUIRED = {"carriageway", "stop_line", "crosswalk", "lane_dir"}   # подстройте под свою сцену

src = json.load(open(sys.argv[1], encoding="utf-8"))
W, H = src["imageWidth"], src["imageHeight"]
shapes = defaultdict(list)
for s in src["shapes"]:
    pts = [[x / W, y / H] for x, y in s["points"]]
    if s.get("shape_type") == "rectangle" and len(pts) == 2:      # 2 угла -> 4 вершины
        (x1, y1), (x2, y2) = pts
        x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    shapes[s["label"]].append({
        "type": s.get("shape_type"),
        "group_id": s.get("group_id"),
        "description": s.get("description") or "",
        "flags": s.get("flags") or {},
        "points": [[round(x, 6), round(y, 6)] for x, y in pts],   # нормировано к [0, 1]
    })
missing = REQUIRED - shapes.keys()
if missing:
    print("WARNING: missing labels:", sorted(missing))
json.dump({"source_image": src.get("imagePath"), "image_size": [W, H], "shapes": shapes},
          open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print({k: len(v) for k, v in shapes.items()})
