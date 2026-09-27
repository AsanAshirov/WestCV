# setup_tasks.py — create the WestCV projects and tasks in our CVAT through the REST API.
#
#   python tools/cvat/setup_tasks.py --user admin            (password: env CVAT_PASSWORD or prompt)
#
# * project "WestCV events": 14 event labels (tools/annotation/cvat_labels_events.json);
#   one task per video from the share folder (DataSets/proxies, mounted read-only) with
#   consensus replicas: 2 independent copies of the annotation job (C3905: 3, calibration),
#   each assigned to a different annotator in the UI. The video and its chunk cache are
#   stored once per video, not once per annotator.
# * project "WestCV scene": scene geometry labels + one image task with the reference frames
#   DataSets/proxies/ref/*.png (skipped if there are none yet).
# Each video task is checked: CVAT frame count must equal the original's, frame 0 = frame 0.
# Idempotent: existing projects/tasks with the same name are reused.
import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
PROXIES = REPO / "DataSets" / "proxies"
N_FRAMES = {"C3896": 10200, "C3897": 9525, "C3902": 9525, "C3905": 3825}  # ffprobe of the originals
REPLICAS = {"C3905": 3, "C3896": 2, "C3897": 2, "C3902": 2}  # annotators per video


class Cvat:
    def __init__(self, host: str, user: str, password: str):
        self.host, self.s = host.rstrip("/"), requests.Session()
        r = self.s.post(f"{self.host}/api/auth/login", json={"username": user, "password": password})
        r.raise_for_status()
        self.s.headers["Authorization"] = f"Token {r.json()['key']}"

    def get(self, path, **params):
        r = self.s.get(f"{self.host}/api/{path}", params=params)
        r.raise_for_status()
        return r.json()

    def post(self, path, payload):
        r = self.s.post(f"{self.host}/api/{path}", json=payload)
        if not r.ok:
            sys.exit(f"POST {path}: {r.status_code} {r.text[:500]}")
        return r.json()

    def find(self, kind, name):
        res = self.get(kind, name=name, page_size=100)["results"]
        return next((x for x in res if x["name"] == name), None)

    def project(self, name, labels_file):
        p = self.find("projects", name)
        if p:
            return p["id"]
        labels = json.loads(Path(labels_file).read_text(encoding="utf-8"))
        return self.post("projects", {"name": name, "labels": labels})["id"]

    def task(self, name, project_id, data: dict, **fields) -> int:
        t = self.find("tasks", name)
        if t and t.get("size"):
            return t["id"]
        if t:  # left over from a failed data upload: data cannot be re-attached, start over
            self.s.delete(f"{self.host}/api/tasks/{t['id']}").raise_for_status()
        tid = self.post("tasks", {"name": name, "project_id": project_id, **fields})["id"]
        rq = self.post(f"tasks/{tid}/data", data)["rq_id"]
        while True:
            st = self.get(f"requests/{rq}")
            if st["status"] in ("finished", "failed"):
                break
            print(f"  {name}: {st['status']} {st.get('message') or ''}".rstrip(), end="\r")
            time.sleep(3)
        if st["status"] == "failed":
            sys.exit(f"{name}: data upload failed: {st.get('message')}")
        return tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8080")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--proxy", default="1080p", help="proxy suffix in DataSets/proxies: 1080p or 720p")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality of the chunks CVAT serves")
    args = ap.parse_args()
    password = os.environ.get("CVAT_PASSWORD") or getpass.getpass(f"CVAT password for {args.user}: ")
    cv = Cvat(args.host, args.user, password)

    pid = cv.project("WestCV events", REPO / "tools/annotation/cvat_labels_events.json")
    ok = True
    for video, replicas in REPLICAS.items():
        src = f"{video}_{args.proxy}.mp4"
        if not (PROXIES / src).exists():
            print(f"skip {video}: no {src} yet")
            continue
        tid = cv.task(video, pid, {"server_files": [src], "image_quality": args.quality,
                                   "use_cache": True, "sorting_method": "natural"},
                      segment_size=N_FRAMES[video], consensus_replicas=replicas)
        meta = cv.get(f"tasks/{tid}/data/meta")
        size = meta["size"]
        good = size == N_FRAMES[video] and meta["start_frame"] == 0 and not meta.get("frame_filter")
        jobs = cv.get("jobs", task_id=tid, page_size=100)["results"]
        n_rep = sum(j["type"] == "consensus_replica" for j in jobs)
        good &= n_rep == replicas
        ok &= good
        print(f"{video}: task {tid}, {size} frames (original {N_FRAMES[video]}), "
              f"{n_rep} replica jobs {'OK' if good else 'MISMATCH'}")

    refs = sorted((PROXIES / "ref").glob("*.png")) if (PROXIES / "ref").is_dir() else []
    if refs:
        sid = cv.project("WestCV scene", REPO / "tools/annotation/cvat_labels_scene.json")
        tid = cv.task("scene · reference frames", sid, {"server_files": [f"ref/{p.name}" for p in refs],
                                                         "image_quality": 95, "use_cache": True,
                                                         "sorting_method": "natural"})
        print(f"scene task {tid}: {len(refs)} reference frames")
    else:
        print("no reference frames in DataSets/proxies/ref yet: scene task skipped")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
