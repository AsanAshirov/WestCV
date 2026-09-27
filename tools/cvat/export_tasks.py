# export_tasks.py — download every annotator's copy (consensus replica job) of every
# "WestCV events" task as CVAT for video 1.1 XML: labels/cvat/<assignee>/<video>.xml
# (unassigned replicas go to labels/cvat/replica<N>/).
#
#   python tools/cvat/export_tasks.py --user admin      (password: env CVAT_PASSWORD or prompt)
#   then: python tools/annotation/cvat_to_gt.py labels/cvat/A/*.xml --videos DataSets --out labels/gt_A.json
import argparse
import getpass
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

from setup_tasks import REPO, Cvat

FORMAT = "CVAT for video 1.1"


def export_xml(cv: Cvat, job_id: int) -> bytes:
    r = cv.s.post(f"{cv.host}/api/jobs/{job_id}/dataset/export",
                  params={"format": FORMAT, "save_images": "false"})
    if not r.ok:
        sys.exit(f"export of job {job_id}: {r.status_code} {r.text[:300]}")
    rq = r.json()["rq_id"]
    while (st := cv.get(f"requests/{rq}"))["status"] not in ("finished", "failed"):
        time.sleep(2)
    if st["status"] == "failed":
        sys.exit(f"export of job {job_id} failed: {st.get('message')}")
    blob = cv.s.get(st["result_url"])
    blob.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(blob.content)) as zf:
        return scrub(zf.read("annotations.xml"))


def scrub(xml: bytes) -> bytes:
    """Drop account e-mails (owner/assignee blocks, e-mail usernames): the exports go to a public repo."""
    xml = re.sub(rb"\s*<email>[^<]*</email>", b"", xml)
    return re.sub(rb"<username>[^<]*@[^<]*</username>", b"<username>redacted</username>", xml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8080")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--out", default=str(REPO / "labels" / "cvat"))
    args = ap.parse_args()
    password = os.environ.get("CVAT_PASSWORD") or getpass.getpass(f"CVAT password for {args.user}: ")
    cv = Cvat(args.host, args.user, password)
    project = cv.find("projects", "WestCV events")
    if project is None:
        sys.exit("no project 'WestCV events'; run setup_tasks.py first")
    for t in cv.get("tasks", project_id=project["id"], page_size=100)["results"]:
        jobs = cv.get("jobs", task_id=t["id"], type="consensus_replica", page_size=100)["results"]
        for n, job in enumerate(sorted(jobs, key=lambda j: j["id"]), 1):
            a = job.get("assignee") or {}
            u = a.get("username")
            who = (f"user{a['id']}" if u and "@" in u else u) or f"replica{n}"  # no e-mails in paths
            dst = Path(args.out) / who / f"{t['name']}.xml"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(export_xml(cv, job["id"]))
            print(f"{t['name']} job {job['id']} ({who}) -> {dst.relative_to(REPO)}")


if __name__ == "__main__":
    main()
