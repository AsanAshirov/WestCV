"""Private Kaggle GPU run on the organizers' sample videos. Keep this kernel PRIVATE.

Pushed by kaggle/push.sh together with a private dataset holding the code. Steps:
install requirements, tests, decode benchmark, the official harness on all samples
(timing + predictions_samples.json), perception cache and review videos for
threshold tuning. Results land in /kaggle/working; fetch them with
`kaggle kernels output <user>/antigradient-samples-t4 -p kaggle_out`.

Videos come from an attached private dataset if there is one, otherwise they are
downloaded from the organizers' Google Drive links into /kaggle/tmp (not saved).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/tmp")
INPUT = Path("/kaggle/input")
DRIVE_IDS = ["1kR9jODA2Wotw4gwkvpRKdqFADNJNc1nS", "1hp8DYeqtYHSwfM6qAo9FPSRHlpMFrIN_",
             "10cHEReCWzO3u-Vk1CnNgHAx6egGy5MwJ", "1aJ-QsAZVYJtLKHiRvKKeBq1D3GWNobRd"]
SUMMARY: dict = {"steps": {}}


def run(name: str, cmd: list[str], cwd: Path) -> int:
    print(f"\n===== {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=cwd).returncode
    except OSError as exc:  # a missing binary must not stop the remaining steps
        print(exc, flush=True)
        rc = 127
    SUMMARY["steps"][name] = {"rc": rc, "sec": round(time.time() - t0, 1)}
    (WORK / "summary.json").write_text(json.dumps(SUMMARY, indent=1))
    return rc


def find_repo() -> Path:
    dst = TMP / "WestCV"
    for p in INPUT.rglob("solution.py"):
        if (p.parent / "run_submission.py").exists():
            shutil.copytree(p.parent, dst, dirs_exist_ok=True)
            return dst
    for z in INPUT.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            if "solution.py" in zf.namelist():
                zf.extractall(dst)
                return dst
    sys.exit("code dataset is not attached (run kaggle/push.sh)")


def find_videos() -> Path:
    for p in sorted(INPUT.rglob("*.MP4")) + sorted(INPUT.rglob("*.mp4")):
        return p.parent
    target = TMP / "samples"
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
    import gdown

    for file_id in DRIVE_IDS:
        gdown.download(id=file_id, output=str(target) + "/", quiet=False)
    return target


def main() -> None:
    TMP.mkdir(exist_ok=True)
    repo = find_repo()
    videos = find_videos()
    vids = sorted(p for p in videos.iterdir() if p.suffix in (".mp4", ".MP4"))
    SUMMARY["commit"] = (repo / "COMMIT").read_text().strip() if (repo / "COMMIT").exists() else "unknown"
    SUMMARY["videos"] = [p.name for p in vids]
    print("repo", repo, "commit", SUMMARY["commit"], "videos", SUMMARY["videos"], flush=True)

    run("install", [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt", "pytest"], repo)
    run("gpu", ["nvidia-smi"], repo)
    run("cpu", ["nproc"], repo)
    run("tests", [sys.executable, "-m", "pytest", "-q", "tests"], repo)
    shortest = min(vids, key=lambda p: p.stat().st_size)
    run("bench_decode", [sys.executable, "tools/bench_decode.py", str(shortest), "--seconds", "20"], repo)
    pred = WORK / "predictions_samples.json"
    run("harness", [sys.executable, "run_submission.py", "--videos", str(videos), "--out", str(pred),
                    "--team", "Antigradient"], repo)
    run("validate", [sys.executable, "evaluate.py", "--pred", str(pred), "--validate-only"], repo)
    if pred.exists():
        log = json.loads(pred.read_text())["log"]
        SUMMARY["timing"] = {v: {"x_duration": round(e["total_sec"] / max(e["duration"], 1e-6), 2), **e}
                             for v, e in log.items()}
        SUMMARY["events"] = {v: len(e["events"]) for v, e in json.loads(pred.read_text())["videos"].items()}
    run("cache", [sys.executable, "tools/cache_perception.py", "--videos", str(videos), "--out",
                  str(WORK / "cache")], repo)
    for v in vids:
        run(f"review_{v.stem}", [sys.executable, "tools/render_review.py", str(v), "--cache", str(WORK / "cache"),
                                 "--out", str(WORK / "review")], repo)
    (WORK / "summary.json").write_text(json.dumps(SUMMARY, indent=1))
    print(json.dumps(SUMMARY, indent=1))


if __name__ == "__main__":
    main()
