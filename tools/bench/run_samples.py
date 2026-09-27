# run_samples.py — the official harness on each sample video separately, then predictions_samples.json.
#
#   python tools/bench/run_samples.py [--videos DataSets] [--time-factor 3]
#
# Every finished video is saved to renders/sub/parts/<video>.json at once, so a crash or a power loss
# only loses the video in progress: run the same command again and it continues with the next one.
# A part that ran over its budget, or where Part A hit its decode brake (a throttled laptop), is
# recomputed. When all videos are done, the parts are merged into
# predictions_samples.json in the repo root and validated with evaluate.py.
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARTS = ROOT / "renders/sub/parts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(ROOT / "DataSets"))
    ap.add_argument("--time-factor", type=float, default=3.0, help="10 on a slow laptop (development only)")
    ap.add_argument("--team", default="westcv")
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in Path(args.videos).iterdir() if p.suffix in (".mp4", ".MP4"))
    env = {**os.environ, "WESTCV_TIME_FACTOR": str(args.time_factor), "PYTHONIOENCODING": "utf-8"}
    for v in videos:
        part = PARTS / f"{v.name}.json"
        stderr = PARTS / f"{v.name}.log"
        if part.exists():
            log = json.loads(part.read_text())["log"][v.name]
            braked = "'braked': True" in stderr.read_text(encoding="utf-8", errors="replace") if stderr.exists() else True
            if not log["errors"] and not braked:
                print(f"{v.name}: done ({log['total_sec']} s), skipped", flush=True)
                continue
            print(f"{v.name}: previous run incomplete (errors {log['errors']}, braked {braked}), recomputing", flush=True)
        print(f"{v.name}: running", flush=True)
        with open(stderr, "w", encoding="utf-8") as err:
            subprocess.run([sys.executable, "run_submission.py", "--videos", str(v), "--out", str(part),
                            "--team", args.team, "--time-factor", str(args.time_factor)], cwd=ROOT, env=env,
                           stderr=err, check=True)
    result = {"team": args.team, "videos": {}, "log": {}}
    for v in videos:
        d = json.loads((PARTS / f"{v.name}.json").read_text())
        result["videos"].update(d["videos"])
        result["log"].update(d["log"])
    out = ROOT / "predictions_samples.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}", flush=True)
    subprocess.run([sys.executable, "evaluate.py", "--pred", str(out), "--validate-only"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
