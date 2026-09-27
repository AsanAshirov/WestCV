# rehearsal.py — the submission as the judges will run it, on a Kaggle T4 (private kernel).
# Clean install from requirements.txt, weights from weights/download.sh, the four samples from the
# organizers' Google Drive, the official harness on all of them -> predictions_samples.json, then a
# second run of one sample to check that the output is identical.
# Input: private dataset westcv-code (a snapshot of the files the tagged commit would contain).
import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile

DRIVE = ["1kR9jODA2Wotw4gwkvpRKdqFADNJNc1nS", "1hp8DYeqtYHSwfM6qAo9FPSRHlpMFrIN_",
         "10cHEReCWzO3u-Vk1CnNgHAx6egGy5MwJ", "1aJ-QsAZVYJtLKHiRvKKeBq1D3GWNobRd"]  # Videos.pdf
OUT = "/kaggle/working"


def sh(cmd, cwd=None, check=True):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    print(r.stdout[-3000:], r.stderr[-3000:], sep="\n", flush=True)
    if check and r.returncode:
        sys.exit(f"failed ({r.returncode}): {cmd}")
    return r


src = os.path.dirname(glob.glob("/kaggle/input/**/solution.py", recursive=True)[0])
code = "/tmp/code"
shutil.copytree(src, code, dirs_exist_ok=True)
for z in glob.glob(f"{code}/*.zip"):  # the Kaggle CLI uploads folders as zips
    with zipfile.ZipFile(z) as f:
        f.extractall(os.path.join(code, os.path.splitext(os.path.basename(z))[0]))
    os.remove(z)
sh("python --version; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nproc; df -h /tmp | tail -1")
sh(f"{sys.executable} -m pip install -q -r requirements.txt", cwd=code)
sh(f"{sys.executable} -c \"import torch, av, cv2, ultralytics; print(torch.__version__, torch.cuda.is_available(), av.__version__, cv2.__version__, ultralytics.__version__)\"")
sh(f"{sys.executable} -m pip list 2>/dev/null | grep -i -E 'opencv|numpy|torch|^av '")
sh("bash weights/download.sh", cwd=code)

videos = "/tmp/videos"
os.makedirs(videos, exist_ok=True)
for v in glob.glob("/kaggle/input/**/*.MP4", recursive=True) + glob.glob("/kaggle/input/**/*.mp4", recursive=True):
    os.symlink(v, f"{videos}/{os.path.basename(v)}")  # samples uploaded as private Kaggle datasets
if len(os.listdir(videos)) < 4:
    sh(f"{sys.executable} -m pip install -q gdown")
    for fid in DRIVE:
        sh(f"gdown --quiet {fid} -O {videos}/", check=False)
sh(f"ls -laL {videos}")
if not os.listdir(videos):
    sys.exit("no sample videos")

sh(f"{sys.executable} run_submission.py --videos {videos} --out {OUT}/predictions_samples.json --team westcv", cwd=code)
sh(f"{sys.executable} evaluate.py --pred {OUT}/predictions_samples.json --validate-only", cwd=code)
first = json.load(open(f"{OUT}/predictions_samples.json"))
print(json.dumps(first.get("log"), indent=1))

again = "/tmp/again"
os.makedirs(again, exist_ok=True)
os.symlink(f"{videos}/C3905.MP4", f"{again}/C3905.MP4")
sh(f"{sys.executable} run_submission.py --videos {again} --out {OUT}/rerun_C3905.json --team westcv", cwd=code)
second = json.load(open(f"{OUT}/rerun_C3905.json"))
same = first["videos"]["C3905.MP4"] == second["videos"]["C3905.MP4"]
print("DETERMINISM C3905:", "identical" if same else "DIFFERENT")
