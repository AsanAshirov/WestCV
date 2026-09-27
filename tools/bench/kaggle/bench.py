# bench.py — runtime table on a Kaggle T4 (private kernel) for the 3x time budget.
# Measures on the 4K sample: the harness's Part B decode (cv2.read of every frame), our
# Part A decode (PyAV, non-reference frames dropped, scaled to 1920), and YOLO latency.
# Push: kaggle kernels push -p tools/bench/kaggle ; logs: kaggle kernels output zakhraluna/westcv-bench-t4
import glob
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.163", "av==18.1.0"], check=True)

import av  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

video = glob.glob("/kaggle/input/**/C3905.MP4", recursive=True)[0]
cap = cv2.VideoCapture(video)
fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dur = n / fps
print(f"video {video}: {n} frames, {dur:.1f}s; cpus={os.cpu_count()}; gpu={torch.cuda.get_device_name(0)}")
print(f"cv2 {cv2.__version__}, torch {torch.__version__}")
rows = []

t = time.perf_counter()
k = 0
while cap.read()[0]:
    k += 1
rows.append(("harness Part B decode: cv2.read all frames", k, time.perf_counter() - t))
cap.release()

for skip in ("DEFAULT", "NONREF"):
    t = time.perf_counter()
    k = 0
    with av.open(video) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        st.codec_context.skip_frame = skip
        for fr in c.decode(st):
            fr.to_ndarray(width=1920, height=1080, format="bgr24", interpolation="AREA")
            k += 1
    rows.append((f"Part A decode PyAV skip_frame={skip} -> 1920 BGR", k, time.perf_counter() - t))

frame = np.random.randint(0, 255, (1080, 1920, 3), np.uint8)
for weights, imgsz, batch in (("yolo26m.pt", 1920, 1), ("yolo26m.pt", 1920, 4), ("yolo26m.pt", 1280, 4),
                              ("yolo26l.pt", 1920, 4), ("yolo26s.pt", 960, 1)):
    m = YOLO(weights)
    for _ in range(3):
        m.predict([frame] * batch, imgsz=imgsz, quantize=16, device=0, verbose=False)
    torch.cuda.synchronize()
    t = time.perf_counter()
    reps = 20
    for _ in range(reps):
        m.predict([frame] * batch, imgsz=imgsz, quantize=16, device=0, verbose=False)
    torch.cuda.synchronize()
    per = (time.perf_counter() - t) / (reps * batch)
    rows.append((f"{weights} imgsz={imgsz} batch={batch} fp16: per frame", reps * batch, per * (n / 3)))

print(f"\n{'what':62s} {'frames':>7s} {'sec':>8s} {'x video':>8s}")
for name, k, sec in rows:
    print(f"{name:62s} {k:7d} {sec:8.1f} {sec / dur:8.2f}")
print("(YOLO rows: time to run the model on every 3rd frame of this video)")
