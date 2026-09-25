import os, sys, time
if len(sys.argv) > 2: os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = sys.argv[2]
import cv2
f = sys.argv[1]
cap = cv2.VideoCapture(f); n = 0; t = time.perf_counter()
while True:
    ok, fr = cap.read()
    if not ok: break
    n += 1; last = fr
dt = time.perf_counter() - t
print(f"opts={sys.argv[2] if len(sys.argv)>2 else None!r:<40} {n/dt:5.1f} fps  shape={last.shape if n else None} dtype=uint8  cap_threads_used? cv2.getNumThreads={cv2.getNumThreads()}")
