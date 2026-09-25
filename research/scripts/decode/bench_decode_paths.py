import cv2, time, subprocess, sys, numpy as np
f = sys.argv[1]
def run(name, fn):
    t = time.perf_counter(); n = fn(); dt = time.perf_counter() - t
    print(f"{name:<52} {n:4d} frames {dt:6.2f}s  {n/dt:6.1f} fps  ({n/dt/29.97:4.2f}x realtime)", flush=True)
def cv_read():
    cap = cv2.VideoCapture(f); n = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        n += 1
    return n
def cv_read_resize():
    cap = cv2.VideoCapture(f); n = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        small = cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA); n += 1
    return n
def cv_grab_every3():
    cap = cv2.VideoCapture(f); n = 0; i = 0
    while cap.grab():
        if i % 3 == 0:
            ok, fr = cap.retrieve(); n += 1
        i += 1
    return i
def ff_pipe(w, h, extra=()):
    def inner():
        cmd = ["ffmpeg", "-loglevel", "error", "-threads", "0", *extra, "-i", f, "-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE); sz = w*h*3; n = 0
        while True:
            b = p.stdout.read(sz)
            if len(b) < sz: break
            n += 1
        p.wait(); return n
    return inner
def ff_null():
    t = subprocess.run(["ffmpeg", "-loglevel", "error", "-threads", "0", "-i", f, "-f", "null", "-"]); return 360
print("OpenCV", cv2.__version__, "threads", cv2.getNumThreads())
run("ffmpeg decode only (-f null)", ff_null)
run("OpenCV cap.read() every frame (= harness Part B)", cv_read)
run("OpenCV cap.read() + resize 960x540", cv_read_resize)
run("OpenCV grab() all, retrieve() every 3rd", cv_grab_every3)
run("ffmpeg pipe scale 1280x720 bgr24", ff_pipe(1280, 720))
run("ffmpeg pipe scale 960x540 bgr24", ff_pipe(960, 540))
run("ffmpeg -skip_frame nokey pipe 1280x720 (keyframes only)", ff_pipe(1280, 720, ("-skip_frame", "nokey")))
