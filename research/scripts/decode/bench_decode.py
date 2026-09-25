import cv2, numpy as np, time, os
print("cv2", cv2.__version__, "threads", cv2.getNumThreads(), "cpus", os.cpu_count())
W,H,FPS,N = 1920,1080,25,750  # 30 s synthetic clip
path = "synth.mp4"
if not os.path.exists(path):
    for fourcc in ("avc1","H264","mp4v"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), FPS, (W,H))
        if vw.isOpened():
            print("writer fourcc", fourcc); break
    rng = np.random.default_rng(0)
    bg = (rng.integers(0,255,(H//8,W//8,3),dtype=np.uint8))
    bg = cv2.resize(bg,(W,H),interpolation=cv2.INTER_CUBIC)
    for i in range(N):
        f = bg.copy()
        for k in range(20):
            x = int((i*7 + k*97) % (W-120)); y = int((k*53) % (H-80))
            cv2.rectangle(f,(x,y),(x+120,y+60),(int(k*12),200,255-k*10),-1)
        vw.write(f)
    vw.release()
cap = cv2.VideoCapture(path)
print("backend", cap.getBackendName(), "frames", cap.get(cv2.CAP_PROP_FRAME_COUNT), "fps", cap.get(cv2.CAP_PROP_FPS), "codec", int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4,'little'))
t=time.perf_counter(); n=0
while True:
    ok,f = cap.read()
    if not ok: break
    n+=1
dt=time.perf_counter()-t; print(f"read all: {n} frames {dt:.2f}s -> {n/dt:.0f} fps")
cap.release()
cap = cv2.VideoCapture(path)
t=time.perf_counter(); n=0; kept=0
while True:
    ok = cap.grab()
    if not ok: break
    if n % 3 == 0:
        ok,f = cap.retrieve(); kept+=1
    n+=1
dt=time.perf_counter()-t; print(f"grab + retrieve every 3rd: {n} frames ({kept} kept) {dt:.2f}s -> {n/dt:.0f} fps")
cap.release()
# seek accuracy test
cap = cv2.VideoCapture(path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 400)
ok,f1 = cap.read(); pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
cap2 = cv2.VideoCapture(path)
for i in range(400): cap2.grab()
ok2,f2 = cap2.read()
print("seek pos after read", pos, "seek==sequential frame:", bool(ok and ok2 and np.array_equal(f1,f2)))
