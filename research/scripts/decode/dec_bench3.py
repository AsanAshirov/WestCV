import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pylib'))
import av, cv2
def ocv(path, stride=1):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG); n=0; t0=time.perf_counter()
    while True:
        if stride == 1:
            ok, f = cap.read()
        else:
            ok = cap.grab()
            if ok and n % stride == 0: ok, f = cap.retrieve()
        if not ok: break
        n+=1
    return n/(time.perf_counter()-t0)
def pa(path, stride=1, bgr=True):
    c=av.open(path); s=c.streams.video[0]; s.thread_type='AUTO'; n=0; t0=time.perf_counter()
    for i,fr in enumerate(c.decode(s)):
        if bgr and i%stride==0: fr.to_ndarray(format='bgr24')
        n+=1
    c.close(); return n/(time.perf_counter()-t0)
for path in ['c1080.mp4','t2160.mp4']:
    for rep in range(3):
        print(path, rep, f"ocv read={ocv(path):.0f} ocv grab/3={ocv(path,3):.0f} pyav bgr={pa(path):.0f} pyav bgr/3={pa(path,3):.0f} pyav dec-only={pa(path,bgr=False):.0f}", flush=True)
