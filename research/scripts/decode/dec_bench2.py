import sys, time, os, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pylib'))
import av, numpy as np, cv2
print('av', av.__version__)
def pyav(path, stride=1, thread_type='AUTO', to_bgr=True, keyframes_only=False):
    c = av.open(path); s = c.streams.video[0]; s.thread_type = thread_type
    n = 0; t0 = time.perf_counter()
    for i, fr in enumerate(c.decode(s)):
        if i % stride == 0 and to_bgr:
            a = fr.to_ndarray(format='bgr24')
        n += 1
    dt = time.perf_counter() - t0; c.close(); return n, dt
def pyav_scaled(path, stride, w, h):
    c = av.open(path); s = c.streams.video[0]; s.thread_type = 'AUTO'
    n = 0; t0 = time.perf_counter()
    for i, fr in enumerate(c.decode(s)):
        if i % stride == 0:
            a = fr.reformat(width=w, height=h, format='bgr24').to_ndarray()
        n += 1
    dt = time.perf_counter() - t0; c.close(); return n, dt
def ffpipe(path, w, h):
    cmd = ['ffmpeg','-loglevel','error','-i',path,'-f','rawvideo','-pix_fmt','bgr24','-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w*h*3*4)
    sz = w*h*3; n = 0; t0 = time.perf_counter()
    while True:
        b = p.stdout.read(sz)
        if len(b) < sz: break
        a = np.frombuffer(b, np.uint8).reshape(h, w, 3); n += 1
    p.wait(); return n, time.perf_counter() - t0
for path, w, h in [('t1080.mp4',1920,1080),('c1080.mp4',1920,1080),('t2160.mp4',3840,2160)]:
    for label, fn in [('pyav AUTO bgr every frame', lambda: pyav(path,1)),
                      ('pyav AUTO decode only', lambda: pyav(path,1,to_bgr=False)),
                      ('pyav AUTO bgr stride3', lambda: pyav(path,3)),
                      ('pyav AUTO scaled 1280x720 stride1', lambda: pyav_scaled(path,1,1280,720)),
                      ('ffmpeg pipe bgr24 every frame', lambda: ffpipe(path,w,h))]:
        n, dt = fn(); print(f'{path} {label}: {n} frames {dt:.2f}s -> {n/dt:.0f} fps')
