import time, sys, cv2, av, os
os.environ.setdefault("OPENCV_LOG_LEVEL","ERROR")
for f in sys.argv[1:]:
    cap = cv2.VideoCapture(f)
    print(f, "cv2 opened", cap.isOpened(), "fps", cap.get(cv2.CAP_PROP_FPS), "n", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "w,h", cap.get(3), cap.get(4))
    t=time.perf_counter(); n=0; shp=None; dt=None
    while True:
        ok, fr = cap.read()
        if not ok: break
        n+=1; shp=fr.shape; dt=fr.dtype
    el=time.perf_counter()-t; print(f"  cv2.read: {n} frames {shp} {dt} -> {n/el:.1f} fps")
    c = av.open(f); s = c.streams.video[0]; s.thread_type="AUTO"
    t=time.perf_counter(); n=0
    for fr in c.decode(s):
        img = fr.to_ndarray(format="bgr24"); n+=1
    el=time.perf_counter()-t; print(f"  PyAV full-res bgr24 (threads AUTO): {n} frames {img.shape} -> {n/el:.1f} fps"); c.close()
    c = av.open(f); s = c.streams.video[0]; s.thread_type="AUTO"
    t=time.perf_counter(); n=0
    for fr in c.decode(s):
        img = fr.reformat(width=1280, height=720, format="bgr24").to_ndarray(); n+=1
    el=time.perf_counter()-t; print(f"  PyAV -> 1280x720 bgr24: {n} frames -> {n/el:.1f} fps"); c.close()
