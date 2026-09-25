import cv2, sys, time
c = cv2.VideoCapture(sys.argv[1]); n = 0; t0 = time.perf_counter()
while c.read()[0]: n += 1
print(f"  cv2 read: {n} frames, {n/(time.perf_counter()-t0):.1f} fps")
