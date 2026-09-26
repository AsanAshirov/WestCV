import numpy as np

from trafficwatch.detector import PERSON, VEHICLE, superclass_nms
from trafficwatch.risk import time_to_collision
from trafficwatch.tracker import ByteTracker


def test_tracker_keeps_ids_for_fast_objects():
    tr = ByteTracker()
    ids = set()
    for k in range(30):
        t = k * 0.1
        x = 100 + 25 * k  # 250 px/s, box 80 px wide: consecutive IoU ~0.5
        dets = np.array([[x, 400, x + 80, 450, 0.9, VEHICLE],
                         [900 - 10 * k, 200, 930 - 10 * k, 280, 0.8, PERSON]], np.float32)
        out = tr.update(t, dets)
        ids.update(out[:, 0].astype(int).tolist())
    assert ids == {1, 2}


def test_tracker_does_not_match_across_superclasses():
    tr = ByteTracker()
    tr.update(0.0, np.array([[0, 0, 50, 50, 0.9, VEHICLE]], np.float32))
    out = tr.update(0.1, np.array([[0, 0, 50, 50, 0.9, PERSON]], np.float32))
    assert int(out[0, 0]) == 2


def test_superclass_nms_removes_car_truck_duplicate():
    det = np.array([[0, 0, 100, 60, 0.9, VEHICLE], [2, 1, 101, 61, 0.6, VEHICLE],
                    [0, 0, 100, 60, 0.7, PERSON]], np.float32)
    assert len(superclass_nms(det, 0.7)) == 2


def test_time_to_collision_head_on():
    a = np.array([100, 380, 180, 400.0])
    b = np.array([400, 380, 480, 400.0])
    ttc = time_to_collision(a, np.array([100.0, 0]), b, np.array([-100.0, 0]), 3.0, 0.1, 0.3)
    assert abs(ttc - 1.1) < 0.11  # gap 220 px closing at 200 px/s
    assert time_to_collision(a, np.zeros(2), b, np.zeros(2), 3.0, 0.1, 0.3) is None
