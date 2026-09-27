"""Rules on synthetic scenes: a straight road with one zebra, hand-made tracks."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from westcv import rules  # noqa: E402
from westcv.scene import Scene  # noqa: E402

FPS = 30.0
META = {"fps": FPS, "n_frames": 600, "width": 800, "height": 600}
H = np.eye(3)
GEOMETRY = {  # road = band y 200..400, sidewalks above and below, zebra at x 100..160
    "crosswalks": {"1": [[100, 200], [160, 200], [160, 400], [100, 400]]},
    "islands": {}, "median": {},
    "sidewalks": {"top": [[0, 0], [800, 0], [800, 200], [0, 200]], "bottom": [[0, 400], [800, 400], [800, 600], [0, 600]]},
    "stop_lines": {"4": [[300, 200], [300, 400]]},
}
NO_DETS = np.zeros((0, 7), np.float32)


def walk(tid, x, y0, y1, f0, f1, h=40, w=16, cls=0):
    """Rows (frame, id, x1, y1, x2, y2, conf, cls) of a box whose feet go from (x, y0) to (x, y1)."""
    frames = np.arange(f0, f1, 3)
    ys = np.linspace(y0, y1, len(frames))
    return np.array([[f, tid, x - w / 2, y - h, x + w / 2, y, 0.9, cls] for f, y in zip(frames, ys)], np.float32)


def scene(rows):
    return Scene(rows, NO_DETS, META, H, None, GEOMETRY)


def test_jaywalking_outside_zebra_is_an_event():
    sc = scene(walk(3, 500, 180, 420, 0, 180))  # crosses the road at x=500, 6 s
    ev = rules.jaywalking(sc)
    assert len(ev) == 1
    start, end = ev[0]
    assert 0.5 < start < 1.5 and 4.5 < end < 6.0  # on the road from y=200 to y=400


def test_crossing_on_the_zebra_is_not_jaywalking():
    sc = scene(walk(3, 130, 180, 420, 0, 180))  # inside the zebra x 100..160
    assert rules.jaywalking(sc) == []


def test_person_inside_a_vehicle_is_not_a_pedestrian():
    car = np.array([[f, 1, 470, 250, 560, 330, 0.9, 2] for f in range(0, 180, 3)], np.float32)
    driver = np.array([[f, 3, 500, 270, 520, 310, 0.9, 0] for f in range(0, 180, 3)], np.float32)
    sc = Scene(np.concatenate([car, driver]), car[:, [0, 2, 3, 4, 5, 6, 7]], META, H, None, GEOMETRY)
    assert rules.jaywalking(sc) == []


def test_unknown_camera_runs_only_geometry_free_classes():
    rows = walk(3, 500, 180, 420, 0, 180)
    sc = Scene(rows, NO_DETS, META, None, None, None)
    assert sc.auto
    assert rules.detect(sc, ["jaywalking", "stop_line", "congestion"]) == []
