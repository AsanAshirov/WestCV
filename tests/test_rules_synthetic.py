"""Hand-made trajectories must give exactly the expected segments.
This is the only check for classes that do not occur in the sample videos."""
import numpy as np
from conftest import DT, FRAME, drive, make_track

from trafficwatch.detector import PERSON, VEHICLE
from trafficwatch.rules import RULES
from trafficwatch.rules.common import RuleContext
from trafficwatch.scene import Scene, build_maps


def context(cfg, tracks, duration, scene=None):
    maps = build_maps(tracks, FRAME, scene, cfg["scene"])
    return RuleContext(tracks, maps, duration, DT, cfg["tracks"])


def eastbound_traffic(cfg, start_id, t_from, t_to, every, y, speed_px=200):
    tracks = []
    for k, t0 in enumerate(np.arange(t_from, t_to, every)):
        t, foot = drive(t0, t0 + 1280 / speed_px, -40, 1320, y)
        tracks.append(make_track(start_id + k, VEHICLE, t, foot, cfg=cfg))
    return tracks


def westbound_traffic(cfg, start_id, t_from, t_to, every, y, speed_px=200):
    tracks = []
    for k, t0 in enumerate(np.arange(t_from, t_to, every)):
        t, foot = drive(t0, t0 + 1280 / speed_px, 1320, -40, y)
        tracks.append(make_track(start_id + k, VEHICLE, t, foot, cfg=cfg))
    return tracks


def test_stopped_vehicle_starts_when_it_stops(cfg):
    traffic = eastbound_traffic(cfg, 100, 0, 60, 3, y=400) + eastbound_traffic(cfg, 200, 1, 60, 3, y=460)
    t1, f1 = drive(5, 10, 0, 600, 400)
    t2 = np.round(np.arange(10.1, 30.0, DT), 3)
    f2 = np.column_stack([np.full(len(t2), 600.0), np.full(len(t2), 400.0)])
    t3, f3 = drive(30, 33, 600, 1300, 400)
    t = np.concatenate([t1, t2, t3])
    car = make_track(1, VEHICLE, t, np.vstack([f1, f2, f3]), cfg=cfg)
    ctx = context(cfg, traffic + [car], 60)
    out = RULES["stopped_vehicle"](ctx, cfg["classes"]["stopped_vehicle"])
    assert len(out) == 1
    s, e = out[0]
    assert abs(s - 10) < 1.0 and abs(e - 30) < 1.0


def test_stopped_vehicle_ignores_signal_queue(cfg):
    # five cars stand side by side for 30 s, nobody passes them, then all leave
    tracks = eastbound_traffic(cfg, 100, 0, 5, 1, y=400)
    for k in range(5):
        t = np.round(np.arange(10, 40, DT), 3)
        x = np.where(t < 38, 300 + 90 * k, 300 + 90 * k + (t - 38) * 300)
        tracks.append(make_track(k + 1, VEHICLE, t, np.column_stack([x, np.full(len(t), 400)]), cfg=cfg))
    ctx = context(cfg, tracks, 60)
    assert RULES["stopped_vehicle"](ctx, cfg["classes"]["stopped_vehicle"]) == []


def test_wrong_way_only_for_the_car_against_its_lane(cfg):
    tracks = eastbound_traffic(cfg, 100, 0, 60, 2, y=400) + westbound_traffic(cfg, 200, 0, 60, 2, y=250)
    t, foot = drive(20, 26, 1300, 0, 400)
    tracks.append(make_track(1, VEHICLE, t, foot, cfg=cfg))
    ctx = context(cfg, tracks, 60)
    out = RULES["wrong_way"](ctx, cfg["classes"]["wrong_way"])
    assert len(out) == 1
    s, e = out[0]
    assert 19.5 <= s <= 21.5 and 24 <= e <= 26.5


def test_jaywalking_outside_crossing_only(cfg):
    tracks = (eastbound_traffic(cfg, 100, 0, 60, 2, y=400) + eastbound_traffic(cfg, 300, 1, 60, 2, y=440)
              + westbound_traffic(cfg, 200, 0, 60, 2, y=300) + westbound_traffic(cfg, 400, 1, 60, 2, y=340))
    scene = Scene({"crosswalk": [np.array([[0.7, 0.2], [0.8, 0.2], [0.8, 0.8], [0.7, 0.8]])]})
    t = np.round(np.arange(10, 20, DT), 3)
    y = np.interp(t, [10, 20], [150, 650])
    jay = make_track(1, PERSON, t, np.column_stack([np.full(len(t), 300), y]), size=(20, 50), cfg=cfg)
    legal = make_track(2, PERSON, t + 20, np.column_stack([np.full(len(t), 960), y]), size=(20, 50), cfg=cfg)
    ctx = context(cfg, tracks + [jay, legal], 60, scene)
    out = RULES["jaywalking"](ctx, cfg["classes"]["jaywalking"])
    assert len(out) == 1 and 10 <= out[0][0] < 20
    ctx_no_scene = context(cfg, tracks + [jay], 60)
    assert RULES["jaywalking"](ctx_no_scene, cfg["classes"]["jaywalking"]) == []


def test_congestion_long_standstill(cfg):
    tracks = eastbound_traffic(cfg, 100, 0, 30, 1.5, y=400)
    for k in range(6):  # a jam from 40 s to 130 s
        t = np.round(np.arange(40, 130, DT), 3)
        x = 150 + 180 * k + 0.02 * (t - 40)
        tracks.append(make_track(k + 1, VEHICLE, t, np.column_stack([x, np.full(len(t), 400)]), cfg=cfg))
    ctx = context(cfg, tracks, 140)
    out = RULES["congestion"](ctx, cfg["classes"]["congestion"])
    assert len(out) == 1
    s, e = out[0]
    assert abs(s - 40) <= 2 and abs(e - 130) <= 2


def test_accident_side_impact(cfg):
    tracks = eastbound_traffic(cfg, 100, 0, 40, 3, y=400)
    t = np.round(np.arange(10, 25, DT), 3)
    xa = np.where(t < 13, 200 + 200 * (t - 10), 800)          # eastbound, stops dead at 13 s
    a = make_track(1, VEHICLE, t, np.column_stack([xa, np.full(len(t), 400)]), cfg=cfg)
    yb = np.where(t < 13, 100 + 100 * (t - 10), 400)           # southbound into it
    xb = np.full(len(t), 850.0)
    b = make_track(2, VEHICLE, t, np.column_stack([xb, yb]), cfg=cfg)
    ctx = context(cfg, tracks + [a, b], 40)
    out = RULES["accident"](ctx, cfg["classes"]["accident"])
    assert len(out) >= 1
    assert 11.5 <= out[0][0] <= 13.5
