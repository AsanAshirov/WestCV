from trafficwatch.postprocess import finalize, merge_intervals, sanitize


def test_merge_intervals_joins_small_gaps():
    assert merge_intervals([(0, 5), (5.5, 10), (20, 21)], gap_s=1.0) == [(0, 10), (20, 21)]


def test_sanitize_rounds_caps_and_drops_blips():
    events = [[1.00001, 1.0003, "accident"], [-1, 5.12345, "jaywalking"], [300.0, 400.0, "congestion"]]
    out = sanitize(events, duration=340.34)
    assert out == [[0.0, 5.123, "jaywalking"], [300.0, 340.34, "congestion"]]


def test_sanitize_merges_same_class_overlap_but_not_touching_classes():
    out = sanitize([[0, 5, "wrong_way"], [4, 8, "wrong_way"], [4, 8, "accident"]], duration=60)
    assert out == [[0.0, 8.0, "wrong_way"], [4.0, 8.0, "accident"]]


def test_finalize_applies_min_duration(cfg):
    out = finalize({"jaywalking": [(1.0, 1.5), (10.0, 14.0)]}, duration=60, classes_cfg=cfg["classes"])
    assert out == [[10.0, 14.0, "jaywalking"]]
