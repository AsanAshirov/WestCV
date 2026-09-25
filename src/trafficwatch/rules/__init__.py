"""One module per event class. Each exposes `detect(ctx, cfg) -> list[(start_sec, end_sec)]`."""
from . import accident, congestion, jaywalking, stopped_vehicle, wrong_way

RULES = {
    "stopped_vehicle": stopped_vehicle.detect,
    "jaywalking": jaywalking.detect,
    "wrong_way": wrong_way.detect,
    "congestion": congestion.detect,
    "accident": accident.detect,
}
