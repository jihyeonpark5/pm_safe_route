"""Uncalibrated, symmetric turn-delay assumptions for sensitivity analysis."""
from dataclasses import dataclass
import math


def turn_angle(previous, following):
    a, b = float(previous.end_heading_deg), float(following.start_heading_deg)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Finite directed headings required")
    difference = abs(a-b) % 360
    return min(difference, 360-difference)


@dataclass(frozen=True)
class TurnDelayProfile:
    seconds_at_90: float = 3.0

    def __post_init__(self):
        if not math.isfinite(self.seconds_at_90) or self.seconds_at_90 < 0:
            raise ValueError("seconds_at_90 must be finite and nonnegative")

    def __call__(self, previous, following):
        angle = turn_angle(previous, following)
        # <=30 degrees: no added delay; 90 degrees: configured delay;
        # >=150 degrees: twice the configured delay. No left/right asymmetry.
        return self.seconds_at_90 * min(2.0, max(0.0, (angle-30.0)/60.0))


def valid_walk(path, start, end):
    node = start
    for arc in path:
        if arc.tail != node:
            return False
        node = arc.head
    return node == end
