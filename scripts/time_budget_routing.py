"""Minimize additive distance exposure subject to a travel-time budget.

Uses nondominated (time, distance exposure, optional time exposure) labels.
Reverse shortest-path bounds prune labels without discretizing time. A search
limit returns a feasible incumbent and a lower bound, never a false optimum.
Inputs must already encode allowed travel directions and road access.
An optional static turn-delay callback adds incoming-arc states, travel delay,
and waiting exposure using max(incoming RS, outgoing RS). Reverse time and
time-exposure bounds use those incoming-arc states when turn delay is enabled.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
import itertools
import math
import time
from typing import Any

EPS = 1e-9


@dataclass
class BudgetRoute:
    path_arcs: list[Any]
    status: str
    stop_reason: str
    fastest_time_sec: float
    time_budget_sec: float
    travel_time_sec: float
    distance_exposure_m: float
    lower_bound_exposure_m: float
    labels_generated: int
    labels_expanded: int
    elapsed_sec: float
    time_exposure_sec: float = 0.0
    max_time_exposure_sec: float | None = None
    fastest_path_arcs: list[Any] | None = None

    @property
    def relative_gap(self):
        if self.distance_exposure_m == 0:
            return 0.0
        return max(0.0, (self.distance_exposure_m - self.lower_bound_exposure_m) / self.distance_exposure_m)


@dataclass
class _Label:
    node: Any
    seconds: float
    exposure: float
    parent: int | None
    arc_idx: int | None
    time_exposure: float = 0.0
    active: bool = True


def allowed_extra_time(fastest_seconds, max_extra_ratio=0.1, max_extra_seconds=120.0):
    """When both limits are supplied, use the stricter limit."""
    if not math.isfinite(fastest_seconds) or fastest_seconds < 0:
        raise ValueError("Invalid fastest travel time")
    limits = []
    for value, multiplier in ((max_extra_ratio, fastest_seconds), (max_extra_seconds, 1)):
        if value is not None:
            if not math.isfinite(value) or value < 0:
                raise ValueError("Extra-time limits must be finite and nonnegative")
            limits.append(value * multiplier)
    if not limits:
        raise ValueError("At least one extra-time limit is required")
    return min(limits)


class ExposureGraph:
    def __init__(self, arcs, turn_delay=None):
        if turn_delay is not None and not callable(turn_delay):
            raise ValueError("turn_delay must be callable or None")
        self.turn_delay = turn_delay
        self.arcs = list(arcs)
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        self.seconds, self.exposure, self.time_exposure = [], [], []
        for index, arc in enumerate(self.arcs):
            seconds, length, rs = float(arc.time_sec), float(arc.length_m), float(arc.rs_value)
            if not all(math.isfinite(v) for v in (seconds, length, rs)):
                raise ValueError("Non-finite edge time, length, or RS")
            if seconds <= 0 or length < 0 or not 0 <= rs <= 100:
                raise ValueError("Require positive edge time, nonnegative length, and RS in [0,100]")
            self.seconds.append(seconds)
            self.exposure.append(length * rs / 100)
            self.time_exposure.append(seconds * rs / 100)
            self.outgoing[arc.tail].append(index)
            self.incoming[arc.head].append(index)

    def _state(self, node, previous_index):
        # Arrival direction matters when the next transition has a turn cost.
        return (node, previous_index) if self.turn_delay is not None else node

    def _transition(self, previous_index, next_index):
        if self.turn_delay is None or previous_index is None:
            return 0.0, 0.0
        previous, following = self.arcs[previous_index], self.arcs[next_index]
        delay = float(self.turn_delay(previous, following))
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("Turn delay must be finite and nonnegative")
        return delay, delay*max(float(previous.rs_value), float(following.rs_value))/100

    def path_metrics(self, path):
        moving_seconds = sum(float(a.time_sec) for a in path)
        distance_exposure = sum(float(a.length_m)*float(a.rs_value)/100 for a in path)
        moving_time_exposure = sum(float(a.time_sec)*float(a.rs_value)/100 for a in path)
        delay_seconds = delay_exposure = 0.0
        if self.turn_delay is not None:
            for previous, following in zip(path, path[1:]):
                delay = float(self.turn_delay(previous, following))
                if not math.isfinite(delay) or delay < 0:
                    raise ValueError("Turn delay must be finite and nonnegative")
                delay_seconds += delay
                delay_exposure += delay*max(float(previous.rs_value), float(following.rs_value))/100
        return {"distance_exposure_m": distance_exposure, "distance_m": sum(float(a.length_m) for a in path),
                "moving_time_sec": moving_seconds, "turn_delay_sec": delay_seconds,
                "travel_time_sec": moving_seconds+delay_seconds,
                "moving_time_exposure_sec": moving_time_exposure,
                "turn_time_exposure_sec": delay_exposure,
                "time_exposure_sec": moving_time_exposure+delay_exposure}

    def shortest_path(self, start, end, metric="time"):
        """Lexicographic Dijkstra; tie-break shortest time by exposure and vice versa."""
        if metric not in {"time", "exposure"}:
            raise ValueError("metric must be time or exposure")
        primary, secondary = (self.seconds, self.exposure) if metric == "time" else (self.exposure, self.seconds)
        serial = itertools.count()
        initial = self._state(start, None)
        queue = [(0.0, 0.0, next(serial), initial)]
        distance = {initial: (0.0, 0.0)}
        previous = {}
        while queue:
            first, second, _, state = heapq.heappop(queue)
            if (first, second) != distance[state]:
                continue
            node, previous_index = state if self.turn_delay is not None else (state, None)
            if node == end:
                path = []
                while state != initial:
                    parent, index = previous[state]
                    path.append(self.arcs[index])
                    state = parent
                return list(reversed(path)), first
            for index in self.outgoing.get(node, []):
                target = self._state(self.arcs[index].head, index)
                delay, _ = self._transition(previous_index, index)
                candidate = (first+primary[index]+delay, second+secondary[index]) if metric == "time" else (first+primary[index], second+secondary[index]+delay)
                if candidate < distance.get(target, (math.inf, math.inf)):
                    distance[target] = candidate
                    previous[target] = state, index
                    heapq.heappush(queue, (*candidate, next(serial), target))
        raise ValueError(f"No directed path between {start} and {end}")

    def _remaining(self, end, costs):
        serial = itertools.count()
        queue, distance = [(0.0, next(serial), end)], {end: 0.0}
        while queue:
            value, _, node = heapq.heappop(queue)
            if value != distance[node]:
                continue
            for index in self.incoming.get(node, []):
                source = self.arcs[index].tail
                candidate = value + costs[index]
                if candidate < distance.get(source, math.inf):
                    distance[source] = candidate
                    heapq.heappush(queue, (candidate, next(serial), source))
        return distance

    def _remaining_turn(self, end, time_exposure=False):
        """Exact unconstrained suffix bound for each incoming-arc state."""
        costs = self.time_exposure if time_exposure else self.seconds
        serial = itertools.count()
        distance = {index: 0.0 for index in self.incoming.get(end, [])}
        queue = [(0.0, next(serial), index) for index in distance]
        heapq.heapify(queue)
        while queue:
            value, _, following_index = heapq.heappop(queue)
            if value != distance[following_index]:
                continue
            following = self.arcs[following_index]
            for previous_index in self.incoming.get(following.tail, []):
                delay, waiting_exposure = self._transition(previous_index, following_index)
                increment = waiting_exposure if time_exposure else delay
                candidate = value+costs[following_index]+increment
                if candidate < distance.get(previous_index, math.inf):
                    distance[previous_index] = candidate
                    heapq.heappush(queue, (candidate, next(serial), previous_index))
        return distance

    def solve(self, start, end, *, max_extra_ratio=0.1, max_extra_seconds=120.0,
              max_labels=1_000_000, max_runtime_sec=60.0, prevent_time_exposure_increase=False):
        if not isinstance(prevent_time_exposure_increase, bool):
            raise ValueError("prevent_time_exposure_increase must be a boolean")
        if isinstance(max_labels, bool) or not isinstance(max_labels, int) or max_labels < 1:
            raise ValueError("max_labels must be a positive integer")
        if not math.isfinite(max_runtime_sec) or max_runtime_sec <= 0:
            raise ValueError("max_runtime_sec must be positive and finite")
        began = time.monotonic()
        fastest, fastest_seconds = self.shortest_path(start, end, "time")
        budget = fastest_seconds + allowed_extra_time(fastest_seconds, max_extra_ratio, max_extra_seconds)
        best_path = fastest
        best_exposure = sum(float(a.length_m) * float(a.rs_value) / 100 for a in fastest)
        best_seconds = fastest_seconds
        best_time_exposure = self.path_metrics(fastest)["time_exposure_sec"]
        time_exposure_budget = best_time_exposure if prevent_time_exposure_increase else None
        remaining_time = self._remaining_turn(end) if self.turn_delay is not None else self._remaining(end, self.seconds)
        remaining_risk = self._remaining(end, self.exposure)
        remaining_time_exposure = (self._remaining_turn(end, time_exposure=True) if self.turn_delay is not None
                                  else self._remaining(end, self.time_exposure)) if prevent_time_exposure_increase else None
        labels = [_Label(start, 0.0, 0.0, None, None)]
        frontiers = defaultdict(list, {self._state(start, None): [0]})
        serial = itertools.count()
        queue = [(remaining_risk[start], 0.0, next(serial), 0)]
        expanded = 0
        stop_reason = "exhausted"
        pending_bound = best_exposure

        def recover(label_id):
            path = []
            while labels[label_id].parent is not None:
                label = labels[label_id]
                path.append(self.arcs[label.arc_idx])
                label_id = label.parent
            return list(reversed(path))

        while queue:
            bound, _, _, label_id = heapq.heappop(queue)
            label = labels[label_id]
            if not label.active:
                continue
            # No route below this bound can improve the feasible incumbent.
            if bound >= best_exposure - EPS:
                continue
            pending_bound = bound
            if time.monotonic() - began >= max_runtime_sec:
                stop_reason = "runtime_limit"
                break
            expanded += 1
            for index in self.outgoing.get(label.node, []):
                arc = self.arcs[index]
                delay, delay_exposure = self._transition(label.arc_idx, index)
                seconds = label.seconds + self.seconds[index] + delay
                suffix_key = index if self.turn_delay is not None else arc.head
                if seconds + remaining_time.get(suffix_key, math.inf) > budget + EPS:
                    continue
                time_exposure = label.time_exposure + self.time_exposure[index] + delay_exposure
                if prevent_time_exposure_increase and time_exposure + remaining_time_exposure.get(suffix_key, math.inf) > time_exposure_budget + EPS:
                    continue
                exposure = label.exposure + self.exposure[index]
                lower_bound = exposure + remaining_risk.get(arc.head, math.inf)
                if lower_bound >= best_exposure - EPS:
                    continue
                state = self._state(arc.head, index)
                frontier = frontiers[state]
                if any(labels[i].seconds <= seconds and labels[i].exposure <= exposure
                       and (not prevent_time_exposure_increase or labels[i].time_exposure <= time_exposure)
                       for i in frontier):
                    continue
                if len(labels) >= max_labels:
                    stop_reason = "label_limit"
                    break
                kept = []
                for i in frontier:
                    old = labels[i]
                    if seconds <= old.seconds and exposure <= old.exposure and (not prevent_time_exposure_increase or time_exposure <= old.time_exposure):
                        old.active = False
                    else:
                        kept.append(i)
                new_id = len(labels)
                labels.append(_Label(arc.head, seconds, exposure, label_id, index, time_exposure))
                frontiers[state] = kept + [new_id]
                if arc.head == end:
                    best_exposure, best_seconds = exposure, seconds
                    best_time_exposure = time_exposure
                    best_path = recover(new_id)
                else:
                    heapq.heappush(queue, (lower_bound, seconds, next(serial), new_id))
            if stop_reason == "label_limit":
                break
        limited = stop_reason in {"label_limit", "runtime_limit"}
        lower_bound = min(best_exposure, pending_bound, queue[0][0] if queue else math.inf) if limited else best_exposure
        return BudgetRoute(best_path, "feasible_limit" if limited else "optimal", stop_reason,
                           fastest_seconds, budget, best_seconds, best_exposure, lower_bound,
                           len(labels), expanded, time.monotonic() - began, best_time_exposure, time_exposure_budget, fastest)
