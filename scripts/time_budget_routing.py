"""Minimize additive distance exposure subject to a travel-time budget.

Uses nondominated (time, exposure) labels on the complete directed graph.
Reverse shortest-path bounds prune labels without discretizing time. A search
limit returns a feasible incumbent and a lower bound, never a false optimum.
Inputs must already encode allowed travel directions and road access.
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
    def __init__(self, arcs):
        self.arcs = list(arcs)
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        self.seconds, self.exposure = [], []
        for index, arc in enumerate(self.arcs):
            seconds, length, rs = float(arc.time_sec), float(arc.length_m), float(arc.rs_value)
            if not all(math.isfinite(v) for v in (seconds, length, rs)):
                raise ValueError("Non-finite edge time, length, or RS")
            if seconds <= 0 or length < 0 or not 0 <= rs <= 100:
                raise ValueError("Require positive edge time, nonnegative length, and RS in [0,100]")
            self.seconds.append(seconds)
            self.exposure.append(length * rs / 100)
            self.outgoing[arc.tail].append(index)
            self.incoming[arc.head].append(index)

    def shortest_path(self, start, end, metric="time"):
        """Lexicographic Dijkstra; tie-break shortest time by exposure and vice versa."""
        if metric not in {"time", "exposure"}:
            raise ValueError("metric must be time or exposure")
        primary, secondary = (self.seconds, self.exposure) if metric == "time" else (self.exposure, self.seconds)
        serial = itertools.count()
        queue = [(0.0, 0.0, next(serial), start)]
        distance = {start: (0.0, 0.0)}
        previous = {}
        while queue:
            first, second, _, node = heapq.heappop(queue)
            if (first, second) != distance[node]:
                continue
            if node == end:
                path = []
                while node != start:
                    index = previous[node]
                    path.append(self.arcs[index])
                    node = self.arcs[index].tail
                return list(reversed(path)), first
            for index in self.outgoing.get(node, []):
                target = self.arcs[index].head
                candidate = first + primary[index], second + secondary[index]
                if candidate < distance.get(target, (math.inf, math.inf)):
                    distance[target] = candidate
                    previous[target] = index
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

    def solve(self, start, end, *, max_extra_ratio=0.1, max_extra_seconds=120.0,
              max_labels=1_000_000, max_runtime_sec=60.0):
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
        remaining_time = self._remaining(end, self.seconds)
        remaining_risk = self._remaining(end, self.exposure)
        labels = [_Label(start, 0.0, 0.0, None, None)]
        frontiers = defaultdict(list, {start: [0]})
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
                seconds = label.seconds + self.seconds[index]
                if seconds + remaining_time.get(arc.head, math.inf) > budget + EPS:
                    continue
                exposure = label.exposure + self.exposure[index]
                lower_bound = exposure + remaining_risk.get(arc.head, math.inf)
                if lower_bound >= best_exposure - EPS:
                    continue
                frontier = frontiers[arc.head]
                if any(labels[i].seconds <= seconds and labels[i].exposure <= exposure for i in frontier):
                    continue
                if len(labels) >= max_labels:
                    stop_reason = "label_limit"
                    break
                kept = []
                for i in frontier:
                    old = labels[i]
                    if seconds <= old.seconds and exposure <= old.exposure:
                        old.active = False
                    else:
                        kept.append(i)
                new_id = len(labels)
                labels.append(_Label(arc.head, seconds, exposure, label_id, index))
                frontiers[arc.head] = kept + [new_id]
                if arc.head == end:
                    best_exposure, best_seconds = exposure, seconds
                    best_path = recover(new_id)
                else:
                    heapq.heappush(queue, (lower_bound, seconds, next(serial), new_id))
            if stop_reason == "label_limit":
                break
        limited = stop_reason in {"label_limit", "runtime_limit"}
        lower_bound = min(best_exposure, pending_bound, queue[0][0] if queue else math.inf) if limited else best_exposure
        return BudgetRoute(best_path, "feasible_limit" if limited else "optimal", stop_reason,
                           fastest_seconds, budget, best_seconds, best_exposure, lower_bound,
                           len(labels), expanded, time.monotonic() - began)
