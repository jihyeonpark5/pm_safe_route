from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from time_budget_routing import ExposureGraph, allowed_extra_time


def edge(tail, head, seconds, exposure, name=""):
    return SimpleNamespace(tail=tail, head=head, time_sec=seconds, length_m=100.,
                           rs_value=exposure, edge_id=name)


def tradeoff_graph():
    # The middle route is not optimal for any linear weighted-sum objective,
    # but is the best feasible route at a five-second budget.
    return ExposureGraph([edge("s", "t", 1, 10, "fast"),
                          edge("s", "t", 5, 6, "middle"),
                          edge("s", "t", 6, 0, "slow")])


def test_finds_constrained_optimum_not_reachable_by_weight_sweep():
    result = tradeoff_graph().solve("s", "t", max_extra_ratio=None, max_extra_seconds=4)
    assert result.status == "optimal"
    assert result.distance_exposure_m == 6
    assert result.travel_time_sec == result.time_budget_sec == 5
    assert result.path_arcs[0].edge_id == "middle"
    assert result.relative_gap == 0


def test_zero_budget_returns_fastest_and_larger_budget_reduces_exposure():
    graph = tradeoff_graph()
    values = [graph.solve("s", "t", max_extra_ratio=None, max_extra_seconds=s) for s in [0, 4, 5]]
    assert [v.distance_exposure_m for v in values] == [10, 6, 0]
    assert all(v.travel_time_sec <= v.time_budget_sec for v in values)


def test_both_limits_use_the_stricter_allowance():
    assert allowed_extra_time(1000, .1, 120) == 100
    assert allowed_extra_time(2000, .1, 120) == 120
    assert allowed_extra_time(1000, None, 120) == 120
    assert allowed_extra_time(2000, .1, None) == 200


def test_limit_returns_feasible_incumbent_with_honest_bound():
    result = tradeoff_graph().solve("s", "t", max_extra_ratio=None, max_extra_seconds=4, max_labels=1)
    assert result.status == "feasible_limit"
    assert result.stop_reason == "label_limit"
    assert result.travel_time_sec <= result.time_budget_sec
    assert result.lower_bound_exposure_m <= 6 <= result.distance_exposure_m
    assert result.relative_gap > 0


def test_runtime_limit_is_reported(monkeypatch):
    import time_budget_routing
    clock = iter([0., 2., 3.])
    monkeypatch.setattr(time_budget_routing.time, "monotonic", lambda: next(clock))
    result = tradeoff_graph().solve("s", "t", max_extra_seconds=4, max_extra_ratio=None, max_runtime_sec=1)
    assert result.status == "feasible_limit"
    assert result.stop_reason == "runtime_limit"
    assert result.travel_time_sec <= result.time_budget_sec


def test_direction_zero_risk_cycles_and_same_endpoint():
    graph = ExposureGraph([edge(0, 1, 1, 0), edge(1, 0, 1, 0), edge(1, 2, 1, 5)])
    assert graph.solve(0, 2).distance_exposure_m == 5
    assert graph.solve(0, 0).path_arcs == []
    with pytest.raises(ValueError, match="No directed path"):
        graph.solve(2, 0)


def enumerate_paths(arcs, start, end):
    def visit(node, visited, seconds, risk):
        if node == end:
            yield seconds, risk
            return
        for a in arcs:
            if a.tail == node and a.head not in visited:
                yield from visit(a.head, visited | {a.head}, seconds+a.time_sec, risk+a.length_m*a.rs_value/100)
    return list(visit(start, {start}, 0, 0))


@pytest.mark.parametrize("seed", range(12))
def test_matches_exhaustive_simple_path_search_on_cyclic_graphs(seed):
    rng = random.Random(seed)
    arcs = [edge(i, i+1, rng.randint(1, 8), rng.randint(0, 20)) for i in range(5)]
    arcs += [edge(i, j, rng.randint(1, 12), rng.randint(0, 25))
             for i in range(6) for j in range(6) if i != j and rng.random() < .25]
    paths = enumerate_paths(arcs, 0, 5)
    fastest = min(t for t, _ in paths)
    graph = ExposureGraph(arcs)
    for allowance in [0, 3, 10]:
        result = graph.solve(0, 5, max_extra_ratio=None, max_extra_seconds=allowance)
        expected = min(risk for t, risk in paths if t <= fastest+allowance)
        assert result.status == "optimal"
        assert result.distance_exposure_m == pytest.approx(expected)
        assert result.travel_time_sec <= result.time_budget_sec + 1e-9
        assert sum(a.time_sec for a in result.path_arcs) == pytest.approx(result.travel_time_sec)


def test_same_road_segmentation_does_not_change_optimized_exposure():
    direct = ExposureGraph([edge(0, 2, 2, 8), edge(0, 2, 1, 10)])
    first, second = edge(0, 1, 1, 8), edge(1, 2, 1, 8)
    first.length_m = second.length_m = 50
    split = ExposureGraph([first, second, edge(0, 2, 1, 10)])
    results = [g.solve(0, 2, max_extra_ratio=None, max_extra_seconds=1) for g in (direct, split)]
    assert [r.distance_exposure_m for r in results] == [8, 8]
    assert [r.travel_time_sec for r in results] == [2, 2]


@pytest.mark.parametrize("kwargs", [{"max_extra_ratio": -1}, {"max_extra_seconds": float("nan")},
                                   {"max_labels": 0}, {"max_runtime_sec": 0},
                                   {"max_extra_ratio": None, "max_extra_seconds": None}])
def test_invalid_solver_settings_fail(kwargs):
    with pytest.raises(ValueError):
        tradeoff_graph().solve("s", "t", **kwargs)


@pytest.mark.parametrize("seconds,length,score", [(0, 1, 10), (1, -1, 10), (1, 1, 101), (1, 1, float("nan"))])
def test_invalid_edge_data_fails(seconds, length, score):
    a = edge(0, 1, seconds, score)
    a.length_m = length
    with pytest.raises(ValueError):
        ExposureGraph([a])
