from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from time_budget_routing import ExposureGraph
from turn_delay import TurnDelayProfile, valid_walk


def arc(a, b, seconds=1., length=10., rs=10., start=0., end=0.):
    return SimpleNamespace(tail=a, head=b, time_sec=seconds, length_m=length, rs_value=rs,
                           start_heading_deg=start, end_heading_deg=end)


def test_incoming_arc_is_part_of_the_shortest_path_state():
    arcs = [arc("s", "x", end=90), arc("s", "b"), arc("b", "x"), arc("x", "t")]
    graph = ExposureGraph(arcs, TurnDelayProfile(3))
    path, seconds = graph.shortest_path("s", "t")
    assert [a.head for a in path] == ["b", "x", "t"]
    assert seconds == 3


def test_delay_and_waiting_exposure_are_counted_consistently():
    arcs = [arc(0, 1, length=20, rs=20, end=90), arc(1, 2, rs=60)]
    graph = ExposureGraph(arcs, TurnDelayProfile(3))
    metrics = graph.path_metrics(arcs)
    assert metrics["moving_time_sec"] == 2
    assert metrics["turn_delay_sec"] == 3
    assert metrics["travel_time_sec"] == 5
    assert metrics["distance_exposure_m"] == 10
    assert metrics["moving_time_exposure_sec"] == pytest.approx(.8)
    assert metrics["turn_time_exposure_sec"] == pytest.approx(1.8)
    result = graph.solve(0, 2, prevent_time_exposure_increase=True)
    assert result.time_exposure_sec == pytest.approx(2.6)
    assert result.max_time_exposure_sec == pytest.approx(2.6)
    assert result.fastest_time_sec == 5
    assert result.fastest_path_arcs == arcs


def test_valid_state_path_can_revisit_a_physical_node():
    arcs = [arc("s", "x", end=90), arc("x", "t"),
            arc("x", "y", seconds=.1, start=90, end=180),
            arc("y", "x", seconds=.1, start=180, end=0)]
    path, seconds = ExposureGraph(arcs, TurnDelayProfile(3)).shortest_path("s", "t")
    assert seconds == pytest.approx(2.2)
    assert [a.head for a in path] == ["x", "y", "x", "t"]
    assert valid_walk(path, "s", "t")
    assert not valid_walk(path, "y", "t")


@pytest.mark.parametrize("seed", range(8))
def test_turn_aware_constrained_optimum_matches_arc_state_enumeration(seed):
    rng = random.Random(seed)
    arcs = [arc(i, i+1, seconds=rng.randint(1, 5), length=rng.randint(1, 30), rs=rng.randint(0, 100),
                start=rng.choice([0, 90, 180]), end=rng.choice([0, 90, 180])) for i in range(3)]
    arcs += [arc(0, 2, seconds=3, rs=40, end=90), arc(1, 0, seconds=1, rs=10), arc(2, 1, seconds=2, rs=20)]
    graph = ExposureGraph(arcs, TurnDelayProfile(3))
    walks = []
    def visit(node, used_states, path):
        if node == 3:
            walks.append(graph.path_metrics(path))
            return
        for index, edge in enumerate(arcs):
            if edge.tail == node and index not in used_states:
                visit(edge.head, used_states | {index}, path+[edge])
    visit(0, set(), [])
    baseline, fastest = graph.shortest_path(0, 3)
    assert fastest == pytest.approx(min(p["travel_time_sec"] for p in walks))
    time_exposure_cap = graph.path_metrics(baseline)["time_exposure_sec"]
    for guard in [False, True]:
        result = graph.solve(0, 3, max_extra_ratio=None, max_extra_seconds=5, prevent_time_exposure_increase=guard)
        feasible = [p for p in walks if p["travel_time_sec"] <= fastest+5+1e-9
                    and (not guard or p["time_exposure_sec"] <= time_exposure_cap+1e-9)]
        assert result.status == "optimal"
        assert result.distance_exposure_m == pytest.approx(min(p["distance_exposure_m"] for p in feasible))
        assert result.travel_time_sec == pytest.approx(graph.path_metrics(result.path_arcs)["travel_time_sec"])
        assert valid_walk(result.path_arcs, 0, 3)


def test_turn_profile_boundaries_and_invalid_values():
    profile = TurnDelayProfile(3)
    first = arc(0, 1)
    for angle, expected in [(0, 0), (30, 0), (90, 3), (150, 6), (180, 6), (270, 3)]:
        assert profile(first, arc(1, 2, start=angle)) == expected
    with pytest.raises(ValueError):
        TurnDelayProfile(-1)
    with pytest.raises(ValueError):
        profile(first, arc(1, 2, start=float("nan")))
    with pytest.raises(ValueError):
        ExposureGraph([arc(0, 1), arc(1, 2)], lambda *args: -1).shortest_path(0, 2)


def test_turn_aware_limit_keeps_a_feasible_incumbent():
    graph = ExposureGraph([arc(0, 1, length=100, rs=50), arc(0, 1, seconds=2, length=20, rs=10)], TurnDelayProfile(3))
    result = graph.solve(0, 1, max_extra_ratio=None, max_extra_seconds=2, max_labels=1,
                         prevent_time_exposure_increase=True)
    assert result.status == "feasible_limit"
    assert result.travel_time_sec <= result.time_budget_sec+1e-9
    assert result.time_exposure_sec <= result.max_time_exposure_sec+1e-9
    assert result.relative_gap > 0


def test_reverse_state_bounds_include_turns_and_can_use_a_loop():
    arcs = [arc("s", "x", end=90), arc("x", "t"),
            arc("x", "y", seconds=.1, start=90, end=180),
            arc("y", "x", seconds=.1, start=180, end=0)]
    graph = ExposureGraph(arcs, TurnDelayProfile(3))
    assert graph._remaining_turn("t")[0] == pytest.approx(1.2)
    assert graph._remaining_turn("t", time_exposure=True)[0] == pytest.approx(.12)
