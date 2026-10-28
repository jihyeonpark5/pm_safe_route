from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from time_budget_routing import ExposureGraph


def arc(a, b, seconds, distance_exposure, time_exposure, name=""):
    score = 100*time_exposure/seconds
    length = distance_exposure*100/score if score else 100.
    return SimpleNamespace(tail=a, head=b, time_sec=seconds, length_m=length, rs_value=score, edge_id=name)


def test_three_resource_dominance_keeps_a_necessary_time_exposure_label():
    arcs = [arc("s", "t", 1, 10, .5, "baseline"),
            arc("s", "x", 2, 2, .45, "two_resource_dominator"),
            arc("s", "x", 3, 3, .1, "needed_label"),
            arc("x", "t", 1, 1, .2, "short_suffix"),
            arc("x", "t", 1, 20, .01, "low_time_exposure_suffix")]
    graph = ExposureGraph(arcs)
    old = graph.solve("s", "t", max_extra_ratio=None, max_extra_seconds=5)
    new = graph.solve("s", "t", max_extra_ratio=None, max_extra_seconds=5, prevent_time_exposure_increase=True)
    assert old.distance_exposure_m == pytest.approx(3)
    assert old.time_exposure_sec > .5
    assert new.status == "optimal" and new.distance_exposure_m == pytest.approx(4)
    assert new.time_exposure_sec == pytest.approx(.3)
    assert new.max_time_exposure_sec == pytest.approx(.5)
    assert new.path_arcs[0].edge_id == "needed_label"


def test_returns_baseline_when_all_distance_improvements_worsen_time_exposure():
    graph = ExposureGraph([arc(0, 1, 1, 10, .1), arc(0, 1, 2, 5, .2)])
    result = graph.solve(0, 1, max_extra_ratio=None, max_extra_seconds=2, prevent_time_exposure_increase=True)
    assert result.distance_exposure_m == pytest.approx(10)
    assert result.time_exposure_sec == pytest.approx(.1)
    assert result.status == "optimal"


@pytest.mark.parametrize("seed", range(12))
def test_joint_optimum_matches_exhaustive_simple_paths(seed):
    rng = random.Random(seed)
    arcs = []
    for i in range(6):
        for j in range(6):
            if i != j and (j == i+1 or rng.random() < .25):
                arcs.append(SimpleNamespace(tail=i, head=j, time_sec=float(rng.randint(1, 10)),
                                           length_m=float(rng.randint(1, 100)), rs_value=float(rng.randint(0, 100))))
    graph = ExposureGraph(arcs)
    baseline, fastest = graph.shortest_path(0, 5)
    time_cap = sum(a.time_sec*a.rs_value/100 for a in baseline)
    paths = []
    def visit(node, seen, seconds, distance_exposure, time_exposure):
        if node == 5:
            paths.append((seconds, distance_exposure, time_exposure))
            return
        for a in arcs:
            if a.tail == node and a.head not in seen:
                visit(a.head, seen | {a.head}, seconds+a.time_sec,
                      distance_exposure+a.length_m*a.rs_value/100, time_exposure+a.time_sec*a.rs_value/100)
    visit(0, {0}, 0, 0, 0)
    for allowance in [0, 5, 15]:
        result = graph.solve(0, 5, max_extra_ratio=None, max_extra_seconds=allowance, prevent_time_exposure_increase=True)
        expected = min(d for t, d, e in paths if t <= fastest+allowance+1e-9 and e <= time_cap+1e-9)
        assert result.status == "optimal"
        assert result.distance_exposure_m == pytest.approx(expected)
        assert result.time_exposure_sec <= time_cap+1e-9
        assert result.travel_time_sec <= result.time_budget_sec+1e-9


def test_joint_limit_incumbent_is_feasible_and_zero_exposure_is_supported():
    graph = ExposureGraph([arc(0, 1, 1, 10, .5), arc(0, 1, 2, 5, .2)])
    result = graph.solve(0, 1, max_extra_ratio=None, max_extra_seconds=2,
                         max_labels=1, prevent_time_exposure_increase=True)
    assert result.status == "feasible_limit"
    assert result.lower_bound_exposure_m <= 5 <= result.distance_exposure_m
    assert result.time_exposure_sec <= result.max_time_exposure_sec+1e-9
    zero = ExposureGraph([arc(0, 1, 1, 0, 0)]).solve(0, 1, prevent_time_exposure_increase=True)
    assert zero.time_exposure_sec == zero.max_time_exposure_sec == 0
    assert zero.relative_gap == 0
    same = graph.solve(0, 0, prevent_time_exposure_increase=True)
    assert same.path_arcs == [] and same.time_exposure_sec == 0


def test_joint_flag_requires_boolean():
    with pytest.raises(ValueError):
        ExposureGraph([]).solve(0, 0, prevent_time_exposure_increase="yes")


def test_joint_constraint_allows_equality_and_is_segmentation_invariant():
    baseline = arc(0, 2, 1, 10, .1)
    alternative = arc(0, 2, 2, 5, .1)
    whole = ExposureGraph([baseline, alternative])
    split = ExposureGraph([baseline, arc(0, 1, 1, 2.5, .05), arc(1, 2, 1, 2.5, .05)])
    results = [g.solve(0, 2, max_extra_ratio=None, max_extra_seconds=1, prevent_time_exposure_increase=True)
               for g in [whole, split]]
    assert all(r.distance_exposure_m == pytest.approx(5) for r in results)
    assert all(r.time_exposure_sec == pytest.approx(.1) for r in results)
    assert all(r.status == "optimal" for r in results)
