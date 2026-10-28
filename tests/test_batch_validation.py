import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("batch_validation", SCRIPTS / "17_batch_validate_routes.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def test_sampling_balanced_reproducible_and_independent_of_input_order():
    nodes = gpd.GeoDataFrame({"osmid": np.arange(80)}, geometry=[Point(i*100, 0) for i in range(80)], crs=5179)
    first, meta = batch.sample_pairs(nodes, set(nodes.osmid), per_band=3, seed=42)
    again, _ = batch.sample_pairs(nodes.sample(frac=1, random_state=7), set(nodes.osmid), per_band=3, seed=42)
    pd.testing.assert_frame_equal(first, again)
    assert meta["counts"] == {"short": 3, "medium": 3, "long": 3}
    assert not first[["start_node", "end_node"]].duplicated().any()
    assert first.start_node.ne(first.end_node).all()
    for band, lower, upper in batch.BANDS:
        distances = first.loc[first.band.eq(band), "straight_distance_m"]
        assert distances.ge(lower).all() and distances.lt(upper).all()


def test_largest_strong_component_excludes_one_way_sources_and_sinks():
    arcs = [SimpleNamespace(tail=a, head=b) for a, b in [(0, 1), (1, 2), (2, 1), (2, 3)]]
    selected, meta = batch.largest_strong_component(arcs)
    assert selected == {1, 2}
    assert meta["graph_nodes"] == 4
    assert meta["eligible_component_nodes"] == 2


def test_summary_retains_errors_limits_and_undefined_denominators():
    rows = pd.DataFrame([
        dict(hour="18h", scenario="S1", method="budget_distance", status="ok", solver_status="feasible_limit",
             within_time_budget=True, distance_exposure_m_reduction_vs_fastest=.1,
             time_exposure_sec_reduction_vs_fastest=-.02, extra_time_sec_vs_fastest=5., solve_elapsed_sec=1., relative_gap=.3),
        dict(hour="18h", scenario="S1", method="budget_distance", status="ok", solver_status="optimal",
             within_time_budget=True, distance_exposure_m_reduction_vs_fastest=np.nan,
             time_exposure_sec_reduction_vs_fastest=np.nan, extra_time_sec_vs_fastest=0., solve_elapsed_sec=.1, relative_gap=0.),
        dict(hour="18h", scenario="S1", method="budget_distance", status="error", error="test failure"),
    ])
    result = batch.summarize_results(rows).iloc[0]
    assert result.scheduled == 3 and result.successful == 2 and result.errors == 1
    assert result.optimal == result.feasible_limit == 1
    assert result.undefined_distance_ratios == result.undefined_time_ratios == 1
    assert result.time_exposure_worsened == 1
    assert result.distance_reduction_mean == .1
    assert result.relative_gap_max == .3


def test_all_error_condition_can_still_be_summarized():
    rows = pd.DataFrame([dict(hour="10h", scenario="S0", method="budget_distance", status="error", error="test")])
    result = batch.summarize_results(rows).iloc[0]
    assert result.scheduled == result.errors == 1
    assert result.successful == 0 and np.isnan(result.distance_reduction_mean)


def test_solver_failure_does_not_hide_other_methods(monkeypatch):
    arc = SimpleNamespace(tail=1, head=2, time_sec=10., length_m=100., rs_value=30., risk_level="normal", edge_id="E1")
    class Graph:
        def shortest_path(self, *args):
            return [arc], 10.
        def solve(self, *args, **kwargs):
            raise RuntimeError("injected failure")
    monkeypatch.setattr(batch.route, "build_subgraph_arcs", lambda *args: ([arc], {}))
    monkeypatch.setattr(batch.route, "optimize_route", lambda *args: SimpleNamespace(path_valid=True, path_arcs=[arc], optimization_status="Optimal"))
    rows = batch.evaluate_pair([arc], Graph(), 1, 2, dict(max_extra_ratio=.1, max_extra_seconds=120.), 10, 1.)
    statuses = {r["method"]: r["status"] for r in rows}
    assert statuses == {"fastest": "ok", "budget_distance": "error", "legacy_lambda_0.5": "ok", "legacy_lambda_1": "ok"}
    assert "injected failure" in next(r["error"] for r in rows if r["method"] == "budget_distance")
