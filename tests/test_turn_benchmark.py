import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("turn_benchmark", SCRIPTS / "20_validate_turn_delay.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def fixture():
    def arc(a, b, seconds, length, name, end=0):
        return SimpleNamespace(tail=a, head=b, time_sec=seconds, length_m=length, rs_value=10.,
                               edge_id=name, risk_level="normal", start_heading_deg=0., end_heading_deg=end)
    arcs = [arc(0, 1, 1, 10, "A", end=90), arc(1, 2, 1, 10, "B"), arc(0, 2, 3, 30, "C")]
    graphs = {0.: benchmark.ExposureGraph(arcs), 3.: benchmark.ExposureGraph(arcs, benchmark.TurnDelayProfile(3))}
    old = graphs[0.].solve(0, 2, prevent_time_exposure_increase=True)
    references = {"ODtest": {"solver_status": old.status, "distance_exposure_m": old.distance_exposure_m,
                             "path_signature": benchmark.route.path_signature(old.path_arcs)}}
    pair = dict(pair_id="ODtest", start_node=0, end_node=2, band="short")
    benchmark.initialize_worker(graphs, dict(max_extra_ratio=.1, max_extra_seconds=120.), 1000, 5., references)
    return pair


def test_end_to_end_profile_comparison_uses_adjusted_baseline_and_records_old_violations():
    _, rows = benchmark.evaluate_pair(fixture())
    assert len(rows) == 6 and all(row["status"] == "ok" for row in rows)
    lookup = {(r["turn_seconds_at_90"], r["method"]): r for r in rows}
    assert lookup[0., "joint_turn"]["v4_path_match"]
    new = lookup[3., "joint_turn"]
    assert new["within_time_budget"] and new["within_time_exposure_reference"]
    assert new["travel_time_sec"] == 3
    old = lookup[3., "v4_joint_replayed"]
    assert old["travel_time_sec"] == 5
    assert not old["within_time_budget"] and not old["within_time_exposure_reference"]
    assert old["moving_time_sec"] == 2 and old["turn_delay_sec"] == 3


def test_profile_summary_keeps_profiles_separate():
    _, rows = benchmark.evaluate_pair(fixture())
    data = pd.DataFrame([{**r, "hour": "18h", "scenario": "S1"} for r in rows])
    summary = benchmark.summarize(data)
    assert len(summary) == 6
    old = summary[(summary.turn_seconds_at_90 == 3) & summary.method.eq("v4_joint_replayed")].iloc[0]
    assert old.time_cap_violations == old.time_exposure_cap_violations == 1
    assert summary.scheduled.eq(1).all()
