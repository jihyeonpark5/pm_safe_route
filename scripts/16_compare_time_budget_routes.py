"""Compare full-graph time-budget routing with fastest and legacy lambda routes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

from route_metrics import exposure_metrics
from time_budget_routing import ExposureGraph, allowed_extra_time
from turn_delay import TurnDelayProfile, valid_walk

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report_cases", Path(__file__).with_name("15_compare_report_routes.py"))
report = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = report
spec.loader.exec_module(report)
route = report.route


def budget_options(mode, ratio, seconds):
    return dict(max_extra_ratio=None if mode == "seconds" else ratio,
                max_extra_seconds=None if mode == "ratio" else seconds)


def compare_request(edges, nodes, request, config, options, max_labels, max_runtime, prevent_time_exposure_increase=False, turn_seconds_at_90=0.):
    start, start_snap, _ = route.nearest_node(nodes, request.start_lon_lat)
    end, end_snap, _ = route.nearest_node(nodes, request.end_lon_lat)
    arcs = route.build_arcs(route.prepare_edge_costs(edges, request, config.scenario))
    graph = ExposureGraph(arcs, None if turn_seconds_at_90 == 0 else TurnDelayProfile(turn_seconds_at_90))
    fastest, fastest_time = graph.shortest_path(start, end, "time")
    fastest_metrics = exposure_metrics(fastest)
    fastest_metrics.update(graph.path_metrics(fastest))
    budget_result = graph.solve(start, end, **options, max_labels=max_labels, max_runtime_sec=max_runtime,
                                prevent_time_exposure_increase=prevent_time_exposure_increase)
    if start == end:
        legacy, legacy_diagnostics = {}, {"skipped": "same snapped endpoint"}
    else:
        legacy, _, legacy_diagnostics = route.optimize_request(edges, nodes, request, config)
    candidates = [("fastest", fastest, "optimal", None)]
    candidates += [(f"legacy_lambda_{value:g}", result.path_arcs, result.optimization_status, value)
                   for value, result in legacy.items()]
    budget_method = "budget_joint" if prevent_time_exposure_increase else "budget_distance"
    if turn_seconds_at_90:
        budget_method += "_turn"
    candidates.append((budget_method, budget_result.path_arcs, budget_result.status, None))
    rows, segments, features = [], [], []
    for method, path, status, lambda_value in candidates:
        valid = valid_walk(path, start, end) if turn_seconds_at_90 else ((start == end and not path) or route.validate_solution_path(path, start, end)[0])
        if not valid:
            raise RuntimeError(f"Invalid {method} route for {request.case_id}")
        metrics = exposure_metrics(path)
        metrics.update(graph.path_metrics(path))
        seconds = metrics.pop("travel_time_sec")
        metrics.pop("distance_m")
        angles = [route.angle_difference_degrees(a.end_heading_deg, b.start_heading_deg)
                  for a, b in zip(path, path[1:])]
        row = dict(case_id=request.case_id, start=request.start_label, end=request.end_label,
                   hour=request.hour, scenario=config.scenario, method=method, lambda_value=lambda_value,
                   solver_status=status, distance_m=sum(a.length_m for a in path), travel_time_sec=seconds,
                   time_budget_sec=budget_result.time_budget_sec, within_time_budget=seconds <= budget_result.time_budget_sec + 1e-9,
                   time_exposure_reference_sec=fastest_metrics["time_exposure_sec"],
                   within_time_exposure_reference=metrics["time_exposure_sec"] <= fastest_metrics["time_exposure_sec"]+1e-9,
                   extra_time_sec_vs_fastest=seconds-fastest_time,
                   extra_time_ratio_vs_fastest=seconds/fastest_time-1 if fastest_time else 0.0,
                   start_snap_m=start_snap, end_snap_m=end_snap, path_valid=valid,
                   turns_ge_90deg=sum(angle >= 90 for angle in angles),
                   turns_ge_150deg=sum(angle >= 150 for angle in angles),
                   optimality_scope="legacy_subgraph" if method.startswith("legacy") else "full_graph",
                   path_signature=route.path_signature(path), **metrics)
        for name in ("distance_exposure_m", "time_exposure_sec", "high_risk_distance_m"):
            baseline = fastest_metrics[name]
            row[name + "_reduction_vs_fastest"] = 1 - metrics[name]/baseline if baseline > 0 else float("nan")
        if method == budget_method:
            row.update(lower_bound_exposure_m=budget_result.lower_bound_exposure_m,
                       relative_gap=budget_result.relative_gap, stop_reason=budget_result.stop_reason,
                       labels_generated=budget_result.labels_generated, labels_expanded=budget_result.labels_expanded,
                       search_elapsed_sec=budget_result.elapsed_sec)
            if not row["within_time_budget"] or metrics["distance_exposure_m"] > fastest_metrics["distance_exposure_m"] + 1e-7:
                raise AssertionError("Budget result violates feasibility or baseline exposure bound")
            if prevent_time_exposure_increase and not row["within_time_exposure_reference"]:
                raise AssertionError("Joint result violates the time-exposure constraint")
        rows.append(row)
        for order, arc in enumerate(path):
            segments.append(dict(case_id=request.case_id, method=method, order=order, edge_id=arc.edge_id,
                                 tail=arc.tail, head=arc.head, length_m=arc.length_m, time_sec=arc.time_sec,
                                 rs=arc.rs_value, risk_level=arc.risk_level))
        # GeoJSON visualization is generated separately; coordinates below use EPSG:4326.
        if path:
            geoms = route.arcs_to_gdf(path, route.TARGET_CRS).to_crs(4326)
            features.append({"type": "Feature", "properties": {"case_id": request.case_id, "method": method},
                             "geometry": {"type": "MultiLineString", "coordinates": [list(g.coords) for g in geoms.geometry]}})
    diagnostics = dict(case_id=request.case_id, start_node=int(start), end_node=int(end),
                       full_graph_arcs=len(arcs), legacy_diagnostics=legacy_diagnostics)
    return rows, segments, features, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-mode", choices=["stricter", "ratio", "seconds"], default="stricter")
    parser.add_argument("--max-extra-ratio", type=float, default=.1)
    parser.add_argument("--max-extra-seconds", type=float, default=120)
    parser.add_argument("--max-labels", type=int, default=1_000_000)
    parser.add_argument("--max-runtime-sec", type=float, default=60)
    parser.add_argument("--prevent-time-exposure-increase", action="store_true")
    parser.add_argument("--turn-seconds-at-90", type=float, default=0., help="Uncalibrated turn-delay assumption; 0 disables it")
    parser.add_argument("--scenario", choices=["S0", "S1", "S2"], default="S1")
    parser.add_argument("--hour", choices=["10h", "18h", "22h"], default="18h")
    parser.add_argument("--case-id", default="all")
    parser.add_argument("--start-coords", default="")
    parser.add_argument("--end-coords", default="")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/output/validation_v2")
    args = parser.parse_args()
    TurnDelayProfile(args.turn_seconds_at_90)
    options = budget_options(args.budget_mode, args.max_extra_ratio, args.max_extra_seconds)
    allowed_extra_time(1, **options)  # Fail on invalid budgets before loading data.
    if bool(args.start_coords) != bool(args.end_coords):
        parser.error("Provide both --start-coords and --end-coords")
    if args.start_coords:
        requests = [route.RouteRequest("custom", "custom", "Custom coordinates", args.start_coords, args.end_coords,
                                      route.parse_lon_lat(args.start_coords), route.parse_lon_lat(args.end_coords), args.hour)]
    else:
        requests = [route.RouteRequest(**{**asdict(c), "hour": args.hour}) for c in report.CASES
                    if args.case_id == "all" or c.case_id == args.case_id]
        if not requests:
            parser.error("Unknown --case-id; use report1, report2, report3 or all")
    config = route.RunConfig(args.scenario, [0., .5, 1.], requests, route.DEFAULT_SUBGRAPH_MODE, True, "validation_v2")
    edges, nodes = route.load_inputs()
    all_rows, all_segments, all_features, all_diagnostics = [], [], [], []
    for request in requests:
        rows, segments, features, diagnostics = compare_request(edges, nodes, request, config, options, args.max_labels, args.max_runtime_sec,
                                                               args.prevent_time_exposure_increase, args.turn_seconds_at_90)
        all_rows.extend(rows)
        all_segments.extend(segments)
        all_features.extend(features)
        all_diagnostics.append(diagnostics)
        candidate = rows[-1]
        print(f"{request.case_id}: {candidate['solver_status']}, "
              f"time={candidate['travel_time_sec']/60:.2f}min, "
              f"distance exposure reduction={candidate['distance_exposure_m_reduction_vs_fastest']:.1%}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(all_rows)
    summary.to_csv(args.output_dir / "route_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_segments).to_csv(args.output_dir / "route_segments.csv", index=False)
    (args.output_dir / "routes.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": all_features}), encoding="utf-8")
    metadata = {
        "source_base_commit": "5a2f213b8a9a69ed96446ad33a6b63eca3e5a5a7",
        "objective": "minimize sum(RS/100 * edge_length_m) subject to total edge travel time <= budget",
        "prevent_time_exposure_increase": args.prevent_time_exposure_increase,
        "turn_seconds_at_90": args.turn_seconds_at_90,
        "turn_exposure_model": "delay * max(incoming RS, outgoing RS)/100",
        "budget_options": options, "budget_mode": args.budget_mode, "max_labels": args.max_labels,
        "max_runtime_sec": args.max_runtime_sec, "scenario": args.scenario, "hour": args.hour,
        "requests": [asdict(c) for c in requests], "diagnostics": all_diagnostics,
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (route.EDGE_RISK_PATH, route.NODES_PATH)},
        "limits": ["Model-based exposure reduction, not measured crash reduction",
                   "Travel times include configured turn-delay assumptions; signals, acceleration and access legs excluded",
                   "New solver uses full directed graph and optional turn delay, without legacy comfort cost",
                   "Existing edge scores and access filtering inherited without model change",
                   "Case 1 uses repository Nonhyeon coordinates despite presentation Sinnonhyeon label",
                   "Optimality tolerance 1e-9 exposure units; feasible_limit is not certified optimal",
                   "Runtime limit checked during label search after shortest-path bounds are prepared"],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = ["case_id", "method", "travel_time_sec", "within_time_budget", "distance_exposure_m_reduction_vs_fastest", "time_exposure_sec_reduction_vs_fastest", "high_risk_distance_m", "solver_status"]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
