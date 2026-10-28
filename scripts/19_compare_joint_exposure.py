"""Replay fixed v3 OD pairs and compare distance-only versus joint constraints."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from route_metrics import exposure_metrics
from time_budget_routing import ExposureGraph

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("batch_reference", Path(__file__).with_name("17_batch_validate_routes.py"))
batch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = batch
spec.loader.exec_module(batch)
route = batch.route
METHODS = ["fastest", "budget_distance", "budget_joint"]


def compare_pair(graph, start, end, options, max_labels, max_runtime, reference):
    began = time.monotonic()
    fastest, fastest_seconds = graph.shortest_path(start, end)
    fastest_elapsed = time.monotonic()-began
    baseline = exposure_metrics(fastest)
    rows = []
    solutions = {}

    def record(method, path, elapsed, result=None):
        if not route.validate_solution_path(path, start, end)[0]:
            raise AssertionError("Invalid directed path")
        metrics = exposure_metrics(path)
        seconds = sum(float(a.time_sec) for a in path)
        time_budget = result.time_budget_sec if result else fastest_seconds+batch.allowed_extra_time(fastest_seconds, **options)
        row = dict(method=method, status="ok", error="", solver_status=result.status if result else "optimal",
                   travel_time_sec=seconds, distance_m=sum(a.length_m for a in path),
                   time_budget_sec=time_budget, within_time_budget=seconds <= time_budget+1e-9,
                   time_exposure_reference_sec=baseline["time_exposure_sec"],
                   within_time_exposure_reference=metrics["time_exposure_sec"] <= baseline["time_exposure_sec"]+1e-9,
                   extra_time_sec_vs_fastest=seconds-fastest_seconds,
                   solve_elapsed_sec=elapsed, relative_gap=result.relative_gap if result else 0.,
                   labels_generated=result.labels_generated if result else 0,
                   stop_reason=result.stop_reason if result else "dijkstra",
                   lower_bound_exposure_m=result.lower_bound_exposure_m if result else metrics["distance_exposure_m"],
                   path_signature=route.path_signature(path), **metrics)
        for name in ["distance_exposure_m", "time_exposure_sec", "high_risk_distance_m"]:
            row[name+"_reduction_vs_fastest"] = 1-metrics[name]/baseline[name] if baseline[name] > 0 else float("nan")
        if not row["within_time_budget"] or metrics["distance_exposure_m"] > baseline["distance_exposure_m"]+1e-7:
            raise AssertionError("Travel-time or distance-exposure baseline bound violated")
        if method == "budget_joint" and not row["within_time_exposure_reference"]:
            raise AssertionError("Time-exposure reference bound violated")
        if method in reference:
            old = reference[method]
            row["v3_path_match"] = row["path_signature"] == old["path_signature"]
            if row["solver_status"] == old["solver_status"] == "optimal":
                if not np.isclose(metrics["distance_exposure_m"], old["distance_exposure_m"], rtol=0, atol=1e-7):
                    raise AssertionError("Replayed v3 optimum changed")
        rows.append(row)
        solutions[method] = row

    record("fastest", fastest, fastest_elapsed)
    for method, guard in [("budget_distance", False), ("budget_joint", True)]:
        try:
            result = graph.solve(start, end, **options, max_labels=max_labels, max_runtime_sec=max_runtime,
                                 prevent_time_exposure_increase=guard)
            record(method, result.path_arcs, result.elapsed_sec, result)
        except Exception as exc:
            rows.append(dict(method=method, status="error", error=f"{type(exc).__name__}: {exc}"))
    if all(name in solutions and solutions[name]["solver_status"] == "optimal" for name in ["budget_distance", "budget_joint"]):
        if solutions["budget_joint"]["distance_exposure_m"] < solutions["budget_distance"]["distance_exposure_m"]-1e-7:
            raise AssertionError("Adding a constraint cannot improve the certified optimum")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "data/output/validation_v3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/output/validation_v4")
    parser.add_argument("--max-labels", type=int, default=100000)
    parser.add_argument("--max-runtime-sec", type=float, default=5.)
    args = parser.parse_args()
    if args.reference_dir.resolve() == args.output_dir.resolve():
        parser.error("Output must differ from reference directory")
    if args.max_labels < 1 or not np.isfinite(args.max_runtime_sec) or args.max_runtime_sec <= 0:
        parser.error("Positive limits required")
    meta_path = args.reference_dir / "metadata.json"
    pairs_path = args.reference_dir / "sampled_pairs.csv"
    previous_path = args.reference_dir / "results.csv"
    source = json.loads(meta_path.read_text())
    for p in [route.EDGE_RISK_PATH, route.NODES_PATH]:
        if hashlib.sha256(p.read_bytes()).hexdigest() != source["input_sha256"][p.name]:
            raise ValueError(f"Input changed since v3: {p.name}")
    pairs = pd.read_csv(pairs_path)
    previous = pd.read_csv(previous_path)
    keys = ["pair_id", "hour", "scenario", "method"]
    if pairs.pair_id.duplicated().any() or previous.duplicated(keys).any():
        raise ValueError("Duplicate reference keys")
    expected = len(pairs)*len(source["hours"])*len(source["scenarios"])
    for method in ["fastest", "budget_distance"]:
        part = previous.loc[previous.method.eq(method)]
        if len(part) != expected or part.status.ne("ok").any():
            raise ValueError(f"Incomplete or failed v3 reference: {method}")
    reference = {tuple(row[k] for k in keys): row for row in previous.to_dict("records")}
    edges, nodes = route.load_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "sampled_pairs.csv", index=False)
    all_rows = []
    for hour in source["hours"]:
        request = route.RouteRequest(**{**batch.comparison.report.CASES[0].__dict__, "hour": hour})
        for scenario in source["scenarios"]:
            graph = ExposureGraph(route.build_arcs(route.prepare_edge_costs(edges, request, scenario)))
            for index, pair in enumerate(pairs.itertuples(index=False), 1):
                context = dict(pair_id=pair.pair_id, band=pair.band, hour=hour, scenario=scenario,
                               start_node=pair.start_node, end_node=pair.end_node)
                refs = {method: reference[pair.pair_id, hour, scenario, method] for method in ["fastest", "budget_distance"]}
                try:
                    rows = compare_pair(graph, pair.start_node, pair.end_node, source["budget"], args.max_labels, args.max_runtime_sec, refs)
                except Exception as exc:
                    rows = [dict(method=m, status="error", error=f"{type(exc).__name__}: {exc}") for m in METHODS]
                all_rows.extend({**context, **r} for r in rows)
                if index % 20 == 0 or index == len(pairs):
                    print(f"{hour}/{scenario}: {index}/{len(pairs)}", flush=True)
            results = pd.DataFrame(all_rows)
            results.to_csv(args.output_dir / "results.csv", index=False)
            batch.summarize_results(results).to_csv(args.output_dir / "summary.csv", index=False)
    errors = results.loc[results.status.ne("ok")]
    errors.to_csv(args.output_dir / "errors.csv", index=False)
    metadata = dict(source_base_commit="964d9e438a2c2b355c440bbf10a49c587ee028a4",
                    reference_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [meta_path, pairs_path, previous_path]},
                    input_sha256=source["input_sha256"], budget=source["budget"],
                    time_exposure_constraint="sum(RS/100 * seconds) <= same-condition fastest route time exposure",
                    hours=source["hours"], scenarios=source["scenarios"], pair_count=len(pairs), scheduled_trials=expected,
                    recorded_rows=len(results), error_rows=len(errors), max_labels=args.max_labels, max_runtime_sec=args.max_runtime_sec,
                    runtime_versions={"python": platform.python_version(), **{n: version(n) for n in ["numpy", "pandas", "scipy", "geopandas", "shapely", "ortools"]}},
                    limits=["Model-based comparison, not measured crash reduction", "Same fixed, correlated OD sample as v3",
                            "Travel time excludes waits, turns, acceleration and access legs", "Turn/comfort costs not in objective",
                            "Joint optimum minimizes distance exposure, not time exposure", "Absolute feasibility tolerance 1e-9",
                            "Search limit returns a feasible incumbent and lower bound, not certified optimum"])
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(batch.summarize_results(results).to_string(index=False))
    if len(errors):
        raise SystemExit(f"Recorded {len(errors)} errors; see errors.csv")


if __name__ == "__main__":
    main()
