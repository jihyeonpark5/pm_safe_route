"""Reproducible, stratified OD benchmark; does not validate actual crash risk.

Sample by straight-line distance before solving, then use the same directed OD
pairs across all hours/scenarios. Failures and limited searches remain in output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
from importlib.metadata import version
import json
import multiprocessing
import platform
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from route_metrics import exposure_metrics
from time_budget_routing import ExposureGraph, allowed_extra_time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("budget_comparison", Path(__file__).with_name("16_compare_time_budget_routes.py"))
comparison = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = comparison
spec.loader.exec_module(comparison)
route = comparison.route
BANDS = [("short", 500., 1500.), ("medium", 1500., 3000.), ("long", 3000., 5000.)]
METHODS = ["fastest", "legacy_lambda_0.5", "legacy_lambda_1", "budget_distance"]
_WORKER_STATE = None


def _initialize_worker(arcs, graph, options, max_labels, max_runtime):
    global _WORKER_STATE
    _WORKER_STATE = arcs, graph, options, max_labels, max_runtime


def _evaluate_worker(pair):
    arcs, graph, options, max_labels, max_runtime = _WORKER_STATE
    try:
        rows = evaluate_pair(arcs, graph, pair["start_node"], pair["end_node"], options, max_labels, max_runtime)
    except Exception as exc:
        rows = [dict(method=m, status="error", error=f"{type(exc).__name__}: {exc}") for m in METHODS]
    return pair, rows


def largest_strong_component(arcs):
    ids = sorted({a.tail for a in arcs} | {a.head for a in arcs})
    if not ids:
        raise ValueError("Empty graph")
    positions = {node: i for i, node in enumerate(ids)}
    matrix = csr_matrix((np.ones(len(arcs), dtype=bool),
                         ([positions[a.tail] for a in arcs], [positions[a.head] for a in arcs])), shape=(len(ids), len(ids)))
    count, labels = connected_components(matrix, directed=True, connection="strong")
    biggest = int(np.argmax(np.bincount(labels)))
    selected = {node for node, label in zip(ids, labels) if label == biggest}
    return selected, dict(graph_nodes=len(ids), strong_components=int(count), eligible_component_nodes=len(selected))


def sample_pairs(nodes, eligible, per_band=20, seed=42):
    if per_band < 1:
        raise ValueError("per_band must be positive")
    candidates = nodes.loc[nodes.osmid.isin(eligible)].sort_values("osmid").reset_index(drop=True)
    if len(candidates) < 2 or candidates.osmid.duplicated().any():
        raise ValueError("Need unique eligible node coordinates")
    if set(candidates.osmid) != set(eligible):
        raise ValueError("Missing coordinates for eligible graph nodes")
    ids = candidates.osmid.to_numpy()
    xy = np.column_stack([candidates.geometry.x, candidates.geometry.y])
    rng = np.random.default_rng(seed)
    counts = {band: 0 for band, _, _ in BANDS}
    records, used = [], set()
    attempts = 0
    while len(records) < per_band * len(BANDS):
        attempts += 1
        if attempts > max(100000, per_band * 10000):
            raise ValueError(f"Unable to fill prespecified distance bands: {counts}")
        i, j = rng.choice(len(candidates), size=2, replace=False)
        pair = int(ids[i]), int(ids[j])
        if pair in used:
            continue
        distance = float(np.linalg.norm(xy[i] - xy[j]))
        for band, lower, upper in BANDS:
            if lower <= distance < upper and counts[band] < per_band:
                records.append(dict(pair_id=f"OD{len(records)+1:03d}", band=band, start_node=pair[0], end_node=pair[1],
                                    straight_distance_m=distance, start_x=float(xy[i, 0]), start_y=float(xy[i, 1]),
                                    end_x=float(xy[j, 0]), end_y=float(xy[j, 1])))
                used.add(pair)
                counts[band] += 1
                break
    return pd.DataFrame(records), dict(seed=seed, attempts=attempts, per_band=per_band, counts=counts)


def summarize_results(results):
    rows = []
    keys = ["hour", "scenario", "method"]
    for values, part in results.groupby(keys, sort=True):
        ok = part.loc[part.status.eq("ok")].copy()
        for column in ["distance_exposure_m_reduction_vs_fastest", "time_exposure_sec_reduction_vs_fastest",
                       "solver_status", "within_time_budget", "extra_time_sec_vs_fastest", "solve_elapsed_sec", "relative_gap"]:
            if column not in ok:
                ok[column] = np.nan
        finite = ok.loc[ok.distance_exposure_m_reduction_vs_fastest.notna()]
        time_finite = ok.loc[ok.time_exposure_sec_reduction_vs_fastest.notna()]
        row = dict(zip(keys, values))
        row.update(scheduled=len(part), successful=len(ok), errors=int(part.status.ne("ok").sum()),
                   optimal=int(ok.solver_status.isin(["optimal", "Optimal"]).sum()),
                   feasible_limit=int(ok.solver_status.eq("feasible_limit").sum()),
                   time_cap_violations=int(ok.within_time_budget.eq(False).sum()),
                   undefined_distance_ratios=len(ok)-len(finite), undefined_time_ratios=len(ok)-len(time_finite),
                   distance_improved=int(finite.distance_exposure_m_reduction_vs_fastest.gt(1e-8).sum()),
                   distance_worsened=int(finite.distance_exposure_m_reduction_vs_fastest.lt(-1e-8).sum()),
                   time_exposure_worsened=int(time_finite.time_exposure_sec_reduction_vs_fastest.lt(-1e-8).sum()),
                   distance_reduction_mean=float(finite.distance_exposure_m_reduction_vs_fastest.mean()),
                   distance_reduction_median=float(finite.distance_exposure_m_reduction_vs_fastest.median()),
                   distance_reduction_p10=float(finite.distance_exposure_m_reduction_vs_fastest.quantile(.1)),
                   time_exposure_reduction_mean=float(time_finite.time_exposure_sec_reduction_vs_fastest.mean()),
                   extra_time_sec_median=float(ok.extra_time_sec_vs_fastest.median()),
                   solve_elapsed_sec_median=float(ok.solve_elapsed_sec.median()),
                   solve_elapsed_sec_p95=float(ok.solve_elapsed_sec.quantile(.95)),
                   relative_gap_max=float(ok.relative_gap.max()))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_pair(arcs, graph, start, end, options, max_labels, max_runtime):
    began = time.monotonic()
    fastest, fastest_seconds = graph.shortest_path(start, end)
    fastest_elapsed = time.monotonic() - began
    baseline = exposure_metrics(fastest)
    budget = fastest_seconds + allowed_extra_time(fastest_seconds, **options)
    output = []

    def record(method, path, solver_status, elapsed, gap=None, **extra):
        if not route.validate_solution_path(path, start, end)[0]:
            raise ValueError("Invalid connected path")
        metrics = exposure_metrics(path)
        seconds = sum(float(a.time_sec) for a in path)
        angles = [route.angle_difference_degrees(a.end_heading_deg, b.start_heading_deg) for a, b in zip(path, path[1:])]
        row = dict(method=method, status="ok", error="", solver_status=solver_status,
                   travel_time_sec=seconds, distance_m=sum(a.length_m for a in path),
                   time_budget_sec=budget, within_time_budget=seconds <= budget+1e-9,
                   extra_time_sec_vs_fastest=seconds-fastest_seconds, solve_elapsed_sec=elapsed, relative_gap=gap,
                   turns_ge_90deg=sum(angle >= 90 for angle in angles),
                   path_signature=route.path_signature(path), **metrics, **extra)
        for metric in ("distance_exposure_m", "time_exposure_sec", "high_risk_distance_m"):
            row[metric + "_reduction_vs_fastest"] = 1-metrics[metric]/baseline[metric] if baseline[metric] > 0 else float("nan")
        if method == "budget_distance" and (not row["within_time_budget"] or metrics["distance_exposure_m"] > baseline["distance_exposure_m"]+1e-7):
            raise AssertionError("Constrained route violates time/exposure invariant")
        output.append(row)

    record("fastest", fastest, "optimal", fastest_elapsed, 0.)
    try:
        result = graph.solve(start, end, **options, max_labels=max_labels, max_runtime_sec=max_runtime)
        record("budget_distance", result.path_arcs, result.status, result.elapsed_sec, result.relative_gap,
               labels_generated=result.labels_generated, stop_reason=result.stop_reason,
               lower_bound_exposure_m=result.lower_bound_exposure_m)
    except Exception as exc:
        output.append(dict(method="budget_distance", status="error", error=f"{type(exc).__name__}: {exc}"))
    try:
        began = time.monotonic()
        subgraph, _ = route.build_subgraph_arcs(arcs, start, end)
        build_seconds = time.monotonic()-began
        for value in [.5, 1.]:
            method = f"legacy_lambda_{value:g}"
            try:
                began = time.monotonic()
                result = route.optimize_route(subgraph, start, end, value)
                fallback = not result.path_valid
                if fallback:
                    result = route.optimize_route(arcs, start, end, value)
                record(method, result.path_arcs, result.optimization_status, time.monotonic()-began,
                       legacy_subgraph_build_sec=build_seconds, legacy_full_graph_fallback=fallback)
            except Exception as exc:
                output.append(dict(method=method, status="error", error=f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        output.extend(dict(method=f"legacy_lambda_{value:g}", status="error", error=f"{type(exc).__name__}: {exc}") for value in [.5, 1.])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-band", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hours", nargs="+", choices=["10h", "18h", "22h"], default=["10h", "18h", "22h"])
    parser.add_argument("--scenarios", nargs="+", choices=["S0", "S1", "S2"], default=["S0", "S1", "S2"])
    parser.add_argument("--max-extra-ratio", type=float, default=.1)
    parser.add_argument("--max-extra-seconds", type=float, default=120.)
    parser.add_argument("--max-labels", type=int, default=100000)
    parser.add_argument("--max-runtime-sec", type=float, default=5.)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/output/validation_v3")
    args = parser.parse_args()
    options = dict(max_extra_ratio=args.max_extra_ratio, max_extra_seconds=args.max_extra_seconds)
    allowed_extra_time(1, **options)
    if args.max_labels < 1 or not np.isfinite(args.max_runtime_sec) or args.max_runtime_sec <= 0:
        parser.error("Use positive max-labels and finite positive max-runtime-sec")
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    if len(set(args.hours)) != len(args.hours) or len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Do not repeat hours or scenarios")
    edges, nodes = route.load_inputs()
    request = comparison.report.CASES[0]
    base_arcs = route.build_arcs(route.prepare_edge_costs(edges, request, "S1"))
    eligible, component_info = largest_strong_component(base_arcs)
    pairs, sampling_info = sample_pairs(nodes, eligible, args.pairs_per_band, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "sampled_pairs.csv", index=False)
    all_rows, condition_timings = [], []
    for hour in args.hours:
        request = route.RouteRequest(**{**request.__dict__, "hour": hour})
        for scenario in args.scenarios:
            began = time.monotonic()
            arcs = route.build_arcs(route.prepare_edge_costs(edges, request, scenario))
            graph = ExposureGraph(arcs)
            condition_timings.append(dict(hour=hour, scenario=scenario, setup_sec=time.monotonic()-began))
            worker_args = arcs, graph, options, args.max_labels, args.max_runtime_sec
            def consume(iterator):
                for number, (pair, rows) in enumerate(iterator, 1):
                    context = {key: pair[key] for key in ["pair_id", "band", "start_node", "end_node"]}
                    context.update(hour=hour, scenario=scenario)
                    all_rows.extend({**context, **r} for r in rows)
                    if number % 10 == 0 or number == len(pairs):
                        print(f"{hour}/{scenario}: {number}/{len(pairs)} pairs", flush=True)
            if args.workers == 1:
                _initialize_worker(*worker_args)
                consume(map(_evaluate_worker, pairs.to_dict("records")))
            else:
                # fork shares the read-only graph on Linux; spawn supports Windows.
                start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context(start_method),
                                         initializer=_initialize_worker, initargs=worker_args) as executor:
                    consume(executor.map(_evaluate_worker, pairs.to_dict("records")))
            # Checkpoint each complete condition. Failures remain in all tables.
            results = pd.DataFrame(all_rows)
            results.to_csv(args.output_dir / "results.csv", index=False)
            summarize_results(results).to_csv(args.output_dir / "summary.csv", index=False)
    results = pd.DataFrame(all_rows)
    by_band = []
    for band, part in results.groupby("band"):
        by_band.append(summarize_results(part).assign(band=band))
    pd.concat(by_band, ignore_index=True).to_csv(args.output_dir / "summary_by_band.csv", index=False)
    errors = results.loc[results.status.ne("ok")]
    errors.to_csv(args.output_dir / "errors.csv", index=False)
    metadata = dict(source_base_commit="43fc9fff2f79067d6c086593d8bfb0a6245ef4a2", sampling=sampling_info,
                    component=component_info, hours=args.hours, scenarios=args.scenarios, methods=METHODS,
                    budget=options, max_labels=args.max_labels, max_runtime_sec=args.max_runtime_sec,
                    scheduled_trials=len(pairs)*len(args.hours)*len(args.scenarios), recorded_rows=len(results),
                    error_rows=len(errors), workers=args.workers,
                    process_start_method="single" if args.workers == 1 else ("fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"),
                    runtime_versions={"python": platform.python_version(), **{name: version(name) for name in
                                      ["numpy", "pandas", "scipy", "geopandas", "shapely", "scikit-learn", "ortools"]}},
                    condition_timings=condition_timings,
                    input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (route.EDGE_RISK_PATH, route.NODES_PATH)},
                    limits=["Model-based comparison, not external safety validation",
                            "Same OD pairs reused across conditions; rows are correlated",
                            "Equal counts per Euclidean distance band, not observed trip distribution",
                            "Largest strongly connected component only; other nodes excluded before sampling",
                            "Positive reduction means improvement; zero baseline gives an undefined ratio",
                            "New solver excludes turn/comfort penalties; times exclude waits and access legs",
                            "Per-query elapsed excludes shared graph preparation; legacy subgraph preparation is recorded separately",
                            "Search caps apply to the new solver only; feasible_limit is retained without claiming optimality"])
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summarize_results(results).to_string(index=False))
    if len(errors):
        raise SystemExit(f"Recorded {len(errors)} errors; see errors.csv")


if __name__ == "__main__":
    main()
