"""Turn-delay sensitivity with fixed v4 OD pairs; assumptions are not calibrated."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from route_metrics import exposure_metrics
from time_budget_routing import ExposureGraph
from turn_delay import TurnDelayProfile, turn_angle, valid_walk

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("joint_reference", Path(__file__).with_name("19_compare_joint_exposure.py"))
joint = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = joint
spec.loader.exec_module(joint)
batch, route = joint.batch, joint.route
METHODS = ["fastest_turn", "v4_joint_replayed", "joint_turn"]
_STATE = None


def initialize_worker(graphs, options, max_labels, max_runtime, references):
    global _STATE
    _STATE = graphs, options, max_labels, max_runtime, references


def summarize(results):
    summaries = []
    for profile, part in results.groupby("turn_seconds_at_90"):
        table = batch.summarize_results(part).assign(turn_seconds_at_90=profile)
        extra = []
        for keys, group in part.groupby(["hour", "scenario", "method"]):
            ok = group.loc[group.status.eq("ok")]
            extra.append(dict(hour=keys[0], scenario=keys[1], method=keys[2],
                              time_exposure_cap_violations=int(ok.get("within_time_exposure_reference", pd.Series(dtype=bool)).eq(False).sum()),
                              mean_turn_delay_sec=float(ok.get("turn_delay_sec", pd.Series(dtype=float)).mean()),
                              mean_turns_ge_90deg=float(ok.get("turns_ge_90deg", pd.Series(dtype=float)).mean()),
                              repeated_node_routes=int(ok.get("repeated_nodes", pd.Series(dtype=float)).gt(0).sum())))
        summaries.append(table.merge(pd.DataFrame(extra), on=["hour", "scenario", "method"], validate="one_to_one"))
    return pd.concat(summaries, ignore_index=True)


def evaluate_pair(pair):
    graphs, options, max_labels, max_runtime, references = _STATE
    start, end = pair["start_node"], pair["end_node"]
    rows = []
    try:
        previous = graphs[0.].solve(start, end, **options, max_labels=max_labels, max_runtime_sec=max_runtime,
                                    prevent_time_exposure_increase=True)
        old_path = previous.path_arcs
        reference = references[pair["pair_id"]]
        if previous.status == "optimal" and reference["solver_status"] == "optimal":
            if not np.isclose(previous.distance_exposure_m, reference["distance_exposure_m"], rtol=0, atol=1e-7):
                raise AssertionError("Zero-delay v4 optimum changed")
    except Exception as exc:
        return pair, [dict(turn_seconds_at_90=profile, method=method, status="error", error=f"{type(exc).__name__}: {exc}")
                      for profile in graphs for method in METHODS]
    for profile, graph in graphs.items():
        try:
            result = previous if profile == 0 else graph.solve(start, end, **options, max_labels=max_labels,
                       max_runtime_sec=max_runtime, prevent_time_exposure_increase=True)
            fastest = result.fastest_path_arcs
            baseline = exposure_metrics(fastest)
            baseline.update(graph.path_metrics(fastest))
            for method, path in [("fastest_turn", fastest), ("v4_joint_replayed", old_path), ("joint_turn", result.path_arcs)]:
                if not valid_walk(path, start, end):
                    raise AssertionError("Disconnected directed walk")
                metrics = exposure_metrics(path)
                metrics.update(graph.path_metrics(path))
                angles = [turn_angle(a, b) for a, b in zip(path, path[1:])]
                nodes = [start]+[a.head for a in path]
                row = dict(turn_seconds_at_90=profile, method=method, status="ok", error="",
                           solver_status=result.status if method == "joint_turn" else ("optimal" if method == "fastest_turn" else "replayed"),
                           time_budget_sec=result.time_budget_sec,
                           within_time_budget=metrics["travel_time_sec"] <= result.time_budget_sec+1e-9,
                           time_exposure_reference_sec=baseline["time_exposure_sec"],
                           within_time_exposure_reference=metrics["time_exposure_sec"] <= baseline["time_exposure_sec"]+1e-9,
                           extra_time_sec_vs_fastest=metrics["travel_time_sec"]-baseline["travel_time_sec"],
                           solve_elapsed_sec=result.elapsed_sec if method == "joint_turn" else float("nan"),
                           relative_gap=result.relative_gap if method == "joint_turn" else float("nan"),
                           labels_generated=result.labels_generated if method == "joint_turn" else 0,
                           stop_reason=result.stop_reason if method == "joint_turn" else "not_searched",
                           lower_bound_exposure_m=result.lower_bound_exposure_m if method == "joint_turn" else float("nan"),
                           turns_ge_90deg=sum(a >= 90 for a in angles), turns_ge_150deg=sum(a >= 150 for a in angles),
                           repeated_nodes=len(nodes)-len(set(nodes)), path_signature=route.path_signature(path), **metrics)
                for name in ["distance_exposure_m", "time_exposure_sec", "high_risk_distance_m"]:
                    row[name+"_reduction_vs_fastest"] = 1-metrics[name]/baseline[name] if baseline[name] > 0 else float("nan")
                if method == "joint_turn" and (not row["within_time_budget"] or not row["within_time_exposure_reference"]
                                               or metrics["distance_exposure_m"] > baseline["distance_exposure_m"]+1e-7):
                    raise AssertionError("Turn-aware constrained solution violates a baseline bound")
                if profile == 0 and method == "joint_turn":
                    row["v4_path_match"] = row["path_signature"] == reference["path_signature"]
                rows.append(row)
        except Exception as exc:
            rows = [r for r in rows if r["turn_seconds_at_90"] != profile]
            rows.extend(dict(turn_seconds_at_90=profile, method=m, status="error", error=f"{type(exc).__name__}: {exc}") for m in METHODS)
    return pair, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "data/output/validation_v4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/output/validation_v5")
    parser.add_argument("--turn-profiles", nargs="+", type=float, default=[0., 3., 6.])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-labels", type=int, default=100000)
    parser.add_argument("--max-runtime-sec", type=float, default=5.)
    parser.add_argument("--hours", nargs="+", choices=["10h", "18h", "22h"])
    parser.add_argument("--scenarios", nargs="+", choices=["S0", "S1", "S2"])
    parser.add_argument("--pair-limit", type=int, default=None, help="Optional first-N smoke subset; omit for full benchmark")
    args = parser.parse_args()
    if args.reference_dir.resolve() == args.output_dir.resolve():
        parser.error("Output must differ from reference")
    if not 1 <= args.workers <= 8 or args.max_labels < 1 or not np.isfinite(args.max_runtime_sec) or args.max_runtime_sec <= 0:
        parser.error("Invalid workers or solver limits")
    if len(set(args.turn_profiles)) != len(args.turn_profiles) or 0. not in args.turn_profiles:
        parser.error("Distinct profiles including zero required")
    for value in args.turn_profiles:
        TurnDelayProfile(value)
    source_paths = [args.reference_dir/name for name in ["metadata.json", "sampled_pairs.csv", "results.csv"]]
    metadata = json.loads(source_paths[0].read_text())
    for path in [route.EDGE_RISK_PATH, route.NODES_PATH]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["input_sha256"][path.name]:
            raise ValueError(f"Input changed since v4: {path.name}")
    pairs = pd.read_csv(source_paths[1])
    if args.pair_limit is not None:
        if args.pair_limit < 1:
            parser.error("pair-limit must be positive")
        pairs = pairs.head(args.pair_limit)
    hours, scenarios = args.hours or metadata["hours"], args.scenarios or metadata["scenarios"]
    if len(set(hours)) != len(hours) or len(set(scenarios)) != len(scenarios):
        parser.error("Repeated conditions are not allowed")
    old = pd.read_csv(source_paths[2]); old = old.loc[old.method.eq("budget_joint")]
    if old.duplicated(["pair_id", "hour", "scenario"]).any() or old.status.ne("ok").any():
        raise ValueError("Invalid v4 reference")
    edges, _ = route.load_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "sampled_pairs.csv", index=False)
    all_rows = []
    for hour in hours:
        request = route.RouteRequest(**{**batch.comparison.report.CASES[0].__dict__, "hour": hour})
        for scenario in scenarios:
            arcs = route.build_arcs(route.prepare_edge_costs(edges, request, scenario))
            graphs = {value: ExposureGraph(arcs, None if value == 0 else TurnDelayProfile(value)) for value in args.turn_profiles}
            references = {r["pair_id"]: r for r in old.loc[old.hour.eq(hour) & old.scenario.eq(scenario)].to_dict("records")}
            if not set(pairs.pair_id).issubset(references):
                raise ValueError("Missing v4 reference pairs")
            def consume(iterator):
                for number, (pair, rows) in enumerate(iterator, 1):
                    context = {k: pair[k] for k in ["pair_id", "band", "start_node", "end_node"]}
                    context.update(hour=hour, scenario=scenario)
                    all_rows.extend({**context, **row} for row in rows)
                    if number % 10 == 0 or number == len(pairs):
                        print(f"{hour}/{scenario}: {number}/{len(pairs)} pairs, {len(graphs)} profiles", flush=True)
            worker_args = graphs, metadata["budget"], args.max_labels, args.max_runtime_sec, references
            if args.workers == 1:
                initialize_worker(*worker_args)
                consume(map(evaluate_pair, pairs.to_dict("records")))
            else:
                context = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn")
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=context, initializer=initialize_worker, initargs=worker_args) as executor:
                    consume(executor.map(evaluate_pair, pairs.to_dict("records")))
            results = pd.DataFrame(all_rows)
            results.to_csv(args.output_dir / "results.csv", index=False)
            summarize(results).to_csv(args.output_dir / "summary.csv", index=False)
    errors = results.loc[results.status.ne("ok")]
    errors.to_csv(args.output_dir / "errors.csv", index=False)
    output_metadata = dict(source_base_commit="ce6626c6c493f254aed281bb3966917d6787f0af",
        reference_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        input_sha256=metadata["input_sha256"], budget=metadata["budget"], turn_profiles=args.turn_profiles,
        profile_model="<=30deg: 0; 90deg: configured seconds; >=150deg: twice configured seconds; linear between",
        turn_exposure_model="turn delay * max(incoming RS, outgoing RS)/100",
        hours=hours, scenarios=scenarios, pair_count=len(pairs), workers=args.workers,
        scheduled_trials=len(pairs)*len(hours)*len(scenarios)*len(graphs), recorded_rows=len(results), error_rows=len(errors),
        max_labels=args.max_labels, max_runtime_sec=args.max_runtime_sec,
        runtime_versions={"python": platform.python_version(), **{n: version(n) for n in ["numpy", "pandas", "scipy", "geopandas", "shapely", "ortools"]}},
        limits=["Uncalibrated turn-delay assumptions, not observed travel times", "Model exposure, not actual crash risk",
                "No traffic-light timing, left/right asymmetry, acceleration or access-leg model",
                "Turn-aware states may revisit physical nodes; repeated-node routes are reported",
                "References and constraints use fastest route under each profile; cross-profile percentages use different baselines",
                "Search limits can return feasible_limit; optimality must be checked separately"])
    (args.output_dir / "metadata.json").write_text(json.dumps(output_metadata, indent=2), encoding="utf-8")
    print(summarize(results).to_string(index=False))
    if len(errors):
        raise SystemExit(f"Recorded {len(errors)} errors; see errors.csv")


if __name__ == "__main__":
    main()
