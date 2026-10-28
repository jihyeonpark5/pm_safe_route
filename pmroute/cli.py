"""Supported train/build/route/benchmark commands for the integrated project."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

from . import __version__
from .data import build_scores, sha256
from .model import train
from .network import compare_routes, load_layers, make_graph, parse_coords, snap


def add_solver_options(parser):
    parser.add_argument("--turn-seconds-at-90", type=float, default=0.)
    parser.add_argument("--max-extra-ratio", type=float, default=.1)
    parser.add_argument("--max-extra-seconds", type=float, default=120.)
    parser.add_argument("--max-labels", type=int, default=100000)
    parser.add_argument("--max-runtime-sec", type=float, default=5.)
    parser.add_argument("--legacy-scores", action="store_true")


def solver_options(args):
    return dict(max_extra_ratio=args.max_extra_ratio, max_extra_seconds=args.max_extra_seconds,
                max_labels=args.max_labels, max_runtime_sec=args.max_runtime_sec)


def route_command(root, args):
    edges, nodes, metadata = load_layers(root, args.legacy_scores)
    graph = make_graph(edges, args.hour, args.scenario, args.turn_seconds_at_90)
    start, start_snap = snap(nodes, parse_coords(args.start), args.max_snap_distance_m)
    end, end_snap = snap(nodes, parse_coords(args.end), args.max_snap_distance_m)
    rows, paths = compare_routes(graph, start, end, **solver_options(args))
    output = args.output_dir or root/"data/output/final/route"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/"summary.csv", index=False)
    features = []
    transformer_crs = edges.crs
    for method, path in paths.items():
        if not path:
            continue
        geometries = gpd.GeoSeries([a.geometry for a in path], crs=transformer_crs).to_crs(4326)
        for order, (arc, geometry) in enumerate(zip(path, geometries)):
            features.append(dict(type="Feature", geometry=mapping(geometry), properties=dict(method=method, order=order,
                edge_id=arc.edge_id, from_node=arc.tail, to_node=arc.head, rs_index=arc.rs_value, risk_level=arc.risk_level)))
    (output/"routes.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=features)), encoding="utf-8")
    document = dict(version=__version__, model=metadata, hour=args.hour, scenario=args.scenario,
                    turn_seconds_at_90=args.turn_seconds_at_90, solver_options=solver_options(args),
                    start_node=start, end_node=end, start_snap_m=start_snap, end_snap_m=end_snap,
                    access_leg_time_included=False, results=rows)
    (output/"result.json").write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    print(pd.DataFrame(rows)[["method", "solver_status", "travel_time_sec", "distance_exposure_m_reduction", "time_exposure_sec_reduction"]].to_string(index=False))
    print("Output:", output)


def benchmark_command(root, args):
    edges, _, metadata = load_layers(root, args.legacy_scores)
    pairs_path = root/"data/output/validation_v3/sampled_pairs.csv"
    pairs = pd.read_csv(pairs_path)
    if args.pair_limit is not None:
        if args.pair_limit < 1:
            raise ValueError("pair-limit must be positive")
        pairs = pairs.head(args.pair_limit)
    output = args.output_dir or root/"data/output/final/routes"
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for hour in ["10h", "18h", "22h"]:
        for scenario in ["S0", "S1", "S2"]:
            graph = make_graph(edges, hour, scenario, args.turn_seconds_at_90)
            for number, pair in enumerate(pairs.itertuples(index=False), 1):
                try:
                    rows, _ = compare_routes(graph, pair.start_node, pair.end_node, **solver_options(args))
                    all_rows.extend(dict(pair_id=pair.pair_id, band=pair.band, hour=hour, scenario=scenario, status="ok", error="", **row) for row in rows)
                except Exception as exc:
                    all_rows.extend(dict(pair_id=pair.pair_id, band=pair.band, hour=hour, scenario=scenario,
                                         method=method, status="error", error=f"{type(exc).__name__}: {exc}") for method in ["fastest", "joint"])
                if number % 20 == 0 or number == len(pairs):
                    print(f"final model {hour}/{scenario}: {number}/{len(pairs)}", flush=True)
            pd.DataFrame(all_rows).to_csv(output/"results.csv", index=False)
    data = pd.DataFrame(all_rows)
    errors = data.loc[data.status.ne("ok")]
    errors.to_csv(output/"errors.csv", index=False)
    summaries = []
    for (hour, scenario), part in data.loc[data.method.eq("joint")].groupby(["hour", "scenario"]):
        ok = part.loc[part.status.eq("ok")]
        summaries.append(dict(hour=hour, scenario=scenario, scheduled=len(part), successful=len(ok), errors=len(part)-len(ok),
            optimal=int(ok.get("solver_status", pd.Series(dtype=str)).eq("optimal").sum()),
            feasible_limit=int(ok.get("solver_status", pd.Series(dtype=str)).eq("feasible_limit").sum()),
            time_violations=int(ok.get("within_time_budget", pd.Series(dtype=bool)).eq(False).sum()),
            time_exposure_violations=int(ok.get("within_time_exposure_reference", pd.Series(dtype=bool)).eq(False).sum()),
            distance_reduction_mean=float(ok.get("distance_exposure_m_reduction", pd.Series(dtype=float)).mean()),
            time_exposure_reduction_mean=float(ok.get("time_exposure_sec_reduction", pd.Series(dtype=float)).mean())))
    pd.DataFrame(summaries).to_csv(output/"summary.csv", index=False)
    document = dict(version=__version__, model=metadata, pair_count=len(pairs), pair_source_sha256=sha256(pairs_path),
                    turn_seconds_at_90=args.turn_seconds_at_90, solver_options=solver_options(args),
                    trials=len(pairs)*9, result_rows=len(data), errors=len(errors),
                    limitations=["Internal model-based routing check, not external safety validation",
                                 "New score scale differs from historical RS; percentages across models have different denominators",
                                 "Hourly scenario indices are not validated hourly accident probabilities"])
    (output/"metadata.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    if len(errors):
        raise ValueError(f"{len(errors)} error rows recorded in {output/'errors.csv'}")


def main():
    parser = argparse.ArgumentParser(prog="pm-safe-route", description="Research pipeline for calibrated scoring and constrained routes")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="Compare models with grouped validation and fit selected JSON model")
    commands.add_parser("build", help="Build separate grid/edge score artifacts")
    route_parser = commands.add_parser("route", help="Find a joint-constrained route for coordinates")
    route_parser.add_argument("--start", required=True, help="longitude,latitude")
    route_parser.add_argument("--end", required=True, help="longitude,latitude")
    route_parser.add_argument("--hour", choices=["10h", "18h", "22h"], default="18h")
    route_parser.add_argument("--scenario", choices=["S0", "S1", "S2"], default="S1")
    route_parser.add_argument("--max-snap-distance-m", type=float, default=300.)
    route_parser.add_argument("--output-dir", type=Path)
    add_solver_options(route_parser)
    benchmark = commands.add_parser("benchmark", help="Check final scores on the fixed 60-pair sample")
    benchmark.add_argument("--pair-limit", type=int)
    benchmark.add_argument("--output-dir", type=Path)
    add_solver_options(benchmark)
    args = parser.parse_args()
    root = args.project_root.resolve()
    try:
        if args.command == "train":
            train(root/"data/processed/merged_grid_variables.parquet", root/"data/output/final/model")
        elif args.command == "build":
            build_scores(root)
        elif args.command == "route":
            route_command(root, args)
        else:
            benchmark_command(root, args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
