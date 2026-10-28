"""Road-layer adapter shared by the supported command-line entry points."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point

from .data import HOURS, LEVELS, SPEEDS, sha256
from .metrics import exposure_metrics
from .routing import ExposureGraph
from .turn_delay import TurnDelayProfile, turn_angle, valid_walk

ALIASES = {"일반": "normal", "주의": "caution", "위험": "risk", "고위험": "high_risk", **dict(zip(LEVELS, LEVELS))}


@dataclass(frozen=True)
class Arc:
    tail: int
    head: int
    edge_id: str
    length_m: float
    time_sec: float
    rs_value: float
    risk_level: str
    start_heading_deg: float
    end_heading_deg: float
    geometry: object


def load_layers(root, legacy=False):
    root = Path(root)
    if legacy:
        path = root/"data/processed/edge_risk_score_no_vehicle.parquet"
        metadata = {"model_candidate": "legacy_abs_rs", "score_semantics": "Historical derived index", "edge_sha256": sha256(path)}
    else:
        artifact = root/"artifacts/final"
        manifest_path = artifact/"manifest.json"
        if not manifest_path.exists():
            raise ValueError("Final scores not built. Run: python -m pmroute build")
        metadata = json.loads(manifest_path.read_text())
        path = artifact/"edge_scores.parquet"
        if sha256(path) != metadata["output_sha256"][path.name]:
            raise ValueError("Built road scores changed; rebuild them")
        if sha256(root/"data/output/final/model/model.json") != metadata["model_sha256"]:
            raise ValueError("Model changed since build; rerun build")
        for name, expected in metadata["input_sha256"].items():
            if sha256(root/"data/processed"/name) != expected:
                raise ValueError(f"Source changed since build: {name}; rerun train/build as appropriate")
    edges = gpd.read_parquet(path)
    nodes = gpd.read_parquet(root/"data/processed/04_osm_nodes_raw.parquet")
    if edges.crs is None or nodes.crs is None:
        raise ValueError("Road/node CRS required")
    edges, nodes = edges.to_crs(5179), nodes.to_crs(5179)
    available = set(edges.u) | set(edges.v)
    nodes = nodes.loc[nodes.osmid.isin(available)].copy()
    return edges, nodes, metadata


def headings(geometry):
    coordinates = list(geometry.coords)
    if len(coordinates) < 2:
        raise ValueError("At least two road geometry points required")
    first, second = coordinates[:2]
    before, last = coordinates[-2:]
    return (math.degrees(math.atan2(second[1]-first[1], second[0]-first[0])),
            math.degrees(math.atan2(last[1]-before[1], last[0]-before[0])))


def oneway(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    text = str(value).lower().strip()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    raise ValueError(f"Unknown oneway value: {value}")


def make_graph(edges, hour="18h", scenario="S1", turn_seconds=0.):
    if hour not in HOURS or scenario not in SPEEDS:
        raise ValueError("Unknown hour or speed scenario")
    arcs = []
    for row in edges.itertuples(index=False):
        level = ALIASES.get(getattr(row, f"risk_level_{hour}"))
        if level is None:
            raise ValueError("Missing/unknown road risk level")
        score = float(getattr(row, f"RS_{hour}"))
        length = float(row.length)
        if not np.isfinite(length) or length <= 0:
            length = float(row.geometry.length)
        if length <= 0:
            raise ValueError("Positive road length required")
        speed = SPEEDS[scenario][LEVELS.index(level)]
        # S1 originally uses the separately stored modal speed; retain it when present.
        if scenario == "S1" and hasattr(row, f"speed_S1_{hour}"):
            speed = float(getattr(row, f"speed_S1_{hour}"))
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Positive modeled speed required")
        start, end = headings(row.geometry)
        arcs.append(Arc(int(row.u), int(row.v), str(row.edge_id), length, length/(speed/3.6), score, level, start, end, row.geometry))
        if not oneway(row.oneway):
            reversed_geometry = LineString(list(row.geometry.coords)[::-1])
            start, end = headings(reversed_geometry)
            arcs.append(Arc(int(row.v), int(row.u), str(row.edge_id), length, length/(speed/3.6), score, level, start, end, reversed_geometry))
    profile = None if turn_seconds == 0 else TurnDelayProfile(turn_seconds)
    return ExposureGraph(arcs, profile)


def parse_coords(value):
    try:
        lon, lat = (float(v) for v in value.split(","))
    except (ValueError, TypeError):
        raise ValueError("Coordinates must be longitude,latitude") from None
    if not np.isfinite([lon, lat]).all() or not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError("Invalid longitude/latitude")
    return lon, lat


def snap(nodes, coords, max_distance=300.):
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("Positive max snap distance required")
    x, y = Transformer.from_crs(4326, 5179, always_xy=True).transform(*coords)
    distances = nodes.geometry.distance(Point(x, y))
    position = int(np.argmin(distances.to_numpy()))
    distance = float(distances.iloc[position])
    if distance > max_distance:
        raise ValueError(f"Endpoint is {distance:.1f}m from the road graph; exceeds configured {max_distance:g}m snap limit")
    return int(nodes.iloc[position].osmid), distance


def signature(path):
    return hashlib.sha256("|".join(f"{a.tail}>{a.head}:{a.edge_id}" for a in path).encode()).hexdigest()[:16]


def compare_routes(graph, start, end, *, max_extra_ratio=.1, max_extra_seconds=120., max_labels=100000, max_runtime_sec=5.):
    result = graph.solve(start, end, max_extra_ratio=max_extra_ratio, max_extra_seconds=max_extra_seconds,
                         max_labels=max_labels, max_runtime_sec=max_runtime_sec, prevent_time_exposure_increase=True)
    baseline = graph.path_metrics(result.fastest_path_arcs)
    rows = []
    paths = {"fastest": result.fastest_path_arcs, "joint": result.path_arcs}
    for method, path in paths.items():
        if not valid_walk(path, start, end):
            raise AssertionError("Invalid directed walk")
        values = exposure_metrics(path)
        values.update(graph.path_metrics(path))
        row = {"method": method, "solver_status": result.status if method == "joint" else "optimal", **values,
               "within_time_budget": values["travel_time_sec"] <= result.time_budget_sec+1e-9,
               "within_time_exposure_reference": values["time_exposure_sec"] <= baseline["time_exposure_sec"]+1e-9,
               "time_budget_sec": result.time_budget_sec,
               "extra_time_sec": values["travel_time_sec"]-baseline["travel_time_sec"],
               "turns_ge_90deg": sum(turn_angle(a, b) >= 90 for a, b in zip(path, path[1:])),
               "path_signature": signature(path),
               "repeated_nodes": len([start]+[a.head for a in path])-len(set([start]+[a.head for a in path])),
               "optimality_scope": "travel_time" if method == "fastest" else "joint_constrained_distance_exposure",
               "lower_bound_exposure_m": result.lower_bound_exposure_m if method == "joint" else None,
               "relative_gap": result.relative_gap if method == "joint" else 0.,
               "labels_generated": result.labels_generated if method == "joint" else 0,
               "stop_reason": result.stop_reason if method == "joint" else "dijkstra"}
        for name in ["distance_exposure_m", "time_exposure_sec"]:
            row[name+"_reduction"] = 1-values[name]/baseline[name] if baseline[name] > 0 else None
        if not row["within_time_budget"] or not row["within_time_exposure_reference"]:
            raise AssertionError("A supported route violates its joint constraints")
        if values["distance_exposure_m"] > baseline["distance_exposure_m"]+1e-7:
            raise AssertionError("Distance exposure increased")
        rows.append(row)
    return rows, paths
