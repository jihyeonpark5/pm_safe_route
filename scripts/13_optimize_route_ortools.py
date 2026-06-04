from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from ortools.graph.python import min_cost_flow
from pyproj import Transformer
from shapely.geometry import Point

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

EDGE_RISK_PATH = PROCESSED_DIR / "edge_risk_score_no_vehicle.parquet"
NODES_PATH = PROCESSED_DIR / "04_osm_nodes_raw.parquet"
ADDRESS_CACHE_PATH = PROCESSED_DIR / "address_coordinate_cache.csv"

DEFAULT_SCENARIO = "S1"
DEFAULT_LAMBDAS = [0.0, 0.5, 1.0]
DEFAULT_SUBGRAPH_MODE = "extreme_paths"
SUBGRAPH_RADIUS_RATIO = 1.25
COST_SCALE = 1_000_000
TARGET_CRS = "EPSG:5179"

HOUR_LABELS = {
    "10h": "10:00",
    "18h": "18:00",
    "22h": "22:00",
}
RISK_LEVELS = ["normal", "caution", "risk", "high_risk"]
RISK_LEVEL_ALIASES = {
    "normal": "normal",
    "caution": "caution",
    "risk": "risk",
    "high_risk": "high_risk",
    "일반": "normal",
    "주의": "caution",
    "위험": "risk",
    "고위험": "high_risk",
    "?쇰컲": "normal",
    "二쇱쓽": "caution",
    "?꾪뿕": "risk",
    "怨좎쐞??": "high_risk",
}
RISK_COLORS = {
    "normal": "#d9d9d9",
    "caution": "#ffd54f",
    "risk": "#fb8c00",
    "high_risk": "#d32f2f",
}
RISK_LABELS = {
    "normal": "Normal",
    "caution": "Caution",
    "risk": "Risk",
    "high_risk": "High risk",
}
ROUTE_COLORS = {
    0.0: "#1f77b4",
    0.5: "#2ca02c",
    1.0: "#d62728",
}
ROUTE_LINESTYLES = {
    0.0: "solid",
    0.5: "dashed",
    1.0: "solid",
}
ROUTE_LINEWIDTHS = {
    0.0: 6.2,
    0.5: 3.0,
    1.0: 4.2,
}
ROUTE_ZORDERS = {
    0.0: 4,
    0.5: 6,
    1.0: 5,
}


@dataclass(frozen=True)
class RouteRequest:
    case_id: str
    slug: str
    name: str
    start_label: str
    end_label: str
    start_lon_lat: tuple[float, float]
    end_lon_lat: tuple[float, float]
    hour: str


@dataclass(frozen=True)
class Arc:
    tail: object
    head: object
    edge_idx: int
    edge_id: str
    length_m: float
    time_sec: float
    risk_cost: float
    time_norm: float
    risk_norm: float
    risk_level: str
    geometry: object


@dataclass(frozen=True)
class OptimizationResult:
    lambda_value: float
    solver: str
    optimization_status: str
    objective_value: float
    normalized_objective: float
    subgraph_node_count: int
    subgraph_edge_count: int
    path_arcs: list[Arc]
    path_valid: bool
    solver_selected_arc_count: int


@dataclass(frozen=True)
class RunConfig:
    scenario: str
    lambdas: list[float]
    requests: list[RouteRequest]
    subgraph_mode: str
    full_graph_fallback: bool
    output_prefix: str


DEFAULT_CASES = [
    RouteRequest(
        case_id="1",
        slug="cheongdam_to_samseong",
        name="Case 1: Cheongdam -> Samseong",
        start_label="Cheongdam Community Service Center",
        end_label="Samseong 1-dong Community Service Center",
        start_lon_lat=(127.04931, 37.52503),
        end_lon_lat=(127.06278, 37.51444),
        hour="18h",
    ),
    RouteRequest(
        case_id="2",
        slug="sinsa_to_cheongdam",
        name="Case 2: Sinsa -> Cheongdam",
        start_label="Sinsa Community Service Center",
        end_label="Cheongdam Community Service Center",
        start_lon_lat=(127.02283, 37.52401),
        end_lon_lat=(127.04931, 37.52503),
        hour="22h",
    ),
    RouteRequest(
        case_id="3",
        slug="gaepo_to_gangnam_office",
        name="Case 3: Gaepo -> Gangnam Office",
        start_label="Gaepo 1-dong Community Service Center",
        end_label="Gangnam-gu Office",
        start_lon_lat=(127.0543931, 37.48328034),
        end_lon_lat=(127.04752, 37.51767),
        hour="18h",
    ),
]


def log(message: str) -> None:
    print(message, flush=True)


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def slugify(text: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower()).strip("_")
    return slug or "route"


def normalize_risk_level(value: object) -> str:
    if pd.isna(value):
        return "normal"
    text = str(value).strip()
    return RISK_LEVEL_ALIASES.get(text, "normal")


def parse_lambdas(value: str) -> list[float]:
    lambdas: list[float] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        lambda_value = float(token)
        if not 0.0 <= lambda_value <= 1.0:
            raise argparse.ArgumentTypeError(f"lambda must be between 0 and 1: {token}")
        lambdas.append(round(lambda_value, 4))
    if not lambdas:
        raise argparse.ArgumentTypeError("at least one lambda must be provided")
    return lambdas


def parse_case_ids(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    case_ids = {item.strip() for item in value.split(",") if item.strip()}
    if not case_ids:
        raise argparse.ArgumentTypeError("at least one case id must be provided")
    return case_ids


def parse_lon_lat(value: str) -> tuple[float, float]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"coordinate must be 'lon,lat': {value}")
    lon = float(parts[0])
    lat = float(parts[1])
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ValueError(f"invalid lon/lat range: {value}")
    return lon, lat


def load_address_cache(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        return {}
    cache: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            address = str(row.get("address", "")).strip()
            if not address:
                continue
            try:
                lon = float(row["lon"])
                lat = float(row["lat"])
            except (KeyError, TypeError, ValueError):
                continue
            cache[address] = (lon, lat)
    return cache


def save_address_cache(path: Path, cache: dict[str, tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["address", "lon", "lat"])
        writer.writeheader()
        for address in sorted(cache):
            lon, lat = cache[address]
            writer.writerow({"address": address, "lon": lon, "lat": lat})


def geocode_address_nominatim(address: str) -> tuple[float, float]:
    params = urlencode(
        {
            "q": address,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "kr",
        }
    )
    request = Request(
        url=f"https://nominatim.openstreetmap.org/search?{params}",
        headers={"User-Agent": "pm-safe-route/1.0"},
    )
    with urlopen(request, timeout=20) as response:
        payload = response.read().decode("utf-8")
    rows = pd.read_json(payload)
    if rows.empty:
        raise ValueError(f"address not found: {address}")
    lat = float(rows.loc[0, "lat"])
    lon = float(rows.loc[0, "lon"])
    return lon, lat


def resolve_address(address: str, cache: dict[str, tuple[float, float]]) -> tuple[float, float]:
    if address in cache:
        return cache[address]
    lon_lat = geocode_address_nominatim(address)
    cache[address] = lon_lat
    save_address_cache(ADDRESS_CACHE_PATH, cache)
    return lon_lat


def build_custom_request(args: argparse.Namespace, cache: dict[str, tuple[float, float]]) -> RouteRequest:
    if not args.hour:
        raise SystemExit("--hour is required when using address or coordinate input")

    if args.start_coords:
        start_lon_lat = parse_lon_lat(args.start_coords)
        start_label = args.start_address or args.start_coords
    elif args.start_address:
        start_lon_lat = resolve_address(args.start_address, cache)
        start_label = args.start_address
    else:
        raise SystemExit("custom route requires --start-address or --start-coords")

    if args.end_coords:
        end_lon_lat = parse_lon_lat(args.end_coords)
        end_label = args.end_address or args.end_coords
    elif args.end_address:
        end_lon_lat = resolve_address(args.end_address, cache)
        end_label = args.end_address
    else:
        raise SystemExit("custom route requires --end-address or --end-coords")

    slug = slugify(f"{start_label}_to_{end_label}_{args.hour}")
    return RouteRequest(
        case_id="custom",
        slug=slug,
        name=f"Custom route: {start_label} -> {end_label}",
        start_label=start_label,
        end_label=end_label,
        start_lon_lat=start_lon_lat,
        end_lon_lat=end_lon_lat,
        hour=args.hour,
    )


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="OR-Tools route optimization with risk-aware costs")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--lambdas", default="0.0,0.5,1.0")
    parser.add_argument("--case-id", default="all")
    parser.add_argument("--subgraph-mode", choices=[DEFAULT_SUBGRAPH_MODE], default=DEFAULT_SUBGRAPH_MODE)
    parser.add_argument("--full-graph-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-prefix", default="")

    parser.add_argument("--start-address", default="")
    parser.add_argument("--end-address", default="")
    parser.add_argument("--start-coords", default="")
    parser.add_argument("--end-coords", default="")
    parser.add_argument("--hour", choices=sorted(HOUR_LABELS), default="")

    args = parser.parse_args()
    lambdas = parse_lambdas(args.lambdas)
    case_ids = parse_case_ids(args.case_id)
    cache = load_address_cache(ADDRESS_CACHE_PATH)

    has_custom_input = any(
        bool(value.strip())
        for value in [args.start_address, args.end_address, args.start_coords, args.end_coords]
    )
    if has_custom_input:
        requests = [build_custom_request(args, cache)]
    else:
        requests = [request for request in DEFAULT_CASES if case_ids is None or request.case_id in case_ids]
        if not requests:
            raise SystemExit(f"no matching case_id found: {args.case_id}")

    return RunConfig(
        scenario=args.scenario,
        lambdas=lambdas,
        requests=requests,
        subgraph_mode=args.subgraph_mode,
        full_graph_fallback=args.full_graph_fallback,
        output_prefix=args.output_prefix.strip(),
    )


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not EDGE_RISK_PATH.exists():
        raise FileNotFoundError(f"missing file: {EDGE_RISK_PATH}")
    if not NODES_PATH.exists():
        raise FileNotFoundError(f"missing file: {NODES_PATH}")

    edges = gpd.read_parquet(EDGE_RISK_PATH)
    nodes = gpd.read_parquet(NODES_PATH)
    if edges.crs is None:
        edges = edges.set_crs(TARGET_CRS)
    if nodes.crs is None:
        nodes = nodes.set_crs(TARGET_CRS)
    if edges.crs.to_epsg() != 5179:
        edges = edges.to_crs(TARGET_CRS)
    if nodes.crs.to_epsg() != 5179:
        nodes = nodes.to_crs(TARGET_CRS)
    return edges, nodes


def nearest_node(nodes: gpd.GeoDataFrame, lon_lat: tuple[float, float]) -> tuple[object, float]:
    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    x, y = transformer.transform(*lon_lat)
    point = Point(x, y)

    try:
        nearest_result = nodes.sindex.nearest(point, return_all=False)
        nearest_array = np.asarray(nearest_result)
        if nearest_array.ndim == 2:
            nearest_pos = int(nearest_array[1][0])
        else:
            nearest_pos = int(np.ravel(nearest_array)[0])
        nearest_row = nodes.iloc[nearest_pos]
        return nearest_row["osmid"], float(nearest_row.geometry.distance(point))
    except Exception:
        distances = nodes.geometry.distance(point)
        nearest_idx = distances.idxmin()
        return nodes.loc[nearest_idx, "osmid"], float(distances.loc[nearest_idx])


def normalize_oneway(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def prepare_edge_costs(edges: gpd.GeoDataFrame, request: RouteRequest, scenario: str) -> gpd.GeoDataFrame:
    speed_col = f"speed_{scenario}_{request.hour}"
    rs_col = f"RS_{request.hour}"
    level_col = f"risk_level_{request.hour}"
    required_cols = {"edge_id", "u", "v", "oneway", "geometry", "length", speed_col, rs_col, level_col}
    missing_cols = required_cols - set(edges.columns)
    if missing_cols:
        raise ValueError(f"missing required edge columns: {sorted(missing_cols)}")

    cost_df = edges.copy().reset_index(drop=True)
    cost_df["edge_idx"] = cost_df.index
    cost_df["edge_length_m"] = pd.to_numeric(cost_df["length"], errors="coerce")
    invalid_length = cost_df["edge_length_m"].isna() | cost_df["edge_length_m"].le(0)
    cost_df.loc[invalid_length, "edge_length_m"] = cost_df.loc[invalid_length].geometry.length

    speed_kmh = pd.to_numeric(cost_df[speed_col], errors="coerce").fillna(25).clip(lower=1)
    speed_mps = speed_kmh * 1000.0 / 3600.0
    rs = pd.to_numeric(cost_df[rs_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    cost_df["time_sec"] = cost_df["edge_length_m"] / speed_mps
    cost_df["risk_cost"] = (rs / 100.0) * cost_df["edge_length_m"]

    max_time = float(cost_df["time_sec"].max())
    max_risk = float(cost_df["risk_cost"].max())
    cost_df["time_norm"] = cost_df["time_sec"] / max_time if max_time > 0 else 0.0
    cost_df["risk_norm"] = cost_df["risk_cost"] / max_risk if max_risk > 0 else 0.0
    cost_df["risk_level"] = cost_df[level_col].apply(normalize_risk_level)
    cost_df["oneway_bool"] = cost_df["oneway"].apply(normalize_oneway)
    return cost_df


def build_arcs(cost_df: pd.DataFrame) -> list[Arc]:
    arcs: list[Arc] = []
    for row in cost_df.itertuples(index=False):
        forward = Arc(
            tail=row.u,
            head=row.v,
            edge_idx=int(row.edge_idx),
            edge_id=str(row.edge_id),
            length_m=float(row.edge_length_m),
            time_sec=float(row.time_sec),
            risk_cost=float(row.risk_cost),
            time_norm=float(row.time_norm),
            risk_norm=float(row.risk_norm),
            risk_level=str(row.risk_level),
            geometry=row.geometry,
        )
        arcs.append(forward)
        if not bool(row.oneway_bool):
            arcs.append(
                Arc(
                    tail=row.v,
                    head=row.u,
                    edge_idx=int(row.edge_idx),
                    edge_id=str(row.edge_id),
                    length_m=float(row.edge_length_m),
                    time_sec=float(row.time_sec),
                    risk_cost=float(row.risk_cost),
                    time_norm=float(row.time_norm),
                    risk_norm=float(row.risk_norm),
                    risk_level=str(row.risk_level),
                    geometry=row.geometry,
                )
            )
    return arcs


def build_weighted_adjacency(
    arcs: Iterable[Arc],
    weight_getter,
) -> dict[object, list[tuple[object, float, Arc]]]:
    adjacency: dict[object, list[tuple[object, float, Arc]]] = defaultdict(list)
    for arc in arcs:
        adjacency[arc.tail].append((arc.head, float(weight_getter(arc)), arc))
    return adjacency


def reverse_weighted_adjacency(
    arcs: Iterable[Arc],
    weight_getter,
) -> dict[object, list[tuple[object, float, Arc]]]:
    adjacency: dict[object, list[tuple[object, float, Arc]]] = defaultdict(list)
    for arc in arcs:
        adjacency[arc.head].append((arc.tail, float(weight_getter(arc)), arc))
    return adjacency


def dijkstra_distances(
    adjacency: dict[object, list[tuple[object, float, Arc]]],
    start_nodes: Iterable[object],
    cutoff: float | None = None,
) -> dict[object, float]:
    queue: list[tuple[float, object]] = []
    distances: dict[object, float] = {}

    for node in start_nodes:
        if node in distances:
            continue
        distances[node] = 0.0
        heapq.heappush(queue, (0.0, node))

    while queue:
        current_dist, node = heapq.heappop(queue)
        if current_dist > distances.get(node, math.inf):
            continue
        if cutoff is not None and current_dist > cutoff:
            continue
        for next_node, edge_weight, _ in adjacency.get(node, []):
            new_dist = current_dist + edge_weight
            if cutoff is not None and new_dist > cutoff:
                continue
            if new_dist < distances.get(next_node, math.inf):
                distances[next_node] = new_dist
                heapq.heappush(queue, (new_dist, next_node))
    return distances


def shortest_path(arcs: list[Arc], start_node: object, end_node: object, weight_name: str) -> tuple[list[Arc], float]:
    adjacency = build_weighted_adjacency(arcs, lambda arc: getattr(arc, weight_name))
    queue: list[tuple[float, object]] = [(0.0, start_node)]
    distances = {start_node: 0.0}
    previous: dict[object, tuple[object, Arc]] = {}

    while queue:
        current_dist, node = heapq.heappop(queue)
        if node == end_node:
            break
        if current_dist > distances.get(node, math.inf):
            continue
        for next_node, edge_weight, arc in adjacency.get(node, []):
            new_dist = current_dist + edge_weight
            if new_dist < distances.get(next_node, math.inf):
                distances[next_node] = new_dist
                previous[next_node] = (node, arc)
                heapq.heappush(queue, (new_dist, next_node))

    if end_node not in distances:
        raise ValueError(f"no path found between {start_node} and {end_node}")

    path_arcs: list[Arc] = []
    current = end_node
    while current != start_node:
        prev_node, arc = previous[current]
        path_arcs.append(arc)
        current = prev_node
    path_arcs.reverse()
    return path_arcs, float(distances[end_node])


def build_subgraph_arcs(arcs: list[Arc], start_node: object, end_node: object) -> tuple[list[Arc], dict[str, float | int | bool]]:
    time_path_arcs, _ = shortest_path(arcs, start_node, end_node, "time_norm")
    risk_path_arcs, _ = shortest_path(arcs, start_node, end_node, "risk_norm")

    baseline_time_distance_m = sum(arc.length_m for arc in time_path_arcs)
    baseline_risk_distance_m = sum(arc.length_m for arc in risk_path_arcs)
    cutoff = max(baseline_time_distance_m, baseline_risk_distance_m) * SUBGRAPH_RADIUS_RATIO

    extreme_nodes = {start_node, end_node}
    for arc in time_path_arcs + risk_path_arcs:
        extreme_nodes.add(arc.tail)
        extreme_nodes.add(arc.head)

    forward_length = build_weighted_adjacency(arcs, lambda arc: arc.length_m)
    reverse_length = reverse_weighted_adjacency(arcs, lambda arc: arc.length_m)
    forward_distances = dijkstra_distances(forward_length, extreme_nodes, cutoff=cutoff)
    reverse_distances = dijkstra_distances(reverse_length, extreme_nodes, cutoff=cutoff)
    eligible_nodes = set(forward_distances) | set(reverse_distances)
    subgraph_arcs = [arc for arc in arcs if arc.tail in eligible_nodes and arc.head in eligible_nodes]

    diagnostics: dict[str, float | int | bool] = {
        "baseline_time_distance_m": round(baseline_time_distance_m, 3),
        "baseline_risk_distance_m": round(baseline_risk_distance_m, 3),
        "subgraph_cutoff_m": round(cutoff, 3),
        "subgraph_node_count": len(eligible_nodes),
        "subgraph_edge_count": len(subgraph_arcs),
        "full_graph_node_count": len({arc.tail for arc in arcs} | {arc.head for arc in arcs}),
        "full_graph_edge_count": len(arcs),
        "used_full_graph_fallback": False,
    }
    return subgraph_arcs, diagnostics


def arc_cost(arc: Arc, lambda_value: float) -> int:
    raw_cost = (1.0 - lambda_value) * arc.time_norm + lambda_value * arc.risk_norm
    return max(0, int(round(raw_cost * COST_SCALE)))


def validate_solution_path(
    selected_arcs: list[Arc],
    start_node: object,
    end_node: object,
) -> tuple[bool, list[Arc]]:
    if not selected_arcs:
        return False, []

    outgoing: dict[object, list[Arc]] = defaultdict(list)
    incoming_count: dict[object, int] = defaultdict(int)
    outgoing_count: dict[object, int] = defaultdict(int)
    touched_nodes: set[object] = set()

    for arc in selected_arcs:
        outgoing[arc.tail].append(arc)
        outgoing_count[arc.tail] += 1
        incoming_count[arc.head] += 1
        touched_nodes.add(arc.tail)
        touched_nodes.add(arc.head)

    for node in touched_nodes:
        in_count = incoming_count.get(node, 0)
        out_count = outgoing_count.get(node, 0)
        if node == start_node:
            if in_count != 0 or out_count != 1:
                return False, []
        elif node == end_node:
            if in_count != 1 or out_count != 0:
                return False, []
        else:
            if in_count != 1 or out_count != 1:
                return False, []

    path_arcs: list[Arc] = []
    visited_nodes = {start_node}
    current = start_node
    while current != end_node:
        next_arcs = outgoing.get(current, [])
        if len(next_arcs) != 1:
            return False, []
        arc = next_arcs[0]
        path_arcs.append(arc)
        current = arc.head
        if current in visited_nodes and current != end_node:
            return False, []
        visited_nodes.add(current)

    if len(path_arcs) != len(selected_arcs):
        return False, []
    return True, path_arcs


def status_to_name(solver: min_cost_flow.SimpleMinCostFlow, status: int) -> str:
    status_names = {
        getattr(solver, "OPTIMAL", None): "Optimal",
        getattr(solver, "INFEASIBLE", None): "Infeasible",
        getattr(solver, "UNBALANCED", None): "Unbalanced",
        getattr(solver, "BAD_RESULT", None): "BadResult",
        getattr(solver, "BAD_COST_RANGE", None): "BadCostRange",
    }
    if status in status_names:
        return status_names[status]
    return f"Status_{status}"


def optimize_route(subgraph_arcs: list[Arc], start_node: object, end_node: object, lambda_value: float) -> OptimizationResult:
    node_ids = sorted({start_node, end_node} | {arc.tail for arc in subgraph_arcs} | {arc.head for arc in subgraph_arcs})
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    solver = min_cost_flow.SimpleMinCostFlow()
    arc_lookup: list[Arc] = []
    for arc in subgraph_arcs:
        solver.add_arc_with_capacity_and_unit_cost(
            node_to_idx[arc.tail],
            node_to_idx[arc.head],
            1,
            arc_cost(arc, lambda_value),
        )
        arc_lookup.append(arc)

    for node_id, node_idx in node_to_idx.items():
        if node_id == start_node:
            solver.set_node_supply(node_idx, 1)
        elif node_id == end_node:
            solver.set_node_supply(node_idx, -1)
        else:
            solver.set_node_supply(node_idx, 0)

    status = solver.solve()
    status_name = status_to_name(solver, status)
    if status != solver.OPTIMAL:
        return OptimizationResult(
            lambda_value=lambda_value,
            solver="ortools_min_cost_flow",
            optimization_status=status_name,
            objective_value=float("nan"),
            normalized_objective=float("nan"),
            subgraph_node_count=len(node_ids),
            subgraph_edge_count=len(subgraph_arcs),
            path_arcs=[],
            path_valid=False,
            solver_selected_arc_count=0,
        )

    selected_arcs: list[Arc] = []
    for arc_idx in range(solver.num_arcs()):
        if solver.flow(arc_idx) == 1:
            selected_arcs.append(arc_lookup[arc_idx])

    path_valid, path_arcs = validate_solution_path(selected_arcs, start_node, end_node)
    objective_value = solver.optimal_cost() / COST_SCALE
    return OptimizationResult(
        lambda_value=lambda_value,
        solver="ortools_min_cost_flow",
        optimization_status=status_name if path_valid else "InvalidPath",
        objective_value=objective_value,
        normalized_objective=objective_value if path_valid else float("nan"),
        subgraph_node_count=len(node_ids),
        subgraph_edge_count=len(subgraph_arcs),
        path_arcs=path_arcs if path_valid else [],
        path_valid=path_valid,
        solver_selected_arc_count=len(selected_arcs),
    )


def route_difference(base_arcs: list[Arc], other_arcs: list[Arc]) -> float:
    base_set = {(arc.tail, arc.head, arc.edge_id) for arc in base_arcs}
    other_set = {(arc.tail, arc.head, arc.edge_id) for arc in other_arcs}
    union_count = len(base_set | other_set)
    if union_count == 0:
        return 0.0
    return 1.0 - len(base_set & other_set) / union_count


def summarize_route(
    request: RouteRequest,
    result: OptimizationResult,
    base_arcs: list[Arc],
    start_snap_m: float,
    end_snap_m: float,
    diagnostics: dict[str, float | int | bool],
    subgraph_mode: str,
) -> dict[str, object]:
    counts = pd.Series([arc.risk_level for arc in result.path_arcs], dtype="object").value_counts()
    distance_m = sum(arc.length_m for arc in result.path_arcs)
    raw_time_min = sum(arc.time_sec for arc in result.path_arcs) / 60.0
    raw_risk_sum = sum(arc.risk_cost for arc in result.path_arcs)
    return {
        "case_id": request.case_id,
        "case_name": request.name,
        "start": request.start_label,
        "end": request.end_label,
        "hour": HOUR_LABELS[request.hour],
        "scenario": diagnostics["scenario"],
        "lambda": result.lambda_value,
        "solver": result.solver,
        "optimization_status": result.optimization_status,
        "objective_value": round(result.objective_value, 6) if math.isfinite(result.objective_value) else result.objective_value,
        "normalized_objective": round(result.normalized_objective, 6)
        if math.isfinite(result.normalized_objective)
        else result.normalized_objective,
        "subgraph_mode": subgraph_mode,
        "subgraph_node_count": result.subgraph_node_count,
        "subgraph_edge_count": result.subgraph_edge_count,
        "distance_km": round(distance_m / 1000.0, 3),
        "model_time_min": round(raw_time_min, 2),
        "raw_time_min": round(raw_time_min, 2),
        "risk_sum": round(raw_risk_sum, 2),
        "raw_risk_sum": round(raw_risk_sum, 2),
        "normal_edges": int(counts.get("normal", 0)),
        "caution_edges": int(counts.get("caution", 0)),
        "risk_edges": int(counts.get("risk", 0)),
        "high_risk_edges": int(counts.get("high_risk", 0)),
        "route_diff_vs_lambda0": round(route_difference(base_arcs, result.path_arcs), 3),
        "start_snap_m": round(start_snap_m, 1),
        "end_snap_m": round(end_snap_m, 1),
        "path_valid": result.path_valid,
        "path_arc_count": len(result.path_arcs),
        "solver_selected_arc_count": result.solver_selected_arc_count,
        "used_full_graph_fallback": bool(diagnostics["used_full_graph_fallback"]),
        "baseline_time_distance_m": diagnostics["baseline_time_distance_m"],
        "baseline_risk_distance_m": diagnostics["baseline_risk_distance_m"],
    }


def output_stem(request: RouteRequest, config: RunConfig) -> str:
    prefix = f"{config.output_prefix}_" if config.output_prefix else ""
    return f"{prefix}route_ortools_{request.case_id}_{request.slug}_{request.hour}_{config.scenario}"


def request_output_path(request: RouteRequest, config: RunConfig) -> Path:
    return OUTPUT_DIR / f"{output_stem(request, config)}.png"


def summary_output_path(config: RunConfig) -> Path:
    prefix = f"{config.output_prefix}_" if config.output_prefix else ""
    suffix = "custom" if len(config.requests) == 1 and config.requests[0].case_id == "custom" else "cases"
    return OUTPUT_DIR / f"{prefix}route_ortools_summary_{suffix}_{config.scenario}.csv"


def arcs_to_gdf(arcs: list[Arc], crs: object) -> gpd.GeoDataFrame:
    if not arcs:
        return gpd.GeoDataFrame(columns=["edge_id", "geometry"], geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(
        {"edge_id": [arc.edge_id for arc in arcs], "geometry": [arc.geometry for arc in arcs]},
        geometry="geometry",
        crs=crs,
    )


def lon_lat_to_projected_point(lon_lat: tuple[float, float]) -> Point:
    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    x, y = transformer.transform(*lon_lat)
    return Point(x, y)


def plot_order(lambda_values: list[float]) -> list[float]:
    preferred = [0.0, 1.0, 0.5]
    ordered = [value for value in preferred if value in lambda_values]
    ordered.extend(value for value in lambda_values if value not in ordered)
    return ordered


def save_route_map(
    edges: gpd.GeoDataFrame,
    request: RouteRequest,
    results_by_lambda: dict[float, OptimizationResult],
    config: RunConfig,
    diagnostics: dict[str, float | int | bool],
) -> Path:
    output_path = request_output_path(request, config)
    fig, ax = plt.subplots(figsize=(14, 14), dpi=500)

    color_series = edges[f"risk_level_{request.hour}"].apply(normalize_risk_level).map(RISK_COLORS).fillna("#eeeeee")
    edges.plot(ax=ax, color="#e6e6e6", linewidth=0.65, alpha=0.8, zorder=1)
    edges.plot(ax=ax, color=color_series, linewidth=0.45, alpha=0.72, zorder=2)

    for lambda_value in plot_order(config.lambdas):
        route_gdf = arcs_to_gdf(results_by_lambda[lambda_value].path_arcs, edges.crs)
        if route_gdf.empty:
            continue
        route_gdf.plot(
            ax=ax,
            color=ROUTE_COLORS.get(lambda_value, "#333333"),
            linewidth=ROUTE_LINEWIDTHS.get(lambda_value, 3.4),
            linestyle=ROUTE_LINESTYLES.get(lambda_value, "solid"),
            alpha=0.9,
            zorder=ROUTE_ZORDERS.get(lambda_value, 5),
        )

    start_point = lon_lat_to_projected_point(request.start_lon_lat)
    end_point = lon_lat_to_projected_point(request.end_lon_lat)
    ax.scatter(start_point.x, start_point.y, s=150, marker="o", color="#00a651", edgecolor="black", linewidth=1.0, zorder=10)
    ax.scatter(end_point.x, end_point.y, s=180, marker="*", color="#7b1fa2", edgecolor="black", linewidth=0.9, zorder=10)
    ax.annotate("Start", (start_point.x, start_point.y), xytext=(8, 8), textcoords="offset points", fontsize=11, fontweight="bold", color="#007a3d", zorder=11)
    ax.annotate("End", (end_point.x, end_point.y), xytext=(8, 8), textcoords="offset points", fontsize=11, fontweight="bold", color="#5e1678", zorder=11)

    route_gdfs = [arcs_to_gdf(result.path_arcs, edges.crs) for result in results_by_lambda.values() if result.path_arcs]
    if route_gdfs:
        route_bounds = pd.concat(route_gdfs, ignore_index=True).total_bounds
        minx, miny, maxx, maxy = route_bounds
        minx = min(minx, start_point.x, end_point.x)
        miny = min(miny, start_point.y, end_point.y)
        maxx = max(maxx, start_point.x, end_point.x)
        maxy = max(maxy, start_point.y, end_point.y)
        width = maxx - minx
        height = maxy - miny
        pad_left = max(width * 0.22, 350)
        pad_right = max(width * 0.06, 120)
        pad_bottom = max(height * 0.16, 250)
        pad_top = max(height * 0.10, 180)
        ax.set_xlim(minx - pad_left, maxx + pad_right)
        ax.set_ylim(miny - pad_bottom, maxy + pad_top)

    fallback_label = "yes" if diagnostics["used_full_graph_fallback"] else "no"
    ax.set_title(request.name, fontsize=17, pad=18)
    subtitle = (
        f"{HOUR_LABELS[request.hour]} / {config.scenario} / fallback={fallback_label} / "
        f"lambda={','.join(f'{value:g}' for value in config.lambdas)}"
    )
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=12)
    ax.set_axis_off()

    route_handles = [
        Line2D(
            [0],
            [0],
            color=ROUTE_COLORS.get(value, "#333333"),
            lw=ROUTE_LINEWIDTHS.get(value, 3.4),
            linestyle=ROUTE_LINESTYLES.get(value, "solid"),
            label=f"lambda={value:g}",
        )
        for value in config.lambdas
    ]
    risk_handles = [
        Line2D([0], [0], color=RISK_COLORS[level], lw=4, label=RISK_LABELS[level])
        for level in RISK_LEVELS
    ]
    point_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#00a651", markeredgecolor="black", markersize=8, label="Start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#7b1fa2", markeredgecolor="black", markersize=11, label="End"),
    ]
    ax.legend(
        handles=route_handles + risk_handles + point_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=min(len(route_handles) + len(risk_handles) + len(point_handles), 8),
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def optimize_request(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    request: RouteRequest,
    config: RunConfig,
) -> tuple[dict[float, OptimizationResult], list[dict[str, object]], dict[str, float | int | bool]]:
    start_node, start_snap_m = nearest_node(nodes, request.start_lon_lat)
    end_node, end_snap_m = nearest_node(nodes, request.end_lon_lat)
    cost_df = prepare_edge_costs(edges, request, config.scenario)
    arcs = build_arcs(cost_df)
    subgraph_arcs, diagnostics = build_subgraph_arcs(arcs, start_node, end_node)
    diagnostics["scenario"] = config.scenario

    log(f"  start node: {start_node} / end node: {end_node}")
    log(f"  snap distance: start {start_snap_m:.1f}m, end {end_snap_m:.1f}m")
    log(
        "  baseline distances: "
        f"time-only {diagnostics['baseline_time_distance_m']}m, "
        f"risk-only {diagnostics['baseline_risk_distance_m']}m"
    )
    shrink_ratio = (
        float(diagnostics["subgraph_edge_count"]) / float(diagnostics["full_graph_edge_count"])
        if diagnostics["full_graph_edge_count"]
        else 0.0
    )
    log(
        "  subgraph shrink: "
        f"{int(diagnostics['subgraph_edge_count']):,}/{int(diagnostics['full_graph_edge_count']):,} "
        f"({shrink_ratio:.1%})"
    )

    lambda_values = list(dict.fromkeys([0.0] + config.lambdas))
    results_by_lambda = {
        lambda_value: optimize_route(subgraph_arcs, start_node, end_node, lambda_value)
        for lambda_value in lambda_values
    }

    if config.full_graph_fallback and any(not result.path_valid for result in results_by_lambda.values()):
        log("  subgraph solve failed for at least one lambda; retrying on full graph")
        results_by_lambda = {
            lambda_value: optimize_route(arcs, start_node, end_node, lambda_value)
            for lambda_value in lambda_values
        }
        diagnostics["used_full_graph_fallback"] = True
        diagnostics["subgraph_node_count"] = diagnostics["full_graph_node_count"]
        diagnostics["subgraph_edge_count"] = diagnostics["full_graph_edge_count"]

    base_arcs = results_by_lambda[0.0].path_arcs
    summary_rows = [
        summarize_route(
            request,
            results_by_lambda[lambda_value],
            base_arcs,
            start_snap_m,
            end_snap_m,
            diagnostics,
            config.subgraph_mode,
        )
        for lambda_value in config.lambdas
    ]

    for lambda_value in config.lambdas:
        result = results_by_lambda[lambda_value]
        log(
            f"  lambda={lambda_value:g}: "
            f"{result.optimization_status}, "
            f"path_valid={result.path_valid}, "
            f"selected_arc={result.solver_selected_arc_count}"
        )

    return {lambda_value: results_by_lambda[lambda_value] for lambda_value in config.lambdas}, summary_rows, diagnostics


def main() -> int:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = parse_args()

    log("=== Step 13: OR-Tools risk-aware route optimization ===")
    log(
        f"config: scenario={config.scenario}, "
        f"lambdas={','.join(f'{value:g}' for value in config.lambdas)}, "
        f"subgraph_mode={config.subgraph_mode}, "
        f"full_graph_fallback={config.full_graph_fallback}"
    )

    try:
        edges, nodes = load_inputs()
        all_summary_rows: list[dict[str, object]] = []

        for request in config.requests:
            log(f"\n[{request.case_id}] {request.name}")
            log(f"  start: {request.start_label} ({request.start_lon_lat[0]:.6f}, {request.start_lon_lat[1]:.6f})")
            log(f"  end:   {request.end_label} ({request.end_lon_lat[0]:.6f}, {request.end_lon_lat[1]:.6f})")
            results_by_lambda, summary_rows, diagnostics = optimize_request(edges, nodes, request, config)
            output_path = save_route_map(edges, request, results_by_lambda, config, diagnostics)
            all_summary_rows.extend(summary_rows)
            log(f"  map saved: {output_path}")
            log(f"  fallback used: {diagnostics['used_full_graph_fallback']}")
            for row in summary_rows:
                log(
                    f"  lambda={row['lambda']}: "
                    f"{row['optimization_status']}, "
                    f"{row['distance_km']:.3f}km, "
                    f"{row['raw_time_min']:.2f}min, "
                    f"risk={row['raw_risk_sum']:.2f}"
                )

        summary = pd.DataFrame(all_summary_rows)
        summary_path = summary_output_path(config)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        log(f"\nsummary saved: {summary_path}")
        log(summary.to_string(index=False))
        return 0
    except Exception as exc:
        log(f"\n[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
