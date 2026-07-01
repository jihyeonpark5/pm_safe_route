from __future__ import annotations

import argparse
import csv
import heapq
import hashlib
import json
import math
import os
import re
import ast
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
SCENARIO_SPEED_LIMITS = {
    "S0": {"normal": 25.0, "caution": 25.0, "risk": 25.0, "high_risk": 25.0},
    "S1": {"normal": 25.0, "caution": 20.0, "risk": 15.0, "high_risk": 10.0},
    "S2": {"normal": 20.0, "caution": 15.0, "risk": 10.0, "high_risk": 7.0},
}
DEFAULT_LAMBDAS = [round(step / 10.0, 1) for step in range(11)]
DEFAULT_SUBGRAPH_MODE = "extreme_paths"
SUBGRAPH_RADIUS_RATIO = 1.6
SUBGRAPH_CORRIDOR_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SUBGRAPH_BUFFER_M = 180.0
COST_SCALE = 1_000_000
TARGET_CRS = "EPSG:5179"
ROBUST_NORM_QUANTILE = 0.95
HIGH_RISK_FIXED_PENALTY = 70.0
ADVANCED_PENALTY_WEIGHT = 0.42
SHORT_EDGE_THRESHOLD_M = 35.0
TURN_COMPLEXITY_FACTOR = 1.15
TURN_PENALTY_SCALE = 0.24
TURN_FREE_ANGLE_DEG = 22.0
UTURN_ANGLE_DEG = 150.0
UTURN_PENALTY_MULTIPLIER = 1.8
RISK_LEVEL_PENALTIES = {
    "normal": 0.0,
    "caution": 5.0,
    "risk": 22.0,
    "high_risk": HIGH_RISK_FIXED_PENALTY,
}

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
    0.0: (0, (3, 3)),
    0.5: (0, (1.4, 2.2)),
    1.0: (0, (5, 2, 1.4, 2)),
}
ROUTE_LINEWIDTHS = {
    0.0: 4.8,
    0.5: 3.8,
    1.0: 3.2,
}
ROUTE_ZORDERS = {
    0.0: 7,
    0.5: 5,
    1.0: 6,
}
MAP_BACKGROUND = "#f5f3ee"
ROAD_STYLE_ORDER = ["trunk", "primary", "secondary", "tertiary", "local", "service", "path", "other"]
ROAD_STYLE_CONFIG = {
    "trunk": {"fill": "#f7c66b", "casing": "#d8a14c", "fill_width": 3.8, "casing_width": 5.6, "alpha": 0.95},
    "primary": {"fill": "#f5d58a", "casing": "#d5b46b", "fill_width": 3.2, "casing_width": 4.8, "alpha": 0.94},
    "secondary": {"fill": "#f4dfaa", "casing": "#ceb989", "fill_width": 2.5, "casing_width": 3.9, "alpha": 0.93},
    "tertiary": {"fill": "#fbf2d2", "casing": "#d4c7a8", "fill_width": 1.8, "casing_width": 2.8, "alpha": 0.92},
    "local": {"fill": "#ffffff", "casing": "#dad7d2", "fill_width": 1.25, "casing_width": 2.0, "alpha": 0.96},
    "service": {"fill": "#f8f7f3", "casing": "#d8d4cb", "fill_width": 0.9, "casing_width": 1.45, "alpha": 0.94},
    "path": {"fill": "#efe9dc", "casing": "#cec5b6", "fill_width": 0.65, "casing_width": 1.0, "alpha": 0.88},
    "other": {"fill": "#f3f0ea", "casing": "#ddd8cf", "fill_width": 0.8, "casing_width": 1.3, "alpha": 0.9},
}
ROAD_CLASS_LOOKUP = {
    "motorway": "trunk",
    "motorway_link": "trunk",
    "trunk": "trunk",
    "trunk_link": "trunk",
    "primary": "primary",
    "primary_link": "primary",
    "secondary": "secondary",
    "secondary_link": "secondary",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary",
    "residential": "local",
    "living_street": "local",
    "unclassified": "local",
    "cycleway": "path",
    "footway": "path",
    "pedestrian": "path",
    "path": "path",
    "steps": "path",
    "track": "path",
    "service": "service",
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
    rs_value: float
    risk_cost: float
    comfort_cost: float
    time_norm: float
    risk_norm: float
    comfort_norm: float
    risk_level: str
    road_class: str
    start_heading_deg: float
    end_heading_deg: float
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
    # RouteRequest(
    #     case_id="1",
    #     slug="cheongdam_to_samseong",
    #     name="Case 1: Cheongdam -> Samseong",
    #     start_label="Cheongdam Community Service Center",
    #     end_label="Samseong 1-dong Community Service Center",
    #     start_lon_lat=(127.04931, 37.52503),
    #     end_lon_lat=(127.06278, 37.51444),
    #     hour="18h",
    # ),
    # RouteRequest(
    #     case_id="2",
    #     slug="sinsa_to_cheongdam",
    #     name="Case 2: Sinsa -> Cheongdam",
    #     start_label="Sinsa Community Service Center",
    #     end_label="Cheongdam Community Service Center",
    #     start_lon_lat=(127.02283, 37.52401),
    #     end_lon_lat=(127.04931, 37.52503),
    #     hour="22h",
    # ),
    # RouteRequest(
    #     case_id="3",
    #     slug="gaepo_to_gangnam_office",
    #     name="Case 3: Gaepo -> Gangnam Office",
    #     start_label="Gaepo 1-dong Community Service Center",
    #     end_label="Gangnam-gu Office",
    #     start_lon_lat=(127.0543931, 37.48328034),
    #     end_lon_lat=(127.04752, 37.51767),
    #     hour="18h",
    # ),
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


def normalize_highway_class(value: object) -> str:
    if value is None or pd.isna(value):
        return "other"
    if isinstance(value, (list, tuple, set)):
        highway_values = [str(item).strip().lower() for item in value if str(item).strip()]
    else:
        text = str(value).strip()
        highway_values: list[str]
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = [text]
            if isinstance(parsed, (list, tuple, set)):
                highway_values = [str(item).strip().lower() for item in parsed if str(item).strip()]
            else:
                highway_values = [str(parsed).strip().lower()]
        else:
            highway_values = [text.lower()]

    for road_type in highway_values:
        normalized = ROAD_CLASS_LOOKUP.get(road_type)
        if normalized:
            return normalized
    return "other"


def robust_normalize(series: pd.Series, quantile: float = ROBUST_NORM_QUANTILE) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    scale = float(numeric.quantile(quantile))
    if not math.isfinite(scale) or scale <= 0:
        max_value = float(numeric.max()) if len(numeric) else 0.0
        scale = max_value if max_value > 0 else 1.0
    return (numeric / scale).clip(lower=0.0, upper=1.0)


def parse_numeric_width(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return float("nan")
    return float(match.group(1))


def geometry_straightness_penalty(geometry: object, length_m: float) -> float:
    if geometry is None or getattr(geometry, "is_empty", True) or length_m <= 0:
        return 0.0
    coords = list(getattr(geometry, "coords", []))
    if len(coords) < 2:
        return 0.0
    start_x, start_y = coords[0]
    end_x, end_y = coords[-1]
    straight_distance = math.hypot(end_x - start_x, end_y - start_y)
    if straight_distance <= 0:
        return 0.0
    sinuosity = max(1.0, length_m / straight_distance)
    return max(0.0, (sinuosity - 1.0) * 8.0)


def effective_lambda(lambda_value: float) -> float:
    return lambda_value ** 0.72


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
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid geocoding response for address '{address}'") from exc
    if not rows:
        raise ValueError(f"address not found: {address}")
    first_row = rows[0]
    lat = float(first_row["lat"])
    lon = float(first_row["lon"])
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
    parser.add_argument("--lambdas", default=",".join(f"{value:.1f}" for value in DEFAULT_LAMBDAS))
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


def nearest_node(nodes: gpd.GeoDataFrame, lon_lat: tuple[float, float]) -> tuple[object, float, Point]:
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
        return nearest_row["osmid"], float(nearest_row.geometry.distance(point)), nearest_row.geometry
    except Exception:
        distances = nodes.geometry.distance(point)
        nearest_idx = distances.idxmin()
        return nodes.loc[nearest_idx, "osmid"], float(distances.loc[nearest_idx]), nodes.loc[nearest_idx, "geometry"]


def normalize_oneway(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def scenario_speed_kmh(cost_df: pd.DataFrame, scenario: str, hour: str) -> pd.Series:
    level_col = f"risk_level_{hour}"
    if level_col not in cost_df.columns:
        raise ValueError(f"missing required edge column: {level_col}")

    normalized_levels = cost_df[level_col].apply(normalize_risk_level)
    scenario_name = scenario.upper()
    if scenario_name == "S1":
        speed_col = f"speed_S1_{hour}"
        if speed_col in cost_df.columns:
            return pd.to_numeric(cost_df[speed_col], errors="coerce").fillna(25.0).clip(lower=1.0)

    if scenario_name not in SCENARIO_SPEED_LIMITS:
        raise ValueError(f"unsupported scenario: {scenario}")

    return pd.to_numeric(
        normalized_levels.map(SCENARIO_SPEED_LIMITS[scenario_name]),
        errors="coerce",
    ).fillna(25.0).clip(lower=1.0)


def prepare_edge_costs(edges: gpd.GeoDataFrame, request: RouteRequest, scenario: str) -> gpd.GeoDataFrame:
    rs_col = f"RS_{request.hour}"
    level_col = f"risk_level_{request.hour}"
    required_cols = {"edge_id", "u", "v", "oneway", "geometry", "length", rs_col, level_col}
    missing_cols = required_cols - set(edges.columns)
    if missing_cols:
        raise ValueError(f"missing required edge columns: {sorted(missing_cols)}")

    cost_df = edges.copy().reset_index(drop=True)
    cost_df["edge_idx"] = cost_df.index
    cost_df["edge_length_m"] = pd.to_numeric(cost_df["length"], errors="coerce")
    invalid_length = cost_df["edge_length_m"].isna() | cost_df["edge_length_m"].le(0)
    cost_df.loc[invalid_length, "edge_length_m"] = cost_df.loc[invalid_length].geometry.length

    speed_kmh = scenario_speed_kmh(cost_df, scenario, request.hour)
    speed_mps = speed_kmh * 1000.0 / 3600.0
    rs = pd.to_numeric(cost_df[rs_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    cost_df["road_class"] = cost_df["highway"].apply(normalize_highway_class)
    degree_series = pd.concat([cost_df["u"], cost_df["v"]], ignore_index=True).value_counts()
    cost_df["node_degree_u"] = cost_df["u"].map(degree_series).fillna(1).astype(float)
    cost_df["node_degree_v"] = cost_df["v"].map(degree_series).fillna(1).astype(float)
    cost_df["width_m"] = cost_df["width"].apply(parse_numeric_width)

    cost_df["time_sec"] = cost_df["edge_length_m"] / speed_mps
    cost_df["rs_value"] = rs
    cost_df["risk_level"] = cost_df[level_col].apply(normalize_risk_level)
    cost_df["risk_level_penalty"] = pd.to_numeric(
        cost_df["risk_level"].map(RISK_LEVEL_PENALTIES),
        errors="coerce",
    ).fillna(0.0)
    cost_df["risk_cost"] = ((rs / 100.0) * cost_df["time_sec"]).astype(float) + cost_df["risk_level_penalty"].astype(float)
    road_discomfort = cost_df["road_class"].map(
        {
            "trunk": 11.0,
            "primary": 8.0,
            "secondary": 4.5,
            "tertiary": 1.2,
            "local": 0.0,
            "service": 1.0,
            "path": 0.6,
            "other": 1.4,
        }
    ).fillna(1.4)
    short_edge_penalty = ((SHORT_EDGE_THRESHOLD_M - cost_df["edge_length_m"]).clip(lower=0.0) / SHORT_EDGE_THRESHOLD_M) * 5.0
    intersection_penalty = (
        (cost_df["node_degree_u"] - 2.0).clip(lower=0.0)
        + (cost_df["node_degree_v"] - 2.0).clip(lower=0.0)
    ) * TURN_COMPLEXITY_FACTOR
    width_penalty = (cost_df["width_m"].fillna(0.0) / 8.0).clip(lower=0.0, upper=2.5)
    bridge_tunnel_penalty = (
        cost_df["bridge"].fillna("").astype(str).ne("").astype(float) * 1.8
        + cost_df["tunnel"].fillna("").astype(str).ne("").astype(float) * 1.2
    )
    curvature_penalty = pd.Series(
        [geometry_straightness_penalty(geom, length_m) for geom, length_m in zip(cost_df.geometry, cost_df["edge_length_m"])],
        index=cost_df.index,
        dtype="float64",
    )
    cost_df["comfort_cost"] = (
        road_discomfort
        + short_edge_penalty
        + intersection_penalty
        + width_penalty
        + bridge_tunnel_penalty
        + curvature_penalty
    )
    cost_df["time_norm"] = robust_normalize(cost_df["time_sec"])
    cost_df["risk_norm"] = robust_normalize(cost_df["risk_cost"])
    cost_df["comfort_norm"] = robust_normalize(cost_df["comfort_cost"])
    cost_df["oneway_bool"] = cost_df["oneway"].apply(normalize_oneway)
    return cost_df


def edge_heading_degrees(geometry, reverse: bool = False) -> tuple[float, float]:
    if geometry is None or geometry.is_empty:
        return 0.0, 0.0
    coords = list(geometry.coords)
    if len(coords) < 2:
        return 0.0, 0.0
    if reverse:
        coords = list(reversed(coords))

    start_x0, start_y0 = coords[0]
    start_x1, start_y1 = coords[1]
    end_x0, end_y0 = coords[-2]
    end_x1, end_y1 = coords[-1]

    start_heading = math.degrees(math.atan2(start_y1 - start_y0, start_x1 - start_x0)) % 360.0
    end_heading = math.degrees(math.atan2(end_y1 - end_y0, end_x1 - end_x0)) % 360.0
    return start_heading, end_heading


def angle_difference_degrees(angle_a: float, angle_b: float) -> float:
    diff = abs(angle_a - angle_b) % 360.0
    return min(diff, 360.0 - diff)


def build_arcs(cost_df: pd.DataFrame) -> list[Arc]:
    arcs: list[Arc] = []
    for row in cost_df.itertuples(index=False):
        start_heading_deg, end_heading_deg = edge_heading_degrees(row.geometry, reverse=False)
        forward = Arc(
            tail=row.u,
            head=row.v,
            edge_idx=int(row.edge_idx),
            edge_id=str(row.edge_id),
            length_m=float(row.edge_length_m),
            time_sec=float(row.time_sec),
            rs_value=float(row.rs_value),
            risk_cost=float(row.risk_cost),
            comfort_cost=float(row.comfort_cost),
            time_norm=float(row.time_norm),
            risk_norm=float(row.risk_norm),
            comfort_norm=float(row.comfort_norm),
            risk_level=str(row.risk_level),
            road_class=str(row.road_class),
            start_heading_deg=float(start_heading_deg),
            end_heading_deg=float(end_heading_deg),
            geometry=row.geometry,
        )
        arcs.append(forward)
        if not bool(row.oneway_bool):
            rev_start_heading_deg, rev_end_heading_deg = edge_heading_degrees(row.geometry, reverse=True)
            arcs.append(
                Arc(
                    tail=row.v,
                    head=row.u,
                    edge_idx=int(row.edge_idx),
                    edge_id=str(row.edge_id),
                    length_m=float(row.edge_length_m),
                    time_sec=float(row.time_sec),
                    rs_value=float(row.rs_value),
                    risk_cost=float(row.risk_cost),
                    comfort_cost=float(row.comfort_cost),
                    time_norm=float(row.time_norm),
                    risk_norm=float(row.risk_norm),
                    comfort_norm=float(row.comfort_norm),
                    risk_level=str(row.risk_level),
                    road_class=str(row.road_class),
                    start_heading_deg=float(rev_start_heading_deg),
                    end_heading_deg=float(rev_end_heading_deg),
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


def shortest_path(
    arcs: list[Arc],
    start_node: object,
    end_node: object,
    weight_name: str | None = None,
    weight_getter=None,
) -> tuple[list[Arc], float]:
    if weight_getter is None:
        if weight_name is None:
            raise ValueError("weight_name or weight_getter is required")
        weight_getter = lambda arc: getattr(arc, weight_name)
    adjacency = build_weighted_adjacency(arcs, weight_getter)
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
    corridor_paths: list[list[Arc]] = []
    for lambda_value in SUBGRAPH_CORRIDOR_LAMBDAS:
        path_arcs, _ = shortest_path(
            arcs,
            start_node,
            end_node,
            weight_getter=lambda arc, value=lambda_value: (
                (1.0 - effective_lambda(value)) * arc.time_norm
                + effective_lambda(value) * arc.risk_norm
                + ADVANCED_PENALTY_WEIGHT * arc.comfort_norm * (0.3 + 0.9 * effective_lambda(value))
            ),
        )
        corridor_paths.append(path_arcs)

    time_path_arcs = corridor_paths[0]
    risk_path_arcs = corridor_paths[-1]
    baseline_time_distance_m = sum(arc.length_m for arc in time_path_arcs)
    baseline_risk_distance_m = sum(arc.length_m for arc in risk_path_arcs)
    max_corridor_distance_m = max(sum(arc.length_m for arc in path) for path in corridor_paths)
    cutoff = max_corridor_distance_m * SUBGRAPH_RADIUS_RATIO + SUBGRAPH_BUFFER_M

    extreme_nodes = {start_node, end_node}
    for path_arcs in corridor_paths:
        for arc in path_arcs:
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
        "corridor_path_count": len(corridor_paths),
        "corridor_seed_node_count": len(extreme_nodes),
        "subgraph_node_count": len(eligible_nodes),
        "subgraph_edge_count": len(subgraph_arcs),
        "full_graph_node_count": len({arc.tail for arc in arcs} | {arc.head for arc in arcs}),
        "full_graph_edge_count": len(arcs),
        "used_full_graph_fallback": False,
    }
    return subgraph_arcs, diagnostics


def arc_cost(arc: Arc, lambda_value: float) -> int:
    adjusted_lambda = effective_lambda(lambda_value)
    comfort_weight = ADVANCED_PENALTY_WEIGHT * (0.3 + 0.9 * adjusted_lambda)
    raw_cost = (1.0 - adjusted_lambda) * arc.time_norm + adjusted_lambda * arc.risk_norm + comfort_weight * arc.comfort_norm
    return max(0, int(round(raw_cost * COST_SCALE)))


def turn_transition_cost(prev_arc: Arc | None, next_arc: Arc, lambda_value: float) -> int:
    if prev_arc is None:
        return 0
    turn_angle = angle_difference_degrees(prev_arc.end_heading_deg, next_arc.start_heading_deg)
    if turn_angle <= TURN_FREE_ANGLE_DEG:
        return 0

    adjusted_lambda = effective_lambda(lambda_value)
    normalized_turn = ((turn_angle - TURN_FREE_ANGLE_DEG) / (180.0 - TURN_FREE_ANGLE_DEG)) ** 1.35
    turn_weight = TURN_PENALTY_SCALE * (0.85 + 0.35 * adjusted_lambda)
    penalty = normalized_turn * turn_weight
    if turn_angle >= UTURN_ANGLE_DEG:
        penalty *= UTURN_PENALTY_MULTIPLIER
    return max(0, int(round(penalty * COST_SCALE)))


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
    adjacency: dict[object, list[tuple[int, Arc]]] = defaultdict(list)
    for arc_idx, arc in enumerate(subgraph_arcs):
        adjacency[arc.tail].append((arc_idx, arc))

    start_state = (start_node, -1)
    queue: list[tuple[int, object, int]] = [(0, start_node, -1)]
    distances: dict[tuple[object, int], int] = {start_state: 0}
    previous: dict[tuple[object, int], tuple[tuple[object, int], Arc]] = {}
    final_state: tuple[object, int] | None = None

    while queue:
        current_cost, node, prev_arc_idx = heapq.heappop(queue)
        state = (node, prev_arc_idx)
        if current_cost > distances.get(state, math.inf):
            continue
        if node == end_node:
            final_state = state
            break

        prev_arc = subgraph_arcs[prev_arc_idx] if prev_arc_idx >= 0 else None
        for next_arc_idx, next_arc in adjacency.get(node, []):
            transition_cost = arc_cost(next_arc, lambda_value) + turn_transition_cost(prev_arc, next_arc, lambda_value)
            next_state = (next_arc.head, next_arc_idx)
            new_cost = current_cost + transition_cost
            if new_cost < distances.get(next_state, math.inf):
                distances[next_state] = new_cost
                previous[next_state] = (state, next_arc)
                heapq.heappush(queue, (new_cost, next_arc.head, next_arc_idx))

    if final_state is None:
        return OptimizationResult(
            lambda_value=lambda_value,
            solver="dijkstra_turn_penalty",
            optimization_status="Infeasible",
            objective_value=float("nan"),
            normalized_objective=float("nan"),
            subgraph_node_count=len(node_ids),
            subgraph_edge_count=len(subgraph_arcs),
            path_arcs=[],
            path_valid=False,
            solver_selected_arc_count=0,
        )

    path_arcs: list[Arc] = []
    trace_state = final_state
    while trace_state != start_state:
        prev_state, arc = previous[trace_state]
        path_arcs.append(arc)
        trace_state = prev_state
    path_arcs.reverse()
    objective_value = distances[final_state] / COST_SCALE
    return OptimizationResult(
        lambda_value=lambda_value,
        solver="dijkstra_turn_penalty",
        optimization_status="Optimal",
        objective_value=objective_value,
        normalized_objective=objective_value,
        subgraph_node_count=len(node_ids),
        subgraph_edge_count=len(subgraph_arcs),
        path_arcs=path_arcs,
        path_valid=True,
        solver_selected_arc_count=len(path_arcs),
    )


def route_difference(base_arcs: list[Arc], other_arcs: list[Arc]) -> float:
    base_set = {(arc.tail, arc.head, arc.edge_id) for arc in base_arcs}
    other_set = {(arc.tail, arc.head, arc.edge_id) for arc in other_arcs}
    union_count = len(base_set | other_set)
    if union_count == 0:
        return 0.0
    return 1.0 - len(base_set & other_set) / union_count


def path_signature(arcs: list[Arc]) -> str:
    if not arcs:
        return "empty"
    joined = "|".join(f"{arc.tail}>{arc.head}:{arc.edge_id}" for arc in arcs)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]


def summarize_route(
    request: RouteRequest,
    result: OptimizationResult,
    base_arcs: list[Arc],
    start_snap_m: float,
    end_snap_m: float,
    diagnostics: dict[str, float | int | bool],
    subgraph_mode: str,
    route_label: str,
    route_signature_value: str,
) -> dict[str, object]:
    counts = pd.Series([arc.risk_level for arc in result.path_arcs], dtype="object").value_counts()
    distance_m = sum(arc.length_m for arc in result.path_arcs)
    raw_time_min = sum(arc.time_sec for arc in result.path_arcs) / 60.0
    raw_exposure_sum = sum(arc.rs_value for arc in result.path_arcs)
    raw_risk_sum = sum(arc.risk_cost for arc in result.path_arcs)
    raw_comfort_sum = sum(arc.comfort_cost for arc in result.path_arcs)
    base_distance_m = sum(arc.length_m for arc in base_arcs)
    base_time_min = sum(arc.time_sec for arc in base_arcs) / 60.0
    base_exposure_sum = sum(arc.rs_value for arc in base_arcs)
    base_risk_sum = sum(arc.risk_cost for arc in base_arcs)
    base_comfort_sum = sum(arc.comfort_cost for arc in base_arcs)
    high_risk_edges = int(counts.get("high_risk", 0))
    path_arc_count = len(result.path_arcs)
    return {
        "case_id": request.case_id,
        "case_name": request.name,
        "start": request.start_label,
        "end": request.end_label,
        "hour": HOUR_LABELS[request.hour],
        "scenario": diagnostics["scenario"],
        "lambda": result.lambda_value,
        "route_label": route_label,
        "route_signature": route_signature_value,
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
        "exposure_sum": round(raw_exposure_sum, 2),
        "comfort_sum": round(raw_comfort_sum, 2),
        "normal_edges": int(counts.get("normal", 0)),
        "caution_edges": int(counts.get("caution", 0)),
        "risk_edges": int(counts.get("risk", 0)),
        "high_risk_edges": high_risk_edges,
        "high_risk_edge_ratio": round(high_risk_edges / path_arc_count, 3) if path_arc_count else 0.0,
        "route_diff_vs_lambda0": round(route_difference(base_arcs, result.path_arcs), 3),
        "distance_detour_ratio_vs_lambda0": round(distance_m / base_distance_m, 3) if base_distance_m > 0 else float("nan"),
        "time_increase_ratio_vs_lambda0": round(raw_time_min / base_time_min, 3) if base_time_min > 0 else float("nan"),
        "exposure_reduction_ratio_vs_lambda0": round(1.0 - (raw_exposure_sum / base_exposure_sum), 3) if base_exposure_sum > 0 else float("nan"),
        "risk_reduction_ratio_vs_lambda0": round(1.0 - (raw_risk_sum / base_risk_sum), 3) if base_risk_sum > 0 else float("nan"),
        "comfort_reduction_ratio_vs_lambda0": round(1.0 - (raw_comfort_sum / base_comfort_sum), 3) if base_comfort_sum > 0 else float("nan"),
        "start_snap_m": round(start_snap_m, 1),
        "end_snap_m": round(end_snap_m, 1),
        "path_valid": result.path_valid,
        "path_arc_count": path_arc_count,
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


def add_road_style_columns(edges: gpd.GeoDataFrame, hour: str) -> gpd.GeoDataFrame:
    styled = edges.copy()
    styled["road_class"] = styled["highway"].apply(normalize_highway_class)
    styled["risk_level_plot"] = styled[f"risk_level_{hour}"].apply(normalize_risk_level)
    styled["risk_overlay_color"] = styled["risk_level_plot"].map(RISK_COLORS).fillna(RISK_COLORS["normal"])
    return styled


def select_plot_lambdas(results_by_lambda: dict[float, OptimizationResult], requested_lambdas: list[float]) -> list[float]:
    preferred = [0.0, 0.5, 1.0]
    selected = [
        lambda_value
        for lambda_value in preferred
        if lambda_value in requested_lambdas
        and lambda_value in results_by_lambda
        and results_by_lambda[lambda_value].path_valid
    ]
    if selected:
        return selected
    priority = plot_order(requested_lambdas)
    return [
        lambda_value
        for lambda_value in priority[: min(3, len(priority))]
        if lambda_value in results_by_lambda and results_by_lambda[lambda_value].path_valid
    ]


def render_plot_lambdas(plot_lambdas: list[float]) -> list[float]:
    render_priority = [0.5, 1.0, 0.0]
    ordered = [value for value in render_priority if value in plot_lambdas]
    ordered.extend(value for value in plot_lambdas if value not in ordered)
    return ordered


def compact_route_title(request: RouteRequest) -> str:
    title = request.name.strip()
    if len(title) <= 58:
        return title
    if "->" in title:
        left, right = [part.strip() for part in title.split("->", 1)]
        return f"{left}\n-> {right}"
    return title


def build_route_summary_lines(plot_lambdas: list[float], results_by_lambda: dict[float, OptimizationResult]) -> list[str]:
    lines: list[str] = []
    for lambda_value in plot_lambdas:
        result = results_by_lambda.get(lambda_value)
        if result is None or not result.path_valid:
            continue
        distance_km = sum(arc.length_m for arc in result.path_arcs) / 1000.0
        time_min = sum(arc.time_sec for arc in result.path_arcs) / 60.0
        high_risk_count = sum(1 for arc in result.path_arcs if arc.risk_level == "high_risk")
        lines.append(f"l={lambda_value:g}  {distance_km:.2f}km | {time_min:.1f}m | HR {high_risk_count}")
    return lines


def compute_viewport_bounds(
    route_bounds: tuple[float, float, float, float],
    network_bounds: tuple[float, float, float, float],
    start_point: Point,
    end_point: Point,
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = route_bounds
    net_minx, net_miny, net_maxx, net_maxy = network_bounds

    minx = min(minx, start_point.x, end_point.x)
    miny = min(miny, start_point.y, end_point.y)
    maxx = max(maxx, start_point.x, end_point.x)
    maxy = max(maxy, start_point.y, end_point.y)

    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    pad_x = max(width * 0.12, 220.0)
    pad_y = max(height * 0.12, 220.0)

    minx -= pad_x
    maxx += pad_x
    miny -= pad_y
    maxy += pad_y

    # Keep the viewport balanced so empty outer margins do not dominate one side.
    width = maxx - minx
    height = maxy - miny
    target_span = max(width, height)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    minx = cx - target_span / 2.0
    maxx = cx + target_span / 2.0
    miny = cy - target_span / 2.0
    maxy = cy + target_span / 2.0

    # Clamp to the available road network extent so the outer frame still shows roads.
    if minx < net_minx:
        shift = net_minx - minx
        minx += shift
        maxx += shift
    if maxx > net_maxx:
        shift = maxx - net_maxx
        minx -= shift
        maxx -= shift
    if miny < net_miny:
        shift = net_miny - miny
        miny += shift
        maxy += shift
    if maxy > net_maxy:
        shift = maxy - net_maxy
        miny -= shift
        maxy -= shift

    minx = max(minx, net_minx)
    miny = max(miny, net_miny)
    maxx = min(maxx, net_maxx)
    maxy = min(maxy, net_maxy)
    return minx, miny, maxx, maxy


def save_route_map(
    edges: gpd.GeoDataFrame,
    request: RouteRequest,
    results_by_lambda: dict[float, OptimizationResult],
    config: RunConfig,
    diagnostics: dict[str, float | int | bool],
) -> Path:
    output_path = request_output_path(request, config)
    fig, ax = plt.subplots(figsize=(14, 14), dpi=500)
    fig.patch.set_facecolor(MAP_BACKGROUND)
    ax.set_facecolor(MAP_BACKGROUND)

    styled_edges = add_road_style_columns(edges, request.hour)
    for road_class in ROAD_STYLE_ORDER:
        class_edges = styled_edges[styled_edges["road_class"] == road_class]
        if class_edges.empty:
            continue
        road_style = ROAD_STYLE_CONFIG[road_class]
        class_edges.plot(
            ax=ax,
            color=road_style["casing"],
            linewidth=road_style["casing_width"],
            alpha=road_style["alpha"],
            zorder=1,
        )
        class_edges.plot(
            ax=ax,
            color=road_style["fill"],
            linewidth=road_style["fill_width"],
            alpha=min(1.0, road_style["alpha"] + 0.03),
            zorder=2,
        )

    risk_edges = styled_edges[styled_edges["risk_level_plot"] != "normal"]
    if not risk_edges.empty:
        risk_linewidth = risk_edges["road_class"].map(
            lambda road_class: max(0.45, ROAD_STYLE_CONFIG.get(road_class, ROAD_STYLE_CONFIG["other"])["fill_width"] * 0.34)
        )
        risk_edges.plot(
            ax=ax,
            color=risk_edges["risk_overlay_color"],
            linewidth=risk_linewidth,
            alpha=0.75,
            zorder=3,
        )

    plot_lambdas = select_plot_lambdas(results_by_lambda, config.lambdas)
    for lambda_value in render_plot_lambdas(plot_lambdas):
        route_gdf = arcs_to_gdf(results_by_lambda[lambda_value].path_arcs, edges.crs)
        if route_gdf.empty:
            continue
        route_color = ROUTE_COLORS.get(lambda_value, "#333333")
        route_width = ROUTE_LINEWIDTHS.get(lambda_value, 3.4)
        route_gdf.plot(
            ax=ax,
            color="white",
            linewidth=route_width + 1.6,
            linestyle=ROUTE_LINESTYLES.get(lambda_value, "solid"),
            alpha=0.62,
            zorder=max(4, ROUTE_ZORDERS.get(lambda_value, 5) - 1),
        )
        route_gdf.plot(
            ax=ax,
            color=route_color,
            linewidth=route_width,
            linestyle=ROUTE_LINESTYLES.get(lambda_value, "solid"),
            alpha=0.78,
            zorder=ROUTE_ZORDERS.get(lambda_value, 5),
        )

    start_point = Point(
        float(diagnostics.get("start_snap_x", lon_lat_to_projected_point(request.start_lon_lat).x)),
        float(diagnostics.get("start_snap_y", lon_lat_to_projected_point(request.start_lon_lat).y)),
    )
    end_point = Point(
        float(diagnostics.get("end_snap_x", lon_lat_to_projected_point(request.end_lon_lat).x)),
        float(diagnostics.get("end_snap_y", lon_lat_to_projected_point(request.end_lon_lat).y)),
    )
    ax.scatter(start_point.x, start_point.y, s=150, marker="o", color="#00a651", edgecolor="black", linewidth=1.0, zorder=10)
    ax.scatter(end_point.x, end_point.y, s=180, marker="*", color="#7b1fa2", edgecolor="black", linewidth=0.9, zorder=10)
    ax.annotate("Start", (start_point.x, start_point.y), xytext=(8, 8), textcoords="offset points", fontsize=11, fontweight="bold", color="#007a3d", zorder=11)
    ax.annotate("End", (end_point.x, end_point.y), xytext=(8, 8), textcoords="offset points", fontsize=11, fontweight="bold", color="#5e1678", zorder=11)

    route_gdfs = [arcs_to_gdf(result.path_arcs, edges.crs) for result in results_by_lambda.values() if result.path_arcs]
    if route_gdfs:
        route_bounds = pd.concat(route_gdfs, ignore_index=True).total_bounds
        minx, miny, maxx, maxy = compute_viewport_bounds(
            tuple(route_bounds),
            tuple(edges.total_bounds),
            start_point,
            end_point,
        )
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    ax.set_title(compact_route_title(request), fontsize=16, pad=30)
    subtitle = f"{HOUR_LABELS[request.hour]} / {config.scenario}"
    ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=11.5, color="#5f5a53")
    summary_lines = build_route_summary_lines(plot_lambdas, results_by_lambda)
    if summary_lines:
        ax.text(
            0.02,
            0.98,
            "Routes\n" + "\n".join(summary_lines) + "\n\nbg risk: yellow caution / orange risk / red high-risk",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.3,
            color="#2f2a24",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fffdf8", "edgecolor": "#d9d1c3", "alpha": 0.96},
            zorder=20,
        )
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
        for value in plot_lambdas
    ]
    point_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#00a651", markeredgecolor="black", markersize=8, label="Start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#7b1fa2", markeredgecolor="black", markersize=11, label="End"),
    ]
    ax.legend(
        handles=route_handles + point_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=min(len(route_handles) + len(point_handles), 5),
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
    start_node, start_snap_m, start_snap_point = nearest_node(nodes, request.start_lon_lat)
    end_node, end_snap_m, end_snap_point = nearest_node(nodes, request.end_lon_lat)
    cost_df = prepare_edge_costs(edges, request, config.scenario)
    arcs = build_arcs(cost_df)
    subgraph_arcs, diagnostics = build_subgraph_arcs(arcs, start_node, end_node)
    diagnostics["scenario"] = config.scenario
    diagnostics["start_snap_x"] = float(start_snap_point.x)
    diagnostics["start_snap_y"] = float(start_snap_point.y)
    diagnostics["end_snap_x"] = float(end_snap_point.x)
    diagnostics["end_snap_y"] = float(end_snap_point.y)

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
    route_labels_by_signature: dict[str, str] = {}
    for lambda_value in config.lambdas:
        result = results_by_lambda[lambda_value]
        signature = path_signature(result.path_arcs)
        route_labels_by_signature.setdefault(signature, f"R{len(route_labels_by_signature) + 1}")
    diagnostics["unique_route_count"] = len(route_labels_by_signature)
    summary_rows = [
        summarize_route(
            request,
            results_by_lambda[lambda_value],
            base_arcs,
            start_snap_m,
            end_snap_m,
            diagnostics,
            config.subgraph_mode,
            route_labels_by_signature[path_signature(results_by_lambda[lambda_value].path_arcs)],
            path_signature(results_by_lambda[lambda_value].path_arcs),
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
    log(f"  unique route count: {diagnostics['unique_route_count']}")

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
