from __future__ import annotations

import heapq
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import geopandas as gpd
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D
from pyproj import Transformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

EDGE_RISK_PATH = PROCESSED_DIR / "edge_risk_score_no_vehicle.parquet"
NODES_PATH = PROCESSED_DIR / "04_osm_nodes_raw.parquet"

SCENARIO = "S1"
LAMBDAS = [0.0, 0.5, 1.0]
CANDIDATE_LAMBDAS = [round(value / 10.0, 1) for value in range(0, 11)]
RISK_WEIGHT = 0.75
RISK_WEIGHT_BY_LAMBDA = {
    0.0: 0.75,
    0.5: 0.8,
    1.0: 0.75,
}
RISK_WEIGHT_CANDIDATES = [0.8, 0.75, 0.5, 0.35, 0.2, 0.0]
MAX_DETOUR_RATIO = {
    0.5: 1.45,
    1.0: 1.50,
}
REPRESENTATIVE_MID_LAMBDA = 0.5
HOUR_LABELS = {"10h": "10시", "18h": "18시", "22h": "22시"}

RISK_COLORS = {
    "일반": "#d9d9d9",
    "주의": "#ffd54f",
    "위험": "#fb8c00",
    "고위험": "#d32f2f",
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
    0.5: 2.8,
    1.0: 4.2,
}
ROUTE_ZORDERS = {
    0.0: 4,
    0.5: 6,
    1.0: 5,
}


@dataclass(frozen=True)
class RouteCase:
    case_id: int
    slug: str
    name: str
    start_label: str
    end_label: str
    start_lon_lat: tuple[float, float]
    end_lon_lat: tuple[float, float]
    hour: str
    scenario: str = SCENARIO


CASES = [
    RouteCase(
        case_id=1,
        slug="cheongdam_to_samseong",
        name="사례 1: 청담동 주민센터 -> 삼성1동 주민센터",
        start_label="청담동 주민센터",
        end_label="삼성1동 주민센터",
        start_lon_lat=(127.04931, 37.52503),
        end_lon_lat=(127.06278, 37.51444),
        hour="18h",
    ),
    RouteCase(
        case_id=2,
        slug="sinsa_to_cheongdam",
        name="사례 2: 신사동 주민센터 -> 청담동 주민센터",
        start_label="신사동 주민센터",
        end_label="청담동 주민센터",
        start_lon_lat=(127.02283, 37.52401),
        end_lon_lat=(127.04931, 37.52503),
        hour="22h",
    ),
    RouteCase(
        case_id=3,
        slug="gaepo_to_gangnam_office",
        name="사례 3: 개포1동 주민센터 -> 강남구청",
        start_label="개포1동 주민센터",
        end_label="강남구청",
        start_lon_lat=(127.0543931, 37.48328034),
        end_lon_lat=(127.04752, 37.51767),
        hour="18h",
    ),
]


def log(message: str) -> None:
    print(message)


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not EDGE_RISK_PATH.exists():
        raise FileNotFoundError(
            f"{EDGE_RISK_PATH} 파일이 없습니다. 먼저 edge risk score를 생성해야 합니다."
        )
    if not NODES_PATH.exists():
        raise FileNotFoundError(f"{NODES_PATH} 파일이 없습니다.")

    edges = gpd.read_parquet(EDGE_RISK_PATH)
    nodes = gpd.read_parquet(NODES_PATH)
    if edges.crs is None:
        edges = edges.set_crs("EPSG:5179")
    if nodes.crs is None:
        nodes = nodes.set_crs("EPSG:5179")
    return edges, nodes


def nearest_node(nodes: gpd.GeoDataFrame, lon_lat: tuple[float, float]) -> tuple[object, float]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    x, y = transformer.transform(*lon_lat)
    point = gpd.points_from_xy([x], [y], crs="EPSG:5179")[0]
    distances = nodes.geometry.distance(point)
    nearest_idx = distances.idxmin()
    return nodes.loc[nearest_idx, "osmid"], float(distances.loc[nearest_idx])


def prepare_edge_costs(edges: gpd.GeoDataFrame, case: RouteCase) -> gpd.GeoDataFrame:
    hour = case.hour
    speed_col = f"speed_{case.scenario}_{hour}"
    rs_col = f"RS_{hour}"
    required_cols = {"edge_id", "u", "v", "oneway", "geometry", "length", speed_col, rs_col}
    missing_cols = required_cols - set(edges.columns)
    if missing_cols:
        raise ValueError(f"edge risk data missing columns: {sorted(missing_cols)}")

    cost_df = edges.copy().reset_index(drop=True)
    cost_df["edge_idx"] = cost_df.index
    cost_df["edge_length_m"] = pd.to_numeric(cost_df["length"], errors="coerce")
    invalid_length = cost_df["edge_length_m"].isna() | cost_df["edge_length_m"].le(0)
    cost_df.loc[invalid_length, "edge_length_m"] = cost_df.loc[invalid_length].geometry.length

    speed_kmh = pd.to_numeric(cost_df[speed_col], errors="coerce").fillna(25).clip(lower=1)
    speed_mps = speed_kmh * 1000.0 / 3600.0
    cost_df["time_sec"] = cost_df["edge_length_m"] / speed_mps
    cost_df["risk_cost"] = (
        pd.to_numeric(cost_df[rs_col], errors="coerce").fillna(0.0)
        / 100.0
        * cost_df["edge_length_m"]
    )

    max_time = cost_df["time_sec"].max()
    max_risk = cost_df["risk_cost"].max()
    cost_df["time_norm"] = cost_df["time_sec"] / max_time if max_time > 0 else 0.0
    cost_df["risk_norm"] = cost_df["risk_cost"] / max_risk if max_risk > 0 else 0.0
    return cost_df


def build_adjacency(
    cost_df: pd.DataFrame,
    lambda_value: float,
    risk_weight: float = RISK_WEIGHT,
) -> dict[object, list[tuple[object, float, int]]]:
    adjacency: dict[object, list[tuple[object, float, int]]] = defaultdict(list)
    for row in cost_df.itertuples(index=False):
        cost = float(row.time_norm) + lambda_value * risk_weight * float(row.risk_norm)
        adjacency[row.u].append((row.v, cost, int(row.edge_idx)))
        if not bool(row.oneway):
            adjacency[row.v].append((row.u, cost, int(row.edge_idx)))
    return adjacency


def default_risk_weight(lambda_value: float) -> float:
    return RISK_WEIGHT_BY_LAMBDA.get(lambda_value, RISK_WEIGHT)


def max_detour_ratio(lambda_value: float) -> float | None:
    if lambda_value <= 0.0:
        return None
    if lambda_value <= 0.5:
        return 1.0 + (MAX_DETOUR_RATIO[0.5] - 1.0) * (lambda_value / 0.5)
    return MAX_DETOUR_RATIO[0.5] + (
        (MAX_DETOUR_RATIO[1.0] - MAX_DETOUR_RATIO[0.5]) * ((lambda_value - 0.5) / 0.5)
    )


def dijkstra(
    adjacency: dict[object, list[tuple[object, float, int]]],
    start_node: object,
    end_node: object,
) -> list[int]:
    queue: list[tuple[float, object]] = [(0.0, start_node)]
    dist = {start_node: 0.0}
    prev: dict[object, tuple[object, int]] = {}

    while queue:
        current_dist, node = heapq.heappop(queue)
        if node == end_node:
            break
        if current_dist > dist.get(node, math.inf):
            continue
        for next_node, edge_cost, edge_idx in adjacency.get(node, []):
            new_dist = current_dist + edge_cost
            if new_dist < dist.get(next_node, math.inf):
                dist[next_node] = new_dist
                prev[next_node] = (node, edge_idx)
                heapq.heappush(queue, (new_dist, next_node))

    if end_node not in prev and start_node != end_node:
        raise ValueError(f"No route found between {start_node} and {end_node}.")

    edge_indices: list[int] = []
    node = end_node
    while node != start_node:
        previous_node, edge_idx = prev[node]
        edge_indices.append(edge_idx)
        node = previous_node
    edge_indices.reverse()
    return edge_indices


def route_distance_m(cost_df: pd.DataFrame, edge_indices: list[int]) -> float:
    return float(cost_df.loc[edge_indices, "edge_length_m"].sum())


def select_route_with_detour_limit(
    cost_df: pd.DataFrame,
    start_node: object,
    end_node: object,
    lambda_value: float,
    baseline_distance_m: float,
    baseline_indices: list[int],
) -> tuple[list[int], float, bool]:
    max_ratio = max_detour_ratio(lambda_value)
    primary_risk_weight = default_risk_weight(lambda_value)
    if max_ratio is None or baseline_distance_m <= 0:
        adjacency = build_adjacency(cost_df, lambda_value, primary_risk_weight)
        return dijkstra(adjacency, start_node, end_node), primary_risk_weight, False

    candidate_weights = [primary_risk_weight] + [
        weight for weight in RISK_WEIGHT_CANDIDATES if weight != primary_risk_weight
    ]
    for risk_weight in candidate_weights:
        adjacency = build_adjacency(cost_df, lambda_value, risk_weight)
        edge_indices = dijkstra(adjacency, start_node, end_node)
        distance_ratio = route_distance_m(cost_df, edge_indices) / baseline_distance_m
        if distance_ratio <= max_ratio:
            return edge_indices, risk_weight, risk_weight != primary_risk_weight

    return baseline_indices, 0.0, True


def summarize_route(
    case: RouteCase,
    route_edges: gpd.GeoDataFrame,
    lambda_value: float,
    baseline_risk_sum: float,
    baseline_distance_m: float,
    route_diff_from_lambda0: float,
    effective_risk_weight: float,
    detour_limited: bool,
    same_as_lambda0: bool,
    same_as_lambda1: bool,
) -> dict[str, object]:
    level_col = f"risk_level_{case.hour}"
    counts = route_edges[level_col].astype(str).value_counts()
    risk_sum = float(route_edges["risk_cost"].sum())
    distance_m = float(route_edges["edge_length_m"].sum())
    distance_ratio = distance_m / baseline_distance_m if baseline_distance_m > 0 else 0.0
    risk_reduction_pct = (
        (baseline_risk_sum - risk_sum) / baseline_risk_sum * 100.0
        if baseline_risk_sum > 0
        else 0.0
    )
    return {
        "case_id": case.case_id,
        "case_name": case.name,
        "start": case.start_label,
        "end": case.end_label,
        "hour": HOUR_LABELS[case.hour],
        "scenario": case.scenario,
        "lambda": lambda_value,
        "distance_km": round(distance_m / 1000.0, 3),
        "distance_ratio_vs_lambda0": round(distance_ratio, 3),
        "model_time_min": round(float(route_edges["time_sec"].sum()) / 60.0, 2),
        "risk_sum": round(risk_sum, 2),
        "risk_reduction_vs_lambda0_pct": round(risk_reduction_pct, 1),
        "effective_risk_weight": round(effective_risk_weight, 2),
        "detour_limited": bool(detour_limited),
        "same_as_lambda0": bool(same_as_lambda0),
        "same_as_lambda1": bool(same_as_lambda1),
        "normal_edges": int(counts.get("일반", 0)),
        "caution_edges": int(counts.get("주의", 0)),
        "risk_edges": int(counts.get("위험", 0)),
        "high_risk_edges": int(counts.get("고위험", 0)),
        "edge_count": int(len(route_edges)),
        "route_diff_vs_lambda0": round(route_diff_from_lambda0, 3),
    }


def route_difference(base_indices: list[int], other_indices: list[int]) -> float:
    base_set = set(base_indices)
    other_set = set(other_indices)
    union_count = len(base_set | other_set)
    if union_count == 0:
        return 0.0
    return 1.0 - len(base_set & other_set) / union_count


def choose_representative_mid_lambda(
    candidate_indices: dict[float, list[int]],
    candidate_layers: dict[float, gpd.GeoDataFrame],
) -> float:
    lambda0_indices = candidate_indices[0.0]
    lambda1_indices = candidate_indices[1.0]
    baseline_risk_sum = float(candidate_layers[0.0]["risk_cost"].sum())
    baseline_distance_m = float(candidate_layers[0.0]["edge_length_m"].sum())

    best_lambda = REPRESENTATIVE_MID_LAMBDA
    best_score = -math.inf
    for lambda_value in CANDIDATE_LAMBDAS:
        if lambda_value in {0.0, 1.0}:
            continue

        edge_indices = candidate_indices[lambda_value]
        diff_from_time = route_difference(lambda0_indices, edge_indices)
        diff_from_safety = route_difference(lambda1_indices, edge_indices)
        if diff_from_time == 0.0 or diff_from_safety == 0.0:
            continue

        route_edges = candidate_layers[lambda_value]
        distance_m = float(route_edges["edge_length_m"].sum())
        risk_sum = float(route_edges["risk_cost"].sum())
        distance_ratio = distance_m / baseline_distance_m if baseline_distance_m > 0 else 1.0
        risk_reduction = (
            (baseline_risk_sum - risk_sum) / baseline_risk_sum if baseline_risk_sum > 0 else 0.0
        )

        # Prefer a visible but balanced compromise route near the middle of the lambda range.
        score = (
            min(diff_from_time, diff_from_safety) * 2.0
            + risk_reduction
            - abs(lambda_value - REPRESENTATIVE_MID_LAMBDA) * 0.25
            - max(distance_ratio - 1.35, 0.0)
        )
        if score > best_score:
            best_score = score
            best_lambda = lambda_value

    return best_lambda


def case_output_path(case: RouteCase) -> Path:
    return OUTPUT_DIR / f"route_case_{case.case_id}_{case.slug}_{case.hour}_{case.scenario}.png"


def save_case_map(
    edges: gpd.GeoDataFrame,
    case: RouteCase,
    route_layers: dict[float, gpd.GeoDataFrame],
    route_meta: dict[float, dict[str, object]],
) -> Path:
    output_path = case_output_path(case)
    fig, ax = plt.subplots(figsize=(14, 14), dpi=500)
    color_series = edges[f"risk_level_{case.hour}"].astype(str).map(RISK_COLORS).fillna("#eeeeee")
    edges.plot(ax=ax, color=color_series, linewidth=0.5, alpha=0.65)

    for lambda_value in [0.0, 1.0, 0.5]:
        route_edges = route_layers[lambda_value]
        route_edges.plot(
            ax=ax,
            color=ROUTE_COLORS[lambda_value],
            linewidth=ROUTE_LINEWIDTHS[lambda_value],
            linestyle=ROUTE_LINESTYLES[lambda_value],
            alpha=0.9,
            label=f"λ={float(route_meta[lambda_value]['actual_lambda']):g}",
            zorder=ROUTE_ZORDERS[lambda_value],
        )

    route_bounds = pd.concat(route_layers.values(), ignore_index=True).total_bounds
    minx, miny, maxx, maxy = route_bounds
    pad_x = max((maxx - minx) * 0.10, 150)
    pad_y = max((maxy - miny) * 0.10, 150)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_title(case.name, fontsize=19, pad=18)
    subtitle = f"{HOUR_LABELS[case.hour]} / {case.scenario} / 안전 가중치별 경로 비교"
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=13)
    ax.set_axis_off()

    route_handles = [
        Line2D(
            [0],
            [0],
            color=ROUTE_COLORS[value],
            lw=ROUTE_LINEWIDTHS[value],
            linestyle=ROUTE_LINESTYLES[value],
            label=f"λ={float(route_meta[value]['actual_lambda']):g}",
        )
        for value in LAMBDAS
    ]
    risk_handles = [
        Line2D([0], [0], color=color, lw=4, label=label)
        for label, color in RISK_COLORS.items()
    ]
    ax.legend(
        handles=route_handles + risk_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=7,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_case_routes(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    case: RouteCase,
) -> tuple[dict[float, gpd.GeoDataFrame], dict[float, dict[str, object]], list[dict[str, object]]]:
    start_node, start_snap_m = nearest_node(nodes, case.start_lon_lat)
    end_node, end_snap_m = nearest_node(nodes, case.end_lon_lat)
    cost_df = prepare_edge_costs(edges, case)

    candidate_indices: dict[float, list[int]] = {}
    candidate_layers: dict[float, gpd.GeoDataFrame] = {}
    candidate_meta: dict[float, dict[str, object]] = {}

    baseline_adjacency = build_adjacency(cost_df, 0.0, default_risk_weight(0.0))
    baseline_indices = dijkstra(baseline_adjacency, start_node, end_node)
    baseline_distance_m = route_distance_m(cost_df, baseline_indices)

    for lambda_value in CANDIDATE_LAMBDAS:
        if lambda_value == 0.0:
            edge_indices = baseline_indices
            effective_risk_weight = default_risk_weight(0.0)
            detour_limited = False
        else:
            edge_indices, effective_risk_weight, detour_limited = select_route_with_detour_limit(
                cost_df,
                start_node,
                end_node,
                lambda_value,
                baseline_distance_m,
                baseline_indices,
            )

        candidate_indices[lambda_value] = edge_indices
        candidate_meta[lambda_value] = {
            "effective_risk_weight": effective_risk_weight,
            "detour_limited": detour_limited,
        }
        route_edges = cost_df.loc[edge_indices].copy()
        candidate_layers[lambda_value] = gpd.GeoDataFrame(route_edges, geometry="geometry", crs=edges.crs)

    mid_lambda = choose_representative_mid_lambda(candidate_indices, candidate_layers)
    selected_lambdas = {
        0.0: 0.0,
        0.5: mid_lambda,
        1.0: 1.0,
    }
    route_indices = {slot: candidate_indices[actual] for slot, actual in selected_lambdas.items()}
    route_layers = {slot: candidate_layers[actual] for slot, actual in selected_lambdas.items()}
    route_meta = {slot: candidate_meta[actual] | {"actual_lambda": actual} for slot, actual in selected_lambdas.items()}

    baseline_risk_sum = float(route_layers[0.0]["risk_cost"].sum())
    lambda0_set = set(route_indices[0.0])
    lambda1_set = set(route_indices[1.0])
    summary_rows = []
    for lambda_value in LAMBDAS:
        current_set = set(route_indices[lambda_value])
        route_diff = route_difference(route_indices[0.0], route_indices[lambda_value])
        row = summarize_route(
            case,
            route_layers[lambda_value],
            float(route_meta[lambda_value]["actual_lambda"]),
            baseline_risk_sum,
            baseline_distance_m,
            route_diff,
            float(route_meta[lambda_value]["effective_risk_weight"]),
            bool(route_meta[lambda_value]["detour_limited"]),
            current_set == lambda0_set,
            current_set == lambda1_set,
        )
        row["display_slot"] = lambda_value
        row["start_snap_m"] = round(start_snap_m, 1)
        row["end_snap_m"] = round(end_snap_m, 1)
        summary_rows.append(row)

    return route_layers, route_meta, summary_rows


def main() -> int:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== Step 12: presentation route case visualization ===")

    try:
        edges, nodes = load_inputs()
        all_summary_rows: list[dict[str, object]] = []

        for case in CASES:
            log(f"\n[{case.case_id}] {case.name}")
            route_layers, route_meta, summary_rows = build_case_routes(edges, nodes, case)
            output_path = save_case_map(edges, case, route_layers, route_meta)
            all_summary_rows.extend(summary_rows)
            log(f"saved map: {output_path}")
            for row in summary_rows:
                log(
                    f"  lambda={row['lambda']}: "
                    f"{row['distance_km']:.3f}km, "
                    f"{row['model_time_min']:.2f}min, "
                    f"risk={row['risk_sum']:.2f}, "
                    f"reduction={row['risk_reduction_vs_lambda0_pct']:.1f}%"
                )

        summary = pd.DataFrame(all_summary_rows)
        summary_path = OUTPUT_DIR / f"route_demo_summary_cases_{SCENARIO}.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        log(f"\nsummary saved: {summary_path}")
        log(summary.to_string(index=False))
        return 0
    except Exception as exc:
        log(f"\n[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
