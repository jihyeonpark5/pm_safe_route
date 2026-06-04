from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EDGE_INPUT_PATH = PROCESSED_DIR / "06_osm_edges_filtered.parquet"
GRID_ROAD_MAP_PATH = PROCESSED_DIR / "08_grid_road_map.parquet"
RISK_GRID_PATH = PROCESSED_DIR / "risk_score_grid_no_vehicle.parquet"
OUTPUT_PATH = PROCESSED_DIR / "edge_risk_score_no_vehicle.parquet"

HOURS = ["10h", "18h", "22h"]
SCENARIO = "S1"


def log(message: str) -> None:
    print(message)


def add_edge_id(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Recreate the edge_id sequence used by scripts/04_map_grid_roads.py."""
    edges = edges.copy().reset_index(drop=True)
    if "edge_id" not in edges.columns:
        edges.insert(0, "edge_id", [f"E{idx:06d}" for idx in range(1, len(edges) + 1)])
    else:
        edges["edge_id"] = edges["edge_id"].astype("string")
    return edges


def validate_inputs(edges: gpd.GeoDataFrame, road_map: gpd.GeoDataFrame, risk: pd.DataFrame) -> None:
    required_road_map = {"edge_id", "grid_id", "length_in_grid"}
    required_risk = {"grid_id"}
    for hour in HOURS:
        required_risk.update({f"RS_{hour}", f"risk_level_{hour}", f"speed_{SCENARIO}_{hour}"})

    missing_road_map = required_road_map - set(road_map.columns)
    missing_risk = required_risk - set(risk.columns)
    if missing_road_map:
        raise ValueError(f"grid-road map missing columns: {sorted(missing_road_map)}")
    if missing_risk:
        raise ValueError(f"risk score grid missing columns: {sorted(missing_risk)}")

    mapped_edge_count = road_map["edge_id"].nunique()
    if len(edges) != mapped_edge_count:
        raise ValueError(
            "Filtered edge count and grid-road edge_id count differ. "
            f"edges={len(edges):,}, mapped edge_ids={mapped_edge_count:,}"
        )


def weighted_mode_by_length(df: pd.DataFrame, value_col: str) -> pd.Series:
    weights = (
        df.dropna(subset=[value_col])
        .groupby(["edge_id", value_col], observed=False)["length_in_grid"]
        .sum()
        .reset_index()
    )
    if weights.empty:
        return pd.Series(dtype="object", name=value_col)

    idx = weights.groupby("edge_id")["length_in_grid"].idxmax()
    result = weights.loc[idx, ["edge_id", value_col]].set_index("edge_id")[value_col]
    result.name = value_col
    return result


def build_edge_risk() -> gpd.GeoDataFrame:
    edges = add_edge_id(gpd.read_parquet(EDGE_INPUT_PATH))
    road_map = gpd.read_parquet(GRID_ROAD_MAP_PATH)
    risk = pd.read_parquet(RISK_GRID_PATH)
    validate_inputs(edges, road_map, risk)

    road_map = road_map[["edge_id", "grid_id", "length_in_grid"]].copy()
    road_map["length_in_grid"] = pd.to_numeric(road_map["length_in_grid"], errors="coerce")
    road_map = road_map.loc[road_map["length_in_grid"].gt(0)].copy()

    joined = road_map.merge(risk, on="grid_id", how="left", validate="many_to_one")
    if joined[[f"RS_{hour}" for hour in HOURS]].isna().all(axis=None):
        raise ValueError("No Risk Score values were joined to grid-road map.")

    length_sum = joined.groupby("edge_id")["length_in_grid"].sum().rename("mapped_length")
    edge_values = pd.DataFrame(index=length_sum.index)
    edge_values["mapped_length"] = length_sum

    for hour in HOURS:
        rs_col = f"RS_{hour}"
        level_col = f"risk_level_{hour}"
        speed_col = f"speed_{SCENARIO}_{hour}"

        weighted_rs_sum = (joined[rs_col] * joined["length_in_grid"]).groupby(joined["edge_id"]).sum()
        edge_values[rs_col] = (weighted_rs_sum / edge_values["mapped_length"]).replace([np.inf, -np.inf], np.nan)
        edge_values[level_col] = weighted_mode_by_length(joined, level_col)
        edge_values[speed_col] = weighted_mode_by_length(joined, speed_col)

    edge_values = edge_values.reset_index()
    result = edges.merge(edge_values, on="edge_id", how="left", validate="one_to_one")

    for hour in HOURS:
        rs_col = f"RS_{hour}"
        level_col = f"risk_level_{hour}"
        speed_col = f"speed_{SCENARIO}_{hour}"
        result[rs_col] = result[rs_col].fillna(0.0).round(4)
        result[level_col] = result[level_col].fillna("일반")
        result[speed_col] = pd.to_numeric(result[speed_col], errors="coerce").fillna(25).astype(int)

    return result


def print_summary(edge_risk: gpd.GeoDataFrame) -> None:
    log("\n=== Edge Risk Score summary ===")
    log(f"edge rows: {len(edge_risk):,}")
    log(f"output columns: {edge_risk.columns.tolist()}")
    for hour in HOURS:
        log(f"\n[{hour}]")
        log(
            f"RS min/max/mean: "
            f"{edge_risk[f'RS_{hour}'].min():.2f} / "
            f"{edge_risk[f'RS_{hour}'].max():.2f} / "
            f"{edge_risk[f'RS_{hour}'].mean():.2f}"
        )
        log(edge_risk[f"risk_level_{hour}"].value_counts().to_string())


def main() -> int:
    log("=== Step 11: no_vehicle edge Risk Score build ===")
    try:
        edge_risk = build_edge_risk()
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        edge_risk.to_parquet(OUTPUT_PATH, index=False)
        print_summary(edge_risk)
        log(f"\nSaved: {OUTPUT_PATH}")
        return 0
    except Exception as exc:
        log(f"\n[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
