"""Build a separately versioned score layer; missing scores fail explicitly."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .model import RiskModel, load_training

HOURS = ["10h", "18h", "22h"]
SPEEDS = {"S0": [25, 25, 25, 25], "S1": [25, 20, 15, 10], "S2": [20, 15, 10, 7]}
LEVELS = ["normal", "caution", "risk", "high_risk"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_scores(root):
    root = Path(root)
    processed, output = root/"data/processed", root/"artifacts/final"
    model_path = root/"data/output/final/model/model.json"
    source = processed/"merged_grid_variables.parquet"
    model = RiskModel.load(model_path)
    if model.state["source_sha256"] != sha256(source):
        raise ValueError("Raw grids changed; rerun train before build")
    frame = load_training(source)
    risk = pd.DataFrame({"grid_id": frame.grid_id.to_numpy(), "historical_probability": model.predict(frame)})
    for hour in HOURS:
        risk[f"RS_{hour}"] = 100*model.predict(frame, hour)
    thresholds = np.quantile(risk[[f"RS_{h}" for h in HOURS]].to_numpy().ravel(), [.5, .75, .9])
    # Empty classes are permitted if scores tie; all rows must still have a class.
    for hour in HOURS:
        category = np.searchsorted(thresholds, risk[f"RS_{hour}"], side="left")
        risk[f"risk_level_{hour}"] = np.asarray(LEVELS)[category]
        for scenario, speeds in SPEEDS.items():
            risk[f"speed_{scenario}_{hour}"] = np.asarray(speeds)[category]
    base_path = processed/"edge_risk_score_no_vehicle.parquet"
    mapping_path = processed/"08_grid_road_map.parquet"
    nodes_path = processed/"04_osm_nodes_raw.parquet"
    base = gpd.read_parquet(base_path)
    # The original file supplies road geometry/attributes only; discard all old scores.
    base = base.drop(columns=[c for c in base if c.startswith(("RS_", "risk_level_", "speed_")) or c == "mapped_length"])
    mapping = pd.read_parquet(mapping_path)[["edge_id", "grid_id", "length_in_grid"]]
    mapping["length_in_grid"] = pd.to_numeric(mapping.length_in_grid, errors="raise")
    if not np.isfinite(mapping.length_in_grid).all() or mapping.length_in_grid.lt(0).any():
        raise ValueError("Mapping lengths must be finite and nonnegative")
    mapping = mapping.loc[mapping.length_in_grid.gt(0)].copy()
    if not np.isfinite(mapping.length_in_grid).all() or base.edge_id.duplicated().any():
        raise ValueError("Invalid mapping lengths or duplicate edge IDs")
    joined = mapping.merge(risk, on="grid_id", how="left", validate="many_to_one")
    if joined[[f"RS_{h}" for h in HOURS]].isna().any().any():
        raise ValueError("Mapped road segment has no model score; refusing a zero-risk fallback")
    if set(base.edge_id) != set(mapping.edge_id):
        raise ValueError("Road geometry and mapping edge IDs differ")
    length = joined.groupby("edge_id").length_in_grid.sum()
    values = pd.DataFrame({"edge_id": length.index, "mapped_length": length.to_numpy()}).set_index("edge_id")
    for hour in HOURS:
        values[f"RS_{hour}"] = (joined[f"RS_{hour}"]*joined.length_in_grid).groupby(joined.edge_id).sum()/length
        weights = joined.groupby(["edge_id", f"risk_level_{hour}"], observed=True).length_in_grid.sum().reset_index()
        weights["severity"] = weights[f"risk_level_{hour}"].map(dict(zip(LEVELS, range(4))))
        winners = weights.sort_values(["edge_id", "length_in_grid", "severity"], ascending=[True, False, False]).drop_duplicates("edge_id").set_index("edge_id")[f"risk_level_{hour}"]
        values[f"risk_level_{hour}"] = winners
        values[f"speed_S1_{hour}"] = winners.map(dict(zip(LEVELS, SPEEDS["S1"])))
    edges = base.merge(values.reset_index(), on="edge_id", how="left", validate="one_to_one")
    if edges[[f"RS_{h}" for h in HOURS]].isna().any().any():
        raise ValueError("Incomplete edge scores")
    output.mkdir(parents=True, exist_ok=True)
    risk.to_parquet(output/"grid_scores.parquet", index=False)
    edges.to_parquet(output/"edge_scores.parquet", index=False)
    manifest = dict(model_candidate=model.state["candidate"], model_sha256=sha256(model_path),
                    input_sha256={p.name: sha256(p) for p in [source, base_path, mapping_path, nodes_path]},
                    score_semantics="Hour-specific scenario index; not validated hourly or per-trip accident probability",
                    historical_probability_semantics=model.state["target"], thresholds=thresholds.tolist(),
                    grids=len(risk), edges=len(edges),
                    output_sha256={name: sha256(output/name) for name in ["grid_scores.parquet", "edge_scores.parquet"]},
                    limitations=["Grid score mapped by segment length; edge class by dominant mapped length",
                                 "Hourly pedestrian substitution is unvalidated; calibration uses average pedestrian input",
                                 "Missing scored segments raise an error"])
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest
