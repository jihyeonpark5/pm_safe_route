"""Out-of-fold evaluation of logistic predictions AND the derived RS index.

Raw features are normalized within each training fold. Spatial blocks test
transfer to held-out 1 km blocks, without a separation buffer. This is not
temporal or external validation, nor validation of hour-specific accident risk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CONTINUOUS = ["intersection_count", "road_type_score", "pedestrian_10h", "pedestrian_18h", "pedestrian_22h"]
BINARY = ["is_school_zone", "is_hospital_zone", "is_elderly_zone"]
FEATURES = ["intersection_count", "road_type_score", *BINARY, "pedestrian_avg"]


def feature_matrix(frame, normalizer, fit=False):
    values = frame[CONTINUOUS].to_numpy(dtype=float)
    normalized = normalizer.fit_transform(values) if fit else normalizer.transform(values)
    return np.column_stack([normalized[:, :2], frame[BINARY].to_numpy(dtype=float), normalized[:, 2:].mean(axis=1)])


def fit_predict(train, test):
    normalizer = MinMaxScaler(clip=True)
    x_train = feature_matrix(train, normalizer, fit=True)
    x_test = feature_matrix(test, normalizer)
    standardizer = StandardScaler().fit(x_train)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(standardizer.transform(x_train), train.pm_accident)
    coefficients = model.coef_[0]
    denominator = np.abs(coefficients).sum()
    if denominator <= 0:
        raise ValueError("Cannot derive RS from zero coefficients")
    weights = np.abs(coefficients) / denominator
    return model.predict_proba(standardizer.transform(x_test))[:, 1], np.clip(x_test @ weights, 0, 1), coefficients, weights


def ranking_metrics(y, scores, ids, fraction=0.1):
    # Exact top-k, with stable grid_id tie breaking; no test labels used to rank.
    k = max(1, int(np.ceil(len(y) * fraction)))
    selected = np.lexsort((np.asarray(ids, dtype=str), -np.asarray(scores)))[:k]
    hits = int(np.asarray(y)[selected].sum())
    positives = int(np.asarray(y).sum())
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "prevalence": float(np.mean(y)),
        "top10_selected": k,
        "top10_accidents_captured": hits,
        "top10_recall": hits / positives,
        "top10_precision": hits / k,
        "top10_lift": (hits / k) / np.mean(y),
    }


def evaluate(frame, block_size=1000, folds=5, seed=42):
    frame = frame.reset_index(drop=True)
    if folds < 2 or block_size <= 0:
        raise ValueError("folds >= 2 and block_size > 0 required")
    y = frame.pm_accident.to_numpy(dtype=int)
    centers = frame.geometry.centroid
    groups = (np.floor(centers.x / block_size).astype(int).astype(str)
              + ":" + np.floor(centers.y / block_size).astype(int).astype(str)).to_numpy()
    rows, predictions, coefficients = [], [], []
    strategies = {
        "random": StratifiedKFold(folds, shuffle=True, random_state=seed),
        "spatial": StratifiedGroupKFold(folds, shuffle=True, random_state=seed),
    }
    for strategy, splitter in strategies.items():
        for fold, (train_idx, test_idx) in enumerate(splitter.split(frame, y, groups if strategy == "spatial" else None), 1):
            train, test = frame.iloc[train_idx], frame.iloc[test_idx]
            if len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
                raise ValueError(f"{strategy} fold {fold} has a single class; revise block size/folds")
            if strategy == "spatial" and set(groups[train_idx]) & set(groups[test_idx]):
                raise AssertionError("Spatial train/test block overlap")
            probability, rs, beta, weights = fit_predict(train, test)
            for name, scores in [("logistic", probability), ("derived_rs", rs)]:
                row = dict(strategy=strategy, fold=fold, score=name, train_n=len(train), test_n=len(test),
                           train_blocks=len(set(groups[train_idx])), test_blocks=len(set(groups[test_idx])),
                           **ranking_metrics(y[test_idx], scores, test.grid_id))
                if name == "logistic":
                    row.update(precision_at_05=float(precision_score(y[test_idx], scores >= .5, zero_division=0)),
                               recall_at_05=float(recall_score(y[test_idx], scores >= .5)))
                rows.append(row)
            predictions.append(pd.DataFrame({"strategy": strategy, "fold": fold, "grid_id": test.grid_id.to_numpy(),
                                             "spatial_block": groups[test_idx], "target": y[test_idx],
                                             "logistic_score": probability, "derived_rs": rs * 100}))
            coefficients.extend(dict(strategy=strategy, fold=fold, feature=feature, coefficient=float(b), rs_weight=float(w))
                                for feature, b, w in zip(FEATURES, beta, weights))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True), pd.DataFrame(coefficients)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size-m", type=float, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/output/validation_v1")
    args = parser.parse_args()
    source = ROOT / "data/processed/merged_grid_variables.parquet"
    frame = gpd.read_parquet(source)
    if frame.crs is None or frame.crs.to_epsg() != 5179:
        raise ValueError("Raw grid must use EPSG:5179")
    frame = frame.loc[frame.road_type_score.notna()].copy()
    required = [*CONTINUOUS, *BINARY, "pm_accident"]
    if frame.grid_id.duplicated().any() or not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Duplicate IDs or missing/non-finite raw features")
    if not set(frame.pm_accident.unique()).issubset({0, 1}):
        raise ValueError("Target must be binary")
    metrics, predictions, coefficients = evaluate(frame, args.block_size_m, args.folds, args.seed)
    summary = metrics.groupby(["strategy", "score"])[["roc_auc", "average_precision", "top10_recall", "top10_precision", "top10_lift"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in [("fold_metrics", metrics), ("oof_predictions", predictions), ("fold_coefficients", coefficients), ("summary", summary.reset_index())]:
        result.to_csv(args.output_dir / f"{name}.csv", index=False)
    metadata = {
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "rows": len(frame),
        "positive_grids": int(frame.pm_accident.sum()), "block_size_m": args.block_size_m,
        "folds": args.folds, "seed": args.seed, "normalization": "training-fold MinMaxScaler(clip=True), then training-fold StandardScaler",
        "target": "pooled historical grid-level accident presence; not hourly risk",
        "limits": ["No temporal holdout", "No buffer between spatial blocks", "RS is not calibrated accident probability",
                   "Top 10% computed separately within each held-out fold", "Scores do not measure causal effects"],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
