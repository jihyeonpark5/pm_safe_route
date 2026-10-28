"""Signed logistic scoring with leakage-safe, grouped sigmoid calibration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

CONTINUOUS = ["intersection_count", "road_type_score", "pedestrian_10h", "pedestrian_18h", "pedestrian_22h"]
BINARY = ["is_school_zone", "is_hospital_zone", "is_elderly_zone"]
FEATURES = ["intersection_count", "road_type_score", *BINARY, "pedestrian_avg"]
CANDIDATES = ["balanced_logistic", "unweighted_logistic", "balanced_sigmoid", "unweighted_sigmoid"]


class GridFeatures(TransformerMixin, BaseEstimator):
    def fit(self, frame, y=None):
        self.scaler_ = MinMaxScaler(clip=True).fit(frame[CONTINUOUS])
        return self

    def transform(self, frame):
        normalized = self.scaler_.transform(frame[CONTINUOUS])
        pedestrian = normalized[:, 2:].mean(axis=1)
        if "_scenario_hour" in frame:
            for hour, index in [("10h", 2), ("18h", 3), ("22h", 4)]:
                mask = frame._scenario_hour.eq(hour).to_numpy()
                pedestrian[mask] = normalized[mask, index]
            if not frame._scenario_hour.isin(["10h", "18h", "22h"]).all():
                raise ValueError("Unknown scenario hour")
        return np.column_stack([normalized[:, :2], frame[BINARY].to_numpy(dtype=float), pedestrian])


def pipeline(balanced):
    return Pipeline([("features", GridFeatures()), ("standard", StandardScaler()),
                     ("logistic", LogisticRegression(class_weight="balanced" if balanced else None,
                                                      max_iter=1000, random_state=42))])


def spatial_groups(frame, block_size=1000.):
    centers = frame.geometry.centroid
    return (np.floor(centers.x/block_size).astype(int).astype(str)+":"+
            np.floor(centers.y/block_size).astype(int).astype(str)).to_numpy()


def grouped_splits(frame, folds=3, seed=42, block_size=1000.):
    groups = spatial_groups(frame, block_size)
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(frame, frame.pm_accident, groups))
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Spatial block overlap")
        if frame.iloc[train].pm_accident.nunique() != 2 or frame.iloc[test].pm_accident.nunique() != 2:
            raise ValueError("A grouped fold has only one class")
    return splits


def fit_family(train, balanced, inner_folds=3):
    estimator = pipeline(balanced)
    splits = grouped_splits(train, folds=inner_folds)
    margins = cross_val_predict(estimator, train, train.pm_accident, cv=splits, method="decision_function")
    calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=42)
    calibrator.fit(margins.reshape(-1, 1), train.pm_accident)
    estimator.fit(train, train.pm_accident)
    return estimator, calibrator


def predict_family(estimator, calibrator, frame):
    margin = estimator.decision_function(frame)
    return expit(margin), calibrator.predict_proba(margin.reshape(-1, 1))[:, 1]


def metrics(y, scores, grid_ids, probability=True):
    k = max(1, int(np.ceil(len(y)*.1)))
    order = np.lexsort((np.asarray(grid_ids, dtype=str), -np.asarray(scores)))[:k]
    hits = int(np.asarray(y)[order].sum())
    result = dict(roc_auc=float(roc_auc_score(y, scores)), average_precision=float(average_precision_score(y, scores)),
                  top10_recall=hits/int(np.sum(y)), top10_precision=hits/k, prevalence=float(np.mean(y)),
                  mean_score=float(np.mean(scores)))
    if probability:
        result.update(brier=float(brier_score_loss(y, scores)), log_loss=float(log_loss(y, scores, labels=[0, 1])))
    return result


def load_training(path):
    frame = gpd.read_parquet(path)
    if frame.crs is None or frame.crs.to_epsg() != 5179:
        raise ValueError("EPSG:5179 raw grids required")
    frame = frame.loc[frame.road_type_score.notna()].reset_index(drop=True)
    columns = [*CONTINUOUS, *BINARY, "pm_accident"]
    if frame.grid_id.duplicated().any() or not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("Duplicate grid IDs or missing/non-finite model inputs")
    if not set(frame.pm_accident.unique()).issubset({0, 1}):
        raise ValueError("Binary target required")
    if not frame[BINARY].isin([0, 1]).all().all():
        raise ValueError("Zone indicators must be binary")
    return frame


def validate_models(frame, folds=5):
    records, predictions, coefficients = [], [], []
    for fold, (train_idx, test_idx) in enumerate(grouped_splits(frame, folds=folds), 1):
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        scores = {"train_prevalence_baseline": np.full(len(test), train.pm_accident.mean())}
        for balanced, prefix in [(True, "balanced"), (False, "unweighted")]:
            estimator, calibrator = fit_family(train, balanced)
            raw, calibrated = predict_family(estimator, calibrator, test)
            scores[prefix+"_logistic"] = raw
            scores[prefix+"_sigmoid"] = calibrated
            beta = estimator.named_steps["logistic"].coef_[0]
            if balanced:
                weights = np.abs(beta)/np.abs(beta).sum()
                scores["legacy_abs_rs"] = np.clip(estimator.named_steps["features"].transform(test) @ weights, 0, 1)
            coefficients.extend(dict(fold=fold, family=prefix, feature=name, coefficient=float(value))
                                for name, value in zip(FEATURES, beta))
        for name, values in scores.items():
            result = metrics(test.pm_accident.to_numpy(), values, test.grid_id, probability=name!="legacy_abs_rs")
            if name == "train_prevalence_baseline":
                # Constant scores have no meaningful top-k ordering.
                result.update(top10_recall=float("nan"), top10_precision=float("nan"))
            records.append(dict(fold=fold, candidate=name, train_n=len(train), test_n=len(test), **result))
            predictions.append(pd.DataFrame(dict(grid_id=test.grid_id.to_numpy(), fold=fold, candidate=name,
                                                  target=test.pm_accident.to_numpy(), score=values)))
        print(f"model validation fold {fold}/{folds}", flush=True)
    return pd.DataFrame(records), pd.concat(predictions, ignore_index=True), pd.DataFrame(coefficients)


def choose_candidate(fold_metrics):
    eligible = fold_metrics.loc[fold_metrics.candidate.isin(CANDIDATES)]
    summary = eligible.groupby("candidate").agg(ap_mean=("average_precision", "mean"), ap_std=("average_precision", "std"),
                                                brier_mean=("brier", "mean"), n=("fold", "count"))
    best = summary.ap_mean.idxmax()
    tolerance = float(summary.loc[best, "ap_std"]/np.sqrt(summary.loc[best, "n"]))
    threshold = float(summary.loc[best, "ap_mean"]-tolerance)
    pool = summary.loc[summary.ap_mean.ge(threshold)]
    selected = pool.sort_values(["brier_mean", "ap_mean"], ascending=[True, False]).index[0]
    return selected, dict(rule="Within one standard error of best mean AP, then smallest mean Brier",
                          best_ap_candidate=best, ap_threshold=threshold, shortlisted=list(pool.index))


def export_model(estimator, calibrator, selected, source_hash):
    features = estimator.named_steps["features"].scaler_
    standard = estimator.named_steps["standard"]
    logistic = estimator.named_steps["logistic"]
    return dict(schema_version=1, candidate=selected, source_sha256=source_hash,
                continuous_columns=CONTINUOUS, binary_columns=BINARY, feature_names=FEATURES,
                minmax_scale=features.scale_.tolist(), minmax_min=features.min_.tolist(),
                standard_mean=standard.mean_.tolist(), standard_scale=standard.scale_.tolist(),
                coefficients=logistic.coef_[0].tolist(), intercept=float(logistic.intercept_[0]),
                calibration_slope=float(calibrator.coef_[0, 0]) if selected.endswith("_sigmoid") else 1.,
                calibration_intercept=float(calibrator.intercept_[0]) if selected.endswith("_sigmoid") else 0.,
                target="Pooled historical grid accident presence, not hourly or per-trip accident probability",
                hourly_score="Scenario index from hour-specific pedestrian substitution; hourly calibration unverified")


class RiskModel:
    def __init__(self, state):
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported model schema")
        for name, size in [("minmax_scale", 5), ("minmax_min", 5), ("standard_mean", 6), ("standard_scale", 6), ("coefficients", 6)]:
            values = np.asarray(state[name], dtype=float)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid model parameters: {name}")
        if np.any(np.asarray(state["standard_scale"]) <= 0):
            raise ValueError("Standard scales must be positive")
        if not all(np.isfinite(state[key]) for key in ["intercept", "calibration_slope", "calibration_intercept"]):
            raise ValueError("Invalid model intercept or calibration")
        self.state = state

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def predict(self, frame, hour=None):
        s = self.state
        raw = frame[s["continuous_columns"]].to_numpy(dtype=float)
        if not np.isfinite(raw).all() or not np.isfinite(frame[s["binary_columns"]].to_numpy(dtype=float)).all():
            raise ValueError("Finite model inputs required")
        if not frame[s["binary_columns"]].isin([0, 1]).all().all():
            raise ValueError("Zone indicators must be binary")
        normal = np.clip(raw*np.asarray(s["minmax_scale"])+np.asarray(s["minmax_min"]), 0, 1)
        if hour is not None and hour not in ["10h", "18h", "22h"]:
            raise ValueError("Unknown hour")
        pedestrian = normal[:, 2:].mean(axis=1) if hour is None else normal[:, ["10h", "18h", "22h"].index(hour)+2]
        x = np.column_stack([normal[:, :2], frame[s["binary_columns"]].to_numpy(dtype=float), pedestrian])
        z = (x-np.asarray(s["standard_mean"]))/np.asarray(s["standard_scale"])
        margin = z @ np.asarray(s["coefficients"])+s["intercept"]
        return expit(s["calibration_slope"]*margin+s["calibration_intercept"])


def train(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_training(source)
    fold_metrics, oof, coefficients = validate_models(frame)
    selected, rule = choose_candidate(fold_metrics)
    estimator, calibrator = fit_family(frame, selected.startswith("balanced_"))
    state = export_model(estimator, calibrator, selected, hashlib.sha256(source.read_bytes()).hexdigest())
    model = RiskModel(state)
    expected = predict_family(estimator, calibrator, frame)[int(selected.endswith("_sigmoid"))]
    if not np.allclose(model.predict(frame), expected, rtol=0, atol=1e-12):
        raise AssertionError("JSON model inference differs from fitted estimator")
    for hour in ["10h", "18h", "22h"]:
        scenario = frame.assign(_scenario_hour=hour)
        expected = predict_family(estimator, calibrator, scenario)[int(selected.endswith("_sigmoid"))]
        if not np.allclose(model.predict(frame, hour), expected, rtol=0, atol=1e-12):
            raise AssertionError("Scenario inference mismatch")
    fold_metrics.to_csv(output/"fold_metrics.csv", index=False)
    coefficients.to_csv(output/"coefficients.csv", index=False)
    oof.to_csv(output/"oof_predictions.csv", index=False)
    summary = fold_metrics.groupby("candidate")[["roc_auc", "average_precision", "top10_recall", "brier", "log_loss", "mean_score"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(output/"summary.csv")
    (output/"model.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    metadata = dict(selected_model=selected, selection=rule, rows=len(frame), positives=int(frame.pm_accident.sum()),
                    outer_cv="5-fold 1km spatial blocks", inner_calibration="3-fold spatial OOF margins; regularized sigmoid C=1e6",
                    source_sha256=state["source_sha256"],
                    limitations=["Model selection uses this CV; final candidate has no independent external holdout",
                                 "No spatial buffer or temporal holdout", "Hourly scores are scenario indices, not validated hourly probabilities"])
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string()); print("Selected:", selected)
    return metadata
