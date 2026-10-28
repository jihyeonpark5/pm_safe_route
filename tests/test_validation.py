import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from route_metrics import exposure_metrics, exposure_comparison

spec = importlib.util.spec_from_file_location("validation", SCRIPTS / "14_validate_risk_score.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def arc(length, seconds, score=30, level="high_risk"):
    return SimpleNamespace(length_m=length, time_sec=seconds, rs_value=score, risk_level=level)


def test_exposure_unchanged_when_same_road_is_split():
    whole = exposure_metrics([arc(100, 20)])
    split = exposure_metrics([arc(25, 5), arc(75, 15)])
    assert whole == pytest.approx(split)
    assert whole["distance_exposure_m"] == 30
    assert whole["time_exposure_sec"] == 6
    assert whole["high_risk_distance_m"] == 100


def test_exposure_distance_and_time_capture_different_detours():
    base = [arc(100, 20, 60)]
    result = exposure_comparison([arc(200, 40, 40, "normal")], base)
    assert result["distance_exposure_m_reduction_vs_lambda0"] == pytest.approx(-1 / 3)
    assert result["high_risk_distance_m_reduction_vs_lambda0"] == 1
    assert np.isnan(exposure_comparison([], [arc(0, 0, 0)])["time_exposure_sec_reduction_vs_lambda0"])


@pytest.mark.parametrize("bad", [arc(1, 1, np.nan), arc(-1, 1), arc(1, 1, 101)])
def test_exposure_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError):
        exposure_metrics([bad])


def synthetic_frame():
    rng = np.random.default_rng(42)
    n = 120
    data = pd.DataFrame({name: rng.uniform(0, 50, n) for name in validation.CONTINUOUS})
    for name in validation.BINARY:
        data[name] = rng.integers(0, 2, n)
    data["grid_id"] = [f"G{i:04d}" for i in range(n)]
    data["pm_accident"] = np.tile([0, 0, 0, 1], n // 4)
    return gpd.GeoDataFrame(data, geometry=[Point((i // 10) * 1000 + 50, 50 + i % 10) for i in range(n)], crs=5179)


def test_test_features_and_labels_do_not_change_fitted_model():
    data = synthetic_frame()
    train, test = data.iloc[:100], data.iloc[100:].copy()
    prob, rs, beta, weights = validation.fit_predict(train, test)
    altered = test.copy()
    altered.loc[altered.index[-1], validation.CONTINUOUS] = 1e12
    altered["pm_accident"] = 1 - altered.pm_accident
    p2, r2, b2, w2 = validation.fit_predict(train, altered)
    np.testing.assert_allclose(beta, b2)
    np.testing.assert_allclose(weights, w2)
    np.testing.assert_allclose(prob[:-1], p2[:-1])
    np.testing.assert_allclose(rs[:-1], r2[:-1])


def test_oof_coverage_and_spatial_block_separation():
    frame = synthetic_frame()
    metrics, predictions, _ = validation.evaluate(frame, folds=3)
    for _, part in predictions.groupby("strategy"):
        assert len(part) == len(frame)
        assert part.grid_id.is_unique
    spatial = predictions.loc[predictions.strategy.eq("spatial")]
    assert spatial.groupby("spatial_block").fold.nunique().eq(1).all()
    assert len(metrics) == 12


def test_top_k_uses_ids_to_break_score_ties():
    result = validation.ranking_metrics(np.array([0, 1, 0, 1]), np.ones(4), ["d", "a", "c", "b"], fraction=.25)
    assert result["top10_selected"] == 1
    assert result["top10_accidents_captured"] == 1
