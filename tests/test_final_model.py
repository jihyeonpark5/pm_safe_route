import copy
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from pmroute.data import build_scores, sha256
from pmroute.model import GridFeatures, RiskModel, choose_candidate, export_model, fit_family, predict_family
from pmroute.network import make_graph, parse_coords, snap


def frame():
    rng = np.random.default_rng(42)
    n = 120
    data = pd.DataFrame({name: rng.uniform(0, 100, n) for name in
                         ["intersection_count", "road_type_score", "pedestrian_10h", "pedestrian_18h", "pedestrian_22h"]})
    for column in ["is_school_zone", "is_hospital_zone", "is_elderly_zone"]:
        data[column] = rng.integers(0, 2, n)
    data["grid_id"] = [f"G{i:03d}" for i in range(n)]
    data["pm_accident"] = np.tile([0, 0, 0, 1], 30)
    return gpd.GeoDataFrame(data, geometry=[Point((i//10)*1000+50, i%10+50) for i in range(n)], crs=5179)


@pytest.fixture(scope="module")
def fitted():
    data = frame()
    estimator, calibrator = fit_family(data, False)
    state = export_model(estimator, calibrator, "unweighted_sigmoid", "test")
    return data, estimator, calibrator, state


def test_json_inference_matches_estimator_and_hour_substitutions(fitted):
    data, estimator, calibrator, state = fitted
    model = RiskModel(json.loads(json.dumps(state)))
    np.testing.assert_allclose(model.predict(data), predict_family(estimator, calibrator, data)[1], atol=1e-12)
    for hour in ["10h", "18h", "22h"]:
        expected = predict_family(estimator, calibrator, data.assign(_scenario_hour=hour))[1]
        np.testing.assert_allclose(model.predict(data, hour), expected, atol=1e-12)


def test_model_inference_does_not_read_target_and_preserves_row_order(fitted):
    data, _, _, state = fitted
    model = RiskModel(state)
    altered = data.copy(); altered["pm_accident"] = 1-data.pm_accident
    np.testing.assert_array_equal(model.predict(data), model.predict(altered))
    shuffled = data.sample(frac=1, random_state=7)
    np.testing.assert_allclose(model.predict(shuffled), model.predict(data)[shuffled.index])


def test_negative_coefficient_keeps_its_direction(fitted):
    data, _, _, state = fitted
    state = copy.deepcopy(state)
    state.update(coefficients=[0, 0, -1, 0, 0, 0], calibration_slope=1., calibration_intercept=0.)
    absent, present = data.copy(), data.copy()
    absent["is_school_zone"] = 0; present["is_school_zone"] = 1
    model = RiskModel(state)
    assert (model.predict(present) < model.predict(absent)).all()


def test_feature_fit_ignores_test_distribution():
    data = frame()
    transform = GridFeatures().fit(data.iloc[:100])
    original = transform.scaler_.scale_.copy()
    test = data.iloc[100:].copy(); test["pedestrian_10h"] = 1e9
    assert np.isfinite(transform.transform(test)).all()
    np.testing.assert_array_equal(original, transform.scaler_.scale_)


def test_selection_uses_prespecified_ap_gate_then_calibration():
    rows = []
    for name, ap, brier in [("balanced_logistic", .1, .2), ("balanced_sigmoid", .1, .03),
                           ("unweighted_logistic", .099, .025), ("unweighted_sigmoid", .099, .02),
                           ("legacy_abs_rs", 1., 0.)]:
        for fold, offset in enumerate([-.02, -.01, 0, .01, .02], 1):
            rows.append(dict(candidate=name, fold=fold, average_precision=ap+offset, brier=brier))
    selected, rule = choose_candidate(pd.DataFrame(rows))
    assert selected == "unweighted_sigmoid" and "legacy_abs_rs" not in rule["shortlisted"]


def test_bad_model_state_and_unknown_hour_fail(fitted):
    data, _, _, state = fitted
    invalid = copy.deepcopy(state); invalid["standard_scale"][0] = 0
    with pytest.raises(ValueError):
        RiskModel(invalid)
    with pytest.raises(ValueError):
        RiskModel(state).predict(data, "unknown")
    invalid_data = data.copy(); invalid_data.loc[0, "pedestrian_10h"] = np.nan
    with pytest.raises(ValueError):
        RiskModel(state).predict(invalid_data)


def test_missing_road_grid_score_never_becomes_zero_risk(tmp_path, fitted):
    data, _, _, state = fitted
    processed = tmp_path/"data/processed"; processed.mkdir(parents=True)
    model_dir = tmp_path/"data/output/final/model"; model_dir.mkdir(parents=True)
    source = processed/"merged_grid_variables.parquet"; data.to_parquet(source)
    state = copy.deepcopy(state); state["source_sha256"] = sha256(source)
    (model_dir/"model.json").write_text(json.dumps(state))
    edges = gpd.GeoDataFrame({"edge_id": ["E1"], "u": [1], "v": [2], "oneway": [False], "length": [100.]},
                             geometry=[LineString([(0, 0), (100, 0)])], crs=5179)
    edges.to_parquet(processed/"edge_risk_score_no_vehicle.parquet")
    pd.DataFrame(dict(edge_id=["E1"], grid_id=["missing"], length_in_grid=[100.])).to_parquet(processed/"08_grid_road_map.parquet")
    with pytest.raises(ValueError, match="no model score"):
        build_scores(tmp_path)
    assert not (tmp_path/"artifacts/final/edge_scores.parquet").exists()


def test_native_adapter_respects_direction_and_joint_constraints():
    edge = dict(edge_id="E1", u=1, v=2, oneway=False, length=100., geometry=LineString([(0, 0), (100, 0)]),
                RS_18h=10., risk_level_18h="normal", speed_S1_18h=25.)
    edges = gpd.GeoDataFrame([edge], crs=5179)
    graph = make_graph(edges)
    assert len(graph.arcs) == 2
    reverse, _ = graph.shortest_path(2, 1)
    assert list(reverse[0].geometry.coords)[0] == (100., 0.)
    edges["oneway"] = True
    with pytest.raises(ValueError, match="No directed path"):
        make_graph(edges).shortest_path(2, 1)


@pytest.mark.parametrize("value", ["nan,37", "200,37", "127", "127,100"])
def test_bad_coordinates_rejected(value):
    with pytest.raises(ValueError):
        parse_coords(value)


def test_distant_endpoint_rejected():
    nodes = gpd.GeoDataFrame({"osmid": [1]}, geometry=[Point(0, 0)], crs=5179)
    with pytest.raises(ValueError, match="snap limit"):
        snap(nodes, (127., 37.), max_distance=300.)
