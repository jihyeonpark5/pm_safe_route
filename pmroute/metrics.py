"""Route exposure indices, measured independently of edge segmentation.

These are model-based indices, not accident probabilities or crash counts.
"""
from __future__ import annotations

import math


def exposure_metrics(arcs):
    distance = time = distance_exposure = time_exposure = 0.0
    high_distance = high_time = 0.0
    for arc in arcs:
        length, seconds, score = float(arc.length_m), float(arc.time_sec), float(arc.rs_value)
        if not all(math.isfinite(v) for v in (length, seconds, score)):
            raise ValueError("Non-finite route exposure input")
        if length < 0 or seconds < 0 or not 0 <= score <= 100:
            raise ValueError("Invalid length, time, or RS")
        distance += length
        time += seconds
        distance_exposure += score / 100 * length
        time_exposure += score / 100 * seconds
        if arc.risk_level in {"high_risk", "고위험"}:
            high_distance += length
            high_time += seconds
    return {
        "distance_exposure_m": distance_exposure,
        "time_exposure_sec": time_exposure,
        "distance_weighted_rs": distance_exposure / distance * 100 if distance else 0.0,
        "high_risk_distance_m": high_distance,
        "high_risk_time_sec": high_time,
        "high_risk_distance_ratio": high_distance / distance if distance else 0.0,
    }


def exposure_comparison(arcs, baseline_arcs):
    current, baseline = exposure_metrics(arcs), exposure_metrics(baseline_arcs)
    for key in ("distance_exposure_m", "time_exposure_sec", "high_risk_distance_m"):
        current[key + "_reduction_vs_lambda0"] = (
            1 - current[key] / baseline[key] if baseline[key] > 0 else float("nan")
        )
    return current
