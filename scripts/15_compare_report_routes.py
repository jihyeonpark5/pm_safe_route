"""Recalculate three report routes using pinned coordinates and additive metrics.

The routing objective is unchanged. This audits the existing routes rather
than claiming that a new routing algorithm has improved safety.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("route_optimizer", Path(__file__).with_name("13_optimize_route_ortools.py"))
route = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = route
spec.loader.exec_module(route)

# Case 1 deliberately uses the repository's cached 논현역 coordinate. The
# presentation labels it 신논현역; this unresolved source discrepancy is reported.
CASES = [
    route.RouteRequest("report1", "nonhyeon_seolleung", "Repository Nonhyeon -> Seolleung", "논현역 (repository cache)", "선릉역",
                       (127.0231433, 37.5072398), (127.051217, 37.505447), "18h"),
    route.RouteRequest("report2", "yeoksam1_gangnamhall", "Report case 2 coordinates", "역삼1동복합문화센터 (source coords)", "강남구민회관 (source coords)",
                       (127.0335, 37.4957), (127.0625, 37.4969), "18h"),
    route.RouteRequest("report3", "ilwon1_daechi2", "Ilwon1 -> Daechi2", "일원1동주민센터", "대치2동주민센터",
                       (127.0878769, 37.4917394), (127.0666992, 37.5019139), "18h"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/output/validation_v1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges, nodes = route.load_inputs()
    config = route.RunConfig("S1", [0.0, 0.5, 1.0], CASES, route.DEFAULT_SUBGRAPH_MODE, True, "validation_v1")
    summaries, segments, diagnostics = [], [], []
    for request in CASES:
        results, rows, diag = route.optimize_request(edges, nodes, request, config)
        if not all(result.path_valid for result in results.values()):
            raise RuntimeError(f"Invalid route in {request.case_id}")
        summaries.extend(rows)
        diagnostics.append({"case_id": request.case_id, **diag})
        for value, result in results.items():
            segments.extend(dict(case_id=request.case_id, lambda_value=value, order=i, edge_id=arc.edge_id,
                                 tail=arc.tail, head=arc.head, length_m=arc.length_m, time_sec=arc.time_sec,
                                 rs=arc.rs_value, risk_level=arc.risk_level)
                            for i, arc in enumerate(result.path_arcs))
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "route_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(segments).to_csv(args.output_dir / "route_segments.csv", index=False)
    metadata = {
        "scenario": "S1", "hour": "18h", "lambdas": config.lambdas,
        "coordinates": [dict(case_id=c.case_id, start=c.start_lon_lat, end=c.end_lon_lat) for c in CASES],
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (route.EDGE_RISK_PATH, route.NODES_PATH)},
        "objective": "unchanged legacy objective, including lambda**0.72, comfort and turn penalties",
        "limits": ["Case 1 station name differs between presentation and repository cache",
                   "Case 2 uses coordinates embedded in repository reportfinal CSV",
                   "Not evidence of crash reduction; no new routing objective",
                   "Subgraph optimum; full graph fallback only on infeasibility",
                   "High-risk distance uses existing edge-level majority labels, not exact grid fragments"],
        "diagnostics": diagnostics,
    }
    (args.output_dir / "route_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    selected = ["case_id", "lambda", "distance_km", "raw_time_min", "exposure_sum", "risk_sum",
                "distance_exposure_m", "time_exposure_sec", "high_risk_distance_m",
                "distance_exposure_m_reduction_vs_lambda0", "time_exposure_sec_reduction_vs_lambda0"]
    print(summary[selected].to_string(index=False))


if __name__ == "__main__":
    main()
