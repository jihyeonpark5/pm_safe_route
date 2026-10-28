"""Plot model-based distance improvement alongside time-exposure tradeoffs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/output/validation_v3")
    parser.add_argument("--method", default="budget_distance")
    parser.add_argument("--turn-seconds-at-90", type=float)
    args = parser.parse_args()
    summary = pd.read_csv(args.input_dir / "summary.csv")
    selected = summary.loc[summary.method.eq(args.method)].copy()
    if "turn_seconds_at_90" in selected.columns:
        if args.turn_seconds_at_90 is None:
            parser.error("Choose --turn-seconds-at-90 for a turn-sensitivity summary")
        selected = selected.loc[selected.turn_seconds_at_90.eq(args.turn_seconds_at_90)]
    elif args.turn_seconds_at_90 is not None:
        parser.error("Input summary has no turn-delay profiles")
    if selected.empty or selected.errors.ne(0).any() or selected.successful.eq(0).any():
        raise ValueError("Plot requires successful budget results; inspect errors before plotting")
    selected["time_worsened_pct"] = 100 * selected.time_exposure_worsened / (selected.successful-selected.undefined_time_ratios)
    hours = sorted(selected.hour.unique())
    scenarios = sorted(selected.scenario.unique())
    matrices = [selected.pivot(index="hour", columns="scenario", values=column).reindex(index=hours, columns=scenarios)
                for column in ["distance_reduction_mean", "time_worsened_pct"]]
    if any(not np.isfinite(matrix.to_numpy()).all() for matrix in matrices):
        raise ValueError("Missing condition or undefined metric in chart")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "svg.hashsalt": "pm_safe_route_validation_v3"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    settings = [(matrices[0]*100, "Mean distance-exposure reduction", "Greens", None),
                (matrices[1], "Routes with increased time exposure", "Oranges", 100)]
    for ax, (matrix, title, cmap, maximum) in zip(axes, settings):
        values = matrix.to_numpy()
        upper = maximum if maximum is not None else max(1, float(values.max())*1.1)
        ax.imshow(values, cmap=cmap, vmin=0, vmax=upper)
        ax.set_xticks(range(len(scenarios)), scenarios)
        ax.set_yticks(range(len(hours)), hours)
        ax.set_title(title, fontsize=12, pad=14)
        ax.set_xlabel("Speed scenario")
        for row in range(len(hours)):
            for col in range(len(scenarios)):
                value = values[row, col]
                ax.text(col, row, f"{value:.1f}%", ha="center", va="center",
                        color="white" if value/upper > .65 else "#17202a", fontsize=13)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    title = "Time-budget routing: improvement and tradeoff"
    if args.turn_seconds_at_90 is not None:
        title += f" (90-degree turn: {args.turn_seconds_at_90:g}s assumed)"
    fig.suptitle(title, fontsize=13 if args.turn_seconds_at_90 is not None else 15)
    fig.supxlabel("Same sampled OD pairs across conditions; model indices, not measured crash reduction", fontsize=9)
    filename = "overview.svg" if args.turn_seconds_at_90 is None else f"overview_turn_{args.turn_seconds_at_90:g}s.svg"
    target = args.input_dir / filename
    fig.savefig(target, metadata={"Date": None})
    target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines())+"\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
