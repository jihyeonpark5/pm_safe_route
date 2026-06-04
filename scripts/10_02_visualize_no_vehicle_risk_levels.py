from __future__ import annotations

from pathlib import Path
import os
import sys

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

GRID_PATH = PROCESSED_DIR / "02_gangnam_grid_50m.parquet"
RISK_PATH = PROCESSED_DIR / "risk_score_grid_no_vehicle.parquet"
MODEL_INPUT_PATH = PROCESSED_DIR / "model_input_variables.parquet"

HOURS = ["10h", "18h", "22h"]
HOUR_LABELS = {
    "10h": "10시",
    "18h": "18시",
    "22h": "22시",
}
OUTPUT_PATHS = {
    hour: OUTPUT_DIR / f"risk_level_no_vehicle_{hour}.png"
    for hour in HOURS
}
ACCIDENT_OUTPUT_PATHS = {
    hour: OUTPUT_DIR / f"risk_level_no_vehicle_{hour}_with_accidents.png"
    for hour in HOURS
}

RISK_LEVELS = ["일반", "주의", "위험", "고위험"]
RISK_COLORS = {
    "일반": "#d9d9d9",
    "주의": "#fee08b",
    "위험": "#fdae61",
    "고위험": "#d73027",
}

NO_SCORE_COLOR = "#f7f7f7"
NO_SCORE_EDGE_COLOR = "#eeeeee"
ACCIDENT_POINT_COLOR = "#111111"

FIG_DPI = 450
FIG_SIZE = (9, 10.5)


def log(message: str) -> None:
    print(message)


def configure_font() -> None:
    """Configure a Korean-capable font for report figures on Windows."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def ensure_input_files_exist() -> None:
    missing_files = [
        str(path)
        for path in [GRID_PATH, RISK_PATH, MODEL_INPUT_PATH]
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing input files: {missing_files}")


def load_data() -> gpd.GeoDataFrame:
    ensure_input_files_exist()

    grid = gpd.read_parquet(GRID_PATH)[["grid_id", "geometry"]].copy()
    risk = pd.read_parquet(RISK_PATH)
    model_input = pd.read_parquet(MODEL_INPUT_PATH)

    required_risk_columns = {"grid_id", *[f"risk_level_{hour}" for hour in HOURS]}
    missing_columns = required_risk_columns - set(risk.columns)
    if missing_columns:
        raise ValueError(f"Risk score file is missing columns: {sorted(missing_columns)}")
    required_model_columns = {"grid_id", "pm_accident"}
    missing_model_columns = required_model_columns - set(model_input.columns)
    if missing_model_columns:
        raise ValueError(f"Model input file is missing columns: {sorted(missing_model_columns)}")

    merged = grid.merge(
        risk[list(required_risk_columns)],
        on="grid_id",
        how="left",
        validate="1:1",
    )
    merged = merged.merge(
        model_input[["grid_id", "pm_accident"]],
        on="grid_id",
        how="left",
        validate="1:1",
    )

    if merged.empty:
        raise ValueError("Merged grid is empty.")

    return merged


def get_scored_grids(merged: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    risk_cols = [f"risk_level_{hour}" for hour in HOURS]
    scored = merged.loc[merged[risk_cols].notna().all(axis=1)].copy()
    if scored.empty:
        raise ValueError("No grids have complete no_vehicle risk levels.")
    return scored


def build_legend_handles() -> list[Patch]:
    handles = [
        Patch(facecolor=NO_SCORE_COLOR, edgecolor="#d0d0d0", label="점수 없음"),
    ]
    handles.extend(
        Patch(facecolor=RISK_COLORS[level], edgecolor="none", label=level)
        for level in RISK_LEVELS
    )
    return handles


def build_count_text(scored: gpd.GeoDataFrame, level_col: str) -> str:
    counts = scored[level_col].astype(str).value_counts().to_dict()
    total = len(scored)
    parts = []
    for level in RISK_LEVELS:
        count = int(counts.get(level, 0))
        parts.append(f"{level} {count:,}개 ({count / total * 100:.1f}%)")
    return " | ".join(parts)


def get_accident_points(merged: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    accident_grids = merged.loc[merged["pm_accident"].fillna(0).astype(int).eq(1)].copy()
    if accident_grids.empty:
        raise ValueError("No pm_accident = 1 grids are available for validation map.")
    return accident_grids.set_geometry(accident_grids.geometry.centroid)


def save_hour_map(
    merged: gpd.GeoDataFrame,
    scored: gpd.GeoDataFrame,
    hour: str,
    accident_points: gpd.GeoDataFrame | None = None,
    output_path: Path | None = None,
) -> Path:
    level_col = f"risk_level_{hour}"
    if output_path is None:
        output_path = OUTPUT_PATHS[hour]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(top=0.83, bottom=0.17)

    merged.plot(
        ax=ax,
        color=NO_SCORE_COLOR,
        edgecolor=NO_SCORE_EDGE_COLOR,
        linewidth=0.015,
    )

    for level in RISK_LEVELS:
        subset = scored.loc[scored[level_col].astype(str).eq(level)]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            color=RISK_COLORS[level],
            edgecolor="#ffffff",
            linewidth=0.035,
        )

    if accident_points is not None:
        accident_points.plot(
            ax=ax,
            marker="o",
            facecolor="none",
            edgecolor=ACCIDENT_POINT_COLOR,
            linewidth=0.5,
            markersize=16,
            alpha=0.95,
            zorder=5,
        )

    fig.suptitle("위험등급 분포", fontsize=22, fontweight="bold", y=0.97)
    ax.set_title(HOUR_LABELS[hour], fontsize=16, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    ax.set_axis_off()

    ax.text(
        0.5,
        -0.055,
        build_count_text(scored, level_col),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    legend_handles = build_legend_handles()
    if accident_points is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="none",
                markeredgecolor=ACCIDENT_POINT_COLOR,
                markeredgewidth=0.8,
                markersize=9,
                label="사고 발생 격자",
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.035),
    )

    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> int:
    try:
        configure_font()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        merged = load_data()
        scored = get_scored_grids(merged)
        accident_points = get_accident_points(merged)

        log(f"전체 격자 수: {len(merged):,}")
        log(f"위험등급 산출 격자 수: {len(scored):,}")
        log(f"사고 발생 격자 수: {len(accident_points):,}")

        for hour in HOURS:
            output_path = save_hour_map(merged, scored, hour)
            log(f"{HOUR_LABELS[hour]} 위험등급 지도 저장 완료: {output_path}")
            accident_output_path = save_hour_map(
                merged,
                scored,
                hour,
                accident_points=accident_points,
                output_path=ACCIDENT_OUTPUT_PATHS[hour],
            )
            log(f"{HOUR_LABELS[hour]} 사고 격자 검증 지도 저장 완료: {accident_output_path}")

        return 0
    except Exception as exc:
        log(f"위험등급 지도 생성 실패: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
