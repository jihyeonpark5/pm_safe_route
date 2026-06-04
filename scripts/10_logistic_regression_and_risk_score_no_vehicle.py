from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

INPUT_PATH = PROCESSED_DIR / "model_input_variables.parquet"
OUTPUT_PATH = PROCESSED_DIR / "risk_score_grid_no_vehicle.parquet"


# ---------------------------------------------------------------------------
# 변수 정의
# ---------------------------------------------------------------------------
# 종속변수: PM 사고 발생 여부
TARGET_COL = "pm_accident"

# 모델 내부에서만 생성하는 대표 보행자 변수
# pedestrian_avg = (10시 보행자 + 18시 보행자 + 22시 보행자) / 3
PEDESTRIAN_AVG_COL = "pedestrian_avg"

# 정적 변수: 시간대와 관계없이 고정되는 격자 특성
STATIC_FEATURE_COLS = [
    "intersection_count_norm",  # 교차로 수
    "road_type_score_norm",  # 도로 유형 점수
    "is_school_zone",  # 어린이보호구역 여부
    "is_hospital_zone",  # 병원 인접 구역 여부
    "is_elderly_zone",  # 노인/장애인 보호구역 여부
]

# 동적 변수: 시간대별로 바뀌는 보행자 변수
# no_vehicle 버전에서는 vehicle_*_norm을 의도적으로 제외한다.
DYNAMIC_FEATURES: dict[str, str] = {
    "10h": "pedestrian_10h_norm",
    "18h": "pedestrian_18h_norm",
    "22h": "pedestrian_22h_norm",
}

# VIF 임계값: 이 값을 초과하면 다중공선성 경고를 출력한다.
VIF_THRESHOLD = 10.0

# Risk Score 위험 등급 분위수 기준
RISK_QUANTILES = [0.50, 0.75, 0.90]
RISK_LABELS = ["일반", "주의", "위험", "고위험"]

# 시나리오별 등급 속도 제한(km/h)
# 등급 순서: 일반 / 주의 / 위험 / 고위험
SCENARIO_SPEED: dict[str, list[int]] = {
    "S0": [25, 25, 25, 25],  # 기준안: 전 구간 25 km/h 고정
    "S1": [25, 20, 15, 10],  # 기본 차등 제한안
    "S2": [20, 15, 10, 7],  # 안전 강화 제한안
}

# 교차검증 fold 수
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    """실행 상태를 콘솔에 바로 출력한다."""
    print(message)


def ensure_output_directory() -> None:
    """출력 폴더가 없으면 자동으로 생성한다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 데이터 로드 및 검증
# ---------------------------------------------------------------------------
def load_input_data() -> pd.DataFrame:
    """
    모델 입력 파일을 읽고, 필요한 컬럼과 기본 구조를 검증한다.

    no_vehicle 모델은 parquet 원본에 pedestrian_avg를 저장하지 않고,
    이 함수 안에서 시간대별 보행자 정규화 값의 평균으로 파생 생성한다.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "모델 입력 파일이 없습니다. "
            "먼저 scripts/09_normalize_variables.py를 실행해 주세요. "
            f"필요 파일: {INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    required_cols = {TARGET_COL, "grid_id", *STATIC_FEATURE_COLS, *DYNAMIC_FEATURES.values()}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"입력 파일에 필요한 컬럼이 없습니다: {sorted(missing_cols)}")

    if df.empty:
        raise ValueError("입력 파일이 비어 있습니다.")

    duplicate_count = int(df["grid_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"grid_id 중복이 있습니다: {duplicate_count}개")

    df = df.copy()
    df[PEDESTRIAN_AVG_COL] = df[list(DYNAMIC_FEATURES.values())].mean(axis=1)

    pedestrian_avg_na = int(df[PEDESTRIAN_AVG_COL].isna().sum())
    if pedestrian_avg_na:
        raise ValueError(f"{PEDESTRIAN_AVG_COL} 결측이 있습니다: {pedestrian_avg_na}개")

    return df


# ---------------------------------------------------------------------------
# VIF 다중공선성 진단
# ---------------------------------------------------------------------------
def run_vif_analysis(X: pd.DataFrame) -> pd.DataFrame:
    """
    입력 변수 간 다중공선성을 VIF(분산팽창지수)로 진단한다.

    VIF > 10이면 변수 제거 또는 피처 그룹 통합을 검토해야 한다.
    이진 변수(0/1)는 VIF 계산 대상에서 제외한다.
    """
    binary_cols = [c for c in X.columns if set(X[c].dropna().unique()).issubset({0, 1})]
    numeric_cols = [c for c in X.columns if c not in binary_cols]

    log("\n=== VIF 다중공선성 진단 ===")

    if not numeric_cols:
        log("연속형 변수가 없어 VIF 분석을 건너뜁니다.")
        return pd.DataFrame(columns=["feature", "VIF"])

    X_num = X[numeric_cols].copy()
    vif_values = [
        variance_inflation_factor(X_num.values, i)
        for i in range(X_num.shape[1])
    ]
    vif_df = pd.DataFrame({"feature": numeric_cols, "VIF": vif_values})
    vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

    for _, row in vif_df.iterrows():
        flag = " [주의] (VIF > 10)" if row["VIF"] > VIF_THRESHOLD else ""
        log(f"  {row['feature']}: VIF = {row['VIF']:.4f}{flag}")

    high_vif = vif_df[vif_df["VIF"] > VIF_THRESHOLD]
    if not high_vif.empty:
        log(f"\n  [WARN] VIF > {VIF_THRESHOLD} 변수 {len(high_vif)}개 발견 - 보고서에 기재 필요")
    else:
        log(f"  [OK] 모든 연속형 변수의 VIF <= {VIF_THRESHOLD} - 다중공선성 문제 없음")

    return vif_df


# ---------------------------------------------------------------------------
# 로지스틱 회귀 학습 및 가중치 산출
# ---------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    no_vehicle 모델 입력 행렬을 만든다.

    컬럼 순서: 정적 변수 5개 + pedestrian_avg = 총 6개
    """
    feature_cols = STATIC_FEATURE_COLS + [PEDESTRIAN_AVG_COL]
    X = df[feature_cols].copy().fillna(0.0)
    return X, feature_cols


def train_logistic_regression(
    df: pd.DataFrame,
) -> tuple[LogisticRegression, StandardScaler, np.ndarray, list[str]]:
    """
    pedestrian_avg 기반 로지스틱 회귀를 학습하고 표준화 계수 기반 가중치를 반환한다.

    - class_weight='balanced': pm_accident 불균형을 보정
    - StandardScaler: 표준화 후 계수로 변수 간 기여도를 비교
    - 가중치 정규화: w_i = |beta_i| / sum(|beta_i|)
    """
    X_df, feature_cols = build_feature_matrix(df)
    y = df[TARGET_COL].astype(int)

    # 최종 모델 학습용 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # 로지스틱 회귀 학습
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_scaled, y)

    # 표준화 회귀계수의 절댓값을 1.0 합계 가중치로 정규화
    betas = np.abs(model.coef_[0])
    weights = betas / betas.sum()

    return model, scaler, weights, feature_cols


def evaluate_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """
    학습된 모델의 분류 성능과 교차검증 AUC-ROC를 출력한다.

    교차검증에서는 스케일링 누수를 막기 위해 StandardScaler와 모델을
    Pipeline으로 묶고, fold마다 훈련 fold에만 scaler를 fit한다.
    """
    X_df = df[feature_cols].fillna(0.0)
    X_scaled = scaler.transform(X_df)
    y = df[TARGET_COL].astype(int)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    log("\n=== 모델 성능 평가 (pedestrian_avg 기반, vehicle 제외) ===")
    log(classification_report(y, y_pred, target_names=["사고 없음", "사고 발생"]))

    auc = roc_auc_score(y, y_prob)
    log(f"  AUC-ROC: {auc:.4f}")

    # 층화 k-fold 교차검증
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
    )
    cv_scores = cross_val_score(cv_model, X_df, y, cv=cv, scoring="roc_auc")
    log(f"  {CV_FOLDS}-Fold 교차검증 AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")


# ---------------------------------------------------------------------------
# Risk Score 산출
# ---------------------------------------------------------------------------
def compute_risk_scores(
    df: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    """
    3개 시간대(10h / 18h / 22h)별 RS를 산출한다.

    학습은 pedestrian_avg 대표 변수로 수행하지만, 시간대별 RS 계산에는
    각 시간대의 pedestrian_*_norm 값을 대입한다.

    RS(격자, 시간대) = Σ(wᵢ × Nᵢ) × 100
    - wᵢ: 로지스틱 회귀로 도출한 가중치 (Σwᵢ = 1.0)
    - Nᵢ: 정규화된 변수값 (0~1)
    - 결과 RS 범위: 0~100
    """
    result = df[["grid_id"]].copy()

    static_weights = weights[: len(STATIC_FEATURE_COLS)]
    pedestrian_weight = weights[-1]
    static_matrix = df[STATIC_FEATURE_COLS].fillna(0.0).to_numpy()
    static_score = (static_matrix * static_weights).sum(axis=1)

    for hour, pedestrian_col in DYNAMIC_FEATURES.items():
        pedestrian_values = df[pedestrian_col].fillna(0.0).to_numpy()
        rs = (static_score + pedestrian_weight * pedestrian_values) * 100.0
        rs = np.clip(rs, 0.0, 100.0)
        result[f"RS_{hour}"] = np.round(rs, 4)

    return result


# ---------------------------------------------------------------------------
# 위험 등급 분류 및 시나리오 속도 매핑
# ---------------------------------------------------------------------------
def assign_risk_levels(result: pd.DataFrame, risk_bins: list[float]) -> pd.DataFrame:
    """
    RS를 4단계 위험등급으로 분류하고 시나리오별 제한속도를 매핑한다.

    등급 구분 (통합 RS 실제 분포 기준):
      일반   : 하위 50%
      주의   : 50~75%
      위험   : 75~90%
      고위험 : 상위 10%
    """
    for hour in DYNAMIC_FEATURES:
        rs_col = f"RS_{hour}"
        level_col = f"risk_level_{hour}"

        result[level_col] = pd.cut(
            result[rs_col],
            bins=risk_bins,
            labels=RISK_LABELS,
            include_lowest=True,
        )

        # 시나리오별 제한속도 컬럼 추가
        for scenario, speeds in SCENARIO_SPEED.items():
            speed_col = f"speed_{scenario}_{hour}"
            result[speed_col] = result[level_col].map(
                {label: speeds[i] for i, label in enumerate(RISK_LABELS)}
            ).astype("Int64")

    return result


# ---------------------------------------------------------------------------
# 출력 및 요약
# ---------------------------------------------------------------------------
def print_weight_summary(weights: np.ndarray, feature_cols: list[str]) -> None:
    """변수별 가중치를 내림차순으로 출력한다."""
    log("\n=== 로지스틱 회귀 가중치 (pedestrian_avg 기반, vehicle 제외) ===")
    sorted_idx = np.argsort(weights)[::-1]
    for i in sorted_idx:
        bar = "#" * int(weights[i] * 40)
        log(f"  {feature_cols[i]:<35s}: {weights[i]:.4f}  {bar}")
    log(f"  가중치 합계: {weights.sum():.6f}  (1.0 기준)")


def calculate_risk_bins(result: pd.DataFrame) -> list[float]:
    """통합 RS 분포의 q50/q75/q90을 등급 경계값으로 계산하고 출력한다."""
    rs_cols = [f"RS_{hour}" for hour in DYNAMIC_FEATURES]
    combined_rs = pd.concat([result[col] for col in rs_cols], ignore_index=True)

    q50, q75, q90 = [float(combined_rs.quantile(q)) for q in RISK_QUANTILES]
    risk_bins = [0.0, q50, q75, q90, 100.0]

    if len(set(risk_bins)) != len(risk_bins):
        raise ValueError(f"Risk Score 분위수 경계값이 중복되었습니다: {risk_bins}")

    log("\n=== Risk Score 분위수 기준 확인 ===")
    log(f"  통합 RS q50: {q50:.2f}")
    log(f"  통합 RS q75: {q75:.2f}")
    log(f"  통합 RS q90: {q90:.2f}")
    log(f"  적용 등급 경계값: {[round(value, 4) for value in risk_bins]}")

    return risk_bins


def print_rs_summary(result: pd.DataFrame) -> None:
    """시간대별 RS 및 위험등급 분포를 출력한다."""
    log("\n=== Risk Score 요약 ===")
    for hour in DYNAMIC_FEATURES:
        rs_col = f"RS_{hour}"
        level_col = f"risk_level_{hour}"
        log(f"\n  [{hour}]")
        log(
            f"    RS min/max/mean: "
            f"{result[rs_col].min():.2f} / "
            f"{result[rs_col].max():.2f} / "
            f"{result[rs_col].mean():.2f}"
        )
        log(f"    위험등급 분포:\n{result[level_col].value_counts().to_string()}")


def save_result(result: pd.DataFrame) -> None:
    """최종 Risk Score 결과를 Parquet으로 저장한다."""
    ensure_output_directory()
    result.to_parquet(OUTPUT_PATH, index=False)
    log(f"\n결과 저장 완료: {OUTPUT_PATH}")
    log(f"저장된 컬럼: {result.columns.tolist()}")


# ---------------------------------------------------------------------------
# 메인 실행 흐름
# ---------------------------------------------------------------------------
def main() -> int:
    """Step 10 no_vehicle: 가중치 산출 및 Risk Score 계산 전체 흐름."""
    log("=== Step 10: pedestrian_avg 기반, vehicle 제외 위험도 모델 및 Risk Score 산출 ===\n")

    try:
        # 1. 데이터 로드
        df = load_input_data()
        log(f"입력 데이터 로드 완료: {len(df)}개 격자")
        log(f"사고 발생 격자: {int(df[TARGET_COL].sum())}개 ({df[TARGET_COL].mean() * 100:.1f}%)")
        log(f"{PEDESTRIAN_AVG_COL} 결측 개수: {int(df[PEDESTRIAN_AVG_COL].isna().sum())}")

        # 2. VIF 다중공선성 진단
        X_train_df, feature_cols = build_feature_matrix(df)
        log(f"학습 feature 수: {len(feature_cols)}개")
        log(f"학습 feature 목록: {feature_cols}")
        run_vif_analysis(X_train_df)

        # 3. 로지스틱 회귀 학습 + 가중치 산출
        log("\n=== 로지스틱 회귀 학습 (pedestrian_avg 기반, vehicle 제외) ===")
        model, scaler, weights, feature_cols = train_logistic_regression(df)
        print_weight_summary(weights, feature_cols)

        # 4. 모델 성능 평가
        evaluate_model(model, scaler, df, feature_cols)

        # 5. 시간대별 Risk Score 산출
        log("\n=== 시간대별 Risk Score 산출 (10h / 18h / 22h, vehicle 제외) ===")
        result = compute_risk_scores(df, weights)
        risk_bins = calculate_risk_bins(result)

        # 6. 위험 등급 분류 + 시나리오 속도 매핑
        result = assign_risk_levels(result, risk_bins)
        print_rs_summary(result)

        # 7. 저장
        save_result(result)

        log("\nStep 10(no_vehicle)이 정상 완료되었습니다.")
        return 0
    except Exception as exc:
        log(f"\nStep 10(no_vehicle) 실행 실패: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
