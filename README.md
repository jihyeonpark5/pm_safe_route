# pm_safe_route

공유 전동킥보드의 위험 점수 산출과 시간·노출 제약 경로 탐색을 위한 연구용 프로젝트입니다.
지원하는 실행 진입점은 `python -m pmroute`입니다. 기존 번호별 실험 스크립트도 재현용으로 보존합니다.

## 빠른 실행

Python 3.11 이상을 사용합니다. 검증 환경은 Python 3.12입니다.
저장소 루트에서 다음 명령을 실행합니다.

```bash
python -m pip install -r requirements-validation.txt
python -m pmroute train
python -m pmroute build
python -m pmroute route --start "127.0335,37.4957" --end "127.0625,37.4969" --hour 18h --scenario S1
```

모델 JSON이 이미 있고 입력 원본이 동일하면 `train`을 생략하고 `build`부터 실행할 수 있습니다.
입력·모델·생성 점수의 해시가 달라지면 다시 학습하거나 생성하도록 오류를 표시합니다.
다른 작업 디렉터리에서는 `python -m pmroute --project-root "저장소 경로" ...`로 지정합니다.
선택적으로 `python -m pip install -e . --no-deps`를 실행하면 `pm-safe-route` 명령도 사용할 수 있습니다.

## 실행 명령

| 명령 | 역할 | 기본 출력 |
|---|---|---|
| `train` | 공간 검증, 후보 비교, 최종 JSON 모델 학습 | `data/output/final/model/` |
| `build` | 별도 격자·도로 점수 생성 | `artifacts/final/` |
| `route` | 좌표 입력의 공동 제약 경로 | `data/output/final/route/` |
| `benchmark` | 고정 60쌍, 3시간, 3시나리오 비교 | `data/output/final/routes/` |

```bash
# 기본 공동 제약: min(최단시간의 10%, 120초), 시간 노출도 기준 이하
python -m pmroute route --start "127.0335,37.4957" --end "127.0625,37.4969" --hour 18h --scenario S1

# 회전 지연 3초 가정은 실측 보정된 값이 아닌 민감도 옵션
python -m pmroute route --start "127.0335,37.4957" --end "127.0625,37.4969" --turn-seconds-at-90 3 --output-dir data/output/final/turn_route

# 같은 출발·도착 쌍의 전체 내부 비교
python -m pmroute benchmark

# 기존 점수로 비교하려면 명시적으로 선택
python -m pmroute benchmark --legacy-scores --output-dir data/output/final/legacy_check

python -m pytest -q
```

좌표 순서는 경도,위도입니다. 기본 도로 스냅 허용 거리는 300m이며 `--max-snap-distance-m`으로 조정합니다.
경로의 접근 구간 시간은 포함하지 않습니다. 멀리 떨어진 좌표를 임의로 도로에 붙이지 않고 오류를 표시합니다.
출력은 `result.json`, `summary.csv`, `routes.geojson`이며 선택한 모델·가정·최적성 상태를 함께 기록합니다.

## 데이터 준비

원래 저장소에 포함된 다음 파일을 사용합니다.

| `data/processed/` 파일 | 용도 |
|---|---|
| `merged_grid_variables.parquet` | 원시 격자 변수와 사고 발생 여부 |
| `08_grid_road_map.parquet` | 격자·도로 교차 길이 |
| `edge_risk_score_no_vehicle.parquet` | 기존 실험 점수, 새 빌드에서는 도로 geometry·속성 원본으로 사용 |
| `04_osm_nodes_raw.parquet` | 노드 좌표와 도로 스냅 |

새 빌드는 기존 점수 열을 제거한 도로 속성에 새 점수를 매핑합니다.
기존 `data/processed/` 결과는 덮어쓰지 않습니다. 점수가 없는 도로 조각을 RS=0으로 처리하지 않고 실패합니다.
도로의 점수는 격자 교차 길이 가중평균, 위험등급은 교차 길이가 가장 긴 등급을 사용합니다.
길이가 같은 등급은 더 높은 위험등급을 선택합니다. 짧은 고위험 구간이 대표등급에 묻힐 수 있는 한계는 남습니다.

## 최종 위험모델

5-fold 1km 공간 블록 검증을 사용했습니다. 확률 보정은 각 외부 학습 fold 안의
3-fold 공간 OOF 예측으로 학습합니다. 정규화·표준화·회귀 학습도 fold 안에서 수행합니다.

| 후보 | ROC-AUC 평균 | AP 평균 | 상위 10% 사고 포착률 | Brier 평균 |
|---|---:|---:|---:|---:|
| 기존 계수 절댓값 RS | 0.7370 | 0.0601 | 25.7% | 적용 대상 아님 |
| 균형 로지스틱 | 0.7822 | 0.0811 | 33.9% | 0.19478 |
| 균형 로지스틱 + sigmoid 보정 | 0.7822 | 0.0811 | 33.9% | 0.02319 |
| **비가중 로지스틱 + sigmoid 보정** | **0.7699** | **0.0747** | **28.9%** | **0.02300** |

최고 평균 AP에서 1 standard error 이내인 후보 중 Brier가 가장 작은 모델을 선택하도록
미리 정한 규칙을 사용했습니다. 최종 후보는 `unweighted_sigmoid`입니다.
이 규칙은 모델 선택 휴리스틱이며 통계적 동등성 검정은 아닙니다.
탐지 순위만 보면 균형 보정 모델이 더 높지만, 여기서는 해당 허용 범위 안에서 확률 보정을 우선했습니다.
선택에도 이 교차검증을 사용했으므로 독립 외부 검증 성능으로 해석하면 안 됩니다.

새 점수는 표준화·회귀계수의 부호·절편·sigmoid 보정을 그대로 사용합니다.
계수 절댓값을 합계 1로 바꾸는 기존 RS 계산은 최종 모델에 사용하지 않습니다.
모델은 실행 코드가 들어가지 않는 JSON 파라미터로 저장하고, 학습한 추론값과 1e-12 이내로 일치하는지 검사합니다.

- `historical_probability`: 시간대 평균 보행자 입력에 대한 누적 과거 격자 사고 발생 추정값.
- `RS_10h`, `RS_18h`, `RS_22h`: 해당 시간 보행자 값을 대입한 0~100 시나리오 지수.

시간대별 사고 정답이 없으므로 시간대별 보정 또는 1회 주행 사고 확률을 검증한 값은 아닙니다.
외부 지역·미래 연도 검증, 공간 블록 간 완충구역, 실제 주행시간 보정은 남아 있습니다.
새 점수와 기존 RS는 척도가 달라 서로 다른 모델의 경로 노출 감소율을 직접 비교하면 안 됩니다.
`data/output/final/model/`의 fold 지표·계수·선택 메타데이터로 후보 비교를 확인할 수 있습니다.
전체 OOF 예측 CSV는 실행 시 생성하며 큰 중간 출력으로 Git에서는 제외합니다.

## 경로 모델과 제약

거리 노출 `sum(RS/100 × 길이)`를 최소화합니다.
같은 점수·속도·회전 가정의 최단시간 경로 대비 이동시간 상한과 시간 노출 비악화 제약을 적용합니다.
`optimal`은 주어진 그래프 모델에서 최적성을 확인한 결과이고,
`feasible_limit`은 탐색 한도에 도달한 제약 충족 후보입니다. 후자는 gap과 종료 사유를 확인해야 합니다.

기본 회전 지연은 0초입니다. 옵션으로 90도에 설정값, 30도 이하 0초, 150도 이상 2배의 지연을 가정할 수 있습니다.
회전 대기 노출은 지연에 인접 두 도로 RS 중 큰 값을 곱합니다. 노드 재방문 경로가 발생할 수 있습니다.
신호·좌우 회전 비대칭·가감속·법적 회전 허용 여부와 실제 사고 감소 효과를 검증한 모델은 아닙니다.
S0/S1/S2 속도와 시간 상한도 연구 시나리오입니다.

## 코드와 과거 실험

- `pmroute/model.py`: 전처리·공간 검증·보정·JSON 추론
- `pmroute/data.py`: 별도 점수 레이어 생성과 해시
- `pmroute/network.py`: 도로·좌표 어댑터, 공동 제약 비교
- `pmroute/routing.py`: 시간·거리 노출·시간 노출의 상태 탐색
- `pmroute/turn_delay.py`, `pmroute/metrics.py`: 지연 가정과 노출 지표
- `pmroute/cli.py`: 지원 명령

기존 `scripts/`의 실험은 보존했습니다. 공통 경로 모듈 3개는 호환 진입점으로 새 패키지를 사용합니다.
검증 기록은 [지표·RS](docs/validation_v1.md), [시간 상한](docs/validation_v2.md),
[다수 경로](docs/validation_v3.md), [공동 제약](docs/validation_v4.md), [회전 가정](docs/validation_v5.md)에 있습니다.
과거 기록의 수치는 당시의 기존 RS로 계산한 결과입니다.
