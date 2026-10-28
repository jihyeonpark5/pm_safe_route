# pm_safe_route

공유 전동킥보드의 시공간 위험도 정량화 및 안전 가중치 기반 최적 경로 추천 시스템

## 거리·시간 위험 노출 공동 제약

추가시간 상한과 함께 시간 기준 위험 노출도 최단시간 경로보다 늘어나지 않도록 하는
옵션을 추가했습니다. [4차 개선 결과](docs/validation_v4.md)에 같은 540개 조건의 비교를 정리했습니다.

```bash
python scripts/16_compare_time_budget_routes.py --prevent-time-exposure-increase --output-dir data/output/joint_report_cases
python scripts/19_compare_joint_exposure.py
```

공동 제약을 적용하려면 `--prevent-time-exposure-increase`를 지정합니다.
기준 점수·속도에 대한 모델상 제약이며 실제 사고 감소를 보장하는 값은 아닙니다.

## 다수 출발·도착 및 조건별 비교 실험

고정된 60개 출발·도착 쌍을 3개 시간대·3개 속도 시나리오에서 비교했습니다.
[3차 비교 결과](docs/validation_v3.md)에 540개 조건의 개선폭, 시간 노출 상충,
기존 λ 방식 비교와 재현 한계를 정리했습니다.

```bash
python scripts/17_batch_validate_routes.py
python scripts/18_plot_batch_validation.py
```

결과는 `data/output/validation_v3/`에 저장됩니다. 실제 사고 감소율을 검증한 결과는 아닙니다.

## 추가시간 상한 기반 경로 추천

최단시간보다 `min(10%, 120초)`까지만 더 걸리는 범위에서 거리 기준 위험 노출을
최소화하는 전체 그래프 탐색을 추가했습니다.
[2차 개선 결과 및 제약](docs/validation_v2.md)에 기존 λ 방식과의 비교를 정리했습니다.

```bash
pip install -r requirements-validation.txt
python scripts/16_compare_time_budget_routes.py
```

3개 사례 또는 `--start-coords 경도,위도 --end-coords 경도,위도` 입력을 지원합니다.
결과는 `data/output/validation_v2/`에 CSV·GeoJSON으로 저장됩니다.
거리 기준 노출을 최소화하며, 시간 기준 노출의 동시 감소나 실제 사고 감소를 보장하는 값은 아닙니다.

## 위험 노출 지표 및 검증 개선

거리·시간 기반 위험 노출, 실제 RS의 교차검증, 기존 3개 경로의 재계산 결과는
[1차 검증 보고서](docs/validation_v1.md)에 정리했습니다.
기존 RS 단순 합계 감소와 거리·시간 기준 노출 감소는 다를 수 있습니다.
보고서에 제시된 회귀 AUC와 파생 RS 자체의 성능도 구분해야 합니다.

```bash
pip install -r requirements-validation.txt
python scripts/14_validate_risk_score.py
python scripts/15_compare_report_routes.py
python -m pytest -q
```

기존 경로 탐색 CSV에는 `distance_exposure_m`, `time_exposure_sec`,
`high_risk_distance_m` 등의 지표가 추가됩니다. 기존 경로 목적함수는 유지됩니다.

## 폴더 구조

```text
pm_safe_route/
├── data/
│   ├── raw/
│   │   ├── boundary/
│   │   ├── facilities/
│   │   ├── population/
│   │   ├── traffic/
│   │   ├── slope/
│   │   └── accident/
│   ├── processed/
│   └── output/
├── notebooks/
├── scripts/
│   └── 00_check_environment.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 사용 패키지

- `osmnx`
- `geopandas`
- `pandas`
- `numpy`
- `scikit-learn`
- `folium`
- `shapely`
- `pyproj`
- `statsmodels`
- `pyarrow`
- `matplotlib`
- `ortools`

## 패키지 역할 메모

- `osmnx`, `geopandas`, `shapely`, `pyproj`는 공간 데이터 처리와 도로 네트워크 분석에 사용합니다.
- `pandas`, `numpy`, `pyarrow`는 표 형태 데이터 처리 및 파일 입출력에 사용합니다.
- `scikit-learn`, `statsmodels`는 위험도 정량화 과정의 분석 및 모델링 단계에서 활용할 수 있습니다.
- `folium`, `matplotlib`는 시각화에 사용합니다.
- `ortools`는 **현재 전처리 단계용이 아니라 이후 경로 최적화 단계**에서 사용할 예정입니다.

## Windows PowerShell 기준 실행 방법

### 1. 프로젝트 폴더로 이동

```powershell
cd C:\Users\user\Documents\pm_safe_route
```

### 2. 가상환경 생성

```powershell
python -m venv .venv
```

### 3. 가상환경 활성화

```powershell
.\.venv\Scripts\Activate.ps1
```

PowerShell 실행 정책 때문에 활성화가 막히면 아래 명령을 한 번 실행한 뒤 다시 활성화합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4. 패키지 설치

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. 환경 점검 스크립트 실행

```powershell
python .\scripts\00_check_environment.py
```

정상 실행되면 주요 패키지 import 결과와 필수 폴더 존재 여부가 출력됩니다.

## processed 산출물 설명

`data/processed` 폴더에는 각 단계에서 생성된 전처리 결과 파일이 저장됩니다. 파일명 앞의 번호는 생성 순서를 의미합니다.

| 파일명 | 생성 단계 | 설명 |
| --- | --- | --- |
| `01_gangnam_boundary.parquet` | Step 2 | 강남구 행정경계 polygon입니다. 모든 공간분석의 기준 경계로 사용합니다. |
| `02_gangnam_grid_50m.parquet` | Step 2 | 강남구 경계와 교차하는 50m x 50m 격자 polygon입니다. `grid_id`, `centroid_x`, `centroid_y`, `geometry`를 포함합니다. |
| `03_gangnam_grid_50m.geojson` | Step 2 | 50m 격자를 QGIS 등에서 확인하기 위한 GeoJSON 파일입니다. 분석보다는 시각 확인용입니다. |
| `04_osm_nodes_raw.parquet` | Step 3 | OSMnx로 수집한 bike network의 원본 node 데이터입니다. 교차로 수 계산 시 node 위치를 가져오는 데 사용합니다. (도로망의 점) |
| `05_osm_edges_raw.parquet` | Step 3 | OSMnx로 수집한 bike network의 원본 edge 데이터입니다. 필터링 전 도로망입니다.(점과 점을 잇는 도로 선) |
| `06_osm_edges_filtered.parquet` | Step 3 | PM 주행이 부적절한 후보 구간을 제거한 edge 데이터입니다. 이후 도로 기반 변수 생성의 기본 도로망으로 사용합니다. |
| `07_osm_edges_filtered.geojson` | Step 3 | 필터링된 도로망을 QGIS 등에서 확인하기 위한 GeoJSON 파일입니다. |
| `08_grid_road_map.parquet` | Step 4 | 50m 격자와 필터링된 도로 edge를 교차시킨 결과입니다. 각 도로가 각 격자 안에서 차지하는 길이 `length_in_grid`를 포함합니다. |
| `09_facility_zone_grid.parquet` | Step 5 | 50m 격자에 학교, 병원, 노인시설 보호구역 포함 여부 변수를 붙인 결과입니다. |
| `10_dynamic_variables.parquet` | Step 6 | 동적 변수 생성 결과입니다. 보호구역 여부, 시간대별 보행량, 시간대별 차량량, `pm_accident`를 포함하며 geometry는 50m grid polygon을 유지합니다. |
| `static_grid.parquet` | Step 5 | 정적 변수 생성 결과입니다. `intersection_count`, `road_type_score`, `geometry`를 포함합니다. |
| `merged_grid_variables.parquet` | Step 7 | 정적 변수와 동적 변수를 `grid_id` 기준으로 통합한 전체 격자 데이터입니다. |
| `final_grid_variables_normalized.parquet` | Step 7 | 전체 격자를 유지한 정규화 결과입니다. 원본 변수, `_norm` 컬럼, grid polygon geometry를 함께 포함합니다. |
| `model_input_variables.parquet` | Step 7 | 모델 학습 전용 최소 입력 파일입니다. `road_type_score` 결측 행을 제외하고 `grid_id`, 이진 변수, `pm_accident`, `_norm` 연속형 변수만 유지합니다. geometry와 원본 연속형 값은 포함하지 않습니다. |
| `risk_score_grid.parquet` | Step 10 | 로지스틱 회귀 기반 위험도 모델 산출 결과입니다. 격자별·시간대별(10h/18h/22h) Risk Score, 4단계 위험등급, 시나리오(S0/S1/S2)별 제한속도를 포함합니다. |

모든 공간 산출물은 거리 계산과 buffer 계산을 위해 `EPSG:5179` 좌표계로 통일합니다.

## processed 주요 변수 설명

아래 설명은 `data/processed`에 생성된 parquet 파일의 주요 컬럼을 기준으로 정리한 것입니다. `03_gangnam_grid_50m.geojson`, `07_osm_edges_filtered.geojson`은 각각 같은 단계의 parquet을 QGIS 등에서 보기 쉽게 저장한 시각 확인용 파일입니다.

### 공통 변수

| 변수명 | 설명 |
| --- | --- |
| `geometry` | 공간 객체입니다. 파일에 따라 polygon, point, line이 들어갑니다. 모든 geometry는 `EPSG:5179` 기준입니다. |

### `01_gangnam_boundary.parquet`

| 변수명 | 설명 |
| --- | --- |
| `district_name` | 추출된 자치구 이름입니다. 현재 값은 `강남구`입니다. |
| `source_rule` | 원천 경계 데이터에서 강남구를 어떤 조건으로 추출했는지 기록한 값입니다. |
| `geometry` | 강남구 행정경계 polygon입니다. |

### `02_gangnam_grid_50m.parquet`

| 변수명 | 설명 |
| --- | --- |
| `grid_id` | 50m 격자의 고유 ID입니다. 예: `G000001` |
| `centroid_x` | 격자 중심점의 x 좌표입니다. 참고용 좌표입니다. |
| `centroid_y` | 격자 중심점의 y 좌표입니다. 참고용 좌표입니다. |
| `geometry` | 50m x 50m 격자 polygon입니다. |

### `04_osm_nodes_raw.parquet`

| 변수명 | 설명 |
| --- | --- |
| `osmid` | OSM node 고유 ID입니다. edge의 `u`, `v`와 연결됩니다. |
| `x` | 원본 OSM node의 경도 좌표입니다. |
| `y` | 원본 OSM node의 위도 좌표입니다. |
| `street_count` | OSMnx가 계산한 node 주변 도로 연결 수입니다. |
| `junction` | OSM의 교차점 관련 태그입니다. 값이 없을 수 있습니다. |
| `highway` | node에 부여된 OSM highway 태그입니다. 값이 없을 수 있습니다. |
| `geometry` | `EPSG:5179`로 변환된 node point입니다. |

### `05_osm_edges_raw.parquet`, `06_osm_edges_filtered.parquet`

| 변수명 | 설명 |
| --- | --- |
| `u` | edge 시작 node의 OSM ID입니다. |
| `v` | edge 끝 node의 OSM ID입니다. |
| `key` | 같은 `u`, `v` 사이에 여러 edge가 있을 때 구분하는 값입니다. |
| `osmid` | OSM way ID입니다. 하나의 edge에 여러 ID가 들어갈 수 있습니다. |
| `highway` | 도로 유형입니다. 예: `residential`, `secondary`, `cycleway` |
| `name` | 도로명입니다. 값이 없을 수 있습니다. |
| `oneway` | 일방통행 여부입니다. |
| `reversed` | OSMnx 그래프 생성 과정에서 방향이 반전되었는지 나타내는 값입니다. |
| `length` | OSMnx가 계산한 edge 길이입니다. 단위는 미터입니다. |
| `width` | OSM에 기록된 도로 폭입니다. 숫자, 문자열, 결측치가 섞여 있을 수 있습니다. |
| `bridge` | 교량 여부 또는 관련 태그입니다. 값이 없을 수 있습니다. |
| `tunnel` | 터널 여부 또는 관련 태그입니다. 값이 없을 수 있습니다. |
| `lanes` | 차로 수입니다. 값이 없거나 문자열일 수 있습니다. |
| `maxspeed` | 제한속도 태그입니다. 값이 없거나 문자열일 수 있습니다. |
| `ref` | 도로 번호 또는 참조 코드입니다. 값이 없을 수 있습니다. |
| `access` | 접근 제한 태그입니다. 예: `no`, `private` |
| `service` | service road의 세부 유형입니다. 값이 없을 수 있습니다. |
| `geometry` | 도로 edge의 line geometry입니다. |

`05_osm_edges_raw.parquet`은 필터링 전 원본 도로망이고, `06_osm_edges_filtered.parquet`은 PM 주행이 어려운 후보 구간을 제거한 도로망입니다.

### `08_grid_road_map.parquet`

| 변수명 | 설명 |
| --- | --- |
| `grid_id` | 도로 segment가 교차하는 50m 격자 ID입니다. |
| `edge_id` | 격자-도로 매핑을 위해 생성한 edge 고유 ID입니다. |
| `highway` | 해당 도로 segment의 OSM 도로 유형입니다. |
| `access` | 해당 도로 segment의 접근 제한 태그입니다. |
| `width` | 해당 도로 segment의 폭 관련 태그입니다. |
| `length_in_grid` | 도로 edge가 해당 격자 내부에서 차지하는 길이입니다. 단위는 미터입니다. |
| `geometry` | 격자 내부에 포함된 도로 segment line geometry입니다. |

### `09_facility_zone_grid.parquet`

| 변수명 | 설명 |
| --- | --- |
| `grid_id` | 50m 격자의 고유 ID입니다. |
| `is_school_zone` | 격자 polygon이 학교 보호구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `is_hospital_zone` | 격자 polygon이 병원 인접 구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `is_elderly_zone` | 격자 polygon이 노인시설 보호구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `geometry` | 원본 50m 격자 polygon입니다. |

### `10_dynamic_variables.parquet`

| 변수명 | 설명 |
| --- | --- |
| `grid_id` | 50m 격자의 고유 ID입니다. |
| `is_school_zone` | 격자 polygon이 학교 보호구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `is_hospital_zone` | 격자 polygon이 병원 인접 구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `is_elderly_zone` | 격자 polygon이 노인시설 보호구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `pedestrian_10h` | 해당 격자의 10시 기준 보행량 변수입니다. |
| `pedestrian_18h` | 해당 격자의 18시 기준 보행량 변수입니다. |
| `pedestrian_22h` | 해당 격자의 22시 기준 보행량 변수입니다. |
| `vehicle_10h` | 해당 격자의 10시 기준 차량량 변수입니다. |
| `vehicle_18h` | 해당 격자의 18시 기준 차량량 변수입니다. |
| `vehicle_22h` | 해당 격자의 22시 기준 차량량 변수입니다. |
| `pm_accident` | 해당 격자에 PM 사고가 있으면 `1`, 아니면 `0`입니다. |
| `geometry` | 원본 50m 격자 polygon입니다. |

### `merged_grid_variables.parquet`

| 변수명 | 설명 |
| --- | --- |
| `grid_id` | 50m 격자의 고유 ID입니다. |
| `intersection_count` | 해당 격자 내부의 교차로 수입니다. |
| `road_type_score` | 해당 격자의 도로 유형 점수입니다. |
| `is_school_zone` | 격자 polygon이 학교 보호구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `is_hospital_zone` | 격자 polygon이 병원 인접 구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `is_elderly_zone` | 격자 polygon이 노인시설 보호구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `pedestrian_10h` | 해당 격자의 10시 기준 보행량 변수입니다. |
| `pedestrian_18h` | 해당 격자의 18시 기준 보행량 변수입니다. |
| `pedestrian_22h` | 해당 격자의 22시 기준 보행량 변수입니다. |
| `vehicle_10h` | 해당 격자의 10시 기준 차량량 변수입니다. |
| `vehicle_18h` | 해당 격자의 18시 기준 차량량 변수입니다. |
| `vehicle_22h` | 해당 격자의 22시 기준 차량량 변수입니다. |
| `pm_accident` | 해당 격자에 PM 사고가 있으면 `1`, 아니면 `0`입니다. |
| `geometry` | 원본 50m 격자 polygon입니다. |

### `final_grid_variables_normalized.parquet`

| 변수명 | 설명 |
| --- | --- |
| `grid_id` | 50m 격자의 고유 ID입니다. |
| `intersection_count` | 해당 격자 내부의 교차로 수 원본값입니다. |
| `road_type_score` | 해당 격자의 도로 유형 점수 원본값입니다. |
| `is_school_zone` | 격자 polygon이 학교 보호구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `is_hospital_zone` | 격자 polygon이 병원 인접 구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `is_elderly_zone` | 격자 polygon이 노인시설 보호구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `pedestrian_10h` | 해당 격자의 10시 기준 보행량 원본값입니다. |
| `pedestrian_18h` | 해당 격자의 18시 기준 보행량 원본값입니다. |
| `pedestrian_22h` | 해당 격자의 22시 기준 보행량 원본값입니다. |
| `vehicle_10h` | 해당 격자의 10시 기준 차량량 원본값입니다. |
| `vehicle_18h` | 해당 격자의 18시 기준 차량량 원본값입니다. |
| `vehicle_22h` | 해당 격자의 22시 기준 차량량 원본값입니다. |
| `pm_accident` | 해당 격자에 PM 사고가 있으면 `1`, 아니면 `0`입니다. |
| `intersection_count_norm` | `intersection_count`를 Min-Max 0~1 정규화한 값입니다. |
| `road_type_score_norm` | `road_type_score`를 Min-Max 0~1 정규화한 값입니다. |
| `pedestrian_10h_norm` | `pedestrian_10h`를 Min-Max 0~1 정규화한 값입니다. |
| `pedestrian_18h_norm` | `pedestrian_18h`를 Min-Max 0~1 정규화한 값입니다. |
| `pedestrian_22h_norm` | `pedestrian_22h`를 Min-Max 0~1 정규화한 값입니다. |
| `vehicle_10h_norm` | `vehicle_10h`를 Min-Max 0~1 정규화한 값입니다. |
| `vehicle_18h_norm` | `vehicle_18h`를 Min-Max 0~1 정규화한 값입니다. |
| `vehicle_22h_norm` | `vehicle_22h`를 Min-Max 0~1 정규화한 값입니다. |
| `geometry` | 원본 50m 격자 polygon입니다. |

### `model_input_variables.parquet`

| 변수명 | 설명 |
| --- | --- |
| `grid_id` | 50m 격자의 고유 ID입니다. |
| `is_school_zone` | 격자 polygon이 학교 보호구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `is_hospital_zone` | 격자 polygon이 병원 인접 구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `is_elderly_zone` | 격자 polygon이 노인시설 보호구역 buffer와 겹치면 `1`, 아니면 `0`입니다. |
| `pm_accident` | 해당 격자에 PM 사고가 있으면 `1`, 아니면 `0`입니다. |
| `intersection_count_norm` | `intersection_count`를 Min-Max 0~1 정규화한 값입니다. |
| `road_type_score_norm` | `road_type_score`를 Min-Max 0~1 정규화한 값입니다. |
| `pedestrian_10h_norm` | `pedestrian_10h`를 Min-Max 0~1 정규화한 값입니다. |
| `pedestrian_18h_norm` | `pedestrian_18h`를 Min-Max 0~1 정규화한 값입니다. |
| `pedestrian_22h_norm` | `pedestrian_22h`를 Min-Max 0~1 정규화한 값입니다. |
| `vehicle_10h_norm` | `vehicle_10h`를 Min-Max 0~1 정규화한 값입니다. |
| `vehicle_18h_norm` | `vehicle_18h`를 Min-Max 0~1 정규화한 값입니다. |
| `vehicle_22h_norm` | `vehicle_22h`를 Min-Max 0~1 정규화한 값입니다. |

### `risk_score_grid.parquet`

| 변수명 | 설명 |
| --- | --- |
| `grid_id` | 50m 격자의 고유 ID입니다. |
| `RS_10h` | 10시 기준 Risk Score입니다. 범위는 0~100입니다. |
| `RS_18h` | 18시 기준 Risk Score입니다. 범위는 0~100입니다. |
| `RS_22h` | 22시 기준 Risk Score입니다. 범위는 0~100입니다. |
| `risk_level_10h` | 10시 RS를 4단계로 분류한 위험등급입니다. 일반(0\~25) / 주의(25\~50) / 위험(50\~75) / 고위험(75\~100). |
| `risk_level_18h` | 18시 RS를 4단계로 분류한 위험등급입니다. |
| `risk_level_22h` | 22시 RS를 4단계로 분류한 위험등급입니다. |
| `speed_S0_{hour}` | S0 시나리오(전 구간 25km/h 고정) 제한속도입니다. hour는 10h/18h/22h. |
| `speed_S1_{hour}` | S1 시나리오(등급별 25/20/15/10 km/h) 제한속도입니다. |
| `speed_S2_{hour}` | S2 시나리오(등급별 20/15/10/7 km/h) 제한속도입니다. |

## output files

| File | Description |
| --- | --- |
| `gangnam_grid_preview.png` | Step 2 격자 생성 결과를 확인하는 preview image입니다. |
| `osm_roads_preview.png` | 도로 수집 및 필터링 결과를 확인하는 preview image입니다. |
| `missing_road_type_score_grids.geojson` | `road_type_score`가 결측인 격자만 추출한 GIS 확인용 파일입니다. |
| `missing_road_type_score_map.html` | `road_type_score` 결측 격자를 브라우저에서 확인하는 지도 파일입니다. |
