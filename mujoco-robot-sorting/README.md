# MuJoCo Robot Sorting

MuJoCo 기반 로봇 팔이 RGB-D 카메라 입력을 사용해 테이블 위 물체를 검사하고, 3D pose와 grasp pose를 생성한 뒤 정상 bin 또는 불량 bin으로 분류하는 시뮬레이션입니다. 로컬과 Docker 모두 `uv` 기반으로 실행합니다.

## 요구사항 매핑

| 요구사항 | 구현 내용 |
|---|---|
| MuJoCo 로봇 팔 시뮬레이션 | deterministic XML scene, robot arm, table, bins, objects |
| 외부 검사 모듈 | 독립 FastAPI `inspection-api` 서비스 |
| RGB 이미지 업로드 | `POST /inspect-image` multipart upload |
| RGB-D 업로드 | `POST /inspect-rgbd` multipart RGB + `.npy` depth upload |
| 3D perception | camera calibration, depth processor, point cloud, pose estimator |
| grasp planning | top-down grasp pose generator |
| collision-aware planning | waypoint trajectory planner + base cylinder collision checker |
| dynamic bin detection | seed-generated blue/red bins detected through RGB-D vision |
| in-bin placement | non-overlapping detected-bin placement slots |
| table safety | shoulder/elbow/wrist/end-effector link clearance checks |
| 0.5초 SLA | inspection response 이후 task/trajectory/command queue latency 측정 |
| Docker 통신 | `inspection-api`와 `sim`이 bridge network로 HTTP 통신 |
| 테스트 | unit, API, integration, safety, latency tests |

## 로컬 실행

```powershell
cd C:\Users\SAMSUNG\Desktop\telepix\Telepix_Robotics\mujoco-robot-sorting
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run mypy src
uv run robot-sort run --headless --objects 5 --seed 42 --output outputs/run
```

외부 FastAPI 서버를 로컬에서 따로 띄워서 HTTP 통신까지 확인하려면:

```powershell
uv run uvicorn robot_sorting.api:app --host 0.0.0.0 --port 8000
uv run robot-sort run --headless --objects 5 --seed 42 --output outputs/run-api --inspection-api-url http://localhost:8000
```

## Docker 실행

```powershell
docker compose build
docker compose run --rm test
docker compose run --rm lint
docker compose run --rm typecheck
docker compose run --rm sim
```

Compose 구성:

```text
inspection-api  FastAPI 서버, 8000 포트 노출
sim             MuJoCo 시뮬레이션, INSPECTION_API_URL=http://inspection-api:8000
test            전체 pytest 실행, 같은 bridge network에서 API 통신 테스트
lint            ruff
typecheck       mypy
```

모든 서비스는 `robot-sorting-net` bridge network에 연결됩니다.

## 아키텍처

```text
MuJoCo RGB-D Renderer
 ↓
FastAPI /inspect-rgbd
 ↓
3D Object Pose + Grasp Pose
 ↓
Task Planner
 ↓
Trajectory Planner + Collision Checker
 ↓
Robot Command Queue
 ↓
Robot Controller
 ↓
Result Logger
```

Fallback 순서:

```text
1. /inspect-rgbd
2. /inspect-image
3. MuJoCo ground-truth object poses
```

렌더링 환경에서 depth 품질이 불안정하면 다음 메시지와 함께 deterministic RGB-D fallback을 사용합니다.

```text
Depth rendering unavailable. Falling back to MuJoCo ground-truth object poses.
```

## MuJoCo Viewer 테스트 방법

Viewer는 GUI가 필요하므로 Docker/headless 환경이 아니라 로컬 데스크톱에서 실행하세요.

```powershell
uv run robot-sort view --objects 5 --seed 42
```

위 명령은 기본적으로 deterministic RGB-D perception, grasp planning, collision-aware trajectory를 만든 뒤 MuJoCo viewer에서 로봇 팔 이동을 재생합니다. 더 천천히 보고 싶으면 delay를 키웁니다.

```powershell
uv run robot-sort view --objects 5 --seed 42 --animate --delay 0.05
```

정적 장면만 확인하려면:

```powershell
uv run robot-sort view --objects 5 --seed 42 --static
```

전체 pipeline 출력 파일까지 만들면서 viewer에서 움직임을 보려면:

```powershell
uv run robot-sort run --viewer --objects 5 --seed 42 --output outputs/run-viewer
```

viewer 창에서 확인할 것:

1. 테이블, 로봇 base/body, shoulder/elbow/wrist joint, two-finger gripper가 보이는지 확인합니다.
2. 파란 물체는 정상, 빨간 물체는 불량입니다.
3. 정상 bin과 불량 bin은 물체 색상 검출과 겹치지 않도록 다른 색상입니다.
4. 카메라는 top-down 고정 카메라입니다.
5. 애니메이션이 끝나면 viewer 창을 닫아 CLI를 종료합니다.

시뮬레이션 실행 결과를 확인하려면:

```powershell
uv run robot-sort run --headless --objects 5 --seed 42 --output outputs/run
```

## 출력 파일

```text
outputs/run/result_log.csv
outputs/run/detected_objects.json
outputs/run/planned_tasks.json
outputs/run/summary.json
outputs/run/camera_rgb.png
outputs/run/camera_depth.npy
```

`result_log.csv`에는 object pose, grasp pose, trajectory safety, collision check, workspace check, command latency가 포함됩니다.

`summary.json`에는 다음 3D 지표가 포함됩니다.

```json
{
  "rgbd_used": true,
  "depth_fallback_used": false,
  "ground_truth_fallback_used": false,
  "pose_estimation_success_count": 0,
  "grasp_generation_success_count": 0,
  "trajectory_collision_failures": 0
}
```

## 주요 가정

그리퍼는 안정적인 테스트를 위해 suction-style logical attachment로 구현했습니다.

Vision은 딥러닝 대신 OpenCV color thresholding을 사용합니다. 과제의 결정성과 테스트 가능성을 우선했습니다.

Pixel-depth-to-world 변환은 pinhole camera model과 top-down workspace calibration을 사용합니다.

MuJoCo depth rendering은 실행 환경의 OpenGL/OSMesa 상태에 따라 달라질 수 있어 deterministic fallback을 제공합니다.

0.5초 SLA는 inspection response 이후 task, trajectory, command queue까지의 software latency입니다. 실제 로봇 팔 이동 완료 시간은 포함하지 않습니다.

Bin 좌표는 고정 config 좌표가 아니라 seed 기반으로 생성된 blue/red bin을 RGB-D/vision으로 감지해 사용합니다. 정상 제품은 blue `normal_bin`, 불량 제품은 red `defect_bin` 안의 비어 있는 slot으로 배치되며, 같은 bin 안의 이전 배치물과 겹치지 않도록 `placed_objects.json`에 state를 남깁니다.

로봇팔 이동은 end-effector뿐 아니라 shoulder, elbow, wrist와 link segment 샘플이 table safety height 아래로 내려가지 않는지 검사합니다. 실패 시 `link_table_penetration_risk`가 result log와 summary에 기록됩니다.
