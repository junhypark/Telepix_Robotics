# MuJoCo Robot Sorting

MuJoCo 기반 로봇 팔이 테이블 위 물체를 외부 검사 모듈 결과에 따라 정상 bin 또는 불량 bin으로 분류하는 headless 시뮬레이션입니다. 로컬과 Docker 모두 `uv`로 실행하도록 구성했습니다.

## 과제 요구사항 매핑

| 과제 요구사항 | 구현 내용 |
|---|---|
| 작업 시나리오 구현 | 정상/불량 물체 검사 및 분류 |
| 로봇 목표 위치 이동 | pick/place 위치로 end-effector 이동 |
| end-effector 작업 수행 | suction-style logical attachment 기반 pick-and-place |
| MuJoCo 환경 | XML scene builder + `MujocoSortingEnv` simulation loop |
| 외부 모듈 연동 | FastAPI inspection service + OpenCV vision module + task planner |
| 0.5초 명령 생성 SLA | inspection result에서 command queue까지 latency 측정 |
| 안전 검증 | workspace, base exclusion, self-collision risk 테스트 |
| 테스트 | unit/integration/safety/latency tests |
| Docker | Dockerfile + docker-compose |
| uv | `pyproject.toml` + `uv.lock` |

## 로컬 실행

PowerShell:

```powershell
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run mypy src
uv run robot-sort run --headless --objects 5 --seed 42 --output outputs/run
```

외부 검사 API를 로컬에서 별도 프로세스로 실행하려면:

```powershell
uv run uvicorn robot_sorting.api:app --host 0.0.0.0 --port 8000
uv run robot-sort run --headless --objects 5 --seed 42 --output outputs/run --inspection-api-url http://localhost:8000
```

Make:

```bash
make sync
make test
make lint
make typecheck
make run
```

## Docker 실행

```powershell
docker compose up --build sim
docker compose run --rm test
docker compose run --rm lint
docker compose run --rm typecheck
```

Compose 실행 시 `inspection-api`와 `sim`은 `robot-sorting-net` bridge network에 함께 연결됩니다. `sim` 컨테이너는 `INSPECTION_API_URL=http://inspection-api:8000` 환경 변수로 외부 검사 API를 호출합니다.

## 아키텍처

```text
CLI
 ↓
SimulationConfig
 ↓
SceneBuilder → MujocoSortingEnv → Renderer
 ↓
FastAPI Inspection API → VisionModule → InspectionModule → TaskPlanner
 ↓
CommandQueue → RobotController
 ↓
ResultLogger
```

## 모듈 설명

`simulation/scene_builder.py`는 테이블, 단순 로봇 팔, 정상/불량 물체, bin, top-down camera를 포함한 MuJoCo XML을 결정적으로 생성합니다.

`simulation/mujoco_env.py`는 MuJoCo `model`과 `data`를 보관하고 step, named lookup, end-effector 위치, object 위치, fallback detection을 제공합니다. headless 모드에서는 viewer를 열지 않습니다.

`simulation/renderer.py`는 top-down camera RGB 이미지를 렌더링합니다. OpenGL/OSMesa가 없으면 `Renderer unavailable. Falling back to simulation ground-truth object positions.` 메시지와 함께 안전하게 fallback합니다.

`api.py`는 FastAPI로 색상 분류와 불량품 검사를 HTTP endpoint로 노출합니다. Docker Compose에서는 별도 `inspection-api` 컨테이너로 실행됩니다.

`modules/inspection_client.py`는 simulation 쪽에서 FastAPI 외부 검사 서비스를 호출하는 HTTP client입니다.

`modules/vision_module.py`는 RGB 이미지에서 OpenCV HSV thresholding으로 파란색 정상 물체와 빨간색 불량 물체를 검출합니다. robot controller에는 의존하지 않습니다.

`modules/inspection_module.py`는 vision 결과를 외부 검사 결과로 변환합니다. 현재는 vision label을 사용하지만, 추후 rule 또는 ML classifier로 교체하기 쉽도록 분리했습니다.

`modules/task_planner.py`는 검사 결과를 거리순 pick-and-place task로 바꾸며, 낮은 confidence, workspace 밖 좌표, base exclusion zone 내부 좌표를 거부합니다.

`modules/command_queue.py`는 task를 robot command로 만들고 command latency를 기록합니다. 0.5초 요구사항은 실제 물리 이동 완료 시간이 아니라 검사 결과가 나온 뒤 명령이 queue에 들어가기까지의 software latency입니다.

`robot/kinematics.py`는 yaw + shoulder + elbow 구조의 간단한 해석 IK를 제공합니다.

`robot/safety.py`는 workspace, base exclusion zone, self-collision risk 검사를 제공합니다.

`robot/controller.py`는 command를 받아 접근, 하강, logical attach, bin 이동, release 순서로 실행합니다.

## 출력 파일

시뮬레이션 실행 후 `outputs/run` 아래에 생성됩니다.

```text
outputs/run/result_log.csv
outputs/run/detected_objects.json
outputs/run/planned_tasks.json
outputs/run/summary.json
outputs/run/camera_rgb.png
```

`result_log.csv`에는 failure reason, command latency, self-collision/workspace check 결과가 포함됩니다. `summary.json`에는 전체 물체 수, 정상/불량 수, 성공률, 평균/최대 latency, 안전 실패 수가 포함됩니다.

## 알려진 가정

그리퍼는 안정적인 테스트를 위해 suction-style logical attachment로 구현했습니다. end-effector가 물체에 충분히 가까우면 물체를 논리적으로 attach하고, release 시 bin 위치로 pose를 갱신합니다.

Vision은 딥러닝 대신 색상 thresholding을 사용합니다. 과제 목적상 결정성과 테스트 가능성을 우선했습니다.

Pixel-to-world 변환은 top-down camera calibration이 workspace bounds와 선형으로 대응된다는 근사입니다.

Headless rendering은 OpenGL/OSMesa 환경에 따라 실패할 수 있습니다. 이 경우 MuJoCo ground-truth object positions를 fallback detection으로 사용합니다.

Docker Compose에서는 FastAPI 외부 검사 서비스와 simulation을 별도 컨테이너로 분리하고, user-defined bridge network를 통해 HTTP 통신합니다.

Axis movement 테스트는 단순화된 로봇 팔의 작은 결합 오차를 고려해 `0.03m` tolerance를 사용합니다.

0.5초 요구사항은 command generation latency입니다. 실제 simulated arm movement completion time이 아닙니다.

## Troubleshooting

렌더링이 실패하면 Docker에서 `MUJOCO_GL=osmesa`가 설정되어 있는지 확인하세요. 로컬 Windows/Anaconda 환경에서도 프로젝트 실행은 conda가 아니라 `uv run ...`을 사용합니다.

`uv sync --dev`가 Python 3.12를 찾지 못하면 uv managed Python 설치가 필요할 수 있습니다.

Docker base image `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`을 받을 수 없는 환경에서는 Astral uv의 동일 Python 3.12 계열 slim 이미지를 사용해도 됩니다.
