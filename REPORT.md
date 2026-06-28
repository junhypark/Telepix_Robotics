# MuJoCo 로봇팔 기반 컨베이어 검사 및 분류 시뮬레이션 보고서

## 1. 프로그램 구조

본 프로젝트는 단순 pick-and-place 예제가 아니라, 현업의 검사 및 분류 자동화 셀을 PoC 수준에서 재현하는 것을 목표로 구성했다.

전체 흐름은 다음과 같다.

```text
Random Scenario Generator
  -> Conveyor Object Feeder
  -> Conveyor Controller
  -> MuJoCo Scene / Robot Simulation
  -> RGB-D Renderer
  -> FastAPI External Inspection API
  -> Object / Bin Detection
  -> Bin Interior Placement Planner
  -> Transport Mode Decision
  -> Collision-Aware Trajectory Planner
  -> Robot Command Queue
  -> Robot Controller
  -> Result Logger
  -> Dashboard / REPORT.md
```

수동으로 최종 flowchart 이미지를 만들 경우 아래 스케치를 기준으로 그리면 된다.

```text
[시드 기반 시나리오 생성]
        |
[컨베이어 투입]
        |
[검사 위치 도달 후 컨베이어 정지]
        |
[MuJoCo RGB-D 이미지 캡처]
        |
[FastAPI 외부 검사 모듈]
        |
[물품 정상/불량 판단 + 빨간/파란 박스 위치 검출]
        |
[박스 내부 빈 위치 계산]
        |
[이동 방식 판단]
   |----------------------|
   |                      |
[수평 유지 이동]     [상부 회피/회전 이동]
   |----------------------|
        |
[충돌 회피 경로 생성]
        |
[로봇팔 Pick & Place]
        |
[결과 로그 / 대시보드 / REPORT.md 생성]
```

### 영상 삽입 위치

최종 제출 시 이 섹션 바로 아래에 MuJoCo viewer에서 로봇팔이 컨베이어 물체를 집고 bin에 배치하는 영상을 넣는 것이 가장 자연스럽다.

```markdown
[MuJoCo Viewer 동작 영상](docs/media/mujoco-viewer-demo.mp4)
```

권장 녹화용 실행 명령:

```powershell
cd Telepix_Robotics\mujoco-robot-sorting
uv run robot-sort view --scenario balanced_conveyor_batch --animate --delay 0.05
```

## 2. 프로그램 구조 설계 이유

현업 자동화 셀은 로봇 제어, 비전 검사, 컨베이어 제어, 작업 계획, 결과 기록이 한 코드 덩어리로 섞이면 장애 원인 추적이 어렵다. 그래서 본 프로젝트는 기능을 다음처럼 분리했다.

- 시뮬레이션 로직과 외부 검사 로직을 분리했다.
- FastAPI를 사용해 검사 모듈이 실제 외부 서비스처럼 HTTP로 동작하도록 했다.
- RGB-D perception을 로봇 제어와 분리해 perception만 독립 테스트할 수 있게 했다.
- task planning, transport policy, trajectory planning을 분리해 의사결정 과정을 추적 가능하게 했다.
- 컨베이어 제어를 추가해 테이블 위 임의 물체 정렬이 아니라 현업의 검사/분류 cell에 가까운 흐름을 만들었다.
- result logging, dashboard, report를 추가해 실행 결과를 사람이 검증할 수 있게 했다.

## 3. 선택한 로봇 모델

사용한 로봇은 상용 UR5, Franka Panda asset이 아니라 프로젝트 내부에서 생성하는 custom educational MuJoCo MJCF robot arm이다.

주요 구조는 다음과 같다.

- base yaw
- shoulder
- elbow
- wrist / end-effector link
- two-finger 형태의 시각적 gripper

제어 방식은 분석적 IK와 waypoint 기반 이동이다. 작업 중에는 workspace limit, base/body collision risk, table/conveyor penetration risk, trajectory collision risk를 검사한다.

이 로봇은 실제 산업용 로봇의 full digital twin이 아니다. 대신 PoC 단계에서 로봇 task execution, 외부 모듈 연동, 경로 계획, 안전성 체크를 안정적으로 보여주기 위한 교육용 시뮬레이션 모델이다.

## 4. End-Effector 동작 방식

End-effector는 실제 진공압과 마찰력을 정밀하게 모델링하지 않고, suction-style logical attachment 방식으로 구현했다.

동작 순서는 다음과 같다.

1. pre-grasp 위치로 이동
2. 물체 상단 grasp 위치로 수직 접근
3. 거리 threshold 확인
4. 물체 logical attach
5. transport mode에 따라 이동
6. bin 내부 placement 위치로 접근
7. release
8. placed object state 갱신

viewer에서는 two-finger gripper geometry가 보이지만, 실제 부착 상태는 시뮬레이션 내부 state와 로그로 관리한다.

## 5. 로봇 모델 및 End-Effector 선택 이유

이번 과제의 핵심은 산업용 contact physics를 완벽히 재현하는 것이 아니라, 검사 모듈과 로봇 작업 흐름을 안정적으로 통합하는 것이다.

custom educational MJCF arm을 선택한 이유는 다음과 같다.

- 외부 mesh, license, asset download 없이 실행 가능하다.
- Docker와 CI 환경에서 재현성이 높다.
- IK, waypoint, safety check를 설명하고 테스트하기 쉽다.
- 복잡한 상용 모델보다 디버깅과 실패 원인 추적이 쉽다.

suction-style logical attachment를 선택한 이유는 다음과 같다.

- 컨베이어 위 물체를 top-down으로 집는 작업과 잘 맞는다.
- MuJoCo contact/friction 불안정성 때문에 발생하는 비본질적 실패를 줄인다.
- 자동 테스트에서 deterministic한 결과를 얻기 쉽다.
- target pose 이동, pick/place 실행, end-effector 기반 조작은 충분히 보여준다.

## 6. 외부 모듈 리스트 및 주요 기능

| 모듈 | 위치 | 주요 기능 |
|---|---|---|
| FastAPI Inspection API | `src/robot_sorting/api.py` | RGB/RGB-D 입력을 받아 정상/불량 물체와 bin 위치 판단 |
| Vision Module | `src/robot_sorting/modules/vision_module.py` | 색상, contour 기반 물체/bin 검출 |
| RGB-D Pose Estimator | `src/robot_sorting/perception` | pixel-depth 정보를 3D world pose로 변환 |
| Bin Placement Planner | `src/robot_sorting/planning/bin_placement_planner.py` | bin 내부 non-overlap placement 위치 계산 |
| Transport Policy | `src/robot_sorting/cli.py` | `level_parallel` / `overhead_rotate` 이동 방식 판단 |
| Trajectory Planner | `src/robot_sorting/planning/trajectory_planner.py` | 충돌 회피 waypoint 경로 생성 |
| Conveyor Controller | `src/robot_sorting/conveyor` | 컨베이어 이동, 정지, station state 관리 |
| Dashboard Generator | `src/robot_sorting/dashboard` | 실행 결과를 HTML dashboard로 시각화 |
| Result Logger | `src/robot_sorting/modules/result_logger.py` | CSV/JSON 결과 저장 |

## 7. 외부 모듈 및 도구 선택 이유

### FastAPI

FastAPI를 사용한 이유는 검사 모듈을 실제 외부 서비스처럼 분리하기 위해서다. 현업에서는 비전 검사 모델과 로봇 제어 시스템이 별도 프로세스 또는 별도 장비에서 HTTP/gRPC 등으로 통신하는 경우가 많다. 본 프로젝트에서는 FastAPI를 통해 이 경계를 PoC 단계에서 재현했다.

### OpenCV

OpenCV는 색상과 contour 기반 검출을 deterministic하게 구현하기 쉽다. deep learning 모델보다 성능은 제한적이지만, PoC 단계에서는 입력과 출력이 설명 가능하고 테스트하기 쉽다는 장점이 있다.

### MuJoCo

MuJoCo는 MJCF scene 생성, 로봇 관절 제어, RGB-D rendering, viewer 확인에 적합하다. 복잡한 공장 전체를 모델링하기보다는 로봇팔 task execution과 cell-level 동작을 검증하는 데 사용했다.

### Docker

Docker를 사용한 이유는 현업 기준에서 실행 환경 재현성이 매우 중요하기 때문이다.

- Python, MuJoCo, FastAPI, OpenCV 의존성을 동일하게 고정할 수 있다.
- `inspection-api`, `sim`, `dashboard`를 분리된 서비스로 실행할 수 있다.
- Docker Compose bridge network로 컨테이너 간 HTTP 통신을 검증할 수 있다.
- 평가자 또는 다른 개발자가 로컬 환경 차이 없이 같은 명령으로 실행할 수 있다.

현업 PoC에서는 “내 PC에서만 실행됨”이 아니라 “다른 환경에서도 같은 방식으로 실행됨”이 중요하다. 이 프로젝트에서 Docker는 그 기준을 맞추기 위한 선택이다.

### uv

로컬 실행에서는 `uv`를 사용했다. Anaconda가 설치된 환경에서도 프로젝트 의존성은 `uv`로 설치하고 실행하도록 분리해, Python 환경 충돌을 줄였다.

### Pytest

테스트 파일들은 PoC 단계에서 현업 기준을 충족하려고 노력한 부분이다. 단순히 동작 화면만 보여주는 것이 아니라, 아래 항목을 자동 검증한다.

- API 정상/오류 응답
- RGB/RGB-D 입력 처리
- object/bin detection
- conveyor state transition
- pick 중 conveyor 정지
- non-overlap bin placement
- trajectory safety
- table penetration 방지
- dashboard data generation
- Docker service communication

즉 테스트는 “기능이 있다”가 아니라 “반복 실행해도 기준을 만족한다”는 것을 보여주기 위한 장치다.

### Dashboard

Dashboard는 현업에서 결과를 빠르게 확인하기 위한 관찰 가능성 도구다. 개발자는 `summary.json`이나 CSV를 볼 수 있지만, 평가자나 운영자는 success rate, latency, failure reason, station timeline을 한 화면에서 확인해야 한다.

본 프로젝트의 Dashboard는 PoC 단계에서 다음을 보여준다.

- 총 물체 수
- 정상/불량 수
- 배치 성공/실패 수
- 성공률
- SLA 통과 여부
- conveyor timeline
- object result table
- failure reason
- annotated detection image
- trajectory preview image

## 8. 모듈 동작 방식

### 8.1 Random Scenario Generator

- 입력: object count, seed, scenario id
- 처리: 정상/불량 label, spawn position, bin position 생성
- 출력: deterministic scenario
- 실패 처리: workspace 밖 위치나 겹침이 있으면 안전한 위치를 재탐색

### 8.2 Conveyor Controller

- 입력: conveyor speed, entry position, inspection zone, pick zone
- 처리: `object_x += conveyor_speed_mps * dt` 방식으로 이동
- 출력: station state, conveyor event
- 실패 처리: inspection zone 도달 실패 시 failed state 기록

### 8.3 External Inspection API

- 입력: RGB image, RGB-D frame, detections
- 처리: 색상 threshold, contour, depth 기반 판정
- 출력: normal/defect object, detected bins, confidence
- 실패 처리: RGB-D 실패 시 image fallback 또는 ground-truth fallback 사용

### 8.4 RGB-D Pose Estimation

- 입력: pixel mask, depth, camera calibration
- 처리: pixel-depth를 world coordinate로 변환
- 출력: object 3D pose, size, confidence
- 실패 처리: depth 불안정 시 deterministic fallback

### 8.5 Dynamic Bin Detection

- 입력: RGB/RGB-D image
- 처리: blue/red bin contour 검출
- 출력: normal_bin, defect_bin 위치
- 실패 처리: 검출 실패 시 MuJoCo ground-truth bin 정보 사용

### 8.6 Bin Interior Placement Planning

- 입력: detected bin, object size, placed object records
- 처리: bin 내부 grid 후보 중 겹치지 않는 위치 선택
- 출력: placement position
- 실패 처리: 빈 공간이 없으면 `no_free_space_inside_bin` 기록

### 8.7 Transport Mode Decision

- 입력: pick position, place position
- 처리: 이동 거리와 방향을 기준으로 `level_parallel` 또는 `overhead_rotate` 결정
- 출력: transport mode
- 실패 처리: mode 자체가 실패를 만들지 않도록 trajectory planner에서 최종 안전성 검사

### 8.8 Collision-Aware Trajectory Planning

- 입력: grasp pose, place pose, workspace, robot geometry
- 처리: waypoint 생성, base collision, table clearance 검사
- 출력: safe trajectory
- 실패 처리: unsafe waypoint는 failure reason과 함께 거부

### 8.9 Robot Controller

- 입력: RobotCommand
- 처리: IK 계산, waypoint 추종, logical attach/release
- 출력: TaskExecutionResult
- 실패 처리: workspace, self-collision, table penetration risk를 result log에 기록

### 8.10 Logger / Dashboard

- 입력: detections, tasks, trajectories, results, timeline
- 처리: CSV/JSON/HTML artifact 생성
- 출력: `summary.json`, `result_log.csv`, `dashboard/index.html`
- 실패 처리: optional artifact가 없어도 dashboard/report가 중단되지 않도록 설계

## 9. 시뮬레이션 연동 인터페이스

### Simulation to FastAPI

```http
POST /inspect-rgbd
Content-Type: multipart/form-data
```

요청에는 다음 정보가 포함된다.

```text
rgb image file
depth .npy file
camera_calibration JSON
workspace_config JSON
```

응답 예시:

```json
{
  "objects": [
    {
      "object_id": "object_0",
      "label": "normal",
      "position": [0.25, 0.10, 0.08],
      "confidence": 0.91
    }
  ],
  "bins": [
    {
      "bin_id": "normal_bin",
      "label": "normal_bin",
      "position": [0.55, 0.20, 0.05],
      "confidence": 0.95
    }
  ],
  "processing_time_seconds": 0.04
}
```

### Planner to Controller

```json
{
  "command_type": "pick_and_place",
  "object_id": "object_0",
  "pick_position": [0.30, -0.22, 0.035],
  "place_position": [0.45, 0.28, 0.055],
  "transport_mode": "level_parallel"
}
```

외부 검사 응답을 받은 뒤 task planning, transport decision, trajectory planning, robot command queue 생성까지 0.5초 이내에 완료되도록 설계했다. 실제 로봇팔 이동 완료 시간이 아니라 post-inspection command generation path가 SLA 대상이다.

## 10. 시뮬레이션 결과 및 한계 분석

최근 Docker Compose 기반 실행에서 다음 결과를 확인했다.

```text
docker compose run --rm sim
Placed 6/6, conveyor stops=12
```

결과 테이블 예시는 다음과 같다. 실제 제출 시 `outputs/run/summary.json`의 값을 확인해 필요하면 갱신한다.

| 지표 | 값 |
|---|---:|
| 전체 물체 수 | 6 |
| 배치 성공 수 | 6 |
| 실패 수 | 0 |
| 성공률 | 100.0% |
| 컨베이어 정지 횟수 | 12 |
| 검사 station 처리 횟수 | 6 |
| table penetration failures | 0 |
| trajectory collision failures | 0 |
| no free space inside bin | 0 |

### 한계

- 로봇 모델은 교육용 simplified MJCF arm이며 산업용 로봇 digital twin은 아니다.
- suction end-effector는 실제 진공압이 아니라 logical attachment이다.
- 컨베이어는 마찰 기반 belt physics가 아니라 kinematic abstraction이다.
- vision은 deep learning이 아니라 rule-based color/contour/depth logic이다.
- collision avoidance는 보수적인 기하 검사이며 full mesh-level motion planning은 아니다.
- RGB-D rendering은 OSMesa/OpenGL 환경에 영향을 받을 수 있다.
- 실제 현장 적용에는 카메라 calibration, hardware latency 검증, 안전 인증이 필요하다.

### 개선 방향

- 실제 gripper contact physics 적용
- 정교한 motion planner 적용
- occlusion, 겹침, 누락 검출 상황 처리
- ML 기반 defect detection 적용
- 실제 camera calibration 데이터 적용
- conveyor multi-object continuous flow 확장
- dashboard 장기 run persistence 및 실시간 stream 추가

## 11. 영상 및 이미지 삽입 위치 안내

아래 위치에 사용자가 직접 영상과 이미지를 넣으면 된다.

### 11.1 MuJoCo Viewer 동작 영상

추천 위치: `## 1. 프로그램 구조`의 flowchart 다음 또는 `## 10. 시뮬레이션 결과 및 한계 분석` 시작 부분.

Markdown 삽입 예시:

```markdown
[MuJoCo Viewer 동작 영상](docs/media/mujoco-viewer-demo.mp4)
```

녹화용 실행 명령:

```powershell
cd Telepix_Robotics\mujoco-robot-sorting
uv run robot-sort view --scenario balanced_conveyor_batch --animate --delay 0.05
```

### 11.2 Dashboard 스크린샷

추천 위치: `## 7. 외부 모듈 및 도구 선택 이유`의 Dashboard 설명 아래.

Markdown 삽입 예시:

```markdown
![Dashboard screenshot](docs/media/dashboard-screenshot.png)
```

Dashboard 실행 명령:

```powershell
cd Telepix_Robotics\mujoco-robot-sorting
uv run robot-sort run --headless --objects 6 --seed 42 --random-data --enable-conveyor --enable-dashboard --output outputs/run
uv run robot-sort dashboard --output outputs/run --host 0.0.0.0 --port 8080
```

브라우저에서 확인:

```text
http://localhost:8080/dashboard
```

### 11.3 Detection / Trajectory 이미지

추천 위치: `## 10. 시뮬레이션 결과 및 한계 분석`의 결과 테이블 바로 아래.

Markdown 삽입 예시:

```markdown
![Annotated detection](mujoco-robot-sorting/outputs/run/annotated_detection.png)
![Trajectory preview](mujoco-robot-sorting/outputs/run/trajectory_preview.png)
```

이미지 생성 명령:

```powershell
cd Telepix_Robotics\mujoco-robot-sorting
uv run robot-sort run --headless --objects 6 --seed 42 --random-data --enable-conveyor --enable-dashboard --output outputs/run
```

## Appendix A. 주요 실행 산출물

| 파일 | 설명 |
|---|---|
| `detected_objects.json` | 검사 모듈이 검출한 물체 목록 |
| `planned_tasks.json` | pick/place task 목록 |
| `placed_objects.json` | bin 내부 최종 배치 상태 |
| `result_log.csv` | 각 물체별 처리 결과 |
| `summary.json` | 전체 실행 요약 |
| `station_timeline.csv` | conveyor / inspection station 상태 변화 |
| `conveyor_events.json` | conveyor event log |
| `annotated_detection.png` | 인식 결과 시각화 이미지 |
| `trajectory_preview.png` | 경로 preview 이미지 |
| `dashboard/index.html` | 실행 결과 dashboard |
| `REPORT.md` | 제출용 보고서 |

일부 optional artifact가 없는 경우에는 해당 실행에서 생성되지 않은 산출물로 간주한다.

## Appendix B. 실행 방법

### uv 실행

```powershell
cd Telepix_Robotics\mujoco-robot-sorting
uv sync --dev

uv run robot-sort run --headless --objects 6 --seed 42 --random-data --enable-conveyor --enable-dashboard --output outputs/run

uv run robot-sort dashboard --output outputs/run --host 0.0.0.0 --port 8080

uv run pytest -q
uv run ruff check .
uv run mypy src
```

### Docker 실행

```powershell
cd Telepix_Robotics\mujoco-robot-sorting
docker compose build
docker compose run --rm test
docker compose run --rm sim
docker compose up dashboard
```

Docker를 사용하는 이유는 현업 배포 및 평가 환경에서 실행 재현성을 확보하기 위해서다. `inspection-api`, `sim`, `dashboard`가 분리된 컨테이너로 실행되므로, 실제 서비스 간 통신 구조와 유사하게 PoC를 검증할 수 있다.

