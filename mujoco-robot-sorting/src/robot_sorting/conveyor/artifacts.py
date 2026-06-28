"""Output artifacts for conveyor automation-cell runs."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from robot_sorting.conveyor.conveyor_schemas import ConveyorConfig
from robot_sorting.schemas import DetectedObject, PickPlaceTask, SimulationConfig


def write_conveyor_images(
    output_dir: Path,
    detections: list[DetectedObject],
    tasks: list[PickPlaceTask],
    config: SimulationConfig,
) -> tuple[Path, Path]:
    """Write deterministic annotated detection and trajectory preview images."""

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = np.full((config.height, config.width, 3), 245, dtype=np.uint8)
    trajectory = np.full((config.height, config.width, 3), 250, dtype=np.uint8)
    _draw_workspace(annotated, config)
    _draw_workspace(trajectory, config)
    for detection in detections:
        px, py = _world_to_pixel(detection.world_position, config)
        color = (230, 60, 40) if detection.label == "defect" else (60, 90, 230)
        cv2.circle(annotated, (px, py), 10, color, thickness=-1)
        cv2.putText(
            annotated,
            detection.object_id,
            (px + 12, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 26, 33),
            1,
            cv2.LINE_AA,
        )
    for task in tasks:
        start = _world_to_pixel(task.pick_position, config)
        end = _world_to_pixel(task.place_position, config)
        color = (230, 60, 40) if task.label == "defect" else (60, 90, 230)
        cv2.circle(trajectory, start, 7, color, thickness=-1)
        cv2.circle(trajectory, end, 7, color, thickness=2)
        cv2.line(trajectory, start, end, color, thickness=2)
    annotated_path = output_dir / "annotated_detection.png"
    trajectory_path = output_dir / "trajectory_preview.png"
    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(trajectory_path), trajectory)
    return annotated_path, trajectory_path


def write_report(output_dir: Path, conveyor_config: ConveyorConfig, dashboard_dir: Path | None = None) -> Path:
    """Write a Korean Markdown report for the latest run."""

    summary = _read_json(output_dir / "summary.json")
    events = _read_json(output_dir / "conveyor_events.json").get("events", [])
    report = f"""# Robot Sorting Automation Cell Report

## 1. 작업 시나리오 개요

컨베이어 feeder가 제품을 한 개씩 공급하고, inspection station에서 멈춘 뒤 외부 FastAPI 검사 결과에 따라 정상품은 blue bin, 불량품은 red bin으로 분류합니다.

## 2. 컨베이어 기반 자동화 셀 구조

컨베이어는 물리 마찰 기반 벨트가 아니라 안정적인 kinematic abstraction입니다. 위치 갱신은 `object_position += conveyor_speed_mps * dt` 규칙을 따릅니다.

## 3. 전체 Flowchart

```text
FEEDING -> MOVING_TO_INSPECTION -> INSPECTING -> WAITING_FOR_PICK -> PICKING -> PLACING -> COMPLETED
```

## 4. 로봇 모델 및 end-effector 방식

로봇은 custom educational MuJoCo MJCF arm이며, end-effector는 suction-style logical attachment로 물체를 부착/해제합니다.

## 5. 외부 검사 모듈 연동 방식

FastAPI 검사 모듈은 `/inspect-detections`, `/inspect-image`, `/inspect-rgbd` 계약을 제공하며 Docker Compose bridge network에서 `inspection-api` 서비스로 접근합니다.

## 6. RGB-D/색상 기반 인식 방식

정상품은 파란색, 불량품은 빨간색으로 분류합니다. RGB-D가 불안정한 환경에서는 deterministic fallback을 사용합니다.

## 7. 컨베이어 상태 흐름

총 이벤트 수: {len(events)}

## 8. Pick-and-place 절차

검사 완료 후 컨베이어를 멈춘 상태에서 pick pose, lift pose, target bin pose 순서로 trajectory를 실행합니다.

## 9. Transport mode decision

pick-place 거리와 y축 이동량에 따라 `overhead_rotate` 또는 `level_parallel`을 선택하고 dashboard/event log에 기록합니다.

## 10. 충돌 회피 및 table penetration 방지

trajectory planner와 robot safety module이 workspace, self-collision, link table clearance를 검사합니다.

## 11. 박스 내 non-overlap placement

detected bin 내부의 free slot을 탐색해 이전 배치 물체와 겹치지 않는 위치를 선택합니다.

## 12. 대시보드 설명

정적 dashboard는 `{dashboard_dir or output_dir / "dashboard"}` 에 생성됩니다. 총 개수, 성공률, SLA, timeline, event, object table, preview image를 표시합니다.

## 13. 결과 지표

- total_objects: {summary.get("total_objects", 0)}
- placed_count: {summary.get("placed_count", 0)}
- failed_count: {summary.get("failed_count", 0)}
- success_rate: {summary.get("success_rate", 0.0)}
- conveyor_stop_count: {summary.get("conveyor_stop_count", 0)}
- inspection_station_count: {summary.get("inspection_station_count", 0)}
- max_command_latency_seconds: {summary.get("max_command_latency_seconds", 0.0)}

## 14. 실패 케이스 분석

실패 건수는 `{summary.get("failed_count", 0)}`건입니다. 실패 사유는 `result_log.csv`의 `failure_reason`에서 확인할 수 있습니다.

## 15. 한계 및 개선 방향

컨베이어는 안정적 테스트를 위해 kinematic abstraction으로 구현되어 실제 벨트 마찰, 제품 미끄러짐, 센서 노이즈를 완전히 모델링하지 않습니다. 향후 실제 로봇 asset, force-based gripper, ML vision detector, 실시간 dashboard persistence를 추가할 수 있습니다.

## Conveyor Config

- speed_mps: {conveyor_config.conveyor_speed_mps}
- axis: {conveyor_config.conveyor_axis}
- entry_position: {conveyor_config.entry_position}
- inspection_zone_center: {conveyor_config.inspection_zone_center}
- pick_zone_center: {conveyor_config.pick_zone_center}
"""
    path = output_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    return path


def _draw_workspace(image: np.ndarray, config: SimulationConfig) -> None:
    cv2.rectangle(image, (12, 12), (config.width - 12, config.height - 12), (150, 157, 166), 1)
    for position, color in (
        (config.conveyor_entry_position, (245, 175, 60)),
        (config.inspection_zone_center, (80, 160, 240)),
        (config.pick_zone_center, (60, 190, 120)),
        (config.normal_bin_position, (60, 90, 230)),
        (config.defect_bin_position, (230, 60, 40)),
    ):
        cv2.circle(image, _world_to_pixel(position, config), 8, color, thickness=2)


def _world_to_pixel(position: tuple[float, float, float], config: SimulationConfig) -> tuple[int, int]:
    workspace = config.workspace
    x_norm = (position[0] - workspace.x_min) / (workspace.x_max - workspace.x_min)
    y_norm = (workspace.y_max - position[1]) / (workspace.y_max - workspace.y_min)
    px = int(np.clip(round(x_norm * (config.width - 1)), 0, config.width - 1))
    py = int(np.clip(round(y_norm * (config.height - 1)), 0, config.height - 1))
    return (px, py)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    return cast(dict[str, Any], loaded)
