"""Unit tests for dashboard data loading."""

from __future__ import annotations

import json

import pytest

from robot_sorting.dashboard.run_data_loader import build_dashboard_data

pytestmark = pytest.mark.unit


def test_dashboard_builder_loads_required_cards_without_images(tmp_path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "total_objects": 2,
                "normal_count": 1,
                "defect_count": 1,
                "placed_count": 2,
                "failed_count": 0,
                "success_rate": 1.0,
                "average_command_latency_seconds": 0.01,
                "max_command_latency_seconds": 0.02,
                "conveyor_enabled": True,
                "conveyor_stop_count": 2,
                "overhead_rotate_count": 1,
                "level_parallel_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "result_log.csv").write_text(
        "object_id,label,status,target_bin,failure_reason,command_latency_seconds\n"
        "object_0,normal,completed,normal_bin,,0.01\n"
        "object_1,defect,completed,defect_bin,,0.02\n",
        encoding="utf-8",
    )
    (tmp_path / "station_timeline.csv").write_text(
        "timestamp,state,active_object_id,event\n0.0,FEEDING,object_0,object_spawned\n",
        encoding="utf-8",
    )
    (tmp_path / "conveyor_events.json").write_text(
        json.dumps({"events": [{"object_id": "object_0", "transport_mode": "level_parallel"}]}),
        encoding="utf-8",
    )

    data = build_dashboard_data(tmp_path)

    assert data.summary.total_objects == 2
    assert data.summary.sla_passed is True
    assert data.summary.normal_bin_detected is True
    assert data.summary.defect_bin_detected is True
    assert len(data.objects) == 2
    assert data.images["annotated_detection"] == "annotated_detection.png"
