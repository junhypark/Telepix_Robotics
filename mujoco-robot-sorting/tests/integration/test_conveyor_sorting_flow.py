"""Integration tests for the conveyor-enabled sorting flow."""

from __future__ import annotations

import csv
import json

import pytest
from typer.testing import CliRunner

from robot_sorting.cli import app

pytestmark = pytest.mark.integration


def test_conveyor_sorting_flow_creates_required_outputs(tmp_path) -> None:
    output_dir = tmp_path / "conveyor-run"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--headless",
            "--objects",
            "2",
            "--seed",
            "11",
            "--random-data",
            "--enable-conveyor",
            "--enable-dashboard",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    with (output_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    assert summary["conveyor_enabled"] is True
    assert summary["inspection_station_count"] >= 1
    assert summary["conveyor_stop_count"] >= 1
    assert summary["pick_station_success_count"] == summary["total_objects"]
    assert (output_dir / "station_timeline.csv").exists()
    assert (output_dir / "conveyor_events.json").exists()
    assert (output_dir / "dashboard_data.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "annotated_detection.png").exists()
    assert (output_dir / "trajectory_preview.png").exists()

    with (output_dir / "result_log.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_label = {row["label"]: row["target_bin"] for row in rows}
    assert by_label["normal"] == "normal_bin"
    assert by_label["defect"] == "defect_bin"
