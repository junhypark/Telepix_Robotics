"""API tests for the dashboard FastAPI app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from robot_sorting.cli import app as cli_app
from robot_sorting.dashboard.dashboard_app import create_dashboard_api

pytestmark = pytest.mark.api


def test_dashboard_api_serves_latest_run(tmp_path) -> None:
    output_dir = tmp_path / "api-dashboard-run"
    run_result = CliRunner().invoke(
        cli_app,
        [
            "run",
            "--headless",
            "--objects",
            "2",
            "--seed",
            "17",
            "--enable-conveyor",
            "--enable-dashboard",
            "--output",
            str(output_dir),
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    client = TestClient(create_dashboard_api(output_dir))

    assert client.get("/health").status_code == 200
    dashboard_response = client.get("/dashboard")
    assert dashboard_response.status_code == 200
    assert "Robot Sorting Automation Cell Dashboard" in dashboard_response.text
    latest = client.get("/api/runs/latest")
    assert latest.status_code == 200
    assert latest.json()["summary"]["total_objects"] == 2
    assert client.get(f"/api/runs/{output_dir.name}/summary").status_code == 200
    assert client.get(f"/api/runs/{output_dir.name}/timeline").status_code == 200
    assert client.get(f"/api/runs/{output_dir.name}/objects").status_code == 200
    assert client.get(f"/api/runs/{output_dir.name}/events").status_code == 200
    assert client.get("/api/runs/missing/summary").status_code == 404
