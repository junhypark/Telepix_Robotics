"""Integration tests for static dashboard generation."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from robot_sorting.cli import app

pytestmark = pytest.mark.integration


def test_static_dashboard_generation_after_run(tmp_path) -> None:
    output_dir = tmp_path / "dashboard-run"
    run_result = CliRunner().invoke(
        app,
        [
            "run",
            "--headless",
            "--objects",
            "2",
            "--seed",
            "13",
            "--enable-conveyor",
            "--enable-dashboard",
            "--output",
            str(output_dir),
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    dashboard_result = CliRunner().invoke(app, ["dashboard", "--output", str(output_dir), "--static"])

    assert dashboard_result.exit_code == 0, dashboard_result.output
    dashboard_dir = output_dir / "dashboard"
    html = (dashboard_dir / "index.html").read_text(encoding="utf-8")
    assert (dashboard_dir / "dashboard_data.json").exists()
    assert "success_rate" in html
    assert "Robot Sorting Automation Cell Dashboard" in html
