"""Integration tests for the full production scenario runner."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from robot_sorting.cli import app
from robot_sorting.scenarios import scenario_ids

pytestmark = pytest.mark.integration


def test_run_scenarios_passes_all_catalog_entries(tmp_path) -> None:
    output_dir = tmp_path / "scenarios"
    result = CliRunner().invoke(
        app,
        [
            "run-scenarios",
            "--output",
            str(output_dir),
            "--no-save-images",
        ],
    )

    assert result.exit_code == 0, result.output
    results_path = output_dir / "scenario_results.json"
    assert results_path.exists()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert {item["scenario_id"] for item in results} == set(scenario_ids())
    assert all(item["passed"] for item in results)
