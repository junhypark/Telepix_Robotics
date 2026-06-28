"""Integration test for non-overlapping detected-bin sorting execution."""

from __future__ import annotations

import csv
import json
from itertools import combinations

import pytest
from typer.testing import CliRunner

from robot_sorting.cli import app
from robot_sorting.planning.bin_placement_planner import objects_overlap_2d
from robot_sorting.schemas import PlacedObjectRecord, SimulationConfig

pytestmark = pytest.mark.integration


def test_sorting_execution_saves_non_overlapping_placed_objects(
    tmp_path,
    two_object_config: SimulationConfig,
) -> None:
    output_dir = tmp_path / "run"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--headless",
            "--objects",
            "4",
            "--seed",
            "7",
            "--output",
            str(output_dir),
            "--no-save-images",
        ],
    )

    assert result.exit_code == 0, result.output
    placed_path = output_dir / "placed_objects.json"
    assert placed_path.exists()
    placed_objects = [
        PlacedObjectRecord.model_validate(item)
        for item in json.loads(placed_path.read_text(encoding="utf-8"))
    ]
    grouped_by_bin: dict[str, list[PlacedObjectRecord]] = {}
    for placed in placed_objects:
        grouped_by_bin.setdefault(placed.target_bin_id, []).append(placed)

    for placed_in_bin in grouped_by_bin.values():
        for first, second in combinations(placed_in_bin, 2):
            assert not objects_overlap_2d(
                first.position,
                first.size_xyz,
                second.position,
                second.size_xyz,
                margin_meters=two_object_config.bin_placement.object_spacing_margin_meters,
            )

    with (output_dir / "result_log.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["target_bin_id"] for row in rows)
    assert all(row["placement_strategy"] for row in rows)
    assert all(row["table_clearance_checked"] == "True" for row in rows)
