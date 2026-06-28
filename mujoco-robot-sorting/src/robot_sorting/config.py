"""Configuration helpers for CLI and tests."""

from __future__ import annotations

from pathlib import Path

from robot_sorting.schemas import SimulationConfig


def create_simulation_config(
    *,
    headless: bool = True,
    objects: int = 5,
    seed: int = 42,
    width: int = 640,
    height: int = 480,
    output_dir: Path = Path("outputs/run"),
    save_images: bool = True,
) -> SimulationConfig:
    """Create a validated simulation configuration from user-facing options."""

    return SimulationConfig(
        headless=headless,
        objects=objects,
        seed=seed,
        width=width,
        height=height,
        output_dir=output_dir,
        save_images=save_images,
    )

