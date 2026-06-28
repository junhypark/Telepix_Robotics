"""Realistic deterministic scenario catalog for sorting runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robot_sorting.config import create_simulation_config
from robot_sorting.schemas import ObjectLabel, SimulationConfig

PICK_Z = 0.035
BIN_Z = 0.04


@dataclass(frozen=True)
class ScenarioDefinition:
    """A deterministic real-world-inspired sorting scenario."""

    scenario_id: str
    name_ko: str
    description_ko: str
    seed: int
    labels: tuple[ObjectLabel, ...]
    object_positions: tuple[tuple[float, float, float], ...]
    normal_bin_position: tuple[float, float, float] | None = None
    defect_bin_position: tuple[float, float, float] | None = None

    @property
    def objects(self) -> int:
        return len(self.labels)


SCENARIOS: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(
        scenario_id="balanced_conveyor_batch",
        name_ko="균형 혼합 배치",
        description_ko="컨베이어 검사 셀에서 정상/불량 제품이 균형 있게 섞여 들어오는 일반 생산 배치",
        seed=42,
        labels=("normal", "defect", "normal", "defect", "normal", "defect"),
        object_positions=(
            (0.20, -0.18, PICK_Z),
            (0.30, -0.18, PICK_Z),
            (0.40, -0.18, PICK_Z),
            (0.20, -0.08, PICK_Z),
            (0.30, -0.08, PICK_Z),
            (0.40, -0.08, PICK_Z),
        ),
    ),
    ScenarioDefinition(
        scenario_id="high_defect_rework_batch",
        name_ko="불량 편중 재작업 배치",
        description_ko="공정 이상 후 불량률이 높아져 red bin 적재가 많은 재작업/격리 배치",
        seed=84,
        labels=("defect", "defect", "normal", "defect", "defect", "normal"),
        object_positions=(
            (0.18, 0.16, PICK_Z),
            (0.28, 0.16, PICK_Z),
            (0.38, 0.16, PICK_Z),
            (0.18, 0.06, PICK_Z),
            (0.28, 0.06, PICK_Z),
            (0.38, 0.06, PICK_Z),
        ),
    ),
    ScenarioDefinition(
        scenario_id="normal_heavy_end_of_shift",
        name_ko="정상 편중 마감 배치",
        description_ko="라인 안정화 후 정상 제품이 대부분이고 불량이 소수만 섞인 마감 배치",
        seed=126,
        labels=("normal", "normal", "normal", "normal", "normal", "defect"),
        object_positions=(
            (0.19, -0.20, PICK_Z),
            (0.29, -0.20, PICK_Z),
            (0.39, -0.20, PICK_Z),
            (0.19, -0.10, PICK_Z),
            (0.29, -0.10, PICK_Z),
            (0.39, -0.10, PICK_Z),
        ),
    ),
    ScenarioDefinition(
        scenario_id="crowded_pick_zone_batch",
        name_ko="Pick Zone 밀집 배치",
        description_ko="제품 간 간격이 좁아 vision 분리와 bin 내부 non-overlap 적재를 검증하는 배치",
        seed=168,
        labels=("normal", "defect", "normal", "normal", "defect"),
        object_positions=(
            (0.19, 0.18, PICK_Z),
            (0.29, 0.18, PICK_Z),
            (0.39, 0.18, PICK_Z),
            (0.24, 0.08, PICK_Z),
            (0.34, 0.08, PICK_Z),
        ),
    ),
    ScenarioDefinition(
        scenario_id="bin_changeover_shift",
        name_ko="Bin 위치 변경 배치",
        description_ko="작업자가 bin 위치를 바꾼 뒤 vision 기반 동적 bin 감지와 배치 좌표 결정을 검증하는 배치",
        seed=210,
        labels=("normal", "defect", "normal", "defect", "normal"),
        object_positions=(
            (0.18, -0.16, PICK_Z),
            (0.28, -0.16, PICK_Z),
            (0.38, -0.16, PICK_Z),
            (0.23, -0.04, PICK_Z),
            (0.33, -0.04, PICK_Z),
        ),
        normal_bin_position=(0.42, 0.23, BIN_Z),
        defect_bin_position=(-0.28, 0.25, BIN_Z),
    ),
    ScenarioDefinition(
        scenario_id="small_batch_single_defect",
        name_ko="소량 단일 불량 배치",
        description_ko="시제품/샘플 검사처럼 소량 제품 중 단일 불량만 분리하는 작은 배치",
        seed=252,
        labels=("normal", "defect", "normal"),
        object_positions=(
            (0.22, 0.14, PICK_Z),
            (0.32, 0.14, PICK_Z),
            (0.42, 0.14, PICK_Z),
        ),
    ),
)


def get_scenario(scenario_id: str) -> ScenarioDefinition:
    """Return a scenario definition by id."""

    for scenario in SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    valid_ids = ", ".join(scenario.scenario_id for scenario in SCENARIOS)
    raise KeyError(f"Unknown scenario '{scenario_id}'. Valid scenarios: {valid_ids}")


def scenario_ids() -> list[str]:
    """Return every available scenario id."""

    return [scenario.scenario_id for scenario in SCENARIOS]


def create_config_for_scenario(
    scenario_id: str,
    *,
    headless: bool = True,
    width: int = 640,
    height: int = 480,
    output_dir: Path = Path("outputs/run"),
    save_images: bool = True,
) -> SimulationConfig:
    """Create a simulation config with deterministic scenario overrides applied."""

    scenario = get_scenario(scenario_id)
    config = create_simulation_config(
        headless=headless,
        objects=scenario.objects,
        seed=scenario.seed,
        width=width,
        height=height,
        output_dir=output_dir,
        save_images=save_images,
    )
    config.scenario_id = scenario.scenario_id
    config.object_label_sequence = scenario.labels
    config.object_spawn_positions = scenario.object_positions
    config.normal_bin_spawn_position = scenario.normal_bin_position
    config.defect_bin_spawn_position = scenario.defect_bin_position
    return config
