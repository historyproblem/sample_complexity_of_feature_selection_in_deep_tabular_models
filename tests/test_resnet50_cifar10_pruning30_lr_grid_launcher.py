import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_resnet50_cifar10_pruning30_lr_grid",
    ROOT / "scripts/launch_resnet50_cifar10_pruning30_lr_grid.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_grid_is_two_horizons_by_six_initial_learning_rates():
    rows = launcher.validate_grid_configs()

    assert len(rows) == 12
    assert [row["t_max"] for row in rows] == [30] * 6 + [75] * 6
    assert [row["initial_lr"] for row in rows[:6]] == pytest.approx([
        0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001,
    ])
    assert all(row["eta_min"] == 0.0 for row in rows)
    assert all(row["search_epochs"] == 30 for row in rows)


def test_epoch30_lr_factors_capture_full_and_partial_cosines():
    rows = launcher.validate_grid_configs()
    factors = {row["t_max"]: row["lr_factor_after_epoch_30"] for row in rows}

    assert factors[30] == pytest.approx(0.0, abs=1e-15)
    assert factors[75] == pytest.approx(
        (1.0 + math.cos(math.pi * 30 / 75)) / 2.0
    )


def test_search_protocol_accepts_an_explicit_positive_cosine_horizon():
    from net_complexity.training import one_shot_pruning_config as schema

    config = schema.compose_config(
        launcher.CONFIG,
        overrides=["one_shot.search_scheduler_horizon_epochs=75"],
    )
    assert schema.validate_config(config) == 150
    assert config.one_shot.search_scheduler_horizon_epochs == 75

    config.one_shot.search_scheduler_horizon_epochs = 0
    with pytest.raises(ValueError, match="positive integer"):
        schema.validate_config(config)
