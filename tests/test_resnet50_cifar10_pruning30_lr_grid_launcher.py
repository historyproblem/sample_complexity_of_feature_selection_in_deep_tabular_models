import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_resnet50_cifar10_pruning30_lr_grid",
    ROOT / "scripts/launch_resnet50_cifar10_pruning30_lr_grid.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_grid_is_one_matching_horizon_by_six_initial_learning_rates():
    rows = launcher.validate_grid_configs()

    assert len(rows) == 6
    assert [row["t_max"] for row in rows] == [30] * 6
    assert [row["initial_lr"] for row in rows] == pytest.approx([
        0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001,
    ])
    assert all(row["eta_min"] == 0.0 for row in rows)
    assert all(row["search_epochs"] == 30 for row in rows)


def test_epoch30_lr_factor_is_the_end_of_the_cosine():
    rows = launcher.validate_grid_configs()
    factors = {row["t_max"]: row["lr_factor_after_epoch_30"] for row in rows}

    assert factors[30] == pytest.approx(0.0, abs=1e-15)
