from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "net_complexity" / "tuning" / "search.py"
SPEC = importlib.util.spec_from_file_location("net_complexity.tuning.search", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_grid_values = MODULE.build_grid_values
build_grid_search_space = MODULE.build_grid_search_space
count_grid_trials = MODULE.count_grid_trials


def test_build_grid_values_for_int_range():
    values = build_grid_values({"type": "int", "low": 2, "high": 6, "step": 2})

    assert values == [2, 4, 6]


def test_build_grid_values_for_float_range():
    values = build_grid_values({"type": "float", "low": 0.1, "high": 0.3, "step": 0.1})

    assert values == [0.1, 0.2, 0.3]


def test_build_grid_values_for_categorical():
    values = build_grid_values({"type": "categorical", "choices": [128, 256]})

    assert values == [128, 256]


def test_build_grid_search_space_counts_cartesian_product():
    grid_space = build_grid_search_space(
        {
            "optimizer.lr": {"type": "float", "low": 0.001, "high": 0.002, "step": 0.001},
            "dataloaders.batch_size": {"type": "categorical", "choices": [128, 256]},
        }
    )

    assert grid_space == {
        "optimizer.lr": [0.001, 0.002],
        "dataloaders.batch_size": [128, 256],
    }
    assert count_grid_trials(grid_space) == 4


def test_build_grid_values_rejects_log_numeric_ranges():
    with pytest.raises(ValueError, match="does not support log-scaled numeric ranges"):
        build_grid_values({"type": "float", "low": 0.001, "high": 0.01, "step": 0.001, "log": True})


def test_build_grid_values_requires_step_for_float_ranges():
    with pytest.raises(ValueError, match="must define step"):
        build_grid_values({"type": "float", "low": 0.1, "high": 0.3})
