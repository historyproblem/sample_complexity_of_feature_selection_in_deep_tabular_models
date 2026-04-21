from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "net_complexity" / "tuning" / "repeats.py"
SPEC = importlib.util.spec_from_file_location("net_complexity.tuning.repeats", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

resolve_repeat_seeds = MODULE.resolve_repeat_seeds
resolve_repeat_attempt_seed = MODULE.resolve_repeat_attempt_seed
select_best_repeat = MODULE.select_best_repeat


def test_resolve_repeat_seeds_generates_distinct_sequence():
    assert resolve_repeat_seeds(3, seed_base=42, seed_stride=5) == [42, 47, 52]


def test_resolve_repeat_seeds_allows_missing_seed_base():
    assert resolve_repeat_seeds(2, seed_base=None, seed_stride=1) == [None, None]


def test_resolve_repeat_attempt_seed_uses_distinct_attempt_offsets():
    assert resolve_repeat_attempt_seed(42, attempt_number=1, repeats_per_trial=3, seed_stride=2) == 42
    assert resolve_repeat_attempt_seed(42, attempt_number=2, repeats_per_trial=3, seed_stride=2) == 48


def test_select_best_repeat_for_maximize():
    best = select_best_repeat(
        "maximize",
        [
            {"repeat_number": 1, "objective_value": 0.81},
            {"repeat_number": 2, "objective_value": 0.79},
            {"repeat_number": 3, "objective_value": 0.84},
        ],
    )
    assert best["repeat_number"] == 3


def test_select_best_repeat_for_minimize():
    best = select_best_repeat(
        "minimize",
        [
            {"repeat_number": 1, "objective_value": 0.35},
            {"repeat_number": 2, "objective_value": 0.29},
            {"repeat_number": 3, "objective_value": 0.31},
        ],
    )
    assert best["repeat_number"] == 2


def test_resolve_repeat_seeds_rejects_non_positive_stride():
    with pytest.raises(ValueError, match="seed_stride"):
        resolve_repeat_seeds(2, seed_base=42, seed_stride=0)
