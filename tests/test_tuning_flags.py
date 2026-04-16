from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "net_complexity" / "tuning_flags.py"
SPEC = importlib.util.spec_from_file_location("tuning_flags", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

parse_search_flag = MODULE.parse_search_flag
preprocess_tune_argv = MODULE.preprocess_tune_argv


def test_parse_float_search_flag():
    path, spec = parse_search_flag("optimizer.lr=float:0.0001:0.01:log")

    assert path == "optimizer.lr"
    assert spec == {
        "type": "float",
        "low": 0.0001,
        "high": 0.01,
        "log": True,
    }


def test_parse_int_search_flag_with_step():
    path, spec = parse_search_flag("training_arguments.num_epochs=int:20:100:step=10")

    assert path == "training_arguments.num_epochs"
    assert spec == {
        "type": "int",
        "low": 20,
        "high": 100,
        "step": 10,
    }


def test_parse_categorical_search_flag_with_numeric_choices():
    path, spec = parse_search_flag("dataloaders.batch_size=categorical:128,256,512")

    assert path == "dataloaders.batch_size"
    assert spec == {
        "type": "categorical",
        "choices": [128, 256, 512],
    }


def test_parse_search_flag_resolves_short_aliases():
    path, spec = parse_search_flag("lr=float:0.0001:0.01:log")

    assert path == "optimizer.lr"
    assert spec == {
        "type": "float",
        "low": 0.0001,
        "high": 0.01,
        "log": True,
    }


def test_preprocess_tune_argv_extracts_custom_flags():
    argv, search_reset, search_space, tuning_overrides = preprocess_tune_argv(
        [
            "tune.py",
            "--grid",
            "--search-reset",
            "--float",
            "lr=0.0001:0.01:log",
            "--cat=bs=128,256",
            "--trials",
            "10",
            "--metric",
            "valid_accuracy",
            "--maximize",
            "training_arguments.num_epochs=30",
        ]
    )

    assert argv == ["tune.py", "training_arguments.num_epochs=30"]
    assert search_reset is True
    assert search_space == {
        "optimizer.lr": {
            "type": "float",
            "low": 0.0001,
            "high": 0.01,
            "log": True,
        },
        "dataloaders.batch_size": {
            "type": "categorical",
            "choices": [128, 256],
        },
    }
    assert tuning_overrides == {
        "tuning.mode": "grid",
        "tuning.n_trials": 10,
        "tuning.objective_metric": "valid_accuracy",
        "tuning.direction": "maximize",
    }


def test_parse_search_flag_rejects_unknown_modifier():
    with pytest.raises(ValueError, match="Unsupported modifier"):
        parse_search_flag("optimizer.lr=float:0.0001:0.01:weird")
