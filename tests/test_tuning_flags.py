from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "net_complexity" / "tuning" / "flags.py"
SPEC = importlib.util.spec_from_file_location("net_complexity.tuning.flags", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

preprocess_tune_argv = MODULE.preprocess_tune_argv
SUPPORTED_CLI_OVERRIDES = MODULE.SUPPORTED_CLI_OVERRIDES


def test_supported_cli_overrides_only_include_seed_and_restart_shortcuts():
    assert SUPPORTED_CLI_OVERRIDES == {
        "--seed-base": "tuning.seed_base",
        "--seed-stride": "tuning.seed_stride",
        "--restart-max-attempts": "tuning.restart_guard.max_attempts_per_repeat",
    }


def test_preprocess_tune_argv_extracts_only_seed_and_restart_flags():
    argv, tuning_overrides = preprocess_tune_argv(
        [
            "tune.py",
            "--seed-base",
            "100",
            "--seed-stride",
            "7",
            "--restart-below-acc",
            "20:0.4",
            "--restart-max-attempts",
            "5",
            "tuning.mode=grid",
            "tuning.n_trials=10",
            "training_arguments.num_epochs=30",
        ]
    )

    assert argv == [
        "tune.py",
        "tuning.mode=grid",
        "tuning.n_trials=10",
        "training_arguments.num_epochs=30",
    ]
    assert tuning_overrides == {
        "tuning.seed_base": 100,
        "tuning.seed_stride": 7,
        "tuning.restart_guard.enabled": True,
        "tuning.restart_guard.metric": "valid_accuracy",
        "tuning.restart_guard.mode": "max",
        "tuning.restart_guard.epoch": 20,
        "tuning.restart_guard.threshold": 0.4,
        "tuning.restart_guard.max_attempts_per_repeat": 5,
    }


def test_preprocess_tune_argv_supports_inline_restart_flag_value():
    argv, tuning_overrides = preprocess_tune_argv(
        [
            "tune.py",
            "--restart-below-acc=15:0.55",
            "experiment=stg_cifar10_120",
        ]
    )

    assert argv == ["tune.py", "experiment=stg_cifar10_120"]
    assert tuning_overrides == {
        "tuning.restart_guard.enabled": True,
        "tuning.restart_guard.metric": "valid_accuracy",
        "tuning.restart_guard.mode": "max",
        "tuning.restart_guard.epoch": 15,
        "tuning.restart_guard.threshold": 0.55,
    }
