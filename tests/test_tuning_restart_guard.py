from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "net_complexity" / "tuning" / "restart_guard.py"
SPEC = importlib.util.spec_from_file_location("net_complexity.tuning.restart_guard", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

RepeatRestartGuard = MODULE.RepeatRestartGuard
RepeatRestartRequested = MODULE.RepeatRestartRequested
CollapseDetected = MODULE.CollapseDetected
CollapseGuard = MODULE.CollapseGuard


def test_restart_guard_allows_run_when_threshold_reached_before_target_epoch():
    guard = RepeatRestartGuard(metric_name="valid_accuracy", mode="max", epoch=3, threshold=0.4)
    guard(1, {}, {"valid_accuracy": 0.2}, None, None, None)
    guard(2, {}, {"valid_accuracy": 0.45}, None, None, None)
    guard(3, {}, {"valid_accuracy": 0.35}, None, None, None)


def test_restart_guard_requests_restart_at_target_epoch_when_threshold_not_reached():
    guard = RepeatRestartGuard(metric_name="valid_accuracy", mode="max", epoch=3, threshold=0.4)
    guard(1, {}, {"valid_accuracy": 0.2}, None, None, None)
    guard(2, {}, {"valid_accuracy": 0.35}, None, None, None)
    with pytest.raises(RepeatRestartRequested, match="restart requested"):
        guard(3, {}, {"valid_accuracy": 0.39}, None, None, None)


def test_collapse_guard_stops_after_patience_epochs_once_metrics_match_collapse_pattern():
    guard = CollapseGuard(
        min_epoch=35,
        patience=5,
        acc_threshold_abs=0.15,
        acc_threshold_rel=0.30,
        loss_threshold=2.25,
        zero_threshold=0.86,
    )

    guard(10, {}, {"valid_accuracy": 0.69, "valid_loss": 1.2, "valid_average_zero_prob": 0.45}, None, None, None)
    for epoch in range(35, 39):
        guard(
            epoch,
            {},
            {"valid_accuracy": 0.10, "valid_loss": 2.40, "valid_average_zero_prob": 0.90},
            None,
            None,
            None,
        )

    with pytest.raises(CollapseDetected, match="Collapse detected"):
        guard(
            39,
            {},
            {"valid_accuracy": 0.10, "valid_loss": 2.40, "valid_average_zero_prob": 0.90},
            None,
            None,
            None,
        )


def test_collapse_guard_ignores_pre_min_epoch_sequence():
    guard = CollapseGuard(
        min_epoch=35,
        patience=5,
        acc_threshold_abs=0.15,
        acc_threshold_rel=0.30,
        loss_threshold=2.25,
        zero_threshold=0.86,
    )

    guard(10, {}, {"valid_accuracy": 0.69, "valid_loss": 1.2, "valid_average_zero_prob": 0.45}, None, None, None)
    for epoch in range(30, 35):
        guard(
            epoch,
            {},
            {"valid_accuracy": 0.10, "valid_loss": 2.40, "valid_average_zero_prob": 0.90},
            None,
            None,
            None,
        )


def test_collapse_guard_resets_counter_when_condition_breaks():
    guard = CollapseGuard(
        min_epoch=35,
        patience=5,
        acc_threshold_abs=0.15,
        acc_threshold_rel=0.30,
        loss_threshold=2.25,
        zero_threshold=0.86,
    )

    guard(10, {}, {"valid_accuracy": 0.69, "valid_loss": 1.2, "valid_average_zero_prob": 0.45}, None, None, None)
    guard(35, {}, {"valid_accuracy": 0.10, "valid_loss": 2.40, "valid_average_zero_prob": 0.90}, None, None, None)
    guard(36, {}, {"valid_accuracy": 0.11, "valid_loss": 2.38, "valid_average_zero_prob": 0.91}, None, None, None)
    guard(37, {}, {"valid_accuracy": 0.25, "valid_loss": 2.00, "valid_average_zero_prob": 0.80}, None, None, None)
    for epoch in range(38, 42):
        guard(
            epoch,
            {},
            {"valid_accuracy": 0.10, "valid_loss": 2.40, "valid_average_zero_prob": 0.90},
            None,
            None,
            None,
        )

    with pytest.raises(CollapseDetected, match="Collapse detected"):
        guard(
            42,
            {},
            {"valid_accuracy": 0.10, "valid_loss": 2.40, "valid_average_zero_prob": 0.90},
            None,
            None,
            None,
        )


def test_collapse_guard_waits_until_best_is_stale_enough():
    guard = CollapseGuard(
        min_epoch=35,
        patience=3,
        min_epochs_since_best=5,
        acc_threshold_abs=0.15,
        acc_threshold_rel=0.30,
        loss_threshold=2.25,
        zero_threshold=0.86,
    )

    guard(34, {}, {"valid_accuracy": 0.69, "valid_loss": 1.2, "valid_average_zero_prob": 0.45}, None, None, None)
    guard(35, {}, {"valid_accuracy": 0.10, "valid_loss": 2.40, "valid_average_zero_prob": 0.90}, None, None, None)
    guard(36, {}, {"valid_accuracy": 0.10, "valid_loss": 2.40, "valid_average_zero_prob": 0.90}, None, None, None)
    guard(37, {}, {"valid_accuracy": 0.10, "valid_loss": 2.40, "valid_average_zero_prob": 0.90}, None, None, None)
    guard(38, {}, {"valid_accuracy": 0.10, "valid_loss": 2.40, "valid_average_zero_prob": 0.90}, None, None, None)

    with pytest.raises(CollapseDetected, match="Collapse detected"):
        guard(
            39,
            {},
            {"valid_accuracy": 0.10, "valid_loss": 2.40, "valid_average_zero_prob": 0.90},
            None,
            None,
            None,
        )
