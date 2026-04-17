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
