from __future__ import annotations

import sys
from typing import Any


SUPPORTED_CLI_OVERRIDES = {
    "--seed-base": "tuning.seed_base",
    "--seed-stride": "tuning.seed_stride",
    "--restart-max-attempts": "tuning.restart_guard.max_attempts_per_repeat",
}


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def preprocess_tune_argv(
    argv: list[str],
) -> tuple[list[str], dict[str, Any]]:
    cleaned = [argv[0]]
    tuning_overrides: dict[str, Any] = {}

    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg in SUPPORTED_CLI_OVERRIDES:
            if index + 1 >= len(argv):
                raise SystemExit(f"{arg} requires a value.")
            tuning_overrides[SUPPORTED_CLI_OVERRIDES[arg]] = _parse_scalar(argv[index + 1])
            index += 2
            continue

        if arg == "--restart-below-acc":
            if index + 1 >= len(argv):
                raise SystemExit("--restart-below-acc requires a value like '20:0.4'.")
            value = argv[index + 1]
            if ":" not in value:
                raise SystemExit("--restart-below-acc must look like '20:0.4'.")
            epoch_raw, threshold_raw = value.split(":", 1)
            tuning_overrides["tuning.restart_guard.enabled"] = True
            tuning_overrides["tuning.restart_guard.metric"] = "valid_accuracy"
            tuning_overrides["tuning.restart_guard.mode"] = "max"
            tuning_overrides["tuning.restart_guard.epoch"] = int(epoch_raw)
            tuning_overrides["tuning.restart_guard.threshold"] = float(threshold_raw)
            index += 2
            continue

        if arg.startswith("--restart-below-acc="):
            value = arg.split("=", 1)[1]
            if ":" not in value:
                raise SystemExit("--restart-below-acc must look like '20:0.4'.")
            epoch_raw, threshold_raw = value.split(":", 1)
            tuning_overrides["tuning.restart_guard.enabled"] = True
            tuning_overrides["tuning.restart_guard.metric"] = "valid_accuracy"
            tuning_overrides["tuning.restart_guard.mode"] = "max"
            tuning_overrides["tuning.restart_guard.epoch"] = int(epoch_raw)
            tuning_overrides["tuning.restart_guard.threshold"] = float(threshold_raw)
            index += 1
            continue

        cleaned.append(arg)
        index += 1

    return cleaned, tuning_overrides


def install_tune_cli_flags() -> dict[str, Any]:
    argv, tuning_overrides = preprocess_tune_argv(sys.argv)
    sys.argv[:] = argv
    return tuning_overrides
