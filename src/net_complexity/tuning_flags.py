from __future__ import annotations

import sys
from typing import Any


SEARCH_ALIASES = {
    "lambda": "model.lambda_coef",
    "lambda_coef": "model.lambda_coef",
    "lr": "optimizer.lr",
    "wd": "optimizer.weight_decay",
    "weight_decay": "optimizer.weight_decay",
    "bs": "dataloaders.batch_size",
    "batch_size": "dataloaders.batch_size",
}

TUNING_FLAG_OVERRIDES = {
    "--mode": "tuning.mode",
    "--trials": "tuning.n_trials",
    "--metric": "tuning.objective_metric",
    "--study-name": "tuning.study_name",
    "--jobs": "tuning.n_jobs",
    "--timeout": "tuning.timeout",
    "--output-dir": "tuning.output_dir",
}


def _resolve_search_path(path: str) -> str:
    return SEARCH_ALIASES.get(path, path)


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


def parse_search_flag(spec: str) -> tuple[str, dict[str, Any]]:
    if "=" not in spec:
        raise ValueError(
            "Search flag must look like "
            "'path=float:low:high[:log][:step=value]' or "
            "'path=categorical:choice1,choice2'."
        )

    path, raw_definition = spec.split("=", 1)
    path = _resolve_search_path(path.strip())
    raw_definition = raw_definition.strip()
    if not path:
        raise ValueError("Search flag path must not be empty.")
    if ":" not in raw_definition:
        raise ValueError(f"Search flag '{spec}' is missing a type definition.")

    search_type, raw_args = raw_definition.split(":", 1)
    search_type = search_type.strip().lower()

    if search_type in {"float", "int"}:
        parts = [part.strip() for part in raw_args.split(":") if part.strip()]
        if len(parts) < 2:
            raise ValueError(
                f"Search flag '{spec}' must define low and high values for type '{search_type}'."
            )

        parser = float if search_type == "float" else int
        parsed: dict[str, Any] = {
            "type": search_type,
            "low": parser(parts[0]),
            "high": parser(parts[1]),
        }

        for modifier in parts[2:]:
            lowered = modifier.lower()
            if lowered == "log":
                parsed["log"] = True
                continue
            if lowered.startswith("step="):
                step_value = modifier.split("=", 1)[1].strip()
                parsed["step"] = parser(step_value)
                continue
            raise ValueError(
                f"Unsupported modifier '{modifier}' in search flag '{spec}'. "
                "Use 'log' or 'step=<value>'."
            )

        return path, parsed

    if search_type == "categorical":
        choices = [item.strip() for item in raw_args.split(",") if item.strip()]
        if not choices:
            raise ValueError(f"Search flag '{spec}' must contain at least one categorical choice.")
        return path, {
            "type": "categorical",
            "choices": [_parse_scalar(choice) for choice in choices],
        }

    raise ValueError(
        f"Unsupported search type '{search_type}' in flag '{spec}'. "
        "Supported types: float, int, categorical."
    )


def _parse_typed_search_flag(search_type: str, spec: str) -> tuple[str, dict[str, Any]]:
    if "=" not in spec:
        raise ValueError(
            f"{search_type} search flag must look like "
            f"'name=low:high[:log][:step=value]' for numeric types or "
            f"'name=value1,value2,...' for categorical."
        )

    path, raw_definition = spec.split("=", 1)
    path = _resolve_search_path(path.strip())
    raw_definition = raw_definition.strip()
    if not path:
        raise ValueError(f"{search_type} search flag path must not be empty.")

    if search_type in {"float", "int"}:
        return parse_search_flag(f"{path}={search_type}:{raw_definition}")
    if search_type == "categorical":
        return parse_search_flag(f"{path}=categorical:{raw_definition}")
    raise ValueError(f"Unsupported typed search flag '{search_type}'.")


def preprocess_tune_argv(
    argv: list[str],
) -> tuple[list[str], bool, dict[str, dict[str, Any]], dict[str, Any]]:
    cleaned = [argv[0]]
    search_reset = False
    search_space: dict[str, dict[str, Any]] = {}
    tuning_overrides: dict[str, Any] = {}

    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--search-reset":
            search_reset = True
            index += 1
            continue

        if arg == "--search":
            if index + 1 >= len(argv):
                raise SystemExit("--search requires a value.")
            path, spec = parse_search_flag(argv[index + 1])
            search_space[path] = spec
            index += 2
            continue

        if arg.startswith("--search="):
            path, spec = parse_search_flag(arg.split("=", 1)[1])
            search_space[path] = spec
            index += 1
            continue

        if arg in {"--float", "--int", "--cat", "--categorical"}:
            if index + 1 >= len(argv):
                raise SystemExit(f"{arg} requires a value.")
            typed_flag = {
                "--float": "float",
                "--int": "int",
                "--cat": "categorical",
                "--categorical": "categorical",
            }[arg]
            path, spec = _parse_typed_search_flag(typed_flag, argv[index + 1])
            search_space[path] = spec
            index += 2
            continue

        if any(arg.startswith(prefix + "=") for prefix in ("--float", "--int", "--cat", "--categorical")):
            flag_name, value = arg.split("=", 1)
            typed_flag = {
                "--float": "float",
                "--int": "int",
                "--cat": "categorical",
                "--categorical": "categorical",
            }[flag_name]
            path, spec = _parse_typed_search_flag(typed_flag, value)
            search_space[path] = spec
            index += 1
            continue

        if arg in TUNING_FLAG_OVERRIDES:
            if index + 1 >= len(argv):
                raise SystemExit(f"{arg} requires a value.")
            tuning_overrides[TUNING_FLAG_OVERRIDES[arg]] = _parse_scalar(argv[index + 1])
            index += 2
            continue

        if arg in {"--grid", "--optuna"}:
            tuning_overrides["tuning.mode"] = "grid" if arg == "--grid" else "optuna"
            index += 1
            continue

        if arg in {"--maximize", "--minimize"}:
            tuning_overrides["tuning.direction"] = "maximize" if arg == "--maximize" else "minimize"
            index += 1
            continue

        cleaned.append(arg)
        index += 1

    return cleaned, search_reset, search_space, tuning_overrides


def install_tune_cli_flags() -> tuple[bool, dict[str, dict[str, Any]], dict[str, Any]]:
    argv, search_reset, search_space, tuning_overrides = preprocess_tune_argv(sys.argv)
    sys.argv[:] = argv
    return search_reset, search_space, tuning_overrides
