from __future__ import annotations

import sys
from typing import Any


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
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
    path = path.strip()
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


def preprocess_tune_argv(argv: list[str]) -> tuple[list[str], bool, dict[str, dict[str, Any]]]:
    cleaned = [argv[0]]
    search_reset = False
    search_space: dict[str, dict[str, Any]] = {}

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

        cleaned.append(arg)
        index += 1

    return cleaned, search_reset, search_space


def install_tune_cli_flags() -> tuple[bool, dict[str, dict[str, Any]]]:
    argv, search_reset, search_space = preprocess_tune_argv(sys.argv)
    sys.argv[:] = argv
    return search_reset, search_space
