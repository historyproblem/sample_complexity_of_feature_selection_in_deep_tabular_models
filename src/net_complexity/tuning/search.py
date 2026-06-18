from __future__ import annotations

from decimal import Decimal
from functools import reduce
from operator import mul
from typing import Any, Mapping

from omegaconf import DictConfig, ListConfig, OmegaConf

GRID_POINT_INDEX_PARAM = "__grid_point_index__"


def _get(spec: Any, key: str, default: Any = None) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(key, default)
    return getattr(spec, key, default)


def _expand_float_grid(low: Any, high: Any, step: Any) -> list[float]:
    low_decimal = Decimal(str(low))
    high_decimal = Decimal(str(high))
    step_decimal = Decimal(str(step))
    if step_decimal <= 0:
        raise ValueError("Grid search step must be positive.")

    values: list[float] = []
    current = low_decimal
    epsilon = step_decimal / Decimal("1000000")
    while current <= high_decimal + epsilon:
        values.append(float(current))
        current += step_decimal
    return values


def _expand_int_grid(low: Any, high: Any, step: Any) -> list[int]:
    step_value = int(step)
    if step_value <= 0:
        raise ValueError("Grid search step must be positive.")
    return list(range(int(low), int(high) + 1, step_value))


def build_grid_values(spec: Any) -> list[Any]:
    explicit_values = _get(spec, "values")
    if explicit_values is not None:
        values = list(explicit_values)
        if not values:
            raise ValueError("Grid search values list must not be empty.")
        return values

    search_type = str(_get(spec, "type")).lower()
    if search_type == "categorical":
        values = list(_get(spec, "choices", []))
        if not values:
            raise ValueError("Categorical grid search spec must define at least one choice.")
        return values

    if bool(_get(spec, "log", False)):
        raise ValueError(
            "Grid search does not support log-scaled numeric ranges. "
            "Use categorical choices or explicit values instead."
        )

    low = _get(spec, "low")
    high = _get(spec, "high")
    if low is None or high is None:
        raise ValueError("Numeric grid search spec must define both low and high.")

    if search_type == "int":
        return _expand_int_grid(low, high, _get(spec, "step", 1))

    if search_type == "float":
        step = _get(spec, "step")
        if step is None:
            raise ValueError(
                "Float grid search spec must define step. "
                "Use step=... or categorical choices."
            )
        return _expand_float_grid(low, high, step)

    raise ValueError(f"Unsupported grid search type '{search_type}'.")


def build_grid_search_space(search_space: Mapping[str, Any]) -> dict[str, list[Any]]:
    return {str(path): build_grid_values(spec) for path, spec in search_space.items()}


def _to_plain_value(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=False)
    if isinstance(value, Mapping):
        return {str(key): _to_plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain_value(item) for item in value]
    return value


def build_grid_points(points: Any) -> list[dict[str, Any]]:
    if points is None:
        return []

    resolved_points: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        if not isinstance(point, Mapping):
            raise ValueError(
                "Each explicit grid point must be a mapping of config paths to values. "
                f"Got {type(point).__name__} at index {index}."
            )
        normalized_point = {
            str(path): _to_plain_value(value)
            for path, value in point.items()
        }
        if not normalized_point:
            raise ValueError(f"Explicit grid point at index {index} must not be empty.")
        resolved_points.append(normalized_point)

    return resolved_points


def build_point_grid_search_space(grid_points: list[Mapping[str, Any]]) -> dict[str, list[int]]:
    if not grid_points:
        raise ValueError("Explicit grid points must not be empty.")
    return {GRID_POINT_INDEX_PARAM: list(range(len(grid_points)))}


def count_grid_trials(grid_search_space: Mapping[str, list[Any]]) -> int:
    if not grid_search_space:
        return 0
    return reduce(mul, (len(values) for values in grid_search_space.values()), 1)
