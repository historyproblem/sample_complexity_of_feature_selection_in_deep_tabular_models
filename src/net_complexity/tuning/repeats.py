from __future__ import annotations

from typing import Any


def resolve_repeat_seeds(
    repeats_per_trial: int,
    *,
    seed_base: int | None,
    seed_stride: int,
) -> list[int | None]:
    if repeats_per_trial <= 0:
        raise ValueError("tuning.repeats_per_trial must be >= 1.")
    if seed_stride <= 0:
        raise ValueError("tuning.seed_stride must be >= 1.")
    if seed_base is None:
        return [None] * repeats_per_trial
    return [int(seed_base) + repeat_index * int(seed_stride) for repeat_index in range(repeats_per_trial)]


def select_best_repeat(direction: str, repeat_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not repeat_results:
        raise ValueError("repeat_results must not be empty.")
    if direction == "maximize":
        return max(repeat_results, key=lambda result: float(result["objective_value"]))
    if direction == "minimize":
        return min(repeat_results, key=lambda result: float(result["objective_value"]))
    raise ValueError("direction must be either 'maximize' or 'minimize'.")
