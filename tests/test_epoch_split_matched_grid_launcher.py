import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_epoch_split_matched_grid",
    ROOT / "scripts/launch_epoch_split_matched_grid.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_grid_is_six_strict_auto_step_plain_ce_profiles():
    rows = launcher.validate_grid_configs()

    assert [(row["target"], row["split"]) for row in rows] == [
        ("18m", "45/105"), ("18m", "60/90"),
        ("12m", "45/105"), ("12m", "60/90"),
        ("5m", "45/105"), ("5m", "60/90"),
    ]
    assert all(row["soft_drop"] * 2 == row["hard_drop"] for row in rows)
    assert [row["resolved_log_step"] for row in rows if row["target"] == "18m"] == (
        pytest.approx([1.315762910282312, 0.9210340371976183])
    )
    assert [row["initial_search_warmup"] for row in rows] == [8, 10, 8, 10, 8, 10]


def test_middle_60_90_profile_reuses_the_completed_standard_anchor(tmp_path):
    runs = tmp_path / "runs"
    output = runs / "grid"
    row = next(
        row for row in launcher._profile_rows()
        if row["target"] == "12m" and row["search_epochs"] == 60
    )

    search, recovery, external = launcher._run_paths(output, row, runs, True)

    assert external is True
    assert search == runs / "epoch_split_auto_60_90_depsafe_search"
    assert recovery == runs / "epoch_split_auto_60_90_depsafe_bn200_fixed"


def test_single_gpu_plan_fits_fifteen_hours_when_standard_anchor_exists(tmp_path):
    runs = tmp_path / "runs"
    output = runs / "grid"
    (runs / "epoch_split_auto_60_90_depsafe_search").mkdir(parents=True)
    (runs / "epoch_split_auto_60_90_depsafe_bn200_fixed").mkdir(parents=True)

    plan = launcher._estimate_plan(output, runs, True)

    assert plan["fresh_pairs"] == 5
    assert plan["estimated_single_gpu_hours"] == pytest.approx(14.4)


def test_validation_winner_is_locked_only_for_two_size_matched_feasible_runs():
    def row(split, params, accuracy, feasible=True):
        search_epochs = int(split.split("/")[0])
        return {
            "target": "5m",
            "target_parameters": 5_000_000,
            "split": split,
            "search_epochs": search_epochs,
            "recovery_epochs": 150 - search_epochs,
            "recovery": {
                "physical_parameters": params,
                "quality_feasible": feasible,
                "validation_accuracy": accuracy,
                "validation_ce_loss": .4,
            },
        }

    matched = launcher._comparison_by_target(
        [
            row("45/105", 5_100_000, .935), row("60/90", 4_950_000, .934),
            {**row("45/105", 12_100_000, .94), "target": "12m",
             "target_parameters": 12_000_000},
            {**row("60/90", 11_900_000, .939), "target": "12m",
             "target_parameters": 12_000_000},
            {**row("45/105", 18_100_000, .944), "target": "18m",
             "target_parameters": 18_000_000},
            {**row("60/90", 17_900_000, .943), "target": "18m",
             "target_parameters": 18_000_000},
        ],
        .15,
    )
    assert matched["5m"]["winner_split"] == "45/105"

    unmatched_rows = [
        row("45/105", 7_000_000, .94), row("60/90", 5_000_000, .93),
        {**row("45/105", 12_000_000, .94), "target": "12m",
         "target_parameters": 12_000_000},
        {**row("60/90", 12_000_000, .93), "target": "12m",
         "target_parameters": 12_000_000},
        {**row("45/105", 18_000_000, .94), "target": "18m",
         "target_parameters": 18_000_000},
        {**row("60/90", 18_000_000, .93), "target": "18m",
         "target_parameters": 18_000_000},
    ]
    unmatched = launcher._comparison_by_target(unmatched_rows, .15)
    assert unmatched["5m"]["winner_split"] is None
    assert unmatched["5m"]["matched"] is False
