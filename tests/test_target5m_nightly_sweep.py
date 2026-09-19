from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_target5m_nightly_sweep as nightly


def test_nightly_plan_has_six_searches_and_two_variants_per_threshold_pair():
    assert nightly.SEARCH_CONFIGS == (
        "search60_soft0p5_crit1p0_fixed",
        "search60_soft0p5_crit1p0_auto",
        "search60_soft0p75_crit1p0_fixed",
        "search60_soft0p75_crit1p0_auto",
        "search60_soft1p0_crit1p5_fixed",
        "search60_soft1p0_crit1p5_auto",
    )
    assert set(nightly.SEARCH_DROPS) == set(nightly.SEARCH_CONFIGS)
    assert sorted(nightly.SEARCH_DROPS.values()) == [
        (0.005, 0.010), (0.005, 0.010),
        (0.0075, 0.010), (0.0075, 0.010),
        (0.010, 0.015), (0.010, 0.015),
    ]


def test_winner_uses_validation_only_and_obeys_parameter_target():
    results = [
        {"config": "too_large", "physical_parameters": 5_100_000,
         "physical_validation_accuracy": 0.99},
        {"config": "compact_lower", "physical_parameters": 4_400_000,
         "physical_validation_accuracy": 0.93},
        {"config": "compact_best", "physical_parameters": 4_800_000,
         "physical_validation_accuracy": 0.94},
    ]
    winner, policy = nightly.select_winner(results, 5_000_000)
    assert winner["config"] == "compact_best"
    assert policy == "highest_physical_validation_accuracy_at_or_below_target"


def test_winner_falls_back_to_smallest_when_target_is_not_reached():
    results = [
        {"config": "best_accuracy", "physical_parameters": 5_400_000,
         "physical_validation_accuracy": 0.95},
        {"config": "smallest", "physical_parameters": 5_100_000,
         "physical_validation_accuracy": 0.92},
    ]
    winner, policy = nightly.select_winner(results, 5_000_000)
    assert winner["config"] == "smallest"
    assert policy == "smallest_physical_model_when_no_candidate_met_target"
