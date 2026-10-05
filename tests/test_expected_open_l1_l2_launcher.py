import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_resnet50_expected_open_l1_l2",
    ROOT / "scripts/launch_resnet50_expected_open_l1_l2.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_suite_runs_only_four_missing_l1_l2_models():
    plan = launcher.validate_suite_configs()

    assert plan["model_runs"] == 4
    assert plan["total_new_training_epochs"] == 600
    assert plan["drop_mode"] == "expected_open_count"
    assert plan["entropy_regularization"] == "disabled"
    assert plan["baseline"]["prepare_if_missing"] is False
    assert {
        ((row["soft_drop"], row["hard_drop"]), row["gate_regularization"])
        for row in plan["runs"]
    } == launcher.EXPECTED_NEW_RUNS
    assert {
        ((row["soft_drop"], row["hard_drop"]),
         row["gate_regularization"], row["prior_suite_id"])
        for row in plan["skipped_completed_runs"]
    } == launcher.EXPECTED_SKIPPED_RUNS


def test_each_new_model_has_one_60_search_and_one_90_recovery(tmp_path):
    plan = launcher.validate_suite_configs()
    commands = []
    for row in plan["runs"]:
        search = tmp_path / row["id"] / "search"
        recovery = tmp_path / row["id"] / "recovery"
        search_command = launcher._launcher_command(
            row["search_config"], tmp_path / "dense", search,
            row=row, search_only=True,
        )
        recovery_command = launcher._launcher_command(
            row["recovery_config"], tmp_path / "dense", recovery,
            row=row, reuse_search=search,
        )
        commands.extend((search_command, recovery_command))
        assert search_command.count("--search-only") == 1
        assert recovery_command.count("--reuse-search") == 1
        assert sum(value == "--override" for value in search_command) == 5
        assert sum(value == "--override" for value in recovery_command) == 5
        assert any(row["gate_regularization"] in value for value in search_command)
    assert len(commands) == 8
