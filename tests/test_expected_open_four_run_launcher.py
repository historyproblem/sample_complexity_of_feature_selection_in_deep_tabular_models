import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_resnet50_expected_open_four_runs",
    ROOT / "scripts/launch_resnet50_expected_open_four_runs.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_suite_is_exactly_four_60_plus_90_models():
    plan = launcher.validate_suite_configs()

    assert plan["model_runs"] == 4
    assert plan["search_epochs_per_model"] == 60
    assert plan["recovery_epochs_per_model"] == 90
    assert plan["epochs_per_model"] == 150
    assert plan["total_training_epochs"] == 600
    assert plan["drop_mode"] == "expected_open_count"
    assert plan["checkpoint_retention"] == "online_selection"
    assert plan["gate_threshold_replaced"] == 0.5
    assert {
        ((row["soft_drop"], row["hard_drop"]), row["entropy_beta"])
        for row in plan["runs"]
    } == {
        ((0.0005, 0.001), 0.0),
        ((0.0005, 0.001), 0.3),
        ((0.01, 0.02), 0.0),
        ((0.01, 0.02), 0.3),
    }


def test_each_model_builds_one_search_and_one_recovery_command(tmp_path):
    plan = launcher.validate_suite_configs()
    dense = tmp_path / "dense"

    commands = []
    for row in plan["runs"]:
        search = tmp_path / row["id"] / "search"
        recovery = tmp_path / row["id"] / "recovery"
        search_command = launcher._launcher_command(
            row["search_config"], dense, search, row=row, search_only=True,
        )
        recovery_command = launcher._launcher_command(
            row["recovery_config"], dense, recovery, row=row, reuse_search=search,
        )
        commands.extend((search_command, recovery_command))

        assert search_command.count("--search-only") == 1
        assert "--reuse-search" not in search_command
        assert recovery_command.count("--reuse-search") == 1
        assert "--search-only" not in recovery_command
        assert row["search_config"] in search_command
        assert row["recovery_config"] in recovery_command
        assert sum(value == "--override" for value in search_command) == 4
        assert sum(value == "--override" for value in recovery_command) == 4

    assert len(commands) == 8


def test_checkpoint_cleanup_is_narrow_and_keeps_root_artifacts(tmp_path):
    search = tmp_path / "search"
    recovery = tmp_path / "recovery"
    root_artifacts = []
    for root in (search, recovery):
        root.mkdir()
        for name in ("selected_checkpoint.pt", "deployment.pt"):
            path = root / name
            path.write_bytes(b"keep")
            root_artifacts.append(path)
        nested = root / "stage/checkpoints"
        nested.mkdir(parents=True)
        (nested / "epoch_0001.pt").write_bytes(b"remove")
        (nested / "best.pt").write_bytes(b"remove")
        (nested / "last.pt").write_bytes(b"remove")

    result = launcher._prune_pair_checkpoints(search, recovery)

    assert result == {"files": 6, "bytes": 36}
    assert not list(tmp_path.rglob("checkpoints/*.pt"))
    assert all(path.read_bytes() == b"keep" for path in root_artifacts)
