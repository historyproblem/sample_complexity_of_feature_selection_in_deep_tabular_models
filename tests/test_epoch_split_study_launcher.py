import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_epoch_split_study",
    ROOT / "scripts/launch_epoch_split_study.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_all_study_configs_are_strictly_validated():
    rows = launcher.validate_study_configs()
    assert [row["split"] for row in rows] == ["30/120", "45/105", "60/90", "75/75"]


def test_auto_step_study_configs_resolve_from_each_search_horizon():
    rows = launcher.validate_study_configs("auto")
    assert [row["search_config"] for row in rows] == [
        "experiment/pruning_v3/epoch_split_auto_30_120_search",
        "experiment/pruning_v3/epoch_split_auto_45_105_search",
        "experiment/pruning_v3/epoch_split_auto_60_90_search",
        "experiment/pruning_v3/epoch_split_auto_75_75_search",
    ]
    assert [row["initial_search_warmup"] for row in rows] == [5, 8, 10, 13]
    assert [row["resolved_log_step"] for row in rows] == pytest.approx([
        1.8420680743952367,
        1.315762910282312,
        0.9210340371976183,
        0.7675283643313486,
    ])


def test_validation_winner_excludes_infeasible_and_uses_validation_ties():
    rows = [
        dict(split="30/120", search_epochs=30, quality_feasible=True,
             validation_accuracy=.94, validation_ce_loss=.20, physical_parameters=17_000_000),
        dict(split="45/105", search_epochs=45, quality_feasible=False,
             validation_accuracy=.96, validation_ce_loss=.10, physical_parameters=16_000_000),
        dict(split="60/90", search_epochs=60, quality_feasible=True,
             validation_accuracy=.94, validation_ce_loss=.19, physical_parameters=17_100_000),
        dict(split="75/75", search_epochs=75, quality_feasible=True,
             validation_accuracy=.94, validation_ce_loss=.19, physical_parameters=16_900_000),
    ]
    winner, policy = launcher.select_validation_winner(rows)
    assert winner["split"] == "75/75"
    assert "validation" in policy


def test_validation_winner_refuses_no_quality_feasible_deployment():
    with pytest.raises(RuntimeError, match="No recovery split"):
        launcher.select_validation_winner([
            dict(split="30/120", search_epochs=30, quality_feasible=False,
                 validation_accuracy=.94, validation_ce_loss=.20, physical_parameters=17_000_000),
        ])


def test_completed_reference_requires_exact_budget_and_no_test(tmp_path):
    reference = tmp_path / "reference"
    job = reference / "J1_dense_control"
    for name in ("global_history.csv", "resolved_config.yaml", "selected_checkpoint.pt", "deployment.pt"):
        path = job / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    (reference / "shared_random_seed42.pt").write_bytes(b"initializer")
    state = {
        "status": "completed",
        "reference_protocol": "one_shot_dense_reference_v1",
        "reference_training_epochs_actually_executed": 150,
        "global_epochs_completed": 150,
        "test_evaluated": False,
        "selected_epoch": 100,
        "validation": {"accuracy": .94},
        "common_init_hash": "common",
        "initializer_file_hash": "file",
    }
    write_json(job / "pilot_state.json", state)
    assert launcher._reference_result(reference)["selected_epoch"] == 100

    state["reference_training_epochs_actually_executed"] = 149
    write_json(job / "pilot_state.json", state)
    with pytest.raises(RuntimeError, match="exactly 150"):
        launcher._reference_result(reference)


def test_checkpoint_cleanup_is_narrow_and_keeps_root_artifacts(tmp_path):
    search, recovery = tmp_path / "search", tmp_path / "recovery"
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
    result = launcher._prune_pair_checkpoints(search, recovery)
    assert result["files"] == 2
    assert not list(tmp_path.rglob("checkpoints/*.pt"))
    assert all(path.read_bytes() == b"keep" for path in root_artifacts)


def test_main_runs_reference_four_pairs_validation_selection_and_one_frozen_test(
        tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    output_root = tmp_path / "runs"
    state_path = tmp_path / "study.json"
    validation_by_search = {30: .941, 45: .943, 60: .942, 75: .940}
    commands = []

    monkeypatch.setattr(
        launcher,
        "validate_study_configs",
        lambda profile="fixed": [
            {"split": f"{a}/{b}", "profile": profile}
            for a, b in launcher.SPLITS
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_environment_preflight",
        lambda minimum: {"gpu": "synthetic", "free_gib": 999, "min_free_gib": minimum},
    )

    def touch(path, data=b"artifact"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def fake_run(command, capture=False):
        del capture
        command = list(map(str, command))
        commands.append(command)
        if "--prepare-reference" in command:
            job = reference / "J1_dense_control"
            for path in (
                reference / "shared_random_seed42.pt",
                job / "global_history.csv",
                job / "resolved_config.yaml",
                job / "selected_checkpoint.pt",
                job / "deployment.pt",
            ):
                touch(path)
            write_json(job / "pilot_state.json", {
                "status": "completed",
                "reference_protocol": "one_shot_dense_reference_v1",
                "reference_training_epochs_actually_executed": 150,
                "global_epochs_completed": 150,
                "test_evaluated": False,
                "selected_epoch": 149,
                "validation": {"accuracy": .945},
                "common_init_hash": "common",
                "initializer_file_hash": "initializer",
            })
            return
        if "--search-only" in command:
            run_dir = Path(command[command.index("--output") + 1])
            config = command[command.index("--config-name") + 1]
            search_epochs = int(config.split("epoch_split_")[1].split("_")[0])
            recovery_epochs = 150 - search_epochs
            for path in (
                run_dir / "selected_checkpoint.pt",
                run_dir / "export_only/deployment.pt",
                run_dir / "resolved_config.yaml",
            ):
                touch(path)
            write_json(run_dir / "one_shot_state.json", {
                "status": "search_only_completed", "search_only": True, "branches": {},
                "protocol": launcher.SEARCH_PROTOCOL, "per_branch_total_allocated": 150,
                "shared_search_ledger": {
                    "global_training_epoch": search_epochs,
                    "search_epochs_consumed": search_epochs,
                },
            })
            write_json(run_dir / "selection.json", {
                "reference_epoch": search_epochs,
                "selected_epoch": max(1, search_epochs - 1),
                "policy": "best_feasible_compact",
            })
            write_json(run_dir / "export_only/diagnostics.json", {
                "status": "measured", "bn_calibration_batches": 0,
                "physical_cost": {"physical_total_parameters": 16_000_000 + search_epochs},
                "physical_validation": {"accuracy": validation_by_search[search_epochs] - .01},
                "split": f"{search_epochs}/{recovery_epochs}",
            })
            return
        if "--reuse-search" in command:
            run_dir = Path(command[command.index("--output") + 1])
            search_dir = Path(command[command.index("--reuse-search") + 1]).resolve()
            config = command[command.index("--config-name") + 1]
            search_epochs = int(config.split("epoch_split_")[1].split("_")[0])
            recovery_epochs = 150 - search_epochs
            for path in (
                run_dir / "selected_checkpoint.pt", run_dir / "selection.json",
                run_dir / "export_only/deployment.pt", run_dir / "resolved_config.yaml",
                run_dir / launcher.RECOVERY_BRANCH / "deployment.pt",
            ):
                touch(path)
            branch = {
                "status": "completed", "optimizer_state_initialization": "fresh",
                "scheduler_state_initialization": "fresh_final_stage_cosine",
                "final_training_epochs_executed": recovery_epochs,
                "ledger": {"global_training_epoch": 150,
                           "search_epochs_consumed": search_epochs},
                "bn_calibration": {"requested_batches": 200, "batches": 200,
                                   "trainable_weights_unchanged": True},
                "validation": {
                    "accuracy": validation_by_search[search_epochs],
                    "ce_loss": .2 + search_epochs / 10_000,
                    "correct_count": int(validation_by_search[search_epochs] * 5000),
                    "example_count": 5000,
                },
                "quality_feasible": True,
                "final_cost": {"physical_total_parameters": 16_000_000 + search_epochs},
                "selected_final_epoch": recovery_epochs,
            }
            write_json(run_dir / launcher.RECOVERY_BRANCH / "branch_state.json", branch)
            write_json(run_dir / "one_shot_state.json", {
                "status": "completed", "protocol": launcher.RECOVERY_PROTOCOL,
                "per_branch_total_allocated": 150,
                "reused_search": {"source": str(search_dir)},
                "branches": {launcher.RECOVERY_BRANCH: branch},
                "test_evaluated": False,
            })
            return
        if str(launcher.EVALUATOR) in command:
            run_dir = Path(command[command.index("--run-dir") + 1]).resolve()
            test_dir = Path(command[command.index("--output") + 1])
            write_json(test_dir / "test_summary.json", {
                "status": "completed", "test_evaluated": True,
                "source_run": str(run_dir), "training_performed": False,
                "bn_recalibration": False, "test_based_selection": False,
                "comparison_scope": "exploratory", "runs": [{"test": {"accuracy": .938}}],
            })
            return
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(launcher, "_run", fake_run)
    result = launcher.main([
        "--reference-output", str(reference),
        "--output-root", str(output_root),
        "--state", str(state_path),
        "--min-free-gib", "1",
    ])

    assert result["status"] == "completed"
    assert result["selection"]["data"] == "validation_only"
    assert result["selection"]["winner"]["split"] == "45/105"
    assert result["selection"]["official_test_accessed_before_selection"] is False
    assert result["test"]["status"] == "completed"
    assert len([command for command in commands if "--search-only" in command]) == 4
    assert len([command for command in commands if "--reuse-search" in command]) == 4
    assert len([command for command in commands if str(launcher.EVALUATOR) in command]) == 1
