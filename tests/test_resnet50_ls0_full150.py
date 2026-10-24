import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_resnet50_ls0_full150",
    ROOT / "scripts/launch_resnet50_cifar10_ls0_curve7.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture(scope="module")
def study():
    return launcher.load_study(launcher.FULL150_CONFIG)


@pytest.fixture(scope="module")
def rows(study):
    return launcher.validate_curve_configs(study)


def test_dedicated_command_defaults_to_one_complete_model():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/launch_resnet50_cifar10_ls0_full150.py"),
         "--validate-only"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    preflight = json.loads(result.stdout)
    assert preflight["protocol"] == launcher.FULL150_PROTOCOL
    assert len(preflight["profiles"]) == 1
    assert preflight["total_training_epochs_for_study"] == 150


def test_one_complete_model_has_no_size_target_or_additional_training(study, rows, monkeypatch):
    assert study["protocol"] == launcher.FULL150_PROTOCOL
    assert study["protocol"] in launcher.BN_SELECTION_PROTOCOLS
    assert study["protocol"] not in launcher.KEEP001_PROTOCOLS
    assert study["profiles"] == [{
        "id": "strict_ls0", "soft_drop": 0.00025, "hard_drop": 0.0005,
    }]
    assert len(rows) == 1
    row = rows[0]
    assert row["target_architecturally_reachable"] is None
    assert row["theoretical_parameter_floor"] == 3_683_893
    assert row["resolved_log_step"] == pytest.approx(0.9210340371976183)
    assert row["required_final_validation_accuracy"] == pytest.approx(0.9409)
    assert row["soft_final_validation_accuracy"] == pytest.approx(0.94115)
    monkeypatch.setattr(launcher, "validate_curve_configs", lambda current: rows)
    preflight = launcher.main(["--config", str(launcher.FULL150_CONFIG), "--validate-only"])
    assert preflight["requested_min_parameters"] is None
    assert preflight["total_training_epochs_per_model"] == 150
    assert preflight["total_training_epochs_for_study"] == 150
    assert preflight["dense_reference_training"].startswith("none;")
    assert "time_budget" not in preflight
    bn = preflight["search_selection_bn_calibration"]
    assert (bn["candidate_epoch_start"], bn["candidate_epoch_end"]) == (31, 60)
    assert bn["planned_train_forward_batches_for_study"] == 6000
    assert bn["optimizer_updates"] == 0


@pytest.mark.parametrize("phase", ["search", "recovery"])
def test_full_run_preserves_repro20_search_recipe_and_one_fresh_recovery(study, phase):
    from hydra.utils import instantiate
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    diagnostic = yaml.safe_load((
        ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_repro20.yaml"
    ).read_text())
    for key in ("search_config", "reference_output", "min_keep_ratio",
                "expected_reference_history_sha256", "expected_initializer_sha256"):
        assert study[key] == diagnostic[key]
    profile = study["profiles"][0]
    assert profile["soft_drop"] == diagnostic["soft_drop"]
    assert profile["hard_drop"] == diagnostic["hard_drop"]
    config = schema.compose_config(study[f"{phase}_config"], launcher.threshold_overrides(profile))
    launcher.validate_recipe(config, study, profile, phase)
    assert schema.validate_config(config) == 150
    assert (config.one_shot.search_epochs, config.one_shot.final_epochs) == (60, 90)
    assert config.one_shot.search_selection_last_epochs == 30
    assert config.one_shot.search_selection_bn_calibration_batches == 200
    assert config.accuracy_guided.eligibility.min_keep_ratio == 0.08
    assert instantiate(config.model.criterion).label_smoothing == 0
    assert config.training_arguments.adaptive_lambda.enabled is True
    assert config.training_arguments.adaptive_lambda.log_step == "auto"
    if phase == "search":
        prefix_config = schema.compose_config(
            diagnostic["search_config"], launcher.threshold_overrides(diagnostic),
        )
        assert OmegaConf.to_container(config, resolve=True) == OmegaConf.to_container(
            prefix_config, resolve=True,
        )
    else:
        plan = schema.resolved_branch_plan(config)
        assert len(plan) == 1
        assert plan[0]["training_seed"] == 42
        assert plan[0]["model_state"] == "selected_surviving_state"
        assert plan[0]["optimizer_state"] == "fresh"
        assert plan[0]["scheduler_state"] == schema.FRESH_SCHEDULER


@pytest.mark.parametrize("field,value", [
    ("label_smoothing", 0.05),
    ("min_keep_ratio", 0.001),
    ("search_selection_last_epochs", 60),
    ("search_selection_bn_calibration_batches", 100),
    ("expected_reference_history_sha256", "0" * 64),
    ("expected_initializer_sha256", "0" * 64),
    ("reference_final_validation_accuracy", 0.94),
    ("profiles", []),
])
def test_manifest_rejects_recipe_or_reference_drift(study, tmp_path, field, value):
    changed = copy.deepcopy(study)
    changed[field] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError):
        launcher.load_study(path)


@pytest.mark.parametrize("drift", [None, "history_path", "initializer_path", "final_accuracy"])
def test_actual_reference_inputs_are_pinned_before_launch(study, monkeypatch, tmp_path, drift):
    from net_complexity.training import one_shot_pruning_config as schema
    from net_complexity.training import pruning_audit

    inputs = {
        "status": "ready", "paths": {"history_path": str(tmp_path / "history.csv")},
        "sha256": {"history_path": study["expected_reference_history_sha256"],
                   "initializer_path": study["expected_initializer_sha256"]},
        "initializer_model_state_hash": "shared_state",
    }
    if drift in inputs["sha256"]:
        inputs["sha256"][drift] = "0" * 64
    checked = []

    def validate_inputs(config):
        checked.append(config)
        return copy.deepcopy(inputs)

    def load_reference(path, epochs):
        assert path == inputs["paths"]["history_path"] and epochs == 150
        return {150: 0.94 if drift == "final_accuracy" else 0.9414}

    monkeypatch.setattr(schema, "validate_inputs", validate_inputs)
    monkeypatch.setattr(pruning_audit, "load_adaptive_reference", load_reference)
    if drift is None:
        result = launcher.validate_reference(study, tmp_path)
        assert result["sha256"] == inputs["sha256"]
        assert len(checked) == 2
    else:
        with pytest.raises(ValueError):
            launcher.validate_reference(study, tmp_path)


def test_pair_launches_complete_search_then_one_recovery_without_prefix_resume(
    study, rows, monkeypatch, tmp_path,
):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    commands = []

    def launch(command):
        commands.append(command)
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir(parents=True)
        config_name = command[command.index("--config-name") + 1]
        config = schema.compose_config(config_name, launcher.threshold_overrides(rows[0]))
        OmegaConf.save(config, destination / "resolved_config.yaml")

    monkeypatch.setattr(launcher.epoch_split, "_run", launch)
    monkeypatch.setattr(launcher, "read_search_result", lambda *args: {"selected_epoch": 42})
    monkeypatch.setattr(launcher.epoch_split, "_recovery_result", lambda *args: {"status": "completed"})
    monkeypatch.setattr(launcher, "validate_saved_selection_bn", lambda *args: None)
    args = SimpleNamespace(resume_completed=False, keep_nested_checkpoints=True)
    launcher.run_pair(study, rows[0], tmp_path / "reference", tmp_path / "run", args)
    assert len(commands) == 2
    search, recovery = commands
    assert "--search-only" in search
    assert "--reuse-search" in recovery and "--skip-test" in recovery
    assert recovery[recovery.index("--reuse-search") + 1] == tmp_path / "run/search_strict_ls0"
    for command in commands:
        assert Path(command[1]).name == "launch_one_shot_pruning.py"
        assert "--stop-after-search-epoch" not in command
        assert "--resume-search" not in command
        assert "--search-diagnostics" not in command


@pytest.mark.parametrize("invalid_test", [None, "model_changed", "partial_test"])
def test_one_model_validation_is_locked_before_one_frozen_official_test(
    study, rows, monkeypatch, tmp_path, invalid_test,
):
    output = tmp_path / "run"
    monkeypatch.setattr(launcher, "validate_curve_configs", lambda current: rows)
    monkeypatch.setattr(launcher, "validate_reference", lambda *args: {"sha256": "locked"})
    monkeypatch.setattr(launcher.epoch_split, "_environment_preflight", lambda free_space: {})
    trained, evaluated = [], []

    def run_pair(current_study, row, reference, root, args):
        assert current_study == study
        assert args.resume_completed is False
        trained.append(row["id"])
        return {
            "search": {"run_dir": str(root / "search_strict_ls0")},
            "recovery": {
                "run_dir": str(root / "recovery_strict_ls0"),
                "physical_parameters": 14_848_414,
                "validation_accuracy": 0.939, "validation_ce_loss": 0.4,
                "quality_feasible": False, "selected_final_epoch": 80,
            },
        }

    def evaluate(command):
        assert trained == ["strict_ls0"]
        assert Path(command[1]).name == "evaluate_one_shot_pruning_test.py"
        state = json.loads((output / "study_state.json").read_text())
        assert state["validation_locked_before_test"] is True
        with (output / "validation_results.csv").open() as stream:
            locked = list(csv.DictReader(stream))
        assert len(locked) == 1 and locked[0]["test_accuracy"] == ""
        destination = Path(command[command.index("--output") + 1])
        assert destination.name == launcher.TEST_DIRECTORIES[study["protocol"]]
        destination.mkdir(parents=True)
        evaluated.append(destination.parent.name)

    def test_result(test_dir, recovery_dir):
        return {"runs": [{
            "physical_parameters": 14_848_414,
            "model_state_unchanged": invalid_test != "model_changed",
            "bn_counters_unchanged": True,
            "test": {"accuracy": 0.9372, "ce_loss": 0.4, "correct_count": 9372,
                     "example_count": 5000 if invalid_test == "partial_test" else 10000},
        }]}

    monkeypatch.setattr(launcher, "run_pair", run_pair)
    monkeypatch.setattr(launcher.epoch_split, "_run", evaluate)
    monkeypatch.setattr(launcher.epoch_split, "_test_result", test_result)
    state = launcher.main([
        "--config", str(launcher.FULL150_CONFIG), "--output", str(output),
        "--reference-output", str(tmp_path / "reference"),
    ])
    assert trained == ["strict_ls0"]
    assert evaluated == ["recovery_strict_ls0"]
    assert state["test_comparison_scope"] == "exploratory"
    if invalid_test is None:
        assert state["status"] == "completed"
        assert state["tests"]["strict_ls0"]["example_count"] == 10000
        with (output / "final_results.csv").open() as stream:
            recorded = list(csv.DictReader(stream))
        assert len(recorded) == 1
        assert recorded[0]["quality_feasible"] == "False"
        assert recorded[0]["test_accuracy"] == "0.9372"
    else:
        assert state["status"] == "completed_with_failures"
        assert "strict_ls0" in state["test_failures"]
        assert state["tests"] == {}
