import copy
import csv
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_resnet50_ls0_extra3",
    ROOT / "scripts/launch_resnet50_cifar10_ls0_curve7.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
EXPECTED = [
    ("val_9240_9265", "0.9240", "0.9265", "0.0149", "0.0174"),
    ("val_9290_9315", "0.9290", "0.9315", "0.0099", "0.0124"),
    ("val_9350_9375", "0.9350", "0.9375", "0.0039", "0.0064"),
]


@pytest.fixture(scope="module")
def study():
    return launcher.load_study(launcher.EXTRA3_CONFIG)


@pytest.fixture(scope="module")
def rows(study):
    return launcher.validate_curve_configs(study)


def test_exact_requested_validation_bands_and_order(study, rows):
    assert study["protocol"] == launcher.EXTRA3_PROTOCOL
    assert study["protocol"] in launcher.PINNED_INITIALIZER_PROTOCOLS
    assert study["protocol"] in launcher.BN_SELECTION_PROTOCOLS
    assert study["protocol"] not in launcher.KEEP001_PROTOCOLS
    assert len(rows) == len(study["profiles"]) == 3
    reference = Decimal(str(study["reference_final_validation_accuracy"]))
    assert reference == Decimal("0.9414")
    for profile, row, (identifier, lower, upper, soft, hard) in zip(
        study["profiles"], rows, EXPECTED,
    ):
        assert profile["id"] == row["id"] == identifier
        assert Decimal(str(profile["soft_drop"])) == Decimal(soft)
        assert Decimal(str(profile["hard_drop"])) == Decimal(hard)
        assert reference - Decimal(hard) == Decimal(lower)
        assert reference - Decimal(soft) == Decimal(upper)
        assert Decimal(upper) - Decimal(lower) == Decimal("0.0025")
        assert Decimal(str(row["required_final_validation_accuracy"])) == Decimal(lower)
        assert Decimal(str(row["soft_final_validation_accuracy"])) == Decimal(upper)
        assert "target_parameters" not in row
        assert row["target_architecturally_reachable"] is None
    assert "requested_min_parameters" not in study


def test_dedicated_command_defaults_to_three_models_and_450_epochs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/launch_resnet50_cifar10_ls0_extra3.py"),
         "--validate-only"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    preflight = json.loads(result.stdout)
    assert preflight["protocol"] == launcher.EXTRA3_PROTOCOL
    assert [row["id"] for row in preflight["profiles"]] == [row[0] for row in EXPECTED]
    assert preflight["total_training_epochs_per_model"] == 150
    assert preflight["total_training_epochs_for_study"] == 450
    assert preflight["dense_reference_training"].startswith("none;")
    assert preflight["search_selection_bn_calibration"]["planned_train_forward_batches_for_study"] == 18000


def test_original_first_run_is_unchanged(study):
    original = launcher.load_study(launcher.FULL150_CONFIG)
    assert original["profiles"] == [{
        "id": "strict_ls0", "soft_drop": 0.00025, "hard_drop": 0.0005,
    }]
    assert original["protocol"] == launcher.FULL150_PROTOCOL
    assert original["output"] != study["output"]
    for key in ("search_config", "recovery_config", "reference_output", "min_keep_ratio",
                "label_smoothing", "search_selection_last_epochs",
                "search_selection_bn_calibration_batches", "reference_final_validation_accuracy",
                "expected_reference_history_sha256", "expected_initializer_sha256"):
        assert original[key] == study[key]
    assert launcher.TEST_DIRECTORIES[original["protocol"]] != launcher.TEST_DIRECTORIES[study["protocol"]]


@pytest.mark.parametrize("phase", ["search", "recovery"])
def test_new_recipes_only_change_thresholds_from_the_first_run(study, rows, phase):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    original = launcher.load_study(launcher.FULL150_CONFIG)
    baseline = schema.compose_config(
        original[f"{phase}_config"], launcher.threshold_overrides(original["profiles"][0]),
    )
    for profile, row in zip(study["profiles"], rows):
        config = schema.compose_config(study[f"{phase}_config"], launcher.threshold_overrides(profile))
        launcher.validate_recipe(config, study, profile, phase)
        assert schema.validate_config(config) == 150
        assert (config.one_shot.search_epochs, config.one_shot.final_epochs) == (60, 90)
        assert config.accuracy_guided.eligibility.min_keep_ratio == 0.08
        assert config.training_arguments.adaptive_lambda.enabled is True
        assert config.training_arguments.adaptive_lambda.log_step == "auto"
        assert config.one_shot.search_selection_last_epochs == 30
        assert config.one_shot.search_selection_bn_calibration_batches == 200
        assert row["resolved_log_step"] == pytest.approx(0.9210340371976183)
        if phase == "recovery":
            plan = schema.resolved_branch_plan(config)
            assert len(plan) == 1
            assert plan[0]["optimizer_state"] == "fresh"
            assert plan[0]["scheduler_state"] == schema.FRESH_SCHEDULER
        comparable = OmegaConf.to_container(config, resolve=True)
        expected = OmegaConf.to_container(baseline, resolve=True)
        for key in ("soft_drop", "hard_drop"):
            assert comparable["training_arguments"]["adaptive_lambda"].pop(key) == profile[key]
            expected["training_arguments"]["adaptive_lambda"].pop(key)
        assert comparable == expected


@pytest.mark.parametrize("drift", ["order", "band", "width", "extra_strict", "initializer", "history"])
def test_manifest_rejects_changes_to_requested_runs_or_reference(study, tmp_path, drift):
    changed = copy.deepcopy(study)
    if drift == "order":
        changed["profiles"].reverse()
    elif drift == "band":
        changed["profiles"][0]["soft_drop"] += 0.001
        changed["profiles"][0]["hard_drop"] += 0.001
    elif drift == "width":
        changed["profiles"][0]["soft_drop"] += 0.0001
    elif drift == "extra_strict":
        changed["profiles"].append({"id": "strict_ls0", "soft_drop": 0.00025, "hard_drop": 0.0005})
    elif drift == "initializer":
        changed["expected_initializer_sha256"] = "0" * 64
    else:
        changed["expected_reference_history_sha256"] = "0" * 64
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError):
        launcher.load_study(path)


@pytest.mark.parametrize("input_name", ["initializer_path", "history_path"])
def test_actual_reference_hash_mismatch_is_rejected(study, monkeypatch, tmp_path, input_name):
    from net_complexity.training import one_shot_pruning_config as schema
    from net_complexity.training import pruning_audit

    inputs = {
        "status": "ready", "paths": {"history_path": str(tmp_path / "history.csv")},
        "sha256": {"history_path": study["expected_reference_history_sha256"],
                   "initializer_path": study["expected_initializer_sha256"]},
        "initializer_model_state_hash": "shared_state",
    }
    inputs["sha256"][input_name] = "0" * 64
    monkeypatch.setattr(schema, "validate_inputs", lambda config: copy.deepcopy(inputs))
    monkeypatch.setattr(pruning_audit, "load_adaptive_reference", lambda *args: {150: 0.9414})
    with pytest.raises(ValueError):
        launcher.validate_reference(study, tmp_path)


def test_exactly_three_pairs_lock_validation_before_three_frozen_tests(
    study, rows, monkeypatch, tmp_path,
):
    output = tmp_path / "run"
    identifiers = [row[0] for row in EXPECTED]
    trained, evaluated = [], []
    monkeypatch.setattr(launcher, "validate_curve_configs", lambda current: rows)
    monkeypatch.setattr(launcher, "validate_reference", lambda *args: {"sha256": "locked"})
    monkeypatch.setattr(launcher.epoch_split, "_environment_preflight", lambda free_space: {})

    def run_pair(current_study, row, reference, root, args):
        assert current_study == study
        assert args.resume_completed is False
        trained.append(row["id"])
        return {
            "search": {"run_dir": str(root / f"search_{row['id']}")},
            "recovery": {
                "run_dir": str(root / f"recovery_{row['id']}"),
                "physical_parameters": 4_000_000 + len(trained),
                "validation_accuracy": row["required_final_validation_accuracy"] - 0.001,
                "validation_ce_loss": 0.4, "quality_feasible": False, "selected_final_epoch": 80,
            },
        }

    def evaluate(command):
        assert trained == identifiers
        assert Path(command[1]).name == "evaluate_one_shot_pruning_test.py"
        state = json.loads((output / "study_state.json").read_text())
        assert state["validation_locked_before_test"] is True
        with (output / "validation_results.csv").open() as stream:
            locked = list(csv.DictReader(stream))
        assert [row["id"] for row in locked] == identifiers
        assert all(row["test_accuracy"] == "" for row in locked)
        destination = Path(command[command.index("--output") + 1])
        assert destination.name == launcher.TEST_DIRECTORIES[study["protocol"]]
        destination.mkdir(parents=True)
        evaluated.append(destination.parent.name.removeprefix("recovery_"))

    def test_result(test_dir, recovery_dir):
        identifier = Path(recovery_dir).name.removeprefix("recovery_")
        return {"runs": [{
            "physical_parameters": 4_000_000 + identifiers.index(identifier) + 1,
            "model_state_unchanged": True, "bn_counters_unchanged": True,
            "test": {"accuracy": 0.925, "ce_loss": 0.4,
                     "correct_count": 9250, "example_count": 10000},
        }]}

    monkeypatch.setattr(launcher, "run_pair", run_pair)
    monkeypatch.setattr(launcher.epoch_split, "_run", evaluate)
    monkeypatch.setattr(launcher.epoch_split, "_test_result", test_result)
    state = launcher.main([
        "--config", str(launcher.EXTRA3_CONFIG), "--output", str(output),
        "--reference-output", str(tmp_path / "reference"),
    ])
    assert state["status"] == "completed"
    assert evaluated == trained == identifiers
    assert set(state["runs"]) == set(state["tests"]) == set(identifiers)
    assert all(result["example_count"] == 10000 for result in state["tests"].values())
    assert state["test_comparison_scope"] == "exploratory"
    with (output / "final_results.csv").open() as stream:
        recorded = list(csv.DictReader(stream))
    assert len(recorded) == 3
    assert all(row["test_accuracy"] == "0.925" for row in recorded)
