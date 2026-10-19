import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_resnet50_ls0_quarterpp_grid",
    ROOT / "scripts/launch_resnet50_cifar10_ls0_curve7.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture(scope="module")
def study():
    return launcher.load_study(launcher.QUALITY_GRID_CONFIG)


@pytest.fixture(scope="module")
def rows(study):
    return launcher.validate_curve_configs(study)


def test_eight_contiguous_quarter_point_intervals_use_fractional_accuracy_units(study, rows):
    assert study["protocol"] == launcher.QUALITY_GRID_PROTOCOL
    assert study["protocol"] in launcher.KEEP001_PROTOCOLS
    assert study["protocol"] in launcher.BN_SELECTION_PROTOCOLS
    assert study["protocol"] not in launcher.NIGHT_PROTOCOLS
    assert [row["id"] for row in rows] == [
        "val_9389_9414", "val_9364_9389", "val_9339_9364", "val_9314_9339",
        "val_9289_9314", "val_9264_9289", "val_9239_9264", "val_9214_9239",
    ]
    assert [row["hard_drop"] for row in rows] == pytest.approx([
        0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02,
    ])
    assert rows[0]["soft_drop"] == 0
    assert [row["hard_drop"] - row["soft_drop"] for row in rows] == pytest.approx(
        [0.0025] * 8
    )
    assert [row["required_final_validation_accuracy"] for row in rows] == pytest.approx([
        0.9389, 0.9364, 0.9339, 0.9314, 0.9289, 0.9264, 0.9239, 0.9214,
    ])
    assert [row["soft_final_validation_accuracy"] for row in rows] == pytest.approx([
        0.9414, 0.9389, 0.9364, 0.9339, 0.9314, 0.9289, 0.9264, 0.9239,
    ])
    assert {row["reference_final_validation_accuracy"] for row in rows} == {0.9414}
    assert "requested_min_parameters" not in study
    assert "time_budget" not in study
    assert all("target_parameters" not in row for row in rows)
    assert all(row["target_architecturally_reachable"] is None for row in rows)
    assert len(set(launcher.TEST_DIRECTORIES.values())) == len(launcher.TEST_DIRECTORIES)


@pytest.mark.parametrize("drift", [
    "percentage_units", "negative_soft", "zero_hard", "narrow_band", "wide_band",
    "swapped_profiles", "missing_profile", "duplicate_profile", "fake_target",
    "wrong_reference_accuracy", "wrong_reference_hash", "missing_reference_hash",
    "keep_008", "window_60", "bn_100", "night_budget",
])
def test_manifest_rejects_changes_that_break_the_quality_grid_contract(study, tmp_path, drift):
    changed = copy.deepcopy(study)
    if drift == "percentage_units":
        for profile in changed["profiles"]:
            profile["soft_drop"] *= 100
            profile["hard_drop"] *= 100
    elif drift == "negative_soft":
        changed["profiles"][0]["soft_drop"] = -0.0001
    elif drift == "zero_hard":
        changed["profiles"][0]["hard_drop"] = 0
    elif drift == "narrow_band":
        changed["profiles"][0]["soft_drop"] = 0.0001
    elif drift == "wide_band":
        changed["profiles"][1]["soft_drop"] = 0
    elif drift == "swapped_profiles":
        changed["profiles"][0], changed["profiles"][1] = (
            changed["profiles"][1], changed["profiles"][0]
        )
    elif drift == "missing_profile":
        changed["profiles"].pop()
    elif drift == "duplicate_profile":
        changed["profiles"][-1] = copy.deepcopy(changed["profiles"][0])
    elif drift == "fake_target":
        changed["profiles"][0]["target_parameters"] = 5_000_000
    elif drift == "wrong_reference_accuracy":
        changed["reference_final_validation_accuracy"] = 0.94
    elif drift == "wrong_reference_hash":
        changed["expected_reference_history_sha256"] = "0" * 64
    elif drift == "missing_reference_hash":
        changed.pop("expected_reference_history_sha256")
    elif drift == "keep_008":
        changed["min_keep_ratio"] = 0.08
    elif drift == "window_60":
        changed["search_selection_last_epochs"] = 60
    elif drift == "bn_100":
        changed["search_selection_bn_calibration_batches"] = 100
    else:
        changed["time_budget"] = {"hours": 9, "minutes_per_epoch": 1, "reserve_minutes": 90}
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError):
        launcher.load_study(path)


@pytest.mark.parametrize("phase", ["search", "recovery"])
def test_all_profiles_compose_to_the_same_controlled_recipe(study, rows, phase):
    from hydra.utils import instantiate
    from net_complexity.training import one_shot_pruning_config as schema

    assert len(rows) == 8
    for profile, row in zip(study["profiles"], rows):
        cfg = schema.compose_config(study[f"{phase}_config"], launcher.threshold_overrides(profile))
        launcher.validate_recipe(cfg, study, profile, phase)
        assert schema.validate_config(cfg) == 150
        assert (cfg.one_shot.search_epochs, cfg.one_shot.final_epochs) == (60, 90)
        assert cfg.one_shot.search_selection_last_epochs == 30
        assert cfg.one_shot.search_selection_bn_calibration_batches == 200
        assert cfg.accuracy_guided.eligibility.min_keep_ratio == 0.001
        assert instantiate(cfg.model.criterion).label_smoothing == 0
        assert cfg.training_arguments.adaptive_lambda.enabled is True
        assert cfg.training_arguments.adaptive_lambda.log_step == "auto"
        assert cfg.training_arguments.adaptive_lambda.soft_drop == profile["soft_drop"]
        assert cfg.training_arguments.adaptive_lambda.hard_drop == profile["hard_drop"]
        assert cfg.accuracy_guided.guard.train_bn_calibration_batches == (
            0 if phase == "search" else 200
        )
        assert row["resolved_log_step"] == pytest.approx(0.9210340371976183)
        assert (row["search_selection_epoch_start"], row["search_selection_epoch_end"]) == (31, 60)
        assert row["theoretical_parameter_floor"] < 3_000_000


def test_preflight_reports_full_eight_model_cost_without_fake_size_or_night_budget(rows, monkeypatch):
    monkeypatch.setattr(launcher, "validate_curve_configs", lambda study: rows)
    preflight = launcher.main([
        "--config", str(launcher.QUALITY_GRID_CONFIG), "--validate-only",
    ])
    assert preflight["requested_min_parameters"] is None
    assert preflight["requested_min_architecturally_reachable"] is None
    assert preflight["total_training_epochs_per_model"] == 150
    assert preflight["total_training_epochs_for_study"] == 1200
    assert "time_budget" not in preflight
    bn = preflight["search_selection_bn_calibration"]
    assert bn["candidates_per_model"] == 30
    assert bn["planned_train_forward_batches_per_model"] == 6000
    assert bn["planned_train_forward_batches_for_study"] == 48000
    assert bn["optimizer_updates"] == 0
    assert bn["lambda_feedback_mask_changed"] is False


@pytest.mark.parametrize("drift", [None, "history_hash", "final_accuracy", "phase_inputs", "not_ready"])
def test_reference_is_locked_to_the_original_history_and_final_validation_accuracy(
    study, monkeypatch, tmp_path, drift,
):
    from net_complexity.training import one_shot_pruning_config as schema
    from net_complexity.training import pruning_audit

    history = tmp_path / "dense_history.csv"
    inputs = {
        "status": "ready", "paths": {"history_path": str(history)},
        "sha256": {"history_path": study["expected_reference_history_sha256"],
                   "initializer_path": "shared_initializer"},
        "initializer_model_state_hash": "shared_state",
    }
    if drift == "history_hash":
        inputs["sha256"]["history_path"] = "0" * 64
    elif drift == "not_ready":
        inputs["status"] = "missing"
    calls = []

    def validate_inputs(config):
        calls.append(config)
        result = copy.deepcopy(inputs)
        if drift == "phase_inputs" and len(calls) == 2:
            result["sha256"]["initializer_path"] = "different_initializer"
        return result

    def load_reference(path, epochs):
        assert path == str(history) and epochs == 150
        return {150: 0.94 if drift == "final_accuracy" else 0.9414}

    monkeypatch.setattr(schema, "validate_inputs", validate_inputs)
    monkeypatch.setattr(pruning_audit, "load_adaptive_reference", load_reference)
    if drift is not None:
        with pytest.raises(ValueError):
            launcher.validate_reference(study, tmp_path)
    else:
        result = launcher.validate_reference(study, tmp_path)
        assert len(calls) == 2
        assert result["sha256"] == inputs["sha256"]
        assert result["initializer_model_state_hash"] == "shared_state"


def test_results_csv_exposes_required_quality_even_when_profile_fails_quality(rows, tmp_path):
    row = rows[0]
    result = {row["id"]: {"recovery": {
        "physical_parameters": 4_000_000, "validation_accuracy": 0.93,
        "validation_ce_loss": 0.4, "quality_feasible": False, "selected_final_epoch": 70,
    }}}
    path = tmp_path / "results.csv"
    launcher.write_table(path, rows, result, {}, {row["id"]: {"accuracy": 0.925}})
    with path.open() as stream:
        recorded = list(csv.DictReader(stream))
    assert len(recorded) == 8
    first = recorded[0]
    assert first["target_parameters"] == ""
    assert float(first["reference_final_validation_accuracy"]) == pytest.approx(0.9414)
    assert float(first["required_final_validation_accuracy"]) == pytest.approx(0.9389)
    assert float(first["soft_final_validation_accuracy"]) == pytest.approx(0.9414)
    assert first["quality_feasible"] == "False"
    assert first["test_accuracy"] == "0.925"


def test_all_eight_validation_rows_are_locked_before_testing_including_infeasible_runs(
    study, rows, monkeypatch, tmp_path,
):
    output = tmp_path / "run"
    monkeypatch.setattr(launcher, "validate_curve_configs", lambda study: rows)
    monkeypatch.setattr(launcher, "validate_reference", lambda *args: {"sha256": "locked"})
    monkeypatch.setattr(launcher.epoch_split, "_environment_preflight", lambda free_space: {})
    trained, evaluated = [], []

    def run_pair(current_study, row, reference, root, args):
        assert current_study == study
        trained.append(row["id"])
        return {
            "search": {"run_dir": str(root / f"search_{row['id']}")},
            "recovery": {
                "run_dir": str(root / f"recovery_{row['id']}"),
                "physical_parameters": 4_000_000 + len(trained),
                "validation_accuracy": 0.93, "validation_ce_loss": 0.4,
                "quality_feasible": False, "selected_final_epoch": 80,
            },
        }

    def evaluate(command):
        assert len(trained) == 8
        state = json.loads((output / "study_state.json").read_text())
        assert state["validation_locked_before_test"] is True
        with (output / "validation_results.csv").open() as stream:
            locked = list(csv.DictReader(stream))
        assert len(locked) == 8 and all(row["test_accuracy"] == "" for row in locked)
        destination = Path(command[command.index("--output") + 1])
        assert destination.name == launcher.TEST_DIRECTORIES[study["protocol"]]
        destination.mkdir(parents=True)
        evaluated.append(destination.parent.name.removeprefix("recovery_"))

    def test_result(test_dir, recovery_dir):
        identifier = Path(recovery_dir).name.removeprefix("recovery_")
        index = trained.index(identifier) + 1
        return {"runs": [{
            "physical_parameters": 4_000_000 + index,
            "model_state_unchanged": True, "bn_counters_unchanged": True,
            "test": {"accuracy": 0.925, "ce_loss": 0.4,
                     "correct_count": 9250, "example_count": 10000},
        }]}

    monkeypatch.setattr(launcher, "run_pair", run_pair)
    monkeypatch.setattr(launcher.epoch_split, "_run", evaluate)
    monkeypatch.setattr(launcher.epoch_split, "_test_result", test_result)
    state = launcher.main([
        "--config", str(launcher.QUALITY_GRID_CONFIG), "--output", str(output),
        "--reference-output", str(tmp_path / "reference"),
    ])
    assert state["status"] == "completed"
    assert evaluated == trained == [row["id"] for row in rows]
    with (output / "final_results.csv").open() as stream:
        recorded = list(csv.DictReader(stream))
    assert len(recorded) == 8
    assert all(row["quality_feasible"] == "False" for row in recorded)
    assert all(row["test_accuracy"] == "0.925" for row in recorded)
