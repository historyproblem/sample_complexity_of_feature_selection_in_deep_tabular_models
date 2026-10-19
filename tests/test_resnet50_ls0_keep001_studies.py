import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_resnet50_ls0_keep001_studies",
    ROOT / "scripts/launch_resnet50_cifar10_ls0_curve7.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture(scope="module", params=[
    (launcher.KEEP001_TARGET4P5M_CONFIG, 4_500_000, (0.015, 0.018)),
    (launcher.KEEP001_TARGET14M_CONFIG, 14_000_000, (0.00025, 0.0005)),
], ids=["target4p5m", "target14m"])
def followup(request):
    manifest, target, thresholds = request.param
    study = launcher.load_study(manifest)
    return manifest, study, launcher.validate_curve_configs(study), target, thresholds


def test_one_independent_keep001_pair_retains_post_bn_validation_policy(followup):
    _, study, rows, target, thresholds = followup
    assert study["protocol"] in launcher.BN_SELECTION_PROTOCOLS
    assert study["protocol"] not in launcher.NIGHT_PROTOCOLS
    assert len(rows) == 1
    row = rows[0]
    assert row["target_parameters"] == study["requested_min_parameters"] == target
    assert (row["soft_drop"], row["hard_drop"]) == thresholds
    assert row["min_keep_ratio"] == 0.001
    assert row["theoretical_parameter_floor"] < 3_000_000
    assert row["target_architecturally_reachable"] is True
    assert row["search_selection_bn_calibration_batches"] == 200
    assert (row["search_selection_epoch_start"], row["search_selection_epoch_end"]) == (31, 60)
    assert row["split"] == "60/90" and row["label_smoothing"] == 0
    assert row["resolved_log_step"] == pytest.approx(0.9210340371976183)
    assert "log_step" not in study["profiles"][0]
    assert "time_budget" not in study
    assert study["output"] == f"outputs/runs/{study['protocol']}"


@pytest.mark.parametrize("phase", ["search", "recovery"])
@pytest.mark.parametrize("drift", ["keep_floor", "selection_bn", "selection_window"])
def test_fresh_and_saved_followup_recipe_rejects_floor_or_selection_drift(followup, phase, drift):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    _, study, _, _, _ = followup
    row = study["profiles"][0]
    config = schema.compose_config(study[f"{phase}_config"], launcher.threshold_overrides(row))
    launcher.validate_recipe(config, study, row, phase)
    assert config.accuracy_guided.guard.train_bn_calibration_batches == (
        0 if phase == "search" else 200
    )
    field, changed = {
        "keep_floor": ("accuracy_guided.eligibility.min_keep_ratio", 0.08),
        "selection_bn": ("one_shot.search_selection_bn_calibration_batches", 0),
        "selection_window": ("one_shot.search_selection_last_epochs", 60),
    }[drift]
    OmegaConf.update(config, field, changed)
    with pytest.raises(ValueError):
        launcher.validate_recipe(config, study, row, phase)


@pytest.mark.parametrize("drift", ["floor_008", "floor_001", "missing_bn", "short_bn", "time_budget"])
def test_new_manifest_does_not_silently_reuse_old_floor_or_pre_bn_policy(followup, tmp_path, drift):
    _, study, _, _, _ = followup
    changed = copy.deepcopy(study)
    if drift.startswith("floor"):
        changed["min_keep_ratio"] = 0.08 if drift == "floor_008" else 0.01
    elif drift == "missing_bn":
        changed.pop("search_selection_bn_calibration_batches")
    elif drift == "short_bn":
        changed["search_selection_bn_calibration_batches"] = 100
    else:
        changed["time_budget"] = {"hours": 9, "minutes_per_epoch": 1, "reserve_minutes": 90}
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError):
        launcher.load_study(path)


@pytest.mark.parametrize("path", [
    launcher.DEFAULT_CONFIG, launcher.NARROW_CONFIG, launcher.MODERATE_CONFIG,
    launcher.NIGHT_CONFIG, launcher.BN_SELECTION_CONFIG,
])
def test_lower_floor_is_not_applied_to_published_008_studies(path, tmp_path):
    study = launcher.load_study(path)
    assert study["min_keep_ratio"] == 0.08
    study["min_keep_ratio"] = 0.001
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError, match="keep floor must stay 0.08"):
        launcher.load_study(changed)


def test_single_profile_preflight_reports_6000_calibration_forwards_without_night_budget(
    followup, monkeypatch,
):
    manifest, _, rows, _, _ = followup
    monkeypatch.setattr(launcher, "validate_curve_configs", lambda study: rows)
    preflight = launcher.main(["--config", str(manifest), "--validate-only"])
    assert "time_budget" not in preflight
    assert preflight["total_training_epochs_per_model"] == 150
    calibration = preflight["search_selection_bn_calibration"]
    assert calibration["planned_train_forward_batches_for_study"] == 6000
    assert calibration["candidates_per_model"] == 30
    assert calibration["optimizer_updates"] == 0
    assert calibration["lambda_feedback_mask_changed"] is False


@pytest.mark.parametrize("drift", ["predictor", "batches", "weights"])
def test_followup_resume_requires_physical_post_bn200_selection_evidence(
    followup, tmp_path, drift,
):
    _, study, _, _, _ = followup
    evidence = {
        "selection_predictor": "physical_after_train_bn_calibration",
        "selection_bn_calibration": {
            "requested_batches": 200, "batches": 200, "seed": 42,
            "trainable_weights_unchanged": True,
        },
        "selection_validation": {
            "accuracy": 0.9, "correct_count": 4500, "example_count": 5000,
        },
        "trace": {"selection_predictor": "physical_after_train_bn_calibration",
                  "bn_calibration_batches": 200},
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(evidence))
    launcher.validate_saved_selection_bn(tmp_path, study)
    if drift == "predictor":
        evidence["selection_predictor"] = "dependency_safe_all_physical_survivors_open"
    elif drift == "batches":
        evidence["selection_bn_calibration"]["batches"] = 100
    else:
        evidence["selection_bn_calibration"]["trainable_weights_unchanged"] = False
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        launcher.validate_saved_selection_bn(tmp_path, study)


def test_one_model_locks_validation_and_reports_actual_size_without_target_filter(
    followup, monkeypatch, tmp_path,
):
    manifest, study, rows, target, _ = followup
    row = rows[0]
    actual_parameters = target + 123_456
    output = tmp_path / "run"
    monkeypatch.setattr(launcher, "validate_curve_configs", lambda study: rows)
    monkeypatch.setattr(launcher, "validate_reference", lambda study, reference: {"sha256": "same"})
    monkeypatch.setattr(launcher.epoch_split, "_environment_preflight", lambda free_space: {})
    recovery_dir = output / f"recovery_{row['id']}"
    monkeypatch.setattr(launcher, "run_pair", lambda *args: {
        "search": {"run_dir": str(output / f"search_{row['id']}")},
        "recovery": {
            "run_dir": str(recovery_dir), "physical_parameters": actual_parameters,
            "validation_accuracy": 0.94, "validation_ce_loss": 0.4,
            "quality_feasible": False, "selected_final_epoch": 80,
        },
    })
    evaluated = []

    def evaluate(command):
        state = json.loads((output / "study_state.json").read_text())
        assert state["validation_locked_before_test"] is True
        with (output / "validation_results.csv").open() as stream:
            locked = list(csv.DictReader(stream))
        assert len(locked) == 1 and locked[0]["test_accuracy"] == ""
        assert locked[0]["physical_parameters"] == str(actual_parameters)
        test_dir = Path(command[command.index("--output") + 1])
        assert test_dir.name == launcher.TEST_DIRECTORIES[study["protocol"]]
        test_dir.mkdir(parents=True)
        evaluated.append(test_dir)

    monkeypatch.setattr(launcher.epoch_split, "_run", evaluate)
    monkeypatch.setattr(launcher.epoch_split, "_test_result", lambda *args: {"runs": [{
        "physical_parameters": actual_parameters,
        "model_state_unchanged": True, "bn_counters_unchanged": True,
        "test": {"accuracy": 0.935, "ce_loss": 0.3,
                 "correct_count": 9350, "example_count": 10000},
    }]})
    state = launcher.main([
        "--config", str(manifest), "--output", str(output),
        "--reference-output", str(tmp_path / "reference"),
    ])
    assert state["status"] == "completed" and len(evaluated) == 1
    with (output / "final_results.csv").open() as stream:
        recorded = list(csv.DictReader(stream))
    assert len(recorded) == 1
    result = recorded[0]
    assert result["physical_parameters"] == str(actual_parameters)
    assert result["target_parameters"] == str(target)
    assert result["min_keep_ratio"] == "0.001"
    assert result["search_selection_bn_calibration_batches"] == "200"
    assert result["quality_feasible"] == "False"
    assert result["test_accuracy"] == "0.935"
