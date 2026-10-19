import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_resnet50_ls0_bn_selection_study",
    ROOT / "scripts/launch_resnet50_cifar10_ls0_curve7.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture(scope="module")
def study():
    return launcher.load_study(launcher.BN_SELECTION_CONFIG)


@pytest.fixture(scope="module")
def rows(study):
    return launcher.validate_curve_configs(study)


def test_followup_keeps_previous_night_profiles_and_uses_separate_outputs(study, rows):
    previous = launcher.load_study(launcher.NIGHT_CONFIG)
    assert study["profiles"] == previous["profiles"]
    assert study["output"] != previous["output"]
    assert study["protocol"] == launcher.BN_SELECTION_PROTOCOL
    assert {row["search_selection_bn_calibration_batches"] for row in rows} == {200}
    assert {(row["search_selection_epoch_start"], row["search_selection_epoch_end"])
            for row in rows} == {(31, 60)}
    assert [row["resolved_log_step"] for row in rows] == pytest.approx([
        0.9210340371976183, 0.9210340371976183, 0.46,
    ])


@pytest.mark.parametrize("phase", ["search", "recovery"])
@pytest.mark.parametrize("drift", ["missing", "zero", "different"])
def test_new_phase_and_saved_recipe_reject_missing_or_changed_selection_bn(study, phase, drift):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import open_dict

    row = study["profiles"][0]
    cfg = schema.compose_config(study[f"{phase}_config"], launcher.threshold_overrides(row))
    launcher.validate_recipe(cfg, study, row, phase)
    assert cfg.accuracy_guided.guard.train_bn_calibration_batches == (
        0 if phase == "search" else 200
    )
    if drift == "missing":
        with open_dict(cfg.one_shot):
            del cfg.one_shot.search_selection_bn_calibration_batches
    else:
        cfg.one_shot.search_selection_bn_calibration_batches = 0 if drift == "zero" else 100
    with pytest.raises(ValueError, match="BN-calibration selection policy"):
        launcher.validate_recipe(cfg, study, row, phase)


@pytest.mark.parametrize("invalid", [0, 100, False, None, "200"])
def test_manifest_does_not_silently_enable_a_different_bn_budget(study, tmp_path, invalid):
    changed = copy.deepcopy(study)
    changed["search_selection_bn_calibration_batches"] = invalid
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="200 train-only batches"):
        launcher.load_study(path)


def selection_evidence():
    return {
        "selection_predictor": "physical_after_train_bn_calibration",
        "selection_bn_calibration": {
            "requested_batches": 200, "batches": 200, "seed": 42,
            "trainable_weights_unchanged": True,
        },
        "selection_validation": {
            "accuracy": 0.9, "ce_loss": 0.4, "correct_count": 4500,
            "example_count": 5000,
        },
        "trace": {"selection_predictor": "physical_after_train_bn_calibration",
                  "bn_calibration_batches": 200},
    }


@pytest.mark.parametrize("drift", ["predictor", "batches", "seed", "weights", "validation", "trace"])
def test_resume_requires_saved_post_bn_selection_evidence(study, tmp_path, drift):
    evidence = selection_evidence()
    (tmp_path / "selection.json").write_text(json.dumps(evidence))
    launcher.validate_saved_selection_bn(tmp_path, study)
    if drift == "predictor":
        evidence["selection_predictor"] = "dependency_safe_all_physical_survivors_open"
    elif drift == "batches":
        evidence["selection_bn_calibration"]["batches"] = 0
    elif drift == "seed":
        evidence["selection_bn_calibration"]["seed"] = 43
    elif drift == "weights":
        evidence["selection_bn_calibration"]["trainable_weights_unchanged"] = False
    elif drift == "validation":
        evidence["selection_validation"]["example_count"] = 128
    else:
        evidence["trace"]["bn_calibration_batches"] = 0
    (tmp_path / "selection.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        launcher.validate_saved_selection_bn(tmp_path, study)


def test_preflight_accounts_for_candidate_forward_work_without_promising_duration(rows, monkeypatch):
    monkeypatch.setattr(launcher, "validate_curve_configs", lambda study: rows)
    preflight = launcher.main([
        "--config", str(launcher.BN_SELECTION_CONFIG), "--validate-only",
    ])
    bn = preflight["search_selection_bn_calibration"]
    assert bn["candidates_per_model"] == 30
    assert bn["train_batches_per_candidate"] == 200
    assert bn["planned_train_forward_batches_per_model"] == 6000
    assert bn["planned_train_forward_batches_for_study"] == 18000
    assert bn["optimizer_updates"] == 0
    assert bn["lambda_feedback_mask_changed"] is False
    assert preflight["time_budget"]["planned_training_epochs"] == 450
    assert preflight["time_budget"]["duration_guaranteed"] is False
    assert preflight["time_budget"]["hard_wallclock_timeout"] is False


def test_csv_preserves_explicit_selection_bn_policy(rows, tmp_path):
    path = tmp_path / "results.csv"
    launcher.write_table(path, rows, {}, {}, {})
    with path.open() as stream:
        recorded = list(csv.DictReader(stream))
    assert len(recorded) == 3
    assert {row["search_selection_bn_calibration_batches"] for row in recorded} == {"200"}


@pytest.fixture
def completed_bn_search(tmp_path):
    selection = {
        **selection_evidence(), "reference_epoch": 60, "selected_epoch": 44,
        "policy": "best_feasible_compact",
    }
    state = {
        "status": "search_only_completed", "search_only": True, "branches": {},
        "protocol": launcher.epoch_split.SEARCH_PROTOCOL, "per_branch_total_allocated": 150,
        "shared_search_ledger": {"global_training_epoch": 60, "search_epochs_consumed": 60},
    }
    diagnostics = {
        "status": "measured", "bn_calibration_batches": 200,
        "selection_predictor": "physical_after_train_bn_calibration",
        "bn_calibration": {"requested_batches": 200, "batches": 200,
                           "trainable_weights_unchanged": True},
        "physical_cost": {"physical_total_parameters": 4_000_000},
        "physical_validation": selection["selection_validation"],
    }
    export = tmp_path / "export_only"
    export.mkdir()
    (tmp_path / "one_shot_state.json").write_text(json.dumps(state))
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    (export / "diagnostics.json").write_text(json.dumps(diagnostics))
    for path in (tmp_path / "selected_checkpoint.pt", export / "deployment.pt",
                 tmp_path / "resolved_config.yaml"):
        path.touch()
    return tmp_path


def test_new_search_reader_accepts_calibrated_export_while_legacy_reader_stays_strict(
    study, completed_bn_search,
):
    with pytest.raises(RuntimeError, match="search export diagnostics are incomplete"):
        launcher.epoch_split._search_result(completed_bn_search, 60, 90)
    result = launcher.read_search_result(completed_bn_search, study)
    assert result["selected_epoch"] == 44
    assert result["search_selection_bn_calibration_batches"] == 200
    assert result["physical_parameters"] == 4_000_000
    assert result["physical_validation"] == result["selection_validation"]


@pytest.mark.parametrize("drift", ["bn_missing", "bn_short", "weights", "validation", "ledger", "predictor"])
def test_new_search_reader_rejects_incomplete_calibrated_export(study, completed_bn_search, drift):
    diagnostic_path = completed_bn_search / "export_only/diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    if drift == "bn_missing":
        diagnostic["bn_calibration_batches"] = 0
    elif drift == "bn_short":
        diagnostic["bn_calibration"]["batches"] = 100
    elif drift == "weights":
        diagnostic["bn_calibration"]["trainable_weights_unchanged"] = False
    elif drift == "validation":
        diagnostic["physical_validation"]["example_count"] = 128
    elif drift == "predictor":
        diagnostic["selection_predictor"] = "dependency_safe_all_physical_survivors_open"
    else:
        state_path = completed_bn_search / "one_shot_state.json"
        state = json.loads(state_path.read_text())
        state["shared_search_ledger"]["search_epochs_consumed"] = 59
        state_path.write_text(json.dumps(state))
    diagnostic_path.write_text(json.dumps(diagnostic))
    with pytest.raises(ValueError):
        launcher.read_search_result(completed_bn_search, study)
