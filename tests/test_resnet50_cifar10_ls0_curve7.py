import copy
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_resnet50_cifar10_ls0_curve7",
    ROOT / "scripts/launch_resnet50_cifar10_ls0_curve7.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture(scope="module")
def validated_rows():
    return launcher.validate_curve_configs(launcher.load_study())


@pytest.fixture(scope="module")
def narrow_validated_rows():
    return launcher.validate_curve_configs(launcher.load_study(launcher.NARROW_CONFIG))


@pytest.fixture(scope="module")
def moderate_validated_rows():
    return launcher.validate_curve_configs(launcher.load_study(launcher.MODERATE_CONFIG))


@pytest.fixture(scope="module")
def night_validated_rows():
    return launcher.validate_curve_configs(launcher.load_study(launcher.NIGHT_CONFIG))


def test_seven_independent_150_epoch_profiles_report_real_architecture_floor(validated_rows):
    assert len(validated_rows) == 7
    assert [row["target_parameters"] for row in validated_rows] == [
        23_000_000, 18_000_000, 12_000_000, 8_000_000, 5_000_000, 4_000_000, 3_700_000,
    ]
    assert {row["theoretical_parameter_floor"] for row in validated_rows} == {3_683_893}
    assert launcher.load_study()["requested_min_parameters"] == 3_700_000
    assert {row["min_keep_ratio"] for row in validated_rows} == {0.08}
    assert {(row["search_selection_epoch_start"], row["search_selection_epoch_end"])
            for row in validated_rows} == {(31, 60)}
    assert all(row["label_smoothing"] == 0 and row["split"] == "60/90" for row in validated_rows)
    assert all(row["soft_drop"] * 2 == row["hard_drop"] for row in validated_rows)
    assert all(row["resolved_log_step"] == pytest.approx(0.9210340371976183)
               for row in validated_rows)


@pytest.mark.parametrize("drift", [
    "label_smoothing", "entropy_beta", "gate_selector", "adaptive_lambda", "optimizer",
    "search_window", "keep_floor",
])
def test_saved_or_composed_recipe_drift_is_rejected(drift):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    study = launcher.load_study()
    row = study["profiles"][0]
    config = schema.compose_config(study["recovery_config"], launcher.threshold_overrides(row))
    fields = {
        "label_smoothing": ("model.criterion.label_smoothing", 0.05),
        "entropy_beta": ("model.entropy_regularization_coef", 0.3),
        "gate_selector": ("accuracy_guided.drop_mode", "expected_open_count"),
        "adaptive_lambda": ("training_arguments.adaptive_lambda.enabled", False),
        "optimizer": ("optimizer.lr", 0.002),
        "search_window": ("one_shot.search_selection_last_epochs", 60),
        "keep_floor": ("accuracy_guided.eligibility.min_keep_ratio", 0.001),
    }
    key, value = fields[drift]
    OmegaConf.update(config, key, value, force_add=True)
    with pytest.raises(ValueError):
        launcher.validate_recipe(config, study, row, "recovery")


@pytest.fixture(params=["curve7", "narrow4", "moderate3", "night9h"])
def simulated_study(monkeypatch, tmp_path, validated_rows, narrow_validated_rows,
                    moderate_validated_rows, night_validated_rows, request):
    manifest_path, selected_rows = {
        "curve7": (launcher.DEFAULT_CONFIG, validated_rows),
        "narrow4": (launcher.NARROW_CONFIG, narrow_validated_rows),
        "moderate3": (launcher.MODERATE_CONFIG, moderate_validated_rows),
        "night9h": (launcher.NIGHT_CONFIG, night_validated_rows),
    }[request.param]
    rows = copy.deepcopy(selected_rows)
    monkeypatch.setattr(launcher, "validate_curve_configs", lambda study: rows)
    monkeypatch.setattr(launcher, "validate_reference", lambda study, reference: {
        "root": str(reference), "sha256": {"initializer_path": "same_initializer"},
        "initializer_model_state_hash": "same_state",
    })
    monkeypatch.setattr(launcher.epoch_split, "_environment_preflight", lambda space: {})
    output = tmp_path / "curve"
    evaluated = []

    def fake_pair(study, row, reference, destination, args):
        return {
            "search": {"run_dir": str(destination / f"search_{row['id']}")},
            "recovery": {
                "run_dir": str(destination / f"recovery_{row['id']}"),
                "physical_parameters": row["target_parameters"],
                "validation_accuracy": 0.94, "validation_ce_loss": 0.4,
                "quality_feasible": True, "selected_final_epoch": 90,
            },
        }

    def fake_evaluation(command, **kwargs):
        assert Path(command[1]).name == "evaluate_one_shot_pruning_test.py"
        state = json.loads((output / "study_state.json").read_text())
        assert state["validation_locked_before_test"] is True
        with (output / "validation_results.csv").open() as stream:
            locked = list(csv.DictReader(stream))
        assert len(locked) == len(rows) and all(row["test_accuracy"] == "" for row in locked)
        assert [float(row["resolved_log_step"]) for row in locked] == [
            row["resolved_log_step"] for row in rows
        ]
        test_dir = Path(command[command.index("--output") + 1])
        test_dir.mkdir(parents=True)
        evaluated.append(test_dir.parent.name.removeprefix("recovery_"))

    def fake_test_summary(test_dir, recovery_dir):
        identifier = recovery_dir.name.removeprefix("recovery_")
        row = next(row for row in rows if row["id"] == identifier)
        return {"runs": [{
            "physical_parameters": row["target_parameters"],
            "model_state_unchanged": True, "bn_counters_unchanged": True,
            "test": {"accuracy": 0.93, "ce_loss": 0.5,
                     "correct_count": 9300, "example_count": 10000},
        }]}

    monkeypatch.setattr(launcher, "run_pair", fake_pair)
    monkeypatch.setattr(launcher.epoch_split, "_run", fake_evaluation)
    monkeypatch.setattr(launcher.epoch_split, "_test_result", fake_test_summary)
    return SimpleNamespace(rows=rows, output=output, evaluated=evaluated, fake_pair=fake_pair,
                           manifest_path=manifest_path,
                           argv=["--config", str(manifest_path), "--output", str(output),
                                 "--reference-output", str(tmp_path / "ref")])


def test_one_failed_profile_does_not_discard_other_frozen_test_results(simulated_study, monkeypatch):
    simulation = simulated_study
    failed_id = simulation.rows[-1]["id"]

    def fail_one(study, row, reference, destination, args):
        if row["id"] == failed_id:
            raise RuntimeError("no feasible search")
        return simulation.fake_pair(study, row, reference, destination, args)

    monkeypatch.setattr(launcher, "run_pair", fail_one)
    state = launcher.main(simulation.argv)
    assert state["status"] == "completed_with_failures"
    assert len(state["tests"]) == len(simulation.evaluated) == len(simulation.rows) - 1
    with (simulation.output / "final_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(simulation.rows)
    failed = next(row for row in rows if row["id"] == failed_id)
    assert failed["status"] == "failed" and failed["test_accuracy"] == ""
    assert failed["physical_parameters"] == ""
    assert all(row["test_accuracy"] == "0.93" for row in rows if row["id"] != failed_id)


def test_resume_reuses_existing_frozen_tests_and_rejects_manifest_changes(simulated_study, tmp_path):
    simulation = simulated_study
    launcher.main(simulation.argv)
    assert len(simulation.evaluated) == len(simulation.rows)
    resumed = launcher.main(simulation.argv + ["--resume-completed"])
    assert resumed["status"] == "completed" and len(simulation.evaluated) == len(simulation.rows)
    manifest = launcher.load_study(simulation.manifest_path)
    manifest["profiles"][-1]["hard_drop"] *= 1.1
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="different settings/reference"):
        launcher.main(simulation.argv + ["--resume-completed", "--config", str(changed)])


def test_incomplete_existing_search_is_not_silently_retrained(tmp_path, monkeypatch):
    study = launcher.load_study()
    row = study["profiles"][0]
    (tmp_path / f"search_{row['id']}").mkdir()
    calls = []
    monkeypatch.setattr(launcher.epoch_split, "_run", lambda command: calls.append(command))
    with pytest.raises(FileNotFoundError):
        launcher.run_pair(study, row, tmp_path / "ref", tmp_path,
                          SimpleNamespace(resume_completed=True, keep_nested_checkpoints=False))
    assert calls == []


def test_narrow_profiles_have_independent_thresholds_and_run_smallest_first(narrow_validated_rows):
    assert [row["target_parameters"] for row in narrow_validated_rows] == [
        3_700_000, 6_000_000, 10_000_000, 15_000_000,
    ]
    assert [(row["soft_drop"], row["hard_drop"]) for row in narrow_validated_rows] == [
        (0.05, 0.055), (0.015, 0.0165), (0.0075, 0.00825), (0.00025, 0.00035),
    ]
    assert all(row["soft_drop"] != row["hard_drop"] / 2 for row in narrow_validated_rows)
    assert {row["theoretical_parameter_floor"] for row in narrow_validated_rows} == {3_683_893}
    assert all(row["target_architecturally_reachable"] for row in narrow_validated_rows)
    assert all(row["resolved_log_step"] == pytest.approx(0.9210340371976183)
               for row in narrow_validated_rows)


def test_moderate_profiles_avoid_retraining_the_completed_large_model(moderate_validated_rows):
    assert [row["target_parameters"] for row in moderate_validated_rows] == [
        3_700_000, 6_000_000, 10_000_000,
    ]
    assert [(row["soft_drop"], row["hard_drop"]) for row in moderate_validated_rows] == [
        (0.05, 0.06), (0.015, 0.018), (0.0075, 0.009),
    ]
    assert {row["theoretical_parameter_floor"] for row in moderate_validated_rows} == {3_683_893}
    assert all(row["resolved_log_step"] == pytest.approx(0.9210340371976183)
               for row in moderate_validated_rows)


@pytest.mark.parametrize("phase", ["search", "recovery"])
def test_narrow_saved_config_rejects_a_silently_halved_soft_threshold(phase):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    study = launcher.load_study(launcher.NARROW_CONFIG)
    row = study["profiles"][0]
    config = schema.compose_config(study[f"{phase}_config"], launcher.threshold_overrides(row))
    launcher.validate_recipe(config, study, row, phase)
    OmegaConf.update(config, "training_arguments.adaptive_lambda.soft_drop", row["hard_drop"] / 2)
    with pytest.raises(ValueError, match="Adaptive-lambda settings"):
        launcher.validate_recipe(config, study, row, phase)


@pytest.mark.parametrize("invalid", [
    "missing_soft", "negative_soft", "soft_equal_hard", "soft_above_hard", "nan_soft",
    "boolean_soft", "nan_hard", "boolean_hard", "wide_band", "three_profiles",
    "reversed_order", "target_above_15m", "duplicate_id",
])
def test_invalid_narrow_manifest_is_rejected(tmp_path, invalid):
    study = launcher.load_study(launcher.NARROW_CONFIG)
    row = study["profiles"][0]
    if invalid == "missing_soft":
        row.pop("soft_drop")
    elif invalid == "negative_soft":
        row["soft_drop"] = -0.01
    elif invalid == "soft_equal_hard":
        row["soft_drop"] = row["hard_drop"]
    elif invalid == "soft_above_hard":
        row["soft_drop"] = 0.06
    elif invalid == "nan_soft":
        row["soft_drop"] = float("nan")
    elif invalid == "boolean_soft":
        row["soft_drop"] = False
    elif invalid == "nan_hard":
        row["hard_drop"] = float("nan")
    elif invalid == "boolean_hard":
        row["hard_drop"] = True
    elif invalid == "wide_band":
        row["hard_drop"] = 0.10
    elif invalid == "three_profiles":
        study["profiles"].pop()
    elif invalid == "reversed_order":
        study["profiles"].reverse()
    elif invalid == "target_above_15m":
        study["profiles"][-1]["target_parameters"] = 16_000_000
    elif invalid == "duplicate_id":
        study["profiles"][-1]["id"] = row["id"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError):
        launcher.load_study(path)


def test_night_control_and_half_step_change_only_one_training_parameter(night_validated_rows):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    assert len(night_validated_rows) == 3
    assert [row["resolved_log_step"] for row in night_validated_rows] == pytest.approx([
        0.9210340371976183, 0.9210340371976183, 0.46,
    ])
    study = launcher.load_study(launcher.NIGHT_CONFIG)
    control, half_step = study["profiles"][1:]
    for phase in ("search", "recovery"):
        configs = [schema.compose_config(study[f"{phase}_config"], launcher.threshold_overrides(row))
                   for row in (control, half_step)]
        for config, row in zip(configs, (control, half_step)):
            launcher.validate_recipe(config, study, row, phase)
        left, right = [OmegaConf.to_container(config, resolve=False) for config in configs]
        assert left["training_arguments"]["adaptive_lambda"].pop("log_step") == "auto"
        assert right["training_arguments"]["adaptive_lambda"].pop("log_step") == 0.46
        assert left == right


@pytest.mark.parametrize("phase", ["search", "recovery"])
def test_night_resume_rejects_a_different_lambda_step(phase):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    study = launcher.load_study(launcher.NIGHT_CONFIG)
    row = study["profiles"][-1]
    config = schema.compose_config(study[f"{phase}_config"], launcher.threshold_overrides(row))
    OmegaConf.update(config, "training_arguments.adaptive_lambda.log_step", "auto")
    with pytest.raises(ValueError, match="Adaptive-lambda settings"):
        launcher.validate_recipe(config, study, row, phase)


@pytest.mark.parametrize("invalid_step", [0, -0.1, False, float("nan"), "fixed", None])
def test_invalid_night_step_is_rejected(tmp_path, invalid_step):
    study = launcher.load_study(launcher.NIGHT_CONFIG)
    study["profiles"][-1]["log_step"] = invalid_step
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError, match="log_step"):
        launcher.load_study(path)


def test_night_budget_retains_ninety_minutes_of_overhead(night_validated_rows, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "validate_curve_configs", lambda study: night_validated_rows)
    preflight = launcher.main(["--config", str(launcher.NIGHT_CONFIG), "--validate-only"])
    assert preflight["time_budget"] == {
        "hours": 9, "minutes_per_epoch": 1, "reserve_minutes": 90,
        "planned_training_epochs": 450, "estimated_training_minutes": 450,
        "estimated_total_minutes_with_reserve": 540, "hard_wallclock_timeout": False,
    }
    study = launcher.load_study(launcher.NIGHT_CONFIG)
    study["time_budget"]["reserve_minutes"] = 91
    path = tmp_path / "over_budget.yaml"
    path.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError, match="exceed the time budget"):
        launcher.load_study(path)
