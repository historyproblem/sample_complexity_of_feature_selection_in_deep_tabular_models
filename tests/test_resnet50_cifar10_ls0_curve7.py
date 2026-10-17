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


@pytest.fixture
def simulated_study(monkeypatch, tmp_path, validated_rows):
    rows = copy.deepcopy(validated_rows)
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
        assert len(locked) == 7 and all(row["test_accuracy"] == "" for row in locked)
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
                           argv=["--output", str(output), "--reference-output", str(tmp_path / "ref")])


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
    assert len(state["tests"]) == len(simulation.evaluated) == 6
    with (simulation.output / "final_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7
    failed = next(row for row in rows if row["id"] == failed_id)
    assert failed["status"] == "failed" and failed["test_accuracy"] == ""
    assert failed["physical_parameters"] == ""
    assert all(row["test_accuracy"] == "0.93" for row in rows if row["id"] != failed_id)


def test_resume_reuses_existing_frozen_tests_and_rejects_manifest_changes(simulated_study, tmp_path):
    simulation = simulated_study
    launcher.main(simulation.argv)
    assert len(simulation.evaluated) == 7
    resumed = launcher.main(simulation.argv + ["--resume-completed"])
    assert resumed["status"] == "completed" and len(simulation.evaluated) == 7
    manifest = launcher.load_study()
    manifest["profiles"][0]["hard_drop"] *= 2
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
