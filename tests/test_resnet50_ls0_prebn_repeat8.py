"""Eight fresh repeats of the exact strict floor-.001 pre-BN recipe."""
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
    "prebn_repeat8_launcher", ROOT / "scripts/launch_resnet50_cifar10_ls0_curve7.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture(scope="module")
def study():
    return launcher.load_study(launcher.PREBN_REPEAT_CONFIG)


@pytest.fixture(scope="module")
def rows(study):
    return launcher.validate_curve_configs(study)


def test_dedicated_repeat_command_validates_eight_fresh_150_epoch_paths():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/launch_resnet50_cifar10_ls0_prebn_repeat8.py"),
         "--validate-only"], cwd=ROOT, check=True, capture_output=True, text=True,
    )
    preflight = json.loads(result.stdout)
    assert preflight["protocol"] == launcher.PREBN_REPEAT_PROTOCOL
    assert len(preflight["profiles"]) == 8
    assert preflight["total_training_epochs_per_model"] == 150
    assert preflight["total_training_epochs_for_study"] == 1200
    assert {row["search_selection_bn_calibration_batches"] for row in preflight["profiles"]} == {0}
    assert {(row["search_selection_epoch_start"], row["search_selection_epoch_end"])
            for row in preflight["profiles"]} == {(31, 60)}


def test_manifest_repeats_the_selected_source_point_and_pinned_inputs(study):
    original = launcher.load_study(launcher.PREBN_REPRO_CONFIG)
    source = next(row for row in original["profiles"] if row["id"] == "large_15m_keep001")
    expected = {key: value for key, value in source.items() if key != "id"}
    assert [row["id"] for row in study["profiles"]] == [
        f"large_15m_keep001_repeat_{index:02d}" for index in range(1, 9)
    ]
    assert all({key: value for key, value in row.items() if key != "id"} == expected
               for row in study["profiles"])
    assert study["expected_reference_history_sha256"] == original["expected_reference_history_sha256"]
    assert study["expected_initializer_sha256"] == original["expected_initializer_sha256"]
    assert study["output"] != original["output"]


@pytest.mark.parametrize("phase", ["search", "recovery"])
def test_all_sixteen_phase_configs_exactly_match_source_point_with_seed42(study, phase):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    original = launcher.load_study(launcher.PREBN_REPRO_CONFIG)
    source = next(row for row in original["profiles"] if row["id"] == "large_15m_keep001")
    expected = schema.compose_config(original[f"{phase}_config"], launcher.threshold_overrides(source))
    expected_values = OmegaConf.to_container(expected, resolve=True)
    for row in study["profiles"]:
        cfg = schema.compose_config(study[f"{phase}_config"], launcher.threshold_overrides(row))
        launcher.validate_recipe(cfg, study, row, phase)
        assert OmegaConf.to_container(cfg, resolve=True) == expected_values
        assert cfg.seed == cfg.dataloaders.seed == cfg.dataloaders.loader_seed == 42
        assert cfg.accuracy_guided.eligibility.min_keep_ratio == .001
        assert (cfg.one_shot.search_epochs, cfg.one_shot.final_epochs) == (60, 90)
        assert schema.search_selection_bounds(cfg) == (31, 60)
        assert cfg.one_shot.search_selection_bn_calibration_batches == 0
        assert cfg.accuracy_guided.guard.train_bn_calibration_batches == (0 if phase == "search" else 200)


@pytest.mark.parametrize("drift", [
    "seven", "nine", "duplicate_id", "other_point", "soft", "hard", "floor",
    "post_bn", "bool_bn", "window", "initializer", "reference", "seed_override",
])
def test_manifest_rejects_changed_repeat_count_or_recipe(study, tmp_path, drift):
    changed = copy.deepcopy(study)
    row = changed["profiles"][0]
    if drift == "seven":
        changed["profiles"].pop()
    elif drift == "nine":
        extra = copy.deepcopy(row)
        extra["id"] = "large_15m_keep001_repeat_09"
        changed["profiles"].append(extra)
    elif drift == "duplicate_id":
        changed["profiles"][1]["id"] = row["id"]
    elif drift == "other_point":
        row["point_id"] = "compact_5m"
    elif drift == "soft":
        row["soft_drop"] = .0025
    elif drift == "hard":
        row["hard_drop"] = .005
    elif drift == "floor":
        row["min_keep_ratio"] = .08
    elif drift == "post_bn":
        changed["search_selection_bn_calibration_batches"] = 200
    elif drift == "bool_bn":
        changed["search_selection_bn_calibration_batches"] = False
    elif drift == "window":
        changed["search_selection_last_epochs"] = 60
    elif drift == "initializer":
        changed["expected_initializer_sha256"] = "0" * 64
    elif drift == "reference":
        changed["expected_reference_history_sha256"] = "0" * 64
    else:
        row["seed"] = 43
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError):
        launcher.load_study(path)


def test_each_repeat_launches_fresh_search_and_reuses_only_its_own_result(study, rows, monkeypatch, tmp_path):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    commands = []

    def launch(command):
        commands.append(command)
        destination = Path(command[command.index("--output") + 1])
        assert not destination.exists()
        destination.mkdir(parents=True)
        overrides = [command[i + 1] for i, part in enumerate(command) if part == "--override"]
        cfg = schema.compose_config(command[command.index("--config-name") + 1], overrides)
        assert cfg.seed == 42
        OmegaConf.save(cfg, destination / "resolved_config.yaml")

    monkeypatch.setattr(launcher.epoch_split, "_run", launch)
    monkeypatch.setattr(launcher, "read_search_result", lambda *args: {"selected_epoch": 51})
    monkeypatch.setattr(launcher.epoch_split, "_recovery_result", lambda *args: {"status": "completed"})
    monkeypatch.setattr(launcher, "validate_saved_selection_bn", lambda *args: None)
    args = SimpleNamespace(resume_completed=False, keep_nested_checkpoints=True)
    for row in rows:
        launcher.run_pair(study, row, tmp_path / "reference", tmp_path / "runs", args)
    assert len(commands) == 16
    destinations = [Path(command[command.index("--output") + 1]) for command in commands]
    assert len(set(destinations)) == 16
    for index, row in enumerate(rows):
        search, recovery = commands[2 * index:2 * index + 2]
        assert "--search-only" in search and "--reuse-search" not in search
        assert "--resume-search" not in search
        assert "--reuse-search" in recovery and "--skip-test" in recovery
        expected_search = tmp_path / "runs" / f"search_{row['id']}"
        assert destinations[2 * index] == expected_search
        assert Path(recovery[recovery.index("--reuse-search") + 1]) == expected_search
        assert destinations[2 * index + 1] == tmp_path / "runs" / f"recovery_{row['id']}"


@pytest.mark.parametrize("phase", ["search", "recovery"])
@pytest.mark.parametrize("drift", ["seed", "floor", "window", "candidate_bn", "export_bn"])
def test_repeat_resume_rejects_changed_saved_phase_without_launch(study, rows, phase, drift, monkeypatch, tmp_path):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    row = rows[0]
    for current_phase in ("search", "recovery"):
        destination = tmp_path / f"{current_phase}_{row['id']}"
        destination.mkdir()
        cfg = schema.compose_config(study[f"{current_phase}_config"], launcher.threshold_overrides(row))
        if current_phase == phase:
            if drift == "seed":
                cfg.seed = 43
            elif drift == "floor":
                cfg.accuracy_guided.eligibility.min_keep_ratio = .08
            elif drift == "window":
                cfg.one_shot.search_selection_last_epochs = 60
            elif drift == "candidate_bn":
                cfg.one_shot.search_selection_bn_calibration_batches = 200
            else:
                cfg.accuracy_guided.guard.train_bn_calibration_batches = 200 if phase == "search" else 0
        OmegaConf.save(cfg, destination / "resolved_config.yaml")
    monkeypatch.setattr(launcher, "read_search_result", lambda *args: {"selected_epoch": 51})
    monkeypatch.setattr(launcher.epoch_split, "_recovery_result", lambda *args: {"status": "completed"})
    monkeypatch.setattr(launcher, "validate_saved_selection_bn", lambda *args: None)
    monkeypatch.setattr(launcher.epoch_split, "_run", lambda *args: pytest.fail("Changed saved recipe launched training"))
    with pytest.raises(ValueError):
        launcher.run_pair(study, row, tmp_path / "reference", tmp_path,
                          SimpleNamespace(resume_completed=True, keep_nested_checkpoints=True))


@pytest.mark.parametrize("drift", [None, "post_bn", "ranking", "fallback_window"])
def test_repeat_keeps_strict_saved_selection_guards(study, tmp_path, drift):
    predictor = "dependency_safe_all_physical_survivors_open"
    selection = {
        "selected_epoch": 51, "policy": "best_feasible_compact", "selection_predictor": predictor,
        "trace": {"candidate_epoch_start": 31, "candidate_epoch_end": 60,
                  "selection_predictor": predictor, "policy": "best_feasible_compact",
                  "fallback_used": False},
    }
    if drift == "post_bn":
        selection["selection_predictor"] = "physical_after_train_bn_calibration"
    elif drift == "ranking":
        selection["policy"] = selection["trace"]["policy"] = "best_validation_within_one_percent"
    elif drift == "fallback_window":
        selection["policy"] = selection["trace"]["policy"] = "best_validation_last_epochs_fallback"
        selection["trace"].update(fallback_used=True, fallback_last_epochs=60, fallback_epoch_start=1)
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    if drift is None:
        launcher.validate_saved_selection_bn(tmp_path, study)
    else:
        with pytest.raises(ValueError):
            launcher.validate_saved_selection_bn(tmp_path, study)


@pytest.mark.parametrize("state_change", [None, "bn", "weights"])
def test_all_eight_validation_rows_lock_before_frozen_official_test(study, rows, monkeypatch, tmp_path, state_change):
    output = tmp_path / "study"
    monkeypatch.setattr(launcher, "validate_curve_configs", lambda current: rows)
    monkeypatch.setattr(launcher, "validate_reference", lambda *args: {"sha256": "locked"})
    monkeypatch.setattr(launcher.epoch_split, "_environment_preflight", lambda free_space: {})
    trained, evaluated = [], []

    def run_pair(current_study, row, reference, root, args):
        trained.append(row["id"])
        return {
            "search": {"run_dir": str(root / f"search_{row['id']}")},
            "recovery": {"run_dir": str(root / f"recovery_{row['id']}"),
                         "physical_parameters": 8_666_833, "validation_accuracy": .941,
                         "validation_ce_loss": .4, "quality_feasible": True,
                         "selected_final_epoch": 80},
        }

    def evaluate(command):
        assert trained == [row["id"] for row in rows]
        state = json.loads((output / "study_state.json").read_text())
        assert state["validation_locked_before_test"] is True
        with (output / "validation_results.csv").open() as stream:
            locked = list(csv.DictReader(stream))
        assert len(locked) == 8 and all(row["test_accuracy"] == "" for row in locked)
        assert [row["id"] for row in locked] == [row["id"] for row in rows]
        assert {row["point_id"] for row in locked} == {"large_15m"}
        assert {row["search_selection_bn_calibration_batches"] for row in locked} == {"0"}
        assert Path(command[1]).name == "evaluate_one_shot_pruning_test.py"
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir(parents=True)
        evaluated.append(destination.parent.name)

    def test_result(test_dir, recovery_dir):
        return {"runs": [{"physical_parameters": 8_666_833,
                          "model_state_unchanged": state_change != "weights",
                          "bn_counters_unchanged": state_change != "bn",
                          "test": {"accuracy": .9333, "ce_loss": .4,
                                   "correct_count": 9333, "example_count": 10000}}]}

    monkeypatch.setattr(launcher, "run_pair", run_pair)
    monkeypatch.setattr(launcher.epoch_split, "_run", evaluate)
    monkeypatch.setattr(launcher.epoch_split, "_test_result", test_result)
    state = launcher.main(["--config", str(launcher.PREBN_REPEAT_CONFIG), "--output", str(output),
                           "--reference-output", str(tmp_path / "reference")])
    assert len(trained) == len(evaluated) == 8
    assert state["test_comparison_scope"] == "exploratory"
    if state_change:
        assert state["status"] == "completed_with_failures"
        assert len(state["test_failures"]) == 8 and state["tests"] == {}
    else:
        assert state["status"] == "completed"
        with (output / "final_results.csv").open() as stream:
            recorded = list(csv.DictReader(stream))
        assert len(recorded) == 8
        assert all(row["test_accuracy"] == "0.9333" for row in recorded)
