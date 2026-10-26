"""Pre-BN reproduction: historical thresholds, four independent 150-epoch paths."""
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
    "prebn_repro4_launcher", ROOT / "scripts/launch_resnet50_cifar10_ls0_curve7.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture(scope="module")
def study():
    return launcher.load_study(launcher.PREBN_REPRO_CONFIG)


@pytest.fixture(scope="module")
def rows(study):
    return launcher.validate_curve_configs(study)


def test_dedicated_command_validates_four_historical_paths_without_training():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/launch_resnet50_cifar10_ls0_prebn_repro4.py"),
         "--validate-only"], cwd=ROOT, check=True, capture_output=True, text=True,
    )
    preflight = json.loads(result.stdout)
    assert preflight["protocol"] == launcher.PREBN_REPRO_PROTOCOL
    assert len(preflight["profiles"]) == 4
    assert preflight["total_training_epochs_per_model"] == 150
    assert preflight["total_training_epochs_for_study"] == 600
    assert {row["search_selection_bn_calibration_batches"] for row in preflight["profiles"]} == {0}
    assert {(row["search_selection_epoch_start"], row["search_selection_epoch_end"])
            for row in preflight["profiles"]} == {(31, 60)}


def test_grid_preserves_two_historical_threshold_pairs_and_pinned_inputs(study, rows):
    assert {(p["point_id"], p["soft_drop"], p["hard_drop"], p["min_keep_ratio"])
            for p in study["profiles"]} == {
        ("compact_5m", .015, .03, .001), ("compact_5m", .015, .03, .08),
        ("large_15m", .00025, .0005, .001), ("large_15m", .00025, .0005, .08),
    }
    assert study["expected_reference_history_sha256"] == launcher.QUALITY_GRID_REFERENCE_HASH
    assert study["expected_initializer_sha256"] == launcher.FULL150_INITIALIZER_HASH
    assert len({p["id"] for p in study["profiles"]}) == 4
    assert {r["theoretical_parameter_floor"] for r in rows if r["min_keep_ratio"] == .08} == {3_683_893}
    assert all(r["resolved_log_step"] == pytest.approx(.9210340371976183) for r in rows)


@pytest.mark.parametrize("phase", ["search", "recovery"])
def test_all_eight_phase_configs_disable_search_calibration_and_preserve_recovery(study, phase):
    from hydra.utils import instantiate
    from net_complexity.training import one_shot_pruning_config as schema

    for row in study["profiles"]:
        cfg = schema.compose_config(study[f"{phase}_config"], launcher.threshold_overrides(row))
        launcher.validate_recipe(cfg, study, row, phase)
        assert schema.validate_config(cfg) == 150
        assert (cfg.one_shot.search_epochs, cfg.one_shot.final_epochs) == (60, 90)
        assert schema.search_selection_bounds(cfg) == (31, 60)
        assert cfg.one_shot.search_selection_bn_calibration_batches == 0
        assert cfg.accuracy_guided.guard.train_bn_calibration_batches == (0 if phase == "search" else 200)
        assert cfg.accuracy_guided.eligibility.min_keep_ratio == row["min_keep_ratio"]
        assert instantiate(cfg.model.criterion).label_smoothing == 0
        assert cfg.model.entropy_regularization_coef == 0
        adaptive = cfg.training_arguments.adaptive_lambda
        assert adaptive.enabled is True and adaptive.log_step == "auto"
        assert (adaptive.soft_drop, adaptive.hard_drop) == (row["soft_drop"], row["hard_drop"])
        if phase == "recovery":
            plan = schema.resolved_branch_plan(cfg)
            assert len(plan) == 1
            assert plan[0]["model_state"] == "selected_surviving_state"
            assert plan[0]["optimizer_state"] == "fresh"
            assert plan[0]["scheduler_state"] == schema.FRESH_SCHEDULER


@pytest.mark.parametrize("drift", [
    "missing_cell", "duplicate_cell", "changed_threshold", "third_point", "wrong_floor",
    "post_bn", "bool_bn", "early_window", "reference", "initializer",
])
def test_manifest_rejects_incomplete_grid_or_changed_experiment(study, tmp_path, drift):
    changed = copy.deepcopy(study)
    row = changed["profiles"][0]
    if drift == "missing_cell":
        changed["profiles"].pop()
    elif drift == "duplicate_cell":
        duplicate = copy.deepcopy(row)
        duplicate["id"] = changed["profiles"][1]["id"]
        changed["profiles"][1] = duplicate
    elif drift == "changed_threshold":
        row["hard_drop"] *= .9
    elif drift == "third_point":
        row["point_id"] = "extra3"
    elif drift == "wrong_floor":
        row["min_keep_ratio"] = .01
    elif drift == "post_bn":
        changed["search_selection_bn_calibration_batches"] = 200
    elif drift == "bool_bn":
        changed["search_selection_bn_calibration_batches"] = False
    elif drift == "early_window":
        changed["search_selection_last_epochs"] = 60
    elif drift == "reference":
        changed["expected_reference_history_sha256"] = "0" * 64
    else:
        changed["expected_initializer_sha256"] = "0" * 64
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError):
        launcher.load_study(path)


def test_search_keeps_historical_minimum_size_ranking_and_recent_fallback(study):
    from net_complexity.training import one_shot_pruning_config as schema
    from net_complexity.training.one_shot_pruning import select_search_checkpoint_records

    cfg = schema.compose_config(study["search_config"], launcher.threshold_overrides(study["profiles"][0]))
    candidates = [
        {"epoch": 20, "accuracy": .94, "ce_loss": .2, "physical_cost": 3_000_000},
        {"epoch": 55, "accuracy": .9084, "ce_loss": .4, "physical_cost": 4_069_213},
        {"epoch": 60, "accuracy": .9158, "ce_loss": .3, "physical_cost": 4_082_457},
    ]
    selected, info = select_search_checkpoint_records(cfg, candidates, .9232, .03)
    assert selected["epoch"] == 55
    assert info["policy"] == "best_feasible_compact"
    selected, info = select_search_checkpoint_records(cfg, candidates, .99, .0005)
    assert selected["epoch"] == 60
    assert info["fallback_used"] is True


def test_runtime_skips_search_bn_and_calibrates_imported_recovery_weights(tmp_path, monkeypatch):
    import torch
    from net_complexity.training import one_shot_pruning as runtime
    from net_complexity.training import one_shot_pruning_config as schema
    from net_complexity.training.pruning_measurement import state_hash
    from omegaconf import OmegaConf

    spec = importlib.util.spec_from_file_location("prebn_runtime_fixture", ROOT / "tests/test_one_shot_candidate_bn.py")
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    cfg = fixture_module.short_config(tmp_path / "inputs", batches=0)
    original_calibrate = runtime.calibrate_bn
    monkeypatch.setattr(runtime, "calibrate_bn", lambda *args, **kwargs: pytest.fail("Search requested extra BN calibration"))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        state = runtime.run_one_shot_pruning(cfg, tmp_path / "search", search_only=True)
    finally:
        torch.set_num_threads(previous)
    assert state["status"] == "search_only_completed"
    assert state["branches"] == {}
    export = state["export_only"]
    assert export["bn_calibration_batches"] == 0
    assert export["bn_calibration"]["requested_batches"] == 0
    assert export["weights_and_bn_unchanged"] is True
    assert export["physical_state_hash"] == export["physical_state_hash_before_bn_calibration"]

    recovery_cfg = copy.deepcopy(cfg)
    OmegaConf.update(recovery_cfg, "one_shot", {
        "protocol": schema.QUALITY_RECOVERY_PROTOCOL,
        "search_epochs": 3, "final_epochs": 2, "reuse_search_required": True,
        "search_selection_last_epochs": 2, "search_selection_bn_calibration_batches": 0,
        "search_scheduler_horizon_epochs": 3, "search_scheduler_eta_min": 0.0,
        "execution_order": "single",
        "methods": [{"id": "fresh_optimizer_fresh_scheduler",
                     "model_state": "selected_surviving_state", "optimizer_state": "fresh",
                     "scheduler_state": schema.FRESH_SCHEDULER}],
        "repeats": [{"id": "repeat_1", "training_seed": 42}],
    }, merge=False, force_add=True)
    recovery_cfg.accuracy_guided.stage_plan[2].restart_policy = "branch_specific_optimizer_scheduler_handoff"
    # Tiny synthetic data exercises timing with two batches; the real eight
    # composed phase configs above separately require exactly BN200 in recovery.
    recovery_cfg.accuracy_guided.guard.train_bn_calibration_batches = 2
    original_build, original_train = runtime.build_planned_physical_branches, runtime.run_training
    imported_hashes, calibration_calls, recovery_calls = {}, [], []

    def record_import(*args, **kwargs):
        models, report = original_build(*args, **kwargs)
        for name, model in models.items():
            assert report[name]["initialization"] == "selected_surviving_state"
            imported_hashes[id(model)] = state_hash(model.state_dict())
        return models, report

    def calibrate_imported(model, *args, **kwargs):
        assert id(model) in imported_hashes
        assert state_hash(model.state_dict()) == imported_hashes[id(model)]
        assert recovery_calls == []
        parameter_hash = state_hash(dict(model.named_parameters()))
        result = original_calibrate(model, *args, **kwargs)
        assert state_hash(dict(model.named_parameters())) == parameter_hash
        calibration_calls.append(result)
        return result

    def train_after_calibration(*args, **kwargs):
        assert calibration_calls == [2]
        recovery_calls.append(True)
        return original_train(*args, **kwargs)

    monkeypatch.setattr(runtime, "build_planned_physical_branches", record_import)
    monkeypatch.setattr(runtime, "calibrate_bn", calibrate_imported)
    monkeypatch.setattr(runtime, "run_training", train_after_calibration)
    torch.set_num_threads(1)
    try:
        recovered = runtime.run_one_shot_pruning(
            recovery_cfg, tmp_path / "recovery", reuse_search_from=tmp_path / "search",
        )
    finally:
        torch.set_num_threads(previous)
    assert calibration_calls == [2] and recovery_calls == [True]
    assert recovered["selection"]["selected_epoch"] == state["selection"]["selected_epoch"]
    assert recovered["reused_search"]["training_epochs_executed_by_this_command"] == 0


@pytest.mark.parametrize("profile_index", range(4))
def test_pair_passes_floor_and_thresholds_to_both_processes(study, rows, profile_index, monkeypatch, tmp_path):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    row = rows[profile_index]
    commands = []

    def launch(command):
        commands.append(command)
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir(parents=True)
        overrides = [command[i + 1] for i, part in enumerate(command) if part == "--override"]
        cfg = schema.compose_config(command[command.index("--config-name") + 1], overrides)
        OmegaConf.save(cfg, destination / "resolved_config.yaml")

    monkeypatch.setattr(launcher.epoch_split, "_run", launch)
    monkeypatch.setattr(launcher, "read_search_result", lambda *args: {"selected_epoch": 42})
    monkeypatch.setattr(launcher.epoch_split, "_recovery_result", lambda *args: {"status": "completed"})
    monkeypatch.setattr(launcher, "validate_saved_selection_bn", lambda *args: None)
    launcher.run_pair(study, row, tmp_path / "reference", tmp_path / "run",
                      SimpleNamespace(resume_completed=False, keep_nested_checkpoints=True))
    assert len(commands) == 2
    assert "--search-only" in commands[0]
    assert "--reuse-search" in commands[1] and "--skip-test" in commands[1]
    for command, phase in zip(commands, ("search", "recovery")):
        cfg = OmegaConf.load(Path(command[command.index("--output") + 1]) / "resolved_config.yaml")
        assert cfg.accuracy_guided.eligibility.min_keep_ratio == row["min_keep_ratio"]
        assert cfg.training_arguments.adaptive_lambda.hard_drop == row["hard_drop"]
        assert cfg.one_shot.search_selection_bn_calibration_batches == 0
        assert cfg.accuracy_guided.guard.train_bn_calibration_batches == (0 if phase == "search" else 200)


@pytest.mark.parametrize("phase", ["search", "recovery"])
@pytest.mark.parametrize("drift", ["floor", "window", "post_bn", "export_bn"])
def test_resume_refuses_changed_phase_recipe_without_launching_training(study, rows, phase, drift, monkeypatch, tmp_path):
    from net_complexity.training import one_shot_pruning_config as schema
    from omegaconf import OmegaConf

    row = rows[0]
    for current_phase in ("search", "recovery"):
        path = tmp_path / f"{current_phase}_{row['id']}"
        path.mkdir()
        cfg = schema.compose_config(study[f"{current_phase}_config"], launcher.threshold_overrides(row))
        if current_phase == phase:
            if drift == "floor":
                cfg.accuracy_guided.eligibility.min_keep_ratio = .08 if row["min_keep_ratio"] == .001 else .001
            elif drift == "window":
                cfg.one_shot.search_selection_last_epochs = 60
            elif drift == "post_bn":
                cfg.one_shot.search_selection_bn_calibration_batches = 200
            else:
                cfg.accuracy_guided.guard.train_bn_calibration_batches = 200 if phase == "search" else 0
        OmegaConf.save(cfg, path / "resolved_config.yaml")
    monkeypatch.setattr(launcher, "read_search_result", lambda *args: {"selected_epoch": 42})
    monkeypatch.setattr(launcher.epoch_split, "_recovery_result", lambda *args: {"status": "completed"})
    monkeypatch.setattr(launcher, "validate_saved_selection_bn", lambda *args: None)
    monkeypatch.setattr(launcher.epoch_split, "_run", lambda *args: pytest.fail("Resume launched training"))
    with pytest.raises(ValueError):
        launcher.run_pair(study, row, tmp_path / "reference", tmp_path,
                          SimpleNamespace(resume_completed=True, keep_nested_checkpoints=True))


def selection_evidence():
    predictor = "dependency_safe_all_physical_survivors_open"
    return {
        "reference_epoch": 60, "selected_epoch": 42, "policy": "best_feasible_compact",
        "selection_predictor": predictor,
        "trace": {"candidate_epoch_start": 31, "candidate_epoch_end": 60,
                  "policy": "best_feasible_compact",
                  "selection_predictor": predictor, "fallback_used": False},
    }


@pytest.mark.parametrize("drift", [None, "predictor", "trace_predictor", "calibration", "window", "fallback"])
def test_saved_selection_cannot_relabel_post_bn_or_early_winner(study, tmp_path, drift):
    selection = selection_evidence()
    if drift == "predictor":
        selection["selection_predictor"] = "physical_after_train_bn_calibration"
    elif drift == "trace_predictor":
        selection["trace"]["selection_predictor"] = "physical_after_train_bn_calibration"
    elif drift == "calibration":
        selection["selection_bn_calibration"] = {"requested_batches": 200, "batches": 200}
    elif drift == "window":
        selection["trace"]["candidate_epoch_start"] = 1
    elif drift == "fallback":
        selection["trace"].update(fallback_used=True, fallback_last_epochs=60, fallback_epoch_start=1)
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    if drift is None:
        launcher.validate_saved_selection_bn(tmp_path, study)
    else:
        with pytest.raises(ValueError):
            launcher.validate_saved_selection_bn(tmp_path, study)


@pytest.mark.parametrize("export_batches", [0, 200])
def test_search_reader_rejects_additional_bn_in_selected_export(study, tmp_path, export_batches):
    state = {
        "status": "search_only_completed", "search_only": True, "branches": {},
        "protocol": launcher.epoch_split.SEARCH_PROTOCOL, "per_branch_total_allocated": 150,
        "shared_search_ledger": {"global_training_epoch": 60, "search_epochs_consumed": 60},
    }
    validation = {"accuracy": .92, "ce_loss": .3, "correct_count": 4600, "example_count": 5000}
    diagnostics = {
        "status": "measured", "bn_calibration_batches": export_batches,
        "bn_calibration": {"requested_batches": export_batches, "batches": export_batches},
        "physical_cost": {"physical_total_parameters": 4_000_000},
        "physical_validation": validation,
    }
    export = tmp_path / "export_only"
    export.mkdir()
    (tmp_path / "one_shot_state.json").write_text(json.dumps(state))
    (tmp_path / "selection.json").write_text(json.dumps(selection_evidence()))
    (export / "diagnostics.json").write_text(json.dumps(diagnostics))
    for path in (tmp_path / "selected_checkpoint.pt", export / "deployment.pt", tmp_path / "resolved_config.yaml"):
        path.touch()
    if export_batches:
        with pytest.raises((RuntimeError, ValueError)):
            launcher.read_search_result(tmp_path, study)
    else:
        result = launcher.read_search_result(tmp_path, study)
        assert result["selected_epoch"] == 42
        assert result["physical_validation"] == validation


@pytest.mark.parametrize("changed_bn", [False, True])
def test_four_validation_rows_lock_before_frozen_official_test(study, rows, monkeypatch, tmp_path, changed_bn):
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
                         "physical_parameters": 4_659_889, "validation_accuracy": .9376,
                         "validation_ce_loss": .4, "quality_feasible": False,
                         "selected_final_epoch": 80},
        }

    def evaluate(command):
        assert trained == [row["id"] for row in rows]
        state = json.loads((output / "study_state.json").read_text())
        assert state["validation_locked_before_test"] is True
        with (output / "validation_results.csv").open() as stream:
            locked = list(csv.DictReader(stream))
        assert len(locked) == 4 and all(row["test_accuracy"] == "" for row in locked)
        assert {row["point_id"] for row in locked} == {"compact_5m", "large_15m"}
        assert {row["search_selection_bn_calibration_batches"] for row in locked} == {"0"}
        assert Path(command[1]).name == "evaluate_one_shot_pruning_test.py"
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir(parents=True)
        evaluated.append(destination.parent.name)

    def test_result(test_dir, recovery_dir):
        return {"runs": [{"physical_parameters": 4_659_889,
                          "model_state_unchanged": True, "bn_counters_unchanged": not changed_bn,
                          "test": {"accuracy": .9331, "ce_loss": .4,
                                   "correct_count": 9331, "example_count": 10000}}]}

    monkeypatch.setattr(launcher, "run_pair", run_pair)
    monkeypatch.setattr(launcher.epoch_split, "_run", evaluate)
    monkeypatch.setattr(launcher.epoch_split, "_test_result", test_result)
    state = launcher.main(["--config", str(launcher.PREBN_REPRO_CONFIG), "--output", str(output),
                           "--reference-output", str(tmp_path / "reference")])
    assert len(trained) == len(evaluated) == 4
    assert state["test_comparison_scope"] == "exploratory"
    if changed_bn:
        assert state["status"] == "completed_with_failures"
        assert len(state["test_failures"]) == 4 and state["tests"] == {}
    else:
        assert state["status"] == "completed"
        with (output / "final_results.csv").open() as stream:
            recorded = list(csv.DictReader(stream))
        assert len(recorded) == 4
        assert all(row["quality_feasible"] == "False" for row in recorded)
        assert all(row["test_accuracy"] == "0.9331" for row in recorded)
