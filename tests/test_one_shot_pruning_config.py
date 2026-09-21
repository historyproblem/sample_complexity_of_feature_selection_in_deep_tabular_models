import builtins
import json
import math
from pathlib import Path
import shutil
import sys

from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytest
import torch

from net_complexity.training import one_shot_pruning_config as schema
from net_complexity.training.accuracy_guided_config import compose_config as compose_iterative, validate_config_v3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_one_shot_pruning as launcher


SEARCH_SWEEP = {
    "search60_soft0p5_crit1p0": (0.005, 0.010),
    "search60_soft1p0_crit1p5": (0.010, 0.015),
    "search60_soft0p75_crit1p0": (0.0075, 0.010),
    "search60_soft1p25_crit1p75": (0.0125, 0.0175),
}


def test_only_authorized_full_plan_composes_with_shared_base_contract():
    cfg = schema.compose_config()
    assert schema.validate_config(cfg) == 150
    assert cfg.one_shot.protocol == "pruning_v3_one_shot_60_90"
    assert cfg.one_shot.branches == ["inherited", "scratch"]
    assert cfg.one_shot.inherited_optimizer_state == "mapped_adamw_moments_and_step"
    assert [(s.id, s.epochs) for s in cfg.accuracy_guided.stage_plan] == [
        ("shared_search", 60), ("export_only", 0), ("final_recovery", 90)]
    assert cfg.training_arguments.adaptive_lambda.enabled is True
    assert cfg.training_arguments.adaptive_lambda.control_mode == "accuracy_only"
    assert cfg.training_arguments.batchnorm_recalibration == {"enabled": False}
    assert cfg.accuracy_guided.guard.train_bn_calibration_batches == 0
    assert cfg.model.backbone.resnet_block.regularization_normalization == "initial_channels"
    assert cfg.optimizer.gate_weight_decay_scale == 0
    assert cfg.seed == cfg.dataloaders.seed == cfg.dataloaders.loader_seed == 42
    assert cfg.optimizer.lr == 0.001 and cfg.dataloaders.batch_size == 128
    assert cfg.training_arguments.evaluate_test is False and cfg.dataloaders.include_test is False
    assert cfg.metrics.test_metrics == []
    assert not GlobalHydra.instance().is_initialized()


def test_handoff_ablation_config_is_single_axis_and_method_first():
    cfg = schema.compose_config(schema.HANDOFF_CONFIG_NAME)
    assert schema.validate_config(cfg) == 150
    plan = schema.resolved_branch_plan(cfg)
    assert [branch["id"] for branch in plan] == [
        "fresh_optimizer_fresh_scheduler__repeat_1",
        "mapped_optimizer_fresh_scheduler__repeat_1",
        "mapped_optimizer_resumed_scheduler__repeat_1",
        "fresh_optimizer_fresh_scheduler__repeat_2",
        "mapped_optimizer_fresh_scheduler__repeat_2",
        "mapped_optimizer_resumed_scheduler__repeat_2",
    ]
    assert {branch["model_state"] for branch in plan} == {"selected_surviving_state"}
    assert cfg.one_shot.search_scheduler_horizon_epochs == 150
    assert cfg.accuracy_guided.stage_plan[2].restart_policy == "branch_specific_optimizer_scheduler_handoff"
    assert schema.to_v3_config(cfg).accuracy_guided.stage_plan[2].restart_policy == "adamw_cosine_restart"
    report = schema.resolved_one_shot(cfg, check_inputs=False)
    assert report["execution_policy"]["comparison_axis"] == "optimizer and scheduler state only"
    assert report["budget"]["per_branch_budget_including_shared_search"] == 150
    assert report["budget"]["number_of_physical_branches"] == 6
    assert report["budget"]["total_unique_training_epochs_all_branches"] == 600


def test_target5m_recovery_config_maps_adamw_and_restarts_cosine_twice():
    cfg = schema.compose_config(schema.MAPPED_REPEATS_CONFIG_NAME)
    assert schema.validate_config(cfg) == 150
    plan = schema.resolved_branch_plan(cfg)
    assert [branch["id"] for branch in plan] == [
        "mapped_optimizer_fresh_scheduler__repeat_1",
        "mapped_optimizer_fresh_scheduler__repeat_2",
    ]
    assert [branch["training_seed"] for branch in plan] == [42, 43]
    assert {branch["optimizer_state"] for branch in plan} == {schema.MAPPED_OPTIMIZER}
    assert {branch["scheduler_state"] for branch in plan} == {schema.FRESH_SCHEDULER}
    assert cfg.one_shot.search_scheduler_horizon_epochs == 60
    assert cfg.one_shot.search_scheduler_eta_min == pytest.approx(0.00066443)
    assert cfg.accuracy_guided.eligibility.min_keep_ratio == pytest.approx(0.14)
    assert cfg.training_arguments.adaptive_lambda.enabled is True
    report = schema.resolved_one_shot(cfg, check_inputs=False)
    assert report["execution_policy"]["comparison_axis"] == (
        "recovery seed only; mapped optimizer and fresh scheduler fixed"
    )
    assert report["budget"]["per_branch_budget_including_shared_search"] == 150
    assert report["budget"]["total_unique_training_epochs_all_branches"] == 240


def test_target5m_quality_recovery_is_one_fresh_90_epoch_branch():
    cfg = schema.compose_config(schema.QUALITY_RECOVERY_CONFIG_NAME)
    assert schema.validate_config(cfg) == 150
    assert cfg.one_shot.reuse_search_required is True
    assert cfg.optimizer.lr == pytest.approx(0.001)
    assert cfg.one_shot.search_scheduler_eta_min == 0
    plan = schema.resolved_branch_plan(cfg)
    assert plan == [{
        "id": "fresh_optimizer_fresh_scheduler__repeat_1",
        "method": "fresh_optimizer_fresh_scheduler",
        "repeat": "repeat_1",
        "training_seed": 42,
        "model_state": "selected_surviving_state",
        "optimizer_state": "fresh",
        "scheduler_state": schema.FRESH_SCHEDULER,
    }]
    report = schema.resolved_one_shot(cfg, check_inputs=False)
    assert report["execution_policy"]["comparison_axis"] == (
        "recovery learning rate; inherited compact weights, fresh optimizer and fresh scheduler"
    )
    assert report["budget"]["per_branch_budget_including_shared_search"] == 150
    assert report["budget"]["number_of_physical_branches"] == 1
    assert report["budget"]["total_unique_training_epochs_all_branches"] == 150


@pytest.mark.parametrize("stem,drops", SEARCH_SWEEP.items())
@pytest.mark.parametrize("step_mode", ["fixed", "auto"])
def test_search60_nightly_profiles_are_exact_v3_overrides(stem, drops, step_mode):
    name = f"experiment/pruning_v3/{stem}_{step_mode}"
    cfg = schema.compose_config(name)
    assert schema.validate_config(cfg) == 150
    adaptive = cfg.training_arguments.adaptive_lambda
    assert (adaptive.soft_drop, adaptive.hard_drop) == drops
    assert adaptive.update_every_search_epochs == 1
    assert adaptive.log_step == ("auto" if step_mode == "auto" else math.log(2.0))
    assert cfg.model.lambda_coef == cfg.optimizer.lr == 1.0e-3
    assert cfg.scheduler._target_ == "torch.optim.lr_scheduler.CosineAnnealingLR"
    assert cfg.scheduler.T_max == 200 and cfg.scheduler.eta_min == 0.0
    assert cfg.one_shot.search_epochs == 60
    assert cfg.one_shot.search_scheduler_eta_min == pytest.approx(0.00066443)
    assert [(stage.kind, stage.epochs) for stage in cfg.accuracy_guided.stage_plan] == [
        ("search", 60), ("commit", 0), ("recovery", 90),
    ]
    assert cfg.run_history.log_channel_history is True
    assert cfg.training_arguments.evaluate_test is False


def test_handoff_ablation_rejects_a_less_informative_execution_order():
    cfg = schema.compose_config(schema.HANDOFF_CONFIG_NAME)
    cfg.one_shot.execution_order = "all_repeats_of_method_before_next_method"
    with pytest.raises(ValueError, match="finish every method"):
        schema.validate_config(cfg)


def test_adapter_does_not_mutate_or_relax_iterative_config():
    cfg = schema.compose_config()
    before = OmegaConf.to_container(cfg, resolve=True)
    with pytest.raises(ValueError, match="unknown=.*one_shot"):
        validate_config_v3(cfg)
    adapted = schema.to_v3_config(cfg)
    assert "one_shot" not in adapted and schema.validate_config(cfg) == validate_config_v3(adapted) == 150
    adapted.optimizer.lr = 123
    assert OmegaConf.to_container(cfg, resolve=True) == before
    iterative = compose_iterative()
    assert [s.epochs for s in iterative.accuracy_guided.stage_plan] == [20, 0, 15, 20, 0, 15, 20, 0, 60]


def test_resolved_budget_and_output_paths_are_explicit(tmp_path):
    cfg = schema.compose_config()
    report = schema.resolved_one_shot(cfg, check_inputs=False, output_root=tmp_path / "run")
    assert report["budget"] == {"shared_search_epochs_executed_once": 60,
        "physical_training_epochs_per_branch": 90, "per_branch_budget_including_shared_search": 150,
        "total_unique_training_epochs_both_branches": 240, "selection_does_not_rewind_consumed_budget": True}
    assert report["execution_policy"]["bn_calibration_batches"] == 0
    assert report["execution_policy"]["iterative_recovery_guard_executed"] is False
    assert report["execution_policy"]["shared_search_checkpoint_and_mask"] is True
    assert report["execution_policy"]["no_feasible_search"] == "stop_before_export_and_branch_training"
    assert report["execution_policy"]["inherited_optimizer_state"] == "mapped_adamw_moments_and_step"
    assert report["execution_policy"]["scratch_optimizer_state"] == "fresh"
    assert report["output_paths"]["export_only"] == str(tmp_path / "run/export_only")
    assert report["training_performed"] is False and report["evaluate_test"] is False
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("key,value", [
    ("one_shot.protocol", "other"), ("one_shot.search_epochs", 59), ("one_shot.final_epochs", 91),
    ("one_shot.search_epochs", True), ("one_shot.final_epochs", 0),
    ("one_shot.branches", ["inherited"]), ("one_shot.branches", ["scratch", "inherited"]),
    ("one_shot.branches", ["inherited", "scratch", "reset"]),
    ("one_shot.scratch_initialization", "inherit_bn"),
    ("one_shot.inherited_optimizer_state", "fresh"), ("one_shot.extra", True),
    ("accuracy_guided.guard.train_bn_calibration_batches", 1),
    ("training_arguments.batchnorm_recalibration.enabled", True),
    ("training_arguments.adaptive_lambda.enabled", False),
    ("training_arguments.adaptive_lambda.adaptive_log_step_enabled", True),
    ("accuracy_guided.target_remaining_ratio", 0.18),
    ("accuracy_guided.stage_plan.1.id", "other"), ("accuracy_guided.stage_plan.0.epochs", 20),
    ("training_arguments.evaluate_test", True), ("dataloaders.include_test", True), ("seed", 43),
])
def test_conflicts_are_rejected_before_training(key, value):
    cfg = schema.compose_config()
    OmegaConf.update(cfg, key, value, force_add=True)
    with pytest.raises(ValueError):
        schema.validate_config(cfg)


def test_other_full_schedule_rejected_but_short_marked_fixture_supported():
    cfg = schema.compose_config()
    cfg.one_shot.search_epochs, cfg.one_shot.final_epochs = 30, 120
    cfg.accuracy_guided.stage_plan[0].epochs, cfg.accuracy_guided.stage_plan[2].epochs = 30, 120
    with pytest.raises(ValueError, match="only the authorized full"):
        schema.validate_config(cfg)
    cfg.accuracy_guided.smoke = True
    cfg.one_shot.search_epochs, cfg.one_shot.final_epochs = 3, 2
    cfg.accuracy_guided.stage_plan[0].epochs, cfg.accuracy_guided.stage_plan[2].epochs = 3, 2
    cfg.accuracy_guided.total_epochs = cfg.training_arguments.num_epochs = 5
    assert schema.validate_config(cfg) == 5
    assert schema.resolved_one_shot(cfg, check_inputs=False)["budget"]["total_unique_training_epochs_both_branches"] == 7


@pytest.mark.parametrize("name", ["../accuracy_guided_gates_v3", "/tmp/config", "a/../../b", "a.yaml/b", "a//b"])
def test_config_name_traversal_and_absolute_paths_are_rejected(name):
    with pytest.raises(ValueError, match="relative Hydra"):
        schema.compose_config(name)


def test_input_validation_uses_stripped_shared_schema(tmp_path, monkeypatch):
    cfg = schema.compose_config()
    for key in ("history_path", "state_path", "config_path"):
        path = tmp_path / key
        path.write_text("{}")
        cfg.accuracy_guided.reference[key] = str(path)
    initializer = tmp_path / "initializer.pt"
    initializer.touch()
    cfg.accuracy_guided.initializer.path = str(initializer)
    seen = []
    def inspect(adapted):
        assert "one_shot" not in adapted
        assert validate_config_v3(adapted) == 150
        seen.append(adapted)
        return {"status": "ready", "verified": "shared_validator"}
    monkeypatch.setattr(schema, "_validate_v3_inputs", inspect)
    assert schema.validate_inputs(cfg) == {"status": "ready", "verified": "shared_validator"}
    assert len(seen) == 1 and "one_shot" in cfg


def test_dry_run_has_no_runtime_data_cuda_or_output_side_effects(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA was queried"))
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CUDA initialized"))
    from torchvision.datasets import CIFAR10
    monkeypatch.setattr(CIFAR10, "__init__", lambda *a, **kw: pytest.fail("CIFAR constructed"))
    original_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        assert name != "net_complexity.training.one_shot_pruning", "training runtime imported by preview"
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    report = launcher.main(["--config-name", schema.CONFIG_NAME, "--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["protocol"] == report["protocol"] == schema.PROTOCOL
    assert report["inputs"]["status"] == "blocked_missing_inputs"
    assert len(report["inputs"]["missing_inputs"]) == 4
    assert report["output_paths"]["root"] == str(tmp_path / "outputs/runs" / schema.OUTPUT_NAME)
    assert not (tmp_path / "outputs").exists()


def test_launch_with_missing_reference_never_creates_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    original_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        assert name != "net_complexity.training.one_shot_pruning", "missing inputs reached runtime"
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried before inputs exist"))
    source = tmp_path / "relocated dense reference"
    with pytest.raises(SystemExit) as error:
        launcher.main(["--dense-source", str(source), "--output", str(tmp_path / "run")])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "--dense-source" in stderr and "Traceback" not in stderr
    for key, value in schema.dense_source_paths(source).items():
        assert key in stderr and value in stderr
    assert not (tmp_path / "run").exists()
    assert not source.exists()


def test_existing_output_refused_before_input_validation(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(launcher, "validate_inputs", lambda *a: pytest.fail("existing output reached preflight"))
    with pytest.raises(FileExistsError, match="Refusing existing"):
        launcher.main(["--output", str(output)])
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("layout", ["parent", "named_job", "renamed_job"])
def test_dense_source_resolves_explicit_layout_without_creating_files(tmp_path, monkeypatch, layout):
    monkeypatch.chdir(tmp_path)
    parent = tmp_path / "dense source with spaces"
    job = parent / ("saved dense job" if layout == "renamed_job" else "J1_dense_control")
    if layout == "renamed_job":
        job.mkdir(parents=True)
        (job / "pilot_state.json").write_text("{}")
    source = parent if layout == "parent" else job
    result = schema.dense_source_paths(source.relative_to(tmp_path))
    assert result == {
        "history_path": str(job / "global_history.csv"),
        "state_path": str(job / "pilot_state.json"),
        "config_path": str(job / "resolved_config.yaml"),
        "initializer_path": str(parent / "shared_random_seed42.pt"),
    }
    assert not (parent / "shared_random_seed42.pt").exists()
    if layout == "renamed_job":
        assert list(job.iterdir()) == [job / "pilot_state.json"]
    else:
        assert not parent.exists()


def test_dense_source_expands_home_and_preserves_explicit_override_priority(tmp_path):
    source = f"~/nonexistent one shot reference {tmp_path.name}"
    expanded_source = Path(source).expanduser()
    overridden = tmp_path / "specific validated initializer.pt"
    cfg = schema.compose_config(dense_source=source, overrides=[
        f"accuracy_guided.initializer.path={json.dumps(str(overridden))}"])
    assert cfg.accuracy_guided.reference.history_path == str(expanded_source / "J1_dense_control/global_history.csv")
    assert cfg.accuracy_guided.initializer.path == str(overridden)
    assert schema.validate_config(cfg) == 150
    assert not expanded_source.exists()


@pytest.mark.parametrize("standard_history,standard_config,fallback_history,fallback_config", [
    (False, False, True, True),
    (True, True, True, True),
    (True, False, True, True),
    (False, True, True, True),
    (False, False, True, False),
    (False, False, False, True),
])
@pytest.mark.parametrize("direct_job", [False, True])
def test_historical_reference_bundle_is_selected_as_a_complete_pair(
        tmp_path, standard_history, standard_config, fallback_history, fallback_config, direct_job):
    job = tmp_path / "J1_dense_control"
    job.mkdir()
    for exists, path in (
            (standard_history, job / "global_history.csv"),
            (standard_config, job / "resolved_config.yaml"),
            (fallback_history, tmp_path / "adaptive_reference_history.csv"),
            (fallback_config, tmp_path / "J1_dense_control_resolved.yaml")):
        if exists:
            path.touch()
    result = schema.dense_source_paths(job if direct_job else tmp_path)
    use_fallback = not standard_history and not standard_config and fallback_history and fallback_config
    assert result["history_path"] == str(tmp_path / "adaptive_reference_history.csv" if use_fallback else job / "global_history.csv")
    assert result["config_path"] == str(tmp_path / "J1_dense_control_resolved.yaml" if use_fallback else job / "resolved_config.yaml")
    assert result["state_path"] == str(job / "pilot_state.json")
    assert result["initializer_path"] == str(tmp_path / "shared_random_seed42.pt")


def _relocated_synthetic_inputs(directory, *, historical_bundle=False):
    """Move real zero-epoch fixture artifacts into the user-facing source layout."""
    from net_complexity.training.pruning_synthetic import make_synthetic_config
    old = directory / "original fixture"
    cfg = make_synthetic_config(old)
    one_shot = schema.compose_config()
    OmegaConf.update(cfg, "one_shot", OmegaConf.to_container(one_shot.one_shot), force_add=True)
    cfg.one_shot.search_epochs, cfg.one_shot.final_epochs = 3, 2
    cfg.accuracy_guided.total_epochs = cfg.training_arguments.num_epochs = 5
    cfg.accuracy_guided.guard.train_bn_calibration_batches = 0
    cfg.accuracy_guided.stage_plan = one_shot.accuracy_guided.stage_plan
    cfg.accuracy_guided.stage_plan[0].epochs, cfg.accuracy_guided.stage_plan[2].epochs = 3, 2
    source = directory / "relocated source with spaces"
    job = source / "J1_dense_control"
    job.mkdir(parents=True)
    if historical_bundle:
        (source / "adaptive_reference_history.csv").touch()
        (source / "J1_dense_control_resolved.yaml").touch()
    paths = schema.dense_source_paths(source)
    for key, old_path in dict(cfg.accuracy_guided.reference).items():
        shutil.move(old_path, paths[key])
        cfg.accuracy_guided.reference[key] = paths[key]
    shutil.move(cfg.accuracy_guided.initializer.path, paths["initializer_path"])
    cfg.accuracy_guided.initializer.path = paths["initializer_path"]
    old.rmdir()
    return cfg, source, paths


@pytest.mark.parametrize("historical_bundle", [False, True])
def test_relocated_dense_source_keeps_strict_input_checks_and_never_loads_dense_weights(
        tmp_path, monkeypatch, historical_bundle):
    cfg, source, paths = _relocated_synthetic_inputs(tmp_path, historical_bundle=historical_bundle)
    for name in ("selected_checkpoint.pt", "deployment.pt", "best.pt"):
        (source / "J1_dense_control" / name).write_bytes(b"dense weights must never be loaded")
    actual_load = torch.load
    loaded = []
    def checked_load(path, *args, **kwargs):
        assert Path(path).resolve() == Path(paths["initializer_path"])
        loaded.append(Path(path).resolve())
        return actual_load(path, *args, **kwargs)
    monkeypatch.setattr(torch, "load", checked_load)
    result = schema.validate_inputs(cfg)
    assert result["status"] == "ready"
    assert result["paths"] == paths
    assert result["initializer_trained_epochs"] == 0
    assert result["reference_kind"] == "synthetic_programmed_feedback"
    assert loaded == [Path(paths["initializer_path"])]


@pytest.mark.parametrize("invalid,match", [
    ("trained", "zero-epoch"),
    ("tensor_hash", "tensor hash differs"),
    ("reference_hash", "reference zero-epoch initializer differs"),
])
def test_relocated_source_does_not_relax_initializer_provenance(tmp_path, invalid, match):
    cfg, _, paths = _relocated_synthetic_inputs(tmp_path)
    if invalid == "reference_hash":
        state = json.loads(Path(paths["state_path"]).read_text())
        state["common_init_hash"] = "0" * 64
        Path(paths["state_path"]).write_text(json.dumps(state))
    else:
        initial = torch.load(paths["initializer_path"], map_location="cpu", weights_only=True)
        if invalid == "trained":
            initial["trained_epochs"] = 150
        else:
            initial["model_state_hash"] = "0" * 64
        torch.save(initial, paths["initializer_path"])
    with pytest.raises(ValueError, match=match):
        schema.validate_inputs(cfg)


def test_dense_source_dry_run_reports_resolved_missing_files_without_side_effects(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "reference parent"
    result = launcher.main(["--dense-source", str(source), "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert result["inputs"] == report["inputs"]
    assert report["inputs"]["status"] == "blocked_missing_inputs"
    assert report["inputs"]["paths"] == schema.dense_source_paths(source)
    assert set(report["inputs"]["missing_inputs"]) == set(report["inputs"]["paths"])
    assert report["training_performed"] is False
    assert not source.exists() and not (tmp_path / "outputs").exists()
