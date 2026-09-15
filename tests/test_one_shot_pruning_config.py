import builtins
import json
from pathlib import Path
import sys

from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytest
import torch

from net_complexity.training import one_shot_pruning_config as schema
from net_complexity.training.accuracy_guided_config import compose_config as compose_iterative, validate_config_v3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_one_shot_pruning as launcher


def test_only_authorized_full_plan_composes_with_shared_base_contract():
    cfg = schema.compose_config()
    assert schema.validate_config(cfg) == 150
    assert cfg.one_shot.protocol == "pruning_v3_one_shot_60_90"
    assert cfg.one_shot.branches == ["inherited", "scratch"]
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
    assert report["output_paths"]["export_only"] == str(tmp_path / "run/export_only")
    assert report["training_performed"] is False and report["evaluate_test"] is False
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("key,value", [
    ("one_shot.protocol", "other"), ("one_shot.search_epochs", 59), ("one_shot.final_epochs", 91),
    ("one_shot.search_epochs", True), ("one_shot.final_epochs", 0),
    ("one_shot.branches", ["inherited"]), ("one_shot.branches", ["scratch", "inherited"]),
    ("one_shot.branches", ["inherited", "scratch", "reset"]),
    ("one_shot.scratch_initialization", "inherit_bn"), ("one_shot.extra", True),
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


def test_input_validation_uses_stripped_shared_schema(monkeypatch):
    cfg = schema.compose_config()
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


def test_launch_with_missing_reference_never_creates_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="reference/initializer"):
        launcher.main(["--output", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


def test_existing_output_refused_before_input_validation(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(launcher, "validate_inputs", lambda *a: pytest.fail("existing output reached preflight"))
    with pytest.raises(FileExistsError, match="Refusing existing"):
        launcher.main(["--output", str(output)])
    assert list(output.iterdir()) == []
