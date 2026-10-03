import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest
import torch

from net_complexity.training.accuracy_guided_config import (
    compose_config, inspect_inputs, resolved_v3, validate_config_v3, validate_inputs,
)
from net_complexity.training.pruning_measurement import state_hash, write_json


def test_full_profile_preserves_current_plan_and_runtime_contract():
    cfg = compose_config()
    assert validate_config_v3(cfg) == 150
    assert [(s.kind, s.epochs) for s in cfg.accuracy_guided.stage_plan] == [
        ("search", 20), ("commit", 0), ("recovery", 15),
        ("search", 20), ("commit", 0), ("recovery", 15),
        ("search", 20), ("commit", 0), ("recovery", 60)]
    assert cfg.model.backbone.resnet_block.gate_internal_width is True
    assert cfg.model.backbone.resnet_block.gate_output is False
    assert cfg.accuracy_guided.eligibility.min_keep_ratio == 0.5
    assert cfg.model.backbone.resnet_block.gate_threshold == 0.5
    assert cfg.optimizer.lr == 0.001 and cfg.optimizer.gate_weight_decay_scale == 0
    assert cfg.seed == 42 and cfg.dataloaders.batch_size == 128
    assert cfg.training_arguments.adaptive_lambda.initial_search_warmup == 10


def test_smoke_is_separately_marked_short_profile():
    cfg = compose_config("accuracy_guided_gates_v3_smoke")
    assert validate_config_v3(cfg) == 8
    assert cfg.accuracy_guided.smoke and cfg.device == "cpu"
    report = resolved_v3(cfg)
    assert report["inputs"]["status"] == "blocked_missing_inputs"
    assert "fixture" in report["inputs"]["note"]


@pytest.mark.parametrize("key,value", [
    ("accuracy_guided.total_epochs", 149), ("accuracy_guided.stage_plan.0.epochs", 19),
    ("accuracy_guided.stage_plan.1.epochs", 1), ("accuracy_guided.stage_plan.1.commit_allowed", False),
    ("accuracy_guided.stage_plan.0.kind", "recovery"), ("accuracy_guided.stage_plan.3.id", "search_0"),
    ("accuracy_guided.stage_plan.0.restart_policy", "carry_optimizer"),
    ("accuracy_guided.stage_plan.0.selected_checkpoint", "trained.pt"),
    ("accuracy_guided.max_param_fraction", 0.18), ("accuracy_guided.target_remaining_ratio", 0.5),
    ("accuracy_guided.eligibility.target_prune_rate", 0.2), ("accuracy_guided.drop_mode", "param_budget"),
    ("accuracy_guided.carry_policy", "reset_open"), ("accuracy_guided.normalization", "enabled_channels"),
    ("accuracy_guided.scaling_contract", "global_survivor_ratio"),
    ("accuracy_guided.guard.max_recovered_accuracy_drop", 0.01),
    ("accuracy_guided.selection.policy", "best_accuracy"), ("accuracy_guided.unknown", True),
    ("training_arguments.adaptive_lambda.enabled", False),
    ("training_arguments.adaptive_lambda.control_mode", "legacy"),
    ("training_arguments.adaptive_lambda.adaptive_log_step_enabled", True),
    ("training_arguments.adaptive_lambda.prune_rate_low_per_epoch", 0.01),
    ("training_arguments.adaptive_lambda.recovery", {"enabled": True}),
    ("training_arguments.adaptive_lambda.alpha_init", float("nan")),
    ("training_arguments.adaptive_lambda.alpha_min", 1),
    ("training_arguments.adaptive_lambda.soft_drop", 0.05),
    ("training_arguments.adaptive_lambda.reentry_samples", 1),
    ("training_arguments.evaluate_test", True), ("dataloaders.include_test", True),
    ("model.backbone.resnet_block.regularization_normalization", "enabled_channels"),
    ("model.backbone.resnet_block.force_ones_mask", True),
    ("model.entropy_regularization_coef", 1), ("optimizer.gate_weight_decay_scale", 1),
    ("cyclic_channel_pruning", {"drop_mode": "param_budget"}),
])
def test_conflicts_unknown_fields_and_invalid_schedules_fail(key, value):
    cfg = compose_config()
    OmegaConf.update(cfg, key, value, force_add=True)
    with pytest.raises(ValueError):
        validate_config_v3(cfg)


def test_no_seed_job_name_whitelist():
    cfg = compose_config()
    cfg.seed = 123
    cfg.dataloaders.seed = cfg.dataloaders.loader_seed = cfg.training_arguments.audit_data_seed = 123
    assert validate_config_v3(cfg) == 150


@pytest.fixture
def input_config(tmp_path):
    cfg = compose_config()
    files = {key: str(tmp_path / name) for key, name in
             (("history_path", "reference.csv"), ("state_path", "state.json"), ("config_path", "source.yaml"))}
    cfg.accuracy_guided.reference = files
    cfg.accuracy_guided.initializer.path = str(tmp_path / "init.pt")
    weights = {"weight": torch.tensor([1.0])}
    common_hash = state_hash(weights)
    torch.save({"trained_epochs": 0, "seed": 42, "model_state_dict": weights,
                "model_state_hash": common_hash}, cfg.accuracy_guided.initializer.path)
    Path(files["history_path"]).write_text("global_epoch,valid_accuracy\n" +
                                        "".join(f"{e},0.9\n" for e in range(1, 151)))
    write_json(Path(files["state_path"]), {"status": "completed", "accepted_mask": {}, "test_evaluated": False,
               "global_epochs_completed": 150, "total_epochs_allocated": 150, "seed": 42,
               "validation": {"example_count": 5000}, "common_init_hash": common_hash})
    OmegaConf.save(cfg, files["config_path"])
    return cfg


def test_compatible_inputs_verified_read_only(input_config):
    cfg = input_config
    paths = [Path(p) for p in cfg.accuracy_guided.reference.values()] + [Path(cfg.accuracy_guided.initializer.path)]
    before = {p: p.read_bytes() for p in paths}
    report = validate_inputs(cfg)
    assert report["status"] == "ready" and report["reference_epochs"] == 150
    assert report["initializer_trained_epochs"] == 0
    assert {p: p.read_bytes() for p in paths} == before


def test_reference_recovery_exceptions_are_explicit_and_narrow(input_config):
    input_config.optimizer.lr = 0.002
    input_config.scheduler.eta_min = 0.00025
    OmegaConf.update(input_config, "model.criterion.label_smoothing", 0.10, force_add=True)
    with pytest.raises(ValueError, match="reference compatibility differs: optimizer"):
        validate_inputs(input_config)

    report = validate_inputs(
        input_config,
        allow_optimizer_lr_difference=True,
        allow_scheduler_eta_min_difference=True,
        allow_label_smoothing_difference=True,
    )
    assert report["status"] == "ready"
    assert report["reference_compatibility_allowed_differences"] == [
        "optimizer.lr", "scheduler.eta_min", "model.criterion.label_smoothing",
    ]

    input_config.optimizer.weight_decay = 0.001
    with pytest.raises(ValueError, match="reference compatibility differs: optimizer"):
        validate_inputs(
            input_config,
            allow_optimizer_lr_difference=True,
            allow_scheduler_eta_min_difference=True,
            allow_label_smoothing_difference=True,
        )

    input_config.optimizer.weight_decay = 0.0005
    input_config.scheduler.T_max = 199
    with pytest.raises(ValueError, match="reference compatibility differs: scheduler"):
        validate_inputs(
            input_config,
            allow_optimizer_lr_difference=True,
            allow_scheduler_eta_min_difference=True,
            allow_label_smoothing_difference=True,
        )

    input_config.scheduler.T_max = 200
    OmegaConf.update(input_config, "model.criterion.reduction", "sum", force_add=True)
    with pytest.raises(ValueError, match="reference compatibility differs: model.criterion"):
        validate_inputs(
            input_config,
            allow_optimizer_lr_difference=True,
            allow_scheduler_eta_min_difference=True,
            allow_label_smoothing_difference=True,
        )


@pytest.mark.parametrize("mutation", ["missing", "nan", "short_curve", "trained_init", "source_split", "test_reference"])
def test_bad_reference_or_initializer_is_never_repaired_or_trained(input_config, mutation):
    cfg = input_config
    r = cfg.accuracy_guided.reference
    if mutation == "missing":
        Path(r.history_path).unlink()
        assert inspect_inputs(cfg)["status"] == "blocked_missing_inputs"
    elif mutation in ("nan", "short_curve"):
        Path(r.history_path).write_text("global_epoch,valid_accuracy\n1," + ("nan" if mutation == "nan" else "0.9") + "\n")
    elif mutation == "trained_init":
        initial = torch.load(cfg.accuracy_guided.initializer.path, weights_only=True)
        initial["trained_epochs"] = 150
        torch.save(initial, cfg.accuracy_guided.initializer.path)
    elif mutation == "source_split":
        source = OmegaConf.load(r.config_path)
        source.dataloaders.train_val_ratio = [0.8, 0.2]
        OmegaConf.save(source, r.config_path)
    else:
        state = json.loads(Path(r.state_path).read_text())
        state["test_evaluated"] = True
        write_json(Path(r.state_path), state)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_inputs(cfg)


def test_dry_run_never_calls_training_cuda_or_dataset(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import launch_accuracy_guided_pruning as launcher
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried"))
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CUDA initialized"))
    from torchvision.datasets import CIFAR10
    monkeypatch.setattr(CIFAR10, "__init__", lambda *a, **kw: pytest.fail("dataset constructed"))
    output = tmp_path / "run"
    report = launcher.main(["--dry-run", "--output", str(output)])
    rendered = json.loads(capsys.readouterr().out)
    assert report["total_training_epochs"] == rendered["total_training_epochs"] == 150
    assert report["inputs"]["status"] == "blocked_missing_inputs"
    assert report["pruning"]["mandatory_quota"] is None
    assert report["training_performed"] is False and not output.exists()
