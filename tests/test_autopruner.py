from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from net_complexity.metrics.autopruner import AutoPrunerComplexityMetric
from net_complexity.models.autopruner import (
    AUTHOR_ALPHA_START_RESNET,
    AUTHOR_ALPHA_STOP_RESNET,
    AUTHOR_ALPHA_UPDATE_INTERVAL,
    AUTHOR_CODE_WINDOW_SIZE,
    AUTHOR_FINAL_FINE_TUNE_EPOCHS,
    AUTHOR_FINAL_WEIGHT_DECAY,
    AUTHOR_INITIAL_PRUNING_THRESHOLD,
    AUTHOR_INITIAL_REGULARIZATION,
    AUTHOR_MOMENTUM,
    AUTHOR_PRUNING_EPOCHS_PER_STAGE,
    AUTHOR_PRUNING_LR,
    AUTHOR_PRUNING_WEIGHT_DECAY,
    AUTHOR_REGULARIZATION_SCALE,
    AutoPrunerLayer,
    AutoPrunerResNet,
    AutoPrunerWrapper,
    export_pruned_autopruner_backbone,
    get_autopruner_modules,
    load_autopruner_pretrained_backbone,
)
from net_complexity.training.engine import _assert_runtime_lambda_consistency
from net_complexity.training.run_history import RunHistory


REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_backbone(*, target_keep_ratio: float = 0.5) -> AutoPrunerResNet:
    return AutoPrunerResNet(
        layer_list=(1, 1, 1, 2),
        num_classes=3,
        input_size=8,
        stage_planes=(2, 4, 8, 16),
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
        target_keep_ratio=target_keep_ratio,
    )


def _set_alternating_hard_masks(model: torch.nn.Module) -> None:
    for module in get_autopruner_modules(model).values():
        module.binary_mask.copy_(
            (torch.arange(module.channels, device=module.binary_mask.device) % 2 == 0)
            .to(module.binary_mask)
        )
        module.phase.fill_(module.PHASE_HARD)
        module.coder.requires_grad_(False)


def test_selector_uses_one_deterministic_code_for_the_whole_batch():
    layer = AutoPrunerLayer(
        channels=2,
        activation_size=4,
        stage_index=0,
        target_keep_ratio=0.5,
        max_pool_kernel=2,
    )
    layer.start_soft_pruning()
    layer.train()
    with torch.no_grad():
        layer.coder.weight.zero_()
        layer.coder.bias.copy_(torch.tensor([0.0, math.log(3.0)]))

    input = torch.randn(3, 2, 4, 4)
    output = layer(input)
    expected_code = torch.tensor([0.5, 0.75])

    torch.testing.assert_close(layer.current_code(), expected_code)
    torch.testing.assert_close(
        output,
        input * expected_code.view(1, 2, 1, 1),
    )
    torch.testing.assert_close(
        layer.regularization_error(),
        torch.tensor((0.625 - 0.5) ** 2),
    )
    layer.weighted_regularization().backward()
    assert layer.coder.bias.grad is not None


def test_selector_uses_the_author_coder_initialization():
    torch.manual_seed(7)
    layer = AutoPrunerLayer(
        channels=32,
        activation_size=8,
        stage_index=0,
        max_pool_kernel=2,
    )
    expected_std = 10.0 * math.sqrt(2.0 / (32 * 4 * 4))
    actual_std = float(layer.coder.weight.detach().std(unbiased=True).item())
    assert actual_std == pytest.approx(expected_std, rel=0.04)


def test_consensus_updates_binary_mask_adaptive_lambda_and_alpha_boost():
    layer = AutoPrunerLayer(
        channels=4,
        activation_size=2,
        stage_index=0,
        target_keep_ratio=0.5,
        code_window_size=AUTHOR_CODE_WINDOW_SIZE,
    )
    layer.start_soft_pruning()
    balanced_codes = torch.tensor([[0.9, 0.6, 0.4, 0.1]]).repeat(20, 1)
    layer._update_consensus(balanced_codes, allow_convergence_boost=False)
    torch.testing.assert_close(layer.binary_mask, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert float(layer.adaptive_regularization.item()) == 0.0

    closed_codes = torch.full((20, 4), 0.1)
    layer._update_consensus(closed_codes, allow_convergence_boost=False)
    assert float(layer.adaptive_regularization.item()) == pytest.approx(
        AUTHOR_REGULARIZATION_SCALE * 0.5
    )
    assert float(layer.alpha_boost.item()) == 1.0
    assert float(layer.pruning_threshold.item()) == pytest.approx(0.9)


def test_finalize_discards_a_trailing_partial_consensus_window():
    layer = AutoPrunerLayer(
        channels=4,
        activation_size=2,
        stage_index=0,
        target_keep_ratio=0.5,
    )
    layer.start_soft_pruning()
    layer._update_consensus(
        torch.tensor([[0.9, 0.8, 0.1, 0.2]]).repeat(20, 1),
        allow_convergence_boost=False,
    )
    layer.code_window[0].copy_(torch.tensor([0.1, 0.1, 0.9, 0.9]))
    layer.window_count.fill_(1)

    layer.finalize()

    torch.testing.assert_close(layer.binary_mask, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert int(layer.window_count.item()) == 0


def test_resnet_placement_and_author_epoch_schedule():
    backbone = _tiny_backbone()
    modules = list(get_autopruner_modules(backbone).values())
    assert len(modules) == 8
    assert [module.stage_index for module in modules].count(0) == 2
    assert [module.stage_index for module in modules].count(1) == 2
    assert [module.stage_index for module in modules].count(2) == 2
    assert [module.stage_index for module in modules].count(3) == 2

    wrapper = AutoPrunerWrapper(backbone)
    assert wrapper.num_pruning_phases == 4
    assert wrapper.expected_num_epochs == (
        4 * AUTHOR_PRUNING_EPOCHS_PER_STAGE + AUTHOR_FINAL_FINE_TUNE_EPOCHS
    )
    assert {module.phase_name for module in wrapper._modules_for_stage(0)} == {"soft"}
    assert {module.phase_name for module in wrapper._modules_for_stage(1)} == {"open"}


def test_image_net_final_group_bypasses_spatial_max_pooling():
    backbone = AutoPrunerResNet(
        layer_list=(1, 1, 1, 2),
        num_classes=3,
        input_size=224,
        stage_planes=(2, 4, 8, 16),
    )
    final_modules = [
        module
        for module in get_autopruner_modules(backbone).values()
        if module.stage_index == 3
    ]
    assert [module.activation_size for module in final_modules] == [14, 7]
    assert [module.max_pool_kernel for module in final_modules] == [1, 1]


def test_wrapper_objective_backpropagates_through_active_coders():
    wrapper = AutoPrunerWrapper(_tiny_backbone(), select_best_per_stage=False)
    optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.2, momentum=0.9)
    wrapper.on_train_epoch_start(epoch=1, optimizer=optimizer, batches_per_epoch=5)
    wrapper.train()

    output = wrapper(torch.randn(2, 3, 8, 8), torch.tensor([0, 1]))
    output.loss.backward()

    assert output.loss.item() == pytest.approx(
        (output.ce_loss + output.reg_loss).item()
    )
    assert float(optimizer.param_groups[0]["lr"]) == AUTHOR_PRUNING_LR
    assert any(
        module.coder.weight.grad is not None
        for module in wrapper._modules_for_stage(0)
    )
    assert all(
        module.coder.weight.grad is None
        for module in wrapper._modules_for_stage(1)
    )


def test_stage_transition_restores_best_validation_state_and_resets_sgd():
    wrapper = AutoPrunerWrapper(
        _tiny_backbone(),
        pruning_epochs_per_stage=1,
        final_fine_tune_epochs=1,
        select_best_per_stage=True,
    )
    optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.2, momentum=0.9)
    wrapper.on_train_epoch_start(epoch=1, optimizer=optimizer, batches_per_epoch=2)
    parameter = wrapper.backbone.conv1.weight
    with torch.no_grad():
        parameter.fill_(2.0)
    wrapper.on_validation_epoch_end(
        epoch=1,
        valid_metrics={"valid_accuracy": 0.8},
        optimizer=optimizer,
    )
    optimizer.state[parameter]["momentum_buffer"] = torch.ones_like(parameter)
    with torch.no_grad():
        parameter.fill_(9.0)

    wrapper.on_train_epoch_start(epoch=2, optimizer=optimizer, batches_per_epoch=2)

    torch.testing.assert_close(parameter, torch.full_like(parameter, 2.0))
    assert optimizer.state == {}
    assert int(wrapper.current_stage_position.item()) == 1
    assert {module.phase_name for module in wrapper._modules_for_stage(0)} == {"hard"}
    assert {module.phase_name for module in wrapper._modules_for_stage(1)} == {"soft"}


def test_physical_export_is_equivalent_and_has_fewer_parameters():
    backbone = _tiny_backbone()
    _set_alternating_hard_masks(backbone)
    backbone.eval()
    input = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        expected = backbone(input)
    exported = export_pruned_autopruner_backbone(backbone)
    exported.eval()
    with torch.no_grad():
        actual = exported(input)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert not get_autopruner_modules(exported)
    assert sum(p.numel() for p in exported.parameters()) < sum(
        p.numel()
        for p in export_pruned_autopruner_backbone(
            backbone,
            use_binary_masks=False,
        ).parameters()
    )


def test_complexity_metric_reports_real_export_reductions():
    backbone = _tiny_backbone()
    _set_alternating_hard_masks(backbone)
    metric = AutoPrunerComplexityMetric()
    metric.update(torch.randn(2, 3, 8, 8), None, None, backbone)

    values = metric.compute()

    assert values["autopruner_num_selectors"] == 8
    assert values["autopruner_channel_keep_ratio"] == 0.5
    assert values["autopruner_pruned_deployment_params"] < values[
        "autopruner_dense_deployment_params"
    ]
    assert values["autopruner_pruned_macs_per_image"] < values[
        "autopruner_dense_macs_per_image"
    ]


def test_best_checkpoint_hook_writes_physical_deployment_artifact(tmp_path):
    wrapper = AutoPrunerWrapper(_tiny_backbone(), select_best_per_stage=False)
    _set_alternating_hard_masks(wrapper)

    info = wrapper.on_best_checkpoint_loaded(run_dir=tmp_path)
    checkpoint = tmp_path / "checkpoints" / "autopruner_pruned.pt"

    assert info["autopruner_pruned_checkpoint"] == str(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert "model_state_dict" in payload
    assert "pruning_spec" in payload


def test_project_checkpoint_conv_biases_are_folded_into_batchnorm(tmp_path):
    backbone = _tiny_backbone()
    state = {
        name: tensor.detach().clone()
        for name, tensor in backbone.state_dict().items()
    }
    prefix = "layer1.0"
    state[f"{prefix}.conv1.bias"] = torch.full((2,), 0.2)
    state[f"{prefix}.batch_norm1.running_mean"] = torch.full((2,), 0.3)
    checkpoint = tmp_path / "baseline.pt"
    torch.save({"model_state_dict": state}, checkpoint)

    info = load_autopruner_pretrained_backbone(backbone, checkpoint)

    torch.testing.assert_close(
        backbone.layer1[0].batch_norm1.running_mean,
        torch.full((2,), 0.1),
    )
    assert f"{prefix}.conv1.bias" in info["folded_conv_biases"]


def test_author_configs_pin_resnet_hyperparameters_and_ratio_grid():
    model_cfg = OmegaConf.load(REPO_ROOT / "configs/model/autopruner_resnet50.yaml")
    method_cfg = OmegaConf.load(
        REPO_ROOT / "configs/method/autopruner_author_resnet50.yaml"
    )
    optimizer_cfg = OmegaConf.load(
        REPO_ROOT / "configs/optimizer/sgd_autopruner_author.yaml"
    )
    train_cfg = OmegaConf.load(REPO_ROOT / "configs/train/autopruner_author.yaml")
    tuning_cfg = OmegaConf.load(
        REPO_ROOT / "configs/tuning/autopruner_keep_ratio_grid.yaml"
    )

    assert model_cfg.model.require_pretrained is True
    assert model_cfg.model.backbone.alpha_start == AUTHOR_ALPHA_START_RESNET
    assert model_cfg.model.backbone.alpha_stop == AUTHOR_ALPHA_STOP_RESNET
    assert model_cfg.model.backbone.code_window_size == AUTHOR_CODE_WINDOW_SIZE
    assert method_cfg.model.pruning_epochs_per_stage == AUTHOR_PRUNING_EPOCHS_PER_STAGE
    assert method_cfg.model.final_fine_tune_epochs == AUTHOR_FINAL_FINE_TUNE_EPOCHS
    assert method_cfg.model.alpha_update_interval == AUTHOR_ALPHA_UPDATE_INTERVAL
    assert method_cfg.model.pruning_lr == AUTHOR_PRUNING_LR
    assert method_cfg.model.pruning_weight_decay == AUTHOR_PRUNING_WEIGHT_DECAY
    assert method_cfg.model.final_weight_decay == AUTHOR_FINAL_WEIGHT_DECAY
    assert method_cfg.model.select_best_per_stage is True
    assert optimizer_cfg.optimizer.lr == AUTHOR_PRUNING_LR
    assert optimizer_cfg.optimizer.momentum == AUTHOR_MOMENTUM
    assert optimizer_cfg.optimizer.weight_decay == AUTHOR_PRUNING_WEIGHT_DECAY
    assert AUTHOR_INITIAL_REGULARIZATION == 10.0
    assert AUTHOR_REGULARIZATION_SCALE == 100.0
    assert AUTHOR_INITIAL_PRUNING_THRESHOLD == 0.95
    assert train_cfg.training_arguments.num_epochs == 62
    assert tuning_cfg.tuning.search_space[
        "model.backbone.target_keep_ratio"
    ].choices == [0.5, 0.3]


def test_autopruner_runtime_does_not_require_unrelated_global_lambda(tmp_path):
    model = AutoPrunerWrapper(_tiny_backbone(), select_best_per_stage=False)
    config = OmegaConf.create(
        {
            "model": {},
            "mlflow": {"run_name": "autopruner_without_global_lambda"},
        }
    )
    run_history = SimpleNamespace(
        run_name="autopruner_without_global_lambda",
        run_dir=tmp_path,
    )

    snapshot = _assert_runtime_lambda_consistency(
        config,
        model,
        run_history,
        progress_context={"grid_params": {}, "optuna_trial_params": {}},
    )

    assert snapshot["cfg_model_lambda_coef"] is None
    assert snapshot["model_lambda_coef"] is None


def test_v100_overnight_series_uses_only_author_ratios_and_four_seeds():
    cfg = OmegaConf.load(
        REPO_ROOT
        / "configs/tuning/autopruner_author_resnet50_v100_11h.yaml"
    )

    assert cfg.dataloaders.batch_size == 256
    assert cfg.dataloaders.num_workers == 8
    assert cfg.dataloaders.pin_memory is True
    assert cfg.tuning.mode == "grid"
    assert cfg.tuning.n_jobs == 1
    assert cfg.tuning.repeats_per_trial == 4
    assert cfg.tuning.seed_base == 42
    assert cfg.tuning.seed_stride == 1
    assert cfg.tuning.timeout == 37800
    assert cfg.tuning.points_in_order is True
    assert [
        point["model.backbone.target_keep_ratio"]
        for point in cfg.tuning.points
    ] == [0.5, 0.3]
    assert all(
        point["mlflow.tags.author_hyperparameters"] == "true"
        for point in cfg.tuning.points
    )
    assert all(
        int(point["mlflow.tags.repeats_per_ratio"]) == 4
        for point in cfg.tuning.points
    )
    cfg.tuning.repeats_per_trial = 3
    assert all(
        int(point["mlflow.tags.repeats_per_ratio"]) == 3
        for point in cfg.tuning.points
    )


def test_autopruner_experiment_enables_mlflow_tracking():
    cfg = OmegaConf.load(
        REPO_ROOT / "configs/experiment/autopruner_resnet50_cifar10.yaml"
    )

    assert cfg.mlflow.enabled is True


def test_run_history_can_delay_best_checkpoint_selection_until_fine_tuning(tmp_path):
    config = OmegaConf.create(
        {
            "run_history": {
                "root_dir": str(tmp_path),
                "run_name": "autopruner_min_epoch",
                "monitor": "valid_accuracy",
                "mode": "max",
                "min_epoch": 33,
                "use_hydra_output_dir": False,
            }
        }
    )
    history = RunHistory(config)

    assert history.should_update_best(32, {"valid_accuracy": 0.99}) is False
    assert history.should_update_best(33, {"valid_accuracy": 0.80}) is True
    assert history.best_epoch == 33
