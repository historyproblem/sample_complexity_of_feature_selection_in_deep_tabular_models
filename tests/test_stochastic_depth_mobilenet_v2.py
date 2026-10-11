from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from net_complexity.metrics.inspection import ComputeCollector
from net_complexity.metrics.stochastic_depth import (
    StochasticDepthActiveBlocksMetric,
    StochasticDepthFLOPsMetric,
)
from net_complexity.models.feature_selection import (
    ClassificationFeatureSelectionWrapper,
)
from net_complexity.models.mobilenet_v2 import InvertedResidual, MobileNetV2
from net_complexity.models.stochastic_depth import (
    HuangStochasticDepthInvertedResidual,
    StochasticDepthMobileNetV2,
    get_stochastic_depth_blocks,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_mobilenet_only_identity_shortcuts_are_stochastic():
    model = StochasticDepthMobileNetV2(num_classes=200, final_survival_probability=0.5)
    blocks = get_stochastic_depth_blocks(model)
    assert len(blocks) == 10
    assert list(blocks) == [
        "features.3",
        "features.5",
        "features.6",
        "features.8",
        "features.9",
        "features.10",
        "features.12",
        "features.13",
        "features.15",
        "features.16",
    ]
    assert [block.survival_probability for block in blocks.values()] == pytest.approx(
        [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    )
    assert sum(
        block.survival_probability for block in blocks.values()
    ) == pytest.approx(7.25)
    transitions = [
        block
        for block in model.features
        if isinstance(block, InvertedResidual) and not block.use_res_connect
    ]
    assert len(transitions) == 7
    assert all(
        not isinstance(block, HuangStochasticDepthInvertedResidual)
        for block in transitions
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 2480072


def test_mobile_uniform_matches_linear_expected_active_blocks():
    linear = StochasticDepthMobileNetV2(num_classes=200)
    uniform = StochasticDepthMobileNetV2(
        num_classes=200,
        survival_schedule="uniform",
        final_survival_probability=0.725,
    )
    linear_blocks = get_stochastic_depth_blocks(linear)
    uniform_blocks = get_stochastic_depth_blocks(uniform)
    assert list(uniform_blocks) == list(linear_blocks)
    assert uniform.stochastic_depth_survival_schedule == "uniform"
    assert uniform.stochastic_depth_survival_probabilities == pytest.approx(
        [0.725] * 10
    )
    assert [block.survival_probability for block in uniform_blocks.values()] == (
        pytest.approx([0.725] * 10)
    )
    assert sum(block.survival_probability for block in uniform_blocks.values()) == (
        pytest.approx(
            sum(block.survival_probability for block in linear_blocks.values())
        )
    )
    assert sum(parameter.numel() for parameter in uniform.parameters()) == 2480072


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"survival_schedule": "cosine"}, "survival_schedule"),
        (
            {"survival_schedule": "uniform", "final_survival_probability": -0.1},
            "final_survival_probability",
        ),
        (
            {"survival_schedule": "uniform", "final_survival_probability": 1.1},
            "final_survival_probability",
        ),
    ],
)
def test_mobile_rejects_invalid_survival_schedule_or_probability(kwargs, message):
    with pytest.raises(ValueError, match=message):
        StochasticDepthMobileNetV2(num_classes=200, **kwargs)


def test_closed_mobile_branch_skips_batchnorm_and_gradients():
    block = HuangStochasticDepthInvertedResidual(8, 8, 1, 6, survival_probability=0.0)
    block.train()
    x = torch.randn(3, 8, 8, 8, requires_grad=True)
    batch_norms = [
        module for module in block.modules() if isinstance(module, nn.BatchNorm2d)
    ]
    before = [module.num_batches_tracked.clone() for module in batch_norms]
    output = block(x)
    torch.testing.assert_close(output, x)
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    assert all(parameter.grad is None for parameter in block.branch.parameters())
    assert all(
        torch.equal(old, module.num_batches_tracked)
        for old, module in zip(before, batch_norms)
    )
    assert block.last_survival_mask.shape == torch.Size([])
    assert not block.last_residual_branch_active


def test_mobile_train_uses_unscaled_branch_and_one_batch_mask():
    block = HuangStochasticDepthInvertedResidual(8, 8, 1, 6, survival_probability=0.25)
    block.branch = nn.Identity()
    block.train()
    torch.manual_seed(42)
    x = torch.ones(5, 8, 4, 4)
    observed = set()
    for _ in range(32):
        output = block(x)
        assert block.last_survival_mask.shape == torch.Size([])
        expected = 2 * x if block.last_residual_branch_active else x
        torch.testing.assert_close(output, expected)
        observed.add(block.last_residual_branch_active)
    assert observed == {True, False}


def test_mobile_eval_scales_residual_without_sampling():
    block = HuangStochasticDepthInvertedResidual(8, 8, 1, 6, survival_probability=0.25)
    block.branch = nn.Identity()
    block.eval()
    x = torch.randn(2, 8, 4, 4)
    torch.testing.assert_close(block(x), 1.25 * x)
    torch.testing.assert_close(block(x), 1.25 * x)
    assert block.last_residual_branch_active


@pytest.mark.parametrize("prefix", ["", "backbone.", "module.backbone."])
@pytest.mark.parametrize("survival_schedule", ["linear", "uniform"])
def test_pretrained_mobile_checkpoint_loads_classifier_and_matches_dense_at_pl1(
    tmp_path, prefix, survival_schedule
):
    plain = MobileNetV2(num_classes=200).eval()
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": {
                prefix + key: value for key, value in plain.state_dict().items()
            }
        },
        checkpoint,
    )
    model = StochasticDepthMobileNetV2(
        num_classes=200,
        final_survival_probability=1.0,
        survival_schedule=survival_schedule,
        pretrained_checkpoint=checkpoint,
        require_pretrained=True,
    ).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(x), plain(x))
    assert model.pretrained_parameter_names == frozenset(dict(model.named_parameters()))
    assert torch.equal(model.classifier[1].weight, plain.classifier[1].weight)


def test_mobile_pretrained_rejects_missing_checkpoint_and_wrong_head(tmp_path):
    with pytest.raises(ValueError, match="pretrained_checkpoint"):
        StochasticDepthMobileNetV2(num_classes=200, require_pretrained=True)
    checkpoint = tmp_path / "wrong-head.pt"
    torch.save(MobileNetV2(num_classes=10).state_dict(), checkpoint)
    with pytest.raises(RuntimeError, match="size mismatch"):
        StochasticDepthMobileNetV2(num_classes=200, pretrained_checkpoint=checkpoint)


def test_mobile_wrapper_backward_and_frozen_eval():
    model = ClassificationFeatureSelectionWrapper(
        StochasticDepthMobileNetV2(num_classes=200),
        lambda_coef=0.0,
    ).train()
    x = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 199])
    output = model(x, labels)
    assert output.logits.shape == (2, 200)
    output.loss.backward()
    assert model.backbone.classifier[1].weight.grad is not None
    assert output.reg_loss.item() == 0.0
    model.eval()
    before = {
        name: value.clone()
        for name, value in model.named_buffers()
        if "running_" in name or "num_batches_tracked" in name
    }
    with torch.no_grad():
        first = model(x, labels)
        second = model(x, labels)
    torch.testing.assert_close(first.logits, second.logits)
    assert all(
        torch.equal(value, dict(model.named_buffers())[name])
        for name, value in before.items()
    )


def test_mobile_metrics_account_for_branch_skips_and_full_inference():
    model = StochasticDepthMobileNetV2(num_classes=200, final_survival_probability=0.0)
    blocks = get_stochastic_depth_blocks(model)
    for block in blocks.values():
        block.survival_probability = 0.0
    model.train()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        model(x)
    active = StochasticDepthActiveBlocksMetric()
    metric = StochasticDepthFLOPsMetric(sample_size=1)
    active.update(x, None, None, model)
    metric.update(x, None, None, model)
    values = metric.compute()
    assert values["stochastic_depth_deployment_params"] == 2480072
    assert active.compute()["stochastic_depth_actual_active_train_blocks"] == 0.0
    assert (
        values["stochastic_depth_expected_train_forward_flops_per_sample"]
        < values["stochastic_depth_dense_reference_forward_flops_per_sample"]
    )
    assert values[
        "stochastic_depth_actual_train_forward_flops_per_sample"
    ] == pytest.approx(
        values["stochastic_depth_expected_train_forward_flops_per_sample"]
    )
    assert values[
        "stochastic_depth_actual_train_forward_backward_flops_per_sample"
    ] == pytest.approx(
        3 * values["stochastic_depth_actual_train_forward_flops_per_sample"]
    )
    assert all(block.survival_probability == 0.0 for block in blocks.values())
    dense = MobileNetV2(num_classes=200)
    dense_compute, _ = ComputeCollector().collect_model(dense, x[:1])
    assert (
        values["stochastic_depth_inference_forward_flops_per_sample"]
        == dense_compute.flops
    )
    model.eval()
    with torch.no_grad():
        model(x)
    active.reset()
    active.update(x, None, None, model)
    assert active.compute()["stochastic_depth_expected_inference_active_blocks"] == 10.0


def test_mobile_grid_uses_same_preprocessing_and_validation_protocol_as_autopruner():
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name="tune_stochastic_depth_mobilenetv2_tinyimagenet200_pL_grid_45"
        )
        auto = compose(
            config_name="tune_autopruner_mobilenetv2_tinyimagenet200_params_20_25"
        )
    assert cfg.dataloaders == auto.dataloaders
    assert cfg.model.backbone.num_classes == 200
    assert cfg.model.backbone.stem_stride == 2
    assert cfg.model.backbone.require_pretrained
    assert cfg.training_arguments.num_epochs == cfg.scheduler.T_max == 45
    assert not cfg.training_arguments.batchnorm_recalibration.enabled
    assert not cfg.training_arguments.adaptive_lambda.enabled
    assert cfg.run_history.monitor == cfg.tuning.objective_metric == "valid_accuracy"
    assert cfg.tuning.n_trials == len(cfg.tuning.points) == 7
    assert cfg.tuning.n_jobs == cfg.tuning.repeats_per_trial == 1
    assert cfg.tuning.points_in_order and cfg.tuning.seed_base == 42
    points = OmegaConf.to_container(cfg.tuning.points, resolve=False)
    assert [point["model.backbone.final_survival_probability"] for point in points] == [
        0.3,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
    for point in points:
        p_l = point["model.backbone.final_survival_probability"]
        assert float(point["mlflow.tags.expected_active_blocks"]) == pytest.approx(
            10 - 5.5 * (1 - p_l)
        )
