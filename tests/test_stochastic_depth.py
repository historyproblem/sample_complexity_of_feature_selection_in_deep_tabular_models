from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from net_complexity.metrics.stochastic_depth import (
    StochasticDepthActiveBlocksMetric,
    StochasticDepthFLOPsMetric,
)
from net_complexity.metrics.inspection import ComputeCollector
from net_complexity.models.feature_selection import (
    ClassificationFeatureSelectionWrapper,
    ResNet50,
)
from net_complexity.models.stochastic_depth import (
    HuangStochasticDepthBottleneck,
    StochasticDepthResNet50,
    get_stochastic_depth_blocks,
    linear_survival_probabilities,
)


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _identity_residual_block(
    survival_probability: float,
) -> HuangStochasticDepthBottleneck:
    block = HuangStochasticDepthBottleneck(
        1,
        1,
        survival_probability=survival_probability,
    )
    block.conv1 = nn.Identity()
    block.batch_norm1 = nn.Identity()
    block.conv2 = nn.Identity()
    block.batch_norm2 = nn.Identity()
    block.conv3 = nn.Identity()
    block.batch_norm3 = nn.Identity()
    block.relu = nn.Identity()
    block.i_downsample = None
    return block


def test_huang_linear_survival_schedule_matches_formula():
    probabilities = linear_survival_probabilities(
        num_blocks=16,
        final_survival_probability=0.5,
    )

    assert len(probabilities) == 16
    assert probabilities[0] == pytest.approx(0.96875)
    assert probabilities[-1] == pytest.approx(0.5)
    assert sum(probabilities) == pytest.approx(11.75)
    for block_index, probability in enumerate(probabilities, start=1):
        expected = 1.0 - block_index / 16.0 * (1.0 - 0.5)
        assert probability == pytest.approx(expected)


def test_stochastic_depth_resnet50_has_16_blocks_and_expected_depth():
    model = StochasticDepthResNet50(
        num_classes=10,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
    )

    blocks = get_stochastic_depth_blocks(model)
    probabilities = [block.survival_probability for block in blocks.values()]

    assert len(blocks) == 16
    assert probabilities[0] == pytest.approx(0.96875)
    assert probabilities[-1] == pytest.approx(0.5)
    assert sum(probabilities) == pytest.approx(11.75)


def test_stochastic_depth_uses_one_scalar_mask_for_whole_batch():
    block = HuangStochasticDepthBottleneck(1, 1, survival_probability=0.5)
    block.train()

    with torch.no_grad():
        block(torch.randn(8, 1, 4, 4))

    assert block.last_survival_mask.shape == torch.Size([])


def test_closed_train_block_does_not_call_residual_convs(monkeypatch):
    block = HuangStochasticDepthBottleneck(1, 1, survival_probability=0.0)
    block.train()

    def _fail_if_called(_input):
        raise AssertionError("residual branch should not be computed")

    monkeypatch.setattr(block.conv1, "forward", _fail_if_called)
    x = torch.ones(2, 1, 4, 4)

    with torch.no_grad():
        out = block(x)

    torch.testing.assert_close(out, x)
    assert block.last_residual_branch_active is False


def test_train_open_block_has_no_inverted_survival_scaling():
    block = _identity_residual_block(survival_probability=0.25)
    block.train()
    block._sample_survival_mask = lambda x: x.new_ones(())
    x = torch.full((2, 1, 2, 2), 3.0)

    with torch.no_grad():
        out = block(x)

    torch.testing.assert_close(out, x * 2.0)


def test_eval_block_scales_residual_branch_by_survival_probability():
    block = _identity_residual_block(survival_probability=0.25)
    block.eval()
    x = torch.full((2, 1, 2, 2), 4.0)

    with torch.no_grad():
        out = block(x)

    torch.testing.assert_close(out, x * 1.25)


def test_closed_train_block_still_computes_projection_shortcut(monkeypatch):
    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return torch.full((x.shape[0], 256, 2, 2), 2.0, device=x.device)

    projection = Projection()
    block = HuangStochasticDepthBottleneck(
        64,
        64,
        i_downsample=projection,
        stride=2,
        survival_probability=0.0,
    )
    block.train()
    monkeypatch.setattr(
        block.conv1,
        "forward",
        lambda _input: (_ for _ in ()).throw(
            AssertionError("residual branch should not be computed")
        ),
    )

    with torch.no_grad():
        out = block(torch.randn(3, 64, 4, 4))

    assert projection.calls == 1
    torch.testing.assert_close(out, torch.full_like(out, 2.0))


def test_stochastic_depth_resnet50_matches_plain_resnet50_parameters_and_inference_flops():
    stochastic_depth = StochasticDepthResNet50(
        num_classes=10,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
    )
    plain = ResNet50(
        num_classes=10,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
    )

    assert sum(p.numel() for p in stochastic_depth.parameters()) == sum(
        p.numel()
        for p in plain.parameters()
    )

    input_sample = torch.randn(1, 3, 32, 32)
    collector = ComputeCollector()
    stochastic_depth_compute, _ = collector.collect_model(stochastic_depth, input_sample)
    plain_compute, _ = collector.collect_model(plain, input_sample)

    assert stochastic_depth_compute.flops == plain_compute.flops


def test_stochastic_depth_wrapper_runs_full_forward_backward():
    torch.manual_seed(0)
    model = ClassificationFeatureSelectionWrapper(
        backbone=StochasticDepthResNet50(
            num_classes=4,
            stem_kernel_size=3,
            stem_stride=1,
            stem_padding=1,
            use_maxpool=False,
        ),
        lambda_coef=0.0,
    )
    model.train()
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    output = model(x, y)
    output.loss.backward()

    assert output.logits.shape == (2, 4)
    assert model.backbone.fc.weight.grad is not None


def test_stochastic_depth_metrics_report_expected_and_actual_train_values():
    model = StochasticDepthResNet50(
        num_classes=4,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
    )
    blocks = get_stochastic_depth_blocks(model)
    for block in blocks.values():
        block._sample_survival_mask = lambda x: x.new_ones(())

    model.train()
    x = torch.randn(1, 3, 32, 32)
    logits = model(x)

    active_metric = StochasticDepthActiveBlocksMetric()
    flops_metric = StochasticDepthFLOPsMetric(sample_size=1)
    active_metric.update(x, logits, torch.tensor([0]), model)
    flops_metric.update(x, logits, torch.tensor([0]), model)

    active = active_metric.compute()
    flops = flops_metric.compute()

    assert active["stochastic_depth_num_blocks"] == pytest.approx(16.0)
    assert active["stochastic_depth_expected_active_train_blocks"] == pytest.approx(11.75)
    assert active["stochastic_depth_actual_active_train_blocks"] == pytest.approx(16.0)
    assert flops["stochastic_depth_full_inference_flops"] > 0.0
    assert flops["stochastic_depth_expected_train_flops"] < flops[
        "stochastic_depth_full_inference_flops"
    ]
    assert flops["stochastic_depth_actual_train_flops"] == pytest.approx(
        flops["stochastic_depth_full_inference_flops"]
    )


def test_stochastic_depth_experiment_config_and_hydra_compose():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "stochastic_depth_resnet50_cifar10.yaml")

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "stochastic_depth"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "sgd_resnet50"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "stochastic_depth"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.training_arguments.batchnorm_recalibration.enabled is False

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        composed = compose(
            config_name="train",
            overrides=["experiment=stochastic_depth_resnet50_cifar10"],
        )

    assert composed.model.backbone._target_ == "net_complexity.wrappers.StochasticDepthResNet50"
    assert composed.model.backbone.final_survival_probability == pytest.approx(0.5)
    assert composed.model.backbone.survival_schedule == "linear"
    assert composed.training_arguments.batchnorm_recalibration.enabled is False


def test_stochastic_depth_pl08_experiment_config_composes():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        composed = compose(
            config_name="train",
            overrides=["experiment=stochastic_depth_resnet50_cifar10_pl08"],
        )

    assert composed.model.backbone._target_ == "net_complexity.wrappers.StochasticDepthResNet50"
    assert composed.model.backbone.final_survival_probability == pytest.approx(0.8)
    assert composed.model.backbone.survival_schedule == "linear"
    assert composed.mlflow.run_name == "stochastic_depth_resnet50_cifar10_pL_0.8"
