from functools import partial

import torch
import torch.nn as nn
import pytest
from omegaconf import OmegaConf

from net_complexity.metrics.aig import AIGActivationsMetric
from net_complexity.models.feature_selection import (
    AIGBottleneckLayer,
    ClassificationFeatureSelectionWrapper,
    ResNet50,
    get_AIG_modules,
    get_AIG_regularization_loss,
    parse_AIG_activations,
)
from net_complexity.training.engine import _build_adaptive_lambda


def test_resnet50_accepts_partial_aig_blocks_and_exposes_gate_activations():
    model = ResNet50(
        num_classes=6,
        in_channels=3,
        resnet_block=partial(AIGBottleneckLayer, temperature=0.75),
    )
    model.train()

    logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 6)

    activations = parse_AIG_activations(model)
    assert len(activations) == 16
    assert all(value.shape[0] == 2 for value in activations.values())

    reg_loss = get_AIG_regularization_loss(model)
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0
    assert reg_loss.item() >= 0


def test_resnet50_supports_cifar_style_stem_without_maxpool():
    model = ResNet50(
        num_classes=4,
        in_channels=3,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
    )
    model.eval()

    logits = model(torch.randn(2, 3, 32, 32))

    assert logits.shape == (2, 4)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert model.conv1.padding == (1, 1)
    assert model.max_pool.__class__.__name__ == "Identity"


def test_wrapper_skips_regularization_call_when_lambda_is_zero():
    backbone = nn.Linear(4, 3)
    targets = torch.tensor([1, 0])
    inputs = torch.randn(2, 4)
    calls = {"count": 0}

    def _regularization_loss(_model):
        calls["count"] += 1
        return torch.tensor(7.0)

    wrapper = ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=0.0,
        criterion=nn.CrossEntropyLoss(),
        regularization_loss=_regularization_loss,
    )

    output = wrapper(inputs, targets)

    assert calls["count"] == 0
    assert output.regularization_loss.item() == 0.0
    torch.testing.assert_close(output.loss, output.ce_loss)


def test_aig_wrapper_bypasses_dynamic_gates_only_for_zero_lambda_baseline():
    backbone = ResNet50(
        num_classes=3,
        in_channels=3,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
        resnet_block=partial(AIGBottleneckLayer, temperature=1.0),
    )
    wrapper = ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=0.0,
        bypass_on_zero_lambda=True,
    )
    wrapper.eval()

    aig_modules = get_AIG_modules(wrapper)
    assert len(aig_modules) == 16
    assert all(module.bypass for module in aig_modules.values())

    output = wrapper(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))

    assert output.logits.shape == (2, 3)
    assert all(torch.all(module.activations == 1.0) for module in aig_modules.values())

    wrapper.set_lambda_coef(0.01)

    assert all(not module.bypass for module in aig_modules.values())


def test_aig_metric_exposes_zero_probability_for_adaptive_lambda():
    model = nn.Sequential(AIGBottleneckLayer(in_planes=64, planes=64))
    model[0].activations = torch.tensor([[[[1.0]]], [[[0.0]]]])
    metric = AIGActivationsMetric()

    metric.update(None, None, None, model)
    computed = metric.compute()

    assert computed["average_prob"] == pytest.approx(0.5)
    assert computed["average_zero_prob"] == pytest.approx(0.5)
    assert computed["max_zero_prob"] == pytest.approx(0.5)
    assert computed["min_zero_prob"] == pytest.approx(0.5)


def test_aig_metric_weights_incomplete_batches_by_number_of_examples():
    model = nn.Sequential(AIGBottleneckLayer(in_planes=64, planes=64))
    metric = AIGActivationsMetric()

    model[0].activations = torch.ones(128, 1, 1, 1)
    metric.update(None, None, None, model)
    model[0].activations = torch.zeros(1, 1, 1, 1)
    metric.update(None, None, None, model)

    computed = metric.compute()

    assert computed["average_prob"] == pytest.approx(128.0 / 129.0)
    assert computed["average_zero_prob"] == pytest.approx(1.0 / 129.0)


def test_aig_adaptive_lambda_rejects_gumbel_open_bias_recovery():
    backbone = nn.Sequential(AIGBottleneckLayer(in_planes=64, planes=64))
    wrapper = ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=0.01,
        bypass_on_zero_lambda=False,
    )
    training_arguments = OmegaConf.create(
        {
            "adaptive_lambda": {
                "enabled": True,
                "recovery": {
                    "enabled": True,
                },
            },
        }
    )

    with pytest.raises(ValueError, match="AIG adaptive lambda requires"):
        _build_adaptive_lambda(training_arguments, wrapper)
