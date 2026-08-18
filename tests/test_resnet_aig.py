from functools import partial

import torch
import torch.nn as nn
import pytest
from omegaconf import OmegaConf

from net_complexity.metrics.aig import AIGActivationsMetric
from net_complexity.models.aig import AIGBlockGate
from net_complexity.models.feature_selection import (
    AIGBottleneckLayer,
    ClassificationFeatureSelectionWrapper,
    ResNet50,
    get_AIG_modules,
    get_AIG_posterior_regularization_terms,
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


def test_aig_gate_uses_two_logits_per_block_and_supports_l1_probability_loss():
    gate = AIGBlockGate(
        in_channels=4,
        hidden_channels=2,
        keep_prob_init=0.8,
        regularization="l1_probability",
    )
    gate.eval()

    values = gate(torch.randn(3, 4, 5, 5))

    assert values.shape == (3, 1, 1, 1)
    assert gate.logits.shape == (3, 2, 1, 1)
    assert gate.probabilities.shape == (3, 2, 1, 1)
    torch.testing.assert_close(
        gate.regularization_loss(),
        gate.keep_probabilities.mean(),
    )


@pytest.mark.parametrize("magnitude", [1.0e2, 1.0e4])
def test_aig_posterior_regularization_is_finite_for_extreme_logits(magnitude):
    gate = AIGBlockGate(
        in_channels=4,
        hidden_channels=2,
        regularization="l1_probability",
    )
    gate.logits = torch.tensor(
        [[[[magnitude]], [[-magnitude]]], [[[-magnitude]], [[magnitude]]]],
        requires_grad=True,
    )

    mean_p_open, negative_entropy = gate.posterior_regularization_terms()
    reg_loss = 0.25 * mean_p_open + negative_entropy
    reg_loss.backward()

    assert torch.isfinite(mean_p_open)
    assert torch.isfinite(negative_entropy)
    assert torch.isfinite(reg_loss)
    assert torch.isfinite(gate.logits.grad).all()


@pytest.mark.parametrize(
    ("entropy_regularization", "entropy_sign"),
    [
        ("disabled", 0.0),
        ("plus_negative_entropy", 1.0),
        ("minus_negative_entropy", -1.0),
    ],
)
def test_aig_wrapper_logs_soft_posterior_regularization_components(
    entropy_regularization,
    entropy_sign,
):
    class TinyAIGClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = AIGBlockGate(in_channels=4, regularization="l1_probability")
            self.classifier = nn.Linear(4, 3)

        def forward(self, x):
            x = x * self.gate(x)
            return self.classifier(x.mean(dim=(2, 3)))

    backbone = TinyAIGClassifier()
    wrapper = ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=0.25,
        bypass_on_zero_lambda=False,
        entropy_regularization=entropy_regularization,
        regularization_loss=get_AIG_regularization_loss,
    )
    wrapper.eval()

    output = wrapper(torch.randn(2, 4, 4, 4), torch.tensor([0, 1]))

    expected = 0.25 * output.mean_p_open + entropy_sign * output.negative_entropy
    torch.testing.assert_close(output.regularization_loss, output.mean_p_open)
    torch.testing.assert_close(output.reg_loss, expected)
    torch.testing.assert_close(output.loss, output.ce_loss + expected)


def test_zero_entropy_coef_keeps_plus_negative_entropy_aig_active():
    class TinyAIGClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = AIGBlockGate(in_channels=4, regularization="l1_probability")
            self.classifier = nn.Linear(4, 3)

        def forward(self, x):
            x = x * self.gate(x)
            return self.classifier(x.mean(dim=(2, 3)))

    backbone = TinyAIGClassifier()
    wrapper = ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=0.25,
        bypass_on_zero_lambda=True,
        entropy_regularization="plus_negative_entropy",
        entropy_regularization_coef=0.0,
        regularization_loss=get_AIG_regularization_loss,
    )
    wrapper.eval()

    assert not backbone.gate.bypass

    output = wrapper(torch.randn(2, 4, 4, 4), torch.tensor([0, 1]))

    assert not backbone.gate.bypass
    torch.testing.assert_close(output.regularization_loss, output.mean_p_open)
    torch.testing.assert_close(output.reg_loss, 0.25 * output.mean_p_open)
    torch.testing.assert_close(output.loss, output.ce_loss + output.reg_loss)


def test_aig_wrapper_rejects_unknown_entropy_regularization_mode():
    with pytest.raises(ValueError, match="entropy_regularization must be one of"):
        ClassificationFeatureSelectionWrapper(
            backbone=nn.Linear(4, 3),
            entropy_regularization="unknown",
        )


def test_aig_posterior_terms_unwrap_aig_bottleneck_layers():
    block = AIGBottleneckLayer(
        in_planes=64,
        planes=64,
        gate_regularization="l1_probability",
    )
    block.gate(torch.randn(2, 64, 4, 4))

    terms = get_AIG_posterior_regularization_terms(nn.Sequential(block))

    assert terms is not None
    mean_p_open, negative_entropy = terms
    torch.testing.assert_close(mean_p_open, block.gate.keep_probabilities.mean())
    assert torch.isfinite(negative_entropy)


def test_aig_gate_l2_activation_loss_matches_legacy_target_zero_penalty():
    gate = AIGBlockGate(in_channels=4, regularization="l2_activation")
    gate.activations = torch.tensor([[[[1.0]]], [[[0.0]]]])
    gate.keep_probabilities = torch.tensor([[[[0.9]]], [[[0.2]]]])

    torch.testing.assert_close(
        gate.regularization_loss(),
        gate.activations.mean() ** 2,
    )


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


def test_aig_adaptive_lambda_allows_clean_config_without_recovery_block():
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
            },
        }
    )

    controller = _build_adaptive_lambda(training_arguments, wrapper)

    assert controller is not None
    assert controller.recovery_config.enabled is False
