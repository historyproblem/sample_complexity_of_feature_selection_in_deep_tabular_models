from functools import partial

import torch
import torch.nn as nn
import pytest

from net_complexity.models.feature_selection import (
    AIGBottleneckLayer,
    AIGTargetRateLoss,
    ClassificationFeatureSelectionWrapper,
    GumbelLayer,
    MaskedGumbelLayer,
    ResNet50,
    apply_paper_style_conv_init,
    get_AIG_modules,
    get_gumbel_loss,
    get_gumbel_posterior_regularization_terms,
)


def test_gumbel_layer_posterior_regularization_terms_matches_manual_computation():
    layer = GumbelLayer(input_dim=4)
    layer.logits = nn.Parameter(
        torch.tensor(
            [[0.0, 2.0], [1.0, -1.0], [0.5, 0.5], [-3.0, 3.0]],
        )
    )

    mean_p_open, negative_entropy = layer.posterior_regularization_terms()

    probs = torch.softmax(layer.logits, dim=-1)
    expected_mean_p_open = probs[:, 1].mean()
    expected_negative_entropy = (probs * probs.log()).sum(dim=-1).mean()

    torch.testing.assert_close(mean_p_open, expected_mean_p_open)
    torch.testing.assert_close(negative_entropy, expected_negative_entropy)


def test_gumbel_layer_posterior_terms_are_zero_when_bypassed():
    layer = GumbelLayer(input_dim=3)
    layer.set_bypass(True)

    mean_p_open, negative_entropy = layer.posterior_regularization_terms()

    assert mean_p_open.item() == 0.0
    assert negative_entropy.item() == 0.0


def test_masked_gumbel_layer_posterior_excludes_disabled_channels():
    layer = MaskedGumbelLayer(input_dim=4, disabled_channels=[1, 3])
    layer.logits = nn.Parameter(
        torch.tensor(
            [[0.0, 2.0], [1.0, -1.0], [0.5, 0.5], [-3.0, 3.0]],
        )
    )

    mean_p_open, negative_entropy = layer.posterior_regularization_terms()

    probs = torch.softmax(layer.logits, dim=-1)
    enabled = torch.tensor([True, False, True, False])
    expected_mean_p_open = probs[enabled, 1].mean()
    expected_negative_entropy = (probs * probs.log()).sum(dim=-1)[enabled].mean()

    torch.testing.assert_close(mean_p_open, expected_mean_p_open)
    torch.testing.assert_close(negative_entropy, expected_negative_entropy)


def test_get_gumbel_posterior_regularization_terms_returns_none_without_gumbel_modules():
    assert get_gumbel_posterior_regularization_terms(nn.Linear(4, 3)) is None


@pytest.mark.parametrize(
    ("entropy_regularization", "entropy_sign"),
    [
        ("plus_negative_entropy", 1.0),
        ("minus_negative_entropy", -1.0),
    ],
)
def test_gumbel_wrapper_uses_posterior_path_only_when_entropy_enabled(
    entropy_regularization,
    entropy_sign,
):
    class TinyGumbelClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gumbel_layer = GumbelLayer(input_dim=4)
            self.classifier = nn.Linear(4, 3)

        def forward(self, x):
            x = self.gumbel_layer(x)
            return self.classifier(x.mean(dim=(2, 3)))

    backbone = TinyGumbelClassifier()
    wrapper = ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=0.25,
        bypass_on_zero_lambda=False,
        entropy_regularization=entropy_regularization,
        entropy_regularization_coef=0.5,
        regularization_loss=get_gumbel_loss,
    )
    wrapper.eval()

    output = wrapper(torch.randn(2, 4, 4, 4), torch.tensor([0, 1]))

    expected = 0.25 * output.mean_p_open + entropy_sign * 0.5 * output.negative_entropy
    torch.testing.assert_close(output.reg_loss, expected)
    torch.testing.assert_close(output.loss, output.ce_loss + expected)


def test_gumbel_wrapper_disabled_entropy_keeps_legacy_regularization_loss_path():
    class TinyGumbelClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gumbel_layer = GumbelLayer(input_dim=4)
            self.classifier = nn.Linear(4, 3)

        def forward(self, x):
            x = self.gumbel_layer(x)
            return self.classifier(x.mean(dim=(2, 3)))

    backbone = TinyGumbelClassifier()
    calls = {"count": 0}

    def _tracking_regularization_loss(model):
        calls["count"] += 1
        return get_gumbel_loss(model)

    wrapper = ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=0.25,
        bypass_on_zero_lambda=False,
        entropy_regularization="disabled",
        regularization_loss=_tracking_regularization_loss,
    )
    wrapper.eval()

    output = wrapper(torch.randn(2, 4, 4, 4), torch.tensor([0, 1]))

    assert calls["count"] == 1
    assert output.mean_p_open is None
    assert output.negative_entropy is None
    torch.testing.assert_close(output.reg_loss, 0.25 * output.regularization_loss)


def test_aig_target_rate_loss_matches_manual_squared_deviation():
    model = ResNet50(
        num_classes=4,
        in_channels=3,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
        resnet_block=partial(AIGBottleneckLayer, temperature=1.0),
    )
    model.eval()
    model(torch.randn(2, 3, 16, 16))

    loss_fn = AIGTargetRateLoss(target_rate=0.6)
    loss = loss_fn(model)

    activations = [module.activations for module in get_AIG_modules(model).values()]
    expected = sum((0.6 - act.mean()) ** 2 for act in activations) / len(activations)

    torch.testing.assert_close(loss, expected)


def test_aig_target_rate_loss_excludes_always_on_blocks_from_penalty():
    model = ResNet50(
        num_classes=4,
        in_channels=3,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
        resnet_block=partial(AIGBottleneckLayer, temperature=1.0),
    )
    model.eval()
    model(torch.randn(2, 3, 16, 16))

    activations = [module.activations for module in get_AIG_modules(model).values()]
    always_on = (0, 2, 5)

    loss_fn = AIGTargetRateLoss(target_rate=0.6, always_on_blocks=always_on)
    loss = loss_fn(model)

    expected = sum(
        (0.6 - act.mean()) ** 2
        for index, act in enumerate(activations)
        if index not in always_on
    ) / len(activations)

    torch.testing.assert_close(loss, expected)


def test_aig_target_rate_loss_rejects_out_of_range_target():
    with pytest.raises(ValueError, match="target_rate must be within"):
        AIGTargetRateLoss(target_rate=1.5)


def test_aig_target_rate_loss_returns_zero_without_aig_modules():
    loss = AIGTargetRateLoss(target_rate=0.5)(nn.Linear(4, 3))
    assert loss == 0.0


def test_apply_paper_style_conv_init_preserves_gate_final_conv():
    model = ResNet50(
        num_classes=4,
        in_channels=3,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
        resnet_block=partial(AIGBottleneckLayer, temperature=1.0),
    )
    gate_final_conv = model.layer1[0].gate.router[-1]
    gate_final_conv_weight_before = gate_final_conv.weight.detach().clone()
    other_conv = model.layer1[0].conv1

    apply_paper_style_conv_init(model)

    # Gate's final router conv keeps its own low-variance init untouched.
    torch.testing.assert_close(gate_final_conv.weight, gate_final_conv_weight_before)
    # Every other conv (including the gate's hidden router conv) gets the
    # general kaiming-normal-style reinit, so its std should roughly match
    # sqrt(2 / (kh*kw*out_channels)) rather than PyTorch's default init scale.
    fan = other_conv.kernel_size[0] * other_conv.kernel_size[1] * other_conv.out_channels
    expected_std = (2.0 / fan) ** 0.5
    assert other_conv.weight.std().item() == pytest.approx(expected_std, rel=0.2)


def test_backbone_weight_init_rejects_unknown_mode():
    with pytest.raises(ValueError, match="backbone_weight_init must be one of"):
        ClassificationFeatureSelectionWrapper(
            backbone=nn.Linear(4, 3),
            backbone_weight_init="unknown",
        )


def test_backbone_weight_init_default_leaves_pytorch_default_init_untouched():
    torch.manual_seed(0)
    backbone_a = ResNet50(
        num_classes=4, in_channels=3, stem_kernel_size=3, stem_stride=1,
        stem_padding=1, use_maxpool=False,
    )
    torch.manual_seed(0)
    backbone_b = ResNet50(
        num_classes=4, in_channels=3, stem_kernel_size=3, stem_stride=1,
        stem_padding=1, use_maxpool=False,
    )
    ClassificationFeatureSelectionWrapper(backbone=backbone_b, backbone_weight_init="default")

    torch.testing.assert_close(backbone_a.conv1.weight, backbone_b.conv1.weight)
