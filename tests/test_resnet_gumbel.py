from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from net_complexity.models.feature_selection import (
    ClassificationFeatureSelectionWrapper,
    GumbelBottleneckLayer,
    GumbelLayer,
    ResNet50,
    get_gumbel_loss,
    get_gumbel_modules,
)


def _close_all_gates(block: GumbelBottleneckLayer) -> None:
    with torch.no_grad():
        block.gumbel_layer.logits[:, 0] = 1.0
        block.gumbel_layer.logits[:, 1] = 0.0


def _set_reference_logits(layer: GumbelLayer) -> None:
    with torch.no_grad():
        layer.logits.copy_(torch.tensor([
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ]))


def _build_wrapped_resnet50(
    lambda_coef: float,
    *,
    bypass_on_zero_lambda: bool = True,
    gumbel_init_mode: str = "auto",
) -> ClassificationFeatureSelectionWrapper:
    backbone = ResNet50(
        num_classes=5,
        in_channels=3,
        resnet_block=partial(GumbelBottleneckLayer, temperature=0.75),
    )
    return ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=lambda_coef,
        gumbel_init_mode=gumbel_init_mode,
        bypass_on_zero_lambda=bypass_on_zero_lambda,
    )


def test_resnet50_accepts_partial_gumbel_blocks_and_collects_channel_masks():
    model = _build_wrapped_resnet50(lambda_coef=0.5).backbone
    model.train()

    logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 5)
    assert len(get_gumbel_modules(model)) == 16

    reg_loss = get_gumbel_loss(model)
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0
    assert reg_loss.item() > 0


def test_wrapper_uses_fully_open_gumbel_init_when_lambda_is_zero():
    wrapper = _build_wrapped_resnet50(lambda_coef=0.0)

    gumbel_modules = get_gumbel_modules(wrapper.backbone)
    selection_probs = torch.cat(
        [module.get_selection_probs() for module in gumbel_modules.values()]
    )

    assert torch.all(selection_probs == 1.0)
    assert all(module.bypass for module in gumbel_modules.values())


def test_wrapper_can_disable_zero_lambda_bypass():
    wrapper = _build_wrapped_resnet50(
        lambda_coef=0.0,
        bypass_on_zero_lambda=False,
    )

    gumbel_modules = get_gumbel_modules(wrapper.backbone)
    selection_probs = torch.cat(
        [module.get_selection_probs() for module in gumbel_modules.values()]
    )
    mean_selection_prob = float(selection_probs.mean().item())

    assert 0.84 < mean_selection_prob < 0.90
    assert all(not module.bypass for module in gumbel_modules.values())


def test_wrapper_uses_paper_gumbel_init_when_lambda_is_positive():
    wrapper = _build_wrapped_resnet50(lambda_coef=0.5)

    gumbel_modules = get_gumbel_modules(wrapper.backbone)
    selection_probs = torch.cat(
        [module.get_selection_probs() for module in gumbel_modules.values()]
    )
    mean_selection_prob = float(selection_probs.mean().item())

    assert 0.84 < mean_selection_prob < 0.90
    assert all(not module.bypass for module in gumbel_modules.values())


def test_wrapper_supports_resnet50_specific_paper_init_mode():
    wrapper = _build_wrapped_resnet50(
        lambda_coef=0.0,
        bypass_on_zero_lambda=False,
        gumbel_init_mode="paper_resnet50",
    )

    gumbel_modules = get_gumbel_modules(wrapper.backbone)
    selection_probs = torch.cat(
        [module.get_selection_probs() for module in gumbel_modules.values()]
    )
    mean_selection_prob = float(selection_probs.mean().item())

    assert 0.97 < mean_selection_prob < 0.99
    assert all(not module.bypass for module in gumbel_modules.values())


def test_set_lambda_coef_respects_disabled_zero_lambda_bypass():
    wrapper = _build_wrapped_resnet50(
        lambda_coef=0.5,
        bypass_on_zero_lambda=False,
    )

    wrapper.set_lambda_coef(0.0)

    gumbel_modules = get_gumbel_modules(wrapper.backbone)

    assert all(not module.bypass for module in gumbel_modules.values())


def test_bypassed_gumbel_layer_returns_identity():
    layer = GumbelBottleneckLayer(256, 64, stride=1)
    layer.train()
    layer.gumbel_layer.set_bypass(True)

    x = torch.randn(2, 256, 8, 8)

    with torch.no_grad():
        gated = layer.gumbel_layer(x)

    torch.testing.assert_close(gated, x)


def test_force_ones_mask_returns_identity_without_bypass():
    layer = GumbelLayer(input_dim=256, temperature=0.75, force_ones_mask=True)
    layer.train()

    x = torch.randn(2, 256, 8, 8)

    with torch.no_grad():
        gated = layer(x)

    assert layer.bypass is False
    torch.testing.assert_close(gated, x)


def test_deterministic_soft_mask_uses_on_probability_without_gumbel_sampling():
    layer = GumbelLayer(input_dim=3, temperature=0.75, deterministic_soft_mask=True)
    layer.train()
    _set_reference_logits(layer)

    x = torch.ones(2, 3, 4, 4)

    with torch.no_grad():
        gated = layer(x)

    expected = F.softmax(layer.logits, dim=1)[:, 1].view(1, 3, 1, 1).expand_as(x)

    assert layer.bypass is False
    torch.testing.assert_close(gated, expected)


def test_deterministic_hard_mask_thresholds_on_probability_without_gumbel_sampling():
    layer = GumbelLayer(input_dim=3, temperature=0.75, deterministic_hard_mask=True)
    layer.train()
    _set_reference_logits(layer)

    x = torch.ones(2, 3, 4, 4)

    with torch.no_grad():
        gated = layer(x)

    expected = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1).expand_as(x)

    assert layer.bypass is False
    torch.testing.assert_close(gated, expected)


def test_train_eval_gate_modes_support_soft_train_hard_eval():
    layer = GumbelLayer(
        input_dim=3,
        temperature=0.75,
        train_gate_mode="deterministic_soft",
        eval_gate_mode="deterministic_hard",
    )
    _set_reference_logits(layer)
    x = torch.ones(2, 3, 4, 4)

    layer.train()
    with torch.no_grad():
        train_gated = layer(x)

    layer.eval()
    with torch.no_grad():
        eval_gated = layer(x)

    expected_train = F.softmax(layer.logits, dim=1)[:, 1].view(1, 3, 1, 1).expand_as(x)
    expected_eval = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1).expand_as(x)

    torch.testing.assert_close(train_gated, expected_train)
    torch.testing.assert_close(eval_gated, expected_eval)


def test_ste_hard_uses_hard_forward_and_soft_backward():
    layer = GumbelLayer(
        input_dim=3,
        temperature=0.75,
        train_gate_mode="ste_hard",
        eval_gate_mode="deterministic_hard",
    )
    _set_reference_logits(layer)
    x = torch.ones(2, 3, 4, 4, requires_grad=True)

    layer.train()
    gated = layer(x)
    expected_forward = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1).expand_as(gated)

    torch.testing.assert_close(gated.detach(), expected_forward)

    loss = gated.sum()
    loss.backward()

    assert layer.logits.grad is not None
    assert float(layer.logits.grad.abs().sum().item()) > 0.0


def test_beta_zero_makes_gumbel_hard_train_gates_deterministic():
    layer = GumbelLayer(
        input_dim=2,
        temperature=0.75,
        beta=0.0,
        train_gate_mode="gumbel_hard",
        eval_gate_mode="deterministic_hard",
    )
    layer.train()
    x = torch.randn(4, 2, 3, 3)

    with torch.no_grad():
        layer.logits.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

    torch.manual_seed(0)
    first = layer.compute_gates(x)
    torch.manual_seed(123)
    second = layer.compute_gates(x)

    expected = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand_as(first)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, expected)


def test_beta_scales_gumbel_soft_sample_strength():
    logits = torch.zeros(1, 1, 2)
    low_beta_layer = GumbelLayer(input_dim=1, temperature=1.0, beta=0.25)
    high_beta_layer = GumbelLayer(input_dim=1, temperature=1.0, beta=4.0)

    torch.manual_seed(7)
    low_beta_sample = low_beta_layer._sample_gumbel_softmax(logits)
    torch.manual_seed(7)
    high_beta_sample = high_beta_layer._sample_gumbel_softmax(logits)

    low_beta_deviation = torch.abs(low_beta_sample[..., 1] - 0.5)
    high_beta_deviation = torch.abs(high_beta_sample[..., 1] - 0.5)

    assert torch.all(high_beta_deviation > low_beta_deviation)


def test_gumbel_layer_rejects_conflicting_mask_modes():
    try:
        GumbelLayer(
            input_dim=3,
            force_ones_mask=True,
            deterministic_soft_mask=True,
            deterministic_hard_mask=True,
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("Expected mutually exclusive mask modes to raise ValueError.")


def test_gumbel_layer_rejects_mixing_legacy_flags_with_explicit_modes():
    try:
        GumbelLayer(
            input_dim=3,
            deterministic_soft_mask=True,
            train_gate_mode="ste_hard",
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("Expected legacy flags mixed with explicit modes to raise ValueError.")


def test_closed_gumbel_bottleneck_preserves_identity_shortcut():
    block = GumbelBottleneckLayer(256, 64, stride=1)
    block.eval()
    _close_all_gates(block)

    x = torch.randn(2, 256, 8, 8)

    with torch.no_grad():
        output = block(x)
        expected = F.relu(x)

    torch.testing.assert_close(output, expected)


def test_closed_gumbel_bottleneck_preserves_downsample_shortcut():
    downsample = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=1, stride=2),
        nn.BatchNorm2d(512),
    )
    block = GumbelBottleneckLayer(256, 128, i_downsample=downsample, stride=2)
    block.eval()
    _close_all_gates(block)

    x = torch.randn(2, 256, 8, 8)

    with torch.no_grad():
        output = block(x)
        expected = F.relu(block.i_downsample(x))

    torch.testing.assert_close(output, expected)
