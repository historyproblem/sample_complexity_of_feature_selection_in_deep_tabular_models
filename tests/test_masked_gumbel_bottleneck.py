from functools import partial

import torch

from net_complexity.models.feature_selection import (
    MaskedGumbelBottleneckLayer,
    MaskedGumbelLayer,
    ResNet50,
    get_gumbel_modules,
)


def test_masked_gumbel_bottleneck_default_has_only_the_output_gate():
    # in_channels == out_channels * expansion(4): no downsample needed.
    block = MaskedGumbelBottleneckLayer(in_channels=16, out_channels=4)

    assert block.mid1_gumbel_layer is None
    assert block.mid2_gumbel_layer is None
    assert isinstance(block.gumbel_layer, MaskedGumbelLayer)

    out = block(torch.randn(2, 16, 6, 6))
    assert out.shape == (2, 16, 6, 6)  # out_channels * expansion(4)


def test_masked_gumbel_bottleneck_internal_width_adds_two_more_gates():
    block = MaskedGumbelBottleneckLayer(
        in_channels=16, out_channels=4, gate_internal_width=True,
    )

    assert isinstance(block.mid1_gumbel_layer, MaskedGumbelLayer)
    assert isinstance(block.mid2_gumbel_layer, MaskedGumbelLayer)
    # mid1/mid2 gate the shared "narrow" width (out_channels), not the
    # expanded output width (out_channels * expansion) the output gate uses.
    assert block.mid1_gumbel_layer.channel_mask.shape[0] == 4
    assert block.mid2_gumbel_layer.channel_mask.shape[0] == 4
    assert block.gumbel_layer.channel_mask.shape[0] == 16

    out = block(torch.randn(2, 16, 6, 6))
    assert out.shape == (2, 16, 6, 6)


def test_masked_gumbel_bottleneck_internal_width_zeroing_matches_identity_only_for_output_gate():
    """Closing every channel of the *output* gate degenerates to identity
    (matching MaskedGumbelBottleneckLayer's existing single-gate behavior);
    the mid1/mid2 gates do not have this property (see class docstring) —
    closing all of mid1 makes conv2's input all-zero, which is NOT identity.
    """
    torch.manual_seed(0)
    downsample = torch.nn.Sequential(
        torch.nn.Conv2d(8, 16, kernel_size=1),
        torch.nn.BatchNorm2d(16),
    )
    block = MaskedGumbelBottleneckLayer(
        in_channels=8,
        out_channels=4,
        gate_internal_width=True,
        deterministic_hard_mask=True,
    )
    block.i_downsample = downsample
    with torch.no_grad():
        block.gumbel_layer.logits.copy_(torch.tensor([[0.0, -10.0]] * 16))  # force all closed
    block.eval()

    x = torch.randn(2, 8, 6, 6)
    out = block(x)

    with torch.no_grad():
        identity = downsample(x)
    torch.testing.assert_close(out, torch.relu(identity))


def test_masked_gumbel_bottleneck_gates_are_discovered_by_get_gumbel_modules():
    model = ResNet50(
        num_classes=5,
        in_channels=3,
        resnet_block=partial(MaskedGumbelBottleneckLayer, gate_internal_width=True),
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
    )

    modules = get_gumbel_modules(model)

    # ResNet50 has 16 blocks; each contributes 3 gates (output + mid1 + mid2).
    assert len(modules) == 16 * 3
    assert any(name.endswith(".gumbel_layer") for name in modules)
    assert any(name.endswith(".mid1_gumbel_layer") for name in modules)
    assert any(name.endswith(".mid2_gumbel_layer") for name in modules)

    logits = model(torch.randn(2, 3, 16, 16))
    assert logits.shape == (2, 5)
