from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from net_complexity.models.feature_selection import (
    GumbelBottleneckLayer,
    ResNet50,
    get_gumbel_loss,
    get_gumbel_modules,
)


def _close_all_gates(block: GumbelBottleneckLayer) -> None:
    with torch.no_grad():
        block.gumbel_layer.logits[:, 0] = 1.0
        block.gumbel_layer.logits[:, 1] = 0.0


def test_resnet50_accepts_partial_gumbel_blocks_and_collects_channel_masks():
    model = ResNet50(
        num_classes=5,
        in_channels=3,
        resnet_block=partial(GumbelBottleneckLayer, temperature=0.75),
    )
    model.train()

    logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 5)
    assert len(get_gumbel_modules(model)) == 16

    reg_loss = get_gumbel_loss(model)
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0
    assert reg_loss.item() > 0


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
