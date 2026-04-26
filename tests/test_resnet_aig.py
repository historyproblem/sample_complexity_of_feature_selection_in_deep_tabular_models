from functools import partial

import torch

from net_complexity.models.feature_selection import (
    AIGBottleneckLayer,
    ResNet50,
    get_AIG_regularization_loss,
    parse_AIG_activations,
)


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
