from functools import partial

import torch

from net_complexity.models.feature_selection import (
    ResNet50,
    STGBottleneckLayer,
    STGResNet50,
    get_stg_loss,
    get_stg_modules,
)


def test_stg_resnet50_collects_all_channel_masks():
    model = STGResNet50(num_classes=7, in_channels=3, sigma=0.5)
    model.train()

    logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 7)
    assert len(get_stg_modules(model)) == 49

    reg_loss = get_stg_loss(model)
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0
    assert reg_loss.item() > 0


def test_resnet50_accepts_partial_stg_blocks():
    model = ResNet50(
        num_classes=5,
        in_channels=3,
        resnet_block=partial(STGBottleneckLayer, sigma=0.25),
    )
    model.eval()

    logits = model(torch.randn(1, 3, 64, 64))

    assert logits.shape == (1, 5)
    assert len(get_stg_modules(model)) == 48
