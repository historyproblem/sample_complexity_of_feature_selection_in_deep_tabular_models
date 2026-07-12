import torch

from net_complexity.wrappers import CIFARResNet20, ResNet50


def test_cifar_resnet20_accepts_tinyimagenet_resolution():
    model = CIFARResNet20(num_classes=200, in_channels=3).eval()

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))

    assert tuple(logits.shape) == (2, 200)


def test_resnet50_tinyimagenet_stem_outputs_200_classes():
    model = ResNet50(
        num_classes=200,
        in_channels=3,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
    ).eval()

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))

    assert tuple(logits.shape) == (2, 200)
