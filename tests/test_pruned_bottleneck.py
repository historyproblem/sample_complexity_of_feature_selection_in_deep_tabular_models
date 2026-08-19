import torch
from omegaconf import OmegaConf

from net_complexity.models.channel_pruning import build_pruned_bottleneck_model
from net_complexity.models.pruned_bottleneck import PrunedGumbelBottleneck, PrunedResNet


def test_pruned_gumbel_bottleneck_disabled_channels_receive_only_shortcut_value():
    torch.manual_seed(0)
    downsample = torch.nn.Sequential(
        torch.nn.Conv2d(8, 16, kernel_size=1),
        torch.nn.BatchNorm2d(16),
    )
    block = PrunedGumbelBottleneck(
        in_channels=8,
        out_channels=4,  # full_channels = 4 * expansion(4) = 16
        i_downsample=downsample,
        disabled_channels=[0, 5, 10, 15],
    )
    block.eval()

    x = torch.randn(2, 8, 6, 6)
    out = block(x)

    assert out.shape == (2, 16, 6, 6)
    assert block.n_active == 12

    with torch.no_grad():
        identity = downsample(x)
    torch.testing.assert_close(out[:, 0], torch.relu(identity[:, 0]))
    torch.testing.assert_close(out[:, 5], torch.relu(identity[:, 5]))


def test_pruned_gumbel_bottleneck_requires_at_least_one_active_channel():
    import pytest

    with pytest.raises(ValueError, match="All .* channels are disabled"):
        PrunedGumbelBottleneck(
            in_channels=4,
            out_channels=1,  # full_channels = 4
            disabled_channels=[0, 1, 2, 3],
        )


def test_pruned_resnet_forward_shape_matches_resnet50_topology():
    pruning_spec = {"layer1.0": [0, 1, 2], "layer3.4": [10]}
    model = PrunedResNet(
        pruning_spec=pruning_spec,
        layer_list=[3, 4, 6, 3],  # ResNet50
        num_classes=5,
        in_channels=3,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
    )
    model.eval()

    logits = model(torch.randn(2, 3, 16, 16))

    assert logits.shape == (2, 5)
    assert model.layer1[0].n_active == 256 - 3
    assert model.layer3[4].n_active == 1024 - 1
    # Untouched blocks keep the full channel count.
    assert model.layer1[1].n_active == 256
    assert model.layer4[0].n_active == 2048


def test_build_pruned_bottleneck_model_reads_backbone_config_and_wraps_output():
    config = OmegaConf.create(
        {
            "model": {
                "lambda_coef": 0.0,
                "criterion": {"_target_": "torch.nn.CrossEntropyLoss"},
                "backbone": {
                    "_target_": "net_complexity.wrappers.ResNet50",
                    "num_classes": 7,
                    "in_channels": 3,
                    "stem_kernel_size": 3,
                    "stem_stride": 1,
                    "stem_padding": 1,
                    "use_maxpool": False,
                },
            }
        }
    )

    model = build_pruned_bottleneck_model(config, pruning_spec={"layer2.0": [0, 1]})
    model.eval()

    output = model(torch.randn(2, 3, 16, 16), torch.tensor([0, 1]))

    assert output.logits.shape == (2, 7)
    assert isinstance(model.backbone, PrunedResNet)
    assert model.backbone.layer2[0].n_active == 512 - 2
