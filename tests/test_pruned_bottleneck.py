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


def test_pruned_gumbel_bottleneck_mid1_mid2_narrow_conv1_conv2_without_scatter_back():
    torch.manual_seed(0)
    downsample = torch.nn.Sequential(
        torch.nn.Conv2d(8, 24, kernel_size=1),
        torch.nn.BatchNorm2d(24),
    )
    block = PrunedGumbelBottleneck(
        in_channels=8,
        out_channels=6,  # full_channels = 6 * expansion(4) = 24
        i_downsample=downsample,
        disabled_mid1_channels=[0, 1],  # conv1-out/conv2-in: 6 -> 4
        disabled_mid2_channels=[2],  # conv2-out/conv3-in: 6 -> 5
    )
    block.eval()

    assert block.w1 == 4
    assert block.w2 == 5
    assert block.n_active == 24  # output gate untouched
    assert block.conv1.out_channels == 4
    assert block.conv2.in_channels == 4
    assert block.conv2.out_channels == 5
    assert block.conv3.in_channels == 5

    out = block(torch.randn(2, 8, 6, 6))
    assert out.shape == (2, 24, 6, 6)


def test_pruned_gumbel_bottleneck_requires_at_least_one_active_mid1_and_mid2_channel():
    import pytest

    with pytest.raises(ValueError, match="conv1-output/conv2-input"):
        PrunedGumbelBottleneck(
            in_channels=4,
            out_channels=3,
            disabled_mid1_channels=[0, 1, 2],
        )

    with pytest.raises(ValueError, match="conv2-output/conv3-input"):
        PrunedGumbelBottleneck(
            in_channels=4,
            out_channels=3,
            disabled_mid2_channels=[0, 1, 2],
        )


def test_pruned_resnet_accepts_nested_pruning_spec_for_all_three_boundaries():
    pruning_spec = {
        "layer1.0": {"output": [0, 1, 2], "mid1": [3], "mid2": [4, 5]},
        # legacy flat-list shorthand still means "output only".
        "layer3.4": [10],
    }
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
    block = model.layer1[0]
    assert block.n_active == 256 - 3
    assert block.w1 == 64 - 1  # layer1 planes = 64
    assert block.w2 == 64 - 2
    assert model.layer3[4].n_active == 1024 - 1
    assert model.layer3[4].w1 == 256  # planes for layer3, untouched
    assert model.layer3[4].w2 == 256
    # Untouched blocks keep full channel counts on every boundary.
    assert model.layer1[1].n_active == 256
    assert model.layer1[1].w1 == 64
    assert model.layer1[1].w2 == 64


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
