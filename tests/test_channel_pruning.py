from functools import partial

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from net_complexity.models.channel_pruning import (
    _mask_dict_to_bottleneck_pruning_spec,
    apply_channel_mask,
    build_structurally_pruned_model_from_config,
    transfer_gated_weights_to_structural,
    transfer_structural_weights_to_gated,
)
from net_complexity.models.feature_selection import (
    CIFARMaskedGumbelBasicBlock,
    CIFARResNet20,
    ClassificationFeatureSelectionWrapper,
    MaskedGumbelBottleneckLayer,
    MaskedGumbelLayer,
    ResNet50,
    SkippedCIFARBasicBlock,
    get_gumbel_modules,
)
from net_complexity.models.layer_skipping import apply_layer_skipping
from net_complexity.models.pruned_resnet import (
    CIFARPrunedGumbelBasicBlock,
    PrunedCIFARResNet,
)
from net_complexity.training.channel_history import (
    CifarResNet20GumbelCollector,
    resolve_channel_history_collector,
)


def test_masked_gumbel_layer_masks_gates_probs_and_regularizer():
    layer = MaskedGumbelLayer(
        input_dim=3,
        train_gate_mode="deterministic_soft",
        eval_gate_mode="deterministic_soft",
        disabled_channels=[1],
    )
    with torch.no_grad():
        layer.logits.copy_(
            torch.tensor(
                [
                    [0.0, 2.0],
                    [0.0, 2.0],
                    [2.0, 0.0],
                ]
            )
        )
    layer.eval()

    output = layer(torch.ones(2, 3, 4, 4))

    assert torch.all(output[:, 1] == 0.0)
    probs = F.softmax(layer.logits, dim=1)[:, 1]
    expected_reg = (probs[0] + probs[2]) / 2
    assert torch.allclose(layer.regularization_loss(), expected_reg)
    assert layer.get_selection_probs()[1].item() == 0.0


def test_apply_channel_mask_updates_matching_masked_gumbel_layers():
    backbone = CIFARResNet20(
        num_classes=10,
        resnet_block=partial(CIFARMaskedGumbelBasicBlock),
    )

    apply_channel_mask(backbone, {"layer2.0.gumbel_layer": [0, 3]})

    mask = backbone.layer2[0].gumbel_layer.channel_mask
    assert mask[0].item() == 0.0
    assert mask[3].item() == 0.0
    assert mask[1].item() == 1.0


def test_structural_channel_pruning_builds_wrapped_pruned_model():
    config = OmegaConf.create(
        {
            "model": {
                "lambda_coef": 0.0,
                "backbone": {
                    "_target_": "net_complexity.wrappers.CIFARResNet20",
                    "num_classes": 7,
                    "in_channels": 3,
                    "shortcut_option": "A",
                },
            }
        }
    )
    pruning_cfg = OmegaConf.create(
        {
            "enabled": True,
            "structural": True,
            "mode": "explicit",
            "mask": {"backbone.layer2.0.gumbel_layer": [0, 1]},
        }
    )

    model = build_structurally_pruned_model_from_config(config, pruning_cfg)

    assert isinstance(model.backbone, PrunedCIFARResNet)
    assert isinstance(model.backbone.layer2[0], CIFARPrunedGumbelBasicBlock)
    assert model.backbone.layer2[0].n_active == 30
    output = model(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
    assert output.logits.shape == (2, 7)


def test_layer_skipping_replaces_blocks_and_invalidates_selector_cache():
    backbone = CIFARResNet20(
        num_classes=10,
        resnet_block=partial(CIFARMaskedGumbelBasicBlock),
    )
    wrapper = ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=0.1,
    )
    assert len(get_gumbel_modules(wrapper.backbone)) == 9
    assert hasattr(wrapper.backbone, "_net_complexity_module_cache")

    apply_layer_skipping(wrapper, ["layer1.0"], mode="skip")

    assert isinstance(wrapper.backbone.layer1[0], SkippedCIFARBasicBlock)
    assert not hasattr(wrapper.backbone, "_net_complexity_module_cache")
    assert len(get_gumbel_modules(wrapper.backbone)) == 8


def test_masked_gumbel_resolves_gumbel_channel_history_collector():
    cfg = OmegaConf.create(
        {
            "model": {
                "backbone": {
                    "_target_": "net_complexity.wrappers.CIFARResNet20",
                    "resnet_block": {
                        "_target_": "net_complexity.wrappers.CIFARMaskedGumbelBasicBlock",
                    },
                },
            },
        }
    )

    collector = resolve_channel_history_collector(cfg)

    assert isinstance(collector, CifarResNet20GumbelCollector)


def test_mask_dict_to_bottleneck_pruning_spec_groups_all_three_gates_per_block():
    mask_dict = {
        "backbone.layer2.0.gumbel_layer": [0, 1],
        "backbone.layer2.0.mid1_gumbel_layer": [2],
        "backbone.layer2.0.mid2_gumbel_layer": [3, 4],
        "backbone.layer3.1.mid1_gumbel_layer": [5],
        "not_a_gate_path": [9],
    }

    spec = _mask_dict_to_bottleneck_pruning_spec(mask_dict)

    assert spec == {
        "layer2.0": {"output": [0, 1], "mid1": [2], "mid2": [3, 4]},
        "layer3.1": {"mid1": [5]},
    }


def test_build_structurally_pruned_model_from_config_prunes_all_three_boundaries():
    config = OmegaConf.create(
        {
            "model": {
                "lambda_coef": 0.0,
                "backbone": {
                    "_target_": "net_complexity.wrappers.ResNet50",
                    "num_classes": 5,
                    "in_channels": 3,
                    "stem_kernel_size": 3,
                    "stem_stride": 1,
                    "stem_padding": 1,
                    "use_maxpool": False,
                },
            }
        }
    )
    pruning_cfg = OmegaConf.create(
        {
            "enabled": True,
            "structural": True,
            "mode": "explicit",
            "mask": {
                "backbone.layer2.0.gumbel_layer": [0, 1],
                "backbone.layer2.0.mid1_gumbel_layer": [2],
                "backbone.layer2.0.mid2_gumbel_layer": [3, 4],
            },
        }
    )

    model = build_structurally_pruned_model_from_config(config, pruning_cfg)

    block = model.backbone.layer2[0]
    assert block.n_active == 512 - 2
    assert block.w1 == 128 - 1  # layer2 planes = 128
    assert block.w2 == 128 - 2
    output = model(torch.randn(2, 3, 16, 16), torch.tensor([0, 1]))
    assert output.logits.shape == (2, 5)


def test_masked_gumbel_bottleneck_channels_discovered_by_apply_channel_mask():
    backbone = ResNet50(
        num_classes=5,
        in_channels=3,
        resnet_block=partial(MaskedGumbelBottleneckLayer, gate_internal_width=True),
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
    )

    apply_channel_mask(
        backbone,
        {
            "layer2.0.gumbel_layer": [0, 1],
            "layer2.0.mid1_gumbel_layer": [2],
            "layer2.0.mid2_gumbel_layer": [3],
        },
    )

    block = backbone.layer2[0]
    assert block.gumbel_layer.channel_mask[0].item() == 0.0
    assert block.gumbel_layer.channel_mask[1].item() == 0.0
    assert block.mid1_gumbel_layer.channel_mask[2].item() == 0.0
    assert block.mid2_gumbel_layer.channel_mask[3].item() == 0.0
    # Untouched channels/gates stay enabled.
    assert block.gumbel_layer.channel_mask[2].item() == 1.0
    assert block.mid1_gumbel_layer.channel_mask[0].item() == 1.0


def test_bottleneck_weight_handoff_preserves_surviving_function():
    gated_backbone = ResNet50(
        num_classes=5,
        in_channels=3,
        resnet_block=partial(
            MaskedGumbelBottleneckLayer,
            gate_internal_width=True,
            force_ones_mask=True,
        ),
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
    )
    gated_model = ClassificationFeatureSelectionWrapper(
        backbone=gated_backbone,
        lambda_coef=0.0,
    )
    mask = {
        "backbone.layer2.0.gumbel_layer": [0, 7],
        "backbone.layer2.0.mid1_gumbel_layer": [2, 9],
        "backbone.layer2.0.mid2_gumbel_layer": [3, 11],
        "backbone.layer3.1.gumbel_layer": [5],
        "backbone.layer3.1.mid1_gumbel_layer": [6],
        "backbone.layer3.1.mid2_gumbel_layer": [8],
    }
    apply_channel_mask(gated_model, mask)
    config = OmegaConf.create(
        {
            "model": {
                "lambda_coef": 0.0,
                "backbone": {
                    "_target_": "net_complexity.wrappers.ResNet50",
                    "num_classes": 5,
                    "in_channels": 3,
                    "stem_kernel_size": 3,
                    "stem_stride": 1,
                    "stem_padding": 1,
                    "use_maxpool": False,
                },
            }
        }
    )
    pruning_cfg = OmegaConf.create(
        {
            "enabled": True,
            "structural": True,
            "mode": "explicit",
            "mask": mask,
        }
    )
    structural_model = build_structurally_pruned_model_from_config(config, pruning_cfg)
    transfer_gated_weights_to_structural(gated_model, structural_model)

    gated_model.eval()
    structural_model.eval()
    sample = torch.randn(2, 3, 16, 16)
    targets = torch.tensor([0, 1])
    with torch.no_grad():
        expected = gated_model(sample, targets).logits
        actual = structural_model(sample, targets).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    # Simulate recovery fine-tuning, then scatter only surviving tensors back
    # into the full-width gated carrier used by the next search cycle.
    with torch.no_grad():
        structural_model.backbone.layer2[0].conv2.weight.add_(0.25)
        structural_model.backbone.layer2[0].batch_norm2.bias.add_(0.1)
    next_gated_model = ClassificationFeatureSelectionWrapper(
        backbone=ResNet50(
            num_classes=5,
            in_channels=3,
            resnet_block=partial(
                MaskedGumbelBottleneckLayer,
                gate_internal_width=True,
                force_ones_mask=True,
            ),
            stem_kernel_size=3,
            stem_stride=1,
            stem_padding=1,
            use_maxpool=False,
        ),
        lambda_coef=0.0,
    )
    next_gated_model.load_state_dict(gated_model.state_dict(), strict=True)
    inactive_weight_before = (
        next_gated_model.backbone.layer2[0].conv2.weight[3].detach().clone()
    )
    transfer_structural_weights_to_gated(structural_model, next_gated_model)
    apply_channel_mask(next_gated_model, mask)
    next_gated_model.eval()

    with torch.no_grad():
        carried = next_gated_model(sample, targets).logits
        recovered = structural_model(sample, targets).logits
    torch.testing.assert_close(carried, recovered, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        next_gated_model.backbone.layer2[0].conv2.weight[3],
        inactive_weight_before,
    )
