from functools import partial

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from net_complexity.models.channel_pruning import (
    apply_channel_mask,
    build_structurally_pruned_model_from_config,
)
from net_complexity.models.feature_selection import (
    CIFARMaskedGumbelBasicBlock,
    CIFARResNet20,
    ClassificationFeatureSelectionWrapper,
    MaskedGumbelLayer,
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
