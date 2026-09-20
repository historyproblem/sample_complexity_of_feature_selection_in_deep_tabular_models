from functools import partial
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from net_complexity.models.channel_pruning import (
    _is_mobilenet_backbone_target,
    _mask_dict_to_mobilenet_pruning_spec,
    apply_channel_mask,
    build_structurally_pruned_model_from_config,
)
from net_complexity.models.feature_selection import (
    ClassificationFeatureSelectionWrapper,
    get_AIG_modules,
    get_gumbel_modules,
)
from net_complexity.models.layer_skipping import apply_layer_skipping
from net_complexity.models.mobilenet_v2 import (
    AIGInvertedResidual,
    InvertedResidual,
    MaskedGumbelInvertedResidual,
    MobileNetV2,
    MobileNetV2TinyImageNet200,
    PrunedInvertedResidual,
    SkippedInvertedResidual,
    _make_divisible,
)
from net_complexity.models.pruned_mobilenet_v2 import (
    PrunedGumbelInvertedResidual,
    PrunedMobileNetV2,
)
from net_complexity.training.cyclic_aig import _extract_layer_g_probs, _pruneable_param_counts
from net_complexity.training.cyclic_channel_pruning import _channel_param_cost

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
MOBILENET_FAMILY = "ablation_mobilenetv2_tinyimagenet200"

# The stock MobileNetV2 table has 17 inverted-residual blocks; 10 of them have
# a skip connection (stride == 1 and in == out) and are therefore gateable.
NUM_GATED_BLOCKS = 10
GATED_BLOCK_INDEX = 10  # features.10: 64 -> 64, stride 1
UNGATED_BLOCK_INDEX = 14  # features.14: 96 -> 160, stride 2


def _backbone(block=None, num_classes=200):
    kwargs = {"num_classes": num_classes}
    if block is not None:
        kwargs["block"] = block
    return MobileNetV2TinyImageNet200(**kwargs)


def _inputs(batch_size=2):
    return torch.randn(batch_size, 3, 64, 64), torch.zeros(batch_size, dtype=torch.long)


def test_make_divisible_matches_torchvision_rounding():
    assert _make_divisible(32 * 1.0) == 32
    assert _make_divisible(16 * 0.5) == 8
    # Never round down by more than 10%.
    assert _make_divisible(4) == 8


def test_plain_mobilenet_v2_forward_and_stem_stride():
    model = _backbone()
    x, _ = _inputs()

    assert model(x).shape == (2, 200)
    # TinyImageNet preset drops the stem stride to keep a 4x4 final map.
    assert model.features[0][0].stride == (1, 1)
    assert MobileNetV2(num_classes=10).features[0][0].stride == (2, 2)


def test_aig_mobilenet_gates_only_residual_blocks():
    model = _backbone(block=partial(AIGInvertedResidual))
    x, _ = _inputs()

    gates = get_AIG_modules(model)
    assert len(gates) == NUM_GATED_BLOCKS
    assert all(name.endswith(".gate") for name in gates)
    assert f"features.{GATED_BLOCK_INDEX}.gate" in gates

    assert model.features[GATED_BLOCK_INDEX].gate is not None
    assert model.features[GATED_BLOCK_INDEX].use_res_connect is True
    assert model.features[UNGATED_BLOCK_INDEX].gate is None
    assert model.features[UNGATED_BLOCK_INDEX].use_res_connect is False

    assert model(x).shape == (2, 200)


def test_masked_gumbel_mobilenet_gates_block_output_only_on_residual_blocks():
    model = _backbone(block=partial(MaskedGumbelInvertedResidual))

    gumbel_modules = get_gumbel_modules(model)
    assert len(gumbel_modules) == NUM_GATED_BLOCKS
    assert f"features.{GATED_BLOCK_INDEX}.gumbel_layer" in gumbel_modules

    block = model.features[GATED_BLOCK_INDEX]
    # The gate covers the block's residual output width (oup), not the
    # inverted bottleneck's internal width.
    assert block.gumbel_layer.channel_mask.shape[0] == block.oup
    assert model.features[UNGATED_BLOCK_INDEX].gumbel_layer is None


def test_aig_mobilenet_wrapper_drives_entropy_regularization():
    model = ClassificationFeatureSelectionWrapper(
        backbone=_backbone(block=partial(AIGInvertedResidual, gate_regularization="l1_probability")),
        lambda_coef=1e-4,
        entropy_regularization="plus_negative_entropy",
        entropy_regularization_coef=0.1,
    )
    x, y = _inputs()

    output = model(x, y)

    assert output.logits.shape == (2, 200)
    assert output.mean_p_open is not None
    assert output.negative_entropy is not None


def test_layer_skipping_prunes_and_skips_residual_mobilenet_blocks():
    wrapper = ClassificationFeatureSelectionWrapper(
        backbone=_backbone(block=partial(AIGInvertedResidual)), lambda_coef=0.0
    )
    before = sum(p.numel() for p in wrapper.parameters())

    apply_layer_skipping(wrapper, [f"features.{GATED_BLOCK_INDEX}"], mode="prune")

    assert isinstance(wrapper.backbone.features[GATED_BLOCK_INDEX], PrunedInvertedResidual)
    assert sum(p.numel() for p in wrapper.parameters()) < before

    x, y = _inputs()
    assert wrapper(x, y).logits.shape == (2, 200)


def test_layer_skipping_skip_mode_keeps_mobilenet_block_weights():
    backbone = _backbone(block=partial(AIGInvertedResidual))
    before = sum(p.numel() for p in backbone.parameters())

    apply_layer_skipping(backbone, [f"features.{GATED_BLOCK_INDEX}"], mode="skip")

    assert isinstance(backbone.features[GATED_BLOCK_INDEX], SkippedInvertedResidual)
    assert sum(p.numel() for p in backbone.parameters()) == before
    assert backbone(torch.randn(2, 3, 64, 64)).shape == (2, 200)


def test_layer_skipping_refuses_non_residual_mobilenet_block():
    backbone = _backbone(block=partial(AIGInvertedResidual))

    apply_layer_skipping(backbone, [f"features.{UNGATED_BLOCK_INDEX}"], mode="prune")

    # Untouched: dropping it would change the next block's input width.
    assert isinstance(backbone.features[UNGATED_BLOCK_INDEX], InvertedResidual)
    assert not isinstance(backbone.features[UNGATED_BLOCK_INDEX], PrunedInvertedResidual)


def test_extract_layer_g_probs_parses_mobilenet_gate_paths():
    probs = _extract_layer_g_probs(
        {
            "valid_g_prob_backbone.features.10.gate": 0.05,
            "valid_g_prob_backbone.features.13.gate": 0.9,
            "valid_g_prob_backbone.layer2.0": 0.4,  # ResNet form still works
            "valid_accuracy": 0.42,
        }
    )

    assert probs == {"features.10": 0.05, "features.13": 0.9, "layer2.0": 0.4}


def test_pruneable_param_counts_lists_exactly_the_gated_mobilenet_blocks():
    config = OmegaConf.create(
        {
            "model": {
                "_target_": "net_complexity.wrappers.ClassificationFeatureSelectionWrapper",
                "lambda_coef": 0.0,
                "backbone": {
                    "_target_": "net_complexity.wrappers.MobileNetV2TinyImageNet200",
                    "num_classes": 200,
                    "block": {
                        "_target_": "net_complexity.wrappers.AIGInvertedResidual",
                        "_partial_": True,
                    },
                },
            },
            "layer_skipping": {"enabled": True, "disabled_layers": []},
        }
    )

    freeable, total_params = _pruneable_param_counts(config)

    assert len(freeable) == NUM_GATED_BLOCKS
    assert all(key.startswith("features.") for key in freeable)
    # The stem/head ConvBNReLU entries of `features` are not droppable blocks.
    assert "features.0" not in freeable
    assert f"features.{UNGATED_BLOCK_INDEX}" not in freeable
    assert total_params > 0


def test_channel_param_cost_uses_mobilenet_project_conv():
    backbone = _backbone(block=partial(MaskedGumbelInvertedResidual))
    wrapper = ClassificationFeatureSelectionWrapper(backbone=backbone, lambda_coef=0.0)

    cost = _channel_param_cost(wrapper, f"backbone.features.{GATED_BLOCK_INDEX}.gumbel_layer")

    project = backbone.features[GATED_BLOCK_INDEX].branch.project
    # One freed output row of `project` (bias-free) + project_bn affine.
    assert cost == project.in_channels + 2


def test_channel_param_cost_rejects_internal_width_gate_on_mobilenet():
    backbone = _backbone(block=partial(MaskedGumbelInvertedResidual))
    wrapper = ClassificationFeatureSelectionWrapper(backbone=backbone, lambda_coef=0.0)

    with pytest.raises(ValueError, match="no MobileNetV2 analogue"):
        _channel_param_cost(wrapper, f"backbone.features.{GATED_BLOCK_INDEX}.mid1_gumbel_layer")


def test_mask_dict_to_mobilenet_pruning_spec():
    spec = _mask_dict_to_mobilenet_pruning_spec(
        {
            "backbone.features.10.gumbel_layer": [0, 1],
            "features.13.gumbel_layer": [5],
            "not_a_gate_path": [9],
        }
    )

    assert spec == {
        "features.10": {"output": [0, 1]},
        "features.13": {"output": [5]},
    }


def test_is_mobilenet_backbone_target():
    assert _is_mobilenet_backbone_target("net_complexity.wrappers.MobileNetV2TinyImageNet200")
    assert _is_mobilenet_backbone_target("net_complexity.wrappers.MobileNetV2")
    assert not _is_mobilenet_backbone_target("net_complexity.wrappers.ResNet50")


def test_apply_channel_mask_reaches_mobilenet_gumbel_layers():
    backbone = _backbone(block=partial(MaskedGumbelInvertedResidual))

    apply_channel_mask(backbone, {f"features.{GATED_BLOCK_INDEX}.gumbel_layer": [0, 3]})

    mask = backbone.features[GATED_BLOCK_INDEX].gumbel_layer.channel_mask
    assert mask[0].item() == 0.0
    assert mask[3].item() == 0.0
    assert mask[1].item() == 1.0


def test_pruned_gumbel_inverted_residual_disabled_channels_come_from_shortcut_only():
    torch.manual_seed(0)
    block = PrunedGumbelInvertedResidual(
        inp=16, oup=16, stride=1, expand_ratio=6, disabled_channels=[0, 5],
    )
    block.eval()

    assert block.n_active == 14
    assert block.branch.project.out_channels == 14
    assert block.branch.project_bn.num_features == 14

    x = torch.randn(2, 16, 8, 8)
    out = block(x)

    assert out.shape == (2, 16, 8, 8)
    torch.testing.assert_close(out[:, 0], x[:, 0])
    torch.testing.assert_close(out[:, 5], x[:, 5])


def test_pruned_gumbel_inverted_residual_rejects_non_residual_and_empty_block():
    with pytest.raises(ValueError, match="residual block"):
        PrunedGumbelInvertedResidual(inp=16, oup=24, stride=1, expand_ratio=6, disabled_channels=[0])

    with pytest.raises(ValueError, match="All 4 channels are disabled"):
        PrunedGumbelInvertedResidual(
            inp=4, oup=4, stride=1, expand_ratio=6, disabled_channels=[0, 1, 2, 3]
        )


def test_pruned_mobilenet_v2_accepts_flat_and_nested_spec():
    model = PrunedMobileNetV2(
        pruning_spec={
            f"features.{GATED_BLOCK_INDEX}": [0, 1, 2],
            "features.13": {"output": [5]},
        },
        num_classes=200,
        stem_stride=1,
    )
    model.eval()

    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 200)
    assert model.features[GATED_BLOCK_INDEX].n_active == model.features[GATED_BLOCK_INDEX].oup - 3
    assert model.features[13].n_active == model.features[13].oup - 1
    # Untouched blocks keep every channel.
    assert model.features[12].n_active == model.features[12].oup


def test_internal_width_gate_covers_non_residual_blocks_too():
    model = _backbone(block=partial(MaskedGumbelInvertedResidual, gate_internal_width=True))

    gates = get_gumbel_modules(model)
    output_gates = [n for n in gates if n.endswith(".gumbel_layer") and ".mid_" not in n]
    internal_gates = [n for n in gates if n.endswith(".mid_gumbel_layer")]

    assert len(output_gates) == NUM_GATED_BLOCKS
    # Every inverted-residual block except features.1 (expand_ratio == 1).
    assert len(internal_gates) == 16
    assert f"features.{UNGATED_BLOCK_INDEX}.mid_gumbel_layer" in internal_gates
    assert model.features[UNGATED_BLOCK_INDEX].gumbel_layer is None

    block = model.features[GATED_BLOCK_INDEX]
    assert block.mid_gumbel_layer.channel_mask.shape[0] == block.hidden_dim

    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 200)


def test_expand_ratio_one_block_has_no_internal_gate():
    model = _backbone(block=partial(MaskedGumbelInvertedResidual, gate_internal_width=True))

    # features.1 is the single expand_ratio == 1 block: hidden == inp, so its
    # "internal" width is the block input and narrowing it is not local.
    assert model.features[1].has_expand is False
    assert model.features[1].mid_gumbel_layer is None


def test_internal_width_soft_mask_equals_physical_pruning():
    """The masked block and the narrowed block must compute the same thing.

    This is what justifies the gate's placement after `depthwise` rather than
    after `expand`: `project` is a linear 1x1 conv, so a zero input channel
    contributes nothing and removing it is exact.
    """
    torch.manual_seed(0)
    disabled = [0, 3, 7]
    # Deliberately a NON-residual block (16 != 24) — unreachable by the output gate.
    masked = MaskedGumbelInvertedResidual(
        inp=16,
        oup=24,
        stride=1,
        expand_ratio=6,
        gate_internal_width=True,
        deterministic_hard_mask=True,
    )
    masked.eval()
    with torch.no_grad():
        for channel in disabled:
            masked.mid_gumbel_layer.channel_mask[channel] = 0.0

    pruned = PrunedGumbelInvertedResidual(
        inp=16, oup=24, stride=1, expand_ratio=6, disabled_mid_channels=disabled,
    )
    keep = [c for c in range(masked.hidden_dim) if c not in disabled]
    with torch.no_grad():
        pruned.branch.expand[0].weight.copy_(masked.branch.expand[0].weight[keep])
        pruned.branch.depthwise[0].weight.copy_(masked.branch.depthwise[0].weight[keep])
        pruned.branch.project.weight.copy_(masked.branch.project.weight[:, keep])
        for source, target, index in (
            (masked.branch.expand[1], pruned.branch.expand[1], keep),
            (masked.branch.depthwise[1], pruned.branch.depthwise[1], keep),
            (masked.branch.project_bn, pruned.branch.project_bn, slice(None)),
        ):
            for attr in ("weight", "bias", "running_mean", "running_var"):
                getattr(target, attr).copy_(getattr(source, attr)[index])
    pruned.eval()

    x = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(masked(x), pruned(x), atol=1e-5, rtol=1e-4)

    assert pruned.n_mid_active == masked.hidden_dim - len(disabled)
    assert sum(p.numel() for p in pruned.branch.parameters()) < sum(
        p.numel() for p in masked.branch.parameters()
    )


def test_channel_param_cost_for_the_internal_width_gate():
    backbone = _backbone(block=partial(MaskedGumbelInvertedResidual, gate_internal_width=True))
    wrapper = ClassificationFeatureSelectionWrapper(backbone=backbone, lambda_coef=0.0)

    cost = _channel_param_cost(wrapper, f"backbone.features.{UNGATED_BLOCK_INDEX}.mid_gumbel_layer")

    block = backbone.features[UNGATED_BLOCK_INDEX]
    # expand row (inp, bias-free) + expand BN (2) + depthwise kernel (3*3)
    # + depthwise BN (2) + project's freed input column (oup).
    assert cost == block.inp + 2 + 9 + 2 + block.oup


def test_pruned_gumbel_inverted_residual_narrows_all_three_internal_tensors():
    block = PrunedGumbelInvertedResidual(
        inp=16, oup=24, stride=1, expand_ratio=6, disabled_mid_channels=[0, 1],
    )

    assert block.n_mid_active == 96 - 2
    assert block.branch.expand[0].out_channels == 94
    assert block.branch.depthwise[0].in_channels == 94
    assert block.branch.depthwise[0].out_channels == 94
    assert block.branch.depthwise[0].groups == 94
    assert block.branch.project.in_channels == 94
    # The output boundary is untouched, so the block's output width stands.
    assert block.branch.project.out_channels == 24


def test_pruned_gumbel_inverted_residual_rejects_internal_pruning_without_expand():
    with pytest.raises(ValueError, match="expand conv"):
        PrunedGumbelInvertedResidual(
            inp=32, oup=16, stride=1, expand_ratio=1, disabled_mid_channels=[0],
        )


def test_mask_dict_to_mobilenet_pruning_spec_groups_both_gates():
    spec = _mask_dict_to_mobilenet_pruning_spec(
        {
            "backbone.features.10.gumbel_layer": [0, 1],
            "backbone.features.10.mid_gumbel_layer": [2, 3],
            "backbone.features.14.mid_gumbel_layer": [5],
        }
    )

    assert spec == {
        "features.10": {"output": [0, 1], "mid": [2, 3]},
        "features.14": {"mid": [5]},
    }


def test_pruned_mobilenet_v2_prunes_both_boundaries_including_non_residual():
    model = PrunedMobileNetV2(
        pruning_spec={
            f"features.{GATED_BLOCK_INDEX}": {"output": [0, 1], "mid": [2, 3]},
            f"features.{UNGATED_BLOCK_INDEX}": {"mid": [5]},
        },
        num_classes=200,
        stem_stride=1,
    )
    model.eval()

    gated = model.features[GATED_BLOCK_INDEX]
    assert gated.n_active == gated.oup - 2
    assert gated.n_mid_active == gated.hidden_dim - 2

    non_residual = model.features[UNGATED_BLOCK_INDEX]
    assert non_residual.n_active == non_residual.oup  # output boundary untouched
    assert non_residual.n_mid_active == non_residual.hidden_dim - 1

    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 200)


def test_pruned_mobilenet_v2_rejects_unknown_spec_boundary():
    with pytest.raises(ValueError, match="'output' and/or"):
        PrunedMobileNetV2(pruning_spec={"features.10": {"mid1": [0]}}, num_classes=10)


def test_pruned_mobilenet_v2_rejects_bad_spec_keys():
    with pytest.raises(ValueError, match="features.N"):
        PrunedMobileNetV2(pruning_spec={"layer2.0": [0]}, num_classes=10)


def test_build_structurally_pruned_model_dispatches_to_mobilenet():
    config = OmegaConf.create(
        {
            "model": {
                "lambda_coef": 0.0,
                "backbone": {
                    "_target_": "net_complexity.wrappers.MobileNetV2TinyImageNet200",
                    "num_classes": 200,
                    "in_channels": 3,
                },
            }
        }
    )
    pruning_cfg = OmegaConf.create(
        {
            "enabled": True,
            "structural": True,
            "mode": "explicit",
            "mask": {f"backbone.features.{GATED_BLOCK_INDEX}.gumbel_layer": [0, 1, 2]},
        }
    )

    model = build_structurally_pruned_model_from_config(config, pruning_cfg)

    assert isinstance(model.backbone, PrunedMobileNetV2)
    block = model.backbone.features[GATED_BLOCK_INDEX]
    assert block.n_active == block.oup - 3

    x, y = _inputs()
    assert model(x, y).logits.shape == (2, 200)


@pytest.mark.parametrize(
    "relative_path",
    [
        "plain_baseline.yaml",
        "aig_classic/mobilenetv2_tinyimagenet200.yaml",
        "aig_classic_static/mobilenetv2_tinyimagenet200.yaml",
        "depgraph/mobilenetv2_tinyimagenet200.yaml",
        "ours_non_iterative/mobilenetv2_tinyimagenet200.yaml",
        "ours_iterative_layers/mobilenetv2_tinyimagenet200.yaml",
        "ours_iterative_channels/mobilenetv2_tinyimagenet200.yaml",
    ],
)
def test_mobilenet_ablation_configs_use_the_shared_dataset_and_model(relative_path):
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / MOBILENET_FAMILY / relative_path)

    defaults = OmegaConf.to_container(cfg.defaults, resolve=True)
    assert {"/data": "tinyimagenet200_best_practice"} in defaults
    assert {"/model": "mobilenetv2_tinyimagenet200"} in defaults
    assert {"/metrics": "full"} in defaults


def test_mobilenet_ablation_configs_pick_the_expected_methods():
    family_dir = CONFIGS_DIR / "experiment" / MOBILENET_FAMILY
    expected = {
        "plain_baseline.yaml": "plain",
        "aig_classic/mobilenetv2_tinyimagenet200.yaml": "aig_target_rate_mobilenetv2",
        "aig_classic_static/mobilenetv2_tinyimagenet200.yaml": "aig_target_rate_mobilenetv2",
        "depgraph/mobilenetv2_tinyimagenet200.yaml": "plain",
        "ours_non_iterative/mobilenetv2_tinyimagenet200.yaml": "aig_mobilenetv2",
        "ours_iterative_layers/mobilenetv2_tinyimagenet200.yaml": "aig_mobilenetv2",
        "ours_iterative_channels/mobilenetv2_tinyimagenet200.yaml": "gumbel_masked_mobilenetv2",
    }

    for relative_path, method in expected.items():
        cfg = OmegaConf.load(family_dir / relative_path)
        defaults = OmegaConf.to_container(cfg.defaults, resolve=True)
        assert {"/method": method} in defaults, relative_path


def test_mobilenet_method_configs_override_the_block_not_resnet_block():
    for name in ("aig_target_rate_mobilenetv2", "aig_mobilenetv2", "gumbel_masked_mobilenetv2"):
        cfg = OmegaConf.load(CONFIGS_DIR / "method" / f"{name}.yaml")
        assert "block" in cfg.model.backbone, name
        assert "resnet_block" not in cfg.model.backbone, name
        assert cfg.model.backbone.block._partial_ is True, name
