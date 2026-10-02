"""ImageNet-pretrained MobileNetV2: weight loading for every block type, the
fine-tuning optimizer split, and the pretrained ablation configs.

The loader is exercised against a torchvision-format checkpoint with random
weights written to a temp dir, so no test touches the network.
"""

import math
from functools import partial
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

tv_models = pytest.importorskip("torchvision.models")

from net_complexity.models.channel_pruning import build_pruned_mobilenet_model
from net_complexity.models.feature_selection import (
    ClassificationFeatureSelectionWrapper,
    get_AIG_modules,
    get_gumbel_modules,
)
from net_complexity.models.mobilenet_v2 import (
    AIGInvertedResidual,
    MaskedGumbelInvertedResidual,
    MobileNetV2,
)
from net_complexity.models.pruned_mobilenet_v2 import PrunedMobileNetV2
from net_complexity.training.engine import _build_optimizer

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
PRETRAINED_FAMILY = "ablation_mobilenetv2_tinyimagenet200_pretrained"
PRETRAINED_EXPERIMENTS = [
    "plain_baseline.yaml",
    "aig_classic/mobilenetv2_tinyimagenet200.yaml",
    "aig_classic_static/mobilenetv2_tinyimagenet200.yaml",
    "depgraph/mobilenetv2_tinyimagenet200.yaml",
    "ours_non_iterative/mobilenetv2_tinyimagenet200.yaml",
    "ours_iterative_layers/mobilenetv2_tinyimagenet200.yaml",
    "ours_iterative_channels/mobilenetv2_tinyimagenet200.yaml",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@pytest.fixture(scope="module")
def torchvision_checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    tv = tv_models.mobilenet_v2(weights=None)
    # Non-trivial BN affine params and running stats, so a mis-mapped BN
    # tensor changes the forward pass.
    with torch.no_grad():
        for module in tv.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.2, 0.2)
                module.running_mean.uniform_(-0.5, 0.5)
                module.running_var.uniform_(0.5, 1.5)
    path = tmp_path_factory.mktemp("weights") / "mobilenet_v2_torchvision.pth"
    torch.save(tv.state_dict(), path)
    return tv.eval(), str(path)


def _open_all_gates(model):
    for module in model.modules():
        if module is not model and hasattr(module, "set_bypass"):
            module.set_bypass(True)


def _x(batch_size=2, size=96):
    torch.manual_seed(1)
    return torch.randn(batch_size, 3, size, size)


def test_plain_model_reproduces_torchvision_features(torchvision_checkpoint):
    tv, path = torchvision_checkpoint
    model = MobileNetV2(num_classes=200, pretrained_weights=path).eval()

    with torch.no_grad():
        assert torch.allclose(model.features(_x()), tv.features(_x()), atol=1e-5)
    # The 200-way head cannot come from ImageNet: it keeps its fresh init.
    assert torch.count_nonzero(model.classifier[-1].bias) == 0
    feature_params = {name for name, _ in model.named_parameters() if name.startswith("features.")}
    assert model.pretrained_parameter_names == feature_params


def test_thousand_class_model_also_loads_the_classifier(torchvision_checkpoint):
    tv, path = torchvision_checkpoint
    model = MobileNetV2(num_classes=1000, pretrained_weights=path).eval()

    with torch.no_grad():
        assert torch.allclose(model(_x()), tv(_x()), atol=1e-5)
    assert "classifier.1.weight" in model.pretrained_parameter_names


def test_default_is_random_init():
    model = MobileNetV2(num_classes=200)
    assert model.pretrained_weights is None
    assert model.pretrained_parameter_names == frozenset()


def test_aig_model_loads_branches_and_keeps_gate_init(torchvision_checkpoint):
    tv, path = torchvision_checkpoint
    model = MobileNetV2(
        num_classes=200, block=partial(AIGInvertedResidual), pretrained_weights=path
    ).eval()
    _open_all_gates(model)

    with torch.no_grad():
        assert torch.allclose(model.features(_x()), tv.features(_x()), atol=1e-5)
    for gate in get_AIG_modules(model).values():
        final_conv = gate.router[-1]
        assert final_conv.bias[1].item() == pytest.approx(math.log(0.85 / 0.15))
        assert final_conv.weight.std().item() < 0.01
    assert not any(".gate." in name for name in model.pretrained_parameter_names)


def test_aig_gate_init_survives_backbone_init_without_pretraining():
    model = MobileNetV2(num_classes=200, block=partial(AIGInvertedResidual, keep_prob_init=0.9))
    for gate in get_AIG_modules(model).values():
        final_conv = gate.router[-1]
        assert final_conv.bias[0].item() == 0.0
        assert final_conv.bias[1].item() == pytest.approx(math.log(0.9 / 0.1))
        assert final_conv.weight.std().item() < 0.01


def test_gumbel_model_with_internal_gates_loads_branches(torchvision_checkpoint):
    tv, path = torchvision_checkpoint
    model = MobileNetV2(
        num_classes=200,
        block=partial(MaskedGumbelInvertedResidual, gate_internal_width=True),
        pretrained_weights=path,
    ).eval()
    _open_all_gates(model)

    with torch.no_grad():
        assert torch.allclose(model.features(_x()), tv.features(_x()), atol=1e-5)
    assert len(get_gumbel_modules(model)) > 0
    assert not any("gumbel_layer" in name for name in model.pretrained_parameter_names)


def test_pruned_model_keeps_imagenet_values_of_surviving_channels(torchvision_checkpoint):
    tv, path = torchvision_checkpoint
    out_drop, mid_drop = [0, 5, 17], [1, 2, 300]
    spec = {
        "features.10": {"output": out_drop, "mid": mid_drop},
        "features.14": {"mid": [0, 7]},  # non-residual block: internal width only
    }
    model = PrunedMobileNetV2(spec, num_classes=200, pretrained_weights=path)

    block, tv_block = model.features[10], tv.features[10]
    out_keep = [c for c in range(block.oup) if c not in out_drop]
    mid_keep = [c for c in range(block.hidden_dim) if c not in mid_drop]

    tv_expand_conv, tv_expand_bn = tv_block.conv[0][0], tv_block.conv[0][1]
    tv_depthwise_conv = tv_block.conv[1][0]
    tv_project, tv_project_bn = tv_block.conv[2], tv_block.conv[3]
    assert torch.equal(block.branch.expand[0].weight, tv_expand_conv.weight[mid_keep])
    assert torch.equal(block.branch.expand[1].running_var, tv_expand_bn.running_var[mid_keep])
    assert torch.equal(block.branch.depthwise[0].weight, tv_depthwise_conv.weight[mid_keep])
    assert torch.equal(block.branch.project.weight, tv_project.weight[out_keep][:, mid_keep])
    assert torch.equal(block.branch.project_bn.running_mean, tv_project_bn.running_mean[out_keep])

    # An untouched block and the stem come through unchanged.
    assert torch.equal(model.features[5].branch.project.weight, tv.features[5].conv[2].weight)
    assert torch.equal(model.features[0][0].weight, tv.features[0][0].weight)
    assert model(_x()).shape == (2, 200)


def test_structural_recovery_builder_passes_pretrained_weights(torchvision_checkpoint):
    tv, path = torchvision_checkpoint
    config = OmegaConf.create({
        "model": {
            "backbone": {
                "_target_": "net_complexity.wrappers.MobileNetV2",
                "num_classes": 200,
                "stem_stride": 2,
                "pretrained_weights": path,
            },
            "lambda_coef": 0.0,
        }
    })
    model = build_pruned_mobilenet_model(config, {"features.10": {"output": [0, 1]}})

    assert torch.equal(model.backbone.features[3].branch.expand[0].weight, tv.features[3].conv[0][0].weight)
    assert model.backbone.pretrained_parameter_names


def test_loader_rejects_incompatible_checkpoints_and_models(torchvision_checkpoint, tmp_path):
    _, path = torchvision_checkpoint
    with pytest.raises(ValueError, match="width_mult=1.0"):
        MobileNetV2(num_classes=200, width_mult=0.5, pretrained_weights=path)
    with pytest.raises(FileNotFoundError, match="IMAGENET1K_V1"):
        MobileNetV2(num_classes=200, pretrained_weights=str(tmp_path / "missing.pth"))

    # A training checkpoint of this repository is not a torchvision state_dict.
    repo_style = tmp_path / "repo_checkpoint.pth"
    torch.save({"backbone.features.0.0.weight": torch.zeros(1)}, repo_style)
    with pytest.raises(ValueError, match="torchvision"):
        MobileNetV2(num_classes=200, pretrained_weights=str(repo_style))


def test_paper_init_cannot_overwrite_pretrained_weights(torchvision_checkpoint):
    _, path = torchvision_checkpoint
    with pytest.raises(ValueError, match="pretrained"):
        ClassificationFeatureSelectionWrapper(
            backbone=MobileNetV2(num_classes=200, pretrained_weights=path),
            backbone_weight_init="paper_kaiming_normal",
        )


def _gated_pretrained_wrapper(path):
    backbone = MobileNetV2(
        num_classes=200,
        block=partial(MaskedGumbelInvertedResidual, gate_internal_width=True),
        pretrained_weights=path,
    )
    return ClassificationFeatureSelectionWrapper(backbone=backbone, lambda_coef=1e-4)


def test_optimizer_gives_pretrained_parameters_a_scaled_lr(torchvision_checkpoint):
    _, path = torchvision_checkpoint
    model = _gated_pretrained_wrapper(path)
    config = OmegaConf.create({
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1e-3,
            "weight_decay": 5e-4,
            "pretrained_lr_scale": 0.1,
        }
    })
    optimizer, info = _build_optimizer(config, model)

    new_group, pretrained_group = optimizer.param_groups
    assert new_group["lr"] == pytest.approx(1e-3)
    assert pretrained_group["lr"] == pytest.approx(1e-4)
    assert pretrained_group["name"] == "pretrained"

    pretrained_ids = {id(p) for p in pretrained_group["params"]}
    backbone = model.backbone
    assert pretrained_ids == {id(backbone.get_parameter(n)) for n in backbone.pretrained_parameter_names}
    # Gates and the fresh classifier stay at the base lr.
    new_ids = {id(p) for p in new_group["params"]}
    assert id(backbone.classifier[-1].weight) in new_ids
    assert all(id(m.logits) in new_ids for m in get_gumbel_modules(backbone).values())
    assert info.pretrained_lr_scale == 0.1
    assert info.num_pretrained_param_tensors == len(pretrained_ids)


def test_optimizer_combines_pretrained_and_gate_weight_decay_groups(torchvision_checkpoint):
    _, path = torchvision_checkpoint
    model = _gated_pretrained_wrapper(path)
    config = OmegaConf.create({
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1e-3,
            "weight_decay": 5e-4,
            "pretrained_lr_scale": 0.1,
            "gate_weight_decay_scale": 1.0,
        }
    })
    optimizer, info = _build_optimizer(config, model)

    lrs = [group["lr"] for group in optimizer.param_groups]
    assert lrs == pytest.approx([1e-3, 1e-4, 1e-3])
    assert info.gate_param_group_enabled
    grouped = sum(len(group["params"]) for group in optimizer.param_groups)
    assert grouped == sum(1 for p in model.parameters() if p.requires_grad)


def test_optimizer_without_the_new_key_is_unchanged(torchvision_checkpoint):
    _, path = torchvision_checkpoint
    model = _gated_pretrained_wrapper(path)
    config = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-3, "weight_decay": 5e-4}
    })
    optimizer, info = _build_optimizer(config, model)

    assert len(optimizer.param_groups) == 1
    assert info.pretrained_lr_scale is None


# --------------------------------------------------------------------- configs


def test_pretrained_model_data_and_optimizer_presets():
    model_cfg = OmegaConf.load(CONFIGS_DIR / "model" / "mobilenetv2_tinyimagenet200_224_pretrained.yaml")
    backbone = model_cfg.model.backbone
    assert backbone.pretrained_weights == "IMAGENET1K_V1"
    assert backbone.stem_stride == 2
    assert backbone.num_classes == 200

    data_cfg = OmegaConf.load(CONFIGS_DIR / "data" / "tinyimagenet200_resize224_imagenet_norm.yaml")
    for pipeline in ("train_transform", "eval_transform"):
        normalize = data_cfg.dataloaders[pipeline].transforms[-1]
        assert normalize._target_ == "torchvision.transforms.Normalize"
        assert list(normalize.mean) == IMAGENET_MEAN
        assert list(normalize.std) == IMAGENET_STD
    eval_ops = [op._target_.rsplit(".", 1)[-1] for op in data_cfg.dataloaders.eval_transform.transforms]
    assert eval_ops[:2] == ["Resize", "CenterCrop"]

    optimizer_cfg = OmegaConf.load(CONFIGS_DIR / "optimizer" / "adamw_finetune.yaml")
    assert optimizer_cfg.optimizer.pretrained_lr_scale == 0.1


@pytest.mark.parametrize("relative_path", PRETRAINED_EXPERIMENTS)
def test_pretrained_family_shares_model_data_and_optimizer(relative_path):
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / PRETRAINED_FAMILY / relative_path)
    defaults = OmegaConf.to_container(cfg.defaults, resolve=True)

    assert {"/data": "tinyimagenet200_resize224_imagenet_norm"} in defaults
    assert {"/model": "mobilenetv2_tinyimagenet200_224_pretrained"} in defaults
    assert {"/optimizer": "adamw_finetune"} in defaults
    assert {"/metrics": "full"} in defaults


def test_pretrained_depgraph_traces_at_training_resolution():
    cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / PRETRAINED_FAMILY / "depgraph" / "mobilenetv2_tinyimagenet200.yaml"
    )
    assert list(cfg.depgraph_pruning.example_input_size) == [3, 224, 224]


@pytest.mark.parametrize(
    ("relative_path", "config_name"),
    [
        ("plain_baseline", "train"),
        ("aig_classic/mobilenetv2_tinyimagenet200", "train"),
        ("aig_classic_static/mobilenetv2_tinyimagenet200", "train"),
        ("depgraph/mobilenetv2_tinyimagenet200", "train"),
        ("ours_non_iterative/mobilenetv2_tinyimagenet200", "train"),
        ("ours_iterative_layers/mobilenetv2_tinyimagenet200", "cyclic_train"),
        ("ours_iterative_channels/mobilenetv2_tinyimagenet200", "cyclic_channel_train"),
    ],
)
def test_pretrained_experiments_compose(relative_path, config_name):
    hydra = pytest.importorskip("hydra")
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = hydra.compose(
            config_name=config_name,
            overrides=[f"experiment={PRETRAINED_FAMILY}/{relative_path}"],
        )

    backbone = cfg.model.backbone
    assert backbone._target_ == "net_complexity.wrappers.MobileNetV2"
    assert backbone.pretrained_weights == "IMAGENET1K_V1"
    assert cfg.optimizer.pretrained_lr_scale == 0.1
    assert cfg.optimizer.lr == 1e-3
    assert cfg.dataloaders.eval_transform.transforms[-1].mean == IMAGENET_MEAN
