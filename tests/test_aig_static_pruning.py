import json
from functools import partial

import pytest
import torch
from omegaconf import OmegaConf

from net_complexity.models import aig_static_pruning as asp
from net_complexity.models.aig_static_pruning import build_static_aig_model_from_config
from net_complexity.models.feature_selection import (
    AIGBottleneckLayer,
    ClassificationFeatureSelectionWrapper,
    PrunedBottleneck,
    ResNet50,
)


def _make_backbone(num_classes=5):
    return ResNet50(
        num_classes=num_classes,
        in_channels=3,
        resnet_block=partial(AIGBottleneckLayer, temperature=1.0),
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
    )


def _write_run_dir(tmp_path, model, layer_g_probs, checkpoint_name="best.pt"):
    run_dir = tmp_path / "run"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 3},
        checkpoints_dir / checkpoint_name,
    )

    valid_metrics = {
        f"valid_g_prob_backbone.{name}": prob for name, prob in layer_g_probs.items()
    }
    summary = {
        "best_valid": {"epoch": 3, "metric": "valid_accuracy", "value": 0.5, "metrics": valid_metrics},
        "final_valid": valid_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def _make_config(source_run_dir, **cfg_overrides):
    cfg = {
        "source_run_dir": str(source_run_dir),
        "checkpoint_name": "best.pt",
        "g_prob_threshold": 0.5,
    }
    cfg.update(cfg_overrides)
    return OmegaConf.create(
        {
            "model": {
                "_target_": "net_complexity.wrappers.ClassificationFeatureSelectionWrapper",
                "lambda_coef": 0.0,
                "criterion": {"_target_": "torch.nn.CrossEntropyLoss"},
                "backbone": {
                    "_target_": "net_complexity.wrappers.ResNet50",
                    "num_classes": 5,
                    "in_channels": 3,
                    "stem_kernel_size": 3,
                    "stem_stride": 1,
                    "stem_padding": 1,
                    "use_maxpool": False,
                    "resnet_block": {
                        "_target_": "net_complexity.wrappers.AIGBottleneckLayer",
                        "_partial_": True,
                        "temperature": 1.0,
                    },
                },
            },
        }
    ), OmegaConf.create(cfg)


def test_build_static_aig_model_prunes_low_gprob_blocks_and_bypasses_survivors(tmp_path):
    backbone = _make_backbone()
    source_model = ClassificationFeatureSelectionWrapper(
        backbone=backbone, lambda_coef=0.0, criterion=torch.nn.CrossEntropyLoss(),
    )
    run_dir = _write_run_dir(
        tmp_path, source_model, layer_g_probs={"layer1.0": 0.1, "layer1.1": 0.9},
    )
    config, cfg = _make_config(run_dir, g_prob_threshold=0.5)

    pruned_model = build_static_aig_model_from_config(config, cfg, device="cpu")

    assert isinstance(pruned_model.backbone.layer1[0], PrunedBottleneck)
    assert isinstance(pruned_model.backbone.layer1[1], AIGBottleneckLayer)
    assert pruned_model.backbone.layer1[1].bypass is True
    assert pruned_model.backbone.layer1[1].gate.bypass is True

    output = pruned_model(torch.randn(2, 3, 16, 16), torch.tensor([0, 1]))
    assert output.logits.shape == (2, 5)


def test_build_static_aig_model_keeps_everything_when_all_g_probs_above_threshold(tmp_path):
    backbone = _make_backbone()
    source_model = ClassificationFeatureSelectionWrapper(
        backbone=backbone, lambda_coef=0.0, criterion=torch.nn.CrossEntropyLoss(),
    )
    run_dir = _write_run_dir(tmp_path, source_model, layer_g_probs={"layer1.0": 0.9})
    config, cfg = _make_config(run_dir, g_prob_threshold=0.5)

    pruned_model = build_static_aig_model_from_config(config, cfg, device="cpu")

    assert isinstance(pruned_model.backbone.layer1[0], AIGBottleneckLayer)
    assert pruned_model.backbone.layer1[0].bypass is True


def test_build_static_aig_model_requires_source_run_dir(tmp_path):
    config, _ = _make_config(tmp_path / "unused")
    cfg = OmegaConf.create({"g_prob_threshold": 0.5})

    with pytest.raises(ValueError, match="source_run_dir"):
        build_static_aig_model_from_config(config, cfg, device="cpu")


def test_build_static_aig_model_missing_checkpoint_raises(tmp_path):
    config, cfg = _make_config(tmp_path / "does_not_exist")

    with pytest.raises(FileNotFoundError):
        build_static_aig_model_from_config(config, cfg, device="cpu")


def test_read_block_g_probs_raises_without_summary_json(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="summary.json"):
        asp._read_block_g_probs(run_dir, "best.pt")


def test_read_block_g_probs_uses_best_valid_or_final_valid_matching_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    summary = {
        "best_valid": {"metrics": {"valid_g_prob_backbone.layer1.0": 0.9}},
        "final_valid": {"valid_g_prob_backbone.layer1.0": 0.1},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    assert asp._read_block_g_probs(run_dir, "best.pt") == {"layer1.0": 0.9}
    assert asp._read_block_g_probs(run_dir, "last.pt") == {"layer1.0": 0.1}


def test_read_block_g_probs_raises_when_no_gprob_keys_present(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    summary = {"best_valid": {"metrics": {"valid_accuracy": 0.9}}}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="valid_g_prob_layer"):
        asp._read_block_g_probs(run_dir, "best.pt")


def test_layers_to_drop_selects_names_below_threshold():
    layer_g_probs = {"layer1.0": 0.1, "layer1.1": 0.9}

    assert asp._layers_to_drop(layer_g_probs, threshold=0.5) == ["layer1.0"]
