import pytest
import torch
from omegaconf import OmegaConf

from net_complexity.models import depgraph_pruning as dg
from net_complexity.models.depgraph_pruning import (
    _progressive_prune_to_target_speedup,
    _run_sparsity_learning_phase,
    build_depgraph_pruned_model_from_config,
)
from net_complexity.models.feature_selection import (
    ClassificationFeatureSelectionWrapper,
    ResNet50,
)

import torch_pruning as tp


def _make_backbone(num_classes=5):
    return ResNet50(
        num_classes=num_classes,
        in_channels=3,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
    )


def _make_config(source_checkpoint, **cfg_overrides):
    cfg = {
        "source_checkpoint": str(source_checkpoint),
        "importance": "magnitude_l2",
        "target_speedup": 1.3,
        "iterative_steps": 50,
        "global_pruning": True,
        "example_input_size": [3, 16, 16],
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
                },
            },
        }
    ), OmegaConf.create(cfg)


def _write_plain_checkpoint(tmp_path):
    backbone = _make_backbone()
    model = ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=0.0,
        criterion=torch.nn.CrossEntropyLoss(),
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, checkpoint_path)
    return checkpoint_path


def test_build_depgraph_pruned_model_reduces_params_and_runs_forward(tmp_path):
    checkpoint_path = _write_plain_checkpoint(tmp_path)
    config, cfg = _make_config(checkpoint_path)

    original_params = sum(p.numel() for p in _make_backbone().parameters())

    pruned_model = build_depgraph_pruned_model_from_config(config, cfg, device="cpu")
    pruned_model.eval()

    pruned_params = sum(p.numel() for p in pruned_model.backbone.parameters())
    assert pruned_params < original_params

    output = pruned_model(torch.randn(2, 3, 16, 16), torch.tensor([0, 1]))
    assert output.logits.shape == (2, 5)


def test_build_depgraph_pruned_model_defaults_to_cuda_when_available(tmp_path, monkeypatch):
    """device=None must resolve to the same device engine.py's training loop
    uses (cuda when available), not a hardcoded CPU — pruning ResNet-sized
    models on CPU in a memory-constrained container is what caused the
    NNPACK "out of memory" warning flood / hang this test guards against.

    Monkeypatches torch.cuda.is_available() to True and short-circuits right
    after the device string is resolved, so this stays safe to run on
    CPU-only machines (no real CUDA tensor is ever allocated).
    """
    checkpoint_path = _write_plain_checkpoint(tmp_path)
    config, cfg = _make_config(checkpoint_path)

    captured = {}

    class _StopEarly(Exception):
        pass

    def _fake_load_plain_backbone(config, checkpoint_path, device):
        captured["device"] = device
        raise _StopEarly

    monkeypatch.setattr(dg, "_load_plain_backbone", _fake_load_plain_backbone)
    monkeypatch.setattr(dg.torch.cuda, "is_available", lambda: True)

    with pytest.raises(_StopEarly):
        build_depgraph_pruned_model_from_config(config, cfg, device=None)

    assert captured["device"] == "cuda"


def test_build_depgraph_pruned_model_rejects_unknown_importance(tmp_path):
    checkpoint_path = _write_plain_checkpoint(tmp_path)
    config, cfg = _make_config(checkpoint_path)
    cfg.importance = "not_a_real_criterion"

    with pytest.raises(ValueError, match="importance must be one of"):
        build_depgraph_pruned_model_from_config(config, cfg, device="cpu")


def test_build_depgraph_pruned_model_missing_checkpoint_raises(tmp_path):
    config, cfg = _make_config(tmp_path / "does_not_exist.pt")

    with pytest.raises(FileNotFoundError):
        build_depgraph_pruned_model_from_config(config, cfg, device="cpu")


def test_build_depgraph_pruned_model_rejects_target_speedup_at_or_below_one(tmp_path):
    checkpoint_path = _write_plain_checkpoint(tmp_path)
    config, cfg = _make_config(checkpoint_path, target_speedup=1.0)

    with pytest.raises(ValueError, match="target_speedup must be > 1.0"):
        build_depgraph_pruned_model_from_config(config, cfg, device="cpu")


def test_build_depgraph_pruned_model_rejects_sparsity_learning_for_unsupported_importance(tmp_path):
    checkpoint_path = _write_plain_checkpoint(tmp_path)
    config, cfg = _make_config(checkpoint_path, importance="random", sparsity_learning=True)

    with pytest.raises(ValueError, match="sparsity_learning=true is only supported"):
        build_depgraph_pruned_model_from_config(config, cfg, device="cpu")


def test_progressive_prune_to_target_speedup_reaches_or_exceeds_target():
    backbone = _make_backbone()
    backbone.eval()
    example_inputs = torch.randn(1, 3, 16, 16)
    ignored_layers = [backbone.fc]

    pruner = tp.pruner.BasePruner(
        backbone,
        example_inputs=example_inputs,
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        global_pruning=True,
        pruning_ratio=1.0,
        iterative_steps=50,
        ignored_layers=ignored_layers,
    )

    achieved = _progressive_prune_to_target_speedup(pruner, backbone, example_inputs, target_speedup=1.3)

    assert achieved >= 1.3
    assert pruner.current_step <= 50


def test_run_sparsity_learning_phase_calls_regularize_every_batch_and_ends_in_eval(monkeypatch):
    backbone = _make_backbone(num_classes=3)

    X = torch.randn(6, 3, 8, 8)
    y = torch.randint(0, 3, (6,))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=2)

    class _FakeDataloaders:
        train_dataloader = loader

    monkeypatch.setattr(dg, "instantiate", lambda _cfg_node: _FakeDataloaders())

    regularize_calls = {"count": 0}

    class _FakePruner:
        def regularize(self, model):
            regularize_calls["count"] += 1

    _run_sparsity_learning_phase(
        backbone,
        _FakePruner(),
        config=OmegaConf.create({"dataloaders": {}}),
        cfg=OmegaConf.create({"sl_epochs": 2, "sl_lr": 0.01, "sl_momentum": 0.9}),
        device="cpu",
    )

    # 2 epochs * 3 batches per epoch (6 samples / batch_size 2)
    assert regularize_calls["count"] == 6
    assert backbone.training is False
