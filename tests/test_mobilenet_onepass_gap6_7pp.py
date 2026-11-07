from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from net_complexity.training import cyclic_channel_pruning
from net_complexity.training.adaptive_lambda import AdaptiveLambdaController


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _config():
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(
            config_name="mobilenet_onepass_thr05_gap6_7pp",
            overrides=[
                "training_arguments.adaptive_lambda.baseline_history_dir="
                "/test/imported_dense_reference_20261002"
            ],
        )


def test_wider_gap_config_keeps_the_selected_single_pass_recipe():
    cfg = _config()
    adaptive = cfg.training_arguments.adaptive_lambda
    cyclic = cfg.cyclic_channel_pruning

    assert adaptive.enabled is True
    assert (adaptive.soft_drop, adaptive.hard_drop) == (0.06, 0.07)
    assert cyclic.g_prob_threshold == 0.5
    assert cyclic.threshold_max_param_fraction == 0.1
    assert cyclic.max_cycles == 1
    assert (cyclic.gumbel_epochs, cyclic.final_epochs) == (30, 45)
    assert cfg.dataloaders.batch_size == 128
    assert cfg.dataloaders.num_workers == 0
    assert cfg.model.criterion.label_smoothing == 0
    assert cfg.model.entropy_regularization_coef == 0
    assert cfg.model.backbone.block.gate_internal_width is True
    assert cfg.model.backbone.pretrained_weights == "IMAGENET1K_V2"
    assert cfg.mlflow.tags.reference_history_available_epochs == 86
    assert cfg.mlflow.tags.reference_used_epochs == "1-30"
    assert cfg.run_history.root_dir == "outputs/runs/mobilenet_onepass_thr05_gap6_7pp"
    assert "widegap6_7pp_20261108" in cfg.mlflow.run_name


@pytest.mark.parametrize(
    ("accuracy_drop", "expected_action"),
    [
        (0.055, "increase_lambda"),
        (0.065, "hold"),
        (0.07, "hold"),
        (0.075, "decrease_lambda"),
    ],
)
def test_positive_gap_values_are_absolute_accuracy_drops(accuracy_drop, expected_action):
    cfg = _config()
    adaptive = OmegaConf.to_container(cfg.training_arguments.adaptive_lambda, resolve=True)
    adaptive.pop("enabled")
    adaptive.pop("baseline_history_dir")
    adaptive["recovery_config"] = adaptive.pop("recovery")
    controller = AdaptiveLambdaController(
        initial_lambda_coef=cfg.model.lambda_coef,
        reference_accuracy_by_epoch={epoch: 0.8 for epoch in range(1, 5)},
        **adaptive,
    )
    model = nn.Identity()
    def apply_lambda(model, value):
        model.lambda_coef = value

    controller.apply_initial_state(model, apply_lambda=apply_lambda)
    for epoch in range(1, 5):
        result = controller.on_epoch_end(
            epoch=epoch,
            model=model,
            valid_metrics={
                "valid_accuracy": 0.8 - accuracy_drop,
                "valid_average_zero_prob": 0.2,
            },
            apply_lambda=apply_lambda,
        )
        if epoch <= 3:
            assert result.action == "warmup"

    assert result.action == expected_action
    assert result.metrics["min_allowed_acc"] == pytest.approx(0.74)
    assert result.metrics["hard_min_allowed_acc"] == pytest.approx(0.73)


def test_single_pass_search_selects_channels_then_runs_only_final_structural_recovery(
    monkeypatch, tmp_path
):
    cfg = _config()
    calls = []

    def fake_run_training(stage_cfg, **kwargs):
        assert "model_initializer" not in kwargs
        calls.append(OmegaConf.to_container(stage_cfg, resolve=False))
        return {
            "num_epochs_executed": stage_cfg.training_arguments.num_epochs,
            "last_valid_metrics": {
                "valid_backbone.features.0.gumbel_layer.channel_000_zero_prob": 0.9,
                "valid_backbone.features.0.gumbel_layer.channel_001_zero_prob": 0.1,
            },
        }

    # A small probe exercises the real threshold/cap selection without a model,
    # ImageNet download, training, inference or BatchNorm updates.
    probe = SimpleNamespace(
        backbone=SimpleNamespace(
            features=[
                SimpleNamespace(
                    branch=SimpleNamespace(
                        project=SimpleNamespace(in_channels=10, bias=None)
                    )
                )
            ]
        ),
        parameters=lambda: [SimpleNamespace(numel=lambda: 1000)],
    )
    monkeypatch.setattr(cyclic_channel_pruning, "run_training", fake_run_training)
    monkeypatch.setattr(cyclic_channel_pruning, "hydra_instantiate", lambda _: probe)
    result = cyclic_channel_pruning.run_cyclic_channel_pruning_training(cfg, tmp_path)

    assert len(calls) == 2
    search, recovery = calls
    assert search["training_arguments"]["num_epochs"] == 30
    assert search["scheduler"]["T_max"] == 30
    assert search["training_arguments"]["adaptive_lambda"]["enabled"] is True
    assert search["channel_pruning"]["structural"] is False
    assert recovery["training_arguments"]["num_epochs"] == 45
    assert recovery["scheduler"]["T_max"] == 45
    assert recovery["channel_pruning"]["structural"] is True
    assert recovery["channel_pruning"]["mask"] == {"backbone.features.0.gumbel_layer": [0]}
    assert recovery["model"]["lambda_coef"] == 0
    assert recovery["training_arguments"]["adaptive_lambda"]["enabled"] is False
    for stage in calls:
        assert stage["model"]["criterion"]["label_smoothing"] == 0
        assert stage["model"]["backbone"]["pretrained_weights"] == "IMAGENET1K_V2"
    assert result["num_cycles_completed"] == 1
    assert result["training_cost"]["num_epochs_executed"] == 75
    assert result["weight_handoff"]["enabled"] is False
    assert len(result["recovery_results"]) == 1
