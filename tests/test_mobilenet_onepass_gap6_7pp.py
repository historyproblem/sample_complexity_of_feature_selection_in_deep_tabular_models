from pathlib import Path

import pytest
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from net_complexity.training import cyclic_channel_pruning
from net_complexity.training.adaptive_lambda import AdaptiveLambdaController


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(params=[("6_7", 0.06, 0.07), ("7_8", 0.07, 0.08)], ids=["6_7pp", "7_8pp"])
def recipe(request):
    return request.param


def _config(gap_suffix):
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(
            config_name=f"mobilenet_onepass_thr05_gap{gap_suffix}pp",
            overrides=[
                "training_arguments.adaptive_lambda.baseline_history_dir="
                "/test/imported_dense_reference_20261002"
            ],
        )


def test_wider_gap_config_keeps_the_selected_single_pass_recipe(recipe):
    gap_suffix, soft_drop, hard_drop = recipe
    cfg = _config(gap_suffix)
    adaptive = cfg.training_arguments.adaptive_lambda
    cyclic = cfg.cyclic_channel_pruning

    assert adaptive.enabled is True
    assert (adaptive.soft_drop, adaptive.hard_drop) == (soft_drop, hard_drop)
    assert cyclic.g_prob_threshold == 0.5
    assert cyclic.threshold_max_param_fraction is None
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
    assert cfg.run_history.root_dir == f"outputs/runs/mobilenet_onepass_thr05_gap{gap_suffix}pp"
    assert f"widegap{gap_suffix}pp_20261108" in cfg.mlflow.experiment_name
    assert f"widegap{gap_suffix}pp_20261108" in cfg.mlflow.run_name
    assert cfg.mlflow.run_name.endswith("_nocap")
    assert cfg.mlflow.tags.recipe == (
        f"single_pass_thr05_widegap_soft{soft_drop * 100:g}pp"
        f"_hard{hard_drop * 100:g}pp_user_20261108"
    )


@pytest.mark.parametrize(
    ("drop_position", "expected_action"),
    [
        ("below_soft", "increase_lambda"),
        ("between_gaps", "hold"),
        ("at_hard", "hold"),
        ("above_hard", "decrease_lambda"),
    ],
)
def test_positive_gap_values_are_absolute_accuracy_drops(recipe, drop_position, expected_action):
    gap_suffix, soft_drop, hard_drop = recipe
    cfg = _config(gap_suffix)
    accuracy_drop = {
        "below_soft": soft_drop - 0.005,
        "between_gaps": (soft_drop + hard_drop) / 2,
        "at_hard": hard_drop,
        "above_hard": hard_drop + 0.005,
    }[drop_position]
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
    assert result.metrics["min_allowed_acc"] == pytest.approx(0.8 - soft_drop)
    assert result.metrics["hard_min_allowed_acc"] == pytest.approx(0.8 - hard_drop)


def test_single_pass_search_selects_channels_then_runs_only_final_structural_recovery(
    recipe, monkeypatch, tmp_path
):
    cfg = _config(recipe[0])
    calls = []
    gate = "backbone.features.0.gumbel_layer"
    eligible_channels = list(range(12))
    channel_metrics = {
        f"valid_{gate}.channel_{channel:03d}_zero_prob": 0.9 - 0.01 * channel
        for channel in eligible_channels
    }
    channel_metrics.update({
        f"valid_{gate}.channel_012_zero_prob": 0.1,
        f"valid_{gate}.channel_013_zero_prob": 0.5,
    })

    def fake_run_training(stage_cfg, **kwargs):
        assert "model_initializer" not in kwargs
        calls.append(OmegaConf.to_container(stage_cfg, resolve=False))
        return {
            "num_epochs_executed": stage_cfg.training_arguments.num_epochs,
            "last_valid_metrics": channel_metrics,
        }

    # The real threshold selection must run without constructing a parameter
    # budget probe, ImageNet download, training or BatchNorm updates.
    monkeypatch.setattr(cyclic_channel_pruning, "run_training", fake_run_training)

    def unexpected_budget_probe(_):
        pytest.fail("uncapped threshold selection must not instantiate a budget probe")

    monkeypatch.setattr(cyclic_channel_pruning, "hydra_instantiate", unexpected_budget_probe)
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
    assert recovery["channel_pruning"]["mask"] == {gate: eligible_channels}
    assert recovery["model"]["lambda_coef"] == 0
    assert recovery["training_arguments"]["adaptive_lambda"]["enabled"] is False
    for stage in calls:
        assert stage["model"]["criterion"]["label_smoothing"] == 0
        assert stage["model"]["backbone"]["pretrained_weights"] == "IMAGENET1K_V2"
    assert result["num_cycles_completed"] == 1
    assert result["training_cost"]["num_epochs_executed"] == 75
    assert result["weight_handoff"]["enabled"] is False
    assert len(result["recovery_results"]) == 1
