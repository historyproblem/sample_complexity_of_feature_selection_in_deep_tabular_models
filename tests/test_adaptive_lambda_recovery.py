from __future__ import annotations

import csv
import json
import math

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from net_complexity.models.feature_selection import GumbelLayer
from net_complexity.training.adaptive_lambda import AdaptiveLambdaController
from net_complexity.training.run_history import RunHistory


class TinyRecoveryModel(nn.Module):
    def __init__(self, num_channels: int = 4):
        super().__init__()
        self.gate = GumbelLayer(num_channels)
        self.lambda_coef = 1.0

    def set_gumbel_open_bias(self, value: float, p_min: float = 0.02, p_max: float = 0.50) -> None:
        self.gate.set_open_bias(value, p_min=p_min, p_max=p_max)


def _logit_pair_for_open_prob(prob: float) -> list[float]:
    return [0.0, math.log(prob / (1.0 - prob))]


def _metrics(acc: float, zero_prob: float, loss: float = 1.0) -> dict[str, float]:
    return {
        "valid_accuracy": float(acc),
        "valid_loss": float(loss),
        "valid_average_zero_prob": float(zero_prob),
    }


def _apply_lambda(model: TinyRecoveryModel, lambda_coef: float) -> None:
    model.lambda_coef = float(lambda_coef)


def _make_controller(**recovery_overrides) -> AdaptiveLambdaController:
    update_every_epochs = int(recovery_overrides.pop("update_every_epochs", 1))
    adaptive_log_step_enabled = bool(
        recovery_overrides.pop("adaptive_log_step_enabled", False)
    )
    recovery_config = {
        "enabled": True,
        "min_epoch": 3,
        "drop_min": 0.02,
        "drop_max": 0.20,
        "patience": 1,
        "recovery_slope_window": 1,
        "require_slow_recovery": True,
        "min_acc_delta_over_window": 0.005,
        "use_zero_prob_filter": True,
        "zero_prob_window": 1,
        "zero_prob_delta_min": 0.02,
        "recovery_epochs": 5,
        "freeze_lambda_increase": True,
        "decay_lambda": False,
        "open_bias_start": 0.15,
        "open_bias_decay": 0.90,
        "p_open_min": 0.02,
        "p_open_max": 0.50,
        "max_reopen_delta": 0.02,
        "target_acc_margin": 0.01,
        "cooldown_epochs": 10,
        "max_recovery_attempts": 3,
    }
    recovery_config.update(recovery_overrides)
    return AdaptiveLambdaController(
        initial_lambda_coef=1.0,
        warmup_epochs=0,
        update_every_epochs=update_every_epochs,
        acc_window=1,
        adaptive_log_step_enabled=adaptive_log_step_enabled,
        soft_drop=0.02,
        hard_drop=0.04,
        recovery_config=recovery_config,
    )


def test_initial_log_includes_recovery_config(capsys):
    model = TinyRecoveryModel()
    controller = _make_controller()

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)

    output = capsys.readouterr().out
    assert "Adaptive lambda initialized" in output
    assert "recovery_enabled=true" in output
    assert "recovery_min_epoch=3" in output
    assert "recovery_patience=1" in output
    assert "recovery_epochs=5" in output
    assert "recovery_max_attempts=3" in output


def test_gumbel_open_bias_only_reopens_revive_candidates():
    layer = GumbelLayer(input_dim=4, train_gate_mode="deterministic_soft", eval_gate_mode="deterministic_soft")
    with torch.no_grad():
        layer.logits.copy_(
            torch.tensor(
                [
                    _logit_pair_for_open_prob(0.01),
                    _logit_pair_for_open_prob(0.10),
                    _logit_pair_for_open_prob(0.40),
                    _logit_pair_for_open_prob(0.60),
                ]
            )
        )

    raw_probs = layer.get_selection_probs()
    raw_regularization = float(layer.regularization_loss().item())

    layer.set_open_bias(0.15, p_min=0.02, p_max=0.50)
    biased_probs = layer.get_selection_probs()

    assert biased_probs[0] == pytest.approx(raw_probs[0])
    assert biased_probs[1] > raw_probs[1]
    assert biased_probs[2] > raw_probs[2]
    assert biased_probs[3] == pytest.approx(raw_probs[3])
    assert float(layer.regularization_loss().item()) == pytest.approx(raw_regularization)

    layer.set_open_bias(0.0, p_min=0.02, p_max=0.50)
    assert layer.open_bias == 0.0
    assert torch.allclose(layer.get_selection_probs(), raw_probs)


def test_recovery_starts_on_slow_recovery_and_blocks_only_lambda_increase():
    model = TinyRecoveryModel()
    controller = _make_controller(update_every_epochs=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    controller.apply_initial_state(model, apply_lambda=_apply_lambda)

    controller.on_epoch_end(
        epoch=1,
        model=model,
        valid_metrics=_metrics(0.90, 0.10),
        apply_lambda=_apply_lambda,
    )
    controller.on_epoch_end(
        epoch=2,
        model=model,
        valid_metrics=_metrics(0.877, 0.11),
        apply_lambda=_apply_lambda,
    )
    result = controller.on_epoch_end(
        epoch=3,
        model=model,
        valid_metrics=_metrics(0.880, 0.14),
        apply_lambda=_apply_lambda,
    )

    assert result.action == "recovery_blocked_lambda_increase"
    assert result.metrics["recovery_action"] == "start_recovery"
    assert result.metrics["recovery_active"] is True
    assert result.metrics["recovery_acc_delta_over_window"] == pytest.approx(0.003)
    assert result.metrics["lambda_coef"] == pytest.approx(1.0)
    assert model.lambda_coef == pytest.approx(1.0)
    assert model.gate.open_bias == pytest.approx(0.15)


def test_recovery_disabled_keeps_adaptive_lambda_behavior_unchanged():
    model = TinyRecoveryModel()
    controller = _make_controller(enabled=False, update_every_epochs=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    controller.apply_initial_state(model, apply_lambda=_apply_lambda)

    controller.on_epoch_end(
        epoch=1,
        model=model,
        valid_metrics=_metrics(0.90, 0.10),
        apply_lambda=_apply_lambda,
    )
    controller.on_epoch_end(
        epoch=2,
        model=model,
        valid_metrics=_metrics(0.877, 0.11),
        apply_lambda=_apply_lambda,
    )
    result = controller.on_epoch_end(
        epoch=3,
        model=model,
        valid_metrics=_metrics(0.880, 0.14),
        apply_lambda=_apply_lambda,
    )

    assert result.action == "increase_lambda"
    assert result.metrics["recovery_action"] == "none"
    assert result.metrics["recovery_active"] is False
    assert result.metrics["lambda_coef"] > 1.0
    assert model.gate.open_bias == 0.0


def test_recovery_does_not_block_regular_lambda_decrease():
    model = TinyRecoveryModel()
    controller = _make_controller(recovery_epochs=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    controller.best_val_acc = 0.90
    controller.recovery_active = True
    controller.recovery_epochs_left = 5
    controller.recovery_start_acc = 0.86
    controller.recovery_start_zero_prob = 0.20
    controller.recovery_start_active_ratio = 0.80
    controller.recovery_attempts = 1
    controller._apply_recovery_open_bias(model, 0.15)

    result = controller.on_epoch_end(
        epoch=10,
        model=model,
        valid_metrics=_metrics(0.80, 0.20),
        apply_lambda=_apply_lambda,
    )

    assert result.action == "decrease_lambda"
    assert result.metrics["recovery_action"] == "continue_recovery"
    assert result.metrics["lambda_coef"] < 1.0
    assert model.lambda_coef < 1.0
    assert model.gate.open_bias == pytest.approx(0.15 * 0.90)


def test_recovery_stops_when_reopen_budget_is_exhausted():
    model = TinyRecoveryModel()
    controller = _make_controller(recovery_epochs=5, max_reopen_delta=0.02)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    controller.best_val_acc = 0.90
    controller.recovery_active = True
    controller.recovery_epochs_left = 5
    controller.recovery_start_acc = 0.86
    controller.recovery_start_zero_prob = 0.20
    controller.recovery_start_active_ratio = 0.80
    controller.recovery_attempts = 1
    controller._apply_recovery_open_bias(model, 0.15)

    result = controller.on_epoch_end(
        epoch=10,
        model=model,
        valid_metrics=_metrics(0.85, 0.17),
        apply_lambda=_apply_lambda,
    )

    assert result.metrics["recovery_action"] == "stop_max_reopen_delta"
    assert result.metrics["recovery_active"] is False
    assert result.metrics["recovery_active_delta"] == pytest.approx(0.03)
    assert result.metrics["recovery_cooldown_left"] == 10
    assert model.gate.open_bias == 0.0


def test_recovery_unblocks_lambda_increase_on_stop_epoch():
    model = TinyRecoveryModel()
    controller = _make_controller(recovery_epochs=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    controller.best_val_acc = 0.90
    controller.recovery_active = True
    controller.recovery_epochs_left = 5
    controller.recovery_start_acc = 0.86
    controller.recovery_start_zero_prob = 0.20
    controller.recovery_start_active_ratio = 0.80
    controller.recovery_attempts = 1
    controller._apply_recovery_open_bias(model, 0.15)

    result = controller.on_epoch_end(
        epoch=10,
        model=model,
        valid_metrics=_metrics(0.895, 0.20),
        apply_lambda=_apply_lambda,
    )

    assert result.metrics["recovery_action"] == "stop_recovered_acc"
    assert result.metrics["recovery_active"] is False
    assert result.action == "increase_lambda"
    assert result.metrics["lambda_coef"] > 1.0
    assert model.gate.open_bias == 0.0



def test_recovery_metrics_are_written_to_history_and_summary(tmp_path):
    model = TinyRecoveryModel()
    controller = _make_controller()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    controller.on_epoch_end(
        epoch=1,
        model=model,
        valid_metrics=_metrics(0.90, 0.10),
        apply_lambda=_apply_lambda,
    )
    controller.on_epoch_end(
        epoch=2,
        model=model,
        valid_metrics=_metrics(0.877, 0.11),
        apply_lambda=_apply_lambda,
    )
    result = controller.on_epoch_end(
        epoch=3,
        model=model,
        valid_metrics=_metrics(0.880, 0.14),
        apply_lambda=_apply_lambda,
    )

    run_history = RunHistory(
        OmegaConf.create(
            {
                "run_history": {
                    "root_dir": str(tmp_path),
                    "run_name": "recovery_history",
                    "use_hydra_output_dir": False,
                }
            }
        )
    )
    run_history.set_runtime_metadata({"adaptive_lambda": controller.summary_state()})
    run_history.log_epoch(
        3,
        {"train_loss": 1.0},
        {"valid_accuracy": 0.88, "valid_average_zero_prob": 0.14},
        extra_metrics=result.metrics,
    )
    run_history.save_summary(
        final_train_metrics={"train_loss": 1.0},
        final_valid_metrics={"valid_accuracy": 0.88, "valid_average_zero_prob": 0.14},
    )

    with run_history.history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["recovery_action"] == "start_recovery"
    assert rows[0]["recovery_active"] == "True"
    assert rows[0]["recovery_open_bias"] == "0.15"
    assert "recovery_revive_candidates_frac" in rows[0]
    assert "adaptive_lambda_effective_log_step" in rows[0]
    assert "adaptive_lambda_log_step_boost_level" in rows[0]
    assert "adaptive_lambda_prune_rate_per_epoch" in rows[0]
    assert "adaptive_lambda_step_action" in rows[0]

    summary = json.loads(run_history.summary_path.read_text(encoding="utf-8"))
    assert summary["recovery_num_attempts"] == 1
    assert summary["recovery_total_epochs"] == 0
    assert summary["recovery_was_used"] is True
    assert summary["recovery_config"]["enabled"] is True
