import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from net_complexity.training.adaptive_lambda import AdaptiveLambdaController


class _DummyScheduler:
    def __init__(self, token: int):
        self.token = int(token)

    def state_dict(self) -> dict[str, int]:
        return {"token": self.token}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.token = int(state_dict["token"])


class _DummyModel(nn.Module):
    def __init__(self, lambda_coef: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.lambda_coef = float(lambda_coef)

    def set_lambda_coef(self, lambda_coef: float, *, bypass_gumbel: bool | None = None) -> None:
        del bypass_gumbel
        self.lambda_coef = float(lambda_coef)


def _apply_lambda(model: _DummyModel, lambda_coef: float) -> None:
    model.set_lambda_coef(lambda_coef)


def _make_scheduler_state(token: int, step_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler=_DummyScheduler(token),
        step_count=int(step_count),
    )


def test_adaptive_lambda_increases_when_accuracy_stays_within_soft_band():
    model = _DummyModel(lambda_coef=5.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler_state = _make_scheduler_state(token=7, step_count=4)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=5.0,
        warmup_epochs=0,
        update_every_epochs=1,
        acc_window=1,
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    result = controller.on_epoch_end(
        epoch=1,
        model=model,
        optimizer=optimizer,
        valid_metrics={
            "valid_accuracy": 0.91,
            "valid_average_zero_prob": 0.35,
        },
        scheduler_state=scheduler_state,
        scaler=None,
        apply_lambda=_apply_lambda,
    )

    assert result.action == "increase_lambda"
    assert result.checkpoint_saved is True
    assert model.lambda_coef == pytest.approx(10.0)
    assert controller.latest_safe_epoch == 1
    assert controller.latest_safe_lambda == pytest.approx(5.0)
    assert controller.best_sparse_safe_epoch == 1
    assert controller.best_sparse_safe_lambda == pytest.approx(5.0)
    assert result.metrics["adaptive_lambda_action"] == "increase_lambda"


def test_adaptive_lambda_rolls_back_and_restores_runtime_state_on_hard_degradation():
    model = _DummyModel(lambda_coef=5.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler_state = _make_scheduler_state(token=7, step_count=4)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=5.0,
        warmup_epochs=0,
        update_every_epochs=1,
        acc_window=1,
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    controller.on_epoch_end(
        epoch=1,
        model=model,
        optimizer=optimizer,
        valid_metrics={
            "valid_accuracy": 0.91,
            "valid_average_zero_prob": 0.35,
        },
        scheduler_state=scheduler_state,
        scaler=None,
        apply_lambda=_apply_lambda,
    )

    with torch.no_grad():
        model.weight.fill_(99.0)
    optimizer.param_groups[0]["lr"] = 0.333
    scheduler_state.step_count = 99
    scheduler_state.scheduler.token = 42

    result = controller.on_epoch_end(
        epoch=2,
        model=model,
        optimizer=optimizer,
        valid_metrics={
            "valid_accuracy": 0.70,
            "valid_average_zero_prob": 0.92,
        },
        scheduler_state=scheduler_state,
        scaler=None,
        apply_lambda=_apply_lambda,
    )

    expected_step = max(math.log(1.05), math.log(2.0) * 0.25)
    expected_lambda = 5.0 * math.exp(expected_step)

    assert result.action == "hard_degradation_rollback"
    assert result.rolled_back is True
    assert controller.rollback_count == 1
    assert model.weight.detach().cpu().item() == pytest.approx(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert scheduler_state.step_count == 4
    assert scheduler_state.scheduler.token == 7
    assert model.lambda_coef == pytest.approx(expected_lambda)


def test_adaptive_lambda_freezes_at_latest_safe_lambda_after_rollback_limit():
    model = _DummyModel(lambda_coef=5.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler_state = _make_scheduler_state(token=11, step_count=2)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=5.0,
        warmup_epochs=0,
        update_every_epochs=1,
        acc_window=1,
        max_rollbacks=0,
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    controller.on_epoch_end(
        epoch=1,
        model=model,
        optimizer=optimizer,
        valid_metrics={
            "valid_accuracy": 0.91,
            "valid_average_zero_prob": 0.35,
        },
        scheduler_state=scheduler_state,
        scaler=None,
        apply_lambda=_apply_lambda,
    )

    result = controller.on_epoch_end(
        epoch=2,
        model=model,
        optimizer=optimizer,
        valid_metrics={
            "valid_accuracy": 0.70,
            "valid_average_zero_prob": 0.92,
        },
        scheduler_state=scheduler_state,
        scaler=None,
        apply_lambda=_apply_lambda,
    )

    assert result.action == "rollback_limit_freeze"
    assert controller.frozen is True
    assert model.lambda_coef == pytest.approx(5.0)
    assert controller.latest_safe_lambda == pytest.approx(5.0)
