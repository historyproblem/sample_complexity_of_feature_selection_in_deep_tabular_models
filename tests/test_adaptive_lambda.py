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


def test_adaptive_lambda_uses_baseline_epoch_accuracy_instead_of_best_accuracy_so_far():
    model = _DummyModel(lambda_coef=5.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler_state = _make_scheduler_state(token=7, step_count=4)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=5.0,
        reference_accuracy_by_epoch={1: 0.91, 2: 0.84},
        warmup_epochs=0,
        update_every_epochs=1,
        acc_window=1,
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    first_result = controller.on_epoch_end(
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
    second_result = controller.on_epoch_end(
        epoch=2,
        model=model,
        optimizer=optimizer,
        valid_metrics={
            "valid_accuracy": 0.83,
            "valid_average_zero_prob": 0.35,
        },
        scheduler_state=scheduler_state,
        scaler=None,
        apply_lambda=_apply_lambda,
    )

    assert first_result.action == "increase_lambda"
    assert second_result.action == "increase_lambda"
    assert second_result.metrics["reference_source"] == "baseline_history"
    assert second_result.metrics["reference_epoch"] == 2
    assert second_result.metrics["reference_acc"] == pytest.approx(0.84)
    assert second_result.metrics["min_allowed_acc"] == pytest.approx(0.82)
    assert model.lambda_coef == pytest.approx(20.0)


def test_adaptive_lambda_decreases_lambda_on_hard_degradation_without_rollback():
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

    assert result.action == "decrease_lambda"
    assert result.rolled_back is False
    assert controller.rollback_count == 0
    assert controller.step == pytest.approx(math.log(2.0))
    assert model.weight.detach().cpu().item() == pytest.approx(99.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.333)
    assert scheduler_state.step_count == 99
    assert scheduler_state.scheduler.token == 42
    assert model.lambda_coef == pytest.approx(5.0)


def test_adaptive_lambda_does_not_freeze_when_rollback_limit_is_unreachable():
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

    assert result.action == "decrease_lambda"
    assert controller.frozen is False
    assert controller.rollback_count == 0
    assert controller.step == pytest.approx(math.log(2.0))
    assert model.lambda_coef == pytest.approx(5.0)
    assert controller.latest_safe_lambda == pytest.approx(5.0)


def test_adaptive_lambda_holds_lambda_between_soft_and_hard_thresholds():
    model = _DummyModel(lambda_coef=5.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler_state = _make_scheduler_state(token=7, step_count=4)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=5.0,
        warmup_epochs=0,
        update_every_epochs=1,
        acc_window=1,
        rollback_on_degradation=False,
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
            "valid_accuracy": 0.88,
            "valid_average_zero_prob": 0.35,
        },
        scheduler_state=scheduler_state,
        scaler=None,
        apply_lambda=_apply_lambda,
    )

    assert result.action == "hold"
    assert result.rolled_back is False
    assert controller.rollback_count == 0
    assert model.weight.detach().cpu().item() == pytest.approx(99.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.333)
    assert scheduler_state.step_count == 99
    assert scheduler_state.scheduler.token == 42
    assert model.lambda_coef == pytest.approx(10.0)


def test_adaptive_lambda_keeps_runtime_state_on_collapse_when_rollbacks_are_disabled_temporarily():
    model = _DummyModel(lambda_coef=5.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler_state = _make_scheduler_state(token=7, step_count=4)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=5.0,
        warmup_epochs=0,
        update_every_epochs=1,
        acc_window=1,
        rollback_on_degradation=False,
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
            "valid_accuracy": 0.10,
            "valid_average_zero_prob": 0.92,
        },
        scheduler_state=scheduler_state,
        scaler=None,
        apply_lambda=_apply_lambda,
    )

    assert result.action == "collapse_continue"
    assert result.rolled_back is False
    assert result.collapse_detected is True
    assert controller.rollback_count == 0
    assert controller.step == pytest.approx(math.log(2.0))
    assert model.weight.detach().cpu().item() == pytest.approx(99.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.333)
    assert scheduler_state.step_count == 99
    assert scheduler_state.scheduler.token == 42
    assert model.lambda_coef == pytest.approx(10.0)
