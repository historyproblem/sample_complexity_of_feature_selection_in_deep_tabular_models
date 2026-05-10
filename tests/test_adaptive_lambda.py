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


def test_adaptive_lambda_rolls_back_after_large_drop_over_five_epochs():
    model = _DummyModel(lambda_coef=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler_state = _make_scheduler_state(token=0, step_count=0)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=0.1,
        warmup_epochs=0,
        update_every_epochs=1,
        acc_window=1,
        lambda_max=1000.0,
        rollback_check_every_epochs=5,
        rollback_acc_drop_threshold=0.20,
        rollback_epoch_lookback=20,
        lambda_increase_cooldown_epochs=10,
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    for epoch in range(1, 25):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        optimizer.param_groups[0]["lr"] = float(epoch)
        scheduler_state.step_count = epoch
        scheduler_state.scheduler.token = epoch

        result = controller.on_epoch_end(
            epoch=epoch,
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
        assert result.rolled_back is False
        assert result.resume_epoch is None

    with torch.no_grad():
        model.weight.fill_(25.0)
    optimizer.param_groups[0]["lr"] = 25.0
    scheduler_state.step_count = 25
    scheduler_state.scheduler.token = 25

    result = controller.on_epoch_end(
        epoch=25,
        model=model,
        optimizer=optimizer,
        valid_metrics={
            "valid_accuracy": 0.60,
            "valid_average_zero_prob": 0.35,
        },
        scheduler_state=scheduler_state,
        scaler=None,
        apply_lambda=_apply_lambda,
    )

    assert result.action == "periodic_rollback"
    assert result.rolled_back is True
    assert result.resume_epoch == 6
    assert controller.rollback_count == 1
    assert controller.lambda_increase_cooldown_remaining == 10
    assert model.weight.detach().cpu().item() == pytest.approx(5.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.0)
    assert scheduler_state.step_count == 5
    assert scheduler_state.scheduler.token == 5
    assert controller.observed_accuracy_by_epoch == {1: 0.91, 2: 0.91, 3: 0.91, 4: 0.91, 5: 0.91}
    assert model.lambda_coef == pytest.approx(3.2)


def test_adaptive_lambda_blocks_increase_during_cooldown_and_resumes_after_it_expires():
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
    assert first_result.action == "increase_lambda"
    assert model.lambda_coef == pytest.approx(10.0)

    controller.lambda_increase_cooldown_remaining = 2

    second_result = controller.on_epoch_end(
        epoch=2,
        model=model,
        optimizer=optimizer,
        valid_metrics={
            "valid_accuracy": 0.93,
            "valid_average_zero_prob": 0.35,
        },
        scheduler_state=scheduler_state,
        scaler=None,
        apply_lambda=_apply_lambda,
    )
    assert second_result.action == "hold"
    assert "rollback cooldown" in second_result.reason
    assert controller.lambda_increase_cooldown_remaining == 1
    assert model.lambda_coef == pytest.approx(10.0)

    third_result = controller.on_epoch_end(
        epoch=3,
        model=model,
        optimizer=optimizer,
        valid_metrics={
            "valid_accuracy": 0.94,
            "valid_average_zero_prob": 0.35,
        },
        scheduler_state=scheduler_state,
        scaler=None,
        apply_lambda=_apply_lambda,
    )
    assert third_result.action == "hold"
    assert controller.lambda_increase_cooldown_remaining == 0
    assert model.lambda_coef == pytest.approx(10.0)

    fourth_result = controller.on_epoch_end(
        epoch=4,
        model=model,
        optimizer=optimizer,
        valid_metrics={
            "valid_accuracy": 0.95,
            "valid_average_zero_prob": 0.35,
        },
        scheduler_state=scheduler_state,
        scaler=None,
        apply_lambda=_apply_lambda,
    )
    assert fourth_result.action == "increase_lambda"
    assert model.lambda_coef == pytest.approx(20.0)
