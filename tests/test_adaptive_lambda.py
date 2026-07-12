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


def _run_epoch(
    controller: AdaptiveLambdaController,
    model: _DummyModel,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    accuracy: float,
    zero_prob: float,
    scheduler_state: SimpleNamespace | None = None,
):
    return controller.on_epoch_end(
        epoch=epoch,
        model=model,
        valid_metrics={
            "valid_accuracy": accuracy,
            "valid_average_zero_prob": zero_prob,
        },
        apply_lambda=_apply_lambda,
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
        adaptive_log_step_enabled=False,
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    result = controller.on_epoch_end(
        epoch=1,
        model=model,
        valid_metrics={
            "valid_accuracy": 0.91,
            "valid_average_zero_prob": 0.35,
        },
        apply_lambda=_apply_lambda,
    )

    assert result.action == "increase_lambda"
    assert model.lambda_coef == pytest.approx(10.0)
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
        adaptive_log_step_enabled=False,
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    first_result = controller.on_epoch_end(
        epoch=1,
        model=model,
        valid_metrics={
            "valid_accuracy": 0.91,
            "valid_average_zero_prob": 0.35,
        },
        apply_lambda=_apply_lambda,
    )
    second_result = controller.on_epoch_end(
        epoch=2,
        model=model,
        valid_metrics={
            "valid_accuracy": 0.83,
            "valid_average_zero_prob": 0.35,
        },
        apply_lambda=_apply_lambda,
    )

    assert first_result.action == "increase_lambda"
    assert second_result.action == "increase_lambda"
    assert second_result.metrics["reference_source"] == "baseline_history"
    assert second_result.metrics["reference_epoch"] == 2
    assert second_result.metrics["reference_acc"] == pytest.approx(0.84)
    assert second_result.metrics["min_allowed_acc"] == pytest.approx(0.82)
    assert model.lambda_coef == pytest.approx(20.0)


def test_adaptive_lambda_decreases_lambda_on_hard_degradation_without_mutating_runtime_state():
    model = _DummyModel(lambda_coef=5.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler_state = _make_scheduler_state(token=7, step_count=4)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=5.0,
        warmup_epochs=0,
        update_every_epochs=1,
        acc_window=1,
        adaptive_log_step_enabled=False,
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    controller.on_epoch_end(
        epoch=1,
        model=model,
        valid_metrics={
            "valid_accuracy": 0.91,
            "valid_average_zero_prob": 0.35,
        },
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
        valid_metrics={
            "valid_accuracy": 0.70,
            "valid_average_zero_prob": 0.92,
        },
        apply_lambda=_apply_lambda,
    )

    assert result.action == "decrease_lambda"
    assert controller.step == pytest.approx(math.log(2.0))
    assert model.weight.detach().cpu().item() == pytest.approx(99.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.333)
    assert scheduler_state.step_count == 99
    assert scheduler_state.scheduler.token == 42
    assert model.lambda_coef == pytest.approx(5.0)



def test_adaptive_lambda_holds_lambda_between_soft_and_hard_thresholds():
    model = _DummyModel(lambda_coef=5.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler_state = _make_scheduler_state(token=7, step_count=4)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=5.0,
        warmup_epochs=0,
        update_every_epochs=1,
        acc_window=1,
        adaptive_log_step_enabled=False,
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    controller.on_epoch_end(
        epoch=1,
        model=model,
        valid_metrics={
            "valid_accuracy": 0.91,
            "valid_average_zero_prob": 0.35,
        },
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
        valid_metrics={
            "valid_accuracy": 0.88,
            "valid_average_zero_prob": 0.35,
        },
        apply_lambda=_apply_lambda,
    )

    assert result.action == "hold"
    assert model.weight.detach().cpu().item() == pytest.approx(99.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.333)
    assert scheduler_state.step_count == 99
    assert scheduler_state.scheduler.token == 42
    assert model.lambda_coef == pytest.approx(10.0)






def test_adaptive_log_step_slow_pruning_boosts_to_cap_and_target_resets():
    model = _DummyModel(lambda_coef=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    log_step = math.log(1.25)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=1.0,
        warmup_epochs=0,
        update_every_epochs=2,
        acc_window=1,
        log_step_init=log_step,
        log_step_min=math.log(1.05),
        lambda_max=1000.0,
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    _run_epoch(controller, model, optimizer, epoch=1, accuracy=0.90, zero_prob=0.095)
    first_update = _run_epoch(
        controller,
        model,
        optimizer,
        epoch=2,
        accuracy=0.90,
        zero_prob=0.10,
    )
    lambda_after_first_update = model.lambda_coef

    _run_epoch(controller, model, optimizer, epoch=3, accuracy=0.90, zero_prob=0.105)
    second_update = _run_epoch(
        controller,
        model,
        optimizer,
        epoch=4,
        accuracy=0.90,
        zero_prob=0.11,
    )
    lambda_after_second_update = model.lambda_coef

    _run_epoch(controller, model, optimizer, epoch=5, accuracy=0.90, zero_prob=0.115)
    third_update = _run_epoch(
        controller,
        model,
        optimizer,
        epoch=6,
        accuracy=0.90,
        zero_prob=0.12,
    )
    lambda_after_third_update = model.lambda_coef

    _run_epoch(controller, model, optimizer, epoch=7, accuracy=0.90, zero_prob=0.125)
    fourth_update = _run_epoch(
        controller,
        model,
        optimizer,
        epoch=8,
        accuracy=0.90,
        zero_prob=0.13,
    )
    lambda_after_fourth_update = model.lambda_coef

    _run_epoch(controller, model, optimizer, epoch=9, accuracy=0.90, zero_prob=0.17)
    target_update = _run_epoch(
        controller,
        model,
        optimizer,
        epoch=10,
        accuracy=0.90,
        zero_prob=0.21,
    )
    lambda_after_target_update = model.lambda_coef

    _run_epoch(controller, model, optimizer, epoch=11, accuracy=0.90, zero_prob=0.31)
    fast_update = _run_epoch(
        controller,
        model,
        optimizer,
        epoch=12,
        accuracy=0.90,
        zero_prob=0.41,
    )

    assert first_update.metrics["adaptive_lambda_step_action"] == "step_init_no_prev_zero_prob"
    assert first_update.metrics["adaptive_lambda_log_step_boost_level"] == 0
    assert first_update.metrics["adaptive_lambda_effective_log_step"] == pytest.approx(log_step)

    assert second_update.metrics["adaptive_lambda_step_action"] == "step_boost_slow_pruning"
    assert second_update.metrics["adaptive_lambda_log_step_boost_level"] == 1
    assert second_update.metrics["adaptive_lambda_prune_rate_per_epoch"] == pytest.approx(0.005)
    assert lambda_after_second_update / lambda_after_first_update == pytest.approx(1.25**2)

    assert third_update.metrics["adaptive_lambda_step_action"] == "step_boost_slow_pruning"
    assert third_update.metrics["adaptive_lambda_log_step_boost_level"] == 2
    assert lambda_after_third_update / lambda_after_second_update == pytest.approx(1.25**4)

    assert fourth_update.metrics["adaptive_lambda_step_action"] == "step_boost_slow_pruning"
    assert fourth_update.metrics["adaptive_lambda_log_step_boost_level"] == 2
    assert lambda_after_fourth_update / lambda_after_third_update == pytest.approx(1.25**4)

    assert target_update.metrics["adaptive_lambda_step_action"] == "step_reset_target_pruning"
    assert target_update.metrics["adaptive_lambda_log_step_boost_level"] == 0
    assert target_update.metrics["adaptive_lambda_prune_rate_per_epoch"] == pytest.approx(0.04)
    assert lambda_after_target_update / lambda_after_fourth_update == pytest.approx(1.25)

    assert fast_update.metrics["adaptive_lambda_step_action"] == (
        "step_keep_fast_pruning_no_new_logic"
    )
    assert fast_update.metrics["adaptive_lambda_log_step_boost_level"] == 0
    assert fast_update.metrics["adaptive_lambda_prune_rate_per_epoch"] == pytest.approx(0.10)
    assert model.lambda_coef / lambda_after_target_update == pytest.approx(1.25)


def test_adaptive_log_step_bad_accuracy_resets_boost_and_uses_base_step():
    model = _DummyModel(lambda_coef=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    log_step = math.log(1.25)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=1.0,
        warmup_epochs=0,
        update_every_epochs=1,
        acc_window=1,
        log_step_init=log_step,
        log_step_min=math.log(1.05),
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    _run_epoch(controller, model, optimizer, epoch=1, accuracy=0.90, zero_prob=0.10)
    _run_epoch(controller, model, optimizer, epoch=2, accuracy=0.90, zero_prob=0.105)
    lambda_before_bad_acc = model.lambda_coef
    assert controller.log_step_boost_level == 1

    result = _run_epoch(
        controller,
        model,
        optimizer,
        epoch=3,
        accuracy=0.80,
        zero_prob=0.11,
    )

    assert result.action == "decrease_lambda"
    assert result.metrics["adaptive_lambda_step_action"] == "step_reset_bad_acc"
    assert result.metrics["adaptive_lambda_log_step_boost_level"] == 0
    assert result.metrics["adaptive_lambda_effective_log_step"] == pytest.approx(log_step)
    assert model.lambda_coef == pytest.approx(lambda_before_bad_acc / 1.25)



def test_disabled_adaptive_log_step_keeps_base_log_step():
    model = _DummyModel(lambda_coef=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    log_step = math.log(1.25)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=1.0,
        warmup_epochs=0,
        update_every_epochs=1,
        acc_window=1,
        log_step_init=log_step,
        log_step_min=math.log(1.05),
        adaptive_log_step_enabled=False,
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    first_update = _run_epoch(
        controller,
        model,
        optimizer,
        epoch=1,
        accuracy=0.90,
        zero_prob=0.10,
    )
    second_update = _run_epoch(
        controller,
        model,
        optimizer,
        epoch=2,
        accuracy=0.90,
        zero_prob=0.10,
    )

    assert first_update.metrics["adaptive_lambda_step_action"] == "step_disabled"
    assert second_update.metrics["adaptive_lambda_step_action"] == "step_disabled"
    assert second_update.metrics["adaptive_lambda_log_step_boost_level"] == 0
    assert second_update.metrics["adaptive_lambda_effective_log_step"] == pytest.approx(log_step)
    assert second_update.metrics["adaptive_lambda_prune_rate_per_epoch"] is None
    assert model.lambda_coef == pytest.approx(1.25**2)


def test_lambda_max_none_leaves_lambda_unbounded_above():
    controller = AdaptiveLambdaController(
        initial_lambda_coef=100.0,
        lambda_min=1e-8,
        lambda_max=None,
    )

    assert controller.lambda_coef == pytest.approx(100.0)
    assert controller._clamp_lambda(1e200) == pytest.approx(1e200)
    assert controller._clamp_log_lambda(math.log(1e200)) == pytest.approx(math.log(1e200))


def test_adaptive_log_step_max_epoch_resets_to_base_step_after_cutoff():
    model = _DummyModel(lambda_coef=1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    log_step = math.log(1.25)
    controller = AdaptiveLambdaController(
        initial_lambda_coef=1.0,
        warmup_epochs=0,
        update_every_epochs=1,
        acc_window=1,
        log_step_init=log_step,
        log_step_min=math.log(1.05),
        adaptive_log_step_max_epoch=2,
    )

    controller.apply_initial_state(model, apply_lambda=_apply_lambda)
    _run_epoch(controller, model, optimizer, epoch=1, accuracy=0.90, zero_prob=0.10)
    _run_epoch(controller, model, optimizer, epoch=2, accuracy=0.90, zero_prob=0.105)
    lambda_before_cutoff = model.lambda_coef
    assert controller.log_step_boost_level == 1

    result = _run_epoch(
        controller,
        model,
        optimizer,
        epoch=3,
        accuracy=0.90,
        zero_prob=0.11,
    )

    assert result.action == "increase_lambda"
    assert result.metrics["adaptive_lambda_step_action"] == "step_reset_after_max_epoch"
    assert result.metrics["adaptive_lambda_log_step_boost_level"] == 0
    assert result.metrics["adaptive_lambda_effective_log_step"] == pytest.approx(log_step)
    assert result.metrics["adaptive_log_step_max_epoch"] == 2
    assert model.lambda_coef / lambda_before_cutoff == pytest.approx(1.25)
