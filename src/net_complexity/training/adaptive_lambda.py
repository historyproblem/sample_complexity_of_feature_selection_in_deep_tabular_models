from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn


LambdaApplier = Callable[[nn.Module, float], None]

ACCURACY_METRIC_NAMES = (
    "valid_accuracy",
    "accuracy",
)
LOSS_METRIC_NAMES = (
    "valid_loss",
    "valid_ce_loss",
    "loss",
    "ce_loss",
)
ZERO_PROB_METRIC_NAMES = (
    "valid_average_zero_prob",
    "valid_zero_prob",
    "average_zero_prob",
    "zero_prob",
)
EXPECTED_ACTIVE_CHANNEL_METRIC_NAMES = (
    "valid_expected_active_channels",
    "expected_active_channels",
)


def _to_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value.detach().cpu().mean().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_metric(metrics: Mapping[str, Any], names: tuple[str, ...]) -> tuple[str | None, float | None]:
    for name in names:
        if name not in metrics:
            continue
        value = _to_float(metrics[name])
        if value is not None:
            return name, value
    return None, None


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return value


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer) -> None:
    first_param = None
    for param_group in optimizer.param_groups:
        params = param_group.get("params", [])
        if params:
            first_param = params[0]
            break
    if first_param is None:
        return

    device = first_param.device
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


@dataclass
class AdaptiveLambdaCheckpoint:
    epoch: int
    lambda_coef: float
    log_lambda: float
    pruning_metric_name: str | None
    pruning_metric_value: float | None
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None
    scheduler_step_count: int | None
    scaler_state_dict: dict[str, Any] | None
    controller_state: dict[str, Any]


@dataclass
class AdaptiveLambdaStepResult:
    action: str
    reason: str
    lambda_changed: bool
    rolled_back: bool
    collapse_detected: bool
    checkpoint_saved: bool
    metrics: dict[str, Any]


class AdaptiveLambdaController:
    def __init__(
        self,
        *,
        initial_lambda_coef: float,
        warmup_epochs: int = 10,
        update_every_epochs: int = 3,
        acc_window: int = 3,
        lambda_min: float = 1e-8,
        lambda_max: float = 80.0,
        log_step_init: float = math.log(2.0),
        log_step_min: float = math.log(1.05),
        soft_drop: float = 0.02,
        hard_drop: float = 0.05,
        soft_step_shrink: float = 0.5,
        hard_step_shrink: float = 0.25,
        collapse_acc_threshold: float = 0.15,
        collapse_loss_threshold: float = 2.15,
        collapse_zero_prob_threshold: float = 0.90,
        collapse_acc_drop_threshold: float = 0.40,
        rollback_on_collapse: bool = True,
        max_rollbacks: int = 6,
        freeze_on_rollback_limit: bool = True,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError("adaptive_lambda.warmup_epochs must be >= 0.")
        if update_every_epochs <= 0:
            raise ValueError("adaptive_lambda.update_every_epochs must be >= 1.")
        if acc_window <= 0:
            raise ValueError("adaptive_lambda.acc_window must be >= 1.")
        if lambda_min <= 0.0:
            raise ValueError("adaptive_lambda.lambda_min must be > 0.")
        if lambda_max < lambda_min:
            raise ValueError("adaptive_lambda.lambda_max must be >= lambda_min.")
        if log_step_init <= 0.0:
            raise ValueError("adaptive_lambda.log_step_init must be > 0.")
        if log_step_min <= 0.0:
            raise ValueError("adaptive_lambda.log_step_min must be > 0.")
        if soft_drop < 0.0 or hard_drop < 0.0:
            raise ValueError("adaptive_lambda soft/hard drops must be >= 0.")
        if soft_step_shrink <= 0.0 or hard_step_shrink <= 0.0:
            raise ValueError("adaptive_lambda step shrink factors must be > 0.")
        if max_rollbacks < 0:
            raise ValueError("adaptive_lambda.max_rollbacks must be >= 0.")

        self.warmup_epochs = int(warmup_epochs)
        self.update_every_epochs = int(update_every_epochs)
        self.acc_window = int(acc_window)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.log_step_min = float(log_step_min)
        self.step = max(float(log_step_init), self.log_step_min)
        self.soft_drop = float(soft_drop)
        self.hard_drop = float(hard_drop)
        self.soft_step_shrink = float(soft_step_shrink)
        self.hard_step_shrink = float(hard_step_shrink)
        self.collapse_acc_threshold = float(collapse_acc_threshold)
        self.collapse_loss_threshold = float(collapse_loss_threshold)
        self.collapse_zero_prob_threshold = float(collapse_zero_prob_threshold)
        self.collapse_acc_drop_threshold = float(collapse_acc_drop_threshold)
        self.rollback_on_collapse = bool(rollback_on_collapse)
        self.max_rollbacks = int(max_rollbacks)
        self.freeze_on_rollback_limit = bool(freeze_on_rollback_limit)

        initial_lambda = self._clamp_lambda(float(initial_lambda_coef))
        self.log_lambda = math.log(initial_lambda)
        self.acc_history: deque[float] = deque(maxlen=self.acc_window)
        self.best_val_acc: float | None = None
        self.previous_valid_acc: float | None = None
        self.rollback_count = 0
        self.frozen = False
        self.latest_safe_checkpoint: AdaptiveLambdaCheckpoint | None = None
        self.best_sparse_safe_checkpoint: AdaptiveLambdaCheckpoint | None = None
        self.latest_safe_epoch: int | None = None
        self.latest_safe_lambda: float | None = None
        self.best_sparse_safe_epoch: int | None = None
        self.best_sparse_safe_lambda: float | None = None
        self.last_action = "hold"
        self.last_reason = "initialized"

    @property
    def lambda_coef(self) -> float:
        return self._clamp_lambda(math.exp(self.log_lambda))

    def apply_initial_state(
        self,
        model: nn.Module,
        *,
        apply_lambda: LambdaApplier,
    ) -> None:
        apply_lambda(model, self.lambda_coef)
        print(
            "Adaptive lambda initialized"
            f" | lambda_coef={self.lambda_coef:.12g}"
            f" | log_lambda={self.log_lambda:.12g}"
            f" | step={self.step:.12g}"
        )

    def on_epoch_end(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        valid_metrics: Mapping[str, Any],
        scheduler_state: Any = None,
        scaler: Any = None,
        apply_lambda: LambdaApplier,
    ) -> AdaptiveLambdaStepResult:
        epoch = int(epoch)
        current_lambda_before = self.lambda_coef
        previous_valid_acc = self.previous_valid_acc

        _, valid_acc = _resolve_metric(valid_metrics, ACCURACY_METRIC_NAMES)
        _, valid_loss = _resolve_metric(valid_metrics, LOSS_METRIC_NAMES)
        _, valid_zero_prob = _resolve_metric(valid_metrics, ZERO_PROB_METRIC_NAMES)

        if valid_acc is not None:
            self.acc_history.append(valid_acc)
            if self.best_val_acc is None or valid_acc > self.best_val_acc:
                self.best_val_acc = valid_acc

        acc_ma = self._compute_acc_ma()
        min_allowed_acc = None if self.best_val_acc is None else self.best_val_acc - self.soft_drop
        hard_min_allowed_acc = None if self.best_val_acc is None else self.best_val_acc - self.hard_drop

        checkpoint_saved = self._maybe_save_safe_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            valid_metrics=valid_metrics,
            scheduler_state=scheduler_state,
            scaler=scaler,
            min_allowed_acc=min_allowed_acc,
            acc_ma=acc_ma,
            current_lambda=current_lambda_before,
        )

        action = "hold"
        reason = "waiting_for_next_update"
        lambda_changed = False
        rolled_back = False
        collapse_detected = False

        if epoch <= self.warmup_epochs:
            action = "warmup"
            reason = f"epoch={epoch} <= warmup_epochs={self.warmup_epochs}"
        elif self.frozen:
            action = "hold"
            reason = "adaptive_lambda_frozen"
        else:
            collapse_reason = self._detect_collapse(
                valid_acc=valid_acc,
                valid_loss=valid_loss,
                valid_zero_prob=valid_zero_prob,
                previous_valid_acc=previous_valid_acc,
            )
            if collapse_reason is not None:
                collapse_detected = True
                if self.rollback_on_collapse and self.latest_safe_checkpoint is not None:
                    action, reason, lambda_changed = self._rollback_to_safe_checkpoint(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scheduler_state=scheduler_state,
                        scaler=scaler,
                        apply_lambda=apply_lambda,
                        shrink_factor=self.hard_step_shrink,
                        action_name="collapse_rollback",
                        reason=collapse_reason,
                    )
                    rolled_back = action in {
                        "collapse_rollback",
                        "rollback_limit_freeze",
                    }
                else:
                    action = "hold"
                    if self.latest_safe_checkpoint is None:
                        reason = f"collapse_without_safe_checkpoint ({collapse_reason})"
                    else:
                        reason = f"collapse_detected_but_rollback_disabled ({collapse_reason})"
            elif self._should_update(epoch):
                if acc_ma is None or min_allowed_acc is None or hard_min_allowed_acc is None:
                    action = "hold"
                    reason = "missing_accuracy_feedback"
                elif acc_ma >= min_allowed_acc:
                    safe_log_lambda = (
                        self.latest_safe_checkpoint.log_lambda
                        if self.latest_safe_checkpoint is not None
                        else self.log_lambda
                    )
                    old_lambda = self.lambda_coef
                    self.log_lambda = self._clamp_log_lambda(safe_log_lambda + self.step)
                    new_lambda = self.lambda_coef
                    apply_lambda(model, new_lambda)
                    lambda_changed = abs(new_lambda - old_lambda) > 1e-12
                    action = "increase_lambda"
                    reason = (
                        f"acc_ma={acc_ma:.4f} >= min_allowed_acc={min_allowed_acc:.4f}; "
                        f"lambda {old_lambda:.6g}->{new_lambda:.6g}"
                    )
                    if lambda_changed:
                        self._print_action(epoch, action, reason)
                elif acc_ma >= hard_min_allowed_acc:
                    if self.latest_safe_checkpoint is not None:
                        action, reason, lambda_changed = self._rollback_to_safe_checkpoint(
                            epoch=epoch,
                            model=model,
                            optimizer=optimizer,
                            scheduler_state=scheduler_state,
                            scaler=scaler,
                            apply_lambda=apply_lambda,
                            shrink_factor=self.soft_step_shrink,
                            action_name="soft_degradation_rollback",
                            reason=(
                                f"acc_ma={acc_ma:.4f} < min_allowed_acc={min_allowed_acc:.4f}"
                            ),
                        )
                        rolled_back = action in {
                            "soft_degradation_rollback",
                            "rollback_limit_freeze",
                        }
                    else:
                        action = "hold"
                        reason = "soft_degradation_without_safe_checkpoint"
                else:
                    if self.latest_safe_checkpoint is not None:
                        action, reason, lambda_changed = self._rollback_to_safe_checkpoint(
                            epoch=epoch,
                            model=model,
                            optimizer=optimizer,
                            scheduler_state=scheduler_state,
                            scaler=scaler,
                            apply_lambda=apply_lambda,
                            shrink_factor=self.hard_step_shrink,
                            action_name="hard_degradation_rollback",
                            reason=(
                                f"acc_ma={acc_ma:.4f} < hard_min_allowed_acc={hard_min_allowed_acc:.4f}"
                            ),
                        )
                        rolled_back = action in {
                            "hard_degradation_rollback",
                            "rollback_limit_freeze",
                        }
                    else:
                        action = "hold"
                        reason = "hard_degradation_without_safe_checkpoint"

        if not rolled_back and valid_acc is not None:
            self.previous_valid_acc = valid_acc

        self.last_action = action
        self.last_reason = reason
        metrics = self._build_epoch_metrics(
            action=action,
            reason=reason,
            acc_ma=acc_ma,
            min_allowed_acc=min_allowed_acc,
            hard_min_allowed_acc=hard_min_allowed_acc,
        )
        return AdaptiveLambdaStepResult(
            action=action,
            reason=reason,
            lambda_changed=lambda_changed,
            rolled_back=rolled_back,
            collapse_detected=collapse_detected,
            checkpoint_saved=checkpoint_saved,
            metrics=metrics,
        )

    def summary_state(self) -> dict[str, Any]:
        latest_safe_metric_name = None
        latest_safe_metric_value = None
        if self.latest_safe_checkpoint is not None:
            latest_safe_metric_name = self.latest_safe_checkpoint.pruning_metric_name
            latest_safe_metric_value = self.latest_safe_checkpoint.pruning_metric_value

        best_sparse_metric_name = None
        best_sparse_metric_value = None
        if self.best_sparse_safe_checkpoint is not None:
            best_sparse_metric_name = self.best_sparse_safe_checkpoint.pruning_metric_name
            best_sparse_metric_value = self.best_sparse_safe_checkpoint.pruning_metric_value

        return {
            "enabled": True,
            "lambda_coef": self.lambda_coef,
            "log_lambda": self.log_lambda,
            "adaptive_lambda_step": self.step,
            "adaptive_lambda_action": self.last_action,
            "adaptive_lambda_reason": self.last_reason,
            "adaptive_lambda_rollbacks": self.rollback_count,
            "best_val_acc": self.best_val_acc,
            "latest_safe_epoch": self.latest_safe_epoch,
            "latest_safe_lambda": self.latest_safe_lambda,
            "latest_safe_pruning_metric_name": latest_safe_metric_name,
            "latest_safe_pruning_metric_value": latest_safe_metric_value,
            "best_sparse_safe_epoch": self.best_sparse_safe_epoch,
            "best_sparse_safe_lambda": self.best_sparse_safe_lambda,
            "best_sparse_safe_pruning_metric_name": best_sparse_metric_name,
            "best_sparse_safe_pruning_metric_value": best_sparse_metric_value,
            "frozen": self.frozen,
        }

    def _compute_acc_ma(self) -> float | None:
        if not self.acc_history:
            return None
        return float(sum(self.acc_history) / len(self.acc_history))

    def _clamp_lambda(self, value: float) -> float:
        return float(min(max(value, self.lambda_min), self.lambda_max))

    def _clamp_log_lambda(self, value: float) -> float:
        return math.log(self._clamp_lambda(math.exp(float(value))))

    def _should_update(self, epoch: int) -> bool:
        if epoch <= self.warmup_epochs:
            return False
        return (epoch - self.warmup_epochs) % self.update_every_epochs == 0

    def _detect_collapse(
        self,
        *,
        valid_acc: float | None,
        valid_loss: float | None,
        valid_zero_prob: float | None,
        previous_valid_acc: float | None,
    ) -> str | None:
        if valid_acc is not None and valid_acc < self.collapse_acc_threshold:
            return (
                f"valid_acc={valid_acc:.4f} < "
                f"collapse_acc_threshold={self.collapse_acc_threshold:.4f}"
            )
        if (
            valid_loss is not None
            and valid_zero_prob is not None
            and valid_loss > self.collapse_loss_threshold
            and valid_zero_prob > self.collapse_zero_prob_threshold
        ):
            return (
                f"valid_loss={valid_loss:.4f} > {self.collapse_loss_threshold:.4f}"
                f" and valid_zero_prob={valid_zero_prob:.4f} > {self.collapse_zero_prob_threshold:.4f}"
            )
        if (
            previous_valid_acc is not None
            and valid_acc is not None
            and previous_valid_acc - valid_acc >= self.collapse_acc_drop_threshold
        ):
            return (
                f"previous_valid_acc-valid_acc={previous_valid_acc - valid_acc:.4f}"
                f" >= {self.collapse_acc_drop_threshold:.4f}"
            )
        return None

    def _controller_state_snapshot(self) -> dict[str, Any]:
        return {
            "acc_history": list(self.acc_history),
            "previous_valid_acc": self.previous_valid_acc,
            "best_val_acc": self.best_val_acc,
        }

    def _capture_checkpoint(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        valid_metrics: Mapping[str, Any],
        scheduler_state: Any,
        scaler: Any,
        current_lambda: float,
    ) -> AdaptiveLambdaCheckpoint:
        pruning_metric_name, pruning_metric_value = self._resolve_pruning_metric(valid_metrics)
        scheduler_state_dict = None
        scheduler_step_count = None
        if scheduler_state is not None and getattr(scheduler_state, "scheduler", None) is not None:
            scheduler_state_dict = _clone_to_cpu(deepcopy(scheduler_state.scheduler.state_dict()))
            scheduler_step_count = int(getattr(scheduler_state, "step_count", 0))

        scaler_state_dict = None
        if scaler is not None and hasattr(scaler, "state_dict"):
            scaler_state_dict = _clone_to_cpu(deepcopy(scaler.state_dict()))

        return AdaptiveLambdaCheckpoint(
            epoch=int(epoch),
            lambda_coef=float(current_lambda),
            log_lambda=float(self.log_lambda),
            pruning_metric_name=pruning_metric_name,
            pruning_metric_value=pruning_metric_value,
            model_state_dict=_clone_to_cpu(deepcopy(model.state_dict())),
            optimizer_state_dict=_clone_to_cpu(deepcopy(optimizer.state_dict())),
            scheduler_state_dict=scheduler_state_dict,
            scheduler_step_count=scheduler_step_count,
            scaler_state_dict=scaler_state_dict,
            controller_state=self._controller_state_snapshot(),
        )

    def _resolve_pruning_metric(self, valid_metrics: Mapping[str, Any]) -> tuple[str | None, float | None]:
        metric_name, metric_value = _resolve_metric(valid_metrics, EXPECTED_ACTIVE_CHANNEL_METRIC_NAMES)
        if metric_value is not None:
            return metric_name, metric_value
        return _resolve_metric(valid_metrics, ZERO_PROB_METRIC_NAMES)

    def _maybe_save_safe_checkpoint(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        valid_metrics: Mapping[str, Any],
        scheduler_state: Any,
        scaler: Any,
        min_allowed_acc: float | None,
        acc_ma: float | None,
        current_lambda: float,
    ) -> bool:
        if min_allowed_acc is None or acc_ma is None or acc_ma < min_allowed_acc:
            return False

        checkpoint = self._capture_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            valid_metrics=valid_metrics,
            scheduler_state=scheduler_state,
            scaler=scaler,
            current_lambda=current_lambda,
        )
        self.latest_safe_checkpoint = checkpoint
        self.latest_safe_epoch = checkpoint.epoch
        self.latest_safe_lambda = checkpoint.lambda_coef

        if self._is_better_sparse_checkpoint(checkpoint, self.best_sparse_safe_checkpoint):
            self.best_sparse_safe_checkpoint = checkpoint
            self.best_sparse_safe_epoch = checkpoint.epoch
            self.best_sparse_safe_lambda = checkpoint.lambda_coef
        return True

    def _is_better_sparse_checkpoint(
        self,
        candidate: AdaptiveLambdaCheckpoint,
        current_best: AdaptiveLambdaCheckpoint | None,
    ) -> bool:
        if current_best is None:
            return True

        candidate_name = candidate.pruning_metric_name
        current_name = current_best.pruning_metric_name
        candidate_value = candidate.pruning_metric_value
        current_value = current_best.pruning_metric_value

        if candidate_name in EXPECTED_ACTIVE_CHANNEL_METRIC_NAMES:
            if current_name not in EXPECTED_ACTIVE_CHANNEL_METRIC_NAMES:
                return True
            if current_value is None or candidate_value is None:
                return current_best.epoch < candidate.epoch
            return float(candidate_value) < float(current_value)

        if current_name in EXPECTED_ACTIVE_CHANNEL_METRIC_NAMES:
            return False

        if candidate_value is None:
            return False
        if current_value is None:
            return True
        return float(candidate_value) > float(current_value)

    def _restore_checkpoint(
        self,
        checkpoint: AdaptiveLambdaCheckpoint,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler_state: Any,
        scaler: Any,
        apply_lambda: LambdaApplier,
    ) -> None:
        model.load_state_dict(checkpoint.model_state_dict)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        _move_optimizer_state_to_device(optimizer)

        if (
            scheduler_state is not None
            and getattr(scheduler_state, "scheduler", None) is not None
            and checkpoint.scheduler_state_dict is not None
        ):
            scheduler_state.scheduler.load_state_dict(checkpoint.scheduler_state_dict)
            if checkpoint.scheduler_step_count is not None:
                scheduler_state.step_count = int(checkpoint.scheduler_step_count)

        if scaler is not None and checkpoint.scaler_state_dict is not None and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(checkpoint.scaler_state_dict)

        apply_lambda(model, checkpoint.lambda_coef)

        snapshot = checkpoint.controller_state
        self.acc_history = deque(
            [float(value) for value in snapshot.get("acc_history", [])],
            maxlen=self.acc_window,
        )
        self.previous_valid_acc = _to_float(snapshot.get("previous_valid_acc"))
        checkpoint_best_val_acc = _to_float(snapshot.get("best_val_acc"))
        if self.best_val_acc is None:
            self.best_val_acc = checkpoint_best_val_acc
        elif checkpoint_best_val_acc is not None:
            self.best_val_acc = max(self.best_val_acc, checkpoint_best_val_acc)

    def _rollback_to_safe_checkpoint(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler_state: Any,
        scaler: Any,
        apply_lambda: LambdaApplier,
        shrink_factor: float,
        action_name: str,
        reason: str,
    ) -> tuple[str, str, bool]:
        checkpoint = self.latest_safe_checkpoint
        if checkpoint is None:
            return "hold", f"{action_name}_without_safe_checkpoint", False

        old_lambda = self.lambda_coef
        self._restore_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler_state=scheduler_state,
            scaler=scaler,
            apply_lambda=apply_lambda,
        )

        self.rollback_count += 1
        self.step = max(self.log_step_min, self.step * float(shrink_factor))

        if self.rollback_count > self.max_rollbacks:
            self.frozen = True
            self.log_lambda = float(checkpoint.log_lambda)
            apply_lambda(model, checkpoint.lambda_coef)
            freeze_reason = (
                f"rollback_limit_reached ({self.rollback_count}>{self.max_rollbacks}); "
                f"freezing at latest safe lambda={checkpoint.lambda_coef:.6g}"
            )
            self._print_action(epoch, "rollback_limit_freeze", freeze_reason)
            return "rollback_limit_freeze", freeze_reason, abs(old_lambda - checkpoint.lambda_coef) > 1e-12

        safe_log_lambda = float(checkpoint.log_lambda)
        self.log_lambda = self._clamp_log_lambda(safe_log_lambda + self.step)
        new_lambda = self.lambda_coef
        apply_lambda(model, new_lambda)

        full_reason = (
            f"{reason}; rolled back to epoch={checkpoint.epoch}; "
            f"lambda {old_lambda:.6g}->{new_lambda:.6g}; step={self.step:.6g}"
        )
        self._print_action(epoch, action_name, full_reason)
        return action_name, full_reason, abs(new_lambda - old_lambda) > 1e-12

    def _build_epoch_metrics(
        self,
        *,
        action: str,
        reason: str,
        acc_ma: float | None,
        min_allowed_acc: float | None,
        hard_min_allowed_acc: float | None,
    ) -> dict[str, Any]:
        return {
            "lambda_coef": self.lambda_coef,
            "log_lambda": self.log_lambda,
            "adaptive_lambda_step": self.step,
            "adaptive_lambda_action": action,
            "adaptive_lambda_reason": reason,
            "adaptive_lambda_rollbacks": self.rollback_count,
            "best_val_acc": self.best_val_acc,
            "acc_ma": acc_ma,
            "min_allowed_acc": min_allowed_acc,
            "hard_min_allowed_acc": hard_min_allowed_acc,
            "latest_safe_epoch": self.latest_safe_epoch,
            "latest_safe_lambda": self.latest_safe_lambda,
            "best_sparse_safe_epoch": self.best_sparse_safe_epoch,
            "best_sparse_safe_lambda": self.best_sparse_safe_lambda,
        }

    def _print_action(self, epoch: int, action: str, reason: str) -> None:
        print(
            "Adaptive lambda update"
            f" | epoch={epoch}"
            f" | action={action}"
            f" | lambda_coef={self.lambda_coef:.12g}"
            f" | log_lambda={self.log_lambda:.12g}"
            f" | step={self.step:.12g}"
            f" | reason={reason}"
        )
