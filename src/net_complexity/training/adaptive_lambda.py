from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn


LambdaApplier = Callable[[nn.Module, float], None]

ACCURACY_METRIC_NAMES = (
    "valid_accuracy",
    "accuracy",
)
ZERO_PROB_METRIC_NAMES = (
    "valid_average_zero_prob",
    "valid_zero_prob",
    "average_zero_prob",
    "zero_prob",
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


def _set_model_gumbel_open_bias(
    model: nn.Module,
    value: float,
    *,
    p_min: float,
    p_max: float,
) -> None:
    if hasattr(model, "set_gumbel_open_bias"):
        model.set_gumbel_open_bias(value, p_min=p_min, p_max=p_max)
        return

    for module in model.modules():
        if hasattr(module, "set_open_bias"):
            module.set_open_bias(value, p_min=p_min, p_max=p_max)


def _collect_model_gumbel_open_bias_stats(
    model: nn.Module,
    *,
    p_min: float,
    p_max: float,
) -> dict[str, float | None]:
    total_channels = 0
    total_candidates = 0
    sum_p_open_candidates = 0.0
    sum_p_open_all = 0.0

    for module in model.modules():
        if not hasattr(module, "get_open_bias_candidate_stats"):
            continue
        stats = module.get_open_bias_candidate_stats(p_min=p_min, p_max=p_max)
        num_channels = int(stats.get("num_channels", 0))
        revive_candidates = int(stats.get("revive_candidates", 0))
        total_channels += num_channels
        total_candidates += revive_candidates
        sum_p_open_candidates += float(stats.get("sum_p_open_candidates", 0.0))
        sum_p_open_all += float(stats.get("sum_p_open_all", 0.0))

    if total_channels <= 0:
        return {
            "recovery_revive_candidates_frac": None,
            "recovery_mean_p_open_candidates": None,
            "recovery_mean_p_open_all": None,
        }

    return {
        "recovery_revive_candidates_frac": float(total_candidates / total_channels),
        "recovery_mean_p_open_candidates": (
            float(sum_p_open_candidates / total_candidates)
            if total_candidates > 0
            else None
        ),
        "recovery_mean_p_open_all": float(sum_p_open_all / total_channels),
    }


@dataclass(frozen=True)
class AdaptiveLambdaRecoveryConfig:
    enabled: bool = True
    min_epoch: int = 80
    drop_min: float = 0.02
    drop_max: float = 0.20
    patience: int = 5
    recovery_slope_window: int = 5
    require_slow_recovery: bool = True
    min_acc_delta_over_window: float = 0.005
    use_zero_prob_filter: bool = True
    zero_prob_window: int = 10
    zero_prob_delta_min: float = 0.02
    recovery_epochs: int = 5
    freeze_lambda_increase: bool = True
    decay_lambda: bool = False
    open_bias_start: float = 0.15
    open_bias_decay: float = 0.90
    p_open_min: float = 0.02
    p_open_max: float = 0.50
    max_reopen_delta: float = 0.02
    target_acc_margin: float = 0.01
    cooldown_epochs: int = 10
    max_recovery_attempts: int = 3

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "AdaptiveLambdaRecoveryConfig":
        if value is None:
            return cls(enabled=False)
        data = dict(value)
        config = cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            min_epoch=int(data.get("min_epoch", cls.min_epoch)),
            drop_min=float(data.get("drop_min", cls.drop_min)),
            drop_max=float(data.get("drop_max", cls.drop_max)),
            patience=int(data.get("patience", cls.patience)),
            recovery_slope_window=int(data.get("recovery_slope_window", cls.recovery_slope_window)),
            require_slow_recovery=bool(data.get("require_slow_recovery", cls.require_slow_recovery)),
            min_acc_delta_over_window=float(
                data.get("min_acc_delta_over_window", cls.min_acc_delta_over_window)
            ),
            use_zero_prob_filter=bool(data.get("use_zero_prob_filter", cls.use_zero_prob_filter)),
            zero_prob_window=int(data.get("zero_prob_window", cls.zero_prob_window)),
            zero_prob_delta_min=float(data.get("zero_prob_delta_min", cls.zero_prob_delta_min)),
            recovery_epochs=int(data.get("recovery_epochs", cls.recovery_epochs)),
            freeze_lambda_increase=bool(data.get("freeze_lambda_increase", cls.freeze_lambda_increase)),
            decay_lambda=bool(data.get("decay_lambda", cls.decay_lambda)),
            open_bias_start=float(data.get("open_bias_start", cls.open_bias_start)),
            open_bias_decay=float(data.get("open_bias_decay", cls.open_bias_decay)),
            p_open_min=float(data.get("p_open_min", cls.p_open_min)),
            p_open_max=float(data.get("p_open_max", cls.p_open_max)),
            max_reopen_delta=float(data.get("max_reopen_delta", cls.max_reopen_delta)),
            target_acc_margin=float(data.get("target_acc_margin", cls.target_acc_margin)),
            cooldown_epochs=int(data.get("cooldown_epochs", cls.cooldown_epochs)),
            max_recovery_attempts=int(data.get("max_recovery_attempts", cls.max_recovery_attempts)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.min_epoch < 0:
            raise ValueError("adaptive_lambda.recovery.min_epoch must be >= 0.")
        if not 0.0 <= self.drop_min < self.drop_max:
            raise ValueError("adaptive_lambda.recovery requires 0.0 <= drop_min < drop_max.")
        if self.patience <= 0:
            raise ValueError("adaptive_lambda.recovery.patience must be >= 1.")
        if self.recovery_slope_window <= 0:
            raise ValueError("adaptive_lambda.recovery.recovery_slope_window must be >= 1.")
        if self.zero_prob_window <= 0:
            raise ValueError("adaptive_lambda.recovery.zero_prob_window must be >= 1.")
        if self.recovery_epochs <= 0:
            raise ValueError("adaptive_lambda.recovery.recovery_epochs must be >= 1.")
        if self.open_bias_start < 0.0:
            raise ValueError("adaptive_lambda.recovery.open_bias_start must be >= 0.")
        if not 0.0 < self.open_bias_decay <= 1.0:
            raise ValueError("adaptive_lambda.recovery.open_bias_decay must be within (0.0, 1.0].")
        if not 0.0 <= self.p_open_min < self.p_open_max <= 1.0:
            raise ValueError("adaptive_lambda.recovery requires 0.0 <= p_open_min < p_open_max <= 1.0.")
        if self.max_reopen_delta < 0.0:
            raise ValueError("adaptive_lambda.recovery.max_reopen_delta must be >= 0.")
        if self.target_acc_margin < 0.0:
            raise ValueError("adaptive_lambda.recovery.target_acc_margin must be >= 0.")
        if self.cooldown_epochs < 0:
            raise ValueError("adaptive_lambda.recovery.cooldown_epochs must be >= 0.")
        if self.max_recovery_attempts < 0:
            raise ValueError("adaptive_lambda.recovery.max_recovery_attempts must be >= 0.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_epoch": self.min_epoch,
            "drop_min": self.drop_min,
            "drop_max": self.drop_max,
            "patience": self.patience,
            "recovery_slope_window": self.recovery_slope_window,
            "require_slow_recovery": self.require_slow_recovery,
            "min_acc_delta_over_window": self.min_acc_delta_over_window,
            "use_zero_prob_filter": self.use_zero_prob_filter,
            "zero_prob_window": self.zero_prob_window,
            "zero_prob_delta_min": self.zero_prob_delta_min,
            "recovery_epochs": self.recovery_epochs,
            "freeze_lambda_increase": self.freeze_lambda_increase,
            "decay_lambda": self.decay_lambda,
            "open_bias_start": self.open_bias_start,
            "open_bias_decay": self.open_bias_decay,
            "p_open_min": self.p_open_min,
            "p_open_max": self.p_open_max,
            "max_reopen_delta": self.max_reopen_delta,
            "target_acc_margin": self.target_acc_margin,
            "cooldown_epochs": self.cooldown_epochs,
            "max_recovery_attempts": self.max_recovery_attempts,
        }


@dataclass(frozen=True)
class RecoveryUpdate:
    metrics: dict[str, Any]
    block_lambda_increase: bool = False


@dataclass(frozen=True)
class AdaptiveLogStepUpdate:
    effective_log_step: float
    prune_rate_per_epoch: float | None
    step_action: str


@dataclass
class AdaptiveLambdaStepResult:
    action: str
    reason: str
    lambda_changed: bool
    metrics: dict[str, Any]


class AdaptiveLambdaController:
    def __init__(
        self,
        *,
        initial_lambda_coef: float,
        reference_accuracy_by_epoch: Mapping[int, float] | None = None,
        warmup_epochs: int = 10,
        update_every_epochs: int = 3,
        acc_window: int = 3,
        lambda_min: float = 1e-8,
        lambda_max: float | None = 80.0,
        log_step_init: float = math.log(2.0),
        log_step_min: float = math.log(1.05),
        adaptive_log_step_enabled: bool = True,
        prune_rate_low_per_epoch: float = 0.02,
        prune_rate_high_per_epoch: float = 0.07,
        log_step_boost_factor: float = 2.0,
        log_step_max_boost_level: int = 2,
        adaptive_log_step_max_epoch: int | None = None,
        soft_drop: float = 0.02,
        hard_drop: float = 0.05,
        recovery_config: Mapping[str, Any] | None = None,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError("adaptive_lambda.warmup_epochs must be >= 0.")
        if update_every_epochs <= 0:
            raise ValueError("adaptive_lambda.update_every_epochs must be >= 1.")
        if acc_window <= 0:
            raise ValueError("adaptive_lambda.acc_window must be >= 1.")
        if lambda_min <= 0.0:
            raise ValueError("adaptive_lambda.lambda_min must be > 0.")
        if lambda_max is not None and lambda_max < lambda_min:
            raise ValueError("adaptive_lambda.lambda_max must be >= lambda_min.")
        if log_step_init <= 0.0:
            raise ValueError("adaptive_lambda.log_step_init must be > 0.")
        if log_step_min <= 0.0:
            raise ValueError("adaptive_lambda.log_step_min must be > 0.")
        if prune_rate_low_per_epoch < 0.0:
            raise ValueError("adaptive_lambda.prune_rate_low_per_epoch must be >= 0.")
        if prune_rate_high_per_epoch < prune_rate_low_per_epoch:
            raise ValueError(
                "adaptive_lambda.prune_rate_high_per_epoch must be >= "
                "prune_rate_low_per_epoch."
            )
        if log_step_boost_factor <= 0.0:
            raise ValueError("adaptive_lambda.log_step_boost_factor must be > 0.")
        if log_step_max_boost_level < 0:
            raise ValueError("adaptive_lambda.log_step_max_boost_level must be >= 0.")
        if adaptive_log_step_max_epoch is not None and adaptive_log_step_max_epoch < 0:
            raise ValueError("adaptive_lambda.adaptive_log_step_max_epoch must be >= 0.")
        if soft_drop < 0.0 or hard_drop < 0.0:
            raise ValueError("adaptive_lambda soft/hard drops must be >= 0.")

        self.warmup_epochs = int(warmup_epochs)
        self.update_every_epochs = int(update_every_epochs)
        self.acc_window = int(acc_window)
        self.lambda_min = float(lambda_min)
        self.lambda_max = None if lambda_max is None else float(lambda_max)
        self.log_step_min = float(log_step_min)
        self.step = max(float(log_step_init), self.log_step_min)
        self.adaptive_log_step_enabled = bool(adaptive_log_step_enabled)
        self.prune_rate_low_per_epoch = float(prune_rate_low_per_epoch)
        self.prune_rate_high_per_epoch = float(prune_rate_high_per_epoch)
        self.log_step_boost_factor = float(log_step_boost_factor)
        self.log_step_max_boost_level = int(log_step_max_boost_level)
        self.adaptive_log_step_max_epoch = (
            None
            if adaptive_log_step_max_epoch is None
            else int(adaptive_log_step_max_epoch)
        )
        self.soft_drop = float(soft_drop)
        self.hard_drop = float(hard_drop)
        self.recovery_config = AdaptiveLambdaRecoveryConfig.from_mapping(recovery_config)

        initial_lambda = self._clamp_lambda(float(initial_lambda_coef))
        self.log_lambda = math.log(initial_lambda)
        self.acc_history: deque[float] = deque(maxlen=self.acc_window)
        self.reference_accuracy_by_epoch = (
            {
                int(epoch): float(accuracy)
                for epoch, accuracy in dict(reference_accuracy_by_epoch).items()
            }
            if reference_accuracy_by_epoch is not None
            else {}
        )
        self.best_val_acc: float | None = None
        self.observed_accuracy_by_epoch: dict[int, float] = {}
        self.observed_zero_prob_by_epoch: dict[int, float] = {}
        self.recovery_active = False
        self.recovery_epochs_left = 0
        self.recovery_start_epoch: int | None = None
        self.recovery_start_active_ratio: float | None = None
        self.recovery_start_zero_prob: float | None = None
        self.recovery_start_acc: float | None = None
        self.recovery_open_bias = 0.0
        self.recovery_attempts = 0
        self.recovery_cooldown_left = 0
        self.recovery_condition_epochs = 0
        self.recovery_total_epochs = 0
        self.recovery_first_start_active_ratio: float | None = None
        self.recovery_last_active_ratio: float | None = None
        self.recovery_best_acc_after_recovery: float | None = None
        self.log_step_boost_level = 0
        self.previous_control_zero_prob: float | None = None
        self.previous_control_epoch: int | None = None
        self.latest_effective_log_step = self.step
        self.latest_prune_rate_per_epoch: float | None = None
        self.latest_step_action = (
            "step_disabled"
            if not self.adaptive_log_step_enabled
            else "step_init_no_prev_zero_prob"
        )
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
        self._apply_recovery_open_bias(model, 0.0)
        print(
            "Adaptive lambda initialized"
            f" | lambda_coef={self.lambda_coef:.12g}"
            f" | log_lambda={self.log_lambda:.12g}"
            f" | step={self.step:.12g}"
            f" | recovery_enabled={str(self.recovery_config.enabled).lower()}"
            f" | recovery_min_epoch={self.recovery_config.min_epoch}"
            f" | recovery_patience={self.recovery_config.patience}"
            f" | recovery_epochs={self.recovery_config.recovery_epochs}"
            f" | recovery_max_attempts={self.recovery_config.max_recovery_attempts}"
        )

    def on_epoch_end(
        self,
        *,
        epoch: int,
        model: nn.Module,
        valid_metrics: Mapping[str, Any],
        apply_lambda: LambdaApplier,
    ) -> AdaptiveLambdaStepResult:
        epoch = int(epoch)

        _, valid_acc = _resolve_metric(valid_metrics, ACCURACY_METRIC_NAMES)
        _, valid_zero_prob = _resolve_metric(valid_metrics, ZERO_PROB_METRIC_NAMES)

        if valid_acc is not None:
            self.acc_history.append(valid_acc)
            self.observed_accuracy_by_epoch[epoch] = float(valid_acc)
            if self.best_val_acc is None or valid_acc > self.best_val_acc:
                self.best_val_acc = valid_acc
        if valid_zero_prob is not None:
            self.observed_zero_prob_by_epoch[epoch] = float(valid_zero_prob)
        if self.recovery_attempts > 0 and valid_acc is not None:
            if (
                self.recovery_best_acc_after_recovery is None
                or valid_acc > self.recovery_best_acc_after_recovery
            ):
                self.recovery_best_acc_after_recovery = float(valid_acc)

        acc_ma = self._compute_acc_ma()
        reference_epoch, reference_acc = self._resolve_reference_accuracy(epoch)
        if reference_acc is None:
            reference_acc = self.best_val_acc
        min_allowed_acc = None if reference_acc is None else reference_acc - self.soft_drop
        hard_min_allowed_acc = None if reference_acc is None else reference_acc - self.hard_drop

        action = "hold"
        reason = "waiting_for_next_update"
        lambda_changed = False

        recovery_update = self._update_recovery(
            epoch=epoch,
            model=model,
            valid_acc=valid_acc,
            valid_zero_prob=valid_zero_prob,
        )

        if epoch <= self.warmup_epochs:
            action = "warmup"
            reason = f"epoch={epoch} <= warmup_epochs={self.warmup_epochs}"
        elif self._should_update(epoch):
            if acc_ma is None or min_allowed_acc is None or hard_min_allowed_acc is None:
                action = "hold"
                reason = "missing_accuracy_feedback"
            elif acc_ma >= min_allowed_acc:
                step_update = self._update_adaptive_log_step(
                    epoch=epoch,
                    valid_zero_prob=valid_zero_prob,
                    acc_ok=True,
                )
                effective_log_step = step_update.effective_log_step
                if recovery_update.block_lambda_increase:
                    action = "recovery_blocked_lambda_increase"
                    reason = (
                        f"acc_ma={acc_ma:.4f} >= min_allowed_acc={min_allowed_acc:.4f}; "
                        "lambda increase blocked by recovery"
                    )
                else:
                    old_lambda = self.lambda_coef
                    self.log_lambda = self._clamp_log_lambda(
                        self.log_lambda + effective_log_step
                    )
                    new_lambda = self.lambda_coef
                    apply_lambda(model, new_lambda)
                    lambda_changed = abs(new_lambda - old_lambda) > 1e-12
                    action = "increase_lambda"
                    reason = (
                        f"acc_ma={acc_ma:.4f} >= min_allowed_acc={min_allowed_acc:.4f}; "
                        f"lambda {old_lambda:.6g}->{new_lambda:.6g}; "
                        f"effective_log_step={effective_log_step:.6g}"
                    )
                    if lambda_changed:
                        self._print_action(epoch, action, reason)
            elif acc_ma >= hard_min_allowed_acc:
                self._update_adaptive_log_step(
                    epoch=epoch,
                    valid_zero_prob=valid_zero_prob,
                    acc_ok=False,
                )
                action = "hold"
                reason = (
                    f"hard_min_allowed_acc={hard_min_allowed_acc:.4f} <= "
                    f"acc_ma={acc_ma:.4f} < min_allowed_acc={min_allowed_acc:.4f}; "
                    "lambda unchanged"
                )
            else:
                step_update = self._update_adaptive_log_step(
                    epoch=epoch,
                    valid_zero_prob=valid_zero_prob,
                    acc_ok=False,
                )
                effective_log_step = step_update.effective_log_step
                old_lambda = self.lambda_coef
                self.log_lambda = self._clamp_log_lambda(
                    self.log_lambda - effective_log_step
                )
                new_lambda = self.lambda_coef
                apply_lambda(model, new_lambda)
                lambda_changed = abs(new_lambda - old_lambda) > 1e-12
                action = "decrease_lambda"
                reason = (
                    f"acc_ma={acc_ma:.4f} < hard_min_allowed_acc={hard_min_allowed_acc:.4f}; "
                    f"lambda {old_lambda:.6g}->{new_lambda:.6g}; "
                    f"effective_log_step={effective_log_step:.6g}"
                )
                if lambda_changed:
                    self._print_action(epoch, action, reason)

        self._maybe_reset_adaptive_log_step_after_max_epoch(epoch=epoch)

        self.last_action = action
        self.last_reason = reason
        metrics = self._build_epoch_metrics(
            action=action,
            reason=reason,
            acc_ma=acc_ma,
            reference_epoch=reference_epoch,
            reference_acc=reference_acc,
            min_allowed_acc=min_allowed_acc,
            hard_min_allowed_acc=hard_min_allowed_acc,
        )
        metrics.update(recovery_update.metrics)
        return AdaptiveLambdaStepResult(
            action=action,
            reason=reason,
            lambda_changed=lambda_changed,
            metrics=metrics,
        )

    def summary_state(self) -> dict[str, Any]:
        recovery_final_active_delta = (
            None
            if self.recovery_first_start_active_ratio is None or self.recovery_last_active_ratio is None
            else float(self.recovery_last_active_ratio - self.recovery_first_start_active_ratio)
        )

        return {
            "enabled": True,
            "lambda_coef": self.lambda_coef,
            "log_lambda": self.log_lambda,
            "adaptive_lambda_step": self.step,
            "adaptive_lambda_effective_log_step": self.latest_effective_log_step,
            "adaptive_lambda_log_step_boost_level": self.log_step_boost_level,
            "adaptive_lambda_prune_rate_per_epoch": self.latest_prune_rate_per_epoch,
            "adaptive_lambda_step_action": self.latest_step_action,
            "adaptive_log_step_max_epoch": self.adaptive_log_step_max_epoch,
            "adaptive_lambda_action": self.last_action,
            "adaptive_lambda_reason": self.last_reason,
            "reference_source": (
                "baseline_history"
                if self.reference_accuracy_by_epoch
                else "best_val_acc"
            ),
            "reference_num_epochs": len(self.reference_accuracy_by_epoch),
            "best_val_acc": self.best_val_acc,
            "recovery_num_attempts": self.recovery_attempts,
            "recovery_total_epochs": self.recovery_total_epochs,
            "recovery_was_used": self.recovery_attempts > 0,
            "recovery_final_active_delta": recovery_final_active_delta,
            "recovery_best_acc_after_recovery": self.recovery_best_acc_after_recovery,
            "recovery_config": self.recovery_config.as_dict(),
        }

    def _compute_acc_ma(self) -> float | None:
        if not self.acc_history:
            return None
        return float(sum(self.acc_history) / len(self.acc_history))

    def _resolve_reference_accuracy(self, epoch: int) -> tuple[int | None, float | None]:
        if not self.reference_accuracy_by_epoch:
            return None, None
        if epoch in self.reference_accuracy_by_epoch:
            return int(epoch), float(self.reference_accuracy_by_epoch[epoch])

        candidate_epochs = [candidate for candidate in self.reference_accuracy_by_epoch if candidate <= epoch]
        if not candidate_epochs:
            return None, None

        resolved_epoch = max(candidate_epochs)
        return int(resolved_epoch), float(self.reference_accuracy_by_epoch[resolved_epoch])

    def _clamp_lambda(self, value: float) -> float:
        clamped = max(float(value), self.lambda_min)
        if self.lambda_max is not None:
            clamped = min(clamped, self.lambda_max)
        return float(clamped)

    def _clamp_log_lambda(self, value: float) -> float:
        log_value = float(value)
        log_min = math.log(self.lambda_min)
        if log_value < log_min:
            return log_min
        if self.lambda_max is not None:
            log_max = math.log(self.lambda_max)
            if log_value > log_max:
                return log_max
        return log_value

    def _should_update(self, epoch: int) -> bool:
        if epoch <= self.warmup_epochs:
            return False
        return (epoch - self.warmup_epochs) % self.update_every_epochs == 0

    def _adaptive_log_step_is_after_max_epoch(self, epoch: int) -> bool:
        return (
            self.adaptive_log_step_enabled
            and self.adaptive_log_step_max_epoch is not None
            and int(epoch) > self.adaptive_log_step_max_epoch
        )

    def _base_log_step(self) -> float:
        return max(float(self.step), self.log_step_min)

    def _current_effective_log_step(self) -> float:
        return float(
            self._base_log_step()
            * (self.log_step_boost_factor ** self.log_step_boost_level)
        )

    def _set_log_step_update_state(
        self,
        *,
        effective_log_step: float,
        prune_rate_per_epoch: float | None,
        step_action: str,
    ) -> AdaptiveLogStepUpdate:
        self.latest_effective_log_step = float(effective_log_step)
        self.latest_prune_rate_per_epoch = prune_rate_per_epoch
        self.latest_step_action = str(step_action)
        return AdaptiveLogStepUpdate(
            effective_log_step=float(effective_log_step),
            prune_rate_per_epoch=prune_rate_per_epoch,
            step_action=str(step_action),
        )

    def _store_control_zero_prob(
        self,
        *,
        epoch: int,
        valid_zero_prob: float | None,
    ) -> None:
        if valid_zero_prob is None:
            return
        self.previous_control_zero_prob = float(valid_zero_prob)
        self.previous_control_epoch = int(epoch)

    def _reset_log_step_boost(self, *, step_action: str) -> AdaptiveLogStepUpdate:
        self.log_step_boost_level = 0
        return self._set_log_step_update_state(
            effective_log_step=self._base_log_step(),
            prune_rate_per_epoch=None,
            step_action=step_action,
        )

    def _maybe_reset_adaptive_log_step_after_max_epoch(self, *, epoch: int) -> None:
        if not self._adaptive_log_step_is_after_max_epoch(epoch):
            return
        self._reset_log_step_boost(step_action="step_reset_after_max_epoch")

    def _update_adaptive_log_step(
        self,
        *,
        epoch: int,
        valid_zero_prob: float | None,
        acc_ok: bool,
    ) -> AdaptiveLogStepUpdate:
        if not self.adaptive_log_step_enabled:
            self.log_step_boost_level = 0
            self._store_control_zero_prob(epoch=epoch, valid_zero_prob=valid_zero_prob)
            return self._set_log_step_update_state(
                effective_log_step=self._base_log_step(),
                prune_rate_per_epoch=None,
                step_action="step_disabled",
            )

        if self._adaptive_log_step_is_after_max_epoch(epoch):
            self._store_control_zero_prob(epoch=epoch, valid_zero_prob=valid_zero_prob)
            return self._reset_log_step_boost(step_action="step_reset_after_max_epoch")

        if not acc_ok:
            step_action = (
                "step_reset_bad_acc"
                if self.log_step_boost_level > 0
                else "step_not_acc_ok"
            )
            self._store_control_zero_prob(epoch=epoch, valid_zero_prob=valid_zero_prob)
            return self._reset_log_step_boost(step_action=step_action)

        if (
            valid_zero_prob is None
            or self.previous_control_zero_prob is None
            or self.previous_control_epoch is None
        ):
            self.log_step_boost_level = 0
            self._store_control_zero_prob(epoch=epoch, valid_zero_prob=valid_zero_prob)
            return self._set_log_step_update_state(
                effective_log_step=self._base_log_step(),
                prune_rate_per_epoch=None,
                step_action="step_init_no_prev_zero_prob",
            )

        epochs_delta = int(epoch) - int(self.previous_control_epoch)
        if epochs_delta <= 0:
            self.log_step_boost_level = 0
            self._store_control_zero_prob(epoch=epoch, valid_zero_prob=valid_zero_prob)
            return self._set_log_step_update_state(
                effective_log_step=self._base_log_step(),
                prune_rate_per_epoch=None,
                step_action="step_init_no_prev_zero_prob",
            )

        prune_rate_per_epoch = (
            float(valid_zero_prob) - float(self.previous_control_zero_prob)
        ) / float(epochs_delta)
        if prune_rate_per_epoch < self.prune_rate_low_per_epoch:
            self.log_step_boost_level = min(
                self.log_step_boost_level + 1,
                self.log_step_max_boost_level,
            )
            step_action = "step_boost_slow_pruning"
        elif prune_rate_per_epoch <= self.prune_rate_high_per_epoch:
            self.log_step_boost_level = 0
            step_action = "step_reset_target_pruning"
        else:
            step_action = "step_keep_fast_pruning_no_new_logic"

        self._store_control_zero_prob(epoch=epoch, valid_zero_prob=valid_zero_prob)
        return self._set_log_step_update_state(
            effective_log_step=self._current_effective_log_step(),
            prune_rate_per_epoch=float(prune_rate_per_epoch),
            step_action=step_action,
        )

    def _apply_recovery_open_bias(self, model: nn.Module, value: float) -> None:
        self.recovery_open_bias = float(value)
        _set_model_gumbel_open_bias(
            model,
            self.recovery_open_bias,
            p_min=self.recovery_config.p_open_min,
            p_max=self.recovery_config.p_open_max,
        )

    def _resolve_window_delta(
        self,
        *,
        epoch: int,
        current_value: float | None,
        history: Mapping[int, float],
        window: int,
    ) -> float | None:
        if current_value is None:
            return None
        previous_value = history.get(int(epoch) - int(window))
        if previous_value is None:
            return None
        return float(current_value) - float(previous_value)

    def _build_recovery_metrics(
        self,
        *,
        model: nn.Module,
        action: str,
        reason: str,
        valid_acc: float | None,
        valid_zero_prob: float | None,
        acc_delta_over_window: float | None,
        zero_delta_over_window: float | None,
    ) -> dict[str, Any]:
        current_active_ratio = None if valid_zero_prob is None else float(1.0 - valid_zero_prob)
        active_delta = (
            None
            if current_active_ratio is None or self.recovery_start_active_ratio is None
            else float(current_active_ratio - self.recovery_start_active_ratio)
        )
        acc_drop_from_best = (
            None
            if valid_acc is None or self.best_val_acc is None
            else float(self.best_val_acc - valid_acc)
        )
        stats = _collect_model_gumbel_open_bias_stats(
            model,
            p_min=self.recovery_config.p_open_min,
            p_max=self.recovery_config.p_open_max,
        )
        return {
            "recovery_active": bool(self.recovery_active),
            "recovery_action": action,
            "recovery_reason": reason,
            "recovery_epochs_left": int(self.recovery_epochs_left),
            "recovery_open_bias": float(self.recovery_open_bias),
            "recovery_attempts": int(self.recovery_attempts),
            "recovery_cooldown_left": int(self.recovery_cooldown_left),
            "recovery_start_acc": self.recovery_start_acc,
            "recovery_start_active_ratio": self.recovery_start_active_ratio,
            "recovery_start_zero_prob": self.recovery_start_zero_prob,
            "recovery_current_active_ratio": current_active_ratio,
            "recovery_active_delta": active_delta,
            "recovery_acc_drop_from_best": acc_drop_from_best,
            "recovery_acc_delta_over_window": acc_delta_over_window,
            "recovery_zero_delta_over_window": zero_delta_over_window,
            **stats,
        }

    def _update_recovery(
        self,
        *,
        epoch: int,
        model: nn.Module,
        valid_acc: float | None,
        valid_zero_prob: float | None,
    ) -> RecoveryUpdate:
        cfg = self.recovery_config
        acc_delta_over_window = self._resolve_window_delta(
            epoch=epoch,
            current_value=valid_acc,
            history=self.observed_accuracy_by_epoch,
            window=cfg.recovery_slope_window,
        )
        zero_delta_over_window = self._resolve_window_delta(
            epoch=epoch,
            current_value=valid_zero_prob,
            history=self.observed_zero_prob_by_epoch,
            window=cfg.zero_prob_window,
        )
        current_active_ratio = None if valid_zero_prob is None else float(1.0 - valid_zero_prob)
        if current_active_ratio is not None:
            self.recovery_last_active_ratio = current_active_ratio

        action = "none"
        reason = "recovery_not_triggered"
        block_lambda_increase = False

        if not cfg.enabled:
            if self.recovery_open_bias > 0.0:
                self._apply_recovery_open_bias(model, 0.0)
            return RecoveryUpdate(
                metrics=self._build_recovery_metrics(
                    model=model,
                    action=action,
                    reason="recovery_disabled",
                    valid_acc=valid_acc,
                    valid_zero_prob=valid_zero_prob,
                    acc_delta_over_window=acc_delta_over_window,
                    zero_delta_over_window=zero_delta_over_window,
                ),
                block_lambda_increase=False,
            )

        if self.recovery_active:
            block_lambda_increase = cfg.freeze_lambda_increase
            self.recovery_total_epochs += 1
            self.recovery_epochs_left = max(0, self.recovery_epochs_left - 1)
            active_delta = (
                None
                if current_active_ratio is None or self.recovery_start_active_ratio is None
                else float(current_active_ratio - self.recovery_start_active_ratio)
            )

            stop_action = None
            if (
                valid_acc is not None
                and self.best_val_acc is not None
                and valid_acc >= self.best_val_acc - cfg.target_acc_margin
            ):
                stop_action = "stop_recovered_acc"
                reason = (
                    f"valid_acc={valid_acc:.4f} >= "
                    f"best_val_acc-target_acc_margin={self.best_val_acc - cfg.target_acc_margin:.4f}"
                )
            elif active_delta is not None and active_delta >= cfg.max_reopen_delta:
                stop_action = "stop_max_reopen_delta"
                reason = (
                    f"active_delta={active_delta:.4f} >= "
                    f"max_reopen_delta={cfg.max_reopen_delta:.4f}"
                )
            elif self.recovery_epochs_left <= 0:
                stop_action = "stop_timeout"
                reason = f"recovery_epochs exhausted after {cfg.recovery_epochs} epochs"

            if stop_action is not None:
                self.recovery_active = False
                self.recovery_condition_epochs = 0
                self.recovery_cooldown_left = int(cfg.cooldown_epochs)
                self._apply_recovery_open_bias(model, 0.0)
                return RecoveryUpdate(
                    metrics=self._build_recovery_metrics(
                        model=model,
                        action=stop_action,
                        reason=reason,
                        valid_acc=valid_acc,
                        valid_zero_prob=valid_zero_prob,
                        acc_delta_over_window=acc_delta_over_window,
                        zero_delta_over_window=zero_delta_over_window,
                    ),
                    block_lambda_increase=False,
                )

            next_open_bias = self.recovery_open_bias * cfg.open_bias_decay
            self._apply_recovery_open_bias(model, next_open_bias)
            return RecoveryUpdate(
                metrics=self._build_recovery_metrics(
                    model=model,
                    action="continue_recovery",
                    reason="recovery_active",
                    valid_acc=valid_acc,
                    valid_zero_prob=valid_zero_prob,
                    acc_delta_over_window=acc_delta_over_window,
                    zero_delta_over_window=zero_delta_over_window,
                ),
                block_lambda_increase=block_lambda_increase,
            )

        if self.recovery_cooldown_left > 0:
            self.recovery_condition_epochs = 0
            self.recovery_cooldown_left = max(0, self.recovery_cooldown_left - 1)
            return RecoveryUpdate(
                metrics=self._build_recovery_metrics(
                    model=model,
                    action="blocked_by_cooldown",
                    reason="recovery cooldown active",
                    valid_acc=valid_acc,
                    valid_zero_prob=valid_zero_prob,
                    acc_delta_over_window=acc_delta_over_window,
                    zero_delta_over_window=zero_delta_over_window,
                ),
                block_lambda_increase=False,
            )

        if epoch < cfg.min_epoch:
            self.recovery_condition_epochs = 0
            return RecoveryUpdate(
                metrics=self._build_recovery_metrics(
                    model=model,
                    action="blocked_by_min_epoch",
                    reason=f"epoch={epoch} < recovery.min_epoch={cfg.min_epoch}",
                    valid_acc=valid_acc,
                    valid_zero_prob=valid_zero_prob,
                    acc_delta_over_window=acc_delta_over_window,
                    zero_delta_over_window=zero_delta_over_window,
                ),
                block_lambda_increase=False,
            )

        if self.recovery_attempts >= cfg.max_recovery_attempts:
            self.recovery_condition_epochs = 0
            return RecoveryUpdate(
                metrics=self._build_recovery_metrics(
                    model=model,
                    action="none",
                    reason="max_recovery_attempts_reached",
                    valid_acc=valid_acc,
                    valid_zero_prob=valid_zero_prob,
                    acc_delta_over_window=acc_delta_over_window,
                    zero_delta_over_window=zero_delta_over_window,
                ),
                block_lambda_increase=False,
            )

        acc_drop = (
            None
            if valid_acc is None or self.best_val_acc is None
            else float(self.best_val_acc - valid_acc)
        )
        if acc_drop is None or not (cfg.drop_min <= acc_drop < cfg.drop_max):
            self.recovery_condition_epochs = 0
            return RecoveryUpdate(
                metrics=self._build_recovery_metrics(
                    model=model,
                    action="none",
                    reason="accuracy_drop_outside_recovery_band",
                    valid_acc=valid_acc,
                    valid_zero_prob=valid_zero_prob,
                    acc_delta_over_window=acc_delta_over_window,
                    zero_delta_over_window=zero_delta_over_window,
                ),
                block_lambda_increase=False,
            )

        if (
            cfg.require_slow_recovery
            and (
                acc_delta_over_window is None
                or acc_delta_over_window >= cfg.min_acc_delta_over_window
            )
        ):
            self.recovery_condition_epochs = 0
            return RecoveryUpdate(
                metrics=self._build_recovery_metrics(
                    model=model,
                    action="blocked_by_fast_recovery",
                    reason="accuracy is recovering fast enough or history window is unavailable",
                    valid_acc=valid_acc,
                    valid_zero_prob=valid_zero_prob,
                    acc_delta_over_window=acc_delta_over_window,
                    zero_delta_over_window=zero_delta_over_window,
                ),
                block_lambda_increase=False,
            )

        if (
            cfg.use_zero_prob_filter
            and (
                zero_delta_over_window is None
                or zero_delta_over_window < cfg.zero_prob_delta_min
            )
        ):
            self.recovery_condition_epochs = 0
            return RecoveryUpdate(
                metrics=self._build_recovery_metrics(
                    model=model,
                    action="blocked_by_zero_prob_filter",
                    reason="zero probability did not grow enough over recovery window",
                    valid_acc=valid_acc,
                    valid_zero_prob=valid_zero_prob,
                    acc_delta_over_window=acc_delta_over_window,
                    zero_delta_over_window=zero_delta_over_window,
                ),
                block_lambda_increase=False,
            )

        self.recovery_condition_epochs += 1
        if self.recovery_condition_epochs < cfg.patience:
            return RecoveryUpdate(
                metrics=self._build_recovery_metrics(
                    model=model,
                    action="none",
                    reason=(
                        f"recovery_condition_patience={self.recovery_condition_epochs}/"
                        f"{cfg.patience}"
                    ),
                    valid_acc=valid_acc,
                    valid_zero_prob=valid_zero_prob,
                    acc_delta_over_window=acc_delta_over_window,
                    zero_delta_over_window=zero_delta_over_window,
                ),
                block_lambda_increase=False,
            )

        self.recovery_active = True
        self.recovery_epochs_left = int(cfg.recovery_epochs)
        self.recovery_start_epoch = int(epoch)
        self.recovery_start_active_ratio = current_active_ratio
        self.recovery_start_zero_prob = valid_zero_prob
        self.recovery_start_acc = valid_acc
        self.recovery_attempts += 1
        self.recovery_condition_epochs = 0
        if self.recovery_first_start_active_ratio is None:
            self.recovery_first_start_active_ratio = current_active_ratio
        self._apply_recovery_open_bias(model, cfg.open_bias_start)

        return RecoveryUpdate(
            metrics=self._build_recovery_metrics(
                model=model,
                action="start_recovery",
                reason=(
                    f"acc_drop={acc_drop:.4f}, "
                    f"acc_delta_over_window={acc_delta_over_window}, "
                    f"zero_delta_over_window={zero_delta_over_window}"
                ),
                valid_acc=valid_acc,
                valid_zero_prob=valid_zero_prob,
                acc_delta_over_window=acc_delta_over_window,
                zero_delta_over_window=zero_delta_over_window,
            ),
            block_lambda_increase=cfg.freeze_lambda_increase,
        )

    def _build_epoch_metrics(
        self,
        *,
        action: str,
        reason: str,
        acc_ma: float | None,
        reference_epoch: int | None,
        reference_acc: float | None,
        min_allowed_acc: float | None,
        hard_min_allowed_acc: float | None,
    ) -> dict[str, Any]:
        return {
            "lambda_coef": self.lambda_coef,
            "log_lambda": self.log_lambda,
            "adaptive_lambda_step": self.step,
            "adaptive_lambda_effective_log_step": self.latest_effective_log_step,
            "adaptive_lambda_log_step_boost_level": self.log_step_boost_level,
            "adaptive_lambda_prune_rate_per_epoch": self.latest_prune_rate_per_epoch,
            "adaptive_lambda_step_action": self.latest_step_action,
            "adaptive_log_step_max_epoch": self.adaptive_log_step_max_epoch,
            "adaptive_lambda_action": action,
            "adaptive_lambda_reason": reason,
            "reference_epoch": reference_epoch,
            "reference_acc": reference_acc,
            "reference_source": (
                "baseline_history"
                if self.reference_accuracy_by_epoch
                else "best_val_acc"
            ),
            "best_val_acc": self.best_val_acc,
            "acc_ma": acc_ma,
            "min_allowed_acc": min_allowed_acc,
            "hard_min_allowed_acc": hard_min_allowed_acc,
        }

    def _print_action(self, epoch: int, action: str, reason: str) -> None:
        print(
            "Adaptive lambda update"
            f" | epoch={epoch}"
            f" | action={action}"
            f" | lambda_coef={self.lambda_coef:.12g}"
            f" | log_lambda={self.log_lambda:.12g}"
            f" | step={self.step:.12g}"
            f" | effective_step={self.latest_effective_log_step:.12g}"
            f" | boost_level={self.log_step_boost_level}"
            f" | reason={reason}"
        )
