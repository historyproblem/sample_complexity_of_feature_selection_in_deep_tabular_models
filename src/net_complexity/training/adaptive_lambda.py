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
    resume_epoch: int | None
    collapse_detected: bool
    checkpoint_saved: bool
    metrics: dict[str, Any]


class AdaptiveLambdaController:
    def __init__(
        self,
        *,
        initial_lambda_coef: float,
        reference_accuracy_by_epoch: Mapping[int, float] | None = None,
        warmup_epochs: int = 10,
        lambda_increase_warmup_epochs: int = 0,
        lambda_increase_warmup_every_epochs: int | None = None,
        update_every_epochs: int = 3,
        acc_window: int = 3,
        lambda_min: float = 1e-8,
        lambda_max: float = 80.0,
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
        soft_step_shrink: float = 0.5,
        hard_step_shrink: float = 0.25,
        collapse_acc_threshold: float = 0.15,
        collapse_loss_threshold: float = 2.15,
        collapse_zero_prob_threshold: float = 0.90,
        collapse_acc_drop_threshold: float = 0.40,
        rollback_check_every_epochs: int = 5,
        rollback_acc_drop_threshold: float = 0.20,
        rollback_epoch_lookback: int = 20,
        lambda_increase_cooldown_epochs: int = 10,
        rollback_on_degradation: bool = True,
        rollback_on_collapse: bool = True,
        max_rollbacks: int = 6,
        freeze_on_rollback_limit: bool = True,
        recovery_config: Mapping[str, Any] | None = None,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError("adaptive_lambda.warmup_epochs must be >= 0.")
        if lambda_increase_warmup_epochs < 0:
            raise ValueError("adaptive_lambda.lambda_increase_warmup_epochs must be >= 0.")
        if (
            lambda_increase_warmup_every_epochs is not None
            and lambda_increase_warmup_every_epochs <= 0
        ):
            raise ValueError(
                "adaptive_lambda.lambda_increase_warmup_every_epochs must be >= 1."
            )
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
        if soft_step_shrink <= 0.0 or hard_step_shrink <= 0.0:
            raise ValueError("adaptive_lambda step shrink factors must be > 0.")
        if rollback_check_every_epochs <= 0:
            raise ValueError("adaptive_lambda.rollback_check_every_epochs must be >= 1.")
        if rollback_acc_drop_threshold < 0.0:
            raise ValueError("adaptive_lambda.rollback_acc_drop_threshold must be >= 0.")
        if rollback_epoch_lookback <= 0:
            raise ValueError("adaptive_lambda.rollback_epoch_lookback must be >= 1.")
        if rollback_epoch_lookback % rollback_check_every_epochs != 0:
            raise ValueError(
                "adaptive_lambda.rollback_epoch_lookback must be divisible by "
                "adaptive_lambda.rollback_check_every_epochs."
            )
        if lambda_increase_cooldown_epochs < 0:
            raise ValueError("adaptive_lambda.lambda_increase_cooldown_epochs must be >= 0.")
        if max_rollbacks < 0:
            raise ValueError("adaptive_lambda.max_rollbacks must be >= 0.")

        self.warmup_epochs = int(warmup_epochs)
        self.update_every_epochs = int(update_every_epochs)
        self.lambda_increase_warmup_epochs = int(lambda_increase_warmup_epochs)
        self.lambda_increase_warmup_every_epochs = (
            self.update_every_epochs
            if lambda_increase_warmup_every_epochs is None
            else int(lambda_increase_warmup_every_epochs)
        )
        self.acc_window = int(acc_window)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
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
        self.soft_step_shrink = float(soft_step_shrink)
        self.hard_step_shrink = float(hard_step_shrink)
        self.collapse_acc_threshold = float(collapse_acc_threshold)
        self.collapse_loss_threshold = float(collapse_loss_threshold)
        self.collapse_zero_prob_threshold = float(collapse_zero_prob_threshold)
        self.collapse_acc_drop_threshold = float(collapse_acc_drop_threshold)
        self.rollback_check_every_epochs = int(rollback_check_every_epochs)
        self.rollback_acc_drop_threshold = float(rollback_acc_drop_threshold)
        self.rollback_epoch_lookback = int(rollback_epoch_lookback)
        self.lambda_increase_cooldown_epochs = int(lambda_increase_cooldown_epochs)
        self.rollback_on_degradation = bool(rollback_on_degradation)
        self.rollback_on_collapse = bool(rollback_on_collapse)
        self.max_rollbacks = int(max_rollbacks)
        self.freeze_on_rollback_limit = bool(freeze_on_rollback_limit)
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
        self.previous_valid_acc: float | None = None
        self.observed_accuracy_by_epoch: dict[int, float] = {}
        self.observed_zero_prob_by_epoch: dict[int, float] = {}
        self.rollback_count = 0
        self.frozen = False
        self.lambda_increase_cooldown_remaining = 0
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
        self.latest_safe_checkpoint: AdaptiveLambdaCheckpoint | None = None
        self.best_sparse_safe_checkpoint: AdaptiveLambdaCheckpoint | None = None
        self.rollback_checkpoints_by_epoch: dict[int, AdaptiveLambdaCheckpoint] = {}
        self.latest_safe_epoch: int | None = None
        self.latest_safe_lambda: float | None = None
        self.best_sparse_safe_epoch: int | None = None
        self.best_sparse_safe_lambda: float | None = None
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
        rolled_back = False
        resume_epoch = None
        collapse_detected = False

        rollback_result = self._maybe_rollback_on_recent_acc_drop(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler_state=scheduler_state,
            scaler=scaler,
            valid_acc=valid_acc,
            apply_lambda=apply_lambda,
        )
        checkpoint_saved = False
        if rollback_result is not None:
            action, reason, lambda_changed, resume_epoch = rollback_result
            rolled_back = True
        else:
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

        recovery_update = self._update_recovery(
            epoch=epoch,
            model=model,
            valid_acc=valid_acc,
            valid_zero_prob=valid_zero_prob,
            hard_rollback_active=(
                rolled_back
                or self.frozen
                or self.lambda_increase_cooldown_remaining > 0
            ),
        )

        if rolled_back:
            pass
        elif epoch <= self.warmup_epochs:
            action = "warmup"
            reason = f"epoch={epoch} <= warmup_epochs={self.warmup_epochs}"
        elif epoch <= self.lambda_increase_warmup_epochs:
            self._store_control_zero_prob(epoch=epoch, valid_zero_prob=valid_zero_prob)
            if self._should_increase_warmup_update(epoch):
                old_lambda = self.lambda_coef
                effective_log_step = self._base_log_step()
                self.log_lambda = self._clamp_log_lambda(
                    self.log_lambda + effective_log_step
                )
                new_lambda = self.lambda_coef
                apply_lambda(model, new_lambda)
                lambda_changed = abs(new_lambda - old_lambda) > 1e-12
                action = "increase_lambda_warmup"
                reason = (
                    f"epoch={epoch} <= lambda_increase_warmup_epochs="
                    f"{self.lambda_increase_warmup_epochs}; "
                    f"lambda {old_lambda:.6g}->{new_lambda:.6g}; "
                    f"effective_log_step={effective_log_step:.6g}"
                )
                self._reset_log_step_boost(step_action="step_warmup_fixed_increase")
                if lambda_changed:
                    self._print_action(epoch, action, reason)
            else:
                action = "warmup_hold"
                reason = (
                    f"epoch={epoch} <= lambda_increase_warmup_epochs="
                    f"{self.lambda_increase_warmup_epochs}; waiting for "
                    f"lambda_increase_warmup_every_epochs="
                    f"{self.lambda_increase_warmup_every_epochs}"
                )
                self._reset_log_step_boost(step_action="step_warmup_hold")
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
                # Temporarily keep training state intact on collapse instead of restoring a safe checkpoint.
                action = "collapse_continue"
                if self.latest_safe_checkpoint is None:
                    reason = f"collapse_without_safe_checkpoint ({collapse_reason})"
                else:
                    reason = f"collapse_detected_but_rollback_disabled_temporarily ({collapse_reason})"
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
                    elif self.lambda_increase_cooldown_remaining > 0:
                        action = "hold"
                        reason = (
                            f"acc_ma={acc_ma:.4f} >= min_allowed_acc={min_allowed_acc:.4f}; "
                            "lambda increase blocked by rollback cooldown"
                            f" ({self.lambda_increase_cooldown_remaining} epochs remaining)"
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

        if not rolled_back and valid_acc is not None:
            self.previous_valid_acc = valid_acc
        if not rolled_back and self.lambda_increase_cooldown_remaining > 0:
            self.lambda_increase_cooldown_remaining = max(
                0,
                self.lambda_increase_cooldown_remaining - 1,
            )
        if not rolled_back:
            self._maybe_reset_adaptive_log_step_after_max_epoch(epoch=epoch)

        self.last_action = action
        self.last_reason = reason
        if not rolled_back:
            self._maybe_store_rollback_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                valid_metrics=valid_metrics,
                scheduler_state=scheduler_state,
                scaler=scaler,
                current_lambda=self.lambda_coef,
            )
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
            rolled_back=rolled_back,
            resume_epoch=resume_epoch,
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
            "adaptive_lambda_rollbacks": self.rollback_count,
            "lambda_increase_warmup_epochs": self.lambda_increase_warmup_epochs,
            "lambda_increase_warmup_every_epochs": self.lambda_increase_warmup_every_epochs,
            "rollback_check_every_epochs": self.rollback_check_every_epochs,
            "rollback_acc_drop_threshold": self.rollback_acc_drop_threshold,
            "rollback_epoch_lookback": self.rollback_epoch_lookback,
            "rollback_on_degradation": self.rollback_on_degradation,
            "lambda_increase_cooldown_epochs": self.lambda_increase_cooldown_epochs,
            "lambda_increase_cooldown_remaining": self.lambda_increase_cooldown_remaining,
            "reference_source": (
                "baseline_history"
                if self.reference_accuracy_by_epoch
                else "best_val_acc"
            ),
            "reference_num_epochs": len(self.reference_accuracy_by_epoch),
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
        return float(min(max(value, self.lambda_min), self.lambda_max))

    def _clamp_log_lambda(self, value: float) -> float:
        return math.log(self._clamp_lambda(math.exp(float(value))))

    def _should_update(self, epoch: int) -> bool:
        if epoch <= self.warmup_epochs:
            return False
        return (epoch - self.warmup_epochs) % self.update_every_epochs == 0

    def _should_increase_warmup_update(self, epoch: int) -> bool:
        return int(epoch) % self.lambda_increase_warmup_every_epochs == 0

    def _should_check_rollback(self, epoch: int) -> bool:
        return epoch % self.rollback_check_every_epochs == 0

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
        hard_rollback_active: bool,
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

        if hard_rollback_active:
            if self.recovery_active or self.recovery_open_bias > 0.0:
                self.recovery_active = False
                self.recovery_epochs_left = 0
                self._apply_recovery_open_bias(model, 0.0)
            self.recovery_condition_epochs = 0
            action = "blocked_by_rollback"
            reason = "hard rollback or rollback-freeze has priority"
            return RecoveryUpdate(
                metrics=self._build_recovery_metrics(
                    model=model,
                    action=action,
                    reason=reason,
                    valid_acc=valid_acc,
                    valid_zero_prob=valid_zero_prob,
                    acc_delta_over_window=acc_delta_over_window,
                    zero_delta_over_window=zero_delta_over_window,
                ),
                block_lambda_increase=False,
            )

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
            "step": self.step,
            "log_step_boost_level": self.log_step_boost_level,
            "previous_control_zero_prob": self.previous_control_zero_prob,
            "previous_control_epoch": self.previous_control_epoch,
            "observed_accuracy_by_epoch": dict(self.observed_accuracy_by_epoch),
            "observed_zero_prob_by_epoch": dict(self.observed_zero_prob_by_epoch),
            "lambda_increase_cooldown_remaining": self.lambda_increase_cooldown_remaining,
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

        self.log_lambda = float(checkpoint.log_lambda)
        apply_lambda(model, checkpoint.lambda_coef)

        snapshot = checkpoint.controller_state
        self.acc_history = deque(
            [float(value) for value in snapshot.get("acc_history", [])],
            maxlen=self.acc_window,
        )
        self.previous_valid_acc = _to_float(snapshot.get("previous_valid_acc"))
        checkpoint_best_val_acc = _to_float(snapshot.get("best_val_acc"))
        checkpoint_step = _to_float(snapshot.get("step"))
        if checkpoint_step is not None:
            self.step = max(float(checkpoint_step), self.log_step_min)
        restored_log_step_boost_level = snapshot.get("log_step_boost_level")
        if restored_log_step_boost_level is None:
            self.log_step_boost_level = 0
        else:
            self.log_step_boost_level = min(
                max(0, int(restored_log_step_boost_level)),
                self.log_step_max_boost_level,
            )
        self.previous_control_zero_prob = _to_float(
            snapshot.get("previous_control_zero_prob")
        )
        restored_previous_control_epoch = snapshot.get("previous_control_epoch")
        self.previous_control_epoch = (
            None
            if restored_previous_control_epoch is None
            else int(restored_previous_control_epoch)
        )
        restored_accuracy_by_epoch = snapshot.get("observed_accuracy_by_epoch", {})
        self.observed_accuracy_by_epoch = {
            int(epoch): float(value)
            for epoch, value in dict(restored_accuracy_by_epoch).items()
        }
        restored_zero_prob_by_epoch = snapshot.get("observed_zero_prob_by_epoch", {})
        self.observed_zero_prob_by_epoch = {
            int(epoch): float(value)
            for epoch, value in dict(restored_zero_prob_by_epoch).items()
        }
        restored_cooldown_remaining = snapshot.get("lambda_increase_cooldown_remaining")
        if restored_cooldown_remaining is None:
            self.lambda_increase_cooldown_remaining = 0
        else:
            self.lambda_increase_cooldown_remaining = max(0, int(restored_cooldown_remaining))
        if self.best_val_acc is None:
            self.best_val_acc = checkpoint_best_val_acc
        elif checkpoint_best_val_acc is not None:
            self.best_val_acc = max(self.best_val_acc, checkpoint_best_val_acc)
        self._reset_log_step_boost(step_action="step_reset_bad_acc")

    def _maybe_rollback_on_recent_acc_drop(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler_state: Any,
        scaler: Any,
        valid_acc: float | None,
        apply_lambda: LambdaApplier,
    ) -> tuple[str, str, bool, int] | None:
        if not self.rollback_on_degradation:
            return None
        if valid_acc is None or not self._should_check_rollback(epoch):
            return None
        if epoch <= self.rollback_epoch_lookback:
            return None

        comparison_epoch = epoch - self.rollback_check_every_epochs
        previous_acc = self.observed_accuracy_by_epoch.get(comparison_epoch)
        if previous_acc is None:
            return None

        acc_drop = float(previous_acc) - float(valid_acc)
        if acc_drop <= self.rollback_acc_drop_threshold:
            return None

        target_epoch = epoch - self.rollback_epoch_lookback
        checkpoint = self.rollback_checkpoints_by_epoch.get(target_epoch)
        if checkpoint is None:
            return None

        old_lambda = self.lambda_coef
        self._restore_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler_state=scheduler_state,
            scaler=scaler,
            apply_lambda=apply_lambda,
        )
        self.rollback_checkpoints_by_epoch = {
            saved_epoch: saved_checkpoint
            for saved_epoch, saved_checkpoint in self.rollback_checkpoints_by_epoch.items()
            if saved_epoch <= target_epoch
        }
        self.rollback_count += 1
        self.lambda_increase_cooldown_remaining = int(self.lambda_increase_cooldown_epochs)

        action = "periodic_rollback"
        lambda_changed = abs(old_lambda - checkpoint.lambda_coef) > 1e-12
        reason = (
            f"valid_acc_drop_over_{self.rollback_check_every_epochs}_epochs={acc_drop:.4f}"
            f" > rollback_acc_drop_threshold={self.rollback_acc_drop_threshold:.4f}; "
            f"rolled back to epoch={target_epoch}; "
            f"lambda increase cooldown={self.lambda_increase_cooldown_remaining} epochs"
        )

        if self.freeze_on_rollback_limit and self.rollback_count > self.max_rollbacks:
            self.frozen = True
            action = "rollback_limit_freeze"
            reason = (
                f"rollback_limit_reached ({self.rollback_count}>{self.max_rollbacks}); "
                f"rolled back to epoch={target_epoch}; "
                f"freezing at lambda={checkpoint.lambda_coef:.6g}"
            )

        self._print_action(epoch, action, reason)
        return action, reason, lambda_changed, target_epoch + 1

    def _maybe_store_rollback_checkpoint(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        valid_metrics: Mapping[str, Any],
        scheduler_state: Any,
        scaler: Any,
        current_lambda: float,
    ) -> None:
        if not self._should_check_rollback(epoch):
            return

        checkpoint = self._capture_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            valid_metrics=valid_metrics,
            scheduler_state=scheduler_state,
            scaler=scaler,
            current_lambda=current_lambda,
        )
        self.rollback_checkpoints_by_epoch[int(epoch)] = checkpoint
        min_epoch_to_keep = max(0, int(epoch) - self.rollback_epoch_lookback)
        self.rollback_checkpoints_by_epoch = {
            saved_epoch: saved_checkpoint
            for saved_epoch, saved_checkpoint in self.rollback_checkpoints_by_epoch.items()
            if saved_epoch >= min_epoch_to_keep
        }

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
        # Temporarily keep the adaptive step fixed while rollback tuning is disabled.
        # self.step = max(self.log_step_min, self.step * float(shrink_factor))

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
            "adaptive_lambda_rollbacks": self.rollback_count,
            "lambda_increase_warmup_epochs": self.lambda_increase_warmup_epochs,
            "lambda_increase_warmup_every_epochs": self.lambda_increase_warmup_every_epochs,
            "rollback_check_every_epochs": self.rollback_check_every_epochs,
            "rollback_acc_drop_threshold": self.rollback_acc_drop_threshold,
            "rollback_epoch_lookback": self.rollback_epoch_lookback,
            "rollback_on_degradation": self.rollback_on_degradation,
            "lambda_increase_cooldown_epochs": self.lambda_increase_cooldown_epochs,
            "lambda_increase_cooldown_remaining": self.lambda_increase_cooldown_remaining,
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
            f" | effective_step={self.latest_effective_log_step:.12g}"
            f" | boost_level={self.log_step_boost_level}"
            f" | reason={reason}"
        )
