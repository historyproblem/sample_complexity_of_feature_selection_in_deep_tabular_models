from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from typing import Any, Callable, Mapping

from net_complexity.data.dataloaders import Dataloaders
from net_complexity.metrics.base import BaseMetric, Multimetric
from net_complexity.models.feature_selection import get_gumbel_modules
from net_complexity.training.meta import Metrics
from net_complexity.training.randomness import set_random_seed
from net_complexity.training.run_history import RunHistory
from net_complexity.training.tracking import MLflowLogger
from net_complexity.tuning.restart_guard import CollapseDetected, CollapseGuard


EpochEndCallback = Callable[
    [int, Mapping[str, float], Mapping[str, float], nn.Module, torch.optim.Optimizer, RunHistory | None],
    None,
]
ProgressContext = Mapping[str, Any]
LAMBDA_CONFIG_PATH = "model.lambda_coef"


@dataclass
class SchedulerState:
    scheduler: Any
    interval: str = "epoch"
    monitor: str | None = None
    frequency: int = 1
    step_count: int = 0

    def __post_init__(self) -> None:
        self.interval = str(self.interval).lower()
        if self.interval not in {"epoch", "batch"}:
            raise ValueError("scheduler.interval must be either 'epoch' or 'batch'.")
        if self.frequency <= 0:
            raise ValueError("scheduler.frequency must be >= 1.")

    @property
    def needs_metric(self) -> bool:
        return self.monitor is not None or isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        )

    def step(self, metrics: Mapping[str, Any] | None = None) -> None:
        self.step_count += 1
        if self.step_count % self.frequency != 0:
            return

        if not self.needs_metric:
            self.scheduler.step()
            return

        metric_name = self.monitor or "valid_loss"
        metrics = metrics or {}
        if metric_name not in metrics:
            available_metrics = ", ".join(sorted(metrics.keys()))
            raise KeyError(
                f"Scheduler monitor '{metric_name}' is missing in metrics. "
                f"Available metrics: {available_metrics}"
            )

        metric_value = _to_float(metrics[metric_name])
        if metric_value is None:
            raise TypeError(f"Scheduler monitor '{metric_name}' must be numeric.")
        self.scheduler.step(metric_value)


def _build_scheduler(config: DictConfig, optimizer: torch.optim.Optimizer) -> SchedulerState | None:
    scheduler_cfg = getattr(config, "scheduler", None)
    if scheduler_cfg is None:
        return None
    if not bool(getattr(scheduler_cfg, "enabled", True)):
        return None

    scheduler_kwargs = {
        key: value
        for key, value in scheduler_cfg.items()
        if key not in {"enabled", "interval", "monitor", "frequency"}
    }
    if "_target_" not in scheduler_kwargs:
        raise ValueError("scheduler._target_ must be set when scheduler config is enabled.")

    scheduler = instantiate(scheduler_kwargs, optimizer=optimizer)
    return SchedulerState(
        scheduler=scheduler,
        interval=str(getattr(scheduler_cfg, "interval", "epoch")),
        monitor=getattr(scheduler_cfg, "monitor", None),
        frequency=int(getattr(scheduler_cfg, "frequency", 1)),
    )


def _to_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value.detach().cpu().mean().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_lambda_value(config: DictConfig) -> float | None:
    value = OmegaConf.select(config, LAMBDA_CONFIG_PATH)
    return None if value is None else float(value)


def _resolve_expected_run_name(config: DictConfig) -> str | None:
    resolved_config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved_config, dict):
        return None
    return (
        resolved_config.get("mlflow", {}).get("run_name")
        or resolved_config.get("run_history", {}).get("run_name")
    )


def _build_runtime_debug_snapshot(
    config: DictConfig,
    model: nn.Module,
    run_history: RunHistory,
    progress_context: ProgressContext | None = None,
) -> dict[str, Any]:
    progress_context = progress_context or {}
    display_trial_idx = progress_context.get("display_trial_idx", progress_context.get("trial_number"))
    trial_total = progress_context.get("trial_total")
    repeat_number = progress_context.get("repeat_number")
    repeat_total = progress_context.get("repeat_total")
    attempt_number = progress_context.get("attempt_number")
    attempt_total = progress_context.get("attempt_total")
    grid_params = progress_context.get("grid_params") or {}
    trial_params = progress_context.get("optuna_trial_params") or {}

    return {
        "display_trial": (
            f"{display_trial_idx}/{trial_total}"
            if display_trial_idx is not None and trial_total is not None
            else None
        ),
        "optuna_trial_number": progress_context.get("optuna_trial_number"),
        "repeat": (
            f"{repeat_number}/{repeat_total}"
            if repeat_number is not None and repeat_total is not None
            else None
        ),
        "attempt": (
            f"{attempt_number}/{attempt_total}"
            if attempt_number is not None and attempt_total is not None
            else None
        ),
        "grid_model_lambda_coef": (
            None
            if LAMBDA_CONFIG_PATH not in grid_params
            else float(grid_params[LAMBDA_CONFIG_PATH])
        ),
        "trial_params_model_lambda_coef": (
            None
            if LAMBDA_CONFIG_PATH not in trial_params
            else float(trial_params[LAMBDA_CONFIG_PATH])
        ),
        "cfg_model_lambda_coef": _resolve_lambda_value(config),
        "model_lambda_coef": _resolve_model_lambda_coef(model),
        "gumbel_bypass_enabled": _resolve_model_gumbel_bypass(model),
        "run_name": str(run_history.run_name),
        "run_dir": str(run_history.run_dir),
    }


def _assert_runtime_lambda_consistency(
    config: DictConfig,
    model: nn.Module,
    run_history: RunHistory,
    progress_context: ProgressContext | None = None,
) -> dict[str, Any]:
    cfg_lambda = _resolve_lambda_value(config)
    model_lambda = _resolve_model_lambda_coef(model)
    assert cfg_lambda is not None, f"cfg.{LAMBDA_CONFIG_PATH} is missing."
    assert model_lambda is not None, "Instantiated model is missing lambda_coef."
    assert abs(float(cfg_lambda) - float(model_lambda)) < 1e-12, (
        f"cfg.{LAMBDA_CONFIG_PATH}={cfg_lambda} does not match model.lambda_coef={model_lambda}."
    )

    progress_context = progress_context or {}
    grid_params = progress_context.get("grid_params") or {}
    if LAMBDA_CONFIG_PATH in grid_params:
        grid_lambda = float(grid_params[LAMBDA_CONFIG_PATH])
        assert abs(float(cfg_lambda) - grid_lambda) < 1e-12, (
            f"grid_params['{LAMBDA_CONFIG_PATH}']={grid_lambda} does not match "
            f"cfg.{LAMBDA_CONFIG_PATH}={cfg_lambda}."
        )

    trial_params = progress_context.get("optuna_trial_params") or {}
    if LAMBDA_CONFIG_PATH in trial_params:
        trial_lambda = float(trial_params[LAMBDA_CONFIG_PATH])
        assert abs(float(cfg_lambda) - trial_lambda) < 1e-12, (
            f"trial.params['{LAMBDA_CONFIG_PATH}']={trial_lambda} does not match "
            f"cfg.{LAMBDA_CONFIG_PATH}={cfg_lambda}."
        )

    expected_run_name = _resolve_expected_run_name(config)
    if expected_run_name is not None:
        assert str(run_history.run_name) == str(expected_run_name), (
            f"run_history.run_name={run_history.run_name} does not match resolved run_name={expected_run_name}."
        )

    return _build_runtime_debug_snapshot(
        config,
        model,
        run_history,
        progress_context=progress_context,
    )


def _log_runtime_debug_snapshot(snapshot: Mapping[str, Any]) -> None:
    parts = [
        f"display_trial={snapshot.get('display_trial')}",
        f"optuna_trial_number={snapshot.get('optuna_trial_number')}",
        f"repeat={snapshot.get('repeat')}",
        f"attempt={snapshot.get('attempt')}",
        f"grid_params['{LAMBDA_CONFIG_PATH}']={snapshot.get('grid_model_lambda_coef')}",
        f"trial.params['{LAMBDA_CONFIG_PATH}']={snapshot.get('trial_params_model_lambda_coef')}",
        f"cfg.{LAMBDA_CONFIG_PATH}={snapshot.get('cfg_model_lambda_coef')}",
        f"model.lambda_coef={snapshot.get('model_lambda_coef')}",
        f"gumbel_bypass_enabled={snapshot.get('gumbel_bypass_enabled')}",
        f"run_name={snapshot.get('run_name')}",
        f"run_dir={snapshot.get('run_dir')}",
    ]
    print(" | ".join(parts))


def collect_batch_metrics(output, targets, model: nn.Module | None = None) -> dict[str, float]:
    """Compatibility helper kept for legacy imports and optional batch logging."""
    batch_metrics: dict[str, float] = {}

    for name in ("ce_loss", "regularization_loss", "loss"):
        value = _to_float(getattr(output, name, None))
        if value is not None:
            batch_metrics[name] = value

    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
        batch_metrics["accuracy"] = float(accuracy)

    if model is None:
        return batch_metrics

    gumbel_modules = get_gumbel_modules(model)
    if not gumbel_modules:
        return batch_metrics

    real_means = []
    estim_means = []
    for name, module in gumbel_modules.items():
        value = module.get_selection_probs().detach().cpu()
        estim_prob = float(value.mean().item())
        real_prob = float((value > 0.5).float().mean().item())
        batch_metrics[f"{name}_avg_estim_prob"] = estim_prob
        batch_metrics[f"{name}_avg_real_prob"] = real_prob
        estim_means.append(estim_prob)
        real_means.append(real_prob)

    if real_means:
        batch_metrics["average_real_prob"] = float(sum(real_means) / len(real_means))
        batch_metrics["max_real_prob"] = float(max(real_means))
        batch_metrics["min_real_prob"] = float(min(real_means))
    if estim_means:
        batch_metrics["average_estim_prob"] = float(sum(estim_means) / len(estim_means))
        batch_metrics["max_estim_prob"] = float(max(estim_means))
        batch_metrics["min_estimm_prob"] = float(min(estim_means))

    return batch_metrics


@dataclass
class EarlyStoppingState:
    monitor: str
    mode: str
    patience: int
    min_epochs: int = 0
    min_delta: float = 0.0
    best_value: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0

    def step(self, epoch: int, metrics: Mapping[str, float]) -> bool:
        if self.monitor not in metrics:
            available_metrics = ", ".join(sorted(metrics.keys()))
            raise KeyError(
                f"Early stopping monitor '{self.monitor}' is missing in metrics. "
                f"Available metrics: {available_metrics}"
            )

        current_value = float(metrics[self.monitor])
        if self.best_value is None:
            improved = True
        elif self.mode == "max":
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta

        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return epoch >= self.min_epochs and self.bad_epochs >= self.patience


@dataclass
class LambdaWarmupState:
    start_epoch: int
    initial_lambda_coef: float
    target_lambda_coef: float
    ramp_epochs: int = 1
    bypass_during_warmup: bool = True
    last_applied_lambda_coef: float | None = None
    last_applied_bypass: bool | None = None

    def apply_initial_state(self, model: nn.Module) -> None:
        self._apply(
            model,
            lambda_coef=self.initial_lambda_coef,
            bypass_gumbel=self.bypass_during_warmup,
            reason="init",
        )

    def step(self, epoch: int, model: nn.Module) -> bool:
        lambda_coef, bypass_gumbel, phase = self.resolve_epoch_state(epoch)
        return self._apply(
            model,
            lambda_coef=lambda_coef,
            bypass_gumbel=bypass_gumbel,
            reason=f"epoch={epoch} phase={phase}",
        )

    def resolve_epoch_state(self, epoch: int) -> tuple[float, bool, str]:
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.initial_lambda_coef, self.bypass_during_warmup, "warmup"

        if self.ramp_epochs <= 1:
            return self.target_lambda_coef, False, "active"

        progress_steps = min(epoch - self.start_epoch + 1, self.ramp_epochs)
        progress = float(progress_steps) / float(self.ramp_epochs)
        lambda_coef = (
            self.initial_lambda_coef
            + (self.target_lambda_coef - self.initial_lambda_coef) * progress
        )
        phase = "ramp" if progress_steps < self.ramp_epochs else "active"
        return float(lambda_coef), False, phase

    def _apply(
        self,
        model: nn.Module,
        *,
        lambda_coef: float,
        bypass_gumbel: bool,
        reason: str,
    ) -> bool:
        lambda_coef = float(lambda_coef)
        bypass_gumbel = bool(bypass_gumbel)
        if (
            self.last_applied_lambda_coef is not None
            and self.last_applied_bypass is not None
            and abs(self.last_applied_lambda_coef - lambda_coef) < 1e-12
            and self.last_applied_bypass == bypass_gumbel
        ):
            return False

        _set_model_lambda_coef(
            model,
            lambda_coef,
            bypass_gumbel=bypass_gumbel,
        )
        self.last_applied_lambda_coef = lambda_coef
        self.last_applied_bypass = bypass_gumbel
        print(
            f"Lambda warmup update | {reason} | "
            f"lambda_coef={lambda_coef:.12g} | gumbel_bypass={bypass_gumbel}"
        )
        return True


def _set_model_lambda_coef(
    model: nn.Module,
    lambda_coef: float,
    *,
    bypass_gumbel: bool | None = None,
) -> None:
    if hasattr(model, "set_lambda_coef"):
        model.set_lambda_coef(float(lambda_coef), bypass_gumbel=bypass_gumbel)
        return
    if not hasattr(model, "lambda_coef"):
        raise AttributeError("Configured lambda warmup requires a model with lambda_coef.")
    model.lambda_coef = float(lambda_coef)
    if bypass_gumbel is not None:
        for module in get_gumbel_modules(model).values():
            module.set_bypass(bool(bypass_gumbel))


def _resolve_model_lambda_coef(model: nn.Module) -> float | None:
    return _to_float(getattr(model, "lambda_coef", None))


def _resolve_model_gumbel_bypass(model: nn.Module) -> float | None:
    gumbel_modules = get_gumbel_modules(model)
    if not gumbel_modules:
        return None
    return float(any(module.bypass for module in gumbel_modules.values()))


def _progress_prefix(progress_context: ProgressContext | None = None) -> str | None:
    parts: list[str] = []
    if progress_context is not None:
        trial_number = progress_context.get("display_trial_idx", progress_context.get("trial_number"))
        trial_total = progress_context.get("trial_total")
        if trial_number is not None and trial_total is not None:
            parts.append(f"display_trial={trial_number}/{trial_total}")
        optuna_trial_number = progress_context.get("optuna_trial_number")
        if optuna_trial_number is not None:
            parts.append(f"optuna_trial_number={optuna_trial_number}")
        repeat_number = progress_context.get("repeat_number")
        repeat_total = progress_context.get("repeat_total")
        if repeat_number is not None and repeat_total is not None:
            parts.append(f"repeat={repeat_number}/{repeat_total}")
        attempt_number = progress_context.get("attempt_number")
        attempt_total = progress_context.get("attempt_total")
        if attempt_number is not None and attempt_total is not None:
            parts.append(f"attempt={attempt_number}/{attempt_total}")
    return " | ".join(parts) if parts else None


def _format_metric(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.detach().item()
        else:
            value = value.detach().mean().item()
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "n/a"


def _build_epoch_log_line(
    epoch: int,
    total_epochs: int,
    train_metrics: Mapping[str, Any],
    valid_metrics: Mapping[str, Any],
    train_time: float,
    valid_time: float,
    epoch_time: float,
    progress_context: ProgressContext | None = None,
) -> str:
    parts: list[str] = []
    progress_prefix = _progress_prefix(progress_context)
    if progress_prefix is not None:
        parts.append(progress_prefix)

    parts.extend(
        [
            f"Epoch {epoch}/{total_epochs}",
            f"train_loss={_format_metric(train_metrics.get('train_loss', train_metrics.get('train_ce_loss')))}",
            f"val_loss={_format_metric(valid_metrics.get('valid_loss', valid_metrics.get('valid_ce_loss')))}",
            f"val_acc={_format_metric(valid_metrics.get('valid_accuracy'))}",
            f"train_time={train_time:.2f}s",
            f"val_time={valid_time:.2f}s",
            f"epoch_time={epoch_time:.2f}s",
        ]
    )

    train_zero_prob = train_metrics.get("train_average_zero_prob")
    if train_zero_prob is not None:
        parts.append(f"train_zero={_format_metric(train_zero_prob)}")

    valid_zero_prob = valid_metrics.get("valid_average_zero_prob")
    if valid_zero_prob is not None:
        parts.append(f"val_zero={_format_metric(valid_zero_prob)}")

    return " | ".join(parts)


def _build_early_stopping(
    training_arguments: DictConfig,
    run_history: RunHistory | None,
) -> EarlyStoppingState | None:
    cfg = getattr(training_arguments, "early_stopping", None)
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return None

    run_history_cfg = getattr(run_history.config, "run_history", None) if run_history is not None else None
    monitor = getattr(cfg, "monitor", None)
    if monitor is None and run_history_cfg is not None:
        monitor = getattr(run_history_cfg, "monitor", None)
    if monitor is None:
        raise ValueError(
            "training_arguments.early_stopping.monitor must be set, "
            "or run_history.monitor must be configured."
        )

    mode = str(
        getattr(
            cfg,
            "mode",
            getattr(run_history_cfg, "mode", "min") if run_history_cfg is not None else "min",
        )
    ).lower()
    if mode not in {"min", "max"}:
        raise ValueError("early_stopping.mode must be either 'min' or 'max'.")

    patience = int(getattr(cfg, "patience", 0))
    if patience <= 0:
        raise ValueError("training_arguments.early_stopping.patience must be > 0.")

    return EarlyStoppingState(
        monitor=str(monitor),
        mode=mode,
        patience=patience,
        min_epochs=int(getattr(cfg, "min_epochs", 0)),
        min_delta=float(getattr(cfg, "min_delta", 0.0)),
    )


def _build_lambda_warmup(training_arguments: DictConfig) -> LambdaWarmupState | None:
    cfg = getattr(training_arguments, "lambda_warmup", None)
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return None

    start_epoch = int(getattr(cfg, "start_epoch"))
    if start_epoch <= 0:
        raise ValueError("training_arguments.lambda_warmup.start_epoch must be >= 1.")

    return LambdaWarmupState(
        start_epoch=start_epoch,
        initial_lambda_coef=float(getattr(cfg, "initial_lambda_coef", 0.0)),
        target_lambda_coef=float(getattr(cfg, "target_lambda_coef")),
        ramp_epochs=max(1, int(getattr(cfg, "ramp_epochs", 1))),
        bypass_during_warmup=bool(getattr(cfg, "bypass_during_warmup", True)),
    )


def _build_collapse_guard(training_arguments: DictConfig) -> CollapseGuard | None:
    cfg = getattr(training_arguments, "collapse_guard", None)
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return None

    return CollapseGuard(
        min_epoch=int(getattr(cfg, "min_epoch", 35)),
        patience=int(getattr(cfg, "patience", 20)),
        min_epochs_since_best=int(
            getattr(cfg, "min_epochs_since_best", getattr(cfg, "patience", 20))
        ),
        acc_threshold_abs=float(getattr(cfg, "acc_threshold_abs", 0.15)),
        acc_threshold_rel=float(getattr(cfg, "acc_threshold_rel", 0.30)),
        loss_threshold=float(getattr(cfg, "loss_threshold", 2.25)),
        zero_threshold=float(getattr(cfg, "zero_threshold", 0.86)),
    )


@torch.inference_mode()
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             metric: Multimetric,
             device: str,
             total_epochs: int,
             stage: str = "valid",
             epoch: int = 0,
             run_history: RunHistory | None = None,
             progress_context: ProgressContext | None = None):
    model.eval()

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        output = model(X, y)
        metric.update(X, output, y, model)


def train_epoch(model,
                optimizer: torch.optim.Optimizer,
                dataloaders: Dataloaders,
                metrics: Metrics,
                device: str,
                total_epochs: int,
                epoch: int = 0,
                run_history: RunHistory | None = None,
                scheduler_state: SchedulerState | None = None,
                progress_context: ProgressContext | None = None):
    """Train for one epoch."""
    model.train()

    for X, y in dataloaders.train_dataloader:
        X, y = X.to(device), y.to(device)
        output = model(X, y)

        metrics.train_metrics.update(X, output, y, model)

        output.loss.backward()
        optimizer.step()
        if scheduler_state is not None and scheduler_state.interval == "batch":
            batch_metrics = collect_batch_metrics(output, y, model) if scheduler_state.needs_metric else {}
            scheduler_state.step(batch_metrics)
        optimizer.zero_grad()


def train(model: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler_state: SchedulerState | None,
          dataloaders: Dataloaders,
          training_arguments: DictConfig,
          metrics,
          device: str,
          mlflow_logger=None,
          run_history: RunHistory | None = None,
          epoch_end_callback: EpochEndCallback | None = None,
          progress_context: ProgressContext | None = None) -> dict[str, Any]:

    model.to(device)
    last_train_metrics: dict[str, float] = {}
    last_valid_metrics: dict[str, float] = {}
    total_epochs = int(training_arguments.num_epochs)
    early_stopping = _build_early_stopping(training_arguments, run_history)
    lambda_warmup = _build_lambda_warmup(training_arguments)
    collapse_guard = _build_collapse_guard(training_arguments)
    completed_epochs = 0
    stop_info: dict[str, Any] | None = None

    for epoch in range(total_epochs):
        epoch_num = epoch + 1
        if lambda_warmup is not None:
            lambda_warmup.step(epoch_num, model)
        epoch_started_at = perf_counter()

        train_started_at = perf_counter()
        train_epoch(
            model,
            optimizer,
            dataloaders,
            metrics,
            device,
            total_epochs=total_epochs,
            epoch=epoch_num,
            run_history=run_history,
            scheduler_state=scheduler_state,
            progress_context=progress_context,
        )
        train_time = perf_counter() - train_started_at

        valid_started_at = perf_counter()
        evaluate(
            model,
            dataloaders.valid_dataloader,
            metrics.valid_metrics,
            device,
            total_epochs=total_epochs,
            stage="valid",
            epoch=epoch_num,
            run_history=run_history,
            progress_context=progress_context,
        )
        valid_time = perf_counter() - valid_started_at

        train_metrics = dict(metrics.train_metrics.compute())
        valid_metrics = dict(metrics.valid_metrics.compute())
        train_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        last_train_metrics = train_metrics
        last_valid_metrics = valid_metrics
        if mlflow_logger is not None:
            mlflow_logger.log_metrics(train_metrics, step=epoch_num)
            mlflow_logger.log_metrics(valid_metrics, step=epoch_num)

        epoch_metrics = {
            **train_metrics,
            **valid_metrics,
        }
        if run_history is not None:
            run_history.save_checkpoint(
                "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch_num,
                metrics=epoch_metrics,
            )
            if run_history.should_update_best(epoch_num, valid_metrics):
                run_history.save_checkpoint(
                    "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch_num,
                    metrics=epoch_metrics,
                )
            run_history.log_channel_history(epoch_num, model)
            run_history.log_gate_history(epoch_num, "valid", model)
            run_history.log_epoch(
                epoch_num,
                train_metrics,
                valid_metrics,
                extra_metrics={
                "train_time_sec": float(train_time),
                "valid_time_sec": float(valid_time),
                "epoch_time_sec": float(perf_counter() - epoch_started_at),
                "model_lambda_coef": _resolve_model_lambda_coef(model),
                "gumbel_bypass_enabled": _resolve_model_gumbel_bypass(model),
                },
            )

        if epoch_end_callback is not None:
            epoch_end_callback(
                epoch_num,
                train_metrics,
                valid_metrics,
                model,
                optimizer,
                run_history,
            )

        collapse_detected: CollapseDetected | None = None
        if collapse_guard is not None:
            try:
                collapse_guard(
                    epoch_num,
                    train_metrics,
                    valid_metrics,
                    model,
                    optimizer,
                    run_history,
                )
            except CollapseDetected as exc:
                collapse_detected = exc
                stop_info = exc.as_dict()
                if mlflow_logger is not None:
                    mlflow_logger.log_params(stop_info)

        if scheduler_state is not None and scheduler_state.interval == "epoch":
            scheduler_state.step({**train_metrics, **valid_metrics})

        should_stop = False
        stop_message: str | None = None
        if early_stopping is not None:
            should_stop = early_stopping.step(epoch_num, valid_metrics)
            if should_stop:
                stop_message = (
                    f"Early stopping at epoch {epoch_num}: "
                    f"best {early_stopping.monitor}={early_stopping.best_value:.6f} "
                    f"at epoch {early_stopping.best_epoch}"
                )

        if collapse_detected is not None:
            should_stop = True
            stop_message = (
                f"Collapse detected at epoch {epoch_num}: "
                f"valid_accuracy={collapse_detected.valid_accuracy:.6f} | "
                f"valid_loss={collapse_detected.valid_loss:.6f} | "
                f"valid_average_zero_prob={collapse_detected.valid_average_zero_prob:.6f} | "
                f"best_val_acc_so_far={collapse_detected.best_val_acc_so_far:.6f} | "
                f"epochs_since_best={collapse_detected.epochs_since_best} | "
                f"consecutive_epochs={collapse_detected.consecutive_epochs}"
            )

        epoch_time = perf_counter() - epoch_started_at
        print(
            _build_epoch_log_line(
                epoch=epoch_num,
                total_epochs=total_epochs,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
                train_time=train_time,
                valid_time=valid_time,
                epoch_time=epoch_time,
                progress_context=progress_context,
            )
        )
        if stop_message is not None:
            print(stop_message)

        metrics.train_metrics.reset()
        metrics.valid_metrics.reset()
        completed_epochs = epoch_num

        if should_stop:
            break

    final_epoch = completed_epochs or total_epochs
    evaluate(
        model,
        dataloaders.test_dataloader,
        metrics.test_metrics,
        device,
        total_epochs=final_epoch,
        stage="test",
        epoch=final_epoch,
        run_history=run_history,
        progress_context=progress_context,
    )
    test_metrics = metrics.test_metrics.compute()
    if mlflow_logger is not None:
        mlflow_logger.log_metrics(
            test_metrics,
            step=final_epoch,
        )
        mlflow_logger.log_model(model, model_name="final_model")
    if run_history is not None:
        run_history.save_summary(
            final_train_metrics=last_train_metrics,
            final_valid_metrics=last_valid_metrics,
            test_metrics=test_metrics,
            stop_info=stop_info,
        )
    metrics.test_metrics.reset()
    result = {
        "last_train_metrics": last_train_metrics,
        "last_valid_metrics": last_valid_metrics,
        "test_metrics": dict(test_metrics),
    }
    if stop_info is not None:
        result.update(stop_info)
    return result


def resolve_device(config: DictConfig) -> str:
    requested_device = getattr(config, "device", None)
    if requested_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(requested_device)


def prepare_metrics(metrics: Metrics):
    metrics.train_metrics = BaseMetric() if len(
        metrics.train_metrics) == 0 else Multimetric(metrics.train_metrics, "train")
    metrics.valid_metrics = BaseMetric() if len(
        metrics.valid_metrics) == 0 else Multimetric(metrics.valid_metrics, "valid")
    metrics.test_metrics = BaseMetric() if len(
        metrics.test_metrics) == 0 else Multimetric(metrics.test_metrics, "test")
    return metrics


def log_training_metadata(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    run_history: RunHistory,
    mlflow_logger: MLflowLogger | None,
    config: DictConfig | None = None,
    runtime_snapshot: Mapping[str, Any] | None = None,
) -> None:
    if mlflow_logger is None:
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    params = {
        "run_history.run_id": run_history.run_id,
        "run_history.run_dir": str(run_history.run_dir),
        "model.total_parameters": total_params,
        "model.trainable_parameters": trainable_params,
        "optimizer.type": optimizer.__class__.__name__,
        "optimizer.lr": optimizer.param_groups[0].get("lr"),
        "optimizer.weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
        "seed": getattr(config, "seed", None) if config is not None else None,
    }
    if runtime_snapshot is not None:
        params.update({
            "runtime.display_trial": runtime_snapshot.get("display_trial"),
            "runtime.optuna_trial_number": runtime_snapshot.get("optuna_trial_number"),
            "runtime.repeat": runtime_snapshot.get("repeat"),
            "runtime.attempt": runtime_snapshot.get("attempt"),
            "runtime.grid_model_lambda_coef": runtime_snapshot.get("grid_model_lambda_coef"),
            "runtime.trial_params_model_lambda_coef": runtime_snapshot.get("trial_params_model_lambda_coef"),
            "runtime.cfg_model_lambda_coef": runtime_snapshot.get("cfg_model_lambda_coef"),
            "runtime.model_lambda_coef": runtime_snapshot.get("model_lambda_coef"),
            "runtime.gumbel_bypass_enabled": runtime_snapshot.get("gumbel_bypass_enabled"),
            "runtime.run_name": runtime_snapshot.get("run_name"),
        })
    mlflow_logger.log_params({
        key: value for key, value in params.items() if value is not None
    })


def log_run_artifacts(config: DictConfig, run_history: RunHistory, mlflow_logger: MLflowLogger | None) -> None:
    if mlflow_logger is None:
        return
    if not getattr(config, "mlflow", None) or not getattr(config.mlflow, "log_artifacts", False):
        return

    for artifact_path, artifact_dir in (
        (run_history.config_path, "run_history"),
        (run_history.history_path, "run_history"),
        (run_history.channel_history_path, "run_history"),
        (run_history.gate_history_path, "run_history"),
        (run_history.batch_history_path, "run_history"),
        (run_history.summary_path, "run_history"),
        (run_history.checkpoints_dir / "last.pt", "run_history/checkpoints"),
        (run_history.checkpoints_dir / "best.pt", "run_history/checkpoints"),
    ):
        if artifact_path.exists():
            mlflow_logger.log_artifact(str(artifact_path), artifact_dir)


def run_training(
    config: DictConfig,
    epoch_end_callback: EpochEndCallback | None = None,
    progress_context: ProgressContext | None = None,
) -> dict[str, Any]:
    resolved_seed = set_random_seed(getattr(config, "seed", None))
    device = resolve_device(config)
    model = instantiate(config.model).to(device)
    lambda_warmup = _build_lambda_warmup(config.training_arguments)
    if lambda_warmup is not None:
        lambda_warmup.apply_initial_state(model)
    dataloaders = instantiate(config.dataloaders)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    scheduler_state = _build_scheduler(config, optimizer)
    metrics = prepare_metrics(instantiate(config.metrics))
    mlflow_cfg = getattr(config, "mlflow", None)
    mlflow_enabled = bool(getattr(mlflow_cfg, "enabled", True)) if mlflow_cfg is not None else False
    mlflow_logger = MLflowLogger(config) if mlflow_enabled else None
    run_history = RunHistory(config)
    runtime_snapshot = _assert_runtime_lambda_consistency(
        config,
        model,
        run_history,
        progress_context=progress_context,
    )
    run_history.set_runtime_metadata(runtime_snapshot)
    _log_runtime_debug_snapshot(runtime_snapshot)

    result: dict[str, Any]
    if mlflow_logger is not None:
        mlflow_logger.setup()
        log_training_metadata(
            model,
            optimizer,
            run_history,
            mlflow_logger,
            config=config,
            runtime_snapshot=runtime_snapshot,
        )

    try:
        result = train(
            model,
            optimizer,
            scheduler_state,
            dataloaders,
            config.training_arguments,
            metrics,
            device,
            mlflow_logger=mlflow_logger,
            run_history=run_history,
            epoch_end_callback=epoch_end_callback,
            progress_context=progress_context,
        )
    finally:
        log_run_artifacts(config, run_history, mlflow_logger)
        if mlflow_logger is not None:
            mlflow_logger.close()

    result.update({
        "run_id": run_history.run_id,
        "run_dir": str(run_history.run_dir),
        "best_metric_name": run_history.best_metric_name,
        "best_metric_value": run_history.best_metric_value,
        "best_epoch": run_history.best_epoch,
        "seed": resolved_seed,
    })
    return result
