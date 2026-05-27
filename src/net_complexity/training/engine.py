from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from typing import Any, Callable, Mapping

from net_complexity.data.dataloaders import Dataloaders
from net_complexity.metrics.base import BaseMetric, Multimetric
from net_complexity.models.feature_selection import get_gumbel_modules, get_stg_modules
from net_complexity.training.adaptive_lambda import ACCURACY_METRIC_NAMES, AdaptiveLambdaController
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
REPO_ROOT = Path(__file__).resolve().parents[3]


def _is_mlflow_enabled(config: DictConfig) -> bool:
    mlflow_cfg = getattr(config, "mlflow", None)
    if mlflow_cfg is None:
        return False
    return bool(getattr(mlflow_cfg, "enabled", True))


@dataclass(frozen=True)
class OptimizerBuildInfo:
    gate_weight_decay_scale: float | None = None
    gate_weight_decay: float | None = None
    gate_param_group_enabled: bool = False
    num_gates: int = 0
    num_gate_param_tensors: int = 0


@dataclass(frozen=True)
class GateParameterSpec:
    parameter: nn.Parameter
    num_gates: int


@dataclass(frozen=True)
class BaselineAccuracyReference:
    root_dir: Path
    history_path: Path
    metric_name: str
    accuracy_by_epoch: dict[int, float]


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


def _iter_gate_parameter_specs(model: nn.Module) -> list[GateParameterSpec]:
    gate_specs: list[GateParameterSpec] = []
    seen_parameter_ids: set[int] = set()

    for module in get_gumbel_modules(model).values():
        parameter = module.logits
        parameter_id = id(parameter)
        if parameter.requires_grad and parameter_id not in seen_parameter_ids:
            gate_specs.append(
                GateParameterSpec(
                    parameter=parameter,
                    num_gates=int(parameter.shape[0]),
                )
            )
            seen_parameter_ids.add(parameter_id)

    for module in get_stg_modules(model).values():
        parameter = module.mu
        parameter_id = id(parameter)
        if parameter.requires_grad and parameter_id not in seen_parameter_ids:
            gate_specs.append(
                GateParameterSpec(
                    parameter=parameter,
                    num_gates=int(parameter.numel()),
                )
            )
            seen_parameter_ids.add(parameter_id)

    return gate_specs


def _build_optimizer(
    config: DictConfig,
    model: nn.Module,
) -> tuple[torch.optim.Optimizer, OptimizerBuildInfo]:
    optimizer_cfg = getattr(config, "optimizer", None)
    if optimizer_cfg is None:
        raise ValueError("optimizer config must be defined.")

    optimizer_kwargs = {
        key: value
        for key, value in optimizer_cfg.items()
        if key != "gate_weight_decay_scale"
    }
    gate_weight_decay_scale = getattr(optimizer_cfg, "gate_weight_decay_scale", None)
    if gate_weight_decay_scale is None:
        return (
            instantiate(optimizer_kwargs, params=model.parameters(), _convert_="all"),
            OptimizerBuildInfo(),
        )

    gate_weight_decay_scale = float(gate_weight_decay_scale)
    if gate_weight_decay_scale < 0.0:
        raise ValueError("optimizer.gate_weight_decay_scale must be >= 0.")

    gate_specs = _iter_gate_parameter_specs(model)
    num_gates = sum(spec.num_gates for spec in gate_specs)
    if num_gates <= 0:
        return (
            instantiate(optimizer_kwargs, params=model.parameters(), _convert_="all"),
            OptimizerBuildInfo(gate_weight_decay_scale=gate_weight_decay_scale),
        )

    gate_parameter_ids = {id(spec.parameter) for spec in gate_specs}
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in gate_parameter_ids
    ]
    gate_parameters = [spec.parameter for spec in gate_specs]

    base_weight_decay = float(getattr(optimizer_cfg, "weight_decay", 0.0))
    gate_weight_decay = base_weight_decay * gate_weight_decay_scale / float(num_gates)
    param_groups: list[dict[str, Any]] = []
    if base_parameters:
        param_groups.append({
            "params": base_parameters,
            "weight_decay": base_weight_decay,
        })
    param_groups.append({
        "params": gate_parameters,
        "weight_decay": gate_weight_decay,
    })

    optimizer = instantiate(optimizer_kwargs, params=param_groups, _convert_="all")
    return (
        optimizer,
        OptimizerBuildInfo(
            gate_weight_decay_scale=gate_weight_decay_scale,
            gate_weight_decay=gate_weight_decay,
            gate_param_group_enabled=True,
            num_gates=num_gates,
            num_gate_param_tensors=len(gate_parameters),
        ),
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


def _adaptive_lambda_enabled(training_arguments: DictConfig) -> bool:
    cfg = getattr(training_arguments, "adaptive_lambda", None)
    return bool(getattr(cfg, "enabled", False)) if cfg is not None else False


def _config_has_path(config: DictConfig | Mapping[str, Any], dot_path: str) -> bool:
    node: Any = config
    for part in dot_path.split("."):
        if isinstance(node, DictConfig):
            if part not in node:
                return False
            node = node[part]
            continue
        if isinstance(node, Mapping):
            if part not in node:
                return False
            node = node[part]
            continue
        return False
    return True


def _resolve_baseline_history_root(training_arguments: DictConfig) -> Path | None:
    adaptive_cfg = getattr(training_arguments, "adaptive_lambda", None)
    if adaptive_cfg is None or not bool(getattr(adaptive_cfg, "enabled", False)):
        return None

    configured_root = getattr(adaptive_cfg, "baseline_history_dir", None)
    if configured_root in {None, ""}:
        return None

    baseline_root = Path(str(configured_root))
    return baseline_root if baseline_root.is_absolute() else REPO_ROOT / baseline_root


def _iter_baseline_history_candidates(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []

    candidates: list[Path] = []
    direct_history_path = root_dir / "history.csv"
    if direct_history_path.exists():
        candidates.append(direct_history_path)

    for child in root_dir.iterdir():
        if not child.is_dir():
            continue
        history_path = child / "history.csv"
        if history_path.exists():
            candidates.append(history_path)

    return sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime, str(path)),
    )


def _load_baseline_accuracy_history(history_path: Path) -> tuple[str, dict[int, float]]:
    with history_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Baseline history has no header: {history_path}")

        metric_name = next(
            (candidate for candidate in ACCURACY_METRIC_NAMES if candidate in reader.fieldnames),
            None,
        )
        if metric_name is None:
            available_columns = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Baseline history {history_path} does not contain an accuracy column. "
                f"Available columns: {available_columns}"
            )

        accuracy_by_epoch: dict[int, float] = {}
        for row in reader:
            epoch_raw = row.get("epoch")
            accuracy_raw = row.get(metric_name)
            if epoch_raw in {None, ""} or accuracy_raw in {None, ""}:
                continue

            try:
                epoch = int(float(epoch_raw))
                accuracy = float(accuracy_raw)
            except ValueError:
                continue

            accuracy_by_epoch[epoch] = accuracy

    if not accuracy_by_epoch:
        raise ValueError(f"Baseline history {history_path} does not contain any valid accuracy records.")

    return metric_name, accuracy_by_epoch


def _load_baseline_accuracy_reference(root_dir: Path) -> BaselineAccuracyReference | None:
    history_candidates = _iter_baseline_history_candidates(root_dir)
    if not history_candidates:
        return None

    history_path = history_candidates[-1]
    metric_name, accuracy_by_epoch = _load_baseline_accuracy_history(history_path)
    return BaselineAccuracyReference(
        root_dir=root_dir,
        history_path=history_path,
        metric_name=metric_name,
        accuracy_by_epoch=accuracy_by_epoch,
    )


def _build_baseline_training_config(config: DictConfig, baseline_root_dir: Path) -> DictConfig:
    baseline_config = deepcopy(config)

    OmegaConf.update(baseline_config, "training_arguments.adaptive_lambda.enabled", False, merge=False)
    if _config_has_path(baseline_config, "training_arguments.lambda_warmup.enabled"):
        OmegaConf.update(baseline_config, "training_arguments.lambda_warmup.enabled", False, merge=False)

    OmegaConf.update(baseline_config, "model.lambda_coef", 0.0, merge=False)
    if _config_has_path(baseline_config, "model.bypass_on_zero_lambda"):
        OmegaConf.update(baseline_config, "model.bypass_on_zero_lambda", True, merge=False)
    if _config_has_path(baseline_config, "model.gumbel_init_mode"):
        OmegaConf.update(baseline_config, "model.gumbel_init_mode", "fully_open", merge=False)

    OmegaConf.update(baseline_config, "run_history.use_hydra_output_dir", False, merge=False)
    OmegaConf.update(baseline_config, "run_history.root_dir", str(baseline_root_dir), merge=False)

    baseline_run_name = (
        f"{_resolve_expected_run_name(config) or 'run'}_baseline_no_pruning"
    )
    OmegaConf.update(baseline_config, "run_history.run_name", baseline_run_name, merge=False)
    if _config_has_path(baseline_config, "mlflow.run_name"):
        OmegaConf.update(baseline_config, "mlflow.run_name", baseline_run_name, merge=False)
    if _config_has_path(baseline_config, "mlflow.enabled"):
        OmegaConf.update(baseline_config, "mlflow.enabled", False, merge=False)

    return baseline_config


def _ensure_adaptive_baseline_reference(
    config: DictConfig,
    progress_context: ProgressContext | None = None,
) -> BaselineAccuracyReference | None:
    baseline_root_dir = _resolve_baseline_history_root(config.training_arguments)
    if baseline_root_dir is None:
        return None

    baseline_reference = _load_baseline_accuracy_reference(baseline_root_dir)
    if baseline_reference is not None:
        print(
            "Adaptive lambda baseline"
            f" | loaded_from={baseline_reference.history_path}"
            f" | metric={baseline_reference.metric_name}"
            f" | epochs={len(baseline_reference.accuracy_by_epoch)}"
        )
        return baseline_reference

    baseline_root_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Adaptive lambda baseline"
        f" | root_dir={baseline_root_dir}"
        " | no baseline history found, running baseline without pruning"
    )

    baseline_config = _build_baseline_training_config(config, baseline_root_dir)
    baseline_progress_context = dict(progress_context or {})
    baseline_progress_context["baseline_reference_run"] = True
    baseline_progress_context.pop("grid_params", None)
    baseline_progress_context.pop("optuna_trial_params", None)
    run_training(
        baseline_config,
        epoch_end_callback=None,
        progress_context=baseline_progress_context,
    )

    baseline_reference = _load_baseline_accuracy_reference(baseline_root_dir)
    if baseline_reference is None:
        raise FileNotFoundError(
            f"Baseline run finished but no history.csv was found under: {baseline_root_dir}"
        )

    print(
        "Adaptive lambda baseline"
        f" | generated_at={baseline_reference.history_path}"
        f" | metric={baseline_reference.metric_name}"
        f" | epochs={len(baseline_reference.accuracy_by_epoch)}"
    )
    return baseline_reference


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
        "grid_point_index": progress_context.get("grid_point_index"),
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
        "gumbel_train_gate_mode": _resolve_model_gumbel_gate_modes(model)[0],
        "gumbel_eval_gate_mode": _resolve_model_gumbel_gate_modes(model)[1],
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
        f"grid_point_index={snapshot.get('grid_point_index')}",
        f"repeat={snapshot.get('repeat')}",
        f"attempt={snapshot.get('attempt')}",
        f"grid_params['{LAMBDA_CONFIG_PATH}']={snapshot.get('grid_model_lambda_coef')}",
        f"trial.params['{LAMBDA_CONFIG_PATH}']={snapshot.get('trial_params_model_lambda_coef')}",
        f"cfg.{LAMBDA_CONFIG_PATH}={snapshot.get('cfg_model_lambda_coef')}",
        f"model.lambda_coef={snapshot.get('model_lambda_coef')}",
        f"gumbel_bypass_enabled={snapshot.get('gumbel_bypass_enabled')}",
        f"gumbel_train_gate_mode={snapshot.get('gumbel_train_gate_mode')}",
        f"gumbel_eval_gate_mode={snapshot.get('gumbel_eval_gate_mode')}",
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
    zero_means = []
    total_channels = 0
    total_real_active_channels = 0.0
    total_estim_active_channels = 0.0
    total_real_zero_channels = 0.0
    total_estim_zero_channels = 0.0
    for name, module in gumbel_modules.items():
        value = module.get_selection_probs().detach().cpu().reshape(-1)
        hard_active = (value > 0.5).float()
        num_channels = int(value.numel())
        estim_prob = float(value.mean().item())
        real_prob = float(hard_active.mean().item())
        zero_prob = 1.0 - estim_prob
        estim_active_channels = float(value.sum().item())
        estim_zero_channels = float(num_channels - estim_active_channels)
        real_active_channels = float(hard_active.sum().item())
        real_zero_channels = float(num_channels - real_active_channels)
        batch_metrics[f"{name}_avg_estim_prob"] = estim_prob
        batch_metrics[f"{name}_avg_real_prob"] = real_prob
        batch_metrics[f"{name}_avg_zero_prob"] = zero_prob
        batch_metrics[f"{name}_num_channels"] = num_channels
        batch_metrics[f"{name}_estim_active_channels"] = estim_active_channels
        batch_metrics[f"{name}_estim_zero_channels"] = estim_zero_channels
        batch_metrics[f"{name}_real_active_channels"] = real_active_channels
        batch_metrics[f"{name}_real_zero_channels"] = real_zero_channels
        estim_means.append(estim_prob)
        real_means.append(real_prob)
        zero_means.append(zero_prob)
        total_channels += num_channels
        total_real_active_channels += real_active_channels
        total_estim_active_channels += estim_active_channels
        total_real_zero_channels += real_zero_channels
        total_estim_zero_channels += estim_zero_channels

    if real_means:
        batch_metrics["average_layer_real_prob"] = float(sum(real_means) / len(real_means))
        batch_metrics["average_real_prob"] = float(total_real_active_channels / total_channels)
        batch_metrics["max_real_prob"] = float(max(real_means))
        batch_metrics["min_real_prob"] = float(min(real_means))
    if estim_means:
        batch_metrics["average_layer_estim_prob"] = float(sum(estim_means) / len(estim_means))
        batch_metrics["average_estim_prob"] = float(total_estim_active_channels / total_channels)
        batch_metrics["max_estim_prob"] = float(max(estim_means))
        batch_metrics["min_estimm_prob"] = float(min(estim_means))
    if zero_means:
        batch_metrics["average_layer_zero_prob"] = float(sum(zero_means) / len(zero_means))
        batch_metrics["average_zero_prob"] = float(total_estim_zero_channels / total_channels)
        batch_metrics["max_zero_prob"] = float(max(zero_means))
        batch_metrics["min_zero_prob"] = float(min(zero_means))
        batch_metrics["total_channels"] = total_channels
        batch_metrics["real_active_channels"] = total_real_active_channels
        batch_metrics["real_zero_channels"] = total_real_zero_channels
        batch_metrics["estim_active_channels"] = total_estim_active_channels
        batch_metrics["estim_zero_channels"] = total_estim_zero_channels

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


@dataclass
class GateModeScheduleState:
    start_epoch: int
    initial_train_gate_mode: str
    target_train_gate_mode: str
    initial_eval_gate_mode: str
    target_eval_gate_mode: str
    last_applied_train_gate_mode: str | None = None
    last_applied_eval_gate_mode: str | None = None

    def apply_initial_state(self, model: nn.Module) -> None:
        self._apply(
            model,
            train_gate_mode=self.initial_train_gate_mode,
            eval_gate_mode=self.initial_eval_gate_mode,
            reason="init",
        )

    def step(self, epoch: int, model: nn.Module) -> bool:
        train_gate_mode, eval_gate_mode, phase = self.resolve_epoch_state(epoch)
        return self._apply(
            model,
            train_gate_mode=train_gate_mode,
            eval_gate_mode=eval_gate_mode,
            reason=f"epoch={epoch} phase={phase}",
        )

    def resolve_epoch_state(self, epoch: int) -> tuple[str, str, str]:
        epoch = int(epoch)
        if epoch < self.start_epoch:
            return self.initial_train_gate_mode, self.initial_eval_gate_mode, "warmup"
        return self.target_train_gate_mode, self.target_eval_gate_mode, "active"

    def _apply(
        self,
        model: nn.Module,
        *,
        train_gate_mode: str,
        eval_gate_mode: str,
        reason: str,
    ) -> bool:
        if (
            self.last_applied_train_gate_mode == str(train_gate_mode)
            and self.last_applied_eval_gate_mode == str(eval_gate_mode)
        ):
            return False

        _set_model_gumbel_gate_modes(
            model,
            train_gate_mode=str(train_gate_mode),
            eval_gate_mode=str(eval_gate_mode),
        )
        self.last_applied_train_gate_mode = str(train_gate_mode)
        self.last_applied_eval_gate_mode = str(eval_gate_mode)
        print(
            f"Gate mode schedule update | {reason} | "
            f"train_gate_mode={train_gate_mode} | eval_gate_mode={eval_gate_mode}"
        )
        return True


@dataclass
class BatchNormRecalibrationState:
    num_batches: int = 200
    reset_running_stats: bool = True
    train_gate_mode: str | None = "deterministic_hard"
    eval_gate_mode: str | None = "deterministic_hard"

    def __post_init__(self) -> None:
        self.num_batches = int(self.num_batches)
        if self.num_batches <= 0:
            raise ValueError("training_arguments.batchnorm_recalibration.num_batches must be >= 1.")

    def apply(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        *,
        device: str,
        progress_context: ProgressContext | None = None,
    ) -> dict[str, Any]:
        batchnorm_modules = [
            module
            for module in model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        if not batchnorm_modules:
            info = {
                "enabled": True,
                "applied": False,
                "num_batches_requested": self.num_batches,
                "num_batches_processed": 0,
                "num_examples_processed": 0,
                "num_batchnorm_modules": 0,
                "reset_running_stats": bool(self.reset_running_stats),
                "train_gate_mode": self.train_gate_mode,
                "eval_gate_mode": self.eval_gate_mode,
                "reason": "no_batchnorm_modules",
            }
            parts: list[str] = []
            progress_prefix = _progress_prefix(progress_context)
            if progress_prefix is not None:
                parts.append(progress_prefix)
            parts.append(
                "BatchNorm recalibration"
                " | applied=False"
                f" | batches=0/{self.num_batches}"
                " | examples=0"
                " | bn_modules=0"
                " | reason=no_batchnorm_modules"
            )
            print(" | ".join(parts))
            return info

        gumbel_modules = get_gumbel_modules(model)
        original_training = bool(model.training)
        original_train_gate_mode, original_eval_gate_mode = _resolve_model_gumbel_gate_modes(model)
        batches_processed = 0
        examples_processed = 0

        try:
            if self.reset_running_stats:
                for module in batchnorm_modules:
                    module.reset_running_stats()

            if gumbel_modules and (self.train_gate_mode is not None or self.eval_gate_mode is not None):
                _set_model_gumbel_gate_modes(
                    model,
                    train_gate_mode=self.train_gate_mode,
                    eval_gate_mode=self.eval_gate_mode,
                )

            model.train()

            with torch.no_grad():
                for batch in dataloader:
                    X, y = _extract_batch_tensors(batch)
                    X, y = X.to(device), y.to(device)
                    model(X, y)
                    batches_processed += 1
                    examples_processed += int(X.shape[0])
                    if batches_processed >= self.num_batches:
                        break
        finally:
            if gumbel_modules and (original_train_gate_mode is not None or original_eval_gate_mode is not None):
                _set_model_gumbel_gate_modes(
                    model,
                    train_gate_mode=original_train_gate_mode,
                    eval_gate_mode=original_eval_gate_mode,
                )

            if original_training:
                model.train()
            else:
                model.eval()

        info = {
            "enabled": True,
            "applied": True,
            "num_batches_requested": self.num_batches,
            "num_batches_processed": batches_processed,
            "num_examples_processed": examples_processed,
            "num_batchnorm_modules": len(batchnorm_modules),
            "reset_running_stats": bool(self.reset_running_stats),
            "train_gate_mode": self.train_gate_mode,
            "eval_gate_mode": self.eval_gate_mode,
        }

        parts: list[str] = []
        progress_prefix = _progress_prefix(progress_context)
        if progress_prefix is not None:
            parts.append(progress_prefix)
        parts.append(
            "BatchNorm recalibration"
            f" | applied={info['applied']}"
            f" | batches={batches_processed}/{self.num_batches}"
            f" | examples={examples_processed}"
            f" | bn_modules={len(batchnorm_modules)}"
            f" | train_gate_mode={self.train_gate_mode}"
            f" | eval_gate_mode={self.eval_gate_mode}"
        )
        print(" | ".join(parts))
        return info


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


def _set_model_gumbel_gate_modes(
    model: nn.Module,
    *,
    train_gate_mode: str | None = None,
    eval_gate_mode: str | None = None,
) -> None:
    if hasattr(model, "set_gumbel_gate_modes"):
        model.set_gumbel_gate_modes(
            train_gate_mode=train_gate_mode,
            eval_gate_mode=eval_gate_mode,
        )
        return

    gumbel_modules = get_gumbel_modules(model)
    if not gumbel_modules:
        raise AttributeError("Configured gate mode schedule requires a model with Gumbel modules.")
    for module in gumbel_modules.values():
        module.set_gate_modes(
            train_gate_mode=train_gate_mode,
            eval_gate_mode=eval_gate_mode,
        )


def _resolve_model_lambda_coef(model: nn.Module) -> float | None:
    return _to_float(getattr(model, "lambda_coef", None))


def _resolve_model_gumbel_bypass(model: nn.Module) -> float | None:
    gumbel_modules = get_gumbel_modules(model)
    if not gumbel_modules:
        return None
    return float(any(module.bypass for module in gumbel_modules.values()))


def _resolve_model_gumbel_gate_modes(model: nn.Module) -> tuple[str | None, str | None]:
    gumbel_modules = get_gumbel_modules(model)
    if not gumbel_modules:
        return None, None
    first_module = next(iter(gumbel_modules.values()))
    return (
        str(getattr(first_module, "train_gate_mode", None)),
        str(getattr(first_module, "eval_gate_mode", None)),
    )


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


def _prefix_metric_keys(metrics: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    normalized_prefix = str(prefix).strip()
    if not normalized_prefix:
        return dict(metrics)
    return {f"{normalized_prefix}_{key}": value for key, value in metrics.items()}


def _format_channel_count(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "n/a"
    rounded = round(numeric)
    if abs(numeric - rounded) < 1e-9:
        return str(int(rounded))
    return f"{numeric:.2f}"


def _format_bool(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.detach().item()
        else:
            value = bool(value.detach().cpu().any().item())
    if isinstance(value, str):
        return "true" if value.strip().lower() in {"1", "true", "yes", "on"} else "false"
    return str(bool(value)).lower()


def _device_is_cuda(device: str | torch.device) -> bool:
    try:
        return torch.device(device).type == "cuda" and torch.cuda.is_available()
    except (RuntimeError, TypeError):
        return False


def _cuda_device(device: str | torch.device) -> torch.device | None:
    if not _device_is_cuda(device):
        return None
    return torch.device(device)


def _build_epoch_log_line(
    epoch: int,
    total_epochs: int,
    train_metrics: Mapping[str, Any],
    valid_metrics: Mapping[str, Any],
    train_time: float,
    valid_time: float,
    epoch_time: float,
    extra_metrics: Mapping[str, Any] | None = None,
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

    extra_metrics = extra_metrics or {}
    train_samples_per_sec = extra_metrics.get("train_samples_per_sec")
    if train_samples_per_sec is not None:
        parts.append(f"samples/sec={_format_metric(train_samples_per_sec)}")
    train_batch_size = extra_metrics.get("train_batch_size")
    if train_batch_size is not None:
        parts.append(f"batch={_format_channel_count(train_batch_size)}")
    train_max_memory_mb = extra_metrics.get("train_max_memory_allocated_mb")
    if train_max_memory_mb is not None:
        parts.append(f"max_memory={_format_metric(train_max_memory_mb)}MB")

    train_zero_prob = train_metrics.get("train_average_zero_prob")
    if train_zero_prob is not None:
        parts.append(f"train_zero={_format_metric(train_zero_prob)}")
    train_active_channels = train_metrics.get("train_estim_active_channels")
    train_total_channels = train_metrics.get("train_total_channels")
    if train_active_channels is not None and train_total_channels is not None:
        parts.append(
            f"train_open={_format_channel_count(train_active_channels)}/"
            f"{_format_channel_count(train_total_channels)}"
        )

    valid_zero_prob = valid_metrics.get("valid_average_zero_prob")
    if valid_zero_prob is not None:
        parts.append(f"val_zero={_format_metric(valid_zero_prob)}")
    valid_active_channels = valid_metrics.get("valid_estim_active_channels")
    valid_total_channels = valid_metrics.get("valid_total_channels")
    if valid_active_channels is not None and valid_total_channels is not None:
        parts.append(
            f"val_open={_format_channel_count(valid_active_channels)}/"
            f"{_format_channel_count(valid_total_channels)}"
        )

    lambda_coef = extra_metrics.get("lambda_coef")
    if lambda_coef is not None:
        parts.append(f"lambda={_format_metric(lambda_coef)}")

    adaptive_step = extra_metrics.get("adaptive_lambda_step")
    if adaptive_step is not None:
        parts.append(f"lambda_step={_format_metric(adaptive_step)}")

    adaptive_action = extra_metrics.get("adaptive_lambda_action")
    if adaptive_action:
        parts.append(f"lambda_action={adaptive_action}")

    recovery_active = extra_metrics.get("recovery_active")
    recovery_action = extra_metrics.get("recovery_action")
    recovery_active_flag = False
    if recovery_active is not None:
        recovery_active_flag = _format_bool(recovery_active) == "true"
        recovery_mode = "active" if recovery_active_flag else "idle"
        parts.append(f"recovery={recovery_mode}")
    if recovery_action:
        parts.append(f"recovery_action={recovery_action}")

    recovery_should_log_details = (
        recovery_active_flag
        or (recovery_action not in {None, "", "none"})
    )
    if recovery_should_log_details:
        recovery_epochs_left = extra_metrics.get("recovery_epochs_left")
        if recovery_epochs_left is not None:
            parts.append(f"recovery_left={_format_channel_count(recovery_epochs_left)}")
        recovery_open_bias = extra_metrics.get("recovery_open_bias")
        if recovery_open_bias is not None:
            parts.append(f"recovery_bias={_format_metric(recovery_open_bias)}")
        recovery_attempts = extra_metrics.get("recovery_attempts")
        if recovery_attempts is not None:
            parts.append(f"recovery_attempts={_format_channel_count(recovery_attempts)}")

    reference_acc = extra_metrics.get("reference_acc")
    if reference_acc is not None:
        parts.append(f"ref_acc={_format_metric(reference_acc)}")

    return " | ".join(parts)


def _build_post_recalibration_valid_log_line(
    valid_metrics: Mapping[str, Any],
    valid_time: float,
    progress_context: ProgressContext | None = None,
) -> str:
    parts: list[str] = []
    progress_prefix = _progress_prefix(progress_context)
    if progress_prefix is not None:
        parts.append(progress_prefix)

    parts.extend(
        [
            "Post-recalibration valid",
            f"val_loss={_format_metric(valid_metrics.get('valid_loss', valid_metrics.get('valid_ce_loss')))}",
            f"val_acc={_format_metric(valid_metrics.get('valid_accuracy'))}",
            f"val_time={valid_time:.2f}s",
        ]
    )

    valid_zero_prob = valid_metrics.get("valid_average_zero_prob")
    if valid_zero_prob is not None:
        parts.append(f"val_zero={_format_metric(valid_zero_prob)}")

    return " | ".join(parts)


def _extract_batch_tensors(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise TypeError(
            "Expected dataloader batch to be a tuple/list of at least (inputs, targets)."
        )
    X, y = batch[0], batch[1]
    if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("BatchNorm recalibration expects tensor inputs and tensor targets.")
    return X, y


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


def _build_adaptive_lambda(
    training_arguments: DictConfig,
    model: nn.Module,
    *,
    baseline_accuracy_by_epoch: Mapping[int, float] | None = None,
) -> AdaptiveLambdaController | None:
    cfg = getattr(training_arguments, "adaptive_lambda", None)
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return None

    lambda_warmup_cfg = getattr(training_arguments, "lambda_warmup", None)
    if lambda_warmup_cfg is not None and bool(getattr(lambda_warmup_cfg, "enabled", False)):
        raise ValueError("adaptive_lambda and lambda_warmup cannot be enabled at the same time.")

    initial_lambda_coef = _resolve_model_lambda_coef(model)
    if initial_lambda_coef is None:
        raise ValueError("adaptive_lambda requires a model with a numeric lambda_coef.")
    recovery_cfg = getattr(cfg, "recovery", None)
    recovery_config = (
        OmegaConf.to_container(recovery_cfg, resolve=True)
        if recovery_cfg is not None
        else None
    )
    adaptive_log_step_max_epoch = getattr(cfg, "adaptive_log_step_max_epoch", None)

    return AdaptiveLambdaController(
        initial_lambda_coef=initial_lambda_coef,
        reference_accuracy_by_epoch=baseline_accuracy_by_epoch,
        warmup_epochs=int(getattr(cfg, "warmup_epochs", 10)),
        update_every_epochs=int(getattr(cfg, "update_every_epochs", 3)),
        acc_window=int(getattr(cfg, "acc_window", 3)),
        lambda_min=float(getattr(cfg, "lambda_min", 1e-8)),
        lambda_max=float(getattr(cfg, "lambda_max", 80.0)),
        log_step_init=float(getattr(cfg, "log_step_init", 0.6931471805599453)),
        log_step_min=float(getattr(cfg, "log_step_min", 0.04879016416943205)),
        adaptive_log_step_enabled=bool(getattr(cfg, "adaptive_log_step_enabled", True)),
        prune_rate_low_per_epoch=float(getattr(cfg, "prune_rate_low_per_epoch", 0.02)),
        prune_rate_high_per_epoch=float(getattr(cfg, "prune_rate_high_per_epoch", 0.07)),
        log_step_boost_factor=float(getattr(cfg, "log_step_boost_factor", 2.0)),
        log_step_max_boost_level=int(getattr(cfg, "log_step_max_boost_level", 2)),
        adaptive_log_step_max_epoch=(
            None
            if adaptive_log_step_max_epoch is None
            else int(adaptive_log_step_max_epoch)
        ),
        soft_drop=float(getattr(cfg, "soft_drop", 0.02)),
        hard_drop=float(getattr(cfg, "hard_drop", 0.05)),
        soft_step_shrink=float(getattr(cfg, "soft_step_shrink", 0.5)),
        hard_step_shrink=float(getattr(cfg, "hard_step_shrink", 0.25)),
        collapse_acc_threshold=float(getattr(cfg, "collapse_acc_threshold", 0.15)),
        collapse_loss_threshold=float(getattr(cfg, "collapse_loss_threshold", 2.15)),
        collapse_zero_prob_threshold=float(getattr(cfg, "collapse_zero_prob_threshold", 0.90)),
        collapse_acc_drop_threshold=float(getattr(cfg, "collapse_acc_drop_threshold", 0.40)),
        rollback_check_every_epochs=int(getattr(cfg, "rollback_check_every_epochs", 1)),
        rollback_acc_drop_threshold=float(getattr(cfg, "rollback_acc_drop_threshold", 0.20)),
        rollback_compare_epoch_lookback=(
            None
            if getattr(cfg, "rollback_compare_epoch_lookback", None) is None
            else int(getattr(cfg, "rollback_compare_epoch_lookback"))
        ),
        rollback_epoch_lookback=int(getattr(cfg, "rollback_epoch_lookback", 20)),
        lambda_increase_cooldown_epochs=int(
            getattr(cfg, "lambda_increase_cooldown_epochs", 10)
        ),
        rollback_on_degradation=bool(getattr(cfg, "rollback_on_degradation", True)),
        rollback_on_collapse=bool(getattr(cfg, "rollback_on_collapse", True)),
        max_rollbacks=int(getattr(cfg, "max_rollbacks", 6)),
        freeze_on_rollback_limit=bool(getattr(cfg, "freeze_on_rollback_limit", True)),
        recovery_config=recovery_config,
    )


def _build_gate_mode_schedule(training_arguments: DictConfig) -> GateModeScheduleState | None:
    cfg = getattr(training_arguments, "gate_mode_schedule", None)
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return None

    start_epoch = int(getattr(cfg, "start_epoch"))
    if start_epoch <= 0:
        raise ValueError("training_arguments.gate_mode_schedule.start_epoch must be >= 1.")

    initial_train_gate_mode = str(getattr(cfg, "initial_train_gate_mode"))
    target_train_gate_mode = str(getattr(cfg, "target_train_gate_mode"))
    initial_eval_gate_mode = str(
        getattr(cfg, "initial_eval_gate_mode", getattr(cfg, "initial_train_gate_mode"))
    )
    target_eval_gate_mode = str(
        getattr(cfg, "target_eval_gate_mode", getattr(cfg, "target_train_gate_mode"))
    )

    return GateModeScheduleState(
        start_epoch=start_epoch,
        initial_train_gate_mode=initial_train_gate_mode,
        target_train_gate_mode=target_train_gate_mode,
        initial_eval_gate_mode=initial_eval_gate_mode,
        target_eval_gate_mode=target_eval_gate_mode,
    )


def _build_batchnorm_recalibration(
    training_arguments: DictConfig,
) -> BatchNormRecalibrationState | None:
    cfg = getattr(training_arguments, "batchnorm_recalibration", None)
    if cfg is not None and not bool(getattr(cfg, "enabled", True)):
        return None

    return BatchNormRecalibrationState(
        num_batches=(
            200
            if cfg is None
            else int(getattr(cfg, "num_batches", 200))
        ),
        reset_running_stats=(
            True
            if cfg is None
            else bool(getattr(cfg, "reset_running_stats", True))
        ),
        train_gate_mode=(
            "deterministic_hard"
            if cfg is None
            else getattr(cfg, "train_gate_mode", "deterministic_hard")
        ),
        eval_gate_mode=(
            "deterministic_hard"
            if cfg is None
            else getattr(cfg, "eval_gate_mode", "deterministic_hard")
        ),
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
    train_dataloader = dataloaders.train_dataloader
    cuda_device = _cuda_device(device)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
    train_started_at = perf_counter()
    n_seen = 0

    for X, y in train_dataloader:
        X = X.to(device, non_blocking=cuda_device is not None)
        y = y.to(device, non_blocking=cuda_device is not None)
        n_seen += int(X.shape[0])
        optimizer.zero_grad(set_to_none=True)
        output = model(X, y)

        metrics.train_metrics.update(X, output, y, model)

        output.loss.backward()
        optimizer.step()
        if scheduler_state is not None and scheduler_state.interval == "batch":
            batch_metrics = collect_batch_metrics(output, y, model) if scheduler_state.needs_metric else {}
            scheduler_state.step(batch_metrics)

    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        max_memory_allocated_mb = float(torch.cuda.max_memory_allocated(cuda_device) / 1024**2)
    else:
        max_memory_allocated_mb = None
    train_time = perf_counter() - train_started_at
    samples_per_sec = float(n_seen / train_time) if train_time > 0.0 else 0.0
    return {
        "train_time_sec": float(train_time),
        "train_samples_seen": int(n_seen),
        "train_samples_per_sec": samples_per_sec,
        "train_batch_size": int(getattr(train_dataloader, "batch_size", 0) or 0),
        "train_max_memory_allocated_mb": max_memory_allocated_mb,
    }


def train(model: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler_state: SchedulerState | None,
          dataloaders: Dataloaders,
          training_arguments: DictConfig,
          metrics,
          device: str,
          baseline_accuracy_reference: BaselineAccuracyReference | None = None,
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
    gate_mode_schedule = _build_gate_mode_schedule(training_arguments)
    batchnorm_recalibration = _build_batchnorm_recalibration(training_arguments)
    collapse_guard = _build_collapse_guard(training_arguments)
    adaptive_lambda = _build_adaptive_lambda(
        training_arguments,
        model,
        baseline_accuracy_by_epoch=(
            baseline_accuracy_reference.accuracy_by_epoch
            if baseline_accuracy_reference is not None
            else None
        ),
    )
    completed_epochs = 0
    stop_info: dict[str, Any] | None = None
    recalibration_info: dict[str, Any] | None = None

    def _apply_adaptive_lambda(target_model: nn.Module, lambda_coef: float) -> None:
        _set_model_lambda_coef(target_model, lambda_coef, bypass_gumbel=False)

    if adaptive_lambda is not None:
        adaptive_lambda.apply_initial_state(
            model,
            apply_lambda=_apply_adaptive_lambda,
        )
        if run_history is not None:
            runtime_metadata = dict(run_history.runtime_metadata)
            if baseline_accuracy_reference is not None:
                runtime_metadata["adaptive_lambda_baseline"] = {
                    "root_dir": str(baseline_accuracy_reference.root_dir),
                    "history_path": str(baseline_accuracy_reference.history_path),
                    "metric_name": baseline_accuracy_reference.metric_name,
                    "num_epochs": len(baseline_accuracy_reference.accuracy_by_epoch),
                }
            runtime_metadata["adaptive_lambda"] = adaptive_lambda.summary_state()
            run_history.set_runtime_metadata(runtime_metadata)

    epoch_num = 1
    while epoch_num <= total_epochs:
        if lambda_warmup is not None:
            lambda_warmup.step(epoch_num, model)
        if gate_mode_schedule is not None:
            gate_mode_schedule.step(epoch_num, model)
        epoch_started_at = perf_counter()

        train_profile_metrics = train_epoch(
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
        train_time = float(train_profile_metrics["train_time_sec"])

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
        observed_epoch_metrics = {
            **train_metrics,
            **valid_metrics,
        }

        best_checkpoint_extra_state = {
            "model_lambda_coef": _resolve_model_lambda_coef(model),
            "gumbel_bypass_enabled": _resolve_model_gumbel_bypass(model),
        }

        if run_history is not None:
            run_history.log_channel_history(epoch_num, model)
            run_history.log_gate_history(epoch_num, "valid", model)

        if scheduler_state is not None and scheduler_state.interval == "epoch":
            scheduler_state.step(observed_epoch_metrics)

        if run_history is not None and run_history.should_update_best(epoch_num, valid_metrics):
            run_history.save_checkpoint(
                "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch_num,
                metrics=observed_epoch_metrics,
                scheduler_state=scheduler_state,
                extra_state=best_checkpoint_extra_state,
            )

        controller_metrics: dict[str, Any] = {}
        adaptive_collapse_handled = False
        next_epoch_num = epoch_num + 1
        if adaptive_lambda is not None:
            adaptive_step_result = adaptive_lambda.on_epoch_end(
                epoch=epoch_num,
                model=model,
                optimizer=optimizer,
                valid_metrics=valid_metrics,
                scheduler_state=scheduler_state,
                scaler=None,
                apply_lambda=_apply_adaptive_lambda,
            )
            controller_metrics = dict(adaptive_step_result.metrics)
            adaptive_collapse_handled = (
                adaptive_step_result.rolled_back
                or (
                    adaptive_step_result.collapse_detected
                    and adaptive_step_result.action in {"collapse_rollback", "rollback_limit_freeze"}
                )
            )
            if adaptive_step_result.resume_epoch is not None:
                next_epoch_num = int(adaptive_step_result.resume_epoch)
            if run_history is not None:
                runtime_metadata = dict(run_history.runtime_metadata)
                runtime_metadata["adaptive_lambda"] = adaptive_lambda.summary_state()
                run_history.set_runtime_metadata(runtime_metadata)

        post_epoch_extra_metrics = {
            **train_profile_metrics,
            "valid_time_sec": float(valid_time),
            "epoch_time_sec": float(perf_counter() - epoch_started_at),
            "model_lambda_coef": _resolve_model_lambda_coef(model),
            "gumbel_bypass_enabled": _resolve_model_gumbel_bypass(model),
            **controller_metrics,
        }

        if mlflow_logger is not None:
            mlflow_logger.log_metrics(train_metrics, step=epoch_num)
            mlflow_logger.log_metrics(valid_metrics, step=epoch_num)
            controller_numeric_metrics = {
                key: value
                for key, value in controller_metrics.items()
                if isinstance(value, (int, float))
            }
            if controller_numeric_metrics:
                mlflow_logger.log_metrics(controller_numeric_metrics, step=epoch_num)

        if run_history is not None:
            run_history.save_checkpoint(
                "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch_num,
                metrics={
                    **observed_epoch_metrics,
                    **controller_metrics,
                },
                scheduler_state=scheduler_state,
                extra_state=post_epoch_extra_metrics,
            )
            run_history.log_epoch(
                epoch_num,
                train_metrics,
                valid_metrics,
                extra_metrics=post_epoch_extra_metrics,
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
        if collapse_guard is not None and not adaptive_collapse_handled:
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

        should_stop = False
        stop_message: str | None = None
        if early_stopping is not None and not adaptive_collapse_handled:
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
                extra_metrics=post_epoch_extra_metrics,
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
        epoch_num = next_epoch_num

    final_epoch = completed_epochs or total_epochs
    if batchnorm_recalibration is not None:
        recalibration_info = batchnorm_recalibration.apply(
            model,
            dataloaders.train_dataloader,
            device=device,
            progress_context=progress_context,
        )
        if recalibration_info.get("applied"):
            recalibrated_valid_started_at = perf_counter()
            evaluate(
                model,
                dataloaders.valid_dataloader,
                metrics.valid_metrics,
                device,
                total_epochs=final_epoch,
                stage="valid",
                epoch=final_epoch,
                run_history=run_history,
                progress_context=progress_context,
            )
            recalibrated_valid_time = perf_counter() - recalibrated_valid_started_at
            recalibrated_valid_metrics = dict(metrics.valid_metrics.compute())
            metrics.valid_metrics.reset()
            last_valid_metrics = recalibrated_valid_metrics
            recalibration_info["post_recalibration_valid_metrics"] = dict(
                recalibrated_valid_metrics
            )
            recalibration_info["post_recalibration_valid_time_sec"] = float(
                recalibrated_valid_time
            )
            print(
                _build_post_recalibration_valid_log_line(
                    recalibrated_valid_metrics,
                    recalibrated_valid_time,
                    progress_context=progress_context,
                )
            )
            if mlflow_logger is not None:
                mlflow_logger.log_metrics(
                    _prefix_metric_keys(recalibrated_valid_metrics, "recalibrated"),
                    step=final_epoch,
                )
        if run_history is not None:
            runtime_metadata = dict(run_history.runtime_metadata)
            runtime_metadata["batchnorm_recalibration"] = recalibration_info
            run_history.set_runtime_metadata(runtime_metadata)
            if recalibration_info.get("applied"):
                run_history.update_last_epoch(
                    {
                        **_prefix_metric_keys(last_valid_metrics, "recalibrated"),
                        "recalibrated_valid_time_sec": float(
                            recalibration_info.get("post_recalibration_valid_time_sec", 0.0)
                        ),
                    }
                )
                run_history.save_checkpoint(
                    "last.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=final_epoch,
                    metrics={
                        **last_train_metrics,
                        **last_valid_metrics,
                    },
                    scheduler_state=scheduler_state,
                    extra_state={
                        "model_lambda_coef": _resolve_model_lambda_coef(model),
                        "gumbel_bypass_enabled": _resolve_model_gumbel_bypass(model),
                        **(adaptive_lambda.summary_state() if adaptive_lambda is not None else {}),
                    },
                )

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
    if adaptive_lambda is not None:
        result["adaptive_lambda"] = adaptive_lambda.summary_state()
    if recalibration_info is not None:
        result["batchnorm_recalibration"] = recalibration_info
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
    optimizer_build_info: OptimizerBuildInfo | None = None,
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
    if optimizer_build_info is not None and optimizer_build_info.gate_weight_decay_scale is not None:
        params.update({
            "optimizer.gate_weight_decay_scale": optimizer_build_info.gate_weight_decay_scale,
            "optimizer.gate_param_group_enabled": optimizer_build_info.gate_param_group_enabled,
            "optimizer.num_gates": optimizer_build_info.num_gates,
            "optimizer.num_gate_param_tensors": optimizer_build_info.num_gate_param_tensors,
            "optimizer.gate_weight_decay": optimizer_build_info.gate_weight_decay,
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
    baseline_accuracy_reference = _ensure_adaptive_baseline_reference(
        config,
        progress_context=progress_context,
    )
    resolved_seed = set_random_seed(getattr(config, "seed", None))
    device = resolve_device(config)
    model = instantiate(config.model).to(device)
    lambda_warmup = _build_lambda_warmup(config.training_arguments)
    if lambda_warmup is not None and _adaptive_lambda_enabled(config.training_arguments):
        raise ValueError("adaptive_lambda and lambda_warmup cannot be enabled at the same time.")
    if lambda_warmup is not None:
        lambda_warmup.apply_initial_state(model)
    gate_mode_schedule = _build_gate_mode_schedule(config.training_arguments)
    if gate_mode_schedule is not None:
        gate_mode_schedule.apply_initial_state(model)
    dataloaders = instantiate(config.dataloaders)
    optimizer, optimizer_build_info = _build_optimizer(config, model)
    scheduler_state = _build_scheduler(config, optimizer)
    metrics = prepare_metrics(instantiate(config.metrics))
    mlflow_logger = MLflowLogger(config) if _is_mlflow_enabled(config) else None
    run_history = RunHistory(config)
    runtime_snapshot = _assert_runtime_lambda_consistency(
        config,
        model,
        run_history,
        progress_context=progress_context,
    )
    if baseline_accuracy_reference is not None:
        runtime_snapshot["adaptive_lambda_baseline"] = {
            "root_dir": str(baseline_accuracy_reference.root_dir),
            "history_path": str(baseline_accuracy_reference.history_path),
            "metric_name": baseline_accuracy_reference.metric_name,
            "num_epochs": len(baseline_accuracy_reference.accuracy_by_epoch),
        }
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
            optimizer_build_info=optimizer_build_info,
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
            baseline_accuracy_reference=baseline_accuracy_reference,
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
    if baseline_accuracy_reference is not None:
        result["adaptive_lambda_baseline"] = {
            "root_dir": str(baseline_accuracy_reference.root_dir),
            "history_path": str(baseline_accuracy_reference.history_path),
            "metric_name": baseline_accuracy_reference.metric_name,
            "num_epochs": len(baseline_accuracy_reference.accuracy_by_epoch),
        }
    return result
