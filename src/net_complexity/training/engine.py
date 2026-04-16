from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Any, Callable, Mapping

from net_complexity.data.dataloaders import Dataloaders
from net_complexity.metrics.base import BaseMetric, Multimetric
from net_complexity.models.feature_selection import get_gumbel_modules
from net_complexity.training.meta import Metrics
from net_complexity.training.run_history import RunHistory
from net_complexity.training.tracking import MLflowLogger


EpochEndCallback = Callable[
    [int, Mapping[str, float], Mapping[str, float], nn.Module, torch.optim.Optimizer, RunHistory | None],
    None,
]
ProgressContext = Mapping[str, Any]


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


def _to_float(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value.detach().cpu().mean().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def collect_batch_metrics(output, targets, model: nn.Module | None = None) -> dict[str, float]:
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
        batch_metrics["min_estim_prob"] = float(min(estim_means))

    return batch_metrics


def _planned_bar_count(total_epochs: int) -> int:
    return total_epochs * 2 + 1


def _bar_index(stage: str, epoch: int, total_epochs: int) -> int:
    if stage == "train":
        return (epoch - 1) * 2 + 1
    if stage == "valid":
        return (epoch - 1) * 2 + 2
    return _planned_bar_count(total_epochs)


def _progress_desc(
    stage: str,
    epoch: int,
    total_epochs: int,
    progress_context: ProgressContext | None = None,
) -> str:
    parts = []
    if progress_context is not None:
        trial_number = progress_context.get("trial_number")
        trial_total = progress_context.get("trial_total")
        if trial_number is not None and trial_total is not None:
            parts.append(f"trial {trial_number}/{trial_total}")

    parts.append(f"bar {_bar_index(stage, epoch, total_epochs)}/{_planned_bar_count(total_epochs)}")
    parts.append(f"epoch {min(epoch, total_epochs)}/{total_epochs}")
    parts.append(stage)
    return " | ".join(f"[{part}]" for part in parts)


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

    for batch_index, (X, y) in enumerate(
        tqdm(
            dataloader,
            desc=_progress_desc(stage, epoch, total_epochs, progress_context),
            dynamic_ncols=True,
        ),
        start=1,
    ):
        X, y = X.to(device), y.to(device)
        output = model(X, y)
        metric.update(X, output, y, model)
        if run_history is not None:
            run_history.log_batch(
                stage=stage,
                epoch=epoch,
                batch_in_epoch=batch_index,
                metrics=collect_batch_metrics(output, y, model),
            )


def train_epoch(model,
                optimizer: torch.optim.Optimizer,
                dataloaders: Dataloaders,
                metrics: Metrics,
                device: str,
                total_epochs: int,
                epoch: int = 0,
                run_history: RunHistory | None = None,
                progress_context: ProgressContext | None = None):
    """Train for one epoch."""
    model.train()

    for batch_index, (X, y) in enumerate(
        tqdm(
            dataloaders.train_dataloader,
            desc=_progress_desc("train", epoch, total_epochs, progress_context),
            dynamic_ncols=True,
        ),
        start=1,
    ):
        X, y = X.to(device), y.to(device)
        output = model(X, y)

        metrics.train_metrics.update(X, output, y, model)
        if run_history is not None:
            run_history.log_batch(
                stage="train",
                epoch=epoch,
                batch_in_epoch=batch_index,
                metrics=collect_batch_metrics(output, y, model),
            )

        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def train(model: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler | None,
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
    completed_epochs = 0

    for epoch in range(total_epochs):
        epoch_num = epoch + 1
        train_epoch(
            model,
            optimizer,
            dataloaders,
            metrics,
            device,
            total_epochs=total_epochs,
            epoch=epoch_num,
            run_history=run_history,
            progress_context=progress_context,
        )
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
            run_history.log_epoch(epoch_num, train_metrics, valid_metrics)
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

        if epoch_end_callback is not None:
            epoch_end_callback(
                epoch_num,
                train_metrics,
                valid_metrics,
                model,
                optimizer,
                run_history,
            )

        if scheduler is not None:
            scheduler.step()

        should_stop = False
        if early_stopping is not None:
            should_stop = early_stopping.step(epoch_num, valid_metrics)
            if should_stop:
                tqdm.write(
                    f"Early stopping at epoch {epoch_num}: "
                    f"best {early_stopping.monitor}={early_stopping.best_value:.6f} "
                    f"at epoch {early_stopping.best_epoch}"
                )

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
        run_history.save_summary(test_metrics)
    metrics.test_metrics.reset()
    return {
        "last_train_metrics": last_train_metrics,
        "last_valid_metrics": last_valid_metrics,
        "test_metrics": dict(test_metrics),
    }


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
) -> None:
    if mlflow_logger is None:
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    mlflow_logger.log_params({
        "run_history.run_id": run_history.run_id,
        "run_history.run_dir": str(run_history.run_dir),
        "model.total_parameters": total_params,
        "model.trainable_parameters": trainable_params,
        "optimizer.type": optimizer.__class__.__name__,
        "optimizer.lr": optimizer.param_groups[0].get("lr"),
        "optimizer.weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
    })


def log_run_artifacts(config: DictConfig, run_history: RunHistory, mlflow_logger: MLflowLogger | None) -> None:
    if mlflow_logger is None:
        return
    if not getattr(config, "mlflow", None) or not getattr(config.mlflow, "log_artifacts", False):
        return

    for artifact_path, artifact_dir in (
        (run_history.config_path, "run_history"),
        (run_history.history_path, "run_history"),
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
    device = resolve_device(config)
    model = instantiate(config.model).to(device)
    print(model)
    dataloaders = instantiate(config.dataloaders)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    scheduler = instantiate(config.scheduler, optimizer=optimizer) if getattr(config, "scheduler", None) is not None else None
    metrics = prepare_metrics(instantiate(config.metrics))
    mlflow_logger = MLflowLogger(config) if getattr(config, "mlflow", None) is not None else None
    run_history = RunHistory(config)
    print(f"Run artifacts: {run_history.run_dir}")

    result: dict[str, Any]
    if mlflow_logger is not None:
        mlflow_logger.setup()
        log_training_metadata(model, optimizer, run_history, mlflow_logger)

    try:
        result = train(
            model,
            optimizer,
            scheduler,
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
    })
    return result
