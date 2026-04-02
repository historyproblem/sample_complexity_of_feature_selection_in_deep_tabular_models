from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from net_complexity.dataloaders import Dataloaders
from net_complexity.logger import MLflowLogger
from net_complexity.meta import Metrics
from net_complexity.metrics.base import BaseMetric, Multimetric
from net_complexity.run_history import RunHistory
from net_complexity.wrappers import get_gumbel_modules


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


@torch.inference_mode()
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             metric: Multimetric,
             device: str,
             stage: str = "valid",
             epoch: int = 0,
             run_history: RunHistory | None = None):
    model.eval()

    for batch_index, (X, y) in enumerate(tqdm(dataloader), start=1):
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
                epoch: int = 0,
                run_history: RunHistory | None = None):
    """Train for one epoch."""
    model.train()

    for batch_index, (X, y) in enumerate(tqdm(dataloaders.train_dataloader), start=1):
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
          dataloaders: Dataloaders,
          training_arguments: DictConfig,
          metrics,
          device: str,
          mlflow_logger=None,
          run_history: RunHistory | None = None):

    model.to(device)
    for epoch in range(training_arguments.num_epochs):
        epoch_num = epoch + 1
        train_epoch(
            model,
            optimizer,
            dataloaders,
            metrics,
            device,
            epoch=epoch_num,
            run_history=run_history,
        )
        evaluate(model, dataloaders.valid_dataloader,
                 metrics.valid_metrics, device,
                 stage="valid",
                 epoch=epoch_num,
                 run_history=run_history)

        train_metrics = metrics.train_metrics.compute()
        valid_metrics = metrics.valid_metrics.compute()
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

        metrics.train_metrics.reset()
        metrics.valid_metrics.reset()

    evaluate(
        model,
        dataloaders.test_dataloader,
        metrics.test_metrics,
        device,
        stage="test",
        epoch=training_arguments.num_epochs,
        run_history=run_history,
    )
    test_metrics = metrics.test_metrics.compute()
    if mlflow_logger is not None:
        mlflow_logger.log_metrics(
            test_metrics,
            step=training_arguments.num_epochs,
        )
        mlflow_logger.log_model(model, model_name="final_model")
    if run_history is not None:
        run_history.save_summary(test_metrics)
    metrics.test_metrics.reset()


@hydra.main(config_path="../../configs/", config_name="main_gumbel", version_base=None)
def main(config: DictConfig):
    requested_device = getattr(config, "device", None)
    if requested_device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = str(requested_device)
    model = instantiate(config.model).to(device)
    print(model)
    dataloaders = instantiate(config.dataloaders)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    metrics = instantiate(config.metrics)
    mlflow_logger = MLflowLogger(config)
    mlflow_logger.setup()
    run_history = RunHistory(config)
    print(f"Run artifacts: {run_history.run_dir}")
    if mlflow_logger:
        # Log model architecture summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
        mlflow_logger.log_params({
            "run_history.run_id": run_history.run_id,
            "run_history.run_dir": str(run_history.run_dir),
            "model.total_parameters": total_params,
            "model.trainable_parameters": trainable_params,
            "optimizer.type": optimizer.__class__.__name__,
            "optimizer.lr": optimizer.param_groups[0]['lr']
        })

    metrics.train_metrics = BaseMetric() if len(
        metrics.train_metrics) == 0 else Multimetric(metrics.train_metrics, 'train')
    metrics.valid_metrics = BaseMetric() if len(
        metrics.valid_metrics) == 0 else Multimetric(metrics.valid_metrics, 'valid')
    metrics.test_metrics = BaseMetric() if len(
        metrics.test_metrics) == 0 else Multimetric(metrics.test_metrics, 'test')

    train(
        model,
        optimizer,
        dataloaders,
        config.training_arguments,
        metrics,
        device,
        mlflow_logger=mlflow_logger,
        run_history=run_history,
    )

    if getattr(config, "mlflow", None) and getattr(config.mlflow, "log_artifacts", False):
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

    if mlflow_logger:
        mlflow_logger.close()


if __name__ == "__main__":
    main()
