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


@torch.inference_mode()
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             metric: Multimetric,
             device: str):
    model.eval()

    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        output = model(X, y)
        metric.update(X, output, y, model)


def train_epoch(model,
                optimizer: torch.optim.Optimizer,
                dataloaders: Dataloaders,
                metrics: Metrics,
                device: str):
    """Train for one epoch."""
    model.train()

    for X, y in tqdm(dataloaders.train_dataloader):
        X, y = X.to(device), y.to(device)
        output = model(X, y)

        metrics.train_metrics.update(X, output, y, model)

        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def train(model: nn.Module,
          optimizer: torch.optim.Optimizer,
          dataloaders: Dataloaders,
          training_arguments: DictConfig,
          metrics,
          device: str,
          mlflow_logger=None):

    model.to(device)
    for epoch in range(training_arguments.num_epochs):
        train_epoch(model, optimizer, dataloaders, metrics, device)
        evaluate(model, dataloaders.valid_dataloader,
                 metrics.valid_metrics, device)

        train_metrics = metrics.train_metrics.compute()
        mlflow_logger.log_metrics(train_metrics, step=epoch)

        valid_metrics = metrics.valid_metrics.compute()
        mlflow_logger.log_metrics(valid_metrics, step=epoch)

        metrics.train_metrics.reset()
        metrics.valid_metrics.reset()

    evaluate(model, dataloaders.test_dataloader, metrics.test_metrics, device)
    mlflow_logger.log_metrics(metrics.test_metrics.compute(), step=0)
    metrics.test_metrics.reset()


@hydra.main(config_path="../../configs/test_configs", config_name="test", version_base=None)
def main(config: DictConfig):
    # device = config.device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = instantiate(config.model).to(device)
    print(model)
    dataloaders = instantiate(config.dataloaders)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    metrics = instantiate(config.metrics)
    mlflow_logger = MLflowLogger(config)
    mlflow_logger.setup()
    if mlflow_logger:
        # Log model architecture summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
        mlflow_logger.log_params({
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
        mlflow_logger=mlflow_logger
    )

    if mlflow_logger:
        mlflow_logger.close()


if __name__ == "__main__":
    main()
