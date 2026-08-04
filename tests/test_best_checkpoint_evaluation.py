from __future__ import annotations

import json

import torch
from omegaconf import OmegaConf

from net_complexity.data.dataloaders import Dataloaders
from net_complexity.training import engine
from net_complexity.training.meta import Metrics
from net_complexity.training.run_history import RunHistory


class _MetricState:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def compute(self) -> dict[str, float]:
        return dict(self.values)

    def reset(self) -> None:
        self.values = {}


def test_train_evaluates_and_reports_best_validation_checkpoint(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metric_state = Metrics(
        train_metrics=_MetricState(),
        valid_metrics=_MetricState(),
        test_metrics=_MetricState(),
    )
    observed_test_weights: list[float] = []

    def fake_train_epoch(model, *args, epoch, **kwargs):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        metric_state.train_metrics.values = {"train_loss": float(epoch)}

    def fake_evaluate(model, _dataloader, metrics, _device, *, stage, epoch, **kwargs):
        if stage == "valid":
            metrics.values = {
                "valid_accuracy": 0.9 if epoch == 1 else 0.8,
            }
        else:
            observed_test_weights.append(float(model.weight.detach().item()))
            metrics.values = {"test_accuracy": observed_test_weights[-1]}

    monkeypatch.setattr(engine, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(engine, "evaluate", fake_evaluate)

    config = OmegaConf.create(
        {
            "seed": 1,
            "run_history": {
                "root_dir": str(tmp_path),
                "run_name": "best-checkpoint-test",
                "monitor": "valid_accuracy",
                "mode": "max",
            },
        }
    )
    run_history = RunHistory(config)
    dataloaders = Dataloaders()

    result = engine.train(
        model,
        optimizer,
        scheduler_state=None,
        dataloaders=dataloaders,
        training_arguments=OmegaConf.create({"num_epochs": 2}),
        metrics=metric_state,
        device="cpu",
        run_history=run_history,
    )

    assert observed_test_weights == [1.0]
    assert result["test_metrics"] == {"test_accuracy": 1.0}
    summary = json.loads(run_history.summary_path.read_text(encoding="utf-8"))
    assert summary["test"] == {"test_accuracy": 1.0}
    assert summary["runtime"]["test_evaluation"] == {
        "checkpoint": "checkpoints/best.pt",
        "checkpoint_epoch": 1,
    }
