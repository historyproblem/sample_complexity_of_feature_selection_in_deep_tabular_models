from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from net_complexity.data.dataloaders import Dataloaders
from net_complexity.metrics.base import MultiLossMetric
from net_complexity.models.outputs import ClassifModelOutput
from net_complexity.training import engine
from net_complexity.training.adaptive_lambda import AdaptiveLambdaStepResult
from net_complexity.training.meta import Metrics


class TinyAdaptiveRollbackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, X, y):
        logits = self.linear(X.flatten(start_dim=1))
        ce_loss = F.cross_entropy(logits, y)
        return ClassifModelOutput(
            ce_loss=ce_loss,
            regularization_loss=logits.new_zeros(()),
            loss=ce_loss,
            logits=logits,
        )


class _FakeAdaptiveController:
    def __init__(self):
        self.calls: list[int] = []

    def apply_initial_state(self, model, *, apply_lambda):
        del model, apply_lambda

    def on_epoch_end(
        self,
        *,
        epoch,
        model,
        optimizer,
        valid_metrics,
        scheduler_state,
        scaler,
        apply_lambda,
    ) -> AdaptiveLambdaStepResult:
        del model, optimizer, valid_metrics, scheduler_state, scaler, apply_lambda
        self.calls.append(int(epoch))
        if self.calls == [1, 2]:
            return AdaptiveLambdaStepResult(
                action="periodic_rollback",
                reason="rewind",
                lambda_changed=False,
                rolled_back=True,
                resume_epoch=1,
                collapse_detected=False,
                checkpoint_saved=False,
                metrics={},
            )
        return AdaptiveLambdaStepResult(
            action="hold",
            reason="continue",
            lambda_changed=False,
            rolled_back=False,
            resume_epoch=None,
            collapse_detected=False,
            checkpoint_saved=False,
            metrics={},
        )

    def summary_state(self):
        return {"enabled": True, "calls": list(self.calls)}


def test_train_rewinds_epoch_loop_when_adaptive_lambda_requests_resume(monkeypatch):
    model = TinyAdaptiveRollbackModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    dataset = TensorDataset(
        torch.ones(4, 1, 2, 2),
        torch.zeros(4, dtype=torch.long),
    )
    dataloaders = Dataloaders()
    dataloaders.train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    dataloaders.valid_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    dataloaders.test_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    metrics = engine.prepare_metrics(
        Metrics(
            train_metrics=[MultiLossMetric()],
            valid_metrics=[MultiLossMetric()],
            test_metrics=[MultiLossMetric()],
        )
    )
    training_arguments = OmegaConf.create(
        {
            "num_epochs": 2,
            "adaptive_lambda": {
                "enabled": True,
            },
        }
    )

    fake_controller = _FakeAdaptiveController()
    monkeypatch.setattr(engine, "_build_adaptive_lambda", lambda *args, **kwargs: fake_controller)

    observed_epochs: list[int] = []

    engine.train(
        model,
        optimizer,
        scheduler_state=None,
        dataloaders=dataloaders,
        training_arguments=training_arguments,
        metrics=metrics,
        device="cpu",
        mlflow_logger=None,
        run_history=None,
        epoch_end_callback=lambda epoch, *args: observed_epochs.append(int(epoch)),
    )

    assert observed_epochs == [1, 2, 1, 2]
    assert fake_controller.calls == [1, 2, 1, 2]
