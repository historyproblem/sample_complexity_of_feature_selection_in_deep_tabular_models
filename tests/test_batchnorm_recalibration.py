import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from net_complexity.data.dataloaders import Dataloaders
from net_complexity.metrics.base import BaseMetric, MultiLossMetric
from net_complexity.models.feature_selection import GumbelLayer
from net_complexity.models.outputs import ClassifModelOutput
from net_complexity.training.engine import _build_batchnorm_recalibration, prepare_metrics, train
from net_complexity.training.meta import Metrics


class TinyBatchNormRecalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gumbel = GumbelLayer(
            input_dim=1,
            train_gate_mode="gumbel_hard",
            eval_gate_mode="deterministic_hard",
        )
        self.bn = nn.BatchNorm2d(1, momentum=1.0)
        with torch.no_grad():
            self.gumbel.logits.copy_(torch.tensor([[-10.0, 10.0]]))

    def forward(self, X, y=None):
        return self.bn(self.gumbel(X))


class TinyBatchNormTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gumbel = GumbelLayer(
            input_dim=1,
            train_gate_mode="deterministic_hard",
            eval_gate_mode="deterministic_hard",
        )
        self.bn = nn.BatchNorm2d(1, momentum=1.0)
        self.classifier = nn.Linear(4, 2)
        with torch.no_grad():
            self.gumbel.logits.copy_(torch.tensor([[-10.0, 10.0]]))

    def forward(self, X, y):
        features = self.bn(self.gumbel(X))
        logits = self.classifier(features.flatten(start_dim=1))
        ce_loss = F.cross_entropy(logits, y)
        return ClassifModelOutput(
            ce_loss=ce_loss,
            regularization_loss=logits.new_zeros(()),
            loss=ce_loss,
            logits=logits,
        )


class UpdateCountingMetric(BaseMetric):
    def __init__(self):
        self.total_updates = 0
        self.current_updates = 0

    def update(self, input, output, targets, model=None):
        self.total_updates += 1
        self.current_updates += 1

    def compute(self):
        return {"num_updates": float(self.current_updates)}

    def reset(self):
        self.current_updates = 0


def test_batchnorm_recalibration_updates_bn_stats_and_restores_gate_modes():
    state = _build_batchnorm_recalibration(
        OmegaConf.create(
            {
                "batchnorm_recalibration": {
                    "enabled": True,
                    "num_batches": 2,
                    "reset_running_stats": True,
                    "train_gate_mode": "deterministic_hard",
                    "eval_gate_mode": "deterministic_hard",
                }
            }
        )
    )

    model = TinyBatchNormRecalibrationModel()
    model.eval()
    model.bn.running_mean.fill_(99.0)
    model.bn.running_var.fill_(99.0)

    dataloader = DataLoader(
        TensorDataset(
            torch.full((4, 1, 2, 2), 3.0),
            torch.zeros(4, dtype=torch.long),
        ),
        batch_size=2,
        shuffle=False,
    )

    info = state.apply(model, dataloader, device="cpu")

    assert info["applied"] is True
    assert info["num_batches_processed"] == 2
    assert info["num_examples_processed"] == 4
    assert info["num_batchnorm_modules"] == 1
    assert model.training is False
    assert model.gumbel.train_gate_mode == "gumbel_hard"
    assert model.gumbel.eval_gate_mode == "deterministic_hard"
    assert model.bn.num_batches_tracked.item() == 2
    torch.testing.assert_close(model.bn.running_mean, torch.tensor([3.0]))


def test_train_runs_final_validation_again_after_batchnorm_recalibration():
    model = TinyBatchNormTrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    dataset = TensorDataset(
        torch.full((4, 1, 2, 2), 3.0),
        torch.zeros(4, dtype=torch.long),
    )
    dataloaders = Dataloaders()
    dataloaders.train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    dataloaders.valid_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    dataloaders.test_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    valid_metric = UpdateCountingMetric()
    test_metric = UpdateCountingMetric()
    metrics = prepare_metrics(
        Metrics(
            train_metrics=[MultiLossMetric()],
            valid_metrics=[valid_metric],
            test_metrics=[test_metric],
        )
    )
    training_arguments = OmegaConf.create(
        {
            "num_epochs": 1,
            "batchnorm_recalibration": {
                "enabled": True,
                "num_batches": 1,
                "reset_running_stats": True,
                "train_gate_mode": "deterministic_hard",
                "eval_gate_mode": "deterministic_hard",
            },
        }
    )

    result = train(
        model,
        optimizer,
        scheduler_state=None,
        dataloaders=dataloaders,
        training_arguments=training_arguments,
        metrics=metrics,
        device="cpu",
        mlflow_logger=None,
        run_history=None,
    )

    assert valid_metric.total_updates == 4
    assert test_metric.total_updates == 2
    assert result["last_valid_metrics"]["valid_num_updates"] == 2.0
    assert result["batchnorm_recalibration"]["applied"] is True
    assert (
        result["batchnorm_recalibration"]["post_recalibration_valid_metrics"]["valid_num_updates"]
        == 2.0
    )
