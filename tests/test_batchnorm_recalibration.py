import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from net_complexity.models.feature_selection import GumbelLayer
from net_complexity.training.engine import _build_batchnorm_recalibration


class RecordingGumbelLayer(GumbelLayer):
    def __init__(self):
        super().__init__(input_dim=1, temperature=1.0)
        self.training_modes_seen: list[bool] = []
        with torch.no_grad():
            self.logits.copy_(torch.tensor([[-10.0, 10.0]]))

    def forward(self, x):
        self.training_modes_seen.append(bool(self.training))
        return super().forward(x)


class TinyBatchNormRecalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gumbel = RecordingGumbelLayer()
        self.bn = nn.BatchNorm2d(1, momentum=1.0)

    def forward(self, X, y=None):
        return self.bn(self.gumbel(X))


def test_batchnorm_recalibration_updates_bn_stats_and_restores_gate_modes():
    state = _build_batchnorm_recalibration(
        OmegaConf.create(
            {
                "batchnorm_recalibration": {
                    "enabled": True,
                    "num_batches": 2,
                    "reset_running_stats": True,
                    "deterministic_gumbel": True,
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
    assert info["num_gumbel_modules"] == 1
    assert model.training is False
    assert model.gumbel.training is False
    assert model.gumbel.training_modes_seen == [False, False]
    assert model.bn.num_batches_tracked.item() == 2
    torch.testing.assert_close(model.bn.running_mean, torch.tensor([3.0]))
