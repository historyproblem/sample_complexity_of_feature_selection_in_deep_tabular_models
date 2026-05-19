import torch.nn as nn
from omegaconf import OmegaConf

from net_complexity.models.feature_selection import GumbelLayer
from net_complexity.training.engine import _build_optimizer


class TinyGateOptimizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 3)
        self.gate = GumbelLayer(input_dim=3)
        self.head = nn.Linear(3, 2)


def test_build_optimizer_scales_gate_weight_decay_by_number_of_gates():
    model = TinyGateOptimizerModel()
    config = OmegaConf.create(
        {
            "optimizer": {
                "_target_": "torch.optim.SGD",
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "nesterov": False,
                "gate_weight_decay_scale": 20.0,
            }
        }
    )

    optimizer, build_info = _build_optimizer(config, model)

    assert len(optimizer.param_groups) == 2
    assert build_info.gate_param_group_enabled is True
    assert build_info.num_gates == 3
    assert build_info.num_gate_param_tensors == 1
    assert build_info.gate_weight_decay_scale == 20.0
    assert build_info.gate_weight_decay == 20.0 * 1e-4 / 3.0

    gate_group = next(
        group
        for group in optimizer.param_groups
        if any(parameter is model.gate.logits for parameter in group["params"])
    )
    base_group = next(
        group
        for group in optimizer.param_groups
        if all(parameter is not model.gate.logits for parameter in group["params"])
    )

    assert gate_group["weight_decay"] == build_info.gate_weight_decay
    assert base_group["weight_decay"] == 1e-4


def test_build_optimizer_keeps_single_group_when_gate_scaling_is_not_configured():
    model = TinyGateOptimizerModel()
    config = OmegaConf.create(
        {
            "optimizer": {
                "_target_": "torch.optim.SGD",
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "nesterov": False,
            }
        }
    )

    optimizer, build_info = _build_optimizer(config, model)

    assert len(optimizer.param_groups) == 1
    assert build_info.gate_param_group_enabled is False
    assert build_info.gate_weight_decay_scale is None


def test_build_optimizer_rejects_negative_gate_weight_decay_scale():
    model = TinyGateOptimizerModel()
    config = OmegaConf.create(
        {
            "optimizer": {
                "_target_": "torch.optim.SGD",
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "nesterov": False,
                "gate_weight_decay_scale": -1.0,
            }
        }
    )

    try:
        _build_optimizer(config, model)
    except ValueError as error:
        assert "gate_weight_decay_scale" in str(error)
    else:
        raise AssertionError("Expected negative gate_weight_decay_scale to raise ValueError.")
