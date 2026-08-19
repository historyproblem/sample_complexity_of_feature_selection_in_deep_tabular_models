import torch.nn as nn
from omegaconf import OmegaConf

from net_complexity.models.aig import AIGBlockGate
from net_complexity.models.feature_selection import GumbelLayer
from net_complexity.training.engine import _build_optimizer


class TinyGateOptimizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 3)
        self.gate = GumbelLayer(input_dim=3)
        self.head = nn.Linear(3, 2)


class TinyAIGGateOptimizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 3)
        self.gate = AIGBlockGate(in_channels=4, hidden_channels=2)
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


def test_build_optimizer_splits_aig_gate_final_conv_into_its_own_group():
    model = TinyAIGGateOptimizerModel()
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

    final_conv = model.gate.router[-1]
    assert len(optimizer.param_groups) == 2
    assert build_info.gate_param_group_enabled is True
    # AIG gates make one block-level decision -> num_gates=1, not per-channel.
    assert build_info.num_gates == 1
    assert build_info.num_gate_param_tensors == 2  # final conv weight + bias
    assert build_info.gate_weight_decay == 20.0 * 1e-4 / 1.0

    gate_group = next(
        group
        for group in optimizer.param_groups
        if any(parameter is final_conv.weight for parameter in group["params"])
    )
    base_group = next(
        group
        for group in optimizer.param_groups
        if all(parameter is not final_conv.weight for parameter in group["params"])
    )

    assert any(parameter is final_conv.bias for parameter in gate_group["params"])
    # The gate's hidden layer (router[0]/router[1]) stays in the base group,
    # matching the reference optimizer which only splits out 'fc2'.
    assert any(parameter is model.gate.router[0].weight for parameter in base_group["params"])
    assert gate_group["weight_decay"] == build_info.gate_weight_decay
    assert base_group["weight_decay"] == 1e-4


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
