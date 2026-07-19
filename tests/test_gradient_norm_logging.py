import pytest
import torch
import torch.nn as nn

from net_complexity.training.gradient_norms import GradientNormLogger


class TinyTwoLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 1, bias=False)


def test_gradient_norm_logger_separates_ce_regularization_and_total_gradients():
    model = TinyTwoLossModel()
    logger = GradientNormLogger(model)
    x = torch.tensor([[1.0, -2.0]])
    prediction = model.second(model.first(x))
    ce_loss = prediction.square().mean()
    regularization_term = 0.25 * model.first.weight.square().mean()
    total_loss = ce_loss + regularization_term

    logger.collect_autograd("ce", ce_loss)
    logger.collect_autograd("regularization", regularization_term)
    total_loss.backward()
    logger.collect_total()
    metrics = logger.compute()

    assert metrics["grad_norm_ce_total_mean"] > 0.0
    assert metrics["grad_norm_regularization_total_mean"] > 0.0
    assert metrics["grad_norm_total_total_mean"] > 0.0
    assert metrics["grad_norm_regularization_layer_first_mean"] > 0.0
    assert metrics["grad_norm_regularization_layer_second_mean"] == 0.0
    assert metrics["grad_norm_total_total_mean"] != pytest.approx(
        metrics["grad_norm_ce_total_mean"]
    )


def test_gradient_norm_logger_aggregates_epoch_mean_and_max():
    model = TinyTwoLossModel()
    logger = GradientNormLogger(model, log_per_layer=False)

    for scale in (1.0, 2.0):
        loss = model.first.weight.sum() * scale
        logger.collect_autograd("ce", loss)

    metrics = logger.compute()
    assert metrics["grad_norm_ce_total_max"] > metrics["grad_norm_ce_total_mean"]
    assert not any("_layer_" in key for key in metrics)
