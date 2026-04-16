import csv
import math

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from net_complexity.metrics.gumbel import GumbelProbMetric
from net_complexity.models import feature_selection as feature_selection_module
from net_complexity.models.feature_selection import GumbelLayer
from net_complexity.training.engine import _build_epoch_log_line
from net_complexity.training.run_history import RunHistory

STGChannelLayer = getattr(feature_selection_module, "STGChannelLayer", None)

try:
    from net_complexity.metrics.stg import STGProbMetric
except Exception:  # pragma: no cover - only used when optional STG support is absent
    STGProbMetric = None


class TinyGumbelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layer1 = nn.Module()
        self.backbone.layer1.gumbel_layer = GumbelLayer(input_dim=3)


class TinySTGModel(nn.Module):
    def __init__(self):
        super().__init__()
        if STGChannelLayer is None:
            raise RuntimeError("STG support is not available in this tree.")
        self.backbone = nn.Module()
        self.backbone.layer1 = nn.Module()
        self.backbone.layer1.stg = STGChannelLayer(input_dim=3, sigma=1.0)


def test_gumbel_metric_logs_per_channel_zero_probabilities():
    model = TinyGumbelModel()
    metric = GumbelProbMetric()

    with torch.no_grad():
        model.backbone.layer1.gumbel_layer.logits.copy_(
            torch.tensor([[0.0, 2.0], [1.0, 0.0], [-1.0, 1.0]])
        )
    metric.update(None, None, None, model)

    with torch.no_grad():
        model.backbone.layer1.gumbel_layer.logits.zero_()
    metric.update(None, None, None, model)

    computed = metric.compute()

    first_probs = [
        math.exp(2.0) / (math.exp(0.0) + math.exp(2.0)),
        math.exp(0.0) / (math.exp(1.0) + math.exp(0.0)),
        math.exp(1.0) / (math.exp(-1.0) + math.exp(1.0)),
    ]
    mean_selection_probs = [(prob + 0.5) / 2 for prob in first_probs]
    mean_zero_probs = [1.0 - prob for prob in mean_selection_probs]

    assert computed["backbone.layer1.gumbel_layer.channel_000_zero_prob"] == pytest.approx(
        mean_zero_probs[0]
    )
    assert computed["backbone.layer1.gumbel_layer.channel_001_zero_prob"] == pytest.approx(
        mean_zero_probs[1]
    )
    assert computed["backbone.layer1.gumbel_layer.channel_002_zero_prob"] == pytest.approx(
        mean_zero_probs[2]
    )
    assert computed["backbone.layer1.gumbel_layer_avg_zero_prob"] == pytest.approx(
        sum(mean_zero_probs) / len(mean_zero_probs)
    )
    assert computed["average_zero_prob"] == pytest.approx(
        computed["backbone.layer1.gumbel_layer_avg_zero_prob"]
    )
    assert computed["backbone.layer1.gumbel_layer_avg_real_prob"] == pytest.approx(1 / 3)


@pytest.mark.skipif(
    STGChannelLayer is None or STGProbMetric is None,
    reason="STG support is not available in this tree.",
)
def test_stg_metric_logs_per_channel_zero_probabilities():
    model = TinySTGModel()
    metric = STGProbMetric()

    with torch.no_grad():
        model.backbone.layer1.stg.mu.copy_(torch.tensor([0.0, 1.0, -1.0]))
    metric.update(None, None, None, model)

    with torch.no_grad():
        model.backbone.layer1.stg.mu.zero_()
    metric.update(None, None, None, model)

    computed = metric.compute()

    def cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    first_probs = [cdf(0.0), cdf(1.0), cdf(-1.0)]
    mean_selection_probs = [(prob + 0.5) / 2 for prob in first_probs]
    mean_zero_probs = [1.0 - prob for prob in mean_selection_probs]

    assert computed["backbone.layer1.stg.channel_000_zero_prob"] == pytest.approx(mean_zero_probs[0])
    assert computed["backbone.layer1.stg.channel_001_zero_prob"] == pytest.approx(mean_zero_probs[1])
    assert computed["backbone.layer1.stg.channel_002_zero_prob"] == pytest.approx(mean_zero_probs[2])
    assert computed["backbone.layer1.stg_avg_zero_prob"] == pytest.approx(
        sum(mean_zero_probs) / len(mean_zero_probs)
    )


def test_run_history_writes_channel_zero_prob_columns(tmp_path):
    config = OmegaConf.create(
        {
            "run_history": {
                "root_dir": str(tmp_path),
                "run_name": "channel_zero_prob_test",
            }
        }
    )
    run_history = RunHistory(config)

    run_history.log_epoch(
        1,
        {
            "train_average_zero_prob": 0.25,
            "train_backbone.layer1.gumbel_layer.channel_000_zero_prob": 0.1,
        },
        {
            "valid_average_zero_prob": 0.5,
            "valid_backbone.layer1.gumbel_layer.channel_000_zero_prob": 0.2,
        },
    )

    with run_history.history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["train_backbone.layer1.gumbel_layer.channel_000_zero_prob"] == "0.1"
    assert rows[0]["valid_backbone.layer1.gumbel_layer.channel_000_zero_prob"] == "0.2"


def test_epoch_log_line_includes_zero_probability_summary():
    line = _build_epoch_log_line(
        epoch=3,
        total_epochs=10,
        train_metrics={"train_loss": 1.2, "train_average_zero_prob": 0.25},
        valid_metrics={
            "valid_loss": 0.9,
            "valid_accuracy": 0.8,
            "valid_average_zero_prob": 0.4,
        },
        train_time=1.5,
        valid_time=0.5,
        epoch_time=2.0,
    )

    assert "train_zero=0.2500" in line
    assert "val_zero=0.4000" in line
