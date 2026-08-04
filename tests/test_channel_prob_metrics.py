import csv
import gzip
import json
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


class TinyTwoLayerGumbelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.small = GumbelLayer(input_dim=2)
        self.backbone.large = GumbelLayer(input_dim=4)


class TinySTGModel(nn.Module):
    def __init__(self):
        super().__init__()
        if STGChannelLayer is None:
            raise RuntimeError("STG support is not available in this tree.")
        self.backbone = nn.Module()
        self.backbone.layer1 = nn.Module()
        self.backbone.layer1.stg = STGChannelLayer(input_dim=3, sigma=1.0)


class TinyGumbelBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gumbel_layer = GumbelLayer(input_dim=3)


class TinyResNet20LikeGumbelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layer1 = nn.Sequential(TinyGumbelBlock())


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


def test_gumbel_metric_weights_global_channel_ratios_by_layer_size():
    model = TinyTwoLayerGumbelModel()
    metric = GumbelProbMetric(log_channel_zero_probs=False)

    with torch.no_grad():
        model.backbone.small.logits.copy_(
            torch.tensor([[10.0, -10.0], [10.0, -10.0]])
        )
        model.backbone.large.logits.copy_(
            torch.tensor([[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]])
        )

    metric.update(None, None, None, model)
    computed = metric.compute()

    assert computed["backbone.small_num_channels"] == 2
    assert computed["backbone.large_num_channels"] == 4
    assert computed["average_layer_zero_prob"] == pytest.approx(0.5)
    assert computed["average_zero_prob"] == pytest.approx(2.0 / 6.0)
    assert computed["total_channels"] == 6
    assert computed["estim_zero_channels"] == pytest.approx(2.0)
    assert computed["estim_active_channels"] == pytest.approx(4.0)
    assert computed["average_zero_prob"] * computed["total_channels"] == pytest.approx(
        computed["estim_zero_channels"]
    )
    assert (1.0 - computed["average_zero_prob"]) * computed["total_channels"] == pytest.approx(
        computed["estim_active_channels"]
    )


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


def test_run_history_splits_scalar_history_and_channel_history(tmp_path):
    model = TinyResNet20LikeGumbelModel()
    with torch.no_grad():
        model.backbone.layer1[0].gumbel_layer.logits.copy_(
            torch.tensor([[0.0, 2.0], [1.0, 0.0], [-1.0, 1.0]])
        )

    config = OmegaConf.create(
        {
            "run_history": {
                "root_dir": str(tmp_path),
                "run_name": "channel_zero_prob_test",
                "monitor": "valid_loss",
                "mode": "min",
                "log_channel_history": True,
            },
            "model": {
                "backbone": {
                    "_target_": "net_complexity.wrappers.CIFARResNet20",
                    "resnet_block": {
                        "_target_": "net_complexity.wrappers.CIFARGumbelBasicBlock",
                    },
                },
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
            "valid_loss": 0.9,
        },
    )
    run_history.log_channel_history(1, model)
    run_history.update_last_epoch({
        "train_time_sec": 1.5,
        "valid_time_sec": 0.5,
        "epoch_time_sec": 2.1,
    })
    assert run_history.should_update_best(1, {"valid_average_zero_prob": 0.5, "valid_loss": 0.9}) is True
    run_history.save_summary(
        final_train_metrics={"train_average_zero_prob": 0.25, "train_loss": 1.2},
        final_valid_metrics={"valid_average_zero_prob": 0.5, "valid_loss": 0.9},
        test_metrics={"test_accuracy": 0.8},
    )

    with run_history.history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert rows[0]["epoch_pass"] == "0"
    assert rows[0]["epoch_label"] == "epoch_001_pass_00"
    assert rows[0]["train_average_zero_prob"] == "0.25"
    assert rows[0]["valid_average_zero_prob"] == "0.5"
    assert rows[0]["epoch_time_sec"] == "2.1"
    assert all(".channel_" not in key for key in rows[0])

    with gzip.open(run_history.channel_history_path, "rt", newline="", encoding="utf-8") as handle:
        channel_rows = list(csv.DictReader(handle))

    assert len(channel_rows) == 3
    assert channel_rows[0]["layer_name"] == "backbone.layer1.0.gumbel_layer"
    assert channel_rows[0]["stage_name"] == "layer1"
    assert channel_rows[0]["block_index"] == "0"
    assert channel_rows[0]["channel_index"] == "0"
    assert channel_rows[0]["selection_prob"] != ""
    assert channel_rows[0]["zero_prob"] != ""
    assert channel_rows[0]["logit_margin"] != ""
    assert channel_rows[0]["beta"] == "1.0"
    assert channel_rows[0]["mu"] == ""

    summary = json.loads(run_history.summary_path.read_text(encoding="utf-8"))
    assert summary["identity"]["run_id"] == run_history.run_id
    assert summary["final_valid"]["valid_loss"] == 0.9
    assert summary["best_valid"]["metric"] == "valid_loss"
    assert summary["best_valid"]["value"] == 0.9
    assert summary["artifacts"]["history"] == "history.csv"
    assert summary["artifacts"]["channel_history"] == "channel_history.csv.gz"
    assert summary["timing"]["num_epochs_executed"] == 1
    assert summary["timing"]["full_train_time_sec"] == 2.1
    assert summary["timing"]["train_forward_backward_time_sec"] == 1.5
    assert summary["timing"]["validation_time_sec"] == 0.5


def test_run_history_assigns_unique_epoch_labels_to_replayed_epochs(tmp_path):
    config = OmegaConf.create(
        {
            "run_history": {
                "root_dir": str(tmp_path),
                "run_name": "epoch_label_test",
            },
        }
    )
    run_history = RunHistory(config)

    run_history.log_epoch(3, {"train_loss": 1.0}, {"valid_loss": 0.9})
    run_history.log_epoch(3, {"train_loss": 0.8}, {"valid_loss": 0.7})

    with run_history.history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["epoch"] == "3"
    assert rows[0]["epoch_pass"] == "0"
    assert rows[0]["epoch_label"] == "epoch_003_pass_00"
    assert rows[1]["epoch"] == "3"
    assert rows[1]["epoch_pass"] == "1"
    assert rows[1]["epoch_label"] == "epoch_003_pass_01"


def test_run_history_logs_gumbel_gate_history_as_jsonl_without_duplicates(tmp_path):
    model = TinyResNet20LikeGumbelModel()
    with torch.no_grad():
        model.backbone.layer1[0].gumbel_layer.logits.copy_(
            torch.tensor([[0.0, 2.0], [1.0, 0.0], [-1.0, 1.0]])
        )

    config = OmegaConf.create(
        {
            "run_history": {
                "root_dir": str(tmp_path),
                "run_name": "gate_history_test",
                "log_gate_history": True,
            },
        }
    )
    run_history = RunHistory(config)

    assert run_history.log_gate_history(1, "valid", model) == 1
    assert run_history.log_gate_history(1, "valid", model) == 0

    with gzip.open(run_history.gate_history_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    assert len(rows) == 1
    assert rows[0]["epoch"] == 1
    assert rows[0]["split"] == "valid"
    assert rows[0]["beta"] == 1.0
    assert rows[0]["layer_name"] == "backbone.layer1.0.gumbel_layer"
    assert rows[0]["num_channels"] == 3
    assert rows[0]["polarized_active_mask"] == [1, 0, 1]
    assert rows[0]["polarized_active_count"] == 2
    assert len(rows[0]["selection_probs"]) == 3
    assert len(rows[0]["logit_margin"]) == 3

    run_history.save_summary(test_metrics={"test_accuracy": 0.7})
    summary = json.loads(run_history.summary_path.read_text(encoding="utf-8"))
    assert summary["artifacts"]["gate_history"] == "gate_history.jsonl.gz"


def test_epoch_log_line_includes_zero_probability_summary():
    line = _build_epoch_log_line(
        epoch=3,
        total_epochs=10,
        train_metrics={
            "train_loss": 1.2,
            "train_average_zero_prob": 0.25,
            "train_estim_active_channels": 7.5,
            "train_total_channels": 10,
        },
        valid_metrics={
            "valid_loss": 0.9,
            "valid_accuracy": 0.8,
            "valid_average_zero_prob": 0.4,
            "valid_estim_active_channels": 6,
            "valid_total_channels": 10,
        },
        train_time=1.5,
        valid_time=0.5,
        epoch_time=2.0,
    )

    assert "train_zero=0.2500" in line
    assert "train_open=7.50/10" in line
    assert "val_zero=0.4000" in line
    assert "val_open=6/10" in line


def test_epoch_log_line_includes_recovery_state():
    line = _build_epoch_log_line(
        epoch=82,
        total_epochs=200,
        train_metrics={"train_loss": 1.2},
        valid_metrics={"valid_loss": 0.9, "valid_accuracy": 0.78},
        train_time=1.5,
        valid_time=0.5,
        epoch_time=2.0,
        extra_metrics={
            "adaptive_lambda_action": "recovery_blocked_lambda_increase",
            "recovery_active": True,
            "recovery_action": "continue_recovery",
            "recovery_epochs_left": 4,
            "recovery_open_bias": 0.135,
            "recovery_attempts": 1,
        },
    )

    assert "lambda_action=recovery_blocked_lambda_increase" in line
    assert "recovery=active" in line
    assert "recovery_action=continue_recovery" in line
    assert "recovery_left=4" in line
    assert "recovery_bias=0.1350" in line
    assert "recovery_attempts=1" in line
