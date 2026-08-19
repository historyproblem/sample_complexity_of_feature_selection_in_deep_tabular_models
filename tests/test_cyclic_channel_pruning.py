import pytest
from omegaconf import OmegaConf

from net_complexity.training.cyclic_channel_pruning import (
    _channels_to_drop,
    _channels_to_drop_by_param_budget,
    _extract_channel_probs,
    _merge_disabled_channels,
    _read_initial_disabled_channels,
    _set_disabled_channels_soft,
    _set_disabled_channels_structural,
)


def test_extract_channel_probs_parses_zero_prob_metric_keys():
    valid_metrics = {
        "valid_backbone.layer2.0.gumbel_layer.channel_000_zero_prob": 0.9,
        "valid_backbone.layer2.0.gumbel_layer.channel_001_zero_prob": 0.1,
        "valid_backbone.layer3.1.gumbel_layer.channel_000_zero_prob": 0.5,
        "valid_accuracy": 0.87,  # unrelated key, must be ignored
    }

    probs = _extract_channel_probs(valid_metrics)

    assert probs.keys() == {"backbone.layer2.0.gumbel_layer", "backbone.layer3.1.gumbel_layer"}
    assert probs["backbone.layer2.0.gumbel_layer"][0] == pytest.approx(0.1)
    assert probs["backbone.layer2.0.gumbel_layer"][1] == pytest.approx(0.9)
    assert probs["backbone.layer3.1.gumbel_layer"][0] == pytest.approx(0.5)


def test_channels_to_drop_below_threshold_and_not_already_disabled():
    channel_probs = {
        "backbone.layer2.0.gumbel_layer": {0: 0.05, 1: 0.5, 2: 0.02},
    }
    already_disabled = {"backbone.layer2.0.gumbel_layer": [2]}

    new_drop = _channels_to_drop(channel_probs, threshold=0.1, already_disabled=already_disabled)

    assert new_drop == {"backbone.layer2.0.gumbel_layer": [0]}


def test_channels_to_drop_keeps_at_least_one_active_channel():
    channel_probs = {
        "backbone.layer2.0.gumbel_layer": {0: 0.01, 1: 0.02, 2: 0.03},
    }

    new_drop = _channels_to_drop(channel_probs, threshold=0.5, already_disabled={})

    # All 3 channels are below threshold, but at least one must stay active —
    # the highest-probability candidate (channel 2) should survive.
    assert new_drop == {"backbone.layer2.0.gumbel_layer": [0, 1]}


def test_merge_disabled_channels_unions_and_sorts():
    base = {"backbone.layer2.0.gumbel_layer": [1, 3]}
    additional = {"backbone.layer2.0.gumbel_layer": [3, 5], "backbone.layer3.0.gumbel_layer": [0]}

    merged = _merge_disabled_channels(base, additional)

    assert merged == {
        "backbone.layer2.0.gumbel_layer": [1, 3, 5],
        "backbone.layer3.0.gumbel_layer": [0],
    }


def test_channels_to_drop_by_param_budget_respects_budget_and_min_active():
    class FakeConv:
        in_channels = 100
        bias = None

    class FakeBlock:
        conv3 = FakeConv()

    class FakeModel:
        def __init__(self):
            self.backbone = self

        class layer2(list):
            pass

    model = FakeModel()
    model.backbone = model
    setattr(model, "layer2", [FakeBlock()])

    channel_probs = {
        "backbone.layer2.0.gumbel_layer": {0: 0.01, 1: 0.02, 2: 0.9},
    }
    # cost per channel = 100 (in_channels) + 0 (no bias) + 2 (BN) = 102
    new_drop = _channels_to_drop_by_param_budget(
        model,
        channel_probs,
        already_disabled={},
        total_params=1000,
        max_param_fraction=0.15,  # budget = 150 -> fits exactly one channel (102)
    )

    assert new_drop == {"backbone.layer2.0.gumbel_layer": [0]}


def test_set_disabled_channels_soft_and_structural_toggle_structural_flag():
    cfg = OmegaConf.create({})
    _set_disabled_channels_soft(cfg, {"backbone.layer2.0.gumbel_layer": [0, 1]})
    assert cfg.channel_pruning.structural is False
    assert cfg.channel_pruning.mode == "explicit"

    cfg2 = OmegaConf.create({})
    _set_disabled_channels_structural(cfg2, {"backbone.layer2.0.gumbel_layer": [0, 1]})
    assert cfg2.channel_pruning.structural is True


def test_read_initial_disabled_channels_returns_empty_when_not_configured():
    assert _read_initial_disabled_channels(OmegaConf.create({})) == {}


def test_read_initial_disabled_channels_reads_explicit_mask():
    config = OmegaConf.create(
        {
            "channel_pruning": {
                "enabled": True,
                "mode": "explicit",
                "mask": {"backbone.layer2.0.gumbel_layer": [0, 2]},
            }
        }
    )
    assert _read_initial_disabled_channels(config) == {"backbone.layer2.0.gumbel_layer": [0, 2]}
