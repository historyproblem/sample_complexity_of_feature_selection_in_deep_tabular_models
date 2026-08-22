from net_complexity.training.run_history import _filter_channel_metrics


def test_filter_channel_metrics_drops_only_per_channel_keys():
    metrics = {
        "valid_accuracy": 0.9,
        "valid_g_prob_backbone.layer2.0.gumbel_layer.channel_000_zero_prob": 0.1,
        "valid_g_prob_backbone.layer2.0.gumbel_layer.channel_001_zero_prob": 0.9,
        "valid_backbone.layer2.0.gumbel_layer_avg_zero_prob": 0.5,
        "recalibrated_backbone.layer3.1.gumbel_layer.channel_012_zero_prob": 0.2,
    }

    filtered = _filter_channel_metrics(metrics)

    assert filtered == {
        "valid_accuracy": 0.9,
        "valid_backbone.layer2.0.gumbel_layer_avg_zero_prob": 0.5,
    }
