from net_complexity.training.cyclic_aig import _cap_layers_to_drop_by_param_budget


def test_cap_layers_to_drop_by_param_budget_skips_expensive_picks_cheaper():
    # Simulates a threshold pass that already selected these layers
    # (e.g. g_prob_threshold=0.5); the cap must further restrict them.
    layers_to_drop = ["layer2.0", "layer3.0", "layer3.1"]
    layer_probs = {"layer2.0": 0.01, "layer3.0": 0.02, "layer3.1": 0.5}
    freeable_counts = {"layer2.0": 102, "layer3.0": 22, "layer3.1": 22}

    capped = _cap_layers_to_drop_by_param_budget(
        layers_to_drop, layer_probs, freeable_counts, total_params=1000, max_param_fraction=0.05,
    )

    # budget = 50. Ascending g_prob order: layer2.0 (0.01, cost 102) is
    # checked first but doesn't fit -> skipped (not a stopping condition).
    # layer3.0 (0.02, cost 22) fits -> selected, remaining 28.
    # layer3.1 (0.5, cost 22) fits -> selected, remaining 6.
    assert capped == ["layer3.0", "layer3.1"]


def test_cap_layers_to_drop_by_param_budget_keeps_everything_when_budget_is_sufficient():
    layers_to_drop = ["layer2.0", "layer3.0"]
    layer_probs = {"layer2.0": 0.01, "layer3.0": 0.02}
    freeable_counts = {"layer2.0": 10, "layer3.0": 10}

    capped = _cap_layers_to_drop_by_param_budget(
        layers_to_drop, layer_probs, freeable_counts, total_params=1000, max_param_fraction=0.5,
    )

    assert capped == ["layer2.0", "layer3.0"]


def test_cap_layers_to_drop_by_param_budget_ignores_layers_missing_from_freeable_counts():
    layers_to_drop = ["layer2.0", "layer3.0"]
    layer_probs = {"layer2.0": 0.01, "layer3.0": 0.02}
    freeable_counts = {"layer3.0": 10}  # layer2.0 has no known freeable cost

    capped = _cap_layers_to_drop_by_param_budget(
        layers_to_drop, layer_probs, freeable_counts, total_params=1000, max_param_fraction=0.5,
    )

    assert capped == ["layer3.0"]
