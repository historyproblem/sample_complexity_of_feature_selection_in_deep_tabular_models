"""CPU contract tests: no dataset download or trained initializer required."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from net_complexity.metrics._channel_prob import survivor_channel_metrics
from net_complexity.metrics.gumbel import GumbelProbMetric
from net_complexity.models.channel_pruning import (
    apply_channel_mask, transfer_gated_weights_to_structural,
    transfer_structural_weights_to_gated,
)
from net_complexity.models.feature_selection import (
    GumbelLayer, MaskedGumbelLayer, MaskedGumbelBottleneckLayer,
    get_gate_normalization_metadata, validate_gate_normalization_metadata,
    get_gate_regularization_diagnostics, get_gumbel_loss, get_gumbel_modules,
    get_gumbel_posterior_regularization_terms,
    ClassificationFeatureSelectionWrapper,
)
from net_complexity.models.pruned_bottleneck import PrunedGumbelBottleneck
from net_complexity.models.pruning_budget import select_learned_closed, PhysicalBudget


def set_prob(gate, probabilities):
    values = torch.as_tensor(probabilities, dtype=gate.logits.dtype)
    with torch.no_grad():
        gate.logits[:, 0].zero_()
        gate.logits[:, 1].copy_(torch.logit(values))


def boundary_model():
    model = nn.Module()
    model.small = MaskedGumbelLayer(4, disabled_channels=[1], regularization_normalization="initial_channels").double()
    model.large = MaskedGumbelLayer(7, disabled_channels=[0, 2, 5, 6], regularization_normalization="initial_channels").double()
    set_prob(model.small, [.1, .2, .4, .9])
    set_prob(model.large, [.05, .3, .1, .8, .9, .1, .2])
    return model


def test_loss_and_gradient_equivalence_for_uneven_boundaries():
    model, alpha = boundary_model(), .001
    canonical = alpha * get_gumbel_loss(model)
    equivalent = 0
    for gate in get_gumbel_modules(model).values():
        mask = gate.channel_mask
        nt = mask.sum()
        raw = gate.logits.softmax(-1)[:, 1]
        equivalent += alpha * (nt / gate.initial_channels) * ((raw * mask).sum() / nt) / 2
    torch.testing.assert_close(canonical, equivalent)
    left = torch.autograd.grad(canonical, tuple(model.parameters()), retain_graph=True)
    right = torch.autograd.grad(equivalent, tuple(model.parameters()))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b)
    for gate, gradient in zip(get_gumbel_modules(model).values(), left):
        assert torch.count_nonzero(gradient[~gate.get_permanent_survivor_mask()]) == 0


def test_100_to_50_effective_lambda_without_second_decay():
    model = nn.Sequential(MaskedGumbelLayer(100, regularization_normalization="initial_channels").double())
    gate, alpha = model[0], .001
    set_prob(gate, [.7] * 100)
    before = torch.autograd.grad(alpha * get_gumbel_loss(model), gate.logits)[0]
    gate.channel_mask[:50] = 0
    after = torch.autograd.grad(alpha * get_gumbel_loss(model), gate.logits)[0]
    diagnostics = get_gate_regularization_diagnostics(model, alpha)
    assert diagnostics["alpha_base"] == alpha
    assert diagnostics["boundaries"]["0"]["lambda_effective"] == .0005
    assert diagnostics["boundaries"]["0"]["lambda_effective"] / 50 == alpha / 100
    assert diagnostics["actual_L_gate"] == pytest.approx(alpha * .7 * .5)
    torch.testing.assert_close(before[50:], after[50:])
    assert get_gate_regularization_diagnostics(model, alpha) == diagnostics


def test_empty_boundary_remains_in_M0_and_entropy_bypass():
    model = boundary_model()
    model.small.channel_mask.zero_()
    metadata = get_gate_normalization_metadata(model)
    assert metadata["M0"] == 2
    assert torch.isfinite(get_gumbel_loss(model))
    torch.testing.assert_close(get_gumbel_loss(model), model.large.regularization_loss() / 2)
    p, entropy = get_gumbel_posterior_regularization_terms(model)
    raw = model.large.logits.softmax(-1)
    expected_entropy = (raw * raw.log()).sum(-1) * model.large.channel_mask
    torch.testing.assert_close(entropy, expected_entropy.sum() / 7 / 2)
    for gate in get_gumbel_modules(model).values():
        gate.set_bypass(True)
    assert get_gumbel_loss(model).item() == 0
    assert all(value.item() == 0 for value in get_gumbel_posterior_regularization_terms(model))
    assert get_gumbel_loss(nn.Linear(2, 2)) == 0


def test_original_width_load_validation_and_legacy_migration():
    model = boundary_model()
    metadata = get_gate_normalization_metadata(model)
    target = boundary_model()
    target.load_state_dict(model.state_dict())
    validate_gate_normalization_metadata(target, metadata)
    bad = deepcopy(model.state_dict())
    bad["small.initial_channels_count"] = torch.tensor(3)
    with pytest.raises(RuntimeError, match="original channel width"):
        boundary_model().load_state_dict(bad)
    bad["small.initial_channels_count"] = torch.tensor(4.2)
    with pytest.raises(RuntimeError, match="original channel width"):
        boundary_model().load_state_dict(bad)
    legacy = {key: value for key, value in model.state_dict().items() if not key.endswith("initial_channels_count")}
    migrated = boundary_model()
    migrated.load_state_dict(legacy)
    assert migrated.small.normalization_metadata_status == "legacy_full_width_inferred"
    validate_gate_normalization_metadata(migrated, metadata)
    model.extra = MaskedGumbelLayer(2)
    with pytest.raises(ValueError, match="boundaries/widths"):
        get_gate_normalization_metadata(model)


def test_survivor_metrics_exclude_removed_mass_and_are_channel_weighted():
    model = boundary_model()
    metrics = survivor_channel_metrics(model)
    expected = ((1 - .1) + (1 - .4) + (1 - .9) + (1 - .3) + (1 - .8) + (1 - .9)) / 6
    assert metrics["surviving_channels"] == 6
    assert metrics["original_channels"] == 11
    assert metrics["survivor_zero_prob_raw"] == pytest.approx(expected)
    assert metrics["original_coordinate_zero_mass"] == pytest.approx(1 - 3.4 / 11)
    model.large.channel_mask[3] = 0  # removing a high-p member changes survivor composition
    after = survivor_channel_metrics(model)
    assert after["survivor_zero_prob_raw"] != pytest.approx(metrics["survivor_zero_prob_raw"])
    with torch.no_grad():
        model.large.logits[~model.large.get_permanent_survivor_mask()] = 999
    assert survivor_channel_metrics(model) == after
    model.small.channel_mask.zero_()
    model.large.channel_mask.zero_()
    empty = survivor_channel_metrics(model)
    assert empty["survivor_zero_prob_raw"] is None
    assert empty["survivor_status"] == "no_surviving_channels"
    assert empty["original_coordinate_zero_mass"] == 1


def test_same_retained_subset_before_after_removal_preserves_probabilities():
    model = boundary_model()
    retained = {"small": [0, 3], "large": [1, 4]}
    modules = get_gumbel_modules(model)
    before_raw = torch.cat([modules[name].get_raw_selection_probs()[ids] for name, ids in retained.items()])
    before_effective = torch.cat([modules[name].get_effective_selection_probs()[ids] for name, ids in retained.items()])
    whole_before = survivor_channel_metrics(model)["survivor_zero_prob_raw"]
    for name, ids in retained.items():
        mask = modules[name].channel_mask
        mask.zero_()
        mask[ids] = 1
    after = survivor_channel_metrics(model)
    assert after["surviving_channels"] == sum(map(len, retained.values()))
    assert after["survivor_zero_prob_raw"] == pytest.approx(float((1 - before_raw).mean()))
    assert after["survivor_zero_prob_effective"] == pytest.approx(float((1 - before_effective).mean()))
    assert after["survivor_zero_prob_raw"] != pytest.approx(whole_before)


def test_raw_effective_disagreement_threshold_ties_and_legacy_keys():
    model = nn.Sequential(MaskedGumbelLayer(4, gate_threshold=.5))
    gate = model[0]
    set_prob(gate, [.5, .2, .9, .1])
    gate.channel_mask[3] = 0
    gate.set_open_bias(3, p_min=.02, p_max=.5)
    assert gate.get_hard_gate_decisions(raw=True).tolist() == [False, False, True, False]
    assert gate.get_hard_gate_decisions().tolist() == [False, True, True, False]
    model.eval()
    actual = gate.compute_gates(torch.ones(1, 4)).flatten().bool()
    assert torch.equal(actual, gate.get_hard_gate_decisions())
    metrics = survivor_channel_metrics(model)
    assert metrics["survivor_zero_prob_raw"] > metrics["survivor_zero_prob_effective"]
    metric = GumbelProbMetric(log_channel_zero_probs=False)
    metric.update(None, None, None, model)
    logged = metric.compute()
    assert logged["average_zero_prob"] == pytest.approx(metrics["original_coordinate_zero_mass"])
    assert logged["survivor_hard_closed_fraction"] == pytest.approx(1 / 3)


class TinyTransferNet(nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 1)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 2)
        if mask is None:
            block = MaskedGumbelBottleneckLayer(16, 4, gate_internal_width=True, gate_output=False,
                       train_gate_mode="ste_hard", eval_gate_mode="deterministic_hard",
                       regularization_normalization="initial_channels")
        else:
            block = PrunedGumbelBottleneck(16, 4,
                disabled_mid1_channels=mask.get("layer1.0.mid1_gumbel_layer", []),
                disabled_mid2_channels=mask.get("layer1.0.mid2_gumbel_layer", []))
        self.layer1 = nn.Sequential(block)
        self.layer2, self.layer3, self.layer4 = nn.Sequential(), nn.Sequential(), nn.Sequential()

    def forward(self, x):
        return self.fc(self.layer1(torch.relu(self.batch_norm1(self.conv1(x)))).mean((2, 3)))


def tiny_open():
    model = TinyTransferNet().eval()
    for gate in get_gumbel_modules(model).values():
        set_prob(gate, [.9] * 4)
    return model


def test_actual_wrapper_zero_entropy_runtime_matches_canonical_loss_and_gradients():
    wrapper = ClassificationFeatureSelectionWrapper(
        TinyTransferNet().double(), lambda_coef=.001, bypass_on_zero_lambda=False,
        entropy_regularization="plus_negative_entropy", entropy_regularization_coef=0,
        regularization_loss=get_gumbel_loss,
    ).double().eval()
    first = wrapper.backbone.layer1[0].mid1_gumbel_layer
    second = wrapper.backbone.layer1[0].mid2_gumbel_layer
    set_prob(first, [.1, .4, .7, .9])
    set_prob(second, [.2, .3, .6, .8])
    first.channel_mask[[0, 2]] = 0
    second.channel_mask[1] = 0
    x, y = torch.randn(2, 3, 4, 4, dtype=torch.double), torch.tensor([0, 1])
    output = wrapper(x, y)
    canonical = wrapper.lambda_coef * get_gumbel_loss(wrapper.backbone)
    diagnostics = get_gate_regularization_diagnostics(wrapper, wrapper.lambda_coef)
    torch.testing.assert_close(output.reg_loss, canonical)
    torch.testing.assert_close(output.loss, output.ce_loss + canonical)
    assert diagnostics["actual_L_gate"] == pytest.approx(float(output.reg_loss.detach()))
    actual_grad = torch.autograd.grad(output.loss - output.ce_loss, (first.logits, second.logits), retain_graph=True)
    canonical_grad = torch.autograd.grad(canonical, (first.logits, second.logits))
    for actual, expected in zip(actual_grad, canonical_grad):
        torch.testing.assert_close(actual, expected)
    assert diagnostics["boundaries"]["backbone.layer1.0.mid1_gumbel_layer"]["lambda_effective"] == .0005
    assert wrapper.lambda_coef == .001
    # Crucially check the actual wrapper's posterior path freezes M0, too.
    wrapper.backbone.layer1[0].mid2_gumbel_layer = nn.Identity()
    with pytest.raises(ValueError, match="boundaries/widths"):
        wrapper(x, y)


def test_selector_all_open_is_noop_and_two_closed_only():
    model = tiny_open()
    mask, report = select_learned_closed(model, {}, .5)
    assert mask == {} and report["no_op_reason"] == "no_new_pruning"
    gate = model.layer1[0].mid1_gumbel_layer
    set_prob(gate, [.1, .2, .9, .9])
    snapshot = deepcopy(model.state_dict())
    mask, report = select_learned_closed(model, {}, .5)
    assert mask == {"layer1.0.mid1_gumbel_layer": [0, 1]}
    assert report["learned_candidates_materialized"] == 2
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, snapshot[name])
    apply_channel_mask(model, mask)
    assert select_learned_closed(model, mask, .5)[0] == mask
    assert select_learned_closed(model, mask, .5)[1]["eligible"] == []


def test_selector_blocks_bias_open_floors_dependencies_deterministically():
    model = tiny_open()
    gate = model.layer1[0].mid1_gumbel_layer
    set_prob(gate, [.1, .2, .3, .9])
    gate.set_open_bias(5)
    assert select_learned_closed(model, {}, .5)[0] == {}
    gate.set_open_bias(0)
    mask, report = select_learned_closed(model, {}, .5)
    assert mask["layer1.0.mid1_gumbel_layer"] == [0, 1]
    assert report["blocked"][0]["original_id"] == 2
    assert report["blocked"][0]["reason"] == "blocked_by_floor"
    group = [[("layer1.0.mid1_gumbel_layer", 0), ("layer1.0.mid2_gumbel_layer", 0)]]
    mask, report = select_learned_closed(model, {}, .5, dependency_groups=group)
    assert mask["layer1.0.mid1_gumbel_layer"] == [1, 2]
    assert report["blocked"][0]["reason"] == "dependency_contains_open_survivor"
    assert select_learned_closed(model, {}, .5, dependency_groups=group) == (mask, report)


@pytest.mark.parametrize("blocked", [False, True])
def test_real_tiny_transfer_cost_and_two_distinct_equivalences(blocked):
    torch.manual_seed(24)
    model = tiny_open()
    set_prob(model.layer1[0].mid1_gumbel_layer, [.1, .2, .3 if blocked else .9, .9])
    selected_logits = model(torch.randn(1, 3, 3, 3))  # first-forward gates already final
    metadata = get_gate_normalization_metadata(model)
    mask, report = select_learned_closed(model, {}, .5)
    physical = TinyTransferNet(mask).eval()
    transfer_gated_weights_to_structural(model, physical)
    assert report["params_after"] == sum(p.numel() for p in physical.parameters())
    assert report["params_after"] == PhysicalBudget(model, mask).total()
    inputs = torch.randn(3, 3, 5, 5)
    with torch.no_grad():
        gated_logits = model(inputs)
        physical_logits = physical(inputs)
    if blocked:
        assert len(report["blocked"]) == 1
        assert not torch.allclose(gated_logits, physical_logits, atol=1e-7, rtol=1e-7)
    else:
        torch.testing.assert_close(gated_logits, physical_logits, atol=1e-6, rtol=1e-5)
    apply_channel_mask(model, mask)
    for gate in get_gumbel_modules(model).values():
        gate.set_bypass(True)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), physical_logits, atol=1e-6, rtol=1e-5)
    logits_before = {name: gate.logits.clone() for name, gate in get_gumbel_modules(model).items()}
    with torch.no_grad():
        physical.layer1[0].batch_norm1.num_batches_tracked.add_(7)
        physical.layer1[0].conv1.weight.add_(.01)
    transfer_structural_weights_to_gated(physical, model)
    validate_gate_normalization_metadata(model, metadata)
    for name, gate in get_gumbel_modules(model).items():
        assert torch.equal(gate.logits, logits_before[name])
    assert model.layer1[0].batch_norm1.num_batches_tracked == 7
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), physical(inputs), atol=1e-6, rtol=1e-5)


def test_disabled_logits_have_no_task_or_regularizer_gradient():
    gate = MaskedGumbelLayer(3, disabled_channels=[1], train_gate_mode="ste_hard",
                             regularization_normalization="initial_channels")
    loss = gate(torch.ones(2, 3)).sum() + gate.regularization_loss()
    loss.backward()
    assert torch.count_nonzero(gate.logits.grad[1]) == 0
