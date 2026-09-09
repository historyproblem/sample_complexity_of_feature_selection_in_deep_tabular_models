"""Fixed original-width normalization must not strengthen surviving gates."""

import io

import pytest
import torch
from torch import nn

from net_complexity.models.channel_pruning import apply_channel_mask
from net_complexity.models.feature_selection import (
    MaskedGumbelBottleneckLayer,
    MaskedGumbelLayer,
    get_gumbel_loss,
    get_gumbel_modules,
    get_gumbel_posterior_regularization_terms,
)


TERMS = ("regularization", "posterior_open", "posterior_entropy")


def _gate(width=100, mode="initial_channels"):
    gate = MaskedGumbelLayer(
        input_dim=width, regularization_normalization=mode,
    ).double()
    with torch.no_grad():
        gate.logits[:, 0].fill_(-0.25)
        gate.logits[:, 1].copy_(torch.linspace(0.1, 1.5, width, dtype=torch.float64))
    return gate


def _term(gate, name):
    if name == "regularization":
        return gate.regularization_loss()
    return gate.posterior_regularization_terms()[0 if name == "posterior_open" else 1]


@pytest.mark.parametrize("term", TERMS)
def test_original_width_and_historical_normalization_match_before_pruning(term):
    initial = _gate()
    historical = _gate(mode="enabled_channels")
    initial_loss = _term(initial, term)
    historical_loss = _term(historical, term)
    torch.testing.assert_close(initial_loss, historical_loss)
    torch.testing.assert_close(
        torch.autograd.grad(initial_loss, initial.logits)[0],
        torch.autograd.grad(historical_loss, historical.logits)[0],
    )


@pytest.mark.parametrize("term", TERMS)
def test_halving_enabled_channels_preserves_each_survivor_gradient(term):
    gate = _gate()
    before = torch.autograd.grad(2.048 * _term(gate, term), gate.logits)[0]
    with torch.no_grad():
        gate.channel_mask[:50].zero_()
    after = torch.autograd.grad(2.048 * _term(gate, term), gate.logits)[0]
    torch.testing.assert_close(after[50:], before[50:])
    assert torch.count_nonzero(after[:50]).item() == 0


@pytest.mark.parametrize("term", TERMS)
def test_historical_default_still_doubles_survivor_gradient_after_halving(term):
    gate = MaskedGumbelLayer(input_dim=100).double()
    assert gate.regularization_normalization == "enabled_channels"
    before = torch.autograd.grad(_term(gate, term), gate.logits)[0]
    with torch.no_grad():
        gate.channel_mask[:50].zero_()
    after = torch.autograd.grad(_term(gate, term), gate.logits)[0]
    torch.testing.assert_close(after[50:], 2.0 * before[50:])
    assert torch.count_nonzero(after[:50]).item() == 0


@pytest.mark.parametrize("term", TERMS)
def test_cumulative_masks_keep_original_width_denominator(term):
    gate = _gate()
    model = nn.ModuleDict({"gate": gate})
    before = torch.autograd.grad(_term(gate, term), gate.logits)[0]
    for disabled in (list(range(50)), list(range(50, 75))):
        apply_channel_mask(model, {"gate": disabled})
        after = torch.autograd.grad(_term(gate, term), gate.logits)[0]
        enabled = gate.channel_mask.bool()
        torch.testing.assert_close(after[enabled], before[enabled])
        assert torch.count_nonzero(after[~enabled]).item() == 0
    assert gate.logits.shape == (100, 2)
    assert int(gate.channel_mask.sum().item()) == 25


@pytest.mark.parametrize("posterior", [False, True])
def test_unequal_retention_is_normalized_per_layer_not_by_global_channel_count(posterior):
    model = nn.ModuleDict({"narrow": _gate(4), "wide": _gate(8)})

    def loss():
        if posterior:
            open_term, entropy_term = get_gumbel_posterior_regularization_terms(model)
            return 0.7 * open_term + 0.25 * entropy_term
        return 0.7 * get_gumbel_loss(model)

    logits = (model["narrow"].logits, model["wide"].logits)
    before = torch.autograd.grad(loss(), logits)
    apply_channel_mask(model, {"narrow": [0], "wide": list(range(6))})
    after = torch.autograd.grad(loss(), logits)
    for gate, expected, actual in zip(model.values(), before, after):
        enabled = gate.channel_mask.bool()
        torch.testing.assert_close(actual[enabled], expected[enabled])
        assert torch.count_nonzero(actual[~enabled]).item() == 0


def test_probability_and_entropy_terms_use_same_original_width_without_changing_metrics():
    gate = _gate(4)
    with torch.no_grad():
        gate.channel_mask[[0, 2]] = 0
    log_probs = torch.log_softmax(gate.logits, dim=-1)
    probs = log_probs.exp()
    open_term, entropy_term = gate.posterior_regularization_terms()
    torch.testing.assert_close(open_term, probs[[1, 3], 1].sum() / 4)
    torch.testing.assert_close(entropy_term, (probs * log_probs)[[1, 3]].sum() / 4)
    torch.testing.assert_close(gate.regularization_loss(), open_term)
    torch.testing.assert_close(gate.get_selection_probs(), probs[:, 1].detach() * gate.channel_mask)
    historical = _gate(4, mode="enabled_channels")
    historical.load_state_dict(gate.state_dict(), strict=True)
    torch.testing.assert_close(gate.get_selection_probs(), historical.get_selection_probs())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["initial_channels", "enabled_channels"])
def test_all_disabled_terms_are_finite_zeros_on_gate_dtype_and_device(dtype, mode):
    gate = _gate(4, mode=mode).to(dtype=dtype)
    with torch.no_grad():
        gate.channel_mask.zero_()
    for term in TERMS:
        value = _term(gate, term)
        assert value.dtype == dtype
        assert value.device == gate.logits.device
        assert value.shape == ()
        assert value.item() == 0.0
        if value.requires_grad:
            gradient = torch.autograd.grad(value, gate.logits)[0]
            assert torch.count_nonzero(gradient).item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_initial_normalization_bypass_has_no_gate_penalty(dtype):
    gate = _gate(4).to(dtype=dtype)
    gate.set_bypass(True)
    for term in TERMS:
        value = _term(gate, term)
        assert value.dtype == dtype
        assert value.device == gate.logits.device
        assert value.item() == 0.0


def test_historical_regularization_bypass_behavior_is_unchanged():
    gate = _gate(4, mode="enabled_channels")
    before = gate.regularization_loss()
    gate.set_bypass(True)
    torch.testing.assert_close(gate.regularization_loss(), before)
    assert all(term.item() == 0.0 for term in gate.posterior_regularization_terms())


def test_fixed_width_checkpoint_strict_roundtrip_has_no_new_state_keys():
    gate = _gate(10)
    with torch.no_grad():
        gate.channel_mask[:6].zero_()
    archive = io.BytesIO()
    torch.save(gate.state_dict(), archive)
    archive.seek(0)
    state = torch.load(archive, weights_only=True)
    restored = _gate(10)
    restored.load_state_dict(state, strict=True)
    assert restored.regularization_normalization == "initial_channels"
    for term in TERMS:
        torch.testing.assert_close(_term(restored, term), _term(gate, term))
    historical = _gate(10, mode="enabled_channels")
    assert set(state) == set(historical.state_dict())
    # The option belongs to resolved config, not the initializer tensor hash.
    historical.load_state_dict(state, strict=True)
    assert historical.regularization_normalization == "enabled_channels"
    for key, value in historical.state_dict().items():
        assert torch.equal(value, state[key])


def test_normalization_mode_does_not_change_initial_state():
    torch.manual_seed(42)
    historical = MaskedGumbelLayer(8)
    torch.manual_seed(42)
    initial = MaskedGumbelLayer(8, regularization_normalization="initial_channels")
    assert historical.state_dict().keys() == initial.state_dict().keys()
    for key, value in historical.state_dict().items():
        assert torch.equal(value, initial.state_dict()[key])


@pytest.mark.parametrize("gate_output", [False, True])
def test_actual_bottleneck_forwards_normalization_to_every_gate(gate_output):
    block = MaskedGumbelBottleneckLayer(
        in_channels=16, out_channels=4,
        gate_internal_width=True, gate_output=gate_output,
        regularization_normalization="initial_channels",
        disabled_mid1_channels=[0, 1], disabled_mid2_channels=[1],
        disabled_channels=[0, 1, 2] if gate_output else None,
    )
    gates = get_gumbel_modules(block)
    assert len(gates) == (3 if gate_output else 2)
    for gate in gates.values():
        assert gate.regularization_normalization == "initial_channels"
        expected = (torch.softmax(gate.logits, dim=1)[:, 1] * gate.channel_mask).sum()
        torch.testing.assert_close(gate.regularization_loss(), expected / gate.logits.shape[0])
    assert block(torch.randn(2, 16, 6, 6)).shape == (2, 16, 6, 6)


@pytest.mark.parametrize("mode", ["unknown", "parameters", None])
def test_invalid_normalization_is_rejected_even_when_block_has_no_gates(mode):
    with pytest.raises(ValueError, match="regularization_normalization"):
        MaskedGumbelLayer(4, regularization_normalization=mode)
    with pytest.raises(ValueError, match="regularization_normalization"):
        MaskedGumbelBottleneckLayer(
            in_channels=16, out_channels=4, gate_output=False,
            gate_internal_width=False, regularization_normalization=mode,
        )


def test_normalization_mode_is_read_only():
    gate = _gate(4)
    with pytest.raises(AttributeError):
        gate.regularization_normalization = "enabled_channels"
