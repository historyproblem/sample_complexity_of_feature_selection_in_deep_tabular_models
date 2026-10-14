import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from net_complexity.evaluation.stochastic_depth import (
    evaluate_stochastic_depth,
    state_digest,
    summarize_stochastic_repeats,
)
from net_complexity.models.stochastic_depth import (
    HuangStochasticDepthInvertedResidual,
    StochasticDepthMobileNetV2,
    get_stochastic_depth_blocks,
)


class SmallClassifier(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.block = HuangStochasticDepthInvertedResidual(
            2, 2, 1, 1, survival_probability=probability
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.9)
        self.head = nn.Linear(2, 2)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.copy_(torch.tensor([1.0, 0.0]))

    def forward(self, x):
        return self.head(self.dropout(self.pool(self.block(x)).flatten(1)))


def test_stochastic_eval_really_skips_a_closed_branch_with_frozen_batchnorm():
    block = HuangStochasticDepthInvertedResidual(
        2, 2, 1, 6, survival_probability=0.0, eval_mode="stochastic"
    ).eval()
    calls = []
    handle = block.branch.register_forward_hook(lambda *_: calls.append(True))
    before = state_digest(block)
    x = torch.randn(3, 2, 8, 8)
    torch.testing.assert_close(block(x), x)
    handle.remove()
    assert calls == []
    assert not block.last_residual_branch_active
    assert state_digest(block) == before
    assert all(not module.training for module in block.modules())


def test_stochastic_eval_has_one_batch_mask_and_no_probability_scaling():
    block = HuangStochasticDepthInvertedResidual(
        2, 2, 1, 1, survival_probability=0.25, eval_mode="stochastic"
    ).eval()
    block.branch = nn.Identity()
    x = torch.ones(5, 2, 4, 4)
    torch.manual_seed(42)
    observed = set()
    for _ in range(32):
        output = block(x)
        assert block.last_survival_mask.shape == torch.Size([])
        torch.testing.assert_close(
            output, 2 * x if block.last_residual_branch_active else x
        )
        observed.add(block.last_residual_branch_active)
    assert observed == {True, False}


def test_factory_propagates_stochastic_mode_only_to_eligible_blocks():
    model = StochasticDepthMobileNetV2(num_classes=200, eval_mode="stochastic")
    assert len(get_stochastic_depth_blocks(model)) == 10
    assert all(
        block.eval_mode == "stochastic"
        for block in get_stochastic_depth_blocks(model).values()
    )
    with pytest.raises(ValueError, match="eval_mode"):
        StochasticDepthMobileNetV2(num_classes=200, eval_mode="invalid")


def test_frozen_pass_counts_samples_and_actual_skips_and_restores_modes():
    model = SmallClassifier(0.0).train()
    # Preserve a mixed incoming training state as well.
    model.head.eval()
    modes = [module.training for module in model.modules()]
    before = state_digest(model)
    observed_training = []
    handles = [
        module.register_forward_pre_hook(
            lambda module, _inputs: observed_training.append(module.training)
        )
        for module in model.modules()
        if isinstance(module, (nn.Dropout, nn.BatchNorm2d))
    ]
    loader = DataLoader(
        TensorDataset(torch.randn(9, 2, 8, 8), torch.tensor([0] * 8 + [1])),
        batch_size=4,
    )
    sampled = evaluate_stochastic_depth(model, loader, mode="stochastic")
    expected = evaluate_stochastic_depth(model, loader, mode="expected")
    for handle in handles:
        handle.remove()
    assert observed_training and not any(observed_training)
    assert sampled["accuracy"] == pytest.approx(8 / 9)
    assert sampled["correct_count"] == 8
    assert sampled["example_count"] == 9
    assert sampled["mean_active_blocks"] == 0
    assert sampled["mean_active_parameters"] < sampled["stored_parameters"]
    assert (
        sampled["mean_forward_flops_per_image"]
        < expected["mean_forward_flops_per_image"]
    )
    assert expected["mean_active_blocks"] == 1
    assert expected["mean_active_parameters"] == expected["stored_parameters"]
    assert model.block.eval_mode == "expected"
    assert [module.training for module in model.modules()] == modes
    assert state_digest(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_seeded_pass_is_reproducible_and_restores_rng_without_ensembling():
    model = SmallClassifier(0.5)
    loader = DataLoader(
        TensorDataset(torch.randn(45, 2, 8, 8), torch.zeros(45).long()), batch_size=4
    )
    torch.manual_seed(123)
    rng = torch.get_rng_state().clone()
    first = evaluate_stochastic_depth(model, loader, seed=42)
    assert torch.equal(torch.get_rng_state(), rng)
    second = evaluate_stochastic_depth(model, loader, seed=42)
    assert (
        first["branch_sample_execution_counts"]
        == second["branch_sample_execution_counts"]
    )
    assert first["mean_active_parameters"] == second["mean_active_parameters"]
    assert 0 < first["mean_active_blocks"] < 1
    assert first["example_count"] == len(loader.dataset)
    summary = summarize_stochastic_repeats([first, second])
    assert summary["repeats"] == 2
    assert summary["accuracy_mean"] == 1.0
    assert summary["accuracy_std"] == 0.0


def test_failed_evaluation_also_restores_modes_and_persistent_state():
    model = SmallClassifier(0.5).train()
    before = state_digest(model)
    empty = DataLoader(TensorDataset(torch.empty(0, 2, 8, 8), torch.empty(0).long()))
    with pytest.raises(ValueError, match="empty"):
        evaluate_stochastic_depth(model, empty)
    assert all(module.training for module in model.modules())
    assert model.block.eval_mode == "expected"
    assert state_digest(model) == before
