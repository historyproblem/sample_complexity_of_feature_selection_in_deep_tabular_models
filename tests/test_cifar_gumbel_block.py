import torch
import torch.nn.functional as F

from net_complexity.models.feature_selection import (
    CIFARGumbelBasicBlock,
    CIFARSTGBasicBlock,
    GumbelLayer,
)


def _close_all_gates(block: CIFARGumbelBasicBlock) -> None:
    with torch.no_grad():
        block.gumbel_layer.logits[:, 0] = 1.0
        block.gumbel_layer.logits[:, 1] = 0.0


def _close_all_stg_gates(block: CIFARSTGBasicBlock) -> None:
    with torch.no_grad():
        block.stg_layer.mu.fill_(-1.0)


def test_closed_gumbel_gates_preserve_identity_shortcut():
    block = CIFARGumbelBasicBlock(16, 16, stride=1)
    block.eval()
    _close_all_gates(block)

    x = torch.randn(2, 16, 8, 8)

    with torch.no_grad():
        output = block(x)
        expected = F.relu(block.shortcut(x))

    torch.testing.assert_close(output, expected)


def test_closed_stg_gates_preserve_identity_shortcut():
    block = CIFARSTGBasicBlock(16, 16, stride=1)
    block.eval()
    _close_all_stg_gates(block)

    x = torch.randn(2, 16, 8, 8)

    with torch.no_grad():
        output = block(x)
        expected = F.relu(block.shortcut(x))

    torch.testing.assert_close(output, expected)


def test_closed_stg_gates_preserve_downsample_shortcut():
    block = CIFARSTGBasicBlock(16, 32, stride=2, option="A")
    block.eval()
    _close_all_stg_gates(block)

    x = torch.randn(2, 16, 8, 8)

    with torch.no_grad():
        output = block(x)
        expected = F.relu(block.shortcut(x))

    torch.testing.assert_close(output, expected)


def test_closed_gumbel_gates_preserve_downsample_shortcut():
    block = CIFARGumbelBasicBlock(16, 32, stride=2, option="A")
    block.eval()
    _close_all_gates(block)

    x = torch.randn(2, 16, 8, 8)

    with torch.no_grad():
        output = block(x)
        expected = F.relu(block.shortcut(x))

    torch.testing.assert_close(output, expected)


def test_gumbel_layer_beta_zero_makes_train_gates_deterministic():
    layer = GumbelLayer(input_dim=2, temperature=1.0, beta=0.0)
    layer.train()
    x = torch.randn(4, 2, 3, 3)

    with torch.no_grad():
        layer.logits.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

    torch.manual_seed(0)
    first = layer.compute_gates(x)
    torch.manual_seed(123)
    second = layer.compute_gates(x)

    expected = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand_as(first)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, expected)


def test_gumbel_layer_beta_scales_soft_sample_strength():
    logits = torch.zeros(1, 1, 2)
    layer_low_beta = GumbelLayer(input_dim=1, temperature=1.0, beta=0.25)
    layer_high_beta = GumbelLayer(input_dim=1, temperature=1.0, beta=4.0)

    torch.manual_seed(7)
    low_beta_sample = layer_low_beta._sample_gumbel_softmax(logits)
    torch.manual_seed(7)
    high_beta_sample = layer_high_beta._sample_gumbel_softmax(logits)

    low_beta_deviation = torch.abs(low_beta_sample[..., 1] - 0.5)
    high_beta_deviation = torch.abs(high_beta_sample[..., 1] - 0.5)

    assert torch.all(high_beta_deviation > low_beta_deviation)
