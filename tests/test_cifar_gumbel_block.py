import torch
import torch.nn.functional as F

from net_complexity.models.feature_selection import CIFARGumbelBasicBlock


def _close_all_gates(block: CIFARGumbelBasicBlock) -> None:
    with torch.no_grad():
        block.gumbel_layer.logits[:, 0] = 1.0
        block.gumbel_layer.logits[:, 1] = 0.0


def test_closed_gumbel_gates_preserve_identity_shortcut():
    block = CIFARGumbelBasicBlock(16, 16, stride=1)
    block.eval()
    _close_all_gates(block)

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
