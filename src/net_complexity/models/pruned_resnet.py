"""Structural pruning of CIFARResNet residual blocks.

After a Gumbel training run identifies dispensable channels, these can be
physically removed from the residual branch of each BasicBlock.  The
block's input/output dimensionality stays the same so that inter-block
and inter-layer shapes are unchanged; disabled channels simply receive
their value from the shortcut (identity pass-through) instead of the
learned residual.

Compute savings vs the full block (n_active = planes - |disabled|):
  conv1 : n_active / planes          (fewer output filters)
  conv2 : (n_active / planes)^2      (fewer input AND output filters)
  bn1, bn2 : proportional to n_active
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cifar_resnet import LambdaLayer, _weights_init


class CIFARPrunedGumbelBasicBlock(nn.Module):
    """CIFAR BasicBlock whose residual branch is physically pruned.

    Channels in disabled_channels are completely excluded from conv/BN
    computation.  Their output values come from the shortcut connection
    only — the same value that would have been passed through the identity
    (or zero-padded) skip connection in the original block.

    Because input and output retain the full `planes` dimensionality, this
    block is a drop-in replacement for CIFARBasicBlock inside CIFARResNet.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        option: str = "A",
        disabled_channels: list[int] | None = None,
    ):
        super().__init__()

        disabled = set(disabled_channels or [])
        active = [ch for ch in range(planes) if ch not in disabled]
        if not active:
            raise ValueError(
                f"All {planes} channels are disabled — the residual branch "
                "would be empty. Provide at least one active channel."
            )
        n_active = len(active)
        self.planes = planes
        self.n_active = n_active

        # Pruned residual branch — operates on n_active channels only
        self.conv1 = nn.Conv2d(
            in_planes, n_active, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(n_active)
        self.conv2 = nn.Conv2d(
            n_active, n_active, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(n_active)

        # Shortcut — same logic as CIFARBasicBlock (option A or B)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            if option == "A":
                pad_channels = planes - in_planes
                pad_front = pad_channels // 2
                pad_back = pad_channels - pad_front
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::stride, ::stride],
                        (0, 0, 0, 0, pad_front, pad_back),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            else:
                raise ValueError("option must be 'A' or 'B'.")

        # active_selection[ch, j] = 1 iff active[j] == ch
        # Used to scatter [B, n_active, H, W] → [B, planes, H, W] via einsum.
        # Registered as buffer so it moves with .to(device) and is
        # saved in checkpoints, but is not a trainable parameter.
        active_sel = torch.zeros(planes, n_active)
        for j, ch in enumerate(active):
            active_sel[ch, j] = 1.0
        self.register_buffer("active_selection", active_sel)  # [planes, n_active]

        # Human-readable list of active channel indices (for reporting)
        self.register_buffer(
            "active_indices", torch.tensor(active, dtype=torch.long)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual branch: compute only for the n_active channels
        res = F.relu(self.bn1(self.conv1(x)))  # [B, n_active, H, W]
        res = self.bn2(self.conv2(res))         # [B, n_active, H, W]

        # Scatter residual back to full planes dimension.
        # einsum is autograd-compatible and device-agnostic.
        # active_selection: [planes, n_active], res: [B, n_active, H, W]
        # → full_res: [B, planes, H, W]
        full_res = torch.einsum("pn,bnhw->bphw", self.active_selection, res)

        return F.relu(self.shortcut(x) + full_res)


class PrunedCIFARResNet(nn.Module):
    """CIFARResNet with per-block structural channel pruning.

    pruning_spec maps "layerN.B" keys to lists of disabled channel indices,
    e.g. {"layer2.0": [14, 23, 11, 10], "layer2.1": [14]}.
    Blocks absent from pruning_spec are built with all channels active.
    """

    def __init__(
        self,
        pruning_spec: dict[str, list[int]],
        num_blocks: list[int] | None = None,
        num_classes: int = 10,
        in_channels: int = 3,
        shortcut_option: str = "A",
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [3, 3, 3]

        self.in_planes = 16
        self.shortcut_option = shortcut_option

        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(pruning_spec, "layer1", 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(pruning_spec, "layer2", 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(pruning_spec, "layer3", 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(
        self,
        pruning_spec: dict[str, list[int]],
        layer_name: str,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for block_idx, s in enumerate(strides):
            key = f"{layer_name}.{block_idx}"
            disabled = pruning_spec.get(key, [])
            layers.append(
                CIFARPrunedGumbelBasicBlock(
                    self.in_planes,
                    planes,
                    stride=s,
                    option=self.shortcut_option,
                    disabled_channels=disabled,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        return self.linear(out.view(out.size(0), -1))
