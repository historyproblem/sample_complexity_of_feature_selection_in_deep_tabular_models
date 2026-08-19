"""Structural channel pruning of Bottleneck residual blocks (ResNet50/101/152).

Bottleneck analogue of ``pruned_resnet.py``'s ``CIFARPrunedGumbelBasicBlock`` /
``PrunedCIFARResNet``: once the iterative channel-pruning search (see
``training.cyclic_channel_pruning``) has identified which output channels of
each block's residual branch are dispensable, they are physically removed
from ``conv3``/``batch_norm3`` here. Block input/output dimensionality is
unchanged (pruned channels are scattered back to zero via the shortcut only),
so this is a drop-in replacement for ``Bottleneck`` inside ``ResNet``.

Compute savings vs. the full block (n_active = out_channels*expansion - |disabled|):
  conv3, batch_norm3: proportional to n_active (fewer output filters)
  conv1, conv2 and their batch norms are unaffected — the internal bottleneck
  width is untouched, only the block's residual *output* is narrowed. This
  mirrors ``GumbelBottleneckLayer``/``MaskedGumbelBottleneckLayer``, which
  gate the same tensor (the output of conv3/batch_norm3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunedGumbelBottleneck(nn.Module):
    """Bottleneck whose residual branch output is physically narrowed to n_active channels.

    Channels in ``disabled_channels`` (0-based indices into the block's full
    ``out_channels * expansion`` output) are completely excluded from
    ``conv3``/``batch_norm3`` computation. Their contribution to the block
    output comes from the shortcut only, matching what a fully-closed
    ``GumbelBottleneckLayer`` gate would produce for that channel.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        i_downsample: nn.Module | None = None,
        stride: int = 1,
        disabled_channels: list[int] | None = None,
    ):
        super().__init__()
        full_channels = out_channels * self.expansion
        disabled = set(disabled_channels or [])
        active = [ch for ch in range(full_channels) if ch not in disabled]
        if not active:
            raise ValueError(
                f"All {full_channels} channels are disabled for this block — "
                "the residual branch would be empty. Provide at least one "
                "active channel, or drop the whole block via layer_skipping instead."
            )
        n_active = len(active)
        self.out_channels = out_channels
        self.n_active = n_active

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, n_active, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(n_active)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

        # active_selection[ch, j] = 1 iff active[j] == ch — scatters the
        # narrowed [B, n_active, H, W] residual back to [B, full_channels, H, W].
        active_sel = torch.zeros(full_channels, n_active)
        for j, ch in enumerate(active):
            active_sel[ch, j] = 1.0
        self.register_buffer("active_selection", active_sel)
        self.register_buffer("active_indices", torch.tensor(active, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.relu(self.batch_norm2(self.conv2(out)))
        out = self.batch_norm3(self.conv3(out))  # [B, n_active, H, W]

        full_out = torch.einsum("pn,bnhw->bphw", self.active_selection, out)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        return self.relu(identity + full_out)


class PrunedResNet(nn.Module):
    """Bottleneck-based ResNet (ResNet50/101/152 topology) with per-block channel pruning.

    ``pruning_spec`` maps ``"layerN.B"`` keys (1-based stage index, 0-based
    block index — matching ``layer_skipping``'s convention) to lists of
    disabled *original* channel indices (``0 .. out_channels*expansion-1``) of
    that block's residual branch output. Blocks absent from ``pruning_spec``
    keep all channels active. Mirrors ``net_complexity.models.resnet.ResNet``'s
    stem/stage layout so it is a drop-in structural-pruning replacement for a
    plain or Gumbel-gated ``ResNet50``/``ResNet101``/``ResNet152``.
    """

    def __init__(
        self,
        pruning_spec: dict[str, list[int]],
        layer_list: list[int],
        num_classes: int,
        in_channels: int = 3,
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        stem_padding: int = 3,
        use_maxpool: bool = True,
    ):
        super().__init__()
        self.in_channels = 64
        self.pruning_spec = dict(pruning_spec)

        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            padding=stem_padding,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = (
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if use_maxpool else nn.Identity()
        )

        self.layer1 = self._make_layer("layer1", layer_list[0], planes=64, stride=1)
        self.layer2 = self._make_layer("layer2", layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer("layer3", layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer("layer4", layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * PrunedGumbelBottleneck.expansion, num_classes)

    def _make_layer(self, layer_name: str, blocks: int, planes: int, stride: int) -> nn.Sequential:
        expansion = PrunedGumbelBottleneck.expansion
        layers: list[nn.Module] = []
        i_downsample = None
        if stride != 1 or self.in_channels != planes * expansion:
            i_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * expansion),
            )

        layers.append(
            PrunedGumbelBottleneck(
                self.in_channels,
                planes,
                i_downsample=i_downsample,
                stride=stride,
                disabled_channels=self.pruning_spec.get(f"{layer_name}.0", []),
            )
        )
        self.in_channels = planes * expansion

        for block_idx in range(1, blocks):
            layers.append(
                PrunedGumbelBottleneck(
                    self.in_channels,
                    planes,
                    disabled_channels=self.pruning_spec.get(f"{layer_name}.{block_idx}", []),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)
