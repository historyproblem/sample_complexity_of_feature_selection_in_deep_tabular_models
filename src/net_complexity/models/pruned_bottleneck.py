"""Structural channel pruning of Bottleneck residual blocks (ResNet50/101/152).

Bottleneck analogue of ``pruned_resnet.py``'s ``CIFARPrunedGumbelBasicBlock`` /
``PrunedCIFARResNet``: once the iterative channel-pruning search (see
``training.cyclic_channel_pruning``) has identified which channels of each
block are dispensable, they are physically removed here. Three independent
boundaries can be pruned, matching ``MaskedGumbelBottleneckLayer``'s three
gates:

* ``disabled_channels`` ("output" gate) — conv3/batch_norm3's *output*.
  Removed channels are scattered back to zero via the shortcut only, so
  block input/output dimensionality (``out_channels * expansion``) is
  unchanged — a drop-in replacement for ``Bottleneck`` inside ``ResNet``.
* ``disabled_mid1_channels`` ("mid1" gate) — conv1's output / conv2's input
  boundary. No scatter-back: these channels are simply absent, narrowing
  conv1's output and conv2's input width directly.
* ``disabled_mid2_channels`` ("mid2" gate) — conv2's output / conv3's input
  boundary. Same idea, narrowing conv2's output and conv3's input width.

Compute savings vs. the full block (default: no channels disabled anywhere):
  conv3, batch_norm3: proportional to n_active (fewer output filters)
  conv2, batch_norm2: proportional to w1 (input) and w2 (output) width
  conv1, batch_norm1: proportional to w1 (output filters)
  With no ``mid1``/``mid2`` channels disabled, conv1/conv2 stay at full
  width — only the block's residual *output* is narrowed, exactly as
  before this module gained the mid1/mid2 boundaries.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunedGumbelBottleneck(nn.Module):
    """Bottleneck whose residual branch is physically narrowed at up to three boundaries.

    Channels in ``disabled_channels`` (0-based indices into the block's full
    ``out_channels * expansion`` output) are completely excluded from
    ``conv3``/``batch_norm3`` computation. Their contribution to the block
    output comes from the shortcut only, matching what a fully-closed
    ``GumbelBottleneckLayer``/``MaskedGumbelBottleneckLayer`` "output" gate
    would produce for that channel.

    ``disabled_mid1_channels``/``disabled_mid2_channels`` (0-based indices
    into ``[0, out_channels)``, independent of each other and of
    ``disabled_channels`` — see module docstring) additionally narrow conv1's
    output/conv2's input (mid1) and conv2's output/conv3's input (mid2). No
    scatter-back is needed for these: unlike the output boundary, they never
    reach the shortcut sum, so removed channels are simply absent from the
    narrowed tensor that flows into the next conv.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        i_downsample: nn.Module | None = None,
        stride: int = 1,
        disabled_channels: list[int] | None = None,
        disabled_mid1_channels: list[int] | None = None,
        disabled_mid2_channels: list[int] | None = None,
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

        mid1_disabled = set(disabled_mid1_channels or [])
        mid1_active = [ch for ch in range(out_channels) if ch not in mid1_disabled]
        if not mid1_active:
            raise ValueError(
                f"All {out_channels} conv1-output/conv2-input channels are disabled "
                "for this block — conv2 would receive no input. Provide at least "
                "one active channel."
            )
        w1 = len(mid1_active)

        mid2_disabled = set(disabled_mid2_channels or [])
        mid2_active = [ch for ch in range(out_channels) if ch not in mid2_disabled]
        if not mid2_active:
            raise ValueError(
                f"All {out_channels} conv2-output/conv3-input channels are disabled "
                "for this block — conv3 would receive no input. Provide at least "
                "one active channel."
            )
        w2 = len(mid2_active)

        self.out_channels = out_channels
        self.n_active = n_active
        self.w1 = w1
        self.w2 = w2

        self.conv1 = nn.Conv2d(in_channels, w1, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(w1)
        self.conv2 = nn.Conv2d(w1, w2, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(w2)
        self.conv3 = nn.Conv2d(w2, n_active, kernel_size=1, stride=1, padding=0)
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
        self.register_buffer("mid1_active_indices", torch.tensor(mid1_active, dtype=torch.long))
        self.register_buffer("mid2_active_indices", torch.tensor(mid2_active, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.batch_norm1(self.conv1(x)))       # [B, w1, H, W]
        out = self.relu(self.batch_norm2(self.conv2(out)))     # [B, w2, H, W]
        out = self.batch_norm3(self.conv3(out))                 # [B, n_active, H, W]

        full_out = torch.einsum("pn,bnhw->bphw", self.active_selection, out)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        return self.relu(identity + full_out)


class PrunedResNet(nn.Module):
    """Bottleneck-based ResNet (ResNet50/101/152 topology) with per-block channel pruning.

    ``pruning_spec`` maps ``"layerN.B"`` keys (1-based stage index, 0-based
    block index — matching ``layer_skipping``'s convention) to that block's
    disabled channels. Each value is either:

      * a plain list of ints — legacy shorthand for the "output" gate only
        (``disabled_channels``, indices into ``0 .. out_channels*expansion-1``),
        matching the pre-mid1/mid2 pruning_spec format; or
      * a dict with up to three keys, ``{"output": [...], "mid1": [...],
        "mid2": [...]}`` (all optional, default ``[]``) — see
        ``PrunedGumbelBottleneck`` for what each boundary prunes.

    Blocks absent from ``pruning_spec`` keep all channels active. Mirrors
    ``net_complexity.models.resnet.ResNet``'s stem/stage layout so it is a
    drop-in structural-pruning replacement for a plain or Gumbel-gated
    ``ResNet50``/``ResNet101``/``ResNet152``.
    """

    def __init__(
        self,
        pruning_spec: dict[str, list[int] | dict[str, list[int]]],
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

    def _spec_for_block(self, key: str) -> tuple[list[int], list[int], list[int]]:
        raw = self.pruning_spec.get(key, [])
        if isinstance(raw, dict):
            return (
                list(raw.get("output", [])),
                list(raw.get("mid1", [])),
                list(raw.get("mid2", [])),
            )
        return list(raw), [], []

    def _make_layer(self, layer_name: str, blocks: int, planes: int, stride: int) -> nn.Sequential:
        expansion = PrunedGumbelBottleneck.expansion
        layers: list[nn.Module] = []
        i_downsample = None
        if stride != 1 or self.in_channels != planes * expansion:
            i_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * expansion),
            )

        output_ch, mid1_ch, mid2_ch = self._spec_for_block(f"{layer_name}.0")
        layers.append(
            PrunedGumbelBottleneck(
                self.in_channels,
                planes,
                i_downsample=i_downsample,
                stride=stride,
                disabled_channels=output_ch,
                disabled_mid1_channels=mid1_ch,
                disabled_mid2_channels=mid2_ch,
            )
        )
        self.in_channels = planes * expansion

        for block_idx in range(1, blocks):
            output_ch, mid1_ch, mid2_ch = self._spec_for_block(f"{layer_name}.{block_idx}")
            layers.append(
                PrunedGumbelBottleneck(
                    self.in_channels,
                    planes,
                    disabled_channels=output_ch,
                    disabled_mid1_channels=mid1_ch,
                    disabled_mid2_channels=mid2_ch,
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
