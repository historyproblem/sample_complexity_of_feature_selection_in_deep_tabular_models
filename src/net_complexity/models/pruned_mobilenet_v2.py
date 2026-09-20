"""Structural channel pruning of MobileNetV2 inverted-residual blocks.

MobileNetV2 counterpart of ``pruned_bottleneck.py``: once the iterative
channel-pruning search (``training.cyclic_channel_pruning``) has decided
which output channels of a block's residual branch are dispensable, they are
physically removed from the block's ``project`` conv and ``project_bn`` here.

Block input/output dimensionality is unchanged — removed channels are
scattered back to zero and their value comes from the residual connection
only, exactly what a fully-closed ``MaskedGumbelInvertedResidual`` gate
produces for that channel. That keeps this a drop-in replacement inside
``MobileNetV2.features``.

Compute savings vs. the full block (``n_active = oup - |disabled|``):
  ``project`` and ``project_bn`` shrink proportionally to ``n_active``;
  ``expand``/``depthwise`` are untouched — the inverted bottleneck's internal
  width is not narrowed here, only the block's residual *output*. This
  mirrors the ResNet path, where ``conv1``/``conv2`` likewise stay at full
  width while ``conv3``'s output is narrowed.

Only residual blocks (``stride == 1 and inp == oup``) can be pruned: a
non-residual block's output feeds the next block directly, so narrowing it
would change that block's input width.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .mobilenet_v2 import (
    MOBILENET_V2_INVERTED_RESIDUAL_SETTING,
    ConvBNReLU,
    InvertedResidual,
    MobileNetV2,
)


class PrunedGumbelInvertedResidual(InvertedResidual):
    """InvertedResidual narrowed at up to two independent boundaries.

    ``disabled_channels`` — the "output" boundary (indices into ``oup``).
        Excluded from ``project``/``project_bn`` entirely; their contribution
        to the block output comes from the residual connection only, so the
        block's output width is unchanged (scatter-back). Residual blocks
        only.

    ``disabled_mid_channels`` — the internal width (indices into
        ``hidden_dim``), i.e. ``expand``'s output == ``depthwise``'s channels
        == ``project``'s input. Removing one narrows all three at once and
        needs no scatter-back: nothing outside the block sees this width, so
        it works on non-residual blocks too. Requires an ``expand`` conv
        (``expand_ratio != 1``).

    The two are independent: ``project.weight`` is ``[oup, hidden]`` and each
    boundary shrinks a different axis of it.
    """

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: float,
        norm_layer: Callable[..., nn.Module] | None = None,
        disabled_channels: list[int] | None = None,
        disabled_mid_channels: list[int] | None = None,
    ) -> None:
        super().__init__(inp, oup, stride, expand_ratio, norm_layer=norm_layer)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        disabled = set(int(channel) for channel in (disabled_channels or []))
        if disabled and not self.use_res_connect:
            raise ValueError(
                "PrunedGumbelInvertedResidual: output channels can only be pruned from "
                "a residual block (stride == 1 and inp == oup); this block has "
                f"stride={stride}, inp={inp}, oup={oup}. A non-residual block's "
                "output width is consumed by the next block, so it cannot be narrowed."
            )

        active = [channel for channel in range(oup) if channel not in disabled]
        if not active:
            raise ValueError(
                f"All {oup} channels are disabled for this block — the residual "
                "branch would be empty. Provide at least one active channel, or "
                "drop the whole block via layer_skipping instead."
            )
        n_active = len(active)
        self.n_active = n_active

        mid_disabled = set(int(channel) for channel in (disabled_mid_channels or []))
        if mid_disabled and not self.has_expand:
            raise ValueError(
                "PrunedGumbelInvertedResidual: the internal width can only be pruned "
                "when the block has an expand conv (expand_ratio != 1); here "
                f"expand_ratio={expand_ratio}, so hidden_dim == inp == {inp} is the "
                "block's input width, not a local one."
            )
        mid_active = [channel for channel in range(self.hidden_dim) if channel not in mid_disabled]
        if not mid_active:
            raise ValueError(
                f"All {self.hidden_dim} internal channels are disabled for this block — "
                "the branch would be empty. Provide at least one active channel."
            )
        n_mid_active = len(mid_active)
        self.n_mid_active = n_mid_active

        if n_mid_active != self.hidden_dim:
            self.branch.expand = ConvBNReLU(
                inp, n_mid_active, kernel_size=1, norm_layer=norm_layer
            )
            self.branch.depthwise = ConvBNReLU(
                n_mid_active,
                n_mid_active,
                kernel_size=3,
                stride=stride,
                groups=n_mid_active,
                norm_layer=norm_layer,
            )
        if n_mid_active != self.hidden_dim or n_active != oup:
            self.branch.project = nn.Conv2d(n_mid_active, n_active, 1, 1, 0, bias=False)
        if n_active != oup:
            self.branch.project_bn = norm_layer(n_active)

        active_selection = torch.zeros(oup, n_active)
        for narrow_index, channel in enumerate(active):
            active_selection[channel, narrow_index] = 1.0
        self.register_buffer("active_selection", active_selection)
        self.register_buffer("active_indices", torch.tensor(active, dtype=torch.long))
        self.register_buffer("mid_active_indices", torch.tensor(mid_active, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.branch(x)  # [B, n_active, H, W]
        if not self.use_res_connect:
            return branch
        if self.n_active != self.oup:
            branch = torch.einsum("pn,bnhw->bphw", self.active_selection, branch)
        return x + branch


class PrunedMobileNetV2(MobileNetV2):
    """MobileNetV2 with per-block structural channel pruning of the residual branch.

    ``pruning_spec`` maps ``"features.N"`` keys (the block's index inside
    ``features``, matching ``layer_skipping``'s addressing) to that block's
    disabled channels. Each value is either a plain list of ints — shorthand
    for the "output" boundary only — or a dict ``{"output": [...], "mid":
    [...]}`` (both optional), mirroring ``PrunedResNet``'s nested shape.
    ``mid`` is the inverted bottleneck's internal width; MobileNetV2 needs
    one key there where the Bottleneck needs two (mid1/mid2), because
    ``depthwise`` preserves channel identity. Blocks absent from
    ``pruning_spec`` keep all channels.
    """

    def __init__(
        self,
        pruning_spec: dict[str, list[int] | dict[str, list[int]]],
        num_classes: int = 1000,
        in_channels: int = 3,
        width_mult: float = 1.0,
        inverted_residual_setting: list[list[int]] | None = None,
        round_nearest: int = 8,
        norm_layer: Callable[..., nn.Module] | None = None,
        dropout: float = 0.2,
        stem_stride: int = 2,
    ) -> None:
        block_kwargs_by_index = {
            index: {
                "disabled_channels": boundaries["output"],
                "disabled_mid_channels": boundaries["mid"],
            }
            for index, boundaries in _pruning_spec_by_feature_index(pruning_spec).items()
        }
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            width_mult=width_mult,
            inverted_residual_setting=inverted_residual_setting,
            round_nearest=round_nearest,
            block=PrunedGumbelInvertedResidual,
            norm_layer=norm_layer,
            dropout=dropout,
            stem_stride=stem_stride,
            block_kwargs_by_index=block_kwargs_by_index,
        )
        self.pruning_spec = dict(pruning_spec)


def _pruning_spec_by_feature_index(
    pruning_spec: dict[str, list[int] | dict[str, list[int]]],
) -> dict[int, dict[str, list[int]]]:
    """Convert ``{"features.N": spec}`` keys to ``{N: {"output": [...], "mid": [...]}}``."""
    by_index: dict[int, dict[str, list[int]]] = {}
    for key, value in pruning_spec.items():
        parts = str(key).split(".")
        if len(parts) != 2 or parts[0] != "features" or not parts[1].isdigit():
            raise ValueError(
                "PrunedMobileNetV2 pruning_spec keys must look like 'features.N' "
                f"(N = index inside MobileNetV2.features); got {key!r}."
            )
        if isinstance(value, dict):
            unknown = set(value) - {"output", "mid"}
            if unknown:
                raise ValueError(
                    "PrunedMobileNetV2 pruning_spec boundaries must be 'output' and/or "
                    f"'mid'; got unknown key(s) {sorted(unknown)} for {key!r}."
                )
            output_channels = value.get("output", [])
            mid_channels = value.get("mid", [])
        else:
            output_channels, mid_channels = value, []
        by_index[int(parts[1])] = {
            "output": [int(channel) for channel in output_channels],
            "mid": [int(channel) for channel in mid_channels],
        }
    return by_index


__all__ = [
    "MOBILENET_V2_INVERTED_RESIDUAL_SETTING",
    "PrunedGumbelInvertedResidual",
    "PrunedMobileNetV2",
]
