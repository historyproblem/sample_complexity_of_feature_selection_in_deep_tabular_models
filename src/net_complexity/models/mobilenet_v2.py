"""MobileNetV2 backbone and its gated variants.

Self-contained port of ``torchvision.models.mobilenetv2`` (see
``baselines/vision-main/torchvision/models/mobilenetv2.py``) adapted to this
repository's conventions, so the whole ablation suite — classic AIG, DepGraph,
and the "ours" block/channel pruning families — can run on MobileNetV2 the
same way it runs on ResNet50/101.

Differences from the torchvision original, all additive:

* No torchvision imports (``Conv2dNormActivation``/``_make_divisible`` are
  reimplemented here), so the model has no dependency outside torch.
* ``in_channels``/``stem_stride`` are configurable. TinyImageNet-200 is
  64x64, where the stock stride-2 stem plus the stock stage strides would
  leave a 2x2 final feature map; ``stem_stride=1`` gives 4x4 instead, the
  same kind of small-input adaptation ``ResNet50(stem_stride=1,
  use_maxpool=false)`` already uses for the ResNet configs.
* The block's conv stack lives in ``self.branch`` (not ``self.conv``), an
  ``nn.Sequential`` with *named* entries (``expand``/``depthwise``/
  ``project``/``project_bn``). The name matters: ``metrics.aig``'s
  ``AIGFLOPsMetric`` already attributes gated FLOPs/params by the
  ``<block>.branch.`` prefix (the convention introduced by
  ``efficientnet_v2_aig.py``), so AIG compute accounting works unchanged.
* ``block_kwargs_by_index`` lets a caller pass per-block constructor kwargs
  keyed by the block's ``features`` index — used by
  ``pruned_mobilenet_v2.PrunedMobileNetV2`` to hand each block its own list
  of physically removed channels.

Gating follows the same rule as ``EfficientNetV2AIGBlock``: only blocks with
a residual connection (``stride == 1 and inp == oup``) carry a gate, because
only there does "gate closed" degenerate cleanly to identity.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn

from .aig import AIGBlockGate
from .feature_selection import MaskedGumbelLayer


# (expand_ratio t, out_channels c, num_blocks n, stride s) — the stock
# MobileNetV2 table from the paper / torchvision.
MOBILENET_V2_INVERTED_RESIDUAL_SETTING: list[list[int]] = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


def _make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round ``value`` to the nearest multiple of ``divisor`` (torchvision's rule)."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:  # never round down by more than 10%
        new_value += divisor
    return int(new_value)


class ConvBNReLU(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU6, torchvision's ``Conv2dNormActivation`` subset."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_channels),
        ]
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted-residual block (expand -> depthwise -> project).

    ``branch`` holds the conv stack; ``use_res_connect`` says whether the
    block's output is added to its input. Only residual blocks are gateable
    and prunable — a non-residual block changes shape, so removing it (or
    narrowing its output) would break the next block's input width.
    """

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: float,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.inp = int(inp)
        self.oup = int(oup)
        self.stride = int(stride)
        self.expand_ratio = expand_ratio
        self.hidden_dim = hidden_dim
        self.use_res_connect = self.stride == 1 and inp == oup
        # expand_ratio == 1 means no pointwise expansion: hidden_dim == inp,
        # so the block's "internal" width is really its input width and is
        # not independently prunable (see MaskedGumbelInvertedResidual).
        self.has_expand = expand_ratio != 1

        layers: list[tuple[str, nn.Module]] = []
        if expand_ratio != 1:
            layers.append(
                ("expand", ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
            )
        layers.append(
            (
                "depthwise",
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                ),
            )
        )
        layers.append(("project", nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)))
        layers.append(("project_bn", norm_layer(oup)))

        self.branch = nn.Sequential(OrderedDict(layers))
        self.out_channels = oup

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.branch(x)
        return self.branch(x)


class AIGInvertedResidual(InvertedResidual):
    """InvertedResidual with a per-block AIG gate on the residual branch.

    MobileNetV2 counterpart of ``AIGBottleneckLayer`` /
    ``EfficientNetV2AIGBlock``: a closed gate makes the block an exact
    identity for that input. Non-residual blocks get no gate (``gate is
    None``) and behave exactly like the plain block.
    """

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: float,
        norm_layer: Callable[..., nn.Module] | None = None,
        gate_hidden_channels: int = 16,
        keep_prob_init: float | None = None,
        gate_threshold: float = 0.5,
        temperature: float = 1.0,
        gate_regularization: str = "l2_gate",
    ) -> None:
        super().__init__(inp, oup, stride, expand_ratio, norm_layer=norm_layer)
        self.bypass = False
        self.gate = (
            AIGBlockGate(
                inp,
                hidden_channels=gate_hidden_channels,
                keep_prob_init=keep_prob_init,
                threshold=gate_threshold,
                temperature=temperature,
                regularization=gate_regularization,
            )
            if self.use_res_connect
            else None
        )

    def set_bypass(self, enabled: bool) -> None:
        self.bypass = bool(enabled)
        if self.gate is not None:
            self.gate.set_bypass(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.branch(x)
        if not self.use_res_connect:
            return branch
        if self.gate is None:
            return x + branch
        return x + self.gate(x) * branch


class MaskedGumbelInvertedResidual(InvertedResidual):
    """InvertedResidual with per-channel MaskedGumbelLayer gate(s).

    Two independent gates, both optional:

    ``gumbel_layer`` — the "output" gate, on the projection output right
        before the residual sum. Closing a channel makes that channel pure
        identity, exactly what ``PrunedGumbelInvertedResidual`` reproduces
        after physical pruning. Only residual blocks (``stride == 1 and
        inp == oup``) can have it: without a skip connection there is no
        identity to fall back to.

    ``mid_gumbel_layer`` — the optional internal-width gate
        (``gate_internal_width=True``), covering the inverted bottleneck's
        hidden width: ``expand``'s output == ``depthwise``'s channels ==
        ``project``'s input. Narrowing it is purely local — the block's
        output width does not change — so it works on *non-residual* blocks
        too, which the output gate cannot reach at all.

    Why a single internal gate, unlike ``MaskedGumbelBottleneckLayer``'s two
    (mid1/mid2): there conv2 is a full 3x3 conv that mixes every input
    channel into every output channel, so its input and output are unrelated
    channel spaces needing separate decisions. Here ``depthwise`` has
    ``groups == hidden_dim`` and preserves channel identity — input channel i
    maps to output channel i — so expand-out, depthwise and project-in are
    one coupled group decided by one gate.

    Placement matters: the gate sits *after* ``depthwise`` (after its BN and
    ReLU6), not after ``expand``. A masked channel must be exactly zero where
    ``project`` consumes it, and ``project`` is a linear 1x1 conv, so a zero
    input channel contributes nothing — soft masking is then numerically
    identical to physically removing that channel. Gating before
    ``depthwise`` instead would leak: the depthwise BN maps 0 to
    ``beta - gamma*mean/std != 0``, so the "disabled" channel would reach
    ``project`` as a non-zero constant and the search phase would not match
    the pruned model.

    Blocks with ``expand_ratio == 1`` (no ``expand`` conv, ``hidden == inp``)
    get no internal gate: their hidden width *is* the block input, so
    narrowing it would not be local.
    """

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: float,
        norm_layer: Callable[..., nn.Module] | None = None,
        temperature: float = 1.0,
        beta: float = 1.0,
        force_ones_mask: bool = False,
        deterministic_soft_mask: bool = False,
        deterministic_hard_mask: bool = False,
        train_gate_mode: str | None = None,
        eval_gate_mode: str | None = None,
        gate_threshold: float = 0.5,
        disabled_channels: list[int] | None = None,
        gate_internal_width: bool = False,
        disabled_mid_channels: list[int] | None = None,
    ) -> None:
        super().__init__(inp, oup, stride, expand_ratio, norm_layer=norm_layer)

        gate_kwargs = dict(
            temperature=temperature,
            beta=beta,
            force_ones_mask=force_ones_mask,
            deterministic_soft_mask=deterministic_soft_mask,
            deterministic_hard_mask=deterministic_hard_mask,
            train_gate_mode=train_gate_mode,
            eval_gate_mode=eval_gate_mode,
            gate_threshold=gate_threshold,
        )

        self.gumbel_layer = (
            MaskedGumbelLayer(input_dim=oup, disabled_channels=disabled_channels, **gate_kwargs)
            if self.use_res_connect
            else None
        )

        self.gate_internal_width = bool(gate_internal_width)
        self.mid_gumbel_layer = (
            MaskedGumbelLayer(
                input_dim=self.hidden_dim,
                disabled_channels=disabled_mid_channels,
                **gate_kwargs,
            )
            if self.gate_internal_width and self.has_expand
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mid_gumbel_layer is None:
            branch = self.branch(x)
        else:
            # Same ops as self.branch, with the internal gate spliced in
            # between depthwise and project (see the class docstring).
            branch = self.branch.expand(x)
            branch = self.branch.depthwise(branch)
            branch = self.mid_gumbel_layer(branch)
            branch = self.branch.project(branch)
            branch = self.branch.project_bn(branch)

        if not self.use_res_connect:
            return branch
        if self.gumbel_layer is not None:
            branch = self.gumbel_layer(branch)
        return x + branch


class SkippedInvertedResidual(nn.Module):
    """Residual InvertedResidual whose forward is short-circuited to identity.

    MobileNetV2 counterpart of ``SkippedBottleneck``: the original block's
    weights are kept (``inner`` holds it) so checkpoints stay loadable, but
    the branch is never evaluated. Only valid for residual blocks.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PrunedInvertedResidual(nn.Module):
    """Parameter-free stub replacing a dropped residual InvertedResidual.

    MobileNetV2 counterpart of ``PrunedBottleneck``. Simpler than the
    ResNet case: a gated MobileNetV2 block always has ``stride == 1`` and
    ``inp == oup``, so there is no downsample projection to keep — the whole
    block collapses to identity.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


_MOBILENET_BLOCK_TYPES = (
    InvertedResidual,
    AIGInvertedResidual,
    MaskedGumbelInvertedResidual,
)


class MobileNetV2(nn.Module):
    """MobileNetV2 backbone returning logits, usable as a ``backbone=`` argument.

    Mirrors ``ResNet50``'s role in this repo: a plain ``nn.Module`` whose
    ``forward(x) -> logits`` is wrapped by
    ``ClassificationFeatureSelectionWrapper``. Pass ``block`` (typically a
    ``functools.partial``) to swap in a gated block type, the same way
    ``ResNet50(resnet_block=partial(AIGBottleneckLayer, ...))`` does.

    Block addressing is ``features.N`` (N is the index inside
    ``self.features``), matching the ``layerN.B`` convention used by
    ``layer_skipping``/``cyclic_aig`` for ResNet — ``features.0`` is the stem
    and the last entry is the 1x1 head conv, so only the indices in between
    are gateable blocks.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        width_mult: float = 1.0,
        inverted_residual_setting: list[list[int]] | None = None,
        round_nearest: int = 8,
        block: Callable[..., nn.Module] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
        dropout: float = 0.2,
        stem_stride: int = 2,
        block_kwargs_by_index: dict[int, dict] | None = None,
    ) -> None:
        super().__init__()
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if inverted_residual_setting is None:
            inverted_residual_setting = MOBILENET_V2_INVERTED_RESIDUAL_SETTING
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                "inverted_residual_setting should be a non-empty list of 4-element "
                f"(t, c, n, s) entries, got {inverted_residual_setting}"
            )

        block_kwargs_by_index = dict(block_kwargs_by_index or {})
        input_channel = _make_divisible(32 * width_mult, round_nearest)
        self.last_channel = _make_divisible(1280 * max(1.0, width_mult), round_nearest)

        features: list[nn.Module] = [
            ConvBNReLU(
                in_channels,
                input_channel,
                kernel_size=3,
                stride=stem_stride,
                norm_layer=norm_layer,
            )
        ]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                feature_index = len(features)
                extra_kwargs = dict(block_kwargs_by_index.get(feature_index, {}))
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                        **extra_kwargs,
                    )
                )
                input_channel = output_channel
        features.append(
            ConvBNReLU(
                input_channel,
                self.last_channel,
                kernel_size=1,
                norm_layer=norm_layer,
            )
        )

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


def MobileNetV2TinyImageNet200(
    num_classes: int = 200,
    in_channels: int = 3,
    **kwargs,
) -> MobileNetV2:
    """MobileNetV2 with a stride-1 stem, for 64x64 TinyImageNet-200 inputs.

    The stock stride-2 stem would take 64x64 down to 2x2 before the head;
    with ``stem_stride=1`` the final feature map is 4x4 instead. Same
    adaptation ``resnet50_tinyimagenet200.yaml`` makes via
    ``stem_stride: 1, use_maxpool: false``.
    """
    kwargs.setdefault("stem_stride", 1)
    return MobileNetV2(num_classes=num_classes, in_channels=in_channels, **kwargs)


def mobilenet_v2_aig_block(**gate_kwargs) -> Callable[..., AIGInvertedResidual]:
    """``partial(AIGInvertedResidual, **gate_kwargs)`` — convenience for Hydra configs."""
    return partial(AIGInvertedResidual, **gate_kwargs)
