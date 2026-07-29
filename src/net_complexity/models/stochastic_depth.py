from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .resnet import Bottleneck, ResNet


def linear_survival_probabilities(
    num_blocks: int,
    final_survival_probability: float = 0.5,
) -> tuple[float, ...]:
    """Huang et al. linear survival schedule p_l = 1 - l / L * (1 - p_L)."""
    num_blocks = int(num_blocks)
    final_survival_probability = float(final_survival_probability)

    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    if not 0.0 <= final_survival_probability <= 1.0:
        raise ValueError("final_survival_probability must be in [0, 1].")

    return tuple(
        1.0 - (block_index / num_blocks) * (1.0 - final_survival_probability)
        for block_index in range(1, num_blocks + 1)
    )


class HuangStochasticDepthBottleneck(Bottleneck):
    """ResNet bottleneck with the original Huang et al. stochastic depth rule."""

    def __init__(
        self,
        in_channels,
        out_channels,
        i_downsample=None,
        stride=1,
        survival_probability: float = 1.0,
        block_index: int | None = None,
        total_blocks: int | None = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            i_downsample=i_downsample,
            stride=stride,
        )
        survival_probability = float(survival_probability)
        if not 0.0 <= survival_probability <= 1.0:
            raise ValueError("survival_probability must be in [0, 1].")

        self.survival_probability = survival_probability
        self.block_index = None if block_index is None else int(block_index)
        self.total_blocks = None if total_blocks is None else int(total_blocks)
        self.last_residual_branch_active = True
        self.register_buffer("last_survival_mask", torch.ones(()), persistent=False)

    def _sample_survival_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.survival_probability <= 0.0:
            return x.new_zeros(())
        if self.survival_probability >= 1.0:
            return x.new_ones(())
        return (
            torch.rand((), device=x.device) < self.survival_probability
        ).to(dtype=x.dtype)

    def _residual_branch(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.relu(self.batch_norm2(self.conv2(out)))
        return self.batch_norm3(self.conv3(out))

    def forward(self, x):
        identity = x
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        if self.training:
            survival_mask = self._sample_survival_mask(x)
            self.last_survival_mask = survival_mask.detach()
            self.last_residual_branch_active = bool(survival_mask.item())
            if self.last_residual_branch_active:
                out = identity + self._residual_branch(x)
            else:
                out = identity
        else:
            self.last_survival_mask = x.new_ones(())
            self.last_residual_branch_active = True
            out = identity + self._residual_branch(x) * self.survival_probability

        return self.relu(out)


class HuangStochasticDepthBlockFactory:
    """Callable ResNet block factory that assigns block-wise survival probabilities."""

    expansion = HuangStochasticDepthBottleneck.expansion

    def __init__(
        self,
        *,
        num_blocks: int,
        final_survival_probability: float = 0.5,
        survival_schedule: str = "linear",
        block_cls: Callable[..., HuangStochasticDepthBottleneck] = HuangStochasticDepthBottleneck,
    ) -> None:
        if survival_schedule != "linear":
            raise ValueError("Only the Huang et al. linear survival schedule is supported.")
        self.num_blocks = int(num_blocks)
        self.final_survival_probability = float(final_survival_probability)
        self.survival_schedule = survival_schedule
        self.block_cls = block_cls
        self.survival_probabilities = linear_survival_probabilities(
            self.num_blocks,
            self.final_survival_probability,
        )
        self._next_block_index = 0

    def __call__(self, *args, **kwargs):
        if self._next_block_index >= self.num_blocks:
            raise RuntimeError(
                "HuangStochasticDepthBlockFactory was called more times than num_blocks."
            )

        block_index = self._next_block_index + 1
        survival_probability = self.survival_probabilities[self._next_block_index]
        self._next_block_index += 1
        return self.block_cls(
            *args,
            **kwargs,
            survival_probability=survival_probability,
            block_index=block_index,
            total_blocks=self.num_blocks,
        )


def StochasticDepthResNet50(
    num_classes,
    in_channels=3,
    final_survival_probability: float = 0.5,
    survival_schedule: str = "linear",
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    stem_padding: int = 3,
    use_maxpool: bool = True,
):
    layer_list = [3, 4, 6, 3]
    num_blocks = sum(layer_list)
    block_factory = HuangStochasticDepthBlockFactory(
        num_blocks=num_blocks,
        final_survival_probability=final_survival_probability,
        survival_schedule=survival_schedule,
    )
    model = ResNet(
        block_factory,
        layer_list,
        num_classes,
        in_channels,
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        stem_padding=stem_padding,
        use_maxpool=use_maxpool,
    )
    model.stochastic_depth_final_survival_probability = float(final_survival_probability)
    model.stochastic_depth_survival_schedule = survival_schedule
    model.stochastic_depth_survival_probabilities = block_factory.survival_probabilities
    return model


def get_stochastic_depth_blocks(model: nn.Module) -> dict[str, HuangStochasticDepthBottleneck]:
    blocks: dict[str, HuangStochasticDepthBottleneck] = {}

    def _collect(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, HuangStochasticDepthBottleneck):
                blocks[full_name] = child
                continue
            _collect(child, full_name)

    _collect(model)
    return blocks
