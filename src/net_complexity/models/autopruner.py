"""AutoPruner layers, ResNet-50 integration, and physical channel export.

The implementation follows Luo & Wu, arXiv:1805.08941, and the authors'
official PyTorch repository at commit 4618a775bbf48d1166012b9b671f84c71b114e26:
https://github.com/Roll920/AutoPruner

The original code is MIT licensed (copyright 2020 Jian-Hao Luo). This module
is a modern, device-agnostic reimplementation rather than a verbatim copy.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .outputs import ClassifModelOutput


AUTHOR_ALPHA_START_RESNET = 1.0
AUTHOR_ALPHA_STOP_RESNET = 100.0
AUTHOR_ALPHA_UPDATE_INTERVAL = 100
AUTHOR_CODE_WINDOW_SIZE = 20
AUTHOR_INITIAL_REGULARIZATION = 10.0
AUTHOR_REGULARIZATION_SCALE = 100.0
AUTHOR_INITIAL_PRUNING_THRESHOLD = 0.95
AUTHOR_PRUNING_THRESHOLD_STEP = 0.05
AUTHOR_TWO_SIDED_LOW = 0.2
AUTHOR_TWO_SIDED_HIGH = 0.8
AUTHOR_TWO_SIDED_TARGET = 0.9
AUTHOR_PRUNING_EPOCHS_PER_STAGE = 8
AUTHOR_FINAL_FINE_TUNE_EPOCHS = 30
AUTHOR_PRUNING_LR = 1e-3
AUTHOR_PRUNING_WEIGHT_DECAY = 5e-4
AUTHOR_FINAL_WEIGHT_DECAY = 1e-4
AUTHOR_MOMENTUM = 0.9


class AutoPrunerLayer(nn.Module):
    """The paper's batch-pooled, activation-conditioned channel selector."""

    PHASE_OPEN = 0
    PHASE_SOFT = 1
    PHASE_HARD = 2

    def __init__(
        self,
        channels: int,
        activation_size: int,
        *,
        stage_index: int,
        target_keep_ratio: float = 0.5,
        alpha_start: float = AUTHOR_ALPHA_START_RESNET,
        alpha_stop: float = AUTHOR_ALPHA_STOP_RESNET,
        code_window_size: int = AUTHOR_CODE_WINDOW_SIZE,
        initial_regularization: float = AUTHOR_INITIAL_REGULARIZATION,
        max_pool_kernel: int | None = None,
    ) -> None:
        super().__init__()
        channels = int(channels)
        activation_size = int(activation_size)
        code_window_size = int(code_window_size)
        target_keep_ratio = float(target_keep_ratio)
        alpha_start = float(alpha_start)
        alpha_stop = float(alpha_stop)

        if channels <= 0:
            raise ValueError("channels must be positive.")
        if activation_size <= 0:
            raise ValueError("activation_size must be positive.")
        if not 0.0 < target_keep_ratio <= 1.0:
            raise ValueError("target_keep_ratio must be in (0, 1].")
        if alpha_start <= 0.0 or alpha_stop < alpha_start:
            raise ValueError("alpha values must satisfy 0 < alpha_start <= alpha_stop.")
        if code_window_size <= 0:
            raise ValueError("code_window_size must be positive.")

        if max_pool_kernel is None:
            max_pool_kernel = min(2, activation_size)
        max_pool_kernel = int(max_pool_kernel)
        if max_pool_kernel <= 0 or max_pool_kernel > activation_size:
            raise ValueError("max_pool_kernel must be within the activation size.")

        pooled_size = activation_size // max_pool_kernel
        if pooled_size <= 0:
            raise ValueError("The spatial pooling result must be non-empty.")

        self.channels = channels
        self.activation_size = activation_size
        self.stage_index = int(stage_index)
        self.target_keep_ratio = target_keep_ratio
        self.alpha_start = alpha_start
        self.alpha_stop = alpha_stop
        self.code_window_size = code_window_size
        self.max_pool_kernel = max_pool_kernel
        self.pooled_size = pooled_size

        self.pool = nn.MaxPool2d(
            kernel_size=max_pool_kernel,
            stride=max_pool_kernel,
        )
        # A full-spatial Conv2d is exactly the C x (C H' W') coding layer
        # described in section 3.1.2, while retaining the tensor layout.
        self.coder = nn.Conv2d(
            channels,
            channels,
            kernel_size=pooled_size,
            stride=1,
            padding=0,
            bias=True,
        )
        self._reset_coder_parameters()

        self.register_buffer("phase", torch.tensor(self.PHASE_OPEN, dtype=torch.long))
        self.register_buffer("alpha", torch.tensor(alpha_start, dtype=torch.float32))
        self.register_buffer("alpha_boost", torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "adaptive_regularization",
            torch.tensor(float(initial_regularization), dtype=torch.float32),
        )
        self.register_buffer(
            "pruning_threshold",
            torch.tensor(AUTHOR_INITIAL_PRUNING_THRESHOLD, dtype=torch.float32),
        )
        self.register_buffer("binary_mask", torch.ones(channels))
        self.register_buffer("last_code", torch.ones(channels))
        self.register_buffer(
            "code_window",
            torch.zeros(code_window_size, channels),
        )
        self.register_buffer("window_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("has_consensus", torch.zeros((), dtype=torch.bool))
        self._current_code: torch.Tensor | None = None
        self.set_open()

    def _reset_coder_parameters(self) -> None:
        # Equation (2): zero-mean Gaussian with std 10 * sqrt(2 / n),
        # n = C * H' * W'. The author code intentionally leaves the Conv2d
        # bias at PyTorch's default initialization.
        fan_in = self.channels * self.pooled_size * self.pooled_size
        nn.init.normal_(
            self.coder.weight,
            mean=0.0,
            std=10.0 * math.sqrt(2.0 / fan_in),
        )

    @property
    def phase_name(self) -> str:
        return {
            self.PHASE_OPEN: "open",
            self.PHASE_SOFT: "soft",
            self.PHASE_HARD: "hard",
        }[int(self.phase.item())]

    def set_open(self) -> None:
        self.phase.fill_(self.PHASE_OPEN)
        self.binary_mask.fill_(1.0)
        self.last_code.fill_(1.0)
        self.code_window.zero_()
        self.window_count.zero_()
        self.has_consensus.zero_()
        self._current_code = None
        self.coder.requires_grad_(False)

    def start_soft_pruning(self) -> None:
        self.phase.fill_(self.PHASE_SOFT)
        self.alpha.fill_(self.alpha_start)
        self.alpha_boost.zero_()
        self.adaptive_regularization.fill_(AUTHOR_INITIAL_REGULARIZATION)
        self.pruning_threshold.fill_(AUTHOR_INITIAL_PRUNING_THRESHOLD)
        self.binary_mask.fill_(1.0)
        self.last_code.fill_(1.0)
        self.code_window.zero_()
        self.window_count.zero_()
        self.has_consensus.zero_()
        self._current_code = None
        self.coder.requires_grad_(True)

    def set_alpha_base(self, value: float) -> None:
        self.alpha.fill_(float(value) + float(self.alpha_boost.item()))

    def _soft_code(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 4 or input.shape[1] != self.channels:
            raise ValueError(
                f"Expected NCHW input with {self.channels} channels, "
                f"got {tuple(input.shape)}."
            )
        if tuple(input.shape[-2:]) != (self.activation_size, self.activation_size):
            raise ValueError(
                "AutoPrunerLayer has a full-spatial coding layer and requires "
                f"{self.activation_size}x{self.activation_size} activations; "
                f"got {tuple(input.shape[-2:])}."
            )

        pooled_batch = input.mean(dim=0, keepdim=True)
        pooled_spatial = self.pool(pooled_batch)
        logits = self.coder(pooled_spatial).reshape(self.channels)
        return torch.sigmoid(self.alpha.to(logits) * logits)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        phase = int(self.phase.item())
        if phase == self.PHASE_OPEN:
            code = input.new_ones(self.channels)
        elif phase == self.PHASE_HARD or not self.training:
            code = self.binary_mask.to(input)
        else:
            code = self._soft_code(input)

        self._current_code = code
        self.last_code.copy_(code.detach().to(self.last_code))
        return input * code.view(1, -1, 1, 1)

    def current_code(self) -> torch.Tensor:
        if self._current_code is None:
            return self.last_code
        return self._current_code

    def regularization_error(self) -> torch.Tensor:
        return (self.current_code().abs().mean() - self.target_keep_ratio).square()

    def weighted_regularization(self) -> torch.Tensor:
        return self.adaptive_regularization.to(self.current_code()) * self.regularization_error()

    @torch.no_grad()
    def observe_current_code(self, *, allow_convergence_boost: bool) -> bool:
        """Update the author's 20-batch consensus and adaptive lambda.

        Returns ``True`` when a complete consensus window was processed.
        """

        if int(self.phase.item()) != self.PHASE_SOFT or not self.training:
            return False
        code = self.current_code().detach().to(self.code_window)
        index = int(self.window_count.item())
        self.code_window[index].copy_(code)
        self.window_count.add_(1)
        if int(self.window_count.item()) < self.code_window_size:
            return False

        self._update_consensus(
            self.code_window,
            allow_convergence_boost=allow_convergence_boost,
        )
        self.code_window.zero_()
        self.window_count.zero_()
        return True

    @torch.no_grad()
    def _update_consensus(
        self,
        codes: torch.Tensor,
        *,
        allow_convergence_boost: bool,
    ) -> None:
        per_batch_binary = (codes >= 0.5).to(codes.dtype)
        consensus = (per_batch_binary.mean(dim=0) >= 0.5).to(codes.dtype)
        self.binary_mask.copy_(consensus.to(self.binary_mask))
        self.has_consensus.fill_(True)

        preserved_ratio = float(consensus.mean().item())
        self.adaptive_regularization.fill_(
            AUTHOR_REGULARIZATION_SCALE
            * abs(preserved_ratio - self.target_keep_ratio)
        )

        pruning_ratio = 1.0 - preserved_ratio
        if pruning_ratio >= float(self.pruning_threshold.item()):
            self.alpha_boost.add_(1.0)
            minimum_threshold = 1.0 - self.target_keep_ratio
            self.pruning_threshold.fill_(
                max(
                    float(self.pruning_threshold.item())
                    - AUTHOR_PRUNING_THRESHOLD_STEP,
                    minimum_threshold,
                )
            )

        first_code = codes[0]
        two_sided_ratio = float(
            ((first_code > AUTHOR_TWO_SIDED_HIGH) | (first_code < AUTHOR_TWO_SIDED_LOW))
            .float()
            .mean()
            .item()
        )
        if allow_convergence_boost and two_sided_ratio < AUTHOR_TWO_SIDED_TARGET:
            self.alpha_boost.add_(1.0)

    @torch.no_grad()
    def finalize(self) -> None:
        if (
            int(self.phase.item()) == self.PHASE_SOFT
            and not bool(self.has_consensus.item())
        ):
            self.binary_mask.copy_((self.last_code >= 0.5).to(self.binary_mask))
        # The author's code only accepts complete 20-batch windows. Discard a
        # trailing partial window and retain the most recent full consensus.
        self.code_window.zero_()
        self.window_count.zero_()
        if not bool(self.binary_mask.any()):
            # The target r is positive, so this is only a numerical/optimization
            # failure. Keep the strongest channel to make export well-defined.
            strongest = int(self.last_code.argmax().item())
            self.binary_mask[strongest] = 1.0
        self.phase.fill_(self.PHASE_HARD)
        self._current_code = None
        self.coder.requires_grad_(False)

    def get_selection_probs(self) -> torch.Tensor:
        phase = int(self.phase.item())
        if phase == self.PHASE_OPEN:
            return torch.ones_like(self.binary_mask)
        if phase == self.PHASE_HARD or not self.training:
            return self.binary_mask
        return self.last_code

    def get_binary_mask(self) -> torch.Tensor:
        return self.binary_mask


class AutoPrunerBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stage_index: int,
        first_activation_size: int,
        second_activation_size: int,
        i_downsample: nn.Module | None = None,
        stride: int = 1,
        enable_pruners: bool = True,
        target_keep_ratio: float = 0.5,
        alpha_start: float = AUTHOR_ALPHA_START_RESNET,
        alpha_stop: float = AUTHOR_ALPHA_STOP_RESNET,
        code_window_size: int = AUTHOR_CODE_WINDOW_SIZE,
        pruner_max_pool_kernel: int | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.i_downsample = i_downsample
        self.stride = int(stride)
        self.stage_index = int(stage_index)
        self.relu = nn.ReLU(inplace=False)

        pruner_kwargs = {
            "stage_index": stage_index,
            "target_keep_ratio": target_keep_ratio,
            "alpha_start": alpha_start,
            "alpha_stop": alpha_stop,
            "code_window_size": code_window_size,
        }
        if enable_pruners:
            self.pruner1: nn.Module = AutoPrunerLayer(
                out_channels,
                first_activation_size,
                max_pool_kernel=pruner_max_pool_kernel,
                **pruner_kwargs,
            )
            self.pruner2: nn.Module = AutoPrunerLayer(
                out_channels,
                second_activation_size,
                max_pool_kernel=pruner_max_pool_kernel,
                **pruner_kwargs,
            )
        else:
            self.pruner1 = nn.Identity()
            self.pruner2 = nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        identity = input
        output = self.relu(self.batch_norm1(self.conv1(input)))
        output = self.pruner1(output)
        output = self.relu(self.batch_norm2(self.conv2(output)))
        output = self.pruner2(output)
        output = self.batch_norm3(self.conv3(output))
        if self.i_downsample is not None:
            identity = self.i_downsample(input)
        return self.relu(output + identity)


class AutoPrunerResNet(nn.Module):
    """ResNet bottleneck network with AutoPruner on two inner convolutions."""

    def __init__(
        self,
        layer_list: Sequence[int],
        num_classes: int,
        *,
        input_size: int,
        in_channels: int = 3,
        stage_planes: Sequence[int] = (64, 128, 256, 512),
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        stem_padding: int = 3,
        use_maxpool: bool = True,
        stages_to_prune: Sequence[int] = (0, 1, 2, 3),
        leave_last_block_unpruned: bool = True,
        target_keep_ratio: float = 0.5,
        alpha_start: float = AUTHOR_ALPHA_START_RESNET,
        alpha_stop: float = AUTHOR_ALPHA_STOP_RESNET,
        code_window_size: int = AUTHOR_CODE_WINDOW_SIZE,
    ) -> None:
        super().__init__()
        if len(layer_list) != 4 or len(stage_planes) != 4:
            raise ValueError("AutoPrunerResNet requires four ResNet stages.")
        if int(input_size) <= 0:
            raise ValueError("input_size must be positive.")

        self.layer_list = tuple(int(value) for value in layer_list)
        self.stage_planes = tuple(int(value) for value in stage_planes)
        self.stages_to_prune = tuple(sorted({int(value) for value in stages_to_prune}))
        self.leave_last_block_unpruned = bool(leave_last_block_unpruned)
        self.target_keep_ratio = float(target_keep_ratio)
        self.inplanes = self.stage_planes[0]

        self.conv1 = nn.Conv2d(
            in_channels,
            self.inplanes,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            padding=stem_padding,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.max_pool = (
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if use_maxpool
            else nn.Identity()
        )

        spatial_size = self._conv_output_size(
            int(input_size),
            kernel_size=int(stem_kernel_size),
            stride=int(stem_stride),
            padding=int(stem_padding),
        )
        if use_maxpool:
            spatial_size = self._conv_output_size(
                spatial_size,
                kernel_size=3,
                stride=2,
                padding=1,
            )

        block_kwargs = {
            "target_keep_ratio": target_keep_ratio,
            "alpha_start": alpha_start,
            "alpha_stop": alpha_stop,
            "code_window_size": code_window_size,
        }
        self.layer1, spatial_size = self._make_layer(
            stage_index=0,
            blocks=self.layer_list[0],
            planes=self.stage_planes[0],
            input_spatial_size=spatial_size,
            stride=1,
            block_kwargs=block_kwargs,
        )
        self.layer2, spatial_size = self._make_layer(
            stage_index=1,
            blocks=self.layer_list[1],
            planes=self.stage_planes[1],
            input_spatial_size=spatial_size,
            stride=2,
            block_kwargs=block_kwargs,
        )
        self.layer3, spatial_size = self._make_layer(
            stage_index=2,
            blocks=self.layer_list[2],
            planes=self.stage_planes[2],
            input_spatial_size=spatial_size,
            stride=2,
            block_kwargs=block_kwargs,
        )
        self.layer4, _ = self._make_layer(
            stage_index=3,
            blocks=self.layer_list[3],
            planes=self.stage_planes[3],
            input_spatial_size=spatial_size,
            stride=2,
            block_kwargs=block_kwargs,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.stage_planes[3] * AutoPrunerBottleneck.expansion, num_classes)
        self._initialize_deployment_parameters()

    @staticmethod
    def _conv_output_size(size: int, *, kernel_size: int, stride: int, padding: int) -> int:
        return (size + 2 * padding - kernel_size) // stride + 1

    def _make_layer(
        self,
        *,
        stage_index: int,
        blocks: int,
        planes: int,
        input_spatial_size: int,
        stride: int,
        block_kwargs: Mapping[str, Any],
    ) -> tuple[nn.Sequential, int]:
        downsample = None
        if stride != 1 or self.inplanes != planes * AutoPrunerBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * AutoPrunerBottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * AutoPrunerBottleneck.expansion),
            )

        output_spatial_size = self._conv_output_size(
            input_spatial_size,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        # In the author's ImageNet implementation the whole final 7x7 group
        # bypasses max pooling, including the 14x14 activation before its first
        # stride-2 convolution. Other groups use 2x2 pooling.
        pruner_max_pool_kernel = 1 if output_spatial_size == 7 else None
        layers: list[nn.Module] = []
        for block_index in range(int(blocks)):
            block_stride = stride if block_index == 0 else 1
            first_size = input_spatial_size if block_index == 0 else output_spatial_size
            is_final_block = (
                stage_index == 3
                and block_index == int(blocks) - 1
                and self.leave_last_block_unpruned
            )
            enable_pruners = stage_index in self.stages_to_prune and not is_final_block
            layers.append(
                AutoPrunerBottleneck(
                    self.inplanes,
                    planes,
                    stage_index=stage_index,
                    first_activation_size=first_size,
                    second_activation_size=output_spatial_size,
                    i_downsample=downsample if block_index == 0 else None,
                    stride=block_stride,
                    enable_pruners=enable_pruners,
                    pruner_max_pool_kernel=pruner_max_pool_kernel,
                    **block_kwargs,
                )
            )
            self.inplanes = planes * AutoPrunerBottleneck.expansion
        return nn.Sequential(*layers), output_spatial_size

    def _initialize_deployment_parameters(self) -> None:
        coder_ids = {
            id(module.coder)
            for module in self.modules()
            if isinstance(module, AutoPrunerLayer)
        }
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and id(module) not in coder_ids:
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.relu(self.batch_norm1(self.conv1(input)))
        output = self.max_pool(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output).flatten(1)
        return self.fc(output)


def AutoPrunerResNet50(**kwargs) -> AutoPrunerResNet:
    return AutoPrunerResNet(layer_list=(3, 4, 6, 3), **kwargs)


def get_autopruner_modules(model: nn.Module) -> dict[str, AutoPrunerLayer]:
    backbone = getattr(model, "backbone", model)
    return {
        name: module
        for name, module in backbone.named_modules()
        if isinstance(module, AutoPrunerLayer)
    }


def _strip_checkpoint_prefix(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module."):]
    if name.startswith("backbone."):
        name = name[len("backbone."):]
    return name


def load_autopruner_pretrained_backbone(
    backbone: nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load matching deployment parameters from a project checkpoint."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    raw_state = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, Mapping)
        else checkpoint
    )
    if not isinstance(raw_state, Mapping):
        raise TypeError("The pretrained checkpoint must contain a state dict.")

    source_state = {
        _strip_checkpoint_prefix(str(name)): tensor
        for name, tensor in raw_state.items()
        if torch.is_tensor(tensor)
    }
    target_state = backbone.state_dict()
    matched = {
        name: source_state[name]
        for name, target in target_state.items()
        if "pruner" not in name
        and name in source_state
        and tuple(source_state[name].shape) == tuple(target.shape)
    }
    # The project's older ResNet bottlenecks have Conv2d biases whereas the
    # paper/torchvision ResNet-50 uses bias=False before BatchNorm. Fold such a
    # bias into BatchNorm's running mean so loading a project baseline remains
    # functionally exact in eval mode: BN(Wx + b; mu) = BN(Wx; mu - b).
    folded_biases: list[str] = []
    bias_to_running_mean: dict[str, str] = {}
    for name in source_state:
        if name.endswith(".conv1.bias"):
            bias_to_running_mean[name] = (
                name.removesuffix(".conv1.bias") + ".batch_norm1.running_mean"
            )
        elif name.endswith(".conv2.bias"):
            bias_to_running_mean[name] = (
                name.removesuffix(".conv2.bias") + ".batch_norm2.running_mean"
            )
        elif name.endswith(".conv3.bias"):
            bias_to_running_mean[name] = (
                name.removesuffix(".conv3.bias") + ".batch_norm3.running_mean"
            )
        elif name.endswith(".i_downsample.0.bias"):
            bias_to_running_mean[name] = name.removesuffix(".0.bias") + ".1.running_mean"
    for bias_name, running_mean_name in bias_to_running_mean.items():
        bias = source_state[bias_name]
        running_mean = source_state.get(running_mean_name)
        target_mean = target_state.get(running_mean_name)
        if (
            running_mean is not None
            and target_mean is not None
            and tuple(bias.shape) == tuple(running_mean.shape) == tuple(target_mean.shape)
        ):
            matched[running_mean_name] = running_mean - bias
            folded_biases.append(bias_name)
    deployment_keys = {name for name in target_state if "pruner" not in name}
    missing_deployment = sorted(deployment_keys - set(matched))
    required_parameter_keys = {
        name
        for name, _ in backbone.named_parameters()
        if "pruner" not in name
    }
    missing_parameters = sorted(required_parameter_keys - set(matched))
    if missing_parameters:
        preview = ", ".join(missing_parameters[:5])
        raise ValueError(
            "The checkpoint does not contain a complete compatible ResNet "
            f"backbone; missing parameters include: {preview}."
        )
    incompatible = backbone.load_state_dict(matched, strict=False)
    return {
        "loaded_tensors": len(matched),
        "folded_conv_biases": folded_biases,
        "missing_deployment_keys": missing_deployment,
        "missing_parameter_keys": missing_parameters,
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


class AutoPrunerWrapper(nn.Module):
    """Stage-wise AutoPruner objective and author training-state controller."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        criterion: nn.Module | None = None,
        pretrained_checkpoint: str | None = None,
        require_pretrained: bool = False,
        pruning_epochs_per_stage: int = AUTHOR_PRUNING_EPOCHS_PER_STAGE,
        final_fine_tune_epochs: int = AUTHOR_FINAL_FINE_TUNE_EPOCHS,
        alpha_update_interval: int = AUTHOR_ALPHA_UPDATE_INTERVAL,
        pruning_lr: float = AUTHOR_PRUNING_LR,
        pruning_weight_decay: float = AUTHOR_PRUNING_WEIGHT_DECAY,
        final_weight_decay: float = AUTHOR_FINAL_WEIGHT_DECAY,
        stagewise: bool = True,
        reset_optimizer_each_stage: bool = True,
        select_best_per_stage: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.pruning_epochs_per_stage = int(pruning_epochs_per_stage)
        self.final_fine_tune_epochs = int(final_fine_tune_epochs)
        self.alpha_update_interval = int(alpha_update_interval)
        self.pruning_lr = float(pruning_lr)
        self.pruning_weight_decay = float(pruning_weight_decay)
        self.final_weight_decay = float(final_weight_decay)
        self.stagewise = bool(stagewise)
        self.reset_optimizer_each_stage = bool(reset_optimizer_each_stage)
        self.select_best_per_stage = bool(select_best_per_stage)
        if self.pruning_epochs_per_stage <= 0:
            raise ValueError("pruning_epochs_per_stage must be positive.")
        if self.alpha_update_interval <= 0:
            raise ValueError("alpha_update_interval must be positive.")

        self.pretrained_checkpoint = pretrained_checkpoint
        self.pretrained_load_info: dict[str, Any] | None = None
        if pretrained_checkpoint:
            self.pretrained_load_info = load_autopruner_pretrained_backbone(
                backbone,
                pretrained_checkpoint,
            )
        elif require_pretrained:
            raise ValueError(
                "AutoPruner is a fine-tuning method. Set model.pretrained_checkpoint "
                "to a trained ResNet checkpoint."
            )

        stage_indices = sorted(
            {
                module.stage_index
                for module in get_autopruner_modules(self).values()
            }
        )
        if not stage_indices:
            raise ValueError("AutoPrunerWrapper requires at least one AutoPrunerLayer.")
        self.stage_indices = tuple(stage_indices)
        self.register_buffer("current_stage_position", torch.zeros((), dtype=torch.long))
        self.register_buffer("batch_step_in_stage", torch.zeros((), dtype=torch.long))
        self.register_buffer("alpha_index", torch.zeros((), dtype=torch.long))
        self.register_buffer("batches_per_epoch", torch.zeros((), dtype=torch.long))
        self._stage_best_accuracy: float | None = None
        self._stage_best_state: dict[str, torch.Tensor] | None = None
        self._set_stage_phases(stage_position=0, reset_active=True)

    @property
    def num_pruning_phases(self) -> int:
        return len(self.stage_indices) if self.stagewise else 1

    @property
    def expected_num_epochs(self) -> int:
        return (
            self.num_pruning_phases * self.pruning_epochs_per_stage
            + self.final_fine_tune_epochs
        )

    def _modules_for_stage(self, stage_index: int) -> list[AutoPrunerLayer]:
        return [
            module
            for module in get_autopruner_modules(self).values()
            if module.stage_index == stage_index
        ]

    def _active_modules(self) -> list[AutoPrunerLayer]:
        position = int(self.current_stage_position.item())
        if position >= self.num_pruning_phases:
            return []
        if not self.stagewise:
            return list(get_autopruner_modules(self).values())
        return self._modules_for_stage(self.stage_indices[position])

    def _set_stage_phases(self, *, stage_position: int, reset_active: bool) -> None:
        modules = get_autopruner_modules(self)
        if not self.stagewise:
            if stage_position == 0:
                for module in modules.values():
                    if reset_active:
                        module.start_soft_pruning()
            else:
                for module in modules.values():
                    module.finalize()
            return

        for module in modules.values():
            module_position = self.stage_indices.index(module.stage_index)
            if module_position < stage_position:
                module.finalize()
            elif module_position == stage_position and stage_position < len(self.stage_indices):
                if reset_active:
                    module.start_soft_pruning()
            else:
                module.set_open()

    def _desired_stage_position(self, epoch: int) -> int:
        return min(
            (int(epoch) - 1) // self.pruning_epochs_per_stage,
            self.num_pruning_phases,
        )

    def _set_optimizer_recipe(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        stage_position: int,
    ) -> None:
        if stage_position < self.num_pruning_phases:
            epoch_within_stage = (int(epoch) - 1) % self.pruning_epochs_per_stage
            decay_epoch = max(self.pruning_epochs_per_stage // 2, 1)
            lr = self.pruning_lr * (0.1 ** (epoch_within_stage // decay_epoch))
            weight_decay = self.pruning_weight_decay
        else:
            fine_tune_epoch = (
                int(epoch)
                - self.num_pruning_phases * self.pruning_epochs_per_stage
                - 1
            )
            decay_epoch = max(self.final_fine_tune_epochs // 3, 1)
            lr = self.pruning_lr * (0.1 ** (fine_tune_epoch // decay_epoch))
            weight_decay = self.final_weight_decay
        for group in optimizer.param_groups:
            group["lr"] = float(lr)
            group["weight_decay"] = float(weight_decay)

    def on_train_epoch_start(
        self,
        *,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        batches_per_epoch: int,
    ) -> None:
        if int(batches_per_epoch) <= 0:
            raise ValueError("AutoPruner requires a non-empty training loader.")
        self.batches_per_epoch.fill_(int(batches_per_epoch))
        desired_position = self._desired_stage_position(epoch)
        current_position = int(self.current_stage_position.item())
        if desired_position != current_position:
            if (
                self.select_best_per_stage
                and current_position < self.num_pruning_phases
                and desired_position == current_position + 1
            ):
                self._restore_best_stage_state()
            self._set_stage_phases(
                stage_position=desired_position,
                reset_active=desired_position < self.num_pruning_phases,
            )
            self.current_stage_position.fill_(desired_position)
            self.batch_step_in_stage.zero_()
            self.alpha_index.zero_()
            self._stage_best_accuracy = None
            self._stage_best_state = None
            if self.reset_optimizer_each_stage:
                optimizer.state.clear()
        self._set_optimizer_recipe(
            optimizer,
            epoch=epoch,
            stage_position=desired_position,
        )

    @torch.no_grad()
    def on_validation_epoch_end(
        self,
        *,
        epoch: int,
        valid_metrics: Mapping[str, Any],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Cache the author's best validation model for the active group."""

        del epoch, optimizer
        if not self.select_best_per_stage:
            return
        if int(self.current_stage_position.item()) >= self.num_pruning_phases:
            return
        value = valid_metrics.get("valid_accuracy", valid_metrics.get("accuracy"))
        if not isinstance(value, (int, float)):
            return
        accuracy = float(value)
        if self._stage_best_accuracy is not None and accuracy <= self._stage_best_accuracy:
            return
        self._stage_best_accuracy = accuracy
        self._stage_best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.backbone.state_dict().items()
        }

    def _restore_best_stage_state(self) -> None:
        if self._stage_best_state is None:
            raise RuntimeError(
                "AutoPruner cannot advance stages before receiving validation metrics."
            )
        self.backbone.load_state_dict(self._stage_best_state, strict=True)

    def on_best_checkpoint_loaded(self, *, run_dir: str | Path) -> dict[str, str]:
        """Persist the physically narrowed deployment checkpoint."""

        target = Path(run_dir) / "checkpoints" / "autopruner_pruned.pt"
        save_pruned_autopruner_checkpoint(self, target)
        return {"autopruner_pruned_checkpoint": str(target)}

    def _alpha_schedule_value(self, index: int) -> float:
        modules = self._active_modules()
        if not modules:
            return 1.0
        total_steps = (
            self.pruning_epochs_per_stage * int(self.batches_per_epoch.item())
        )
        num_points = max(total_steps // self.alpha_update_interval, 1)
        if num_points == 1:
            return modules[0].alpha_start
        capped_index = min(max(int(index), 0), num_points - 1)
        fraction = capped_index / (num_points - 1)
        return modules[0].alpha_start + fraction * (
            modules[0].alpha_stop - modules[0].alpha_start
        )

    def _advance_alpha_before_forward(self) -> None:
        modules = self._active_modules()
        if not modules:
            return
        batch_step = int(self.batch_step_in_stage.item())
        if batch_step % self.alpha_update_interval != 0:
            return
        index = int(self.alpha_index.item())
        base_alpha = self._alpha_schedule_value(index)
        for module in modules:
            module.set_alpha_base(base_alpha)
        self.alpha_index.add_(1)

    def forward(self, input: torch.Tensor, targets: torch.Tensor) -> ClassifModelOutput:
        active_modules = self._active_modules()
        if self.training and active_modules:
            self._advance_alpha_before_forward()

        logits = self.backbone(input)
        ce_loss = self.criterion(logits, targets)
        if self.training and active_modules:
            raw_terms = [module.regularization_error() for module in active_modules]
            weighted_terms = [module.weighted_regularization() for module in active_modules]
            raw_regularization = sum(raw_terms)
            reg_loss = sum(weighted_terms)

            total_steps = (
                self.pruning_epochs_per_stage * int(self.batches_per_epoch.item())
            )
            num_points = max(total_steps // self.alpha_update_interval, 1)
            first_epoch_points = max(
                num_points // self.pruning_epochs_per_stage,
                1,
            )
            allow_boost = int(self.alpha_index.item()) >= first_epoch_points
            for module in active_modules:
                module.observe_current_code(
                    allow_convergence_boost=allow_boost,
                )
            self.batch_step_in_stage.add_(1)
        else:
            raw_regularization = logits.new_zeros(())
            reg_loss = logits.new_zeros(())

        modules = list(get_autopruner_modules(self).values())
        mean_p_open = torch.cat(
            [module.get_selection_probs().reshape(-1) for module in modules]
        ).mean()
        return ClassifModelOutput(
            ce_loss=ce_loss,
            regularization_loss=raw_regularization,
            reg_loss=reg_loss,
            mean_p_open=mean_p_open,
            loss=ce_loss + reg_loss,
            logits=logits,
        )


def _copy_batch_norm(source: nn.BatchNorm2d, indices: torch.Tensor) -> nn.BatchNorm2d:
    target = nn.BatchNorm2d(
        int(indices.numel()),
        eps=source.eps,
        momentum=source.momentum,
        affine=source.affine,
        track_running_stats=source.track_running_stats,
    ).to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        if source.affine:
            target.weight.copy_(source.weight.index_select(0, indices))
            target.bias.copy_(source.bias.index_select(0, indices))
        if source.track_running_stats:
            target.running_mean.copy_(source.running_mean.index_select(0, indices))
            target.running_var.copy_(source.running_var.index_select(0, indices))
            target.num_batches_tracked.copy_(source.num_batches_tracked)
    return target


def _copy_conv2d(
    source: nn.Conv2d,
    *,
    output_indices: torch.Tensor | None = None,
    input_indices: torch.Tensor | None = None,
) -> nn.Conv2d:
    output_indices = (
        torch.arange(source.out_channels, device=source.weight.device)
        if output_indices is None
        else output_indices.to(source.weight.device)
    )
    input_indices = (
        torch.arange(source.in_channels, device=source.weight.device)
        if input_indices is None
        else input_indices.to(source.weight.device)
    )
    if source.groups != 1:
        raise ValueError("Physical AutoPruner export currently requires groups=1.")
    target = nn.Conv2d(
        int(input_indices.numel()),
        int(output_indices.numel()),
        kernel_size=source.kernel_size,
        stride=source.stride,
        padding=source.padding,
        dilation=source.dilation,
        groups=1,
        bias=source.bias is not None,
        padding_mode=source.padding_mode,
    ).to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        weight = source.weight.index_select(0, output_indices)
        weight = weight.index_select(1, input_indices)
        target.weight.copy_(weight)
        if source.bias is not None:
            target.bias.copy_(source.bias.index_select(0, output_indices))
    return target


class PrunedAutoPrunerBottleneck(nn.Module):
    """Physically narrowed deployment form of an AutoPruner bottleneck."""

    expansion = 4

    def __init__(self, source: AutoPrunerBottleneck, *, use_binary_masks: bool) -> None:
        super().__init__()
        device = source.conv1.weight.device
        first_indices = self._active_indices(
            source.pruner1,
            source.conv1.out_channels,
            device=device,
            use_binary_masks=use_binary_masks,
        )
        second_indices = self._active_indices(
            source.pruner2,
            source.conv2.out_channels,
            device=device,
            use_binary_masks=use_binary_masks,
        )
        self.conv1 = _copy_conv2d(source.conv1, output_indices=first_indices)
        self.batch_norm1 = _copy_batch_norm(source.batch_norm1, first_indices)
        self.conv2 = _copy_conv2d(
            source.conv2,
            output_indices=second_indices,
            input_indices=first_indices,
        )
        self.batch_norm2 = _copy_batch_norm(source.batch_norm2, second_indices)
        self.conv3 = _copy_conv2d(source.conv3, input_indices=second_indices)
        all_output_indices = torch.arange(
            source.batch_norm3.num_features,
            device=device,
        )
        self.batch_norm3 = _copy_batch_norm(source.batch_norm3, all_output_indices)
        self.i_downsample = copy.deepcopy(source.i_downsample)
        self.stride = source.stride
        self.relu = nn.ReLU(inplace=False)
        self.register_buffer("first_active_indices", first_indices.detach().clone())
        self.register_buffer("second_active_indices", second_indices.detach().clone())
        self.train(source.training)

    @staticmethod
    def _active_indices(
        pruner: nn.Module,
        channels: int,
        *,
        device: torch.device,
        use_binary_masks: bool,
    ) -> torch.Tensor:
        if not use_binary_masks or not isinstance(pruner, AutoPrunerLayer):
            return torch.arange(channels, device=device)
        indices = torch.nonzero(pruner.get_binary_mask() > 0.5, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("Cannot export a bottleneck with no active channels.")
        return indices.to(device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        identity = input
        output = self.relu(self.batch_norm1(self.conv1(input)))
        output = self.relu(self.batch_norm2(self.conv2(output)))
        output = self.batch_norm3(self.conv3(output))
        if self.i_downsample is not None:
            identity = self.i_downsample(input)
        return self.relu(output + identity)


def export_pruned_autopruner_backbone(
    model: nn.Module,
    *,
    use_binary_masks: bool = True,
) -> nn.Module:
    """Remove AutoPruner layers and physically narrow adjacent convolutions."""

    backbone = getattr(model, "backbone", model)
    exported = copy.deepcopy(backbone)

    def replace(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, AutoPrunerBottleneck):
                setattr(
                    parent,
                    name,
                    PrunedAutoPrunerBottleneck(
                        child,
                        use_binary_masks=use_binary_masks,
                    ),
                )
            else:
                replace(child)

    replace(exported)
    exported.train(backbone.training)
    return exported


def autopruner_pruning_spec(model: nn.Module) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = {}
    for name, module in get_autopruner_modules(model).items():
        active = torch.nonzero(module.binary_mask > 0.5, as_tuple=False).flatten()
        disabled = torch.nonzero(module.binary_mask <= 0.5, as_tuple=False).flatten()
        result[name] = {
            "active": [int(value) for value in active.cpu().tolist()],
            "disabled": [int(value) for value in disabled.cpu().tolist()],
        }
    return result


def save_pruned_autopruner_checkpoint(model: nn.Module, path: str | Path) -> Path:
    exported = export_pruned_autopruner_backbone(model)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": exported.state_dict(),
            "pruning_spec": autopruner_pruning_spec(model),
        },
        target,
    )
    return target
