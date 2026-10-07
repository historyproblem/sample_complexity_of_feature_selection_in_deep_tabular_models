"""AutoPruner selectors on MobileNetV2's independently narrowable inner widths."""

from __future__ import annotations

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from .autopruner import AutoPrunerLayer, _copy_batch_norm, _copy_conv2d
from .mobilenet_v2 import InvertedResidual, MobileNetV2


class AutoPrunerInvertedResidual(InvertedResidual):
    """Select channels after depthwise BN/ReLU6, immediately before projection."""

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: float,
        *,
        activation_size: int,
        target_keep_ratio: float = 0.5,
        norm_layer=None,
    ) -> None:
        super().__init__(inp, oup, stride, expand_ratio, norm_layer=norm_layer)
        self.pruner: nn.Module = (
            AutoPrunerLayer(
                self.hidden_dim,
                activation_size,
                stage_index=0,
                target_keep_ratio=target_keep_ratio,
                max_pool_kernel=max(1, activation_size // 2),
            )
            if self.has_expand
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_expand:
            branch = self.branch(x)
        else:
            branch = self.branch.expand(x)
            branch = self.branch.depthwise(branch)
            branch = self.pruner(branch)
            branch = self.branch.project(branch)
            branch = self.branch.project_bn(branch)
        return x + branch if self.use_res_connect else branch


class AutoPrunerMobileNetV2(MobileNetV2):
    """Stock MobileNetV2 with one AutoPruner search phase over internal widths."""

    def __init__(
        self,
        *,
        input_size: int = 224,
        target_keep_ratio: float = 0.5,
        stem_stride: int = 2,
        **kwargs,
    ) -> None:
        if int(input_size) <= 0:
            raise ValueError("input_size must be positive")
        if kwargs.get("pretrained_weights"):
            raise ValueError(
                "Use model.pretrained_checkpoint with a project best.pt; "
                "do not also load torchvision pretrained_weights."
            )
        spatial_size = (int(input_size) + 1) // int(stem_stride)

        def make_block(inp, oup, stride, expand_ratio, norm_layer=None, **unused):
            nonlocal spatial_size
            spatial_size = (spatial_size + int(stride) - 1) // int(stride)
            return AutoPrunerInvertedResidual(
                inp, oup, stride, expand_ratio,
                activation_size=spatial_size,
                target_keep_ratio=target_keep_ratio,
                norm_layer=norm_layer,
            )

        super().__init__(stem_stride=stem_stride, block=make_block, **kwargs)
        # MobileNetV2's general Conv2d initializer also visited the coders.
        for module in self.modules():
            if isinstance(module, AutoPrunerLayer):
                module._reset_coder_parameters()


class PrunedAutoPrunerInvertedResidual(nn.Module):
    """Deployment block with an actually narrower expand/depthwise/project path."""

    def __init__(
        self,
        source: AutoPrunerInvertedResidual,
        *,
        use_binary_masks: bool,
    ) -> None:
        super().__init__()
        self.use_res_connect = source.use_res_connect
        if not source.has_expand:
            self.branch = copy.deepcopy(source.branch)
        else:
            device = source.branch.expand[0].weight.device
            mask = source.pruner.get_binary_mask()
            indices = (
                torch.nonzero(mask > 0.5, as_tuple=False).flatten().to(device)
                if use_binary_masks
                else torch.arange(source.hidden_dim, device=device)
            )
            if indices.numel() == 0:
                raise ValueError("AutoPruner cannot export an empty inner width")
            expand = nn.Sequential(
                _copy_conv2d(source.branch.expand[0], output_indices=indices),
                _copy_batch_norm(source.branch.expand[1], indices),
                copy.deepcopy(source.branch.expand[2]),
            )
            old_depthwise = source.branch.depthwise[0]
            depthwise = nn.Conv2d(
                int(indices.numel()), int(indices.numel()),
                kernel_size=old_depthwise.kernel_size,
                stride=old_depthwise.stride,
                padding=old_depthwise.padding,
                dilation=old_depthwise.dilation,
                groups=int(indices.numel()),
                bias=old_depthwise.bias is not None,
            ).to(device=old_depthwise.weight.device, dtype=old_depthwise.weight.dtype)
            with torch.no_grad():
                depthwise.weight.copy_(old_depthwise.weight.index_select(0, indices))
                if old_depthwise.bias is not None:
                    depthwise.bias.copy_(old_depthwise.bias.index_select(0, indices))
            depthwise_stack = nn.Sequential(
                depthwise,
                _copy_batch_norm(source.branch.depthwise[1], indices),
                copy.deepcopy(source.branch.depthwise[2]),
            )
            self.branch = nn.Sequential(OrderedDict([
                ("expand", expand),
                ("depthwise", depthwise_stack),
                ("project", _copy_conv2d(source.branch.project, input_indices=indices)),
                ("project_bn", copy.deepcopy(source.branch.project_bn)),
            ]))
            self.register_buffer("active_inner_indices", indices.detach().clone())
        self.train(source.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.branch(x)
        return x + branch if self.use_res_connect else branch
