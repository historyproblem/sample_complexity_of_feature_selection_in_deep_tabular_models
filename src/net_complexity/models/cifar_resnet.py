from __future__ import annotations

"""
Adapted from akamaster/pytorch_resnet_cifar10
https://github.com/akamaster/pytorch_resnet_cifar10

Original work copyright (c) 2018, Yerlan Idelbayev.
Distributed under the BSD-2-Clause license.
"""

from collections.abc import Callable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(module.weight)


def _block_attr(block: Callable[..., nn.Module], name: str, default):
    target = getattr(block, "func", block)
    return getattr(target, name, default)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, option: str = "A"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()

        if stride != 1 or in_planes != planes * self.expansion:
            if option == "A":
                out_channels = planes * self.expansion
                pad_channels = out_channels - in_planes
                if pad_channels < 0:
                    raise ValueError("Option A shortcut only supports non-decreasing channels.")
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
                    nn.Conv2d(
                        in_planes,
                        planes * self.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * self.expansion),
                )
            else:
                raise ValueError("shortcut option must be either 'A' or 'B'.")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet(nn.Module):
    def __init__(
        self,
        block: Callable[..., nn.Module],
        num_blocks: list[int],
        num_classes: int = 10,
        in_channels: int = 3,
        shortcut_option: str = "A",
    ):
        super().__init__()
        self.in_planes = 16
        self.shortcut_option = shortcut_option
        self.block_expansion = _block_attr(block, "expansion", 1)

        self.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * self.block_expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(
        self,
        block: Callable[..., nn.Module],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for block_stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride=block_stride,
                    option=self.shortcut_option,
                )
            )
            self.in_planes = planes * self.block_expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
