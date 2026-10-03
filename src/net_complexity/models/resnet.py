from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


def _block_target(block: Callable[..., nn.Module]):
    target = block
    while hasattr(target, "func"):
        target = target.func
    return target


def _block_attr(block: Callable[..., nn.Module], name: str, default):
    return getattr(_block_target(block), name, default)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        ResBlock: Callable[..., nn.Module],
        layer_list,
        num_classes,
        in_channels=3,
        stem_feature_selector_factory: Callable[[int], nn.Module] | None = None,
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        stem_padding: int = 3,
        use_maxpool: bool = True,
        base_width: int = 64,
    ):
        super().__init__()
        if type(base_width) is not int or base_width < 1:
            raise ValueError("base_width must be a positive integer.")
        self.in_channels = base_width
        self.block_expansion = _block_attr(ResBlock, "expansion", 1)

        self.conv1 = nn.Conv2d(
            in_channels,
            base_width,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            padding=stem_padding,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU()
        self.stem_feature_selector = (
            stem_feature_selector_factory(base_width)
            if stem_feature_selector_factory is not None
            else nn.Identity()
        )
        self.max_pool = (
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if use_maxpool
            else nn.Identity()
        )

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=base_width)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=base_width * 2, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=base_width * 4, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=base_width * 8, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * self.block_expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.stem_feature_selector(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * self.block_expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * self.block_expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * self.block_expansion),
            )

        layers.append(
            ResBlock(
                self.in_channels,
                planes,
                i_downsample=ii_downsample,
                stride=stride,
            )
        )
        self.in_channels = planes * self.block_expansion

        for _ in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)
