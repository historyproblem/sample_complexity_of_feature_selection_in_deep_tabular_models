"""Utilities for disabling (skipping) residual blocks from config.

A skipped block participates in the forward pass but contributes only its
shortcut connection - the residual branch is bypassed entirely.  Spatial
dimensions (including downsampling at stride-2 blocks) are preserved.

Config format::

    layer_skipping:
      enabled: true
      disabled_layers:
        - layer1.0
        - layer2.1

Layer naming follows the CIFARResNet convention: ``layerN.B`` where N is
the stage index (1-3) and B is the zero-based block index within that stage.
For ResNet20 each stage has 3 blocks (B in {0, 1, 2}).
"""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .cifar_resnet import CIFARBasicBlock
from .feature_selection import SkippedBottleneck, SkippedCIFARBasicBlock
from .resnet import Bottleneck


def _clear_module_cache(model: nn.Module) -> None:
    backbone = getattr(model, "backbone", model)
    if hasattr(model, "_net_complexity_module_cache"):
        delattr(model, "_net_complexity_module_cache")
    if hasattr(backbone, "_net_complexity_module_cache"):
        delattr(backbone, "_net_complexity_module_cache")


def _replace_block_with_skipped(backbone: nn.Module, key: str) -> bool:
    """Navigate backbone to 'layerN.B' and replace the block with SkippedCIFARBasicBlock.

    Returns True if the replacement was successful.
    """
    parts = key.split(".")
    if len(parts) != 2:
        print(f"[layer_skipping] Warning: '{key}' must be 'layerN.B' format - skipped.")
        return False

    layer_name, block_idx_str = parts
    try:
        block_idx = int(block_idx_str)
    except ValueError:
        print(f"[layer_skipping] Warning: block index '{block_idx_str}' is not an integer - skipped.")
        return False

    layer = getattr(backbone, layer_name, None)
    if layer is None:
        print(f"[layer_skipping] Warning: '{layer_name}' not found in backbone - skipped.")
        return False

    if block_idx >= len(layer):
        print(
            f"[layer_skipping] Warning: index {block_idx} out of range for "
            f"'{layer_name}' (len={len(layer)}) - skipped."
        )
        return False

    old_block = layer[block_idx]
    if isinstance(old_block, CIFARBasicBlock):
        in_planes = old_block.conv1.in_channels
        planes = old_block.conv2.out_channels
        stride = old_block.conv1.stride[0]
        option = "B" if isinstance(old_block.shortcut, nn.Sequential) else "A"
        new_block = SkippedCIFARBasicBlock(
            in_planes=in_planes,
            planes=planes,
            stride=stride,
            option=option,
        )
        layer[block_idx] = new_block
        print(
            f"[layer_skipping] '{key}': disabled "
            f"(in_planes={in_planes}, planes={planes}, stride={stride})"
        )
        return True
    elif isinstance(old_block, Bottleneck):
        in_channels = old_block.conv1.in_channels
        out_channels = old_block.conv1.out_channels
        stride = old_block.stride
        new_block = SkippedBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            i_downsample=old_block.i_downsample,
            stride=stride,
        )
        layer[block_idx] = new_block
        print(
            f"[layer_skipping] '{key}': disabled "
            f"(in_channels={in_channels}, out_channels={out_channels}, stride={stride})"
        )
        return True
    else:
        print(
            f"[layer_skipping] Warning: block at '{key}' is "
            f"{type(old_block).__name__}, not CIFARBasicBlock or Bottleneck - skipped."
        )
        return False


def apply_layer_skipping(model: nn.Module, disabled_layers: list[str]) -> None:
    """Replace specified residual blocks with skip-only blocks.

    Args:
        model: Model (optionally wrapped in ClassificationFeatureSelectionWrapper).
               If the model has a ``backbone`` attribute it is unwrapped automatically.
        disabled_layers: List of ``'layerN.B'`` keys, e.g. ``["layer1.0", "layer2.1"]``.
    """
    backbone = getattr(model, "backbone", model)
    applied = sum(_replace_block_with_skipped(backbone, k) for k in disabled_layers)
    if applied:
        _clear_module_cache(model)
    print(f"[layer_skipping] {applied}/{len(disabled_layers)} blocks disabled.")


def apply_layer_skipping_from_config(model: nn.Module, cfg: DictConfig) -> None:
    """Apply layer skipping from the ``layer_skipping`` config section.

    Args:
        model: Model to modify in-place.
        cfg: The ``layer_skipping`` sub-config (not the root config).
    """
    raw = OmegaConf.to_container(cfg.disabled_layers, resolve=True)
    if not isinstance(raw, list):
        raise ValueError(
            "layer_skipping.disabled_layers must be a list of 'layerN.B' strings."
        )
    apply_layer_skipping(model, [str(k) for k in raw])
