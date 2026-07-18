"""Utilities for disabling (skipping) residual blocks from config.

Two modes are supported and selected via ``layer_skipping.mode``:

``prune`` (default)
    The residual block is replaced with a parameter-free stub
    (``PrunedBottleneck`` / ``PrunedCIFARBasicBlock``).  Only the downsample
    projection is kept when the block changes spatial dimensions or channel
    count.  Model parameter count and memory footprint decrease.

``skip``
    The residual block is replaced with ``SkippedBottleneck`` /
    ``SkippedCIFARBasicBlock`` which inherits all conv/BN weights but
    short-circuits the forward pass.  Parameter count is unchanged; useful
    when checkpoint compatibility with the full architecture must be preserved.

Config format::

    layer_skipping:
      enabled: true
      mode: prune          # "prune" (default) or "skip"
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
from .feature_selection import (
    PrunedBottleneck,
    PrunedCIFARBasicBlock,
    SkippedBottleneck,
    SkippedCIFARBasicBlock,
)
from .resnet import Bottleneck

_VALID_MODES = {"prune", "skip"}


def _replace_block(backbone: nn.Module, key: str, mode: str) -> bool:
    """Navigate backbone to ``layerN.B`` and replace the block.

    Args:
        backbone: The bare backbone (without wrapper).
        key: Block address in ``'layerN.B'`` format.
        mode: ``'prune'`` to physically remove parameters;
              ``'skip'`` to keep weights but bypass forward.

    Returns:
        ``True`` on success, ``False`` if the key is invalid or the block
        type is unsupported.
    """
    parts = key.split(".")
    if len(parts) != 2:
        print(f"[layer_skipping] Warning: '{key}' must be 'layerN.B' format — ignored.")
        return False

    layer_name, block_idx_str = parts
    try:
        block_idx = int(block_idx_str)
    except ValueError:
        print(f"[layer_skipping] Warning: block index '{block_idx_str}' is not an integer — ignored.")
        return False

    layer = getattr(backbone, layer_name, None)
    if layer is None:
        print(f"[layer_skipping] Warning: '{layer_name}' not found in backbone — ignored.")
        return False

    if block_idx >= len(layer):
        print(
            f"[layer_skipping] Warning: index {block_idx} out of range for "
            f"'{layer_name}' (len={len(layer)}) — ignored."
        )
        return False

    old_block = layer[block_idx]
    if isinstance(old_block, CIFARBasicBlock):
        in_planes = old_block.conv1.in_channels
        planes = old_block.conv2.out_channels
        stride = old_block.conv1.stride[0]
        if mode == "prune":
            shortcut = old_block.shortcut if isinstance(old_block.shortcut, nn.Sequential) else None
            new_block = PrunedCIFARBasicBlock(shortcut=shortcut)
        else:
            option = "B" if isinstance(old_block.shortcut, nn.Sequential) else "A"
            new_block = SkippedCIFARBasicBlock(
                in_planes=in_planes, planes=planes, stride=stride, option=option
            )
        layer[block_idx] = new_block
        print(
            f"[layer_skipping] '{key}': {'pruned' if mode == 'prune' else 'skipped'} "
            f"(in_planes={in_planes}, planes={planes}, stride={stride})"
        )
        return True

    elif isinstance(old_block, Bottleneck):
        in_channels = old_block.conv1.in_channels
        out_channels = old_block.conv1.out_channels
        stride = old_block.stride
        if mode == "prune":
            new_block = PrunedBottleneck(i_downsample=old_block.i_downsample)
        else:
            new_block = SkippedBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                i_downsample=old_block.i_downsample,
                stride=stride,
            )
        layer[block_idx] = new_block
        print(
            f"[layer_skipping] '{key}': {'pruned' if mode == 'prune' else 'skipped'} "
            f"(in_channels={in_channels}, out_channels={out_channels}, stride={stride})"
        )
        return True

    else:
        print(
            f"[layer_skipping] Warning: block at '{key}' is "
            f"{type(old_block).__name__}, not CIFARBasicBlock or Bottleneck — ignored."
        )
        return False


def apply_layer_skipping(
    model: nn.Module,
    disabled_layers: list[str],
    mode: str = "prune",
) -> None:
    """Replace specified residual blocks according to ``mode``.

    Args:
        model: Model (optionally wrapped in ClassificationFeatureSelectionWrapper).
               If the model has a ``backbone`` attribute it is unwrapped automatically.
        disabled_layers: List of ``'layerN.B'`` keys, e.g. ``["layer1.0", "layer2.1"]``.
        mode: ``'prune'`` (default) to physically remove parameters;
              ``'skip'`` to keep weights but bypass forward.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"layer_skipping mode must be one of {_VALID_MODES}, got {mode!r}")
    backbone = getattr(model, "backbone", model)
    applied = sum(_replace_block(backbone, k, mode) for k in disabled_layers)
    verb = "pruned" if mode == "prune" else "skipped"
    print(f"[layer_skipping] {applied}/{len(disabled_layers)} blocks {verb}.")


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
    mode = str(getattr(cfg, "mode", "prune"))
    apply_layer_skipping(model, [str(k) for k in raw], mode=mode)
