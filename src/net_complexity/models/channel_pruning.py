"""Utilities for loading channel masks from prior Gumbel runs and applying them
to MaskedGumbelLayer buffers (soft masking) or building a structurally pruned
PrunedCIFARResNet (structural pruning) before a new training run."""

from __future__ import annotations

import csv
import gzip
import re
from collections import defaultdict
from pathlib import Path

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .feature_selection import MaskedGumbelLayer


def _collect_masked_layers(model: nn.Module) -> dict[str, MaskedGumbelLayer]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, MaskedGumbelLayer)
    }


def apply_channel_mask(model: nn.Module, mask_dict: dict[str, list[int]]) -> None:
    """Set channel_mask to 0 for disabled channels in every MaskedGumbelLayer.

    Args:
        model: Model containing MaskedGumbelLayer instances.
        mask_dict: {layer_name: [channel_indices_to_disable]}.
                   layer_name is matched by exact full path or by suffix.
    """
    masked_layers = _collect_masked_layers(model)
    if not masked_layers:
        print(
            "[channel_pruning] No MaskedGumbelLayer found in model. "
            "Use CIFARMaskedGumbelBasicBlock as the resnet_block type."
        )
        return

    applied = 0
    for target_name, channels in mask_dict.items():
        # Match by exact name or by trailing suffix (e.g. "gumbel_layer" matches
        # "backbone.layer1.0.gumbel_layer").
        match = next(
            (
                (full_name, layer)
                for full_name, layer in masked_layers.items()
                if full_name == target_name or full_name.endswith("." + target_name)
            ),
            None,
        )
        if match is None:
            print(f"[channel_pruning] Warning: no MaskedGumbelLayer matching '{target_name}'")
            continue

        full_name, layer = match
        n_ch = layer.channel_mask.shape[0]
        for ch in channels:
            if 0 <= ch < n_ch:
                layer.channel_mask[ch] = 0.0
            else:
                print(
                    f"[channel_pruning] Warning: channel {ch} out of range "
                    f"[0, {n_ch}) for layer '{full_name}'"
                )

        n_disabled = int((layer.channel_mask == 0).sum().item())
        print(f"[channel_pruning] '{full_name}': {n_disabled}/{n_ch} channels disabled")
        applied += 1

    if applied == 0:
        print("[channel_pruning] Warning: mask_dict had no matches. Check layer names.")


def build_mask_from_channel_history(
    run_dir: str | Path,
    threshold: float = 0.1,
    last_n_epochs: int | None = None,
) -> dict[str, list[int]]:
    """Build a channel disable mask from channel_history.csv.gz of a previous run.

    Reads per-channel selection_prob values and disables channels whose mean
    selection_prob (over the last last_n_epochs epochs, or all epochs if None)
    falls below threshold.

    Args:
        run_dir: Path to the previous run directory.
        threshold: Channels with mean selection_prob < threshold are disabled.
        last_n_epochs: Average only over the last N epochs. None = all epochs.

    Returns:
        {layer_name: [sorted list of channel indices to disable]}
    """
    run_dir = Path(run_dir)
    history_path = run_dir / "channel_history.csv.gz"
    if not history_path.exists():
        raise FileNotFoundError(
            f"channel_history.csv.gz not found in '{run_dir}'. "
            "Make sure the source run used run_history.log_channel_history=true."
        )

    # Accumulate (epoch, selection_prob) per (layer_name, channel_index)
    records: dict[str, dict[int, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with gzip.open(history_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_name = row["layer_name"]
            ch = int(row["channel_index"])
            prob = float(row["selection_prob"])
            epoch = int(row["epoch"])
            records[layer_name][ch].append((epoch, prob))

    mask_dict: dict[str, list[int]] = {}
    for layer_name, channel_data in records.items():
        disabled: list[int] = []
        for ch, epoch_probs in channel_data.items():
            pairs = sorted(epoch_probs, key=lambda t: t[0])
            if last_n_epochs is not None:
                pairs = pairs[-last_n_epochs:]
            mean_prob = sum(p for _, p in pairs) / len(pairs)
            if mean_prob < threshold:
                disabled.append(ch)
        if disabled:
            mask_dict[layer_name] = sorted(disabled)

    total = sum(len(v) for v in mask_dict.values())
    print(
        f"[channel_pruning] Loaded mask from '{run_dir}': "
        f"{len(mask_dict)} layers, {total} channels will be disabled "
        f"(threshold={threshold})"
    )
    return mask_dict


def apply_channel_mask_from_config(model: nn.Module, cfg: DictConfig) -> None:
    """Apply channel masking from the channel_pruning config section.

    Supported modes:
      threshold - read channel_history.csv.gz from source_run_dir and disable
                  channels with mean selection_prob < threshold.
      explicit  - specify mask as a dict {layer_name: [channel_indices]} in YAML.
    """
    mode = str(getattr(cfg, "mode", "threshold")).lower()

    if mode == "threshold":
        source_run_dir = str(cfg.source_run_dir)
        threshold = float(getattr(cfg, "threshold", 0.1))
        last_n_epochs = getattr(cfg, "last_n_epochs", None)
        last_n_epochs = int(last_n_epochs) if last_n_epochs is not None else None
        mask_dict = build_mask_from_channel_history(
            source_run_dir, threshold=threshold, last_n_epochs=last_n_epochs
        )

    elif mode == "explicit":
        raw = OmegaConf.to_container(cfg.mask, resolve=True)
        if not isinstance(raw, dict):
            raise ValueError(
                "channel_pruning.mask must be a dict mapping layer names to "
                "lists of channel indices."
            )
        mask_dict = {k: list(v) for k, v in raw.items()}

    else:
        raise ValueError(
            f"Unknown channel_pruning.mode: '{mode}'. Must be 'threshold' or 'explicit'."
        )

    apply_channel_mask(model, mask_dict)


# ---------------------------------------------------------------------------
# Structural pruning builds a new physically pruned model.
# ---------------------------------------------------------------------------

# channel_history layer name -> PrunedCIFARResNet pruning_spec key
# "backbone.layer2.0.gumbel_layer" -> "layer2.0"
_LAYER_NAME_RE = re.compile(
    r"(?:backbone\.)?(?P<key>layer\d+\.\d+)\.gumbel_layer$"
)

# Bottleneck-only: same idea, but also recognizes the optional mid1/mid2
# internal-width gates (MaskedGumbelBottleneckLayer(gate_internal_width=True),
# see feature_selection.py) alongside the "output" gate every
# MaskedGumbelBottleneckLayer has.
_BOTTLENECK_LAYER_NAME_RE = re.compile(
    r"(?:backbone\.)?(?P<key>layer\d+\.\d+)\.(?P<gate>gumbel_layer|mid1_gumbel_layer|mid2_gumbel_layer)$"
)

_BOTTLENECK_GATE_SPEC_KEY = {
    "gumbel_layer": "output",
    "mid1_gumbel_layer": "mid1",
    "mid2_gumbel_layer": "mid2",
}

_NUM_BLOCKS_BY_TARGET = {
    "CIFARResNet20": [3, 3, 3],
    "CIFARResNet32": [5, 5, 5],
    "CIFARResNet44": [7, 7, 7],
    "CIFARResNet56": [9, 9, 9],
    "CIFARResNet110": [18, 18, 18],
}

# Bottleneck-based ResNet50/101/152 (net_complexity.models.resnet.ResNet) —
# distinct from the CIFARResNet family above, which uses BasicBlock and a
# 3-stage layout.
_BOTTLENECK_LAYER_LIST_BY_TARGET = {
    "ResNet50": [3, 4, 6, 3],
    "ResNet101": [3, 4, 23, 3],
    "ResNet152": [3, 8, 36, 3],
}


def _mask_dict_to_pruning_spec(mask_dict: dict[str, list[int]]) -> dict[str, list[int]]:
    """Convert channel_history layer-name keys to PrunedCIFARResNet pruning_spec keys."""
    spec: dict[str, list[int]] = {}
    for layer_name, channels in mask_dict.items():
        m = _LAYER_NAME_RE.search(layer_name)
        if m:
            spec[m.group("key")] = channels
    return spec


def _mask_dict_to_bottleneck_pruning_spec(
    mask_dict: dict[str, list[int]],
) -> dict[str, dict[str, list[int]]]:
    """Convert channel_history layer-name keys to PrunedResNet's nested pruning_spec.

    Bottleneck-only counterpart of ``_mask_dict_to_pruning_spec``: recognizes
    all three MaskedGumbelBottleneckLayer gates (output/mid1/mid2, see
    ``_BOTTLENECK_LAYER_NAME_RE``) instead of only the output gate, and
    groups them per block into ``{"layerN.B": {"output": [...], "mid1":
    [...], "mid2": [...]}}`` (only keys with a non-empty channel list are
    included) so ``PrunedResNet``/``PrunedGumbelBottleneck`` can prune each
    boundary independently.
    """
    spec: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for layer_name, channels in mask_dict.items():
        m = _BOTTLENECK_LAYER_NAME_RE.search(layer_name)
        if m:
            spec[m.group("key")][_BOTTLENECK_GATE_SPEC_KEY[m.group("gate")]] = channels
    return dict(spec)


def _load_mask_dict(cfg: DictConfig) -> dict[str, list[int]]:
    """Shared mask loading logic (threshold or explicit)."""
    mode = str(getattr(cfg, "mode", "threshold")).lower()
    if mode == "threshold":
        source_run_dir = str(cfg.source_run_dir)
        threshold = float(getattr(cfg, "threshold", 0.1))
        last_n_epochs = getattr(cfg, "last_n_epochs", None)
        last_n_epochs = int(last_n_epochs) if last_n_epochs is not None else None
        return build_mask_from_channel_history(
            source_run_dir, threshold=threshold, last_n_epochs=last_n_epochs
        )
    elif mode == "explicit":
        raw = OmegaConf.to_container(cfg.mask, resolve=True)
        if not isinstance(raw, dict):
            raise ValueError(
                "channel_pruning.mask must be a dict mapping layer names to "
                "lists of channel indices."
            )
        return {k: list(v) for k, v in raw.items()}
    else:
        raise ValueError(
            f"Unknown channel_pruning.mode: '{mode}'. Must be 'threshold' or 'explicit'."
        )


def _is_bottleneck_backbone_target(backbone_target: str) -> bool:
    return any(name in backbone_target for name in _BOTTLENECK_LAYER_LIST_BY_TARGET)


def build_pruned_bottleneck_model(
    config: DictConfig,
    pruning_spec: dict[str, list[int] | dict[str, list[int]]],
) -> nn.Module:
    """Build a Bottleneck-based ``PrunedResNet`` wrapped in ``ClassificationFeatureSelectionWrapper``.

    Bottleneck (ResNet50/101/152) counterpart of the CIFARResNet path in
    ``build_structurally_pruned_model_from_config``. Unlike that function,
    this one takes the pruning spec directly (already in ``{"layerN.B":
    [channel_indices]}`` form) so callers that already hold it in memory —
    e.g. the iterative channel-pruning cycle — do not need to round-trip it
    through a mask file.
    """
    from .feature_selection import ClassificationFeatureSelectionWrapper
    from .pruned_bottleneck import PrunedResNet

    num_classes = int(OmegaConf.select(config, "model.backbone.num_classes") or 1000)
    in_channels = int(OmegaConf.select(config, "model.backbone.in_channels") or 3)
    stem_kernel_size = int(OmegaConf.select(config, "model.backbone.stem_kernel_size") or 7)
    stem_stride = int(OmegaConf.select(config, "model.backbone.stem_stride") or 2)
    stem_padding = int(OmegaConf.select(config, "model.backbone.stem_padding") or 3)
    use_maxpool_cfg = OmegaConf.select(config, "model.backbone.use_maxpool")
    use_maxpool = True if use_maxpool_cfg is None else bool(use_maxpool_cfg)

    backbone_target = str(OmegaConf.select(config, "model.backbone._target_") or "")
    layer_list = next(
        (v for k, v in _BOTTLENECK_LAYER_LIST_BY_TARGET.items() if k in backbone_target),
        None,
    )
    if layer_list is None:
        raise ValueError(
            "build_pruned_bottleneck_model: could not infer ResNet50/101/152 depth from "
            f"model.backbone._target_={backbone_target!r}."
        )

    backbone = PrunedResNet(
        pruning_spec=pruning_spec,
        layer_list=layer_list,
        num_classes=num_classes,
        in_channels=in_channels,
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        stem_padding=stem_padding,
        use_maxpool=use_maxpool,
    )

    lambda_coef = float(OmegaConf.select(config, "model.lambda_coef") or 0.0)
    criterion_cfg = OmegaConf.select(config, "model.criterion")
    criterion = instantiate(criterion_cfg) if criterion_cfg is not None else nn.CrossEntropyLoss()

    total_disabled = sum(len(v) for v in pruning_spec.values())
    print(
        f"[channel_pruning] Structural pruning applied (Bottleneck): "
        f"{len(pruning_spec)} blocks affected, "
        f"{total_disabled} channels removed from residual branches."
    )

    return ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=lambda_coef,
        criterion=criterion,
        regularization_loss=lambda m: 0,
    )


def _build_pruned_cifar_model(
    config: DictConfig,
    pruning_spec: dict[str, list[int]],
) -> nn.Module:
    from .feature_selection import ClassificationFeatureSelectionWrapper
    from .pruned_resnet import PrunedCIFARResNet

    num_classes = int(OmegaConf.select(config, "model.backbone.num_classes") or 10)
    in_channels = int(OmegaConf.select(config, "model.backbone.in_channels") or 3)
    shortcut_option = str(
        OmegaConf.select(config, "model.backbone.shortcut_option") or "A"
    )

    backbone_target = str(
        OmegaConf.select(config, "model.backbone._target_") or ""
    )
    num_blocks = next(
        (v for k, v in _NUM_BLOCKS_BY_TARGET.items() if k in backbone_target),
        [3, 3, 3],
    )

    backbone = PrunedCIFARResNet(
        pruning_spec=pruning_spec,
        num_blocks=num_blocks,
        num_classes=num_classes,
        in_channels=in_channels,
        shortcut_option=shortcut_option,
    )

    lambda_coef = float(OmegaConf.select(config, "model.lambda_coef") or 0.0)
    criterion_cfg = OmegaConf.select(config, "model.criterion")
    criterion = instantiate(criterion_cfg) if criterion_cfg is not None else nn.CrossEntropyLoss()

    total_disabled = sum(len(v) for v in pruning_spec.values())
    print(
        f"[channel_pruning] Structural pruning applied: "
        f"{len(pruning_spec)} blocks affected, "
        f"{total_disabled} channels removed from residual branches."
    )

    return ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=lambda_coef,
        criterion=criterion,
        regularization_loss=lambda m: 0,
    )


def build_structurally_pruned_model_from_config(
    config: DictConfig,
    cfg: DictConfig,
) -> nn.Module:
    """Build a physically channel-pruned model wrapped in ClassificationFeatureSelectionWrapper.

    Reads the channel mask (from channel_history file or explicit YAML),
    converts it to a per-block pruning_spec, and constructs a fresh model
    whose residual branches are physically narrowed to the active channels.
    Dispatches to the CIFARResNet (BasicBlock) or Bottleneck (ResNet50/101/152)
    builder based on ``model.backbone._target_``.

    The returned model has the same interface as the standard training model:
    forward(X, y) -> ClassifModelOutput.
    """
    mask_dict = _load_mask_dict(cfg)
    backbone_target = str(OmegaConf.select(config, "model.backbone._target_") or "")

    if _is_bottleneck_backbone_target(backbone_target):
        pruning_spec = _mask_dict_to_bottleneck_pruning_spec(mask_dict)
        if not pruning_spec:
            print(
                "[channel_pruning] WARNING: no prunable layers found in mask - "
                "the pruned model will be equivalent to the full model."
            )
        return build_pruned_bottleneck_model(config, pruning_spec)

    pruning_spec = _mask_dict_to_pruning_spec(mask_dict)
    if not pruning_spec:
        print(
            "[channel_pruning] WARNING: no prunable layers found in mask - "
            "the pruned model will be equivalent to the full model."
        )
    return _build_pruned_cifar_model(config, pruning_spec)
