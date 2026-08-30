"""Cyclic channel-dropping training loop — channel-granularity analogue of
``cyclic_aig.py``'s layer-dropping loop.

Where ``cyclic_aig`` iteratively drops whole residual *blocks* based on
per-block AIG gate probabilities, this module iteratively drops individual
*channels* within each block's residual branch, based on per-channel Gumbel
selector probabilities (``MaskedGumbelBottleneckLayer`` / ``MaskedGumbelLayer``).

Two channel-selection modes are supported via ``cyclic_channel_pruning.drop_mode``:

``threshold`` (default)
    Disable every channel whose mean validation selection probability is
    below ``g_prob_threshold``. Converges when no channel falls below the
    threshold. A block is never fully emptied — if every remaining channel of
    a block would be disabled, the single highest-probability candidate is
    kept active instead (matching ``PrunedGumbelBottleneck``'s requirement of
    at least one active channel).

    Optionally, ``threshold_max_param_fraction`` caps how many of the params
    freed by the threshold-selected channels may actually be dropped in a
    single cycle: the threshold-selected channels are ranked by ascending
    selection probability (most prunable first) and accepted greedily until
    the cap (as a fraction of the current model's total params) would be
    exceeded. A candidate that doesn't fit is skipped, not a stopping
    condition, so cheaper lower-ranked candidates can still be picked. Unset
    (default: ``null``) means no cap — every channel below threshold is
    dropped, as before.

``param_budget``
    Dynamically select the subset of channels to disable each cycle so that
    the freed parameter count stays within ``max_param_fraction`` of the
    current model size. Channels are considered in ascending probability
    order (most prunable first) and greedily added until the budget would be
    exceeded.

Each cycle re-uses the existing ``channel_pruning`` mechanism end to end:
  * search phase — soft masking (``channel_pruning.structural=false``,
    ``mode=explicit``) so gates on the *not-yet-disabled* channels stay live
    and trainable while already-disabled channels are permanently masked off
    via ``MaskedGumbelLayer.channel_mask`` (see ``models/channel_pruning.py``).
  * recovery phase — physical structural pruning
    (``channel_pruning.structural=true``) of every channel disabled so far,
    fine-tuned *without* Gumbel regularization (``lambda_coef=0``), run after
    *every* cycle's drop decision (not only at the end), so every pruning
    level in the trajectory gets a clean accuracy/compute reading. The last
    recovery pass uses ``final_epochs`` (falling back to ``recovery_epochs``).

With ``weight_handoff.enabled=true``, each structural recovery is initialized
by slicing the current gated checkpoint. Its fine-tuned surviving Conv/BN
tensors are then scattered into the full-width, permanently masked carrier of
the next search cycle. Gate logits also continue from the preceding search.
Optimizer state is deliberately rebuilt because parameter shapes differ.

Config section (``cyclic_channel_pruning``)::

    cyclic_channel_pruning:
      enabled: true
      max_cycles: 10

      drop_mode: threshold          # "threshold" (default) or "param_budget"
      g_prob_threshold: 0.1         # used when drop_mode: threshold
      threshold_max_param_fraction: null  # optional per-cycle cap, drop_mode: threshold only
      max_param_fraction: 0.10      # used when drop_mode: param_budget

      gumbel_epochs: 40             # epochs per search cycle
      recovery_epochs: 15           # epochs for each intermediate recovery pass
      final_epochs: 100             # epochs for the *last* recovery pass
      disable_mlflow_for_cycles: false
      weight_handoff:
        enabled: true
        checkpoint_name: best.pt
        initial_checkpoint: null     # optional dense/plain warm start
        initial_run_dir: null

Requires ``model.backbone.resnet_block`` to be
``net_complexity.wrappers.MaskedGumbelBottleneckLayer`` (see
``configs/method/gumbel_masked_bottleneck.yaml``) on a Bottleneck-based
backbone (ResNet50/101/152).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

from .cyclic_aig import _configure_run_history, _set_num_epochs, _set_run_name
from .engine import run_training

_VALID_DROP_MODES = {"threshold", "param_budget"}

# Metric key format: "valid_backbone.layer2.0.gumbel_layer.channel_007_zero_prob"
# (see metrics._channel_prob.ChannelZeroProbMetric with log_channel_zero_probs=true).
_CHANNEL_PROB_RE = re.compile(
    r"^valid_(?P<layer_name>.+)\.channel_(?P<channel_index>\d+)_zero_prob$"
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _extract_channel_probs(valid_metrics: Mapping[str, Any]) -> dict[str, dict[int, float]]:
    """Return {layer_name: {channel_index: selection_prob}} from valid_metrics.

    ``layer_name`` keeps the "backbone." prefix so it matches both
    ``apply_channel_mask``'s module-name lookup and
    ``build_structurally_pruned_model_from_config``'s pruning-spec regex
    directly, without any extra name translation.
    """
    result: dict[str, dict[int, float]] = defaultdict(dict)
    for key, value in valid_metrics.items():
        match = _CHANNEL_PROB_RE.match(key)
        if match is None:
            continue
        layer_name = match.group("layer_name")
        channel_index = int(match.group("channel_index"))
        zero_prob = float(value)
        result[layer_name][channel_index] = 1.0 - zero_prob
    return dict(result)


def _merge_disabled_channels(
    base: dict[str, list[int]],
    additional: dict[str, list[int]],
) -> dict[str, list[int]]:
    merged: dict[str, set[int]] = {
        layer_name: set(channels) for layer_name, channels in base.items()
    }
    for layer_name, channels in additional.items():
        merged.setdefault(layer_name, set()).update(channels)
    return {layer_name: sorted(channels) for layer_name, channels in merged.items() if channels}


def _channels_to_drop(
    channel_probs: dict[str, dict[int, float]],
    threshold: float,
    already_disabled: dict[str, list[int]],
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for layer_name, probs in channel_probs.items():
        disabled_here = set(already_disabled.get(layer_name, []))
        candidates = sorted(
            ch for ch, prob in probs.items() if prob < threshold and ch not in disabled_here
        )
        if not candidates:
            continue

        total_channels = len(probs)
        remaining_active = total_channels - len(disabled_here) - len(candidates)
        if remaining_active < 1:
            # Never fully empty a block's residual branch — keep the
            # highest-probability candidate(s) active instead.
            keep_count = 1 - remaining_active
            candidates_by_prob_desc = sorted(candidates, key=lambda ch: probs[ch], reverse=True)
            keep = set(candidates_by_prob_desc[:keep_count])
            candidates = [ch for ch in candidates if ch not in keep]

        if candidates:
            result[layer_name] = candidates
    return result


def _cap_channels_to_drop_by_param_budget(
    model,
    channels_to_drop: dict[str, list[int]],
    channel_probs: dict[str, dict[int, float]],
    total_params: int,
    max_param_fraction: float,
) -> dict[str, list[int]]:
    """Cap an already threshold-selected drop set to a per-cycle param budget.

    Ranks only the channels already selected by ``_channels_to_drop`` (i.e.
    those below ``g_prob_threshold``) by ascending selection probability
    (most prunable first) and greedily keeps them while the freed param cost
    fits within ``max_param_fraction`` of ``total_params``. A candidate that
    doesn't fit is skipped, not a stopping condition, so cheaper candidates
    further down the ranking can still be kept.
    """
    budget = int(total_params * max_param_fraction)
    candidates: list[tuple[str, int, float, int]] = []
    for layer_name, channels in channels_to_drop.items():
        cost = _channel_param_cost(model, layer_name)
        probs = channel_probs[layer_name]
        for channel_index in channels:
            candidates.append((layer_name, channel_index, probs[channel_index], cost))

    candidates.sort(key=lambda item: item[2])  # ascending probability

    remaining = budget
    selected: dict[str, list[int]] = defaultdict(list)
    for layer_name, channel_index, _prob, cost in candidates:
        if cost > remaining:
            continue
        selected[layer_name].append(channel_index)
        remaining -= cost

    return {layer_name: sorted(channels) for layer_name, channels in selected.items()}


# Suffix of a MaskedGumbelBottleneckLayer gate's module path -> which
# boundary it controls (see feature_selection.py's MaskedGumbelBottleneckLayer
# docstring). The three suffixes are mutually exclusive (each is preceded by
# a "." separator in the module path, so ".gumbel_layer" never matches a
# "...mid1_gumbel_layer"/"...mid2_gumbel_layer" path), so check order here
# doesn't matter.
_GATE_SUFFIXES = (".gumbel_layer", ".mid1_gumbel_layer", ".mid2_gumbel_layer")


def _resolve_block(model, block_path: str):
    block = model
    for part in block_path.split("."):
        block = block[int(part)] if part.isdigit() else getattr(block, part)
    return block


def _channel_param_cost(model, layer_name: str) -> int:
    """Approximate parameter cost of disabling one channel of ``layer_name``.

    ``layer_name`` is a MaskedGumbelBottleneckLayer gate's module path (e.g.
    "backbone.layer2.0.gumbel_layer", "...mid1_gumbel_layer", or
    "...mid2_gumbel_layer") — each gate sits on a different boundary of the
    block and therefore frees weights from a different pair of layers when a
    channel is physically pruned (see MaskedGumbelBottleneckLayer's
    docstring for what each boundary is):

      ``.gumbel_layer`` (output, conv3's output before the identity sum):
        one freed output row of conv3 (weight + bias) + batch_norm3 affine.
      ``.mid1_gumbel_layer`` (conv1-output / conv2-input boundary):
        one freed output row of conv1 (weight + bias) + batch_norm1 affine,
        plus the corresponding input column of conv2 (weight only — conv2's
        bias/batch_norm2 are per *output* channel, unaffected by narrowing
        its input).
      ``.mid2_gumbel_layer`` (conv2-output / conv3-input boundary):
        one freed output row of conv2 (weight + bias) + batch_norm2 affine,
        plus the corresponding input column of conv3 (weight only, same
        reasoning as above).
    """
    for suffix in _GATE_SUFFIXES:
        if layer_name.endswith(suffix):
            block = _resolve_block(model, layer_name[: -len(suffix)])
            break
    else:
        raise ValueError(
            f"_channel_param_cost: unrecognized channel-gate module path {layer_name!r} "
            f"(expected one of the suffixes {_GATE_SUFFIXES})."
        )

    if suffix == ".gumbel_layer":
        conv3 = block.conv3
        cost = int(conv3.in_channels)  # weight: [out, in, 1, 1] -> in per out-channel
        if conv3.bias is not None:
            cost += 1
        cost += 2  # batch_norm3 weight + bias per channel
        return cost

    if suffix == ".mid1_gumbel_layer":
        conv1, conv2 = block.conv1, block.conv2
        kh, kw = conv2.kernel_size
        cost = int(conv1.in_channels)  # conv1's freed output row
        if conv1.bias is not None:
            cost += 1
        cost += 2  # batch_norm1
        cost += int(conv2.out_channels) * kh * kw  # conv2's freed input column
        return cost

    # ".mid2_gumbel_layer"
    conv2, conv3 = block.conv2, block.conv3
    kh, kw = conv2.kernel_size
    cost = int(conv2.in_channels) * kh * kw  # conv2's freed output row
    if conv2.bias is not None:
        cost += 1
    cost += 2  # batch_norm2
    cost += int(conv3.out_channels)  # conv3's freed input column (1x1 conv)
    return cost


def _channels_to_drop_by_param_budget(
    model,
    channel_probs: dict[str, dict[int, float]],
    already_disabled: dict[str, list[int]],
    total_params: int,
    max_param_fraction: float,
) -> dict[str, list[int]]:
    budget = int(total_params * max_param_fraction)
    candidates: list[tuple[str, int, float, int]] = []  # (layer, channel, prob, cost)
    for layer_name, probs in channel_probs.items():
        disabled_here = set(already_disabled.get(layer_name, []))
        cost = _channel_param_cost(model, layer_name)
        for channel_index, prob in probs.items():
            if channel_index in disabled_here:
                continue
            candidates.append((layer_name, channel_index, prob, cost))

    candidates.sort(key=lambda item: item[2])  # ascending probability

    remaining = budget
    selected: dict[str, list[int]] = defaultdict(list)
    remaining_active: dict[str, int] = {
        layer_name: len(probs) - len(already_disabled.get(layer_name, []))
        for layer_name, probs in channel_probs.items()
    }
    for layer_name, channel_index, _prob, cost in candidates:
        if cost > remaining:
            continue
        if remaining_active[layer_name] <= 1:
            continue  # keep at least one active channel per block
        selected[layer_name].append(channel_index)
        remaining_active[layer_name] -= 1
        remaining -= cost

    return {layer_name: sorted(channels) for layer_name, channels in selected.items()}


# ---------------------------------------------------------------------------
# Config builders for each sub-run
# ---------------------------------------------------------------------------

def _set_disabled_channels_soft(cfg: DictConfig, disabled_channels: dict[str, list[int]]) -> None:
    OmegaConf.update(cfg, "channel_pruning.enabled", True, merge=False, force_add=True)
    OmegaConf.update(cfg, "channel_pruning.mode", "explicit", merge=False, force_add=True)
    OmegaConf.update(cfg, "channel_pruning.structural", False, merge=False, force_add=True)
    OmegaConf.update(cfg, "channel_pruning.mask", dict(disabled_channels), merge=False, force_add=True)


def _set_disabled_channels_structural(
    cfg: DictConfig, disabled_channels: dict[str, list[int]]
) -> None:
    OmegaConf.update(cfg, "channel_pruning.enabled", True, merge=False, force_add=True)
    OmegaConf.update(cfg, "channel_pruning.mode", "explicit", merge=False, force_add=True)
    OmegaConf.update(cfg, "channel_pruning.structural", True, merge=False, force_add=True)
    OmegaConf.update(cfg, "channel_pruning.mask", dict(disabled_channels), merge=False, force_add=True)


def _build_search_cycle_config(
    base_config: DictConfig,
    cycle_idx: int,
    disabled_channels: dict[str, list[int]],
    output_root: Path,
) -> DictConfig:
    cfg = deepcopy(base_config)
    cyclic_cfg = cfg.cyclic_channel_pruning

    gumbel_epochs = getattr(cyclic_cfg, "gumbel_epochs", None)
    if gumbel_epochs is not None:
        _set_num_epochs(cfg, int(gumbel_epochs))

    _set_disabled_channels_soft(cfg, disabled_channels)
    _configure_run_history(cfg, output_root / f"cycle_{cycle_idx}")

    base_name = (
        (getattr(cfg.run_history, "run_name", None) if hasattr(cfg, "run_history") else None)
        or (getattr(cfg.mlflow, "run_name", None) if hasattr(cfg, "mlflow") else None)
        or "run"
    )
    _set_run_name(cfg, f"{base_name}_cycle_{cycle_idx}")

    if bool(getattr(cyclic_cfg, "disable_mlflow_for_cycles", True)):
        if hasattr(cfg, "mlflow"):
            OmegaConf.update(cfg, "mlflow.enabled", False, merge=False)

    return cfg


def _run_recovery_pass(
    base_config: DictConfig,
    disabled_channels: dict[str, list[int]],
    output_root: Path,
    *,
    cycle_idx: int,
    is_final: bool,
    progress_context: dict[str, Any] | None,
    model_initializer: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    cyclic_cfg = base_config.cyclic_channel_pruning
    stage_name = "final" if is_final else f"cycle_{cycle_idx}_recovery"

    epochs = None
    if is_final:
        epochs = getattr(cyclic_cfg, "final_epochs", None)
    if epochs is None:
        epochs = getattr(cyclic_cfg, "recovery_epochs", None)

    cfg = deepcopy(base_config)
    if epochs is not None:
        _set_num_epochs(cfg, int(epochs))

    OmegaConf.update(cfg, "model.lambda_coef", 0.0, merge=False)
    OmegaConf.update(cfg, "training_arguments.adaptive_lambda.enabled", False, merge=False)
    _set_disabled_channels_structural(cfg, disabled_channels)
    _configure_run_history(cfg, output_root / stage_name)

    base_name = (
        (getattr(cfg.run_history, "run_name", None) if hasattr(cfg, "run_history") else None)
        or (getattr(cfg.mlflow, "run_name", None) if hasattr(cfg, "mlflow") else None)
        or "run"
    )
    _set_run_name(cfg, f"{base_name}_{stage_name}")

    if hasattr(cfg, "mlflow"):
        OmegaConf.update(cfg, "mlflow.enabled", True, merge=False)

    if model_initializer is None:
        return run_training(cfg, progress_context=progress_context)
    return run_training(
        cfg,
        progress_context=progress_context,
        model_initializer=model_initializer,
    )


def _read_initial_disabled_channels(config: DictConfig) -> dict[str, list[int]]:
    channel_pruning_cfg = getattr(config, "channel_pruning", None)
    if channel_pruning_cfg is None or not bool(getattr(channel_pruning_cfg, "enabled", False)):
        return {}
    mask = getattr(channel_pruning_cfg, "mask", None)
    if mask is None:
        return {}
    raw = OmegaConf.to_container(mask, resolve=True)
    if not isinstance(raw, dict):
        return {}
    return {str(k): sorted(int(ch) for ch in v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Cross-cycle model-weight handoff.
# ---------------------------------------------------------------------------

def _weight_handoff_config(config: DictConfig) -> DictConfig | None:
    cyclic_cfg = config.cyclic_channel_pruning
    handoff_cfg = getattr(cyclic_cfg, "weight_handoff", None)
    if handoff_cfg is None or not bool(getattr(handoff_cfg, "enabled", False)):
        return None
    return handoff_cfg


def _handoff_checkpoint_name(config: DictConfig) -> str:
    handoff_cfg = _weight_handoff_config(config)
    return str(getattr(handoff_cfg, "checkpoint_name", "best.pt")) if handoff_cfg else "best.pt"


def _result_checkpoint_path(result: Mapping[str, Any], checkpoint_name: str) -> Path:
    run_dir = result.get("run_dir")
    if not run_dir:
        raise ValueError(
            "Cyclic channel weight handoff requires run_training() to return run_dir."
        )
    checkpoint_path = Path(str(run_dir)) / "checkpoints" / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Weight-handoff checkpoint was not created: {checkpoint_path}"
        )
    return checkpoint_path


def _checkpoint_model_state(checkpoint_path: Path) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError(
            f"Checkpoint {checkpoint_path} does not contain a model_state_dict mapping."
        )
    return state


def _load_checkpoint_strict(model, checkpoint_path: Path) -> None:
    model.load_state_dict(_checkpoint_model_state(checkpoint_path), strict=True)


def _resolve_initial_checkpoint(config: DictConfig) -> Path | None:
    handoff_cfg = _weight_handoff_config(config)
    if handoff_cfg is None:
        return None
    explicit = getattr(handoff_cfg, "initial_checkpoint", None)
    if explicit:
        checkpoint_path = Path(str(explicit))
    else:
        initial_run_dir = getattr(handoff_cfg, "initial_run_dir", None)
        if not initial_run_dir:
            return None
        checkpoint_path = (
            Path(str(initial_run_dir))
            / "checkpoints"
            / _handoff_checkpoint_name(config)
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Initial cyclic channel checkpoint not found: {checkpoint_path}"
        )
    return checkpoint_path


def _initial_checkpoint_initializer(checkpoint_path: Path) -> Callable[[Any], None]:
    state = _checkpoint_model_state(checkpoint_path)

    def initialize(model) -> None:
        incompatible = model.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing_non_gate = [
            key for key in incompatible.missing_keys if "gumbel_layer" not in key
        ]
        if unexpected or missing_non_gate:
            raise ValueError(
                "Initial checkpoint is not compatible with the gated ResNet: "
                f"unexpected={unexpected[:5]}, missing_non_gate={missing_non_gate[:5]}."
            )
        print(
            f"[cyclic_channel_pruning] Warm-started gated search from {checkpoint_path} "
            f"({len(incompatible.missing_keys)} gate tensor(s) initialized from config)."
        )

    return initialize


def _search_to_structural_initializer(
    search_config: DictConfig,
    search_checkpoint: Path,
) -> Callable[[Any], None]:
    from net_complexity.models.channel_pruning import transfer_gated_weights_to_structural

    source_model = hydra_instantiate(search_config.model)
    _load_checkpoint_strict(source_model, search_checkpoint)

    def initialize(structural_model) -> None:
        transfer_gated_weights_to_structural(source_model, structural_model)
        print(
            "[cyclic_channel_pruning] Initialized structural recovery with "
            f"surviving weights from {search_checkpoint}."
        )

    return initialize


def _structural_model_from_checkpoint(
    base_config: DictConfig,
    disabled_channels: dict[str, list[int]],
    checkpoint_path: Path,
):
    from net_complexity.models.channel_pruning import (
        build_structurally_pruned_model_from_config,
    )

    structural_cfg = deepcopy(base_config)
    _set_disabled_channels_structural(structural_cfg, disabled_channels)
    model = build_structurally_pruned_model_from_config(
        structural_cfg,
        structural_cfg.channel_pruning,
    )
    _load_checkpoint_strict(model, checkpoint_path)
    return model


def _next_search_initializer(
    base_config: DictConfig,
    disabled_channels: dict[str, list[int]],
    previous_search_checkpoint: Path,
    previous_recovery_checkpoint: Path,
) -> Callable[[Any], None]:
    from net_complexity.models.channel_pruning import transfer_structural_weights_to_gated

    recovered_model = _structural_model_from_checkpoint(
        base_config,
        disabled_channels,
        previous_recovery_checkpoint,
    )
    previous_search_state = _checkpoint_model_state(previous_search_checkpoint)

    def initialize(gated_model) -> None:
        # Restore gate logits and the full-width carrier first, then overwrite
        # every surviving Conv/BN tensor with its fine-tuned recovery value.
        gated_model.load_state_dict(previous_search_state, strict=True)
        transfer_structural_weights_to_gated(recovered_model, gated_model)
        print(
            "[cyclic_channel_pruning] Carried recovered surviving weights into "
            "the next gated search cycle."
        )

    return initialize


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cyclic_channel_pruning_training(
    config: DictConfig,
    output_root: Path,
    progress_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run cyclic channel-dropping followed by per-cycle recovery passes.

    Returns a dict with the same shape as ``cyclic_aig.run_cyclic_aig_training``:
    ``num_cycles_completed``, ``final_disabled_channels``, ``cycle_results``,
    ``recovery_results``, ``final_result``, ``training_cost``.
    """
    cyclic_cfg = config.cyclic_channel_pruning
    max_cycles = int(cyclic_cfg.max_cycles)
    drop_mode = str(getattr(cyclic_cfg, "drop_mode", "threshold"))

    if drop_mode not in _VALID_DROP_MODES:
        raise ValueError(
            f"cyclic_channel_pruning.drop_mode must be one of {_VALID_DROP_MODES}, "
            f"got {drop_mode!r}"
        )

    threshold_max_param_fraction: float | None = None
    if drop_mode == "threshold":
        threshold = float(cyclic_cfg.g_prob_threshold)
        raw_cap = getattr(cyclic_cfg, "threshold_max_param_fraction", None)
        if raw_cap is not None:
            threshold_max_param_fraction = float(raw_cap)
    else:
        max_param_fraction = float(cyclic_cfg.max_param_fraction)

    disabled_channels: dict[str, list[int]] = _read_initial_disabled_channels(config)
    cycle_results: list[dict[str, Any]] = []
    recovery_results: list[dict[str, Any]] = []
    handoff_enabled = _weight_handoff_config(config) is not None
    checkpoint_name = _handoff_checkpoint_name(config)
    initial_checkpoint = _resolve_initial_checkpoint(config)
    previous_search_checkpoint: Path | None = None
    previous_recovery_checkpoint: Path | None = None

    _banner = "=" * 60

    for cycle in range(max_cycles):
        total_disabled = sum(len(v) for v in disabled_channels.values())
        cap_suffix = (
            f" | cap={threshold_max_param_fraction:.1%}"
            if threshold_max_param_fraction is not None
            else ""
        )
        print(f"\n{_banner}")
        print(
            f"Cyclic Channel Pruning | cycle {cycle + 1}/{max_cycles}"
            f" | disabled_channels={total_disabled} | mode={drop_mode}{cap_suffix}"
        )
        print(_banner)

        cycle_cfg = _build_search_cycle_config(config, cycle, disabled_channels, output_root)
        search_initializer = None
        if handoff_enabled:
            if previous_search_checkpoint is not None and previous_recovery_checkpoint is not None:
                search_initializer = _next_search_initializer(
                    config,
                    disabled_channels,
                    previous_search_checkpoint,
                    previous_recovery_checkpoint,
                )
            elif initial_checkpoint is not None:
                search_initializer = _initial_checkpoint_initializer(initial_checkpoint)

        if search_initializer is None:
            result = run_training(cycle_cfg, progress_context=progress_context)
        else:
            result = run_training(
                cycle_cfg,
                progress_context=progress_context,
                model_initializer=search_initializer,
            )
        cycle_results.append(result)
        search_checkpoint = (
            _result_checkpoint_path(result, checkpoint_name)
            if handoff_enabled
            else None
        )

        valid_metrics = result.get("last_valid_metrics", {})
        channel_probs = _extract_channel_probs(valid_metrics)

        if drop_mode == "threshold":
            new_drop = _channels_to_drop(channel_probs, threshold, disabled_channels)
            if threshold_max_param_fraction is not None and new_drop:
                probe_model = hydra_instantiate(cycle_cfg.model)
                total_params = sum(p.numel() for p in probe_model.parameters())
                capped = _cap_channels_to_drop_by_param_budget(
                    probe_model, new_drop, channel_probs, total_params, threshold_max_param_fraction,
                )
                num_before = sum(len(v) for v in new_drop.values())
                num_after = sum(len(v) for v in capped.values())
                if num_after < num_before:
                    print(
                        f"Cycle {cycle + 1} | threshold selected {num_before} channel(s), "
                        f"capped to {num_after} by threshold_max_param_fraction="
                        f"{threshold_max_param_fraction:.1%}."
                    )
                new_drop = capped
        else:
            probe_model = hydra_instantiate(cycle_cfg.model)
            total_params = sum(p.numel() for p in probe_model.parameters())
            new_drop = _channels_to_drop_by_param_budget(
                probe_model, channel_probs, disabled_channels, total_params, max_param_fraction,
            )

        num_new = sum(len(v) for v in new_drop.values())
        converged = num_new == 0
        if converged:
            print(f"Cycle {cycle + 1} | no channels selected for pruning — converged.")
        else:
            print(f"Cycle {cycle + 1} | dropping {num_new} channel(s) across {len(new_drop)} block(s).")
            disabled_channels = _merge_disabled_channels(disabled_channels, new_drop)

        is_final_recovery = converged or cycle == max_cycles - 1
        total_disabled = sum(len(v) for v in disabled_channels.values())
        print(
            f"\nCycle {cycle + 1} | recovery pass (structural, no regularization)"
            f" | disabled_channels={total_disabled}"
            f"{' | final' if is_final_recovery else ''}"
        )
        recovery_initializer = (
            _search_to_structural_initializer(cycle_cfg, search_checkpoint)
            if search_checkpoint is not None
            else None
        )
        recovery_result = _run_recovery_pass(
            config,
            disabled_channels,
            output_root,
            cycle_idx=cycle,
            is_final=is_final_recovery,
            progress_context=progress_context,
            model_initializer=recovery_initializer,
        )
        recovery_results.append(recovery_result)
        if handoff_enabled:
            previous_search_checkpoint = search_checkpoint
            previous_recovery_checkpoint = _result_checkpoint_path(
                recovery_result,
                checkpoint_name,
            )

        if converged:
            break

    final_result = recovery_results[-1]

    print(f"\n{_banner}")
    total_disabled = sum(len(v) for v in disabled_channels.values())
    print(f"Final Training | {total_disabled} channel(s) permanently disabled")
    print(_banner)

    cycle_train_time_sec = sum(float(r.get("full_train_time_sec", 0.0)) for r in cycle_results)
    recovery_train_time_sec = sum(float(r.get("full_train_time_sec", 0.0)) for r in recovery_results)
    final_retrain_time_sec = float(final_result.get("full_train_time_sec", 0.0))
    full_train_time_sec = cycle_train_time_sec + recovery_train_time_sec
    num_epochs_executed = sum(
        int(r.get("num_epochs_executed", 0)) for r in (*cycle_results, *recovery_results)
    )
    baseline_costs: dict[str, Mapping[str, Any]] = {}
    for sub_result in [*cycle_results, *recovery_results]:
        baseline = sub_result.get("adaptive_lambda_baseline")
        if isinstance(baseline, Mapping):
            baseline_costs[str(baseline.get("history_path"))] = baseline
    one_time_baseline_cost_sec = sum(
        float(b.get("full_train_time_sec", 0.0)) for b in baseline_costs.values()
    )
    one_time_baseline_epochs = sum(
        int(b.get("num_epochs_executed", 0)) for b in baseline_costs.values()
    )

    cyclic_result = {
        "num_cycles_completed": len(cycle_results),
        "final_disabled_channels": disabled_channels,
        "cycle_results": cycle_results,
        "recovery_results": recovery_results,
        "final_result": final_result,
        "weight_handoff": {
            "enabled": handoff_enabled,
            "checkpoint_name": checkpoint_name,
            "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
            "optimizer_state_transferred": False,
        },
        "training_cost": {
            "cycle_train_time_sec": cycle_train_time_sec,
            "recovery_train_time_sec": recovery_train_time_sec,
            "final_retrain_time_sec": final_retrain_time_sec,
            "full_train_time_sec": full_train_time_sec,
            "num_epochs_executed": num_epochs_executed,
            "one_time_baseline_cost_sec": one_time_baseline_cost_sec,
            "one_time_baseline_epochs": one_time_baseline_epochs,
            "baseline_included_in_full_train_time": False,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cyclic_summary.json").write_text(
        json.dumps(cyclic_result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return cyclic_result
