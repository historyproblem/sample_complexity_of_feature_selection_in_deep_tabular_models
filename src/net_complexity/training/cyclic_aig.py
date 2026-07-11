"""Cyclic AIG layer-dropping training loop.

Two layer-selection modes are supported via ``cyclic_layer_dropping.drop_mode``:

``threshold`` (default, legacy)
    Drop every layer whose mean validation gate probability is below
    ``g_prob_threshold``.  Converges when no layer falls below the threshold.

``param_budget``
    Dynamically select the subset of layers to drop each cycle so that the
    freed parameter count stays within ``max_param_fraction`` of the current
    model size.  Layers are considered in ascending g_prob order (most prunable
    first) and greedily added until the budget would be exceeded.  Converges
    when no candidate layer fits within the remaining budget.

Config section (``cyclic_layer_dropping``)::

    cyclic_layer_dropping:
      enabled: true
      max_cycles: 10

      # --- layer-selection mode ---
      drop_mode: threshold      # "threshold" (default) or "param_budget"

      # used when drop_mode: threshold
      g_prob_threshold: 0.1

      # used when drop_mode: param_budget
      max_param_fraction: 0.10  # remove at most 10 % of current params per cycle

      # --- training schedule ---
      aig_epochs: 200           # epochs per AIG cycle (overrides training_arguments.num_epochs)
      final_epochs: 200         # epochs for the final plain-training run
      use_plain_model_for_final: true
      disable_mlflow_for_cycles: false
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict

from .engine import run_training

_VALID_DROP_MODES = {"threshold", "param_budget"}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _extract_layer_g_probs(valid_metrics: Mapping[str, Any]) -> dict[str, float]:
    """Return {layer_name: g_prob} for every ``valid_g_prob_*layer*`` metric.

    Strips an optional ``backbone.`` prefix so that metric keys like
    ``valid_g_prob_backbone.layer2.0`` are stored as ``layer2.0``, matching
    the format expected by ``layer_skipping.disabled_layers``.
    """
    result: dict[str, float] = {}
    prefix = "valid_g_prob_"
    for key, value in valid_metrics.items():
        if not key.startswith(prefix):
            continue
        layer_name = key[len(prefix):]
        if layer_name.startswith("backbone."):
            layer_name = layer_name[len("backbone."):]
        if layer_name.startswith("layer"):
            result[layer_name] = float(value)
    return result


def _layers_to_drop(
    layer_probs: dict[str, float],
    threshold: float,
    already_disabled: list[str],
) -> list[str]:
    disabled_set = set(already_disabled)
    return [
        name
        for name, prob in sorted(layer_probs.items())
        if prob < threshold and name not in disabled_set
    ]


# ---------------------------------------------------------------------------
# Param-budget helpers
# ---------------------------------------------------------------------------

def _pruneable_param_counts(config: DictConfig) -> tuple[dict[str, int], int]:
    """Instantiate a temporary CPU model and return per-block freeable param counts.

    Freeable params = block's total params minus what ``PrunedBottleneck``
    would keep (only the downsample projection, if any).

    Args:
        config: Cycle config — must already have ``layer_skipping.disabled_layers``
                set to the current set of disabled layers.

    Returns:
        ``(freeable, total_params)`` where ``freeable`` maps ``'layerN.B'``
        to the number of parameters that would be freed by pruning that block.
    """
    from hydra.utils import instantiate as hydra_instantiate

    from net_complexity.models.cifar_resnet import CIFARBasicBlock
    from net_complexity.models.feature_selection import PrunedBottleneck, PrunedCIFARBasicBlock
    from net_complexity.models.layer_skipping import apply_layer_skipping_from_config
    from net_complexity.models.resnet import Bottleneck

    model = hydra_instantiate(config.model)

    layer_skipping_cfg = getattr(config, "layer_skipping", None)
    if layer_skipping_cfg is not None and bool(getattr(layer_skipping_cfg, "enabled", True)):
        apply_layer_skipping_from_config(model, layer_skipping_cfg)

    total_params = sum(p.numel() for p in model.parameters())
    backbone = getattr(model, "backbone", model)

    raw_disabled = OmegaConf.to_container(
        getattr(layer_skipping_cfg, "disabled_layers", []), resolve=True
    ) if layer_skipping_cfg is not None else []
    disabled_set = {str(k) for k in raw_disabled}

    freeable: dict[str, int] = {}
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        stage = getattr(backbone, stage_name, None)
        if stage is None:
            continue
        for block_idx, block in enumerate(stage):
            key = f"{stage_name}.{block_idx}"
            if key in disabled_set or isinstance(block, (PrunedBottleneck, PrunedCIFARBasicBlock)):
                continue
            block_params = sum(p.numel() for p in block.parameters())
            kept = 0
            if isinstance(block, Bottleneck):
                ds = getattr(block, "i_downsample", None)
                if ds is not None:
                    kept = sum(p.numel() for p in ds.parameters())
            elif isinstance(block, CIFARBasicBlock):
                sc = getattr(block, "shortcut", None)
                if sc is not None and not isinstance(sc, nn.Identity):
                    kept = sum(p.numel() for p in sc.parameters())
            freeable[key] = block_params - kept

    return freeable, total_params


def _layers_to_drop_by_param_budget(
    layer_probs: dict[str, float],
    already_disabled: list[str],
    freeable_counts: dict[str, int],
    total_params: int,
    max_param_fraction: float,
) -> list[str]:
    """Select layers to drop without exceeding ``max_param_fraction`` of params.

    Candidates are sorted by g_prob ascending (most prunable first).  Layers
    that individually exceed the remaining budget are skipped so that smaller
    layers can still fill the slot.

    Args:
        layer_probs: ``{layer_name: g_prob}`` from the last validation pass.
        already_disabled: Layers already pruned in previous cycles.
        freeable_counts: ``{layer_name: params_freed_if_pruned}`` from
                         ``_pruneable_param_counts``.
        total_params: Current total parameter count of the model.
        max_param_fraction: Upper bound on the fraction of params to remove
                            (e.g. 0.10 = at most 10 %).

    Returns:
        Sorted list of layer names to drop this cycle.
    """
    disabled_set = set(already_disabled)
    budget = int(total_params * max_param_fraction)

    candidates = sorted(
        (
            (name, prob)
            for name, prob in layer_probs.items()
            if name not in disabled_set and name in freeable_counts
        ),
        key=lambda x: x[1],  # ascending g_prob
    )

    selected: list[str] = []
    remaining = budget
    for name, _ in candidates:
        cost = freeable_counts[name]
        if cost <= remaining:
            selected.append(name)
            remaining -= cost

    return selected


# ---------------------------------------------------------------------------
# Config builders for each sub-run
# ---------------------------------------------------------------------------

def _set_run_name(cfg: DictConfig, run_name: str) -> None:
    OmegaConf.update(cfg, "run_history.run_name", run_name, merge=False, force_add=True)
    if hasattr(cfg, "mlflow") and hasattr(cfg.mlflow, "run_name"):
        OmegaConf.update(cfg, "mlflow.run_name", run_name, merge=False)


def _configure_run_history(cfg: DictConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.update(cfg, "run_history.use_hydra_output_dir", False, merge=False, force_add=True)
    OmegaConf.update(cfg, "run_history.root_dir", str(run_dir), merge=False, force_add=True)


def _set_disabled_layers(cfg: DictConfig, disabled_layers: list[str]) -> None:
    OmegaConf.update(cfg, "layer_skipping.enabled", True, merge=False, force_add=True)
    OmegaConf.update(cfg, "layer_skipping.disabled_layers", list(disabled_layers), merge=False, force_add=True)


def _build_cycle_config(
    base_config: DictConfig,
    cycle_idx: int,
    disabled_layers: list[str],
    output_root: Path,
) -> DictConfig:
    cfg = deepcopy(base_config)
    cyclic_cfg = cfg.cyclic_layer_dropping

    aig_epochs = getattr(cyclic_cfg, "aig_epochs", None)
    if aig_epochs is not None:
        OmegaConf.update(cfg, "training_arguments.num_epochs", int(aig_epochs), merge=False)

    _set_disabled_layers(cfg, disabled_layers)
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


def _build_final_config(
    base_config: DictConfig,
    disabled_layers: list[str],
    output_root: Path,
) -> DictConfig:
    cfg = deepcopy(base_config)
    cyclic_cfg = cfg.cyclic_layer_dropping

    final_epochs = getattr(cyclic_cfg, "final_epochs", None)
    if final_epochs is not None:
        OmegaConf.update(cfg, "training_arguments.num_epochs", int(final_epochs), merge=False)

    OmegaConf.update(cfg, "model.lambda_coef", 0.0, merge=False)
    OmegaConf.update(cfg, "training_arguments.adaptive_lambda.enabled", False, merge=False)

    if bool(getattr(cyclic_cfg, "use_plain_model_for_final", True)):
        if (
            hasattr(cfg, "model")
            and hasattr(cfg.model, "backbone")
            and "resnet_block" in cfg.model.backbone
        ):
            with open_dict(cfg):
                del cfg.model.backbone["resnet_block"]

    _set_disabled_layers(cfg, disabled_layers)
    _configure_run_history(cfg, output_root / "final")

    base_name = (
        (getattr(cfg.run_history, "run_name", None) if hasattr(cfg, "run_history") else None)
        or (getattr(cfg.mlflow, "run_name", None) if hasattr(cfg, "mlflow") else None)
        or "run"
    )
    _set_run_name(cfg, f"{base_name}_final")

    if hasattr(cfg, "mlflow"):
        OmegaConf.update(cfg, "mlflow.enabled", True, merge=False)

    return cfg


# ---------------------------------------------------------------------------
# Initial disabled_layers from config
# ---------------------------------------------------------------------------

def _read_initial_disabled_layers(config: DictConfig) -> list[str]:
    ls_cfg = getattr(config, "layer_skipping", None)
    if ls_cfg is None:
        return []
    raw = OmegaConf.to_container(
        getattr(ls_cfg, "disabled_layers", []), resolve=True
    )
    if not isinstance(raw, list):
        return []
    return [str(k) for k in raw]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cyclic_aig_training(
    config: DictConfig,
    output_root: Path,
    progress_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run cyclic AIG layer-dropping followed by a final plain training run.

    Args:
        config: Full Hydra config (must contain a ``cyclic_layer_dropping`` section).
        output_root: Directory under which per-cycle and final sub-run outputs
            are written (typically the Hydra run output dir).
        progress_context: Optional dict forwarded to ``run_training``.

    Returns:
        Dict with keys:
            ``num_cycles_completed``, ``final_disabled_layers``,
            ``cycle_results``, ``final_result``.
    """
    cyclic_cfg = config.cyclic_layer_dropping
    max_cycles = int(cyclic_cfg.max_cycles)
    drop_mode = str(getattr(cyclic_cfg, "drop_mode", "threshold"))

    if drop_mode not in _VALID_DROP_MODES:
        raise ValueError(
            f"cyclic_layer_dropping.drop_mode must be one of {_VALID_DROP_MODES}, "
            f"got {drop_mode!r}"
        )

    if drop_mode == "threshold":
        threshold = float(cyclic_cfg.g_prob_threshold)
    else:
        max_param_fraction = float(cyclic_cfg.max_param_fraction)

    disabled_layers: list[str] = _read_initial_disabled_layers(config)
    cycle_results: list[dict[str, Any]] = []

    _banner = "=" * 60

    for cycle in range(max_cycles):
        print(f"\n{_banner}")
        if drop_mode == "threshold":
            print(
                f"Cyclic AIG Training | cycle {cycle + 1}/{max_cycles}"
                f" | disabled={len(disabled_layers)}"
                f" | mode=threshold | threshold={threshold}"
            )
        else:
            print(
                f"Cyclic AIG Training | cycle {cycle + 1}/{max_cycles}"
                f" | disabled={len(disabled_layers)}"
                f" | mode=param_budget | max_param_fraction={max_param_fraction:.1%}"
            )
        if disabled_layers:
            print(f"  Currently disabled: {disabled_layers}")
        print(_banner)

        cycle_cfg = _build_cycle_config(
            base_config=config,
            cycle_idx=cycle,
            disabled_layers=disabled_layers,
            output_root=output_root,
        )
        result = run_training(cycle_cfg, progress_context=progress_context)
        cycle_results.append(result)

        valid_metrics = result.get("last_valid_metrics", {})
        layer_probs = _extract_layer_g_probs(valid_metrics)

        print(f"\nCycle {cycle + 1} | layer g_probs: {layer_probs}")

        if drop_mode == "threshold":
            new_drop = _layers_to_drop(layer_probs, threshold, disabled_layers)
            if not new_drop:
                print(
                    f"Cycle {cycle + 1} | no layers with g_prob < {threshold} — converged."
                )
                break
            print(f"Cycle {cycle + 1} | dropping {len(new_drop)} layer(s): {new_drop}")

        else:  # param_budget
            freeable_counts, total_params = _pruneable_param_counts(cycle_cfg)
            new_drop = _layers_to_drop_by_param_budget(
                layer_probs, disabled_layers, freeable_counts, total_params, max_param_fraction,
            )
            if not new_drop:
                print(
                    f"Cycle {cycle + 1} | no layers fit within "
                    f"{max_param_fraction:.1%} budget — converged."
                )
                break
            freed = sum(freeable_counts[n] for n in new_drop)
            print(
                f"Cycle {cycle + 1} | dropping {len(new_drop)} layer(s)"
                f" — {freed:,} params freed ({freed / total_params:.1%} of {total_params:,}): "
                f"{new_drop}"
            )

        disabled_layers = disabled_layers + new_drop

    print(f"\n{_banner}")
    print(f"Final Training | {len(disabled_layers)} layer(s) permanently disabled")
    if disabled_layers:
        print(f"  Disabled: {disabled_layers}")
    print(_banner)

    final_cfg = _build_final_config(
        base_config=config,
        disabled_layers=disabled_layers,
        output_root=output_root,
    )
    final_result = run_training(final_cfg, progress_context=progress_context)

    return {
        "num_cycles_completed": len(cycle_results),
        "final_disabled_layers": disabled_layers,
        "cycle_results": cycle_results,
        "final_result": final_result,
    }
