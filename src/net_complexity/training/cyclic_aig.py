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

After every cycle's drop decision (converged or not), the current
architecture is retrained *without* AIG regularization ("recovery pass") —
not only once at the very end. This gives every pruning level in the
trajectory a clean, regularization-free accuracy/compute reading, instead of
only the final one. The last recovery pass (on convergence or once
``max_cycles`` is reached) plays the role of the previous "final" run and
uses ``final_epochs`` if set.

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
      recovery_epochs: 30       # epochs for each intermediate no-regularization recovery pass
      final_epochs: 200         # epochs for the *last* recovery pass (falls back to recovery_epochs)
      use_plain_model_for_final: true
      disable_mlflow_for_cycles: false
"""
from __future__ import annotations

import json
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


def _set_num_epochs(cfg: DictConfig, num_epochs: int) -> None:
    OmegaConf.update(cfg, "training_arguments.num_epochs", int(num_epochs), merge=False)
    if OmegaConf.select(cfg, "scheduler.T_max") is not None:
        OmegaConf.update(cfg, "scheduler.T_max", int(num_epochs), merge=False)


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


def _build_recovery_config(
    base_config: DictConfig,
    disabled_layers: list[str],
    output_root: Path,
    *,
    stage_name: str,
    epochs: int | None,
) -> DictConfig:
    """Build a no-regularization ("recovery") sub-run config for the given architecture.

    Used both after every intermediate cycle (to report an accuracy/compute
    operating point for that pruning level without AIG regularization noise)
    and for the final polish run at the end of the loop (``stage_name="final"``).
    """
    cfg = deepcopy(base_config)
    cyclic_cfg = cfg.cyclic_layer_dropping

    if epochs is not None:
        _set_num_epochs(cfg, int(epochs))

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
    _configure_run_history(cfg, output_root / stage_name)

    base_name = (
        (getattr(cfg.run_history, "run_name", None) if hasattr(cfg, "run_history") else None)
        or (getattr(cfg.mlflow, "run_name", None) if hasattr(cfg, "mlflow") else None)
        or "run"
    )
    _set_run_name(cfg, f"{base_name}_{stage_name}")

    # Recovery/final runs are always logged, regardless of disable_mlflow_for_cycles
    # (that flag only suppresses the noisy AIG search cycles).
    if hasattr(cfg, "mlflow"):
        OmegaConf.update(cfg, "mlflow.enabled", True, merge=False)

    return cfg


def _run_recovery_pass(
    base_config: DictConfig,
    disabled_layers: list[str],
    output_root: Path,
    *,
    cycle_idx: int,
    is_final: bool,
    progress_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Train the current architecture without AIG regularization ("recovery").

    Runs after every cycle's drop decision — not only at the very end — so
    that each pruning level in the trajectory gets a clean, regularization-free
    accuracy/compute measurement. ``recovery_epochs`` controls the budget for
    intermediate cycles; the last recovery pass (``is_final=True``) uses
    ``final_epochs`` when set, falling back to ``recovery_epochs`` otherwise,
    for a possibly longer final polish.
    """
    cyclic_cfg = base_config.cyclic_layer_dropping
    stage_name = "final" if is_final else f"cycle_{cycle_idx}_recovery"

    epochs = None
    if is_final:
        epochs = getattr(cyclic_cfg, "final_epochs", None)
    if epochs is None:
        epochs = getattr(cyclic_cfg, "recovery_epochs", None)

    recovery_cfg = _build_recovery_config(
        base_config,
        disabled_layers,
        output_root,
        stage_name=stage_name,
        epochs=epochs,
    )
    return run_training(recovery_cfg, progress_context=progress_context)


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
    recovery_results: list[dict[str, Any]] = []

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
            converged = not new_drop
            if converged:
                print(
                    f"Cycle {cycle + 1} | no layers with g_prob < {threshold} — converged."
                )
            else:
                print(f"Cycle {cycle + 1} | dropping {len(new_drop)} layer(s): {new_drop}")

        else:  # param_budget
            freeable_counts, total_params = _pruneable_param_counts(cycle_cfg)
            new_drop = _layers_to_drop_by_param_budget(
                layer_probs, disabled_layers, freeable_counts, total_params, max_param_fraction,
            )
            converged = not new_drop
            if converged:
                print(
                    f"Cycle {cycle + 1} | no layers fit within "
                    f"{max_param_fraction:.1%} budget — converged."
                )
            else:
                freed = sum(freeable_counts[n] for n in new_drop)
                print(
                    f"Cycle {cycle + 1} | dropping {len(new_drop)} layer(s)"
                    f" — {freed:,} params freed ({freed / total_params:.1%} of {total_params:,}): "
                    f"{new_drop}"
                )

        if not converged:
            disabled_layers = disabled_layers + new_drop

        # Recovery pass: retrain the current (post-drop) architecture without AIG
        # regularization, every cycle — not only at the very end — so each
        # pruning level in the trajectory gets a clean accuracy/compute reading.
        is_final_recovery = converged or cycle == max_cycles - 1
        print(
            f"\nCycle {cycle + 1} | recovery pass (no regularization)"
            f" | disabled={len(disabled_layers)}"
            f"{' | final' if is_final_recovery else ''}"
        )
        recovery_result = _run_recovery_pass(
            config,
            disabled_layers,
            output_root,
            cycle_idx=cycle,
            is_final=is_final_recovery,
            progress_context=progress_context,
        )
        recovery_results.append(recovery_result)

        if converged:
            break

    final_result = recovery_results[-1]

    print(f"\n{_banner}")
    print(f"Final Training | {len(disabled_layers)} layer(s) permanently disabled")
    if disabled_layers:
        print(f"  Disabled: {disabled_layers}")
    print(_banner)

    cycle_train_time_sec = sum(
        float(result.get("full_train_time_sec", 0.0))
        for result in cycle_results
    )
    recovery_train_time_sec = sum(
        float(result.get("full_train_time_sec", 0.0))
        for result in recovery_results
    )
    final_retrain_time_sec = float(final_result.get("full_train_time_sec", 0.0))
    full_train_time_sec = cycle_train_time_sec + recovery_train_time_sec
    num_epochs_executed = (
        sum(int(result.get("num_epochs_executed", 0)) for result in cycle_results)
        + sum(int(result.get("num_epochs_executed", 0)) for result in recovery_results)
    )
    baseline_costs: dict[str, Mapping[str, Any]] = {}
    for sub_result in [*cycle_results, *recovery_results]:
        baseline = sub_result.get("adaptive_lambda_baseline")
        if isinstance(baseline, Mapping):
            baseline_costs[str(baseline.get("history_path"))] = baseline
    one_time_baseline_cost_sec = sum(
        float(baseline.get("full_train_time_sec", 0.0))
        for baseline in baseline_costs.values()
    )
    one_time_baseline_epochs = sum(
        int(baseline.get("num_epochs_executed", 0))
        for baseline in baseline_costs.values()
    )

    cyclic_result = {
        "num_cycles_completed": len(cycle_results),
        "final_disabled_layers": disabled_layers,
        "cycle_results": cycle_results,
        "recovery_results": recovery_results,
        "final_result": final_result,
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
