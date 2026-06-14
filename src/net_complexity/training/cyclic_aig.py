"""Cyclic AIG layer-dropping training loop.

Algorithm:
  1. Train with AIG for ``aig_epochs`` epochs.
  2. Collect per-layer gate probabilities (``valid_g_prob_<layer>``).
  3. Drop every layer whose gate probability is below ``g_prob_threshold``.
  4. Repeat until no new layers are dropped (convergence) or ``max_cycles`` is reached.
  5. Run a final plain training on the surviving-layer model.

Config section (``cyclic_layer_dropping``)::

    cyclic_layer_dropping:
      enabled: true
      max_cycles: 5
      g_prob_threshold: 0.5
      aig_epochs: 200         # epochs per AIG cycle (overrides training_arguments.num_epochs)
      final_epochs: 200       # epochs for the final plain-training run
      use_plain_model_for_final: true   # replace AIGBottleneck with Bottleneck in final run
      disable_mlflow_for_cycles: true   # suppress MLflow logging in intermediate cycles
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf, open_dict

from .engine import run_training


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _extract_layer_g_probs(valid_metrics: Mapping[str, Any]) -> dict[str, float]:
    """Return {layer_name: g_prob} for every ``valid_g_prob_layer*`` metric."""
    result: dict[str, float] = {}
    prefix = "valid_g_prob_"
    for key, value in valid_metrics.items():
        if not key.startswith(prefix):
            continue
        layer_name = key[len(prefix):]
        # Skip aggregate keys: average_prob, max_prob, min_prob, …
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

    # Epoch count for this AIG cycle
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

    # Epoch count for the final run
    final_epochs = getattr(cyclic_cfg, "final_epochs", None)
    if final_epochs is not None:
        OmegaConf.update(cfg, "training_arguments.num_epochs", int(final_epochs), merge=False)

    # Disable AIG regularization — purely CE-driven final training
    OmegaConf.update(cfg, "model.lambda_coef", 0.0, merge=False)

    # Optionally swap AIGBottleneckLayer → standard Bottleneck so the gate
    # adapter is absent and there is no stochastic noise during final training.
    # We delete the key so ResNet50() falls back to its default Bottleneck arg.
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

    # Re-enable MLflow for the final (summary) run
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
    threshold = float(cyclic_cfg.g_prob_threshold)

    disabled_layers: list[str] = _read_initial_disabled_layers(config)
    cycle_results: list[dict[str, Any]] = []

    _banner = "=" * 60

    for cycle in range(max_cycles):
        print(f"\n{_banner}")
        print(
            f"Cyclic AIG Training | cycle {cycle + 1}/{max_cycles}"
            f" | disabled={len(disabled_layers)}"
            f" | threshold={threshold}"
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

        new_drop = _layers_to_drop(layer_probs, threshold, disabled_layers)

        if not new_drop:
            print(
                f"Cycle {cycle + 1} | no layers with g_prob < {threshold} — converged."
            )
            break

        print(f"Cycle {cycle + 1} | dropping {len(new_drop)} layer(s): {new_drop}")
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
