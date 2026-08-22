"""Post-hoc static pruning for classic ConvNet-AIG (ablation/aig_classic) checkpoints.

Classic AIG makes a *per-input* keep/skip decision for every gated block at
eval time (``AIGBlockGate.forward``: ``gate = keep_probabilities >= threshold``)
— the architecture and parameter count are identical for every value of the
target-rate ``t``, by design (same capacity, adaptive average compute; see
ConvNet-AIG, Veit & Belongie 2020). That means a `t`-sweep's ``valid_accuracy``
cannot be read as "accuracy after dropping X% of params" — the honest
compute-axis metric already logged for those runs is
``valid_aig_flops_active_ratio`` / ``valid_aig_active_flops_per_sample``
(``metrics.aig.AIGFLOPsMetric``).

For a data point that IS comparable on a params/FLOPs axis to the DepGraph
and "ours" ablation baselines (both of which produce a genuinely static,
pruned architecture), this module converts a trained classic-AIG checkpoint
into one:

  1. Read each block's average validation keep-probability
     (``valid_g_prob_backbone.layerN.B``) from the source run's persisted
     ``summary.json`` — the *same* epoch the loaded checkpoint corresponds to
     (``best_valid`` for ``best.pt``, ``final_valid`` for ``last.pt``).
  2. Blocks whose average keep-probability falls below ``g_prob_threshold``
     are physically removed via the existing structural ``layer_skipping``
     mechanism (``mode="prune"``).
  3. Every surviving block's AIG gate is bypassed to always-on
     (``ClassificationFeatureSelectionWrapper.set_aig_bypass(True)``), so the
     resulting forward pass is fully deterministic (no residual per-input
     branching) — a real static subnetwork, not just "mostly static".

Usage (see configs/experiment/ablation/aig_classic_static/*.yaml): point
``source_run_dir`` at a completed ``ablation/aig_classic`` run, pick a
``g_prob_threshold``. The resulting model is then fine-tuned like any other
model via the normal training loop (matching the DepGraph baseline's
"prune once, then fine-tune through the standard pipeline" convention).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

_G_PROB_PREFIX = "valid_g_prob_"


def _resolve_checkpoint_path(cfg: DictConfig) -> tuple[Path, Path, str]:
    source_run_dir = getattr(cfg, "source_run_dir", None)
    if not source_run_dir:
        raise ValueError(
            "aig_static_pruning requires 'source_run_dir' (a completed "
            "ablation/aig_classic run directory) — unlike depgraph_pruning, "
            "it also needs that run's summary.json for per-block gate rates, "
            "so a bare source_checkpoint is not enough."
        )
    checkpoint_name = str(getattr(cfg, "checkpoint_name", "best.pt"))
    run_dir = Path(str(source_run_dir))
    return run_dir, run_dir / "checkpoints" / checkpoint_name, checkpoint_name


def _extract_layer_g_probs(valid_metrics) -> dict[str, float]:
    """Return {layer_name: mean_keep_probability} from a valid_metrics mapping.

    Strips the "backbone." prefix so keys look like "layer1.0", matching
    layer_skipping's disabled_layers format. Mirrors
    training.cyclic_aig._extract_layer_g_probs, duplicated locally to keep
    this module (like the rest of net_complexity.models) independent of
    net_complexity.training.
    """
    result: dict[str, float] = {}
    for key, value in valid_metrics.items():
        if not key.startswith(_G_PROB_PREFIX):
            continue
        layer_name = key[len(_G_PROB_PREFIX):]
        if layer_name.startswith("backbone."):
            layer_name = layer_name[len("backbone."):]
        if layer_name.startswith("layer"):
            result[layer_name] = float(value)
    return result


def _read_block_g_probs(run_dir: Path, checkpoint_name: str) -> dict[str, float]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"aig_static_pruning: expected 'summary.json' under source_run_dir="
            f"'{run_dir}', not found at '{summary_path}'. Point source_run_dir "
            "at a completed ablation/aig_classic training run."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # best.pt <-> best_valid, last.pt <-> final_valid: the g_prob snapshot
    # must come from the same epoch as the checkpoint being loaded, otherwise
    # the drop decision and the loaded weights would describe different runs.
    if checkpoint_name == "last.pt":
        metrics = summary.get("final_valid") or {}
    else:
        metrics = (summary.get("best_valid") or {}).get("metrics") or {}

    layer_g_probs = _extract_layer_g_probs(metrics)
    if not layer_g_probs:
        raise ValueError(
            f"aig_static_pruning: no 'valid_g_prob_layer*' keys found in "
            f"'{summary_path}' for checkpoint '{checkpoint_name}' — was this "
            "run trained with an AIG-gated backbone and a metrics config "
            "that includes AIGActivationsMetric (e.g. /metrics: full)?"
        )
    return layer_g_probs


def _layers_to_drop(layer_g_probs: dict[str, float], threshold: float) -> list[str]:
    return sorted(name for name, prob in layer_g_probs.items() if prob < threshold)


def build_static_aig_model_from_config(
    config: DictConfig,
    cfg: DictConfig,
    device: str | None = None,
) -> nn.Module:
    """Build a statically pruned model from a trained classic-AIG checkpoint.

    ``config.model`` must describe the *same* AIG-gated architecture the
    checkpoint was trained with (matching depgraph_pruning's convention of
    reusing the current run's own composed ``model`` section rather than
    reloading the source run's config), so ``model_state_dict`` keys line up.

    Returns a ``ClassificationFeatureSelectionWrapper`` ready for fine-tuning
    through the normal training loop, with the low-gate-rate blocks
    physically removed and every surviving block's AIG gate bypassed to
    always-on (see module docstring).
    """
    from .feature_selection import ClassificationFeatureSelectionWrapper
    from .layer_skipping import apply_layer_skipping

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir, checkpoint_path, checkpoint_name = _resolve_checkpoint_path(cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"aig_static_pruning source checkpoint not found: '{checkpoint_path}'. "
            "Point aig_static_pruning.source_run_dir at a completed "
            "ablation/aig_classic training run."
        )
    layer_g_probs = _read_block_g_probs(run_dir, checkpoint_name)

    g_prob_threshold = float(getattr(cfg, "g_prob_threshold", 0.5))
    disabled_layers = _layers_to_drop(layer_g_probs, g_prob_threshold)

    model = instantiate(config.model)
    if not isinstance(model, ClassificationFeatureSelectionWrapper):
        raise TypeError(
            "aig_static_pruning expects model._target_ to be "
            "ClassificationFeatureSelectionWrapper, matching the checkpointed run."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    params_before = sum(p.numel() for p in model.parameters())
    if disabled_layers:
        apply_layer_skipping(model, disabled_layers, mode="prune")
    model.set_aig_bypass(True)
    params_after = sum(p.numel() for p in model.parameters())

    print(
        f"[aig_static_pruning] g_prob_threshold={g_prob_threshold} | "
        f"blocks dropped: {disabled_layers or 'none'} | "
        f"params {params_before:,} -> {params_after:,} "
        f"({(1.0 - params_after / max(params_before, 1)):.1%} removed) | "
        "remaining blocks' AIG gates bypassed to always-on"
    )

    return model
