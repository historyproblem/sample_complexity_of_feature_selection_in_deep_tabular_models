"""DepGraph (torch-pruning) structural pruning baseline.

Post-hoc structural pruning of an already-trained *plain* backbone (no AIG /
Gumbel / STG gates — this is a pruning baseline, not a gating method), using
the Dependency Graph mechanism from Fang et al., "DepGraph: Towards Any
Structural Pruning" (CVPR 2023), https://arxiv.org/abs/2301.12900,
implemented by the ``torch-pruning`` package (https://github.com/VainF/Torch-Pruning).

Unlike AIG/Gumbel/STG, DepGraph does not learn which channels to keep during
training — it estimates channel importance and removes the least important
ones, guided by a dependency graph that keeps coupled channels (e.g. a
Bottleneck's conv3 output and its downsample projection) consistent. The
pruned model is a plain nn.Module with no gates, wrapped in
``ClassificationFeatureSelectionWrapper`` purely so it fits the existing
training/evaluation pipeline (checkpointing, metrics, MLflow logging) — its
``lambda_coef`` is always 0 and it carries no regularization loss.

Two pruning protocols are supported, mirroring the reference reproduction
script (``Torch-Pruning/reproduce/main.py``) rather than a single fixed
channel ratio:

* Progressive pruning to a target FLOPs *speedup* (always used, matching
  ``reproduce/main.py``'s ``progressive_pruning``): the pruner takes many
  small steps (``iterative_steps``) with an aggregate ceiling of
  ``pruning_ratio=1.0``, stopping as soon as ``count_ops_and_params`` shows
  the measured speedup has reached ``target_speedup`` (or ``iterative_steps``
  is exhausted). This lands much closer to a specific target compute budget
  than a single one-shot cut at a fixed channel-count ratio, which matters
  for comparing against AIG operating points on the same GMAC/image axis.
* Optional sparsity learning (``sparsity_learning: true``): before pruning,
  runs a short extra training phase that calls ``pruner.regularize(model)``
  after every backward pass (L1/group-lasso pressure on the importance
  criterion), matching the reference's "slim"/"group_sl" recipes
  (``BNScalePruner``/``GroupNormPruner``). Without this phase (the default),
  the result corresponds to the DepGraph paper's own reported
  **"Ours w/o SL"** ablation — a legitimate, weaker baseline, not their
  headline numbers, which require the sparsity-learning phase.

Usage (see configs/experiment/ablation/depgraph/*.yaml): point
``source_checkpoint`` (or ``source_run_dir``) at a *plain*-trained run's
checkpoint, pick an ``importance`` criterion, and set a ``target_speedup``.
The resulting model is then fine-tuned like any other model via the normal
training loop.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch_pruning as tp

_IMPORTANCE_FACTORIES = {
    "magnitude_l1": lambda: tp.importance.GroupMagnitudeImportance(p=1),
    "magnitude_l2": lambda: tp.importance.GroupMagnitudeImportance(p=2),
    "random": lambda: tp.importance.RandomImportance(),
    "bn_scale": lambda: tp.importance.BNScaleImportance(),
    "lamp": lambda: tp.importance.LAMPImportance(),
}

# importance name -> sparsity-learning-capable pruner class, matching
# reproduce/main.py's get_pruner(): "slim"/"group_slim" -> BNScalePruner,
# "group_sl" -> GroupNormPruner. "random"/"lamp" have no SL recipe upstream.
_SPARSITY_LEARNING_PRUNER_CLASSES = {
    "bn_scale": tp.pruner.BNScalePruner,
    "magnitude_l1": tp.pruner.GroupNormPruner,
    "magnitude_l2": tp.pruner.GroupNormPruner,
}


def _build_importance(name: str):
    key = str(name).strip().lower()
    if key not in _IMPORTANCE_FACTORIES:
        allowed = ", ".join(sorted(_IMPORTANCE_FACTORIES))
        raise ValueError(f"depgraph_pruning.importance must be one of: {allowed}. Got: {name!r}")
    return _IMPORTANCE_FACTORIES[key]()


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    explicit = getattr(cfg, "source_checkpoint", None)
    if explicit:
        return Path(str(explicit))

    source_run_dir = getattr(cfg, "source_run_dir", None)
    if not source_run_dir:
        raise ValueError(
            "depgraph_pruning requires either 'source_checkpoint' (a .pt file) "
            "or 'source_run_dir' (a run directory containing checkpoints/<checkpoint_name>)."
        )
    checkpoint_name = str(getattr(cfg, "checkpoint_name", "best.pt"))
    return Path(str(source_run_dir)) / "checkpoints" / checkpoint_name


def _resolve_example_input_size(config: DictConfig, cfg: DictConfig) -> tuple[int, int, int]:
    explicit = getattr(cfg, "example_input_size", None)
    if explicit is not None:
        values = OmegaConf.to_container(explicit, resolve=True)
        c, h, w = (int(v) for v in values)
        return c, h, w

    in_channels = int(OmegaConf.select(config, "model.backbone.in_channels") or 3)
    image_size = OmegaConf.select(config, "data.dataloaders.image_size")
    if image_size is None:
        raise ValueError(
            "depgraph_pruning.example_input_size is not set and "
            "data.dataloaders.image_size could not be resolved — set one of them explicitly."
        )
    size = int(image_size)
    return in_channels, size, size


def _load_plain_backbone(config: DictConfig, checkpoint_path: Path, device: str) -> nn.Module:
    """Instantiate the configured backbone and load a plain-trained checkpoint into it.

    The checkpoint is expected to come from a run whose model was built from
    the *same* ``model`` config section (typically ``/method: plain``, i.e. no
    resnet_block override — a stock ``Bottleneck``/``CIFARBasicBlock``), so
    that ``model_state_dict`` keys line up with a fresh ``backbone.*`` prefix.
    """
    from .feature_selection import ClassificationFeatureSelectionWrapper

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DepGraph source checkpoint not found: '{checkpoint_path}'. "
            "Point depgraph_pruning.source_checkpoint / source_run_dir at a "
            "completed plain-training run."
        )

    model = instantiate(config.model)
    if not isinstance(model, ClassificationFeatureSelectionWrapper):
        raise TypeError(
            "depgraph_pruning expects model._target_ to be "
            "ClassificationFeatureSelectionWrapper, matching the checkpointed run."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.backbone


def _run_sparsity_learning_phase(
    backbone: nn.Module,
    pruner,
    config: DictConfig,
    cfg: DictConfig,
    *,
    device: str,
) -> None:
    """Run the pre-pruning sparsity-learning phase (reference: train_model()).

    Mirrors reproduce/main.py's train_model(): SGD with weight_decay=0 (the
    pruner's own regularizer supplies the sparsity pressure instead), calling
    ``pruner.regularize(model)`` after every backward pass and before the
    optimizer step.
    """
    dataloaders = instantiate(config.dataloaders)
    epochs = int(getattr(cfg, "sl_epochs", 20))
    lr = float(getattr(cfg, "sl_lr", 0.01))
    momentum = float(getattr(cfg, "sl_momentum", 0.9))

    optimizer = torch.optim.SGD(backbone.parameters(), lr=lr, momentum=momentum, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    backbone.to(device)
    backbone.train()

    for epoch in range(epochs):
        last_loss = None
        for batch in dataloaders.train_dataloader:
            X, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(backbone(X), y)
            loss.backward()
            pruner.regularize(backbone)
            optimizer.step()
            last_loss = loss
        loss_str = f"{last_loss.item():.4f}" if last_loss is not None else "n/a"
        print(f"[depgraph_pruning] sparsity learning epoch {epoch + 1}/{epochs} | last_batch_loss={loss_str}")

    backbone.eval()


def _progressive_prune_to_target_speedup(
    pruner,
    backbone: nn.Module,
    example_inputs: torch.Tensor,
    target_speedup: float,
) -> float:
    """Take small pruning steps until the measured FLOPs speedup hits the target.

    Mirrors reproduce/main.py's progressive_pruning(): the pruner is built
    with a coarse ceiling (pruning_ratio=1.0) and many iterative_steps; we
    stop as soon as the target is reached rather than exhausting every step,
    so the *result* is a specific compute budget rather than a specific
    channel-count ratio.
    """
    backbone.eval()
    base_ops, _ = tp.utils.count_ops_and_params(backbone, example_inputs=example_inputs)
    current_speedup = 1.0
    while current_speedup < target_speedup:
        pruner.step()
        pruned_ops, _ = tp.utils.count_ops_and_params(backbone, example_inputs=example_inputs)
        current_speedup = float(base_ops) / float(pruned_ops)
        if pruner.current_step >= pruner.iterative_steps:
            break
    return current_speedup


def build_depgraph_pruned_model_from_config(
    config: DictConfig,
    cfg: DictConfig,
) -> nn.Module:
    """Build a DepGraph-pruned model wrapped in ClassificationFeatureSelectionWrapper.

    Reads a plain-trained checkpoint, optionally runs a sparsity-learning
    phase, then progressively prunes the backbone's dependency graph until
    ``cfg.target_speedup`` (measured FLOPs speedup) is reached, using the
    importance criterion named by ``cfg.importance``. Returns the pruned
    model ready for fine-tuning through the normal training loop
    (forward(X, y) -> ClassifModelOutput, same as every other model in this
    codebase).
    """
    from .feature_selection import ClassificationFeatureSelectionWrapper

    device = "cpu"  # tracing/pruning happens on CPU; engine.py moves the result to config.device
    checkpoint_path = _resolve_checkpoint_path(cfg)
    backbone = _load_plain_backbone(config, checkpoint_path, device=device)
    backbone.eval()

    example_shape = _resolve_example_input_size(config, cfg)
    example_inputs = torch.randn(1, *example_shape)

    importance_name = str(getattr(cfg, "importance", "magnitude_l2")).strip().lower()
    importance = _build_importance(importance_name)

    target_speedup = float(getattr(cfg, "target_speedup", 2.0))
    if target_speedup <= 1.0:
        raise ValueError("depgraph_pruning.target_speedup must be > 1.0.")

    ignored_layers = []
    classifier_head = getattr(backbone, "fc", None) or getattr(backbone, "linear", None)
    if classifier_head is not None:
        ignored_layers.append(classifier_head)

    random_seed = getattr(cfg, "random_seed", None)
    if random_seed is not None:
        torch.manual_seed(int(random_seed))

    global_pruning = bool(getattr(cfg, "global_pruning", True))
    iterative_steps = int(getattr(cfg, "iterative_steps", 200))
    sparsity_learning = bool(getattr(cfg, "sparsity_learning", False))

    pruner_kwargs = dict(
        model=backbone,
        example_inputs=example_inputs,
        importance=importance,
        global_pruning=global_pruning,
        pruning_ratio=1.0,  # ceiling; progressive pruning stops early at target_speedup
        iterative_steps=iterative_steps,
        ignored_layers=ignored_layers,
    )

    if sparsity_learning:
        pruner_class = _SPARSITY_LEARNING_PRUNER_CLASSES.get(importance_name)
        if pruner_class is None:
            allowed = ", ".join(sorted(_SPARSITY_LEARNING_PRUNER_CLASSES))
            raise ValueError(
                "depgraph_pruning.sparsity_learning=true is only supported for "
                f"importance in {{{allowed}}} (matching the reference reproduce/main.py "
                f"'slim'/'group_sl' recipes); got importance={importance_name!r}."
            )
        reg = float(getattr(cfg, "reg", 1e-4))
        pruner = pruner_class(reg=reg, **pruner_kwargs)
        _run_sparsity_learning_phase(backbone, pruner, config, cfg, device=device)
    else:
        pruner = tp.pruner.BasePruner(**pruner_kwargs)

    params_before = sum(p.numel() for p in backbone.parameters())
    achieved_speedup = _progressive_prune_to_target_speedup(
        pruner, backbone, example_inputs, target_speedup
    )
    params_after = sum(p.numel() for p in backbone.parameters())

    print(
        f"[depgraph_pruning] importance={importance_name} sparsity_learning={sparsity_learning} "
        f"target_speedup={target_speedup} achieved_speedup={achieved_speedup:.3f} "
        f"global_pruning={global_pruning} | params {params_before:,} -> {params_after:,} "
        f"({(1.0 - params_after / max(params_before, 1)):.1%} removed)"
    )

    lambda_coef = float(OmegaConf.select(config, "model.lambda_coef") or 0.0)
    criterion_cfg = OmegaConf.select(config, "model.criterion")
    criterion = instantiate(criterion_cfg) if criterion_cfg is not None else nn.CrossEntropyLoss()

    return ClassificationFeatureSelectionWrapper(
        backbone=backbone,
        lambda_coef=lambda_coef,
        criterion=criterion,
        regularization_loss=lambda m: 0,
    )
