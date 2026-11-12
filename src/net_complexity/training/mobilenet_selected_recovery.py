"""Recover the explicitly selected MobileNet, without repeating mask search."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


MANIFEST = Path("configs/mobilenet_recovery45_then_wide_search.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_hash(mask: dict) -> str:
    return hashlib.sha256(json.dumps(mask, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inspect_source(repo: Path, source_profile: Path | None = None) -> dict:
    """Fail before training if source artifacts differ from the selected result."""
    spec = json.loads((repo / MANIFEST).read_text())["recovery_source"]
    profile = (source_profile or repo / spec["source_profile"]).resolve()
    summary = json.loads((profile / "cyclic_summary.json").read_text())
    search = summary["cycle_results"][0]
    if search["best_epoch"] != spec["expected_epoch"] or search["num_epochs_executed"] != 30:
        raise ValueError("Selected recovery requires the completed epoch30 search checkpoint")
    if summary["final_result"] is not None or summary["recovery_results"]:
        raise ValueError("Expected the original search-only source")
    run = profile / "cycle_0" / Path(search["run_dir"]).name
    mask = json.loads((profile / "channel_mask.json").read_text())
    bundled = json.loads((repo / spec["mask_file"]).read_text())
    if mask != bundled or mask_hash(mask) != spec["mask_sha256"]:
        raise ValueError("Source mask differs from the selected 962-channel architecture")
    arch = json.loads((profile / "architecture.json").read_text())
    if (arch["total_params"] != spec["expected_params"]
            or arch["channels_removed"] != spec["expected_channels"]
            or sum(map(len, mask.values())) != spec["expected_channels"]):
        raise ValueError("Source physical parameter/channel counts differ")
    for filename, key in (("config_resolved.yaml", "config_sha256"), ("history.csv", "history_sha256")):
        if sha256(run / filename) != spec[key]:
            raise ValueError(f"Selected source {filename} differs from the verified result")
    with (run / "history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in history] != list(range(1, 31)):
        raise ValueError("Source history must contain epochs1-30 exactly")
    checkpoint = run / "checkpoints/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected trained weights are missing: {checkpoint}; no ImageNet fallback")
    held_state = search["adaptive_lambda_state"]
    return {
        "source_profile": str(profile), "source_run_dir": str(run),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "mask_sha256": mask_hash(mask), "mask": mask,
        "expected_params": spec["expected_params"], "source_epoch": 30,
        "held_adaptive_lambda_state": held_state,
        "controller_semantics": "preserved source state; structural recovery has no gates; new searches reset independently",
        "optimizer_semantics": "fresh AdamW and cosine45; backbone lr0.0001, classifier lr0.001",
    }


def recover_selected_model(config) -> dict:
    """Physically slice learned weights, then train the gate-free model45epochs."""
    import torch
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from .engine import run_training
    from ..models.channel_pruning import transfer_gated_weights_to_structural
    from ..models.feature_selection import get_gumbel_modules

    repo = Path(__file__).resolve().parents[3]
    source_arg = config.selected_recovery.source_profile
    identity = inspect_source(repo, Path(source_arg) if source_arg else None)
    if (int(config.training_arguments.num_epochs) != 45
            or config.training_arguments.adaptive_lambda.enabled
            or float(config.model.lambda_coef) != 0
            or float(config.model.criterion.label_smoothing) != 0):
        raise ValueError("Recovery requires45epochs, no gate controller and label_smoothing0")
    checkpoint = torch.load(identity["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != 30:
        raise ValueError("Checkpoint is not the selected epoch30")
    held = checkpoint.get("extra_state", {}).get("adaptive_lambda_state")
    # Epoch dictionaries use integer keys in torch checkpoints and string keys in JSON.
    if json.loads(json.dumps(held, allow_nan=False)) != identity["held_adaptive_lambda_state"]:
        raise ValueError("Checkpoint controller state differs from selected source metadata")
    source_cfg = OmegaConf.load(Path(identity["source_run_dir"]) / "config_resolved.yaml")
    source_cfg.model.backbone.pretrained_weights = None
    with torch.device("cpu"):
        source_model = instantiate(source_cfg.model)
    source_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    config.model.backbone.pretrained_weights = None
    OmegaConf.update(config, "channel_pruning", {
        "enabled": True, "structural": True, "mode": "explicit", "mask": identity["mask"],
    }, merge=False, force_add=True)

    def initialize(model):
        if get_gumbel_modules(model) or sum(p.numel() for p in model.parameters()) != identity["expected_params"]:
            raise ValueError("Recovery must use the gate-free2251882parameter architecture")
        transfer_gated_weights_to_structural(source_model, model)
        # Preserve the recipe's differential learning rates despite avoiding a weights download.
        model.backbone.pretrained_parameter_names = frozenset(
            name for name, _ in model.backbone.named_parameters() if name.startswith("features.")
        )
        print("[recovery] Surviving trained weights and BN statistics copied from epoch30", flush=True)

    directory = Path(HydraConfig.get().runtime.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source_provenance.json").write_text(json.dumps(identity, indent=2) + "\n")
    result = run_training(config, model_initializer=initialize)
    if result["num_epochs_executed"] != 45 or not result["test_metrics"]:
        raise RuntimeError("Recovery did not finish45epochs with frozen-checkpoint test")
    record = {"result": result, "source": identity, "new_epochs": 45,
              "lifetime_training_epochs": 75, "total_params": identity["expected_params"]}
    (directory / "recovery_result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
