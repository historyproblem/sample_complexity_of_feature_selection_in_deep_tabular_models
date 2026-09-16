"""Explicit preparation of a new dense validation reference for a clean clone.

This is not an implicit fallback in adaptive training. The caller opts into a
new reference and accounts for its training separately from each pruning branch.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import math
from pathlib import Path
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from net_complexity.models.pruning_budget import gates
from .accuracy_guided_config import code_provenance, file_hash
from .accuracy_guided_pruning import atomic_checkpoint
from .cyclic_aig import _configure_run_history, _set_num_epochs
from .engine import run_training
from .interruption import cooperative_signals, TrainingInterrupted
from .one_shot_pruning_config import dense_source_paths, to_v3_config, validate_config
from .one_shot_progress import epoch_progress, phase_progress, progress_message
from .pruning_measurement import (
    deployment_cost, evaluate_deployment, isolated_diagnostic_rng, mask_hash, state_hash, write_json,
)
from .randomness import set_random_seed

REFERENCE_PROTOCOL = "one_shot_dense_reference_v1"


def prepare_dense_reference(config, output_root):
    """Create one zero-epoch initializer, then train an ungated dense reference.

    The full profile trains exactly 150 epochs. Short measured synthetic runs
    require the explicit smoke marker and cannot serve a full experiment.
    """
    total = validate_config(config)
    output_root = Path(output_root).expanduser().resolve()
    if output_root.name == "J1_dense_control":
        raise ValueError("New reference output must be a parent directory, not named J1_dense_control")
    if output_root.exists():
        raise FileExistsError(f"Refusing existing dense reference directory: {output_root}")
    smoke = bool(config.accuracy_guided.smoke)
    if not smoke and (
        config.dataloaders._target_ not in ("net_complexity.dataloaders.ClassicCVDataloaders",
                                          "net_complexity.data.dataloaders.ClassicCVDataloaders")
        or str(config.dataloaders.taskname).lower() != "cifar10"
        or config.model.backbone._target_ != "net_complexity.wrappers.ResNet50"
        or config.model.backbone.num_classes != 10
        or OmegaConf.select(config, "model.backbone.base_width", default=64) != 64
    ):
        raise ValueError("Full dense reference requires CIFAR10 and the standard ResNet50; synthetic inputs require smoke=True")
    device, seed = str(config.device), int(config.seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing full dense training on CPU")

    # Keep the comparison plan separate from its executed stage config, as in
    # the pruning runner. The stage's actual cosine horizon is always `total`.
    plan = to_v3_config(config)
    del plan.accuracy_guided
    plan.model.lambda_coef = 0.0
    plan.model.backbone.resnet_block = {
        "_target_": "net_complexity.models.resnet.Bottleneck", "_partial_": True}
    plan.training_arguments.adaptive_lambda.enabled = False
    OmegaConf.update(plan, "channel_pruning", {"enabled": False}, merge=False, force_add=True)
    plan.mlflow.run_name = plan.run_history.run_name = "J1_dense_control"
    plan.run_history.log_channel_history = False
    runtime = deepcopy(plan)
    _set_num_epochs(runtime, total)

    with phase_progress("dense_reference", "preparing dataset: download/cache, split and validation loader"), isolated_diagnostic_rng():
        data = instantiate(config.dataloaders, include_test=False, loader_seed=seed)
        sample = next(iter(data.valid_dataloader))
    train_count, valid_count = len(data.train_dataloader.dataset), len(data.valid_dataloader.dataset)
    if not smoke and (train_count, valid_count) != (45000, 5000):
        raise ValueError("Dense reference requires the matched CIFAR10 45000/5000 split")
    split_hash = mask_hash({name: list(getattr(loader.dataset, "indices", range(len(loader.dataset))))
        for name, loader in (("train", data.train_dataloader), ("valid", data.valid_dataloader))})

    # No gate construction before the shared backbone initialization. This is
    # the same plain-backbone initialization procedure as the historical pilot.
    with phase_progress("dense_reference", "creating shared zero-epoch initializer"):
        set_random_seed(seed)
        source = instantiate(plan.model).cpu()
    if gates(source):
        raise AssertionError("Dense reference initialization unexpectedly contains gates")
    initial_state = {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}
    common_hash = state_hash(initial_state)
    paths = dense_source_paths(output_root)
    job = Path(paths["state_path"]).parent
    output_root.mkdir(parents=True, exist_ok=False)
    job.mkdir(exist_ok=False)
    _configure_run_history(runtime, job / "training")
    atomic_checkpoint(paths["initializer_path"], {
        "model_state_dict": initial_state, "model_state_hash": common_hash,
        "trained_epochs": 0, "seed": seed, "reference_protocol": REFERENCE_PROTOCOL,
        "origin": "new_shared_seed42_initializer", "prior_checkpoint_loaded": False})
    initializer_file_hash = file_hash(paths["initializer_path"])
    progress_message("dense_reference", f"zero-epoch initializer saved: {paths['initializer_path']}")
    OmegaConf.save(plan, paths["config_path"], resolve=True)
    OmegaConf.save(runtime, job / "training_config.yaml", resolve=True)
    ledger = dict(global_training_epoch=0, search_epochs_consumed=0,
                  optimizer_updates=0, consumed_training_examples=0)
    with phase_progress("dense_reference", "measuring initial physical model cost"):
        initial_cost = deployment_cost(source, image_shape=tuple(sample[0].shape[1:]))
    state = {"root": str(output_root), "status": "running", "reference_protocol": REFERENCE_PROTOCOL,
        "reference_origin": "new_shared_seed42_initializer", "seed": seed,
        "common_init_hash": common_hash, "initializer_file_hash": initializer_file_hash,
        "initialization_state_hash": common_hash, "training_initializer_verified": False,
        "split_indices_hash": split_hash, "accepted_mask": {}, "test_evaluated": False,
        "total_epochs_allocated": total, "global_epochs_completed": 0, "ledger": ledger,
        "synthetic_reference": smoke,
        "reference_kind": "measured_synthetic_dense_reference" if smoke else "measured_dense_validation_reference",
        "reference_training_epochs_actually_executed": 0,
        "reference_training_cost": "external to each 150-epoch pruning branch; explicitly consumed here",
        "initial_cost": initial_cost,
        "runtime": {"optimizer": OmegaConf.to_container(runtime.optimizer, resolve=True),
                    "scheduler": OmegaConf.to_container(runtime.scheduler, resolve=True),
                    "gate_count": 0, "lambda_coef": 0.0, "adaptive_controller_enabled": False},
        "configuration_scope": "resolved_config.yaml is the comparison plan; training_config.yaml is the executed dense stage",
        "training_configuration_path": str(job / "training_config.yaml"),
        "code": code_provenance(), "torch_version": str(torch.__version__), "device": device}
    started = time.perf_counter()
    rows = []
    best_key = None
    best_path = job / "selected_checkpoint.pt"

    def persist():
        state.update(global_epochs_completed=ledger["global_training_epoch"],
            reference_training_epochs_actually_executed=ledger["global_training_epoch"],
            optimizer_steps_total=ledger["optimizer_updates"], wall_seconds=time.perf_counter() - started)
        write_json(Path(paths["state_path"]), state)

    def initialize(model):
        model.load_state_dict(initial_state, strict=True)

    def initialized(model, controller):
        if gates(model) or controller is not None or model.lambda_coef != 0.0:
            raise AssertionError("Dense reference must have no gates, penalty or adaptive controller")
        if state_hash(model.state_dict()) != common_hash:
            raise AssertionError("Dense runtime differs from the shared zero-epoch initializer")
        state["training_initializer_verified"] = True

    def epoch_end(epoch, train, valid, model, optimizer, history):
        nonlocal best_key
        accuracy, ce = float(valid["valid_accuracy"]), float(valid["valid_ce_loss"])
        correct, examples = int(valid["valid_correct_count"]), int(valid["valid_example_count"])
        if (examples != valid_count or not 0 <= correct <= examples
                or not math.isfinite(ce) or ce < 0
                or abs(accuracy - correct / examples) > 1e-12):
            raise ValueError("Dense validation must use finite, sample-weighted full-split metrics")
        if epoch != len(rows) + 1 or epoch != ledger["global_training_epoch"]:
            raise AssertionError("Dense reference epoch sequence or consumed ledger differs")
        rows.append({"global_epoch": epoch, "valid_accuracy": accuracy, "valid_ce_loss": ce,
            "valid_correct_count": correct, "valid_example_count": examples,
            "optimizer_steps_total": ledger["optimizer_updates"],
            "consumed_training_examples": ledger["consumed_training_examples"]})
        history_path = Path(paths["history_path"])
        with history_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        key = (-accuracy, ce, epoch)
        if best_key is None or key < best_key:
            best_key = key
            atomic_checkpoint(best_path, {"model_state_dict": model.state_dict(), "epoch": epoch,
                "validation": {"accuracy": accuracy, "ce_loss": ce,
                    "correct_count": correct, "example_count": examples},
                "model_state_hash": state_hash(model.state_dict()), "common_init_hash": common_hash})
        persist()
        epoch_progress("dense_reference", epoch, total, valid, ledger, time.perf_counter() - started)

    persist()
    try:
        with cooperative_signals():
            with phase_progress("dense_reference", f"training {total} epochs on {device}; state={paths['state_path']}",
                                ledger=ledger, total_epochs=total):
                result = run_training(runtime, model_initializer=initialize, runtime_initialized_callback=initialized,
                                      training_ledger=ledger, epoch_end_callback=epoch_end)
            if (result["num_epochs_executed"] != total or ledger["global_training_epoch"] != total
                    or len(rows) != total or ledger["search_epochs_consumed"] != 0
                    or not state["training_initializer_verified"]):
                raise RuntimeError("Incomplete dense reference; the full training budget must be executed")
            if result.get("test_metrics") or not result.get("test_evaluation_disabled"):
                raise AssertionError("Dense reference training accessed test data")
            progress_message("dense_reference", "validating selected frozen dense checkpoint")
            selected = torch.load(best_path, map_location="cpu", weights_only=True)
            source.load_state_dict(selected["model_state_dict"], strict=True)
            before = state_hash(source.state_dict())
            with isolated_diagnostic_rng(data.valid_dataloader):
                validation = evaluate_deployment(source, data.valid_dataloader, device)
            if before != state_hash(source.state_dict()):
                raise AssertionError("Frozen dense validation changed weights or BN state")
            if (validation["correct_count"] != selected["validation"]["correct_count"]
                    or abs(validation["ce_loss"] - selected["validation"]["ce_loss"]) > 1e-5):
                raise AssertionError("Frozen dense deployment differs from its selected validation checkpoint")
            if file_hash(paths["initializer_path"]) != initializer_file_hash:
                raise AssertionError("Dense training changed the immutable zero-epoch initializer")
            state.update(status="completed", validation=validation, selected_epoch=selected["epoch"],
                selection_policy="best_validation_accuracy_then_ce_then_earlier_epoch",
                training_run_dir=result["run_dir"], model_state_hash=before,
                final_cost=deployment_cost(source, image_shape=tuple(sample[0].shape[1:])),
                history_sha256=file_hash(paths["history_path"]),
                training_configuration_sha256=file_hash(job / "training_config.yaml"))
            atomic_checkpoint(job / "deployment.pt", {**state,
                "model_state_dict": source.cpu().state_dict(), "artifact_type": "dense_reference_only"})
            persist()
            progress_message("dense_reference", f"ready: epoch={state['selected_epoch']}; validation={validation['accuracy']:.4%}; source={output_root}")
            return state
    except Exception as exc:
        state.update(status="interrupted" if isinstance(exc, TrainingInterrupted) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        persist()
        progress_message("dense_reference", f"{state['status']}: {state['error']}; state={paths['state_path']}")
        raise
