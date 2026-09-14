"""Historical pilot reuse plus versioned epoch runtime validation and restore.

The historical validators remain restricted to reuse before the first pruning
transaction. New epoch-event helpers support exact same-stage epoch-boundary
continuation; they do not claim general iterative or partial-epoch resume.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .pruning_measurement import mask_hash, state_hash


def _read_json(path):
    return json.loads(path.read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsafe pruning resume: {message}")


def _same_config(config, source):
    saved = OmegaConf.to_container(OmegaConf.load(source / "resolved_config.yaml"), resolve=True)
    current = OmegaConf.to_container(config, resolve=True)
    _require(saved == current, "saved and requested configurations differ")


def validate_reused_dense(config, source, *, expected_epochs):
    source = Path(source).resolve()
    _same_config(config, source)
    state = _read_json(source / "pilot_state.json")
    _require(state["status"] == "completed" and state["global_epochs_completed"] == expected_epochs,
             f"dense control is not a completed {expected_epochs}-epoch run")
    _require(not state["test_evaluated"] and not state["accepted_mask"],
             "dense control has test access or a pruned mask")
    _require(state["seed"] == int(config.seed), "dense seed differs")
    initial = torch.load(str(config.cyclic_channel_pruning.weight_handoff.initial_checkpoint),
                         map_location="cpu", weights_only=True)
    _require(initial.get("trained_epochs") == 0 and initial.get("seed") == int(config.seed),
             "shared initializer is not the original zero-epoch initializer")
    _require(state_hash(initial["model_state_dict"]) == state["common_init_hash"],
             "shared initializer hash differs from dense control")
    del initial
    checkpoint = torch.load(source / "deployment.pt", map_location="cpu", weights_only=True)
    _require(checkpoint["model_state_hash"] == state_hash(checkpoint["model_state_dict"]),
             "dense deployment checkpoint hash is invalid")
    _require(checkpoint["common_init_hash"] == state["common_init_hash"]
             and checkpoint["global_epochs_consumed"] == expected_epochs
             and checkpoint["validation"] == state["validation"]
             and not checkpoint["pruning_mask"], "dense deployment provenance differs")
    return {**state, "reused_from": str(source)}


def load_completed_search(config, source, *, common_hash, split_hash, steps_per_epoch):
    source = Path(source).resolve()
    _same_config(config, source)
    c = config.cyclic_channel_pruning
    epochs = int(c.gumbel_epochs)
    _require(c.max_cycles == 1, "only a one-cycle pilot can reuse the first search")
    state = _read_json(source / "pilot_state.json")
    _require(state["status"] in {"failed", "interrupted"}, "source job is not stopped")
    _require(state["global_epochs_completed"] == epochs
             and state["total_epochs_allocated"] == epochs + c.final_epochs,
             "search epoch budget is incomplete or differs")
    _require(not state["decisions"] and not state["accepted_mask"]
             and not (source / "deployment.pt").exists()
             and not list(source.glob("*recovery*"))
             and not (source / "rollback_finetune").exists(),
             "a pruning transaction or recovery has already started")
    _require(not state["test_evaluated"] and state["seed"] == int(config.seed),
             "source test policy or seed differs")
    _require(state["common_init_hash"] == common_hash and state["split_indices_hash"] == split_hash,
             "initializer or data split differs")
    _require(len(state["stages"]) == 1, "expected exactly one completed search stage")
    stage = state["stages"][0]
    _require(stage["name"] == "cycle_0_search" and stage["epochs"] == epochs,
             "completed stage is not the full first search")
    run_dir = Path(stage["run_dir"]).resolve()
    _require(run_dir.parent == source / "cycle_0_search", "search directory is outside the source job")
    summary = _read_json(run_dir / "summary.json")
    _require(summary["timing"]["num_epochs_executed"] == epochs and not summary["test"],
             "search summary is incomplete or contains test metrics")
    history_path = source / "global_history.csv"
    with history_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == epochs, "search history length differs")
    for epoch, row in enumerate(rows, 1):
        _require(int(row["global_epoch"]) == epoch and int(row["local_epoch"]) == epoch
                 and row["stage"] == "cycle_0_search"
                 and int(row["optimizer_steps_total"]) == epoch * steps_per_epoch
                 and row["mask_hash"] == mask_hash({})
                 and float(row["lambda_used"]) == float(config.model.lambda_coef),
                 "search history epochs, steps, mask or lambda differ")
    best_row = max(rows, key=lambda row: (float(row["valid_accuracy"]), -float(row["valid_ce_loss"])))
    best_epoch = int(best_row["local_epoch"])
    _require(stage["best_epoch"] == best_epoch == summary["best_valid"]["epoch"],
             "selected checkpoint epoch differs from history")
    hashes = {}
    for name, expected_epoch in (("last", epochs), ("best", best_epoch)):
        checkpoint = torch.load(run_dir / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=True)
        _require(checkpoint["epoch"] == expected_epoch
                 and checkpoint["global_epochs_completed"] == expected_epoch
                 and not checkpoint["pruning_mask"] and checkpoint["mask_hash"] == mask_hash({}),
                 f"{name} checkpoint epoch or mask differs")
        hashes[name] = state_hash(checkpoint["model_state_dict"])
        _require(hashes[name] == checkpoint["model_state_hash"], f"{name} checkpoint hash is invalid")
        if name == "best":
            _require(float(checkpoint["metrics"]["valid_accuracy"]) == float(best_row["valid_accuracy"])
                     and float(checkpoint["metrics"]["valid_ce_loss"]) == float(best_row["valid_ce_loss"]),
                     "best checkpoint metrics differ from history")
        del checkpoint
    return {"stage": dict(stage), "history_path": str(history_path), "metadata": {
        "source_job": str(source), "epochs_reused": epochs,
        "selected_checkpoint_state_hash": hashes["best"], "last_checkpoint_state_hash": hashes["last"],
        "source_state_sha256": hashlib.sha256((source / "pilot_state.json").read_bytes()).hexdigest(),
        "source_history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "mode": "completed_first_search_only_not_mid_epoch_resume",
    }}


# New snapshots coexist with the deliberately narrow historical reuse validator.
EPOCH_EVENT_VERSION = 1
FORWARD_RUNTIME_ATTRIBUTES = (
    "temperature", "beta", "gate_threshold", "train_gate_mode", "eval_gate_mode",
    "force_ones_mask", "deterministic_soft_mask", "deterministic_hard_mask",
    "_bypass", "_open_bias", "_open_bias_p_min", "_open_bias_p_max",
    "lambda_coef", "bypass_on_zero_lambda", "entropy_regularization",
    "entropy_regularization_coef", "_regularization_normalization",
)


def capture_eval_runtime(model):
    """Non-state-dict forward settings, including historical nonzero open bias."""
    return {name: {key: getattr(module, key) for key in FORWARD_RUNTIME_ATTRIBUTES
                   if hasattr(module, key) and isinstance(getattr(module, key), (str, int, float, bool))}
            for name, module in model.named_modules()
            if any(hasattr(module, key) for key in FORWARD_RUNTIME_ATTRIBUTES)}


def restore_eval_runtime(model, runtime):
    import math
    modules = dict(model.named_modules())
    expected = capture_eval_runtime(model)
    if set(runtime) != set(expected):
        raise ValueError("Epoch runtime topology differs from constructed model")
    for name, attrs in runtime.items():
        if set(attrs) != set(expected[name]):
            raise ValueError(f"Incomplete forward runtime for {name}")
        for key, value in attrs.items():
            if key not in FORWARD_RUNTIME_ATTRIBUTES:
                raise ValueError(f"Unknown forward runtime setting {name}.{key}")
            if (not isinstance(value, (str, int, float, bool))
                    or isinstance(value, (int, float)) and not math.isfinite(value)):
                raise ValueError(f"Nonfinite or invalid forward runtime setting {name}.{key}")
            setattr(modules[name], key, value)


def checkpoint_runtime_status(checkpoint):
    """Explicit status; never infer an exact forward from incomplete legacy data."""
    event = checkpoint.get("epoch_event")
    if event is None:
        return {"status": "legacy_incomplete_runtime", "exact_eval": False, "exact_resume": False}
    required = {"version", "eval_runtime", "continuation_runtime", "alpha_used", "alpha_next",
                "controller_after_feedback", "clocks", "consumed_ledger", "topology", "provenance"}
    if not isinstance(event, dict) or not required <= set(event) or event["version"] != EPOCH_EVENT_VERSION:
        raise ValueError("Incomplete or unsupported epoch-event snapshot")
    return {"status": "complete_epoch_event", "exact_eval": True,
            "exact_resume": checkpoint.get("extra_state", {}).get("status") != "interrupted"}


def restore_exact_epoch_checkpoint(checkpoint, *, model, optimizer, scheduler_state,
                                   controller, training_ledger, dataloader, expected_stage):
    """Resume at an epoch boundary; partial worker/sampler state is not guessed.

    For num_workers>0 only nonpersistent workers are supported; their next epoch
    is deterministically seeded by the saved DataLoader generator state.
    """
    import random
    import numpy as np
    from copy import deepcopy
    status = checkpoint_runtime_status(checkpoint)
    if not status["exact_resume"]:
        raise ValueError("Exact resume requires complete epoch-boundary runtime; legacy/partial snapshots refused")
    event = checkpoint["epoch_event"]
    if event["provenance"].get("stage") != expected_stage:
        raise ValueError("Exact resume stage/config/reference/initializer provenance mismatch")
    if getattr(dataloader, "persistent_workers", False):
        raise ValueError("Exact resume of persistent DataLoader worker RNG is unsupported")
    if set(event["consumed_ledger"]) != set(training_ledger):
        raise ValueError("Exact resume consumed-ledger schema mismatch")
    if checkpoint.get("model_state_hash") != state_hash(checkpoint["model_state_dict"]):
        raise ValueError("Exact resume model hash mismatch")
    required = {"python", "numpy", "torch", "cuda"}
    if not required <= set(checkpoint.get("rng_state", {})):
        raise ValueError("Exact resume RNG snapshot incomplete")
    saved = event["consumed_ledger"]
    clock_keys = {"global_training_epoch", "search_epochs_consumed"}
    if (not clock_keys <= set(saved)
            or any(not isinstance(value, int) or value < 0 for value in saved.values())
            or any(event["clocks"].get(key) != saved[key] for key in clock_keys)
            or saved["search_epochs_consumed"] > saved["global_training_epoch"]
            or checkpoint.get("consumed_ledger") != saved):
        raise ValueError("Inconsistent exact-resume consumed clocks/ledger")
    if (checkpoint.get("mask_hash") != mask_hash(checkpoint.get("pruning_mask", {}))
            or event["topology"].get("mask_hash") != checkpoint.get("mask_hash")):
        raise ValueError("Exact resume topology/mask snapshot mismatch")
    if any(training_ledger[key] not in (0, saved[key]) for key in training_ledger):
        raise ValueError("Exact resume cannot rewind a later consumed ledger")
    state = event["controller_after_feedback"]
    if not event.get("controller_updates_enabled", state is not None):
        state = None  # Held parent state is provenance, not a physical-stage controller.
    if (state is None) != (controller is None):
        raise ValueError("Exact resume controller presence mismatch")
    if controller is not None:
        controller.load_state_dict(state)
        if state.get("control_mode") == "accuracy_only":
            runtime = state["runtime"]
            if (any(runtime[key] != saved[key] for key in clock_keys)
                    or runtime["local_search_epoch"] != event["clocks"]["local_search_epoch"]
                    or runtime["lambda_coef"] != event["alpha_next"]):
                raise ValueError("Exact resume controller/alpha/clock mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    from net_complexity.models.feature_selection import (
        get_gate_normalization_metadata, validate_gate_normalization_metadata,
    )
    topology = event["topology"]
    normalization = {key: topology[key] for key in get_gate_normalization_metadata(model)
                     if key in topology}
    validate_gate_normalization_metadata(model, normalization)
    restore_eval_runtime(model, event["continuation_runtime"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler_state is not None:
        if "scheduler_state_dict" not in checkpoint:
            raise ValueError("Exact resume scheduler snapshot absent")
        scheduler_state.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scheduler_state.step_count = int(checkpoint["scheduler_step_count"])
    elif "scheduler_state_dict" in checkpoint:
        raise ValueError("Exact resume scheduler configuration mismatch")
    training_ledger.update(deepcopy(saved))
    rng = checkpoint["rng_state"]
    random.setstate(rng["python"])
    ns = rng["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(rng["torch"].cpu())
    if rng["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Exact resume CUDA RNG cannot be restored on CPU")
        torch.cuda.set_rng_state_all(rng["cuda"])
    generator = getattr(dataloader, "generator", None)
    if generator is not None:
        if "train_dataloader" not in rng:
            raise ValueError("Exact resume DataLoader generator state absent")
        generator.set_state(rng["train_dataloader"].cpu())
    return int(checkpoint["epoch"]) + 1


def migrate_legacy_controller_to_accuracy_only(legacy_state, controller, *, transition_id,
                                               global_training_epoch, search_epochs_consumed,
                                               previous_mask_hash, new_mask_hash, phase_id):
    """Explicit handoff migration, never advertised as exact legacy resume."""
    import math
    from copy import deepcopy
    if legacy_state.get("version") != 1 or set(legacy_state) != {"version", "config", "runtime"}:
        raise ValueError("Incomplete legacy adaptive state; migration refused")
    from .adaptive_lambda import AdaptiveLambdaController
    legacy = AdaptiveLambdaController(initial_lambda_coef=0.001, **legacy_state["config"])
    legacy.load_state_dict(legacy_state)
    runtime = legacy_state["runtime"]
    if runtime["recovery_active"] or runtime["recovery_open_bias"] != 0:
        raise ValueError("Active legacy anti-collapse/open-bias episode cannot migrate to accuracy_only")
    migrated = controller.state_dict()
    migrated["runtime"]["lambda_coef"] = math.exp(runtime["log_lambda"])
    controller.load_state_dict(migrated)
    controller.rebase(transition_id=transition_id, phase_id=phase_id,
        previous_mask_hash=previous_mask_hash, new_mask_hash=new_mask_hash,
        reason="explicit_legacy_to_accuracy_only_handoff_not_exact_resume",
        global_training_epoch=global_training_epoch, search_epochs_consumed=search_epochs_consumed)
    return {"status": "migrated_handoff_not_exact_resume", "source_version": 1,
            "controller_state": deepcopy(controller.state_dict())}


def validate_epoch_eval_checkpoint(checkpoint, model):
    """Fail closed before evaluating a selected epoch; never change its mask.

    Call after loading model_state_dict into the matching carrier. This validates
    metadata and the evaluated alpha, and restores the saved eval runtime.
    """
    import math
    from net_complexity.models.feature_selection import (
        get_gate_normalization_metadata, validate_gate_normalization_metadata,
    )
    status = checkpoint_runtime_status(checkpoint)
    if not status["exact_eval"]:
        raise ValueError("Legacy incomplete runtime cannot claim exact selected-checkpoint evaluation")
    event = checkpoint["epoch_event"]
    if checkpoint.get("model_state_hash") != state_hash(checkpoint["model_state_dict"]):
        raise ValueError("Selected checkpoint model-state hash mismatch")
    if state_hash(model.state_dict()) != checkpoint["model_state_hash"]:
        raise ValueError("Selected checkpoint model weights have not been loaded exactly")
    if (checkpoint.get("mask_hash") != mask_hash(checkpoint.get("pruning_mask", {}))
            or event["topology"].get("mask_hash") != checkpoint.get("mask_hash")
            or event["topology"].get("permanent_mask") != checkpoint.get("pruning_mask", {})):
        raise ValueError("Selected checkpoint mask/topology metadata mismatch")
    expected = get_gate_normalization_metadata(model)
    validate_gate_normalization_metadata(model, {key: event["topology"].get(key) for key in expected})
    for key in ("alpha_used", "alpha_next"):
        if not isinstance(event[key], (int, float)) or not math.isfinite(event[key]) or event[key] < 0:
            raise ValueError(f"Invalid selected checkpoint {key}")
    used_root = event["eval_runtime"].get("", {})
    if event.get("alpha_semantics") != "held_base_no_gate_penalty":
        if used_root.get("lambda_coef") != event["alpha_used"]:
            raise ValueError("Selected checkpoint alpha_used disagrees with evaluated runtime")
    elif used_root.get("lambda_coef", 0) != 0:
        raise ValueError("Physical recovery runtime must have zero model gate penalty")
    if event["topology"].get("alpha_base") != used_root.get("lambda_coef", 0):
        raise ValueError("Selected checkpoint effective lambda map alpha mismatch")
    restore_eval_runtime(model, event["eval_runtime"])
    return event
