"""Read-only validation for resuming BEFORE the first pruning transaction.

Not a general or mid-epoch resume: only a completed first search in a one-cycle
pilot, with no commit/recovery started, can be inherited into a NEW directory.
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


def validate_reused_dense(config, source):
    source = Path(source).resolve()
    _same_config(config, source)
    state = _read_json(source / "pilot_state.json")
    _require(state["status"] == "completed" and state["global_epochs_completed"] == 25,
             "dense control is not a completed 25-epoch run")
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
             and checkpoint["global_epochs_consumed"] == 25
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
