"""Frozen CIFAR-10 test for the explicitly budgeted 150 + 150 one-shot protocol.

Search checkpoints never enter this evaluator. Every selected deployment is
validated before test data is loaded; inference must leave all model state intact.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
from omegaconf import OmegaConf
import torch

from evaluate_pruning_test import (
    ROOT, build_test_loader, evaluate_fixed, file_hash, require,
    validate_config_and_mask,
)
from net_complexity.models.channel_pruning import build_structurally_pruned_model_from_config
from net_complexity.models.feature_selection import get_gumbel_modules
from net_complexity.training.pruning_measurement import deployment_cost, mask_hash, state_hash, write_json

PROTOCOL = "one_shot_reinit_v1"
DENSE = "D0_dense_reference"


def validate_epoch_accounting(state):
    require(state.get("protocol") == PROTOCOL, "Expected the one-shot reinitialization protocol")
    require(state.get("status") == "completed", "Refusing test of unfinished training")
    kind = state.get("kind")
    require(kind in ("dense_reference", "one_shot_reinit"), "Unknown deployment kind")
    expected = ((0, 0, 150) if kind == "dense_reference" else (150, 150, 0))
    keys = ("search_epochs_completed", "retrain_epochs_completed", "dense_epochs_completed")
    require(all(type(state.get(key)) is int and state[key] == value
                for key, value in zip(keys, expected)), "Incomplete or incorrect phase epoch budget")
    total = sum(expected)
    require(type(state.get("global_epochs_completed")) is int
            and state["global_epochs_completed"] == total
            and state.get("total_epochs_allocated") == total,
            "Incorrect total epoch accounting: dense 150; search + scratch 300")
    return kind


def resolve_job(run_dir, job, reference_source=None):
    require(isinstance(job, str) and re.fullmatch(r"[A-Za-z0-9_-]+", job) is not None,
            "Invalid job identifier")
    job_dir = Path(run_dir).resolve() / job
    if job == DENSE and reference_source is not None:
        source = Path(reference_source).resolve()
        if not (source / "one_shot_state.json").is_file():
            source = source / DENSE
    else:
        source = job_dir
        if (source / "one_shot_state.json").is_file():
            pointer = json.loads((source / "one_shot_state.json").read_text())
            if job == DENSE and pointer.get("reused_from") and not (source / "deployment.pt").is_file():
                source = Path(pointer["reused_from"]).resolve()
    for filename in ("one_shot_state.json", "deployment.pt", "resolved_config.yaml"):
        require((source / filename).is_file(), f"Missing {job}/{filename}; use --reference-source for relocated dense")
    return source


def prepare_job(run_dir, job, reference_source=None):
    source = resolve_job(run_dir, job, reference_source)
    state_path, checkpoint_path = source / "one_shot_state.json", source / "deployment.pt"
    config_path = source / "resolved_config.yaml"
    state = json.loads(state_path.read_text())
    kind = validate_epoch_accounting(state)
    require((job == DENSE) == (kind == "dense_reference"), "Dense and pruned job kinds cannot be exchanged")
    require(state.get("deployment_sha256") == file_hash(checkpoint_path)
            and state.get("resolved_config_sha256") == file_hash(config_path),
            "Saved deployment/config file hash differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint.get("protocol") == PROTOCOL and checkpoint.get("kind") == kind,
            "Deployment protocol/kind differs")
    mask = checkpoint["pruning_mask"]
    require(isinstance(mask, dict) and mask == state["accepted_mask"]
            and mask_hash(mask) == checkpoint["mask_hash"] == state["accepted_mask_hash"],
            "Deployment mask differs from frozen learned architecture")
    require(checkpoint["validation"] == state["validation"], "Deployment validation differs from selected result")
    validation = state["validation"]
    require(validation.get("example_count") == 5000
            and type(validation.get("correct_count")) is int
            and 0 <= validation["correct_count"] <= 5000
            and validation["accuracy"] == validation["correct_count"] / 5000,
            "Expected sample-weighted validation on all 5,000 images")
    require(checkpoint.get("common_init_hash") == state["common_init_hash"]
            and checkpoint.get("scratch_init_hash") == state.get("scratch_init_hash"),
            "Initialization provenance differs between manifest and deployment")
    require(checkpoint["global_epochs_consumed"] == state["global_epochs_completed"],
            "Deployment must account for the complete search and scratch training")
    if kind == "dense_reference":
        require(not any(mask.values()), "Dense reference contains pruned channels")
    else:
        require(isinstance(state.get("scratch_init_hash"), str) and bool(state["scratch_init_hash"]),
                "Missing scratch initialization provenance")
        require(state.get("scratch_init_hash") != state.get("common_init_hash"),
                "Scratch model must have its own initialization")
        require(checkpoint.get("provenance", {}).get("phase") == "scratch"
                and checkpoint["provenance"].get("trained_from_scratch") is True
                and checkpoint["provenance"].get("search_state_reused") is False,
                "Final deployment must come from fresh scratch training, not search")
        for key in ("search_weights_reused_for_scratch", "search_bn_reused_for_scratch",
                    "search_optimizer_reused_for_scratch", "search_controller_reused_for_scratch"):
            require(state.get(key) is False, f"Unexpected handoff: {key}")
    config = OmegaConf.load(config_path)
    require(int(config.seed) == state["seed"], "Config/manifest seed differs")
    require(OmegaConf.select(config, "one_shot_pruning.protocol") == PROTOCOL, "Config protocol differs")
    require(OmegaConf.select(config, "training_arguments.evaluate_test") is False
            and OmegaConf.select(config, "dataloaders.include_test") is False,
            "Training configuration must not use test")
    if kind == "one_shot_reinit":
        require(OmegaConf.select(config, "training_arguments.adaptive_lambda.enabled") is True,
                "One-shot search must use adaptive lambda")
        require(OmegaConf.select(config, "one_shot_pruning.search_epochs") == 150
                and OmegaConf.select(config, "one_shot_pruning.retrain_epochs") == 150,
                "Unexpected configured one-shot phase lengths")
        require(OmegaConf.select(config, "one_shot_pruning.probability_source") == "raw_logits"
                and OmegaConf.select(config, "one_shot_pruning.mask_threshold") == 0.5,
                "Unexpected mask selection policy")
        require(OmegaConf.select(config, "one_shot_pruning.reinit_seed") == state["reinit_seed"],
                "Scratch seed differs")
        require(OmegaConf.select(config, "model.backbone.resnet_block.regularization_normalization")
                == state.get("gate_regularization_normalization") == "initial_channels",
                "One-shot gate normalization differs")
    validate_config_and_mask(config, mask)
    weights = checkpoint["model_state_dict"]
    require(all(not value.is_floating_point() or value.dtype == torch.float32 for value in weights.values()),
            "Expected FP32 deployment weights")
    require(all(not value.is_floating_point() or torch.isfinite(value).all() for value in weights.values()),
            "Non-finite deployment weights/buffers")
    digest = state_hash(weights)
    require(digest == checkpoint["model_state_hash"] == state.get("deployment_model_state_hash"),
            "Deployment tensor hash differs")
    structural_config = deepcopy(config)
    structural_config.model.lambda_coef = 0.0
    pruning = OmegaConf.create({"mode": "explicit", "enabled": True, "structural": True, "mask": mask})
    with redirect_stdout(io.StringIO()):
        model = build_structurally_pruned_model_from_config(structural_config, pruning)
    model.load_state_dict(weights, strict=True)
    require(not get_gumbel_modules(model), "Final model still contains gates")
    cost = deployment_cost(model)
    for key in ("physical_total_parameters", "conv_linear_macs_per_image"):
        require(cost[key] == state["final_cost"][key], f"Physical deployment {key} differs")
    require(state_hash(model.state_dict()) == digest, "Loading/cost measurement changed the model")
    model.eval()
    record = {
        "job": job, "kind": kind, "training_protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_hash(checkpoint_path),
        "config": str(config_path), "config_sha256": file_hash(config_path),
        "state_sha256": file_hash(state_path), "model_state_hash": digest,
        "mask_hash": checkpoint["mask_hash"], "seed": state["seed"],
        "reinit_seed": state.get("reinit_seed"), "common_init_hash": state["common_init_hash"],
        "scratch_init_hash": state.get("scratch_init_hash"),
        "split_indices_hash": state["split_indices_hash"],
        "training_signature": state.get("training_signature"),
        "epochs_consumed": state["global_epochs_completed"],
        "search_epochs": state["search_epochs_completed"],
        "retrain_epochs": state["retrain_epochs_completed"],
        "dense_epochs": state["dense_epochs_completed"],
        "gate_regularization_normalization": state.get("gate_regularization_normalization", "not_applicable"),
        "adaptive_soft_drop": OmegaConf.select(config, "training_arguments.adaptive_lambda.soft_drop")
            if kind == "one_shot_reinit" else None,
        "adaptive_hard_drop": OmegaConf.select(config, "training_arguments.adaptive_lambda.hard_drop")
            if kind == "one_shot_reinit" else None,
        "removed_parameter_fraction": state.get("removed_parameter_fraction", 0.0),
        "validation": state["validation"], "selection_provenance": checkpoint["provenance"],
        "physical_parameters": cost["physical_total_parameters"],
        "conv_linear_macs_per_image": cost["conv_linear_macs_per_image"],
    }
    print(f"[check] {job}: {record['physical_parameters']:,} parameters; "
          f"search={record['search_epochs']}, scratch={record['retrain_epochs']}, dense={record['dense_epochs']}", flush=True)
    return model, record


def write_comparison(output, records):
    output = Path(output)
    dense = next((r for r in records if r["kind"] == "dense_reference"), None)
    rows = []
    lines = ["# One-shot pruning — frozen CIFAR-10 TEST", "",
             "Exploratory comparison; no test-based checkpoint selection or training.",
             "Each pruned experiment: 150 search + 150 fresh scratch epochs. Dense reference: separate 150 epochs.", "",
             "| Job | Test accuracy | Physical parameters | Search epochs | Scratch epochs | Dense epochs |",
             "|---|---:|---:|---:|---:|---:|"]
    for record in records:
        test = record["test"]
        rows.append({"Run": record["job"], "training_protocol": PROTOCOL,
                     "model.trainable_parameters": record["physical_parameters"],
                     "test_accuracy": test["accuracy"], "test_correct_count": test["correct_count"],
                     "test_example_count": test["example_count"], "test_ce_loss": test["ce_loss"],
                     "validation_accuracy": record["validation"]["accuracy"],
                     "search_epochs": record["search_epochs"], "retrain_epochs": record["retrain_epochs"],
                     "dense_epochs": record["dense_epochs"], "epochs_consumed": record["epochs_consumed"],
                     "adaptive_soft_drop": record.get("adaptive_soft_drop"),
                     "adaptive_hard_drop": record.get("adaptive_hard_drop"),
                     "reinit_seed": record.get("reinit_seed"),
                     "removed_parameter_fraction": record.get("removed_parameter_fraction"),
                     "conv_linear_macs_per_image": record["conv_linear_macs_per_image"],
                     "test_delta_vs_dense_pp": 100 * (test["accuracy"] - dense["test"]["accuracy"]) if dense else None,
                     "checkpoint_sha256": record["checkpoint_sha256"]})
        lines.append(f"| {record['job']} | {test['accuracy']:.2%} | {record['physical_parameters']:,} | "
                     f"{record['search_epochs']} | {record['retrain_epochs']} | {record['dense_epochs']} |")
    if rows:
        with (output / "test_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / "test_comparison.md").write_text("\n".join(lines) + "\n")


def run(args):
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output or run_dir / "test_evaluation").resolve()
    require(args.jobs and len(args.jobs) == len(set(args.jobs)), "Use distinct jobs")
    if not args.check_only and output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not args.check_only and str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the GPU server or explicitly select --device cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared = [prepare_job(run_dir, job, args.reference_source) for job in args.jobs]
    reference = prepared[0][1]
    if args.reference_source is not None and DENSE not in args.jobs:
        # The launcher evaluates each job in its own process. Still check its
        # provenance against dense even when dense is not being tested again.
        reference_model, reference = prepare_job(run_dir, DENSE, args.reference_source)
        del reference_model
    for _, record in prepared:
        for key in ("seed", "common_init_hash", "split_indices_hash", "training_signature"):
            require(record[key] == reference[key], f"Unmatched {key}")
    if args.check_only:
        print("Deployments verified. Test data not loaded; no outputs written.", flush=True)
        return None
    loader, dataset = build_test_loader(args.data, args.batch_size, args.num_workers, args.device, args.download)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "protocol": "frozen_one_shot_test_v1", "source_run": str(run_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "test_evaluated": False,
        "comparison_scope": "exploratory", "training_performed": False,
        "bn_recalibration": False, "test_based_selection": False,
        "device": str(args.device), "batch_size": args.batch_size, "precision": "fp32",
        "python": sys.version, "torch": str(torch.__version__), "dataset": dataset,
        "evaluation_script_sha256": file_hash(__file__), "planned_jobs": list(args.jobs), "runs": [],
    }
    write_json(output / "evaluation_plan.json", {**report, "selected_deployments": [r for _, r in prepared]})
    write_json(output / "test_summary.json", report)
    try:
        for model, record in prepared:
            metrics, arrays = evaluate_fixed(model, loader, args.device, job=record["job"])
            require(np.array_equal(arrays["label"], np.asarray(loader.dataset.targets)), "Test ordering differs")
            path = output / f"{record['job']}_predictions.npz"
            np.savez_compressed(path, **arrays)
            report["runs"].append({**record, "test": metrics, "predictions": path.name,
                                   "predictions_sha256": file_hash(path), "model_state_unchanged": True})
            report["test_evaluated"] = True
            write_json(output / "test_summary.json", report)
            write_comparison(output, report["runs"])
            print(f"{record['job']}: test={metrics['accuracy']:.2%}, "
                  f"params={record['physical_parameters']:,}", flush=True)
        report.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        report.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "test_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Positive batch size and nonnegative workers required")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    run(args)


if __name__ == "__main__":
    main()
