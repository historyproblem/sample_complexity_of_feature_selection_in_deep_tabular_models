"""Evaluate frozen one-shot physical branches on one official CIFAR-10 test loader.

Training outputs are read-only. Checkpoint validation finishes before test data
is opened; evaluation performs no training, selection, or BN calibration.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import io
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from omegaconf import OmegaConf
import torch

import evaluate_pruning_test as frozen
from net_complexity.models.channel_pruning import build_structurally_pruned_model_from_config
from net_complexity.training.pruning_measurement import deployment_cost, mask_hash, state_hash, write_json

PROTOCOL = "pruning_v3_one_shot_60_90"
BRANCHES = ("inherited", "scratch")
DEFAULT_CONFIG = ROOT / "configs/evaluation/one_shot_test.yaml"
require = frozen.require
read_json = frozen.read_json
file_hash = frozen.file_hash


def evaluation_args_from_config(config_path=DEFAULT_CONFIG, **overrides):
    """Resolve the checked-in server inference profile with explicit overrides."""
    path = Path(config_path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    require(isinstance(raw, dict) and set(raw) == {"one_shot_test"},
            "Test config must contain exactly the one_shot_test mapping")
    values = raw["one_shot_test"]
    fields = {"run_dir", "data", "device", "output_subdir", "batch_size", "num_workers", "download"}
    require(isinstance(values, dict) and set(values) == fields,
            f"Test config fields differ: unknown={sorted(set(values) - fields)}, "
            f"missing={sorted(fields - set(values))}")
    for key, value in overrides.items():
        if value is not None:
            require(key in fields | {"output", "check_only"}, f"Unknown test override: {key}")
            values[key] = value
    run_dir = Path(values["run_dir"]).expanduser()
    data = Path(values["data"]).expanduser()
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not data.is_absolute():
        data = ROOT / data
    output = values.get("output")
    output = Path(output).expanduser() if output is not None else run_dir / str(values["output_subdir"])
    if not output.is_absolute():
        output = ROOT / output
    batch_size, num_workers = int(values["batch_size"]), int(values["num_workers"])
    require(batch_size > 0 and num_workers >= 0, "Use positive batch_size and nonnegative num_workers")
    return SimpleNamespace(
        run_dir=run_dir.resolve(), data=data.resolve(), device=str(values["device"]),
        output=output.resolve(), batch_size=batch_size, num_workers=num_workers,
        download=bool(values["download"]), check_only=bool(values.get("check_only", False)),
        config_path=path,
    )


def _validate_normalization(config, metadata):
    require(isinstance(metadata, dict), "Missing original normalization metadata")
    widths = {}
    block = config.model.backbone.resnet_block
    base = int(OmegaConf.select(config, "model.backbone.base_width", default=64))
    for stage, count in enumerate((3, 4, 6, 3), 1):
        for index in range(count):
            prefix = f"backbone.layer{stage}.{index}."
            width = base * 2 ** (stage - 1)
            if block.gate_internal_width:
                widths.update({prefix + "mid1_gumbel_layer": width, prefix + "mid2_gumbel_layer": width})
            if block.gate_output:
                widths[prefix + "gumbel_layer"] = width * 4
    require(metadata == {"version": 1, "M0": len(widths), "n_b0": widths,
                         "normalization": {name: "initial_channels" for name in widths},
                         "scaling_contract": "survivor_equivalent_v1"},
            "Original boundary ids/widths/M0 or normalization contract differ")


def _validate_ledger(ledger, *, total, search):
    required = {"global_training_epoch", "search_epochs_consumed", "optimizer_updates", "consumed_training_examples"}
    require(isinstance(ledger, dict) and required <= set(ledger)
            and all(type(ledger[key]) is int and ledger[key] >= 0 for key in required),
            "Invalid consumed training ledger")
    require(ledger["global_training_epoch"] == total and ledger["search_epochs_consumed"] == search,
            "Incomplete one-shot training budget or incorrect search clock")
    require(ledger["optimizer_updates"] > 0 and ledger["consumed_training_examples"] > 0,
            "Completed deployment has no recorded training work")


def _validate_validation(metrics):
    require(isinstance(metrics, dict) and {"accuracy", "ce_loss", "correct_count", "example_count"} <= set(metrics),
            "Incomplete selected validation metrics")
    require(type(metrics["example_count"]) is int and metrics["example_count"] > 0
            and type(metrics["correct_count"]) is int and 0 <= metrics["correct_count"] <= metrics["example_count"],
            "Invalid validation sample counts")
    require(type(metrics["accuracy"]) in (int, float) and math.isfinite(metrics["accuracy"])
            and abs(metrics["accuracy"] - metrics["correct_count"] / metrics["example_count"]) < 1e-12
            and type(metrics["ce_loss"]) in (int, float) and math.isfinite(metrics["ce_loss"])
            and metrics["ce_loss"] >= 0, "Invalid selected validation accuracy/CE")


def prepare_branches(run_dir):
    """Validate finalized branches and their common frozen search selection."""
    from net_complexity.training.one_shot_pruning_config import (
        resolved_branch_plan, validate_config,
    )
    from net_complexity.training.one_shot_pruning import physical_architecture_signature
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "resolved_config.yaml"
    config = OmegaConf.load(config_path)
    validate_config(config)
    protocol = str(config.one_shot.protocol)
    branch_plan = resolved_branch_plan(config)
    branches = tuple(branch["id"] for branch in branch_plan)
    branch_specs = {branch["id"]: branch for branch in branch_plan}
    search_epochs = int(config.one_shot.search_epochs)
    final_epochs = int(config.one_shot.final_epochs)
    total_epochs = search_epochs + final_epochs
    require(not config.accuracy_guided.smoke,
            "Synthetic smoke artifacts must never access official test data")
    require(int(config.accuracy_guided.total_epochs) == 150,
            "Official evaluation requires the complete 150-epoch protocol")
    selected_path, selection_path = run_dir / "selected_checkpoint.pt", run_dir / "selection.json"
    selection = read_json(selection_path)
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    require(isinstance(selected, dict) and "model_state_dict" in selected,
            "Missing shared selected checkpoint tensors")
    selected_file_hash = file_hash(selected_path)
    selected_model_hash = state_hash(selected["model_state_dict"])
    require(selected.get("model_state_hash") == selected_model_hash,
            "Shared selected checkpoint tensor hash differs")
    identity = {"selected_checkpoint_id": selection.get("selected_checkpoint_id"),
                "selected_checkpoint_hash": selected_file_hash, "selected_model_state_hash": selected_model_hash}
    require(isinstance(identity["selected_checkpoint_id"], str) and bool(identity["selected_checkpoint_id"]),
            "Missing shared selected checkpoint identity")
    require(all(selection.get(key) == value for key, value in identity.items()),
            "Selection record differs from the frozen shared checkpoint")
    require(type(selection.get("selected_epoch")) is int
            and 1 <= selection["selected_epoch"] <= search_epochs
            and selection["selected_epoch"] == selected.get("epoch")
            and selection.get("reference_epoch") == search_epochs
            and selection.get("policy") == "best_feasible_compact",
            "Invalid common search checkpoint selection policy/epoch/reference")
    require(type(selection.get("quality_threshold")) in (int, float)
            and math.isfinite(selection["quality_threshold"]), "Invalid frozen search quality threshold")
    _validate_ledger(selection.get("search_ledger_consumed"), total=search_epochs, search=search_epochs)
    prepared = []
    for branch in branches:
        branch_spec = branch_specs[branch]
        state_path, checkpoint_path = run_dir / branch / "branch_state.json", run_dir / branch / "deployment.pt"
        state = read_json(state_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for artifact in (state, checkpoint):
            require(artifact.get("protocol") == protocol and artifact.get("artifact_type") == "physical_ungated"
                    and artifact.get("branch") == branch,
                    f"{branch}: unsupported protocol/type/branch; gated artifacts are never reinterpreted")
            require(all(artifact.get(key) == value for key, value in identity.items()),
                    f"{branch}: shared selected-checkpoint identity differs")
        require(state.get("status") in ("completed", "infeasible"), f"{branch}: deployment is not finalized")
        require(checkpoint.get("status") == state["status"], f"{branch}: deployment status differs")
        if protocol != PROTOCOL:
            for artifact in (state, checkpoint):
                require(artifact.get("method") == branch_spec["method"]
                        and artifact.get("repeat") == branch_spec["repeat"]
                        and artifact.get("training_seed") == branch_spec["training_seed"]
                        and artifact.get("optimizer_state_initialization") == branch_spec["optimizer_state"]
                        and artifact.get("scheduler_state_initialization") == branch_spec["scheduler_state"],
                        f"{branch}: configured method/repeat/handoff policy differs")
                optimizer_handoff = artifact.get("optimizer_handoff")
                scheduler_handoff = artifact.get("scheduler_handoff")
                require((optimizer_handoff is None if branch_spec["optimizer_state"] == "fresh"
                         else isinstance(optimizer_handoff, dict)
                         and optimizer_handoff.get("policy") == branch_spec["optimizer_state"]),
                        f"{branch}: optimizer handoff evidence differs")
                require((scheduler_handoff is None
                         if branch_spec["scheduler_state"] == "fresh_final_stage_cosine"
                         else isinstance(scheduler_handoff, dict)
                         and scheduler_handoff.get("policy") == branch_spec["scheduler_state"]),
                        f"{branch}: scheduler handoff evidence differs")
        require(state.get("training_initializer_verified") is True
                and isinstance(state.get("initialization_state_hash"), str) and bool(state["initialization_state_hash"]),
                f"{branch}: first-forward branch initialization provenance is incomplete")
        initialization = branch_spec["model_state"]
        require(state.get("initialization") == checkpoint.get("initialization") == initialization
                and state["initialization_state_hash"] == checkpoint.get("initialization_state_hash")
                and checkpoint.get("training_initializer_verified") is True,
                f"{branch}: branch initialization policy or snapshot differs")
        require(isinstance(state.get("architecture_hash"), str) and bool(state["architecture_hash"])
                and state["architecture_hash"] == checkpoint.get("architecture_hash"),
                f"{branch}: physical architecture identity differs")
        weights, mask = checkpoint["model_state_dict"], checkpoint["pruning_mask"]
        require(not any("gumbel" in name or "gate_logits" in name for name in weights),
                f"{branch}: ungated evaluator refuses gate tensors")
        require(mask == state.get("pruning_mask") and mask_hash(mask) == state.get("mask_hash")
                == checkpoint.get("mask_hash"), f"{branch}: branch mask/state mismatch")
        require(checkpoint.get("validation") == state.get("validation"), f"{branch}: validation provenance differs")
        _validate_validation(checkpoint["validation"])
        require(state.get("selection_policy") == checkpoint.get("selection_policy") == "best_validation_accuracy"
                and state.get("reference_epoch") == checkpoint.get("reference_epoch") == total_epochs
                and type(state.get("selected_final_epoch")) is int
                and 1 <= state["selected_final_epoch"] <= final_epochs
                and state["selected_final_epoch"] == checkpoint.get("selected_final_epoch"),
                f"{branch}: final validation selection protocol differs")
        threshold = state.get("quality_threshold")
        require(type(threshold) in (int, float) and math.isfinite(threshold)
                and threshold == checkpoint.get("quality_threshold")
                and type(state.get("quality_feasible")) is bool
                and state["quality_feasible"] == checkpoint.get("quality_feasible")
                == (checkpoint["validation"]["accuracy"] >= threshold)
                and state["quality_feasible"] == (state["status"] == "completed"),
                f"{branch}: saved validation quality status/threshold differs")
        require(checkpoint.get("ledger") == state.get("ledger"), f"{branch}: consumed ledger differs")
        _validate_ledger(checkpoint["ledger"], total=total_epochs, search=search_epochs)
        require(state.get("final_training_epochs_executed") == checkpoint.get("final_training_epochs_executed") == final_epochs
                and state.get("per_branch_total_allocated") == checkpoint.get("per_branch_total_allocated") == total_epochs,
                f"{branch}: final physical training allocation differs")
        require(all(checkpoint["ledger"][key] > selection["search_ledger_consumed"][key]
                    for key in ("optimizer_updates", "consumed_training_examples")),
                f"{branch}: physical-stage training work is missing from the consumed ledger")
        require(checkpoint.get("provenance") == state.get("provenance"), f"{branch}: deployment provenance differs")
        require(isinstance(checkpoint.get("provenance"), dict) and checkpoint["provenance"],
                f"{branch}: missing deployment provenance")
        provenance = checkpoint["provenance"]
        require(provenance.get("seed") == int(config.seed)
                and isinstance(provenance.get("split_indices_hash"), str) and bool(provenance["split_indices_hash"])
                and provenance.get("resolved_config_sha256") == hashlib.sha256(OmegaConf.to_yaml(config, resolve=True).encode()).hexdigest(),
                f"{branch}: saved configuration/seed/split provenance differs")
        require(provenance.get("source_weights") == "shared_zero_epoch_initializer_only"
                and provenance.get("dense_reference_weights_loaded") is False,
                f"{branch}: zero-epoch search initializer provenance is incomplete")
        normalization = checkpoint.get("normalization_metadata")
        require(normalization == state.get("normalization_metadata"), f"{branch}: normalization provenance differs")
        require(normalization == provenance.get("normalization"), f"{branch}: original normalization differs from provenance")
        _validate_normalization(config, normalization)
        expected_hash = state_hash(weights)
        require(expected_hash == checkpoint.get("model_state_hash") == state.get("model_state_hash"),
                f"{branch}: deployment tensor hash differs")
        require(all(isinstance(value, torch.Tensor) and (not value.is_floating_point()
                    or value.dtype == torch.float32 and bool(torch.isfinite(value).all()))
                    for value in weights.values()), f"{branch}: expected finite FP32 deployment tensors")
        frozen.validate_config_and_mask(config, mask)
        structural_config = deepcopy(config)
        structural_config.model.lambda_coef = 0.0
        pruning = OmegaConf.create({"mode": "explicit", "structural": True, "enabled": True, "mask": mask})
        with redirect_stdout(io.StringIO()):
            model = build_structurally_pruned_model_from_config(structural_config, pruning)
        model.load_state_dict(weights, strict=True)
        model.eval()
        architecture_hash = hashlib.sha256(json.dumps(physical_architecture_signature(model), sort_keys=True).encode()).hexdigest()
        require(architecture_hash == checkpoint["architecture_hash"],
                f"{branch}: saved architecture hash differs from constructed modules/tensors/indices")
        cost = deployment_cost(model)
        for key in ("physical_total_parameters", "conv_linear_macs_per_image"):
            require(cost[key] == state.get("final_cost", {}).get(key), f"{branch}: physical cost differs: {key}")
        require(state_hash(model.state_dict()) == expected_hash, f"{branch}: model loading/cost check mutated state")
        prepared.append((model, {
            "job": branch, "branch": branch, "method": branch_spec["method"],
            "repeat": branch_spec["repeat"], "training_seed": branch_spec["training_seed"],
            "optimizer_state_initialization": branch_spec["optimizer_state"],
            "scheduler_state_initialization": branch_spec["scheduler_state"],
            "initialization": initialization,
            "initialization_state_hash": checkpoint["initialization_state_hash"],
            "pilot_version": 3, "training_protocol": protocol,
            "artifact_type": "physical_ungated", "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_hash(checkpoint_path), "state_sha256": file_hash(state_path),
            "config_sha256": file_hash(config_path), "selection_sha256": file_hash(selection_path),
            "model_state_hash": expected_hash, "mask_hash": checkpoint["mask_hash"],
            "architecture_hash": checkpoint["architecture_hash"],
            "pruning_mask": deepcopy(mask), **identity,
            "normalization_metadata": normalization, "gate_regularization_normalization": "initial_channels",
            "scaling_contract": "survivor_equivalent_v1", "selection_provenance": checkpoint["provenance"],
            "ledger": checkpoint["ledger"], "validation": checkpoint["validation"],
            "quality_feasible": state.get("quality_feasible"),
            "validation_quality_threshold": threshold, "validation_reference_epoch": state["reference_epoch"],
            "final_selection_policy": state["selection_policy"], "selected_final_epoch": state["selected_final_epoch"],
            "physical_parameters": cost["physical_total_parameters"],
            "conv_linear_macs_per_image": cost["conv_linear_macs_per_image"],
        }))
    first = prepared[0][1]
    for key in ("pruning_mask", "mask_hash", "architecture_hash", "normalization_metadata", "selection_provenance", "physical_parameters",
                "conv_linear_macs_per_image", "ledger", "validation_quality_threshold",
                "validation_reference_epoch", "final_selection_policy", *identity):
        require(all(first[key] == record[key] for _, record in prepared[1:]),
                f"Physical branches do not share the required {key}")
    require(all(first["validation"]["example_count"] == record["validation"]["example_count"]
                for _, record in prepared[1:]),
            "Physical branches used different validation sample counts")
    if protocol != PROTOCOL:
        require(all(first["initialization"] == record["initialization"]
                    and first["initialization_state_hash"] == record["initialization_state_hash"]
                    for _, record in prepared[1:]),
                "Handoff-ablation branches do not share identical compact initialization")
    require(selection.get("pruning_mask") == first["pruning_mask"], "Committed mask differs from selected common mask")
    require(selection.get("mask_hash") == first["mask_hash"], "Committed mask hash differs from selection")
    return prepared


def run(args):
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output or run_dir / "one_shot_test_evaluation").resolve()
    if not args.check_only and output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}. Use a fresh --output directory.")
    if not args.check_only and str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the GPU server or explicitly pass --device cpu.")
    prepared = prepare_branches(run_dir)
    if args.check_only:
        print(f"{len(prepared)} frozen branches verified. Official test data not loaded; no outputs written.")
        return None
    protocol = prepared[0][1]["training_protocol"]
    branches = [record["branch"] for _, record in prepared]
    loader, dataset = frozen.build_test_loader(args.data, args.batch_size, args.num_workers, args.device, args.download)
    require(dataset.get("example_count") == 10000 and dataset.get("dataset") == "CIFAR10"
            and dataset.get("split") == "official_test", "Expected the full official CIFAR-10 test set")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "source_run": str(run_dir), "protocol": "frozen_one_shot_branches_test_v1",
        "training_protocol": protocol, "comparison_scope": "exploratory", "test_evaluated": False,
        "training_performed": False, "bn_recalibration": False, "test_based_selection": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": dataset,
        "planned_branches": branches, "device": str(args.device), "precision": "fp32",
        "batch_size": args.batch_size, "python": sys.version, "torch": str(torch.__version__),
        "evaluation_script_sha256": file_hash(__file__), "runs": []}
    write_json(output / "evaluation_plan.json", {**report, "selected_deployments": [row for _, row in prepared]})
    expected_labels = np.asarray(loader.dataset.targets)
    try:
        for model, record in prepared:
            bn_before = {name: value.detach().cpu().clone() for name, value in model.named_buffers()
                         if name.endswith("num_batches_tracked")}
            metrics, arrays = frozen.evaluate_fixed(model, loader, args.device, job=record["branch"])
            require(np.array_equal(arrays["label"], expected_labels), "Official test sample order differs")
            require(all(torch.equal(dict(model.named_buffers())[name].cpu(), value)
                        for name, value in bn_before.items()), "Frozen evaluation changed BatchNorm counters")
            prediction_path = output / f"{record['branch']}_predictions.npz"
            np.savez_compressed(prediction_path, **arrays)
            report["runs"].append({**record, "test": metrics, "model_state_unchanged": True,
                "bn_counters_unchanged": True, "predictions": prediction_path.name,
                "predictions_sha256": file_hash(prediction_path)})
            print(f"[official-test] {record['branch']}: accuracy={metrics['accuracy']:.2%}; "
                  f"ce_loss={metrics['ce_loss']:.6f}; "
                  f"correct={metrics['correct_count']}/{metrics['example_count']}", flush=True)
            report["test_evaluated"] = True
            write_json(output / "test_summary.json", report)
            frozen.write_comparison(output, report["runs"])
            comparison_path = output / "test_comparison.md"
            lines = comparison_path.read_text().splitlines()
            lines[2:2] = ["Exploratory comparison: prior test results informed further experimentation.", ""]
            comparison_path.write_text("\n".join(lines) + "\n")
        report.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        report.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "test_summary.json", report)
    if report["status"] == "completed" and branches == list(BRANCHES):
        by_branch = {row["branch"]: row["test"]["accuracy"] for row in report["runs"]}
        delta = 100 * (by_branch["inherited"] - by_branch["scratch"])
        print(f"[official-test] inherited-minus-scratch={delta:+.2f} pp", flush=True)
    print(f"[official-test] saved={output / 'test_summary.json'}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Server inference profile; command-line values override it")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, help="Fresh output directory; training artifacts stay read-only")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--check-only", action="store_true", help="CPU artifact verification, no official test access or writes")
    cli = parser.parse_args(argv)
    args = evaluation_args_from_config(
        cli.config, run_dir=cli.run_dir, data=cli.data, device=cli.device, output=cli.output,
        batch_size=cli.batch_size, num_workers=cli.num_workers, download=cli.download,
        check_only=cli.check_only,
    )
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return run(args)


if __name__ == "__main__":
    main()
