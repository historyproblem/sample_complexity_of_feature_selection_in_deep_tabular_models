"""Run a validation-only 30-epoch ResNet50 pruning LR/scheduler grid.

The grid crosses two cosine horizons (30 and 75 epochs) with six initial/base
AdamW learning rates. Every job executes exactly 30 adaptive-pruning epochs,
uses eta_min=0, and promotes the epoch-30 checkpoint to ``last_checkpoint.pt``.
The standard validation-selected checkpoint remains untouched. No recovery or
test evaluation is performed by this tuning study.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from time import time

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import launch_epoch_split_study as epoch_split


ROOT = Path(__file__).resolve().parents[1]
ONE_SHOT = ROOT / "scripts/launch_one_shot_pruning.py"
CONFIG = "experiment/pruning_v3/resnet50_cifar10_pruning30_lr_grid_search"
DEFAULT_REFERENCE = Path("outputs/runs/epoch_split_dense_reference_seed42_fresh")
DEFAULT_OUTPUT = Path("outputs/runs/resnet50_cifar10_pruning30_lr_grid_v1")
SEARCH_EPOCHS = 30
RECOVERY_EPOCHS = 120
ETA_MIN = 0.0
INITIAL_LRS = (0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001)
T_MAX_VALUES = (30, 75)
EXPECTED_HOURS_PER_SEARCH = 0.75


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lr_id(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("0.", "").replace(".", "p")


def _profiles():
    return [
        {
            "id": f"tmax{t_max}_lr{_lr_id(lr)}",
            "t_max": t_max,
            "initial_lr": lr,
            "eta_min": ETA_MIN,
            "search_epochs": SEARCH_EPOCHS,
            "recovery_epochs_reserved": RECOVERY_EPOCHS,
        }
        for t_max in T_MAX_VALUES
        for lr in INITIAL_LRS
    ]


def _overrides(row):
    return (
        f"optimizer.lr={row['initial_lr']:.12g}",
        f"scheduler.eta_min={row['eta_min']:.12g}",
        f"one_shot.search_scheduler_eta_min={row['eta_min']:.12g}",
        f"one_shot.search_scheduler_horizon_epochs={row['t_max']}",
    )


def _launcher_command(reference, output, row):
    command = [
        sys.executable,
        str(ONE_SHOT),
        "--config-name", CONFIG,
        "--dense-source", str(reference),
        "--output", str(output),
    ]
    for override in _overrides(row):
        command.extend(["--override", override])
    command.append("--search-only")
    return command


def validate_grid_configs():
    sys.path.insert(0, str(ROOT / "src"))
    from net_complexity.training import one_shot_pruning_config as schema

    verified = []
    for row in _profiles():
        config = schema.compose_config(CONFIG, overrides=list(_overrides(row)))
        _require(schema.validate_config(config) == 150,
                 f"{row['id']}: paired budget is not 150 epochs")
        _require(int(config.one_shot.search_epochs) == SEARCH_EPOCHS
                 and int(config.one_shot.final_epochs) == RECOVERY_EPOCHS,
                 f"{row['id']}: configured epoch split changed")
        _require(int(config.one_shot.search_scheduler_horizon_epochs) == row["t_max"],
                 f"{row['id']}: cosine horizon override was not resolved")
        _require(float(config.optimizer.lr) == row["initial_lr"],
                 f"{row['id']}: initial LR override was not resolved")
        _require(float(config.scheduler.eta_min) == ETA_MIN
                 and float(config.one_shot.search_scheduler_eta_min) == ETA_MIN,
                 f"{row['id']}: eta_min must remain zero")
        _require(config.training_arguments.adaptive_lambda.enabled is True
                 and config.training_arguments.adaptive_lambda.control_mode == "accuracy_only",
                 f"{row['id']}: adaptive lambda changed")
        _require(str(config.training_arguments.adaptive_lambda.log_step).lower() == "auto",
                 f"{row['id']}: lambda step is not automatic")
        _require(float(config.accuracy_guided.eligibility.min_keep_ratio) == 0.001,
                 f"{row['id']}: shared keep floor changed")
        _require(dict(config.model.criterion) == {
            "_target_": "torch.nn.CrossEntropyLoss"
        }, f"{row['id']}: loss is not plain CE")
        _require(float(config.optimizer.weight_decay) == 0.0005,
                 f"{row['id']}: AdamW weight decay changed")
        final_lr_factor = (
            1.0 + math.cos(math.pi * SEARCH_EPOCHS / row["t_max"])
        ) / 2.0
        verified.append({**row, "lr_factor_after_epoch_30": final_lr_factor})
    return verified


def _run_dir(output, row):
    return output / row["id"]


def _nested_last_checkpoint(run_dir):
    matches = sorted(
        (Path(run_dir) / "shared_search" / "training").glob(
            "*/checkpoints/epoch_0030.pt"
        )
    )
    _require(len(matches) == 1,
             f"Expected exactly one nested epoch-30 checkpoint under {run_dir}")
    return matches[0]


def _trace_record(selection, epoch):
    trace = selection.get("trace", {})
    records = trace.get("trace") if isinstance(trace, dict) else trace
    _require(isinstance(records, list), "Selection trace is missing")
    matches = [record for record in records if int(record.get("epoch", -1)) == epoch]
    _require(len(matches) == 1, f"Selection trace has no unique epoch {epoch}")
    return matches[0]


def _promote_last_checkpoint(run_dir, row):
    run_dir = Path(run_dir)
    nested = _nested_last_checkpoint(run_dir)
    payload = torch.load(nested, map_location="cpu", weights_only=True)
    _require(int(payload.get("epoch", -1)) == SEARCH_EPOCHS,
             f"Last checkpoint epoch differs: {nested}")
    _require(int(payload.get("scheduler_state_dict", {}).get("T_max", -1)) == row["t_max"],
             f"Last checkpoint scheduler horizon differs: {nested}")
    _require(int(payload.get("scheduler_step_count", -1)) == SEARCH_EPOCHS,
             f"Last checkpoint scheduler step count differs: {nested}")
    promoted = run_dir / "last_checkpoint.pt"
    temporary = run_dir / "last_checkpoint.pt.tmp"
    shutil.copy2(nested, temporary)
    temporary.replace(promoted)
    _require(_file_hash(promoted) == _file_hash(nested),
             f"Promoted checkpoint hash differs: {promoted}")
    selection = json.loads((run_dir / "selection.json").read_text())
    trace = _trace_record(selection, SEARCH_EPOCHS)
    metrics = payload.get("metrics", {})
    result = {
        **row,
        "run_dir": str(run_dir.resolve()),
        "last_checkpoint": str(promoted.resolve()),
        "last_checkpoint_sha256": _file_hash(promoted),
        "last_epoch": SEARCH_EPOCHS,
        "last_validation_accuracy": float(metrics["valid_accuracy"]),
        "last_validation_ce_loss": float(metrics["valid_ce_loss"]),
        "last_physical_parameters": int(trace["physical_cost"]),
        "last_dependency_safe_accuracy": float(trace["accuracy"]),
        "selected_epoch": int(selection["selected_epoch"]),
        "selected_policy": selection["policy"],
    }
    manifest = run_dir / "last_checkpoint.json"
    epoch_split._write_json(manifest, result)
    return result, nested


def _prune_other_nested_checkpoints(run_dir, keep):
    removed_files = 0
    removed_bytes = 0
    keep = Path(keep).resolve()
    for path in sorted(Path(run_dir).glob("*/training/*/checkpoints/*.pt")):
        if path.resolve() == keep:
            continue
        removed_bytes += path.stat().st_size
        path.unlink()
        removed_files += 1
    return {"files": removed_files, "bytes": removed_bytes,
            "preserved_nested_checkpoint": str(keep)}


def _verify_completed(run_dir, row):
    epoch_split._search_result(run_dir, SEARCH_EPOCHS, RECOVERY_EPOCHS)
    config_path = Path(run_dir) / "resolved_config.yaml"
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    _require(float(config.optimizer.lr) == row["initial_lr"],
             f"Completed run LR differs: {run_dir}")
    _require(int(config.one_shot.search_scheduler_horizon_epochs) == row["t_max"],
             f"Completed run T_max differs: {run_dir}")
    return _promote_last_checkpoint(run_dir, row)


def _write_csv(path, rows):
    fields = (
        "id", "t_max", "initial_lr", "eta_min", "lr_factor_after_epoch_30",
        "last_epoch", "last_validation_accuracy", "last_validation_ce_loss",
        "last_physical_parameters", "last_dependency_safe_accuracy",
        "selected_epoch", "selected_policy", "last_checkpoint",
        "last_checkpoint_sha256", "run_dir",
    )
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-output", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=55.0)
    args = parser.parse_args(argv)

    reference = args.reference_output.expanduser().resolve()
    output = args.output.expanduser().resolve()
    state_path = output / "study_state.json"
    results_csv = output / "validation_results_epoch30.csv"
    profiles = validate_grid_configs()
    environment = epoch_split._environment_preflight(float(args.min_free_gib))
    reference_result = epoch_split._reference_result(reference)
    preflight = {
        "status": "ready",
        "protocol": "resnet50_cifar10_pruning30_initial_lr_tmax_grid_v1",
        "profiles": profiles,
        "fresh_searches": sum(not _run_dir(output, row).exists() for row in profiles),
        "estimated_single_gpu_hours": sum(
            EXPECTED_HOURS_PER_SEARCH
            for row in profiles if not _run_dir(output, row).exists()
        ),
        "environment": environment,
        "reference": reference_result,
        "selection_policy": (
            "rank epoch-30 checkpoints by validation accuracy, then lower CE; no test"
        ),
    }
    print(json.dumps(preflight, indent=2, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return preflight

    if state_path.exists() and not args.resume_completed:
        raise FileExistsError(
            f"Refusing existing study state: {state_path}. Use --resume-completed."
        )
    output.mkdir(parents=True, exist_ok=True)
    state = {
        **preflight,
        "status": "running",
        "started_unix": time(),
        "runs": {},
        "test_evaluated": False,
    }
    epoch_split._write_json(state_path, state)
    try:
        results = []
        for index, row in enumerate(profiles, 1):
            run_dir = _run_dir(output, row)
            print(
                f"[lr-grid30] {index}/{len(profiles)}: T_max={row['t_max']} "
                f"initial_lr={row['initial_lr']:.4g}",
                flush=True,
            )
            if run_dir.exists():
                _require(args.resume_completed,
                         f"Existing output for {row['id']}; pass --resume-completed")
            else:
                epoch_split._run(_launcher_command(reference, run_dir, row))
            result, nested_last = _verify_completed(run_dir, row)
            if not args.keep_all_checkpoints:
                result["checkpoint_cleanup"] = _prune_other_nested_checkpoints(
                    run_dir, nested_last
                )
            results.append(result)
            state["runs"][row["id"]] = result
            epoch_split._write_json(state_path, state)

        ranked = sorted(results, key=lambda row: (
            -row["last_validation_accuracy"],
            row["last_validation_ce_loss"],
            -row["last_physical_parameters"],
            row["t_max"],
            -row["initial_lr"],
        ))
        _write_csv(results_csv, results)
        state["validation_results_csv"] = str(results_csv)
        state["validation_ranking"] = [row["id"] for row in ranked]
        state["winner"] = ranked[0]
        state["validation_locked"] = True
        state["status"] = "completed"
        state["finished_unix"] = time()
        epoch_split._write_json(state_path, state)
        epoch_split._write_json(output / "study_summary.json", state)
        print(
            f"[lr-grid30] completed; validation winner={ranked[0]['id']}; "
            f"results={results_csv}",
            flush=True,
        )
        return state
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["finished_unix"] = time()
        epoch_split._write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
