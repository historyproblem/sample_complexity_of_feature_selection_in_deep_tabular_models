"""Sequential adaptive-lambda pruning, capped at 150 epochs per new model.

The existing dense validation curve is read-only controller reference, not a
trained initializer. Frozen deployments are evaluated on test after the queue;
test results never enter checkpoint selection or subsequent training decisions.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

from pruning_pilot_common import ROOT, config_for
from launch_pruning_pilot import run_child, write_comparison
from evaluate_pruning_test import ADAPTIVE_JOBS, file_hash, prepare_job

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from net_complexity.training.pruning_audit import validate_config
from net_complexity.training.pruning_measurement import state_hash, write_json

PROTOCOL = "adaptive_lambda_v1"
DENSE = "J1_dense_control"
TEST_RESERVE_SECONDS = 180
PREFLIGHT_TESTS = [
    "tests/test_adaptive_pruning_audit.py", "tests/test_adaptive_lambda_handoff.py",
    "tests/test_pruning_audit.py", "tests/test_adaptive_pruning_launcher.py",
    "tests/test_pruning_test_evaluation.py", "tests/test_pruned_bottleneck.py",
    "tests/test_cyclic_channel_weight_handoff.py", "tests/test_best_checkpoint_evaluation.py",
    "tests/test_optimizer_groups.py", "tests/test_dataloaders.py",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_plan(args):
    name = args.config_name.removesuffix(".yaml")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None, "Invalid --config-name")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name=name)
    plan = OmegaConf.to_container(config, resolve=True)
    require(isinstance(plan, dict) and set(plan) == {
        "name", "protocol", "jobs", "dense_source", "hours", "data", "evaluate_test", "run_history"
    }, "Unexpected adaptive launcher fields")
    require(plan["protocol"] == PROTOCOL, "Adaptive protocol must not be disabled")
    require(plan["evaluate_test"] is True, "Final frozen test evaluation is required")
    jobs = plan["jobs"]
    require(isinstance(jobs, list) and jobs and all(isinstance(j, str) and j in ADAPTIVE_JOBS for j in jobs)
            and len(jobs) == len(set(jobs)), "Use distinct adaptive jobs; no dense retraining")
    hours = args.hours if args.hours is not None else plan["hours"]
    require(type(hours) in (float, int) and math.isfinite(hours) and 0 < hours <= 24,
            "hours must be in (0, 24]")
    require(isinstance(plan["name"], str) and re.fullmatch(r"[A-Za-z0-9_-]+", plan["name"]),
            "Invalid output name")
    require(isinstance(plan["run_history"], dict) and set(plan["run_history"]) == {"root_dir"},
            "run_history supports root_dir only")
    for key, value in (("data", args.data or plan["data"]),
                       ("dense_source", args.dense_source or plan["dense_source"]),
                       ("root_dir", plan["run_history"]["root_dir"])):
        require(isinstance(value, (str, Path)) and str(value).strip(), f"{key} must be a path")
    output = args.output
    if output is None:
        require(args.job is None, "Internal jobs need explicit --output")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = Path(plan["run_history"]["root_dir"]) / f"{stamp}_{plan['name']}"
    return {**plan, "config_name": name, "hours": hours,
            "data": str(Path(args.data or plan["data"]).resolve()),
            "dense_source": str(Path(args.dense_source or plan["dense_source"]).resolve()),
            "output": str(output.resolve()), "epochs_per_new_model": 150,
            "test_comparison_scope": "exploratory; prior test results informed experimentation"}


def validate_nightly_config(config):
    require(config.cyclic_channel_pruning.audit_protocol == PROTOCOL, "Adaptive protocol required")
    require(config.training_arguments.adaptive_lambda.enabled is True, "Adaptive lambda must remain enabled")
    require(config.training_arguments.adaptive_lambda.baseline_history_dir is None,
            "Hidden baseline training is forbidden; use the verified dense reference")
    require(config.seed == 42, "Matched nightly comparison requires seed 42")
    require(validate_config(config) == 150, "Each new model must consume exactly 150 total epochs")
    require(config.cyclic_channel_pruning.ranking == "learned", "Night queue requires learned ranking")


def validate_dense_source(source, new_config):
    """Validate source without modifying it, including a relocated run tree."""
    source = Path(source).resolve()
    source_job = source if source.name == DENSE else source / DENSE
    source_parent = source_job.parent
    state = json.loads((source_job / "pilot_state.json").read_text())
    require(state["status"] == "completed" and state["global_epochs_completed"] == 150
            and state["total_epochs_allocated"] == 150 and not state["accepted_mask"],
            "Reference must be the completed, unpruned 150-epoch dense model")
    require(not state["test_evaluated"], "Dense training state must not use test for selection")
    require(state["seed"] == int(new_config.seed), "Dense seed differs")
    require(state["validation"]["example_count"] == 5000, "Dense validation split must contain 5,000 examples")
    source_config = OmegaConf.load(source_job / "resolved_config.yaml")
    require(any(OmegaConf.select(metric, "_target_") == "net_complexity.metrics.classification.Accuracy"
                and OmegaConf.select(metric, "return_counts") is True
                for metric in source_config.metrics.valid_metrics),
            "Dense source must use sample-weighted validation accuracy")
    for key in ("dataloaders.train_val_ratio", "dataloaders.seed", "dataloaders.loader_seed",
                "dataloaders.batch_size", "dataloaders.taskname", "dataloaders._target_",
                "optimizer", "scheduler", "model.backbone.num_classes"):
        require(OmegaConf.select(source_config, key) == OmegaConf.select(new_config, key),
                f"Reference and requested run differ: {key}")
    initializer = source_parent / "shared_random_seed42.pt"
    initial = torch.load(initializer, map_location="cpu", weights_only=True)
    require(initial.get("trained_epochs") == 0 and initial.get("seed") == int(new_config.seed),
            "Initializer must be the original zero-epoch random model")
    require(state_hash(initial["model_state_dict"]) == state["common_init_hash"]
            == initial.get("model_state_hash"), "Dense random initializer hash differs")
    del initial
    # Also validates deployment weights, physical shape, selected metrics, and mask.
    model, deployment_record = prepare_job(source_parent, DENSE)
    del model
    history = source_job / "global_history.csv"
    with history.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 150, "Dense reference must contain all 150 validation epochs")
    steps, remainder = divmod(int(state["optimizer_steps_total"]), 150)
    require(remainder == 0 and steps > 0, "Invalid dense optimizer step count")
    for epoch, row in enumerate(rows, 1):
        require(int(row["global_epoch"]) == epoch
                and int(row["optimizer_steps_total"]) == epoch * steps,
                "Dense reference epoch/step sequence differs")
        require(math.isfinite(float(row["valid_accuracy"])) and 0 <= float(row["valid_accuracy"]) <= 1,
                "Invalid dense validation accuracy")
        correct = float(row["valid_accuracy"]) * 5000
        require(abs(correct - round(correct)) < 1e-8,
                "Dense reference accuracy is not consistent with 5,000 validation examples")
    return {"state": {**state, "reused_from": str(source_job)}, "config": source_config,
            "initializer": initializer, "history": history,
            "provenance": {"source_parent": str(source_parent), "dense_source": str(source_job),
                           "history_sha256": file_hash(history), "initializer_sha256": file_hash(initializer),
                           "deployment": deployment_record}}


def provenance():
    hashes = {str(p.relative_to(ROOT)): file_hash(p)
              for directory in ("src", "scripts", "configs", "tests")
              for p in sorted((ROOT / directory).rglob("*"))
              if p.is_file() and p.suffix in (".py", ".yaml", ".sh")}
    return {"protocol": PROTOCOL, "files_sha256": hashes, "python": sys.version,
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "new_training_epochs_per_model": 150,
            "test_used_for_selection": False, "test_comparisons": "exploratory"}


def enough_time_for_job(remaining_seconds, previous_seconds):
    return remaining_seconds > 0 and (previous_seconds is None or remaining_seconds >= previous_seconds * 1.15)


def run_queue(plan):
    output = Path(plan["output"])
    require(not output.exists(), f"Refusing to overwrite existing output: {output}")
    # Validate configs and source before reserving a results directory or training.
    os.environ["AUDIT_INIT_CHECKPOINT"] = str(output / "shared_random_seed42.pt")
    configurations = {job: config_for(job) for job in plan["jobs"]}
    for cfg in configurations.values():
        cfg.dataloaders.path_to_data = plan["data"]
        cfg.cyclic_channel_pruning.adaptive_reference_history = str(output / "adaptive_reference_history.csv")
        validate_nightly_config(cfg)
    dense = validate_dense_source(plan["dense_source"], configurations[plan["jobs"][0]])
    output.mkdir(parents=True, exist_ok=False)
    print(f"Results: {output}", flush=True)
    deadline = time.monotonic() + plan["hours"] * 3600
    training_deadline = deadline - TEST_RESERVE_SECONDS
    status = {"status": "preflight", "protocol": PROTOCOL, "planned_jobs": plan["jobs"],
              "completed_jobs": [], "skipped_jobs": [], "incomplete_jobs": [], "test_evaluated": False,
              "reused_jobs": [DENSE], "new_training_epochs_per_model": 150}
    previous_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        require(shutil.disk_usage(output).free >= 30 * 1024 ** 3, "Need at least 30 GiB free for checkpoints")
        OmegaConf.save(OmegaConf.create(plan), output / "launcher_config.yaml", resolve=True)
        write_json(output / "provenance.json", provenance())
        shutil.copyfile(dense["initializer"], output / "shared_random_seed42.pt")
        shutil.copyfile(dense["history"], output / "adaptive_reference_history.csv")
        write_json(output / "reuse_dense_provenance.json", dense["provenance"])
        write_json(output / DENSE / "pilot_state.json", dense["state"])
        OmegaConf.save(dense["config"], output / f"{DENSE}_resolved.yaml", resolve=True)
        for job, cfg in configurations.items():
            OmegaConf.save(cfg, output / f"{job}_resolved.yaml", resolve=True)
        write_json(output / "nightly_status.json", status)
        run_child([sys.executable, "-m", "pytest", "-q", *[str(ROOT / p) for p in PREFLIGHT_TESTS]],
                  output / "preflight_tests.log", training_deadline)
        run_child([sys.executable, str(ROOT / "scripts/smoke_adaptive_pruning.py"),
                   "--output", str(output / "gpu_smoke"), "--device", "cuda:0"],
                  output / "preflight_gpu_smoke.log", training_deadline)
        print("[reuse] Dense validation reference and zero-epoch random initializer verified; no dense retraining.", flush=True)
        previous_seconds = None
        for index, job in enumerate(plan["jobs"]):
            if not enough_time_for_job(training_deadline - time.monotonic(), previous_seconds):
                status["skipped_jobs"] = plan["jobs"][index:]
                break
            status["status"] = f"running_{job}"
            write_json(output / "nightly_status.json", status)
            started = time.monotonic()
            try:
                run_child([sys.executable, str(ROOT / "scripts/launch_adaptive_pruning.py"),
                           "--config-name", plan["config_name"], "--job", job,
                           "--job-config", str(output / f"{job}_resolved.yaml"),
                           "--output", str(output / job)], output / f"{job}.log", training_deadline)
            except TimeoutError:
                # A true deadline leaves a reserve to evaluate previous completed
                # models. A manual interruption must still stop the whole queue.
                if time.monotonic() < training_deadline or not status["completed_jobs"]:
                    raise
                status["incomplete_jobs"].append(job)
                status["skipped_jobs"] = plan["jobs"][index + 1:]
                break
            previous_seconds = time.monotonic() - started
            result = json.loads((output / job / "pilot_state.json").read_text())
            require(result["status"] == "completed" and result["global_epochs_completed"] == 150
                    and result["pilot_version"] == 2 and result["protocol"] == PROTOCOL,
                    f"{job}: incomplete or non-adaptive result")
            for key in ("common_init_hash", "split_indices_hash", "optimizer_steps_total"):
                require(result[key] == dense["state"][key], f"{job}: unmatched {key}")
            status["completed_jobs"].append(job)
            write_comparison(output, [DENSE, *status["completed_jobs"]], "adaptive_lambda_nightly")
            print(f"{job}: valid={result['validation']['accuracy']:.2%}; "
                  f"params={result['final_cost']['physical_total_parameters']:,}; "
                  f"wall={previous_seconds / 60:.1f} min. Frozen test evaluation follows the queue.", flush=True)
        require(bool(status["completed_jobs"]), "No new model completed within the wall-clock budget")
        status["status"] = "evaluating_frozen_test"
        write_json(output / "nightly_status.json", status)
        run_child([sys.executable, str(ROOT / "scripts/evaluate_pruning_test.py"),
                   "--run-dir", str(output), "--data", plan["data"],
                   "--jobs", DENSE, *status["completed_jobs"]], output / "test_evaluation.log", deadline)
        test = json.loads((output / "test_evaluation/test_summary.json").read_text())
        require(test["status"] == "completed" and test["test_evaluated"], "Test evaluation did not complete")
        status.update(test_evaluated=True, test_comparisons="exploratory",
                      status=("completed_partial_wall_budget"
                              if status["skipped_jobs"] or status["incomplete_jobs"] else "completed"))
    except BaseException as exc:
        status.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        write_json(output / "nightly_status.json", status)
        print(f"Status: {status['status']}; logs and checkpoints: {output}", flush=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pruning_adaptive_nightly")
    parser.add_argument("--cfg", choices=["job"], help="Print resolved plan without GPU checks or writes")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--output", type=Path, help="Must be new; default outputs/runs/<timestamp>_<name>")
    parser.add_argument("--dense-source", type=Path, help="Completed original dense run parent or J1 folder")
    parser.add_argument("--job", choices=ADAPTIVE_JOBS, help=argparse.SUPPRESS)
    parser.add_argument("--job-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        plan = resolve_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.cfg:
        print(OmegaConf.to_yaml(OmegaConf.create(plan)), end="")
        return
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable; refusing overnight CPU training")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.job:
        require(args.job_config is not None, "Internal --job requires its frozen --job-config")
        config = OmegaConf.load(args.job_config)
        require(config.mlflow.run_name == args.job, "Job/config identity differs")
        validate_nightly_config(config)
        from net_complexity.training.pruning_audit import run_adaptive_pruning_pilot
        run_adaptive_pruning_pilot(config, Path(plan["output"]))
    else:
        run_queue(plan)


if __name__ == "__main__":
    main()
