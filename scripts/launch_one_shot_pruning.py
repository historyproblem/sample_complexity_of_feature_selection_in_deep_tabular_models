"""Explicit one-shot queue: dense reference, 150 mask-search, fresh 150 scratch.

No pretrained weight handoff, param-budget selector, cyclic recovery, or test
selection. Run from repository root with .venv/bin/python; logs stream live.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import signal
import sys
import time

from pruning_pilot_common import ROOT
from launch_pruning_pilot import run_child

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from net_complexity.training.pruning_measurement import write_json

PROTOCOL = "one_shot_reinit_v1"
DENSE = "D0_dense_reference"
JOBS = ("O1_internal_gap05_1", "O2_internal_gap1_2", "O3_internal_gap2_5")
PREFLIGHT_TESTS = [
    "tests/test_one_shot_pruning.py", "tests/test_one_shot_launcher.py",
    "tests/test_one_shot_evaluation.py", "tests/test_adaptive_lambda.py",
    "tests/test_gumbel_regularization_normalization.py", "tests/test_pruned_bottleneck.py",
    "tests/test_best_checkpoint_evaluation.py", "tests/test_optimizer_groups.py",
    "tests/test_dataloaders.py",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def config_for(job):
    require(job in JOBS or job == DENSE, "Unknown one-shot job")
    source_job = JOBS[0] if job == DENSE else job
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name=None, overrides=[f"+experiment=one_shot/{source_job}"])
    config.mlflow.run_name = job
    config.run_history.run_name = job
    return config


def validate_model(config):
    """Read-only structural checks, including before reference files exist."""
    from net_complexity.training.one_shot_pruning import validate_config
    validate_config(config)
    c = config.one_shot_pruning
    require(c.protocol == PROTOCOL, "Wrong one-shot protocol")
    require(c.search_epochs == 150 and c.retrain_epochs == 150,
            "One-shot explicitly requires 150 search + 150 fresh scratch epochs")
    require(c.mask_threshold == 0.5 and c.probability_source == "raw_logits",
            "Physical selection must use raw p_open <= 0.5")
    require(c.selection_checkpoint == "best.pt", "Select checkpoint on validation")
    require(c.reinit_seed == 4242 and config.seed == 42, "Matched seeds must be 42 / 4242")
    require(config.training_arguments.adaptive_lambda.enabled is True,
            "Adaptive lambda must remain enabled during mask search")
    require(config.training_arguments.adaptive_lambda.baseline_history_dir is None,
            "Hidden baseline training forbidden; use explicit dense reference")
    soft = config.training_arguments.adaptive_lambda.soft_drop
    hard = config.training_arguments.adaptive_lambda.hard_drop
    require(0 < soft < hard < 1, "Adaptive gaps must be fractions with 0 < soft < hard < 1")
    require(OmegaConf.select(config, "cyclic_channel_pruning") is None,
            "One-shot must not invoke cyclic pruning")
    require(not config.training_arguments.evaluate_test and not config.dataloaders.include_test
            and list(config.metrics.test_metrics) == [], "Training must not access test")
    require(config.model.backbone.resnet_block.regularization_normalization == "initial_channels",
            "Preserve initial-channel penalty normalization")
    require(config.model.backbone.resnet_block.gate_internal_width is True
            and config.model.backbone.resnet_block.gate_output is False,
            "This queue compares internal-channel pruning only")
    require(config.training_arguments.num_epochs == 150 and config.scheduler.T_max == 150
            and config.scheduler._target_ == "torch.optim.lr_scheduler.CosineAnnealingLR",
            "Each phase needs a single uninterrupted 150-epoch cosine schedule")
    require(config.optimizer.gate_weight_decay_scale == 0, "Keep gate weight decay unchanged")


def resolve_plan(args):
    name = args.config_name.removesuffix(".yaml")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", name), "Invalid --config-name")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        plan = OmegaConf.to_container(compose(config_name=name), resolve=True)
    fields = {"name", "protocol", "jobs", "reference_source", "hours", "data", "device",
              "evaluate_test", "run_history"}
    require(isinstance(plan, dict) and set(plan) == fields, "Unexpected one-shot launcher fields")
    require(plan["protocol"] == PROTOCOL, "Wrong one-shot protocol")
    require(plan["evaluate_test"] is True, "Frozen test after each completed model is required")
    jobs = plan["jobs"]
    require(isinstance(jobs, list) and jobs and all(isinstance(j, str) and j in JOBS for j in jobs)
            and len(jobs) == len(set(jobs)), "Use distinct, predeclared one-shot jobs")
    require(isinstance(plan["name"], str) and re.fullmatch(r"[A-Za-z0-9_-]+", plan["name"]),
            "Invalid run name")
    require(isinstance(plan["run_history"], dict) and set(plan["run_history"]) == {"root_dir"},
            "run_history supports root_dir only")
    hours = args.hours if args.hours is not None else plan["hours"]
    require(hours is None or (type(hours) in (int, float) and math.isfinite(hours) and hours > 0),
            "hours must be null or a positive finite number")
    device = args.device or plan["device"]
    require(isinstance(device, str) and re.fullmatch(r"cuda:\d+", device),
            "Training requires an explicit CUDA device, e.g. cuda:0; no CPU fallback")
    data = args.data or plan["data"]
    reference = args.reference_source if args.reference_source is not None else plan["reference_source"]
    for key, value in (("data", data), ("root_dir", plan["run_history"]["root_dir"])):
        require(isinstance(value, (str, Path)) and str(value).strip(), f"{key} must be a path")
    require(reference is None or isinstance(reference, (str, Path)) and str(reference).strip(),
            "reference_source must be null or a completed one-shot dense path")
    output = args.output
    if output is None:
        require(args.job is None, "Internal jobs require explicit --output")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = Path(plan["run_history"]["root_dir"]) / f"{stamp}_{plan['name']}"
    reference_epochs = 150 if reference is None else 0
    return {**plan, "config_name": name, "data": str(Path(data).resolve()), "device": device,
            "output": str(Path(output).resolve()), "hours": hours,
            "reference_source": str(Path(reference).resolve()) if reference is not None else None,
            "search_epochs_per_job": 150, "scratch_epochs_per_job": 150,
            "new_training_epochs_per_job": 300,
            "dense_reference_epochs": 150, "new_dense_reference_epochs": reference_epochs,
            "total_new_training_epochs_allocated": 300 * len(jobs) + reference_epochs,
            "reference_schedule": "continuous_cosine_150_no_stage_restarts",
            "test_comparison_scope": "exploratory; test never selects checkpoints or changes queue"}


def configurations_for(plan):
    output = Path(plan["output"])
    reference = resolve_reference_source(plan["reference_source"]) if plan["reference_source"] else output / DENSE
    configs = {job: config_for(job) for job in [DENSE, *plan["jobs"]]}
    for job, cfg in configs.items():
        cfg.device = plan["device"]
        cfg.dataloaders.path_to_data = plan["data"]
        if job != DENSE:
            cfg.one_shot_pruning.initial_checkpoint = str(reference / "initializer.pt")
            cfg.one_shot_pruning.reference_history = str(reference / "global_history.csv")
        validate_model(cfg)
    return configs


def resolve_reference_source(path):
    path = Path(path).resolve()
    return path if (path / "one_shot_state.json").is_file() else path / DENSE


def validate_reference(source, config):
    from net_complexity.training.one_shot_pruning import validate_dense_reference
    return validate_dense_reference(resolve_reference_source(source), config)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(plan):
    paths = [path for directory in ("src", "scripts", "configs", "tests")
             for path in sorted((ROOT / directory).rglob("*"))
             if path.is_file() and path.suffix in (".py", ".yaml")]
    return {"protocol": PROTOCOL, "files_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
            "python": sys.version, "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(plan["device"])),
            "training_budget": {key: plan[key] for key in (
                "search_epochs_per_job", "scratch_epochs_per_job", "new_training_epochs_per_job",
                "dense_reference_epochs", "new_dense_reference_epochs", "total_new_training_epochs_allocated")},
            "test_used_for_selection": False, "test_comparisons": "exploratory"}


def write_test_summary(output, status, records):
    if not records:
        return
    evaluation = output / "test_evaluation"
    evaluation.mkdir(exist_ok=True)
    planned = [DENSE, *status["planned_jobs"]]
    write_json(evaluation / "test_summary.json", {
        "status": status["status"], "protocol": PROTOCOL, "test_evaluated": True,
        "comparison_scope": "exploratory", "training_performed": False,
        "bn_recalibration": False, "test_based_selection": False,
        "planned_jobs": planned, "evaluated_jobs": status["evaluated_jobs"],
        "pending_jobs": [j for j in planned if j not in status["evaluated_jobs"]],
        "all_completed_models_test_evaluated": all(j in status["evaluated_jobs"]
                                                   for j in status["completed_jobs"]),
        "total_new_training_epochs_allocated": status["total_new_training_epochs_allocated"],
        "error": status.get("error"), "runs": records,
    })
    from evaluate_one_shot_pruning import write_comparison
    write_comparison(evaluation, records)


def evaluate_completed_job(plan, output, job, reference_source, status, records, deadline):
    destination = output / "test_evaluation" / job
    require(not destination.exists(), f"Refusing to overwrite test evaluation: {destination}")
    require(job not in status["evaluated_jobs"], "Repeated test evaluation is forbidden")
    status["status"] = f"evaluating_frozen_test_{job}"
    write_json(output / "one_shot_queue_status.json", status)
    run_child([sys.executable, str(ROOT / "scripts/evaluate_one_shot_pruning.py"),
               "--run-dir", str(output), "--data", plan["data"], "--output", str(destination),
               "--device", plan["device"], "--reference-source", str(reference_source),
               "--jobs", job], output / f"{job}_test_evaluation.log", deadline)
    report = json.loads((destination / "test_summary.json").read_text())
    require(report["status"] == "completed" and report["test_evaluated"] is True
            and [row["job"] for row in report["runs"]] == [job], "Frozen test did not complete correctly")
    record = report["runs"][0]
    records.append({**record, "evaluation_report": str(Path(job) / "test_summary.json"),
                    "predictions": str(Path(job) / record["predictions"])})
    status["evaluated_jobs"].append(job)
    write_json(output / "one_shot_queue_status.json", status)
    write_test_summary(output, status, records)


def run_queue(plan):
    output = Path(plan["output"])
    require(not output.exists(), f"Refusing to overwrite existing output: {output}")
    configs = configurations_for(plan)
    reference_source = (resolve_reference_source(plan["reference_source"])
                        if plan["reference_source"] else output / DENSE)
    for job in plan["jobs"]:
        configs[job].one_shot_pruning.initial_checkpoint = str(reference_source / "initializer.pt")
        configs[job].one_shot_pruning.reference_history = str(reference_source / "global_history.csv")
    reference = validate_reference(reference_source, configs[plan["jobs"][0]]) if plan["reference_source"] else None
    output.mkdir(parents=True, exist_ok=False)
    print(f"Results: {output}", flush=True)
    print(f"Budget: {len(plan['jobs'])} x (150 search + 150 fresh scratch) + "
          f"{plan['new_dense_reference_epochs']} NEW dense-reference epochs = "
          f"{plan['total_new_training_epochs_allocated']} new epochs. "
          f"Reference: {'reused' if reference else 'new continuous 150-epoch training'}. No trained-weight reuse.", flush=True)
    deadline = time.monotonic() + plan["hours"] * 3600 if plan["hours"] is not None else float("inf")
    status = {"protocol": PROTOCOL, "status": "preflight", "planned_jobs": plan["jobs"],
              "completed_jobs": [], "incomplete_jobs": [], "evaluated_jobs": [],
              "reused_jobs": [DENSE] if reference else [],
              "total_new_training_epochs_allocated": plan["total_new_training_epochs_allocated"],
              "search_epochs_per_job": 150, "scratch_epochs_per_job": 150,
              "dense_reference_epochs": 150, "new_dense_reference_epochs": plan["new_dense_reference_epochs"]}
    records = []
    previous_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    active_job = None
    try:
        OmegaConf.save(OmegaConf.create(plan), output / "launcher_config.yaml", resolve=True)
        for job, cfg in configs.items():
            OmegaConf.save(cfg, output / f"{job}_resolved.yaml", resolve=True)
        write_json(output / "provenance.json", provenance(plan))
        write_json(output / "one_shot_queue_status.json", status)
        require(shutil.disk_usage(output).free >= 20 * 1024 ** 3, "Need at least 20 GiB free for checkpoints")
        run_child([sys.executable, "-m", "pytest", "-q", *[str(ROOT / p) for p in PREFLIGHT_TESTS]],
                  output / "preflight_tests.log", deadline)
        run_child([sys.executable, str(ROOT / "scripts/smoke_one_shot_pruning.py"),
                   "--output", str(output / "gpu_smoke"), "--device", plan["device"]],
                  output / "preflight_gpu_smoke.log", deadline)
        if reference is None:
            active_job = DENSE
            status["status"] = f"running_{DENSE}"
            write_json(output / "one_shot_queue_status.json", status)
            run_child([sys.executable, str(Path(__file__).resolve()), "--config-name", plan["config_name"],
                       "--job", DENSE, "--job-config", str(output / f"{DENSE}_resolved.yaml"),
                       "--output", str(reference_source), "--device", plan["device"]],
                      output / f"{DENSE}.log", deadline)
            reference = validate_reference(reference_source, configs[plan["jobs"][0]])
            active_job = None
        else:
            write_json(output / DENSE / "one_shot_state.json", {**reference, "reused_from": str(reference_source)})
            shutil.copyfile(reference_source / "resolved_config.yaml", output / DENSE / "resolved_config.yaml")
        status["completed_jobs"].append(DENSE)
        write_json(output / "reference_provenance.json", {
            "source": str(reference_source), "trained_weights_used_for_search": False,
            "reference_schedule": plan["reference_schedule"],
            "sha256": {name: file_hash(reference_source / name)
                       for name in ("initializer.pt", "global_history.csv", "one_shot_state.json")},
        })
        evaluate_completed_job(plan, output, DENSE, reference_source, status, records, deadline)
        for job in plan["jobs"]:
            active_job = job
            status["status"] = f"running_{job}"
            write_json(output / "one_shot_queue_status.json", status)
            run_child([sys.executable, str(Path(__file__).resolve()), "--config-name", plan["config_name"],
                       "--job", job, "--job-config", str(output / f"{job}_resolved.yaml"),
                       "--output", str(output / job), "--device", plan["device"]], output / f"{job}.log", deadline)
            result = json.loads((output / job / "one_shot_state.json").read_text())
            require(result["status"] == "completed" and result["protocol"] == PROTOCOL
                    and result["kind"] == "one_shot_reinit"
                    and result["search_epochs_completed"] == result["retrain_epochs_completed"] == 150
                    and result["global_epochs_completed"] == result["total_epochs_allocated"] == 300,
                    f"{job}: incomplete or wrong protocol result")
            status["completed_jobs"].append(job)
            active_job = None
            evaluate_completed_job(plan, output, job, reference_source, status, records, deadline)
        status["status"] = "completed"
    except BaseException as exc:
        if active_job:
            status["incomplete_jobs"].append(active_job)
        status.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        status["all_completed_models_test_evaluated"] = bool(status["completed_jobs"]) and all(
            job in status["evaluated_jobs"] for job in status["completed_jobs"])
        status["all_planned_models_test_evaluated"] = all(job in status["evaluated_jobs"]
                                                         for job in [DENSE, *plan["jobs"]])
        write_json(output / "one_shot_queue_status.json", status)
        write_test_summary(output, status, records)
        print(f"Status: {status['status']}; logs and checkpoints: {output}", flush=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pruning_one_shot")
    parser.add_argument("--cfg", choices=["job"], help="Print full plan/configs without GPU checks or writes")
    parser.add_argument("--reference-source", type=Path, help="Completed matching continuous-150 one-shot dense (not old J1)")
    parser.add_argument("--output", type=Path, help="New path; default outputs/runs/<timestamp>_<name>")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", help="Explicit CUDA device; default cuda:0")
    parser.add_argument("--hours", type=float, help="Optional hard wall-clock cap; default unlimited")
    parser.add_argument("--job", choices=[DENSE, *JOBS], help=argparse.SUPPRESS)
    parser.add_argument("--job-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        plan = resolve_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.cfg:
        config = OmegaConf.create({**plan, "resolved_jobs": {
            job: OmegaConf.to_container(cfg, resolve=True) for job, cfg in configurations_for(plan).items()}})
        print(OmegaConf.to_yaml(config), end="")
        return
    if not torch.cuda.is_available() or torch.device(plan["device"]).index >= torch.cuda.device_count():
        parser.error("Requested CUDA unavailable; refusing CPU fallback")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.job:
        require(args.job_config is not None, "Internal job requires frozen --job-config")
        cfg = OmegaConf.load(args.job_config)
        validate_model(cfg)
        require(cfg.mlflow.run_name == args.job and cfg.device == plan["device"], "Frozen job identity/device differs")
        from net_complexity.training.one_shot_pruning import run_dense_reference, run_one_shot_pruning
        runner = run_dense_reference if args.job == DENSE else run_one_shot_pruning
        runner(cfg, Path(plan["output"]))
    else:
        run_queue(plan)


if __name__ == "__main__":
    main()
