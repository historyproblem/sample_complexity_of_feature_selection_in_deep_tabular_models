"""Sequential, bounded pilot launcher. Run with the server repository .venv/bin/python."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pruning_pilot_common import ROOT, config_for, make_initializer

import torch
from omegaconf import OmegaConf

from net_complexity.training.pruning_audit import run_fixed_pruning_pilot, validate_config
from net_complexity.training.pruning_measurement import write_json

JOBS = ["J1_dense_control", "J2_output_fixed", "J3_internal_fixed"]
DAYTIME_JOBS = ["D1_dense_control", "D2_internal_fixed"]
PROFILE_JOBS = {"nightly": JOBS, "daytime": DAYTIME_JOBS}
RANDOM_JOBS = {"nightly": "J4_internal_random", "daytime": "D3_internal_random"}
PROFILE_HOURS = {"nightly": 11.75, "daytime": 2.0}


def dense_sanity_failed(profile, result):
    # A 25-epoch diagnostic must not inherit a 150-epoch accuracy requirement.
    return profile == "nightly" and result["validation"]["accuracy"] < 0.92


def write_comparison(output, completed_jobs, profile):
    """Small validation-only handoff artifact; partial jobs never enter this table."""
    records = []
    dense = None
    for job in completed_jobs:
        result = json.loads((output / job / "pilot_state.json").read_text())
        if result["status"] != "completed":
            raise ValueError("Cannot compare an incomplete job as a completed result.")
        if dense is None:
            dense = result
        cost = result["final_cost"]
        dense_cost = dense["final_cost"]
        records.append({
            "job": job,
            "epochs_consumed": result["global_epochs_completed"],
            "optimizer_steps": result["optimizer_steps_total"],
            "validation": result["validation"],
            "accuracy_delta_vs_dense_pp": 100 * (result["validation"]["accuracy"] - dense["validation"]["accuracy"]),
            "physical_parameters": cost["physical_total_parameters"],
            "parameter_reduction_vs_dense": 1 - cost["physical_total_parameters"] / dense_cost["physical_total_parameters"],
            "conv_linear_macs_per_image": cost["conv_linear_macs_per_image"],
            "mac_reduction_vs_dense": 1 - cost["conv_linear_macs_per_image"] / dense_cost["conv_linear_macs_per_image"],
            "parameter_target_met": result["parameter_target_met"],
            "latency": cost.get("latency"),
            "decisions": [{
                "cycle": d["cycle"], "status": d["status"], "budget": d["budget"],
                "immediate_accuracy_drop_pp": 100 * (d["old_committed_valid"]["accuracy"] - d["candidate_committed_valid"]["accuracy"]),
                "old_committed_valid": d["old_committed_valid"],
                "candidate_committed_valid": d["candidate_committed_valid"],
                "recovered_valid": d.get("recovered_valid"),
                "decision_checkpoint_epoch": d["decision_checkpoint_epoch"],
            } for d in result["decisions"]],
        })
    report = {
        "profile": profile, "seeds_per_job": 1, "test_evaluated": False,
        "interpretation": ("Early diagnostic only; not converged accuracy or proof of learned ranking."
                           if profile == "daytime" else "Single-seed pilot; not a significance claim."),
        "runs": records,
    }
    write_json(output / "comparison.json", report)
    lines = [
        f"# Pruning comparison — {profile}",
        "", report["interpretation"], "",
        "| Job | Epochs | Valid accuracy | Δ vs dense (pp) | Params | Conv/Linear GMAC | Target met |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in records:
        lines.append(f"| {row['job']} | {row['epochs_consumed']} | "
                     f"{row['validation']['accuracy']:.2%} | {row['accuracy_delta_vs_dense_pp']:+.2f} | "
                     f"{row['physical_parameters']:,} | {row['conv_linear_macs_per_image'] / 1e9:.3f} | "
                     f"{row['parameter_target_met']} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


def run_child(command, log_path, deadline):
    """Bound the entire process group; allow checkpoint flush before SIGKILL."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Overall wall-clock allowance exhausted.")
    with log_path.open("w") as log:
        print("Running:", " ".join(map(str, command)), flush=True)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            returncode = process.wait(timeout=remaining)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # TERM only the trainer first: killing DataLoader workers alongside
            # it can raise a worker error before a cooperative checkpoint is saved.
            os.kill(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError(f"Stopped incomplete process. Inspect {log_path}")
    if returncode:
        raise RuntimeError(f"Child failed (exit {returncode}); inspect {log_path}")


def provenance():
    hashes = {}
    for directory in ("src", "configs", "scripts", "tests"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".sh"}:
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"files_sha256": hashes, "python": sys.version, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "base_commit": "22c5866681680129eda04eeabefa8335e591da0e",
            "protocol": "fixed_lambda_pilot_v1_not_full_adaptive_contract"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="NEW output directory (never reused)")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--profile", choices=list(PROFILE_JOBS), default="nightly")
    parser.add_argument("--hours", type=float, help="Overall cap; default: nightly 11.75h, daytime 2h")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--with-random-control", action="store_true")
    parser.add_argument("--job", choices=JOBS + DAYTIME_JOBS + list(RANDOM_JOBS.values()), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.hours = PROFILE_HOURS[args.profile] if args.hours is None else args.hours
    jobs = list(PROFILE_JOBS[args.profile])
    if args.with_random_control:
        jobs.append(RANDOM_JOBS[args.profile])
    if sys.version_info < (3, 10):
        parser.error("Server runtime requires Python >=3.10; local compatibility smoke is separate.")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable. Refusing to launch overnight training on CPU.")
    if not 0 < args.hours <= 24:
        parser.error("--hours must be in (0, 24].")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output, data = args.output.resolve(), args.data.resolve()
    if args.job:
        if args.job not in PROFILE_JOBS[args.profile] + [RANDOM_JOBS[args.profile]]:
            parser.error("--job does not belong to the requested profile.")
        cfg = config_for(args.job)
        cfg.dataloaders.path_to_data = str(data)
        run_fixed_pruning_pilot(cfg, output)
        return

    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < 30 * 1024 ** 3:
        parser.error("Need at least 30 GiB free for checkpoints and preflight artifacts.")
    deadline = time.monotonic() + args.hours * 3600
    # Also cooperatively stop children when the launcher itself receives TERM.
    previous_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    status = {"status": "preflight", "profile": args.profile, "completed_jobs": [], "skipped_jobs": [],
              "wall_budget_hours": args.hours, "test_evaluated": False}
    try:
        write_json(output / "provenance.json", provenance())
        os.environ["AUDIT_INIT_CHECKPOINT"] = str(output / "shared_random_seed42.pt")
        configurations = {job: config_for(job) for job in jobs}
        for job, cfg in configurations.items():
            validate_config(cfg)
            cfg.dataloaders.path_to_data = str(data)
            OmegaConf.save(cfg, output / f"{job}_resolved.yaml", resolve=True)
        tests = [
            "tests/test_pruning_audit.py", "tests/test_pruning_pilot_launcher.py", "tests/test_pruned_bottleneck.py",
            "tests/test_cyclic_channel_weight_handoff.py", "tests/test_best_checkpoint_evaluation.py",
            "tests/test_optimizer_groups.py", "tests/test_dataloaders.py",
        ]
        run_child([sys.executable, "-m", "pytest", "-q", *[str(ROOT / t) for t in tests]],
                  output / "preflight_tests.log", deadline)
        run_child([sys.executable, str(ROOT / "scripts/smoke_pruning_pilot.py"),
                   "--output", str(output / "gpu_smoke"), "--device", "cuda:0",
                   "--profile", args.profile],
                  output / "preflight_gpu_smoke.log", deadline)
        if args.preflight_only:
            status["status"] = "preflight_passed_no_nightly_training"
            return
        make_initializer(configurations[jobs[0]], output / "shared_random_seed42.pt")
        prior_seconds = None
        for job in jobs:
            remaining = deadline - time.monotonic()
            if prior_seconds is not None and remaining < prior_seconds * 1.20:
                status["skipped_jobs"].extend(jobs[jobs.index(job):])
                status["status"] = "wall_budget_insufficient_for_remaining_jobs"
                break
            status["status"] = f"running_{job}"
            write_json(output / "nightly_status.json", status)
            started = time.monotonic()
            run_child([sys.executable, str(Path(__file__).resolve()), "--job", job,
                       "--profile", args.profile, "--output", str(output / job), "--data", str(data)],
                      output / f"{job}.log", deadline)
            prior_seconds = time.monotonic() - started
            result = json.loads((output / job / "pilot_state.json").read_text())
            expected_epochs = validate_config(configurations[job])
            if result["status"] != "completed" or result["global_epochs_completed"] != expected_epochs:
                raise RuntimeError(f"Job did not complete exactly {expected_epochs} epochs.")
            if status["completed_jobs"]:
                first = json.loads((output / jobs[0] / "pilot_state.json").read_text())
                for key in ("common_init_hash", "split_indices_hash", "optimizer_steps_total"):
                    if result[key] != first[key]:
                        raise RuntimeError(f"Matched-control provenance mismatch: {key}")
            status["completed_jobs"].append(job)
            write_comparison(output, status["completed_jobs"], args.profile)
            print(f"{job}: valid={result['validation']['accuracy']:.2%}, "
                  f"params={result['final_cost']['physical_total_parameters']:,}, "
                  f"wall={prior_seconds / 60:.1f} min; see comparison.md", flush=True)
            if job == jobs[0] and dense_sanity_failed(args.profile, result):
                status["status"] = "dense_sanity_gate_failed"
                raise RuntimeError("Dense validation <92%; not spending remaining night on pruning.")
        else:
            status["status"] = "completed"
    except BaseException as exc:
        status.update(status="failed_or_interrupted", error=str(exc))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        write_json(output / "nightly_status.json", status)
        print(f"Status: {status['status']}; logs and checkpoints: {output}", flush=True)


if __name__ == "__main__":
    main()
