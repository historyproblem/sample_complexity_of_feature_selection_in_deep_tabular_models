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
    parser.add_argument("--hours", type=float, default=11.75)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--with-random-control", action="store_true")
    parser.add_argument("--job", choices=JOBS + ["J4_internal_random"], help=argparse.SUPPRESS)
    args = parser.parse_args()
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
    status = {"status": "preflight", "completed_jobs": [], "skipped_jobs": [],
              "wall_budget_hours": args.hours, "test_evaluated": False}
    try:
        write_json(output / "provenance.json", provenance())
        os.environ["AUDIT_INIT_CHECKPOINT"] = str(output / "shared_random_seed42.pt")
        configurations = {job: config_for(job) for job in JOBS + ["J4_internal_random"]}
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
                   "--output", str(output / "gpu_smoke"), "--device", "cuda:0"],
                  output / "preflight_gpu_smoke.log", deadline)
        if args.preflight_only:
            status["status"] = "preflight_passed_no_nightly_training"
            return
        make_initializer(configurations[JOBS[0]], output / "shared_random_seed42.pt")
        jobs = JOBS + (["J4_internal_random"] if args.with_random_control else [])
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
                       "--output", str(output / job), "--data", str(data)],
                      output / f"{job}.log", deadline)
            prior_seconds = time.monotonic() - started
            result = json.loads((output / job / "pilot_state.json").read_text())
            if result["status"] != "completed" or result["global_epochs_completed"] != 150:
                raise RuntimeError("Job did not complete exactly 150 epochs.")
            if status["completed_jobs"]:
                first = json.loads((output / JOBS[0] / "pilot_state.json").read_text())
                for key in ("common_init_hash", "split_indices_hash", "optimizer_steps_total"):
                    if result[key] != first[key]:
                        raise RuntimeError(f"Matched-control provenance mismatch: {key}")
            status["completed_jobs"].append(job)
            if job == JOBS[0] and result["validation"]["accuracy"] < 0.92:
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
