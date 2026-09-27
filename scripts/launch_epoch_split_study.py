"""Run the fresh-reference four-way 150-epoch pruning split study end to end.

The study trains one dense validation reference, reuses its immutable seed-42
initializer for every search, runs each configured search/recovery pair, selects
the canonical split on validation only, and evaluates only that frozen winner on
the official test set. Existing incomplete outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ONE_SHOT = ROOT / "scripts/launch_one_shot_pruning.py"
EVALUATOR = ROOT / "scripts/evaluate_one_shot_pruning_test.py"
SPLITS = ((30, 120), (45, 105), (60, 90), (75, 75))
RECOVERY_BRANCH = "fresh_optimizer_fresh_scheduler__repeat_1"
SEARCH_PROTOCOL = "pruning_v3_one_shot_epoch_split_150"
RECOVERY_PROTOCOL = "pruning_v3_quality_recovery_epoch_split_150"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text())


def _run(command, *, capture=False):
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if result.returncode:
        detail = f"\n{result.stdout.rstrip()}" if capture and result.stdout else ""
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(map(str, command))}{detail}"
        )
    return result


def _config_name(search_epochs, recovery_epochs, phase):
    return (
        f"experiment/pruning_v3/epoch_split_"
        f"{search_epochs}_{recovery_epochs}_{phase}"
    )


def validate_study_configs():
    """Resolve and strictly verify every checked-in study configuration."""
    sys.path.insert(0, str(SRC))
    from net_complexity.training import one_shot_pruning_config as schema

    resolved = []
    for search_epochs, recovery_epochs in SPLITS:
        search = schema.compose_config(
            _config_name(search_epochs, recovery_epochs, "search")
        )
        recovery = schema.compose_config(
            _config_name(search_epochs, recovery_epochs, "recovery")
        )
        _require(
            schema.validate_config(search) == schema.validate_config(recovery) == 150,
            f"{search_epochs}/{recovery_epochs}: configuration is not a strict 150-epoch pair",
        )
        _require(search.one_shot.protocol == SEARCH_PROTOCOL,
                 f"{search_epochs}/{recovery_epochs}: wrong search protocol")
        _require(recovery.one_shot.protocol == RECOVERY_PROTOCOL,
                 f"{search_epochs}/{recovery_epochs}: wrong recovery protocol")
        _require(recovery.one_shot.reuse_search_required is True,
                 f"{search_epochs}/{recovery_epochs}: recovery does not require its search")
        for config in (search, recovery):
            _require(config.training_arguments.adaptive_lambda.enabled is True,
                     f"{search_epochs}/{recovery_epochs}: adaptive lambda is disabled")
            _require(float(config.optimizer.lr) == 0.001,
                     f"{search_epochs}/{recovery_epochs}: optimizer LR changed")
            _require(float(config.optimizer.weight_decay) == 0.0005,
                     f"{search_epochs}/{recovery_epochs}: weight decay changed")
            _require(dict(config.model.criterion) == {
                "_target_": "torch.nn.CrossEntropyLoss"
            }, f"{search_epochs}/{recovery_epochs}: loss is not plain cross entropy")
            _require(float(config.scheduler.eta_min) == 0.0,
                     f"{search_epochs}/{recovery_epochs}: cosine eta_min changed")
        _require(int(search.accuracy_guided.guard.train_bn_calibration_batches) == 0,
                 f"{search_epochs}/{recovery_epochs}: search unexpectedly calibrates BN")
        _require(int(recovery.accuracy_guided.guard.train_bn_calibration_batches) == 200,
                 f"{search_epochs}/{recovery_epochs}: recovery is not BN200")
        plan = schema.resolved_branch_plan(recovery)
        _require(len(plan) == 1 and plan[0] == {
            "id": RECOVERY_BRANCH,
            "method": "fresh_optimizer_fresh_scheduler",
            "repeat": "repeat_1",
            "training_seed": 42,
            "model_state": "selected_surviving_state",
            "optimizer_state": "fresh",
            "scheduler_state": schema.FRESH_SCHEDULER,
        }, f"{search_epochs}/{recovery_epochs}: recovery handoff policy changed")
        resolved.append({
            "split": f"{search_epochs}/{recovery_epochs}",
            "search_config": _config_name(search_epochs, recovery_epochs, "search"),
            "recovery_config": _config_name(search_epochs, recovery_epochs, "recovery"),
        })
    return resolved


def _environment_preflight(min_free_gib):
    expected_environment = (ROOT / ".venv").resolve()
    _require(Path(sys.prefix).resolve() == expected_environment,
             f"Use the repository environment: {ROOT / '.venv/bin/python'}")
    _run([sys.executable, "-m", "pip", "check"], capture=True)
    import torch
    import torchvision  # noqa: F401 - explicit runtime dependency preflight
    import hydra  # noqa: F401 - explicit runtime dependency preflight
    import omegaconf  # noqa: F401 - explicit runtime dependency preflight
    _require(torch.cuda.is_available(), "CUDA is unavailable; refusing a silent CPU run")
    _require(torch.cuda.device_count() >= 1, "No CUDA device is visible")
    free_gib = shutil.disk_usage(ROOT).free / 2 ** 30
    _require(free_gib >= min_free_gib,
             f"Only {free_gib:.1f} GiB free; require at least {min_free_gib:.1f} GiB")
    print(
        f"[epoch-split] preflight ready: torch={torch.__version__}; "
        f"gpu={torch.cuda.get_device_name(0)}; free={free_gib:.1f} GiB",
        flush=True,
    )
    return {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
            "free_gib": free_gib, "min_free_gib": min_free_gib}


def _reference_result(reference):
    job = Path(reference) / "J1_dense_control"
    state = _read_json(job / "pilot_state.json")
    required = (
        Path(reference) / "shared_random_seed42.pt",
        job / "global_history.csv",
        job / "resolved_config.yaml",
        job / "selected_checkpoint.pt",
        job / "deployment.pt",
    )
    for path in required:
        _require(path.is_file(), f"Completed dense reference is missing {path}")
    _require(state.get("status") == "completed", "Dense reference is not completed")
    _require(state.get("reference_protocol") == "one_shot_dense_reference_v1",
             "Dense reference protocol differs")
    _require(state.get("reference_training_epochs_actually_executed") == 150,
             "Dense reference did not execute exactly 150 epochs")
    _require(state.get("global_epochs_completed") == 150,
             "Dense reference global epoch ledger is incomplete")
    _require(state.get("test_evaluated") is False,
             "Dense reference unexpectedly accessed official test data")
    return {
        "root": str(Path(reference).resolve()),
        "status": state["status"],
        "selected_epoch": int(state["selected_epoch"]),
        "validation": state["validation"],
        "common_init_hash": state["common_init_hash"],
        "initializer_file_hash": state["initializer_file_hash"],
    }


def _search_result(run_dir, search_epochs, recovery_epochs):
    run_dir = Path(run_dir)
    state = _read_json(run_dir / "one_shot_state.json")
    selection = _read_json(run_dir / "selection.json")
    diagnostics = _read_json(run_dir / "export_only/diagnostics.json")
    for path in (run_dir / "selected_checkpoint.pt", run_dir / "export_only/deployment.pt",
                 run_dir / "resolved_config.yaml"):
        _require(path.is_file(), f"Completed search is missing {path}")
    _require(state.get("status") == "search_only_completed",
             f"Search is not completed: {run_dir} status={state.get('status')}")
    _require(state.get("search_only") is True and state.get("branches") == {},
             f"Search-only artifact contains unexpected recovery branches: {run_dir}")
    _require(state.get("protocol") == SEARCH_PROTOCOL,
             f"Search protocol differs: {run_dir}")
    _require(state.get("per_branch_total_allocated") == 150,
             f"Search pair budget differs: {run_dir}")
    ledger = state.get("shared_search_ledger", {})
    _require(ledger.get("global_training_epoch") == search_epochs
             and ledger.get("search_epochs_consumed") == search_epochs,
             f"Search ledger is incomplete: {run_dir}")
    _require(selection.get("reference_epoch") == search_epochs,
             f"Search selection used the wrong validation reference epoch: {run_dir}")
    _require(diagnostics.get("status") == "measured"
             and diagnostics.get("bn_calibration_batches") == 0,
             f"Dependency-safe search export diagnostics are incomplete: {run_dir}")
    return {
        "split": f"{search_epochs}/{recovery_epochs}",
        "run_dir": str(run_dir.resolve()),
        "status": state["status"],
        "selected_epoch": int(selection["selected_epoch"]),
        "selection_policy": selection["policy"],
        "physical_parameters": int(diagnostics["physical_cost"]["physical_total_parameters"]),
        "physical_validation": diagnostics["physical_validation"],
    }


def _recovery_result(run_dir, search_dir, search_epochs, recovery_epochs):
    run_dir, search_dir = Path(run_dir), Path(search_dir).resolve()
    state = _read_json(run_dir / "one_shot_state.json")
    branch = _read_json(run_dir / RECOVERY_BRANCH / "branch_state.json")
    for path in (run_dir / "selected_checkpoint.pt", run_dir / "selection.json",
                 run_dir / "export_only/deployment.pt",
                 run_dir / RECOVERY_BRANCH / "deployment.pt",
                 run_dir / "resolved_config.yaml"):
        _require(path.is_file(), f"Completed recovery is missing {path}")
    _require(state.get("status") == "completed",
             f"Recovery is not completed: {run_dir} status={state.get('status')}")
    _require(state.get("protocol") == RECOVERY_PROTOCOL,
             f"Recovery protocol differs: {run_dir}")
    _require(state.get("per_branch_total_allocated") == 150,
             f"Recovery pair budget differs: {run_dir}")
    _require(Path(state.get("reused_search", {}).get("source", "")).resolve() == search_dir,
             f"Recovery reused the wrong search: {run_dir}")
    _require(set(state.get("branches", {})) == {RECOVERY_BRANCH},
             f"Recovery branch set differs: {run_dir}")
    _require(branch.get("status") in {"completed", "infeasible"},
             f"Recovery deployment is not finalized: {run_dir}")
    _require(branch.get("optimizer_state_initialization") == "fresh",
             f"Recovery did not use fresh AdamW: {run_dir}")
    _require(branch.get("scheduler_state_initialization") == "fresh_final_stage_cosine",
             f"Recovery did not use a fresh cosine scheduler: {run_dir}")
    _require(branch.get("final_training_epochs_executed") == recovery_epochs,
             f"Recovery epoch ledger differs: {run_dir}")
    ledger = branch.get("ledger", {})
    _require(ledger.get("global_training_epoch") == 150
             and ledger.get("search_epochs_consumed") == search_epochs,
             f"Combined 150-epoch ledger is incomplete: {run_dir}")
    calibration = branch.get("bn_calibration", {})
    _require(calibration.get("requested_batches") == 200
             and calibration.get("batches") == 200
             and calibration.get("trainable_weights_unchanged") is True,
             f"BN200 recalibration evidence differs: {run_dir}")
    validation = branch.get("validation", {})
    _require(validation.get("example_count") == 5000,
             f"Recovery validation did not use the full 5,000-example split: {run_dir}")
    _require(state.get("test_evaluated") is False,
             f"Recovery accessed test before validation selection: {run_dir}")
    return {
        "split": f"{search_epochs}/{recovery_epochs}",
        "search_epochs": search_epochs,
        "recovery_epochs": recovery_epochs,
        "run_dir": str(run_dir.resolve()),
        "status": branch["status"],
        "quality_feasible": bool(branch["quality_feasible"]),
        "validation_accuracy": float(validation["accuracy"]),
        "validation_ce_loss": float(validation["ce_loss"]),
        "validation_correct_count": int(validation["correct_count"]),
        "validation_example_count": int(validation["example_count"]),
        "physical_parameters": int(branch["final_cost"]["physical_total_parameters"]),
        "selected_final_epoch": int(branch["selected_final_epoch"]),
    }


def select_validation_winner(results):
    feasible = [row for row in results if row["quality_feasible"]]
    _require(feasible, "No recovery split met its validation quality requirement")
    winner = max(
        feasible,
        key=lambda row: (
            row["validation_accuracy"],
            -row["validation_ce_loss"],
            -row["physical_parameters"],
            -row["search_epochs"],
        ),
    )
    return winner, (
        "highest validation accuracy among quality-feasible frozen deployments; "
        "ties: lower validation CE, fewer physical parameters, shorter search"
    )


def _write_validation_csv(path, rows):
    path = Path(path)
    fieldnames = (
        "split", "search_epochs", "recovery_epochs", "status", "quality_feasible",
        "validation_accuracy", "validation_ce_loss", "validation_correct_count",
        "validation_example_count", "physical_parameters", "selected_final_epoch", "run_dir",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)
    temporary.replace(path)


def _prune_pair_checkpoints(search_dir, recovery_dir):
    # Callers must validate both completed roots immediately before this helper.
    removed_files = 0
    removed_bytes = 0
    for root in (Path(search_dir), Path(recovery_dir)):
        for path in root.rglob("checkpoints/*.pt"):
            _require(path.is_file() and path.parent.name == "checkpoints",
                     f"Refusing unexpected cleanup target: {path}")
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1
    print(
        f"[epoch-split] removed {removed_files} completed nested checkpoints "
        f"({removed_bytes / 2 ** 30:.1f} GiB); root selected/deployment artifacts retained",
        flush=True,
    )
    return {"files": removed_files, "bytes": removed_bytes}


def _test_result(test_dir, winner_dir):
    summary = _read_json(Path(test_dir) / "test_summary.json")
    _require(summary.get("status") == "completed" and summary.get("test_evaluated") is True,
             "Frozen winner test evaluation did not complete")
    _require(Path(summary.get("source_run", "")).resolve() == Path(winner_dir).resolve(),
             "Test summary points to a different recovery run")
    _require(summary.get("training_performed") is False
             and summary.get("bn_recalibration") is False
             and summary.get("test_based_selection") is False,
             "Final test was not a frozen inference-only evaluation")
    return {
        "output": str(Path(test_dir).resolve()),
        "status": summary["status"],
        "comparison_scope": summary["comparison_scope"],
        "runs": summary["runs"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-output", type=Path,
        default=Path("outputs/runs/epoch_split_dense_reference_seed42_fresh"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument(
        "--state", type=Path,
        default=Path("outputs/runs/epoch_split_four_way_state.json"),
    )
    parser.add_argument(
        "--resume-completed", action="store_true",
        help="Verify and reuse only fully completed stages; partial stages still abort",
    )
    parser.add_argument(
        "--prune-completed-checkpoints", action="store_true",
        help="After each verified pair, remove only nested checkpoints/*.pt files",
    )
    parser.add_argument(
        "--min-free-gib", type=float,
        help="Initial disk requirement (default: 200 GiB, or 90 GiB with pruning)",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    reference = args.reference_output.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    state_path = args.state.expanduser().resolve()
    min_free_gib = float(args.min_free_gib if args.min_free_gib is not None else
                         (90.0 if args.prune_completed_checkpoints else 200.0))
    _require(min_free_gib > 0, "--min-free-gib must be positive")

    configs = validate_study_configs()
    preflight = _environment_preflight(min_free_gib)
    if args.preflight_only:
        print(json.dumps({"status": "ready", "configs": configs, "environment": preflight},
                         indent=2, ensure_ascii=False), flush=True)
        return {"status": "ready", "configs": configs, "environment": preflight}

    if state_path.exists() and not args.resume_completed:
        raise FileExistsError(
            f"Refusing existing study state: {state_path}. "
            "Use --resume-completed only to verify and skip completed stages."
        )
    state = {
        "protocol": "pruning_v3_epoch_split_four_way_fresh_reference_v1",
        "status": "running",
        "started_unix": time(),
        "reference": None,
        "splits": {},
        "selection": None,
        "test": None,
        "configs": configs,
        "environment": preflight,
        "test_comparison_scope": "exploratory; prior test results informed this study",
    }
    _write_json(state_path, state)

    def run_or_reuse(path, validator, command):
        path = Path(path)
        if path.exists():
            if not args.resume_completed:
                raise FileExistsError(f"Refusing existing stage output: {path}")
            result = validator()
            print(f"[epoch-split] verified completed stage: {path}", flush=True)
            return result
        _run(command)
        return validator()

    try:
        state["reference"] = run_or_reuse(
            reference,
            lambda: _reference_result(reference),
            [
                sys.executable, ONE_SHOT,
                "--config-name", _config_name(60, 90, "search"),
                "--prepare-reference", reference,
            ],
        )
        _write_json(state_path, state)

        recoveries = []
        for index, (search_epochs, recovery_epochs) in enumerate(SPLITS, 1):
            stem = f"epoch_split_{search_epochs}_{recovery_epochs}"
            search_dir = output_root / f"{stem}_depsafe_search"
            recovery_dir = output_root / f"{stem}_depsafe_bn200_fixed"
            print(
                f"[epoch-split] pair {index}/{len(SPLITS)}: "
                f"{search_epochs}/{recovery_epochs}",
                flush=True,
            )
            search_result = run_or_reuse(
                search_dir,
                lambda s=search_dir, a=search_epochs, b=recovery_epochs:
                    _search_result(s, a, b),
                [
                    sys.executable, ONE_SHOT,
                    "--config-name", _config_name(search_epochs, recovery_epochs, "search"),
                    "--dense-source", reference,
                    "--output", search_dir,
                    "--search-only",
                ],
            )
            recovery_result = run_or_reuse(
                recovery_dir,
                lambda r=recovery_dir, s=search_dir, a=search_epochs, b=recovery_epochs:
                    _recovery_result(r, s, a, b),
                [
                    sys.executable, ONE_SHOT,
                    "--config-name", _config_name(search_epochs, recovery_epochs, "recovery"),
                    "--dense-source", reference,
                    "--reuse-search", search_dir,
                    "--output", recovery_dir,
                    "--skip-test",
                ],
            )
            recoveries.append(recovery_result)
            state["splits"][f"{search_epochs}/{recovery_epochs}"] = {
                "search": search_result,
                "recovery": recovery_result,
            }
            if args.prune_completed_checkpoints:
                state["splits"][f"{search_epochs}/{recovery_epochs}"]["checkpoint_cleanup"] = (
                    _prune_pair_checkpoints(search_dir, recovery_dir)
                )
            _write_json(state_path, state)

        winner, policy = select_validation_winner(recoveries)
        validation_csv = state_path.with_name("epoch_split_validation_summary.csv")
        _write_validation_csv(validation_csv, recoveries)
        state["selection"] = {
            "data": "validation_only",
            "policy": policy,
            "winner": winner,
            "summary_csv": str(validation_csv),
            "official_test_accessed_before_selection": False,
        }
        _write_json(state_path, state)
        print(
            f"[epoch-split] validation winner={winner['split']}; "
            f"accuracy={winner['validation_accuracy']:.4%}; "
            f"parameters={winner['physical_parameters']:,}",
            flush=True,
        )

        winner_dir = Path(winner["run_dir"])
        test_dir = winner_dir / "test_evaluation"
        if test_dir.exists():
            if not args.resume_completed:
                raise FileExistsError(f"Refusing existing test output: {test_dir}")
        else:
            _run([
                sys.executable, EVALUATOR,
                "--run-dir", winner_dir,
                "--output", test_dir,
            ])
        state["test"] = _test_result(test_dir, winner_dir)
        state["status"] = "completed"
        state["finished_unix"] = time()
        _write_json(state_path, state)
        print(f"[epoch-split] completed end to end: {state_path}", flush=True)
        return state
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["finished_unix"] = time()
        _write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
