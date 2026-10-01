"""Run a dense validation-controlled ResNet50/CIFAR-10 60/90 curve.

Every point uses one 60-epoch adaptive-lambda search followed by 90 epochs of
physical ungated recovery. Only the allowed validation-accuracy drop changes;
``soft_drop`` remains one half of ``hard_drop``. The lower keep floor is shared
by every point so the resulting accuracy/parameter curve has one standard
method configuration rather than target-specific structural settings.

The 2M entry is an explicit architectural-floor probe. This implementation
prunes only bottleneck-internal channels, and its exact one-channel-per-gated-
boundary floor is above 2M. Planning targets are labels only and never enter
checkpoint selection. Test inference starts only after the validation table is
written and locked.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from time import time

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import launch_epoch_split_study as epoch_split


ROOT = Path(__file__).resolve().parents[1]
ONE_SHOT = ROOT / "scripts/launch_one_shot_pruning.py"
EVALUATOR = ROOT / "scripts/evaluate_one_shot_pruning_test.py"
SEARCH_CONFIG = "experiment/pruning_v3/resnet50_cifar10_curve60_search"
RECOVERY_CONFIG = "experiment/pruning_v3/resnet50_cifar10_curve60_recovery"
SEARCH_EPOCHS = 60
RECOVERY_EPOCHS = 90
MIN_KEEP_RATIO = 0.001

# Empirical anchors from validation/search artifacts are 17.76M at 0.0005,
# 12.73M at 0.001, about 6.2M at 0.01, 5.1M at 0.015, and 4.15M at 0.03.
# Intermediate values densify the curve. The final two values deliberately
# probe the newly lowered structural floor; their target names are not quotas.
GRID = (
    {"id": "18m", "target_parameters": 18_000_000, "hard_drop": 0.0005},
    {"id": "15m", "target_parameters": 15_000_000, "hard_drop": 0.00075},
    {"id": "12m", "target_parameters": 12_000_000, "hard_drop": 0.001},
    {"id": "10m", "target_parameters": 10_000_000, "hard_drop": 0.0025},
    {"id": "8m", "target_parameters": 8_000_000, "hard_drop": 0.005},
    {"id": "6m", "target_parameters": 6_000_000, "hard_drop": 0.01},
    {"id": "5m", "target_parameters": 5_000_000, "hard_drop": 0.015},
    {"id": "4m", "target_parameters": 4_000_000, "hard_drop": 0.03},
    {"id": "3m", "target_parameters": 3_000_000, "hard_drop": 0.06},
    {"id": "2m_floor_probe", "target_parameters": 2_000_000, "hard_drop": 0.10},
)

DEFAULT_REFERENCE = Path("outputs/runs/epoch_split_dense_reference_seed42_fresh")
DEFAULT_OUTPUT = Path("outputs/runs/resnet50_cifar10_curve60_90_v2")
EXPECTED_HOURS_PER_PAIR = 3.0


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _profiles():
    return [
        {
            **point,
            "soft_drop": float(point["hard_drop"]) / 2.0,
            "search_epochs": SEARCH_EPOCHS,
            "recovery_epochs": RECOVERY_EPOCHS,
            "split": f"{SEARCH_EPOCHS}/{RECOVERY_EPOCHS}",
            "min_keep_ratio": MIN_KEEP_RATIO,
        }
        for point in GRID
    ]


def _threshold_overrides(row):
    return (
        f"training_arguments.adaptive_lambda.soft_drop={row['soft_drop']:.12g}",
        f"training_arguments.adaptive_lambda.hard_drop={row['hard_drop']:.12g}",
    )


def _launcher_command(config, reference, output, *, row, search_only=False,
                      reuse_search=None):
    command = [
        sys.executable,
        str(ONE_SHOT),
        "--config-name", config,
        "--dense-source", str(reference),
        "--output", str(output),
    ]
    for override in _threshold_overrides(row):
        command.extend(["--override", override])
    if search_only:
        command.append("--search-only")
    if reuse_search is not None:
        command.extend(["--reuse-search", str(reuse_search), "--skip-test"])
    return command


def _theoretical_parameter_floor(config):
    from hydra.utils import instantiate
    from net_complexity.models.pruning_budget import PhysicalBudget, gates

    model = instantiate(config.model)
    mask = {}
    for name, gate in gates(model).items():
        width = int(gate.initial_channels)
        keep = max(1, math.ceil(width * MIN_KEEP_RATIO))
        mask[name] = list(range(width - keep))
    return int(PhysicalBudget(model, mask).total())


def validate_curve_configs():
    """Resolve every point and reject non-threshold method drift."""
    sys.path.insert(0, str(ROOT / "src"))
    from net_complexity.training import one_shot_pruning_config as schema
    from net_complexity.training.engine import _resolve_adaptive_lambda_log_step_init
    from omegaconf import OmegaConf

    verified = []
    parameter_floor = None
    for row in _profiles():
        overrides = list(_threshold_overrides(row))
        search = schema.compose_config(SEARCH_CONFIG, overrides=overrides)
        recovery = schema.compose_config(RECOVERY_CONFIG, overrides=overrides)
        _require(
            schema.validate_config(search) == schema.validate_config(recovery) == 150,
            f"{row['id']}: not a strict 150-epoch pair",
        )
        for config in (search, recovery):
            adaptive = config.training_arguments.adaptive_lambda
            _require(adaptive.enabled is True and adaptive.control_mode == "accuracy_only",
                     f"{row['id']}: accuracy-only controller changed")
            _require(str(adaptive.log_step).lower() == "auto",
                     f"{row['id']}: lambda step is not automatic")
            _require(float(adaptive.soft_drop) == row["soft_drop"]
                     and float(adaptive.hard_drop) == row["hard_drop"],
                     f"{row['id']}: accuracy-drop override was not resolved")
            _require(float(adaptive.soft_drop) * 2.0 == float(adaptive.hard_drop),
                     f"{row['id']}: soft:hard relation is not 1:2")
            _require(float(adaptive.alpha_init) == 0.001
                     and float(adaptive.alpha_min) == 1e-8
                     and float(adaptive.alpha_max) == 80.0,
                     f"{row['id']}: lambda bounds changed")
            _require(int(adaptive.initial_search_warmup) == 10
                     and int(adaptive.update_every_search_epochs) == 1
                     and int(adaptive.gap_window) == 3
                     and int(adaptive.reentry_samples) == 3,
                     f"{row['id']}: controller cadence changed")
            _require(float(config.accuracy_guided.eligibility.min_keep_ratio)
                     == MIN_KEEP_RATIO,
                     f"{row['id']}: keep floor changed")
            _require(dict(config.model.criterion) == {
                "_target_": "torch.nn.CrossEntropyLoss"
            }, f"{row['id']}: loss is not plain CE")
            _require(float(config.optimizer.lr) == 0.001
                     and float(config.optimizer.weight_decay) == 0.0005,
                     f"{row['id']}: AdamW settings changed")
            _require(float(config.scheduler.eta_min) == 0.0,
                     f"{row['id']}: cosine eta_min changed")
        _require(int(search.accuracy_guided.guard.train_bn_calibration_batches) == 0,
                 f"{row['id']}: search unexpectedly recalibrates BN")
        _require(int(recovery.accuracy_guided.guard.train_bn_calibration_batches) == 200,
                 f"{row['id']}: recovery is not BN200")
        plan = schema.resolved_branch_plan(recovery)
        _require(len(plan) == 1 and plan[0]["optimizer_state"] == "fresh"
                 and plan[0]["scheduler_state"] == schema.FRESH_SCHEDULER,
                 f"{row['id']}: recovery handoff changed")
        adaptive = search.training_arguments.adaptive_lambda
        resolved_step = _resolve_adaptive_lambda_log_step_init(
            OmegaConf.create({"num_epochs": SEARCH_EPOCHS}),
            OmegaConf.create({"log_step_init": "auto"}),
            initial_lambda_coef=float(adaptive.alpha_init),
            warmup_epochs=int(adaptive.initial_search_warmup),
            update_every_epochs=int(adaptive.update_every_search_epochs),
        )
        if parameter_floor is None:
            parameter_floor = _theoretical_parameter_floor(search)
        verified.append({
            **row,
            "search_config": SEARCH_CONFIG,
            "recovery_config": RECOVERY_CONFIG,
            "resolved_log_step": resolved_step,
            "theoretical_parameter_floor": parameter_floor,
            "target_architecturally_reachable": (
                row["target_parameters"] >= parameter_floor
            ),
        })
    return verified


def _run_paths(output, row):
    return output / f"search_{row['id']}", output / f"recovery_{row['id']}"


def _validate_resolved_config(path, row):
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(path) / "resolved_config.yaml")
    adaptive = config.training_arguments.adaptive_lambda
    _require(float(adaptive.soft_drop) == row["soft_drop"]
             and float(adaptive.hard_drop) == row["hard_drop"],
             f"Completed run has different accuracy thresholds: {path}")
    _require(str(adaptive.log_step).lower() == "auto",
             f"Completed run did not use automatic lambda steps: {path}")
    _require(float(config.accuracy_guided.eligibility.min_keep_ratio)
             == MIN_KEEP_RATIO,
             f"Completed run has a different keep floor: {path}")


def _completed_pair_result(search_dir, recovery_dir, row):
    search = epoch_split._search_result(search_dir, SEARCH_EPOCHS, RECOVERY_EPOCHS)
    recovery = epoch_split._recovery_result(
        recovery_dir, search_dir, SEARCH_EPOCHS, RECOVERY_EPOCHS
    )
    _validate_resolved_config(search_dir, row)
    _validate_resolved_config(recovery_dir, row)
    actual = int(recovery["physical_parameters"])
    return {
        **row,
        "search": search,
        "recovery": recovery,
        "relative_target_error": (
            actual - row["target_parameters"]
        ) / row["target_parameters"],
    }


def _estimate_plan(output):
    jobs = []
    for row in _profiles():
        search_dir, recovery_dir = _run_paths(output, row)
        completed = search_dir.exists() and recovery_dir.exists()
        jobs.append({
            **row,
            "search_dir": str(search_dir),
            "recovery_dir": str(recovery_dir),
            "completed_hint": completed,
            "estimated_hours_if_fresh": EXPECTED_HOURS_PER_PAIR,
        })
    fresh = [job for job in jobs if not job["completed_hint"]]
    return {
        "jobs": jobs,
        "fresh_pairs": len(fresh),
        "estimated_single_gpu_hours": sum(
            job["estimated_hours_if_fresh"] for job in fresh
        ),
    }


def _write_csv(path, rows, tests=None):
    tests = tests or {}
    fields = (
        "id", "target_parameters", "target_architecturally_reachable",
        "theoretical_parameter_floor", "split", "soft_drop", "hard_drop",
        "resolved_log_step", "min_keep_ratio", "physical_parameters",
        "relative_target_error", "quality_feasible", "validation_accuracy",
        "validation_ce_loss", "selected_final_epoch", "test_accuracy",
        "test_ce_loss", "search_dir", "recovery_dir",
    )
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            recovery = row["recovery"]
            test = tests.get(row["id"], {})
            writer.writerow({
                "id": row["id"],
                "target_parameters": row["target_parameters"],
                "target_architecturally_reachable": row[
                    "target_architecturally_reachable"
                ],
                "theoretical_parameter_floor": row["theoretical_parameter_floor"],
                "split": row["split"],
                "soft_drop": row["soft_drop"],
                "hard_drop": row["hard_drop"],
                "resolved_log_step": row["resolved_log_step"],
                "min_keep_ratio": row["min_keep_ratio"],
                "physical_parameters": recovery["physical_parameters"],
                "relative_target_error": row["relative_target_error"],
                "quality_feasible": recovery["quality_feasible"],
                "validation_accuracy": recovery["validation_accuracy"],
                "validation_ce_loss": recovery["validation_ce_loss"],
                "selected_final_epoch": recovery["selected_final_epoch"],
                "test_accuracy": test.get("accuracy", ""),
                "test_ce_loss": test.get("ce_loss", ""),
                "search_dir": row["search"]["run_dir"],
                "recovery_dir": recovery["run_dir"],
            })
    temporary.replace(path)


def _extract_test_metrics(summary):
    runs = summary.get("runs", [])
    _require(len(runs) == 1, "Expected one frozen recovery branch in test summary")
    return dict(runs[0]["test"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-output", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--keep-nested-checkpoints", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-estimated-hours", type=float, default=31.0)
    parser.add_argument("--allow-longer", action="store_true")
    parser.add_argument("--min-free-gib", type=float)
    args = parser.parse_args(argv)

    _require(args.max_estimated_hours > 0, "Maximum estimated hours must be positive")
    reference = args.reference_output.expanduser().resolve()
    output = args.output.expanduser().resolve()
    state_path = output / "study_state.json"
    validation_csv = output / "validation_results.csv"
    final_csv = output / "final_results.csv"

    configs = validate_curve_configs()
    profile_lookup = {row["id"]: row for row in configs}
    plan = _estimate_plan(output)
    min_free_gib = float(args.min_free_gib if args.min_free_gib is not None else
                         (190.0 if args.keep_nested_checkpoints else 55.0))
    environment = epoch_split._environment_preflight(min_free_gib)
    reference_result = epoch_split._reference_result(reference)
    if (plan["estimated_single_gpu_hours"] > args.max_estimated_hours
            and not args.allow_longer):
        raise RuntimeError(
            f"Estimated fresh work is {plan['estimated_single_gpu_hours']:.1f} GPU-hours, "
            f"above --max-estimated-hours={args.max_estimated_hours:.1f}; pass "
            "--allow-longer if this is intentional."
        )
    preflight = {
        "status": "ready",
        "configs": configs,
        "plan": plan,
        "environment": environment,
        "reference": reference_result,
        "theoretical_parameter_floor": configs[0]["theoretical_parameter_floor"],
        "unreachable_planning_targets": [
            row["id"] for row in configs
            if not row["target_architecturally_reachable"]
        ],
        "checkpoint_policy": (
            "retain all nested checkpoints" if args.keep_nested_checkpoints
            else "prune nested checkpoints after each verified pair; retain root selected/deployment"
        ),
        "test_policy": (
            "write and lock the complete validation curve before frozen test inference"
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
        "protocol": "resnet50_cifar10_pruning_v3_curve60_90_low_floor_v2",
        "status": "running",
        "started_unix": time(),
        "reference": reference_result,
        "configs": configs,
        "plan": plan,
        "runs": {},
        "validation_locked_before_test": False,
        "tests": {},
        "test_comparison_scope": "exploratory; prior test results informed this study",
    }
    epoch_split._write_json(state_path, state)

    try:
        results = []
        for index, raw_row in enumerate(_profiles(), 1):
            row = profile_lookup[raw_row["id"]]
            search_dir, recovery_dir = _run_paths(output, row)
            print(
                f"[curve60] pair {index}/{len(configs)}: id={row['id']} "
                f"soft={row['soft_drop']:.6g} hard={row['hard_drop']:.6g}",
                flush=True,
            )
            if search_dir.exists() or recovery_dir.exists():
                _require(args.resume_completed,
                         f"Existing output for {row['id']}; pass --resume-completed")
                result = _completed_pair_result(search_dir, recovery_dir, row)
                print(f"[curve60] verified completed pair: {row['id']}", flush=True)
            else:
                epoch_split._run(_launcher_command(
                    SEARCH_CONFIG, reference, search_dir,
                    row=row, search_only=True,
                ))
                epoch_split._search_result(search_dir, SEARCH_EPOCHS, RECOVERY_EPOCHS)
                epoch_split._run(_launcher_command(
                    RECOVERY_CONFIG, reference, recovery_dir,
                    row=row, reuse_search=search_dir,
                ))
                result = _completed_pair_result(search_dir, recovery_dir, row)
            results.append(result)
            state["runs"][row["id"]] = result
            if not args.keep_nested_checkpoints:
                state["runs"][row["id"]]["checkpoint_cleanup"] = (
                    epoch_split._prune_pair_checkpoints(search_dir, recovery_dir)
                )
            epoch_split._write_json(state_path, state)

        _write_csv(validation_csv, results)
        state["validation_locked_before_test"] = True
        state["validation_locked_unix"] = time()
        state["validation_csv"] = str(validation_csv)
        epoch_split._write_json(state_path, state)
        print(f"[curve60] validation curve locked: {validation_csv}", flush=True)

        tests = {}
        if not args.skip_test:
            for row in results:
                recovery_dir = Path(row["recovery"]["run_dir"])
                test_dir = recovery_dir / "test_evaluation_exploratory_curve60_v2"
                if test_dir.exists():
                    _require(args.resume_completed,
                             f"Existing test output for {row['id']}; pass --resume-completed")
                else:
                    epoch_split._run([
                        sys.executable, EVALUATOR,
                        "--run-dir", recovery_dir,
                        "--output", test_dir,
                    ])
                summary = epoch_split._test_result(test_dir, recovery_dir)
                metrics = _extract_test_metrics(summary)
                tests[row["id"]] = metrics
                state["tests"][row["id"]] = {
                    "output": str(test_dir), "metrics": metrics,
                }
                epoch_split._write_json(state_path, state)
        _write_csv(final_csv, results, tests)
        state["final_csv"] = str(final_csv)
        state["status"] = "completed"
        state["finished_unix"] = time()
        epoch_split._write_json(state_path, state)
        epoch_split._write_json(output / "study_summary.json", state)
        print(f"[curve60] completed end to end: {final_csv}", flush=True)
        return state
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["finished_unix"] = time()
        epoch_split._write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
