"""Run the validation-controlled 45/105 versus 60/90 matched-size study.

The six requested models form two three-point parameter/quality curves.  Every
point uses the same pruning/recovery method; only the validation-accuracy budget
changes.  ``soft_drop`` is fixed to one half of ``hard_drop`` so that the sweep
has one pruning-strength axis rather than two independently tuned thresholds.

The nominal 5M/12M/18M labels are planning targets only.  They never participate
in checkpoint selection.  Search and recovery checkpoints are selected on
validation, then every frozen deployment is evaluated on test only after the
complete validation table has been written and locked.
"""
from __future__ import annotations

import argparse
import csv
import json
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
SPLITS = ((45, 105), (60, 90))

# Thresholds were chosen from validation/search artifacts only.  The 60/90
# 12M anchor is the completed 12.728M standard run.  The 45/105 middle budget is
# slightly looser because its completed 0.00133 budget retained 14.865M.  Every
# point preserves the canonical soft:hard = 1:2 controller relation.
GRID = (
    {
        "target": "18m",
        "target_parameters": 18_000_000,
        "hard_drop_by_search_epochs": {45: 0.0005, 60: 0.0005},
    },
    {
        "target": "12m",
        "target_parameters": 12_000_000,
        "hard_drop_by_search_epochs": {45: 0.002, 60: 0.001},
    },
    {
        "target": "5m",
        "target_parameters": 5_000_000,
        "hard_drop_by_search_epochs": {45: 0.02, 60: 0.02},
    },
)

EXPECTED_HOURS_PER_FRESH_PAIR = {45: 2.8, 60: 3.0}
DEFAULT_REFERENCE = Path("outputs/runs/epoch_split_dense_reference_seed42_fresh")
DEFAULT_OUTPUT = Path("outputs/runs/epoch_split_matched_grid_v1")
STANDARD_60_90_12M_SEARCH = Path(
    "outputs/runs/epoch_split_auto_60_90_depsafe_search"
)
STANDARD_60_90_12M_RECOVERY = Path(
    "outputs/runs/epoch_split_auto_60_90_depsafe_bn200_fixed"
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _profile_rows():
    rows = []
    for point in GRID:
        for search_epochs, recovery_epochs in SPLITS:
            hard_drop = float(point["hard_drop_by_search_epochs"][search_epochs])
            rows.append({
                "target": point["target"],
                "target_parameters": int(point["target_parameters"]),
                "search_epochs": search_epochs,
                "recovery_epochs": recovery_epochs,
                "split": f"{search_epochs}/{recovery_epochs}",
                "soft_drop": hard_drop / 2.0,
                "hard_drop": hard_drop,
            })
    return rows


def _config_name(search_epochs, recovery_epochs, phase):
    return (
        "experiment/pruning_v3/epoch_split_auto_"
        f"{search_epochs}_{recovery_epochs}_{phase}"
    )


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


def validate_grid_configs():
    """Resolve all six profiles and reject any non-threshold method drift."""
    sys.path.insert(0, str(ROOT / "src"))
    from net_complexity.training import one_shot_pruning_config as schema
    from net_complexity.training.engine import _resolve_adaptive_lambda_log_step_init
    from omegaconf import OmegaConf

    verified = []
    for row in _profile_rows():
        overrides = list(_threshold_overrides(row))
        search = schema.compose_config(
            _config_name(row["search_epochs"], row["recovery_epochs"], "search"),
            overrides=overrides,
        )
        recovery = schema.compose_config(
            _config_name(row["search_epochs"], row["recovery_epochs"], "recovery"),
            overrides=overrides,
        )
        _require(
            schema.validate_config(search) == schema.validate_config(recovery) == 150,
            f"{row['split']} {row['target']}: not a strict 150-epoch pair",
        )
        for config in (search, recovery):
            adaptive = config.training_arguments.adaptive_lambda
            _require(adaptive.enabled is True and adaptive.control_mode == "accuracy_only",
                     f"{row['split']} {row['target']}: accuracy-only controller changed")
            _require(str(adaptive.log_step).lower() == "auto",
                     f"{row['split']} {row['target']}: log step is not automatic")
            _require(float(adaptive.soft_drop) == row["soft_drop"]
                     and float(adaptive.hard_drop) == row["hard_drop"],
                     f"{row['split']} {row['target']}: threshold override was not resolved")
            _require(float(adaptive.soft_drop) * 2.0 == float(adaptive.hard_drop),
                     f"{row['split']} {row['target']}: soft:hard relation is not 1:2")
            _require(float(adaptive.alpha_init) == 0.001
                     and float(adaptive.alpha_min) == 1e-8
                     and float(adaptive.alpha_max) == 80.0,
                     f"{row['split']} {row['target']}: alpha bounds changed")
            _require(int(adaptive.update_every_search_epochs) == 1
                     and int(adaptive.gap_window) == 3
                     and int(adaptive.reentry_samples) == 3,
                     f"{row['split']} {row['target']}: controller cadence changed")
            _require(dict(config.model.criterion) == {
                "_target_": "torch.nn.CrossEntropyLoss"
            }, f"{row['split']} {row['target']}: recovery loss is not plain CE")
            _require(float(config.optimizer.lr) == 0.001
                     and float(config.optimizer.weight_decay) == 0.0005,
                     f"{row['split']} {row['target']}: AdamW settings changed")
            _require(float(config.scheduler.eta_min) == 0.0,
                     f"{row['split']} {row['target']}: cosine eta_min changed")
            _require(float(config.accuracy_guided.eligibility.min_keep_ratio) == 0.08,
                     f"{row['split']} {row['target']}: keep floor changed")
        _require(int(search.accuracy_guided.guard.train_bn_calibration_batches) == 0,
                 f"{row['split']} {row['target']}: search unexpectedly recalibrates BN")
        _require(int(recovery.accuracy_guided.guard.train_bn_calibration_batches) == 200,
                 f"{row['split']} {row['target']}: recovery is not BN200")
        plan = schema.resolved_branch_plan(recovery)
        _require(len(plan) == 1 and plan[0]["optimizer_state"] == "fresh"
                 and plan[0]["scheduler_state"] == schema.FRESH_SCHEDULER,
                 f"{row['split']} {row['target']}: recovery handoff changed")
        adaptive = search.training_arguments.adaptive_lambda
        resolved_step = _resolve_adaptive_lambda_log_step_init(
            OmegaConf.create({"num_epochs": row["search_epochs"]}),
            OmegaConf.create({"log_step_init": "auto"}),
            initial_lambda_coef=float(adaptive.alpha_init),
            warmup_epochs=int(adaptive.initial_search_warmup),
            update_every_epochs=int(adaptive.update_every_search_epochs),
        )
        verified.append({
            **row,
            "search_config": _config_name(
                row["search_epochs"], row["recovery_epochs"], "search"
            ),
            "recovery_config": _config_name(
                row["search_epochs"], row["recovery_epochs"], "recovery"
            ),
            "initial_search_warmup": int(adaptive.initial_search_warmup),
            "resolved_log_step": resolved_step,
        })
    return verified


def _run_paths(output, row, runs_root, reuse_standard_anchor):
    if (reuse_standard_anchor and row["search_epochs"] == 60
            and row["target"] == "12m"):
        return (
            runs_root / STANDARD_60_90_12M_SEARCH.name,
            runs_root / STANDARD_60_90_12M_RECOVERY.name,
            True,
        )
    stem = f"{row['target']}_{row['search_epochs']}_{row['recovery_epochs']}"
    return output / f"search_{stem}", output / f"recovery_{stem}", False


def _validate_thresholds_in_resolved_config(path, row):
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(path) / "resolved_config.yaml")
    adaptive = config.training_arguments.adaptive_lambda
    _require(float(adaptive.soft_drop) == row["soft_drop"]
             and float(adaptive.hard_drop) == row["hard_drop"],
             f"Completed run has different accuracy thresholds: {path}")
    _require(str(adaptive.log_step).lower() == "auto",
             f"Completed run did not use automatic lambda steps: {path}")


def _completed_pair_result(search_dir, recovery_dir, row):
    search = epoch_split._search_result(
        search_dir, row["search_epochs"], row["recovery_epochs"]
    )
    recovery = epoch_split._recovery_result(
        recovery_dir, search_dir, row["search_epochs"], row["recovery_epochs"]
    )
    _validate_thresholds_in_resolved_config(search_dir, row)
    _validate_thresholds_in_resolved_config(recovery_dir, row)
    return {**row, "search": search, "recovery": recovery}


def _estimate_plan(output, runs_root, reuse_standard_anchor):
    jobs = []
    for row in _profile_rows():
        search_dir, recovery_dir, external = _run_paths(
            output, row, runs_root, reuse_standard_anchor
        )
        completed_hint = search_dir.exists() and recovery_dir.exists()
        jobs.append({
            **row,
            "search_dir": str(search_dir),
            "recovery_dir": str(recovery_dir),
            "external_standard_anchor": external,
            "completed_hint": completed_hint,
            "estimated_hours_if_fresh": EXPECTED_HOURS_PER_FRESH_PAIR[
                row["search_epochs"]
            ],
        })
    fresh = [job for job in jobs if not job["completed_hint"]]
    return {
        "jobs": jobs,
        "fresh_pairs": len(fresh),
        "estimated_single_gpu_hours": sum(
            job["estimated_hours_if_fresh"] for job in fresh
        ),
    }


def _comparison_by_target(rows, relative_band):
    comparisons = {}
    for point in GRID:
        target = point["target"]
        candidates = [row for row in rows if row["target"] == target]
        _require(len(candidates) == 2, f"Expected two split results for {target}")
        for row in candidates:
            actual = row["recovery"]["physical_parameters"]
            row["relative_target_error"] = (
                actual - row["target_parameters"]
            ) / row["target_parameters"]
            row["within_target_band"] = abs(row["relative_target_error"]) <= relative_band
        matched = [row for row in candidates if row["within_target_band"]]
        feasible = [row for row in matched if row["recovery"]["quality_feasible"]]
        winner = None
        policy = (
            "no validation winner: both splits must be quality-feasible and within "
            f"±{relative_band:.0%} of the planning target"
        )
        if len(feasible) == 2:
            winner = max(feasible, key=lambda row: (
                row["recovery"]["validation_accuracy"],
                -row["recovery"]["validation_ce_loss"],
                -abs(row["relative_target_error"]),
            ))
            policy = (
                "validation-only winner among two quality-feasible deployments within "
                f"±{relative_band:.0%} of target; ties: lower CE, closer size"
            )
        comparisons[target] = {
            "target_parameters": point["target_parameters"],
            "matched": len(matched) == 2,
            "policy": policy,
            "winner_split": None if winner is None else winner["split"],
            "candidates": [{
                "split": row["split"],
                "physical_parameters": row["recovery"]["physical_parameters"],
                "relative_target_error": row["relative_target_error"],
                "within_target_band": row["within_target_band"],
                "quality_feasible": row["recovery"]["quality_feasible"],
                "validation_accuracy": row["recovery"]["validation_accuracy"],
                "validation_ce_loss": row["recovery"]["validation_ce_loss"],
            } for row in candidates],
        }
    return comparisons


def _write_csv(path, rows, tests=None):
    tests = tests or {}
    fields = (
        "target", "target_parameters", "split", "search_epochs", "recovery_epochs",
        "soft_drop", "hard_drop", "resolved_log_step", "physical_parameters",
        "relative_target_error", "within_target_band", "quality_feasible",
        "validation_accuracy", "validation_ce_loss", "selected_final_epoch",
        "test_accuracy", "test_ce_loss", "search_dir", "recovery_dir",
    )
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            recovery = row["recovery"]
            test = tests.get(f"{row['target']}:{row['split']}", {})
            writer.writerow({
                "target": row["target"],
                "target_parameters": row["target_parameters"],
                "split": row["split"],
                "search_epochs": row["search_epochs"],
                "recovery_epochs": row["recovery_epochs"],
                "soft_drop": row["soft_drop"],
                "hard_drop": row["hard_drop"],
                "resolved_log_step": row["resolved_log_step"],
                "physical_parameters": recovery["physical_parameters"],
                "relative_target_error": row.get("relative_target_error", ""),
                "within_target_band": row.get("within_target_band", ""),
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
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--prune-completed-checkpoints", action="store_true")
    parser.add_argument("--no-reuse-standard-60-90-12m", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--target-relative-band", type=float, default=0.15)
    parser.add_argument("--max-estimated-hours", type=float, default=15.0)
    parser.add_argument("--allow-longer", action="store_true")
    parser.add_argument("--min-free-gib", type=float)
    args = parser.parse_args(argv)

    _require(0 < args.target_relative_band < 1, "Target band must be in (0, 1)")
    _require(args.max_estimated_hours > 0, "Maximum estimated hours must be positive")
    reference = args.reference_output.expanduser().resolve()
    output = args.output.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve()
    reuse_anchor = not args.no_reuse_standard_60_90_12m
    state_path = output / "study_state.json"
    validation_csv = output / "validation_results.csv"
    final_csv = output / "final_results.csv"

    configs = validate_grid_configs()
    profile_lookup = {(row["target"], row["search_epochs"]): row for row in configs}
    plan = _estimate_plan(output, runs_root, reuse_anchor)
    min_free_gib = float(args.min_free_gib if args.min_free_gib is not None else
                         (55.0 if args.prune_completed_checkpoints else 190.0))
    environment = epoch_split._environment_preflight(min_free_gib)
    reference_result = epoch_split._reference_result(reference)
    if (plan["estimated_single_gpu_hours"] > args.max_estimated_hours
            and not args.allow_longer):
        raise RuntimeError(
            f"Estimated fresh work is {plan['estimated_single_gpu_hours']:.1f} GPU-hours, "
            f"above --max-estimated-hours={args.max_estimated_hours:.1f}. The completed "
            "60/90 12M anchor is probably absent. Restore it, or pass --allow-longer."
        )
    preflight = {
        "status": "ready",
        "configs": configs,
        "plan": plan,
        "environment": environment,
        "reference": reference_result,
        "test_policy": (
            "validation table and matched-size decisions are locked before any frozen "
            "test inference; all test comparisons are exploratory"
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
        "protocol": "pruning_v3_epoch_split_45_105_vs_60_90_matched_grid_v1",
        "status": "running",
        "started_unix": time(),
        "reference": reference_result,
        "configs": configs,
        "plan": plan,
        "runs": {},
        "validation_comparisons": None,
        "validation_locked_before_test": False,
        "tests": {},
        "test_comparison_scope": "exploratory; prior test results informed this study",
    }
    epoch_split._write_json(state_path, state)

    try:
        results = []
        for index, raw_row in enumerate(_profile_rows(), 1):
            row = profile_lookup[(raw_row["target"], raw_row["search_epochs"])]
            search_dir, recovery_dir, external = _run_paths(
                output, row, runs_root, reuse_anchor
            )
            key = f"{row['target']}:{row['split']}"
            print(
                f"[matched-grid] pair {index}/{len(configs)}: target={row['target']} "
                f"split={row['split']} soft={row['soft_drop']:.6g} "
                f"hard={row['hard_drop']:.6g}",
                flush=True,
            )
            if search_dir.exists() or recovery_dir.exists():
                _require(
                    external or args.resume_completed,
                    f"Existing output for {key}; pass --resume-completed",
                )
                result = _completed_pair_result(search_dir, recovery_dir, row)
                print(f"[matched-grid] verified completed pair: {key}", flush=True)
            else:
                epoch_split._run(_launcher_command(
                    row["search_config"], reference, search_dir,
                    row=row, search_only=True,
                ))
                epoch_split._search_result(
                    search_dir, row["search_epochs"], row["recovery_epochs"]
                )
                epoch_split._run(_launcher_command(
                    row["recovery_config"], reference, recovery_dir,
                    row=row, reuse_search=search_dir,
                ))
                result = _completed_pair_result(search_dir, recovery_dir, row)
            result["external_standard_anchor"] = external
            results.append(result)
            state["runs"][key] = result
            if args.prune_completed_checkpoints and not external:
                state["runs"][key]["checkpoint_cleanup"] = (
                    epoch_split._prune_pair_checkpoints(search_dir, recovery_dir)
                )
            epoch_split._write_json(state_path, state)

        comparisons = _comparison_by_target(results, args.target_relative_band)
        state["validation_comparisons"] = comparisons
        state["validation_locked_before_test"] = True
        state["validation_locked_unix"] = time()
        _write_csv(validation_csv, results)
        state["validation_csv"] = str(validation_csv)
        epoch_split._write_json(state_path, state)
        print(f"[matched-grid] validation decisions locked: {validation_csv}", flush=True)

        tests = {}
        if not args.skip_test:
            for row in results:
                key = f"{row['target']}:{row['split']}"
                recovery_dir = Path(row["recovery"]["run_dir"])
                test_dir = recovery_dir / "test_evaluation_exploratory_matched_grid_v1"
                if test_dir.exists():
                    _require(args.resume_completed,
                             f"Existing test output for {key}; pass --resume-completed")
                else:
                    epoch_split._run([
                        sys.executable, EVALUATOR,
                        "--run-dir", recovery_dir,
                        "--output", test_dir,
                    ])
                summary = epoch_split._test_result(test_dir, recovery_dir)
                metrics = _extract_test_metrics(summary)
                tests[key] = metrics
                state["tests"][key] = {"output": str(test_dir), "metrics": metrics}
                epoch_split._write_json(state_path, state)
        _write_csv(final_csv, results, tests)
        state["final_csv"] = str(final_csv)
        state["status"] = "completed"
        state["finished_unix"] = time()
        epoch_split._write_json(state_path, state)
        epoch_split._write_json(output / "study_summary.json", state)
        print(f"[matched-grid] completed end to end: {final_csv}", flush=True)
        return state
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["finished_unix"] = time()
        epoch_split._write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
