"""One shared adaptive search, then configured compact comparison branches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__" and not any(flag in sys.argv for flag in ("--dry-run", "--help", "-h")):
    # This must precede the package import, which can initialize PyTorch.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError, ValueError):
            pass
    print("[one-shot] Starting launcher; loading configuration dependencies...", flush=True)

from net_complexity.training.one_shot_pruning_config import (
    CONFIG_NAME, compose_config, dense_source_paths, output_paths, resolved_one_shot,
    validate_config, validate_inputs,
)
from net_complexity.training.one_shot_progress import phase_progress, progress_message

DEFAULT_REFERENCE_OUTPUT = Path("outputs/runs/one_shot_dense_reference_seed42")
DEFAULT_TEST_CONFIG = ROOT / "configs/evaluation/one_shot_test.yaml"


def _run_official_test(output, config_path):
    """Evaluate only finalized frozen deployments and persist the official test report."""
    import evaluate_one_shot_pruning_test as evaluation
    args = evaluation.evaluation_args_from_config(
        config_path, run_dir=output, output=output / "test_evaluation")
    return evaluation.run(args)


def _new_reference_preview(config, reference_output, output, *, prepare_only):
    """Resolve the clean-clone plan without importing the reference/training runtime."""
    report = resolved_one_shot(config, check_inputs=False, output_root=output)
    dense_epochs = int(config.accuracy_guided.total_epochs)
    mode = "prepare_reference_only" if prepare_only else "prepare_reference_then_one_shot"
    report["mode"] = mode
    report["inputs"] = {"status": "planned_new_reference", "paths": dense_source_paths(reference_output)}
    report["reference_preparation"] = {
        "output_root": str(reference_output), "output_exists": reference_output.exists(),
        "initializer": "shared_random_seed42.pt", "initializer_origin": "newly_generated",
        "initializer_trained_epochs": 0,
        "seed": int(config.seed), "dense_training_epochs": dense_epochs,
        "dense_gates": False, "dense_adaptive_lambda": False,
        "search_initialization": "same_new_zero_epoch_initializer; no_trained_dense_weights",
        "replaces_missing_historical_reference": True,
        "selection_data": "validation", "test_evaluated": False,
        "existing_output_policy": "refuse_overwrite_or_resume",
    }
    pruning_epochs = report["budget"].get(
        "total_unique_training_epochs_both_branches",
        report["budget"].get("total_unique_training_epochs_all_branches"),
    )
    report["budget"].update({
        "dense_reference_training_epochs": dense_epochs,
        "total_unique_training_epochs_including_reference": dense_epochs + pruning_epochs,
        "total_unique_training_epochs_this_command": dense_epochs if prepare_only else dense_epochs + pruning_epochs,
        "dense_reference_is_separate_from_pruning_branch_budget": True,
    })
    graph = {"dense_reference": {"epochs": dense_epochs}}
    if not prepare_only:
        graph.update(report["execution_graph"])
    report["execution_graph"] = graph
    report["training_performed"] = False
    return report


def _check_new_paths(reference_output, output, *, prepare_only):
    if reference_output.exists():
        raise FileExistsError(f"Refusing existing dense reference output: {reference_output}. Use a fresh directory.")
    if prepare_only:
        return
    if output == reference_output or output in reference_output.parents or reference_output in output.parents:
        raise ValueError("Dense reference and one-shot outputs must be separate directories, neither inside the other.")
    if output.exists():
        raise FileExistsError(f"Refusing existing one-shot output: {output}. Use a fresh --output directory.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=CONFIG_NAME)
    parser.add_argument("--output", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dense-source", type=Path,
                        help="Existing dense run root or J1_dense_control directory; resolves reference metadata and shared zero-epoch initializer")
    source.add_argument("--prepare-reference", type=Path, metavar="PATH",
                        help="Create a NEW shared seed42 initializer and train only its dense150 validation reference in PATH")
    source.add_argument("--from-scratch", action="store_true",
                        help="Create a NEW shared seed42 initializer, train its dense150 reference, then run the configured search and compact branches")
    parser.add_argument("--reference-output", type=Path,
                        help=f"New reference directory for --from-scratch (default: {DEFAULT_REFERENCE_OUTPUT})")
    parser.add_argument("--test-config", type=Path, default=DEFAULT_TEST_CONFIG,
                        help="Official-test inference profile run automatically after frozen deployments are finalized")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths/contracts without training, CUDA, or dataset construction")
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Stop after 60-epoch search, checkpoint selection and physical export; do not run recovery or test",
    )
    parser.add_argument(
        "--reuse-search",
        type=Path,
        metavar="PATH",
        help="Reuse a completed search-only run and execute only its configured physical recovery branches",
    )
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Explicit configuration override; strict one-shot constraints still apply")
    args = parser.parse_args(argv)
    if args.reference_output is not None and not args.from_scratch:
        parser.error("--reference-output requires --from-scratch; use --prepare-reference PATH for preparation alone")
    if args.prepare_reference is not None and args.output is not None:
        parser.error("--output is for one-shot results; --prepare-reference PATH creates only the reference")
    if args.search_only and args.reuse_search is not None:
        parser.error("--search-only and --reuse-search are mutually exclusive")
    new_reference = args.from_scratch or args.prepare_reference is not None
    if new_reference and args.reuse_search is not None:
        parser.error("--reuse-search requires an existing --dense-source; it cannot create a new reference")
    reference_output = ((args.prepare_reference if args.prepare_reference is not None
                         else args.reference_output or DEFAULT_REFERENCE_OUTPUT).expanduser().resolve()
                        if new_reference else None)
    config = compose_config(args.config_name, args.override,
                            dense_source=reference_output if new_reference else args.dense_source)
    output = Path(output_paths(config, args.output)["root"])
    if new_reference:
        validate_config(config)
        expected = dense_source_paths(reference_output)
        if any(reference_output not in Path(value).resolve().parents for value in expected.values()):
            parser.error("New reference output must be the parent run directory, not J1_dense_control; all generated artifacts must remain inside it")
        actual = {**dict(config.accuracy_guided.reference),
                  "initializer_path": str(config.accuracy_guided.initializer.path)}
        if any(Path(actual[key]).expanduser().resolve() != Path(value) for key, value in expected.items()):
            parser.error("New-reference modes require all initializer/reference paths inside their new reference output; remove path overrides")
    if args.dry_run:
        report = (_new_reference_preview(config, reference_output, output,
                                         prepare_only=args.prepare_reference is not None)
                  if new_reference else resolved_one_shot(config, output_root=args.output))
        if args.search_only:
            search_epochs = int(config.one_shot.search_epochs)
            dense_epochs = int(report["budget"].get("dense_reference_training_epochs", 0))
            report["mode"] = "search_only"
            report["budget"].update({
                "recovery_training_epochs_this_command": 0,
                "pruning_training_epochs_this_command": search_epochs,
                "total_unique_training_epochs_this_command": dense_epochs + search_epochs,
            })
        if args.reuse_search is not None:
            final_epochs = int(config.one_shot.final_epochs)
            branches = len(report["execution_policy"]["branch_plan"])
            report["mode"] = "reuse_search_then_recovery"
            report["reused_search"] = str(args.reuse_search.expanduser().resolve())
            report["budget"].update({
                "search_training_epochs_this_command": 0,
                "recovery_training_epochs_this_command": branches * final_epochs,
                "total_unique_training_epochs_this_command": branches * final_epochs,
            })
        requested_execution = {
            "search_only": bool(args.search_only),
            "stops_after": "physical_export" if args.search_only else "configured_recovery",
        }
        if args.reuse_search is not None:
            requested_execution["reuse_search"] = str(args.reuse_search.expanduser().resolve())
        report["requested_execution"] = requested_execution
        report["official_test_evaluation"] = {
            "automatic_after_frozen_deployments": args.prepare_reference is None and not args.search_only,
            "config": str(args.test_config.expanduser().resolve()),
            "output": str((output / "test_evaluation").resolve()),
            "training": False, "batchnorm_updates": False, "selection": False,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return report
    progress_message("one-shot", f"device={config.device}; seed={config.seed}; output={output}")
    if new_reference:
        _check_new_paths(reference_output, output, prepare_only=args.prepare_reference is not None)
        progress_message("one-shot", f"new dense reference: {reference_output}; epochs={config.accuracy_guided.total_epochs}")
        with phase_progress("runtime", "loading dense reference training code"):
            from net_complexity.training.one_shot_reference import prepare_dense_reference
        reference_state = prepare_dense_reference(config, reference_output)
        if args.prepare_reference is not None:
            return reference_state
    if output.exists():
        raise FileExistsError(f"Refusing existing one-shot output: {output}. Use a fresh --output directory.")
    try:
        with phase_progress("preflight", "checking dense validation reference and zero-epoch initializer"):
            validate_inputs(config)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    # Preview/preflight cannot import or accidentally invoke the training engine.
    with phase_progress("runtime", "loading one-shot training code"):
        from net_complexity.training.one_shot_pruning import run_one_shot_pruning
    if args.reuse_search is not None:
        result = run_one_shot_pruning(config, output, reuse_search_from=args.reuse_search)
    elif args.search_only:
        result = run_one_shot_pruning(config, output, search_only=True)
    else:
        result = run_one_shot_pruning(config, output)
    clean_clone_state = None
    if args.from_scratch:
        from net_complexity.training.pruning_measurement import write_json
        reference_epochs = reference_state["reference_training_epochs_actually_executed"]
        pruning_epochs = result["compute_ledger"]["actual_training_epochs_executed"]
        clean_clone_state = {
            "protocol": "pruning_v3_clean_clone", "status": result["status"],
            "reference": {
                "root": str(reference_output),
                "state_path": dense_source_paths(reference_output)["state_path"],
                "status": reference_state["status"],
                "reference_origin": reference_state["reference_origin"],
                "common_init_hash": reference_state["common_init_hash"],
                "initializer_file_hash": reference_state["initializer_file_hash"],
            },
            "pruning": {"root": str(output), "state_path": str(output / "one_shot_state.json")},
            "compute_ledger": {
                "reference_training_epochs_actually_executed": reference_epochs,
                "pruning_training_epochs_actually_executed": pruning_epochs,
                "actual_training_epochs_executed": reference_epochs + pruning_epochs,
            },
            "per_pruning_branch_total_allocated": result["per_branch_total_allocated"],
            "reference_cost_is_external_to_pruning_branch_budget": True,
            "test_evaluated": bool(reference_state["test_evaluated"] or result["test_evaluated"]),
        }
        write_json(output / "clean_clone_state.json", clean_clone_state)
    if result.get("status") == "completed":
        progress_message("official-test", "evaluating all finalized frozen deployments")
        test_report = _run_official_test(output, args.test_config)
        result["official_test_evaluation"] = {
            "status": test_report["status"], "test_evaluated": test_report["test_evaluated"],
            "summary": str(output / "test_evaluation/test_summary.json"),
            "comparison": str(output / "test_evaluation/test_comparison.csv"),
        }
        if clean_clone_state is not None:
            clean_clone_state["test_evaluated"] = bool(test_report["test_evaluated"])
            clean_clone_state["test_evaluation"] = result["official_test_evaluation"]
            write_json(output / "clean_clone_state.json", clean_clone_state)
    else:
        result["official_test_evaluation"] = {
            "status": "not_run", "test_evaluated": False,
            "reason": f"no finalized branch pair: one-shot status={result.get('status', 'unknown')}",
        }
    progress_message("one-shot", f"finished: status={result.get('status', 'returned')}; results={output}")
    return result


if __name__ == "__main__":
    main()
