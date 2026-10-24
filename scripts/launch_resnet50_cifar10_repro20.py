"""Compare two fresh 20-epoch prefixes of the unchanged 60-epoch search recipe."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for directory in (SCRIPT_DIR, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import launch_epoch_split_study as epoch_split
import launch_resnet50_cifar10_ls0_curve7 as curve

DEFAULT_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_repro20.yaml"
HISTORY_FIELDS = (
    "train_accuracy", "valid_accuracy", "train_ce_loss", "valid_ce_loss",
    "train_L_gate_mean", "lambda_used", "lambda_next", "adaptive_lambda_action",
    "adaptive_lambda_reason", "lr", "survivor_hard_closed_channels",
    "quality_reference_accuracy",
)
EXPECTED = {
    "protocol": "resnet50_cifar10_repro20_v1",
    "search_config": "experiment/pruning_v3/resnet50_cifar10_ls0_bn_selection_search",
    "repeats": 2, "stop_after_search_epoch": 20, "search_epochs": 60,
    "recovery_epochs": 90, "soft_drop": 0.00025, "hard_drop": 0.0005,
    "min_keep_ratio": 0.08,
    "expected_reference_history_sha256": curve.QUALITY_GRID_REFERENCE_HASH,
    "expected_initializer_sha256": "82e7232c8832ab6db93d4f551e6a5eb509a5a92089fc820620e05dd80b9b9422",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_study(path=DEFAULT_CONFIG):
    study = yaml.safe_load(Path(path).read_text())
    require(isinstance(study, dict)
            and set(study) == set(EXPECTED) | {"reference_output", "output"},
            "Unexpected reproducibility manifest fields")
    for key, value in EXPECTED.items():
        require(study[key] == value and type(study[key]) is type(value),
                f"Reproducibility recipe changed: {key}")
    for key in ("reference_output", "output"):
        require(isinstance(study[key], str) and study[key], f"Invalid {key}")
    return study


def validate_recipe(study):
    """CPU-only composition; the pause must not shorten cosine or auto-lambda."""
    from net_complexity.training import one_shot_pruning_config as schema
    from net_complexity.training.engine import _resolve_adaptive_lambda_log_step_init
    from omegaconf import OmegaConf

    config = schema.compose_config(study["search_config"], curve.threshold_overrides(study))
    policy = {
        "protocol": curve.BN_SELECTION_PROTOCOL, "min_keep_ratio": 0.08,
        "search_selection_last_epochs": 30, "search_selection_bn_calibration_batches": 200,
    }
    curve.validate_recipe(config, policy, study, "search")
    step = _resolve_adaptive_lambda_log_step_init(
        OmegaConf.create({"num_epochs": 60}),
        OmegaConf.create({"log_step_init": "auto"}),
        initial_lambda_coef=0.001, warmup_epochs=10, update_every_epochs=1,
    )
    require(math.isclose(step, 0.9210340371976183, abs_tol=1e-15),
            "Automatic lambda step no longer matches the 60-epoch search")
    return {
        "protocol": study["protocol"], "repeat_count": 2,
        "search_epochs_per_repeat_this_command": 20,
        "total_training_epochs_this_command": 40,
        "search_horizon_epochs": 60, "scheduler_T_max": 60,
        "planned_per_model_budget": 150, "recovery_epochs_this_command": 0,
        "initial_search_warmup": 10, "resolved_log_step": step,
        "soft_drop": study["soft_drop"], "hard_drop": study["hard_drop"],
        "label_smoothing": 0, "min_keep_ratio": 0.08,
        "reference_training": False, "test_evaluation": False,
        "checkpoint_selection": False, "physical_export": False,
        "backend_policy": "inherited unchanged; actual flags recorded in each fresh process",
        "input_validation": "not performed by --validate-only",
    }


def validate_reference(study, reference):
    from net_complexity.training import one_shot_pruning_config as schema

    config = schema.compose_config(
        study["search_config"], curve.threshold_overrides(study), dense_source=reference,
    )
    try:
        inputs = schema.validate_inputs(config)
    except FileNotFoundError:
        missing = {key: path for key, path in schema.dense_source_paths(reference).items()
                   if not Path(path).is_file()}
        if not missing:
            raise
        details = "\n".join(f"  {key}: {path}" for key, path in missing.items())
        raise FileNotFoundError(
            "Historical repro20 inputs are missing; git clone does not include run artifacts:\n"
            + details + "\nRestore the four original files from resnet50_repro20_inputs.tar.gz "
            "or pass --reference-output PATH to the existing matching reference. "
            "This diagnostic requires the pinned historical inputs; a newly trained reference "
            "cannot replace them. See docs/experiments/resnet50_cifar10_repro20.md."
        ) from None
    require(inputs["status"] == "ready", "Existing reference inputs are not ready")
    for key, expected in (
        ("history_path", study["expected_reference_history_sha256"]),
        ("initializer_path", study["expected_initializer_sha256"]),
    ):
        require(inputs["sha256"][key] == expected,
                f"Wrong existing reference {key}; refusing to train a replacement")
    return {"root": str(reference), "sha256": inputs["sha256"],
            "initializer_model_state_hash": inputs["initializer_model_state_hash"]}


def repeat_command(study, reference, output):
    command = [sys.executable, str(ROOT / "scripts/launch_one_shot_pruning.py"),
               "--config-name", study["search_config"], "--dense-source", str(reference),
               "--output", str(output), "--search-only", "--skip-test",
               "--stop-after-search-epoch", "20", "--search-diagnostics"]
    for override in curve.threshold_overrides(study):
        command.extend(["--override", override])
    return command


def read_repeat(output):
    """Accept only a complete diagnostic prefix, never an exported deployment."""
    output = Path(output)
    state = epoch_split._read_json(output / "one_shot_state.json")
    require(state.get("status") == "search_paused" and state.get("search_only") is True,
            f"Expected search_paused at epoch20: {output}")
    require(state.get("test_evaluated") is False and not state.get("branches"),
            "Diagnostic prefix unexpectedly contains test or recovery")
    require(state.get("per_branch_total_allocated") == 150,
            "Diagnostic prefix changed the full model budget")
    ledger = state.get("shared_search_ledger", {})
    require(ledger.get("global_training_epoch") == 20
            and ledger.get("search_epochs_consumed") == 20,
            "Diagnostic prefix did not consume exactly 20 search epochs")
    require(not (output / "selection.json").exists()
            and not (output / "export_only").exists(),
            "Diagnostic prefix unexpectedly selected or exported a model")
    stage = state["stages"]["shared_search"]
    require(stage.get("epochs_allocated") == 60
            and stage.get("scheduler", {}).get("T_max") == 60,
            "Operational pause changed the 60-epoch scheduler/search horizon")
    require(stage.get("training_initializer_verified") is True,
            "Training initializer verification is missing")
    run_dir = Path(stage["run_dir"])
    if not run_dir.is_dir():
        run_dir = output / "shared_search/training" / run_dir.name
    with (run_dir / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    require([int(row["epoch"]) for row in history] == list(range(1, 21)),
            "History must contain exactly epochs1-20")
    diagnostics = output / "search_diagnostics"
    environment = epoch_split._read_json(diagnostics / "environment.json")
    with (diagnostics / "trace.jsonl").open() as stream:
        trace = [json.loads(line) for line in stream if line.strip()]
    require([row["epoch"] for row in trace if row.get("kind") == "epoch_end"]
            == list(range(1, 21)), "Diagnostic trace is missing completed epochs")
    batches = [row for row in trace if row.get("kind") == "batch_input"]
    batch_keys = [(row["epoch"], row["batch_index"]) for row in batches]
    require(len(batches) == ledger.get("optimizer_updates")
            and len(set(batch_keys)) == len(batch_keys)
            and {row["epoch"] for row in batches} == set(range(1, 21)),
            "Diagnostic trace does not cover the consumed training batches")
    return {"output": str(output), "initialization_state_hash": stage["initialization_state_hash"],
            "history": history, "trace": trace, "environment": environment}


def first_difference(left, right, path=""):
    """Describe the first differing recorded value without claiming its cause."""
    if type(left) is not type(right):
        return {"field": path, "repeat_1": left, "repeat_2": right}
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                return {"field": child, "repeat_1": left.get(key), "repeat_2": right.get(key)}
            difference = first_difference(left[key], right[key], child)
            if difference:
                return difference
        return None
    if isinstance(left, list):
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
        if len(left) != len(right):
            return {"field": f"{path}.length", "repeat_1": len(left), "repeat_2": len(right)}
        return None
    if left != right:
        return {"field": path, "repeat_1": left, "repeat_2": right}
    return None


def compare_repeats(repeats, output):
    require(len(repeats) == 2, "Comparison needs exactly two completed prefixes")
    left, right = repeats
    rows = []
    first_history_difference = None
    for a, b in zip(left["history"], right["history"]):
        fields = [key for key in HISTORY_FIELDS if key in a or key in b]
        changed = [key for key in fields if a.get(key) != b.get(key)]
        row = {"epoch": int(a["epoch"]), "different_fields": ";".join(changed)}
        for key in HISTORY_FIELDS:
            row[f"repeat_1_{key}"] = a.get(key, "")
            row[f"repeat_2_{key}"] = b.get(key, "")
        rows.append(row)
        if changed and first_history_difference is None:
            first_history_difference = {"epoch": int(a["epoch"]), "fields": changed}
    first_trace_difference = None
    for index, (a, b) in enumerate(zip(left["trace"], right["trace"])):
        difference = first_difference(a, b)
        if difference:
            first_trace_difference = {
                "event_index": index, "kind": a.get("kind"), "epoch": a.get("epoch"),
                "batch_index": a.get("batch_index"), **difference,
            }
            break
    if first_trace_difference is None and len(left["trace"]) != len(right["trace"]):
        first_trace_difference = {"field": "event_count", "repeat_1": len(left["trace"]),
                                  "repeat_2": len(right["trace"])}
    environment_difference = first_difference(left["environment"], right["environment"])
    same_initialization = left["initialization_state_hash"] == right["initialization_state_hash"]
    summary = {
        "status": "completed", "comparison": "identical" if (
            same_initialization and first_history_difference is None
            and first_trace_difference is None and environment_difference is None
        ) else "diverged",
        "initialization_identical": same_initialization,
        "first_history_difference": first_history_difference,
        "first_trace_difference": first_trace_difference,
        "first_environment_difference": environment_difference,
        "trace_event_counts": [len(row["trace"]) for row in repeats],
        "training_epochs_executed": 40, "official_test_evaluated": False,
        "interpretation": "A first recorded difference locates divergence; it does not establish its cause.",
    }
    epoch_split._write_json(output / "comparison.json", summary)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# ResNet50/CIFAR10 reproducibility prefix", "",
             f"Outcome: **{summary['comparison']}**. Both fresh processes completed 20 epochs.", "",
             "The training recipe retains a 60-epoch search, cosine horizon 60 and adaptive lambda.",
             "No recovery, checkpoint selection, physical export or official test was run.", "",
             f"Initial model states identical: {same_initialization}.", "",
             "First history difference: " + json.dumps(first_history_difference, ensure_ascii=False), "",
             "First trace difference: " + json.dumps(first_trace_difference, ensure_ascii=False), "",
             "First environment difference: " + json.dumps(environment_difference, ensure_ascii=False), "",
             summary["interpretation"], ""]
    (output / "report.md").write_text("\n".join(lines))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--validate-only", action="store_true",
                        help="CPU-only recipe validation; no inputs, CUDA, data or output creation")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Also verify existing reference hashes and GPU environment, without training")
    parser.add_argument("--min-free-gib", type=float, default=20)
    args = parser.parse_args(argv)
    study = load_study(args.config)
    report = validate_recipe(study)
    if args.validate_only:
        print(json.dumps(report, indent=2), flush=True)
        return report
    output = (args.output or ROOT / study["output"]).expanduser().resolve()
    reference = (args.reference_output or ROOT / study["reference_output"]).expanduser().resolve()
    require(not output.exists(), f"Refusing existing output: {output}; use a fresh directory")
    require(output != reference and output not in reference.parents and reference not in output.parents,
            "Reference and diagnostic outputs must be separate directories")
    reference_info = validate_reference(study, reference)
    require(math.isfinite(args.min_free_gib) and args.min_free_gib > 0,
            "min-free-gib must be positive and finite")
    report["reference"] = reference_info
    report["environment"] = epoch_split._environment_preflight(args.min_free_gib)
    report["input_validation"] = "verified existing metadata and pinned initializer/reference hashes"
    print(json.dumps(report, indent=2), flush=True)
    if args.preflight_only:
        return report
    output.mkdir(parents=True, exist_ok=False)
    state = {"protocol": study["protocol"], "status": "running", "preflight": report,
             "runs": {}, "official_test_evaluated": False}
    epoch_split._write_json(output / "study_state.json", state)
    repeats = []
    try:
        for index in (1, 2):
            identifier = f"repeat_{index}"
            require(validate_reference(study, reference) == reference_info,
                    "Reference inputs changed between fresh processes")
            destination = output / identifier
            print(f"[repro20] {identifier}: fresh process, pause after 20 of 60 search epochs", flush=True)
            command = repeat_command(study, reference, destination)
            epoch_split._run(command)
            repeat = read_repeat(destination)
            repeats.append(repeat)
            state["runs"][identifier] = {
                "status": "search_paused", "output": str(destination),
                "completed_search_epochs": 20, "command": command,
            }
            epoch_split._write_json(output / "study_state.json", state)
        state["comparison"] = compare_repeats(repeats, output)
        state["status"] = "completed"
        epoch_split._write_json(output / "study_state.json", state)
        print(f"[repro20] completed: {output / 'comparison.json'}; "
              f"comparison={state['comparison']['comparison']}", flush=True)
        return state
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        epoch_split._write_json(output / "study_state.json", state)
        raise


if __name__ == "__main__":
    main()
