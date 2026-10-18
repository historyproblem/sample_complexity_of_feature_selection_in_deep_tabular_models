"""Run independent LS=0 ResNet50/CIFAR10 adaptive-lambda 60/90 pairs.

Profiles vary validation-accuracy tolerances and, in the overnight study, the
adaptive controller's log step. Planning sizes never enter checkpoint selection.
The complete validation table is locked
before any frozen official-test inference. A failed profile is recorded without
losing the other profiles' results or silently restarting partial training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from time import time

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import launch_epoch_split_study as epoch_split


ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_ls0_curve7.yaml"
NARROW_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_ls0_curve4_narrow.yaml"
MODERATE_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_ls0_curve3_moderate.yaml"
NIGHT_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_ls0_night9h.yaml"
NIGHT_PROTOCOL = "resnet50_cifar10_ls0_night9h_v1"
PROFILE_COUNTS = {
    "resnet50_cifar10_ls0_curve7_v2": 7,
    "resnet50_cifar10_ls0_curve4_narrow_v1": 4,
    "resnet50_cifar10_ls0_curve3_moderate_v1": 3,
    NIGHT_PROTOCOL: 3,
}
TEST_DIRECTORIES = {
    "resnet50_cifar10_ls0_curve7_v2": "test_evaluation_exploratory_curve7",
    "resnet50_cifar10_ls0_curve4_narrow_v1": "test_evaluation_exploratory_curve4_narrow",
    "resnet50_cifar10_ls0_curve3_moderate_v1": "test_evaluation_exploratory_curve3_moderate",
    NIGHT_PROTOCOL: "test_evaluation_exploratory_night9h",
}
SEARCH_EPOCHS, RECOVERY_EPOCHS = 60, 90
FIELDS = (
    "id", "status", "error", "target_parameters", "physical_parameters",
    "theoretical_parameter_floor", "target_architecturally_reachable",
    "soft_drop", "hard_drop", "resolved_log_step", "min_keep_ratio", "label_smoothing", "split",
    "search_selection_epoch_start", "search_selection_epoch_end",
    "validation_accuracy", "validation_ce_loss", "quality_feasible",
    "selected_final_epoch", "test_accuracy", "test_ce_loss", "search_dir", "recovery_dir",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_study(path=DEFAULT_CONFIG):
    study = yaml.safe_load(Path(path).read_text())
    expected = {
        "protocol", "search_config", "recovery_config", "reference_output", "output",
        "requested_min_parameters", "min_keep_ratio", "search_selection_last_epochs",
        "label_smoothing", "profiles",
    }
    require(isinstance(study, dict), "Invalid study fields")
    night = study.get("protocol") == NIGHT_PROTOCOL
    if night:
        expected.add("time_budget")
    require(set(study) == expected, "Invalid study fields")
    require(study["protocol"] in PROFILE_COUNTS, "Unexpected protocol")
    explicit_thresholds = study["protocol"] != "resnet50_cifar10_ls0_curve7_v2"
    max_band = (0.01 if night or study["protocol"] == "resnet50_cifar10_ls0_curve3_moderate_v1"
                else 0.005)
    require(study["label_smoothing"] == 0, "label_smoothing must stay zero")
    require(study["min_keep_ratio"] == 0.08, "The restored common keep floor must stay 0.08")
    require(type(study["search_selection_last_epochs"]) is int
            and study["search_selection_last_epochs"] == 30,
            "Search checkpoint selection must stay inside epochs 31-60")
    require(isinstance(study["profiles"], list)
            and len(study["profiles"]) == PROFILE_COUNTS[study["protocol"]],
            "Unexpected number of independent models")
    identifiers = set()
    for row in study["profiles"]:
        fields = {"id", "target_parameters", "hard_drop"}
        if explicit_thresholds:
            fields.add("soft_drop")
        if night:
            fields.add("log_step")
        require(isinstance(row, dict) and set(row) == fields, "Invalid profile fields")
        identifier = row["id"]
        require(isinstance(identifier, str) and identifier.replace("_", "").isalnum(),
                "Profile ID must be a simple directory name")
        require(identifier not in identifiers, "Duplicate profile ID")
        identifiers.add(identifier)
        require(type(row["target_parameters"]) is int and row["target_parameters"] > 0,
                "Planning size must be a positive integer")
        require(type(row["hard_drop"]) in (int, float)
                and math.isfinite(row["hard_drop"]) and 0 < row["hard_drop"] < 1,
                "Invalid hard_drop")
        if explicit_thresholds:
            soft = row["soft_drop"]
            require(type(soft) in (int, float) and math.isfinite(soft)
                    and 0 <= soft < row["hard_drop"], "Invalid soft_drop")
            require(row["hard_drop"] - soft <= max_band + 1e-12,
                    f"Soft-hard interval must not exceed {max_band * 100:g} percentage points")
        if night:
            step = row["log_step"]
            require(step == "auto" or (type(step) in (int, float)
                    and math.isfinite(step) and step > 0), "Invalid profile log_step")
    if explicit_thresholds:
        targets = [row["target_parameters"] for row in study["profiles"]]
        ordered = sorted(targets) if night else sorted(set(targets))
        require(targets == ordered, "Profiles must run smallest first")
        require(targets[0] == study["requested_min_parameters"] and targets[-1] <= 15_000_000,
                "Narrow planning range must start at the requested minimum and end at most 15M")
    if night:
        budget = study["time_budget"]
        require(isinstance(budget, dict)
                and set(budget) == {"hours", "minutes_per_epoch", "reserve_minutes"},
                "Invalid time budget fields")
        require(all(type(value) in (int, float) and math.isfinite(value) and value > 0
                    for value in budget.values()), "Time budget values must be positive finite numbers")
        training = len(study["profiles"]) * (SEARCH_EPOCHS + RECOVERY_EPOCHS) * budget["minutes_per_epoch"]
        require(training + budget["reserve_minutes"] <= budget["hours"] * 60 + 1e-12,
                "Planned training and overhead reserve exceed the time budget")
    return study


def profile_soft_drop(row):
    return row["soft_drop"] if "soft_drop" in row else row["hard_drop"] / 2


def threshold_overrides(row):
    overrides = [
        f"training_arguments.adaptive_lambda.soft_drop={profile_soft_drop(row):.12g}",
        f"training_arguments.adaptive_lambda.hard_drop={row['hard_drop']:.12g}",
    ]
    if "log_step" in row:
        overrides.append(f"training_arguments.adaptive_lambda.log_step={row['log_step']}")
    return overrides


def validate_recipe(config, study, row, phase):
    """Validate both fresh composed configs and saved configs used by resume."""
    from net_complexity.training import one_shot_pruning_config as schema

    require(schema.validate_config(config) == 150, "Pair must have a 150-epoch budget")
    require(config.one_shot.search_epochs == 60 and config.one_shot.final_epochs == 90,
            "The common split must stay 60/90")
    require(config.one_shot.get("search_selection_last_epochs")
            == study["search_selection_last_epochs"],
            "Feasible and fallback search selection must both use epochs 31-60")
    criterion = dict(config.model.criterion)
    require(criterion.pop("label_smoothing", 0) == 0
            and criterion == {"_target_": "torch.nn.CrossEntropyLoss"},
            "Both phases require plain CE / label_smoothing=0")
    block = config.model.backbone.resnet_block
    require(config.model.entropy_regularization_coef == 0, "Entropy beta must stay zero")
    require(config.accuracy_guided.drop_mode == "learned_closed_gates"
            and block.gate_threshold == 0.5 and block.gate_internal_width is True
            and block.gate_output is False
            and block.get("gate_regularization", "l1_probability") == "l1_probability"
            and block.regularization_normalization == "initial_channels",
            "The legacy L1 internal-channel / tau=0.5 method changed")
    require(config.accuracy_guided.eligibility.min_keep_ratio == study["min_keep_ratio"],
            "The common keep floor changed")
    require(config.seed == 42 and config.optimizer._target_ == "torch.optim.AdamW"
            and config.scheduler._target_ == "torch.optim.lr_scheduler.CosineAnnealingLR"
            and config.optimizer.lr == 0.001
            and config.optimizer.weight_decay == 0.0005 and config.scheduler.eta_min == 0,
            "Seed / AdamW / cosine recipe changed")
    adaptive = config.training_arguments.adaptive_lambda
    expected = {
        "enabled": True, "control_mode": "accuracy_only", "alpha_init": 0.001,
        "alpha_min": 1e-8, "alpha_max": 80.0, "soft_drop": profile_soft_drop(row),
        "hard_drop": row["hard_drop"], "gap_window": 3,
        "update_every_search_epochs": 1, "initial_search_warmup": 10,
        "reentry_samples": 3, "log_step": row.get("log_step", "auto"),
    }
    require(all(adaptive.get(key) == value for key, value in expected.items()),
            "Adaptive-lambda settings differ from the study")
    require(config.accuracy_guided.guard.train_bn_calibration_batches
            == (0 if phase == "search" else 200), "BN calibration policy changed")
    require(config.one_shot.search_scheduler_eta_min == 0, "Search cosine must end at zero")
    if phase == "recovery":
        plan = schema.resolved_branch_plan(config)
        require(len(plan) == 1 and plan[0]["optimizer_state"] == "fresh"
                and plan[0]["scheduler_state"] == schema.FRESH_SCHEDULER,
                "Recovery must use fresh AdamW and a fresh cosine")


def validate_curve_configs(study):
    """CPU-only configuration and architecture checks; never load training data."""
    sys.path.insert(0, str(ROOT / "src"))
    from hydra.utils import instantiate
    from net_complexity.models.pruning_budget import PhysicalBudget, gates
    from net_complexity.training import one_shot_pruning_config as schema
    from net_complexity.training.engine import _resolve_adaptive_lambda_log_step_init
    from omegaconf import OmegaConf

    rows, floor = [], None
    for profile in study["profiles"]:
        configs = {}
        for phase in ("search", "recovery"):
            config = schema.compose_config(study[f"{phase}_config"], threshold_overrides(profile))
            validate_recipe(config, study, profile, phase)
            configs[phase] = config
        if floor is None:
            model = instantiate(configs["search"].model)
            mask = {}
            for name, gate in gates(model).items():
                width = int(gate.initial_channels)
                keep = max(1, math.ceil(width * study["min_keep_ratio"]))
                mask[name] = list(range(width - keep))
            floor = int(PhysicalBudget(model, mask).total())
        step = _resolve_adaptive_lambda_log_step_init(
            OmegaConf.create({"num_epochs": SEARCH_EPOCHS}),
            OmegaConf.create({"log_step_init": profile.get("log_step", "auto")}),
            initial_lambda_coef=0.001, warmup_epochs=10, update_every_epochs=1,
        )
        rows.append({
            **profile, "soft_drop": profile_soft_drop(profile),
            "min_keep_ratio": study["min_keep_ratio"], "label_smoothing": 0.0,
            "split": "60/90", "resolved_log_step": step,
            "search_selection_epoch_start": SEARCH_EPOCHS - study["search_selection_last_epochs"] + 1,
            "search_selection_epoch_end": SEARCH_EPOCHS,
            "theoretical_parameter_floor": floor,
            "target_architecturally_reachable": profile["target_parameters"] >= floor,
        })
    return rows


def validate_reference(study, reference):
    from net_complexity.training import one_shot_pruning_config as schema

    inputs = None
    for phase in ("search", "recovery"):
        config = schema.compose_config(
            study[f"{phase}_config"], threshold_overrides(study["profiles"][0]),
            dense_source=reference,
        )
        current = schema.validate_inputs(config)
        require(current["status"] == "ready", "Dense reference inputs are not ready")
        if inputs is not None:
            require(current["sha256"] == inputs["sha256"], "Phases use different reference inputs")
        inputs = current
    # Only metadata + the shared zero-epoch initializer are required. Trained
    # dense checkpoints may already have been removed by the user's cleanup.
    return {"root": str(reference), "sha256": inputs["sha256"],
            "initializer_model_state_hash": inputs["initializer_model_state_hash"]}


def write_table(path, rows, results, failures, tests):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            result = results.get(row["id"], {})
            recovery = result.get("recovery", {})
            test = tests.get(row["id"], {})
            writer.writerow({
                **{key: value for key, value in row.items() if key in FIELDS},
                "status": "failed" if row["id"] in failures else "completed",
                "error": failures.get(row["id"], ""),
                "physical_parameters": recovery.get("physical_parameters", ""),
                "validation_accuracy": recovery.get("validation_accuracy", ""),
                "validation_ce_loss": recovery.get("validation_ce_loss", ""),
                "quality_feasible": recovery.get("quality_feasible", ""),
                "selected_final_epoch": recovery.get("selected_final_epoch", ""),
                "test_accuracy": test.get("accuracy", ""),
                "test_ce_loss": test.get("ce_loss", ""),
                "search_dir": result.get("search", {}).get("run_dir", ""),
                "recovery_dir": recovery.get("run_dir", ""),
            })
    temporary.replace(path)


def run_pair(study, row, reference, output, args):
    from omegaconf import OmegaConf

    search_dir, recovery_dir = output / f"search_{row['id']}", output / f"recovery_{row['id']}"
    require(args.resume_completed or not (search_dir.exists() or recovery_dir.exists()),
            "Existing profile output; use --resume-completed to verify/reuse it")

    def launch(phase, destination):
        command = [sys.executable, ROOT / "scripts/launch_one_shot_pruning.py",
                   "--config-name", study[f"{phase}_config"],
                   "--dense-source", reference, "--output", destination]
        for override in threshold_overrides(row):
            command.extend(["--override", override])
        command.extend(["--search-only"] if phase == "search"
                       else ["--reuse-search", search_dir, "--skip-test"])
        epoch_split._run(command)

    if not search_dir.exists():
        launch("search", search_dir)
    search = epoch_split._search_result(search_dir, SEARCH_EPOCHS, RECOVERY_EPOCHS)
    require(row["search_selection_epoch_start"] <= search["selected_epoch"]
            <= row["search_selection_epoch_end"],
            "Saved search checkpoint is outside epochs 31-60")
    validate_recipe(OmegaConf.load(search_dir / "resolved_config.yaml"), study, row, "search")
    if not args.keep_nested_checkpoints:
        # Search is verified complete and root selected_checkpoint.pt is retained.
        # Release its epoch snapshots before allocating recovery snapshots.
        for path in (search_dir / "shared_search").rglob("checkpoints/*.pt"):
            require(path.is_file() and not path.is_symlink(), "Unexpected checkpoint cleanup target")
            path.unlink()
    if not recovery_dir.exists():
        launch("recovery", recovery_dir)
    recovery = epoch_split._recovery_result(recovery_dir, search_dir, SEARCH_EPOCHS, RECOVERY_EPOCHS)
    validate_recipe(OmegaConf.load(recovery_dir / "resolved_config.yaml"), study, row, "recovery")
    if not args.keep_nested_checkpoints:
        epoch_split._prune_pair_checkpoints(search_dir, recovery_dir)
    return {"search": search, "recovery": recovery}


def main(argv=None, *, default_config=DEFAULT_CONFIG):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true", help="CPU config/floor check; no data or GPU")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    parser.add_argument("--keep-nested-checkpoints", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=55.0)
    args = parser.parse_args(argv)
    study = load_study(args.config)
    rows = validate_curve_configs(study)
    floor = rows[0]["theoretical_parameter_floor"]
    preflight = {
        "protocol": study["protocol"], "profiles": rows,
        "requested_min_parameters": study["requested_min_parameters"],
        "theoretical_parameter_floor": floor,
        "requested_min_architecturally_reachable": study["requested_min_parameters"] >= floor,
        "total_training_epochs_per_model": 150,
        "dense_reference_training": "none; reuse immutable metadata and zero-epoch initializer",
        "test_policy": "lock all validation rows before frozen official test; exploratory comparison",
    }
    if "time_budget" in study:
        budget = study["time_budget"]
        planned_epochs = len(rows) * (SEARCH_EPOCHS + RECOVERY_EPOCHS)
        preflight["time_budget"] = {
            **budget, "planned_training_epochs": planned_epochs,
            "estimated_training_minutes": planned_epochs * budget["minutes_per_epoch"],
            "estimated_total_minutes_with_reserve": (
                planned_epochs * budget["minutes_per_epoch"] + budget["reserve_minutes"]
            ),
            "hard_wallclock_timeout": False,
        }
    if args.validate_only:
        print(json.dumps(preflight, indent=2), flush=True)
        return preflight
    reference = (args.reference_output or ROOT / study["reference_output"]).resolve()
    output = (args.output or ROOT / study["output"]).resolve()
    reference_info = validate_reference(study, reference)
    require(args.min_free_gib > 0, "min-free-gib must be positive")
    preflight["environment"] = epoch_split._environment_preflight(args.min_free_gib)
    preflight["reference"] = reference_info
    print(json.dumps(preflight, indent=2), flush=True)
    if not preflight["requested_min_architecturally_reachable"]:
        print(f"[ls0-curve] Requested minimum {study['requested_min_parameters']:,} is below "
              f"the {floor:,} architecture floor.", flush=True)
    if args.preflight_only:
        return preflight

    state_path = output / "study_state.json"
    study_hash = hashlib.sha256(json.dumps(study, sort_keys=True).encode()).hexdigest()
    if state_path.exists():
        require(args.resume_completed, "Existing study; use --resume-completed")
        state = epoch_split._read_json(state_path)
        require(state.get("study_hash") == study_hash and state.get("reference") == reference_info,
                "Refusing resume with different settings/reference inputs")
    else:
        state = {"protocol": study["protocol"], "study_hash": study_hash,
                 "reference": reference_info, "configs": rows, "started_unix": time(),
                 "runs": {}, "failures": {}, "tests": {}, "test_failures": {},
                 "test_comparison_scope": "exploratory", "validation_locked_before_test": False}
    output.mkdir(parents=True, exist_ok=True)
    state["status"] = "running"
    epoch_split._write_json(state_path, state)
    try:
        for index, row in enumerate(rows, 1):
            identifier = row["id"]
            print(f"[ls0-curve] {index}/{len(rows)} {identifier}: soft={row['soft_drop']:g} hard={row['hard_drop']:g}",
                  flush=True)
            try:
                state["runs"][identifier] = run_pair(study, row, reference, output, args)
                state["failures"].pop(identifier, None)
            except Exception as exc:
                state["failures"][identifier] = f"{type(exc).__name__}: {exc}"
                print(f"[ls0-curve] FAILED {identifier}: {exc}; continuing other profiles", flush=True)
            epoch_split._write_json(state_path, state)

        validation_csv = output / "validation_results.csv"
        write_table(validation_csv, rows, state["runs"], state["failures"], {})
        state["validation_locked_before_test"] = True
        state["validation_locked_unix"] = time()
        state["validation_csv"] = str(validation_csv)
        epoch_split._write_json(state_path, state)
        if not args.skip_test:
            for row in rows:
                identifier = row["id"]
                if identifier not in state["runs"] or identifier in state["failures"]:
                    continue
                recovery_dir = Path(state["runs"][identifier]["recovery"]["run_dir"])
                test_dir = recovery_dir / TEST_DIRECTORIES[study["protocol"]]
                try:
                    if not test_dir.exists():
                        epoch_split._run([sys.executable, ROOT / "scripts/evaluate_one_shot_pruning_test.py",
                                          "--run-dir", recovery_dir, "--output", test_dir])
                    summary = epoch_split._test_result(test_dir, recovery_dir)
                    require(len(summary["runs"]) == 1, "Expected one frozen physical recovery branch")
                    run = summary["runs"][0]
                    require(run["physical_parameters"]
                            == state["runs"][identifier]["recovery"]["physical_parameters"],
                            "Frozen test model size differs from the locked validation row")
                    require(run["model_state_unchanged"] is True and run["bn_counters_unchanged"] is True,
                            "Frozen test changed model/BN state")
                    metrics = dict(run["test"])
                    require(metrics["example_count"] == 10000
                            and abs(metrics["accuracy"] - metrics["correct_count"] / 10000) < 1e-12,
                            "Official test accuracy must use all 10,000 weighted examples")
                    state["tests"][identifier] = metrics
                    state["test_failures"].pop(identifier, None)
                except Exception as exc:
                    state["test_failures"][identifier] = f"{type(exc).__name__}: {exc}"
                    print(f"[ls0-curve] TEST FAILED {identifier}: {exc}", flush=True)
                epoch_split._write_json(state_path, state)
        final_csv = output / "final_results.csv"
        failures = {**state["failures"], **state["test_failures"]}
        write_table(final_csv, rows, state["runs"], failures,
                    {} if args.skip_test else state["tests"])
        state["status"] = "completed_with_failures" if failures else "completed"
        state["finished_unix"] = time()
        state["final_csv"] = str(final_csv)
        state["test_skipped"] = args.skip_test
        epoch_split._write_json(state_path, state)
        epoch_split._write_json(output / "study_summary.json", state)
        print(f"[ls0-curve] {state['status']}: {final_csv}", flush=True)
        return state
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        epoch_split._write_json(state_path, state)
        raise


if __name__ == "__main__":
    result = main()
    if result.get("status") == "completed_with_failures":
        raise SystemExit(1)
