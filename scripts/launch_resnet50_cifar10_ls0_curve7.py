"""Run independent LS=0 ResNet50/CIFAR10 adaptive-lambda 60/90 pairs.

Profiles vary validation-accuracy tolerances and, in the overnight study, the
adaptive controller's log step. The pre-BN reproduction crosses two historical
tolerance pairs with two channel floors. Planning sizes never enter selection.
The complete validation table is locked
before any frozen official-test inference. A failed profile is recorded without
losing the other profiles' results or silently restarting partial training.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
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
BN_SELECTION_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_ls0_night9h_bn_selection.yaml"
BN_SELECTION_PROTOCOL = "resnet50_cifar10_ls0_night9h_bn_selection_v1"
KEEP001_TARGET4P5M_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_ls0_keep001_target4p5m.yaml"
KEEP001_TARGET14M_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_ls0_keep001_target14m.yaml"
KEEP001_TARGET4P5M_PROTOCOL = "resnet50_cifar10_ls0_keep001_target4p5m_v1"
KEEP001_TARGET14M_PROTOCOL = "resnet50_cifar10_ls0_keep001_target14m_v1"
QUALITY_GRID_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_ls0_keep001_quarterpp_grid8.yaml"
QUALITY_GRID_PROTOCOL = "resnet50_cifar10_ls0_keep001_quarterpp_grid8_v1"
QUALITY_GRID_REFERENCE_HASH = "47b56d77615d01e9780f7765d10d144ad1f3d6db5897d777687661f4a6f23a76"
FULL150_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_ls0_full150.yaml"
FULL150_PROTOCOL = "resnet50_cifar10_ls0_full150_v1"
FULL150_INITIALIZER_HASH = "82e7232c8832ab6db93d4f551e6a5eb509a5a92089fc820620e05dd80b9b9422"
EXTRA3_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_ls0_extra3.yaml"
EXTRA3_PROTOCOL = "resnet50_cifar10_ls0_extra3_v1"
PREBN_REPRO_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_ls0_prebn_repro4.yaml"
PREBN_REPRO_PROTOCOL = "resnet50_cifar10_ls0_prebn_repro4_v1"
PREBN_REPEAT_CONFIG = ROOT / "configs/experiment/pruning_v3/resnet50_cifar10_ls0_prebn_repeat8.yaml"
PREBN_REPEAT_PROTOCOL = "resnet50_cifar10_ls0_prebn_repeat8_v1"
PREBN_PROTOCOLS = {PREBN_REPRO_PROTOCOL, PREBN_REPEAT_PROTOCOL}
PREBN_REPEAT_IDS = tuple(f"large_15m_keep001_repeat_{index:02d}" for index in range(1, 9))
PREBN_REPRO_POINTS = {
    "compact_5m": (5_000_000, 0.015, 0.03),
    "large_15m": (15_000_000, 0.00025, 0.0005),
}
PINNED_INITIALIZER_PROTOCOLS = {FULL150_PROTOCOL, EXTRA3_PROTOCOL, *PREBN_PROTOCOLS}
PINNED_REFERENCE_PROTOCOLS = {QUALITY_GRID_PROTOCOL, *PINNED_INITIALIZER_PROTOCOLS}
KEEP001_PROTOCOLS = {
    KEEP001_TARGET4P5M_PROTOCOL, KEEP001_TARGET14M_PROTOCOL, QUALITY_GRID_PROTOCOL,
}
BN_SELECTION_PROTOCOLS = {
    BN_SELECTION_PROTOCOL, FULL150_PROTOCOL, EXTRA3_PROTOCOL, *KEEP001_PROTOCOLS,
}
NIGHT_PROTOCOLS = {NIGHT_PROTOCOL, BN_SELECTION_PROTOCOL}
PROFILE_COUNTS = {
    "resnet50_cifar10_ls0_curve7_v2": 7,
    "resnet50_cifar10_ls0_curve4_narrow_v1": 4,
    "resnet50_cifar10_ls0_curve3_moderate_v1": 3,
    NIGHT_PROTOCOL: 3,
    BN_SELECTION_PROTOCOL: 3,
    KEEP001_TARGET4P5M_PROTOCOL: 1,
    KEEP001_TARGET14M_PROTOCOL: 1,
    QUALITY_GRID_PROTOCOL: 8,
    FULL150_PROTOCOL: 1,
    EXTRA3_PROTOCOL: 3,
    PREBN_REPRO_PROTOCOL: 4,
    PREBN_REPEAT_PROTOCOL: 8,
}
TEST_DIRECTORIES = {
    "resnet50_cifar10_ls0_curve7_v2": "test_evaluation_exploratory_curve7",
    "resnet50_cifar10_ls0_curve4_narrow_v1": "test_evaluation_exploratory_curve4_narrow",
    "resnet50_cifar10_ls0_curve3_moderate_v1": "test_evaluation_exploratory_curve3_moderate",
    NIGHT_PROTOCOL: "test_evaluation_exploratory_night9h",
    BN_SELECTION_PROTOCOL: "test_evaluation_exploratory_night9h_bn_selection",
    KEEP001_TARGET4P5M_PROTOCOL: "test_evaluation_exploratory_keep001_target4p5m",
    KEEP001_TARGET14M_PROTOCOL: "test_evaluation_exploratory_keep001_target14m",
    QUALITY_GRID_PROTOCOL: "test_evaluation_exploratory_keep001_quarterpp_grid8",
    FULL150_PROTOCOL: "test_evaluation_exploratory_full150",
    EXTRA3_PROTOCOL: "test_evaluation_exploratory_extra3",
    PREBN_REPRO_PROTOCOL: "test_evaluation_exploratory_prebn_repro4",
    PREBN_REPEAT_PROTOCOL: "test_evaluation_exploratory_prebn_repeat8",
}
SEARCH_EPOCHS, RECOVERY_EPOCHS = 60, 90
FIELDS = (
    "id", "point_id", "status", "error", "target_parameters", "physical_parameters",
    "theoretical_parameter_floor", "target_architecturally_reachable",
    "soft_drop", "hard_drop", "resolved_log_step", "min_keep_ratio", "label_smoothing", "split",
    "reference_final_validation_accuracy", "required_final_validation_accuracy",
    "soft_final_validation_accuracy",
    "search_selection_epoch_start", "search_selection_epoch_end",
    "fallback_epoch_start", "fallback_epoch_end",
    "search_selection_bn_calibration_batches",
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
    quality_grid = study.get("protocol") == QUALITY_GRID_PROTOCOL
    prebn_repro = study.get("protocol") in PREBN_PROTOCOLS
    prebn_repeat = study.get("protocol") == PREBN_REPEAT_PROTOCOL
    pinned_reference = study.get("protocol") in PINNED_REFERENCE_PROTOCOLS
    pinned_initializer = study.get("protocol") in PINNED_INITIALIZER_PROTOCOLS
    if pinned_reference:
        expected.remove("requested_min_parameters")
        expected.update({"reference_final_validation_accuracy", "expected_reference_history_sha256"})
    if pinned_initializer:
        expected.add("expected_initializer_sha256")
    night = study.get("protocol") in NIGHT_PROTOCOLS
    bn_selection = study.get("protocol") in BN_SELECTION_PROTOCOLS
    if night:
        expected.add("time_budget")
    if bn_selection or prebn_repro:
        expected.add("search_selection_bn_calibration_batches")
    require(set(study) == expected, "Invalid study fields")
    require(study["protocol"] in PROFILE_COUNTS, "Unexpected protocol")
    explicit_thresholds = study["protocol"] != "resnet50_cifar10_ls0_curve7_v2"
    max_band = (0.01 if night or study["protocol"] == "resnet50_cifar10_ls0_curve3_moderate_v1"
                else 0.005)
    if prebn_repro:
        max_band = 0.015
    require(study["label_smoothing"] == 0, "label_smoothing must stay zero")
    keep_floor = 0.001 if study["protocol"] in KEEP001_PROTOCOLS or prebn_repeat else 0.08
    require(study["min_keep_ratio"] == keep_floor,
            f"This study's keep floor must stay {keep_floor:g}")
    if bn_selection:
        require(type(study["search_selection_bn_calibration_batches"]) is int
                and study["search_selection_bn_calibration_batches"] == 200,
                "The BN-selection study requires 200 train-only batches per candidate")
    if prebn_repro:
        require(type(study["search_selection_bn_calibration_batches"]) is int
                and study["search_selection_bn_calibration_batches"] == 0,
                "Historical reproduction requires candidate BN calibration to stay off")
    require(type(study["search_selection_last_epochs"]) is int
            and study["search_selection_last_epochs"] == 30,
            "Search checkpoint selection must stay inside epochs 31-60")
    require(isinstance(study["profiles"], list)
            and len(study["profiles"]) == PROFILE_COUNTS[study["protocol"]],
            "Unexpected number of independent models")
    identifiers = set()
    for row in study["profiles"]:
        fields = {"id", "hard_drop"}
        if not pinned_reference or prebn_repro:
            fields.add("target_parameters")
        if prebn_repro:
            fields.update({"point_id", "min_keep_ratio"})
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
        if not pinned_reference or prebn_repro:
            require(type(row["target_parameters"]) is int and row["target_parameters"] > 0,
                    "Planning size must be a positive integer")
        if prebn_repro:
            require(isinstance(row["point_id"], str) and row["point_id"] in PREBN_REPRO_POINTS,
                    "Unknown historical reproduction point")
            require(type(row["min_keep_ratio"]) in (int, float)
                    and row["min_keep_ratio"] in (0.001, 0.08),
                    "Historical reproduction requires channel floors 0.001 and 0.08")
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
    if prebn_repro:
        if prebn_repeat:
            require(tuple(row["id"] for row in study["profiles"]) == PREBN_REPEAT_IDS,
                    "Repeat study requires eight distinct, ordered repeat IDs 01-08")
            require(all(row["point_id"] == "large_15m" and row["min_keep_ratio"] == 0.001
                        for row in study["profiles"]),
                    "Every repeat must preserve the large_15m_keep001 recipe")
            for phase in ("search", "recovery"):
                require(study[f"{phase}_config"]
                        == f"experiment/pruning_v3/resnet50_cifar10_ls0_prebn_repro4_{phase}",
                        "Repeats must reuse the original pre-BN phase configs")
        else:
            require({(row["point_id"], row["min_keep_ratio"]) for row in study["profiles"]}
                    == {(point, floor) for point in PREBN_REPRO_POINTS for floor in (0.001, 0.08)},
                    "Reproduction requires both floors for each of the two historical points")
        for row in study["profiles"]:
            require((row["target_parameters"], row["soft_drop"], row["hard_drop"])
                    == PREBN_REPRO_POINTS[row["point_id"]],
                    "Historical reproduction thresholds and planning sizes must stay fixed")
    if pinned_reference:
        require(study["reference_final_validation_accuracy"] == 0.9414,
                "Accuracy bands require the 94.14% final validation reference")
        require(study["expected_reference_history_sha256"] == QUALITY_GRID_REFERENCE_HASH,
                "Accuracy bands require the original dense validation history")
    if pinned_initializer:
        require(study["expected_initializer_sha256"] == FULL150_INITIALIZER_HASH,
                "This study requires the original diagnostic zero-epoch initializer")
        require(study["output"] == f"outputs/runs/{study['protocol']}",
                "The full run must use its own output directory")
        if study["protocol"] == EXTRA3_PROTOCOL:
            reference = Decimal(str(study["reference_final_validation_accuracy"]))
            bands = (("0.924", "0.9265"), ("0.929", "0.9315"), ("0.935", "0.9375"))
            for row, (lower, upper) in zip(study["profiles"], bands):
                soft, hard = Decimal(str(row["soft_drop"])), Decimal(str(row["hard_drop"]))
                require((reference - hard, reference - soft) == (Decimal(lower), Decimal(upper)),
                        "Extra3 must preserve the three requested validation bands and their order")
    elif quality_grid:
        for index, row in enumerate(study["profiles"], 1):
            soft, hard = Decimal(str(row["soft_drop"])), Decimal(str(row["hard_drop"]))
            require(hard - soft == Decimal("0.0025"),
                    "Accuracy bands must be exactly 0.25 percentage points wide")
            require(hard == Decimal("0.0025") * index,
                    "Run the eight adjacent quarter-point bands strictest first")
        require(study["output"] == f"outputs/runs/{QUALITY_GRID_PROTOCOL}",
                "The accuracy grid must use its own output directory")
    elif explicit_thresholds:
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
    if "point_id" in row:
        overrides.append(f"accuracy_guided.eligibility.min_keep_ratio={row['min_keep_ratio']:.12g}")
    return overrides


def profile_policy(study, row):
    """Keep the selected channel floor consistent through search, recovery and resume."""
    if study["protocol"] in PREBN_PROTOCOLS:
        return {**study, "min_keep_ratio": row["min_keep_ratio"]}
    return study


def validate_recipe(config, study, row, phase):
    """Validate both fresh composed configs and saved configs used by resume."""
    from net_complexity.training import one_shot_pruning_config as schema

    study = profile_policy(study, row)
    require(schema.validate_config(config) == 150, "Pair must have a 150-epoch budget")
    require(config.one_shot.search_epochs == 60 and config.one_shot.final_epochs == 90,
            "The common split must stay 60/90")
    require(config.one_shot.get("search_selection_last_epochs")
            == study["search_selection_last_epochs"],
            "Feasible and fallback search selection must both use epochs 31-60")
    selection_bn_batches = study.get("search_selection_bn_calibration_batches", 0)
    require(config.one_shot.get("search_selection_bn_calibration_batches", 0)
            == selection_bn_batches,
            "Search checkpoint BN-calibration selection policy changed")
    if study["protocol"] in BN_SELECTION_PROTOCOLS or study["protocol"] in PREBN_PROTOCOLS:
        require("search_selection_bn_calibration_batches" in config.one_shot,
                "The recipe must explicitly declare its candidate BN selection policy")
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

    rows, floors, model = [], {}, None
    for profile in study["profiles"]:
        policy = profile_policy(study, profile)
        configs = {}
        for phase in ("search", "recovery"):
            config = schema.compose_config(study[f"{phase}_config"], threshold_overrides(profile))
            validate_recipe(config, study, profile, phase)
            configs[phase] = config
        keep_ratio = policy["min_keep_ratio"]
        if keep_ratio not in floors:
            if model is None:
                model = instantiate(configs["search"].model)
            mask = {}
            for name, gate in gates(model).items():
                width = int(gate.initial_channels)
                keep = max(1, math.ceil(width * keep_ratio))
                mask[name] = list(range(width - keep))
            floors[keep_ratio] = int(PhysicalBudget(model, mask).total())
        floor = floors[keep_ratio]
        step = _resolve_adaptive_lambda_log_step_init(
            OmegaConf.create({"num_epochs": SEARCH_EPOCHS}),
            OmegaConf.create({"log_step_init": profile.get("log_step", "auto")}),
            initial_lambda_coef=0.001, warmup_epochs=10, update_every_epochs=1,
        )
        row = {
            **profile, "soft_drop": profile_soft_drop(profile),
            "min_keep_ratio": keep_ratio, "label_smoothing": 0.0,
            "split": "60/90", "resolved_log_step": step,
            "search_selection_epoch_start": SEARCH_EPOCHS - study["search_selection_last_epochs"] + 1,
            "search_selection_epoch_end": SEARCH_EPOCHS,
            "search_selection_bn_calibration_batches": study.get(
                "search_selection_bn_calibration_batches", 0
            ),
            "theoretical_parameter_floor": floor,
            "target_architecturally_reachable": (
                profile["target_parameters"] >= floor if "target_parameters" in profile else None
            ),
        }
        if study["protocol"] in PREBN_PROTOCOLS:
            row.update(fallback_epoch_start=31, fallback_epoch_end=60)
        if study["protocol"] in PINNED_REFERENCE_PROTOCOLS:
            reference = Decimal(str(study["reference_final_validation_accuracy"]))
            row.update({
                "reference_final_validation_accuracy": float(reference),
                "required_final_validation_accuracy": float(
                    reference - Decimal(str(profile["hard_drop"]))
                ),
                "soft_final_validation_accuracy": float(
                    reference - Decimal(str(profile["soft_drop"]))
                ),
            })
        rows.append(row)
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
    if study["protocol"] in PINNED_REFERENCE_PROTOCOLS:
        from net_complexity.training.pruning_audit import load_adaptive_reference

        require(inputs["sha256"]["history_path"] == study["expected_reference_history_sha256"],
                "Dense reference history differs from the accuracy-grid reference")
        curve = load_adaptive_reference(inputs["paths"]["history_path"], 150)
        require(curve[150] == study["reference_final_validation_accuracy"],
                "Final dense validation accuracy differs from the accuracy-grid reference")
    if study["protocol"] in PINNED_INITIALIZER_PROTOCOLS:
        require(inputs["sha256"]["initializer_path"] == study["expected_initializer_sha256"],
                "Zero-epoch initializer differs from the full-run diagnostic reference")
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


def validate_saved_selection_bn(run_dir, study):
    """Reject saved selections produced with a different candidate-scoring policy."""
    batches = study.get("search_selection_bn_calibration_batches", 0)
    if not batches:
        if study["protocol"] in PREBN_PROTOCOLS:
            selection = epoch_split._read_json(Path(run_dir) / "selection.json")
            trace = selection.get("trace", {})
            predictor = "dependency_safe_all_physical_survivors_open"
            calibration = selection.get("selection_bn_calibration", {})
            require(selection.get("selection_predictor") == predictor
                    and trace.get("selection_predictor") == predictor
                    and trace.get("bn_calibration_batches", 0) == 0
                    and calibration.get("requested_batches", 0) == 0
                    and calibration.get("batches", 0) == 0,
                    "Saved search selection does not match the pre-BN reproduction policy")
            require(trace.get("candidate_epoch_start") == 31
                    and trace.get("candidate_epoch_end") == SEARCH_EPOCHS
                    and 31 <= selection.get("selected_epoch", -1) <= SEARCH_EPOCHS,
                    "Saved search selection must use epochs 31-60")
            expected_policy = ("best_validation_last_epochs_fallback" if trace.get("fallback_used")
                               else "best_feasible_compact")
            require(selection.get("policy") == trace.get("policy") == expected_policy,
                    "Saved search must preserve the historical minimum-size / fallback rule")
            if trace.get("fallback_used"):
                require(trace.get("fallback_last_epochs") == 30
                        and trace.get("fallback_epoch_start") == 31,
                        "Saved fallback selection must use epochs 31-60")
        return
    selection = epoch_split._read_json(Path(run_dir) / "selection.json")
    require(selection.get("selection_predictor") == "physical_after_train_bn_calibration",
            "Saved search selection did not use physical post-BN validation")
    calibration = selection.get("selection_bn_calibration", {})
    require(calibration.get("requested_batches") == batches
            and calibration.get("batches") == batches
            and calibration.get("seed") == 42
            and calibration.get("trainable_weights_unchanged") is True,
            "Saved search selection is missing the required BN200 calibration evidence")
    validation = selection.get("selection_validation", {})
    require(validation.get("example_count") == 5000
            and type(validation.get("correct_count")) is int
            and abs(validation.get("accuracy", -1) - validation["correct_count"] / 5000) < 1e-12,
            "Saved post-BN search selection did not evaluate the full validation split")
    trace = selection.get("trace", {})
    require(trace.get("selection_predictor") == "physical_after_train_bn_calibration"
            and trace.get("bn_calibration_batches") == batches,
            "Saved checkpoint-selection trace has a different BN-calibration policy")


def read_search_result(run_dir, study):
    """Verify the calibrated export without changing legacy artifact readers."""
    batches = study.get("search_selection_bn_calibration_batches", 0)
    if not batches:
        validate_saved_selection_bn(run_dir, study)
        return epoch_split._search_result(run_dir, SEARCH_EPOCHS, RECOVERY_EPOCHS)
    run_dir = Path(run_dir)
    state = epoch_split._read_json(run_dir / "one_shot_state.json")
    selection = epoch_split._read_json(run_dir / "selection.json")
    diagnostics = epoch_split._read_json(run_dir / "export_only/diagnostics.json")
    for path in (run_dir / "selected_checkpoint.pt", run_dir / "export_only/deployment.pt",
                 run_dir / "resolved_config.yaml"):
        require(path.is_file(), f"Completed search is missing {path}")
    require(state.get("status") == "search_only_completed"
            and state.get("search_only") is True and state.get("branches") == {},
            "Calibrated search-only artifact is not completed or contains recovery branches")
    require(state.get("protocol") == epoch_split.SEARCH_PROTOCOL
            and state.get("per_branch_total_allocated") == 150,
            "Calibrated search protocol or training budget differs")
    ledger = state.get("shared_search_ledger", {})
    require(ledger.get("global_training_epoch") == SEARCH_EPOCHS
            and ledger.get("search_epochs_consumed") == SEARCH_EPOCHS,
            "Calibrated search ledger is incomplete")
    require(selection.get("reference_epoch") == SEARCH_EPOCHS,
            "Calibrated search used the wrong validation reference epoch")
    calibration = diagnostics.get("bn_calibration", {})
    require(diagnostics.get("status") == "measured"
            and diagnostics.get("selection_predictor") == "physical_after_train_bn_calibration"
            and diagnostics.get("bn_calibration_batches") == batches
            and calibration.get("requested_batches") == batches
            and calibration.get("batches") == batches
            and calibration.get("trainable_weights_unchanged") is True,
            "Selected physical search export is missing BN200 calibration evidence")
    validation = diagnostics.get("physical_validation", {})
    require(validation.get("example_count") == 5000,
            "Calibrated physical search export did not evaluate the full validation split")
    validate_saved_selection_bn(run_dir, study)
    return {
        "split": "60/90", "run_dir": str(run_dir.resolve()), "status": state["status"],
        "selected_epoch": int(selection["selected_epoch"]),
        "selection_policy": selection["policy"],
        "physical_parameters": int(diagnostics["physical_cost"]["physical_total_parameters"]),
        "physical_validation": validation,
        "selection_validation": selection["selection_validation"],
        "search_selection_bn_calibration_batches": batches,
    }


def run_pair(study, row, reference, output, args):
    from omegaconf import OmegaConf

    study = profile_policy(study, row)
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
    search = read_search_result(search_dir, study)
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
    validate_saved_selection_bn(recovery_dir, study)
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
    floor_values = {row["theoretical_parameter_floor"] for row in rows}
    floor = next(iter(floor_values)) if len(floor_values) == 1 else None
    preflight = {
        "protocol": study["protocol"], "profiles": rows,
        "requested_min_parameters": study.get("requested_min_parameters"),
        "theoretical_parameter_floor": floor,
        "requested_min_architecturally_reachable": (
            study["requested_min_parameters"] >= floor if "requested_min_parameters" in study else None
        ),
        "total_training_epochs_per_model": 150,
        "total_training_epochs_for_study": len(rows) * 150,
        "dense_reference_training": "none; reuse immutable metadata and zero-epoch initializer",
        "test_policy": "lock all validation rows before frozen official test; exploratory comparison",
    }
    if study["protocol"] in PREBN_PROTOCOLS:
        preflight.update(
            theoretical_parameter_floors={
                str(row["min_keep_ratio"]): row["theoretical_parameter_floor"] for row in rows
            },
            target_parameters_role="historical planning labels; never used to select or reject a model",
            parameter_count_guaranteed=False,
            search_selection_policy="minimum physical parameters among feasible epochs 31-60",
            no_feasible_fallback="best validation among epochs 31-60",
            search_selection_bn_calibration_batches=0,
            selected_search_export_bn_calibration_batches=0,
            recovery_bn_calibration_batches=200,
            ordinary_training_bn_updates=True,
        )
    if study["protocol"] == PREBN_REPEAT_PROTOCOL:
        preflight["repeats"] = {
            "count": 8, "seed": 42, "initializer": "same pinned zero-epoch weights",
            "fresh_search_per_repeat": True, "shared_search_across_repeats": False,
            "scope": "same-config, same-seed reproducibility; not a multi-seed estimate",
        }
    if study["protocol"] in BN_SELECTION_PROTOCOLS:
        batches = study["search_selection_bn_calibration_batches"]
        candidates = study["search_selection_last_epochs"]
        preflight["search_selection_bn_calibration"] = {
            "predictor": "physical_after_train_bn_calibration",
            "candidate_epoch_start": SEARCH_EPOCHS - candidates + 1,
            "candidate_epoch_end": SEARCH_EPOCHS,
            "candidates_per_model": candidates,
            "train_batches_per_candidate": batches,
            "planned_train_forward_batches_per_model": candidates * batches,
            "planned_train_forward_batches_for_study": len(rows) * candidates * batches,
            "optimizer_updates": 0,
            "calibration_data": "train only; fixed seed and augmentation batches across candidates",
            "lambda_feedback_mask_changed": False,
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
        if study["protocol"] in BN_SELECTION_PROTOCOLS:
            preflight["time_budget"]["candidate_bn_overhead_minutes"] = None
            preflight["time_budget"]["duration_guaranteed"] = False
            preflight["time_budget"]["note"] = (
                "450 training minutes plus reserve is a planning estimate; "
                "30 x 200 candidate-BN forward batches per model have no measured "
                "wall-clock estimate, so completion within nine hours is not guaranteed."
            )
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
    if preflight["requested_min_architecturally_reachable"] is False:
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
