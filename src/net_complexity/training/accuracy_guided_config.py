"""Strict opt-in v3 schema and read-only input/provenance inspection.

This module never instantiates data, models, CUDA, or the training engine. The
historical pilot validators retain their independent, unchanged whitelists.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PROTOCOL = "accuracy_guided_gates_v3"
SCHEMA_VERSION = 3
ROOT = Path(__file__).resolve().parents[3]


def _require(condition, message):
    if not condition:
        raise ValueError(f"Invalid {PROTOCOL} configuration: {message}")


def _keys(value, expected, path):
    _require(isinstance(value, dict), f"{path} must be a mapping")
    _require(set(value) == set(expected),
             f"{path} missing={sorted(set(expected) - set(value))}, unknown={sorted(set(value) - set(expected))}")


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def compose_config(config_name="accuracy_guided_gates_v3", overrides=None):
    name = str(config_name).removesuffix(".yaml")
    _require(re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None, "invalid config name")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name=name, overrides=list(overrides or []))


def validate_config_v3(config):
    """Validate the launch contract without touching external data or CUDA."""
    cfg = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
    _require(isinstance(cfg, dict), "root must be a mapping")
    expected_root = {"seed", "device", "model", "optimizer", "scheduler", "dataloaders",
                     "training_arguments", "metrics", "accuracy_guided", "mlflow", "run_history"}
    _keys(cfg, expected_root, "root")
    p = cfg["accuracy_guided"]
    _keys(p, {"protocol", "smoke", "total_epochs", "normalization", "scaling_contract", "drop_mode",
              "carry_policy", "eligibility", "selection", "guard", "reference", "initializer", "stage_plan"},
          "accuracy_guided")
    _require(p["protocol"] == PROTOCOL, "protocol version must be explicit")
    _require(type(p["smoke"]) is bool, "smoke must be boolean")
    _require(type(p["total_epochs"]) is int and 0 < p["total_epochs"] <= 150, "invalid epoch budget")
    _require(p["smoke"] or p["total_epochs"] == 150, "a full new run must allocate 150 epochs")
    _require(not p["smoke"] or p["total_epochs"] < 150, "smoke must have a short separate budget")
    for key, expected in {"normalization": "initial_channels", "scaling_contract": "survivor_equivalent_v1",
                          "drop_mode": "learned_closed_gates", "carry_policy": "carry"}.items():
        _require(p[key] == expected, f"{key} must be {expected}")
    _keys(p["eligibility"], {"min_keep_ratio"}, "eligibility")
    ratio = p["eligibility"]["min_keep_ratio"]
    _require(_number(ratio) and 0 < ratio <= 1, "min_keep_ratio must be in (0,1]")
    _keys(p["selection"], {"policy", "reference", "objective", "tie_break"}, "selection")
    _require(p["selection"] == {"policy": "best_feasible_compact", "reference": "stage_end",
             "objective": "physical_total_parameters", "tie_break": ["validation_accuracy", "ce_loss", "earlier_epoch"]},
             "unsupported checkpoint selection contract")
    _keys(p["guard"], {"policy", "reference", "train_bn_calibration_batches", "on_reject"}, "guard")
    _require(p["guard"]["policy"] == "recover_then_quality_guard" and p["guard"]["reference"] == "stage_end"
             and p["guard"]["on_reject"] == "rollback_stop_commits_ungated_fallback", "unsupported quality guard")
    batches = p["guard"]["train_bn_calibration_batches"]
    _require(type(batches) is int and batches >= 0, "calibration batches must be a nonnegative integer")
    _keys(p["reference"], {"history_path", "state_path", "config_path"}, "reference")
    _keys(p["initializer"], {"path", "required_trained_epochs"}, "initializer")
    _require(type(p["initializer"]["required_trained_epochs"]) is int
             and p["initializer"]["required_trained_epochs"] == 0, "trained initializer forbidden")
    for path, value in [*p["reference"].items(), ("initializer.path", p["initializer"]["path"])]:
        _require((p["smoke"] and value is None) or (isinstance(value, str) and bool(value.strip())),
                 f"{path} must be an explicit path")
    stages = p["stage_plan"]
    _require(isinstance(stages, list) and bool(stages) and len(stages) % 3 == 0,
             "stage_plan must contain search -> commit -> recovery groups")
    ids = set()
    for index, stage in enumerate(stages):
        _keys(stage, {"id", "kind", "epochs", "commit_allowed", "selected_checkpoint", "restart_policy"}, "stage")
        _require(isinstance(stage["id"], str) and re.fullmatch(r"[A-Za-z0-9_-]+", stage["id"])
                 and stage["id"] not in ids, "stage ids must be distinct safe names")
        ids.add(stage["id"])
        expected_kind = ("search", "commit", "recovery")[index % 3]
        _require(stage["kind"] == expected_kind, "stage order must be search -> commit -> recovery")
        commit = expected_kind == "commit"
        _require(type(stage["epochs"]) is int and (stage["epochs"] == 0 if commit else stage["epochs"] > 0),
                 "only commit stages have zero epochs")
        _require(type(stage["commit_allowed"]) is bool and stage["commit_allowed"] == commit,
                 "only commit stages may commit")
        _require(stage["restart_policy"] == ("none" if commit else "adamw_cosine_restart"),
                 "stage-wise AdamW/cosine restart must be preserved")
        _require(stage["selected_checkpoint"] is None, "launch plan must not supply selected checkpoints")
    total = sum(stage["epochs"] for stage in stages)
    _require(total == p["total_epochs"], f"stage total {total} differs from budget {p['total_epochs']}")
    t = cfg["training_arguments"]
    _keys(t, {"num_epochs", "evaluate_test", "global_epoch_offset", "audit_data_seed", "lambda_warmup",
              "batchnorm_recalibration", "adaptive_lambda"}, "training_arguments")
    _require(t["num_epochs"] == total and type(t["num_epochs"]) is int, "training budget differs")
    _require(t["evaluate_test"] is False and cfg["dataloaders"].get("include_test") is False
             and cfg["metrics"].get("test_metrics") == [], "test data must be inaccessible during training")
    _require(t["lambda_warmup"] == {"enabled": False} and t["batchnorm_recalibration"] == {"enabled": False},
             "hidden warmup or engine calibration forbidden")
    _require(t["global_epoch_offset"] == 0, "a new run starts at global epoch zero")
    a = t["adaptive_lambda"]
    _keys(a, {"enabled", "control_mode", "alpha_init", "alpha_min", "alpha_max", "soft_drop", "hard_drop",
              "gap_window", "update_every_search_epochs", "initial_search_warmup", "reentry_samples", "log_step"},
          "adaptive_lambda (legacy rate boost, targets and open-bias recovery are unsupported)")
    _require(a["enabled"] is True and a["control_mode"] == "accuracy_only", "adaptive accuracy-only controller required")
    _require(all(_number(a[k]) for k in ("alpha_init", "alpha_min", "alpha_max", "soft_drop", "hard_drop"))
             and (_number(a["log_step"]) or a["log_step"] == "auto"),
             "controller scalars must be finite numbers")
    _require(0 < a["alpha_min"] <= a["alpha_init"] <= a["alpha_max"], "invalid alpha bounds")
    _require(0 <= a["soft_drop"] < a["hard_drop"] <= 1
             and (a["log_step"] == "auto" or a["log_step"] > 0),
             "invalid accuracy drops/log step")
    for key in ("gap_window", "update_every_search_epochs", "reentry_samples", "initial_search_warmup"):
        _require(type(a[key]) is int and a[key] >= (0 if key == "initial_search_warmup" else 1),
                 f"{key} must be an integer in its valid range")
    _require(a["reentry_samples"] >= a["gap_window"], "reentry samples must fill the feedback window")
    block = cfg["model"]["backbone"]["resnet_block"]
    _require(block.get("regularization_normalization") == p["normalization"], "runtime gate normalization differs")
    _require(block.get("train_gate_mode") == "ste_hard" and block.get("eval_gate_mode") == "deterministic_hard",
             "real hard forward is required")
    _require(not any(block.get(k) for k in ("force_ones_mask", "deterministic_soft_mask", "deterministic_hard_mask")),
             "legacy gate mode overrides conflict with explicit runtime modes")
    _require(_number(block.get("gate_threshold")) and 0 <= block["gate_threshold"] <= 1, "invalid hard threshold")
    _require(cfg["model"].get("lambda_coef") == a["alpha_init"], "model alpha differs from controller alpha_init")
    _require(cfg["model"].get("entropy_regularization_coef") == 0.0, "base protocol entropy coefficient must be zero")
    _require(cfg["optimizer"].get("_target_") == "torch.optim.AdamW"
             and cfg["optimizer"].get("gate_weight_decay_scale") == 0.0, "AdamW with zero gate decay is required")
    _require(cfg["scheduler"].get("_target_") == "torch.optim.lr_scheduler.CosineAnnealingLR", "cosine scheduler required")
    _require(type(cfg["seed"]) is int and t["audit_data_seed"] == cfg["dataloaders"].get("seed")
             == cfg["dataloaders"].get("loader_seed"), "explicit, consistent split and loader seeds required")
    return total


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_curve(path, total):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        _require({"global_epoch", "valid_accuracy"} <= set(reader.fieldnames or []), "reference requires validation columns")
        rows = {}
        for row in reader:
            epoch, value = int(row["global_epoch"]), float(row["valid_accuracy"])
            _require(epoch > 0 and epoch not in rows and math.isfinite(value) and 0 <= value <= 1,
                     "invalid reference epoch/validation accuracy")
            rows[epoch] = value
    _require(set(range(1, total + 1)) <= set(rows), "reference does not cover the full budget; extrapolation forbidden")
    return rows


def inspect_inputs(config, *, load_initializer=True):
    """Report missing paths; validate existing inputs without changing any file."""
    total = validate_config_v3(config)
    cfg = OmegaConf.to_container(config, resolve=True)
    p = cfg["accuracy_guided"]
    paths = {**p["reference"], "initializer_path": p["initializer"]["path"]}
    missing = [key for key, value in paths.items() if value is None or not Path(value).is_file()]
    result = {"status": "blocked_missing_inputs" if missing else "ready", "missing_inputs": missing,
              "paths": {key: str(Path(value).resolve()) if value else None for key, value in paths.items()},
              "sha256": {key: file_hash(value) for key, value in paths.items() if value and Path(value).is_file()},
              "reference_split": "validation", "reference_training_cost": "external; not charged to this new model"}
    if "history_path" not in missing:
        curve = _reference_curve(paths["history_path"], total)
        result["reference_epochs"] = len(curve)
    state = None
    if "state_path" not in missing:
        state = json.loads(Path(paths["state_path"]).read_text())
        _require(state.get("status") == "completed" and not state.get("accepted_mask")
                 and state.get("test_evaluated") is False, "reference must be completed dense validation-only training")
        _require(state.get("global_epochs_completed") == state.get("total_epochs_allocated")
                 and state["global_epochs_completed"] >= total, "reference budget is incomplete")
        _require(state.get("seed") == cfg["seed"], "reference seed differs")
        _require(p["smoke"] or state.get("synthetic_reference") is not True,
                 "a full profile cannot use synthetic reference feedback")
        if state.get("synthetic_reference") is True:
            result.update(reference_kind="synthetic_programmed_feedback",
                          reference_training_cost=state.get("reference_training_epochs_actually_executed"),
                          reference_quality_claim=False)
        if not p["smoke"]:
            _require(state.get("validation", {}).get("example_count") == 5000, "reference validation split differs")
            if "history_path" not in missing:
                _require(set(curve) == set(range(1, state["global_epochs_completed"] + 1)),
                         "reference curve must contain every dense training epoch exactly once")
                _require(all(abs(value * 5000 - round(value * 5000)) < 1e-8 for value in curve.values()),
                         "reference accuracy is not consistent with weighted validation counts")
    if "config_path" not in missing:
        source = OmegaConf.load(paths["config_path"])
        for key in ("dataloaders.train_val_ratio", "dataloaders.seed", "dataloaders.loader_seed",
                    "dataloaders.batch_size", "dataloaders.taskname", "dataloaders._target_", "optimizer",
                    "scheduler", "model.backbone.num_classes", "model.backbone._target_",
                    "model.backbone.stem_kernel_size", "model.backbone.stem_stride", "model.backbone.stem_padding",
                    "model.backbone.use_maxpool"):
            _require(OmegaConf.select(source, key) == OmegaConf.select(config, key), f"reference compatibility differs: {key}")
        _require(any(OmegaConf.select(metric, "_target_") == "net_complexity.metrics.classification.Accuracy"
                     and OmegaConf.select(metric, "return_counts") is True for metric in source.metrics.valid_metrics),
                 "reference accuracy must be sample weighted")
    if "initializer_path" not in missing and load_initializer:
        import torch
        from .pruning_measurement import state_hash
        initial = torch.load(paths["initializer_path"], map_location="cpu", weights_only=True)
        _require(initial.get("trained_epochs") == 0 and initial.get("seed") == cfg["seed"],
                 "initializer must be verified zero-epoch weights with matching seed")
        value_hash = state_hash(initial["model_state_dict"])
        _require(value_hash == initial.get("model_state_hash"), "initializer tensor hash differs")
        if state:
            _require(value_hash == state.get("common_init_hash"), "reference zero-epoch initializer differs")
        result["initializer_trained_epochs"] = 0
        result["initializer_model_state_hash"] = value_hash
    if p["smoke"] and missing:
        result["note"] = "Synthetic fixture must supply its own immutable validation reference and zero-epoch initializer."
    return result


def code_provenance(root=ROOT):
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout
    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        diff = git("diff", "HEAD", "--binary")
        # git diff excludes newly authored files: include their hashes explicitly.
        untracked = git("ls-files", "--others", "--exclude-standard").decode().splitlines()
        hashes = {path: file_hash(Path(root) / path) for path in untracked if (Path(root) / path).is_file()}
        return {"commit": commit, "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
                "dirty": bool(diff or hashes), "untracked_file_sha256": hashes}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_diff_sha256": None, "status": "vcs_metadata_unavailable"}


def validate_inputs(config):
    """Training preflight: missing/incompatible reference never triggers training."""
    result = inspect_inputs(config)
    if result["missing_inputs"]:
        raise FileNotFoundError(f"Blocked: required immutable reference/initializer inputs missing: {result['missing_inputs']}")
    return result


def resolved_v3(config, *, check_inputs=True):
    total = validate_config_v3(config)
    cfg = OmegaConf.to_container(config, resolve=True)
    p, a = cfg["accuracy_guided"], cfg["training_arguments"]["adaptive_lambda"]
    return {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "resolved_config": cfg,
            "stage_plan": p["stage_plan"], "total_training_epochs": total,
            "training_performed": False, "evaluate_test": False, "cuda_initialized": False,
            "alpha_policy": {**a, "feedback": "mean of paired reference(global epoch) - validation accuracy",
                              "updates": "increase / hold / decrease by fixed log_step using quality only",
                              "recovery": "alpha held; no search-clock or feedback-window advance"},
            "loss_contract": {"normalization": p["normalization"], "scaling_contract": p["scaling_contract"],
                              "formula": "alpha / M0 * sum_b sum_j(m_bj * p_raw_bj) / n_b0",
                              "effective_lambda_b": "alpha * n_bt / n_b0", "second_alpha_decay": False},
            "pruning": {"drop_mode": p["drop_mode"], "mandatory_quota": None,
                        "eligibility": p["eligibility"], "no_candidates": "legal no-op; continue recovery/plan"},
            "handoff": {"gate_policy": "carry surviving raw logits", "backbone": "selected physical Conv/BN weights",
                        "optimizer": "new stage optimizer/scheduler; weights preserved, optimizer state restarted",
                        "rebase": "preserve alpha and consumed clocks, clear paired-gap window once per transition"},
            "clocks": ["global_training_epoch", "search_epochs_consumed", "local_search_epoch"],
            "conflicting_legacy_options": [],
            "inputs": inspect_inputs(config) if check_inputs else {"status": "not_inspected"},
            "code": code_provenance()}
