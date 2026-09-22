"""Strict adapters for shared-search compact-model comparison protocols.

The iterative protocol schema and runtime remain untouched. One-shot execution
has an explicit terminal no-feasible-search policy and no iterative rollback.
"""
from __future__ import annotations

from pathlib import Path
import json
import re

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .accuracy_guided_config import (
    ROOT, resolved_v3, validate_config_v3, validate_inputs as _validate_v3_inputs,
)

PROTOCOL = "pruning_v3_one_shot_60_90"
HANDOFF_PROTOCOL = "pruning_v3_optimizer_scheduler_handoff_60_90"
MAPPED_REPEATS_PROTOCOL = "pruning_v3_mapped_optimizer_fresh_scheduler_repeats2_60_90"
QUALITY_RECOVERY_PROTOCOL = "pruning_v3_quality_recovery_single_60_90"
CONFIG_NAME = "experiment/pruning_v3/one_shot_60_90_inherited_vs_scratch"
HANDOFF_CONFIG_NAME = "experiment/pruning_v3/optimizer_scheduler_handoff_60_90_repeats2"
MAPPED_REPEATS_CONFIG_NAME = "experiment/pruning_v3/target5m_mapped_recovery_repeats2"
QUALITY_RECOVERY_CONFIG_NAME = "experiment/pruning_v3/target5m_quality_recovery90"
OUTPUT_NAME = "one_shot_60_90_inherited_vs_scratch"
HANDOFF_OUTPUT_NAME = "optimizer_scheduler_handoff_60_90_repeats2"
MAPPED_REPEATS_OUTPUT_NAME = "target5m_mapped_recovery_repeats2"
QUALITY_RECOVERY_OUTPUT_NAME = "target5m_quality_recovery90"
MAPPED_OPTIMIZER = "mapped_adamw_moments_and_step"
FRESH_SCHEDULER = "fresh_final_stage_cosine"
RESUMED_SCHEDULER = "continue_selected_checkpoint_cosine"


def _require(condition, message):
    if not condition:
        raise ValueError(f"Invalid one-shot pruning configuration: {message}")


def dense_source_paths(source):
    """Locate reference metadata and its original initializer, without loading weights.

    Accept the original dense run tree or its self-contained adaptive copy.
    Never follow historical absolute paths or discover/select a different run.
    """
    _require(bool(str(source).strip()), "dense source must be a directory path")
    source = Path(source).expanduser().resolve()
    _require(not source.is_file(), "dense source must be a directory, not a checkpoint file")
    job = source if source.name == "J1_dense_control" or (source / "pilot_state.json").is_file() else source / "J1_dense_control"
    root = job.parent
    history, config = job / "global_history.csv", job / "resolved_config.yaml"
    bundled_history = root / "adaptive_reference_history.csv"
    bundled_config = root / "J1_dense_control_resolved.yaml"
    # Select a complete known layout, never mix partial metadata pairs silently.
    if (not history.exists() and not config.exists()
            and bundled_history.is_file() and bundled_config.is_file()):
        history, config = bundled_history, bundled_config
    return {"history_path": str(history), "state_path": str(job / "pilot_state.json"),
            "config_path": str(config), "initializer_path": str(root / "shared_random_seed42.pt")}


def compose_config(config_name=CONFIG_NAME, overrides=None, *, dense_source=None):
    name = str(config_name).removesuffix(".yaml")
    _require(re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", name) is not None,
             "config name must be a relative Hydra config path without traversal")
    source_overrides = []
    if dense_source is not None:
        for key, value in dense_source_paths(dense_source).items():
            field = ("accuracy_guided.initializer.path" if key == "initializer_path"
                     else f"accuracy_guided.reference.{key}")
            source_overrides.append(f"{field}={json.dumps(value)}")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name=name, overrides=source_overrides + list(overrides or []))


def to_v3_config(config):
    """Copy the shared engine/reference schema, never mutate the caller's config."""
    plain = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    _require(isinstance(plain, dict) and "one_shot" in plain, "one_shot mapping is required")
    one = plain.pop("one_shot")
    if one.get("protocol") in {
        HANDOFF_PROTOCOL, MAPPED_REPEATS_PROTOCOL, QUALITY_RECOVERY_PROTOCOL,
    }:
        # The shared v3 validator predates per-branch state policies and accepts
        # only its historical restart token. The one-shot adapter validates the
        # truthful public token before normalizing this private compatibility copy.
        plain["accuracy_guided"]["stage_plan"][2]["restart_policy"] = "adamw_cosine_restart"
    return OmegaConf.create(plain)


def validate_config(config):
    plain = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
    _require(isinstance(plain, dict), "root must be a mapping")
    one = plain.get("one_shot")
    _require(isinstance(one, dict), "one_shot mapping is required")
    protocol = one.get("protocol")
    if protocol == PROTOCOL:
        required_fields = {
            "protocol", "search_epochs", "final_epochs", "branches",
            "scratch_initialization", "inherited_optimizer_state",
        }
        optional_fields = {"search_scheduler_eta_min"}
        _require(required_fields <= set(one) <= required_fields | optional_fields,
                 f"one_shot unknown={sorted(set(one) - required_fields - optional_fields)}, "
                 f"missing={sorted(required_fields - set(one))}")
        _require(one["branches"] == ["inherited", "scratch"],
                 "exactly inherited and scratch branches required")
        _require(one["scratch_initialization"] == "pytorch_default_all_trainable_and_bn",
                 "scratch must reinitialize every trainable parameter and BN state")
        _require(one["inherited_optimizer_state"] == MAPPED_OPTIMIZER,
                 "inherited branch must map AdamW moments and step from the selected search checkpoint")
        if "search_scheduler_eta_min" in one:
            eta_min = one["search_scheduler_eta_min"]
            _require(type(eta_min) in (float, int) and 0 <= eta_min < plain["optimizer"]["lr"],
                     "search_scheduler_eta_min must be in [0, optimizer.lr)")
    elif protocol in {HANDOFF_PROTOCOL, MAPPED_REPEATS_PROTOCOL, QUALITY_RECOVERY_PROTOCOL}:
        fields = {
            "protocol", "search_epochs", "final_epochs",
            "search_scheduler_horizon_epochs", "execution_order", "methods", "repeats",
        }
        if protocol in {MAPPED_REPEATS_PROTOCOL, QUALITY_RECOVERY_PROTOCOL}:
            fields.add("search_scheduler_eta_min")
        if protocol == QUALITY_RECOVERY_PROTOCOL:
            fields.add("reuse_search_required")
        _require(set(one) == fields,
                 f"one_shot unknown={sorted(set(one) - fields)}, missing={sorted(fields - set(one))}")
        expected_methods = ([
                {"id": "fresh_optimizer_fresh_scheduler",
                 "model_state": "selected_surviving_state",
                 "optimizer_state": "fresh", "scheduler_state": FRESH_SCHEDULER},
                {"id": "mapped_optimizer_fresh_scheduler",
                 "model_state": "selected_surviving_state",
                 "optimizer_state": MAPPED_OPTIMIZER, "scheduler_state": FRESH_SCHEDULER},
                {"id": "mapped_optimizer_resumed_scheduler",
                 "model_state": "selected_surviving_state",
                 "optimizer_state": MAPPED_OPTIMIZER, "scheduler_state": RESUMED_SCHEDULER},
            ] if protocol == HANDOFF_PROTOCOL else [
                {"id": "fresh_optimizer_fresh_scheduler",
                 "model_state": "selected_surviving_state",
                 "optimizer_state": "fresh", "scheduler_state": FRESH_SCHEDULER},
            ] if protocol == QUALITY_RECOVERY_PROTOCOL else [
                {"id": "mapped_optimizer_fresh_scheduler",
                 "model_state": "selected_surviving_state",
                 "optimizer_state": MAPPED_OPTIMIZER, "scheduler_state": FRESH_SCHEDULER},
            ])
        _require(one["methods"] == expected_methods,
                 ("handoff methods must be the three ordered single-change policies"
                  if protocol == HANDOFF_PROTOCOL
                  else "quality recovery requires inherited compact weights with fresh AdamW and cosine"
                  if protocol == QUALITY_RECOVERY_PROTOCOL
                  else "mapped-repeat recovery requires mapped AdamW and a fresh scheduler"))
        expected_order = ("all_methods_once_then_repeats" if protocol == HANDOFF_PROTOCOL
                          else "single" if protocol == QUALITY_RECOVERY_PROTOCOL
                          else "repeat_major")
        _require(one["execution_order"] == expected_order,
                 ("execution order must finish every method before later repeats"
                  if protocol == HANDOFF_PROTOCOL
                  else "quality recovery execution order must be single"
                  if protocol == QUALITY_RECOVERY_PROTOCOL
                  else "mapped-repeat recovery execution order must be repeat_major"))
        _require(plain["accuracy_guided"]["stage_plan"][2]["restart_policy"]
                 == "branch_specific_optimizer_scheduler_handoff",
                 "final recovery restart policy must defer to the explicit method list")
        expected_repeats = ([{"id": "repeat_1", "training_seed": 42}]
                            if protocol == QUALITY_RECOVERY_PROTOCOL else [
                                {"id": "repeat_1", "training_seed": 42},
                                {"id": "repeat_2", "training_seed": 43},
                            ])
        _require(one["repeats"] == expected_repeats,
                 ("quality recovery uses exactly one seed42 branch"
                  if protocol == QUALITY_RECOVERY_PROTOCOL
                  else "the authorized paired plan uses repeat seeds 42 then 43"))
        if protocol == QUALITY_RECOVERY_PROTOCOL:
            _require(one["reuse_search_required"] is True,
                     "quality recovery must reuse the completed 60-epoch search")
        _require(type(one["search_epochs"]) is int and type(one["final_epochs"]) is int,
                 "search_epochs and final_epochs must be integers")
        expected_horizon = (one["search_epochs"] + one["final_epochs"]
                            if protocol == HANDOFF_PROTOCOL else one["search_epochs"])
        _require(one["search_scheduler_horizon_epochs"] == expected_horizon,
                 "search scheduler horizon differs from the protocol policy")
        if protocol in {MAPPED_REPEATS_PROTOCOL, QUALITY_RECOVERY_PROTOCOL}:
            eta_min = one["search_scheduler_eta_min"]
            _require(type(eta_min) in (float, int) and 0 <= eta_min < plain["optimizer"]["lr"],
                     "search_scheduler_eta_min must be in [0, optimizer.lr)")
    else:
        _require(False, "unsupported one-shot protocol")
    for field in ("search_epochs", "final_epochs"):
        _require(type(one[field]) is int and one[field] > 0, f"{field} must be a positive integer")
    shared = to_v3_config(config)
    total = validate_config_v3(shared)
    _require(one["search_epochs"] + one["final_epochs"] == total, "branch budget differs from shared stage plan")
    if not shared.accuracy_guided.smoke:
        _require((one["search_epochs"], one["final_epochs"]) == (60, 90),
                 "only the authorized full 60-search + 90-physical profile is supported")
        _require(shared.seed == shared.dataloaders.seed == shared.dataloaders.loader_seed == 42,
                 "the authorized full profile preserves seed/split 42")
    stages = shared.accuracy_guided.stage_plan
    _require(len(stages) == 3, "one-shot has one search, one export and one physical training stage")
    _require([(stage.id, stage.kind, stage.epochs) for stage in stages] == [
        ("shared_search", "search", one["search_epochs"]), ("export_only", "commit", 0),
        ("final_recovery", "recovery", one["final_epochs"])], "unexpected one-shot stage identities or lengths")
    _require(shared.accuracy_guided.guard.train_bn_calibration_batches == 0,
             "export-only diagnostics and both physical branches forbid BN calibration")
    # The base validator already checks the disabled engine recalibration/warmup,
    # quality-only controller, initial-width normalization, gates and no test access.
    return total


def resolved_branch_plan(config):
    """Expand the checked-in branch order into concrete, auditable runs."""
    validate_config(config)
    one = OmegaConf.to_container(config.one_shot, resolve=True)
    if one["protocol"] == PROTOCOL:
        return [
            {"id": "inherited", "method": "inherited", "repeat": "repeat_1",
             "training_seed": int(config.seed), "model_state": "selected_surviving_state",
             "optimizer_state": one["inherited_optimizer_state"],
             "scheduler_state": FRESH_SCHEDULER},
            {"id": "scratch", "method": "scratch", "repeat": "repeat_1",
             "training_seed": int(config.seed), "model_state": one["scratch_initialization"],
             "optimizer_state": "fresh", "scheduler_state": FRESH_SCHEDULER},
        ]
    return [
        {"id": f"{method['id']}__{repeat['id']}", "method": method["id"],
         "repeat": repeat["id"], "training_seed": repeat["training_seed"],
         "model_state": method["model_state"],
         "optimizer_state": method["optimizer_state"],
         "scheduler_state": method["scheduler_state"]}
        for repeat in one["repeats"]
        for method in one["methods"]
    ]


def _reference_origin_info(config, result):
    raw_path = config.accuracy_guided.reference.state_path
    if raw_path is None:
        return result
    path = Path(raw_path)
    if path.is_file():
        reference = json.loads(path.read_text())
        if reference.get("reference_protocol") == "one_shot_dense_reference_v1":
            result = {**result, "reference_origin": reference["reference_origin"],
                "reference_kind": reference["reference_kind"],
                "reference_training_epochs_actually_executed": reference["reference_training_epochs_actually_executed"]}
    return result


def validate_inputs(config):
    validate_config(config)
    quality_recovery = str(config.one_shot.protocol) == QUALITY_RECOVERY_PROTOCOL
    try:
        kwargs = {"allow_optimizer_lr_difference": True} if quality_recovery else {}
        return _reference_origin_info(config, _validate_v3_inputs(to_v3_config(config), **kwargs))
    except FileNotFoundError:
        paths = {**dict(config.accuracy_guided.reference),
                 "initializer_path": config.accuracy_guided.initializer.path}
        missing = {key: str(Path(value).resolve()) if value is not None else "<not configured>"
                   for key, value in paths.items() if value is None or not Path(value).is_file()}
        if not missing:
            raise
        details = "\n".join(f"  {key}: {value}" for key, value in missing.items())
        raise FileNotFoundError(
            "Blocked: required immutable reference/initializer inputs missing:\n" + details
            + "\nThese run artifacts are not distributed with Git. Pass --dense-source PATH "
              "to the existing dense run directory (containing J1_dense_control/ and "
              "shared_random_seed42.pt), or to J1_dense_control/ itself. "
              "Explicit --override accuracy_guided.reference.*=PATH and "
              "--override accuracy_guided.initializer.path=PATH take precedence.\n"
              "Locate existing artifacts from the repository root:\n"
              "  find outputs -type f \\( -name shared_random_seed42.pt -o -name pilot_state.json \\) -print\n"
              "For a clean clone, --from-scratch explicitly creates a NEW shared initializer and "
              "trains a NEW dense reference for 150 epochs before the one-shot experiment. "
              "Use --from-scratch --dry-run to preview that separate training cost. "
              "Trained dense weights cannot replace the shared zero-epoch initializer."
        ) from None


def output_paths(config, output_root=None):
    protocol = str(config.one_shot.protocol)
    output_name = (HANDOFF_OUTPUT_NAME if protocol == HANDOFF_PROTOCOL
                   else MAPPED_REPEATS_OUTPUT_NAME if protocol == MAPPED_REPEATS_PROTOCOL
                   else QUALITY_RECOVERY_OUTPUT_NAME if protocol == QUALITY_RECOVERY_PROTOCOL
                   else OUTPUT_NAME)
    root = (Path(output_root) if output_root is not None
            else Path(config.run_history.root_dir) / output_name).resolve()
    branch_ids = [branch["id"] for branch in resolved_branch_plan(config)]
    return {"root": str(root), **{name: str(root / name)
            for name in ("shared_search", "export_only", *branch_ids)}}


def resolved_one_shot(config, *, check_inputs=True, output_root=None):
    total = validate_config(config)
    quality_recovery = str(config.one_shot.protocol) == QUALITY_RECOVERY_PROTOCOL
    shared = resolved_v3(
        to_v3_config(config),
        check_inputs=check_inputs,
        allow_optimizer_lr_difference=quality_recovery,
    )
    if check_inputs:
        shared["inputs"] = _reference_origin_info(config, shared["inputs"])
    one = OmegaConf.to_container(config.one_shot, resolve=True)
    branches = resolved_branch_plan(config)
    paths = output_paths(config, output_root)
    # Do not expose the inherited iterative guard as this runner's actual policy.
    execution = {
        "shared_search_checkpoint_and_mask": True,
        "search_checkpoint_selection": "best_feasible_compact; reference fixed at shared search end",
        "no_feasible_search": "stop_before_export_and_branch_training",
        "export_diagnostics": "before_branch_training; eval/no_grad; weights_and_bn_unchanged",
        "export_diagnostic_quality": "measure opening/export jumps without calibration or training",
        "bn_calibration_batches": 0,
        "branch_plan": branches,
        "branch_gate_penalty": 0,
        "quality_failure": "record_infeasible_keep_shared_architecture; no iterative rollback",
        "iterative_recovery_guard_executed": False,
        "final_test": "separate frozen physical evaluation; never training or selection",
    }
    if one["protocol"] == PROTOCOL:
        execution.update({
            "inherited_initialization": "selected search Conv/BN tensors sliced into the compact architecture",
            "scratch_initialization": one["scratch_initialization"],
            "inherited_optimizer_state": one["inherited_optimizer_state"],
            "scratch_optimizer_state": "fresh",
            "branch_scheduler": "independent new cosine schedule for final_epochs",
        })
        handoff = {"source": "one selected adaptive search checkpoint and learned mask",
                   "inherited": execution["inherited_initialization"],
                   "scratch": execution["scratch_initialization"],
                   "gate_reentry": None, "controller_rebase": None,
                   "optimizer": {
                       "inherited": execution["inherited_optimizer_state"],
                       "scratch": execution["scratch_optimizer_state"],
                       "scheduler": execution["branch_scheduler"],
                   }}
        budget = {"shared_search_epochs_executed_once": one["search_epochs"],
                  "physical_training_epochs_per_branch": one["final_epochs"],
                  "per_branch_budget_including_shared_search": total,
                  "total_unique_training_epochs_both_branches": one["search_epochs"] + 2 * one["final_epochs"],
                  "selection_does_not_rewind_consumed_budget": True}
    else:
        continued_search_scheduler = one["protocol"] == HANDOFF_PROTOCOL
        quality_recovery = one["protocol"] == QUALITY_RECOVERY_PROTOCOL
        execution.update({
            "all_model_initialization": "selected search Conv/BN tensors sliced into one shared compact architecture",
            "search_scheduler": ((
                    f"CosineAnnealingLR(T_max={one['search_scheduler_horizon_epochs']}) so selected "
                    "checkpoint state can be continued without crossing a stage-local cosine minimum"
                ) if continued_search_scheduler else
                f"CosineAnnealingLR(T_max={one['search_scheduler_horizon_epochs']}); recovery uses a fresh cosine"),
            "comparison_axis": (
                "optimizer and scheduler state only" if continued_search_scheduler
                else "recovery learning rate; inherited compact weights, fresh optimizer and fresh scheduler"
                if quality_recovery
                else "recovery seed only; mapped optimizer and fresh scheduler fixed"
            ),
            "paired_recovery_seeds": [repeat["training_seed"] for repeat in one["repeats"]],
            "execution_order": [branch["id"] for branch in branches],
        })
        handoff = {"source": "one selected adaptive search checkpoint and learned mask",
                   "model_state": "identical selected surviving state for every branch",
                   "gate_reentry": None, "controller_rebase": None,
                   "methods": one["methods"]}
        budget = {"shared_search_epochs_executed_once": one["search_epochs"],
                  "physical_training_epochs_per_branch": one["final_epochs"],
                  "per_branch_budget_including_shared_search": total,
                  "number_of_physical_branches": len(branches),
                  "total_unique_training_epochs_all_branches": (
                      one["search_epochs"] + len(branches) * one["final_epochs"]),
                  "selection_does_not_rewind_consumed_budget": True}
    return {**shared, "protocol": one["protocol"], "shared_method_protocol": shared["protocol"],
            "stage_plan": OmegaConf.to_container(config.accuracy_guided.stage_plan, resolve=True),
            "resolved_config": OmegaConf.to_container(config, resolve=True),
            "one_shot": one, "execution_policy": execution,
            "handoff": handoff,
            "execution_graph": {"shared": [{"id": "shared_search", "epochs": one["search_epochs"]},
                                             {"id": "export_only", "epochs": 0}],
                                "physical_branches": [{**branch, "epochs": one["final_epochs"]}
                                                      for branch in branches]},
            "budget": budget,
            "output_paths": paths, "output_exists": Path(paths["root"]).exists(),
            "existing_output_policy": "refuse_overwrite_or_resume"}
