"""Separate strict adapter for the authorized shared-search inherited/scratch run.

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
CONFIG_NAME = "experiment/pruning_v3/one_shot_60_90_inherited_vs_scratch"
OUTPUT_NAME = "one_shot_60_90_inherited_vs_scratch"


def _require(condition, message):
    if not condition:
        raise ValueError(f"Invalid {PROTOCOL} configuration: {message}")


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
    plain.pop("one_shot")
    return OmegaConf.create(plain)


def validate_config(config):
    plain = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
    _require(isinstance(plain, dict), "root must be a mapping")
    one = plain.get("one_shot")
    fields = {
        "protocol", "search_epochs", "final_epochs", "branches",
        "scratch_initialization", "inherited_optimizer_state",
    }
    _require(isinstance(one, dict), "one_shot mapping is required")
    _require(set(one) == fields,
             f"one_shot unknown={sorted(set(one) - fields)}, missing={sorted(fields - set(one))}")
    _require(one["protocol"] == PROTOCOL, "unsupported one-shot protocol")
    _require(one["branches"] == ["inherited", "scratch"], "exactly inherited and scratch branches required")
    _require(one["scratch_initialization"] == "pytorch_default_all_trainable_and_bn",
             "scratch must reinitialize every trainable parameter and BN state")
    _require(one["inherited_optimizer_state"] == "mapped_adamw_moments_and_step",
             "inherited branch must map AdamW moments and step from the selected search checkpoint")
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
    try:
        return _reference_origin_info(config, _validate_v3_inputs(to_v3_config(config)))
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
    root = (Path(output_root) if output_root is not None
            else Path(config.run_history.root_dir) / OUTPUT_NAME).resolve()
    return {"root": str(root), **{name: str(root / name)
            for name in ("shared_search", "export_only", "inherited", "scratch")}}


def resolved_one_shot(config, *, check_inputs=True, output_root=None):
    total = validate_config(config)
    shared = resolved_v3(to_v3_config(config), check_inputs=check_inputs)
    if check_inputs:
        shared["inputs"] = _reference_origin_info(config, shared["inputs"])
    one = OmegaConf.to_container(config.one_shot, resolve=True)
    paths = output_paths(config, output_root)
    # Do not expose the inherited iterative guard as this runner's actual policy.
    execution = {
        "shared_search_checkpoint_and_mask": True,
        "search_checkpoint_selection": "best_feasible_compact; reference fixed at shared search end",
        "no_feasible_search": "stop_before_export_and_branch_training",
        "export_diagnostics": "before_branch_training; eval/no_grad; weights_and_bn_unchanged",
        "export_diagnostic_quality": "measure opening/export jumps without calibration or training",
        "bn_calibration_batches": 0,
        "inherited_initialization": "selected search Conv/BN tensors sliced into the compact architecture",
        "scratch_initialization": one["scratch_initialization"],
        "inherited_optimizer_state": one["inherited_optimizer_state"],
        "scratch_optimizer_state": "fresh",
        "branch_scheduler": "independent new cosine schedule for final_epochs",
        "branch_gate_penalty": 0,
        "quality_failure": "record_infeasible_keep_shared_architecture; no iterative rollback",
        "iterative_recovery_guard_executed": False,
        "final_test": "separate frozen physical evaluation; never training or selection",
    }
    return {**shared, "protocol": PROTOCOL, "shared_method_protocol": shared["protocol"],
            "resolved_config": OmegaConf.to_container(config, resolve=True),
            "one_shot": one, "execution_policy": execution,
            "handoff": {"source": "one selected adaptive search checkpoint and learned mask",
                        "inherited": execution["inherited_initialization"],
                        "scratch": execution["scratch_initialization"],
                        "gate_reentry": None, "controller_rebase": None,
                        "optimizer": {
                            "inherited": execution["inherited_optimizer_state"],
                            "scratch": execution["scratch_optimizer_state"],
                            "scheduler": execution["branch_scheduler"],
                        }},
            "execution_graph": {"shared": [{"id": "shared_search", "epochs": one["search_epochs"]},
                                             {"id": "export_only", "epochs": 0}],
                                "physical_branches": [{"id": name, "epochs": one["final_epochs"]}
                                                      for name in one["branches"]]},
            "budget": {"shared_search_epochs_executed_once": one["search_epochs"],
                       "physical_training_epochs_per_branch": one["final_epochs"],
                       "per_branch_budget_including_shared_search": total,
                       "total_unique_training_epochs_both_branches": one["search_epochs"] + 2 * one["final_epochs"],
                       "selection_does_not_rewind_consumed_budget": True},
            "output_paths": paths, "output_exists": Path(paths["root"]).exists(),
            "existing_output_policy": "refuse_overwrite_or_resume"}
