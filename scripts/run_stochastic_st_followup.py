"""Three existing-search recoveries, then two new 60+90 stochastic-ST models."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from launch_pruning_ablations import inspect_search
from pruning_recovery_matrix import SOURCE_FILES, canonical, digest, object_hash, read_json, require, write_json
from net_complexity.training.one_shot_pruning_config import compose_config, validate_config, validate_inputs

DEFAULT_CONFIG = ROOT / "configs/experiment/pruning_v3/stochastic_st_followup5.yaml"
PROTOCOL = "pruning_v3_stochastic_st_followup5_v1"
FULL4_PROTOCOL = "pruning_v3_stochastic_st_full4_v1"
FULL2_PROTOCOL = "pruning_v3_stochastic_st_full2_v1"


def load_study(path):
    study = yaml.safe_load(Path(path).read_text())
    fields = {"protocol", "config", "reference_output", "output", "seed", "recoveries", "searches",
              "reference_sha256", "initializer_model_state_hash", "split_indices_hash"}
    fresh = study.get("protocol") in {FULL4_PROTOCOL, FULL2_PROTOCOL}
    require(set(study) == fields | ({"checkpoint_retention"} if fresh else set()), "Invalid followup fields")
    require(study["protocol"] in {PROTOCOL, FULL4_PROTOCOL, FULL2_PROTOCOL} and study["seed"] == 42
            and study["config"] == "experiment/pruning_v3/stochastic_st_inherited", "Invalid followup recipe")
    counts = {PROTOCOL: (3, 2), FULL4_PROTOCOL: (0, 4), FULL2_PROTOCOL: (0, 2)}
    require((len(study["recoveries"]), len(study["searches"])) == counts[study["protocol"]], "Unexpected number of models")
    if fresh:
        require(study["checkpoint_retention"] == "online_validation_winner_best_last", "Fresh runs require bounded checkpoint storage")
    ids = []
    for row in study["recoveries"] + study["searches"]:
        fields = {"id", "soft_drop", "hard_drop"}
        if row in study["recoveries"]:
            fields.add("source")
        pins = {"expected_parameters", "selected_epoch", "selected_checkpoint_sha256_prefix"}
        require(set(row) in (fields, fields | pins) and ("source" in row or not set(row) & pins), "Unknown row settings")
        require(re.fullmatch(r"[A-Za-z0-9_-]+", row["id"]), "Invalid job ID")
        require(type(row["soft_drop"]) in (int, float) and type(row["hard_drop"]) in (int, float)
                and 0 <= row["soft_drop"] < row["hard_drop"] <= 1, "Invalid accuracy thresholds")
        if "source" in row:
            source = Path(row["source"])
            require(not source.is_absolute() and ".." not in source.parts and source.parts[:2] == ("outputs", "runs")
                    and source.parts[-2:] == ("internal", "search"), "Invalid immutable source path")
        if pins <= set(row):
            require(type(row["expected_parameters"]) is int and row["expected_parameters"] > 0
                    and 31 <= row["selected_epoch"] <= 60
                    and re.fullmatch(r"[a-f0-9]{16}", row["selected_checkpoint_sha256_prefix"]), "Invalid measured source pin")
        ids.append(row["id"])
    require(len(set(ids)) == len(ids), "Duplicate job IDs")
    require(set(study["reference_sha256"]) == {"history_path", "state_path", "config_path", "initializer_path"}, "Reference files must be pinned")
    for value in [*study["reference_sha256"].values(), study["initializer_model_state_hash"], study["split_indices_hash"]]:
        require(re.fullmatch(r"[a-f0-9]{64}", value), "Invalid reference checksum")
    return study


def configured(study, row, args):
    reference = Path(args.dense_source or ROOT / study["reference_output"]).resolve()
    config = compose_config(study["config"], [
        f"training_arguments.adaptive_lambda.soft_drop={row['soft_drop']}",
        f"training_arguments.adaptive_lambda.hard_drop={row['hard_drop']}",
    ], dense_source=reference)
    if args.device:
        config.device = args.device
    validate_config(config)
    require(list(config.one_shot.branches) == ["inherited"], "Only the inherited main-method branch is allowed")
    return config


def review_source(source, config, study, row, *, from_old_queue):
    """Check all saved identities and raw tensors; no dataset or model forward."""
    import torch
    from net_complexity.training.pruning_measurement import mask_hash, state_hash
    saved = OmegaConf.load(source / "resolved_config.yaml")
    validate_config(saved)
    expected = deepcopy(config)
    if from_old_queue:
        expected.one_shot.branches = ["inherited", "scratch", "random"]
    require(canonical(saved) == canonical(expected), "Source model/training settings differ: " + str(source))
    review = inspect_search(source, saved, include_provenance=True)
    require(review["reference_sha256"] == study["reference_sha256"]
            and review["initializer_model_state_hash"] == study["initializer_model_state_hash"]
            and review["split_indices_hash"] == study["split_indices_hash"], "Source reference/initializer/split changed")
    selection = read_json(source / "selection.json")
    require(mask_hash(selection["pruning_mask"]) == review["mask_hash"], "Saved mask contents differ")
    if "expected_parameters" in row:
        require(review["physical_parameters"] == row["expected_parameters"] and review["selected_epoch"] == row["selected_epoch"]
                and review["files"]["selected_checkpoint.pt"].startswith(row["selected_checkpoint_sha256_prefix"]),
                "Measured search identity differs: " + row["id"])
    payload = torch.load(source / "selected_checkpoint.pt", map_location="cpu", weights_only=True)
    require(payload["epoch"] == review["selected_epoch"]
            and state_hash(payload["model_state_dict"]) == review["selected_model_state_hash"] == payload["model_state_hash"],
            "Selected checkpoint tensors/epoch differ")
    require(all(not t.is_floating_point() or (t.dtype == torch.float32 and bool(torch.isfinite(t).all()))
                for t in payload["model_state_dict"].values()), "Expected finite FP32 source tensors")
    return review


def check_completed(path, config, review):
    state = read_json(path / "one_shot_state.json")
    require(state["status"] == "completed" and state["test_evaluated"] is False,
            f"Partial recovery retained; refusing another 90 epochs: {path}")
    require(canonical(OmegaConf.load(path / "resolved_config.yaml")) == canonical(config), "Completed recipe differs")
    require(set(state["branches"]) == {"inherited"}, "Exactly one inherited branch required")
    branch = state["branches"]["inherited"]
    require(branch["status"] in {"completed", "infeasible"} and branch["ledger"]["global_training_epoch"] == 150
            and branch["ledger"]["search_epochs_consumed"] == 60 and branch["final_training_epochs_executed"] == 90,
            "Incomplete 150-epoch training ledger")
    require(branch["optimizer_state_initialization"] == "fresh"
            and branch["scheduler_state_initialization"] == "fresh_final_stage_cosine"
            and branch["bn_calibration"]["batches"] == 200, "Recovery initialization/calibration changed")
    stage = state["stages"]["inherited"]["stage_metadata"]
    require(stage["kind"] == "physical_final" and stage["held_controller_state"] is not None,
            "Physical recovery lost the held adaptive-controller state")
    selection = read_json(path / "selection.json")
    require(selection["mask_hash"] == review["mask_hash"] and selection["selected_epoch"] == review["selected_epoch"]
            and digest(path / "selected_checkpoint.pt") == review["files"]["selected_checkpoint.pt"], "Recovery reselected the source")
    require(branch["final_cost"]["physical_total_parameters"] == review["physical_parameters"], "Physical architecture size changed")
    require(read_json(path / "inherited/branch_state.json") == branch
            and (path / "inherited/deployment.pt").is_file() and (path / "inherited/initial_state.pt").is_file(), "Incomplete deployment files")
    return branch


def preflight(study, args):
    output = Path(args.output or ROOT / study["output"]).resolve()
    cells, missing = [], []
    for row in study["recoveries"] + study["searches"]:
        existing = "source" in row
        source = (ROOT / row["source"]).resolve() if existing else output / row["id"] / "internal/search"
        if existing:
            require(output != source and output not in source.parents and source not in output.parents, "Output overlaps original source")
            absent = [name for name in SOURCE_FILES if not (source / name).is_file()]
            if absent:
                missing.append(f"{row['id']}: {source}; missing: {', '.join(absent)}")
        cells.append({"row": row, "existing": existing, "source": source,
                      "output": output / row["id"] / "internal/recovery", "config": configured(study, row, args)})
    if missing:
        raise FileNotFoundError("Finish all three original searches before followup training:\n" + "\n".join(missing)
                                + "\nNo new search or recovery has started.")
    inputs = validate_inputs(cells[0]["config"])
    require(inputs["sha256"] == study["reference_sha256"]
            and inputs["initializer_model_state_hash"] == study["initializer_model_state_hash"], "Immutable reference inputs changed")
    for cell in cells:
        cell["review"] = review_source(cell["source"], cell["config"], study, cell["row"],
                                       from_old_queue=cell["existing"]) if cell["source"].exists() else None
        cell["completed"] = check_completed(cell["output"], cell["config"], cell["review"]) if cell["output"].exists() else None
    return output, cells


def matches_original_search(argv):
    """Match the Python script arguments, not shell command text containing them."""
    script = next((i for i, value in enumerate(argv) if Path(value).name == "launch_pruning_ablations.py"), None)
    if script is None:
        return False
    values = argv[script + 1:]
    def option(name):
        for i, value in enumerate(values):
            if value == name and i + 1 < len(values):
                return values[i + 1]
            if value.startswith(name + "="):
                return value.split("=", 1)[1]
        return ""
    return (Path(option("--config")).name == "ablations_stochastic_st4.yaml"
            and option("--stage") == "search" and "--dry-run" not in values)


def running_original_searches():
    proc = Path("/proc")
    if proc.is_dir():
        found = []
        for entry in proc.iterdir():
            if not entry.name.isdecimal() or int(entry.name) == os.getpid():
                continue
            try:
                if entry.stat().st_uid != os.getuid():
                    continue
                argv = [value.decode(errors="replace") for value in (entry / "cmdline").read_bytes().split(b"\0") if value]
                if matches_original_search(argv):
                    found.append(int(entry.name))
            except (OSError, ProcessLookupError):
                continue  # A process may exit between listing and reading it.
        return found
    pattern = r"[l]aunch_pruning_ablations\.py .*ablations_stochastic_st4\.yaml .*--stage[ =]+search([[:space:]]|$)"
    result = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", pattern], capture_output=True, text=True)
    require(result.returncode in {0, 1}, "Cannot check original search process; wait manually and use --skip-wait")
    return [int(value) for value in result.stdout.split() if int(value) != os.getpid()]


def wait_for_original_searches():
    while True:
        pids = running_original_searches()
        if not pids:
            return
        print(f"Waiting for original stochastic-ST search queue, PID(s) {pids}; no new training started", flush=True)
        time.sleep(15)


@contextmanager
def queue_lock(output):
    path = output / "followup.lock"
    with path.open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    try:
        yield
    finally:
        path.unlink()


def run_searches(study, cells, output, bounded):
    """Finish only the new searches; retain each selected source for later recovery."""
    from net_complexity.training.one_shot_pruning import run_one_shot_pruning
    from plot_pruning_channels import channel_rows
    state = {"status": "searching", "completed_searches": [], "planned_searches": len(cells),
             "new_training_epochs_allocated": len(cells) * 60, "test_evaluated": False,
             "recovery_epochs_executed": 0, "remaining_training_epochs_per_model": 90}
    summaries = []
    try:
        write_json(output / "search_state.json", state)
        for cell in cells:
            row, config = cell["row"], cell["config"]
            if cell["review"] is None:
                print(f"[search-only] {row['id']}: new stochastic-ST search60; recovery deferred", flush=True)
                run_one_shot_pruning(config, cell["source"], search_only=True,
                                     keep_epoch_checkpoints=False, bounded_checkpoints=bounded)
            else:
                print(f"[search-only] {row['id']}: reusing completed search; no training repeated", flush=True)
            review = review_source(cell["source"], config, study, row, from_old_queue=False)
            require(cell["review"] is None or review == cell["review"], "Search changed after preflight")
            cell["review"] = review
            selection = read_json(cell["source"] / "selection.json")
            channels = channel_rows(selection["pruning_mask"])
            summary = {"id": row["id"], "soft_drop": row["soft_drop"], "hard_drop": row["hard_drop"],
                       "selected_epoch": review["selected_epoch"], "physical_parameters": review["physical_parameters"],
                       "remaining_internal_channels": sum(r["preserved"] for r in channels),
                       "original_internal_channels": sum(r["original"] for r in channels),
                       "search_validation_accuracy": review["validation_accuracy"],
                       "fallback_used": review["fallback_used"]}
            summaries.append(summary)
            write_json(output / "search_summary.json", {"test_evaluated": False, "searches": summaries})
            with (output / "search_summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary))
                writer.writeheader()
                writer.writerows(summaries)
            print(f"[search-summary] {row['id']}: parameters={summary['physical_parameters']:,}; "
                  f"channels_remaining={summary['remaining_internal_channels']}/{summary['original_internal_channels']}; "
                  f"search_validation={summary['search_validation_accuracy']:.2%}; fallback={summary['fallback_used']}", flush=True)
            state["completed_searches"].append(row["id"])
            write_json(output / "search_state.json", state)
        state["status"] = "search_completed"
        write_json(output / "search_state.json", state)
    except BaseException as exc:
        state.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "search_state.json", state)
        raise
    print("\n| ID | soft | hard | Parameters | Channels retained | Search validation | Fallback |", flush=True)
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |", flush=True)
    for row in summaries:
        print(f"| {row['id']} | {row['soft_drop']} | {row['hard_drop']} | {row['physical_parameters']:,} | "
              f"{row['remaining_internal_channels']}/{row['original_internal_channels']} | "
              f"{row['search_validation_accuracy']:.2%} | {row['fallback_used']} |", flush=True)
    print(f"Search complete: {len(cells)}/{len(cells)}; no recovery or official test started. Summary: {output / 'search_summary.csv'}", flush=True)
    return state


def execute(study, args):
    rows = study["recoveries"] + study["searches"]
    model_count = len(rows)
    search_only = args.stage == "search"
    require(not search_only or (study["protocol"] in {FULL4_PROTOCOL, FULL2_PROTOCOL} and not study["recoveries"]),
            "Search-only requires a fresh two- or four-model study without existing-source recoveries")
    new_epochs = len(study["searches"]) * 60 if search_only else len(study["recoveries"]) * 90 + len(study["searches"]) * 150
    bounded = study.get("checkpoint_retention") == "online_validation_winner_best_last"
    configs = [configured(study, row, args) for row in rows]
    block = configs[0].model.backbone.resnet_block
    plan = {"stage": args.stage, "models": model_count, "existing_search_recoveries": len(study["recoveries"]), "new_searches": len(study["searches"]),
            "recovery_branches_per_model": [] if search_only else ["inherited"],
            "new_training_epochs": new_epochs if args.stage in {"run", "search"} else 0,
            "bounded_epoch_checkpoints": bounded,
            "per_model_total_epochs": 150, "test_after_each_model": not search_only, "wait_for_original_search_process": not args.skip_wait,
            "search_gate_modes": {key: block[key] for key in ("train_gate_mode", "eval_gate_mode", "beta", "temperature")},
            "models_in_order": [{"id": row["id"], "new_search_epochs": 0 if "source" in row else 60,
                                  "recovery_epochs": 0 if search_only else 90,
                                  "soft_drop": row["soft_drop"], "hard_drop": row["hard_drop"]} for row in rows]}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    output = Path(args.output or ROOT / study["output"]).resolve()
    if args.stage == "test":
        from stochastic_pruning_test import DeploymentTests
        tester = DeploymentTests(output, [row["id"] for row in rows], args.device, branches=("inherited",))
        targets = [output / row["id"] / "internal/recovery" for row in rows]
        for target in targets:
            require(read_json(target / "one_shot_state.json")["status"] == "completed", "All fits must finish before standalone test")
        for target in targets:
            tester.existing(target)
        return tester.finish()
    if args.stage in {"run", "search"} and not args.skip_wait:
        wait_for_original_searches()
    output, cells = preflight(study, args)
    source_review = {cell["row"]["id"]: cell["review"] for cell in cells if cell["existing"]}
    for name, review in source_review.items():
        print(f"[source] {name}: selected_epoch={review['selected_epoch']}; "
              f"parameters={review['physical_parameters']:,}; physical pre-BN validation={review['validation_accuracy']:.2%}; "
              f"fallback={review['fallback_used']}", flush=True)
    if args.stage == "inspect":
        print(json.dumps(source_review, indent=2))
        return source_review
    output.mkdir(parents=True, exist_ok=True)
    with queue_lock(output):
        manifest, receipt = output / "followup_study.json", output / "source_review.json"
        require(not manifest.exists() or object_hash(read_json(manifest)) == object_hash(study), "Followup settings changed")
        require(not receipt.exists() or read_json(receipt) == source_review, "Original sources changed after preflight")
        write_json(manifest, study)
        write_json(receipt, source_review)
        if search_only:
            return run_searches(study, cells, output, bounded)
        from stochastic_pruning_test import DeploymentTests
        tester = DeploymentTests(output, [row["id"] for row in rows], args.device, branches=("inherited",))
        state = {"status": "running", "completed_models": [], "planned_models": model_count, "new_training_epochs_allocated": new_epochs}
        write_json(output / "followup_state.json", state)
        from net_complexity.training.one_shot_pruning import run_one_shot_pruning
        try:
            for cell in cells:
                row, config = cell["row"], cell["config"]
                if cell["completed"] is not None:
                    print(f"Reusing completed recovery: {row['id']}", flush=True)
                    tester.existing(cell["output"])
                else:
                    if cell["review"] is None:
                        print(f"[followup] {row['id']}: new stochastic-ST search60", flush=True)
                        run_one_shot_pruning(config, cell["source"], search_only=True,
                                             **({"bounded_checkpoints": True, "keep_epoch_checkpoints": False} if bounded else {}))
                        cell["review"] = review_source(cell["source"], config, study, row, from_old_queue=False)
                    else:
                        require(review_source(cell["source"], config, study, row, from_old_queue=cell["existing"])
                                == cell["review"], "Source changed after all-source preflight")
                    print(f"[followup] {row['id']}: inherited recovery90, parameters={cell['review']['physical_parameters']:,}; no search rerun", flush=True)
                    run_one_shot_pruning(config, cell["output"], reuse_search_from=cell["source"],
                                         keep_epoch_checkpoints=args.keep_epoch_checkpoints,
                                         frozen_deployment_callback=tester.callback(config),
                                         **({"bounded_checkpoints": True} if bounded else {}))
                    check_completed(cell["output"], config, cell["review"])
                state["completed_models"].append(row["id"])
                write_json(output / "followup_state.json", state)
            state["status"] = "training_completed"
            write_json(output / "followup_state.json", state)
        except BaseException as exc:
            state.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
            write_json(output / "followup_state.json", state)
            raise
        tester.finish()
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("run", "search", "inspect", "test"), default="run")
    parser.add_argument("--dense-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-wait", action="store_true", help="Only if the original search process has already exited")
    parser.add_argument("--keep-epoch-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    try:
        return execute(load_study(args.config), args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.exit(2, f"Followup blocked: {exc}\n")


if __name__ == "__main__":
    main()
