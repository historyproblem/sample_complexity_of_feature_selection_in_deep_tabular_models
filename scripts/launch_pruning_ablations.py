"""Search -> inspect -> recover; never train a reference or evaluate official test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_CONFIG = ROOT / "configs/experiment/pruning_v3/ablations_grid.yaml"
PROTOCOL = "pruning_v3_size_ablations_grid_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_study(path):
    study = yaml.safe_load(Path(path).read_text())
    require(set(study) == {"protocol", "seed", "split_seed", "reference_output", "output",
                           "configs", "pilot_profiles", "profiles"}, "Unexpected study fields")
    require(study["protocol"] == PROTOCOL and study["seed"] == study["split_seed"] == 42,
            "Expected the single-seed42 size-ablation study")
    require(study["configs"] == {"internal": "experiment/pruning_v3/ablations_internal",
                                  "output": "experiment/pruning_v3/ablations_output"},
            "Study must use the checked size-ablation recipes")
    ids = []
    require(len(study["profiles"]) == 5, "The full study has five tolerance profiles")
    for row in study["profiles"]:
        require(set(row) == {"id", "soft_drop", "hard_drop"}, "Unexpected profile fields")
        require(isinstance(row["id"], str) and re.fullmatch(r"[a-zA-Z0-9_-]+", row["id"]), "Invalid profile id")
        require(type(row["soft_drop"]) in (int, float) and type(row["hard_drop"]) in (int, float)
                and 0 <= row["soft_drop"] < row["hard_drop"] <= 1, "Invalid accuracy tolerances")
        ids.append(row["id"])
    require(len(set(ids)) == 5, "Duplicate profile ids")
    pilot = study["pilot_profiles"]
    require(len(pilot) == len(set(pilot)) == 3 and set(pilot) <= set(ids), "Pilot must name three profiles")
    return study


def compose_cell(study, row, placement, reference, device=None):
    from net_complexity.training.one_shot_pruning_config import compose_config, validate_config
    overrides = [f"training_arguments.adaptive_lambda.{key}={row[key]}"
                 for key in ("soft_drop", "hard_drop")]
    if device:
        overrides.append(f"device={device}")
    cfg = compose_config(study["configs"][placement], overrides, dense_source=reference)
    validate_config(cfg)
    return cfg


def config_hash(config):
    from omegaconf import OmegaConf
    return hashlib.sha256(json.dumps(OmegaConf.to_container(config, resolve=True), sort_keys=True).encode()).hexdigest()


def restore_reference(archive, destination):
    """Extract only four required files by verified names; never arbitrary paths."""
    destination = Path(destination)
    require(not destination.exists(), "Reference destination already exists; use --dense-source to reuse it")
    names = {"shared_random_seed42.pt": "shared_random_seed42.pt",
             "J1_dense_control/global_history.csv": "J1_dense_control/global_history.csv",
             "J1_dense_control/pilot_state.json": "J1_dense_control/pilot_state.json",
             "J1_dense_control/resolved_config.yaml": "J1_dense_control/resolved_config.yaml"}
    with tarfile.open(archive) as source:
        selected = {}
        for member in source:
            if not member.isfile():
                continue
            for suffix in names:
                if member.name == suffix or member.name.endswith("/" + suffix):
                    require(suffix not in selected, "Ambiguous reference archive member: " + suffix)
                    selected[suffix] = member
        require(set(selected) == set(names), "Reference archive lacks required metadata or zero-epoch initializer")
        destination.mkdir(parents=True)
        for suffix, member in selected.items():
            target = destination / names[suffix]
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, target.open("wb") as output:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    output.write(block)


def inspect_search(path, config):
    """Read and pin finished search artifacts without data/model/test construction."""
    from omegaconf import OmegaConf
    from net_complexity.training.one_shot_pruning_config import ABLATION_PROTOCOL
    path = Path(path)
    state = read_json(path / "one_shot_state.json")
    selection = read_json(path / "selection.json")
    export = read_json(path / "export_only/diagnostics.json")
    require(config_hash(OmegaConf.load(path / "resolved_config.yaml")) == config_hash(config),
            "Search configuration changed: " + str(path))
    require(state["protocol"] == ABLATION_PROTOCOL and state["status"] == "search_only_completed"
            and state["test_evaluated"] is False and not state["branches"], "Search is not a completed search-only run")
    require(state["shared_search_ledger"]["global_training_epoch"] == 60
            and state["shared_search_ledger"]["search_epochs_consumed"] == 60,
            "Search has not consumed exactly60 epochs")
    require(31 <= selection["selected_epoch"] <= 60 and selection["reference_epoch"] == 60,
            "Search selected outside epochs31-60")
    require(selection["trace"]["candidate_epoch_start"] == 31
            and selection["trace"]["candidate_epoch_end"] == 60, "Search candidate window changed")
    require(selection == state["selection"], "Selection and saved state differ")
    require(selection["selected_checkpoint_hash"] == digest(path / "selected_checkpoint.pt"),
            "Selected checkpoint checksum differs")
    for key in ("mask_hash", "selected_model_state_hash", "selected_checkpoint_hash"):
        require(export[key] == selection[key], "Search/export identity differs: " + key)
    require(export["gated_equivalent_on_checked_batch"] is True
            and export["bn_calibration_batches"] == 0, "Search export is not equivalent and pre-BN")
    require(config.accuracy_guided.eligibility.min_keep_ratio == .08, "Search floor changed")
    files = ["resolved_config.yaml", "one_shot_state.json", "selection.json", "selected_checkpoint.pt",
             "export_only/diagnostics.json", "export_only/deployment.pt", "export_only/transfer_equivalence.json"]
    pins = {name: digest(path / name) for name in files}
    return {"path": str(path.resolve()), "files": pins, "config_hash": config_hash(config),
            "selected_epoch": selection["selected_epoch"], "mask_hash": selection["mask_hash"],
            "physical_parameters": export["physical_cost"]["physical_total_parameters"],
            "macs_per_image": export["physical_cost"]["conv_linear_macs_per_image"],
            "validation_accuracy": export["physical_validation_before_bn_calibration"]["accuracy"],
            "selection_policy": selection["policy"], "fallback_used": selection["trace"]["fallback_used"],
            "retained_closed_channels": len(export["retained_learned_closed_channels"])}


def execute(study, args):
    from net_complexity.training.one_shot_pruning_config import validate_inputs
    output = Path(args.output or study["output"]).resolve()
    reference = Path(args.dense_source or study["reference_output"]).resolve()
    profiles = [p for p in study["profiles"] if args.subset == "full" or p["id"] in study["pilot_profiles"]]
    cells = [(p, placement, compose_cell(study, p, placement, reference, args.device))
             for p in profiles for placement in ("internal", "output")]
    planned = {"stage": args.stage, "subset": args.subset, "seed": 42, "reference": str(reference),
               "output": str(output), "reference_training_epochs": 0, "official_test": False,
               "searches": len(cells), "recoveries": len(profiles) * 4,
               "new_training_epochs": {"search": len(cells) * 60, "inspect": 0,
                                       "recover": len(profiles) * 4 * 90}[args.stage],
               "per_model_budget": 150, "profiles": profiles,
               "reference_archive": str(args.reference_archive) if args.reference_archive else None}
    if args.dry_run:
        print(json.dumps(planned, indent=2))
        return planned
    if args.reference_archive:
        require(args.stage == "search", "Restore a reference only before search")
        restore_reference(args.reference_archive, reference)
    # All configs and inputs are checked before the first training call.
    for _, _, config in cells:
        validate_inputs(config)
    paths = {f"{row['id']}/{placement}": output / row["id"] / placement
             for row, placement, _ in cells}
    receipt_path = output / f"search_review_{args.subset}.json"
    if args.stage == "inspect":
        rows = {f"{row['id']}/{placement}": inspect_search(paths[f"{row['id']}/{placement}"] / "search", config)
                for row, placement, config in cells}
        output.mkdir(parents=True, exist_ok=True)
        receipt = {"protocol": PROTOCOL, "subset": args.subset, "test_evaluated": False, "searches": rows}
        write_json(receipt_path, receipt)
        with (output / f"search_summary_{args.subset}.csv").open("w", newline="") as stream:
            fields = ["cell", "selected_epoch", "physical_parameters", "macs_per_image", "validation_accuracy",
                      "selection_policy", "fallback_used", "retained_closed_channels"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for name, row in rows.items():
                writer.writerow({"cell": name, **{k: row[k] for k in fields if k != "cell"}})
                print(f"{name}: epoch={row['selected_epoch']}; params={row['physical_parameters']:,}; "
                      f"MACs={row['macs_per_image']:,}; validation={row['validation_accuracy']:.2%}; "
                      f"fallback={row['fallback_used']}", flush=True)
        print(f"Search checks saved: {receipt_path}", flush=True)
        return receipt
    receipt = None
    if args.stage == "recover":
        receipt = read_json(receipt_path)
        require(receipt["protocol"] == PROTOCOL and receipt["test_evaluated"] is False, "Invalid search review")
        expected = {f"{row['id']}/{placement}" for row, placement, _ in cells}
        require(set(receipt["searches"]) == expected, "Search review does not cover this subset")
    # Preflight every selected cell, including later cells, before starting any training.
    for row, placement, config in cells:
        name = f"{row['id']}/{placement}"
        cell = paths[name]
        target = cell / ("search" if args.stage == "search" else "recovery")
        if args.stage == "recover":
            require(inspect_search(cell / "search", config) == receipt["searches"][name],
                    "Search changed after inspection: " + name)
        if target.exists():
            if args.stage == "search":
                inspect_search(target, config)
            else:
                state = read_json(target / "one_shot_state.json")
                require(state["status"] == "completed" and state["test_evaluated"] is False,
                        "Partial recovery retained; refusing to restart/overwrite: " + str(target))
                from omegaconf import OmegaConf
                require(config_hash(OmegaConf.load(target / "resolved_config.yaml")) == config_hash(config),
                        "Existing recovery config changed")
                require(digest(target / "selected_checkpoint.pt") == receipt["searches"][name]["files"]["selected_checkpoint.pt"],
                        "Existing recovery used another checkpoint")
                from net_complexity.training.one_shot_pruning_config import resolved_branch_plan
                expected_branches = {item["id"] for item in resolved_branch_plan(config)}
                require(set(state["branches"]) == expected_branches, "Completed recovery lacks planned branches")
                for branch_id in expected_branches:
                    branch = state["branches"][branch_id]
                    require(branch["status"] in {"completed", "infeasible"}
                            and branch["ledger"]["global_training_epoch"] == 150
                            and branch["final_training_epochs_executed"] == 90,
                            "Completed recovery branch has an incomplete budget")
                    require(read_json(target / branch_id / "branch_state.json") == branch
                            and (target / branch_id / "deployment.pt").is_file()
                            and (target / branch_id / "initial_state.pt").is_file(),
                            "Completed recovery branch artifacts are incomplete")
    from net_complexity.training.one_shot_pruning import run_one_shot_pruning
    for row, placement, config in cells:
        name = f"{row['id']}/{placement}"
        cell = paths[name]
        target = cell / ("search" if args.stage == "search" else "recovery")
        if target.exists():
            print(f"Reusing completed {args.stage}: {target}", flush=True)
            continue
        print(f"Starting {args.stage}: {name}", flush=True)
        run_one_shot_pruning(config, target, search_only=args.stage == "search",
                             reuse_search_from=cell / "search" if args.stage == "recover" else None)
    return planned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["search", "inspect", "recover"], required=True)
    parser.add_argument("--subset", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--dense-source", type=Path)
    parser.add_argument("--reference-archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return execute(load_study(args.config), args)


if __name__ == "__main__":
    main()
