"""Ten independent stronger MobileNets: 45epochs each, inherited recovery and test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys

import run_mobilenet_onepass as single
import run_mobilenet_open98_ablation as previous


MANIFEST = Path("configs/mobilenet_strong10.json")
OUTPUT = "outputs/runs/mobilenet_strong10_20261128"
REFERENCE = OUTPUT + "/reference"
BASE_CONFIG = "mobilenet_strong10"
MIN_LOG_STEP = math.log(1.05)


def load_manifest(repo: Path) -> dict:
    spec = json.loads((repo / MANIFEST).read_text())
    fields = {"protocol", "base_config", "output_root", "reference_dir", "epochs_per_model", "removal_threshold", "seed", "runs"}
    if set(spec) != fields or (spec["protocol"], spec["base_config"], spec["output_root"], spec["reference_dir"]) != (
            "mobilenet_strong10_v1", BASE_CONFIG, OUTPUT, REFERENCE):
        raise ValueError("Invalid strong10 manifest identity or fields.")
    if spec["epochs_per_model"] != 45 or spec["removal_threshold"] != .9 or spec["seed"] != 42 or len(spec["runs"]) != 10:
        raise ValueError("Exactly10 models, 45epochs each, removal0.9 and seed42 are required.")
    row_fields = {"id", "lambda_init", "mask_init", "soft_drop", "hard_drop", "warmup_epochs", "search_epochs", "recovery_epochs"}
    ids, combinations = set(), set()
    for row in spec["runs"]:
        if set(row) != row_fields or not re.fullmatch(r"[A-Za-z0-9_-]+", row["id"]):
            raise ValueError("Only the authorized method parameters and run ID may vary.")
        if not all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in ("lambda_init", "soft_drop", "hard_drop")):
            raise ValueError("Method parameters must be finite numbers.")
        if not (0 < row["lambda_init"] < 10 and 0 < row["soft_drop"] < row["hard_drop"] < 1):
            raise ValueError("Invalid initial lambda or quality thresholds.")
        if row["mask_init"] not in {"paper", "paper_resnet50"}:
            raise ValueError("Use the existing p_open0.87 or0.98 initialization mode.")
        if not all(type(row[key]) is int for key in ("warmup_epochs", "search_epochs", "recovery_epochs")):
            raise ValueError("Epoch counts must be integers.")
        if not (0 <= row["warmup_epochs"] < row["search_epochs"] and row["recovery_epochs"] > 0
                and row["search_epochs"] + row["recovery_epochs"] == 45):
            raise ValueError("Each model needs search+recovery=45 and post-warmup updates.")
        key = tuple(row[name] for name in sorted(row_fields - {"id"}))
        if row["id"] in ids or key in combinations:
            raise ValueError("Duplicate run ID or method configuration.")
        ids.add(row["id"]); combinations.add(key)
    return spec


def initial_probability(row: dict) -> float:
    return 1 / (1 + math.exp(-({"paper": 1.9, "paper_resnet50": 3.9}[row["mask_init"]])))


def auto_details(row: dict) -> dict:
    raw = math.log(10 / row["lambda_init"]) / (row["search_epochs"] - row["warmup_epochs"])
    effective = max(raw, MIN_LOG_STEP)
    return {"auto_raw_log_step": raw, "effective_log_step": effective, "lambda_update_factor": math.exp(effective)}


def common_overrides() -> list[str]:
    return [f"training_arguments.adaptive_lambda.baseline_history_dir={REFERENCE}",
            "cyclic_channel_pruning.g_prob_threshold=0.9", f"run_history.root_dir={OUTPUT}",
            "mlflow.experiment_name=mobilenet_strong10_20261128", "mlflow.tags.recipe=mobilenet_strong10_20261128"]


def run_overrides(row: dict) -> list[str]:
    return [f"model.lambda_coef={row['lambda_init']}", f"model.gumbel_init_mode={row['mask_init']}",
            f"training_arguments.adaptive_lambda.soft_drop={row['soft_drop']}",
            f"training_arguments.adaptive_lambda.hard_drop={row['hard_drop']}",
            f"training_arguments.adaptive_lambda.warmup_epochs={row['warmup_epochs']}",
            f"training_arguments.num_epochs={row['search_epochs']}", f"scheduler.T_max={row['search_epochs']}",
            f"cyclic_channel_pruning.gumbel_epochs={row['search_epochs']}",
            f"cyclic_channel_pruning.recovery_epochs={row['recovery_epochs']}",
            f"cyclic_channel_pruning.final_epochs={row['recovery_epochs']}",
            f"mlflow.run_name={row['id']}",
            "mlflow.tags.initial_open_probability=" + json.dumps(str(initial_probability(row)))]


def validate_configs(repo: Path) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    sys.path.insert(0, str(repo / "src"))
    from net_complexity.training.engine import _resolve_adaptive_lambda_log_step_init
    spec = load_manifest(repo)
    details = []
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        for row in spec["runs"]:
            cfg = compose(config_name=BASE_CONFIG, overrides=run_overrides(row))
            expected = compose(config_name="mobilenet_open98_gumbel", overrides=common_overrides() + run_overrides(row))
            if OmegaConf.to_container(cfg, resolve=False) != OmegaConf.to_container(expected, resolve=False):
                raise ValueError("Training recipe changed beyond the authorized method settings and derived phase lengths.")
            fixed = {"seed": 42, "model.backbone.block.train_gate_mode": "gumbel_hard",
                     "model.backbone.block.eval_gate_mode": "deterministic_hard", "model.backbone.block.gate_threshold": .5,
                     "model.backbone.block.gate_internal_width": True, "model.backbone.block.beta": 1,
                     "model.backbone.block.temperature": 1, "model.criterion.label_smoothing": 0,
                     "training_arguments.adaptive_lambda.enabled": True,
                     "training_arguments.adaptive_lambda.log_step_init": "auto",
                     "training_arguments.adaptive_lambda.auto_log_step_epoch_fraction": 1,
                     "training_arguments.adaptive_lambda.log_step_min": MIN_LOG_STEP,
                     "training_arguments.adaptive_lambda.adaptive_log_step_enabled": False,
                     "training_arguments.adaptive_lambda.lambda_min": 1e-8,
                     "training_arguments.adaptive_lambda.lambda_max": 100,
                     "cyclic_channel_pruning.g_prob_threshold": .9,
                     "cyclic_channel_pruning.threshold_max_param_fraction": None,
                     "cyclic_channel_pruning.search_only": False, "cyclic_channel_pruning.max_cycles": 1,
                     "cyclic_channel_pruning.channel_selection": "best_checkpoint",
                     "cyclic_channel_pruning.weight_handoff.enabled": True,
                     "cyclic_channel_pruning.weight_handoff.checkpoint_name": "best.pt"}
            for key, value in fixed.items():
                if OmegaConf.select(cfg, key) != value:
                    raise ValueError(f"Protected setting changed: {key}.")
            raw = _resolve_adaptive_lambda_log_step_init(cfg.training_arguments, cfg.training_arguments.adaptive_lambda,
                initial_lambda_coef=row["lambda_init"], warmup_epochs=row["warmup_epochs"], update_every_epochs=1)
            if not math.isclose(raw, auto_details(row)["auto_raw_log_step"], rel_tol=1e-12):
                raise ValueError("Resolved automatic lambda step differs from the search horizon.")
            details.append({**row, "initial_p_open": initial_probability(row), **auto_details(row)})
    return {"models": 10, "epochs_per_model": 45, "total_training_epochs": 450,
            "removal_threshold": .9, "search_forward_threshold": .5, "runs": details}


def validate_reference(repo: Path, archive: Path) -> dict:
    spec = load_manifest(repo)
    reference = single.validate_imported_reference(repo / spec["reference_dir"], archive)
    maximum = max(row["search_epochs"] for row in spec["runs"])
    with (repo / spec["reference_dir"] / "history.csv").open(newline="") as stream:
        epochs = {int(float(row["epoch"])) for row in csv.DictReader(stream)}
    if not set(range(1, maximum + 1)) <= epochs:
        raise ValueError("Historical reference does not cover every requested search epoch.")
    return {**reference, "controller_search_epoch_coverage": [1, maximum]}


def build_plan(repo: Path, reference: dict | None = None) -> list[single.Stage]:
    spec = load_manifest(repo)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    paths = [MANIFEST, Path("configs/mobilenet_strong10.yaml"), Path("configs/mobilenet_open98_gumbel.yaml"),
             Path("configs/mobilenet_onepass_thr05_gap8_12pp.yaml"), Path("scripts/run_mobilenet_strong10.py"),
             Path("scripts/run_mobilenet_open98_ablation.py"), Path("scripts/run_mobilenet_onepass.py"),
             Path("scripts/prepare_mobilenet_reference.py")]
    paths += sorted(path.relative_to(repo) for path in (repo / "src/net_complexity").rglob("*.py"))
    digests = {str(path): hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in paths}
    plan = []
    for row in spec["runs"]:
        directory = repo / spec["output_root"] / row["id"]
        command = [".venv/bin/python", "-u", "src/net_complexity/cli/cyclic_train.py",
                   "--config-name", BASE_CONFIG, *run_overrides(row), f"mlflow.tags.source_commit={revision}",
                   "hydra.run.dir=" + json.dumps(str(directory))]
        fingerprint = hashlib.sha256(json.dumps({"row": row, "command": command, "revision": revision,
            "digests": digests, "reference": reference or {"archive_sha256": single.EXPECTED_ARCHIVE_SHA256}}, sort_keys=True).encode()).hexdigest()
        plan.append(single.Stage("gumbel_hard", directory, command, fingerprint))
    return plan


def validate_run(stage: single.Stage, row: dict) -> bool:
    if not previous.validate_stage(stage, search_epochs=row["search_epochs"], recovery_epochs=row["recovery_epochs"]):
        return False
    summary = json.loads((stage.directory / "cyclic_summary.json").read_text())
    for result in summary["cycle_results"] + summary["recovery_results"]:
        if not (Path(result["run_dir"]) / "checkpoints/last.pt").is_file():
            raise RuntimeError("Last checkpoint missing; refusing another fit.")
    return True


def comparison_record(stage: single.Stage, row: dict) -> dict:
    if not validate_run(stage, row):
        raise RuntimeError("Cannot report an incomplete model.")
    result = previous.comparison_record(stage, search_epochs=row["search_epochs"], recovery_epochs=row["recovery_epochs"])
    return {"id": row["id"], **{key: row[key] for key in row if key != "id"},
            "initial_p_open": initial_probability(row), "removal_threshold": .9, "search_forward_threshold": .5,
            **auto_details(row), **result}


def save_results(output: Path, records: list[dict], protocol: dict) -> None:
    previous.write_json(output / "comparison.json", {"protocol": protocol, "results": records})
    temporary = output / "comparison.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    temporary.replace(output / "comparison.csv")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--inspect", action="store_true")
    parser.add_argument("--baseline-archive", type=Path, default=Path("outputs/baselines/mobilenet_baseline_partial.tar.gz"))
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != repo:
        parser.error("Run from the repository root.")
    try:
        spec = load_manifest(repo)
        archive = args.baseline_archive.resolve()
        import_command = [".venv/bin/python", "scripts/prepare_mobilenet_reference.py", "--archive", str(archive),
                          "--output-dir", str(repo / spec["reference_dir"])]
        if args.dry_run:
            print("10 independent models; each search+recovery=45; 450 training epochs; final physical test after each.")
            print(shlex.join(import_command))
            for stage in build_plan(repo): print(shlex.join(stage.command))
            return 0
        protocol = validate_configs(repo)
        with single.exclusive_launcher(repo, output_root=Path(spec["output_root"])):
            subprocess.run(import_command, cwd=repo, check=True)
            reference = validate_reference(repo, archive)
            maximum = max(row["search_epochs"] for row in spec["runs"])
            print(f"Controller reference for this grid: epochs 1-{maximum}; historical baseline reused.", flush=True)
            plan = build_plan(repo, reference)
            completed = [validate_run(stage, row) for stage, row in zip(plan, spec["runs"])]
            if not all(completed):
                single.preflight(repo, config_name="mobilenet_onepass_thr05_gap8_12pp")
            output = repo / spec["output_root"]
            plan_record = {"protocol": protocol, "reference": reference,
                           "stages": [{"id": row["id"], "command": stage.command, "fingerprint": stage.fingerprint}
                                      for stage, row in zip(plan, spec["runs"])]}
            stored = output / "grid_plan.json"
            if stored.exists() and json.loads(stored.read_text()) != plan_record:
                raise RuntimeError("Existing grid plan differs; refusing replacement or retraining.")
            previous.write_json(stored, plan_record)
            if args.inspect:
                print("10 recipes, reference, data and CUDA verified; no training started.", flush=True)
                return 0
            records = []
            for index, (stage, row, done) in enumerate(zip(plan, spec["runs"], completed), 1):
                current = build_plan(repo, validate_reference(repo, archive))
                if [item.fingerprint for item in current] != [item.fingerprint for item in plan]:
                    raise RuntimeError("Code, configs or reference changed after planning.")
                print(f"Model {index}/10 {row['id']}: search {row['search_epochs']} + recovery {row['recovery_epochs']} + physical test", flush=True)
                if done:
                    print(f"Reuse completed: {stage.directory}", flush=True)
                else:
                    stage.directory.mkdir(parents=True, exist_ok=False)
                    single.write_receipt(stage, "running")
                    print(shlex.join(stage.command), flush=True)
                    subprocess.run(stage.command, cwd=repo, check=True)
                    validate_run(stage, row)
                    single.write_receipt(stage, "complete")
                records.append(comparison_record(stage, row))
                save_results(output, records, protocol)
            print(f"Done: 10/10; {output / 'comparison.csv'}", flush=True)
        return 0
    except (OSError, KeyError, IndexError, TypeError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Strong10 blocked: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
