"""Run eight independent MobileNet channel-search profiles sequentially.

Use .venv/bin/python from the repository root. --dry-run needs only the
standard library and prints the reference import plus eight search-only commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import run_mobilenet_onepass as single


CONFIG_NAME = "mobilenet_onepass_grid_thr05_09_search_only_warmup10"
GRID_CONFIG_PATH = Path("configs") / f"{CONFIG_NAME}.json"


def load_grid(repo_root: Path) -> dict:
    grid = json.loads((repo_root / GRID_CONFIG_PATH).read_text(encoding="utf-8"))
    if grid["base_config"] != "mobilenet_onepass_thr05_gap8_12pp":
        raise ValueError("The grid requires the verified auto-step/no-boost base config")
    if grid["output_root"] != f"outputs/runs/{CONFIG_NAME}":
        raise ValueError("The grid must use its own output directory")
    if grid["search_only"] is not True:
        raise ValueError("This grid must disable post-search recovery training")
    if type(grid["warmup_epochs"]) is not int or grid["warmup_epochs"] != 10:
        raise ValueError("This grid requires ten lambda warmup epochs")
    fraction = float(grid["auto_log_step_epoch_fraction"])
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("The auto-step target epoch fraction must be in (0, 1]")
    if math.floor(30 * fraction - grid["warmup_epochs"]) < 1:
        raise ValueError("The auto-step target must leave at least one post-warmup update")
    if grid["thresholds"] != [0.5, 0.9]:
        raise ValueError("The grid must contain channel thresholds 0.5 and 0.9")
    if type(grid["seed"]) is not int or grid["seed"] < 0:
        raise ValueError("The grid seed must be a non-negative integer")
    bands = grid["lambda_bands"]
    if len(bands) != 4:
        raise ValueError("The grid requires four lambda bands")
    pairs = []
    for band in bands:
        soft, hard = float(band["soft_drop"]), float(band["hard_drop"])
        if not (math.isfinite(soft) and math.isfinite(hard) and 0 < soft < hard < 1):
            raise ValueError("Lambda bands must satisfy 0 < soft_drop < hard_drop < 1")
        if not math.isclose(hard - soft, 0.04, abs_tol=1e-12):
            raise ValueError("Each lambda band must be four percentage points wide")
        pairs.append((soft, hard))
    if len(set(pairs)) != 4 or (0.08, 0.12) not in pairs:
        raise ValueError("Use four unique lambda bands, including soft=0.08/hard=0.12")
    return grid


def build_plan(repo_root: Path, reference_identity: dict | None = None) -> list[single.Stage]:
    grid = load_grid(repo_root)
    base_path = repo_root / "configs" / f"{grid['base_config']}.yaml"
    digests = {
        "grid_sha256": hashlib.sha256((repo_root / GRID_CONFIG_PATH).read_bytes()).hexdigest(),
        "base_config_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_launcher_sha256": hashlib.sha256(Path(single.__file__).read_bytes()).hexdigest(),
    }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
    ).strip()
    reference_dir = repo_root / single.OUTPUT_ROOT / single.REFERENCE_DIRECTORY
    stages = []
    for band in grid["lambda_bands"]:
        soft, hard = band["soft_drop"], band["hard_drop"]
        for threshold in grid["thresholds"]:
            name = (
                f"mobilenet_grid_thr{threshold:g}_soft{soft:g}_hard{hard:g}"
                f"_seed{grid['seed']}_nocap_autostep_noboost_search_only_warmup10"
            )
            directory = repo_root / grid["output_root"] / name
            command = [
                ".venv/bin/python", "-u", "src/net_complexity/cli/cyclic_train.py",
                "--config-path", str(repo_root / "configs"),
                "--config-name", grid["base_config"],
                f"++seed={grid['seed']}",
                "training_arguments.adaptive_lambda.baseline_history_dir="
                + json.dumps(str(reference_dir)),
                f"training_arguments.adaptive_lambda.soft_drop={soft:g}",
                f"training_arguments.adaptive_lambda.hard_drop={hard:g}",
                f"training_arguments.adaptive_lambda.warmup_epochs={grid['warmup_epochs']}",
                "++training_arguments.adaptive_lambda.auto_log_step_epoch_fraction="
                + str(grid["auto_log_step_epoch_fraction"]),
                f"cyclic_channel_pruning.g_prob_threshold={threshold:g}",
                "cyclic_channel_pruning.threshold_max_param_fraction=null",
                "++cyclic_channel_pruning.search_only=true",
                "cyclic_channel_pruning.recovery_epochs=0",
                "cyclic_channel_pruning.final_epochs=0",
                f"run_history.root_dir={json.dumps(str(repo_root / grid['output_root']))}",
                "mlflow.experiment_name=mobilenetv2_onepass_grid_thr05_09_search_only_warmup10_20261109",
                f"mlflow.run_name={name}",
                f"mlflow.tags.recipe={CONFIG_NAME}_seed{grid['seed']}",
                f"hydra.run.dir={json.dumps(str(directory))}",
            ]
            identity = {
                **digests, "command": command, "commit": revision,
                "reference": reference_identity or {"archive_sha256": single.EXPECTED_ARCHIVE_SHA256},
            }
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            stages.append(single.Stage("search_only", directory, command, fingerprint))
    if len(stages) != 8 or len({stage.directory for stage in stages}) != 8:
        raise ValueError("The grid must produce exactly eight distinct profiles")
    return stages


def validate_existing_stage(stage: single.Stage) -> bool:
    """Accept only complete 30-epoch searches with their channel architecture."""
    if not stage.directory.exists():
        return False
    receipt_path = stage.directory / "launcher_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"Existing directory has no launch receipt: {stage.directory}; refusing rerun")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("fingerprint") != stage.fingerprint:
        raise RuntimeError(f"Existing stage fingerprint differs: {stage.directory}; refusing rerun")
    try:
        summary = json.loads((stage.directory / "cyclic_summary.json").read_text(encoding="utf-8"))
        if summary["search_only"] is not True or summary["num_cycles_completed"] != 1:
            raise RuntimeError("Expected exactly one search-only cycle")
        if summary["training_cost"]["num_epochs_executed"] != 30:
            raise RuntimeError("Search must finish exactly 30 epochs")
        if len(summary["cycle_results"]) != 1 or summary["recovery_results"]:
            raise RuntimeError("Recovery training must not run in this grid")
        if summary["final_result"] is not None:
            raise RuntimeError("Search-only results must not claim a trained final model")
        search = summary["cycle_results"][0]
        if search["num_epochs_executed"] != 30:
            raise RuntimeError("Search result has the wrong epoch count")
        search_dir = Path(search["run_dir"])
        single.validate_history(search_dir / "history.csv", 30)
        for name in ("best.pt", "last.pt"):
            if not (search_dir / "checkpoints" / name).is_file():
                raise RuntimeError(f"Search checkpoint missing: {name}")
        architecture = summary["architecture"]
        stored = json.loads(Path(architecture["architecture_file"]).read_text(encoding="utf-8"))
        mask = json.loads(Path(architecture["mask_file"]).read_text(encoding="utf-8"))
        if mask != summary["final_disabled_channels"] or stored["channel_mask"] != mask:
            raise RuntimeError("Saved channel mask differs from the selected architecture")
        for key, value in (("structural", True), ("trained", False), ("gate_free", True)):
            if architecture[key] is not value or stored[key] is not value:
                raise RuntimeError(f"Unexpected architecture flag: {key}")
        keys = ("total_params", "dense_total_params", "removed_params", "channels_removed")
        for key in keys:
            if type(architecture[key]) is not int or architecture[key] != stored[key]:
                raise RuntimeError(f"Invalid architecture count: {key}")
        dense, compact = architecture["dense_total_params"], architecture["total_params"]
        if not 0 < compact <= dense or architecture["removed_params"] != dense - compact:
            raise RuntimeError("Physical parameter counts do not reconcile")
        fraction = (dense - compact) / dense
        if not math.isclose(architecture["removed_fraction"], fraction, abs_tol=1e-12):
            raise RuntimeError("Physical removal fraction does not reconcile")
        if architecture["removed_fraction"] != stored["removed_fraction"]:
            raise RuntimeError("Saved physical removal fractions differ")
        if architecture["channels_removed"] != sum(len(channels) for channels in mask.values()):
            raise RuntimeError("Selected channel count does not match the mask")
    except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Stage is incomplete or invalid: {stage.directory}; refusing to repeat training. {exc}"
        ) from exc
    return True


def _write_json(target: Path, data: dict) -> None:
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--baseline-archive", type=Path)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != repo_root:
        parser.error("Run from the repository root")
    archive_path = (args.baseline_archive or repo_root / "mobilenet_baseline_partial.tar.gz").resolve()
    reference_dir = repo_root / single.OUTPUT_ROOT / single.REFERENCE_DIRECTORY
    import_command = [
        ".venv/bin/python", "scripts/prepare_mobilenet_reference.py",
        "--archive", str(archive_path), "--output-dir", str(reference_dir),
    ]
    try:
        grid = load_grid(repo_root)
        if args.dry_run:
            plan = build_plan(repo_root)
            print(shlex.join(import_command))
            for stage in plan:
                print(shlex.join(stage.command))
            return 0
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"Upload the selected baseline archive to {archive_path}, or set --baseline-archive. "
                "No new dense baseline will be trained."
            )
        with single.exclusive_launcher(repo_root, output_root=Path(grid["output_root"])):
            print(shlex.join(import_command), flush=True)
            subprocess.run(import_command, cwd=repo_root, check=True)
            reference_identity = single.validate_imported_reference(reference_dir, archive_path)
            single.preflight(repo_root, config_name=grid["base_config"])
            plan = build_plan(repo_root, reference_identity)
            completed = {stage.directory: validate_existing_stage(stage) for stage in plan}
            plan_record = {
                "grid": grid, "new_epochs_per_model": 30, "total_new_epochs": 240,
                "profiles": [
                    {"directory": str(stage.directory), "command": stage.command,
                     "fingerprint": stage.fingerprint}
                    for stage in plan
                ],
            }
            output_root = repo_root / grid["output_root"]
            plan_path = output_root / "grid_plan.json"
            if plan_path.exists() and json.loads(plan_path.read_text()) != plan_record:
                raise RuntimeError("Existing grid plan differs; refusing to replace or repeat its training")
            _write_json(plan_path, plan_record)
            print(
                f"Grid: 8 sequential profiles; seed={grid['seed']}; warmup={grid['warmup_epochs']}; "
                "each 30 search epochs; no recovery training; 240 new epochs total. "
                "Historical reference reused; no new baseline training.",
                flush=True,
            )
            for index, stage in enumerate(plan, start=1):
                if single.validate_imported_reference(reference_dir, archive_path) != reference_identity:
                    raise RuntimeError("Imported reference changed; refusing to start training")
                print(f"Profile {index}/8: {stage.directory.name}", flush=True)
                if completed[stage.directory]:
                    print(f"Reuse completed stage: {stage.directory}", flush=True)
                    continue
                stage.directory.mkdir(parents=True, exist_ok=False)
                single.write_receipt(stage, "running")
                print(shlex.join(stage.command), flush=True)
                subprocess.run(stage.command, cwd=repo_root, check=True)
                validate_existing_stage(stage)
                single.write_receipt(stage, "complete")
            results = []
            for stage in plan:
                summary_path = stage.directory / "cyclic_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                results.append({
                    "profile": stage.directory.name,
                    "cyclic_summary": str(summary_path),
                    "num_epochs_executed": summary["training_cost"]["num_epochs_executed"],
                    "architecture": summary["architecture"],
                    "last_search_validation_accuracy": summary["cycle_results"][0].get(
                        "last_valid_metrics", {}
                    ).get("valid_accuracy"),
                })
            _write_json(output_root / "grid_summary.json", {
                "grid": grid, "total_new_epochs": 240, "profiles": results,
            })
        print("Done: all eight searches completed; channel masks and architecture sizes saved; no recovery training.", flush=True)
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
