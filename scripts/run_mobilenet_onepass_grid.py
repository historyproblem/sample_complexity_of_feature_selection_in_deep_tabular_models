"""Run eight independent MobileNet search/recovery profiles sequentially.

Use .venv/bin/python from the repository root. --dry-run needs only the
standard library and prints the reference import plus eight full-cycle commands.
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


CONFIG_NAME = "mobilenet_onepass_grid_thr05_09"
GRID_CONFIG_PATH = Path("configs") / f"{CONFIG_NAME}.json"


def load_grid(repo_root: Path) -> dict:
    grid = json.loads((repo_root / GRID_CONFIG_PATH).read_text(encoding="utf-8"))
    if grid["base_config"] != "mobilenet_onepass_thr05_gap8_12pp":
        raise ValueError("The grid requires the verified 30+45 auto-step/no-boost base config")
    if grid["output_root"] != f"outputs/runs/{CONFIG_NAME}":
        raise ValueError("The grid must use its own output directory")
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
                f"_seed{grid['seed']}_nocap_autostep_noboost"
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
                f"cyclic_channel_pruning.g_prob_threshold={threshold:g}",
                "cyclic_channel_pruning.threshold_max_param_fraction=null",
                f"run_history.root_dir={json.dumps(str(repo_root / grid['output_root']))}",
                "mlflow.experiment_name=mobilenetv2_onepass_grid_thr05_09_20261109",
                f"mlflow.run_name={name}",
                f"mlflow.tags.recipe={CONFIG_NAME}_seed{grid['seed']}",
                f"hydra.run.dir={json.dumps(str(directory))}",
            ]
            identity = {
                **digests, "command": command, "commit": revision,
                "reference": reference_identity or {"archive_sha256": single.EXPECTED_ARCHIVE_SHA256},
            }
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            stages.append(single.Stage("onepass", directory, command, fingerprint))
    if len(stages) != 8 or len({stage.directory for stage in stages}) != 8:
        raise ValueError("The grid must produce exactly eight distinct profiles")
    return stages


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
            completed = {stage.directory: single.validate_existing_stage(stage) for stage in plan}
            plan_record = {
                "grid": grid, "new_epochs_per_model": 75, "total_new_epochs": 600,
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
                f"Grid: 8 sequential profiles; seed={grid['seed']}; "
                "each 30 search + 45 recovery = 75 epochs; 600 new epochs total. "
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
                single.validate_existing_stage(stage)
                single.write_receipt(stage, "complete")
            results = []
            for stage in plan:
                summary_path = stage.directory / "cyclic_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                results.append({
                    "profile": stage.directory.name,
                    "cyclic_summary": str(summary_path),
                    "num_epochs_executed": summary["training_cost"]["num_epochs_executed"],
                    "test_metrics": summary["final_result"]["test_metrics"],
                })
            _write_json(output_root / "grid_summary.json", {
                "grid": grid, "total_new_epochs": 600, "profiles": results,
            })
        print("Done: all eight search/recovery profiles have completed frozen-checkpoint test results.", flush=True)
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
