"""Run selected-weight recovery45, then four fresh ImageNet mask searches30."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import run_mobilenet_onepass as single
import run_mobilenet_onepass_grid as previous_grid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
MANIFEST = Path("configs/mobilenet_recovery45_then_wide_search.json")


def inspect_source(repo: Path, source_profile: Path | None = None) -> dict:
    from net_complexity.training.mobilenet_selected_recovery import inspect_source as inspect

    return inspect(repo, source_profile)


def load_workflow(repo: Path) -> dict:
    spec = json.loads((repo / MANIFEST).read_text())
    if (spec["recovery_epochs"], spec["search_epochs"], spec["warmup_epochs"],
            spec["seed"], spec["auto_log_step_epoch_fraction"]) != (45, 30, 10, 42, 1.0):
        raise ValueError("Workflow requires recovery45 + four30epoch searches, warmup10, seed42")
    if spec["thresholds"] != [0.5, 0.9] or spec["lambda_bands"] != [
        {"soft_drop": 0.15, "hard_drop": 0.20}, {"soft_drop": 0.20, "hard_drop": 0.25},
    ]:
        raise ValueError("Expected thresholds0.5/0.9 and15/20,20/25percentage-point bands")
    return spec


def build_plan(repo: Path, reference: dict | None = None, source: dict | None = None,
               source_profile: Path | None = None) -> list[single.Stage]:
    spec = load_workflow(repo)
    profile = (source_profile or repo / spec["recovery_source"]["source_profile"]).resolve()
    output = repo / spec["output_root"]
    reference_dir = repo / single.OUTPUT_ROOT / single.REFERENCE_DIRECTORY
    commands = [("recovery45", output / "recovery45_thr09_gap6_10_epoch30", [
        ".venv/bin/python", "-u", "src/net_complexity/cli/mobilenet_recovery.py",
        "--config-name", spec["recovery_config"],
        "selected_recovery.source_profile=" + json.dumps(str(profile)),
        "mlflow.run_name=mobilenet_thr09_gap6_10_recovery45_transferred",
    ])]
    for band in spec["lambda_bands"]:
        for threshold in spec["thresholds"]:
            name = (f"mobilenet_wide_thr{threshold:g}_soft{band['soft_drop']:g}"
                    f"_hard{band['hard_drop']:g}_seed42_nocap_autostep_noboost_search_only_warmup10")
            commands.append(("search_only", output / name, [
                ".venv/bin/python", "-u", "src/net_complexity/cli/cyclic_train.py",
                "--config-name", spec["base_config"], "++seed=42",
                "training_arguments.adaptive_lambda.baseline_history_dir=" + json.dumps(str(reference_dir)),
                f"training_arguments.adaptive_lambda.soft_drop={band['soft_drop']:g}",
                f"training_arguments.adaptive_lambda.hard_drop={band['hard_drop']:g}",
                "training_arguments.adaptive_lambda.warmup_epochs=10",
                "++training_arguments.adaptive_lambda.auto_log_step_epoch_fraction=1.0",
                "training_arguments.adaptive_lambda.enabled=true",
                "training_arguments.adaptive_lambda.adaptive_log_step_enabled=false",
                "model.lambda_coef=0.0001", "model.backbone.pretrained_weights=IMAGENET1K_V2",
                "model.criterion.label_smoothing=0",
                f"cyclic_channel_pruning.g_prob_threshold={threshold:g}",
                "cyclic_channel_pruning.threshold_max_param_fraction=null",
                "++cyclic_channel_pruning.search_only=true",
                "++cyclic_channel_pruning.weight_handoff.enabled=false",
                "cyclic_channel_pruning.recovery_epochs=0", "cyclic_channel_pruning.final_epochs=0",
                f"mlflow.run_name={name}",
            ]))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    paths = [MANIFEST, Path(__file__).resolve().relative_to(repo),
             Path("configs") / (spec["base_config"] + ".yaml"),
             Path("configs") / (spec["recovery_config"] + ".yaml"),
             Path(spec["recovery_source"]["mask_file"]),
             Path("src/net_complexity/training/mobilenet_selected_recovery.py"),
             Path("src/net_complexity/cli/mobilenet_recovery.py"),
             Path("src/net_complexity/models/channel_pruning.py"),
             Path("src/net_complexity/models/pruned_mobilenet_v2.py"),
             Path("scripts/run_mobilenet_onepass.py"),
             Path("scripts/run_mobilenet_onepass_grid.py"),
             Path("src/net_complexity/training/engine.py")]
    digests = {str(path): hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in paths}
    plan = []
    for kind, directory, command in commands:
        command += [
            "run_history.root_dir=" + json.dumps(str(output)),
            "mlflow.experiment_name=mobilenet_recovery45_then_wide_search_20261112",
            "hydra.run.dir=" + json.dumps(str(directory)),
        ]
        identity = {"command": command, "source": source, "reference": reference,
                    "commit": revision, "digests": digests}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        plan.append(single.Stage(kind, directory, command, fingerprint))
    return plan


def validate_stage(stage: single.Stage) -> bool:
    if stage.kind == "search_only":
        return previous_grid.validate_existing_stage(stage)
    if not stage.directory.exists():
        return False
    try:
        receipt = json.loads((stage.directory / "launcher_receipt.json").read_text())
        if receipt["fingerprint"] != stage.fingerprint:
            raise ValueError("Recovery fingerprint differs")
        record = json.loads((stage.directory / "recovery_result.json").read_text())
        result = record["result"]
        summary = json.loads((stage.directory / "summary.json").read_text())
        if (record["new_epochs"], record["lifetime_training_epochs"],
                record["total_params"], result["num_epochs_executed"]) != (45, 75, 2251882, 45):
            raise ValueError("Unexpected recovery cost or architecture")
        if summary["runtime"]["model_parameters"]["total"] != 2251882 or not result["test_metrics"]:
            raise ValueError("Recovery lacks compact frozen-checkpoint test")
        if summary["runtime"]["test_evaluation"]["checkpoint_epoch"] != summary["best_valid"]["epoch"]:
            raise ValueError("Recovery test is not from the validation-selected checkpoint")
        single.validate_history(stage.directory / "history.csv", 45)
        if not (stage.directory / "checkpoints/best.pt").is_file():
            raise ValueError("Recovery best checkpoint is missing")
        if not (stage.directory / "source_provenance.json").is_file():
            raise ValueError("Held source controller state/provenance is missing")
        if json.loads((stage.directory / "source_provenance.json").read_text()) != record["source"]:
            raise ValueError("Recovery source/controller provenance differs")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Recovery stage incomplete/invalid; refusing rerun: {stage.directory}: {exc}") from exc
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-profile", type=Path)
    parser.add_argument("--baseline-archive", type=Path, default=Path("outputs/baselines/mobilenet_baseline_partial.tar.gz"))
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != repo:
        parser.error("Run from the repository root")
    try:
        spec = load_workflow(repo)
        archive = args.baseline_archive.resolve()
        reference_dir = repo / single.OUTPUT_ROOT / single.REFERENCE_DIRECTORY
        import_command = [".venv/bin/python", "scripts/prepare_mobilenet_reference.py",
                          "--archive", str(archive), "--output-dir", str(reference_dir)]
        if args.dry_run:
            print(shlex.join(import_command))
            for stage in build_plan(repo, source_profile=args.source_profile):
                print(shlex.join(stage.command))
            return 0
        source = inspect_source(repo, args.source_profile)
        with single.exclusive_launcher(repo, output_root=Path(spec["output_root"])):
            subprocess.run(import_command, cwd=repo, check=True)
            reference = single.validate_imported_reference(reference_dir, archive)
            plan = build_plan(repo, reference, source, args.source_profile)
            completed = [validate_stage(stage) for stage in plan]
            single.preflight(repo, config_name=spec["base_config"])
            output = repo / spec["output_root"]
            previous_grid._write_json(output / "workflow_plan.json", {
                "manifest": spec, "source": source, "new_epochs": 165,
                "stages": [{"kind": s.kind, "command": s.command, "fingerprint": s.fingerprint} for s in plan],
            })
            print("Sequence: selected trained weights -> compact recovery45 -> four independent ImageNet searches30;165new epochs", flush=True)
            for stage, done in zip(plan, completed):
                if inspect_source(repo, args.source_profile) != source:
                    raise RuntimeError("Recovery source changed during workflow")
                if single.validate_imported_reference(reference_dir, archive) != reference:
                    raise RuntimeError("Historical reference changed during workflow")
                if done:
                    print(f"Reuse completed: {stage.directory}", flush=True)
                    continue
                stage.directory.mkdir(parents=True, exist_ok=False)
                single.write_receipt(stage, "running")
                print(shlex.join(stage.command), flush=True)
                subprocess.run(stage.command, cwd=repo, check=True)
                validate_stage(stage)
                single.write_receipt(stage, "complete")
            previous_grid._write_json(output / "workflow_summary.json", {
                "new_epochs": 165, "recovery_lifetime_epochs": 75,
                "stages": [{"kind": s.kind, "directory": str(s.directory)} for s in plan],
            })
            print("Done: recovery45 and all four search-only runs completed", flush=True)
        return 0
    except (OSError, KeyError, ValueError, TypeError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
