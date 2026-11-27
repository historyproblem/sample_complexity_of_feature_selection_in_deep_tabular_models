"""Compare Gumbel and deterministic STE: lambda0.001, p_open~0.98, 30+15 epochs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import run_mobilenet_onepass as single


MANIFEST = Path("configs/mobilenet_open98_ablation.json")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_reference(repo: Path, archive: Path) -> dict:
    spec = json.loads((repo / MANIFEST).read_text())
    return single.validate_imported_reference(repo / spec["reference_dir"], archive)


def build_plan(repo: Path, reference: dict | None = None) -> list[single.Stage]:
    spec = json.loads((repo / MANIFEST).read_text())
    if (spec["search_epochs"], spec["recovery_epochs"]) != (30, 15):
        raise ValueError("The MobileNet budget is exactly30search+15recovery.")
    if spec["arms"] != [
        {"config": "mobilenet_open98_gumbel", "mode": "gumbel_hard"},
        {"config": "mobilenet_open98_ste", "mode": "ste_hard"},
    ]:
        raise ValueError("Expected one Gumbel arm and one deterministic STE arm.")
    reference = reference or {"archive_sha256": single.EXPECTED_ARCHIVE_SHA256}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    paths = [MANIFEST, Path("scripts/run_mobilenet_open98_ablation.py"),
             Path("scripts/run_mobilenet_onepass.py"),
             Path("configs/mobilenet_onepass_thr05_gap8_12pp.yaml"),
             Path("configs/mobilenet_open98_gumbel.yaml"),
             Path("configs/mobilenet_open98_ste.yaml")]
    # Bind the small training implementation, including shared imported modules.
    paths += sorted(path.relative_to(repo) for path in (repo / "src/net_complexity").rglob("*.py"))
    digests = {str(path): hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in paths}
    stages = []
    for arm in spec["arms"]:
        directory = repo / spec["output_root"] / arm["mode"]
        command = [".venv/bin/python", "-u", "src/net_complexity/cli/cyclic_train.py",
                   "--config-name", arm["config"], f"mlflow.tags.source_commit={revision}",
                   "hydra.run.dir=" + json.dumps(str(directory))]
        fingerprint = hashlib.sha256(json.dumps({
            "command": command, "commit": revision, "digests": digests,
            "reference": reference,
        }, sort_keys=True).encode()).hexdigest()
        stages.append(single.Stage(arm["mode"], directory, command, fingerprint))
    return stages


def validate_configs(repo: Path) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    sys.path.insert(0, str(repo / "src"))
    from net_complexity.training.engine import _resolve_adaptive_lambda_log_step_init

    expected = {
        "model.lambda_coef": .001, "model.gumbel_init_mode": "paper_resnet50",
        "model.entropy_regularization": "disabled", "model.entropy_regularization_coef": 0,
        "model.criterion.label_smoothing": 0,
        "model.backbone.pretrained_weights": "IMAGENET1K_V2",
        "model.backbone.block.gate_internal_width": True,
        "model.backbone.block.eval_gate_mode": "deterministic_hard",
        "model.backbone.block.gate_threshold": .5,
        "model.backbone.block.beta": 1, "model.backbone.block.temperature": 1,
        "optimizer._target_": "torch.optim.AdamW", "optimizer.lr": .001,
        "optimizer.pretrained_lr_scale": .1, "optimizer.weight_decay": .0005,
        "dataloaders.batch_size": 128, "model.backbone.num_classes": 200,
        "training_arguments.num_epochs": 30,
        "training_arguments.adaptive_lambda.enabled": True,
        "training_arguments.adaptive_lambda.baseline_history_dir": "outputs/runs/mobilenet_open98_gumbel_ablation_20261120/reference",
        "training_arguments.adaptive_lambda.warmup_epochs": 10,
        "training_arguments.adaptive_lambda.update_every_epochs": 1,
        "training_arguments.adaptive_lambda.acc_window": 3,
        "training_arguments.adaptive_lambda.log_step_init": "auto",
        "training_arguments.adaptive_lambda.auto_log_step_epoch_fraction": 1,
        "training_arguments.adaptive_lambda.adaptive_log_step_enabled": False,
        "training_arguments.adaptive_lambda.soft_drop": .06,
        "training_arguments.adaptive_lambda.hard_drop": .10,
        "training_arguments.adaptive_lambda.recovery.enabled": False,
        "training_arguments.batchnorm_recalibration.enabled": False,
        "scheduler.T_max": 30, "dataloaders.num_workers": 0, "seed": 42,
        "cyclic_channel_pruning.max_cycles": 1,
        "cyclic_channel_pruning.search_only": False,
        "cyclic_channel_pruning.channel_selection": "best_checkpoint",
        "cyclic_channel_pruning.g_prob_threshold": .5,
        "cyclic_channel_pruning.threshold_max_param_fraction": None,
        "cyclic_channel_pruning.gumbel_epochs": 30,
        "cyclic_channel_pruning.recovery_epochs": 15,
        "cyclic_channel_pruning.final_epochs": 15,
        "cyclic_channel_pruning.weight_handoff.enabled": True,
        "cyclic_channel_pruning.weight_handoff.checkpoint_name": "best.pt",
    }
    recipes = []
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        for name, mode in [("mobilenet_open98_gumbel", "gumbel_hard"), ("mobilenet_open98_ste", "ste_hard")]:
            cfg = compose(config_name=name)
            for key, value in expected.items():
                if OmegaConf.select(cfg, key) != value:
                    raise ValueError(f"Unexpected {name}:{key}; expected {value}.")
            if cfg.model.backbone.block.train_gate_mode != mode:
                raise ValueError("Gate mode does not match the noise arm.")
            alpha = cfg.training_arguments.adaptive_lambda
            step = _resolve_adaptive_lambda_log_step_init(
                cfg.training_arguments, alpha, initial_lambda_coef=.001,
                warmup_epochs=10, update_every_epochs=1,
            )
            if not math.isclose(step, .46051701859880917, rel_tol=1e-12):
                raise ValueError("Unexpected automatic lambda step.")
            recipe = OmegaConf.to_container(cfg, resolve=False)
            recipe["model"]["backbone"]["block"]["train_gate_mode"] = "ARM"
            recipe["mlflow"]["run_name"] = "ARM"
            recipe["mlflow"]["tags"]["noise_arm"] = "ARM"
            recipes.append(recipe)
    if recipes[0] != recipes[1]:
        raise ValueError("The two recipes differ beyond the requested gate-noise mode.")
    return {"search_epochs": 30, "recovery_epochs": 15, "lambda_log_step": step,
            "initial_p_open": 1 / (1 + math.exp(-3.9)), "soft_drop": .06, "hard_drop": .10}


def validate_stage(stage: single.Stage, *, search_epochs: int = 30, recovery_epochs: int = 15) -> bool:
    if not stage.directory.exists():
        return False
    try:
        receipt = json.loads((stage.directory / "launcher_receipt.json").read_text())
        if receipt["fingerprint"] != stage.fingerprint:
            raise ValueError("Existing arm has a different source/config fingerprint.")
        summary = json.loads((stage.directory / "cyclic_summary.json").read_text())
        if summary["search_only"] or summary["num_cycles_completed"] != 1:
            raise ValueError("Expected one complete search/recovery cycle.")
        if summary["channel_selection_source"] != "best_checkpoint":
            raise ValueError("The mask was not selected from the best validation checkpoint.")
        if summary["training_cost"]["num_epochs_executed"] != 45:
            raise ValueError("The experiment did not execute exactly45epochs.")
        if len(summary["cycle_results"]) != 1 or len(summary["recovery_results"]) != 1:
            raise ValueError("Unexpected extra search/recovery passes.")
        search, recovery = summary["cycle_results"][0], summary["recovery_results"][0]
        if search_epochs + recovery_epochs != 45:
            raise ValueError("The requested MobileNet phases must total45epochs.")
        if search["num_epochs_executed"] != search_epochs or recovery["num_epochs_executed"] != recovery_epochs:
            raise ValueError(f"Expected{search_epochs}search+{recovery_epochs}recovery.")
        if summary["final_result"] != recovery:
            raise ValueError("Final model differs from the recovery result.")
        handoff = summary["weight_handoff"]
        if not handoff["enabled"] or not handoff["surviving_weights_transferred"]:
            raise ValueError("Trained search weights were not inherited by recovery.")
        for result, epochs in [(search, search_epochs), (recovery, recovery_epochs)]:
            root = Path(result["run_dir"])
            if not root.resolve().is_relative_to(stage.directory.resolve()):
                raise ValueError("Result belongs to another output directory.")
            single.validate_history(root / "history.csv", epochs)
            if not (root / "checkpoints/best.pt").is_file():
                raise ValueError("Selected checkpoint is missing.")
        selection = summary["search_selections"][0]
        selected = Path(search["run_dir"]) / "checkpoints/best.pt"
        if selection["epoch"] != search["best_epoch"] or selection["checkpoint"] != str(selected):
            raise ValueError("Search mask/weights come from different checkpoints.")
        if hashlib.sha256(selected.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
            raise ValueError("Selected search checkpoint changed.")
        if not selection["held_adaptive_lambda_state"]:
            raise ValueError("Source adaptive state was not retained.")
        root = Path(recovery["run_dir"])
        physical = json.loads((root / "summary.json").read_text())
        if (physical["runtime"]["test_evaluation"]["checkpoint_epoch"] != recovery["best_epoch"]
                or physical["best_valid"]["epoch"] != recovery["best_epoch"]):
            raise ValueError("Test did not evaluate the best validation deployment.")
        accuracy = float(recovery["test_metrics"]["test_accuracy"])
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Frozen official test accuracy is unavailable.")
        if recovery["model_parameters"]["total"] != physical["runtime"]["model_parameters"]["total"]:
            raise ValueError("Compact parameter counts differ.")
    except (OSError, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Incomplete/different arm; refusing repeat training: {stage.directory}: {exc}") from exc
    return True


def comparison_record(stage: single.Stage, *, search_epochs: int = 30, recovery_epochs: int = 15) -> dict:
    if not validate_stage(stage, search_epochs=search_epochs, recovery_epochs=recovery_epochs):
        raise RuntimeError("Cannot summarize an uncompleted arm.")
    summary = json.loads((stage.directory / "cyclic_summary.json").read_text())
    final = summary["final_result"]
    physical = json.loads((Path(final["run_dir"]) / "summary.json").read_text())
    mask = summary["final_disabled_channels"]
    internal = sum(len(channels) for name, channels in mask.items() if "mid_gumbel_layer" in name)
    removed = sum(len(channels) for channels in mask.values())
    write_json(stage.directory / "selected_mask.json", mask)
    return {
        "mode": stage.kind, "seed": 42, "total_epochs": 45,
        "search_epoch": summary["search_selections"][0]["epoch"],
        "recovery_epoch": final["best_epoch"],
        "compact_parameters": final["model_parameters"]["total"],
        "removed_internal_channels": internal, "removed_output_channels": removed - internal,
        "validation_accuracy": physical["best_valid"]["metrics"]["valid_accuracy"],
        "test_accuracy": final["test_metrics"]["test_accuracy"],
        "test_scope": "exploratory frozen official TinyImageNet test",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--baseline-archive", type=Path,
                        default=Path("outputs/baselines/mobilenet_baseline_partial.tar.gz"))
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != repo:
        parser.error("Run from the repository root.")
    try:
        spec = json.loads((repo / MANIFEST).read_text())
        archive = args.baseline_archive.resolve()
        import_command = [".venv/bin/python", "scripts/prepare_mobilenet_reference.py",
                          "--archive", str(archive),
                          "--output-dir", str(repo / spec["reference_dir"])]
        if args.dry_run:
            print(shlex.join(import_command))
            for stage in build_plan(repo):
                print(shlex.join(stage.command))
            return 0
        protocol = validate_configs(repo)
        with single.exclusive_launcher(repo, output_root=Path(spec["output_root"])):
            # The historical metrics remain local input, outside version control.
            subprocess.run(import_command, cwd=repo, check=True)
            reference = validate_reference(repo, archive)
            plan = build_plan(repo, reference)
            completed = [validate_stage(stage) for stage in plan]
            if not all(completed):
                single.preflight(repo, config_name="mobilenet_onepass_thr05_gap8_12pp")
            output = repo / spec["output_root"]
            write_json(output / "ablation_plan.json", {
                "protocol": protocol, "new_epochs_total": 90, "epochs_per_model": 45,
                "reference": reference,
                "stages": [{"mode": stage.kind, "command": stage.command, "fingerprint": stage.fingerprint} for stage in plan],
            })
            print("Two independent ImageNet models; each30search+15trained-weight recovery;90new epochs total.", flush=True)
            for stage, done in zip(plan, completed):
                current = build_plan(repo, validate_reference(repo, archive))
                if [item.fingerprint for item in current] != [item.fingerprint for item in plan]:
                    raise RuntimeError("Code, configuration or reference changed after planning.")
                if done:
                    print(f"Reuse completed: {stage.directory}", flush=True)
                    continue
                stage.directory.mkdir(parents=True, exist_ok=False)
                single.write_receipt(stage, "running")
                print(shlex.join(stage.command), flush=True)
                subprocess.run(stage.command, cwd=repo, check=True)
                validate_stage(stage)
                single.write_receipt(stage, "complete")
            records = [comparison_record(stage) for stage in plan]
            write_json(output / "comparison.json", {"protocol": protocol, "results": records})
            with (output / "comparison.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            print(f"Done: {output / 'comparison.csv'}", flush=True)
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
