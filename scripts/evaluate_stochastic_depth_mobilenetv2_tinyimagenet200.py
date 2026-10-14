"""Reevaluate saved SD checkpoints with actual branch skipping and frozen BN."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from net_complexity.evaluation.stochastic_depth import (
    evaluate_stochastic_depth,
    summarize_stochastic_repeats,
)


def write_report(output: Path, report: dict) -> None:
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "summary.json")
    rows = []
    for run in report["runs"]:
        expected = run["expected"]
        sampled = run["stochastic_summary"]
        rows.append(
            {
                "variant": run["variant"],
                "checkpoint_epoch": run["checkpoint_epoch"],
                "stored_parameters": expected["stored_parameters"],
                "expected_accuracy": expected["accuracy"],
                "stochastic_accuracy_mean": sampled["accuracy_mean"],
                "stochastic_accuracy_std": sampled["accuracy_std"],
                "stochastic_mean_active_parameters": sampled[
                    "mean_active_parameters_mean"
                ],
                "stochastic_mean_active_blocks": sampled["mean_active_blocks_mean"],
                "stochastic_mean_forward_flops_per_image": sampled[
                    "mean_forward_flops_per_image_mean"
                ],
                "full_forward_flops_per_image": expected[
                    "full_forward_flops_per_image"
                ],
            }
        )
    if rows:
        with (output / "comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(args) -> dict:
    study = args.study_dir.expanduser().resolve()
    study_summary = json.loads((study / "summary.json").read_text())
    if (
        study_summary["failed_trials"]
        or study_summary["pruned_trials"]
        or study_summary["completed_trials"] != study_summary["effective_trials"]
    ):
        raise ValueError("Use a completed study with all checkpoints finalized.")
    run_dirs = sorted((study / "runs").glob("*/summary.json"))
    if not run_dirs or len(run_dirs) != study_summary["completed_trials"]:
        raise ValueError("Run summaries do not match the completed study trial count.")
    prepared = []
    for summary_path in run_dirs:
        run_dir = summary_path.parent
        checkpoint = run_dir / "checkpoints/best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing {checkpoint}. Run on the server with the original "
                "checkpoints, not the downloaded archive without weights."
            )
        summary = json.loads(summary_path.read_text())
        config = OmegaConf.load(run_dir / "config_resolved.yaml")
        if (
            summary["runtime"]["test_evaluation"]["checkpoint_epoch"]
            != summary["best_valid"]["epoch"]
        ):
            raise ValueError(f"Checkpoint selection metadata differs: {run_dir}")
        if (
            config.model.backbone._target_.split(".")[-1]
            != "StochasticDepthMobileNetV2"
        ):
            raise ValueError(f"Expected a stochastic-depth MobileNetV2 run: {run_dir}")
        prepared.append((run_dir, checkpoint, summary, config))
    prepared.sort(key=lambda item: item[2]["runtime"]["optuna_trial_number"])
    data_signature = OmegaConf.to_container(prepared[0][3].dataloaders, resolve=True)
    if any(
        OmegaConf.to_container(item[3].dataloaders, resolve=True) != data_signature
        for item in prepared[1:]
    ):
        raise ValueError(
            "All runs must use the same data split and evaluation pipeline."
        )
    if args.check_only:
        print(
            f"{len(prepared)} finalized run directories found; no data or weights loaded."
        )
        return {"status": "paths_checked", "runs": len(prepared)}
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose the GPU server or --device cpu.")
    data_config = OmegaConf.create(data_signature)
    data_config.num_workers = args.num_workers
    if args.batch_size is not None:
        data_config.batch_size = args.batch_size
    if args.data is not None:
        data_config.path_to_data = str(args.data.expanduser().resolve())
    dataloaders = instantiate(data_config)
    loader = dataloaders.test_dataloader
    if not args.smoke and (
        len(loader.dataset) != 10000
        or data_config.num_classes != 200
        or data_config.taskname != "tinyimagenet200"
    ):
        raise ValueError("Expected the 10,000-image TinyImageNet-200 test split.")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else study
        / "stochastic_inference"
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "source_study": str(study),
        "evaluation_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "dataset": "synthetic_smoke" if args.smoke else "TinyImageNet-200_official_val",
        "device": str(args.device),
        "precision": "fp32",
        "batch_size": data_config.batch_size,
        "num_workers": args.num_workers,
        "data_path": str(data_config.path_to_data),
        "eval_transform": OmegaConf.to_container(
            data_config.eval_transform, resolve=True
        ),
        "accuracy_aggregation": "total_correct_divided_by_total_examples",
        "stochastic_mask": "one_independent_Bernoulli_per_branch_per_batch",
        "active_branch_scaling": "none_in_stochastic_mode",
        "training_performed": False,
        "batchnorm_updates": False,
        "classifier_dropout": False,
        "ensemble_predictions": False,
        "physical_export": False,
        "comparison_scope": "exploratory_custom_stochastic_inference",
        "checkpoint_selection": "existing_best_from_legacy_batch_mean_validation",
        "validation_selection_repeated": False,
        "std_convention": "sample_standard_deviation_across_independent_passes",
        "flops_convention": "Conv/Linear forward operations, 2*MAC plus bias additions",
        "active_parameters_convention": "sample_weighted_executed_parameters_not_storage_size",
        "seeds": list(range(args.seed, args.seed + args.repeats)),
        "runs": [],
    }
    write_report(output, report)
    print(f"[SD inference] output={output}", flush=True)
    try:
        for run_dir, checkpoint, summary, config in prepared:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload.get("epoch") != summary["best_valid"]["epoch"]:
                raise ValueError(
                    f"Checkpoint epoch differs from saved selection: {checkpoint}"
                )
            del payload
            model_config = OmegaConf.create(
                OmegaConf.to_container(config.model.backbone, resolve=True)
            )
            model_config.pretrained_checkpoint = str(checkpoint)
            model_config.require_pretrained = True
            model_config.eval_mode = "expected"
            model = instantiate(model_config).to(args.device)
            variant = str(config.mlflow.tags.sd_variant)
            expected = evaluate_stochastic_depth(
                model, loader, device=args.device, mode="expected", seed=args.seed
            )
            print(
                f"[{variant}] expected: accuracy={expected['accuracy']:.4%}", flush=True
            )
            repeats = []
            for seed in report["seeds"]:
                row = evaluate_stochastic_depth(
                    model, loader, device=args.device, mode="stochastic", seed=seed
                )
                repeats.append(row)
                print(
                    f"[{variant}] stochastic seed={seed}: accuracy={row['accuracy']:.4%}; "
                    f"active_params={row['mean_active_parameters']:.0f}; "
                    f"active_blocks={row['mean_active_blocks']:.3f}",
                    flush=True,
                )
            report["runs"].append(
                {
                    "variant": variant,
                    "source_run": str(run_dir),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    "checkpoint_epoch": summary["best_valid"]["epoch"],
                    "source_provenance": summary.get("provenance"),
                    "survival_schedule": str(model.stochastic_depth_survival_schedule),
                    "survival_probabilities": list(
                        model.stochastic_depth_survival_probabilities
                    ),
                    "expected": expected,
                    "stochastic_repeats": repeats,
                    "stochastic_summary": summarize_stochastic_repeats(repeats),
                }
            )
            write_report(output, report)
            del model
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed_or_interrupted"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_report(output, report)
    print(f"[SD inference] completed: {output / 'comparison.csv'}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Allow synthetic test fixtures"
    )
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.num_workers < 0
        or (args.batch_size is not None and args.batch_size < 1)
    ):
        parser.error("Use positive repeats/batch-size and nonnegative num-workers.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    run(args)


if __name__ == "__main__":
    main()
