"""One-shot CIFAR-10 test evaluation of already selected pruning deployments.

No training, checkpoint selection, gate changes, or BatchNorm recalibration.
The original run is read-only; outputs go to a fresh test_evaluation directory.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from net_complexity.data.dataloaders import _build_cifar_transforms, _resolve_num_workers
from net_complexity.models.channel_pruning import build_structurally_pruned_model_from_config
from net_complexity.training.pruning_measurement import deployment_cost, mask_hash, state_hash, write_json

JOBS = ("J1_dense_control", "J2_output_fixed", "J3_internal_fixed", "J4_internal_random")
ADAPTIVE_JOBS = ("A1_internal_p18", "A2_internal_p12", "A3_internal_p23", "A4_output_p05")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_files(run_dir, job, dense_source=None):
    """Resolve reused J1, including a run tree relocated to another machine."""
    run_dir = Path(run_dir).resolve()
    job_dir = run_dir / job
    state = read_json(job_dir / "pilot_state.json")
    sources = [job_dir]
    if job == JOBS[0] and dense_source is not None:
        dense_source = Path(dense_source).resolve()
        sources = [dense_source / job, dense_source, job_dir]
    if state.get("reused_from"):
        original = Path(state["reused_from"])
        sources += [original, run_dir.parent / original.parent.name / job]
    source = next((p for p in sources if (p / "deployment.pt").is_file()), None)
    if source is None:
        hint = " Pass --dense-source PATH_TO_DENSE_RUN (or its J1 folder)." if job == JOBS[0] else ""
        raise FileNotFoundError(f"Missing {job}/deployment.pt. Checked: {sources}.{hint}")
    configs = [job_dir / "resolved_config.yaml", run_dir / f"{job}_resolved.yaml",
               source / "resolved_config.yaml"]
    config_path = next((p for p in configs if p.is_file()), None)
    if config_path is None:
        raise FileNotFoundError(f"No saved resolved config for {job}: {configs}")
    return state, source / "deployment.pt", config_path


def validate_config_and_mask(config, mask):
    expected = {
        "dataloaders.taskname": "CIFAR10", "model.backbone.num_classes": 10,
        "model.backbone.in_channels": 3, "model.backbone.stem_kernel_size": 3,
        "model.backbone.stem_stride": 1, "model.backbone.stem_padding": 1,
        "model.backbone.use_maxpool": False,
        "model.backbone._target_": "net_complexity.wrappers.ResNet50",
        "model.criterion._target_": "torch.nn.CrossEntropyLoss",
    }
    for key, value in expected.items():
        require(OmegaConf.select(config, key) == value, f"Unsupported saved config: {key}")
    counts = (3, 4, 6, 3)
    for name, indices in mask.items():
        match = re.fullmatch(r"backbone\.layer([1-4])\.(\d+)\.(gumbel_layer|mid[12]_gumbel_layer)", name)
        require(match is not None, f"Unknown mask key: {name}")
        stage, block = int(match[1]), int(match[2])
        suffix = match[3]
        require(block < counts[stage - 1], f"Invalid block: {name}")
        flag = "gate_output" if suffix == "gumbel_layer" else "gate_internal_width"
        require(bool(OmegaConf.select(config, f"model.backbone.resnet_block.{flag}")), f"Mask targets disabled gate: {name}")
        width = 64 * 2 ** (stage - 1) * (4 if suffix == "gumbel_layer" else 1)
        require(isinstance(indices, list) and len(indices) == len(set(indices))
                and all(type(i) is int and 0 <= i < width for i in indices)
                and len(indices) < width, f"Invalid channel indices: {name}")


def prepare_job(run_dir, job, dense_source=None):
    state, checkpoint_path, config_path = resolve_files(run_dir, job, dense_source)
    adaptive = state.get("pilot_version") == 2 and state.get("protocol") == "adaptive_lambda_v1"
    require(state["status"] == "completed" and (state.get("pilot_version") == 1 or adaptive),
            f"{job}: expected a completed audited fixed-v1 or adaptive-v2 deployment")
    require(job not in ADAPTIVE_JOBS or adaptive, f"{job}: an adaptive job cannot be labelled fixed-lambda")
    require(state["global_epochs_completed"] == state["total_epochs_allocated"], f"{job}: incomplete epoch budget")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    mask = checkpoint["pruning_mask"]
    require(mask == state["accepted_mask"] and mask_hash(mask) == checkpoint["mask_hash"]
            == state["accepted_mask_hash"], f"{job}: deployment mask differs from accepted mask")
    require(checkpoint["validation"] == state["validation"], f"{job}: selected deployment validation differs")
    require(checkpoint["common_init_hash"] == state["common_init_hash"], f"{job}: initializer differs")
    # A rollback may retain an earlier deployment while still consuming the full budget.
    require(0 < checkpoint["global_epochs_consumed"] <= state["global_epochs_completed"], f"{job}: invalid checkpoint epoch accounting")
    weights = checkpoint["model_state_dict"]
    require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in weights.values()), f"{job}: expected FP32 deployment")
    expected_hash = state_hash(weights)
    require(expected_hash == checkpoint["model_state_hash"], f"{job}: checkpoint tensor hash mismatch")
    config = OmegaConf.load(config_path)
    require(int(config.seed) == state["seed"], f"{job}: config seed differs")
    normalization = OmegaConf.select(config, "model.backbone.resnet_block.regularization_normalization",
                                     default="enabled_channels")
    require(normalization in ("enabled_channels", "initial_channels"),
            f"{job}: unsupported gate regularization normalization")
    require(state.get("gate_regularization_normalization", "enabled_channels") == normalization
            == checkpoint.get("gate_regularization_normalization", "enabled_channels"),
            f"{job}: gate regularization normalization differs between config/state/checkpoint")
    if normalization == "initial_channels":
        require(adaptive, f"{job}: initial_channels requires the adaptive training protocol")
        widths = state.get("initial_gate_channels")
        require(isinstance(widths, dict) and bool(widths)
                and all(isinstance(name, str) and type(width) is int and width > 0
                        for name, width in widths.items())
                and widths == checkpoint.get("initial_gate_channels"),
                f"{job}: original gate normalization widths missing or inconsistent")
    if adaptive:
        require(OmegaConf.select(config, "cyclic_channel_pruning.audit_protocol") == "adaptive_lambda_v1"
                and OmegaConf.select(config, "training_arguments.adaptive_lambda.enabled") is True
                and state.get("adaptive_lambda_enabled") is True,
                f"{job}: adaptive deployment has a non-adaptive saved configuration")
        require(state["global_epochs_completed"] <= 150, f"{job}: adaptive training exceeded 150 epochs")
    validate_config_and_mask(config, mask)
    structural_config = deepcopy(config)
    structural_config.model.lambda_coef = 0.0
    pruning = OmegaConf.create({"mode": "explicit", "structural": True, "enabled": True, "mask": mask})
    with redirect_stdout(io.StringIO()):
        model = build_structurally_pruned_model_from_config(structural_config, pruning)
    model.load_state_dict(weights, strict=True)
    model.eval()
    cost = deployment_cost(model)  # CPU, synthetic input, eval mode; never test data.
    for key in ("physical_total_parameters", "conv_linear_macs_per_image"):
        require(cost[key] == state["final_cost"][key], f"{job}: deployment {key} differs")
    require(state_hash(model.state_dict()) == expected_hash, f"{job}: loading/cost check changed model state")
    record = {
        "job": job, "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_hash(checkpoint_path),
        "pilot_version": state["pilot_version"], "training_protocol": state.get("protocol", "fixed_lambda_pilot_v1"),
        "gate_regularization_normalization": normalization,
        "model_state_hash": expected_hash, "mask_hash": checkpoint["mask_hash"],
        "config": str(config_path), "config_sha256": file_hash(config_path),
        "state_sha256": file_hash(Path(run_dir) / job / "pilot_state.json"),
        "seed": state["seed"], "common_init_hash": state["common_init_hash"],
        "split_indices_hash": state["split_indices_hash"],
        "epochs_consumed": state["global_epochs_completed"],
        "optimizer_steps_total": state["optimizer_steps_total"],
        "deployment_epochs_consumed": checkpoint["global_epochs_consumed"],
        "selection_provenance": checkpoint["provenance"], "validation": state["validation"],
        "physical_parameters": cost["physical_total_parameters"],
        "conv_linear_macs_per_image": cost["conv_linear_macs_per_image"],
    }
    print(f"[check] {job}: {record['physical_parameters']:,} physical parameters; {checkpoint_path}", flush=True)
    return model, record


def build_test_loader(data_dir, batch_size, num_workers, device, download=False):
    _, test_transform = _build_cifar_transforms()
    # Do not construct a training/validation dataset, let alone calibrate BN on test.
    dataset = CIFAR10(root=str(data_dir), train=False, transform=test_transform, download=download)
    require(len(dataset) == 10000, "Expected all 10,000 official CIFAR-10 test images")
    digest = hashlib.sha256(dataset.data.tobytes())
    digest.update(np.asarray(dataset.targets, dtype="<i8").tobytes())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                        num_workers=_resolve_num_workers(num_workers), pin_memory=str(device).startswith("cuda"))
    info = {"dataset": "CIFAR10", "split": "official_test", "example_count": len(dataset),
            "ordered_data_and_labels_sha256": digest.hexdigest(), "transform": repr(test_transform),
            "classes": dataset.classes, "shuffle": False, "drop_last": False}
    return loader, info


@torch.inference_mode()
def evaluate_fixed(model, loader, device, *, job, expected_examples=10000):
    original_hash = state_hash(model.state_dict())
    model.to(device).eval()
    labels, predictions, probabilities, losses = [], [], [], []
    seen = 0
    started = time.perf_counter()
    for batch, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        logits = model(x, y).logits
        require(logits.shape == (len(y), 10), f"{job}: invalid classifier output shape")
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"{job}: non-finite test logits")
        per_example_ce = F.cross_entropy(logits, y, reduction="none")
        if not torch.isfinite(per_example_ce).all():
            raise FloatingPointError(f"{job}: non-finite test CE")
        labels.append(y.cpu().numpy())
        predictions.append(logits.argmax(-1).cpu().numpy())
        probabilities.append(logits.softmax(-1).cpu().numpy())
        losses.append(per_example_ce.cpu().numpy())
        seen += len(y)
        if batch % 20 == 0 or seen == expected_examples:
            print(f"[test] {job}: {seen}/{expected_examples}", flush=True)
    require(seen == expected_examples and seen > 0, f"{job}: incomplete test pass ({seen}/{expected_examples})")
    arrays = {"index": np.arange(seen), "label": np.concatenate(labels),
              "prediction": np.concatenate(predictions), "probabilities": np.concatenate(probabilities),
              "ce_loss": np.concatenate(losses)}
    correct = int((arrays["prediction"] == arrays["label"]).sum())
    result = {"accuracy": correct / seen, "ce_loss": float(arrays["ce_loss"].mean(dtype=np.float64)),
              "correct_count": correct, "example_count": seen, "wall_seconds": time.perf_counter() - started}
    model.cpu()
    require(state_hash(model.state_dict()) == original_hash, f"{job}: inference changed model weights or BN buffers")
    return result, arrays


def write_comparison(output, records):
    dense = next((r for r in records if r["job"] == JOBS[0]), None)
    rows = []
    lines = ["# Frozen pruning deployments — CIFAR-10 test", "",
             "No training, model selection, or BN recalibration. One fixed checkpoint per job.", "",
             "| Job | Test accuracy | Validation accuracy | Parameters | Conv/Linear GMAC |",
             "|---|---:|---:|---:|---:|"]
    if any(r.get("training_protocol") == "adaptive_lambda_v1" for r in records):
        lines[4:4] = ["Exploratory comparison: prior test results informed further experimentation.", ""]
    for r in records:
        rows.append({"Run": r["job"], "model.trainable_parameters": r["physical_parameters"],
                     "gate_regularization_normalization": r.get("gate_regularization_normalization", "enabled_channels"),
                     "test_accuracy": r["test"]["accuracy"], "validation_accuracy": r["validation"]["accuracy"],
                     "test_ce_loss": r["test"]["ce_loss"], "test_correct_count": r["test"]["correct_count"],
                     "test_example_count": r["test"]["example_count"],
                     "conv_linear_macs_per_image": r["conv_linear_macs_per_image"],
                     "test_delta_vs_dense_pp": 100 * (r["test"]["accuracy"] - dense["test"]["accuracy"]) if dense else None,
                     "checkpoint_sha256": r["checkpoint_sha256"]})
        lines.append(f"| {r['job']} | {r['test']['accuracy']:.2%} | {r['validation']['accuracy']:.2%} | "
                     f"{r['physical_parameters']:,} | {r['conv_linear_macs_per_image'] / 1e9:.3f} |")
    with (output / "test_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "test_comparison.md").write_text("\n".join(lines) + "\n")


def run(args):
    run_dir = args.run_dir.resolve()
    output = (args.output or run_dir / "test_evaluation").resolve()
    if not args.check_only and output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}. Use a fresh --output directory.")
    if not args.check_only and str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run on the GPU server, or explicitly pass --device cpu.")
    # Freeze and validate ALL chosen deployments before touching any test image.
    prepared = [prepare_job(run_dir, job, args.dense_source) for job in args.jobs]
    reference = prepared[0][1]
    for _, record in prepared:
        for key in ("seed", "common_init_hash", "split_indices_hash", "epochs_consumed", "optimizer_steps_total"):
            require(record[key] == reference[key], f"{record['job']}: unmatched {key}")
    if args.check_only:
        print("All checkpoints verified. Test data not loaded; no outputs written.")
        return None
    try:
        loader, dataset_info = build_test_loader(args.data, args.batch_size, args.num_workers, args.device, args.download)
    except RuntimeError as exc:
        raise RuntimeError(f"Could not load CIFAR-10 test from {args.data}. Check --data; use --download if necessary.") from exc
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "source_run": str(run_dir), "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "frozen_deployments_test_v1", "test_evaluated": False,
              "comparison_scope": ("exploratory" if any(r["pilot_version"] == 2 for _, r in prepared)
                                   else "frozen_deployment_evaluation"),
              "training_performed": False, "bn_recalibration": False, "test_based_selection": False,
              "device": str(args.device), "batch_size": args.batch_size, "precision": "fp32",
              "python": sys.version, "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else None,
              "evaluation_script_sha256": file_hash(__file__), "dataset": dataset_info,
              "planned_jobs": list(args.jobs), "runs": []}
    write_json(output / "evaluation_plan.json", {**report, "selected_deployments": [r for _, r in prepared]})
    write_json(output / "test_summary.json", report)
    print(f"Results: {output}", flush=True)
    try:
        for model, record in prepared:
            metrics, arrays = evaluate_fixed(model, loader, args.device, job=record["job"])
            require(np.array_equal(arrays["label"], np.asarray(loader.dataset.targets)), "Test sample order differs")
            prediction_path = output / f"{record['job']}_predictions.npz"
            np.savez_compressed(prediction_path, **arrays)
            row = {**record, "test": metrics, "predictions": prediction_path.name,
                   "predictions_sha256": file_hash(prediction_path), "model_state_unchanged": True}
            report["runs"].append(row)
            report["test_evaluated"] = True
            write_json(output / "test_summary.json", report)
            write_comparison(output, report["runs"])
            print(f"{record['job']}: test={metrics['accuracy']:.2%}, valid={record['validation']['accuracy']:.2%}, "
                  f"params={record['physical_parameters']:,}; {metrics['wall_seconds']:.1f}s", flush=True)
        report.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        report.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "test_summary.json", report)
    print(f"Completed. See {output / 'test_comparison.csv'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed nightly run directory")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--jobs", nargs="+", choices=JOBS + ADAPTIVE_JOBS, default=list(JOBS))
    parser.add_argument("--dense-source", type=Path, help="Original dense parent/J1 folder, if relocated")
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/test_evaluation; must not exist")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--download", action="store_true", help="Allow CIFAR-10 download if absent")
    parser.add_argument("--check-only", action="store_true", help="CPU checkpoint checks; no test access or writes")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or len(args.jobs) != len(set(args.jobs)):
        parser.error("Use positive batch size, nonnegative workers, and distinct jobs")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    run(args)


if __name__ == "__main__":
    main()
