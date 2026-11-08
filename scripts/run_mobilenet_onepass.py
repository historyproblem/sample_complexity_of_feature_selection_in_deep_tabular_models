"""Import the accepted historical reference and run MobileNet search + recovery.

Run from the repository root with .venv/bin/python. --dry-run requires only
the Python standard library and prints the exact commands without training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


CONFIG_NAME = "mobilenet_onepass_thr05_gap05_1pp"
DEFAULT_CAPS = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6")
OUTPUT_ROOT = Path("outputs/runs") / CONFIG_NAME
REFERENCE_DIRECTORY = "imported_dense_reference_20261002"
EXPECTED_ARCHIVE_SHA256 = "7ada53ae4c56902dcfc48d3625273d66955dd8b9ca7702727d9854d342bf9cfd"


@dataclass(frozen=True)
class Experiment:
    soft_drop: float
    hard_drop: float
    default_caps: tuple[str | None, ...]
    auto_step_no_boost: bool = False


EXPERIMENTS = {
    CONFIG_NAME: Experiment(0.005, 0.01, DEFAULT_CAPS),
    "mobilenet_onepass_thr05_gap6_7pp": Experiment(0.06, 0.07, (None,)),
    "mobilenet_onepass_thr05_gap7_8pp": Experiment(0.07, 0.08, (None,), auto_step_no_boost=True),
}


@dataclass(frozen=True)
class Stage:
    kind: str
    directory: Path
    command: list[str]
    fingerprint: str


def build_plan(
    repo_root: Path, caps: list[str | None], reference_identity: dict | None = None,
    *, config_name: str = CONFIG_NAME,
) -> list[Stage]:
    """Build immutable stage identities, including code and config digests."""
    experiment = EXPERIMENTS[config_name]
    config_path = repo_root / "configs" / f"{config_name}.yaml"
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
    ).strip()
    baseline_dir = repo_root / OUTPUT_ROOT / REFERENCE_DIRECTORY
    commands = []
    for cap in caps:
        cap_label = "nocap" if cap is None else f"cap{cap}"
        cap_value = "null" if cap is None else cap
        name = (
            f"mobilenet_onepass_thr0.5_soft{experiment.soft_drop:g}"
            f"_hard{experiment.hard_drop:g}_{cap_label}"
        )
        if experiment.auto_step_no_boost:
            name += "_autostep_noboost"
        directory = repo_root / "outputs/runs" / config_name / name
        command = [
            ".venv/bin/python", "-u", "src/net_complexity/cli/cyclic_train.py",
            "--config-path", str(repo_root / "configs"),
            "--config-name", config_name,
            "training_arguments.adaptive_lambda.baseline_history_dir="
            + json.dumps(str(baseline_dir)),
            f"cyclic_channel_pruning.threshold_max_param_fraction={cap_value}",
            f"mlflow.run_name={name}",
            f"hydra.run.dir={json.dumps(str(directory))}",
        ]
        commands.append(("onepass", directory, command))
    result = []
    for kind, directory, command in commands:
        identity = json.dumps({
            "kind": kind, "command": command, "config_sha256": config_hash,
            "runner_sha256": runner_hash, "commit": revision,
            "reference": reference_identity or {"archive_sha256": EXPECTED_ARCHIVE_SHA256},
        }, sort_keys=True)
        result.append(Stage(kind, directory, command, hashlib.sha256(identity.encode()).hexdigest()))
    return result


def validate_history(path: Path, expected_epochs: int) -> None:
    """Require a complete validation history, rather than a partial run."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(float(row["epoch"])) for row in rows]
    if epochs != list(range(1, expected_epochs + 1)):
        raise RuntimeError(f"Expected exactly epochs 1–{expected_epochs} in {path}; refusing reuse")
    if not all(
        math.isfinite(float(row["valid_accuracy"]))
        and 0 <= float(row["valid_accuracy"]) <= 1
        for row in rows
    ):
        raise RuntimeError(f"Invalid validation accuracy in {path}")


def validate_imported_reference(reference_dir: Path, archive_path: Path) -> dict:
    """Keep the full historical curve and identify its actual training recipe."""
    provenance_path = reference_dir / "reference_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    expected = {
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "actual_training_horizon_epochs": 150,
        "actual_cosine_T_max": 150,
        "available_history_epochs": 86,
        "selected_reference_epochs": [1, 30],
        "training_performed": False,
    }
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("The supplied archive differs from the historical reference selected by the user")
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise RuntimeError(f"Imported reference has unexpected {key}; refusing automatic baseline training")
    history_path = reference_dir / "history.csv"
    validate_history(history_path, 86)
    history_hash = hashlib.sha256(history_path.read_bytes()).hexdigest()
    if history_hash != provenance["history_sha256"]:
        raise RuntimeError("Imported reference history checksum differs")
    return {
        "archive_sha256": archive_hash,
        "history_sha256": history_hash,
        "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "available_history_epochs": 86,
        "selected_reference_epochs": [1, 30],
        "actual_training_horizon_epochs": 150,
        "actual_cosine_T_max": 150,
    }


def validate_existing_stage(stage: Stage) -> bool:
    """Reuse only this launcher's completed matching stage; never restart one."""
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
        if summary["num_cycles_completed"] != 1 or summary["training_cost"]["num_epochs_executed"] != 75:
            raise RuntimeError("Single-pass run did not finish exactly 30+45 epochs")
        if len(summary["cycle_results"]) != 1 or len(summary["recovery_results"]) != 1:
            raise RuntimeError("Unexpected extra search/recovery stages")
        if summary["cycle_results"][0]["num_epochs_executed"] != 30:
            raise RuntimeError("Search did not finish exactly 30 epochs")
        if summary["recovery_results"][0]["num_epochs_executed"] != 45:
            raise RuntimeError("Recovery did not finish exactly 45 epochs")
        final_dir = Path(summary["final_result"]["run_dir"])
        validate_history(final_dir / "history.csv", 45)
        if not (final_dir / "checkpoints/best.pt").is_file():
            raise RuntimeError("Recovery checkpoint missing")
        if not summary["final_result"]["test_metrics"]:
            raise RuntimeError("Recovery has no frozen-checkpoint test results")
    except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Stage is incomplete or invalid: {stage.directory}; refusing to repeat training. {exc}"
        ) from exc
    return True


def preflight(repo_root: Path, *, config_name: str = CONFIG_NAME) -> None:
    """Check the intended config, CUDA wheel compatibility and extracted data."""
    if Path(sys.prefix).resolve() != (repo_root / ".venv").resolve():
        raise RuntimeError("Run this command through the repository .venv/bin/python")
    from omegaconf import OmegaConf

    experiment = EXPERIMENTS[config_name]
    config = OmegaConf.load(repo_root / "configs" / f"{config_name}.yaml")
    expected = {
        "dataloaders.num_workers": 0,
        "cyclic_channel_pruning.max_cycles": 1,
        "cyclic_channel_pruning.g_prob_threshold": 0.5,
        "cyclic_channel_pruning.threshold_max_param_fraction": (
            None if experiment.default_caps[0] is None else float(experiment.default_caps[0])
        ),
        "cyclic_channel_pruning.gumbel_epochs": 30,
        "cyclic_channel_pruning.final_epochs": 45,
        "training_arguments.adaptive_lambda.enabled": True,
        "training_arguments.adaptive_lambda.soft_drop": experiment.soft_drop,
        "training_arguments.adaptive_lambda.hard_drop": experiment.hard_drop,
        "model.criterion.label_smoothing": 0,
        "model.backbone.block.gate_internal_width": True,
    }
    if experiment.auto_step_no_boost:
        expected.update({
            "training_arguments.num_epochs": 30,
            "training_arguments.adaptive_lambda.warmup_epochs": 3,
            "training_arguments.adaptive_lambda.update_every_epochs": 1,
            "training_arguments.adaptive_lambda.log_step_init": "auto",
            "training_arguments.adaptive_lambda.adaptive_log_step_enabled": False,
        })
    for key, value in expected.items():
        if OmegaConf.select(config, key) != value:
            raise RuntimeError(f"Unexpected {key}: expected {value}")
    print("DataLoader preflight: num_workers=0; loading batches in the main process", flush=True)
    data_root = repo_root / "data/tiny-imagenet-200"
    for relative in ("wnids.txt", "train", "val/images", "val/val_annotations.txt"):
        if not (data_root / relative).exists():
            raise RuntimeError(f"Extract TinyImageNet first; missing {data_root / relative}")
    wnids = (data_root / "wnids.txt").read_text(encoding="utf-8").split()
    if len(set(wnids)) != 200:
        raise RuntimeError("TinyImageNet must have200distinct classes")
    for wnid in wnids:
        images = data_root / "train" / wnid / "images"
        if not images.is_dir() or not any(images.iterdir()):
            raise RuntimeError(f"Missing train images: {images}")

    import torch
    import torchvision

    if torch.__version__.split("+")[0] != "2.6.0" or torchvision.__version__.split("+")[0] != "0.21.0":
        raise RuntimeError("Install the verified torch2.6.0/torchvision0.21.0 CUDA stack first")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback for training")
    if torch.cuda.get_device_capability(0) == (7, 0) and "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("This PyTorch wheel lacks V100(sm70) support")
    tensor = torch.ones((1, 3, 16, 16), device="cuda:0", requires_grad=True)
    kernel = torch.ones((4, 3, 3, 3), device="cuda:0", requires_grad=True)
    torch.nn.functional.conv2d(tensor, kernel).mean().backward()
    torchvision.ops.nms(
        torch.tensor([[0., 0., 1., 1.]], device="cuda:0"),
        torch.tensor([1.], device="cuda:0"), 0.5,
    )
    torch.cuda.synchronize()
    print(f"CUDA preflight passed: {torch.cuda.get_device_name(0)}; torch{torch.__version__}", flush=True)


def write_receipt(stage: Stage, status: str) -> None:
    target = stage.directory / "launcher_receipt.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "kind": stage.kind, "fingerprint": stage.fingerprint,
        "status": status, "command": stage.command,
    }, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


@contextmanager
def exclusive_launcher(repo_root: Path, *, output_root: Path = OUTPUT_ROOT):
    """Prevent simultaneous launchers; OS releases the lock after an exit."""
    import fcntl

    root = repo_root / output_root
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".launcher.lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another MobileNet launcher holds the output lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without importing torch or training")
    parser.add_argument("--config-name", choices=tuple(EXPERIMENTS), default=CONFIG_NAME)
    parser.add_argument("--caps", nargs="+", choices=DEFAULT_CAPS)
    parser.add_argument(
        "--baseline-archive", type=Path,
        help="Accepted historical mobilenet_baseline_partial.tar.gz (default: repository root)",
    )
    args = parser.parse_args(argv)
    if args.caps is None:
        args.caps = list(EXPERIMENTS[args.config_name].default_caps)
    if len(set(args.caps)) != len(args.caps):
        parser.error("--caps must not repeat a profile")
    repo_root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != repo_root:
        parser.error("Run from the repository root")
    archive_path = (args.baseline_archive or repo_root / "mobilenet_baseline_partial.tar.gz").resolve()
    reference_dir = repo_root / OUTPUT_ROOT / REFERENCE_DIRECTORY
    import_command = [
        ".venv/bin/python", "scripts/prepare_mobilenet_reference.py",
        "--archive", str(archive_path), "--output-dir", str(reference_dir),
    ]
    try:
        if args.dry_run:
            plan = build_plan(repo_root, args.caps, config_name=args.config_name)
            print(shlex.join(import_command))
            for stage in plan:
                print(shlex.join(stage.command))
            return 0
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"Upload the selected baseline archive to {archive_path}, or set --baseline-archive. "
                "No new dense baseline will be trained."
            )
        with exclusive_launcher(repo_root, output_root=Path("outputs/runs") / args.config_name):
            print(shlex.join(import_command), flush=True)
            subprocess.run(import_command, cwd=repo_root, check=True)
            reference_identity = validate_imported_reference(reference_dir, archive_path)
            plan = build_plan(repo_root, args.caps, reference_identity, config_name=args.config_name)
            print(
                "Reference: imported all 86 history epochs from the historical 150-epoch/cosine150 recipe; "
                "search uses matching epochs 1–30. No new reference training or weight transfer.",
                flush=True,
            )
            preflight(repo_root, config_name=args.config_name)
            completed = {stage.directory: validate_existing_stage(stage) for stage in plan}
            for stage in plan:
                if validate_imported_reference(reference_dir, archive_path) != reference_identity:
                    raise RuntimeError("Imported reference changed after planning; refusing to start training")
                if completed[stage.directory]:
                    print(f"Reuse completed stage: {stage.directory}", flush=True)
                    continue
                stage.directory.mkdir(parents=True, exist_ok=False)
                write_receipt(stage, "running")
                print(shlex.join(stage.command), flush=True)
                subprocess.run(stage.command, cwd=repo_root, check=True)
                validate_existing_stage(stage)
                write_receipt(stage, "complete")
        print("Done: historical reference imported; each selected profile uses 30 search + 45 recovery = 75 new epochs.", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
