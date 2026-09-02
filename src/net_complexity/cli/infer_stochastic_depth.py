from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Any

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from net_complexity.metrics.base import Multimetric
from net_complexity.models.stochastic_depth import get_stochastic_depth_blocks
from net_complexity.training.randomness import set_random_seed


CONFIGS_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@dataclass(frozen=True)
class CheckpointTarget:
    study_dir: Path
    run_dir: Path
    checkpoint_path: Path
    config_path: Path


def _resolve_path(path: str | Path, root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = root / resolved
    return resolved.resolve()


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _run_dirs_from_source(source: Path) -> tuple[Path, list[Path]]:
    if (source / "config_resolved.yaml").is_file():
        study_dir = source.parent.parent if source.parent.name == "runs" else source.parent
        return study_dir, [source]

    runs_dir = source / "runs" if (source / "runs").is_dir() else source
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Stochastic-depth inference source does not exist: {source}")

    run_dirs = sorted(
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "config_resolved.yaml").is_file()
    )
    study_dir = runs_dir.parent if runs_dir.name == "runs" else source
    return study_dir, run_dirs


def discover_checkpoint_targets(
    sources: Iterable[str | Path],
    *,
    checkpoint_name: str,
    root: Path,
) -> list[CheckpointTarget]:
    checkpoint_relative = Path(checkpoint_name)
    if checkpoint_relative.is_absolute() or ".." in checkpoint_relative.parts:
        raise ValueError("checkpoint_name must be a relative path inside checkpoints/.")

    targets: list[CheckpointTarget] = []
    seen: set[Path] = set()
    for source_value in sources:
        source = _resolve_path(source_value, root)
        study_dir, run_dirs = _run_dirs_from_source(source)
        for run_dir in run_dirs:
            checkpoint_path = run_dir / "checkpoints" / checkpoint_relative
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint '{checkpoint_name}' is missing for run: {run_dir}"
                )
            checkpoint_path = checkpoint_path.resolve()
            if checkpoint_path in seen:
                continue
            seen.add(checkpoint_path)
            targets.append(
                CheckpointTarget(
                    study_dir=study_dir.resolve(),
                    run_dir=run_dir.resolve(),
                    checkpoint_path=checkpoint_path,
                    config_path=(run_dir / "config_resolved.yaml").resolve(),
                )
            )

    if not targets:
        raise FileNotFoundError("No stochastic-depth checkpoints were discovered.")
    return targets


def _resolve_device(requested_device: str | None) -> str:
    if requested_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    requested_device = str(requested_device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; falling back from {requested_device} to cpu.")
        return "cpu"
    return requested_device


def _set_stochastic_depth_inference_mode(
    model: torch.nn.Module,
    eval_mode: str,
) -> dict[str, torch.nn.Module]:
    eval_mode = str(eval_mode)
    if eval_mode not in {"expected", "stochastic"}:
        raise ValueError(
            "inference.eval_mode must be either 'expected' or 'stochastic'."
        )

    model.eval()
    blocks = get_stochastic_depth_blocks(model)
    if not blocks:
        raise ValueError("The loaded model does not contain stochastic-depth blocks.")
    for block in blocks.values():
        # Set only the block's own flag. Its BatchNorm children remain in eval,
        # while the block forward follows the train-time Bernoulli skip path.
        block.training = eval_mode == "stochastic"
        block.eval_mode = eval_mode

    batchnorm_modules = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if any(module.training for module in batchnorm_modules):
        raise RuntimeError("BatchNorm must remain in eval mode during inference.")
    return blocks


def _build_model(
    target: CheckpointTarget,
    *,
    eval_mode: str,
    device: str,
    strict: bool,
) -> tuple[torch.nn.Module, Mapping[str, Any], DictConfig]:
    run_config = OmegaConf.load(target.config_path)
    model_config = OmegaConf.create(
        OmegaConf.to_container(run_config.model, resolve=True)
    )
    model = instantiate(model_config).to(device)
    checkpoint = torch.load(
        target.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    _set_stochastic_depth_inference_mode(model, eval_mode)
    return model, checkpoint, run_config


@torch.inference_mode()
def _evaluate_once(
    model: torch.nn.Module,
    dataloader,
    metrics_config: DictConfig,
    *,
    device: str,
) -> tuple[dict[str, float], int]:
    metric_bundle = instantiate(metrics_config)
    metric = Multimetric(metric_bundle.test_metrics, prefix="test")
    blocks = get_stochastic_depth_blocks(model)
    eval_modes = {getattr(block, "eval_mode", "expected") for block in blocks.values()}
    if len(eval_modes) != 1:
        raise RuntimeError(f"Inconsistent stochastic-depth eval modes: {eval_modes}")
    _set_stochastic_depth_inference_mode(model, eval_modes.pop())
    num_batches = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs, targets)
        metric.update(inputs, output, targets, model)
        num_batches += 1
    return dict(metric.compute()), num_batches


def _numeric(value: Any) -> float | int | str | bool | None:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata_keys = {
        "study",
        "run_id",
        "run_dir",
        "checkpoint",
        "checkpoint_epoch",
        "p_L",
        "eval_mode",
        "num_test_batches",
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["run_dir"]), []).append(row)

    aggregated: list[dict[str, Any]] = []
    for run_rows in grouped.values():
        first = run_rows[0]
        summary = {key: first[key] for key in metadata_keys if key in first}
        summary["num_repeats"] = len(run_rows)
        metric_keys = [
            key
            for key, value in first.items()
            if key not in metadata_keys
            and key not in {"repeat", "inference_seed", "duration_sec"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        for key in metric_keys:
            values = [float(row[key]) for row in run_rows]
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary["duration_sec_total"] = sum(
            float(row["duration_sec"]) for row in run_rows
        )
        aggregated.append(summary)
    return aggregated


def run_inference(config: DictConfig) -> dict[str, Any]:
    root = Path(get_original_cwd()).resolve()
    inference_config = config.inference
    device = _resolve_device(getattr(config, "device", None))
    targets = discover_checkpoint_targets(
        inference_config.sources,
        checkpoint_name=str(inference_config.checkpoint_name),
        root=root,
    )
    num_repeats = int(inference_config.num_repeats)
    if num_repeats <= 0:
        raise ValueError("inference.num_repeats must be positive.")

    dataloaders = instantiate(config.dataloaders)
    output_dir = _resolve_path(inference_config.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    print(
        f"Stochastic-depth inference | checkpoints={len(targets)}"
        f" | repeats={num_repeats} | device={device}"
    )
    for target_index, target in enumerate(targets, start=1):
        model, checkpoint, run_config = _build_model(
            target,
            eval_mode=str(inference_config.eval_mode),
            device=device,
            strict=bool(inference_config.strict_checkpoint_loading),
        )
        p_l = OmegaConf.select(
            run_config,
            "model.backbone.final_survival_probability",
        )
        if p_l is None:
            raise ValueError(
                "Run config is missing model.backbone.final_survival_probability: "
                f"{target.config_path}"
            )
        for repeat_index in range(num_repeats):
            inference_seed = int(inference_config.seed_base) + repeat_index
            set_random_seed(inference_seed)
            started_at = perf_counter()
            metrics, num_batches = _evaluate_once(
                model,
                dataloaders.test_dataloader,
                config.metrics,
                device=device,
            )
            duration_sec = perf_counter() - started_at
            row = {
                "study": target.study_dir.name,
                "run_id": target.run_dir.name,
                "run_dir": _display_path(target.run_dir, root),
                "checkpoint": _display_path(target.checkpoint_path, root),
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "p_L": float(p_l),
                "eval_mode": str(inference_config.eval_mode),
                "repeat": repeat_index + 1,
                "inference_seed": inference_seed,
                "num_test_batches": num_batches,
                "duration_sec": duration_sec,
                **{key: _numeric(value) for key, value in metrics.items()},
            }
            rows.append(row)
            actual_active_blocks = metrics.get(
                "test_stochastic_depth_actual_inference_active_blocks",
                float("nan"),
            )
            print(
                f"[{target_index}/{len(targets)}] {target.run_dir.name}"
                f" | repeat={repeat_index + 1}/{num_repeats}"
                f" | accuracy={float(metrics.get('test_accuracy', float('nan'))):.6f}"
                f" | active_blocks={float(actual_active_blocks):.3f}"
                f" | duration={duration_sec:.1f}s"
            )
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate_rows = _aggregate_rows(rows)
    per_repeat_path = output_dir / "per_repeat.csv"
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "inference.json"
    _write_csv(per_repeat_path, rows)
    _write_csv(summary_path, aggregate_rows)
    metadata = {
        "eval_mode": str(inference_config.eval_mode),
        "num_checkpoints": len(targets),
        "num_repeats": num_repeats,
        "seed_base": int(inference_config.seed_base),
        "device": device,
        "per_repeat_csv": str(per_repeat_path),
        "summary_csv": str(summary_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved stochastic-depth inference results to {output_dir}")
    return metadata


@hydra.main(
    config_path=CONFIGS_PATH,
    config_name="infer_stochastic_depth",
    version_base=None,
)
def main(config: DictConfig) -> None:
    run_inference(config)


if __name__ == "__main__":
    main()
