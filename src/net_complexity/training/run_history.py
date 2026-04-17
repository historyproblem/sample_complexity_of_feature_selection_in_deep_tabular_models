from __future__ import annotations

import csv
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import DictConfig, OmegaConf


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "run"


class RunHistory:
    def __init__(self, config: DictConfig):
        self.config = config
        self.repo_root = Path(__file__).resolve().parents[3]
        self.started_at = datetime.now()
        self.history_records: list[dict[str, Any]] = []
        self.batch_records: list[dict[str, Any]] = []
        self.global_batch_step = 0
        self.stage_batch_steps: dict[str, int] = {}
        self.best_metric_name: str | None = None
        self.best_metric_value: float | None = None
        self.best_epoch: int | None = None

        resolved_config = OmegaConf.to_container(config, resolve=True)
        run_name = "run"
        if isinstance(resolved_config, dict):
            run_name = (
                resolved_config.get("mlflow", {}).get("run_name")
                or resolved_config.get("run_history", {}).get("run_name")
                or run_name
            )

        run_history_cfg = getattr(config, "run_history", None)
        root_dir = self.repo_root / "outputs" / "runs"
        if run_history_cfg is not None and getattr(run_history_cfg, "root_dir", None):
            configured_root = Path(str(run_history_cfg.root_dir))
            root_dir = configured_root if configured_root.is_absolute() else self.repo_root / configured_root

        timestamp = self.started_at.strftime("%Y%m%d_%H%M%S")
        self.run_name = str(run_name)
        self.run_id = f"{timestamp}_{_slugify(self.run_name)}"
        self.run_dir = self._make_unique_dir(root_dir, self.run_id)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.history_path = self.run_dir / "history.csv"
        self.batch_history_path = self.run_dir / "batch_history.csv"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config_resolved.yaml"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=False)

        OmegaConf.save(config=OmegaConf.create(resolved_config), f=str(self.config_path))
        self._write_json(
            self.run_dir / "run_info.json",
            {
                "run_id": self.run_id,
                "run_name": self.run_name,
                "seed": getattr(config, "seed", None),
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "run_dir": str(self.run_dir),
            },
        )

    def _make_unique_dir(self, root_dir: Path, base_name: str) -> Path:
        root_dir.mkdir(parents=True, exist_ok=True)
        candidate = root_dir / base_name
        suffix = 1
        while candidate.exists():
            candidate = root_dir / f"{base_name}_{suffix}"
            suffix += 1
        return candidate

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _to_cpu(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: self._to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._to_cpu(item) for item in value)
        return value

    def _rewrite_history(self) -> None:
        if not self.history_records:
            return

        fieldnames = ["epoch"]
        for record in self.history_records:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)

        with self.history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history_records)

    def _rewrite_batch_history(self) -> None:
        if not self.batch_records:
            return

        fieldnames = ["global_batch_step", "stage", "stage_batch_step", "epoch", "batch_in_epoch"]
        for record in self.batch_records:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)

        with self.batch_history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.batch_records)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Mapping[str, Any],
        valid_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        record = {
            "epoch": int(epoch),
            **dict(train_metrics),
            **dict(valid_metrics),
        }
        self.history_records.append(record)
        self._rewrite_history()
        return record

    def log_batch(
        self,
        stage: str,
        epoch: int,
        batch_in_epoch: int,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        stage = str(stage)
        self.global_batch_step += 1
        self.stage_batch_steps[stage] = self.stage_batch_steps.get(stage, 0) + 1

        record = {
            "global_batch_step": self.global_batch_step,
            "stage": stage,
            "stage_batch_step": self.stage_batch_steps[stage],
            "epoch": int(epoch),
            "batch_in_epoch": int(batch_in_epoch),
            **dict(metrics),
        }
        self.batch_records.append(record)
        self._rewrite_batch_history()
        return record

    def resolve_monitor(self, valid_metrics: Mapping[str, Any]) -> tuple[str | None, str]:
        run_history_cfg = getattr(self.config, "run_history", None)
        monitor = None
        mode = "min"
        if run_history_cfg is not None:
            monitor = getattr(run_history_cfg, "monitor", None)
            mode = str(getattr(run_history_cfg, "mode", "min")).lower()

        if monitor is None:
            for candidate in ("valid_loss", "valid_ce_loss", "valid_accuracy"):
                if candidate in valid_metrics:
                    monitor = candidate
                    break

        return str(monitor) if monitor is not None else None, mode

    def should_update_best(self, epoch: int, valid_metrics: Mapping[str, Any]) -> bool:
        monitor, mode = self.resolve_monitor(valid_metrics)
        if monitor is None:
            return False

        current_value = valid_metrics.get(monitor)
        if not isinstance(current_value, (int, float)):
            return False

        if mode not in {"min", "max"}:
            raise ValueError("run_history.mode must be either 'min' or 'max'.")

        if self.best_metric_value is None:
            improved = True
        elif mode == "min":
            improved = float(current_value) < self.best_metric_value
        else:
            improved = float(current_value) > self.best_metric_value

        if improved:
            self.best_metric_name = monitor
            self.best_metric_value = float(current_value)
            self.best_epoch = int(epoch)

        return improved

    def save_checkpoint(
        self,
        file_name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Mapping[str, Any],
    ) -> Path:
        checkpoint_path = self.checkpoints_dir / file_name
        payload = {
            "epoch": int(epoch),
            "model_state_dict": self._to_cpu(deepcopy(model.state_dict())),
            "optimizer_state_dict": self._to_cpu(deepcopy(optimizer.state_dict())),
            "metrics": dict(metrics),
            "run_id": self.run_id,
            "run_name": self.run_name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def save_summary(
        self,
        test_metrics: Mapping[str, Any] | None = None,
        stop_info: Mapping[str, Any] | None = None,
    ) -> None:
        summary = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "seed": getattr(self.config, "seed", None),
            "run_dir": str(self.run_dir),
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "num_epochs_logged": len(self.history_records),
            "num_batches_logged": len(self.batch_records),
            "stage_batch_steps": dict(self.stage_batch_steps),
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
            "best_epoch": self.best_epoch,
            "test_metrics": dict(test_metrics or {}),
        }
        if stop_info is not None:
            summary.update(dict(stop_info))
        self._write_json(self.summary_path, summary)
