from __future__ import annotations

import csv
import gzip
import json
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .channel_history import CHANNEL_HISTORY_FIELDNAMES, resolve_channel_history_collector


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
        self.last_train_metrics: dict[str, Any] = {}
        self.last_valid_metrics: dict[str, Any] = {}
        self.best_valid_metrics: dict[str, Any] = {}

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

        use_hydra_output_dir = True
        if run_history_cfg is not None and getattr(run_history_cfg, "use_hydra_output_dir", None) is not None:
            use_hydra_output_dir = bool(run_history_cfg.use_hydra_output_dir)

        timestamp = self.started_at.strftime("%Y%m%d_%H%M%S")
        self.run_name = str(run_name)
        hydra_output_dir = self._resolve_hydra_output_dir() if use_hydra_output_dir else None
        if hydra_output_dir is not None:
            self.run_dir = hydra_output_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.run_id = self.run_dir.name
        else:
            self.run_id = f"{timestamp}_{_slugify(self.run_name)}"
            self.run_dir = self._make_unique_dir(root_dir, self.run_id)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.history_path = self.run_dir / "history.csv"
        self.batch_history_path = self.run_dir / "batch_history.csv"
        self.channel_history_path = self.run_dir / "channel_history.csv.gz"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config_resolved.yaml"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=False)

        self.channel_history_enabled = bool(
            getattr(run_history_cfg, "log_channel_history", False)
        )
        self.channel_history_collector = (
            resolve_channel_history_collector(config)
            if self.channel_history_enabled
            else None
        )

        OmegaConf.save(config=OmegaConf.create(resolved_config), f=str(self.config_path))
        if self.channel_history_enabled:
            with gzip.open(self.channel_history_path, "wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CHANNEL_HISTORY_FIELDNAMES)
                writer.writeheader()
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

    def _resolve_hydra_output_dir(self) -> Path | None:
        if not HydraConfig.initialized():
            return None
        runtime_output_dir = getattr(HydraConfig.get().runtime, "output_dir", None)
        if runtime_output_dir is None:
            return None
        return Path(str(runtime_output_dir))

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

    def _relative_to_run_dir(self, path: Path) -> str:
        return str(path.relative_to(self.run_dir))

    def _existing_relative_to_run_dir(self, path: Path) -> str | None:
        if not path.exists():
            return None
        return self._relative_to_run_dir(path)

    def _build_provenance(self) -> dict[str, Any]:
        commit_hash = None
        git_dirty = None
        try:
            commit_hash = subprocess.run(
                ["git", "-C", str(self.repo_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip() or None
        except Exception:
            commit_hash = None

        try:
            status_output = subprocess.run(
                ["git", "-C", str(self.repo_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            git_dirty = bool(status_output.strip())
        except Exception:
            git_dirty = None

        return {
            "git_commit": commit_hash,
            "git_dirty": git_dirty,
        }

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
        self.last_train_metrics = dict(train_metrics)
        self.last_valid_metrics = dict(valid_metrics)
        record = {
            "epoch": int(epoch),
            **self.last_train_metrics,
            **self.last_valid_metrics,
        }
        self.history_records.append(record)
        self._rewrite_history()
        return record

    def update_last_epoch(self, extra_metrics: Mapping[str, Any]) -> None:
        if not self.history_records:
            return
        self.history_records[-1].update(dict(extra_metrics))
        self._rewrite_history()

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
            self.best_valid_metrics = dict(valid_metrics)

        return improved

    def log_channel_history(self, epoch: int, model: torch.nn.Module) -> int:
        if not self.channel_history_enabled or self.channel_history_collector is None:
            return 0

        rows = self.channel_history_collector.collect(model, epoch)
        if not rows:
            return 0

        with gzip.open(self.channel_history_path, "at", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CHANNEL_HISTORY_FIELDNAMES)
            writer.writerows(rows)
        return len(rows)

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
        final_train_metrics: Mapping[str, Any] | None = None,
        final_valid_metrics: Mapping[str, Any] | None = None,
        test_metrics: Mapping[str, Any] | None = None,
        stop_info: Mapping[str, Any] | None = None,
    ) -> None:
        finished_at = datetime.now()
        duration_sec = (finished_at - self.started_at).total_seconds()
        summary = {
            "schema_version": 2,
            "identity": {
                "run_id": self.run_id,
                "run_name": self.run_name,
                "seed": getattr(self.config, "seed", None),
                "run_dir": str(self.run_dir),
            },
            "timing": {
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "finished_at": finished_at.isoformat(timespec="seconds"),
                "duration_sec": duration_sec,
                "num_epochs_logged": len(self.history_records),
                "num_batches_logged": len(self.batch_records),
            },
            "final_train": dict(final_train_metrics or self.last_train_metrics),
            "final_valid": dict(final_valid_metrics or self.last_valid_metrics),
            "best_valid": {
                "epoch": self.best_epoch,
                "metric": self.best_metric_name,
                "value": self.best_metric_value,
                "metrics": dict(self.best_valid_metrics),
            },
            "test": dict(test_metrics or {}),
            "artifacts": {
                "config_resolved": self._existing_relative_to_run_dir(self.config_path),
                "history": self._existing_relative_to_run_dir(self.history_path),
                "channel_history": self._existing_relative_to_run_dir(self.channel_history_path),
                "summary": self._relative_to_run_dir(self.summary_path),
                "checkpoints": {
                    "best": self._existing_relative_to_run_dir(self.checkpoints_dir / "best.pt"),
                    "last": self._existing_relative_to_run_dir(self.checkpoints_dir / "last.pt"),
                },
            },
            "provenance": self._build_provenance(),
        }
        if stop_info is not None:
            summary["stop_info"] = dict(stop_info)
        self._write_json(self.summary_path, summary)
