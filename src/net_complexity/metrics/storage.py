from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_RESULTS_DIR = Path("experiment_results")
RESERVED_COLUMNS = {"record_id", "recorded_at", "source_file"}


class ExperimentMetricsStore:
    def __init__(self, root_dir: str | Path = DEFAULT_RESULTS_DIR):
        self.root_dir = Path(root_dir)

    def _path(self, file_name: str | Path) -> Path:
        path = Path(file_name)
        path = path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")
        return path if path.is_absolute() else self.root_dir / path

    def _normalize(
        self,
        values: Mapping[str, Any] | None,
        *,
        reserved: set[str] | None = None,
    ) -> dict[str, int | float | str | bool]:
        result: dict[str, int | float | str | bool] = {}
        reserved = reserved or set()

        for key, value in (values or {}).items():
            if value is None:
                continue

            key = str(key)
            if key in reserved:
                raise ValueError(f"Metadata contains reserved column names: {key}.")

            if isinstance(value, bool):
                result[key] = value
            elif isinstance(value, Real):
                result[key] = value.item() if hasattr(value, "item") else value
            elif isinstance(value, (str, Path)):
                result[key] = str(value)

        return result

    def _next_record_id(self, path: Path) -> int:
        if not path.exists() or path.stat().st_size == 0:
            return 0
        frame = pd.read_csv(path, usecols=["record_id"])
        return 0 if frame.empty else int(frame["record_id"].max()) + 1

    def save(
        self,
        file_name: str | Path,
        metrics: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self._path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        metrics = self._normalize(metrics)
        if not metrics:
            raise ValueError("Metrics are empty. Nothing to save.")

        row = pd.DataFrame([{
            "record_id": self._next_record_id(path),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            **self._normalize(metadata, reserved=RESERVED_COLUMNS),
            **metrics,
        }])

        if path.exists():
            row = pd.concat([pd.read_csv(path), row], ignore_index=True, sort=False)

        row.to_csv(path, index=False)
        return path

    def load(self, file_names: str | Path | Sequence[str | Path]) -> pd.DataFrame:
        names = [file_names] if isinstance(file_names, (str, Path)) else list(file_names)
        if not names:
            raise ValueError("At least one file name is required.")

        frames = []
        for name in names:
            path = self._path(name)
            if not path.exists():
                raise FileNotFoundError(f"Metrics file not found: {path}")
            frame = pd.read_csv(path)
            frame["source_file"] = path.name
            frames.append(frame)

        return pd.concat(frames, ignore_index=True, sort=False)

    def list_files(self) -> list[Path]:
        return sorted(self.root_dir.rglob("*.csv")) if self.root_dir.exists() else []

    def show_metrics(
        self,
        file_names: str | Path | Sequence[str | Path],
        metrics: str | Sequence[str] | None = None,
        x: str = "record_id",
        kind: str = "line",
        plot: bool = True,
        title: str | None = None,
    ) -> pd.DataFrame:
        frame = self.load(file_names)

        if x not in frame.columns:
            raise KeyError(f"Column '{x}' is not present in the loaded metrics.")

        if metrics is None:
            excluded = RESERVED_COLUMNS | {x}
            metrics = [
                c for c in frame.select_dtypes(include="number").columns
                if c not in excluded
            ]
        else:
            metrics = [metrics] if isinstance(metrics, str) else list(metrics)
            missing = [m for m in metrics if m not in frame.columns]
            if missing:
                raise KeyError(f"Metric columns not found: {', '.join(missing)}")

        frame = frame.sort_values(["source_file", x]).reset_index(drop=True)

        if plot:
            self._plot(frame, metrics, x=x, kind=kind, title=title)

        return frame

    def _plot(
        self,
        frame: pd.DataFrame,
        metrics: Sequence[str],
        *,
        x: str,
        kind: str,
        title: str | None,
    ) -> None:
        if not metrics:
            raise ValueError("No numeric metrics available for plotting.")
        if kind not in {"line", "scatter", "bar"}:
            raise ValueError("kind must be one of: 'line', 'scatter', 'bar'.")

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plotting. Use plot=False to only load the data."
            ) from exc

        _, ax = plt.subplots(figsize=(10, 5))
        multi_source = frame["source_file"].nunique() > 1

        for source_file, group in frame.groupby("source_file", sort=False):
            for metric in metrics:
                label = f"{source_file}:{metric}" if multi_source else metric
                if kind == "line":
                    ax.plot(group[x], group[metric], marker="o", label=label)
                elif kind == "scatter":
                    ax.scatter(group[x], group[metric], label=label)
                else:
                    ax.bar(group[x], group[metric], alpha=0.7, label=label)

        ax.set_xlabel(x)
        ax.set_ylabel("value")
        ax.set_title(title or "Experiment metrics")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()


def save_metrics(
    file_name: str | Path,
    metrics: Mapping[str, Any],
    root_dir: str | Path = DEFAULT_RESULTS_DIR,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    return ExperimentMetricsStore(root_dir).save(file_name, metrics, metadata)


def load_metrics(
    file_names: str | Path | Sequence[str | Path],
    root_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    return ExperimentMetricsStore(root_dir).load(file_names)


def list_metric_files(root_dir: str | Path = DEFAULT_RESULTS_DIR) -> list[Path]:
    return ExperimentMetricsStore(root_dir).list_files()


def show_metrics(
    file_names: str | Path | Sequence[str | Path],
    metrics: str | Sequence[str] | None = None,
    root_dir: str | Path = DEFAULT_RESULTS_DIR,
    x: str = "record_id",
    kind: str = "line",
    plot: bool = True,
    title: str | None = None,
) -> pd.DataFrame:
    return ExperimentMetricsStore(root_dir).show_metrics(
        file_names=file_names,
        metrics=metrics,
        x=x,
        kind=kind,
        plot=plot,
        title=title,
    )