from __future__ import annotations

from typing import Any, Mapping


class RepeatRestartRequested(RuntimeError):
    def __init__(
        self,
        *,
        metric_name: str,
        epoch: int,
        value: float,
        threshold: float,
        mode: str,
        run_dir: str | None = None,
        run_id: str | None = None,
    ) -> None:
        comparator = ">=" if mode == "max" else "<="
        super().__init__(
            f"Repeat restart requested at epoch {epoch}: "
            f"{metric_name}={value:.6f} did not reach {comparator} {threshold:.6f}."
        )
        self.metric_name = metric_name
        self.epoch = int(epoch)
        self.value = float(value)
        self.threshold = float(threshold)
        self.mode = mode
        self.run_dir = run_dir
        self.run_id = run_id


class RepeatRestartGuard:
    def __init__(self, *, metric_name: str, mode: str, epoch: int, threshold: float):
        self.metric_name = metric_name
        self.mode = mode
        self.epoch = int(epoch)
        self.threshold = float(threshold)
        self.reached_threshold = False
        if self.mode not in {"max", "min"}:
            raise ValueError("restart guard mode must be either 'max' or 'min'.")
        if self.epoch <= 0:
            raise ValueError("restart guard epoch must be >= 1.")

    def _passes_threshold(self, value: float) -> bool:
        if self.mode == "max":
            return value >= self.threshold
        return value <= self.threshold

    def __call__(
        self,
        epoch: int,
        train_metrics: Mapping[str, Any],
        valid_metrics: Mapping[str, Any],
        model,
        optimizer,
        run_history,
    ) -> None:
        if self.metric_name not in valid_metrics:
            available_metrics = ", ".join(sorted(valid_metrics.keys()))
            raise KeyError(
                f"Restart guard metric '{self.metric_name}' is missing in validation metrics. "
                f"Available metrics: {available_metrics}"
            )

        current_value = float(valid_metrics[self.metric_name])
        if self._passes_threshold(current_value):
            self.reached_threshold = True
            return

        if int(epoch) >= self.epoch and not self.reached_threshold:
            raise RepeatRestartRequested(
                metric_name=self.metric_name,
                epoch=int(epoch),
                value=current_value,
                threshold=self.threshold,
                mode=self.mode,
                run_dir=str(run_history.run_dir) if run_history is not None else None,
                run_id=getattr(run_history, "run_id", None) if run_history is not None else None,
            )
