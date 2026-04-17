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


class CollapseDetected(RuntimeError):
    def __init__(
        self,
        *,
        epoch: int,
        consecutive_epochs: int,
        best_epoch_so_far: int,
        epochs_since_best: int,
        best_val_acc_so_far: float,
        valid_accuracy: float,
        valid_loss: float,
        valid_average_zero_prob: float,
        acc_threshold: float,
        loss_threshold: float,
        zero_threshold: float,
    ) -> None:
        super().__init__(
            f"Collapse detected at epoch {epoch}: "
            f"valid_accuracy={valid_accuracy:.6f}, "
            f"valid_loss={valid_loss:.6f}, "
            f"valid_average_zero_prob={valid_average_zero_prob:.6f}, "
            f"best_val_acc_so_far={best_val_acc_so_far:.6f}, "
            f"best_epoch_so_far={best_epoch_so_far}, "
            f"epochs_since_best={epochs_since_best}, "
            f"consecutive_epochs={consecutive_epochs}"
        )
        self.epoch = int(epoch)
        self.consecutive_epochs = int(consecutive_epochs)
        self.best_epoch_so_far = int(best_epoch_so_far)
        self.epochs_since_best = int(epochs_since_best)
        self.best_val_acc_so_far = float(best_val_acc_so_far)
        self.valid_accuracy = float(valid_accuracy)
        self.valid_loss = float(valid_loss)
        self.valid_average_zero_prob = float(valid_average_zero_prob)
        self.acc_threshold = float(acc_threshold)
        self.loss_threshold = float(loss_threshold)
        self.zero_threshold = float(zero_threshold)

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "stop_reason": "collapse_detected",
            "collapse_epoch": self.epoch,
            "collapse_consecutive_epochs": self.consecutive_epochs,
            "collapse_best_epoch_so_far": self.best_epoch_so_far,
            "collapse_epochs_since_best": self.epochs_since_best,
            "collapse_best_val_acc_so_far": self.best_val_acc_so_far,
            "collapse_valid_accuracy": self.valid_accuracy,
            "collapse_valid_loss": self.valid_loss,
            "collapse_valid_average_zero_prob": self.valid_average_zero_prob,
            "collapse_acc_threshold": self.acc_threshold,
            "collapse_loss_threshold": self.loss_threshold,
            "collapse_zero_threshold": self.zero_threshold,
        }


class CollapseGuard:
    def __init__(
        self,
        *,
        min_epoch: int,
        patience: int,
        min_epochs_since_best: int | None = None,
        acc_threshold_abs: float,
        acc_threshold_rel: float,
        loss_threshold: float,
        zero_threshold: float,
        accuracy_metric_name: str = "valid_accuracy",
        loss_metric_name: str = "valid_loss",
        zero_metric_name: str = "valid_average_zero_prob",
    ) -> None:
        if min_epoch <= 0:
            raise ValueError("collapse guard min_epoch must be >= 1.")
        if patience <= 0:
            raise ValueError("collapse guard patience must be > 0.")
        if min_epochs_since_best is not None and min_epochs_since_best <= 0:
            raise ValueError("collapse guard min_epochs_since_best must be > 0.")
        self.min_epoch = int(min_epoch)
        self.patience = int(patience)
        self.min_epochs_since_best = int(min_epochs_since_best or patience)
        self.acc_threshold_abs = float(acc_threshold_abs)
        self.acc_threshold_rel = float(acc_threshold_rel)
        self.loss_threshold = float(loss_threshold)
        self.zero_threshold = float(zero_threshold)
        self.accuracy_metric_name = accuracy_metric_name
        self.loss_metric_name = loss_metric_name
        self.zero_metric_name = zero_metric_name
        self.best_val_acc_so_far: float | None = None
        self.best_epoch_so_far: int | None = None
        self.consecutive_collapse_epochs = 0

    def _require_metric(self, metrics: Mapping[str, Any], metric_name: str) -> float:
        if metric_name not in metrics:
            available_metrics = ", ".join(sorted(metrics.keys()))
            raise KeyError(
                f"Collapse guard metric '{metric_name}' is missing in validation metrics. "
                f"Available metrics: {available_metrics}"
            )
        return float(metrics[metric_name])

    def __call__(
        self,
        epoch: int,
        train_metrics: Mapping[str, Any],
        valid_metrics: Mapping[str, Any],
        model,
        optimizer,
        run_history,
    ) -> None:
        del train_metrics, model, optimizer, run_history

        current_accuracy = self._require_metric(valid_metrics, self.accuracy_metric_name)
        current_loss = self._require_metric(valid_metrics, self.loss_metric_name)
        current_zero = self._require_metric(valid_metrics, self.zero_metric_name)

        if self.best_val_acc_so_far is None or current_accuracy > self.best_val_acc_so_far:
            self.best_val_acc_so_far = current_accuracy
            self.best_epoch_so_far = int(epoch)

        if int(epoch) < self.min_epoch:
            self.consecutive_collapse_epochs = 0
            return

        if self.best_epoch_so_far is None:
            self.best_epoch_so_far = int(epoch)
        acc_threshold = max(
            self.acc_threshold_abs,
            self.acc_threshold_rel * self.best_val_acc_so_far,
        )
        epochs_since_best = int(epoch) - self.best_epoch_so_far
        collapse_condition = (
            current_accuracy <= acc_threshold
            and current_loss >= self.loss_threshold
            and current_zero >= self.zero_threshold
        )

        if collapse_condition:
            self.consecutive_collapse_epochs += 1
        else:
            self.consecutive_collapse_epochs = 0
            return

        if (
            self.consecutive_collapse_epochs >= self.patience
            and epochs_since_best >= self.min_epochs_since_best
        ):
            raise CollapseDetected(
                epoch=int(epoch),
                consecutive_epochs=self.consecutive_collapse_epochs,
                best_epoch_so_far=self.best_epoch_so_far,
                epochs_since_best=epochs_since_best,
                best_val_acc_so_far=self.best_val_acc_so_far,
                valid_accuracy=current_accuracy,
                valid_loss=current_loss,
                valid_average_zero_prob=current_zero,
                acc_threshold=acc_threshold,
                loss_threshold=self.loss_threshold,
                zero_threshold=self.zero_threshold,
            )
