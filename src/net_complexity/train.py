from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from net_complexity.cli.train import main
from net_complexity.training.engine import (
    EpochEndCallback,
    ProgressContext,
    collect_batch_metrics,
    evaluate,
    log_run_artifacts,
    log_training_metadata,
    prepare_metrics,
    resolve_device,
    run_training,
    train,
    train_epoch,
)

__all__ = [
    "EpochEndCallback",
    "ProgressContext",
    "collect_batch_metrics",
    "evaluate",
    "log_run_artifacts",
    "log_training_metadata",
    "main",
    "prepare_metrics",
    "resolve_device",
    "run_training",
    "train",
    "train_epoch",
]


if __name__ == "__main__":
    main()
