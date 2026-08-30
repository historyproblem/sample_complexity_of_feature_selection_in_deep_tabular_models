#!/usr/bin/env python3
"""Run the 300-epoch dense baseline, then the two adaptive-lambda trials."""

from __future__ import annotations

import csv
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "outputs/baselines/resnet50_adamw_cifar10_300ep"
BASELINE_HISTORY = BASELINE_DIR / "history.csv"
BASELINE_CHECKPOINT = BASELINE_DIR / "checkpoints/best.pt"
EXPECTED_EPOCHS = set(range(1, 301))


def _history_epochs(history_path: Path) -> set[int]:
    if not history_path.is_file():
        return set()

    with history_path.open(newline="", encoding="utf-8") as handle:
        return {
            int(float(row["epoch"]))
            for row in csv.DictReader(handle)
            if row.get("epoch") not in {None, ""}
        }


def _baseline_is_complete() -> bool:
    return BASELINE_CHECKPOINT.is_file() and _history_epochs(BASELINE_HISTORY) == EXPECTED_EPOCHS


def _run(command: list[str]) -> None:
    print(f"Running: {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "This project requires Python >= 3.10; "
            f"the selected interpreter is {sys.version.split()[0]}."
        )

    if not _baseline_is_complete():
        if BASELINE_DIR.exists():
            raise RuntimeError(
                f"Incomplete baseline output already exists at {BASELINE_DIR}. "
                "Move it aside or finish that run before starting the pipeline again."
            )

        _run([
            sys.executable,
            "src/net_complexity/train.py",
            "experiment=best_practice_resnet50_adamw_300ep_on_cifar10",
            "hydra.run.dir=outputs/baselines/resnet50_adamw_cifar10_300ep",
        ])
        if not _baseline_is_complete():
            raise RuntimeError(
                "Baseline training returned without a complete 300-epoch history "
                f"and best checkpoint under {BASELINE_DIR}."
            )
    else:
        print(f"Reusing complete baseline: {BASELINE_DIR}", flush=True)

    _run([
        sys.executable,
        "src/net_complexity/tune.py",
        "--config-name=tune_resnet50_aig_adaptive_lambda_checkpoint_history_gap_1_2_2_4_300ep",
    ])


if __name__ == "__main__":
    main()
