from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the controlled CIFAR-10 ResNet-50 baseline and run the "
            "AutoPruner keep-ratio series on its best checkpoint."
        )
    )
    parser.add_argument(
        "--repeats-per-ratio",
        type=int,
        default=3,
        help=(
            "Repeats for each author-reported keep ratio. The default of 3 "
            "leaves room for baseline training in one 11-hour V100 job."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.repeats_per_ratio <= 0:
        raise ValueError("--repeats-per-ratio must be positive.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    baseline_dir = (
        REPO_ROOT
        / "outputs"
        / "runs"
        / f"{run_id}_best_practice_resnet50_on_cifar10"
    ).resolve()
    checkpoint = baseline_dir / "checkpoints" / "best.pt"

    _run(
        [
            sys.executable,
            "src/net_complexity/train.py",
            "experiment=best_practice_resnet50_on_cifar10",
            f"hydra.run.dir={baseline_dir}",
        ]
    )

    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(
            "Baseline training finished without a non-empty best checkpoint: "
            f"{checkpoint}"
        )
    print(f"Verified baseline checkpoint: {checkpoint}", flush=True)

    _run(
        [
            sys.executable,
            "src/net_complexity/tune.py",
            "--config-name=tune_autopruner_resnet50_cifar10_v100_11h",
            f"model.pretrained_checkpoint={checkpoint}",
            f"tuning.repeats_per_trial={args.repeats_per_ratio}",
        ]
    )


if __name__ == "__main__":
    main()
