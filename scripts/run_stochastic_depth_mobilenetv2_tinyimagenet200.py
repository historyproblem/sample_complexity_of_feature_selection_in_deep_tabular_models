"""Fine-tune pretrained MobileNetV2 on TinyImageNet-200 with seven p_L values."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_CHECKPOINT = (
    "outputs/runs/"
    "20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200/"
    "checkpoints/best.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command without training"
    )
    args = parser.parse_args()
    root = Path.cwd()
    if not (root / "configs/train.yaml").is_file():
        parser.error("Run this script from the repository root.")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        parser.error(f"Pretrained checkpoint is missing or empty: {checkpoint}")
    command = [
        sys.executable,
        "-u",
        "src/net_complexity/tune.py",
        "--config-name=tune_stochastic_depth_mobilenetv2_tinyimagenet200_pL_grid_45",
        f"model.backbone.pretrained_checkpoint={checkpoint}",
        f"dataloaders.batch_size={args.batch_size}",
    ]
    print(f"[StochasticDepth] Seven 45-epoch runs: {' '.join(command)}", flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
