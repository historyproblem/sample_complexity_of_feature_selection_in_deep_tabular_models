"""Run the 15+30 pilot, then six ordered 30+15 AutoPruner runs."""

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
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--pilot-only", action="store_true")
    scope.add_argument("--sweep-only", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    if not (root / "configs" / "train.yaml").is_file():
        parser.error("Run this script from the repository root.")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        parser.error(f"Pretrained checkpoint is missing or empty: {checkpoint}")

    override = f"model.pretrained_checkpoint={checkpoint}"
    stages = []
    if not args.sweep_only:
        stages.append((
            "pilot 15+30, keep 0.5",
            "src/net_complexity/train.py",
            "train_autopruner_mobilenetv2_tinyimagenet200_pilot_15_30",
        ))
    if not args.pilot_only:
        stages.append((
            "six-run 30+15 keep-ratio sweep",
            "src/net_complexity/tune.py",
            "tune_autopruner_mobilenetv2_tinyimagenet200_30_15",
        ))

    for label, entrypoint, config_name in stages:
        command = [
            sys.executable, "-u", entrypoint,
            f"--config-name={config_name}", override,
        ]
        print(f"[AutoPruner] Starting {label}: {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=root, check=True)
        print(f"[AutoPruner] Completed {label}", flush=True)


if __name__ == "__main__":
    main()
