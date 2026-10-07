"""Compare five 45-epoch AutoPruner schedules at one fixed model size."""

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
    args = parser.parse_args()

    root = Path.cwd()
    if not (root / "configs" / "train.yaml").is_file():
        parser.error("Run this script from the repository root.")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        parser.error(
            f"Pretrained checkpoint is missing or empty: {checkpoint}. "
            "Upload mobilenet_baseline_partial.tar.gz and restore its "
            "checkpoints/best.pt member as described in "
            "docs/autopruner_mobilenetv2_tinyimagenet200_45ep.md."
        )

    override = f"model.pretrained_checkpoint={checkpoint}"
    command = [
        sys.executable, "-u", "src/net_complexity/tune.py",
        "--config-name=tune_autopruner_mobilenetv2_tinyimagenet200_schedule_45",
        override,
    ]
    print(f"[AutoPruner] Starting fixed-size schedule sweep: {' '.join(command)}", flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=root, check=True)
    print("[AutoPruner] Completed fixed-size schedule sweep", flush=True)


if __name__ == "__main__":
    main()
