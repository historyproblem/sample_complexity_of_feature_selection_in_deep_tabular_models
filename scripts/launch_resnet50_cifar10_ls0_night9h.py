"""Run three complete pruning experiments planned for a nine-hour night."""
from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from launch_resnet50_cifar10_ls0_curve7 import main as run_curve, NIGHT_CONFIG


def main(argv=None):
    return run_curve(argv, default_config=NIGHT_CONFIG)


if __name__ == "__main__":
    result = main()
    if result.get("status") == "completed_with_failures":
        raise SystemExit(1)
