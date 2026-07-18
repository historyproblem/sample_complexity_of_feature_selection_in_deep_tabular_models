from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@hydra.main(
    config_path="../configs",
    config_name="experiment/efficientnetv2_s_aig_adaptive_lambda_cifar10",
    version_base=None,
)
def main(config: DictConfig) -> None:
    from net_complexity.training.engine import run_training

    run_training(config)


if __name__ == "__main__":
    main()
