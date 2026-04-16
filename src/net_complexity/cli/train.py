from __future__ import annotations

from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
from omegaconf import DictConfig

from net_complexity.training.engine import run_training


CONFIGS_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(config_path=CONFIGS_PATH, config_name="train", version_base=None)
def main(config: DictConfig):
    run_training(config)


if __name__ == "__main__":
    main()
