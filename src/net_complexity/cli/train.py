from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
from omegaconf import DictConfig

from net_complexity.training.engine import run_training


@hydra.main(config_path="../../../configs/", config_name="train", version_base=None)
def main(config: DictConfig):
    run_training(config)


if __name__ == "__main__":
    main()
