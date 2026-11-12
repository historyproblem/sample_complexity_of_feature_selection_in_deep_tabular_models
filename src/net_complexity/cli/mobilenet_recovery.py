from __future__ import annotations

from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
from omegaconf import DictConfig

from net_complexity.training.mobilenet_selected_recovery import recover_selected_model


@hydra.main(config_path=str(Path(__file__).resolve().parents[3] / "configs"),
            config_name="mobilenet_selected_recovery45", version_base=None)
def main(config: DictConfig):
    recover_selected_model(config)


if __name__ == "__main__":
    main()
