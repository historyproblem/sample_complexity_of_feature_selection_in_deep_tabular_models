"""CLI entry point for cyclic AIG layer-dropping training.

Usage::

    python src/net_complexity/cli/cyclic_train.py experiment=cyclic_aig_resnet50_cifar10
    python src/net_complexity/cli/cyclic_train.py experiment=cyclic_aig_resnet50_cifar10 \\
        cyclic_layer_dropping.max_cycles=3 \\
        cyclic_layer_dropping.g_prob_threshold=0.5
"""
from __future__ import annotations

from pathlib import Path

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from net_complexity.training.cyclic_aig import run_cyclic_aig_training


CONFIGS_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(config_path=CONFIGS_PATH, config_name="cyclic_train", version_base=None)
def main(config: DictConfig) -> None:
    output_root = Path(HydraConfig.get().runtime.output_dir)
    run_cyclic_aig_training(config, output_root=output_root)


if __name__ == "__main__":
    main()
