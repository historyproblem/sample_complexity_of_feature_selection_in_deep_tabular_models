"""CLI entry point for cyclic layer- or channel-dropping training.

Dispatches on which cyclic config section the composed config carries:
``cyclic_layer_dropping`` -> AIG block-level cyclic pruning (cyclic_aig.py);
``cyclic_channel_pruning`` -> Gumbel channel-level cyclic pruning
(cyclic_channel_pruning.py). Exactly one of the two must be present and
enabled.

Usage::

    python src/net_complexity/cli/cyclic_train.py experiment=cyclic_aig_resnet50_cifar10
    python src/net_complexity/cli/cyclic_train.py experiment=cyclic_aig_resnet50_cifar10 \\
        cyclic_layer_dropping.max_cycles=3 \\
        cyclic_layer_dropping.g_prob_threshold=0.5

    python src/net_complexity/cli/cyclic_train.py \\
        --config-name cyclic_channel_train \\
        experiment=ours_iterative_channels_resnet50_cifar10
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
from net_complexity.training.cyclic_channel_pruning import run_cyclic_channel_pruning_training


CONFIGS_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(config_path=CONFIGS_PATH, config_name="cyclic_train", version_base=None)
def main(config: DictConfig) -> None:
    output_root = Path(HydraConfig.get().runtime.output_dir)

    layer_cfg = getattr(config, "cyclic_layer_dropping", None)
    layer_enabled = layer_cfg is not None and bool(getattr(layer_cfg, "enabled", False))

    channel_cfg = getattr(config, "cyclic_channel_pruning", None)
    channel_enabled = channel_cfg is not None and bool(getattr(channel_cfg, "enabled", False))

    if layer_enabled and channel_enabled:
        raise ValueError(
            "Both cyclic_layer_dropping and cyclic_channel_pruning are enabled — "
            "enable exactly one per run."
        )
    if channel_enabled:
        run_cyclic_channel_pruning_training(config, output_root=output_root)
    elif layer_enabled:
        run_cyclic_aig_training(config, output_root=output_root)
    else:
        raise ValueError(
            "Neither cyclic_layer_dropping nor cyclic_channel_pruning is enabled — "
            "set one of them in the experiment config."
        )


if __name__ == "__main__":
    main()
