from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def config_for(job):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="cyclic_channel_train", overrides=[f"experiment=audit/{job}"])


def make_initializer(config, destination):
    from copy import deepcopy
    from net_complexity.training.randomness import set_random_seed
    from net_complexity.training.pruning_measurement import state_hash
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    set_random_seed(int(config.seed))
    cfg = deepcopy(config.model)
    cfg.lambda_coef = 0.0
    # Plain backbone: gate construction must not perturb random backbone weights.
    cfg.backbone.resnet_block = {
        "_target_": "net_complexity.models.resnet.Bottleneck", "_partial_": True,
    }
    model = instantiate(cfg)
    state = model.state_dict()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state, "trained_epochs": 0, "seed": int(config.seed),
                "model_state_hash": state_hash(state)}, destination)
    return destination
