"""Deterministic technical fixtures, never a CIFAR quality experiment or baseline."""
from copy import deepcopy
import json
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset

from .accuracy_guided_config import compose_config
from .pruning_measurement import state_hash, mask_hash, isolated_diagnostic_rng


class SyntheticDataloaders:
    def __init__(self, batch_size=4, include_test=False, loader_seed=42, **kwargs):
        if include_test:
            raise AssertionError("Synthetic training fixture forbids constructing official test data.")
        generator = torch.Generator().manual_seed(1729)
        train = TensorDataset(torch.randn(8, 3, 8, 8, generator=generator), torch.arange(8) % 3)
        valid = TensorDataset(torch.randn(5, 3, 8, 8, generator=generator), torch.arange(5) % 3)
        self.train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(loader_seed))
        self.valid_dataloader = DataLoader(valid, batch_size=batch_size,
            generator=torch.Generator().manual_seed(loader_seed + 1))

    @property
    def test_dataloader(self):
        raise AssertionError("Official test loader must be inaccessible to selection.")


def make_synthetic_config(directory, *, learned_closed=False, reject_after=None):
    """Explicit zero-trained-epoch fixtures to exercise both selector branches.

    The synthetic reference is a programmed validation feedback signal, not a
    measured dense accuracy curve and not evidence about model quality.
    """
    from net_complexity.models.pruning_budget import gates
    from .accuracy_guided_pruning import atomic_checkpoint
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cfg = compose_config("accuracy_guided_gates_v3_smoke")
    OmegaConf.update(cfg, "model.backbone.base_width", 2, force_add=True)
    cfg.model.backbone.num_classes = 3
    cfg.dataloaders = {"_target_": "net_complexity.training.pruning_synthetic.SyntheticDataloaders",
                      "seed": 42, "loader_seed": 42, "include_test": False, "batch_size": 4}
    cfg.accuracy_guided.reference = {"history_path": str(directory / "reference.csv"),
        "state_path": str(directory / "reference_state.json"), "config_path": str(directory / "reference_config.yaml")}
    cfg.accuracy_guided.initializer.path = str(directory / "zero_epoch_initializer.pt")
    with isolated_diagnostic_rng():
        torch.manual_seed(int(cfg.seed))
        model = instantiate(cfg.model)
        with torch.no_grad():
            for gate in gates(model).values():
                gate.logits[:, 0].zero_()
                gate.logits[:, 1].fill_(float(torch.log(torch.tensor(9.0))))
            if learned_closed:
                # Two original coordinates, each in a width-two boundary: legal at floor .5.
                for gate in list(gates(model).values())[:2]:
                    gate.logits[0, 1].fill_(-float(torch.log(torch.tensor(9.0))))
        tensor_state = model.state_dict()
        init_hash = state_hash(tensor_state)
        atomic_checkpoint(cfg.accuracy_guided.initializer.path, {"trained_epochs": 0, "seed": int(cfg.seed),
            "model_state_dict": tensor_state, "model_state_hash": init_hash,
            "fixture": "synthetic_preset_gate_probabilities; no optimizer steps"})
    total = int(cfg.accuracy_guided.total_epochs)
    Path(cfg.accuracy_guided.reference.history_path).write_text("global_epoch,valid_accuracy\n" + "".join(
        f"{epoch},{1.0 if reject_after is not None and epoch >= reject_after else 0.0}\n"
        for epoch in range(1, total + 1)))
    Path(cfg.accuracy_guided.reference.state_path).write_text(json.dumps({"status": "completed",
        "accepted_mask": {}, "test_evaluated": False, "global_epochs_completed": total,
        "total_epochs_allocated": total, "seed": int(cfg.seed), "common_init_hash": init_hash,
        "split_indices_hash": mask_hash({"train": list(range(8)), "valid": list(range(5))}),
        "validation": {"example_count": 5}, "synthetic_reference": True,
        "reference_training_epochs_actually_executed": 0,
        "description": "Artificial feedback fixture covering logical epochs; no dense model was trained."}))
    OmegaConf.save(deepcopy(cfg), cfg.accuracy_guided.reference.config_path, resolve=True)
    return cfg
