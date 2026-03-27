import torch
from dataclasses import dataclass


@dataclass
class ClassifModelOutput:
    ce_loss: torch.Tensor = None
    regularization_loss: torch.Tensor = None
    loss: torch.Tensor = None
    logits: torch.Tensor = None
    mean_activations: list[torch.Tensor] = None
